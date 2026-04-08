#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
CK_CLI_V8 = BUILD_DIR / "ck-cli-v8"
LIBCK = BUILD_DIR / "libckernel_engine.so"
V8_BUILD_PATH = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
V8_CODEGEN_PATH = ROOT / "version" / "v8" / "scripts" / "codegen_v8.py"
V8_BRIDGE_RUNNER_PATH = ROOT / "version" / "v8" / "scripts" / "run_multimodal_bridge_v8.py"


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_ir_v8 = _load_module("build_ir_v8_native_bridge_host_tests", V8_BUILD_PATH)
bridge_runner_v8 = _load_module("run_multimodal_bridge_v8_host_tests", V8_BRIDGE_RUNNER_PATH)
vision_bridge_runtime_v8 = _load_module(
    "vision_bridge_runtime_v8_host_tests",
    ROOT / "version" / "v8" / "scripts" / "vision_bridge_runtime_v8.py",
)


def _entry(name: str, dtype: str, shape: list[int], offset: int) -> dict:
    nbytes_per = {"fp32": 4, "fp16": 2, "q8_0": 1}.get(dtype, 4)
    size = 1
    for dim in shape:
        size *= int(dim)
    size *= nbytes_per
    return {
        "name": name,
        "dtype": dtype,
        "offset": offset,
        "shape": shape,
        "size": size,
        "nbytes": size,
    }


def _make_qwen3_decoder_manifest() -> dict:
    offset = 8
    entries = []

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        item = _entry(name, dtype, shape, offset)
        entries.append(item)
        offset += int(item["size"])

    add("token_emb", "q8_0", [64, 16])
    add("layer.0.ln1_gamma", "fp32", [16])
    add("layer.0.ln2_gamma", "fp32", [16])
    add("layer.0.q_norm", "fp32", [4])
    add("layer.0.k_norm", "fp32", [4])
    add("layer.0.wq", "q8_0", [16, 16])
    add("layer.0.wk", "q8_0", [16, 16])
    add("layer.0.wv", "q8_0", [16, 16])
    add("layer.0.wo", "q8_0", [16, 16])
    add("layer.0.w1", "q8_0", [32, 16])
    add("layer.0.w2", "q8_0", [16, 32])
    add("layer.0.w3", "q8_0", [32, 16])
    add("final_ln_weight", "fp32", [16])

    return {
        "config": {
            "model": "qwen3",
            "arch": "qwen3",
            "num_layers": 1,
            "embed_dim": 16,
            "num_heads": 4,
            "num_kv_heads": 4,
            "head_dim": 4,
            "intermediate_size": 32,
            "context_length": 32,
            "max_seq_len": 32,
            "vocab_size": 64,
        },
        "quant_summary": {
            "token_emb": "q8_0",
            "layer.0": {
                "wq": "q8_0",
                "wk": "q8_0",
                "wv": "q8_0",
                "wo": "q8_0",
                "w1": "q8_0",
                "w2": "q8_0",
                "w3": "q8_0",
            },
            "final_ln_weight": "fp32",
        },
        "entries": entries,
        "template": build_ir_v8._load_builtin_template_doc("qwen3"),
    }


def _write_zero_bump(manifest: dict, bump_path: Path) -> None:
    total_size = 0
    for entry in manifest["entries"]:
        end = int(entry["offset"]) + int(entry["size"])
        total_size = max(total_size, end)
    total_size = max(total_size, 8)
    bump_path.write_bytes(b"BUMPWGT5" + (b"\0" * (total_size - 8)))


def _compile_generated_model(c_path: Path, so_path: Path) -> None:
    cmd = [
        "cc",
        "-shared",
        "-fPIC",
        "-O3",
        "-fopenmp",
        "-Iinclude",
        "-Iversion/v8/src",
        str(c_path),
        "version/v8/src/ckernel_model_load_v8.c",
        "version/v8/src/ck_parallel_decode_v8.c",
        "version/v8/src/ck_parallel_prefill_v8.c",
        "-Lbuild",
        "-lckernel_engine",
        f"-Wl,-rpath,{BUILD_DIR}",
        "-o",
        str(so_path),
        "-lm",
        "-lpthread",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"generated model compile failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def _build_tiny_decoder_runtime(workdir: Path) -> tuple[Path, Path, Path]:
    manifest = _make_qwen3_decoder_manifest()
    manifest_path = workdir / "weights_manifest.json"
    bump_path = workdir / "weights.bump"
    manifest_map = workdir / "weights_manifest.map"
    prefill_ir1 = workdir / "ir1_prefill.json"
    prefill_layout = workdir / "layout_prefill.json"
    prefill_lowered = workdir / "lowered_prefill.json"
    prefill_call = workdir / "call_prefill.json"
    decode_ir1 = workdir / "ir1_decode.json"
    decode_layout = workdir / "layout_decode.json"
    decode_lowered = workdir / "lowered_decode.json"
    decode_call = workdir / "call_decode.json"
    c_path = workdir / "decoder_v8.c"
    so_path = workdir / "libdecoder_v8.so"

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_zero_bump(manifest, bump_path)

    for mode, ir1_path, layout_path, lowered_path, call_path in (
        ("prefill", prefill_ir1, prefill_layout, prefill_lowered, prefill_call),
        ("decode", decode_ir1, decode_layout, decode_lowered, decode_call),
    ):
        args = [
            "--manifest",
            str(manifest_path),
            "--mode",
            mode,
            "--output",
            str(ir1_path),
            "--layout-output",
            str(layout_path),
            "--lowered-output",
            str(lowered_path),
            "--call-output",
            str(call_path),
        ]
        if mode == "decode":
            args.extend(["--manifest-map-output", str(manifest_map)])
        rc = build_ir_v8.main(args)
        if rc != 0:
            raise AssertionError(f"build_ir_v8 failed for mode={mode}")

    result = subprocess.run(
        [
            sys.executable,
            str(V8_CODEGEN_PATH),
            "--ir",
            str(decode_call),
            "--prefill",
            str(prefill_call),
            "--layout",
            str(decode_layout),
            "--output",
            str(c_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"codegen_v8 failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    _compile_generated_model(c_path, so_path)
    return so_path, bump_path, manifest_map


class V8NativeBridgeHostTests(unittest.TestCase):
    def test_vision_bridge_contract_prefers_projector_width_output(self) -> None:
        layout = {
            "config": {
                "embed_dim": 1152,
                "vision_merged_tokens": 576,
                "projector_out_dim": 4096,
                "projector_total_out_dim": 16384,
                "projection_dim": 4096,
            }
        }
        activation_buffers = {
            "embedded_input": {"name": "embedded_input", "size_bytes": 2304 * 1152 * 4},
            "vision_output": {"name": "vision_output", "size_bytes": 576 * 16384 * 4},
        }
        bridge = vision_bridge_runtime_v8.resolve_vision_bridge_contract(layout, activation_buffers)
        self.assertEqual(bridge["named_activation"], "vision_bridge_output")
        self.assertEqual(bridge["fallback_buffer_name"], "embedded_input")
        self.assertEqual(bridge["embed_dim"], 4096)
        self.assertEqual(bridge["prefix_tokens"], 576)
        self.assertEqual(bridge["used_nbytes"], 576 * 4096 * 4)
        self.assertEqual(bridge["reason"], "projector_output")

    def test_format_prompt_with_qwen3vl_contract_matches_chatml_shell(self) -> None:
        template_doc = build_ir_v8._load_builtin_template_doc("qwen3vl")
        self.assertIsNotNone(template_doc)
        contract = template_doc["contract"]["chat_contract"]

        formatted = bridge_runner_v8._format_prompt_with_chat_contract(
            "Describe the image.",
            contract,
            thinking_mode="auto",
        )

        self.assertEqual(
            formatted,
            "<|im_start|>user\nDescribe the image.<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_format_prompt_with_qwen3vl_contract_honors_suppressed_thinking(self) -> None:
        template_doc = build_ir_v8._load_builtin_template_doc("qwen3vl")
        self.assertIsNotNone(template_doc)
        contract = template_doc["contract"]["chat_contract"]

        formatted = bridge_runner_v8._format_prompt_with_chat_contract(
            "Describe the image.",
            contract,
            thinking_mode="suppressed",
        )

        self.assertEqual(
            formatted,
            "<|im_start|>user\n/no_think\nDescribe the image.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_format_multimodal_prompt_segments_wraps_qwen3vl_image_chunk(self) -> None:
        template_doc = build_ir_v8._load_builtin_template_doc("qwen3vl")
        self.assertIsNotNone(template_doc)
        contract = template_doc["contract"]["chat_contract"]

        segments = bridge_runner_v8._format_multimodal_prompt_segments(
            "Describe the image.",
            contract,
            include_image=True,
        )

        self.assertTrue(segments["uses_image_chunks"])
        self.assertEqual(segments["before_text"], "<|im_start|>user\n<|vision_start|>")
        self.assertEqual(segments["after_text"], "<|vision_end|>Describe the image.<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(
            segments["formatted_prompt"],
            "<|im_start|>user\n<|vision_start|><image_embeds><|vision_end|>Describe the image.<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_fallback_chat_contract_from_template_text_preserves_vision_markers(self) -> None:
        template = (
            "{%- for message in messages %}"
            "{{- '<|im_start|>' + message.role + '\\n' }}"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "{{- '<|im_end|>\\n' }}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
        )

        contract = bridge_runner_v8._fallback_chat_contract_from_template_text(template)

        self.assertIsNotNone(contract)
        self.assertEqual(contract["name"], "chatml_auto")
        self.assertEqual(contract["image_begin_marker"], "<|vision_start|>")
        self.assertEqual(contract["image_end_marker"], "<|vision_end|>")
        self.assertEqual(contract["assistant_generation_prefix_by_thinking_mode"], {})
        self.assertEqual(contract["last_user_prefix_by_thinking_mode"], {})
        self.assertIn("<|vision_start|>", contract["template_markers"])
        self.assertIn("<|vision_end|>", contract["template_markers"])

    def test_resolve_decoder_chat_contract_auto_prefers_gguf_template_over_arch_builtin(self) -> None:
        gguf_template = (
            "{%- for message in messages %}"
            "{{- '<|im_start|>' + message.role + '\\n' }}"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "{{- '<|im_end|>\\n' }}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
        )

        with mock.patch.object(
            bridge_runner_v8,
            "_read_gguf_metadata",
            return_value={
                "general.architecture": "qwen3vl",
                "tokenizer.chat_template": gguf_template,
            },
        ):
            contract = bridge_runner_v8._resolve_decoder_chat_contract(
                Path("/tmp/fake-qwen3vl.gguf"),
                chat_template_mode="auto",
            )

        self.assertIsNotNone(contract)
        self.assertEqual(contract["name"], "chatml_auto")
        formatted = bridge_runner_v8._format_multimodal_prompt_segments(
            "Explain this image.",
            contract,
            include_image=True,
            thinking_mode="suppressed",
        )
        self.assertEqual(formatted["before_text"], "<|im_start|>user\n<|vision_start|>")
        self.assertEqual(
            formatted["after_text"],
            "<|vision_end|>Explain this image.<|im_end|>\n<|im_start|>assistant\n",
        )
        self.assertEqual(
            formatted["formatted_prompt"],
            "<|im_start|>user\n<|vision_start|><image_embeds><|vision_end|>Explain this image.<|im_end|>\n<|im_start|>assistant\n",
        )

    def test_resolve_stop_token_policy_uses_eos_and_chat_contract_markers(self) -> None:
        class FakeTokenizer:
            eos_id = 151645
            token_to_id = {"<|im_end|>": 151645, "<|eot_id|>": 151643}

            def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
                token_id = self.token_to_id.get(text)
                return [] if token_id is None else [token_id]

        policy = bridge_runner_v8._resolve_stop_token_policy(
            FakeTokenizer(),
            {
                "token_stop_markers": ["<|im_end|>", "<|eot_id|>"],
            },
        )

        self.assertEqual(policy["eos_id"], 151645)
        self.assertEqual(policy["stop_ids"], [151643, 151645])
        self.assertEqual(policy["stop_markers"], ["<|im_end|>", "<|eot_id|>"])

    def test_bridge_runner_auto_formats_prompt_from_decoder_arch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_prompt_") as tmpdir:
            tmp = Path(tmpdir)
            workdir = tmp / "work"
            fake_decoder_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def __init__(self) -> None:
                    self.calls: list[str] = []

                def encode(self, text: str) -> list[int]:
                    self.calls.append(text)
                    return [11, 22]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return f"<tok:{ids[0]}>"

            tokenizer = FakeTokenizer()
            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": workdir / "decoder" / "libdecoder_v8.so",
                "c_path": workdir / "decoder" / "decoder_v8.c",
            }
            fake_logits = array("f", [0.1, 0.9, -0.4, 0.0])

            with mock.patch.object(bridge_runner_v8, "_ensure_engine_lib") as ensure_engine, \
                 mock.patch.object(bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": fake_logits}) as run_decoder, \
                 mock.patch.object(bridge_runner_v8, "_read_gguf_metadata", return_value={"general.architecture": "qwen3vl"}), \
                 mock.patch.object(bridge_runner_v8.GGUFTokenizer, "from_gguf", return_value=tokenizer):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = bridge_runner_v8.main(
                        [
                            "--decoder-gguf",
                            str(fake_decoder_gguf),
                            "--workdir",
                            str(workdir),
                            "--prompt",
                            "Describe the image.",
                            "--synthetic-prefix-tokens",
                            "3",
                            "--top-k",
                            "2",
                        ]
                    )

            self.assertEqual(rc, 0)
            ensure_engine.assert_called_once_with(openmp=False)
            self.assertEqual(
                tokenizer.calls,
                [
                    "<|im_start|>user\n<|vision_start|>",
                    "<|vision_end|>Describe the image.<|im_end|>\n<|im_start|>assistant\n",
                ],
            )
            _, decoder_kwargs = run_decoder.call_args
            self.assertEqual(decoder_kwargs["tokens_before"], [11, 22])
            report = json.loads((workdir / "bridge_report.json").read_text(encoding="utf-8"))
            self.assertEqual(
                report["formatted_prompt"],
                "<|im_start|>user\n<|vision_start|><image_embeds><|vision_end|>Describe the image.<|im_end|>\n<|im_start|>assistant\n",
            )
            self.assertEqual(report["chat_contract_name"], "qwen3vl")

    @classmethod
    def setUpClass(cls) -> None:
        result = subprocess.run(
            ["make", "build/libckernel_engine.so", "ck-cli-v8"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                "failed to build ck-cli-v8 prerequisites\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        if not CK_CLI_V8.exists():
            raise AssertionError(f"missing built cli: {CK_CLI_V8}")
        if not LIBCK.exists():
            raise AssertionError(f"missing engine library: {LIBCK}")

    def test_ck_cli_v8_runs_forward_mixed_with_prompt_tokens_and_prefix_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_cli_") as tmpdir:
            tmp = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(tmp)
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(array("f", [0.0] * (3 * 16)).tobytes())

            result = subprocess.run(
                [
                    str(CK_CLI_V8),
                    "--lib",
                    str(so_path),
                    "--weights",
                    str(bump_path),
                    "--manifest",
                    str(manifest_map),
                    "--prompt-tokens",
                    "1,2,3,4",
                    "--prefix-f32",
                    str(prefix_path),
                    "--max-tokens",
                    "1",
                    "--quiet-output",
                    "--no-timing",
                    "--verbose",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            combined = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, msg=combined)
            self.assertIn(
                "[DEBUG] Running ck_model_forward_mixed with prefix_tokens=3 embed_dim=16 prompt_tokens=4",
                combined,
            )
            self.assertNotIn("Model does not have built-in tokenizer", combined)

    def test_ck_cli_v8_first_token_matches_direct_forward_mixed_argmax(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_decoder_") as tmpdir:
            tmp = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(tmp)
            prefix = array("f", [0.0] * (3 * 16))
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(prefix.tobytes())
            token_ids = [1, 2, 3, 4]

            direct_forward = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import ctypes, sys;"
                        "build_lib, so_path, weights_path, manifest_path, prefix_path, prefix_tokens, token_csv = sys.argv[1:8];"
                        "ctypes.CDLL(build_lib, mode=ctypes.RTLD_GLOBAL);"
                        "lib = ctypes.CDLL(so_path);"
                        "lib.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p];"
                        "lib.ck_model_init_with_manifest.restype = ctypes.c_int;"
                        "lib.ck_model_forward_mixed.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_float)];"
                        "lib.ck_model_forward_mixed.restype = ctypes.c_int;"
                        "lib.ck_model_get_vocab_size.argtypes = [];"
                        "lib.ck_model_get_vocab_size.restype = ctypes.c_int;"
                        "lib.ck_model_free.argtypes = [];"
                        "lib.ck_model_free.restype = None;"
                        "rc = lib.ck_model_init_with_manifest(weights_path.encode(), manifest_path.encode());"
                        "assert rc == 0, rc;"
                        "blob = open(prefix_path, 'rb').read();"
                        "prefix = (ctypes.c_float * (len(blob) // 4)).from_buffer_copy(blob);"
                        "token_ids = [int(x) for x in token_csv.split(',') if x];"
                        "tokens = (ctypes.c_int32 * len(token_ids))(*token_ids);"
                        "vocab = lib.ck_model_get_vocab_size();"
                        "logits = (ctypes.c_float * vocab)();"
                        "rc = lib.ck_model_forward_mixed(prefix, int(prefix_tokens), tokens, len(token_ids), logits);"
                        "assert rc == 0, rc;"
                        "best = max(range(vocab), key=lambda idx: float(logits[idx]));"
                        "print(best);"
                        "lib.ck_model_free();"
                    ),
                    str(LIBCK),
                    str(so_path),
                    str(bump_path),
                    str(manifest_map),
                    str(prefix_path),
                    "3",
                    ",".join(str(tok) for tok in token_ids),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                direct_forward.returncode,
                0,
                msg=f"STDOUT:\n{direct_forward.stdout}\nSTDERR:\n{direct_forward.stderr}",
            )
            expected_token = int(direct_forward.stdout.strip())

            result = subprocess.run(
                [
                    str(CK_CLI_V8),
                    "--lib",
                    str(so_path),
                    "--weights",
                    str(bump_path),
                    "--manifest",
                    str(manifest_map),
                    "--prompt-tokens",
                    ",".join(str(tok) for tok in token_ids),
                    "--prefix-f32",
                    str(prefix_path),
                    "--max-tokens",
                    "1",
                    "--quiet-output",
                    "--no-timing",
                    "--verbose",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            combined = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, msg=combined)

            match = re.search(r"\[DEBUG\] Token 0: (\d+)", combined)
            self.assertIsNotNone(match, msg=combined)
            self.assertEqual(int(match.group(1)), expected_token)

    def test_bridge_runner_dump_prefix_writes_expected_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_runner_") as tmpdir:
            tmp = Path(tmpdir)
            workdir = tmp / "work"
            dump_path = tmp / "prefix.f32"
            fake_decoder_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    self.last_text = text
                    return [11, 22]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return f"<tok:{ids[0]}>"

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": workdir / "decoder" / "libdecoder_v8.so",
                "c_path": workdir / "decoder" / "decoder_v8.c",
            }
            fake_logits = array("f", [0.1, 0.9, -0.4, 0.0])

            with mock.patch.object(bridge_runner_v8, "_ensure_engine_lib") as ensure_engine, \
                 mock.patch.object(bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": fake_logits}), \
                 mock.patch.object(bridge_runner_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = bridge_runner_v8.main(
                        [
                            "--decoder-gguf",
                            str(fake_decoder_gguf),
                            "--workdir",
                            str(workdir),
                            "--prompt",
                            "Describe the image.",
                            "--synthetic-prefix-tokens",
                            "3",
                            "--dump-prefix-f32",
                            str(dump_path),
                            "--top-k",
                            "2",
                        ]
                    )

            self.assertEqual(rc, 0)
            ensure_engine.assert_called_once_with(openmp=False)
            self.assertTrue(dump_path.exists())
            self.assertEqual(dump_path.stat().st_size, 3 * 64 * 4)

            report = json.loads((workdir / "bridge_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["prefix_source"], "synthetic_zero")
            self.assertEqual(report["prefix_tokens"], 3)
            self.assertEqual(report["prefix_embed_dim"], 64)
            self.assertEqual(report["decoder_embed_dim"], 16)
            self.assertEqual(report["decoder_context_len"], 32)
            self.assertEqual(report["prompt_tokens"], [11, 22])
            self.assertEqual(report["prefix_dump_path"], str(dump_path.resolve()))

    def test_bridge_runner_auto_caps_decoder_context_to_prefix_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_ctx_") as tmpdir:
            tmp = Path(tmpdir)
            workdir = tmp / "work"
            fake_decoder_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    self.last_text = text
                    return [10, 20, 30, 40, 50]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return f"<tok:{ids[0]}>"

            fake_runtime = {
                "embed_dim": 16,
                "vocab_size": 4,
                "so_path": workdir / "decoder" / "libdecoder_v8.so",
                "c_path": workdir / "decoder" / "decoder_v8.c",
            }
            fake_logits = array("f", [0.1, 0.9, -0.4, 0.0])

            with mock.patch.object(bridge_runner_v8, "_ensure_engine_lib") as ensure_engine, \
                 mock.patch.object(bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime) as prepare_decoder, \
                 mock.patch.object(bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": fake_logits}), \
                 mock.patch.object(bridge_runner_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = bridge_runner_v8.main(
                        [
                            "--decoder-gguf",
                            str(fake_decoder_gguf),
                            "--workdir",
                            str(workdir),
                            "--prompt",
                            "Describe the image.",
                            "--synthetic-prefix-tokens",
                            "40",
                            "--top-k",
                            "2",
                        ]
                    )

            self.assertEqual(rc, 0)
            ensure_engine.assert_called_once_with(openmp=False)
            prepare_decoder.assert_called_once_with(
                fake_decoder_gguf.resolve(),
                workdir.resolve() / "decoder",
                context_override=61,
            )

            report = json.loads((workdir / "bridge_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["decoder_context_len"], 61)
            self.assertEqual(report["total_prefill_tokens"], 45)

    def test_bridge_runner_encoder_path_delays_dim_check_until_decoder_ready(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_encoder_") as tmpdir:
            tmp = Path(tmpdir)
            workdir = tmp / "work"
            fake_decoder_gguf = tmp / "decoder.gguf"
            fake_encoder_gguf = tmp / "encoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [101, 202]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return f"<tok:{ids[0]}>"

            fake_decoder_runtime = {
                "embed_dim": 16,
                "vocab_size": 4,
                "so_path": workdir / "decoder" / "libdecoder_v8.so",
                "c_path": workdir / "decoder" / "decoder_v8.c",
            }
            fake_encoder_runtime = {
                "embed_dim": 16,
                "so_path": workdir / "encoder" / "libencoder_v8.so",
            }
            fake_encoder_report = {
                "embed_dim": 16,
                "prefix_tokens": 3,
                "embeddings": array("f", [0.0] * (3 * 16)),
                "prefix_grid_x": 3,
                "prefix_grid_y": 1,
                "prefix_text_pos": 3,
                "bridge_activation": "vision_bridge_output",
                "bridge_reason": "projector_output",
                "image_source": "synthetic",
                "image_mode": "checker",
                "image_path": None,
                "source_image_size": [768, 768],
                "preprocess": "synthetic_generator",
                "image_size": 768,
                "image_height": 768,
                "image_width": 768,
                "interleaved_image": [0.0, 0.0, 0.0],
            }
            fake_logits = array("f", [0.1, 0.9, -0.4, 0.0])

            with mock.patch.object(bridge_runner_v8, "_ensure_engine_lib") as ensure_engine, \
                 mock.patch.object(bridge_runner_v8, "_prepare_encoder_runtime", return_value=fake_encoder_runtime), \
                 mock.patch.object(bridge_runner_v8, "_run_encoder", return_value=fake_encoder_report), \
                 mock.patch.object(bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_decoder_runtime), \
                 mock.patch.object(bridge_runner_v8, "_run_decoder", return_value={"vocab_size": 4, "logits": fake_logits}) as run_decoder, \
                 mock.patch.object(bridge_runner_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = bridge_runner_v8.main(
                        [
                            "--decoder-gguf",
                            str(fake_decoder_gguf),
                            "--encoder-gguf",
                            str(fake_encoder_gguf),
                            "--workdir",
                            str(workdir),
                            "--prompt",
                            "Describe the image.",
                            "--top-k",
                            "2",
                        ]
                    )

            self.assertEqual(rc, 0)
            ensure_engine.assert_called_once_with(openmp=True)
            _, decoder_kwargs = run_decoder.call_args
            self.assertEqual(decoder_kwargs["prefix_grid"], (3, 1))
            self.assertEqual(decoder_kwargs["prefix_text_pos"], 3)
            report = json.loads((workdir / "bridge_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["prefix_source"], "encoder")
            self.assertEqual(report["prefix_tokens"], 3)
            self.assertEqual(report["prefix_grid_x"], 3)
            self.assertEqual(report["prefix_grid_y"], 1)
            self.assertEqual(report["decoder_context_len"], 32)

    def test_run_decoder_uses_decode_runtime_for_mixed_prefix_continuation(self) -> None:
        used_paths: list[Path] = []

        class FakeLib:
            def __init__(self, path: Path) -> None:
                self.path = path

            def ck_model_init_with_manifest(self, weights: bytes, manifest: bytes) -> int:
                return 0

            def ck_model_get_vocab_size(self) -> int:
                return 4

            def ck_model_forward_mixed(self, prefix_ptr, prefix_tokens: int, token_arr, token_count: int, logits) -> int:
                logits[0] = 0.1
                logits[1] = 0.2
                logits[2] = 0.3
                logits[3] = 0.4
                return 0

            def ck_model_free(self) -> None:
                return None

        def fake_loader(path: Path) -> FakeLib:
            resolved = Path(path)
            used_paths.append(resolved)
            return FakeLib(resolved)

        runtime = {
            "so_path": ROOT / "decode_runtime.so",
            "prefill_so_path": ROOT / "prefill_runtime.so",
            "weights_bump": ROOT / "weights.bump",
            "manifest_map": ROOT / "weights_manifest.map",
            "vocab_size": 4,
        }

        with mock.patch.object(bridge_runner_v8, "_load_decoder_lib", side_effect=fake_loader):
            result = bridge_runner_v8._run_decoder(
                runtime,
                array("f", [0.0] * 16),
                1,
                [1, 2],
            )

        self.assertEqual(used_paths, [ROOT / "decode_runtime.so"])
        self.assertEqual(result["runtime_mode"], "decode")
        self.assertEqual(result["vocab_size"], 4)

    def test_run_decoder_generation_stops_on_resolved_stop_token(self) -> None:
        class FakeTokenizer:
            def decode(self, ids: list[int], skip_special: bool = True) -> str:
                mapping = {
                    1: "This",
                    2: " image",
                    9: "<|im_end|>",
                }
                text = "".join(mapping.get(int(token_id), f"<tok:{int(token_id)}>") for token_id in ids)
                return text.replace("<|im_end|>", "") if skip_special else text

        class FakeLib:
            def __init__(self) -> None:
                self.decode_calls = 0

            def ck_model_init_with_manifest(self, weights: bytes, manifest: bytes) -> int:
                return 0

            def ck_model_get_vocab_size(self) -> int:
                return 10

            def ck_model_forward_mixed(self, prefix_ptr, prefix_tokens: int, token_arr, token_count: int, logits) -> int:
                for idx in range(10):
                    logits[idx] = 0.0
                logits[1] = 5.0
                return 0

            def ck_model_decode(self, token: int, logits) -> int:
                for idx in range(10):
                    logits[idx] = 0.0
                if self.decode_calls == 0:
                    logits[2] = 7.0
                else:
                    logits[9] = 11.0
                self.decode_calls += 1
                return 0

            def ck_model_free(self) -> None:
                return None

        runtime = {
            "so_path": ROOT / "decode_runtime.so",
            "weights_bump": ROOT / "weights.bump",
            "manifest_map": ROOT / "weights_manifest.map",
            "vocab_size": 10,
        }

        with mock.patch.object(bridge_runner_v8, "_load_decoder_lib", return_value=FakeLib()):
            result = bridge_runner_v8._run_decoder(
                runtime,
                array("f", [0.0] * 16),
                1,
                [1, 2],
                tokenizer=FakeTokenizer(),
                stop_token_ids=[9],
                max_tokens=8,
            )

        self.assertEqual(result["generated_token_ids"], [1, 2])
        self.assertEqual(result["generated_text"], "This image")
        self.assertEqual(result["generated_text_raw"], "This image")
        self.assertEqual(result["generation_stop_reason"], "stop_token")


if __name__ == "__main__":
    unittest.main()
