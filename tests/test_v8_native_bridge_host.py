#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
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
V8_RUNNER_PATH = ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"


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
ck_run_v8 = _load_module("ck_run_v8_host_tests", V8_RUNNER_PATH)
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

    vocab_tokens = [f"<|tok_{idx}|>" for idx in range(64)]
    vocab_tokens[0] = "<|pad|>"
    vocab_tokens[1] = "Hello"
    vocab_tokens[2] = " world"
    vocab_tokens[9] = "<|endoftext|>"

    vocab_offsets: list[int] = []
    vocab_strings = bytearray()
    for token in vocab_tokens:
        vocab_offsets.append(len(vocab_strings))
        vocab_strings.extend(token.encode("utf-8"))
        vocab_strings.append(0)

    add("vocab_offsets", "i32", [len(vocab_offsets)])
    add("vocab_strings", "u8", [len(vocab_strings)])

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
        "test_vocab_offsets": vocab_offsets,
        "test_vocab_strings": list(vocab_strings),
    }


def _write_zero_bump(manifest: dict, bump_path: Path) -> None:
    total_size = 0
    for entry in manifest["entries"]:
        end = int(entry["offset"]) + int(entry["size"])
        total_size = max(total_size, end)
    total_size = max(total_size, 8)
    blob = bytearray(b"BUMPWGT5" + (b"\0" * (total_size - 8)))

    vocab_offsets = manifest.get("test_vocab_offsets")
    vocab_strings = manifest.get("test_vocab_strings")
    if vocab_offsets is not None and vocab_strings is not None:
        for entry in manifest["entries"]:
            name = entry.get("name")
            offset = int(entry["offset"])
            if name == "vocab_offsets":
                payload = array("i", vocab_offsets).tobytes()
                blob[offset : offset + len(payload)] = payload
            elif name == "vocab_strings":
                payload = bytes(vocab_strings)
                blob[offset : offset + len(payload)] = payload

    bump_path.write_bytes(bytes(blob))


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
    def test_ck_run_v8_step_run_chat_uses_detected_default_threads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_step_run_chat_threads_") as tmpdir:
            tmp = Path(tmpdir)
            args = argparse.Namespace(
                temperature=0.7,
                max_tokens=16,
                top_k=40,
                top_p=1.0,
                min_p=0.0,
                repeat_penalty=1.0,
                repeat_last_n=64,
                prompt="Hello",
                no_chat_template=False,
                chat_template="auto",
                allow_raw_prompt=False,
                thinking_mode="suppressed",
                python_tokenizer=False,
                memory=False,
            )

            with mock.patch.object(ck_run_v8, "_sync_runtime_lib"), \
                 mock.patch.object(ck_run_v8, "_detect_default_ck_threads", return_value=12), \
                 mock.patch.dict(ck_run_v8.os.environ, {}, clear=True), \
                 mock.patch.object(ck_run_v8.os, "execvpe", side_effect=SystemExit) as execvpe:
                with self.assertRaises(SystemExit):
                    ck_run_v8.step_run_chat(tmp, args, gguf_path=None)

        env = execvpe.call_args.args[2]
        self.assertEqual(env["CK_NUM_THREADS"], "12")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["OMP_DYNAMIC"], "FALSE")

    def test_ck_run_v8_step_compile_prefers_icx_when_available(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_step_compile_icx_") as tmpdir:
            tmp = Path(tmpdir)
            build_dir = tmp / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "libckernel_engine.so").write_bytes(b"so")
            (build_dir / "libckernel_tokenizer.so").write_bytes(b"so")
            model_c = tmp / "model_v8.c"
            model_c.write_text("/* generated */", encoding="utf-8")
            out_dir = tmp / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            calls: list[list[str]] = []

            def fake_run_cmd(cmd: list[str], *, cwd=None, env=None, capture=False):
                calls.append([str(part) for part in cmd])
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"so")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            def fake_sync_runtime_lib(*args, **kwargs):
                return None

            def fake_which(binary: str) -> str | None:
                if binary == "icx":
                    return "/opt/intel/oneapi/compiler/latest/bin/icx"
                if binary == "gcc":
                    return "/usr/bin/gcc"
                return None

            with mock.patch.object(ck_run_v8, "BUILD_DIR", build_dir), \
                 mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd), \
                 mock.patch.object(ck_run_v8, "_sync_runtime_lib", side_effect=fake_sync_runtime_lib), \
                 mock.patch.object(ck_run_v8.shutil, "which", side_effect=fake_which), \
                 mock.patch.dict(ck_run_v8.os.environ, {}, clear=True):
                lib_path = ck_run_v8.step_compile(model_c, out_dir, force=False)

        self.assertEqual(lib_path, out_dir / "libmodel.so")
        self.assertEqual(len(calls), 1)
        cmd = calls[0]
        self.assertEqual(cmd[0], "icx")
        self.assertIn("-qopenmp", cmd)

    def test_ck_run_v8_parser_accepts_visualizer_and_llama_template(self) -> None:
        with mock.patch.object(ck_run_v8, "_ensure_v8_python_requirements"), \
             mock.patch.object(ck_run_v8, "run_pipeline", return_value=0) as run_pipeline:
            rc = ck_run_v8.main(
                [
                    "run",
                    "/tmp/fake.gguf",
                    "--chat-template",
                    "llama",
                    "--generate-visualizer",
                ]
            )

        self.assertEqual(rc, 0)
        args = run_pipeline.call_args.args[0]
        self.assertEqual(args.chat_template, "llama")
        self.assertTrue(args.generate_visualizer)

    def test_ck_run_v8_parser_accepts_canonical_v7_surface_examples(self) -> None:
        commands = [
            [
                "run",
                "hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf",
                "--context-len",
                "1024",
                "--force-compile",
                "--force-convert",
                "--chat-template",
                "none",
                "--generate-visualizer",
            ],
            [
                "run",
                "hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf",
                "--context-len",
                "1024",
                "--force-compile",
                "--force-convert",
                "--generate-visualizer",
            ],
            [
                "run",
                "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
                "--context-len",
                "1024",
                "--force-compile",
                "--force-convert",
                "--generate-visualizer",
            ],
            [
                "run",
                "hf://unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
                "--force-convert",
                "--force-compile",
                "--context-len",
                "1034",
            ],
            [
                "run",
                "hf://mradermacher/Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf",
                "--context-len",
                "1024",
                "--force-compile",
                "--force-convert",
                "--chat-template",
                "auto",
                "--generate-visualizer",
            ],
        ]

        with mock.patch.object(ck_run_v8, "_ensure_v8_python_requirements"), \
             mock.patch.object(ck_run_v8, "run_pipeline", return_value=0) as run_pipeline:
            for argv in commands:
                rc = ck_run_v8.main(argv)
                self.assertEqual(rc, 0, msg=f"argv not accepted: {argv}")

        self.assertEqual(run_pipeline.call_count, len(commands))
        gemma_args = run_pipeline.call_args_list[0].args[0]
        qwen35_args = run_pipeline.call_args_list[3].args[0]
        nanbeige_args = run_pipeline.call_args_list[4].args[0]
        self.assertEqual(gemma_args.chat_template, "none")
        self.assertTrue(gemma_args.generate_visualizer)
        self.assertEqual(qwen35_args.context_len, 1034)
        self.assertEqual(nanbeige_args.chat_template, "auto")
        self.assertTrue(nanbeige_args.generate_visualizer)

    def test_ck_run_v8_reexecs_into_repo_venv_when_direct_python_lacks_deps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_reexec_venv_") as tmpdir:
            tmp = Path(tmpdir)
            fake_venv = tmp / "python"
            fake_venv.write_text("", encoding="utf-8")

            with mock.patch.object(ck_run_v8, "REPO_VENV_PY", fake_venv), \
                 mock.patch.object(ck_run_v8, "_missing_python_packages", return_value=["torch"]), \
                 mock.patch.object(ck_run_v8.os, "execve", side_effect=SystemExit(0)) as execve, \
                 mock.patch.dict(ck_run_v8.os.environ, {}, clear=True), \
                 mock.patch.object(ck_run_v8.sys, "argv", ["ck_run_v8.py", "run", "model.gguf"]), \
                 mock.patch.object(ck_run_v8.sys, "executable", "/usr/bin/python3"):
                with self.assertRaises(SystemExit) as ctx:
                    ck_run_v8._ensure_v8_python_requirements("run")

        self.assertEqual(ctx.exception.code, 0)
        execve.assert_called_once()
        self.assertEqual(execve.call_args.args[0], str(fake_venv))
        self.assertEqual(execve.call_args.args[1], [str(fake_venv), ck_run_v8.__file__, "run", "model.gguf"])

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

    def test_bridge_runner_prints_concise_generated_text_by_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_stdout_") as tmpdir:
            tmp = Path(tmpdir)
            workdir = tmp / "work"
            fake_decoder_gguf = tmp / "decoder.gguf"

            class FakeTokenizer:
                def encode(self, text: str) -> list[int]:
                    return [11, 22]

                def decode(self, ids: list[int], skip_special: bool = False) -> str:
                    return "This image shows a logo."

            fake_runtime = {
                "embed_dim": 16,
                "input_embed_dim": 64,
                "vocab_size": 4,
                "so_path": workdir / "decoder" / "libdecoder_v8.so",
                "c_path": workdir / "decoder" / "decoder_v8.c",
            }
            fake_logits = array("f", [0.1, 0.9, -0.4, 0.0])
            fake_decoder_report = {
                "vocab_size": 4,
                "logits": fake_logits,
                "generated_token_ids": [1, 2, 3],
                "generated_text": "This image shows a logo.",
                "generated_text_raw": "This image shows a logo.",
                "generation_stop_reason": "stop_token",
            }

            stdout = io.StringIO()
            with mock.patch.object(bridge_runner_v8, "_ensure_engine_lib"), \
                 mock.patch.object(bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
                 mock.patch.object(bridge_runner_v8, "_run_decoder", return_value=fake_decoder_report), \
                 mock.patch.object(bridge_runner_v8, "_read_gguf_metadata", return_value={"general.architecture": "qwen3vl"}), \
                 mock.patch.object(bridge_runner_v8.GGUFTokenizer, "from_gguf", return_value=FakeTokenizer()):
                with contextlib.redirect_stdout(stdout):
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
                            "--max-tokens",
                            "8",
                        ]
                    )

            self.assertEqual(rc, 0)
            rendered = stdout.getvalue()
            self.assertIn("This image shows a logo.", rendered)
            self.assertNotIn('"generated_text"', rendered)

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

    def test_ck_cli_v8_replays_bridge_report_through_native_multimodal_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_report_cli_") as tmpdir:
            tmp = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(tmp)
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(array("f", [0.0] * (3 * 16)).tobytes())
            report_path = tmp / "bridge_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "prefix_dump_path": str(prefix_path),
                        "prefix_embed_dim": 16,
                        "prompt_tokens_before_image": [],
                        "prompt_tokens_after_image": [1, 2, 3, 4],
                        "stop_token_ids": [],
                        "prefix_grid_x": None,
                        "prefix_grid_y": None,
                        "prefix_text_pos": None,
                        "multimodal_prompt_segmented": False,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(CK_CLI_V8),
                    "--lib",
                    str(so_path),
                    "--weights",
                    str(bump_path),
                    "--manifest",
                    str(manifest_map),
                    "--bridge-report",
                    str(report_path),
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
            self.assertRegex(
                combined,
                r"\[DEBUG\] Running ck_model_forward_mixed(?:_ex)? with prefix_tokens=3 embed_dim=16 prompt_tokens=4",
            )
            self.assertNotIn("Model does not have built-in tokenizer", combined)

    def test_ck_cli_v8_bridge_report_decodes_output_from_vocab_tables_without_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_report_vocab_cli_") as tmpdir:
            tmp = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(tmp)
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(array("f", [0.0] * (3 * 16)).tobytes())
            report_path = tmp / "bridge_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "prefix_dump_path": str(prefix_path),
                        "prefix_embed_dim": 16,
                        "prompt_tokens_before_image": [],
                        "prompt_tokens_after_image": [1, 2, 3, 4],
                        "stop_token_ids": [9],
                        "prefix_grid_x": None,
                        "prefix_grid_y": None,
                        "prefix_text_pos": None,
                        "multimodal_prompt_segmented": False,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(CK_CLI_V8),
                    "--lib",
                    str(so_path),
                    "--weights",
                    str(bump_path),
                    "--manifest",
                    str(manifest_map),
                    "--bridge-report",
                    str(report_path),
                    "--max-tokens",
                    "2",
                    "--temperature",
                    "0",
                    "--no-timing",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )

            combined = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, msg=combined)
            self.assertIn("<|pad|>", result.stdout)

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

    def test_run_decoder_generation_enforces_min_response_tokens_before_stop(self) -> None:
        class FakeTokenizer:
            def decode(self, ids: list[int], skip_special: bool = True) -> str:
                mapping = {
                    1: "This",
                    9: "<|im_end|>",
                }
                text = "".join(mapping.get(int(token_id), f"<tok:{int(token_id)}>") for token_id in ids)
                return text.replace("<|im_end|>", "") if skip_special else text

        class FakeLib:
            def ck_model_init_with_manifest(self, weights: bytes, manifest: bytes) -> int:
                return 0

            def ck_model_get_vocab_size(self) -> int:
                return 10

            def ck_model_forward_mixed(self, prefix_ptr, prefix_tokens: int, token_arr, token_count: int, logits) -> int:
                for idx in range(10):
                    logits[idx] = 0.0
                logits[9] = 10.0
                logits[1] = 9.0
                return 0

            def ck_model_decode(self, token: int, logits) -> int:
                for idx in range(10):
                    logits[idx] = 0.0
                logits[9] = 11.0
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
                max_tokens=4,
                min_response_tokens=1,
            )

        self.assertEqual(result["generated_token_ids"], [1])
        self.assertEqual(result["generated_text"], "This")
        self.assertEqual(result["generation_stop_reason"], "stop_token")

    def test_run_decoder_generation_repeat_penalty_breaks_greedy_loop(self) -> None:
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
                logits[1] = 10.0
                logits[2] = 9.0
                return 0

            def ck_model_decode(self, token: int, logits) -> int:
                for idx in range(10):
                    logits[idx] = 0.0
                if self.decode_calls == 0:
                    logits[1] = 10.0
                    logits[2] = 9.0
                    logits[9] = 8.0
                else:
                    logits[9] = 12.0
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
                repeat_penalty=2.0,
                repeat_last_n=8,
            )

        self.assertEqual(result["generated_token_ids"], [1, 2])
        self.assertEqual(result["generated_text"], "This image")
        self.assertEqual(result["generation_stop_reason"], "stop_token")

    def test_run_decoder_streams_text_and_suppresses_progress_by_default(self) -> None:
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

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(bridge_runner_v8, "_load_decoder_lib", return_value=FakeLib()):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = bridge_runner_v8._run_decoder(
                    runtime,
                    array("f", [0.0] * 16),
                    1,
                    [1, 2],
                    tokenizer=FakeTokenizer(),
                    stop_token_ids=[9],
                    max_tokens=8,
                    stream_output=True,
                    generation_progress_every=0,
                )

        self.assertEqual(result["generated_token_ids"], [1, 2])
        self.assertTrue(result["streamed_output"])
        self.assertIn("This image", stdout.getvalue())
        self.assertNotIn("generation progress", stderr.getvalue())

    def test_run_decoder_clamps_generation_to_remaining_context_budget(self) -> None:
        class FakeTokenizer:
            def decode(self, ids: list[int], skip_special: bool = True) -> str:
                return "".join("A" for _ in ids)

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
                logits[1] = 5.0
                self.decode_calls += 1
                return 0

            def ck_model_free(self) -> None:
                return None

        runtime = {
            "so_path": ROOT / "decode_runtime.so",
            "weights_bump": ROOT / "weights.bump",
            "manifest_map": ROOT / "weights_manifest.map",
            "vocab_size": 10,
            "context_length": 8,
        }

        stderr = io.StringIO()
        fake_lib = FakeLib()
        with mock.patch.object(bridge_runner_v8, "_load_decoder_lib", return_value=fake_lib):
            with contextlib.redirect_stderr(stderr):
                result = bridge_runner_v8._run_decoder(
                    runtime,
                    array("f", [0.0] * 16),
                    1,
                    [1, 2],
                    tokens_before=[10, 20],
                    tokenizer=FakeTokenizer(),
                    max_tokens=99,
                    stream_output=False,
                )

        self.assertEqual(len(result["generated_token_ids"]), 3)
        self.assertIn("decoder: generation clamp requested=99 effective=3 context=8 prefill_tokens=5", stderr.getvalue())
        self.assertEqual(fake_lib.decode_calls, 2)

    def test_ck_run_v8_multimodal_forwards_sampling_controls(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_multimodal_sampling_") as tmpdir:
            tmp = Path(tmpdir)
            decoder = tmp / "decoder.gguf"
            encoder = tmp / "encoder.gguf"
            image = tmp / "image.png"
            decoder.write_bytes(b"gguf")
            encoder.write_bytes(b"gguf")
            image.write_bytes(b"png")

            args = argparse.Namespace(
                model=str(decoder),
                run_dir=str(tmp / "run"),
                mmproj=str(encoder),
                image_path=str(image),
                synthetic_prefix_tokens=0,
                force_download=False,
                prompt="Explain this image.",
                image_mode="checker",
                vision_top_k=7,
                vision_activation_pref=["out_proj=q8", "mlp_down=q8"],
                max_tokens=32,
                no_chat_template=False,
                chat_template="auto",
                allow_raw_prompt=False,
                thinking_mode="suppressed",
                context_len=1024,
                temperature=0.7,
                top_k=15,
                top_p=0.92,
                min_p=0.05,
                repeat_penalty=1.15,
                repeat_last_n=48,
                force_convert=False,
                force_compile=False,
                memory=False,
                python_tokenizer=False,
                generate_only=False,
            )

            captured_cmds: list[list[str]] = []

            def fake_run_cmd(cmd: list[str], **kwargs):
                captured_cmds.append([str(part) for part in cmd])
                return None

            with mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd):
                rc = ck_run_v8.run_pipeline(args)

        self.assertEqual(rc, 0)
        self.assertEqual(len(captured_cmds), 1)
        cmd = captured_cmds[0]
        self.assertIn("--report-top-k", cmd)
        self.assertIn("7", cmd)
        self.assertIn("--temperature", cmd)
        self.assertIn("0.7", cmd)
        self.assertIn("--sample-top-k", cmd)
        self.assertIn("15", cmd)
        self.assertIn("--top-p", cmd)
        self.assertIn("0.92", cmd)
        self.assertIn("--min-p", cmd)
        self.assertIn("0.05", cmd)
        self.assertIn("--repeat-penalty", cmd)
        self.assertIn("1.15", cmd)
        self.assertIn("--repeat-last-n", cmd)
        self.assertIn("48", cmd)
        self.assertEqual(cmd.count("--vision-activation-pref"), 2)
        self.assertIn("out_proj=q8", cmd)
        self.assertIn("mlp_down=q8", cmd)

    def test_ck_run_v8_generate_visualizer_refreshes_report_for_text_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_generate_visualizer_") as tmpdir:
            tmp = Path(tmpdir)
            gguf = tmp / "model.gguf"
            gguf.write_bytes(b"gguf")

            work_dir = tmp / "run"
            weights_path = work_dir / "weights.bump"
            config_path = work_dir / "config.json"
            manifest_path = work_dir / "weights_manifest.json"
            decode_layout = work_dir / "layout_decode.json"
            model_c_path = work_dir / "model_v8.c"
            lib_path = work_dir / "libmodel.so"

            args = argparse.Namespace(
                model=str(gguf),
                run_dir=str(work_dir),
                mmproj=None,
                image_path=None,
                synthetic_prefix_tokens=0,
                force_download=False,
                prompt=None,
                image_mode="checker",
                vision_top_k=8,
                max_tokens=16,
                no_chat_template=False,
                chat_template="auto",
                allow_raw_prompt=False,
                thinking_mode="auto",
                context_len=1024,
                temperature=0.7,
                top_k=40,
                top_p=1.0,
                min_p=0.0,
                repeat_penalty=1.0,
                repeat_last_n=64,
                force_convert=False,
                force_compile=False,
                memory=False,
                python_tokenizer=False,
                generate_visualizer=True,
                generate_only=True,
                logits_layout="auto",
            )

            with mock.patch.object(ck_run_v8, "step_convert_gguf", return_value=(weights_path, config_path, manifest_path)), \
                 mock.patch.object(
                     ck_run_v8,
                     "step_build_ir",
                     return_value={
                         "prefill_ir": work_dir / "ir1_prefill.json",
                         "prefill_layout": work_dir / "layout_prefill.json",
                         "prefill_lowered": work_dir / "lowered_prefill.json",
                         "prefill_call": work_dir / "lowered_prefill_call.json",
                         "decode_ir": work_dir / "ir1_decode.json",
                         "decode_layout": decode_layout,
                         "decode_lowered": work_dir / "lowered_decode.json",
                         "decode_call": work_dir / "lowered_decode_call.json",
                         "manifest_map": work_dir / "weights_manifest.map",
                     },
                 ), \
                 mock.patch.object(ck_run_v8, "step_codegen", return_value=model_c_path), \
                 mock.patch.object(ck_run_v8, "step_compile", return_value=lib_path), \
                 mock.patch.object(ck_run_v8, "_generate_visualizer_html", return_value=work_dir / "ir_report.html") as gen_viz:
                rc = ck_run_v8.run_pipeline(args)

        self.assertEqual(rc, 0)
        gen_viz.assert_called_once_with(work_dir)

    def test_ck_run_v8_step_build_ir_requests_init_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_build_ir_init_") as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "weights_manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            output_dir = tmp / "run"
            calls: list[list[str]] = []

            def fake_run_cmd(cmd: list[str], *, cwd=None, env=None, capture=False):
                calls.append(list(cmd))
                if "--init-output" in cmd:
                    init_ir = Path(cmd[cmd.index("--init-output") + 1])
                    init_ir.parent.mkdir(parents=True, exist_ok=True)
                    init_ir.write_text("{}", encoding="utf-8")
                    (init_ir.parent / "init_call.json").write_text("{}", encoding="utf-8")
                Path(cmd[cmd.index("--output") + 1]).write_text("{}", encoding="utf-8")
                Path(cmd[cmd.index("--layout-output") + 1]).write_text("{}", encoding="utf-8")
                Path(cmd[cmd.index("--lowered-output") + 1]).write_text("{}", encoding="utf-8")
                Path(cmd[cmd.index("--call-output") + 1]).write_text("{}", encoding="utf-8")
                if "--manifest-map-output" in cmd:
                    Path(cmd[cmd.index("--manifest-map-output") + 1]).write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd), \
                 mock.patch.object(ck_run_v8, "step_regenerate_kernel_registry", return_value=ck_run_v8.KERNEL_REGISTRY_PATH):
                outputs = ck_run_v8.step_build_ir(manifest_path, output_dir, force=True, context_len=1024)

        self.assertEqual(len(calls), 2)
        decode_cmd = calls[1]
        self.assertIn("--init-output", decode_cmd)
        self.assertEqual(outputs["init_ir"].name, "init.json")
        self.assertEqual(outputs["init_call"].name, "init_call.json")

    def test_ck_run_v8_step_codegen_passes_prefill_layout_for_hybrid_runtime(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_codegen_prefill_layout_") as tmpdir:
            tmp = Path(tmpdir)
            output_dir = tmp / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            ir_paths = {
                "decode_call": output_dir / "lowered_decode_call.json",
                "prefill_call": output_dir / "lowered_prefill_call.json",
                "decode_layout": output_dir / "layout_decode.json",
                "prefill_layout": output_dir / "layout_prefill.json",
            }
            for path in ir_paths.values():
                path.write_text("{}", encoding="utf-8")

            calls: list[list[str]] = []

            def fake_run_cmd(cmd: list[str], *, cwd=None, env=None, capture=False):
                calls.append(list(cmd))
                Path(cmd[cmd.index("--output") + 1]).write_text("/* generated */", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd):
                model_c = ck_run_v8.step_codegen(output_dir, ir_paths, force=True)

        self.assertEqual(model_c, output_dir / "model_v8.c")
        self.assertEqual(len(calls), 1)
        cmd = calls[0]
        self.assertIn("--prefill-layout", cmd)
        self.assertEqual(Path(cmd[cmd.index("--prefill-layout") + 1]), ir_paths["prefill_layout"])

    def test_ck_run_v8_step_convert_preserves_model_native_context_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_convert_native_ctx_") as tmpdir:
            tmp = Path(tmpdir)
            gguf_path = tmp / "model.gguf"
            gguf_path.write_bytes(b"gguf")
            output_dir = tmp / "run"
            calls: list[list[str]] = []

            def fake_run_cmd(cmd: list[str], *, cwd=None, env=None, capture=False):
                calls.append(list(cmd))
                Path(cmd[cmd.index("--output") + 1]).write_bytes(b"bump")
                Path(cmd[cmd.index("--config-out") + 1]).write_text("{}", encoding="utf-8")
                Path(cmd[cmd.index("--manifest-out") + 1]).write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd):
                ck_run_v8.step_convert_gguf(gguf_path, output_dir, force=True, context_len=1024)

        self.assertEqual(len(calls), 1)
        self.assertNotIn("--context", calls[0])

    def test_bridge_converter_preserves_model_native_context_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_bridge_convert_native_ctx_") as tmpdir:
            tmp = Path(tmpdir)
            gguf_path = tmp / "decoder.gguf"
            gguf_path.write_bytes(b"gguf")
            output_dir = tmp / "work"

            captured_argv: list[str] = []

            def fake_main() -> int:
                captured_argv[:] = list(sys.argv)
                manifest_path = Path(sys.argv[sys.argv.index("--manifest-out") + 1])
                config_path = Path(sys.argv[sys.argv.index("--config-out") + 1])
                bump_path = Path(sys.argv[sys.argv.index("--output") + 1])
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                bump_path.write_bytes(b"BUMPWGT5")
                payload = {"config": {"model": "qwen35", "context_length": 262144, "max_seq_len": 262144}}
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                config_path.write_text(json.dumps(payload["config"]), encoding="utf-8")
                return 0

            with mock.patch.object(bridge_runner_v8.convert_gguf_to_bump_v8, "main", side_effect=fake_main):
                bridge_runner_v8._run_converter(gguf_path, output_dir, context_override=1024)

        self.assertNotIn("--context", captured_argv)

    def test_bridge_run_converter_reuses_cached_outputs_with_matching_stamp(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_bridge_convert_reuse_") as tmpdir:
            tmp = Path(tmpdir)
            gguf_path = tmp / "decoder.gguf"
            gguf_path.write_bytes(b"gguf")
            output_dir = tmp / "work"
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {"config": {"model": "qwen3vl", "context_length": 262144}}
            manifest_path = output_dir / "weights_manifest.json"
            config_path = output_dir / "config.json"
            bump_path = output_dir / "weights.bump"
            stamp_path = output_dir / "convert.cache.json"

            bridge_runner_v8._json_write(manifest_path, manifest)
            bridge_runner_v8._json_write(config_path, manifest["config"])
            bump_path.write_bytes(b"BUMPWGT5")
            bridge_runner_v8._json_write(stamp_path, bridge_runner_v8._converter_fingerprint(gguf_path))

            with mock.patch.object(bridge_runner_v8.convert_gguf_to_bump_v8, "main") as convert_main:
                resolved_manifest, resolved_manifest_path, resolved_bump_path, resolved_config_path = bridge_runner_v8._run_converter(
                    gguf_path,
                    output_dir,
                )

        convert_main.assert_not_called()
        self.assertEqual(resolved_manifest, manifest)
        self.assertEqual(resolved_manifest_path, manifest_path)
        self.assertEqual(resolved_bump_path, bump_path)
        self.assertEqual(resolved_config_path, config_path)

    def test_bridge_prepare_encoder_runtime_reuses_cached_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_bridge_encoder_reuse_") as tmpdir:
            tmp = Path(tmpdir)
            gguf_path = tmp / "encoder.gguf"
            gguf_path.write_bytes(b"gguf")
            output_dir = tmp / "encoder"
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "config": {
                    "model": "vision_stub",
                    "arch": "vision_stub",
                    "embed_dim": 1536,
                    "image_size": 96,
                    "image_height": 96,
                    "image_width": 96,
                    "spatial_merge_size": 2,
                }
            }
            manifest_path = output_dir / "weights_manifest.json"
            config_path = output_dir / "config.json"
            bump_path = output_dir / "weights.bump"
            layout_path = output_dir / "layout.json"
            lowered_path = output_dir / "lowered.json"
            call_path = output_dir / "call.json"
            ir1_path = output_dir / "ir1.json"
            c_path = output_dir / "encoder_v8.c"
            so_path = output_dir / "libencoder_v8.so"
            stamp_path = output_dir / "encoder_runtime.cache.json"

            bridge_runner_v8._json_write(manifest_path, manifest)
            bridge_runner_v8._json_write(config_path, manifest["config"])
            bump_path.write_bytes(b"BUMPWGT5")
            bridge_runner_v8._json_write(layout_path, {"config": {"embed_dim": 1536}})
            bridge_runner_v8._json_write(lowered_path, {})
            bridge_runner_v8._json_write(call_path, {})
            bridge_runner_v8._json_write(ir1_path, {})
            c_path.write_text("/* generated */", encoding="utf-8")
            bridge_runner_v8._json_write(
                stamp_path,
                bridge_runner_v8._runtime_fingerprint(
                    manifest_path=manifest_path,
                    mode="encoder_prefill",
                ),
            )

            def fake_run_converter(path: Path, out_dir: Path, context_override: int | None = None):
                return manifest, manifest_path, bump_path, config_path

            def fake_compile_generated_model(c_src: Path, so_dst: Path) -> Path:
                so_dst.write_bytes(b"so")
                return so_dst

            with mock.patch.object(bridge_runner_v8, "_run_converter", side_effect=fake_run_converter), \
                 mock.patch.object(bridge_runner_v8.build_ir_v8, "main") as build_ir_main, \
                 mock.patch.object(bridge_runner_v8.codegen_v8, "main") as codegen_main, \
                 mock.patch.object(bridge_runner_v8, "_compile_generated_model", side_effect=fake_compile_generated_model) as compile_model:
                runtime = bridge_runner_v8._prepare_encoder_runtime(gguf_path, output_dir)

        build_ir_main.assert_not_called()
        codegen_main.assert_not_called()
        compile_model.assert_called_once_with(c_path, so_path)
        self.assertEqual(runtime["embed_dim"], 1536)
        self.assertEqual(runtime["so_path"], so_path)

    def test_bridge_prepare_encoder_runtime_applies_activation_overrides(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_bridge_encoder_overrides_") as tmpdir:
            tmp = Path(tmpdir)
            gguf_path = tmp / "encoder.gguf"
            gguf_path.write_bytes(b"gguf")
            output_dir = tmp / "encoder"

            manifest = {
                "config": {
                    "model": "qwen3_vl_vision",
                    "arch": "qwen3_vl_vision",
                    "embed_dim": 1536,
                    "image_size": 96,
                    "image_height": 96,
                    "image_width": 96,
                    "spatial_merge_size": 2,
                    "patch_size": 14,
                    "activation_preference_by_op": {"branch_fc1": "fp32"},
                }
            }
            manifest_path = output_dir / "weights_manifest.json"
            config_path = output_dir / "config.json"
            bump_path = output_dir / "weights.bump"
            layout_path = output_dir / "layout.json"
            lowered_path = output_dir / "lowered.json"
            call_path = output_dir / "call.json"
            ir1_path = output_dir / "ir1.json"
            manifest_map = output_dir / "weights_manifest.map"
            c_path = output_dir / "encoder_v8.c"
            so_path = output_dir / "libencoder_v8.so"

            def fake_run_converter(path: Path, out_dir: Path, context_override: int | None = None):
                out_dir.mkdir(parents=True, exist_ok=True)
                bridge_runner_v8._json_write(manifest_path, manifest)
                bridge_runner_v8._json_write(config_path, manifest["config"])
                bump_path.write_bytes(b"BUMPWGT5")
                return json.loads(json.dumps(manifest)), manifest_path, bump_path, config_path

            def fake_build_ir(argv: list[str]) -> int:
                bridge_runner_v8._json_write(layout_path, {"config": {"embed_dim": 1536}})
                bridge_runner_v8._json_write(lowered_path, {})
                bridge_runner_v8._json_write(call_path, {})
                bridge_runner_v8._json_write(ir1_path, {})
                manifest_map.write_text("{}", encoding="utf-8")
                return 0

            def fake_codegen_main() -> int:
                c_path.write_text("/* generated */", encoding="utf-8")
                return 0

            def fake_compile_generated_model(c_src: Path, so_dst: Path) -> Path:
                so_dst.write_bytes(b"so")
                return so_dst

            with mock.patch.object(bridge_runner_v8, "_run_converter", side_effect=fake_run_converter), \
                 mock.patch.object(bridge_runner_v8.build_ir_v8, "main", side_effect=fake_build_ir), \
                 mock.patch.object(bridge_runner_v8.codegen_v8, "main", side_effect=fake_codegen_main), \
                 mock.patch.object(bridge_runner_v8, "_compile_generated_model", side_effect=fake_compile_generated_model):
                bridge_runner_v8._prepare_encoder_runtime(
                    gguf_path,
                    output_dir,
                    activation_overrides={"out_proj": "q8", "mlp_down": "q8"},
                )

            resolved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prefs = resolved_manifest["config"]["activation_preference_by_op"]
            self.assertEqual(prefs["branch_fc1"], "fp32")
            self.assertEqual(prefs["out_proj"], "q8")
            self.assertEqual(prefs["mlp_down"], "q8")

    def test_bridge_prepare_decoder_runtime_reuses_cached_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_bridge_decoder_reuse_") as tmpdir:
            tmp = Path(tmpdir)
            gguf_path = tmp / "decoder.gguf"
            gguf_path.write_bytes(b"gguf")
            output_dir = tmp / "decoder"
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "config": {
                    "model": "qwen3vl",
                    "arch": "qwen3vl",
                    "embed_dim": 4096,
                    "input_embed_dim": 16384,
                    "num_deepstack_layers": 3,
                    "context_length": 1024,
                    "context_len": 1024,
                    "max_seq_len": 1024,
                    "vocab_size": 151936,
                }
            }
            manifest_path = output_dir / "weights_manifest.json"
            config_path = output_dir / "config.json"
            bump_path = output_dir / "weights.bump"
            manifest_map = output_dir / "weights_manifest.map"
            prefill_ir1 = output_dir / "ir1_prefill.json"
            prefill_layout = output_dir / "layout_prefill.json"
            prefill_lowered = output_dir / "lowered_prefill.json"
            prefill_call = output_dir / "call_prefill.json"
            decode_ir1 = output_dir / "ir1_decode.json"
            decode_layout = output_dir / "layout_decode.json"
            decode_lowered = output_dir / "lowered_decode.json"
            decode_call = output_dir / "call_decode.json"
            prefill_c_path = output_dir / "decoder_v8_prefill.c"
            prefill_so_path = output_dir / "libdecoder_v8_prefill.so"
            c_path = output_dir / "decoder_v8.c"
            so_path = output_dir / "libdecoder_v8.so"
            stamp_path = output_dir / "decoder_runtime.cache.json"

            bridge_runner_v8._json_write(manifest_path, manifest)
            bridge_runner_v8._json_write(config_path, manifest["config"])
            bump_path.write_bytes(b"BUMPWGT5")
            manifest_map.write_text("{}", encoding="utf-8")
            for path in (
                prefill_ir1,
                prefill_layout,
                prefill_lowered,
                prefill_call,
                decode_ir1,
                decode_layout,
                decode_lowered,
                decode_call,
            ):
                bridge_runner_v8._json_write(
                    path,
                    {
                        "config": {
                            "embed_dim": 4096,
                            "input_embed_dim": 16384,
                            "num_deepstack_layers": 3,
                            "context_length": 1024,
                            "context_len": 1024,
                            "max_seq_len": 1024,
                            "vocab_size": 151936,
                        }
                    },
                )
            prefill_c_path.write_text("/* prefill */", encoding="utf-8")
            c_path.write_text("/* decode */", encoding="utf-8")
            bridge_runner_v8._json_write(
                stamp_path,
                bridge_runner_v8._runtime_fingerprint(
                    manifest_path=manifest_path,
                    mode="decoder_hybrid",
                    context_override=1024,
                    parity_dump=False,
                ),
            )

            def fake_run_converter(path: Path, out_dir: Path, context_override: int | None = None):
                return manifest, manifest_path, bump_path, config_path

            def fake_compile_generated_model(c_src: Path, so_dst: Path) -> Path:
                so_dst.write_bytes(b"so")
                return so_dst

            with mock.patch.object(bridge_runner_v8, "_run_converter", side_effect=fake_run_converter), \
                 mock.patch.object(bridge_runner_v8.build_ir_v8, "main") as build_ir_main, \
                 mock.patch.object(bridge_runner_v8.codegen_v8, "main") as codegen_main, \
                 mock.patch.object(bridge_runner_v8, "_compile_generated_model", side_effect=fake_compile_generated_model) as compile_model:
                runtime = bridge_runner_v8._prepare_decoder_runtime(
                    gguf_path,
                    output_dir,
                    context_override=1024,
                )

        build_ir_main.assert_not_called()
        codegen_main.assert_not_called()
        self.assertEqual(compile_model.call_count, 2)
        compile_model.assert_any_call(c_path, so_path)
        compile_model.assert_any_call(prefill_c_path, prefill_so_path)
        self.assertEqual(runtime["embed_dim"], 4096)
        self.assertEqual(runtime["input_embed_dim"], 16384)
        self.assertEqual(runtime["vocab_size"], 151936)

    def test_bridge_prepare_decoder_runtime_passes_context_override_to_ir_builds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_bridge_decoder_ctx_override_") as tmpdir:
            tmp = Path(tmpdir)
            gguf_path = tmp / "decoder.gguf"
            gguf_path.write_bytes(b"gguf")
            output_dir = tmp / "decoder"
            manifest_path = output_dir / "weights_manifest.json"
            bump_path = output_dir / "weights.bump"
            config_path = output_dir / "config.json"
            manifest = {
                "config": {
                    "model": "qwen3vl",
                    "embed_dim": 4096,
                    "input_embed_dim": 16384,
                    "num_deepstack_layers": 3,
                    "context_length": 262144,
                    "max_seq_len": 262144,
                    "vocab_size": 151936,
                }
            }
            build_calls: list[list[str]] = []
            codegen_argvs: list[list[str]] = []

            def fake_run_converter(path: Path, out_dir: Path, context_override: int | None = None):
                out_dir.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                config_path.write_text(json.dumps(manifest["config"]), encoding="utf-8")
                bump_path.write_bytes(b"BUMPWGT5")
                return manifest, manifest_path, bump_path, config_path

            def fake_build_ir_main(args: list[str]) -> int:
                argv = list(args)
                build_calls.append(argv)
                ctx = int(argv[argv.index("--context-len") + 1]) if "--context-len" in argv else 262144
                layout_path = Path(argv[argv.index("--layout-output") + 1])
                lowered_path = Path(argv[argv.index("--lowered-output") + 1])
                call_path = Path(argv[argv.index("--call-output") + 1])
                output_path = Path(argv[argv.index("--output") + 1])
                payload = {
                    "config": {
                        "embed_dim": 4096,
                        "input_embed_dim": 16384,
                        "num_deepstack_layers": 3,
                        "context_length": ctx,
                        "context_len": ctx,
                        "max_seq_len": ctx,
                        "vocab_size": 151936,
                    }
                }
                for path in (layout_path, lowered_path, call_path, output_path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                layout_path.write_text(json.dumps(payload), encoding="utf-8")
                lowered_path.write_text("{}", encoding="utf-8")
                call_path.write_text(json.dumps(payload), encoding="utf-8")
                output_path.write_text("{}", encoding="utf-8")
                if "--manifest-map-output" in argv:
                    manifest_map = Path(argv[argv.index("--manifest-map-output") + 1])
                    manifest_map.write_text("{}", encoding="utf-8")
                return 0

            def fake_codegen_main() -> int:
                codegen_argvs.append(list(sys.argv))
                output_path = Path(sys.argv[sys.argv.index("--output") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("/* generated */", encoding="utf-8")
                return 0

            def fake_compile_generated_model(c_path: Path, so_path: Path) -> Path:
                so_path.parent.mkdir(parents=True, exist_ok=True)
                so_path.write_bytes(b"so")
                return so_path

            with mock.patch.object(bridge_runner_v8, "_run_converter", side_effect=fake_run_converter), \
                 mock.patch.object(bridge_runner_v8.build_ir_v8, "main", side_effect=fake_build_ir_main), \
                 mock.patch.object(bridge_runner_v8.codegen_v8, "main", side_effect=fake_codegen_main), \
                 mock.patch.object(bridge_runner_v8, "_compile_generated_model", side_effect=fake_compile_generated_model):
                runtime = bridge_runner_v8._prepare_decoder_runtime(
                    gguf_path,
                    output_dir,
                    context_override=1024,
                )

        self.assertEqual(len(build_calls), 2)
        self.assertTrue(all("--context-len" in call for call in build_calls))
        self.assertTrue(all(call[call.index("--context-len") + 1] == "1024" for call in build_calls))
        decode_codegen = [argv for argv in codegen_argvs if Path(argv[argv.index("--output") + 1]).name == "decoder_v8.c"]
        self.assertEqual(len(decode_codegen), 1)
        self.assertIn("--prefill", decode_codegen[0])
        self.assertEqual(
            Path(decode_codegen[0][decode_codegen[0].index("--prefill") + 1]),
            output_dir / "call_prefill.json",
        )
        self.assertNotIn("--prefill-layout", decode_codegen[0])
        self.assertEqual(runtime["embed_dim"], 4096)
        self.assertEqual(runtime["input_embed_dim"], 16384)
        self.assertEqual(runtime["vocab_size"], 151936)

    def test_bridge_compile_generated_model_rebuilds_when_generated_source_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_bridge_compile_stamp_") as tmpdir:
            tmp = Path(tmpdir)
            c_path = tmp / "decoder_v8.c"
            so_path = tmp / "libdecoder_v8.so"
            stamp_path = so_path.with_suffix(so_path.suffix + ".build.json")
            c_path.write_text("/* first */", encoding="utf-8")
            so_path.write_bytes(b"so")
            initial_hash = hashlib.sha256(c_path.read_bytes()).hexdigest()
            stamp_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source_path": str(c_path.resolve()),
                        "source_sha256": initial_hash,
                        "source_size": c_path.stat().st_size,
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(bridge_runner_v8, "_run") as run_cmd:
                bridge_runner_v8._compile_generated_model(c_path, so_path)
            run_cmd.assert_not_called()

            c_path.write_text("/* second */", encoding="utf-8")

            def fake_run(cmd: list[str]) -> None:
                so_path.write_bytes(b"rebuilt")

            with mock.patch.object(bridge_runner_v8, "_run", side_effect=fake_run) as run_cmd:
                bridge_runner_v8._compile_generated_model(c_path, so_path)
                run_cmd.assert_called_once()
                updated = json.loads(stamp_path.read_text(encoding="utf-8"))
                self.assertEqual(updated["source_sha256"], hashlib.sha256(c_path.read_bytes()).hexdigest())

    def test_ck_run_v8_reuses_legacy_v7_hf_gguf_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_legacy_cache_") as tmpdir:
            tmp = Path(tmpdir)
            v8_cache = tmp / "v8"
            v7_cache = tmp / "v7"
            legacy_dir = v7_cache / "Qwen--Qwen3.5-0.8B-GGUF"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_gguf = legacy_dir / "Qwen3.5-0.8B-Q4_K_M.gguf"
            legacy_gguf.write_bytes(b"gguf")

            with mock.patch.object(ck_run_v8, "CACHE_DIR", v8_cache), \
                 mock.patch.object(ck_run_v8, "LEGACY_CACHE_DIR", v7_cache):
                resolved, repo_id = ck_run_v8._resolve_gguf_input(
                    "hf://Qwen/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
                    force_download=False,
                )

        self.assertEqual(resolved, legacy_gguf)
        self.assertEqual(repo_id, "Qwen/Qwen3.5-0.8B-GGUF")

    def test_ck_run_v8_reuses_cached_tokenizer_before_hf_download(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_cached_tokenizer_") as tmpdir:
            tmp = Path(tmpdir)
            v8_cache = tmp / "v8"
            repo_dir = v8_cache / "Qwen--Qwen3-0.6B-GGUF"
            repo_dir.mkdir(parents=True, exist_ok=True)
            cached_tokenizer = repo_dir / "tokenizer.json"
            cached_tokenizer.write_text('{"version":"cached"}', encoding="utf-8")
            work_dir = tmp / "run"

            with mock.patch.object(ck_run_v8, "CACHE_DIR", v8_cache), \
                 mock.patch.object(ck_run_v8, "LEGACY_CACHE_DIR", tmp / "v7"):
                ck_run_v8.ensure_tokenizer_files("Qwen/Qwen3-0.6B-GGUF", work_dir)

            copied = work_dir / "tokenizer.json"
            self.assertTrue(copied.exists())
            self.assertEqual(copied.read_text(encoding="utf-8"), '{"version":"cached"}')


if __name__ == "__main__":
    unittest.main()
