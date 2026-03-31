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
        "-Iversion/v7/src",
        str(c_path),
        "version/v7/src/ckernel_model_load_v7.c",
        "version/v7/src/ck_parallel_decode.c",
        "version/v7/src/ck_parallel_prefill.c",
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
                "vocab_size": 4,
                "so_path": workdir / "decoder" / "libdecoder_v8.so",
                "c_path": workdir / "decoder" / "decoder_v8.c",
            }
            fake_logits = array("f", [0.1, 0.9, -0.4, 0.0])

            with mock.patch.object(bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_runtime), \
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
            self.assertTrue(dump_path.exists())
            self.assertEqual(dump_path.stat().st_size, 3 * 16 * 4)

            report = json.loads((workdir / "bridge_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["prefix_source"], "synthetic_zero")
            self.assertEqual(report["prefix_tokens"], 3)
            self.assertEqual(report["decoder_embed_dim"], 16)
            self.assertEqual(report["prompt_tokens"], [11, 22])
            self.assertEqual(report["prefix_dump_path"], str(dump_path.resolve()))


if __name__ == "__main__":
    unittest.main()
