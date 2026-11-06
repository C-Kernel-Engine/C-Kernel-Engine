#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import io
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from array import array
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
CK_CLI_V8 = BUILD_DIR / "ck-cli-v8"
LIBCK_SESSION_V8 = BUILD_DIR / "libck_session_v8.so"
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
    elements = 1
    for dim in shape:
        elements *= int(dim)
    if dtype == "q8_0":
        if not shape or int(shape[-1]) % 32 != 0:
            raise ValueError(f"Q8_0 fixture row must be 32-value aligned: {name} {shape}")
        size = (elements // 32) * 34
    else:
        size = elements * {"fp32": 4, "fp16": 2, "i32": 4, "u8": 1}.get(dtype, 4)
    return {
        "name": name,
        "dtype": dtype,
        "offset": offset,
        "shape": shape,
        "size": size,
        "nbytes": size,
    }


def _make_qwen3_decoder_manifest(*, include_tokenizer_contract: bool = False) -> dict:
    offset = 8
    entries = []

    def add(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        item = _entry(name, dtype, shape, offset)
        entries.append(item)
        offset += int(item["size"])

    add("token_emb", "q8_0", [64, 32])
    add("layer.0.ln1_gamma", "fp32", [32])
    add("layer.0.ln2_gamma", "fp32", [32])
    add("layer.0.q_norm", "fp32", [8])
    add("layer.0.k_norm", "fp32", [8])
    add("layer.0.wq", "q8_0", [32, 32])
    add("layer.0.wk", "q8_0", [32, 32])
    add("layer.0.wv", "q8_0", [32, 32])
    add("layer.0.wo", "q8_0", [32, 32])
    add("layer.0.w1", "q8_0", [32, 32])
    add("layer.0.w2", "q8_0", [32, 32])
    add("layer.0.w3", "q8_0", [32, 32])
    add("final_ln_weight", "fp32", [32])

    vocab_tokens = [f"<|tok_{idx}|>" for idx in range(64)]
    vocab_tokens[0] = "<|pad|>"
    vocab_tokens[1] = "Hello"
    vocab_tokens[2] = " world"
    vocab_tokens[9] = "<|endoftext|>"
    if include_tokenizer_contract:
        # The formatted chat must encode completely within the 32-token
        # context. This ASCII fixture has no merges, so include the markers
        # and individual characters used by its role labels and message.
        chat_pieces = [
            "<|im_start|>", "<|im_end|>", "\n", "H", "a", "e",
            "i", "l", "n", "o", "r", "s", "t", "u",
        ]
        chat_ids = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
        for token_id, piece in zip(chat_ids, chat_pieces):
            vocab_tokens[token_id] = piece

    vocab_offsets: list[int] = []
    vocab_strings = bytearray()
    for token in vocab_tokens:
        vocab_offsets.append(len(vocab_strings))
        vocab_strings.extend(token.encode("utf-8"))
        vocab_strings.append(0)

    add("vocab_offsets", "i32", [len(vocab_offsets)])
    add("vocab_strings", "u8", [len(vocab_strings)])
    if include_tokenizer_contract:
        add("vocab_merges", "i32", [0])

    manifest = {
        "config": {
            "model": "qwen3",
            "arch": "qwen3",
            "num_layers": 1,
            "embed_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 4,
            "head_dim": 8,
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
    if include_tokenizer_contract:
        manifest["special_tokens"] = {
            "bos_token_id": 1,
            "eos_token_id": 9,
            "tokenizer_model": "bpe",
            "add_bos_token": False,
            "add_eos_token": False,
        }
    return manifest


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
        "-lckernel_tokenizer",
        f"-Wl,-rpath,{BUILD_DIR}",
        "-o",
        str(so_path),
        "-lm",
        "-lpthread",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"generated model compile failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def _build_tiny_decoder_runtime(
    workdir: Path,
    *,
    include_tokenizer_contract: bool = False,
    batched_prefill: bool = False,
    stop_token_ids: list[int] | None = None,
    special_token_overrides: dict[str, object] | None = None,
    inject_prefill_failure: bool = False,
) -> tuple[Path, Path, Path]:
    manifest = _make_qwen3_decoder_manifest(
        include_tokenizer_contract=include_tokenizer_contract
    )
    if stop_token_ids is not None:
        manifest["special_tokens"]["stop_token_ids"] = stop_token_ids
    if special_token_overrides is not None:
        manifest["special_tokens"].update(special_token_overrides)
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
    init_path = workdir / "init.json"
    init_call_path = workdir / "init_call.json"

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
            if include_tokenizer_contract:
                args.extend(["--init-output", str(init_path)])
        rc = build_ir_v8.main(args)
        if rc != 0:
            raise AssertionError(f"build_ir_v8 failed for mode={mode}")

    codegen_args = [
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
    ]
    if include_tokenizer_contract:
        codegen_args.extend(["--init", str(init_call_path)])
    if batched_prefill:
        codegen_args.extend(["--prefill-layout", str(prefill_layout)])
    result = subprocess.run(
        codegen_args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"codegen_v8 failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    if inject_prefill_failure:
        source = c_path.read_text(encoding="utf-8")
        needle = "        ck_decode(g_model, tokens[i]);\n"
        assert source.count(needle) == 1
        source = source.replace(
            needle,
            needle + "        if (getenv(\"CK_TEST_FAIL_AFTER_PREFILL_TOKEN\")) {\n"
            "            ((uint8_t *)g_model->kv_cache)[0] = 0x5a;\n"
            "            return -1;\n"
            "        }\n",
        )
        source += (
            "\nCK_EXPORT int ck_test_kv_first_byte(void) {\n"
            "    return g_model && g_model->kv_cache ? ((uint8_t *)g_model->kv_cache)[0] : -1;\n"
            "}\n"
        )
        c_path.write_text(source, encoding="utf-8")

    _compile_generated_model(c_path, so_path)
    return so_path, bump_path, manifest_map


class V8NativeBridgeHostTests(unittest.TestCase):
    def test_generated_model_compiler_comes_from_engine_build_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stamp = Path(tmpdir) / ".ck_build_flags"
            stamp.write_text(
                "CC=ccache gcc\nCFLAGS=-O3 -mavx2\nLDFLAGS=\n",
                encoding="utf-8",
            )
            self.assertEqual(
                bridge_runner_v8._engine_compiler_argv(stamp),
                ["ccache", "gcc"],
            )

    def test_runtime_engine_copy_records_compiler_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            build = root / "build"
            runtime = root / "runtime"
            build.mkdir()
            runtime.mkdir()
            engine = build / "libckernel_engine.so"
            engine.write_bytes(b"engine")
            (build / ".ck_build_flags").write_text(
                "CC=gcc\nCFLAGS=-O3\nLDFLAGS=\n",
                encoding="utf-8",
            )
            model = runtime / "libdecoder_v8.so"
            model.touch()
            probe = subprocess.CompletedProcess(
                ["gcc", "--version"], 0, "gcc test compiler\n", ""
            )
            with mock.patch.object(bridge_runner_v8, "BUILD_DIR", build), \
                 mock.patch.object(bridge_runner_v8.subprocess, "run", return_value=probe):
                copied = bridge_runner_v8._sync_runtime_engine(engine, model)

            metadata = json.loads(
                copied.with_suffix(copied.suffix + ".build.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["compiler"]["command"], ["gcc"])
            self.assertEqual(metadata["compiler"]["version"], "gcc test compiler")
            self.assertEqual(copied.read_bytes(), b"engine")

    @unittest.skipUnless(shutil.which("cc") and shutil.which("nm"), "C toolchain required")
    def test_prebuilt_encoder_engine_rejects_missing_decoder_symbols_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            encoder_dir = root / "encoder"
            decoder_dir = root / "decoder"
            encoder_dir.mkdir()
            decoder_dir.mkdir()

            def compile_so(directory: Path, name: str, source: str) -> Path:
                c_path = directory / f"{name}.c"
                so_path = directory / f"{name}.so"
                c_path.write_text(source, encoding="utf-8")
                subprocess.run(["cc", "-shared", "-fPIC", str(c_path), "-o", str(so_path)], check=True)
                return so_path

            original = compile_so(
                decoder_dir, "libckernel_engine",
                "void ck_base(void) {}\nvoid rope_forward_q_split_llama_f32(void) {}\n",
            )
            selected = compile_so(encoder_dir, "libckernel_engine", "void ck_base(void) {}\n")
            decode = compile_so(
                decoder_dir, "libdecoder_v8",
                "extern void rope_forward_q_split_llama_f32(void); "
                "void decode(void) { rope_forward_q_split_llama_f32(); }\n",
            )
            prefill = compile_so(
                decoder_dir, "libdecoder_v8_prefill",
                "extern void ck_base(void); void prefill(void) { ck_base(); }\n",
            )
            encoder_runtime = {"engine_so": selected}
            decoder_runtime = {"so_path": decode, "prefill_so_path": prefill}
            original_bytes = original.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "missing_symbols=.*rope_forward_q_split_llama_f32"):
                bridge_runner_v8._select_prebuilt_encoder_engine(encoder_runtime, decoder_runtime)
            self.assertEqual(original.read_bytes(), original_bytes)
            self.assertNotIn("engine_so", decoder_runtime)

            compile_so(decoder_dir, "libdecoder_v8", "extern void ck_base(void); void decode(void) { ck_base(); }\n")
            compile_so(
                decoder_dir, "libdecoder_v8_prefill",
                "extern void rope_forward_q_split_llama_f32(void); "
                "void prefill(void) { rope_forward_q_split_llama_f32(); }\n",
            )
            with self.assertRaisesRegex(RuntimeError, "libdecoder_v8_prefill.*missing_symbols=.*rope_forward_q_split_llama_f32"):
                bridge_runner_v8._select_prebuilt_encoder_engine(encoder_runtime, decoder_runtime)
            self.assertEqual(original.read_bytes(), original_bytes)

            compile_so(
                encoder_dir, "libckernel_engine",
                "void ck_base(void) {}\nvoid rope_forward_q_split_llama_f32(void) {}\n"
                "void ck_extra(void) {}\n",
            )
            chosen = bridge_runner_v8._select_prebuilt_encoder_engine(encoder_runtime, decoder_runtime)
            self.assertEqual(chosen, selected.resolve())
            self.assertEqual(original.read_bytes(), selected.read_bytes())
            self.assertEqual(decoder_runtime["engine_so"], str(selected.resolve()))

    def test_runtime_preflight_builds_engine_and_tokenizer_together(self) -> None:
        with mock.patch.object(bridge_runner_v8, "_run") as run, \
             mock.patch.object(bridge_runner_v8.shutil, "which", return_value="/usr/bin/gcc"), \
             mock.patch.dict(bridge_runner_v8.os.environ, {}, clear=True):
            bridge_runner_v8._ensure_engine_lib(openmp=True)
        run.assert_called_once_with(
            [
                "make",
                "CC=gcc",
                "CK_ENABLE_OPENMP=1",
                "build/libckernel_engine.so",
                "build/libckernel_tokenizer.so",
            ]
        )

    def test_runtime_preflight_propagates_requested_compiler_to_make(self) -> None:
        with mock.patch.object(bridge_runner_v8, "_run") as run, \
             mock.patch.object(bridge_runner_v8.shutil, "which", return_value="/usr/bin/gcc"), \
             mock.patch.dict(bridge_runner_v8.os.environ, {"CK_V8_COMPILER": "gcc"}, clear=True):
            bridge_runner_v8._ensure_engine_lib(openmp=True)
        run.assert_called_once_with(
            [
                "make",
                "CC=gcc",
                "CK_ENABLE_OPENMP=1",
                "build/libckernel_engine.so",
                "build/libckernel_tokenizer.so",
            ]
        )

    def test_runtime_preflight_rejects_missing_requested_compiler(self) -> None:
        with mock.patch.object(bridge_runner_v8, "_run") as run, \
             mock.patch.object(bridge_runner_v8.shutil, "which", return_value=None), \
             mock.patch.dict(bridge_runner_v8.os.environ, {"CK_V8_COMPILER": "missing-cc"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CK_V8_COMPILER not found"):
                bridge_runner_v8._ensure_engine_lib(openmp=True)
        run.assert_not_called()

    def test_pre_push_defaults_to_gcc_and_makes_icx_explicit(self) -> None:
        hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
        self.assertIn("CK_PREPUSH_USE_INTEL=0", hook)
        self.assertIn('CK_PREPUSH_USE_INTEL=1 is explicitly requested', hook)
        self.assertIn('export CK_V8_COMPILER="${CK_V8_COMPILER:-gcc}"', hook)
        self.assertIn('export CK_V8_COMPILER="${CK_V8_COMPILER:-icx}"', hook)
        self.assertNotIn("Prefer Intel path when available", hook)

    def test_engine_has_canonical_soname(self) -> None:
        dynamic = subprocess.run(
            ["readelf", "-d", str(LIBCK)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        self.assertRegex(dynamic, r"\(SONAME\).*\[libckernel_engine\.so\]")

    def test_decoder_execution_loads_runtime_selected_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selected = root / "selected-engine.so"
            selected.touch()
            runtime = {
                "so_path": root / "libdecoder_v8.so",
                "prefill_so_path": root / "libdecoder_v8_prefill.so",
                "engine_so": str(selected),
            }
            sentinel = RuntimeError("loader reached")
            with mock.patch.object(bridge_runner_v8, "_load_decoder_lib", side_effect=sentinel) as loader:
                with self.assertRaisesRegex(RuntimeError, "loader reached"):
                    bridge_runner_v8._run_decoder(runtime, array("f"), 0, [])
            loader.assert_called_once_with(runtime["prefill_so_path"], engine_so=selected)

    def test_encoder_execution_loads_runtime_selected_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selected = root / "selected-engine.so"
            selected.touch()
            (root / "libencoder_v8.so").write_bytes(b"fixture encoder library")
            layout_path = root / "layout.json"
            layout_path.write_text(json.dumps({"config": {"model": "fixture"}}))
            runtime = {
                "so_path": root / "libencoder_v8.so",
                "engine_so": str(selected),
                "layout_path": layout_path,
            }
            sentinel = RuntimeError("loader reached")
            with mock.patch.object(bridge_runner_v8, "_load_encoder_lib", side_effect=sentinel) as loader:
                with self.assertRaisesRegex(RuntimeError, "loader reached"):
                    bridge_runner_v8._run_encoder(runtime, "processor-planar")
            loader.assert_called_once_with(runtime["so_path"], engine_so=selected)

    def test_encoder_loader_uses_requested_engine_when_adjacent_copy_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requested = root / "canonical" / "libckernel_engine.so"
            adjacent = root / "runtime" / "libckernel_engine.so"
            model_so = root / "runtime" / "libencoder_v8.so"
            requested.parent.mkdir()
            adjacent.parent.mkdir()
            requested.write_bytes(b"same engine")
            adjacent.write_bytes(b"same engine")
            model_so.touch()

            fake_lib = mock.Mock()
            loads: list[Path] = []

            def fake_cdll(path: str | None, mode: int = 0) -> object:
                if path is not None:
                    loads.append(Path(path).resolve())
                return fake_lib

            with mock.patch.object(bridge_runner_v8.ctypes, "CDLL", side_effect=fake_cdll):
                bridge_runner_v8._load_encoder_lib(model_so, engine_so=requested)

            self.assertEqual(loads[0], requested.resolve())
            self.assertNotIn(adjacent.resolve(), loads)

    def test_encoder_loader_rejects_mismatched_adjacent_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requested = root / "canonical" / "libckernel_engine.so"
            adjacent = root / "runtime" / "libckernel_engine.so"
            model_so = root / "runtime" / "libencoder_v8.so"
            requested.parent.mkdir()
            adjacent.parent.mkdir()
            requested.write_bytes(b"canonical engine")
            adjacent.write_bytes(b"stale engine")
            model_so.touch()

            with self.assertRaisesRegex(RuntimeError, "different adjacent CK engine"):
                bridge_runner_v8._load_encoder_lib(model_so, engine_so=requested)

    def test_decoder_loader_hard_fails_when_tokenizer_runtime_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_so = root / "libdecoder_v8.so"
            engine_so = root / "libckernel_engine.so"
            model_so.touch()
            engine_so.touch()
            with mock.patch.object(bridge_runner_v8.ctypes, "CDLL", return_value=object()):
                with mock.patch.object(bridge_runner_v8, "BUILD_DIR", root / "missing-build"):
                    with self.assertRaisesRegex(FileNotFoundError, "requires libckernel_tokenizer"):
                        bridge_runner_v8._load_decoder_lib(model_so, engine_so=engine_so)

    def test_decoder_loader_uses_requested_engine_when_adjacent_copy_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requested = root / "canonical" / "libckernel_engine.so"
            adjacent = root / "runtime" / "libckernel_engine.so"
            model_so = root / "runtime" / "libdecoder_v8.so"
            tokenizer_so = root / "runtime" / "libckernel_tokenizer.so"
            requested.parent.mkdir()
            adjacent.parent.mkdir()
            requested.write_bytes(b"same engine")
            adjacent.write_bytes(b"same engine")
            model_so.touch()
            tokenizer_so.touch()

            fake_lib = mock.Mock()
            loads: list[Path] = []

            def fake_cdll(path: str | None, mode: int = 0) -> object:
                if path is not None:
                    loads.append(Path(path).resolve())
                return fake_lib

            with mock.patch.object(bridge_runner_v8.ctypes, "CDLL", side_effect=fake_cdll):
                with mock.patch.object(
                    bridge_runner_v8,
                    "_resolved_symbol_library",
                    return_value=requested.resolve(),
                ):
                    bridge_runner_v8._load_decoder_lib(model_so, engine_so=requested)

            self.assertEqual(loads[0], requested.resolve())
            self.assertNotIn(adjacent.resolve(), loads)

    def test_decoder_loader_rejects_mismatched_adjacent_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            requested = root / "canonical" / "libckernel_engine.so"
            adjacent = root / "runtime" / "libckernel_engine.so"
            model_so = root / "runtime" / "libdecoder_v8.so"
            requested.parent.mkdir()
            adjacent.parent.mkdir()
            requested.write_bytes(b"canonical engine")
            adjacent.write_bytes(b"stale engine")
            model_so.touch()

            with self.assertRaisesRegex(RuntimeError, "different adjacent CK engine"):
                bridge_runner_v8._load_decoder_lib(model_so, engine_so=requested)

    def test_vision_prefix_position_policy_keeps_qwen_mrope_but_gemma_linear(self) -> None:
        self.assertEqual(
            bridge_runner_v8._vision_prefix_position_policy({"model": "qwen3_vl_vision"}),
            "mrope_2d",
        )
        self.assertEqual(
            bridge_runner_v8._local_prefix_text_pos_for_policy("mrope_2d", 196, 14, 14),
            14,
        )
        self.assertEqual(
            bridge_runner_v8._vision_prefix_position_policy({"model": "gemma4_vision"}),
            "linear",
        )
        self.assertEqual(
            bridge_runner_v8._local_prefix_text_pos_for_policy("linear", 196, 14, 14),
            196,
        )

    def test_vision_prefix_decode_policy_respects_gemma_variant(self) -> None:
        self.assertEqual(
            bridge_runner_v8._vision_prefix_decode_policy({"model": "qwen3_vl_vision"}),
            "causal_mixed_prefix",
        )
        self.assertEqual(
            bridge_runner_v8._vision_prefix_decode_policy({
                "model": "gemma4_vision",
                "projector_out_dim": 4096,
                "contract": {"multimodal_bridge": {"decode_policy": "non_causal_visual_chunk"}},
            }),
            "non_causal_visual_chunk",
        )
        self.assertEqual(
            bridge_runner_v8._vision_prefix_decode_policy({"model": "gemma3_vision"}),
            "non_causal_visual_chunk",
        )

    def test_gemma4_e4b_rejects_stale_noncausal_encoder_bundle(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a positive projector_out_dim"):
            bridge_runner_v8._vision_prefix_decode_policy({"model": "gemma4_vision"})
        for width in (1536, 2560):
            layout_cfg = {
                "model": "gemma4_vision",
                "projector_out_dim": width,
                "contract": {"multimodal_bridge": {"decode_policy": "non_causal_visual_chunk"}},
            }
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "missing or stale visual decode policy"):
                bridge_runner_v8._vision_prefix_decode_policy(layout_cfg)
            del layout_cfg["contract"]["multimodal_bridge"]["decode_policy"]
            with self.subTest(width=width, policy="missing"), self.assertRaisesRegex(ValueError, "missing or stale"):
                bridge_runner_v8._vision_prefix_decode_policy(layout_cfg)
            layout_cfg["contract"]["multimodal_bridge"]["decode_policy"] = "causal_mixed_prefix"
            self.assertEqual(
                bridge_runner_v8._vision_prefix_decode_policy(layout_cfg),
                "causal_mixed_prefix",
            )

    def test_gemma4_stale_decode_policy_rejected_before_encoder_load(self) -> None:
        layout = {
            "config": {
                "model": "gemma4_vision",
                "projector_out_dim": 2560,
                "contract": {"multimodal_bridge": {"decode_policy": "non_causal_visual_chunk"}},
            }
        }
        with mock.patch.object(bridge_runner_v8, "_load_layout", return_value=layout), \
             mock.patch.object(bridge_runner_v8, "_load_encoder_lib") as load_lib:
            with self.assertRaisesRegex(ValueError, "missing or stale visual decode policy"):
                bridge_runner_v8._run_encoder({"layout_path": "unused"}, "default")
            load_lib.assert_not_called()

    def test_composition_policy_outranks_encoder_standalone_fallback(self) -> None:
        encoder_report = {
            "prefix_position_policy": "linear",
            "prefix_decode_policy": "causal_mixed_prefix",
        }
        self.assertEqual(
            bridge_runner_v8._resolved_encoder_prefix_policy(
                encoder_report,
                "prefix_position_policy",
                "mrope_2d",
                composition_owned=True,
            ),
            "mrope_2d",
        )
        self.assertEqual(
            bridge_runner_v8._resolved_encoder_prefix_policy(
                encoder_report,
                "prefix_position_policy",
                "mrope_2d",
                composition_owned=False,
            ),
            "linear",
        )

    def test_vision_prefix_policy_prefers_template_bridge_contract(self) -> None:
        layout_cfg = {
            "model": "qwen3_vl_vision",
            "contract": {
                "multimodal_bridge": {
                    "position_policy": "linear",
                    "decode_policy": "non_causal_visual_chunk",
                }
            },
        }
        self.assertEqual(bridge_runner_v8._vision_prefix_position_policy(layout_cfg), "linear")
        self.assertEqual(
            bridge_runner_v8._vision_prefix_decode_policy(layout_cfg),
            "non_causal_visual_chunk",
        )

    def test_vision_templates_declare_multimodal_bridge_contracts(self) -> None:
        gemma = build_ir_v8._load_builtin_template_doc("gemma4_vision")
        qwen = build_ir_v8._load_builtin_template_doc("qwen3_vl_vision")
        self.assertIsNotNone(gemma)
        self.assertIsNotNone(qwen)

        gemma_bridge = gemma["contract"]["multimodal_bridge"]
        self.assertEqual(gemma_bridge["producer_activation"], "vision_output")
        self.assertEqual(gemma_bridge["position_policy"], "linear")
        self.assertEqual(gemma_bridge["decode_policy"], "non_causal_visual_chunk")
        self.assertEqual(gemma_bridge["image_begin_marker"], "<|image>")
        self.assertEqual(gemma_bridge["image_end_marker"], "<image|>")
        self.assertNotIn("image_marker", gemma_bridge)

        qwen_bridge = qwen["contract"]["multimodal_bridge"]
        self.assertEqual(qwen_bridge["producer_activation"], "vision_output")
        self.assertEqual(qwen_bridge["position_policy"], "mrope_2d")
        self.assertEqual(qwen_bridge["decode_policy"], "causal_mixed_prefix")
        self.assertEqual(qwen_bridge["grid_policy"], "merged_hw")

    def test_prebuilt_runtime_pair_infers_declared_composition(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_comp_runtime_") as tmpdir:
            root = Path(tmpdir)
            encoder = root / "encoder"
            decoder = root / "decoder"
            encoder.mkdir()
            decoder.mkdir()
            (encoder / "weights_manifest.json").write_text(
                json.dumps({"template": {"name": "cohere_compass_vision"}}),
                encoding="utf-8",
            )
            (decoder / "weights_manifest.json").write_text(
                json.dumps({"template": {"name": "cohere_compass_text"}}),
                encoding="utf-8",
            )

            circuit = bridge_runner_v8._infer_prebuilt_composition_circuit(
                encoder_runtime=encoder,
                decoder_runtime=decoder,
            )

        self.assertIsNotNone(circuit)
        self.assertEqual(circuit["name"], "cohere_compass")

    def test_prebuilt_qwen_vision_requires_declared_resize_rounding(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_qwen_legacy_runtime_") as tmpdir:
            runtime = Path(tmpdir)
            for filename in (
                "libencoder_v8.so", "libckernel_engine.so", "weights.bump", "weights_manifest.map",
            ):
                (runtime / filename).touch()
            layout_path = runtime / "layout.json"
            layout_path.write_text(json.dumps({"config": {"model": "qwen3_vl_vision", "embed_dim": 1152}}))
            with self.assertRaisesRegex(ValueError, "regenerate the encoder artifact"):
                bridge_runner_v8._load_prebuilt_encoder_runtime(runtime)
            layout_path.write_text(json.dumps({
                "config": {
                    "model": "qwen3_vl_vision", "embed_dim": 1152,
                    "image_resize_rounding_policy": "half_away_from_zero",
                },
            }))
            self.assertEqual(bridge_runner_v8._load_prebuilt_encoder_runtime(runtime)["embed_dim"], 1152)

    def test_qwen_encoder_execution_rejects_legacy_layout_before_library_load(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_qwen_legacy_encoder_") as tmpdir:
            layout_path = Path(tmpdir) / "layout.json"
            layout_path.write_text(json.dumps({"config": {"model": "qwen3_vl_vision"}}))
            with mock.patch.object(bridge_runner_v8, "_load_encoder_lib") as load_library:
                with self.assertRaisesRegex(ValueError, "regenerate the encoder artifact"):
                    bridge_runner_v8._run_encoder({"layout_path": layout_path}, "gradient")
                load_library.assert_not_called()

    def test_unknown_prebuilt_runtime_pair_keeps_safe_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_comp_runtime_unknown_") as tmpdir:
            root = Path(tmpdir)
            encoder = root / "encoder"
            decoder = root / "decoder"
            encoder.mkdir()
            decoder.mkdir()
            (encoder / "weights_manifest.json").write_text(
                json.dumps({"template": {"name": "unknown_vision"}}),
                encoding="utf-8",
            )
            (decoder / "weights_manifest.json").write_text(
                json.dumps({"template": {"name": "unknown_text"}}),
                encoding="utf-8",
            )

            circuit = bridge_runner_v8._infer_prebuilt_composition_circuit(
                encoder_runtime=encoder,
                decoder_runtime=decoder,
            )

        self.assertIsNone(circuit)

    def test_gemma4_vision_patch_projection_uses_fp16_weight_kernel(self) -> None:
        gemma = build_ir_v8._load_builtin_template_doc("gemma4_vision")
        self.assertIsNotNone(gemma)
        self.assertEqual(gemma["kernels"]["patch_proj"], "gemm_nt_f16")

    def test_gemma4_vision_rope_theta_overrides_common_rope_default(self) -> None:
        circuit = build_ir_v8._load_builtin_template_doc("gemma4_vision")
        self.assertIsNotNone(circuit)
        rope_op = next(
            op for op in circuit["block_types"]["vision_encoder"]["body"]["ops"]
            if op.get("id") == "rope_qk"
        )
        params = build_ir_v8._materialize_template_config_params(
            rope_op.get("params"),
            {
                "vision_grid_w": 21,
                "rotary_dim": 72,
                "vision_rope_theta": 100.0,
            },
        )
        self.assertEqual(params["vision_grid_w"], 21)
        self.assertEqual(params["rotary_dim"], 72)
        self.assertEqual(params["freq_base"], 100.0)

        configured = build_ir_v8._materialize_template_config_params(
            rope_op.get("params"),
            {
                "vision_grid_w": 21,
                "rotary_dim": 72,
                "vision_rope_theta": 1234.0,
            },
        )
        self.assertEqual(configured["freq_base"], 1234.0)

        defaulted = build_ir_v8._materialize_template_config_params(
            rope_op.get("params"),
            {"vision_grid_w": 21, "rotary_dim": 72},
        )
        self.assertEqual(defaulted["freq_base"], 100.0)

        source = V8_BUILD_PATH.read_text(encoding="utf-8")
        marker = 'ir_op.get("kernel", "") == "rope_forward_qk_gemma4v_vision_xy"'
        self.assertNotIn(marker, source)

    def test_gemma4_geometry_override_uses_pooled_projector_tokens(self) -> None:
        contract = {
            "patch_size": 16,
            "spatial_merge_size": 3,
            "image_min_pixels": 161280,
            "image_max_pixels": 2580480,
            "image_resize_algorithm": "bicubic",
            "image_resize_padding": "center_ceil",
            "image_resize_rounding_policy": "half_away_from_zero",
        }
        with mock.patch.object(bridge_runner_v8, "_image_source_size", return_value=(512, 256)):
            overrides = bridge_runner_v8._gemma4_geometry_overrides(
                contract, Path("image.ppm"),
            )

        self.assertEqual(overrides["image_width"], 576)
        self.assertEqual(overrides["image_height"], 288)
        self.assertEqual(overrides["vision_grid_w"], 36)
        self.assertEqual(overrides["vision_grid_h"], 18)
        self.assertEqual(overrides["vision_num_patches"], 648)
        self.assertEqual(overrides["vision_merged_tokens"], 72)
        self.assertEqual(overrides["merged_grid_x"], 12)
        self.assertEqual(overrides["merged_grid_y"], 6)
        self.assertEqual(overrides["spatial_merge_factor"], 9)
        self.assertEqual(overrides["image_min_pixels"], 161280)

        with mock.patch.object(bridge_runner_v8, "_image_source_size", return_value=(120, 120)):
            half_step = bridge_runner_v8._gemma4_geometry_overrides(
                {**contract, "image_min_pixels": 1}, Path("square.ppm"),
            )
        self.assertEqual((half_step["image_width"], half_step["image_height"]), (144, 144))

        for invalid in (
            {"image_min_pixels": None},
            {"image_min_pixels": True},
            {"image_max_pixels": 1},
            {"image_resize_algorithm": "bilinear"},
            {"image_resize_padding": "none"},
            {"image_resize_rounding_policy": "ties_to_even"},
        ):
            with self.subTest(invalid=invalid), mock.patch.object(
                bridge_runner_v8, "_image_source_size", return_value=(512, 256)
            ), self.assertRaisesRegex(RuntimeError, "Gemma4 vision"):
                bridge_runner_v8._gemma4_geometry_overrides({**contract, **invalid}, Path("image.ppm"))

    def test_gemma4_bicubic_resize_uses_centered_ceil_padding(self) -> None:
        self.assertIsNotNone(bridge_runner_v8.Image, "Pillow is required for bicubic preprocessing")
        with tempfile.TemporaryDirectory(prefix="v8_gemma4_resize_") as tmpdir:
            image = Path(tmpdir) / "image.ppm"
            image.write_bytes(b"P6\n2 1\n255\n" + bytes((255, 0, 0, 0, 0, 255)))
            result = bridge_runner_v8._load_image_file(
                image, 2, 6,
                image_mean=[0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0],
                resize_algorithm="bicubic",
                resize_padding="center_ceil",
            )
        pixels = result["interleaved"]
        self.assertEqual(len(pixels), 2 * 6 * 3)
        self.assertIn("bicubic_center_ceil_resize_6x2", result["preprocess"])
        for row in range(2):
            self.assertEqual(pixels[(row * 6) * 3:(row * 6 + 1) * 3], [0.0, 0.0, 0.0])
            self.assertEqual(pixels[(row * 6 + 5) * 3:(row * 6 + 6) * 3], [0.0, 0.0, 0.0])
            self.assertGreater(sum(pixels[(row * 6 + 1) * 3:(row * 6 + 2) * 3]), 0.0)

    def test_gemma4_padding_geometry_and_normalization_boundaries(self) -> None:
        self.assertIsNotNone(bridge_runner_v8.Image, "Pillow is required for bicubic preprocessing")
        mean = [0.2, 0.3, 0.4]
        std = [0.5, 0.25, 0.2]
        black = [-0.4, -1.2, -2.0]
        cases = (
            ("portrait", 1, 2, 4, 5, (3, 0)),
            ("landscape", 2, 1, 5, 4, (0, 3)),
            ("square", 2, 2, 4, 4, None),
        )
        with tempfile.TemporaryDirectory(prefix="v8_gemma4_padding_boundaries_") as tmpdir:
            for name, src_w, src_h, out_w, out_h, padded_xy in cases:
                with self.subTest(name=name):
                    image = Path(tmpdir) / f"{name}.ppm"
                    image.write_bytes(
                        f"P6\n{src_w} {src_h}\n255\n".encode()
                        + bytes([255, 255, 255] * (src_w * src_h))
                    )
                    result = bridge_runner_v8._load_image_file(
                        image, out_h, out_w, image_mean=mean, image_std=std,
                        resize_algorithm="bicubic", resize_padding="center_ceil",
                    )
                    self.assertEqual(len(result["interleaved"]), out_w * out_h * 3)
                    self.assertEqual(len(result["planar"]), out_w * out_h * 3)
                    self.assertNotEqual(result["interleaved"][:3], black)
                    if padded_xy is None:
                        self.assertTrue(all(value > 0 for value in result["interleaved"]))
                        continue
                    x, y = padded_xy
                    index = y * out_w + x
                    legacy = bridge_runner_v8._load_image_file(
                        image, out_h, out_w, image_mean=mean, image_std=std,
                        resize_algorithm="bicubic", resize_padding="none",
                    )
                    for channel in range(3):
                        self.assertAlmostEqual(result["interleaved"][index * 3 + channel], black[channel])
                        self.assertAlmostEqual(result["planar"][channel * out_w * out_h + index], black[channel])
                        self.assertNotAlmostEqual(legacy["interleaved"][index * 3 + channel], black[channel])

    def test_gemma4_prebuilt_encoder_rejects_old_resize_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_gemma4_legacy_encoder_") as tmpdir:
            runtime = Path(tmpdir)
            for filename in (
                "libencoder_v8.so", "libckernel_engine.so", "weights.bump", "weights_manifest.map",
            ):
                (runtime / filename).touch()
            config = {
                "model": "gemma4_vision", "embed_dim": 768, "projector_out_dim": 2560,
                "image_min_pixels": 1, "image_max_pixels": 1000,
                "image_resize_algorithm": "bicubic",
                "image_resize_rounding_policy": "half_away_from_zero",
                "contract": {"multimodal_bridge": {"decode_policy": "causal_mixed_prefix"}},
            }
            layout_path = runtime / "layout.json"
            for padding in (None, "none"):
                layout_path.write_text(json.dumps({"config": {**config, "image_resize_padding": padding}}))
                with self.subTest(padding=padding), self.assertRaisesRegex(RuntimeError, "regenerate"):
                    bridge_runner_v8._load_prebuilt_encoder_runtime(runtime)
                with self.subTest(padding=padding), mock.patch.object(
                    bridge_runner_v8, "_load_encoder_lib"
                ) as load_library, self.assertRaisesRegex(RuntimeError, "regenerate"):
                    bridge_runner_v8._run_encoder({"layout_path": layout_path}, "gradient")
                load_library.assert_not_called()
            layout_path.write_text(json.dumps({"config": {**config, "image_resize_padding": "center_ceil"}}))
            self.assertEqual(bridge_runner_v8._load_prebuilt_encoder_runtime(runtime)["embed_dim"], 768)

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

    def test_ck_run_v8_step_compile_defaults_to_gcc_when_icx_is_available(self) -> None:
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
                if cmd[0] != "make":
                    Path(cmd[cmd.index("-o") + 1]).write_bytes(b"so")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            def fake_sync_runtime_lib(source, destination, *args, **kwargs):
                Path(destination).write_bytes(Path(source).read_bytes())

            def fake_which(binary: str) -> str | None:
                if binary == "icx":
                    return "/opt/intel/oneapi/compiler/latest/bin/icx"
                if binary == "gcc":
                    return "/usr/bin/gcc"
                return None

            with mock.patch.object(ck_run_v8, "PROJECT_ROOT", tmp), \
                 mock.patch.object(ck_run_v8, "BUILD_DIR", build_dir), \
                 mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd), \
                 mock.patch.object(ck_run_v8, "_sync_runtime_lib", side_effect=fake_sync_runtime_lib), \
                 mock.patch.object(ck_run_v8, "_validate_runtime_bundle"), \
                 mock.patch.object(ck_run_v8, "_compiler_supports_openmp", return_value=True), \
                 mock.patch.object(ck_run_v8.shutil, "which", side_effect=fake_which), \
                 mock.patch.dict(ck_run_v8.os.environ, {}, clear=True):
                lib_path = ck_run_v8.step_compile(model_c, out_dir, force=False)

        self.assertEqual(lib_path, out_dir / "libmodel.so")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "make")
        cmd = calls[1]
        self.assertEqual(cmd[0], "gcc")
        self.assertIn("-fopenmp", cmd)

    def test_ck_run_v8_step_compile_accepts_explicit_icx(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_step_compile_explicit_icx_") as tmpdir:
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
                if cmd[0] != "make":
                    Path(cmd[cmd.index("-o") + 1]).write_bytes(b"so")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            def fake_sync_runtime_lib(source, destination, *args, **kwargs):
                Path(destination).write_bytes(Path(source).read_bytes())

            with mock.patch.object(ck_run_v8, "PROJECT_ROOT", tmp), \
                 mock.patch.object(ck_run_v8, "BUILD_DIR", build_dir), \
                 mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd), \
                 mock.patch.object(ck_run_v8, "_sync_runtime_lib", side_effect=fake_sync_runtime_lib), \
                 mock.patch.object(ck_run_v8, "_validate_runtime_bundle"), \
                 mock.patch.object(ck_run_v8.shutil, "which", return_value="/opt/intel/bin/icx"), \
                 mock.patch.dict(ck_run_v8.os.environ, {"CK_V8_COMPILER": "icx"}, clear=True):
                ck_run_v8.step_compile(model_c, out_dir, force=False)

        self.assertEqual(calls[0][0], "make")
        self.assertEqual(calls[1][0], "icx")
        self.assertIn("-qopenmp", calls[1])

    def test_ck_run_v8_step_compile_propagates_requested_compiler_to_engine(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_step_compile_gcc_") as tmpdir:
            tmp = Path(tmpdir)
            build_dir = tmp / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            model_c = tmp / "model_v8.c"
            model_c.write_text("/* generated */", encoding="utf-8")
            out_dir = tmp / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            calls: list[list[str]] = []

            def fake_run_cmd(cmd: list[str], *, cwd=None, env=None, capture=False):
                call = [str(part) for part in cmd]
                calls.append(call)
                if call[0] == "make":
                    (build_dir / "libckernel_engine.so").write_bytes(b"engine")
                    (build_dir / "libckernel_tokenizer.so").write_bytes(b"tokenizer")
                else:
                    Path(call[call.index("-o") + 1]).write_bytes(b"model")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            def fake_sync_runtime_lib(source, destination, *args, **kwargs):
                Path(destination).write_bytes(Path(source).read_bytes())

            with mock.patch.object(ck_run_v8, "PROJECT_ROOT", tmp), \
                 mock.patch.object(ck_run_v8, "BUILD_DIR", build_dir), \
                 mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd), \
                 mock.patch.object(ck_run_v8, "_sync_runtime_lib", side_effect=fake_sync_runtime_lib), \
                 mock.patch.object(ck_run_v8, "_validate_runtime_bundle"), \
                 mock.patch.object(ck_run_v8.shutil, "which", return_value="/usr/bin/gcc"), \
                 mock.patch.dict(ck_run_v8.os.environ, {"CK_V8_COMPILER": "gcc"}, clear=True):
                ck_run_v8.step_compile(model_c, out_dir, force=True)

        self.assertEqual(calls[0][:4], ["make", "-B", "--no-print-directory", "CC=gcc"])
        self.assertEqual(calls[1][0], "gcc")

    def test_ck_run_v8_step_compile_overrides_exported_cc_with_default_gcc(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_step_compile_exported_cc_") as tmpdir:
            tmp = Path(tmpdir)
            build_dir = tmp / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            model_c = tmp / "model_v8.c"
            model_c.write_text("/* generated */", encoding="utf-8")
            out_dir = tmp / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            calls: list[list[str]] = []

            def fake_run_cmd(cmd: list[str], *, cwd=None, env=None, capture=False):
                call = [str(part) for part in cmd]
                calls.append(call)
                if call[0] == "make":
                    (build_dir / "libckernel_engine.so").write_bytes(b"engine")
                    (build_dir / "libckernel_tokenizer.so").write_bytes(b"tokenizer")
                else:
                    Path(call[call.index("-o") + 1]).write_bytes(b"model")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            def fake_sync_runtime_lib(source, destination, *args, **kwargs):
                Path(destination).write_bytes(Path(source).read_bytes())

            with mock.patch.object(ck_run_v8, "PROJECT_ROOT", tmp), \
                 mock.patch.object(ck_run_v8, "BUILD_DIR", build_dir), \
                 mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd), \
                 mock.patch.object(ck_run_v8, "_sync_runtime_lib", side_effect=fake_sync_runtime_lib), \
                 mock.patch.object(ck_run_v8, "_validate_runtime_bundle"), \
                 mock.patch.object(ck_run_v8.shutil, "which", return_value="/usr/bin/gcc"), \
                 mock.patch.dict(ck_run_v8.os.environ, {"CC": "icx"}, clear=True):
                ck_run_v8.step_compile(model_c, out_dir, force=True)

        self.assertEqual(calls[0][:4], ["make", "-B", "--no-print-directory", "CC=gcc"])
        self.assertEqual(calls[1][0], "gcc")

    def test_ck_run_v8_step_compile_does_not_publish_invalid_staged_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_step_compile_atomic_") as tmpdir:
            tmp = Path(tmpdir)
            build_dir = tmp / "build"
            build_dir.mkdir(parents=True)
            (build_dir / "libckernel_engine.so").write_bytes(b"new-engine")
            (build_dir / "libckernel_tokenizer.so").write_bytes(b"new-tokenizer")
            model_c = tmp / "model_v8.c"
            model_c.write_text("/* generated */", encoding="utf-8")
            out_dir = tmp / "run"
            out_dir.mkdir()
            old_bundle = {
                "libmodel.so": b"old-model",
                "libckernel_engine.so": b"old-engine",
                "libckernel_tokenizer.so": b"old-tokenizer",
            }
            for name, contents in old_bundle.items():
                (out_dir / name).write_bytes(contents)

            def fake_run_cmd(cmd: list[str], *, cwd=None, env=None, capture=False):
                if cmd[0] != "make":
                    Path(cmd[cmd.index("-o") + 1]).write_bytes(b"new-model")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.object(ck_run_v8, "PROJECT_ROOT", tmp), \
                 mock.patch.object(ck_run_v8, "BUILD_DIR", build_dir), \
                 mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd), \
                 mock.patch.object(ck_run_v8, "_validate_runtime_bundle", side_effect=RuntimeError("undefined symbol")), \
                 mock.patch.object(ck_run_v8.shutil, "which", return_value="/usr/bin/gcc"), \
                 mock.patch.dict(ck_run_v8.os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "undefined symbol"):
                    ck_run_v8.step_compile(model_c, out_dir, force=True)

            for name, contents in old_bundle.items():
                self.assertEqual((out_dir / name).read_bytes(), contents)

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
                    "--gemm-schedule",
                    "static",
                    "--prefill-policy",
                    "batched",
                ]
            )

        self.assertEqual(rc, 0)
        args = run_pipeline.call_args.args[0]
        self.assertEqual(args.chat_template, "llama")
        self.assertTrue(args.generate_visualizer)
        self.assertEqual(args.gemm_schedule, "static")
        self.assertEqual(args.prefill_policy, "batched")

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

    def test_format_multimodal_prompt_segments_wraps_gemma4v_image_chunk(self) -> None:
        template_doc = build_ir_v8._load_builtin_template_doc("gemma4")
        self.assertIsNotNone(template_doc)
        contract = template_doc["contract"]["chat_contract"]

        segments = bridge_runner_v8._format_multimodal_prompt_segments(
            "Explain this image.",
            contract,
            include_image=True,
        )

        self.assertTrue(segments["uses_image_chunks"])
        self.assertEqual(
            segments["before_text"],
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n<|image>",
        )
        self.assertEqual(segments["after_text"], "<image|>Explain this image.<turn|>\n<|turn>model\n")
        self.assertEqual(
            segments["formatted_prompt"],
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n"
            "<|image><image_embeds><image|>Explain this image.<turn|>\n<|turn>model\n",
        )
        suppressed = bridge_runner_v8._format_multimodal_prompt_segments(
            "Explain this image.", contract, include_image=True, thinking_mode="suppressed",
        )
        self.assertEqual(suppressed["before_text"], "<bos><|turn>user\n<|image>")
        self.assertEqual(suppressed["after_text"], segments["after_text"])

    def test_gemma4_text_prompt_respects_thinking_mode(self) -> None:
        template_doc = build_ir_v8._load_builtin_template_doc("gemma4")
        self.assertIsNotNone(template_doc)
        contract = template_doc["contract"]["chat_contract"]

        visible = bridge_runner_v8._format_prompt_with_chat_contract(
            "Hello.", contract, thinking_mode="visible",
        )
        auto = bridge_runner_v8._format_prompt_with_chat_contract(
            "Hello.", contract, thinking_mode="auto",
        )
        suppressed = bridge_runner_v8._format_prompt_with_chat_contract(
            "Hello.", contract, thinking_mode="suppressed",
        )

        self.assertEqual(
            visible,
            "<bos><|turn>system\n<|think|>\n<turn|>\n"
            "<|turn>user\nHello.<turn|>\n<|turn>model\n",
        )
        self.assertEqual(auto, visible)
        self.assertEqual(
            suppressed,
            "<bos><|turn>user\nHello.<turn|>\n<|turn>model\n",
        )
        self.assertEqual(
            bridge_runner_v8._format_prompt_with_chat_contract(
                "Hello.", contract, thinking_mode="suppressed", system_prompt="Policy.",
            ),
            "<bos><|turn>system\nPolicy.<turn|>\n"
            "<|turn>user\nHello.<turn|>\n<|turn>model\n",
        )
        self.assertEqual(
            bridge_runner_v8._format_prompt_with_chat_contract(
                "Hello.", contract, thinking_mode="visible", system_prompt="Policy.",
            ),
            "<bos><|turn>system\n<|think|>\nPolicy.<turn|>\n"
            "<|turn>user\nHello.<turn|>\n<|turn>model\n",
        )

    def test_gemma4_generated_chat_uses_declared_default_mode(self) -> None:
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")
        template_doc = build_ir_v8._load_builtin_template_doc("gemma4")
        contract = template_doc["contract"]["chat_contract"]
        source = build_ir_v8._generate_chat_contract_c_api(contract)
        with tempfile.TemporaryDirectory(prefix="gemma4_chat_contract_") as tmpdir:
            library = Path(tmpdir) / "chat.so"
            result = subprocess.run(
                [cc, "-shared", "-fPIC", "-x", "c", "-", "-o", str(library)],
                input="#include <string.h>\n#define CK_EXPORT\n" + source,
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            formatter = ctypes.CDLL(str(library)).ck_model_format_chat
            formatter.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            formatter.restype = ctypes.c_int
            for system in (None, b"Policy."):
                output = ctypes.create_string_buffer(256)
                written = formatter(system, b"Hello.", output, len(output))
                expected = bridge_runner_v8._format_prompt_with_chat_contract(
                    "Hello.", contract, thinking_mode="auto",
                    system_prompt=system.decode() if system else None,
                )
                self.assertEqual(written, len(expected))
                self.assertEqual(output.value.decode(), expected)

    def test_prompt_segment_encoding_does_not_add_bos_after_image(self) -> None:
        class FakeTokenizer:
            bos_id = 2

            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def encode(self, text: str, add_bos: bool = True) -> list[int]:
                self.calls.append((text, bool(add_bos)))
                ids = [100 + len(self.calls)]
                return ([self.bos_id] if add_bos else []) + ids

        tokenizer = FakeTokenizer()
        before = "<bos><|turn>user\n"
        after = "Explain this image.<turn|>\n<|turn>model\n"
        before_ids = bridge_runner_v8._encode_prompt_segment(
            tokenizer,
            before,
            add_bos=not bridge_runner_v8._segment_text_has_bos_prefix(before),
        )
        after_ids = bridge_runner_v8._encode_prompt_segment(tokenizer, after, add_bos=False)

        self.assertEqual(before_ids, [101])
        self.assertEqual(after_ids, [102])
        self.assertEqual(tokenizer.calls, [(before, False), (after, False)])

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

    def test_resolve_decoder_chat_contract_auto_composes_compatible_arch_policy(self) -> None:
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
        self.assertEqual(contract["semantic_contract_name"], "qwen3vl")
        formatted = bridge_runner_v8._format_multimodal_prompt_segments(
            "Explain this image.",
            contract,
            include_image=True,
            thinking_mode="suppressed",
        )
        self.assertEqual(
            formatted["before_text"],
            "<|im_start|>user\n/no_think\n<|vision_start|>",
        )
        self.assertEqual(
            formatted["after_text"],
            "<|vision_end|>Explain this image.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        self.assertEqual(
            formatted["formatted_prompt"],
            "<|im_start|>user\n/no_think\n<|vision_start|><image_embeds><|vision_end|>"
            "Explain this image.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_auto_chat_contract_does_not_compose_incompatible_arch_policy(self) -> None:
        gguf_contract = bridge_runner_v8._fallback_chat_contract_from_template_text(
            "<|im_start|><|im_end|><|vision_start|><|vision_end|>"
        )
        architecture_contract = dict(gguf_contract)
        architecture_contract["name"] = "incompatible"
        architecture_contract["turn_prefix"] = "<different>{role}"
        architecture_contract["thinking_mode_default"] = "visible"

        composed = bridge_runner_v8._compose_compatible_auto_chat_contracts(
            gguf_contract,
            architecture_contract,
        )

        self.assertNotIn("semantic_contract_name", composed)
        self.assertEqual(composed["thinking_mode_default"], "")

    def test_qwen35_auto_contract_combines_gguf_image_markers_and_thinking_policy(self) -> None:
        gguf_template = "<|im_start|><|im_end|><|vision_start|><|vision_end|>"
        with mock.patch.object(
            bridge_runner_v8,
            "_read_gguf_metadata",
            return_value={
                "general.architecture": "qwen35",
                "tokenizer.chat_template": gguf_template,
            },
        ):
            contract = bridge_runner_v8._resolve_decoder_chat_contract(
                Path("/tmp/fake-qwen36.gguf"),
                chat_template_mode="auto",
            )

        self.assertEqual(contract["semantic_contract_name"], "qwen35")
        self.assertEqual(contract["image_begin_marker"], "<|vision_start|>")
        self.assertEqual(contract["image_end_marker"], "<|vision_end|>")
        self.assertEqual(
            contract["last_user_prefix_by_thinking_mode"]["suppressed"],
            "/no_think\n",
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

    def test_gemma4_stop_policy_includes_both_end_of_generation_tokens(self) -> None:
        class FakeTokenizer:
            eos_id = 106
            token_to_id = {"<turn|>": 106, "<eos>": 1}

        for circuit in ("gemma4", "gemma4_assistant"):
            with self.subTest(circuit=circuit):
                contract = bridge_runner_v8._load_builtin_chat_contract(circuit)
                policy = bridge_runner_v8._resolve_stop_token_policy(FakeTokenizer(), contract)

                self.assertEqual(policy["stop_ids"], [1, 106])
                self.assertEqual(policy["stop_markers"], ["<turn|>", "<eos>"])

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
                 mock.patch.object(bridge_runner_v8, "_gguf_declared_context_limit", return_value=4096), \
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
                 mock.patch.object(bridge_runner_v8, "_gguf_declared_context_limit", return_value=4096), \
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
            ["make", "build/libckernel_engine.so", "ck-cli-v8", "ck-session-v8"],
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
        if not LIBCK_SESSION_V8.exists():
            raise AssertionError(f"missing session library: {LIBCK_SESSION_V8}")

    def test_ck_session_v8_ffi_formats_tokenizes_and_streams_generation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_session_ffi_") as tmpdir:
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(
                Path(tmpdir), include_tokenizer_contract=True,
                inject_prefill_failure=True,
            )

            class SessionConfig(ctypes.Structure):
                _fields_ = [
                    ("struct_size", ctypes.c_uint32),
                    ("abi_version", ctypes.c_uint32),
                    ("model_library_path", ctypes.c_char_p),
                    ("weights_path", ctypes.c_char_p),
                    ("manifest_path", ctypes.c_char_p),
                    ("context_length", ctypes.c_int32),
                    ("num_threads", ctypes.c_int32),
                    ("required_capabilities", ctypes.c_uint64),
                    ("reserved", ctypes.c_uint64 * 8),
                ]

            class GenerateRequest(ctypes.Structure):
                _fields_ = [
                    ("struct_size", ctypes.c_uint32),
                    ("abi_version", ctypes.c_uint32),
                    ("system_text", ctypes.c_char_p),
                    ("user_text", ctypes.c_char_p),
                    ("max_tokens", ctypes.c_int32),
                    ("temperature", ctypes.c_float),
                    ("top_p", ctypes.c_float),
                    ("flags", ctypes.c_uint32),
                    ("reserved0", ctypes.c_uint32),
                    ("reserved", ctypes.c_uint64 * 8),
                ]

            class GenerateResult(ctypes.Structure):
                _fields_ = [
                    ("struct_size", ctypes.c_uint32),
                    ("abi_version", ctypes.c_uint32),
                    ("prompt_tokens", ctypes.c_int32),
                    ("generated_tokens", ctypes.c_int32),
                    ("stop_reason", ctypes.c_int32),
                    ("reserved0", ctypes.c_int32),
                    ("prefill_time_ms", ctypes.c_double),
                    ("decode_time_ms", ctypes.c_double),
                    ("reserved", ctypes.c_uint64 * 8),
                ]

            token_callback_type = ctypes.CFUNCTYPE(
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.c_int32,
            )
            session_lib = ctypes.CDLL(str(LIBCK_SESSION_V8))
            session_lib.ck_session_v8_get_abi_version.restype = ctypes.c_uint32
            session_lib.ck_session_v8_open.argtypes = [
                ctypes.POINTER(SessionConfig),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            session_lib.ck_session_v8_open.restype = ctypes.c_int
            session_lib.ck_session_v8_close.argtypes = [ctypes.c_void_p]
            session_lib.ck_session_v8_format_chat.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int32,
            ]
            session_lib.ck_session_v8_format_chat.restype = ctypes.c_int
            session_lib.ck_session_v8_encode.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int32,
            ]
            session_lib.ck_session_v8_encode.restype = ctypes.c_int
            session_lib.ck_session_v8_generate.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(GenerateRequest),
                token_callback_type,
                ctypes.c_void_p,
                ctypes.POINTER(GenerateResult),
            ]
            session_lib.ck_session_v8_generate.restype = ctypes.c_int
            session_lib.ck_session_v8_reset.argtypes = [ctypes.c_void_p]
            session_lib.ck_session_v8_reset.restype = ctypes.c_int

            self.assertEqual(session_lib.ck_session_v8_get_abi_version(), 1)
            config = SessionConfig(
                struct_size=ctypes.sizeof(SessionConfig),
                abi_version=1,
                model_library_path=str(so_path).encode(),
                weights_path=str(bump_path).encode(),
                manifest_path=str(manifest_map).encode(),
                context_length=32,
                num_threads=1,
                required_capabilities=(1 << 4) | (1 << 5),
            )
            session = ctypes.c_void_p()
            self.assertEqual(
                session_lib.ck_session_v8_open(ctypes.byref(config), ctypes.byref(session)),
                0,
            )
            self.assertTrue(session.value)
            try:
                required = session_lib.ck_session_v8_format_chat(
                    session, None, b"Hello", None, 0
                )
                self.assertGreater(required, 0)
                formatted = ctypes.create_string_buffer(required + 1)
                self.assertEqual(
                    session_lib.ck_session_v8_format_chat(
                        session, None, b"Hello", formatted, len(formatted)
                    ),
                    required,
                )
                self.assertEqual(
                    formatted.value,
                    b"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
                )

                expected_prompt_ids = [
                    3, 17, 15, 8, 14, 5, 6, 8, 11, 11, 13, 4, 5,
                    3, 7, 15, 15, 10, 15, 16, 7, 12, 16, 5,
                ]
                prompt_count = session_lib.ck_session_v8_encode(
                    session, formatted.value, None, 0
                )
                self.assertEqual(prompt_count, len(expected_prompt_ids))
                prompt_ids = (ctypes.c_int32 * prompt_count)()
                self.assertEqual(
                    session_lib.ck_session_v8_encode(
                        session, formatted.value, prompt_ids, prompt_count
                    ),
                    prompt_count,
                )
                self.assertEqual(list(prompt_ids), expected_prompt_ids)
                self.assertEqual(
                    session_lib.ck_session_v8_encode(
                        session, formatted.value, prompt_ids, prompt_count - 1
                    ),
                    -8,  # caller buffer too small
                )
                self.assertEqual(
                    session_lib.ck_session_v8_encode(
                        session, b"u" * 33, None, 0
                    ),
                    33,  # exact count may exceed the generation context
                )

                encoded_count = session_lib.ck_session_v8_encode(
                    session, b"Hello", None, 0
                )
                self.assertGreater(encoded_count, 0)
                encoded = (ctypes.c_int32 * encoded_count)()
                self.assertEqual(
                    session_lib.ck_session_v8_encode(
                        session, b"Hello", encoded, encoded_count
                    ),
                    encoded_count,
                )

                streamed: list[tuple[int, bytes, int]] = []

                @token_callback_type
                def on_token(_user_data, token_id, text, text_len, sequence_index):
                    streamed.append(
                        (token_id, ctypes.string_at(text, text_len), sequence_index)
                    )
                    return 0

                request = GenerateRequest(
                    struct_size=ctypes.sizeof(GenerateRequest),
                    abi_version=1,
                    user_text=b"Hello",
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                )
                generation = GenerateResult(
                    struct_size=ctypes.sizeof(GenerateResult), abi_version=1
                )
                self.assertEqual(
                    session_lib.ck_session_v8_generate(
                        session,
                        ctypes.byref(request),
                        on_token,
                        None,
                        ctypes.byref(generation),
                    ),
                    0,
                )
                self.assertEqual(generation.generated_tokens, 1)
                self.assertEqual(generation.stop_reason, 2)  # token limit
                self.assertEqual(generation.prompt_tokens, len(expected_prompt_ids))
                self.assertGreaterEqual(generation.prefill_time_ms, 0.0)
                self.assertEqual(streamed, [(0, b"<|pad|>", 0)])

                model_lib = ctypes.CDLL(str(so_path))
                model_lib.ck_model_get_active_tokens.restype = ctypes.c_int
                model_lib.ck_model_get_vocab_size.restype = ctypes.c_int
                model_lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
                model_lib.ck_model_debug_export_kv.argtypes = [ctypes.c_char_p, ctypes.c_int]
                model_lib.ck_model_debug_export_kv.restype = ctypes.c_int
                model_lib.ck_test_kv_first_byte.restype = ctypes.c_int

                def state_snapshot(label: str) -> tuple[int, bytes, bytes]:
                    kv_path = Path(tmpdir) / f"{label}.kv"
                    self.assertEqual(model_lib.ck_model_debug_export_kv(str(kv_path).encode(), 0), 0)
                    logits = ctypes.string_at(
                        model_lib.ck_model_get_logits(),
                        model_lib.ck_model_get_vocab_size() * ctypes.sizeof(ctypes.c_float),
                    )
                    return (model_lib.ck_model_get_active_tokens(), kv_path.read_bytes(), logits)

                baseline_state = state_snapshot("baseline")
                self.assertGreater(baseline_state[0], 0)

                # Independent requests reset generated state by default while
                # retaining the loaded model and tokenizer.
                streamed.clear()
                second = GenerateResult(
                    struct_size=ctypes.sizeof(GenerateResult), abi_version=1
                )
                self.assertEqual(
                    session_lib.ck_session_v8_generate(
                        session,
                        ctypes.byref(request),
                        on_token,
                        None,
                        ctypes.byref(second),
                    ),
                    0,
                )
                self.assertEqual(second.prompt_tokens, generation.prompt_tokens)
                self.assertEqual(streamed, [(0, b"<|pad|>", 0)])
                self.assertEqual(state_snapshot("second"), baseline_state)
                with mock.patch.dict("os.environ", {"CK_TEST_FAIL_AFTER_PREFILL_TOKEN": "1"}):
                    failed = GenerateResult(
                        struct_size=ctypes.sizeof(GenerateResult), abi_version=1
                    )
                    self.assertEqual(
                        session_lib.ck_session_v8_generate(
                            session, ctypes.byref(request), on_token, None,
                            ctypes.byref(failed),
                        ),
                        -7,  # a generated forward failed after advancing state
                    )
                self.assertEqual(
                    model_lib.ck_test_kv_first_byte(), 0,
                    "failed generation must clear the injected KV write",
                )
                self.assertEqual(
                    model_lib.ck_model_get_active_tokens(), 0,
                    "failed generation must discard partially written model state",
                )
                for index in range(3):
                    streamed.clear()
                    recovered = GenerateResult(
                        struct_size=ctypes.sizeof(GenerateResult), abi_version=1
                    )
                    self.assertEqual(session_lib.ck_session_v8_generate(
                        session, ctypes.byref(request), on_token, None,
                        ctypes.byref(recovered),
                    ), 0)
                    self.assertEqual(streamed, [(0, b"<|pad|>", 0)])
                    self.assertEqual(recovered.prompt_tokens, generation.prompt_tokens)
                    self.assertEqual(state_snapshot(f"recovered-{index}"), baseline_state)
                self.assertEqual(session_lib.ck_session_v8_reset(session), 0)
            finally:
                session_lib.ck_session_v8_close(session)

    def test_generated_prefill_cancellation_releases_native_session(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_prefill_cancel_") as tmpdir:
            workdir = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(
                workdir, include_tokenizer_contract=True, batched_prefill=True
            )
            source_path = workdir / "decoder_v8.c"
            source = source_path.read_text(encoding="utf-8")
            prefill_start = (
                "static void ck_prefill_range(CKModel *model, const int32_t *tokens, "
                "int num_tokens, int prefill_start_pos) {\n"
            )
            self.assertEqual(source.count(prefill_start), 1)
            source = source.replace(
                prefill_start,
                "static atomic_int ck_test_prefill_entered = 0;\n"
                "__attribute__((visibility(\"default\"))) int ck_test_prefill_was_entered(void) {\n"
                "    return atomic_load(&ck_test_prefill_entered);\n"
                "}\n" + prefill_start +
                "    atomic_store(&ck_test_prefill_entered, 1);\n"
                "    usleep(250000);\n",
                1,
            )
            source_path.write_text(source, encoding="utf-8")
            _compile_generated_model(source_path, so_path)

            class SessionConfig(ctypes.Structure):
                _fields_ = [
                    ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                    ("model_library_path", ctypes.c_char_p), ("weights_path", ctypes.c_char_p),
                    ("manifest_path", ctypes.c_char_p), ("context_length", ctypes.c_int32),
                    ("num_threads", ctypes.c_int32), ("required_capabilities", ctypes.c_uint64),
                    ("reserved", ctypes.c_uint64 * 8),
                ]

            class GenerateRequest(ctypes.Structure):
                _fields_ = [
                    ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                    ("system_text", ctypes.c_char_p), ("user_text", ctypes.c_char_p),
                    ("max_tokens", ctypes.c_int32), ("temperature", ctypes.c_float),
                    ("top_p", ctypes.c_float), ("flags", ctypes.c_uint32),
                    ("reserved0", ctypes.c_uint32), ("reserved", ctypes.c_uint64 * 8),
                ]

            class GenerateResult(ctypes.Structure):
                _fields_ = [
                    ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                    ("prompt_tokens", ctypes.c_int32), ("generated_tokens", ctypes.c_int32),
                    ("stop_reason", ctypes.c_int32), ("reserved0", ctypes.c_int32),
                    ("prefill_time_ms", ctypes.c_double), ("decode_time_ms", ctypes.c_double),
                    ("reserved", ctypes.c_uint64 * 8),
                ]

            session_lib = ctypes.CDLL(str(LIBCK_SESSION_V8))
            session_lib.ck_session_v8_open.argtypes = [ctypes.POINTER(SessionConfig), ctypes.POINTER(ctypes.c_void_p)]
            session_lib.ck_session_v8_open.restype = ctypes.c_int
            session_lib.ck_session_v8_close.argtypes = [ctypes.c_void_p]
            session_lib.ck_session_v8_generate.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(GenerateRequest), ctypes.c_void_p,
                ctypes.c_void_p, ctypes.POINTER(GenerateResult),
            ]
            session_lib.ck_session_v8_generate.restype = ctypes.c_int
            session_lib.ck_session_v8_cancel.argtypes = [ctypes.c_void_p]
            model_lib = ctypes.CDLL(str(so_path))
            model_lib.ck_test_prefill_was_entered.restype = ctypes.c_int

            config = SessionConfig(
                struct_size=ctypes.sizeof(SessionConfig), abi_version=1,
                model_library_path=str(so_path).encode(), weights_path=str(bump_path).encode(),
                manifest_path=str(manifest_map).encode(), context_length=32, num_threads=1,
            )
            session = ctypes.c_void_p()
            self.assertEqual(session_lib.ck_session_v8_open(ctypes.byref(config), ctypes.byref(session)), 0)
            try:
                request = GenerateRequest(
                    struct_size=ctypes.sizeof(GenerateRequest), abi_version=1,
                    user_text=b"Hello world", max_tokens=1, top_p=1.0,
                    flags=1,  # CK_SESSION_REQUEST_RAW_PROMPT
                )
                cancelled = GenerateResult(struct_size=ctypes.sizeof(GenerateResult), abi_version=1)
                status = []
                worker = threading.Thread(target=lambda: status.append(
                    session_lib.ck_session_v8_generate(
                        session, ctypes.byref(request), None, None, ctypes.byref(cancelled)
                    )
                ))
                worker.start()
                deadline = time.monotonic() + 5.0
                while not model_lib.ck_test_prefill_was_entered() and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertTrue(
                    model_lib.ck_test_prefill_was_entered(),
                    f"prefill did not start: status={status}, reason={cancelled.stop_reason}, "
                    f"prompt_tokens={cancelled.prompt_tokens}",
                )
                session_lib.ck_session_v8_cancel(session)
                worker.join(timeout=5.0)
                self.assertFalse(worker.is_alive(), "native prefill did not acknowledge cancellation")
                self.assertEqual(status, [0])
                self.assertEqual(cancelled.stop_reason, 3)  # CK_SESSION_STOP_CANCELLED
                self.assertEqual(cancelled.generated_tokens, 0)

                followup = GenerateResult(struct_size=ctypes.sizeof(GenerateResult), abi_version=1)
                self.assertEqual(session_lib.ck_session_v8_generate(
                    session, ctypes.byref(request), None, None, ctypes.byref(followup)
                ), 0)
                self.assertEqual(followup.stop_reason, 2)  # token limit
                self.assertEqual(followup.generated_tokens, 1)
            finally:
                session_lib.ck_session_v8_close(session)

    def test_decode_layout_prompt_replay_acknowledges_cancellation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_replay_cancel_") as tmpdir:
            workdir = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(workdir)
            source_path = workdir / "decoder_v8.c"
            source = source_path.read_text(encoding="utf-8")
            decode_start = "static void ck_decode(CKModel *model, int32_t token) {\n"
            self.assertEqual(source.count(decode_start), 1)
            source = source.replace(
                decode_start,
                "static atomic_int ck_test_decode_entered = 0;\n"
                "__attribute__((visibility(\"default\"))) int ck_test_decode_was_entered(void) {\n"
                "    return atomic_load(&ck_test_decode_entered);\n"
                "}\n"
                "__attribute__((visibility(\"default\"))) void ck_test_set_cancel(atomic_int *flag, int value) {\n"
                "    atomic_store(flag, value);\n"
                "}\n" + decode_start +
                "    atomic_store(&ck_test_decode_entered, 1);\n"
                "    usleep(250000);\n",
                1,
            )
            source_path.write_text(source, encoding="utf-8")
            _compile_generated_model(source_path, so_path)
            model = ctypes.CDLL(str(so_path))
            model.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            model.ck_model_init_with_manifest.restype = ctypes.c_int
            model.ck_model_set_cancel_flag.argtypes = [ctypes.POINTER(ctypes.c_int)]
            model.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
            model.ck_model_embed_tokens.restype = ctypes.c_int
            model.ck_model_free.argtypes = []
            model.ck_model_kv_cache_reset.argtypes = []
            model.ck_test_decode_was_entered.restype = ctypes.c_int
            model.ck_test_set_cancel.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            self.assertEqual(model.ck_model_init_with_manifest(
                str(bump_path).encode(), str(manifest_map).encode()
            ), 0)
            try:
                flag = ctypes.c_int(0)
                model.ck_model_set_cancel_flag(ctypes.byref(flag))
                ids = (ctypes.c_int32 * 3)(1, 2, 1)
                status = []
                worker = threading.Thread(target=lambda: status.append(model.ck_model_embed_tokens(ids, 3)))
                worker.start()
                deadline = time.monotonic() + 5.0
                while not model.ck_test_decode_was_entered() and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertTrue(model.ck_test_decode_was_entered(), "decode replay did not start")
                model.ck_test_set_cancel(ctypes.byref(flag), 1)
                worker.join(timeout=5.0)
                self.assertFalse(worker.is_alive())
                self.assertEqual(status, [-2])
                model.ck_test_set_cancel(ctypes.byref(flag), 0)
                model.ck_model_kv_cache_reset()
                self.assertEqual(model.ck_model_embed_tokens(ids, 3), 0)
            finally:
                model.ck_model_set_cancel_flag(None)
                model.ck_model_free()

    def test_ck_cli_v8_runs_forward_mixed_with_prompt_tokens_and_prefix_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_cli_") as tmpdir:
            tmp = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(tmp)
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(array("f", [0.0] * (3 * 32)).tobytes())

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
                "[DEBUG] Running ck_model_forward_mixed with prefix_tokens=3 embed_dim=32 prompt_tokens=4",
                combined,
            )
            self.assertNotIn("Model does not have built-in tokenizer", combined)

    def test_ck_cli_v8_reads_prompt_tokens_from_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_prompt_tokens_") as tmpdir:
            tmp = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(tmp)
            token_path = tmp / "prompt_tokens.txt"
            token_path.write_text("1\n2\n3\n4\n", encoding="utf-8")

            result = subprocess.run(
                [
                    str(CK_CLI_V8),
                    "--lib",
                    str(so_path),
                    "--weights",
                    str(bump_path),
                    "--manifest",
                    str(manifest_map),
                    "--prompt-token-file",
                    str(token_path),
                    "--max-tokens",
                    "1",
                    "--quiet-output",
                    "--no-timing",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_generated_runtime_exports_circuit_chat_and_stop_contracts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_chat_contract_") as tmpdir:
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(
                Path(tmpdir), include_tokenizer_contract=True, stop_token_ids=[9, 10]
            )
            ctypes.CDLL(str(LIBCK), mode=ctypes.RTLD_GLOBAL)
            model = ctypes.CDLL(str(so_path))

            model.ck_model_has_chat_template.argtypes = []
            model.ck_model_has_chat_template.restype = ctypes.c_int
            model.ck_model_format_chat.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            model.ck_model_format_chat.restype = ctypes.c_int
            self.assertEqual(model.ck_model_has_chat_template(), 1)

            class RuntimeDescriptor(ctypes.Structure):
                _fields_ = [
                    ("struct_size", ctypes.c_uint32),
                    ("abi_version", ctypes.c_uint32),
                    ("capabilities", ctypes.c_uint64),
                    ("artifact_role", ctypes.c_uint32),
                    ("reserved0", ctypes.c_uint32),
                    ("context_length", ctypes.c_int32),
                    ("vocab_size", ctypes.c_int32),
                    ("encoder_memory_tokens", ctypes.c_int32),
                    ("encoder_memory_dim", ctypes.c_int32),
                    ("primary_input_tokens", ctypes.c_int32),
                    ("primary_input_dim", ctypes.c_int32),
                    ("reserved", ctypes.c_uint64 * 8),
                ]

            model.ck_model_get_abi_version.argtypes = []
            model.ck_model_get_abi_version.restype = ctypes.c_uint32
            model.ck_model_get_capabilities.argtypes = []
            model.ck_model_get_capabilities.restype = ctypes.c_uint64
            model.ck_model_get_runtime_descriptor.argtypes = [
                ctypes.POINTER(RuntimeDescriptor),
                ctypes.c_size_t,
            ]
            model.ck_model_get_runtime_descriptor.restype = ctypes.c_int
            runtime = RuntimeDescriptor()
            self.assertEqual(model.ck_model_get_abi_version(), 1)
            self.assertEqual(
                model.ck_model_get_runtime_descriptor(
                    ctypes.byref(runtime), ctypes.sizeof(runtime)
                ),
                0,
            )
            self.assertEqual(runtime.struct_size, ctypes.sizeof(runtime))
            self.assertEqual(runtime.abi_version, 1)
            self.assertEqual(runtime.capabilities, model.ck_model_get_capabilities())
            self.assertEqual(runtime.artifact_role, 1)  # decoder
            self.assertNotEqual(runtime.capabilities & (1 << 1), 0)  # decode
            self.assertNotEqual(runtime.capabilities & (1 << 2), 0)  # text encode
            self.assertNotEqual(runtime.capabilities & (1 << 4), 0)  # chat
            self.assertNotEqual(runtime.capabilities & (1 << 5), 0)  # stop tokens

            required = model.ck_model_format_chat(None, b"Hello", None, 0)
            self.assertGreater(required, 0)
            output = ctypes.create_string_buffer(required + 1)
            written = model.ck_model_format_chat(None, b"Hello", output, len(output))
            self.assertEqual(written, required)
            self.assertEqual(
                output.value.decode(),
                "<|im_start|>user\nHello<|im_end|>\n"
                "<|im_start|>assistant\n",
            )

            model.ck_model_get_eos_token_id.argtypes = []
            model.ck_model_get_eos_token_id.restype = ctypes.c_int32
            model.ck_model_is_stop_token.argtypes = [ctypes.c_int32]
            model.ck_model_is_stop_token.restype = ctypes.c_int
            eos = int(model.ck_model_get_eos_token_id())
            self.assertGreaterEqual(eos, 0)
            self.assertEqual(model.ck_model_is_stop_token(eos), 1)
            self.assertEqual(model.ck_model_is_stop_token(eos + 1), 1)
            self.assertEqual(model.ck_model_is_stop_token(eos + 2), 0)

            # The HTTP batch owner must read stop IDs from the loaded native
            # session, not from a possibly replaced bundle pathname.
            (Path(tmpdir) / "libmodel.so").symlink_to(so_path.name)
            from server.session_v8 import SessionV8
            session = SessionV8.open(Path(tmpdir), context_length=32, num_threads=2)
            try:
                self.assertEqual(session.generated_stop_ids(), (eos, eos + 1))
            finally:
                session.close()

            # An explicit caller stop must work even when the generated
            # library exports a different (nonempty) stop-token callback.
            common_cli = [
                str(CK_CLI_V8), "--lib", str(so_path), "--weights", str(bump_path),
                "--manifest", str(manifest_map), "--prompt-tokens", "1,2,3",
                "--context", "32", "--no-chat-template", "--quiet-output",
            ]
            (Path(tmpdir) / "vocab.json").write_text(
                json.dumps({"special_tokens": {"eos": eos, "bos": eos + 1}}),
                encoding="utf-8",
            )
            baseline_trace = Path(tmpdir) / "baseline.trace.json"
            baseline = subprocess.run(
                common_cli + ["--ignore-eos", "--max-tokens", "1", "--token-trace-json", str(baseline_trace)],
                cwd=str(ROOT), capture_output=True, text=True,
            )
            self.assertEqual(baseline.returncode, 0, msg=baseline.stdout + baseline.stderr)
            first_token = json.loads(baseline_trace.read_text())["token_ids"][0]
            self.assertNotIn(first_token, (eos, eos + 1))
            override_trace = Path(tmpdir) / "override.trace.json"
            override = subprocess.run(
                common_cli + ["--eos", str(first_token), "--max-tokens", "3", "--token-trace-json", str(override_trace)],
                cwd=str(ROOT), capture_output=True, text=True,
            )
            self.assertEqual(override.returncode, 0, msg=override.stdout + override.stderr)
            self.assertEqual(
                json.loads(override_trace.read_text())["stop_reason"], "eos",
                msg=f"baseline={baseline_trace.read_text()} override={override_trace.read_text()}",
            )
            ignored_trace = Path(tmpdir) / "ignored.trace.json"
            ignored = subprocess.run(
                common_cli + ["--eos", str(first_token), "--ignore-eos", "--max-tokens", "1",
                              "--token-trace-json", str(ignored_trace)],
                cwd=str(ROOT), capture_output=True, text=True,
            )
            self.assertEqual(ignored.returncode, 0, msg=ignored.stdout + ignored.stderr)
            self.assertEqual(json.loads(ignored_trace.read_text())["stop_reason"], "token_limit")
            valid_max = subprocess.run(
                common_cli + ["--eos", "63", "--max-tokens", "1"],
                cwd=str(ROOT), capture_output=True, text=True,
            )
            self.assertEqual(valid_max.returncode, 0, msg=valid_max.stdout + valid_max.stderr)
            for invalid_ids in ("-1", "0junk", "1,", "64", "2147483648", "4294967297"):
                invalid = subprocess.run(
                    common_cli + ["--eos", invalid_ids, "--max-tokens", "1"],
                    cwd=str(ROOT), capture_output=True, text=True,
                )
                self.assertNotEqual(invalid.returncode, 0, msg=invalid_ids)

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
                    "1",
                    "--max-tokens",
                    "1",
                    "--temperature",
                    "0",
                    "--no-timing",
                    "--require-generated-abi",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0, msg=result.stdout + result.stderr
            )
            self.assertIn("<|pad|>", result.stdout)

    def test_generated_stop_ids_validate_bounds_and_preserve_encode_eos(self) -> None:
        for name, override in (
            ("stop_vocab", {"stop_token_ids": [64]}),
            ("stop_int32", {"stop_token_ids": [0x80000000]}),
            ("stop_wrap", {"stop_token_ids": [0x100000001]}),
            ("bos_vocab", {"bos_token_id": 64}),
            ("eos_wrap", {"eos_token_id": 0x100000001}),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(AssertionError, "invalid .* token ID"):
                    _build_tiny_decoder_runtime(
                        Path(tmpdir), include_tokenizer_contract=True,
                        special_token_overrides=override,
                    )
                self.assertFalse((Path(tmpdir) / "libdecoder_v8.so").exists())

        with tempfile.TemporaryDirectory(prefix="v8_native_primary_eos_") as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "tokenizer.json").write_text(json.dumps({
                "model": {"vocab": {"<|pad|>": 0, "Hello": 1,
                                    " world": 2, "<|im_start|>": 3}},
                "added_tokens": [],
            }), encoding="utf-8")
            (tmp / "tokenizer_config.json").write_text(json.dumps({
                "eos_token": " world", "add_eos_token": True,
            }), encoding="utf-8")
            (tmp / "generation_config.json").write_text(
                json.dumps({"eos_token_id": [0, 3, 2, 63]}), encoding="utf-8"
            )
            converter = _load_module(
                "convert_safetensors_to_bump_v8_native_stop_tests",
                ROOT / "version" / "v8" / "scripts" / "convert_safetensors_to_bump_v8.py",
            )
            special = converter._special_tokens_from_tokenizer_config(tmp, "bpe", 64)
            self.assertEqual(special["eos_token_id"], 2)
            self.assertEqual(special["stop_token_ids"], [0, 3, 2, 63])
            so_path, _bump_path, _manifest_map = _build_tiny_decoder_runtime(
                tmp, include_tokenizer_contract=True,
                special_token_overrides=special,
            )
            (tmp / "libmodel.so").symlink_to(so_path.name)
            from server.session_v8 import SessionV8
            session = SessionV8.open(tmp, context_length=32, num_threads=2)
            try:
                self.assertEqual(session.generated_stop_ids(), (0, 3, 2, 63))
                self.assertEqual(session.encode_ids("Hello"), [1, 2])
            finally:
                session.close()

    def test_ck_cli_v8_replays_bridge_report_through_native_multimodal_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_report_cli_") as tmpdir:
            tmp = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(tmp)
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(array("f", [0.0] * (3 * 32)).tobytes())
            report_path = tmp / "bridge_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "prefix_dump_path": str(prefix_path),
                        "prefix_embed_dim": 32,
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
            token_trace_path = tmp / "native_token_trace.json"

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
                    "--token-trace-json",
                    str(token_trace_path),
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
                r"\[DEBUG\] Running ck_model_forward_mixed(?:_ex)? with prefix_tokens=3 embed_dim=32 prompt_tokens=4",
            )
            self.assertNotIn("Model does not have built-in tokenizer", combined)
            trace = json.loads(token_trace_path.read_text(encoding="utf-8"))
            self.assertEqual(trace["schema"], "cke.native_token_trace")
            self.assertEqual(trace["schema_version"], 1)
            self.assertEqual(trace["prompt_tokens"], 4)
            self.assertEqual(trace["generated_tokens"], 1)
            self.assertEqual(trace["stop_reason"], "token_limit")
            self.assertEqual(trace["token_ids"], [0])
            self.assertGreaterEqual(trace["prefill_time_ms"], 0.0)
            self.assertEqual(trace["decode_time_ms"], 0.0)
            self.assertRegex(trace["first_logits_fnv1a64"], r"^[0-9a-f]{16}$")
            self.assertGreater(trace["first_logits_count"], 0)
            self.assertEqual(token_trace_path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(list(tmp.glob("native_token_trace.json.tmp.*")))

    def test_ck_cli_v8_replays_gridless_segmented_bridge_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_gridless_segmented_cli_") as tmpdir:
            tmp = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(tmp)
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(array("f", [0.0] * (3 * 32)).tobytes())
            report_path = tmp / "bridge_report.json"

            for grid_x, grid_y, text_pos in ((None, None, None), (None, None, 9), (3, 1, None)):
                with self.subTest(grid_x=grid_x, grid_y=grid_y, text_pos=text_pos):
                    report_path.write_text(
                        json.dumps(
                            {
                                "prefix_dump_path": str(prefix_path),
                                "prefix_embed_dim": 32,
                                "prompt_tokens_before_image": [1, 2],
                                "prompt_tokens_after_image": [3, 4],
                                "stop_token_ids": [],
                                "prefix_grid_x": grid_x,
                                "prefix_grid_y": grid_y,
                                "prefix_text_pos": text_pos,
                                "multimodal_prompt_segmented": True,
                            }
                        ),
                        encoding="utf-8",
                    )
                    result = subprocess.run(
                        [
                            str(CK_CLI_V8),
                            "--lib", str(so_path),
                            "--weights", str(bump_path),
                            "--manifest", str(manifest_map),
                            "--bridge-report", str(report_path),
                            "--max-tokens", "1",
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
                    expected_grid = f"{grid_x}x{grid_y}" if grid_x is not None else "0x0"
                    self.assertIn(
                        f"before=2 prefix_tokens=3 embed_dim=32 grid={expected_grid} text_pos={text_pos or 5} after=2",
                        combined,
                    )

    def test_ck_cli_v8_bridge_report_decodes_output_from_vocab_tables_without_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_native_bridge_report_vocab_cli_") as tmpdir:
            tmp = Path(tmpdir)
            so_path, bump_path, manifest_map = _build_tiny_decoder_runtime(tmp)
            prefix_path = tmp / "prefix.f32"
            prefix_path.write_bytes(array("f", [0.0] * (3 * 32)).tobytes())
            report_path = tmp / "bridge_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "prefix_dump_path": str(prefix_path),
                        "prefix_embed_dim": 32,
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
            prefix = array("f", [0.0] * (3 * 32))
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
                 mock.patch.object(bridge_runner_v8, "_gguf_declared_context_limit", return_value=4096), \
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
                 mock.patch.object(bridge_runner_v8, "_gguf_declared_context_limit", return_value=4096), \
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
                            "--decoder-context-len",
                            "32",
                            "--max-tokens",
                            "8",
                            "--reserve-generation-tokens",
                            "64",
                            "--top-k",
                            "2",
                        ]
                    )

            self.assertEqual(rc, 0)
            ensure_engine.assert_called_once_with(openmp=False)
            prepare_decoder.assert_called_once_with(
                fake_decoder_gguf.resolve(),
                workdir.resolve() / "decoder",
                context_override=109,
                profile=False,
                runtime_config_overrides=None,
            )

            report = json.loads((workdir / "bridge_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["decoder_context_len"], 109)
            self.assertEqual(report["decoder_context_budget"]["reserved_generation_tokens"], 64)
            self.assertEqual(report["total_prefill_tokens"], 45)

    def test_explicit_decoder_context_reserves_generation_capacity(self) -> None:
        self.assertEqual(
            bridge_runner_v8._derive_decoder_context_len(
                prompt_token_count=5, prefix_tokens=40, requested=32, slack_tokens=8,
            ),
            53,
        )
        self.assertEqual(
            bridge_runner_v8._derive_decoder_context_len(
                prompt_token_count=5, prefix_tokens=40, requested=64, slack_tokens=8,
            ),
            64,
        )

    def test_negative_reserved_generation_is_rejected_before_model_loading(self) -> None:
        with self.assertRaises(SystemExit) as error:
            bridge_runner_v8.main([
                "--decoder-gguf", "unused.gguf", "--workdir", "unused",
                "--reserve-generation-tokens", "-1",
            ])
        self.assertEqual(error.exception.code, 2)

    def test_decoder_context_cap_rejects_image_and_output_overflow(self) -> None:
        derive = bridge_runner_v8._derive_decoder_context_len
        self.assertEqual(
            derive(20, 1024, slack_tokens=16, maximum_context=1060),
            1060,
        )
        with self.assertRaisesRegex(ValueError, "required=1060 limit=1059"):
            derive(20, 1024, slack_tokens=16, maximum_context=1059)
        with self.assertRaisesRegex(ValueError, "required=4258 limit=4096"):
            derive(473, 3657, slack_tokens=128, maximum_context=4096)
        with self.assertRaisesRegex(ValueError, "required=5000 limit=4096"):
            derive(20, 1024, requested=5000, maximum_context=4096)

    def test_decoder_context_limit_comes_from_registered_gguf_metadata(self) -> None:
        def field(key: str, kind: int, payload: bytes) -> bytes:
            encoded = key.encode("utf-8")
            return struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I", kind) + payload

        architecture = b"qwen3vl"
        fields = [
            field("general.architecture", 8, struct.pack("<Q", len(architecture)) + architecture),
            field("qwen3vl.context_length", 4, struct.pack("<I", 4096)),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf = Path(tmpdir) / "decoder.gguf"
            gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields)) + b"".join(fields))
            self.assertEqual(bridge_runner_v8._gguf_declared_context_limit(gguf), 4096)
            gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + fields[0])
            with self.assertRaisesRegex(ValueError, "positive declared context length"):
                bridge_runner_v8._gguf_declared_context_limit(gguf)
            gguf.write_bytes(b"BAD!")
            with self.assertRaisesRegex(ValueError, "invalid GGUF magic"):
                bridge_runner_v8._gguf_declared_context_limit(gguf)

    def test_prebuilt_decoder_context_preflight_uses_packaged_capacity(self) -> None:
        class FakeTokenizer:
            def encode(self, text: str) -> list[int]:
                return [11, 22]

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = Path(tmpdir) / "decoder"
            workdir = Path(tmpdir) / "work"
            for compiled_limit, operator_limit in ((50, None), (100, 50)):
                contract = {"context_length": compiled_limit, "model_context_limit": 128}
                with mock.patch.object(
                    bridge_runner_v8, "_load_prebuilt_decoder_runtime", return_value=contract
                ) as load_runtime, mock.patch.object(
                    bridge_runner_v8, "_load_prebuilt_decoder_tokenizer", return_value=FakeTokenizer()
                ), mock.patch.object(
                    bridge_runner_v8, "_infer_prebuilt_composition_circuit", return_value=None
                ), mock.patch.object(
                    bridge_runner_v8, "_json_read", return_value={}
                ), mock.patch.object(
                    bridge_runner_v8, "_ensure_engine_lib"
                ), mock.patch.object(
                    bridge_runner_v8, "_run_decoder"
                ) as run_decoder:
                    argv = [
                        "--decoder-runtime", str(runtime),
                        "--workdir", str(workdir),
                        "--prompt", "Describe the image.",
                        "--synthetic-prefix-tokens", "40",
                    ]
                    if operator_limit is not None:
                        argv.extend(["--decoder-context-cap", str(operator_limit)])
                    with self.assertRaisesRegex(ValueError, "decoder context exceeds the declared limit"):
                        bridge_runner_v8.main(argv)
                    load_runtime.assert_called_once_with(runtime)
                    run_decoder.assert_not_called()

    def test_invalid_prebuilt_context_fails_before_encoder_or_decoder_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = Path(tmpdir) / "decoder"
            with mock.patch.object(
                bridge_runner_v8, "_load_prebuilt_decoder_runtime",
                side_effect=RuntimeError("prebuilt decoder model manifest has invalid context_length"),
            ), mock.patch.object(
                bridge_runner_v8, "_ensure_engine_lib"
            ) as ensure_engine, mock.patch.object(
                bridge_runner_v8, "_prepare_encoder_runtime"
            ) as prepare_encoder, mock.patch.object(
                bridge_runner_v8, "_run_encoder"
            ) as run_encoder, mock.patch.object(
                bridge_runner_v8, "_run_decoder"
            ) as run_decoder:
                with self.assertRaisesRegex(RuntimeError, "invalid context_length"):
                    bridge_runner_v8.main([
                        "--decoder-runtime", str(runtime),
                        "--encoder-gguf", str(Path(tmpdir) / "encoder.gguf"),
                        "--workdir", str(Path(tmpdir) / "work"),
                        "--prompt", "Describe the image.",
                    ])
                ensure_engine.assert_not_called()
                prepare_encoder.assert_not_called()
                run_encoder.assert_not_called()
                run_decoder.assert_not_called()

    def test_prebuilt_decoder_rejects_loaded_context_mismatch(self) -> None:
        runtime = {
            "runtime_dir": Path("/unused"),
            "so_path": Path("/unused/libmodel.so"),
            "engine_so": Path("/unused/libckernel_engine.so"),
            "context_length": 64,
        }
        library = mock.Mock()
        library.ck_model_get_context_window.return_value = 32
        with mock.patch.object(
            bridge_runner_v8, "_load_decoder_lib", return_value=library
        ):
            with self.assertRaisesRegex(RuntimeError, "loaded=32 packaged=64"):
                bridge_runner_v8._run_decoder(runtime, array("f"), 0, [])
        library.ck_model_init_with_manifest.assert_not_called()
        library.ck_model_get_context_window = None
        with mock.patch.object(
            bridge_runner_v8, "_load_decoder_lib", return_value=library
        ):
            with self.assertRaisesRegex(RuntimeError, "no context-window ABI"):
                bridge_runner_v8._run_decoder(runtime, array("f"), 0, [])

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
                 mock.patch.object(bridge_runner_v8, "_gguf_declared_context_limit", return_value=4096), \
                 mock.patch.object(bridge_runner_v8, "_prepare_encoder_runtime", return_value=fake_encoder_runtime), \
                 mock.patch.object(bridge_runner_v8, "_planned_encoder_prefix_tokens", return_value=3), \
                 mock.patch.object(bridge_runner_v8, "_run_encoder", return_value=fake_encoder_report) as run_encoder, \
                 mock.patch.object(bridge_runner_v8, "_prepare_decoder_runtime", return_value=fake_decoder_runtime) as prepare_decoder, \
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
                with self.assertRaisesRegex(ValueError, "decoder context exceeds the declared limit"):
                    bridge_runner_v8.main(
                        [
                            "--decoder-gguf", str(fake_decoder_gguf),
                            "--encoder-gguf", str(fake_encoder_gguf),
                            "--workdir", str(workdir),
                            "--prompt", "Describe the image.",
                            "--decoder-context-cap", "20",
                        ]
                    )

            self.assertEqual(rc, 0)
            run_encoder.assert_called_once()
            prepare_decoder.assert_called_once()
            self.assertEqual(ensure_engine.call_args_list, [mock.call(openmp=True)] * 2)
            _, decoder_kwargs = run_decoder.call_args
            self.assertIsNone(decoder_kwargs["prefix_grid"])
            self.assertEqual(decoder_kwargs["prefix_text_pos"], 3)
            report = json.loads((workdir / "bridge_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["prefix_source"], "encoder")
            self.assertEqual(report["prefix_tokens"], 3)
            self.assertIsNone(report["prefix_grid_x"])
            self.assertIsNone(report["prefix_grid_y"])
            self.assertEqual(report["decoder_context_len"], 32)

    def test_bridge_memory_snapshot_supports_cgroup_v1(self) -> None:
        def fake_read(path: Path) -> int | None:
            values = {
                "/sys/fs/cgroup/memory/memory.usage_in_bytes": 1234,
                "/sys/fs/cgroup/memory/memory.limit_in_bytes": 5678,
            }
            return values.get(str(path))

        with mock.patch.object(bridge_runner_v8, "_read_int_file", side_effect=fake_read):
            snapshot = bridge_runner_v8._memory_snapshot()

        self.assertEqual(snapshot["source"], "cgroup_v1")
        self.assertEqual(snapshot["cgroup_current_bytes"], 1234)
        self.assertEqual(snapshot["cgroup_limit_bytes"], 5678)

    def test_bridge_weight_storage_evidence_resolves_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_bridge_weights_") as tmpdir:
            root = Path(tmpdir)
            target = root / "persistent.bump"
            target.write_bytes(b"weights")
            link = root / "weights.bump"
            link.symlink_to(target)
            with mock.patch.object(bridge_runner_v8, "_path_filesystem_type", return_value="tmpfs"):
                evidence = bridge_runner_v8._weight_storage_evidence(link)

        self.assertEqual(evidence["resolved_path"], str(target.resolve()))
        self.assertEqual(evidence["size_bytes"], 7)
        self.assertEqual(evidence["filesystem_type"], "tmpfs")
        self.assertTrue(evidence["memory_backed"])

    def test_run_decoder_uses_decode_runtime_for_mixed_prefix_continuation(self) -> None:
        used_paths: list[Path] = []

        class FakeLib:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.ck_set_gemm_schedule = mock.Mock(return_value=0)

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

        self.assertEqual(used_paths, [ROOT / "prefill_runtime.so"])
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
                self.ck_set_gemm_schedule = mock.Mock(return_value=0)

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
            def __init__(self) -> None:
                self.ck_set_gemm_schedule = mock.Mock(return_value=0)

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
                self.ck_set_gemm_schedule = mock.Mock(return_value=0)

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
                self.ck_set_gemm_schedule = mock.Mock(return_value=0)

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
                self.ck_set_gemm_schedule = mock.Mock(return_value=0)

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

        runtime["context_length"] = 5
        with mock.patch.object(bridge_runner_v8, "_load_decoder_lib", return_value=FakeLib()):
            with self.assertRaisesRegex(RuntimeError, "no generation capacity"):
                bridge_runner_v8._run_decoder(
                    runtime,
                    array("f", [0.0] * 16),
                    1,
                    [1, 2],
                    tokens_before=[10, 20],
                    tokenizer=FakeTokenizer(),
                    max_tokens=1,
                    stream_output=False,
                )

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
                image_min_tokens=None,
                image_max_tokens=None,
                bridge_runtime=None,
                bridge_generation_mode=None,
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

    def test_ck_run_v8_auto_resolves_qwen3vl_mmproj_from_decoder_input(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_multimodal_auto_mmproj_") as tmpdir:
            tmp = Path(tmpdir)
            decoder = tmp / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
            encoder = tmp / "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
            image = tmp / "image.png"
            decoder.write_bytes(b"gguf")
            encoder.write_bytes(b"gguf")
            image.write_bytes(b"png")

            args = argparse.Namespace(
                model=str(decoder),
                run_dir=str(tmp / "run"),
                mmproj=None,
                image_path=str(image),
                synthetic_prefix_tokens=0,
                force_download=False,
                prompt="Explain this image.",
                image_mode="checker",
                vision_top_k=8,
                vision_activation_pref=[],
                image_min_tokens=None,
                image_max_tokens=None,
                bridge_runtime=None,
                bridge_generation_mode=None,
                max_tokens=16,
                no_chat_template=False,
                chat_template="auto",
                allow_raw_prompt=False,
                thinking_mode="suppressed",
                context_len=1024,
                temperature=0.0,
                top_k=40,
                top_p=1.0,
                min_p=0.0,
                repeat_penalty=1.1,
                repeat_last_n=64,
                force_convert=False,
                force_compile=False,
                memory=False,
                python_tokenizer=False,
                generate_only=False,
            )

            calls: list[str] = []
            captured_cmds: list[list[str]] = []

            def fake_resolve_gguf_input(spec: str, *, force_download: bool):
                calls.append(spec)
                if spec == str(decoder):
                    return decoder, None
                self.assertEqual(
                    spec,
                    "hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
                )
                return encoder, "Qwen/Qwen3-VL-8B-Instruct-GGUF"

            def fake_run_cmd(cmd: list[str], **kwargs):
                captured_cmds.append([str(part) for part in cmd])
                return None

            with mock.patch.object(ck_run_v8, "_resolve_gguf_input", side_effect=fake_resolve_gguf_input), \
                 mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd):
                rc = ck_run_v8.run_pipeline(args)

        self.assertEqual(rc, 0)
        self.assertEqual(
            calls,
            [
                str(decoder),
                "hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
            ],
        )
        self.assertEqual(len(captured_cmds), 1)
        cmd = captured_cmds[0]
        self.assertIn("--encoder-gguf", cmd)
        self.assertIn(str(encoder), cmd)

    def test_ck_run_v8_routes_local_safetensors_dir_through_safetensors_converter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_local_safetensors_") as tmpdir:
            tmp = Path(tmpdir)
            checkpoint = tmp / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors").write_bytes(b"stub")

            work_dir = tmp / "run"
            weights_path = work_dir / "weights.bump"
            config_path = work_dir / "config.json"
            manifest_path = work_dir / "weights_manifest.json"
            model_c_path = work_dir / "model_v8.c"
            lib_path = work_dir / "libmodel.so"
            ir_paths = {
                "prefill_ir": work_dir / "ir1_prefill.json",
                "prefill_layout": work_dir / "layout_prefill.json",
                "prefill_lowered": work_dir / "lowered_prefill.json",
                "prefill_call": work_dir / "lowered_prefill_call.json",
                "decode_ir": work_dir / "ir1_decode.json",
                "decode_layout": work_dir / "layout_decode.json",
                "decode_lowered": work_dir / "lowered_decode.json",
                "decode_call": work_dir / "lowered_decode_call.json",
                "manifest_map": work_dir / "weights_manifest.map",
            }
            args = argparse.Namespace(
                model=str(checkpoint),
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
                generate_visualizer=False,
                generate_only=True,
                logits_layout="auto",
                profile=False,
                sweep_kernels=False,
                sweep_kernels_full=False,
            )

            with mock.patch.object(ck_run_v8, "step_convert_safetensors", return_value=(weights_path, config_path, manifest_path)) as convert_safe, \
                 mock.patch.object(ck_run_v8, "step_convert_gguf") as convert_gguf, \
                 mock.patch.object(ck_run_v8, "step_build_ir", return_value=ir_paths), \
                 mock.patch.object(ck_run_v8, "step_codegen", return_value=model_c_path), \
                 mock.patch.object(ck_run_v8, "step_compile", return_value=lib_path):
                rc = ck_run_v8.run_pipeline(args)

        self.assertEqual(rc, 0)
        convert_safe.assert_called_once_with(checkpoint.resolve(), work_dir, force=False)
        convert_gguf.assert_not_called()

    def test_ck_run_v8_routes_hf_safetensors_repo_through_safetensors_converter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_hf_safetensors_") as tmpdir:
            tmp = Path(tmpdir)
            checkpoint = tmp / "google--gemma-4-E4B-it-assistant"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

            work_dir = tmp / "run"
            weights_path = work_dir / "weights.bump"
            config_path = work_dir / "config.json"
            manifest_path = work_dir / "weights_manifest.json"
            model_c_path = work_dir / "model_v8.c"
            lib_path = work_dir / "libmodel.so"
            ir_paths = {
                "prefill_ir": work_dir / "ir1_prefill.json",
                "prefill_layout": work_dir / "layout_prefill.json",
                "prefill_lowered": work_dir / "lowered_prefill.json",
                "prefill_call": work_dir / "lowered_prefill_call.json",
                "decode_ir": work_dir / "ir1_decode.json",
                "decode_layout": work_dir / "layout_decode.json",
                "decode_lowered": work_dir / "lowered_decode.json",
                "decode_call": work_dir / "lowered_decode_call.json",
                "manifest_map": work_dir / "weights_manifest.map",
            }
            args = argparse.Namespace(
                model="hf://google/gemma-4-E4B-it-assistant",
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
                force_convert=True,
                force_compile=False,
                memory=False,
                python_tokenizer=False,
                generate_visualizer=False,
                generate_only=True,
                logits_layout="auto",
                profile=False,
                sweep_kernels=False,
                sweep_kernels_full=False,
            )

            with mock.patch.object(ck_run_v8, "step_download", return_value=checkpoint) as download, \
                 mock.patch.object(ck_run_v8, "step_convert_safetensors", return_value=(weights_path, config_path, manifest_path)) as convert_safe, \
                 mock.patch.object(ck_run_v8, "step_convert_gguf") as convert_gguf, \
                 mock.patch.object(ck_run_v8, "step_build_ir", return_value=ir_paths), \
                 mock.patch.object(ck_run_v8, "step_codegen", return_value=model_c_path), \
                 mock.patch.object(ck_run_v8, "step_compile", return_value=lib_path):
                rc = ck_run_v8.run_pipeline(args)

        self.assertEqual(rc, 0)
        download.assert_called_once_with("google/gemma-4-E4B-it-assistant", ck_run_v8.CACHE_DIR, force=False)
        convert_safe.assert_called_once_with(checkpoint, work_dir, force=True)
        convert_gguf.assert_not_called()

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
                outputs = ck_run_v8.step_build_ir(
                    manifest_path,
                    output_dir,
                    force=True,
                    context_len=1024,
                    prefill_policy="batched",
                )

        self.assertEqual(len(calls), 2)
        decode_cmd = calls[1]
        self.assertIn("--init-output", decode_cmd)
        self.assertEqual(
            decode_cmd[decode_cmd.index("--prefill-policy-override") + 1],
            "batched",
        )
        self.assertEqual(outputs["init_ir"].name, "init.json")
        self.assertEqual(outputs["init_call"].name, "init_call.json")

    def test_ck_run_v8_step_build_ir_rejects_stale_planner_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_build_ir_provenance_") as tmpdir:
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
                for flag in (
                    "--output",
                    "--layout-output",
                    "--lowered-output",
                    "--call-output",
                    "--manifest-map-output",
                ):
                    if flag in cmd:
                        Path(cmd[cmd.index(flag) + 1]).write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            common = (
                mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd),
                mock.patch.object(
                    ck_run_v8,
                    "step_regenerate_kernel_registry",
                    return_value=ck_run_v8.KERNEL_REGISTRY_PATH,
                ),
            )
            with common[0], common[1], mock.patch.object(
                ck_run_v8, "_tree_identity", return_value={"planner": "v1"}
            ):
                ck_run_v8.step_build_ir(manifest_path, output_dir, context_len=1024)
                ck_run_v8.step_build_ir(manifest_path, output_dir, context_len=1024)

            self.assertEqual(len(calls), 2, "matching IR provenance must reuse the cache")

            with mock.patch.object(ck_run_v8, "run_cmd", side_effect=fake_run_cmd), \
                 mock.patch.object(
                     ck_run_v8,
                     "step_regenerate_kernel_registry",
                     return_value=ck_run_v8.KERNEL_REGISTRY_PATH,
                 ), \
                 mock.patch.object(
                     ck_run_v8, "_tree_identity", return_value={"planner": "v2"}
                 ):
                ck_run_v8.step_build_ir(manifest_path, output_dir, context_len=1024)

            self.assertEqual(len(calls), 4, "changed planner provenance must rebuild both modes")

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

    def test_runtime_source_fingerprint_invalidates_on_imported_codegen_change(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_runtime_source_fingerprint_") as tmpdir:
            source = Path(tmpdir) / "codegen_core.py"
            source.write_text("VERSION = 1\n", encoding="utf-8")
            before = bridge_runner_v8._source_set_fingerprint([source])
            source.write_text("VERSION = 2\n", encoding="utf-8")
            after = bridge_runner_v8._source_set_fingerprint([source])

        self.assertEqual(before["file_count"], 1)
        self.assertEqual(after["file_count"], 1)
        self.assertNotEqual(before["sha256"], after["sha256"])

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

            def fake_compile_generated_model(c_src: Path, so_dst: Path, *, profile: bool = False) -> Path:
                so_dst.write_bytes(b"so")
                return so_dst

            with mock.patch.object(bridge_runner_v8, "_run_converter", side_effect=fake_run_converter), \
                 mock.patch.object(bridge_runner_v8.build_ir_v8, "main") as build_ir_main, \
                 mock.patch.object(bridge_runner_v8.codegen_v8, "main") as codegen_main, \
                 mock.patch.object(bridge_runner_v8, "_compile_generated_model", side_effect=fake_compile_generated_model) as compile_model:
                runtime = bridge_runner_v8._prepare_encoder_runtime(gguf_path, output_dir)

        build_ir_main.assert_not_called()
        codegen_main.assert_not_called()
        compile_model.assert_called_once_with(c_path, so_path, profile=False)
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

            def fake_compile_generated_model(c_src: Path, so_dst: Path, *, profile: bool = False) -> Path:
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
            init_ir = output_dir / "init.json"
            init_call = output_dir / "init_call.json"
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
                init_ir,
                init_call,
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

            def fake_compile_generated_model(c_src: Path, so_dst: Path, *, profile: bool = False) -> Path:
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
        compile_model.assert_any_call(c_path, so_path, profile=False)
        compile_model.assert_any_call(prefill_c_path, prefill_so_path, profile=False)
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
                if "--init-output" in argv:
                    init_ir = Path(argv[argv.index("--init-output") + 1])
                    init_ir.write_text("{}", encoding="utf-8")
                    (init_ir.parent / "init_call.json").write_text("{}", encoding="utf-8")
                return 0

            def fake_codegen_main() -> int:
                codegen_argvs.append(list(sys.argv))
                output_path = Path(sys.argv[sys.argv.index("--output") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("/* generated */", encoding="utf-8")
                return 0

            def fake_compile_generated_model(c_path: Path, so_path: Path, *, profile: bool = False) -> Path:
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
        decode_build = [call for call in build_calls if call[call.index("--mode") + 1] == "decode"]
        self.assertEqual(len(decode_build), 1)
        self.assertIn("--init-output", decode_build[0])
        self.assertEqual(
            Path(decode_build[0][decode_build[0].index("--init-output") + 1]),
            output_dir / "init.json",
        )
        decode_codegen = [argv for argv in codegen_argvs if Path(argv[argv.index("--output") + 1]).name == "decoder_v8.c"]
        self.assertEqual(len(decode_codegen), 1)
        self.assertTrue(all("--init" in argv for argv in codegen_argvs))
        self.assertTrue(
            all(Path(argv[argv.index("--init") + 1]) == output_dir / "init_call.json" for argv in codegen_argvs)
        )
        self.assertIn("--prefill", decode_codegen[0])
        self.assertEqual(
            Path(decode_codegen[0][decode_codegen[0].index("--prefill") + 1]),
            output_dir / "call_prefill.json",
        )
        self.assertIn("--prefill-layout", decode_codegen[0])
        self.assertEqual(
            Path(decode_codegen[0][decode_codegen[0].index("--prefill-layout") + 1]),
            output_dir / "layout_prefill.json",
        )
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
            def fake_run(cmd: list[str]) -> None:
                so_path.write_bytes(b"rebuilt")

            with mock.patch.object(bridge_runner_v8, "_run", side_effect=fake_run) as run_cmd:
                bridge_runner_v8._compile_generated_model(c_path, so_path)
                run_cmd.assert_called_once()
                self.assertIn("-Wl,-rpath,$ORIGIN", run_cmd.call_args.args[0])

            with mock.patch.object(bridge_runner_v8, "_run") as run_cmd:
                bridge_runner_v8._compile_generated_model(c_path, so_path)
                run_cmd.assert_not_called()

            initial = json.loads(stamp_path.read_text(encoding="utf-8"))
            self.assertEqual(initial["version"], 3)
            self.assertEqual(initial["runtime_dependency"]["runpath"], "$ORIGIN")
            self.assertEqual(
                initial["runtime_dependency"]["sha256"],
                hashlib.sha256((BUILD_DIR / "libckernel_engine.so").read_bytes()).hexdigest(),
            )

            c_path.write_text("/* second */", encoding="utf-8")

            with mock.patch.object(bridge_runner_v8, "_run", side_effect=fake_run) as run_cmd:
                bridge_runner_v8._compile_generated_model(c_path, so_path)
                run_cmd.assert_called_once()
                updated = json.loads(stamp_path.read_text(encoding="utf-8"))
                self.assertEqual(updated["source_sha256"], hashlib.sha256(c_path.read_bytes()).hexdigest())

            changed_support = {
                "sha256": "support-source-changed",
                "file_count": int(initial["compiled_support_source_set"]["file_count"]),
            }
            with mock.patch.object(
                bridge_runner_v8,
                "_compiled_runtime_support_fingerprint",
                return_value=changed_support,
            ), mock.patch.object(bridge_runner_v8, "_run", side_effect=fake_run) as run_cmd:
                bridge_runner_v8._compile_generated_model(c_path, so_path)
                run_cmd.assert_called_once()
                dependency_updated = json.loads(stamp_path.read_text(encoding="utf-8"))
                self.assertEqual(dependency_updated["compiled_support_source_set"], changed_support)

            compiler_descriptor = {
                "command": ["icx-test"],
                "version": "icx-test 1.0",
            }
            with mock.patch.object(
                bridge_runner_v8, "_engine_compiler_argv", return_value=["icx-test"]
            ), mock.patch.object(
                bridge_runner_v8,
                "_engine_compiler_descriptor",
                return_value=compiler_descriptor,
            ), mock.patch.object(
                bridge_runner_v8, "_run", side_effect=fake_run
            ) as run_cmd:
                bridge_runner_v8._compile_generated_model(c_path, so_path)
                run_cmd.assert_called_once()
                self.assertEqual(run_cmd.call_args.args[0][0], "icx-test")
                compiler_updated = json.loads(stamp_path.read_text(encoding="utf-8"))
                self.assertEqual(compiler_updated["compiler"], compiler_descriptor)

    def test_ck_run_v8_reuses_legacy_v7_hf_gguf_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_legacy_cache_") as tmpdir:
            tmp = Path(tmpdir)
            v8_cache = tmp / "v8"
            v7_cache = tmp / "v7"
            legacy_dir = v7_cache / "Qwen--Qwen3.5-0.8B-GGUF"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_gguf = legacy_dir / "Qwen3.5-0.8B-Q4_K_M.gguf"
            legacy_gguf.write_bytes(b"GGUF")

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

    def test_ck_run_v8_falls_back_to_tmp_cache_when_default_is_unwritable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_tmp_cache_") as tmpdir:
            with mock.patch.dict(ck_run_v8.os.environ, {}, clear=True), \
                 mock.patch.object(ck_run_v8, "_can_write_dir", return_value=False), \
                 mock.patch.object(ck_run_v8, "DEFAULT_CACHE_DIR", Path("/readonly/ck-engine-v8/models")), \
                 mock.patch.object(ck_run_v8.tempfile, "gettempdir", return_value=tmpdir):
                resolved = ck_run_v8._get_cache_dir()

        self.assertEqual(resolved, Path(tmpdir) / "ck-engine-v8" / "models")

    def test_ck_run_v8_download_redirects_hf_cache_under_selected_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v8_hf_cache_") as tmpdir:
            tmp = Path(tmpdir)
            cache_dir = tmp / "models"
            seen: dict[str, str] = {}

            def fake_hf_hub_download(**kwargs: str) -> str:
                seen.update({k: str(v) for k, v in kwargs.items()})
                local_dir = Path(kwargs["local_dir"])
                local_dir.mkdir(parents=True, exist_ok=True)
                target = local_dir / Path(kwargs["filename"]).name
                # Must carry the real GGUF magic: step_download_gguf validates
                # the payload before publishing it to the cache path.
                target.write_bytes(b"GGUF" + b"\x00" * 16)
                return str(target)

            fake_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
            with mock.patch.object(ck_run_v8, "CACHE_DIR", cache_dir), \
                 mock.patch.object(ck_run_v8, "DEFAULT_CACHE_DIR", tmp / "default-home-models"), \
                 mock.patch.object(ck_run_v8, "LEGACY_CACHE_DIR", tmp / "legacy-models"), \
                 mock.patch.dict(sys.modules, {"huggingface_hub": fake_module}):
                resolved = ck_run_v8.step_download_gguf(
                    "mradermacher/Nanbeige4.1-3B-GGUF",
                    "Nanbeige4.1-3B.Q4_K_M.gguf",
                    cache_dir,
                    force=False,
                )
                self.assertTrue(resolved.exists())
                self.assertEqual(seen["cache_dir"], str(cache_dir / ".hf-hub"))
                self.assertEqual(seen["local_dir"], str(cache_dir / "mradermacher--Nanbeige4.1-3B-GGUF"))


if __name__ == "__main__":
    unittest.main()
