#!/usr/bin/env python3
from __future__ import annotations

import argparse
from array import array
import ctypes
import heapq
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is optional at import time.
    Image = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
V8_TOOLS = REPO_ROOT / "version" / "v8" / "tools"
BUILD_DIR = REPO_ROOT / "build"
LLAMA_CPP_ROOT = REPO_ROOT / "llama.cpp"
V7_SCRIPTS = REPO_ROOT / "version" / "v7" / "scripts"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(V7_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V7_SCRIPTS))

import parity_qwen3vl_mmproj_v8 as parity_harness  # type: ignore  # noqa: E402
import build_ir_v8  # type: ignore  # noqa: E402
import parity_test  # type: ignore  # noqa: E402
from vision_bridge_runtime_v8 import (  # type: ignore  # noqa: E402
    declare_named_activation_api,
    resolve_vision_bridge_contract,
    try_named_activation_view,
)


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), env=env, check=True)


def _with_env_var(name: str, value: str | None) -> str | None:
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    return old


def _restore_env_var(name: str, old: str | None) -> None:
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


def _ensure_engine_lib() -> None:
    _run(["make", "-B", "CK_ENABLE_OPENMP=1", "build/libckernel_engine.so"])


def _load_runtime_metadata(report: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    result = dict(report)
    config = result.get("config")
    if not isinstance(config, dict) or not config:
        config_path = output_dir / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                result["config"] = loaded
        else:
            layout_path = output_dir / "layout.json"
            if layout_path.exists():
                layout_obj = _load_layout(layout_path)
                loaded = layout_obj.get("config")
                if isinstance(loaded, dict):
                    result["config"] = dict(loaded)

    weights_bump = result.get("weights_bump")
    if not weights_bump:
        bump_path = output_dir / "weights.bump"
        if bump_path.exists():
            result["weights_bump"] = str(bump_path)
    return result


def _ensure_runtime_artifacts(gguf_path: Path, output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "report.json"
    if not report_path.exists():
        parity_harness.run_harness(gguf_path, output_dir)

    runtime_manifest = output_dir / "weights_manifest.runtime.json"
    layout = output_dir / "layout.json"
    ir1 = output_dir / "ir1.json"
    lowered = output_dir / "lowered.json"
    call = output_dir / "call.json"
    manifest_map = output_dir / "weights_manifest.map"

    if not manifest_map.exists():
        rc = build_ir_v8.main(
            [
                "--manifest", str(runtime_manifest),
                "--mode", "prefill",
                "--output", str(ir1),
                "--layout-output", str(layout),
                "--lowered-output", str(lowered),
                "--call-output", str(call),
                "--manifest-map-output", str(manifest_map),
            ]
        )
        if rc != 0:
            raise RuntimeError(f"build_ir_v8 failed with rc={rc}")

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    return _load_runtime_metadata(report, output_dir)


def _compile_generated_model(output_dir: Path) -> Path:
    so_path = output_dir / "libqwen3vl_mmproj_v8.so"
    c_path = output_dir / "qwen3_vl_mmproj_v8.c"
    if so_path.exists() and so_path.stat().st_mtime >= c_path.stat().st_mtime:
        return so_path

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
    _run(cmd)
    return so_path


def _compile_mtmd_shim(output_dir: Path) -> Path:
    shim_src = V8_TOOLS / "mtmd_clip_shim.cpp"
    shim_so = output_dir / "libmtmd_clip_shim.so"
    if shim_so.exists() and shim_so.stat().st_mtime >= shim_src.stat().st_mtime:
        return shim_so

    cmd = [
        "g++",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++17",
        "-Illama.cpp/tools/mtmd",
        "-Illama.cpp/ggml/include",
        "-Illama.cpp/include",
        str(shim_src),
        "-Lllama.cpp/build/bin",
        "-lmtmd",
        f"-Wl,-rpath,{LLAMA_CPP_ROOT / 'build' / 'bin'}",
        "-o",
        str(shim_so),
    ]
    _run(cmd)
    return shim_so


def _load_activation_offsets(layout_path: Path) -> dict[str, dict[str, Any]]:
    with layout_path.open("r", encoding="utf-8") as f:
        layout = json.load(f)
    out: dict[str, dict[str, Any]] = {}
    for buf in layout["memory"]["activations"]["buffers"]:
        out[str(buf["name"])] = buf
    return out


def _load_layout(layout_path: Path) -> dict[str, Any]:
    with layout_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _activation_runtime_base(layout: dict[str, Any]) -> int:
    weights = layout.get("memory", {}).get("weights", {})
    # layout.json stores activation offsets relative to the activation arena,
    # while generated C indexes from g_model->bump after the loaded weights.
    return int(weights.get("base_offset", 0)) + int(weights.get("size", 0))


def _activation_runtime_offset(layout: dict[str, Any], buf: dict[str, Any]) -> int:
    return _activation_runtime_base(layout) + int(buf["offset"])


def _buffer_nbytes(buf: dict[str, Any]) -> int:
    return int(buf.get("size_bytes", buf.get("size", 0)))


def _build_test_image(height: int, width: int, mode: str) -> tuple[list[float], list[float]]:
    interleaved = [0.0] * (height * width * 3)
    planar = [0.0] * (height * width * 3)

    for y in range(height):
        yf = y / max(1, height - 1)
        for x in range(width):
            xf = x / max(1, width - 1)
            idx = y * width + x
            if mode == "gray":
                rgb = (0.5, 0.5, 0.5)
            elif mode == "checker":
                v = 0.8 if ((x // 32) + (y // 32)) % 2 == 0 else 0.2
                rgb = (v, 1.0 - v * 0.5, 0.3 + 0.4 * yf)
            else:
                rgb = (
                    0.15 + 0.7 * xf,
                    0.10 + 0.6 * yf,
                    0.05 + 0.45 * (0.5 * xf + 0.5 * yf),
                )
            base_i = idx * 3
            interleaved[base_i + 0] = rgb[0]
            interleaved[base_i + 1] = rgb[1]
            interleaved[base_i + 2] = rgb[2]
            planar[idx] = rgb[0]
            planar[height * width + idx] = rgb[1]
            planar[2 * height * width + idx] = rgb[2]
    return interleaved, planar


def _load_image_file(image_path: Path, height: int, width: int) -> dict[str, Any]:
    if Image is None:
        raise RuntimeError("Pillow is required for --image-path support")
    if not image_path.exists():
        raise FileNotFoundError(f"image file not found: {image_path}")

    with Image.open(image_path) as src:
        source_width, source_height = src.size
        rgb = src.convert("RGB")
        if rgb.size != (width, height):
            if hasattr(Image, "Resampling"):
                rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
            else:  # pragma: no cover - compatibility with older Pillow.
                rgb = rgb.resize((width, height), Image.BILINEAR)
        pixels = list(rgb.getdata())

    interleaved = [0.0] * (height * width * 3)
    planar = [0.0] * (height * width * 3)
    for idx, (r, g, b) in enumerate(pixels):
        rf = float(r) / 255.0
        gf = float(g) / 255.0
        bf = float(b) / 255.0
        base_i = idx * 3
        interleaved[base_i + 0] = rf
        interleaved[base_i + 1] = gf
        interleaved[base_i + 2] = bf
        planar[idx] = rf
        planar[height * width + idx] = gf
        planar[2 * height * width + idx] = bf
    return {
        "interleaved": interleaved,
        "planar": planar,
        "image_source": "file",
        "image_path": str(image_path.resolve()),
        "source_image_size": [source_width, source_height],
        "preprocess": "rgb_bilinear_resize_to_square_0_1",
    }


def _load_generated_lib(model_so: Path) -> ctypes.CDLL:
    ctypes.CDLL(str(BUILD_DIR / "libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(model_so))
    lib.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.ck_model_init_with_manifest.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.c_void_p]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None
    declare_named_activation_api(lib)
    return lib


def _load_ggml_cpu_global() -> Path | None:
    ggml_libs = [
        [
            LLAMA_CPP_ROOT / "build" / "bin" / "libggml-base.so.0.9.8",
            LLAMA_CPP_ROOT / "build" / "bin" / "libggml-base.so",
        ],
        [
            LLAMA_CPP_ROOT / "build" / "bin" / "libggml.so.0.9.8",
            LLAMA_CPP_ROOT / "build" / "bin" / "libggml.so",
        ],
        [
            LLAMA_CPP_ROOT / "build" / "bin" / "libggml-cpu.so.0.9.8",
            LLAMA_CPP_ROOT / "build" / "bin" / "libggml-cpu.so",
        ],
    ]
    cpu_path: Path | None = None
    for candidates in ggml_libs:
        for path in candidates:
            if path.exists():
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                if "libggml-cpu" in path.name:
                    cpu_path = path
                break
    if cpu_path is not None:
        return cpu_path
    return None


def _load_mtmd_shim(shim_so: Path) -> ctypes.CDLL:
    ctypes.CDLL(str(LLAMA_CPP_ROOT / "build" / "bin" / "libmtmd.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(shim_so))
    lib.ck_mtmd_clip_init.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.ck_mtmd_clip_init.restype = ctypes.c_void_p
    lib.ck_mtmd_clip_free.argtypes = [ctypes.c_void_p]
    lib.ck_mtmd_clip_free.restype = None
    lib.ck_mtmd_clip_n_mmproj_embd.argtypes = [ctypes.c_void_p]
    lib.ck_mtmd_clip_n_mmproj_embd.restype = ctypes.c_int
    lib.ck_mtmd_clip_embd_nbytes_by_img.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.ck_mtmd_clip_embd_nbytes_by_img.restype = ctypes.c_size_t
    lib.ck_mtmd_clip_encode_float_image.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ck_mtmd_clip_encode_float_image.restype = ctypes.c_int
    return lib


def _read_named_llama_dump_tensor(dump_path: Path, op_name: str) -> array:
    dumps = parity_test.read_dump_file(dump_path)
    matches = [dump for dump in dumps if dump.op_name == op_name]
    if not matches:
        available = sorted({dump.op_name for dump in dumps})
        raise RuntimeError(
            f"llama parity dump {dump_path} missing {op_name}; "
            f"available ops: {', '.join(available) if available else '(none)'}"
        )
    flat = matches[-1].data.reshape(-1)
    return array("f", (float(v) for v in flat))


def _normalize_output_name(name: str | None) -> str:
    value = str(name or "auto").strip()
    return value or "auto"


def _llama_reference_output_name(config: dict[str, Any]) -> str | None:
    projector_out = int(config.get("projector_out_dim", config.get("projection_dim", 0)) or 0)
    projector_total = int(config.get("projector_total_out_dim", projector_out) or 0)
    # Qwen3-VL public mtmd image encode already returns the stitched
    # multimodal prefix that includes deepstack slices. Compare that full
    # bridge tensor directly instead of regressing to the legacy 4096-wide seam.
    if projector_out > 0 and projector_total > projector_out:
        return None
    return None


def _resolve_llama_reference_output_name(config: dict[str, Any], output_name: str | None = None) -> str | None:
    requested = _normalize_output_name(output_name)
    if requested == "auto":
        return _llama_reference_output_name(config)
    if requested in {"clip_encode_float_image", "public"}:
        return None
    return requested


def _resolve_ck_output_contract(
    layout: dict[str, Any],
    offsets: dict[str, dict[str, Any]],
    output_name: str | None = None,
) -> dict[str, Any]:
    requested = _normalize_output_name(output_name)
    if requested == "auto":
        contract = dict(resolve_vision_bridge_contract(layout, offsets, prefer_total_output=True))
        contract["requested_output"] = requested
        contract["resolved_output"] = str(
            contract.get("named_activation") or contract.get("fallback_buffer_name") or ""
        )
        return contract

    if requested == "vision_bridge_output":
        contract = dict(resolve_vision_bridge_contract(layout, offsets, prefer_total_output=False))
        if str(contract.get("named_activation") or "") != "vision_bridge_output":
            raise RuntimeError("vision_bridge_output is not available for this encoder layout")
        contract["requested_output"] = requested
        contract["resolved_output"] = requested
        return contract

    contract = {
        "named_activation": requested,
        "fallback_buffer_name": requested if requested in offsets else "",
        "used_nbytes": int(_buffer_nbytes(offsets.get(requested)) if requested in offsets else 0),
        "reason": f"explicit:{requested}",
        "requested_output": requested,
        "resolved_output": requested,
    }
    return contract


def _run_generated_encoder(
    model_so: Path,
    weights_bump: Path,
    manifest_map: Path,
    layout_path: Path,
    planar_image: list[float],
    strict_parity: bool = False,
    strict_mtmd_oracle: bool = False,
    gguf_path: Path | None = None,
    shim_so: Path | None = None,
    output_name: str | None = None,
) -> array:
    lib = _load_generated_lib(model_so)
    restore_env: dict[str, str | None] = {}
    if strict_mtmd_oracle:
        if gguf_path is None or shim_so is None:
            raise RuntimeError("strict mtmd oracle requested without gguf/shim paths")
        env_updates = {
            "CK_STRICT_MTMD_CLIP_ORACLE": "1",
            "CK_STRICT_GGUF_PATH": str(gguf_path),
            "CK_STRICT_MTMD_SHIM_SO": str(shim_so),
        }
        for key, value in env_updates.items():
            restore_env[key] = os.environ.get(key)
            os.environ[key] = value
    strict_mode = bool(strict_parity or strict_mtmd_oracle)
    if strict_mode and not strict_mtmd_oracle:
        ggml_cpu_path = _load_ggml_cpu_global()
        if ggml_cpu_path is None:
            raise RuntimeError("strict parity requested, but libggml-cpu was not found")
        if "CK_GGML_CPU_SO" not in os.environ:
            restore_env["CK_GGML_CPU_SO"] = os.environ.get("CK_GGML_CPU_SO")
            os.environ["CK_GGML_CPU_SO"] = str(ggml_cpu_path)
    if strict_mode:
        lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
        lib.ck_set_strict_parity.restype = None
        lib.ck_set_strict_parity(1)
    else:
        lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
        lib.ck_set_strict_parity.restype = None
        lib.ck_set_strict_parity(0)
    rc = lib.ck_model_init_with_manifest(str(weights_bump).encode(), str(manifest_map).encode())
    if rc != 0:
        raise RuntimeError(f"ck_model_init_with_manifest failed with rc={rc}")

    try:
        layout = _load_layout(layout_path)
        offsets = _load_activation_offsets(layout_path)
        bridge = _resolve_ck_output_contract(layout, offsets, output_name)
        image_buf = offsets["image_input"]
        base_ptr = int(lib.ck_model_get_base_ptr())
        if base_ptr == 0:
            raise RuntimeError("ck_model_get_base_ptr returned null")

        image_len = _buffer_nbytes(image_buf) // ctypes.sizeof(ctypes.c_float)
        if len(planar_image) != image_len:
            raise RuntimeError(f"planar image length mismatch: {len(planar_image)} != {image_len}")

        image_arr = (ctypes.c_float * image_len).from_address(
            base_ptr + _activation_runtime_offset(layout, image_buf)
        )
        image_arr[:] = planar_image

        rc = lib.ck_model_decode(0, None)
        if rc != 0:
            raise RuntimeError(f"ck_model_decode failed with rc={rc}")

        output_ptr = 0
        output_nbytes = 0
        bridge_name = str(bridge.get("named_activation") or "")
        if bridge_name:
            named_view = try_named_activation_view(lib, bridge_name)
            if named_view is not None:
                output_ptr, output_nbytes = named_view
        if output_ptr == 0 or output_nbytes <= 0:
            output_buf = offsets[str(bridge["fallback_buffer_name"])]
            output_ptr = base_ptr + _activation_runtime_offset(layout, output_buf)
            output_nbytes = int(bridge["used_nbytes"])

        output_len = output_nbytes // ctypes.sizeof(ctypes.c_float)
        if output_ptr == 0 or output_len <= 0:
            raise RuntimeError(f"failed to resolve CK encoder output {bridge.get('resolved_output') or bridge.get('fallback_buffer_name')}")
        output_arr = (ctypes.c_float * output_len).from_address(output_ptr)
        return array("f", output_arr)
    finally:
        lib.ck_model_free()
        for key, prior in restore_env.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _run_llamacpp_encoder(
    shim_so: Path,
    gguf_path: Path,
    interleaved_image: list[float],
    height: int,
    width: int,
    n_threads: int,
    named_dump_output: str | None = None,
) -> array:
    lib = _load_mtmd_shim(shim_so)
    dump_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
    dump_path: Path | None = None
    restore_env: dict[str, str | None] = {}
    if named_dump_output is not None:
        dump_dir_ctx = tempfile.TemporaryDirectory(prefix="v8_llama_mmproj_dump_")
        dump_path = Path(dump_dir_ctx.name) / "dump.bin"
        restore_env["CK_LLAMA_PARITY_DIR"] = _with_env_var("CK_LLAMA_PARITY_DIR", str(Path(dump_dir_ctx.name)))
        restore_env["CK_LLAMA_PARITY_ALL"] = _with_env_var("CK_LLAMA_PARITY_ALL", None)
        restore_env["CK_LLAMA_PARITY_NAMES"] = _with_env_var("CK_LLAMA_PARITY_NAMES", named_dump_output)
        restore_env["CK_LLAMA_PARITY_LAYER"] = _with_env_var("CK_LLAMA_PARITY_LAYER", None)
    ctx = lib.ck_mtmd_clip_init(str(gguf_path).encode(), 0, 0, 0, 0, 0)
    if not ctx:
        for key, prior in restore_env.items():
            _restore_env_var(key, prior)
        if dump_dir_ctx is not None:
            dump_dir_ctx.cleanup()
        raise RuntimeError("ck_mtmd_clip_init returned null")

    try:
        n_embd = int(lib.ck_mtmd_clip_n_mmproj_embd(ctx))
        nbytes = int(lib.ck_mtmd_clip_embd_nbytes_by_img(ctx, width, height))
        out_len = nbytes // ctypes.sizeof(ctypes.c_float)
        if n_embd <= 0 or out_len <= 0:
            raise RuntimeError(f"invalid llama.cpp output shape: n_embd={n_embd} out_len={out_len}")

        image_arr = (ctypes.c_float * len(interleaved_image))(*interleaved_image)
        out_arr = (ctypes.c_float * out_len)()
        ok = lib.ck_mtmd_clip_encode_float_image(ctx, n_threads, image_arr, height, width, out_arr)
        if ok != 1:
            raise RuntimeError("ck_mtmd_clip_encode_float_image failed")
        if dump_path is not None:
            if not dump_path.exists():
                raise RuntimeError(f"expected llama parity dump at {dump_path}")
            return _read_named_llama_dump_tensor(dump_path, named_dump_output or "")
        return array("f", out_arr)
    finally:
        lib.ck_mtmd_clip_free(ctx)
        for key, prior in restore_env.items():
            _restore_env_var(key, prior)
        if dump_dir_ctx is not None:
            dump_dir_ctx.cleanup()


def _metrics(ref: array, got: array) -> dict[str, float]:
    if len(ref) != len(got):
        raise RuntimeError(f"length mismatch: ref={len(ref)} got={len(got)}")
    if not ref:
        return {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "rmse": 0.0,
            "cosine": 1.0,
        }

    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    dot = 0.0
    ref_sq = 0.0
    got_sq = 0.0

    for a, b in zip(ref, got):
        d = b - a
        ad = abs(d)
        sum_abs += ad
        sum_sq += d * d
        max_abs = max(max_abs, ad)
        dot += a * b
        ref_sq += a * a
        got_sq += b * b

    denom = math.sqrt(ref_sq) * math.sqrt(got_sq)
    cosine = dot / denom if denom > 0.0 else 0.0
    return {
        "max_abs": max_abs,
        "mean_abs": sum_abs / len(ref),
        "rmse": math.sqrt(sum_sq / len(ref)),
        "cosine": cosine,
    }


def _sample_diffs(ref: array, got: array, count: int = 8) -> list[dict[str, float]]:
    heap: list[tuple[float, int, float, float]] = []
    for idx, (a, b) in enumerate(zip(ref, got)):
        d = abs(b - a)
        item = (d, idx, float(a), float(b))
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif d > heap[0][0]:
            heapq.heapreplace(heap, item)
    diffs = sorted(heap, reverse=True)
    out = []
    for d, idx, a, b in diffs:
        out.append({
            "index": idx,
            "ref": a,
            "got": b,
            "abs_diff": d,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Numeric parity for v8 Qwen3-VL mmproj encoder vs local llama.cpp")
    ap.add_argument("--gguf", type=Path, required=True, help="Path to mmproj-Qwen3VL-*.gguf")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/qwen3vl_mmproj_v8_numeric"), help="Workspace for generated artifacts")
    ap.add_argument("--image-mode", choices=("gradient", "gray", "checker"), default="gradient")
    ap.add_argument("--image-path", type=Path, default=None, help="Optional real image path; overrides --image-mode")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ck-threads", type=int, default=None, help="Thread count for generated CK runtime; defaults to --threads")
    ap.add_argument("--strict-parity", action="store_true", help="Enable parity-only strict mode in CK and load ggml CPU helpers for full-attention replay")
    ap.add_argument("--strict-mtmd-oracle", action="store_true", help="In strict mode, allow the generated CK vision runtime to short-circuit to the local mtmd/ggml encoder oracle")
    ap.add_argument("--ck-output-name", type=str, default="auto", help="CK encoder output to compare: auto, vision_output, vision_bridge_output, embedded_input, or another named activation/buffer")
    ap.add_argument("--llama-output-name", type=str, default="auto", help="llama.cpp reference output to compare: auto, clip_encode_float_image, projector_out, vision_output, or another dumped tensor name")
    ap.add_argument("--report", type=Path, default=None, help="Optional JSON report output")
    args = ap.parse_args(argv)

    ck_threads = int(args.ck_threads or args.threads)
    os.environ["OMP_NUM_THREADS"] = str(ck_threads)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    _ensure_engine_lib()
    report = _ensure_runtime_artifacts(args.gguf, output_dir)
    model_so = _compile_generated_model(output_dir)
    shim_so = _compile_mtmd_shim(output_dir)
    t_artifacts = time.perf_counter()

    config = report["config"]
    height = int(config["image_size"])
    width = int(config["image_size"])
    layout_path = output_dir / "layout.json"
    layout = _load_layout(layout_path)
    offsets = _load_activation_offsets(layout_path)
    ck_output_contract = _resolve_ck_output_contract(layout, offsets, args.ck_output_name)
    ck_resolved_output = str(
        ck_output_contract.get("resolved_output") or ck_output_contract.get("fallback_buffer_name") or ""
    )
    llama_reference_output = _resolve_llama_reference_output_name(config, args.llama_output_name)
    if args.image_path is not None:
        image_report = _load_image_file(args.image_path.resolve(), height, width)
        interleaved = image_report["interleaved"]
        planar = image_report["planar"]
    else:
        interleaved, planar = _build_test_image(height, width, args.image_mode)
        image_report = {
            "image_source": "synthetic",
            "image_mode": args.image_mode,
            "image_path": None,
            "source_image_size": [width, height],
            "preprocess": "synthetic_generator",
        }
    t_image = time.perf_counter()

    ck_out = _run_generated_encoder(
        model_so=model_so,
        weights_bump=Path(report["weights_bump"]),
        manifest_map=output_dir / "weights_manifest.map",
        layout_path=output_dir / "layout.json",
        planar_image=planar,
        strict_parity=args.strict_parity,
        strict_mtmd_oracle=args.strict_mtmd_oracle,
        gguf_path=args.gguf,
        shim_so=shim_so,
        output_name=args.ck_output_name,
    )
    t_ck = time.perf_counter()
    llama_out = _run_llamacpp_encoder(
        shim_so=shim_so,
        gguf_path=args.gguf,
        interleaved_image=interleaved,
        height=height,
        width=width,
        n_threads=args.threads,
        named_dump_output=llama_reference_output,
    )
    t_llama = time.perf_counter()

    metrics = _metrics(llama_out, ck_out)
    t_metrics = time.perf_counter()
    lowering = report.get("lowering", {}) if isinstance(report, dict) else {}
    notes = [
        "llama.cpp reference uses clip_encode_float_image from libmtmd via a local C shim.",
    ]
    if _normalize_output_name(args.ck_output_name) == "auto":
        notes.append(
            "CK output is read from the resolved full vision bridge activation so Qwen3-VL compares the stitched decoder-facing prefix tensor."
        )
    else:
        notes.append(f"CK output is read from the explicit {ck_resolved_output} seam.")
    if llama_reference_output is not None:
        notes.append(
            f"llama.cpp numeric parity is pinned to the dumped {llama_reference_output} seam instead of the public image-encode output."
        )
    if not lowering.get("has_vision_mrope", False):
        notes.append("Vision multi-section RoPE is still not lowered in the current v8 path, so parity is expected to fail.")
    result = {
        "gguf": str(args.gguf),
        "output_dir": str(output_dir),
        "image_source": str(image_report["image_source"]),
        "image_mode": image_report.get("image_mode"),
        "image_path": image_report.get("image_path"),
        "source_image_size": image_report.get("source_image_size"),
        "preprocess": str(image_report["preprocess"]),
        "height": height,
        "width": width,
        "threads": {
            "llama_cpp": args.threads,
            "ck_runtime": ck_threads,
        },
        "strict_parity": bool(args.strict_parity),
        "strict_mtmd_oracle": bool(args.strict_mtmd_oracle),
        "ck_output_name": _normalize_output_name(args.ck_output_name),
        "ck_resolved_output": ck_resolved_output,
        "llama_requested_output": _normalize_output_name(args.llama_output_name),
        "llama_reference_output": llama_reference_output or "clip_encode_float_image",
        "num_values": len(llama_out),
        "metrics": metrics,
        "top_diffs": _sample_diffs(llama_out, ck_out),
        "timings_sec": {
            "artifacts": t_artifacts - t0,
            "image": t_image - t_artifacts,
            "ck_encode": t_ck - t_image,
            "llama_encode": t_llama - t_ck,
            "metrics": t_metrics - t_llama,
            "total": t_metrics - t0,
        },
        "notes": notes,
    }

    if args.report is not None:
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
