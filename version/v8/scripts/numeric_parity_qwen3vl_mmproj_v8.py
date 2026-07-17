#!/usr/bin/env python3
from __future__ import annotations

import argparse
from array import array
import ctypes
import hashlib
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
LLAMA_CPP_ROOT = Path(os.environ.get("CK_LLAMA_CPP_ROOT", str(REPO_ROOT / "llama.cpp"))).resolve()
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
from run_multimodal_bridge_v8 import _qwen3vl_geometry_overrides  # type: ignore  # noqa: E402


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
    _run(["make", "CK_ENABLE_OPENMP=1", "build/libckernel_engine.so"])


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


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _apply_qwen3vl_geometry_to_runtime_manifest(
    output_dir: Path,
    image_path: Path | None,
    image_min_tokens: int | None,
    image_max_tokens: int | None,
) -> bool:
    if image_path is None:
        return False
    if (image_min_tokens is None or int(image_min_tokens) <= 0) and (image_max_tokens is None or int(image_max_tokens) <= 0):
        return False
    runtime_manifest_path = output_dir / "weights_manifest.runtime.json"
    config_path = output_dir / "config.json"
    report_path = output_dir / "report.json"
    if not runtime_manifest_path.exists():
        return False
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    cfg = runtime_manifest.get("config")
    if not isinstance(cfg, dict):
        cfg = {}
        runtime_manifest["config"] = cfg
    overrides = _qwen3vl_geometry_overrides(
        dict(cfg),
        image_path,
        image_min_tokens=image_min_tokens,
        image_max_tokens=image_max_tokens,
    )
    changed = any(cfg.get(k) != v for k, v in overrides.items())
    if not changed:
        return False
    cfg.update(overrides)
    _write_json(runtime_manifest_path, runtime_manifest)
    if config_path.exists():
        config_obj = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config_obj, dict):
            config_obj.update(overrides)
            _write_json(config_path, config_obj)
    if report_path.exists():
        report_obj = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(report_obj, dict):
            report_cfg = report_obj.get("config")
            if isinstance(report_cfg, dict):
                report_cfg.update(overrides)
            _write_json(report_path, report_obj)
    for name in (
        "ir1.json",
        "layout.json",
        "lowered.json",
        "call.json",
        "weights_manifest.map",
        "qwen3_vl_mmproj_v8.c",
        "libqwen3vl_mmproj_v8.so",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
    return True



def _invalidate_generated_runtime_artifacts(output_dir: Path) -> None:
    for name in (
        "ir1.json",
        "layout.json",
        "lowered.json",
        "call.json",
        "weights_manifest.map",
        "qwen3_vl_mmproj_v8.c",
        "libqwen3vl_mmproj_v8.so",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def _parse_activation_preference_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"invalid activation preference override {item!r}; expected op=dtype")
        op_name, pref = item.split("=", 1)
        op_name = op_name.strip()
        pref = pref.strip().lower()
        if not op_name or not pref:
            raise ValueError(f"invalid activation preference override {item!r}; expected op=dtype")
        overrides[op_name] = pref
    return overrides


def _apply_activation_preferences_to_runtime_manifest(
    output_dir: Path,
    overrides: dict[str, str] | None,
) -> bool:
    if not overrides:
        return False
    runtime_manifest_path = output_dir / "weights_manifest.runtime.json"
    config_path = output_dir / "config.json"
    report_path = output_dir / "report.json"
    if not runtime_manifest_path.exists():
        return False

    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    cfg = runtime_manifest.get("config")
    if not isinstance(cfg, dict):
        cfg = {}
        runtime_manifest["config"] = cfg
    prefs = cfg.get("activation_preference_by_op")
    if not isinstance(prefs, dict):
        prefs = {}
    else:
        prefs = dict(prefs)

    changed = False
    for op_name, pref in overrides.items():
        if prefs.get(op_name) != pref:
            prefs[op_name] = pref
            changed = True
    if not changed:
        return False
    cfg["activation_preference_by_op"] = prefs
    _write_json(runtime_manifest_path, runtime_manifest)

    if config_path.exists():
        config_obj = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config_obj, dict):
            config_prefs = config_obj.get("activation_preference_by_op")
            if not isinstance(config_prefs, dict):
                config_prefs = {}
            else:
                config_prefs = dict(config_prefs)
            config_prefs.update(overrides)
            config_obj["activation_preference_by_op"] = config_prefs
            _write_json(config_path, config_obj)
    if report_path.exists():
        report_obj = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(report_obj, dict):
            report_cfg = report_obj.get("config")
            if isinstance(report_cfg, dict):
                report_prefs = report_cfg.get("activation_preference_by_op")
                if not isinstance(report_prefs, dict):
                    report_prefs = {}
                else:
                    report_prefs = dict(report_prefs)
                report_prefs.update(overrides)
                report_cfg["activation_preference_by_op"] = report_prefs
            _write_json(report_path, report_obj)
    _invalidate_generated_runtime_artifacts(output_dir)
    return True


def _apply_runtime_config_overrides_to_runtime_manifest(
    output_dir: Path,
    overrides: dict[str, Any] | None,
) -> bool:
    if not overrides:
        return False
    runtime_manifest_path = output_dir / "weights_manifest.runtime.json"
    config_path = output_dir / "config.json"
    report_path = output_dir / "report.json"
    if not runtime_manifest_path.exists():
        return False

    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    cfg = runtime_manifest.get("config")
    if not isinstance(cfg, dict):
        cfg = {}
        runtime_manifest["config"] = cfg

    changed = False
    for key, value in overrides.items():
        if cfg.get(key) != value:
            cfg[key] = value
            changed = True
    if not changed:
        return False
    _write_json(runtime_manifest_path, runtime_manifest)

    if config_path.exists():
        config_obj = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config_obj, dict):
            for key, value in overrides.items():
                config_obj[key] = value
            _write_json(config_path, config_obj)
    if report_path.exists():
        report_obj = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(report_obj, dict):
            report_cfg = report_obj.get("config")
            if isinstance(report_cfg, dict):
                for key, value in overrides.items():
                    report_cfg[key] = value
            _write_json(report_path, report_obj)
    _invalidate_generated_runtime_artifacts(output_dir)
    return True


def _ensure_runtime_artifacts(
    gguf_path: Path,
    output_dir: Path,
    image_path: Path | None = None,
    image_min_tokens: int | None = None,
    image_max_tokens: int | None = None,
    activation_preferences: dict[str, str] | None = None,
    runtime_config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report_path = output_dir / "report.json"
    if not report_path.exists():
        parity_harness.run_harness(gguf_path, output_dir)

    runtime_manifest = output_dir / "weights_manifest.runtime.json"
    layout = output_dir / "layout.json"
    ir1 = output_dir / "ir1.json"
    lowered = output_dir / "lowered.json"
    call = output_dir / "call.json"
    manifest_map = output_dir / "weights_manifest.map"

    _apply_qwen3vl_geometry_to_runtime_manifest(output_dir, image_path, image_min_tokens, image_max_tokens)
    _apply_activation_preferences_to_runtime_manifest(output_dir, activation_preferences)
    _apply_runtime_config_overrides_to_runtime_manifest(output_dir, runtime_config_overrides)

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

    c_path = output_dir / "qwen3_vl_mmproj_v8.c"
    if not c_path.exists() or c_path.stat().st_mtime < call.stat().st_mtime:
        _run([
            sys.executable,
            str(SCRIPT_DIR / "codegen_v8.py"),
            "--ir", str(call),
            "--layout", str(layout),
            "--output", str(c_path),
        ])

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
    stamp_path = output_dir / "libmtmd_clip_shim.so.build.json"

    clip_header = (LLAMA_CPP_ROOT / "tools" / "mtmd" / "clip.h").read_text(
        encoding="utf-8", errors="ignore"
    )
    clip_impl_header = (LLAMA_CPP_ROOT / "tools" / "mtmd" / "clip-impl.h").read_text(
        encoding="utf-8", errors="ignore"
    )
    uses_object_api = "clip_embd_nbytes_by_img" not in clip_header
    uses_value_api = uses_object_api and "clip_image_f32_ptr" not in (
        clip_header + clip_impl_header
    )
    llama_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(LLAMA_CPP_ROOT),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    build_fingerprint = {
        "version": 1,
        "llama_cpp_root": str(LLAMA_CPP_ROOT),
        "llama_cpp_commit": llama_commit,
        "shim_sha256": hashlib.sha256(shim_src.read_bytes()).hexdigest(),
        "clip_header_sha256": hashlib.sha256(clip_header.encode()).hexdigest(),
        "clip_impl_header_sha256": hashlib.sha256(clip_impl_header.encode()).hexdigest(),
        "object_api": uses_object_api,
        "value_api": uses_value_api,
    }
    if shim_so.is_file() and stamp_path.is_file():
        try:
            if json.loads(stamp_path.read_text(encoding="utf-8")) == build_fingerprint:
                return shim_so
        except (OSError, json.JSONDecodeError):
            pass

    cmd = [
        "g++",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++17",
        *(["-DCK_MTMD_CLIP_OBJECT_API=1"] if uses_object_api else []),
        *(["-DCK_MTMD_CLIP_VALUE_API=1"] if uses_value_api else []),
        f"-I{LLAMA_CPP_ROOT / 'tools' / 'mtmd'}",
        f"-I{LLAMA_CPP_ROOT / 'ggml' / 'include'}",
        f"-I{LLAMA_CPP_ROOT / 'include'}",
        str(shim_src),
        f"-L{LLAMA_CPP_ROOT / 'build' / 'bin'}",
        "-lmtmd",
        f"-Wl,-rpath,{LLAMA_CPP_ROOT / 'build' / 'bin'}",
        "-o",
        str(shim_so),
    ]
    _run(cmd)
    stamp_path.write_text(json.dumps(build_fingerprint, indent=2) + "\n", encoding="utf-8")
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



def _ppm_skip_ws_and_comments(data: bytes, idx: int) -> int:
    while idx < len(data):
        b = data[idx]
        if b in b" \t\r\n":
            idx += 1
            continue
        if b == ord("#"):
            while idx < len(data) and data[idx] not in b"\r\n":
                idx += 1
            continue
        break
    return idx


def _ppm_next_token(data: bytes, idx: int) -> tuple[str, int]:
    idx = _ppm_skip_ws_and_comments(data, idx)
    start = idx
    while idx < len(data) and data[idx] not in b" \t\r\n#":
        idx += 1
    if idx == start:
        raise ValueError("malformed PPM header")
    return data[start:idx].decode("ascii"), idx


def _read_ppm_rgb8(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    data = path.read_bytes()
    magic, idx = _ppm_next_token(data, 0)
    width_s, idx = _ppm_next_token(data, idx)
    height_s, idx = _ppm_next_token(data, idx)
    maxval_s, idx = _ppm_next_token(data, idx)
    width = int(width_s)
    height = int(height_s)
    maxval = int(maxval_s)
    if width <= 0 or height <= 0 or maxval <= 0 or maxval > 255:
        raise ValueError(f"unsupported PPM shape/maxval: {width}x{height} max={maxval}")
    pixels: list[tuple[int, int, int]] = []
    if magic == "P6":
        idx = _ppm_skip_ws_and_comments(data, idx)
        if idx < len(data) and data[idx] in b" \t\r\n":
            idx += 1
        payload = data[idx:]
        expected = width * height * 3
        if len(payload) < expected:
            raise ValueError(f"PPM payload too short: {len(payload)} < {expected}")
        for i in range(0, expected, 3):
            pixels.append((payload[i], payload[i + 1], payload[i + 2]))
    elif magic == "P3":
        for _ in range(width * height):
            r, idx = _ppm_next_token(data, idx)
            g, idx = _ppm_next_token(data, idx)
            b, idx = _ppm_next_token(data, idx)
            pixels.append((int(r), int(g), int(b)))
    else:
        raise ValueError(f"unsupported PPM magic: {magic}")
    if maxval != 255:
        pixels = [tuple(int(round(c * 255.0 / maxval)) for c in px) for px in pixels]
    return width, height, pixels


def _resize_pixels_nearest(pixels: list[tuple[int, int, int]], src_w: int, src_h: int, dst_w: int, dst_h: int) -> list[tuple[int, int, int]]:
    if (src_w, src_h) == (dst_w, dst_h):
        return pixels
    out: list[tuple[int, int, int]] = []
    for y in range(dst_h):
        sy = min(src_h - 1, int((y + 0.5) * src_h / dst_h))
        row = sy * src_w
        for x in range(dst_w):
            sx = min(src_w - 1, int((x + 0.5) * src_w / dst_w))
            out.append(pixels[row + sx])
    return out

def _load_image_file(image_path: Path, height: int, width: int) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"image file not found: {image_path}")

    if image_path.suffix.lower() == ".ppm":
        source_width, source_height, pixels = _read_ppm_rgb8(image_path)
        pixels = _resize_pixels_nearest(pixels, source_width, source_height, width, height)
    else:
        if Image is None:
            raise RuntimeError("Pillow is required for non-PPM --image-path support")
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


def _parse_named_dump_selector(selector: str) -> tuple[str, int | None]:
    name = str(selector).strip()
    if "@" not in name:
        return name, None
    op_name, layer_text = name.rsplit("@", 1)
    op_name = op_name.strip()
    layer_text = layer_text.strip()
    if not op_name or not layer_text:
        raise RuntimeError(f"invalid llama dump selector {selector!r}; expected op or op@layer")
    try:
        layer_id = int(layer_text)
    except ValueError as exc:
        raise RuntimeError(f"invalid llama dump selector {selector!r}; layer must be an integer") from exc
    return op_name, layer_id


def _llama_raw_dump_name(op_name: str) -> str:
    reverse_aliases = {
        "q_proj": "Qcur",
        "k_proj": "Kcur",
        "v_proj": "Vcur",
        "rope_q": "Qcur_rope",
        "rope_k": "Kcur_rope",
        "attn_out_head_major": "kqv_out",
        "attn_output": "attn_out",
    }
    return reverse_aliases.get(op_name, op_name)


def _read_named_llama_dump_tensor(dump_path: Path, op_name: str) -> array:
    requested_op, requested_layer = _parse_named_dump_selector(op_name)
    dumps = parity_test.read_dump_file(dump_path)
    matches = [
        dump for dump in dumps
        if dump.op_name == requested_op and (requested_layer is None or dump.layer_id == requested_layer)
    ]
    if not matches:
        available = sorted({f"{dump.op_name}@{dump.layer_id}" for dump in dumps})
        suffix = f" at layer {requested_layer}" if requested_layer is not None else ""
        raise RuntimeError(
            f"llama parity dump {dump_path} missing {requested_op}{suffix}; "
            f"available ops: {', '.join(available) if available else '(none)'}"
        )
    flat = matches[-1].data.reshape(-1)
    return array("f", (float(v) for v in flat))


def _ck_hidden_dump_path(hidden_dir: Path, selector: str) -> Path:
    requested_name, requested_layer = _parse_named_dump_selector(selector)
    if requested_layer is None:
        pattern = f"tok_*_layer_*_{requested_name}.f32"
    else:
        pattern = f"tok_*_layer_{requested_layer:03d}_{requested_name}.f32"
    matches = sorted(hidden_dir.glob(pattern))
    if not matches and requested_layer == -1:
        matches = sorted(hidden_dir.glob(f"tok_*_layer_-01_{requested_name}.f32"))
    if not matches:
        available = sorted(path.name for path in hidden_dir.glob("tok_*_layer_*.f32"))
        preview = ", ".join(available[:32])
        if len(available) > 32:
            preview += ", ..."
        suffix = f" at layer {requested_layer}" if requested_layer is not None else ""
        raise RuntimeError(
            f"CK hidden dump missing {requested_name}{suffix}; "
            f"available dumps: {preview if preview else '(none)'}"
        )
    return matches[-1]


def _read_ck_hidden_dump_tensor(hidden_dir: Path, selector: str) -> array:
    dump_path = _ck_hidden_dump_path(hidden_dir, selector)
    data = dump_path.read_bytes()
    if len(data) % ctypes.sizeof(ctypes.c_float) != 0:
        raise RuntimeError(f"CK hidden dump {dump_path} has invalid byte length {len(data)}")
    out = array("f")
    out.frombytes(data)
    return out


def _config_int(config: dict[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        value = config.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _ck_head_major_to_llama_token_major(
    data: array,
    *,
    num_heads: int,
    head_dim: int,
) -> array:
    if num_heads <= 0 or head_dim <= 0:
        return data
    row_width = num_heads * head_dim
    if row_width <= 0 or len(data) % row_width != 0:
        return data
    num_tokens = len(data) // row_width
    out = array("f", [0.0]) * len(data)
    for token in range(num_tokens):
        token_base = token * row_width
        for head in range(num_heads):
            src = (head * num_tokens + token) * head_dim
            dst = token_base + head * head_dim
            out[dst:dst + head_dim] = data[src:src + head_dim]
    return out


def _normalize_ck_hidden_dump_tensor_for_llama(
    data: array,
    selector: str,
    layout: dict[str, Any],
) -> array:
    base_name, _ = _parse_named_dump_selector(selector)
    if base_name not in {"q_proj", "k_proj", "v_proj", "rope_q", "rope_k", "attn_out_head_major"}:
        return data
    config_obj = layout.get("config") if isinstance(layout, dict) else None
    config = config_obj if isinstance(config_obj, dict) else {}
    head_dim = _config_int(config, "head_dim", "aligned_head_dim", "rotary_dim")
    if base_name in {"q_proj", "rope_q", "attn_out_head_major"}:
        heads = _config_int(config, "num_heads", "vision_num_heads")
    else:
        heads = _config_int(config, "num_kv_heads", "vision_num_kv_heads", "num_heads", "vision_num_heads")
    return _ck_head_major_to_llama_token_major(data, num_heads=heads, head_dim=head_dim)


def _normalize_output_name(name: str | None) -> str:
    value = str(name or "auto").strip()
    return value or "auto"


_CK_HIDDEN_EXPORT_OUTPUTS = {
    "after_attn",
    "after_attn_last",
    "after_attn_residual",
    "after_attn_residual_last",
    "ffn_inp_normed_last",
    "ffn_inp_normed",
    "out_proj_last",
    "out_proj",
    "ln1_last",
    "ln1",
    "attn_out",
    "attn_out_last",
    "attn_out_head_major",
    "ffn_norm",
    "ffn_norm_last",
    "ffn_residual",
    "ffn_residual_last",
    "layer_out",
    "layer_out_last",
    "mlp_down",
    "mlp_down_last",
    "mlp_up",
    "mlp_up_last",
    "post_attn_norm",
    "post_attn_norm_last",
    "post_ffn_norm",
    "post_ffn_norm_last",
    "q_proj",
    "k_proj",
    "v_proj",
    "qk_norm_k",
    "qk_norm_q",
    "rope_k",
    "rope_q",
    "vision_patch_bias",
    "vision_patchify",
    "vision_patch_proj",
    "vision_patch_proj_aux",
    "vision_patch_sum",
    "vision_position_embeddings",
    "vision_projector_fc1",
    "vision_projector_fc1_last",
    "vision_projector_out",
    "vision_projector_out_last",
    "vision_projector_prep",
    "vision_projector_prep_last",
    "vision_spatial_merge",
}


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
    hidden_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
    requested_output = _normalize_output_name(output_name)
    requested_base, requested_layer = _parse_named_dump_selector(requested_output)
    if requested_output not in {"auto", "vision_bridge_output"}:
        layout_for_dump = _load_layout(layout_path)
        offsets_for_dump = _load_activation_offsets(layout_path)
        if requested_output not in offsets_for_dump:
            named_contract = _resolve_ck_output_contract(layout_for_dump, offsets_for_dump, requested_output)
            named_view_available = bool(str(named_contract.get("named_activation") or ""))
            # Generated debug seams such as layer_out@24 and vision_projector_out are
            # written to CK_DEBUG_EXPORT_HIDDEN, not exposed through named activations.
            if requested_base in _CK_HIDDEN_EXPORT_OUTPUTS or "@" in requested_output or not named_view_available:
                hidden_dir_ctx = tempfile.TemporaryDirectory(prefix="v8_ck_mmproj_hidden_")
                restore_env["CK_DEBUG_EXPORT_HIDDEN"] = os.environ.get("CK_DEBUG_EXPORT_HIDDEN")
                restore_env["CK_DEBUG_EXPORT_HIDDEN_NAME"] = os.environ.get("CK_DEBUG_EXPORT_HIDDEN_NAME")
                restore_env["CK_DEBUG_EXPORT_HIDDEN_LAYER"] = os.environ.get("CK_DEBUG_EXPORT_HIDDEN_LAYER")
                os.environ["CK_DEBUG_EXPORT_HIDDEN"] = hidden_dir_ctx.name
                os.environ["CK_DEBUG_EXPORT_HIDDEN_NAME"] = requested_base
                if requested_layer is None:
                    os.environ.pop("CK_DEBUG_EXPORT_HIDDEN_LAYER", None)
                else:
                    os.environ["CK_DEBUG_EXPORT_HIDDEN_LAYER"] = str(requested_layer)
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

        if hidden_dir_ctx is not None:
            hidden = _read_ck_hidden_dump_tensor(Path(hidden_dir_ctx.name), requested_output)
            return _normalize_ck_hidden_dump_tensor_for_llama(hidden, requested_output, layout)

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
        if hidden_dir_ctx is not None:
            hidden_dir_ctx.cleanup()


def _run_llamacpp_encoder(
    shim_so: Path,
    gguf_path: Path,
    interleaved_image: list[float],
    height: int,
    width: int,
    n_threads: int,
    named_dump_output: str | None = None,
    image_min_tokens: int | None = None,
    image_max_tokens: int | None = None,
    flash_attn_type: int = 0,
) -> array:
    lib = _load_mtmd_shim(shim_so)
    dump_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
    dump_path: Path | None = None
    restore_env: dict[str, str | None] = {}
    resolved_named_dump_output = named_dump_output
    if named_dump_output is not None:
        dump_op_name, dump_layer = _parse_named_dump_selector(named_dump_output)
        dump_op_name = _llama_raw_dump_name(dump_op_name)
        resolved_named_dump_output = f"{dump_op_name}@{dump_layer}" if dump_layer is not None else dump_op_name
        dump_dir_ctx = tempfile.TemporaryDirectory(prefix="v8_llama_mmproj_dump_")
        dump_path = Path(dump_dir_ctx.name) / "dump.bin"
        restore_env["CK_LLAMA_PARITY_DIR"] = _with_env_var("CK_LLAMA_PARITY_DIR", str(Path(dump_dir_ctx.name)))
        restore_env["CK_LLAMA_PARITY_ALL"] = _with_env_var("CK_LLAMA_PARITY_ALL", None)
        restore_env["CK_LLAMA_PARITY_NAMES"] = _with_env_var("CK_LLAMA_PARITY_NAMES", dump_op_name)
        restore_env["CK_LLAMA_PARITY_LAYER"] = _with_env_var("CK_LLAMA_PARITY_LAYER", None)
    ctx = lib.ck_mtmd_clip_init(
        str(gguf_path).encode(),
        0,
        int(flash_attn_type),
        int(image_min_tokens or 0),
        int(image_max_tokens or 0),
        0,
    )
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




def _sample_row_diffs(ref: array, got: array, row_width: int, count: int = 8) -> list[dict[str, float]]:
    if row_width <= 0 or len(ref) != len(got) or len(ref) % row_width != 0:
        return []
    heap: list[tuple[float, int, float, float, float, int]] = []
    num_rows = len(ref) // row_width
    for row in range(num_rows):
        start = row * row_width
        end = start + row_width
        row_ref = ref[start:end]
        row_got = got[start:end]
        metrics = _metrics(row_ref, row_got)
        item = (
            float(metrics["rmse"]),
            row,
            float(metrics["max_abs"]),
            float(metrics["mean_abs"]),
            float(metrics["cosine"]),
            row_width,
        )
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    rows = sorted(heap, reverse=True)
    return [
        {
            "row": row,
            "row_width": width,
            "rmse": rmse,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "cosine": cosine,
        }
        for rmse, row, max_abs, mean_abs, cosine, width in rows
    ]




def _cosine_similarity(a: array, b: array) -> float:
    dot = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        a_sq += av * av
        b_sq += bv * bv
    denom = math.sqrt(a_sq) * math.sqrt(b_sq)
    return dot / denom if denom > 0.0 else 0.0


def _nearest_rows(
    ref: array,
    got: array,
    row_width: int,
    query_rows: list[int],
    *,
    direction: str,
    count: int = 5,
) -> list[dict[str, Any]]:
    if row_width <= 0 or len(ref) != len(got) or len(ref) % row_width != 0:
        return []
    num_rows = len(ref) // row_width
    out: list[dict[str, Any]] = []
    for query_row in query_rows:
        if query_row < 0 or query_row >= num_rows:
            continue
        if direction == "got_to_ref":
            query = got[query_row * row_width:(query_row + 1) * row_width]
            candidates = ref
            candidate_key = "ref_row"
        elif direction == "ref_to_got":
            query = ref[query_row * row_width:(query_row + 1) * row_width]
            candidates = got
            candidate_key = "got_row"
        else:
            continue
        heap: list[tuple[float, int]] = []
        for row in range(num_rows):
            cand = candidates[row * row_width:(row + 1) * row_width]
            cosine = _cosine_similarity(query, cand)
            item = (cosine, row)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif cosine > heap[0][0]:
                heapq.heapreplace(heap, item)
        matches = [
            {candidate_key: row, "cosine": cosine}
            for cosine, row in sorted(heap, reverse=True)
        ]
        out.append({"query_row": query_row, "direction": direction, "matches": matches})
    return out


def _resolve_feature_slice(
    *,
    config: dict[str, Any],
    length: int,
    side: str,
    row_width: int | None,
    offset: int | None,
    dim: int | None,
    common_row_width: int | None,
    common_slice_index: int | None,
    common_slice_dim: int | None,
) -> dict[str, int | str] | None:
    slice_dim = int(dim or common_slice_dim or config.get("projector_out_dim", config.get("projection_dim", 0)) or 0)
    slice_row_width = int(row_width or common_row_width or config.get("projector_total_out_dim", 0) or 0)
    slice_offset = offset
    if slice_offset is None and common_slice_index is not None and slice_dim > 0:
        slice_offset = int(common_slice_index) * slice_dim
    if slice_offset is None and dim is None and row_width is None and common_slice_index is None:
        return None
    if slice_dim <= 0:
        raise RuntimeError(f"{side} feature slice requested without a positive feature dimension")
    if slice_row_width <= 0:
        raise RuntimeError(f"{side} feature slice requested without a positive row width")
    if slice_offset is None:
        slice_offset = 0
    if slice_offset < 0 or slice_offset + slice_dim > slice_row_width:
        raise RuntimeError(
            f"{side} feature slice out of bounds: offset={slice_offset} dim={slice_dim} row_width={slice_row_width}"
        )
    if length % slice_row_width != 0:
        raise RuntimeError(
            f"{side} feature slice row width {slice_row_width} does not divide tensor length {length}"
        )
    return {
        "side": side,
        "row_width": slice_row_width,
        "offset": int(slice_offset),
        "dim": slice_dim,
        "rows": length // slice_row_width,
    }


def _apply_feature_slice(data: array, spec: dict[str, int | str] | None) -> array:
    if spec is None:
        return data
    row_width = int(spec["row_width"])
    offset = int(spec["offset"])
    dim = int(spec["dim"])
    rows = int(spec["rows"])
    out = array("f")
    for row in range(rows):
        start = row * row_width + offset
        out.extend(data[start:start + dim])
    return out


def _resolve_row_slice(
    *,
    length: int,
    side: str,
    row_index: int | None,
    row_width: int | None,
    common_row_index: int | None,
    common_row_width: int | None,
    row_dim: int | None,
) -> dict[str, int | str] | None:
    selected_index = row_index if row_index is not None else common_row_index
    if selected_index is None:
        return None
    selected_width = int(row_width or common_row_width or 0)
    if selected_width <= 0:
        raise RuntimeError(f"{side} row slice requested without a positive row width")
    if length % selected_width != 0:
        raise RuntimeError(
            f"{side} row slice row width {selected_width} does not divide tensor length {length}"
        )
    rows = length // selected_width
    idx = int(selected_index)
    if idx < 0:
        idx += rows
    if idx < 0 or idx >= rows:
        raise RuntimeError(f"{side} row slice index {selected_index} out of bounds for {rows} rows")
    dim = int(row_dim or selected_width)
    if dim <= 0 or dim > selected_width:
        raise RuntimeError(f"{side} row slice has invalid dim={dim} for row_width={selected_width}")
    return {
        "side": side,
        "row_width": selected_width,
        "row_index": idx,
        "dim": dim,
        "rows": rows,
    }


def _apply_row_slice(data: array, spec: dict[str, int | str] | None) -> array:
    if spec is None:
        return data
    row_width = int(spec["row_width"])
    row_index = int(spec["row_index"])
    dim = int(spec["dim"])
    start = row_index * row_width
    return array("f", data[start:start + dim])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Numeric parity for v8 Qwen3-VL mmproj encoder vs local llama.cpp")
    ap.add_argument("--gguf", type=Path, required=True, help="Path to mmproj-Qwen3VL-*.gguf")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/qwen3vl_mmproj_v8_numeric"), help="Workspace for generated artifacts")
    ap.add_argument("--image-mode", choices=("gradient", "gray", "checker"), default="gradient")
    ap.add_argument("--image-path", type=Path, default=None, help="Optional real image path; overrides --image-mode")
    ap.add_argument("--image-min-tokens", type=int, default=None, help="Override minimum merged visual tokens for dynamic-resolution Qwen3-VL images")
    ap.add_argument("--image-max-tokens", type=int, default=None, help="Override maximum merged visual tokens for dynamic-resolution Qwen3-VL images")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument(
        "--llama-flash-attn",
        choices=("disabled", "auto", "enabled"),
        default="disabled",
        help="Reference attention algorithm; select enabled for production flash-attention parity.",
    )
    ap.add_argument("--ck-threads", type=int, default=None, help="Thread count for generated CK runtime; defaults to --threads")
    ap.add_argument("--activation-pref", action="append", default=[], help="Override generated vision activation preference in op=dtype form; may be repeated")
    ap.add_argument("--strict-parity", action="store_true", help="Enable parity-only strict mode in CK and load ggml CPU helpers for full-attention replay")
    ap.add_argument("--strict-mtmd-oracle", action="store_true", help="In strict mode, allow the generated CK vision runtime to short-circuit to the local mtmd/ggml encoder oracle")
    ap.add_argument("--ck-output-name", type=str, default="auto", help="CK encoder output to compare: auto, vision_output, vision_bridge_output, embedded_input, or another named activation/buffer")
    ap.add_argument("--llama-output-name", type=str, default="auto", help="llama.cpp reference output to compare: auto, clip_encode_float_image, projector_out, vision_output, or another dumped tensor name")
    ap.add_argument("--feature-slice-index", type=int, default=None, help="Convenience slice index for both tensors, using --feature-slice-dim")
    ap.add_argument("--feature-slice-dim", type=int, default=None, help="Convenience feature slice width for both tensors; defaults to projector_out_dim")
    ap.add_argument("--feature-row-width", type=int, default=None, help="Convenience row width for both tensors; defaults to projector_total_out_dim")
    ap.add_argument("--ck-feature-offset", type=int, default=None, help="Optional CK-only feature offset for row-wise tensor slicing")
    ap.add_argument("--ck-feature-dim", type=int, default=None, help="Optional CK-only feature width for row-wise tensor slicing")
    ap.add_argument("--ck-feature-row-width", type=int, default=None, help="Optional CK-only row width for row-wise tensor slicing")
    ap.add_argument("--llama-feature-offset", type=int, default=None, help="Optional llama-only feature offset for row-wise tensor slicing")
    ap.add_argument("--llama-feature-dim", type=int, default=None, help="Optional llama-only feature width for row-wise tensor slicing")
    ap.add_argument("--llama-feature-row-width", type=int, default=None, help="Optional llama-only row width for row-wise tensor slicing")
    ap.add_argument("--row-index", type=int, default=None, help="Optional common row index to compare after tensor capture; negative indices count from the end")
    ap.add_argument("--row-width", type=int, default=None, help="Optional common row width for --row-index")
    ap.add_argument("--row-dim", type=int, default=None, help="Optional number of values to compare from the selected row; defaults to row width")
    ap.add_argument("--ck-row-index", type=int, default=None, help="Optional CK-only row index; negative indices count from the end")
    ap.add_argument("--ck-row-width", type=int, default=None, help="Optional CK-only row width")
    ap.add_argument("--llama-row-index", type=int, default=None, help="Optional llama-only row index; negative indices count from the end")
    ap.add_argument("--llama-row-width", type=int, default=None, help="Optional llama-only row width")
    ap.add_argument("--report", type=Path, default=None, help="Optional JSON report output")
    ap.add_argument("--dump-ck-f32", type=Path, default=None, help="Optional raw decoder-facing CK output tensor")
    ap.add_argument("--dump-llama-f32", type=Path, default=None, help="Optional raw decoder-facing llama.cpp output tensor")
    args = ap.parse_args(argv)

    ck_threads = int(args.ck_threads or args.threads)
    activation_preferences = _parse_activation_preference_overrides(args.activation_pref)
    os.environ["OMP_NUM_THREADS"] = str(ck_threads)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    _ensure_engine_lib()
    report = _ensure_runtime_artifacts(
        args.gguf,
        output_dir,
        image_path=args.image_path.resolve() if args.image_path is not None else None,
        image_min_tokens=args.image_min_tokens,
        image_max_tokens=args.image_max_tokens,
        activation_preferences=activation_preferences,
    )
    model_so = _compile_generated_model(output_dir)
    shim_so = _compile_mtmd_shim(output_dir)
    t_artifacts = time.perf_counter()

    config = report["config"]
    height = int(config.get("image_height", config.get("image_size")))
    width = int(config.get("image_width", config.get("image_size")))
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
        image_min_tokens=args.image_min_tokens,
        image_max_tokens=args.image_max_tokens,
        flash_attn_type={"disabled": 0, "auto": -1, "enabled": 1}[args.llama_flash_attn],
    )
    t_llama = time.perf_counter()

    for path, values in ((args.dump_ck_f32, ck_out), (args.dump_llama_f32, llama_out)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                values.tofile(f)

    raw_num_values = {"ck": len(ck_out), "llama": len(llama_out)}
    ck_row_slice = _resolve_row_slice(
        length=len(ck_out),
        side="ck",
        row_index=args.ck_row_index,
        row_width=args.ck_row_width,
        common_row_index=args.row_index,
        common_row_width=args.row_width,
        row_dim=args.row_dim,
    )
    llama_row_slice = _resolve_row_slice(
        length=len(llama_out),
        side="llama",
        row_index=args.llama_row_index,
        row_width=args.llama_row_width,
        common_row_index=args.row_index,
        common_row_width=args.row_width,
        row_dim=args.row_dim,
    )
    ck_out = _apply_row_slice(ck_out, ck_row_slice)
    llama_out = _apply_row_slice(llama_out, llama_row_slice)

    ck_feature_slice = _resolve_feature_slice(
        config=config,
        length=len(ck_out),
        side="ck",
        row_width=args.ck_feature_row_width,
        offset=args.ck_feature_offset,
        dim=args.ck_feature_dim,
        common_row_width=args.feature_row_width,
        common_slice_index=args.feature_slice_index,
        common_slice_dim=args.feature_slice_dim,
    )
    llama_feature_slice = _resolve_feature_slice(
        config=config,
        length=len(llama_out),
        side="llama",
        row_width=args.llama_feature_row_width,
        offset=args.llama_feature_offset,
        dim=args.llama_feature_dim,
        common_row_width=args.feature_row_width,
        common_slice_index=args.feature_slice_index,
        common_slice_dim=args.feature_slice_dim,
    )
    ck_out = _apply_feature_slice(ck_out, ck_feature_slice)
    llama_out = _apply_feature_slice(llama_out, llama_feature_slice)

    metrics = _metrics(llama_out, ck_out)
    row_diagnostics_width = 0
    if ck_row_slice is not None or llama_row_slice is not None:
        row_diagnostics_width = len(llama_out)
    elif ck_feature_slice is not None or llama_feature_slice is not None:
        feature_slice = llama_feature_slice or ck_feature_slice or {}
        row_diagnostics_width = int(feature_slice.get("dim", 0) or 0)
    elif args.row_width:
        row_diagnostics_width = int(args.row_width)
    elif args.llama_row_width:
        row_diagnostics_width = int(args.llama_row_width)
    elif args.ck_row_width:
        row_diagnostics_width = int(args.ck_row_width)
    elif args.feature_row_width:
        row_diagnostics_width = int(args.feature_row_width)
    else:
        row_diagnostics_width = int(config.get("projector_total_out_dim", config.get("projection_dim", 0)) or 0)
    worst_rows = _sample_row_diffs(llama_out, ck_out, row_diagnostics_width)
    worst_row_indices = [int(row["row"]) for row in worst_rows[:3]]
    nearest_rows = {
        "got_to_ref": _nearest_rows(llama_out, ck_out, row_diagnostics_width, worst_row_indices, direction="got_to_ref"),
        "ref_to_got": _nearest_rows(llama_out, ck_out, row_diagnostics_width, worst_row_indices, direction="ref_to_got"),
    }
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
        notes.append("Qwen3-VL vision multi-section M-RoPE is lowered; remaining deltas should be interpreted from the reported tensor metrics.")
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
        "image_min_tokens": args.image_min_tokens,
        "image_max_tokens": args.image_max_tokens,
        "merged_grid": [int(config.get("merged_grid_x", 0) or 0), int(config.get("merged_grid_y", 0) or 0)],
        "threads": {
            "llama_cpp": args.threads,
            "ck_runtime": ck_threads,
        },
        "strict_parity": bool(args.strict_parity),
        "strict_mtmd_oracle": bool(args.strict_mtmd_oracle),
        "activation_preference_overrides": dict(activation_preferences),
        "active_activation_preferences": dict(config.get("activation_preference_by_op", {})) if isinstance(config.get("activation_preference_by_op"), dict) else {},
        "ck_output_name": _normalize_output_name(args.ck_output_name),
        "ck_resolved_output": ck_resolved_output,
        "llama_requested_output": _normalize_output_name(args.llama_output_name),
        "llama_reference_output": llama_reference_output or "clip_encode_float_image",
        "raw_num_values": raw_num_values,
        "row_slices": {
            "ck": ck_row_slice,
            "llama": llama_row_slice,
        },
        "feature_slices": {
            "ck": ck_feature_slice,
            "llama": llama_feature_slice,
        },
        "num_values": len(llama_out),
        "metrics": metrics,
        "top_diffs": _sample_diffs(llama_out, ck_out),
        "row_diagnostics": {
            "row_width": row_diagnostics_width,
            "worst_rows": worst_rows,
            "nearest_rows": nearest_rows,
        },
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
