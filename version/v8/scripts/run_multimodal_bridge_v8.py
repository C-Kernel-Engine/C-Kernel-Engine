#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import heapq
import importlib.util
import json
import subprocess
import sys
from array import array
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is optional at import time.
    Image = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BUILD_DIR = REPO_ROOT / "build"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gguf_tokenizer import GGUFTokenizer  # type: ignore  # noqa: E402


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


convert_gguf_to_bump_v8 = _load_module("convert_gguf_to_bump_v8_bridge", SCRIPT_DIR / "convert_gguf_to_bump_v8.py")
build_ir_v8 = _load_module("build_ir_v8_bridge", SCRIPT_DIR / "build_ir_v8.py")
codegen_v8 = _load_module("codegen_v8_bridge", SCRIPT_DIR / "codegen_v8.py")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _run_converter(gguf_path: Path, output_dir: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bump_path = output_dir / "weights.bump"
    manifest_path = output_dir / "weights_manifest.json"
    config_path = output_dir / "config.json"

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(SCRIPT_DIR / "convert_gguf_to_bump_v8.py"),
            "--gguf",
            str(gguf_path),
            "--output",
            str(bump_path),
            "--manifest-out",
            str(manifest_path),
            "--config-out",
            str(config_path),
        ]
        convert_gguf_to_bump_v8.main()
    finally:
        sys.argv = old_argv

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest, manifest_path, bump_path, config_path


def _compile_generated_model(c_path: Path, so_path: Path) -> Path:
    if so_path.exists() and so_path.stat().st_mtime >= c_path.stat().st_mtime:
        return so_path
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
    _run(cmd)
    return so_path


def _load_layout(layout_path: Path) -> dict[str, Any]:
    with layout_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_activation_offsets(layout_path: Path) -> dict[str, dict[str, Any]]:
    layout = _load_layout(layout_path)
    return {
        str(buf["name"]): buf
        for buf in layout["memory"]["activations"]["buffers"]
    }


def _activation_runtime_base(layout: dict[str, Any]) -> int:
    weights = layout.get("memory", {}).get("weights", {})
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


def _prepare_encoder_runtime(gguf_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest, manifest_path, bump_path, config_path = _run_converter(gguf_path, output_dir)
    layout_path = output_dir / "layout.json"
    call_path = output_dir / "call.json"
    lowered_path = output_dir / "lowered.json"
    ir1_path = output_dir / "ir1.json"
    manifest_map = output_dir / "weights_manifest.map"
    c_path = output_dir / "encoder_v8.c"
    so_path = output_dir / "libencoder_v8.so"

    rc = build_ir_v8.main(
        [
            "--manifest",
            str(manifest_path),
            "--mode",
            "prefill",
            "--output",
            str(ir1_path),
            "--layout-output",
            str(layout_path),
            "--lowered-output",
            str(lowered_path),
            "--call-output",
            str(call_path),
            "--manifest-map-output",
            str(manifest_map),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"build_ir_v8 encoder failed with rc={rc}")

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(SCRIPT_DIR / "codegen_v8.py"),
            "--ir",
            str(call_path),
            "--layout",
            str(layout_path),
            "--output",
            str(c_path),
        ]
        codegen_rc = codegen_v8.main()
    finally:
        sys.argv = old_argv
    if codegen_rc != 0:
        raise RuntimeError(f"codegen_v8 encoder failed with rc={codegen_rc}")

    _compile_generated_model(c_path, so_path)
    layout = _load_layout(layout_path)
    return {
        "gguf": str(gguf_path),
        "manifest": manifest,
        "weights_bump": bump_path,
        "manifest_map": manifest_map,
        "config_path": config_path,
        "layout_path": layout_path,
        "c_path": c_path,
        "so_path": so_path,
        "embed_dim": int(layout.get("config", {}).get("embed_dim", 0)),
    }


def _prepare_decoder_runtime(gguf_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest, manifest_path, bump_path, config_path = _run_converter(gguf_path, output_dir)
    prefill_ir1 = output_dir / "ir1_prefill.json"
    prefill_layout = output_dir / "layout_prefill.json"
    prefill_lowered = output_dir / "lowered_prefill.json"
    prefill_call = output_dir / "call_prefill.json"
    decode_ir1 = output_dir / "ir1_decode.json"
    decode_layout = output_dir / "layout_decode.json"
    decode_lowered = output_dir / "lowered_decode.json"
    decode_call = output_dir / "call_decode.json"
    manifest_map = output_dir / "weights_manifest.map"
    c_path = output_dir / "decoder_v8.c"
    so_path = output_dir / "libdecoder_v8.so"

    rc = build_ir_v8.main(
        [
            "--manifest",
            str(manifest_path),
            "--mode",
            "prefill",
            "--output",
            str(prefill_ir1),
            "--layout-output",
            str(prefill_layout),
            "--lowered-output",
            str(prefill_lowered),
            "--call-output",
            str(prefill_call),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"build_ir_v8 decoder prefill failed with rc={rc}")

    rc = build_ir_v8.main(
        [
            "--manifest",
            str(manifest_path),
            "--mode",
            "decode",
            "--output",
            str(decode_ir1),
            "--layout-output",
            str(decode_layout),
            "--lowered-output",
            str(decode_lowered),
            "--call-output",
            str(decode_call),
            "--manifest-map-output",
            str(manifest_map),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"build_ir_v8 decoder decode failed with rc={rc}")

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(SCRIPT_DIR / "codegen_v8.py"),
            "--ir",
            str(decode_call),
            "--prefill",
            str(prefill_call),
            "--layout",
            str(decode_layout),
            "--output",
            str(c_path),
        ]
        codegen_rc = codegen_v8.main()
    finally:
        sys.argv = old_argv
    if codegen_rc != 0:
        raise RuntimeError(f"codegen_v8 decoder failed with rc={codegen_rc}")

    _compile_generated_model(c_path, so_path)
    layout = _load_layout(decode_layout)
    return {
        "gguf": str(gguf_path),
        "manifest": manifest,
        "weights_bump": bump_path,
        "manifest_map": manifest_map,
        "config_path": config_path,
        "prefill_layout_path": prefill_layout,
        "decode_layout_path": decode_layout,
        "c_path": c_path,
        "so_path": so_path,
        "embed_dim": int(layout.get("config", {}).get("embed_dim", 0)),
        "vocab_size": int(layout.get("config", {}).get("vocab_size", 0)),
    }


def _load_encoder_lib(model_so: Path) -> ctypes.CDLL:
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
    return lib


def _load_decoder_lib(model_so: Path) -> ctypes.CDLL:
    ctypes.CDLL(str(BUILD_DIR / "libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(model_so))
    lib.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.ck_model_init_with_manifest.restype = ctypes.c_int
    lib.ck_model_forward_mixed.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ck_model_forward_mixed.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None
    return lib


def _run_encoder(runtime: dict[str, Any], image_mode: str, image_path: Path | None = None) -> dict[str, Any]:
    lib = _load_encoder_lib(runtime["so_path"])
    rc = lib.ck_model_init_with_manifest(
        str(runtime["weights_bump"]).encode(),
        str(runtime["manifest_map"]).encode(),
    )
    if rc != 0:
        raise RuntimeError(f"encoder init failed with rc={rc}")
    try:
        layout = _load_layout(runtime["layout_path"])
        offsets = _load_activation_offsets(runtime["layout_path"])
        image_buf = offsets["image_input"]
        output_buf = offsets["vision_output"]
        base_ptr = int(lib.ck_model_get_base_ptr())
        if base_ptr == 0:
            raise RuntimeError("encoder base ptr is null")
        image_size = int(layout.get("config", {}).get("image_size", 0))
        if image_size <= 0:
            raise RuntimeError("encoder image_size missing from layout")

        if image_path is not None:
            image_report = _load_image_file(image_path, image_size, image_size)
            interleaved = image_report["interleaved"]
            planar = image_report["planar"]
        else:
            interleaved, planar = _build_test_image(image_size, image_size, image_mode)
            image_report = {
                "image_source": "synthetic",
                "image_mode": image_mode,
                "image_path": None,
                "source_image_size": [image_size, image_size],
                "preprocess": "synthetic_generator",
            }
        image_len = _buffer_nbytes(image_buf) // ctypes.sizeof(ctypes.c_float)
        output_len = _buffer_nbytes(output_buf) // ctypes.sizeof(ctypes.c_float)
        if len(planar) != image_len:
            raise RuntimeError(f"encoder planar image length mismatch: {len(planar)} != {image_len}")

        image_arr = (ctypes.c_float * image_len).from_address(
            base_ptr + _activation_runtime_offset(layout, image_buf)
        )
        output_arr = (ctypes.c_float * output_len).from_address(
            base_ptr + _activation_runtime_offset(layout, output_buf)
        )
        image_arr[:] = planar
        rc = lib.ck_model_decode(0, None)
        if rc != 0:
            raise RuntimeError(f"encoder decode failed with rc={rc}")

        embed_dim = int(runtime["embed_dim"])
        if embed_dim <= 0 or output_len % embed_dim != 0:
            raise RuntimeError(f"invalid encoder output shape: output_len={output_len} embed_dim={embed_dim}")
        return {
            "embed_dim": embed_dim,
            "prefix_tokens": output_len // embed_dim,
            "embeddings": array("f", output_arr),
            "image_source": str(image_report["image_source"]),
            "image_mode": image_report.get("image_mode"),
            "image_path": image_report.get("image_path"),
            "source_image_size": image_report.get("source_image_size"),
            "preprocess": str(image_report["preprocess"]),
            "image_size": image_size,
            "interleaved_image": interleaved,
        }
    finally:
        lib.ck_model_free()


def _run_decoder(runtime: dict[str, Any], prefix_embeddings: array, prefix_tokens: int, token_ids: list[int]) -> dict[str, Any]:
    lib = _load_decoder_lib(runtime["so_path"])
    rc = lib.ck_model_init_with_manifest(
        str(runtime["weights_bump"]).encode(),
        str(runtime["manifest_map"]).encode(),
    )
    if rc != 0:
        raise RuntimeError(f"decoder init failed with rc={rc}")
    try:
        vocab_size = int(lib.ck_model_get_vocab_size())
        if vocab_size <= 0:
            vocab_size = int(runtime["vocab_size"])
        logits = (ctypes.c_float * vocab_size)()
        prefix_ptr: ctypes.Array[ctypes.c_float] | None
        if prefix_tokens > 0:
            prefix_ptr = (ctypes.c_float * len(prefix_embeddings))(*prefix_embeddings)
        else:
            prefix_ptr = None
        token_arr = (ctypes.c_int32 * len(token_ids))(*token_ids) if token_ids else None
        rc = lib.ck_model_forward_mixed(prefix_ptr, prefix_tokens, token_arr, len(token_ids), logits)
        if rc != 0:
            raise RuntimeError(f"decoder forward_mixed failed with rc={rc}")
        logits_arr = array("f", logits)
        return {
            "vocab_size": vocab_size,
            "logits": logits_arr,
        }
    finally:
        lib.ck_model_free()


def _topk(values: array, k: int) -> list[tuple[int, float]]:
    return heapq.nlargest(k, enumerate(values), key=lambda item: float(item[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Explicit v8 multimodal bridge runner")
    ap.add_argument("--decoder-gguf", type=Path, required=True, help="Decoder GGUF to lower/codegen")
    ap.add_argument("--encoder-gguf", type=Path, default=None, help="Optional vision encoder/mmproj GGUF")
    ap.add_argument("--workdir", type=Path, required=True, help="Artifact/output directory")
    ap.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt text for decoder tokenization")
    ap.add_argument("--image-mode", choices=["checker", "gradient", "gray"], default="checker", help="Synthetic image generator to use when --image-path is not provided")
    ap.add_argument("--image-path", type=Path, default=None, help="Optional real image path for encoder input; overrides --image-mode")
    ap.add_argument("--synthetic-prefix-tokens", type=int, default=0, help="Use zero prefix embeddings when a real encoder bridge is unavailable")
    ap.add_argument("--dump-prefix-f32", type=Path, default=None, help="Optional output path for resolved float32 prefix embeddings")
    ap.add_argument("--top-k", type=int, default=8, help="How many top logits to report")
    args = ap.parse_args(argv)

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    encoder_dir = workdir / "encoder"
    decoder_dir = workdir / "decoder"

    decoder_runtime = _prepare_decoder_runtime(args.decoder_gguf.resolve(), decoder_dir)
    tokenizer = GGUFTokenizer.from_gguf(str(args.decoder_gguf.resolve()))
    token_ids = tokenizer.encode(args.prompt)

    prefix_source = "none"
    prefix_tokens = 0
    prefix_embeddings = array("f")
    encoder_report: dict[str, Any] | None = None
    dim_mismatch: dict[str, int] | None = None

    if args.encoder_gguf is not None:
        encoder_runtime = _prepare_encoder_runtime(args.encoder_gguf.resolve(), encoder_dir)
        encoder_report = _run_encoder(
            encoder_runtime,
            args.image_mode,
            image_path=args.image_path.resolve() if args.image_path is not None else None,
        )
        if encoder_report["embed_dim"] == decoder_runtime["embed_dim"]:
            prefix_source = "encoder"
            prefix_tokens = int(encoder_report["prefix_tokens"])
            prefix_embeddings = encoder_report["embeddings"]
        else:
            dim_mismatch = {
                "encoder_embed_dim": int(encoder_report["embed_dim"]),
                "decoder_embed_dim": int(decoder_runtime["embed_dim"]),
            }

    if prefix_source != "encoder" and args.synthetic_prefix_tokens > 0:
        prefix_source = "synthetic_zero"
        prefix_tokens = args.synthetic_prefix_tokens
        prefix_embeddings = array("f", [0.0] * (prefix_tokens * int(decoder_runtime["embed_dim"])))

    if prefix_source == "none":
        raise SystemExit(
            "No usable prefix source: encoder/decode dims do not match and no --synthetic-prefix-tokens was provided"
        )

    dumped_prefix_path: str | None = None
    if args.dump_prefix_f32 is not None:
        dump_path = args.dump_prefix_f32.resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_bytes(prefix_embeddings.tobytes())
        dumped_prefix_path = str(dump_path)

    decoder_report = _run_decoder(decoder_runtime, prefix_embeddings, prefix_tokens, token_ids)
    top = _topk(decoder_report["logits"], max(1, args.top_k))
    top_tokens = [
        {
            "token_id": int(tok_id),
            "logit": float(logit),
            "token_text": tokenizer.decode([int(tok_id)], skip_special=False),
        }
        for tok_id, logit in top
    ]

    report = {
        "status": "ok",
        "prefix_source": prefix_source,
        "prompt": args.prompt,
        "prompt_token_count": len(token_ids),
        "prompt_tokens": token_ids,
        "decoder_embed_dim": int(decoder_runtime["embed_dim"]),
        "prefix_tokens": prefix_tokens,
        "prefix_dump_path": dumped_prefix_path,
        "total_prefill_tokens": prefix_tokens + len(token_ids),
        "decoder_runtime": {
            "gguf": str(args.decoder_gguf.resolve()),
            "workdir": str(decoder_dir),
            "so_path": str(decoder_runtime["so_path"]),
            "c_path": str(decoder_runtime["c_path"]),
        },
        "encoder_runtime": {
            "gguf": str(args.encoder_gguf.resolve()),
            "workdir": str(encoder_dir),
        } if args.encoder_gguf is not None else None,
        "encoder_report": {
            "embed_dim": int(encoder_report["embed_dim"]),
            "prefix_tokens": int(encoder_report["prefix_tokens"]),
            "image_source": str(encoder_report["image_source"]),
            "image_mode": None if encoder_report["image_mode"] is None else str(encoder_report["image_mode"]),
            "image_path": encoder_report["image_path"],
            "source_image_size": encoder_report["source_image_size"],
            "preprocess": str(encoder_report["preprocess"]),
            "image_size": int(encoder_report["image_size"]),
        } if encoder_report is not None else None,
        "dim_mismatch": dim_mismatch,
        "top_logits": top_tokens,
        "notes": [
            "This runner keeps the encoder->decoder bridge in orchestration instead of baking a multimodal special-case into templates.",
            "Synthetic-prefix mode only validates the decoder bridge seam; it is not a substitute for real multimodal parity.",
            "Real-image mode resizes the provided image to the encoder square input in Python; exact llama.cpp parity still depends on matching the reference preprocessing path.",
        ],
    }
    report_path = workdir / "bridge_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
