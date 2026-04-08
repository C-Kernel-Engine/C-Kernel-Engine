#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BUILD_DIR = REPO_ROOT / "build"
LLAMA_CPP_ROOT = REPO_ROOT / "llama.cpp"
V7_SCRIPTS = REPO_ROOT / "version" / "v7" / "scripts"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(V7_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V7_SCRIPTS))

import codegen_v8  # type: ignore  # noqa: E402
import numeric_parity_qwen3vl_mmproj_v8 as npv8  # type: ignore  # noqa: E402
import parity_test  # type: ignore  # noqa: E402


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), env=env, check=True)


def _compile_generated_dump_model(output_dir: Path, c_path: Path) -> Path:
    so_path = output_dir / "libqwen3vl_mmproj_v8_parity_dump.so"
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


def _generate_dump_model_source(output_dir: Path) -> Path:
    c_path = output_dir / "qwen3_vl_mmproj_v8_parity_dump.c"
    rc = codegen_v8.main(
        [
            "--ir", str(output_dir / "call.json"),
            "--layout", str(output_dir / "layout.json"),
            "--output", str(c_path),
            "--parity-dump",
        ]
    )
    if rc != 0:
        raise RuntimeError(f"codegen_v8 parity-dump failed with rc={rc}")
    return c_path


def _with_env_var(name: str, value: str | None):
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


def _merge_ck_dump_artifacts(dump_dir: Path) -> Path:
    primary = dump_dir / "dump.bin"
    strict_internal = dump_dir / "strict_internal.bin"
    if not strict_internal.exists():
        return primary

    merged = dump_dir / "dump.merged.bin"
    with merged.open("wb") as out_f:
        if primary.exists():
            out_f.write(primary.read_bytes())
        out_f.write(strict_internal.read_bytes())
    return merged


def _run_generated_encoder_with_dump(
    model_so: Path,
    weights_bump: Path,
    manifest_map: Path,
    layout_path: Path,
    planar_image: list[float],
    dump_dir: Path,
    strict_parity: bool,
    strict_dump_layer: int | None,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / "dump.bin"
    strict_internal = dump_dir / "strict_internal.bin"
    merged_dump = dump_dir / "dump.merged.bin"
    if dump_path.exists():
        dump_path.unlink()
    if strict_internal.exists():
        strict_internal.unlink()
    if merged_dump.exists():
        merged_dump.unlink()
    old_dump = _with_env_var("CK_PARITY_DIR", str(dump_dir))
    old_dump_layer = _with_env_var(
        "CK_STRICT_ATTN_DUMP_LAYER",
        None if strict_dump_layer is None else str(strict_dump_layer),
    )
    try:
        npv8._run_generated_encoder(
            model_so=model_so,
            weights_bump=weights_bump,
            manifest_map=manifest_map,
            layout_path=layout_path,
            planar_image=planar_image,
            strict_parity=strict_parity,
        )
    finally:
        _restore_env_var("CK_STRICT_ATTN_DUMP_LAYER", old_dump_layer)
        _restore_env_var("CK_PARITY_DIR", old_dump)


def _run_llama_encoder_with_dump(
    shim_so: Path,
    gguf_path: Path,
    interleaved_image: list[float],
    height: int,
    width: int,
    n_threads: int,
    dump_dir: Path,
    dump_names: str | None = None,
    dump_layer: int | None = None,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / "dump.bin"
    if dump_path.exists():
        dump_path.unlink()
    old_dump = _with_env_var("CK_LLAMA_PARITY_DIR", str(dump_dir))
    old_all = _with_env_var("CK_LLAMA_PARITY_ALL", None)
    old_names = _with_env_var("CK_LLAMA_PARITY_NAMES", dump_names)
    old_layer = _with_env_var("CK_LLAMA_PARITY_LAYER", None if dump_layer is None else str(dump_layer))
    try:
        npv8._run_llamacpp_encoder(
            shim_so=shim_so,
            gguf_path=gguf_path,
            interleaved_image=interleaved_image,
            height=height,
            width=width,
            n_threads=n_threads,
        )
    finally:
        _restore_env_var("CK_LLAMA_PARITY_DIR", old_dump)
        _restore_env_var("CK_LLAMA_PARITY_ALL", old_all)
        _restore_env_var("CK_LLAMA_PARITY_NAMES", old_names)
        _restore_env_var("CK_LLAMA_PARITY_LAYER", old_layer)


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "total": len(results),
        "pass": 0,
        "fail": 0,
        "error": 0,
        "warn": 0,
    }
    for row in results:
        status = str(row.get("status", "")).upper()
        if status == "PASS":
            summary["pass"] += 1
        elif status == "FAIL":
            summary["fail"] += 1
        elif status == "ERROR":
            summary["error"] += 1
        elif status == "WARN":
            summary["warn"] += 1
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Activation parity dump for v8 Qwen3-VL vision checkpoints vs local llama.cpp")
    ap.add_argument("--gguf", type=Path, required=True, help="Path to mmproj-Qwen3VL-*.gguf")
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/qwen3vl_mmproj_v8_activation_parity"))
    ap.add_argument("--image-mode", choices=("gradient", "gray", "checker"), default="gradient")
    ap.add_argument("--image-path", type=Path, default=None, help="Optional real image path; overrides --image-mode")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ck-threads", type=int, default=None)
    ap.add_argument("--strict-parity", action="store_true", help="Enable parity-only strict mode in CK during the generated encoder run")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument(
        "--llama-dump-names",
        type=str,
        default=None,
        help="Optional comma-separated llama dump filter, e.g. patch_bias,inp_pos_emb,ln1,Qcur,Qcur_rope,Kcur,Kcur_rope,Vcur,kqv_out,attn_out",
    )
    ap.add_argument("--llama-dump-layer", type=int, default=None, help="Optional exact llama layer id filter; globals remain included")
    ap.add_argument("--ck-strict-dump-layer", type=int, default=None, help="Optional exact CK strict-attention dump layer filter")
    args = ap.parse_args(argv)

    ck_threads = int(args.ck_threads or args.threads)
    os.environ["OMP_NUM_THREADS"] = str(ck_threads)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = npv8._ensure_runtime_artifacts(args.gguf, output_dir)
    c_path = _generate_dump_model_source(output_dir)
    model_so = _compile_generated_dump_model(output_dir, c_path)
    shim_so = npv8._compile_mtmd_shim(output_dir)

    config = report["config"]
    height = int(config["image_size"])
    width = int(config["image_size"])
    if args.image_path is not None:
        image_report = npv8._load_image_file(args.image_path.resolve(), height, width)
        interleaved = image_report["interleaved"]
        planar = image_report["planar"]
    else:
        interleaved, planar = npv8._build_test_image(height, width, args.image_mode)
        image_report = {
            "image_source": "synthetic",
            "image_mode": args.image_mode,
            "image_path": None,
            "source_image_size": [width, height],
            "preprocess": "synthetic_generator",
        }

    ck_dump_dir = output_dir / "ck_parity_dumps"
    llama_dump_dir = output_dir / "llama_parity_dumps"

    _run_generated_encoder_with_dump(
        model_so=model_so,
        weights_bump=Path(report["weights_bump"]),
        manifest_map=output_dir / "weights_manifest.map",
        layout_path=output_dir / "layout.json",
        planar_image=planar,
        dump_dir=ck_dump_dir,
        strict_parity=bool(args.strict_parity),
        strict_dump_layer=args.ck_strict_dump_layer,
    )
    _run_llama_encoder_with_dump(
        shim_so=shim_so,
        gguf_path=args.gguf,
        interleaved_image=interleaved,
        height=height,
        width=width,
        n_threads=args.threads,
        dump_dir=llama_dump_dir,
        dump_names=args.llama_dump_names,
        dump_layer=args.llama_dump_layer,
    )

    ck_dump = _merge_ck_dump_artifacts(ck_dump_dir)
    ref_dump = llama_dump_dir / "dump.bin"
    if not ck_dump.exists():
        raise RuntimeError(f"CK parity dump missing: {ck_dump}")
    if not ref_dump.exists():
        raise RuntimeError(f"llama.cpp parity dump missing: {ref_dump}")

    exit_code, results = parity_test.run_parity_test(
        ck_dump_path=ck_dump,
        ref_dump_path=ref_dump,
        atol=args.atol,
        rtol=args.rtol,
        verbose=not args.quiet,
        model_family="qwen3vl_vision",
        pass_filter="all",
    )

    first_issue = next((row for row in results if str(row.get("status", "")).upper() in {"FAIL", "ERROR"}), None)
    summary = _summarize_results(results)
    artifact_report = {
        "gguf": str(args.gguf),
        "output_dir": str(output_dir),
        "image_source": str(image_report.get("image_source", "synthetic")),
        "image_mode": image_report.get("image_mode"),
        "image_path": image_report.get("image_path"),
        "source_image_size": image_report.get("source_image_size"),
        "preprocess": image_report.get("preprocess"),
        "threads": {
            "llama_cpp": args.threads,
            "ck_runtime": ck_threads,
        },
        "strict_parity": bool(args.strict_parity),
        "llama_dump_names": args.llama_dump_names,
        "llama_dump_layer": args.llama_dump_layer,
        "atol": args.atol,
        "rtol": args.rtol,
        "ck_dump": str(ck_dump),
        "llama_dump": str(ref_dump),
        "summary": summary,
        "first_issue": first_issue,
        "results": results,
    }

    if args.report is not None:
        args.report.write_text(json.dumps(artifact_report, indent=2), encoding="utf-8")
    if args.quiet:
        print(json.dumps(artifact_report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
