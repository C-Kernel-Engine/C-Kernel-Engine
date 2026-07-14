#!/usr/bin/env python3
"""Qwen3-VL GGUF/llama.cpp checkpoint capture adapter.

Ownership boundary:
  * This module is backend-specific capture plumbing, not the public parity
    orchestration surface.
  * New checkpoint selection, bisection, canonicalization, comparison,
    classification, or reporting belongs in the v8 X-ray tooling.
  * Do not add model-family branches here. Add a separate backend adapter that
    emits the canonical X-ray checkpoint ABI.

Direct CLI use is retained for compatibility and adapter development only.
Agents and automation must invoke ``xray_vision_parity_v8.py --backend
llamacpp`` so checkpoint ordering and reports use the standard X-ray surface.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required X-ray binary is missing: {path}")
    comment = subprocess.run(
        ["readelf", "-p", ".comment", str(path)],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "elf_comment": comment.stdout.strip() if comment.returncode == 0 else None,
    }


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
    granular_report = output_dir / "granular_codegen.json"
    rc = codegen_v8.main(
        [
            "--ir", str(output_dir / "call.json"),
            "--layout", str(output_dir / "layout.json"),
            "--output", str(c_path),
            "--granular-test",
            "--granular-report", str(granular_report),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"codegen_v8 granular-test failed with rc={rc}")
    return c_path


def _resolve_ck_stop_op(output_dir: Path, ck_stop_op: int | None, ck_stop_layer: int | None) -> int | None:
    if ck_stop_op is not None:
        return int(ck_stop_op)
    if ck_stop_layer is None:
        return None

    granular_report = output_dir / "granular_codegen.json"
    if not granular_report.exists():
        raise RuntimeError(f"cannot resolve --ck-stop-layer without granular report: {granular_report}")
    data = json.loads(granular_report.read_text(encoding="utf-8"))
    cutpoints = data.get("cutpoints")
    if not isinstance(cutpoints, list):
        raise RuntimeError(f"invalid granular report {granular_report}: missing cutpoints")

    matches: list[int] = []
    available_layers: set[int] = set()
    for row in cutpoints:
        if not isinstance(row, dict):
            continue
        try:
            layer = int(row.get("layer", -999999))
        except (TypeError, ValueError):
            continue
        available_layers.add(layer)
        if layer != int(ck_stop_layer):
            continue
        try:
            matches.append(int(row["index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid cutpoint row in {granular_report}: {row!r}") from exc
    if not matches:
        raise RuntimeError(
            f"granular report {granular_report} has no cutpoints for layer {ck_stop_layer}; "
            f"available layers: {sorted(available_layers)}"
        )
    return max(matches)


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
    dump_layer: int | None,
    ck_stop_op: int | None,
    dump_names: str | None,
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
    # Keep global frontend checkpoints and the requested transformer layer.
    # Without this bound, a late-layer X-ray emits every prior layer and can
    # produce multi-gigabyte dumps before attribution begins.
    old_layer_filter = _with_env_var(
        "CK_PARITY_LAYER_FILTER",
        None if dump_layer is None else f"-1,{int(dump_layer)}",
    )
    old_stop_op = _with_env_var("CK_STOP_OP", None if ck_stop_op is None else str(int(ck_stop_op)))
    old_filter = _with_env_var("CK_PARITY_OP_FILTER", dump_names)
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
        _restore_env_var("CK_PARITY_OP_FILTER", old_filter)
        _restore_env_var("CK_STOP_OP", old_stop_op)
        _restore_env_var("CK_PARITY_LAYER_FILTER", old_layer_filter)
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
    flash_attn_type: int = 0,
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
            flash_attn_type=flash_attn_type,
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
    ap.add_argument("--image-min-tokens", type=int, default=None, help="Override minimum merged visual tokens for dynamic-resolution Qwen3-VL images")
    ap.add_argument("--image-max-tokens", type=int, default=None, help="Override maximum merged visual tokens for dynamic-resolution Qwen3-VL images")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument(
        "--llama-flash-attn",
        choices=("disabled", "auto", "enabled"),
        default="disabled",
        help="Reference attention algorithm; it must match the CK execution contract.",
    )
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
    ap.add_argument("--ck-dump-layer", type=int, default=None, help="Optional exact CK layer id filter; global frontend checkpoints remain included")
    ap.add_argument("--ck-strict-dump-layer", type=int, default=None, help="Optional exact CK strict-attention dump layer filter")
    ap.add_argument(
        "--ck-dump-names",
        type=str,
        default=None,
        help="Optional comma-separated CK dump filter; defaults to --llama-dump-names",
    )
    ck_stop = ap.add_mutually_exclusive_group()
    ck_stop.add_argument("--ck-stop-op", type=int, default=None, help="Return from generated CK encoder immediately after this generated op index")
    ck_stop.add_argument("--ck-stop-layer", type=int, default=None, help="Return from generated CK encoder after the last generated op in this encoder layer")
    args = ap.parse_args(argv)

    ck_threads = int(args.ck_threads or args.threads)
    os.environ["OMP_NUM_THREADS"] = str(ck_threads)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = npv8._ensure_runtime_artifacts(
        args.gguf,
        output_dir,
        image_path=args.image_path.resolve() if args.image_path is not None else None,
        image_min_tokens=args.image_min_tokens,
        image_max_tokens=args.image_max_tokens,
    )
    c_path = _generate_dump_model_source(output_dir)
    ck_stop_op = _resolve_ck_stop_op(output_dir, args.ck_stop_op, args.ck_stop_layer)
    model_so = _compile_generated_dump_model(output_dir, c_path)
    shim_so = npv8._compile_mtmd_shim(output_dir)
    engine_so = BUILD_DIR / "libckernel_engine.so"
    binary_provenance = {
        "engine": _binary_provenance(engine_so),
        "generated_model": _binary_provenance(model_so),
        "llama_shim": _binary_provenance(shim_so),
    }

    config = report["config"]
    height = int(config.get("image_height", config.get("image_size")))
    width = int(config.get("image_width", config.get("image_size")))
    if height <= 0 or width <= 0:
        raise RuntimeError(f"invalid encoder image shape in generated config: height={height} width={width}")
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
        dump_layer=args.ck_dump_layer,
        ck_stop_op=ck_stop_op,
        dump_names=args.ck_dump_names or args.llama_dump_names,
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
        flash_attn_type={"disabled": 0, "auto": -1, "enabled": 1}[args.llama_flash_attn],
    )
    engine_sha_after_capture = _sha256(engine_so)
    if engine_sha_after_capture != binary_provenance["engine"]["sha256"]:
        raise RuntimeError(
            "build/libckernel_engine.so changed during X-ray capture; discard the mixed-build report"
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
        "image_min_tokens": args.image_min_tokens,
        "image_max_tokens": args.image_max_tokens,
        "threads": {
            "llama_cpp": args.threads,
            "ck_runtime": ck_threads,
        },
        "binary_provenance": binary_provenance,
        "strict_parity": bool(args.strict_parity),
        "llama_flash_attn": args.llama_flash_attn,
        "llama_dump_names": args.llama_dump_names,
        "ck_dump_names": args.ck_dump_names or args.llama_dump_names,
        "llama_dump_layer": args.llama_dump_layer,
        "ck_dump_layer": args.ck_dump_layer,
        "atol": args.atol,
        "rtol": args.rtol,
        "ck_dump": str(ck_dump),
        "llama_dump": str(ref_dump),
        "granular_codegen": str(output_dir / "granular_codegen.json"),
        "ck_stop_layer": args.ck_stop_layer,
        "ck_stop_op_requested": args.ck_stop_op,
        "ck_stop_op_resolved": ck_stop_op,
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
