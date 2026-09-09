#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
NUMERIC_PARITY = SCRIPT_DIR / "numeric_parity_qwen3vl_mmproj_v8.py"


def _sanitize_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    text = text.strip("._-")
    return text or "sample"


def _load_image_specs(summary_json: Path | None, image_paths: list[Path], limit: int | None) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    if summary_json is not None:
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        for idx, item in enumerate(data.get("results", []), 1):
            image = item.get("image")
            if not image:
                continue
            sample_id = str(item.get("id") or Path(str(image)).stem or idx)
            specs.append({"id": sample_id, "image": str(image)})
    for image in image_paths:
        specs.append({"id": image.stem, "image": str(image)})
    if limit is not None and limit > 0:
        specs = specs[:limit]
    return specs


def _metric_value(sample: dict[str, Any], name: str, default: float = 0.0) -> float:
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        return default
    try:
        return float(metrics.get(name, default))
    except (TypeError, ValueError):
        return default


def _shape_status(sample: dict[str, Any], embed_dim: int) -> tuple[bool, int | None]:
    grid = sample.get("grid")
    values = sample.get("num_values")
    if not isinstance(grid, list) or len(grid) != 2:
        return False, None
    try:
        gx = int(grid[0])
        gy = int(grid[1])
        got = int(values)
    except (TypeError, ValueError):
        return False, None
    expected = gx * gy * int(embed_dim)
    raw = sample.get("raw_num_values")
    raw_ok = True
    if isinstance(raw, dict):
        try:
            raw_ok = int(raw.get("ck", got)) == got and int(raw.get("llama", got)) == got
        except (TypeError, ValueError):
            raw_ok = False
    return got == expected and raw_ok, expected


def _sample_from_report(spec: dict[str, str], report_path: Path, embed_dim: int) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sample = {
        "id": spec["id"],
        "image": spec["image"],
        "report": str(report_path),
        "grid": report.get("merged_grid"),
        "height": report.get("height"),
        "width": report.get("width"),
        "num_values": report.get("num_values"),
        "raw_num_values": report.get("raw_num_values"),
        "metrics": report.get("metrics", {}),
        "timings_sec": report.get("timings_sec", {}),
        "worst_rows": (report.get("row_diagnostics", {}) or {}).get("worst_rows", [])[:3],
    }
    shape_ok, expected = _shape_status(sample, embed_dim)
    sample["shape_ok"] = shape_ok
    sample["expected_values"] = expected
    return sample


def _run_one(
    *,
    spec: dict[str, str],
    index: int,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    sample_name = f"{index:02d}_{_sanitize_id(spec['id'])}"
    report_path = args.output_dir / "reports" / f"{sample_name}.json"
    log_path = args.output_dir / "logs" / f"{sample_name}.log"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not (args.reuse_reports and report_path.exists()):
        cmd = [
            sys.executable,
            str(NUMERIC_PARITY),
            "--gguf",
            str(args.gguf),
            "--output-dir",
            str(args.runtime_dir),
            "--image-path",
            spec["image"],
            "--threads",
            str(args.threads),
            "--ck-threads",
            str(args.ck_threads),
            "--report",
            str(report_path),
        ]
        if args.image_min_tokens is not None:
            cmd.extend(["--image-min-tokens", str(args.image_min_tokens)])
        if args.image_max_tokens is not None:
            cmd.extend(["--image-max-tokens", str(args.image_max_tokens)])
        start = time.perf_counter()
        with log_path.open("wb") as log_file:
            subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        elapsed = time.perf_counter() - start
    else:
        elapsed = 0.0

    sample = _sample_from_report(spec, report_path, args.embed_dim)
    sample["log"] = str(log_path)
    sample["suite_elapsed_sec"] = elapsed
    return sample


def _log_failure_reason(path: Path, returncode: int) -> str:
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
    except OSError:
        lines = []
    detail = lines[-1] if lines else "child process produced no diagnostic output"
    return f"numeric parity process exited {returncode}: {detail}"


def _evaluate_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for sample in samples:
        sid = str(sample.get("id", "sample"))
        if not sample.get("shape_ok"):
            failures.append(f"{sid}: shape mismatch, got {sample.get('num_values')} expected {sample.get('expected_values')}")
        cosine = _metric_value(sample, "cosine")
        rmse = _metric_value(sample, "rmse")
        max_abs = _metric_value(sample, "max_abs")
        if cosine < float(args.min_cosine):
            failures.append(f"{sid}: cosine {cosine:.9f} < {args.min_cosine:.9f}")
        if rmse > float(args.max_rmse):
            failures.append(f"{sid}: rmse {rmse:.9f} > {args.max_rmse:.9f}")
        if args.max_abs is not None and max_abs > float(args.max_abs):
            failures.append(f"{sid}: max_abs {max_abs:.9f} > {args.max_abs:.9f}")
    return failures


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen3-VL Encoder Prefix Parity",
        "",
        f"- samples: {summary['sample_count']}",
        f"- threads: {summary['threads']}",
        f"- image_max_tokens: {summary.get('image_max_tokens')}",
        f"- min_cosine: {summary['aggregate']['min_cosine']:.9f}",
        f"- max_rmse: {summary['aggregate']['max_rmse']:.6f}",
        f"- max_abs: {summary['aggregate']['max_abs']:.6f}",
        f"- failures: {len(summary['failures'])}",
        "",
        "| sample | grid | values | cosine | rmse | mean_abs | max_abs | ck_s | llama_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in summary["samples"]:
        grid = sample.get("grid") or ["?", "?"]
        metrics = sample.get("metrics") or {}
        timings = sample.get("timings_sec") or {}
        lines.append(
            "| {sid} | {gx}x{gy} | {values} | {cos:.9f} | {rmse:.6f} | {mean:.6f} | {max_abs:.6f} | {ck:.1f} | {llama:.1f} |".format(
                sid=sample.get("id"),
                gx=grid[0],
                gy=grid[1],
                values=sample.get("num_values"),
                cos=float(metrics.get("cosine", 0.0)),
                rmse=float(metrics.get("rmse", 0.0)),
                mean=float(metrics.get("mean_abs", 0.0)),
                max_abs=float(metrics.get("max_abs", 0.0)),
                ck=float(timings.get("ck_encode", 0.0)),
                llama=float(timings.get("llama_encode", 0.0)),
            )
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["failures"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL encoder prefix parity over a small image set.")
    parser.add_argument("--gguf", type=Path, required=True, help="Path to mmproj-Qwen3VL-*.gguf")
    parser.add_argument("--summary-json", type=Path, default=None, help="OCR summary JSON containing result image paths")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Extra image path; may be repeated")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("build/qwen3vl_encoder_prefix_parity"))
    parser.add_argument("--runtime-dir", type=Path, default=None, help="Reusable generated mmproj runtime directory")
    parser.add_argument("--image-min-tokens", type=int, default=None)
    parser.add_argument("--image-max-tokens", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("CK_NUM_THREADS", "20") or "20"))
    parser.add_argument("--ck-threads", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=16384)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--max-rmse", type=float, default=0.03)
    parser.add_argument("--max-abs", type=float, default=None)
    parser.add_argument("--reuse-reports", action="store_true", help="Reuse existing per-image reports instead of recomputing")
    parser.add_argument("--no-fail", action="store_true", help="Write reports but return success even when thresholds fail")
    args = parser.parse_args(argv)

    args.output_dir = args.output_dir.resolve()
    args.runtime_dir = (args.runtime_dir or (args.output_dir / "runtime")).resolve()
    args.ck_threads = int(args.ck_threads or args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_image_specs(args.summary_json, args.image, args.limit)
    if not specs:
        raise SystemExit("no images selected; pass --summary-json or --image")

    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.ck_threads)
    env["OMP_NUM_THREADS"] = str(args.ck_threads)

    samples: list[dict[str, Any]] = []
    execution_failures: list[str] = []
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] encoder parity {spec['id']} -> {spec['image']}", flush=True)
        try:
            samples.append(_run_one(spec=spec, index=index, args=args, env=env))
        except subprocess.CalledProcessError as exc:
            sample_name = f"{index:02d}_{_sanitize_id(spec['id'])}"
            log_path = args.output_dir / "logs" / f"{sample_name}.log"
            execution_failures.append(
                f"{spec['id']}: {_log_failure_reason(log_path, exc.returncode)}"
            )

    failures = execution_failures + _evaluate_samples(samples, args)
    aggregate = {
        "min_cosine": min((_metric_value(sample, "cosine") for sample in samples), default=0.0),
        "max_rmse": max((_metric_value(sample, "rmse") for sample in samples), default=0.0),
        "max_abs": max((_metric_value(sample, "max_abs") for sample in samples), default=0.0),
        "all_shapes_ok": all(bool(sample.get("shape_ok")) for sample in samples),
    }
    summary = {
        "gguf": str(args.gguf),
        "sample_count": len(samples),
        "threads": args.threads,
        "ck_threads": args.ck_threads,
        "image_min_tokens": args.image_min_tokens,
        "image_max_tokens": args.image_max_tokens,
        "embed_dim": args.embed_dim,
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_rmse": args.max_rmse,
            "max_abs": args.max_abs,
        },
        "aggregate": aggregate,
        "failures": failures,
        "samples": samples,
    }
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(report_path, summary)

    print(json.dumps({
        "sample_count": len(samples),
        "aggregate": aggregate,
        "failures": failures,
        "summary": str(summary_path),
        "report": str(report_path),
    }, indent=2))
    return 0 if (not failures or args.no_fail) else 1


if __name__ == "__main__":
    raise SystemExit(main())
