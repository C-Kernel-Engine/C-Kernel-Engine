#!/usr/bin/env python3
"""Artifact-backed vision encoder accuracy gate.

The gate deliberately uses the same implementation for local and nightly runs.
It generates a public, deterministic large image so regressions that only occur
with large visual grids are covered without committing private OCR documents.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
Q4_SUITE = SCRIPT_DIR / "qwen3vl_encoder_prefix_parity_suite_v8.py"
BF16_SUITE = SCRIPT_DIR / "compare_qwen3vl_bf16_vision_hidden_v8.py"


def _cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith(("flags", "Features")) and ":" in line:
            return set(line.split(":", 1)[1].strip().split())
    return set()


def _write_large_form(path: Path, width: int = 1152, height: int = 896) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = bytearray([248]) * (width * height * 3)

    def rect(x0: int, y0: int, x1: int, y1: int, rgb: tuple[int, int, int]) -> None:
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        row = bytes(rgb) * max(0, x1 - x0)
        for y in range(y0, y1):
            start = (y * width + x0) * 3
            pixels[start:start + len(row)] = row

    # Generic form geometry: header, table rules, field strokes, and checkboxes.
    rect(40, 32, width - 40, 38, (25, 48, 72))
    rect(40, 70, 430, 78, (25, 48, 72))
    for y in range(130, height - 48, 82):
        rect(48, y, width - 48, y + 2, (105, 118, 126))
        rect(72, y + 24, 330, y + 29, (42, 52, 58))
        rect(390, y + 20, width - 90, y + 23, (120, 126, 132))
        # Deterministic text-like bars expose spatial ordering without real data.
        for word in range(7):
            x = 390 + word * 86
            bar = 30 + ((y // 2 + word * 17) % 46)
            rect(x, y + 35, min(x + bar, width - 90), y + 39, (34, 38, 42))
    for y in (170, 334, 498, 662):
        rect(78, y, 98, y + 3, (20, 25, 30))
        rect(78, y + 17, 98, y + 20, (20, 25, 30))
        rect(78, y, 81, y + 20, (20, 25, 30))
        rect(95, y, 98, y + 20, (20, 25, 30))
        rect(82, y + 5, 95, y + 8, (35, 92, 55))
        rect(88, y + 8, 92, y + 16, (35, 92, 55))

    with path.open("wb") as out:
        out.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        out.write(pixels)


def _run(cmd: list[str], *, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def _missing(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def _cleanup_raw_tensors(root: Path) -> None:
    for path in root.rglob("*.f32"):
        path.unlink(missing_ok=True)


def _phase(status: str, **values: Any) -> dict[str, Any]:
    return {"status": status, **values}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the promoted v8 vision encoder accuracy gate.")
    ap.add_argument("--family", choices=("qwen3vl",), default="qwen3vl")
    ap.add_argument("--mode", choices=("all", "q4", "bf16"), default="all")
    ap.add_argument("--output-dir", type=Path, default=Path("build/vision_encoder_accuracy"))
    ap.add_argument("--q4-mmproj", type=Path)
    ap.add_argument("--bf16-checkpoint", type=Path)
    ap.add_argument("--bf16-runtime-dir", type=Path)
    ap.add_argument("--bf16-weights-bump", type=Path)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("CK_NUM_THREADS", "20") or "20"))
    ap.add_argument("--full-layers", action="store_true")
    ap.add_argument("--require-artifacts", action="store_true")
    ap.add_argument("--allow-non-avx512", action="store_true")
    ap.add_argument("--keep-artifacts", action="store_true")
    ap.add_argument("--q4-min-cosine", type=float, default=0.99)
    ap.add_argument("--q4-max-rmse", type=float, default=0.03)
    ap.add_argument("--bf16-min-cosine", type=float, default=0.99)
    ap.add_argument("--bf16-max-rmse", type=float, default=0.10)
    ap.add_argument("--bf16-max-abs", type=float, default=2.0)
    args = ap.parse_args(argv)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image = args.image.resolve() if args.image else out_dir / "fixtures" / "synthetic_form_1152x896.ppm"
    if args.image is None:
        _write_large_form(image)

    flags = _cpu_flags()
    has_avx512 = "avx512f" in flags
    report: dict[str, Any] = {
        "schema_version": 1,
        "family": args.family,
        "mode": args.mode,
        "image": str(image),
        "image_geometry": [1152, 896] if args.image is None else None,
        "cpu": {
            "avx512f": has_avx512,
            "avx512_bf16": bool({"avx512_bf16", "avx512bf16"} & flags),
        },
        "layers": list(range(27)) if args.full_layers else [0, 8, 16, 24, 26],
        "phases": {},
        "started_unix": time.time(),
    }
    failures: list[str] = []

    if not has_avx512 and not args.allow_non_avx512:
        reason = "AVX-512F is not visible; pass --allow-non-avx512 for compatibility testing"
        if args.mode in {"all", "q4"}:
            report["phases"]["q8_mmproj_llamacpp"] = _phase("skip", reason=reason)
        if args.mode in {"all", "bf16"}:
            report["phases"]["bf16_pytorch"] = _phase("skip", reason=reason)
        report["status"] = "skip"
        report["skip_reason"] = reason
        (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report["skip_reason"])
        return 1 if args.require_artifacts else 0

    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = str(args.threads)

    if args.mode in {"all", "q4"}:
        missing = _missing([args.q4_mmproj] if args.q4_mmproj else [])
        if args.q4_mmproj is None:
            missing = ["--q4-mmproj"]
        if missing:
            report["phases"]["q8_mmproj_llamacpp"] = _phase("skip", reason="missing " + ", ".join(missing))
            if args.require_artifacts:
                failures.append("q8_mmproj_llamacpp: required artifact missing")
        else:
            q4_out = out_dir / "q8_mmproj_llamacpp"
            q4_runtime = out_dir / "runtime_q4"
            cmd = [
                sys.executable, str(Q4_SUITE),
                "--gguf", str(args.q4_mmproj.resolve()),
                "--image", str(image),
                "--limit", "1",
                "--output-dir", str(q4_out),
                "--runtime-dir", str(q4_runtime),
                "--image-max-tokens", "1024",
                "--threads", str(args.threads),
                "--ck-threads", str(args.threads),
                "--min-cosine", str(args.q4_min_cosine),
                "--max-rmse", str(args.q4_max_rmse),
            ]
            try:
                _run(cmd, env=env, log=out_dir / "q8_mmproj_llamacpp.log")
                report["phases"]["q8_mmproj_llamacpp"] = _phase("pass", report=str(q4_out / "summary.json"))
            except subprocess.CalledProcessError as exc:
                report["phases"]["q8_mmproj_llamacpp"] = _phase("fail", returncode=exc.returncode, log=str(out_dir / "q8_mmproj_llamacpp.log"))
                failures.append("q8_mmproj_llamacpp parity failed")
            finally:
                if not args.keep_artifacts:
                    shutil.rmtree(q4_runtime, ignore_errors=True)

    if args.mode in {"all", "bf16"}:
        required = [args.bf16_checkpoint, args.bf16_runtime_dir, args.bf16_weights_bump]
        missing = ["BF16 checkpoint/runtime/weights"] if any(path is None for path in required) else _missing([path for path in required if path])
        if importlib.util.find_spec("torch") is None:
            missing.append("PyTorch")
        if importlib.util.find_spec("transformers") is None:
            missing.append("transformers")
        if missing:
            report["phases"]["bf16_pytorch"] = _phase("skip", reason="missing " + ", ".join(missing))
            if args.require_artifacts:
                failures.append("bf16_pytorch: required artifact or dependency missing")
        else:
            bf16_out = out_dir / "bf16_pytorch"
            layers = list(range(27)) if args.full_layers else [0, 8, 16, 24, 26]
            selectors = [f"layer_out@{layer}" for layer in layers] + ["vision_output"]
            cmd = [
                sys.executable, str(BF16_SUITE),
                "--checkpoint", str(args.bf16_checkpoint.resolve()),
                "--runtime-dir", str(args.bf16_runtime_dir.resolve()),
                "--weights-bump", str(args.bf16_weights_bump.resolve()),
                "--image", str(image),
                "--out-dir", str(bf16_out),
                "--threads", str(args.threads),
                "--attn-implementation", "eager",
                "--min-cosine", str(args.bf16_min_cosine),
                "--max-relative-rmse", "0.10",
                "--final-max-rmse", str(args.bf16_max_rmse),
                "--final-max-abs", str(args.bf16_max_abs),
            ]
            for selector in selectors:
                cmd.extend(["--selector", selector])
            try:
                _run(cmd, env=env, log=out_dir / "bf16_pytorch.log")
                report["phases"]["bf16_pytorch"] = _phase("pass", report=str(bf16_out / "report.json"), selectors=selectors)
            except subprocess.CalledProcessError as exc:
                report["phases"]["bf16_pytorch"] = _phase("fail", returncode=exc.returncode, log=str(out_dir / "bf16_pytorch.log"), selectors=selectors)
                failures.append("bf16_pytorch parity failed")
            finally:
                if not args.keep_artifacts:
                    _cleanup_raw_tensors(bf16_out)

    ran = [phase for phase in report["phases"].values() if phase["status"] != "skip"]
    report["failures"] = failures
    report["status"] = "fail" if failures else ("pass" if ran else "skip")
    report["elapsed_sec"] = time.time() - report["started_unix"]
    summary = out_dir / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "phases": report["phases"], "summary": str(summary)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
