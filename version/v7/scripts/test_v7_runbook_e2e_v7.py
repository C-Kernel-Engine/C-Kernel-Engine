#!/usr/bin/env python3
"""
v7 runbook (training path) E2E smoke/full validator.

Covers runbook Steps 1-7 in one reproducible flow:
  1) setup paths and clean run-dir
  2) ASCII/SVG dataset cleanup
  3) tokenizer prepare-only + roundtrip/data-lab artifacts
  4) row1/row2 CK-vs-PyTorch canaries
  5) automated parity regimen
  6) full training pipeline
  7) checkpoint promotion, inference build, and visualizer regeneration
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "version" / "v7" / "data" / "svg_assets_train.txt"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("[run]", " ".join(cmd))
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(cmd, cwd=str(cwd), env=full_env, check=True)


def _capture(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    print("[run]", " ".join(cmd))
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.check_output(cmd, cwd=str(cwd), env=full_env, text=True)


def _record(checks: list[Check], name: str, passed: bool, detail: str) -> None:
    checks.append(Check(name=name, passed=passed, detail=detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def _python_exec() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def _latest_ascii_bpe_dir(run_dir: Path) -> Path | None:
    root = run_dir / ".ck_pipeline"
    if not root.exists():
        return None
    cands = sorted([p for p in root.glob("ascii_bpe_*") if p.is_dir()])
    return cands[-1] if cands else None


def _extract_embedded_data(report_html: Path) -> dict[str, Any]:
    html = report_html.read_text(encoding="utf-8", errors="replace")
    prefix = "window.EMBEDDED_IR_DATA = "
    suffix = ";window.dispatchEvent(new Event('ckEmbeddedDataLoaded'));"
    i = html.find(prefix)
    if i < 0:
        raise RuntimeError("embedded data prefix not found")
    i += len(prefix)
    j = html.find(suffix, i)
    if j < 0:
        raise RuntimeError("embedded data suffix not found")
    return json.loads(html[i:j])


def _check_canary_gate(run_dir: Path) -> tuple[bool, list[str], dict[str, Any]]:
    details: list[str] = []
    rows: dict[str, Any] = {}
    ok = True
    th_max = 1e-4
    th_mean = 5e-5
    th_param = 1e-4
    for idx in (1, 2):
        row_root = run_dir / f"parity_svg_row{idx}" / ".ck_pipeline"
        work_dirs = sorted([p for p in row_root.glob("ascii_bpe_*") if p.is_dir()])
        if not work_dirs:
            ok = False
            details.append(f"row{idx}: missing ascii_bpe_* work dir")
            continue
        w = work_dirs[-1]
        ck_path = w / "train_ck.json"
        pt_path = w / "train_torch_ref.json"
        if not ck_path.exists() or not pt_path.exists():
            ok = False
            details.append(f"row{idx}: missing train_ck/train_torch_ref")
            continue
        ck = json.loads(ck_path.read_text(encoding="utf-8"))
        pt = json.loads(pt_path.read_text(encoding="utf-8"))
        c = [float(x.get("loss_ck", 0.0)) for x in ck.get("loss_curve", [])]
        t = [float(x.get("loss", 0.0)) for x in pt.get("loss_curve", [])]
        n = min(len(c), len(t))
        if n == 0:
            ok = False
            details.append(f"row{idx}: empty loss curves")
            continue
        diffs = [abs(c[i] - t[i]) for i in range(n)]
        max_abs = max(diffs)
        mean_abs = mean(diffs)
        final_param = float(ck.get("final_param_max_abs_diff", 1.0))
        row_pass = max_abs <= th_max and mean_abs <= th_mean and final_param <= th_param
        rows[f"row{idx}"] = {
            "steps_compared": n,
            "max_abs_loss_diff": max_abs,
            "mean_abs_loss_diff": mean_abs,
            "final_param_max_abs_diff": final_param,
            "pass": row_pass,
            "work_dir": str(w),
        }
        ok = ok and row_pass
        details.append(
            f"row{idx}: max={max_abs:.3e} mean={mean_abs:.3e} param={final_param:.3e} pass={row_pass}"
        )
    return ok, details, rows


def _mode_cfg(mode: str) -> dict[str, int | float]:
    if mode == "full":
        return {
            "tok_vocab": 320,
            "tok_layers": 2,
            "tok_embed": 64,
            "tok_hidden": 128,
            "tok_epochs": 1,
            "tok_seq": 8,
            "tok_total_tokens": 64,
            "canary_vocab": 2048,
            "canary_layers": 4,
            "canary_embed": 96,
            "canary_hidden": 192,
            "canary_epochs": 10,
            # Step 3.1 canaries are often single SVG rows; keep seq_len short to
            # avoid wrap-heavy false parity drift on tiny token streams.
            "canary_seq": 64,
            "canary_total_tokens": 12288,
            "train_vocab": 320,
            "train_layers": 24,
            "train_embed": 64,
            "train_hidden": 128,
            "train_epochs": 1,
            "train_seq": 32,
            "train_total_tokens": 841472,
            "lr": 5e-4,
        }
    # smoke
    return {
        "tok_vocab": 320,
        "tok_layers": 2,
        "tok_embed": 64,
        "tok_hidden": 128,
        "tok_epochs": 1,
        "tok_seq": 8,
        "tok_total_tokens": 64,
        "canary_vocab": 512,
        "canary_layers": 2,
        "canary_embed": 64,
        "canary_hidden": 128,
        "canary_epochs": 3,
        "canary_seq": 64,
        "canary_total_tokens": 2048,
        "train_vocab": 320,
        "train_layers": 6,
        "train_embed": 64,
        "train_hidden": 128,
        "train_epochs": 1,
        "train_seq": 32,
        "train_total_tokens": 16384,
        "lr": 5e-4,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run v7 runbook steps 1-7 end-to-end")
    ap.add_argument("--run-dir", default="/tmp/v7_runbook_e2e_v7", help="Run directory")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="Input dataset text file")
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--keep-run", action="store_true", help="Do not delete run-dir before starting")
    ap.add_argument("--json-out", default="", help="Optional JSON summary output path")
    args = ap.parse_args()

    py = _python_exec()
    run_dir = Path(args.run_dir).expanduser().resolve()
    data_src = Path(args.data).expanduser().resolve()
    if not data_src.exists():
        print(f"ERROR: data file not found: {data_src}", file=sys.stderr)
        return 2

    cfg = _mode_cfg(args.mode)
    checks: list[Check] = []
    summary: dict[str, Any] = {"mode": args.mode, "run_dir": str(run_dir), "checks": [], "artifacts": {}}

    # Step 1: setup + clean start
    if run_dir.exists() and not args.keep_run:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_txt = run_dir / "svg_train_ascii.txt"
    shutil.copy2(data_src, data_txt)
    _record(checks, "step1_setup", run_dir.exists() and data_txt.exists(), f"run={run_dir} data={data_txt}")

    # Step 0.5: strict ASCII/SVG cleanup
    _run(
        [
            py,
            "version/v7/scripts/prepare_ascii_dataset_v7.py",
            "--input",
            str(data_txt),
            "--output",
            str(data_txt),
            "--input-format",
            "text",
            "--jsonl-text-key",
            "text",
            "--ascii-map-common",
            "--ascii-mode",
            "xml_escape",
            "--svg-only",
        ]
    )
    qc_lines = [ln for ln in data_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    _record(checks, "step0_5_ascii_svg_cleanup", len(qc_lines) >= 2, f"non_empty_lines={len(qc_lines)}")

    # Step 0.7: prepare-only tokenizer/data-lab in canonical run path
    tok_work = run_dir / ".ck_pipeline" / "bpe_preview_e2e"
    _run(
        [
            py,
            "version/v7/scripts/train_data_pipeline_v7.py",
            "--run",
            str(run_dir),
            "--init-if-missing",
            "--init",
            "xavier_uniform",
            "--template",
            "qwen3",
            "--tokenizer",
            "ascii_bpe",
            "--require-svg-rows",
            "--strict-data-gates",
            "--min-valid-svg-rate",
            "0.70",
            "--roundtrip-max-lines",
            "2048",
            "--roundtrip-sample-limit",
            "16",
            "--data",
            str(data_txt),
            "--vocab-size",
            str(int(cfg["tok_vocab"])),
            "--bpe-vocab-size",
            str(int(cfg["tok_vocab"])),
            "--layers",
            str(int(cfg["tok_layers"])),
            "--embed-dim",
            str(int(cfg["tok_embed"])),
            "--hidden-dim",
            str(int(cfg["tok_hidden"])),
            "--epochs",
            str(int(cfg["tok_epochs"])),
            "--seq-len",
            str(int(cfg["tok_seq"])),
            "--total-tokens",
            str(int(cfg["tok_total_tokens"])),
            "--grad-accum",
            "1",
            "--lr",
            str(float(cfg["lr"])),
            "--max-grad-norm",
            "1.0",
            "--seed",
            "42",
            "--work-dir",
            str(tok_work),
            "--prepare-only",
            "--json-out",
            str(run_dir / "step0_prepare_only.json"),
        ]
    )
    data_lab_ok = all((run_dir / p).exists() for p in ("dataset_qc.json", "dataset_profile.json", "tokenizer_roundtrip.json"))
    _record(checks, "step0_7_prepare_only", data_lab_ok, "dataset_qc/dataset_profile/tokenizer_roundtrip present")

    # Step 0.8 roundtrip gate
    _run([py, "version/v7/scripts/test_ascii_bpe_roundtrip_v7.py", "--run", str(run_dir), "--dataset", str(data_txt), "--require-ascii"])
    _record(checks, "step0_8_roundtrip", True, "ascii_bpe roundtrip script passed")

    # Step 3.1 parity canaries
    parity_root = run_dir / "parity_canary"
    parity_root.mkdir(parents=True, exist_ok=True)
    (parity_root / "svg_row1.txt").write_text(qc_lines[0] + "\n", encoding="utf-8")
    (parity_root / "svg_row2.txt").write_text(qc_lines[1] + "\n", encoding="utf-8")
    for idx in (1, 2):
        row_run = run_dir / f"parity_svg_row{idx}"
        row_file = parity_root / f"svg_row{idx}.txt"
        _run(
            [
                py,
                "version/v7/scripts/train_data_pipeline_v7.py",
                "--run",
                str(row_run),
                "--init-if-missing",
                "--init",
                "xavier_uniform",
                "--template",
                "qwen3",
                "--tokenizer",
                "ascii_bpe",
                "--require-svg-rows",
                "--strict-data-gates",
                "--data",
                str(row_file),
                "--vocab-size",
                str(int(cfg["canary_vocab"])),
                "--bpe-vocab-size",
                str(int(cfg["canary_vocab"])),
                "--layers",
                str(int(cfg["canary_layers"])),
                "--embed-dim",
                str(int(cfg["canary_embed"])),
                "--hidden-dim",
                str(int(cfg["canary_hidden"])),
                "--epochs",
                str(int(cfg["canary_epochs"])),
                "--seq-len",
                str(int(cfg["canary_seq"])),
                "--total-tokens",
                str(int(cfg["canary_total_tokens"])),
                "--grad-accum",
                "1",
                "--lr",
                "3e-4",
                "--max-grad-norm",
                "1.0",
                "--seed",
                "42",
                "--train-driver",
                "ck_run",
                "--with-torch-ref",
                "--no-post-train-eval",
                "--no-open-visualizer",
                "--json-out",
                str(row_run / "parity_pipeline.json"),
            ]
        )
    gate_ok, gate_details, gate_rows = _check_canary_gate(run_dir)
    _record(checks, "step3_1_canary_gate", gate_ok, "; ".join(gate_details))
    summary["artifacts"]["canary_gate"] = {"passed": gate_ok, "details": gate_details, "rows": gate_rows}

    # Step 3.2 parity regimen
    _run([py, "version/v7/scripts/run_training_parity_regimen_v7.py", "--run-dir", str(run_dir)])
    regimen_path = run_dir / "training_parity_regimen_latest.json"
    regimen = json.loads(regimen_path.read_text(encoding="utf-8")) if regimen_path.exists() else {}
    regimen_pass = bool(((regimen.get("summary") or {}).get("passed")))
    _record(checks, "step3_2_regimen", regimen_pass, f"summary.passed={regimen_pass}")

    # Step 3.7 full pipeline (functional end-to-end: quality gate non-blocking)
    _run(
        [
            py,
            "version/v7/scripts/train_data_pipeline_v7.py",
            "--run",
            str(run_dir),
            "--init-if-missing",
            "--init",
            "xavier_uniform",
            "--template",
            "qwen3",
            "--tokenizer",
            "ascii_bpe",
            "--require-svg-rows",
            "--roundtrip-max-lines",
            "2048",
            "--roundtrip-sample-limit",
            "16",
            "--data",
            str(data_txt),
            "--vocab-size",
            str(int(cfg["train_vocab"])),
            "--bpe-vocab-size",
            str(int(cfg["train_vocab"])),
            "--layers",
            str(int(cfg["train_layers"])),
            "--embed-dim",
            str(int(cfg["train_embed"])),
            "--hidden-dim",
            str(int(cfg["train_hidden"])),
            "--epochs",
            str(int(cfg["train_epochs"])),
            "--seq-len",
            str(int(cfg["train_seq"])),
            "--total-tokens",
            str(int(cfg["train_total_tokens"])),
            "--grad-accum",
            "1",
            "--lr",
            str(float(cfg["lr"])),
            "--max-grad-norm",
            "1.0",
            "--seed",
            "42",
            "--train-driver",
            "ck_run",
            "--json-out",
            str(run_dir / "pipeline_latest.json"),
        ]
    )
    ck_json = _latest_ascii_bpe_dir(run_dir)
    train_ck_ok = ck_json is not None and (ck_json / "train_ck.json").exists()
    _record(checks, "step3_7_train_pipeline", train_ck_ok, f"latest_work_dir={ck_json}")

    # Step 5 promote
    _run([py, "version/v7/scripts/promote_latest_checkpoint_v7.py", "--run", str(run_dir)])
    promote_ok = (run_dir / "weights.bump").exists() and (run_dir / "weights_manifest.json").exists()
    _record(checks, "step5_promote", promote_ok, "weights.bump + weights_manifest.json present")

    # Step 6 inference build + chat smoke
    _run([py, "version/v7/scripts/ck_run_v7.py", "run", str(run_dir), "--generate-only", "--context-len", "128"])
    chat_out = _capture(
        [
            py,
            "scripts/ck_chat.py",
            "--model-dir",
            str(run_dir / ".ck_build"),
            "--python-tokenizer",
            "--chat-template",
            "none",
            "--show-token-ids",
            "--prompt",
            "<svg",
            "--max-tokens",
            "24",
            "--temperature",
            "0.0",
        ]
    )
    # Basic smoke: command returned and emitted response/prompt section.
    chat_ok = ("Prompt:" in chat_out) and ("Response:" in chat_out)
    _record(checks, "step6_inference_smoke", chat_ok, "ck_chat prompt/response markers present")

    # Step 7 visualizer regeneration
    report_html = run_dir / "ir_report.html"
    _run([py, "version/v7/tools/open_ir_visualizer.py", "--generate", "--run", str(run_dir), "--html-only", "--output", str(report_html)])
    embedded = _extract_embedded_data(report_html)
    files = embedded.get("files", {})
    meta = embedded.get("meta", {})
    run_match = str(meta.get("run_dir", "")) == str(run_dir)
    has_pipeline = isinstance(files.get("training_pipeline"), dict)
    has_regimen = isinstance(files.get("training_parity_regimen"), dict)
    has_canary = isinstance(files.get("training_canary_summary"), dict)
    _record(
        checks,
        "step7_visualizer",
        report_html.exists() and run_match and has_pipeline and has_regimen and has_canary,
        f"report={report_html.exists()} run_match={run_match} pipeline={has_pipeline} regimen={has_regimen} canary={has_canary}",
    )
    summary["artifacts"]["report_html"] = str(report_html)
    summary["artifacts"]["report_generated_at"] = meta.get("generated_at")

    all_passed = all(c.passed for c in checks)
    summary["checks"] = [asdict(c) for c in checks]
    summary["passed"] = all_passed

    out_path = Path(args.json_out).expanduser().resolve() if args.json_out else (run_dir / "runbook_e2e_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] report={out_path}")
    print(f"[result] RUNBOOK_E2E={'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
