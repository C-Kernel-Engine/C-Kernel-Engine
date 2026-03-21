#!/usr/bin/env python3
"""
run_v7_stabilization_nightly_v7.py

Nightly v7 core stabilization matrix:
- strict dataset/tokenizer gates (ascii_bpe + bpe roundtrip)
- CK-vs-PyTorch training parity regimen sweep across depth/token budgets
- machine-readable scorecard for weekly tracking

Outputs:
  - training_stabilization_scorecard_latest.json
  - training_stabilization_scorecard_latest.md
  - training_stabilization_history.jsonl (append-only)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PIPELINE = SCRIPT_DIR / "train_data_pipeline_v7.py"
PARITY_REGIMEN = SCRIPT_DIR / "run_training_parity_regimen_v7.py"


def _default_train_run_root() -> Path:
    cache_env = os.environ.get("CK_CACHE_DIR")
    if cache_env:
        base = Path(cache_env).expanduser()
        if base.name == "train":
            return base / "nightly_stabilization"
        if base.name == "models":
            return base / "train" / "nightly_stabilization"
        return base / "models" / "train" / "nightly_stabilization"
    return Path.home() / ".cache" / "ck-engine-v7" / "models" / "train" / "nightly_stabilization"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pick_python(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _parse_csv_ints(spec: str, *, name: str, min_value: int = 1) -> List[int]:
    out: List[int] = []
    for tok in [t.strip() for t in str(spec).split(",") if t.strip()]:
        try:
            v = int(tok)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Invalid {name} value '{tok}'") from exc
        if v < min_value:
            raise ValueError(f"Invalid {name} value '{tok}' (< {min_value})")
        out.append(v)
    if not out:
        raise ValueError(f"Empty {name} list")
    return sorted(set(out))


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cmd(cmd: List[str], *, log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.perf_counter() - t0
    out = proc.stdout if isinstance(proc.stdout, str) else ""
    log_path.write_text(out, encoding="utf-8")
    return {
        "rc": int(proc.returncode),
        "duration_s": float(dt),
        "log": str(log_path),
    }


def _abs_or_root(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _stage_by_id(report: Dict[str, Any], stage_id: str) -> Optional[Dict[str, Any]]:
    stages = report.get("stages")
    if not isinstance(stages, list):
        return None
    for row in stages:
        if isinstance(row, dict) and str(row.get("id")) == stage_id:
            return row
    return None


def _build_stability_grid(token_budget: int, seq_len: int, grad_accum_values: List[int]) -> str:
    max_accum = max(grad_accum_values)
    cases = [(2, 1), (4, 2), (8, max_accum)]
    uniq: List[tuple[int, int]] = []
    seen = set()
    for e, g in cases:
        k = (e, g)
        if k in seen:
            continue
        uniq.append(k)
        seen.add(k)
    toks = []
    for epochs, g in uniq:
        denom = max(1, seq_len * g * epochs)
        steps = max(4, int(token_budget) // denom)
        toks.append(f"{epochs}x{g}x{steps}")
    return ",".join(toks)


def _run_tokenizer_gate(
    *,
    python_exec: str,
    run_root: Path,
    dataset_path: Path,
    mode: str,
    seq_len: int,
    lr: float,
    seed: int,
    d_model: int,
    hidden: int,
    vocab_size: int,
    bpe_vocab_size: int,
    roundtrip_max_lines: int,
    roundtrip_sample_limit: int,
) -> Dict[str, Any]:
    mode_dir = run_root / "tokenizer_gates" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    json_out = mode_dir / "pipeline_gate.json"
    log_out = mode_dir / "pipeline_gate.log"
    work_dir = mode_dir / ".ck_pipeline" / "gate"

    cmd = [
        python_exec,
        str(TRAIN_PIPELINE),
        "--run",
        str(mode_dir),
        "--init-if-missing",
        "--init",
        "xavier_uniform",
        "--template",
        "qwen3",
        "--tokenizer",
        mode,
        "--require-svg-rows",
        "--strict-data-gates",
        "--prepare-only",
        "--no-open-visualizer",
        "--data",
        str(dataset_path),
        "--work-dir",
        str(work_dir),
        "--vocab-size",
        str(vocab_size),
        "--bpe-vocab-size",
        str(bpe_vocab_size),
        "--layers",
        "1",
        "--embed-dim",
        str(d_model),
        "--hidden-dim",
        str(hidden),
        "--num-heads",
        "8",
        "--num-kv-heads",
        "4",
        "--context-len",
        str(max(128, seq_len)),
        "--epochs",
        "1",
        "--seq-len",
        str(seq_len),
        "--total-tokens",
        str(max(seq_len + 1, 512)),
        "--grad-accum",
        "1",
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--roundtrip-max-lines",
        str(roundtrip_max_lines),
        "--roundtrip-sample-limit",
        str(roundtrip_sample_limit),
        "--json-out",
        str(json_out),
    ]
    cmd.append("--require-ascii-data")

    run_info = _run_cmd(cmd, log_path=log_out)
    payload: Dict[str, Any] = {}
    if json_out.exists():
        payload = _json_load(json_out)

    dataset_qc = payload.get("dataset_qc") if isinstance(payload.get("dataset_qc"), dict) else {}
    tokenizer_roundtrip = payload.get("tokenizer_roundtrip") if isinstance(payload.get("tokenizer_roundtrip"), dict) else {}
    checks = dataset_qc.get("checks") if isinstance(dataset_qc.get("checks"), dict) else {}
    line_eval = tokenizer_roundtrip.get("line_eval") if isinstance(tokenizer_roundtrip.get("line_eval"), dict) else {}

    exact_match = bool(tokenizer_roundtrip.get("exact_match", False))
    ascii_gate = bool(checks.get("ascii_gate", False))
    svg_gate = bool(checks.get("svg_row_gate", False))
    line_rate = float(line_eval.get("exact_match_rate", 0.0) or 0.0)
    gate_pass = bool(run_info["rc"] == 0 and exact_match and svg_gate and (ascii_gate if mode == "ascii_bpe" else True))

    return {
        "mode": mode,
        "status": "PASS" if gate_pass else "FAIL",
        "rc": int(run_info["rc"]),
        "duration_s": float(run_info["duration_s"]),
        "artifact_json": str(json_out),
        "artifact_log": str(log_out),
        "metrics": {
            "dataset_ascii_gate": ascii_gate,
            "dataset_svg_gate": svg_gate,
            "dataset_non_empty_lines": int(dataset_qc.get("non_empty_lines", 0) or 0),
            "roundtrip_exact_match": exact_match,
            "roundtrip_line_exact_rate": line_rate,
            "roundtrip_evaluated_lines": int(line_eval.get("evaluated_lines", 0) or 0),
            "token_count": int(tokenizer_roundtrip.get("token_count", 0) or 0),
        },
    }


def _run_regimen_case(
    *,
    python_exec: str,
    case_dir: Path,
    case_id: str,
    layer: int,
    token_budget: int,
    seq_len: int,
    grad_accum_sweep: str,
    sweep_epochs: int,
    forward_epochs: int,
    lr: float,
    seed: int,
    vocab: int,
    d_model: int,
    hidden: int,
    loss_tol: float,
    param_tol: float,
    ck_loss_backend: str,
    train_text: str,
    runtime_checks: bool,
    backend_xray: bool,
    force_regimen: bool,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    json_out = case_dir / "training_parity_regimen_latest.json"
    md_out = case_dir / "training_parity_regimen_latest.md"
    logs_dir = case_dir / "training_parity_regimen_logs"
    case_log = case_dir / "regimen_runner.log"

    accum_values = _parse_csv_ints(grad_accum_sweep, name="grad_accum_sweep", min_value=1)
    max_accum = max(accum_values)
    sweep_steps_per_epoch = max(2, int(token_budget) // max(1, seq_len * max_accum * sweep_epochs))
    stability_grid = _build_stability_grid(token_budget=int(token_budget), seq_len=int(seq_len), grad_accum_values=accum_values)

    cmd = [
        python_exec,
        str(PARITY_REGIMEN),
        "--json-out",
        str(json_out),
        "--md-out",
        str(md_out),
        "--logs-dir",
        str(logs_dir),
        "--no-stop-on-fail",
        "--no-skip-if-unchanged",
        "--seed",
        str(seed),
        "--seq-len",
        str(seq_len),
        "--vocab",
        str(vocab),
        "--d-model",
        str(d_model),
        "--hidden",
        str(hidden),
        "--num-layers",
        str(layer),
        "--lr",
        str(lr),
        "--loss-tol",
        str(loss_tol),
        "--param-tol",
        str(param_tol),
        "--ck-loss-backend",
        str(ck_loss_backend),
        "--forward-epochs",
        str(forward_epochs),
        "--grad-accum-sweep",
        str(grad_accum_sweep),
        "--sweep-epochs",
        str(sweep_epochs),
        "--sweep-steps-per-epoch",
        str(sweep_steps_per_epoch),
        "--stability-grid",
        str(stability_grid),
        "--train-text",
        str(train_text),
    ]
    if run_dir is not None:
        cmd.extend(["--run-dir", str(run_dir)])
    if not runtime_checks:
        cmd.append("--no-runtime-checks")
    if not backend_xray:
        cmd.append("--no-backend-xray")
    if force_regimen:
        cmd.append("--force")

    run_info = _run_cmd(cmd, log_path=case_log)
    report: Dict[str, Any] = {}
    if json_out.exists():
        report = _json_load(json_out)

    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    stages = report.get("stages") if isinstance(report.get("stages"), list) else []
    a1 = _stage_by_id(report, "A1") or {}
    e1 = _stage_by_id(report, "E1") or {}
    f1 = _stage_by_id(report, "F1") or {}

    final_ck_loss = math.nan
    final_torch_loss = math.nan
    a1_art = _abs_or_root(a1.get("artifact_json") if isinstance(a1.get("artifact_json"), str) else None)
    if a1_art is not None and a1_art.exists():
        try:
            a1_payload = _json_load(a1_art)
            final_ck_loss = float(a1_payload.get("final_ck_loss", math.nan))
            final_torch_loss = float(a1_payload.get("final_torch_loss", math.nan))
        except Exception:
            pass

    case_pass = bool(summary.get("passed", False)) and int(run_info["rc"]) == 0
    replay_pass: Optional[bool]
    replay_accum_pass: Optional[bool]
    if isinstance(e1, dict) and e1:
        replay_pass = str(e1.get("status", "")).upper() == "PASS"
    else:
        replay_pass = None
    if isinstance(f1, dict) and f1:
        replay_accum_pass = str(f1.get("status", "")).upper() == "PASS"
    else:
        replay_accum_pass = None

    return {
        "case_id": case_id,
        "layers": int(layer),
        "token_budget": int(token_budget),
        "seq_len": int(seq_len),
        "sweep_steps_per_epoch": int(sweep_steps_per_epoch),
        "stability_grid": str(stability_grid),
        "grad_accum_sweep": [int(v) for v in accum_values],
        "status": "PASS" if case_pass else "FAIL",
        "rc": int(run_info["rc"]),
        "duration_s": float(run_info["duration_s"]),
        "artifact_json": str(json_out),
        "artifact_md": str(md_out),
        "artifact_log": str(case_log),
        "metrics": {
            "total_stages": int(summary.get("total_stages", len(stages)) or len(stages)),
            "passed_stages": int(summary.get("passed_stages", 0) or 0),
            "failed_stage_ids": list(summary.get("failed_stage_ids", []) or []),
            "a1_max_loss_abs_diff": float(((a1.get("metrics") or {}) if isinstance(a1, dict) else {}).get("max_loss_abs_diff", math.nan)),
            "a1_mean_loss_abs_diff": float(((a1.get("metrics") or {}) if isinstance(a1, dict) else {}).get("mean_loss_abs_diff", math.nan)),
            "a1_final_param_max_abs_diff": float(((a1.get("metrics") or {}) if isinstance(a1, dict) else {}).get("final_param_max_abs_diff", math.nan)),
            "replay_determinism_pass": replay_pass,
            "replay_accum_pass": replay_accum_pass,
            "final_ck_loss": float(final_ck_loss),
            "final_torch_loss": float(final_torch_loss),
        },
    }


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# v7 Core Stabilization Scorecard")
    lines.append("")
    lines.append(f"- Generated: `{payload.get('generated_at', '')}`")
    lines.append(f"- Passed: `{payload.get('summary', {}).get('passed', False)}`")
    lines.append("")

    lines.append("## Tokenizer Gates")
    lines.append("")
    lines.append("| Mode | Status | ascii_gate | svg_gate | roundtrip_exact | line_exact_rate |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload.get("tokenizer_gates", []):
        if not isinstance(row, dict):
            continue
        m = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        lines.append(
            "| {mode} | {status} | {ascii_gate} | {svg_gate} | {exact} | {line_rate:.4f} |".format(
                mode=row.get("mode", ""),
                status=row.get("status", ""),
                ascii_gate=m.get("dataset_ascii_gate", "-"),
                svg_gate=m.get("dataset_svg_gate", "-"),
                exact=m.get("roundtrip_exact_match", "-"),
                line_rate=float(m.get("roundtrip_line_exact_rate", 0.0) or 0.0),
            )
        )
    lines.append("")

    lines.append("## Parity Matrix")
    lines.append("")
    lines.append("| Case | Status | Layers | Token Budget | Final CK Loss | Replay | Failed Stages |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload.get("matrix_cases", []):
        if not isinstance(row, dict):
            continue
        m = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        failed = ",".join(str(x) for x in m.get("failed_stage_ids", [])) or "-"
        ck_loss = m.get("final_ck_loss")
        ck_loss_txt = "nan"
        if isinstance(ck_loss, (int, float)) and math.isfinite(float(ck_loss)):
            ck_loss_txt = f"{float(ck_loss):.6f}"
        lines.append(
            f"| {row.get('case_id','')} | {row.get('status','')} | {row.get('layers','')} | {row.get('token_budget','')} | {ck_loss_txt} | {m.get('replay_determinism_pass','-')} | {failed} |"
        )

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    s = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines.append(f"- parity_pass_rate: `{float(s.get('parity_pass_rate', 0.0) or 0.0):.4f}`")
    replay_rate = s.get("replay_determinism_pass_rate")
    replay_txt = "n/a"
    if isinstance(replay_rate, (int, float)) and math.isfinite(float(replay_rate)):
        replay_txt = f"{float(replay_rate):.4f}"
    lines.append(f"- replay_determinism_pass_rate: `{replay_txt}`")
    lines.append(f"- tokenizer_gate_pass_rate: `{float(s.get('tokenizer_gate_pass_rate', 0.0) or 0.0):.4f}`")
    lines.append(f"- final_ck_loss_mean: `{s.get('final_ck_loss_mean', 'nan')}`")
    quality = payload.get("main_run_quality") if isinstance(payload.get("main_run_quality"), dict) else {}
    if quality:
        lines.append(
            f"- main_run_valid_svg_rate: `{quality.get('valid_svg_rate', 'n/a')}` "
            f"(closure_rate=`{quality.get('closure_success_rate', 'n/a')}`)"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run nightly v7 stabilization matrix and emit scorecard artifacts.")
    ap.add_argument("--run-root", type=Path, default=_default_train_run_root())
    ap.add_argument("--dataset", type=Path, default=ROOT / "version" / "v7" / "data" / "svg_assets_train.txt")
    ap.add_argument("--python-exec", type=str, default=None)

    ap.add_argument("--layers", type=str, default="1,2,3,4", help="Comma list, e.g. 1,2,3,4")
    ap.add_argument("--token-budgets", type=str, default="2048,4096", help="Comma list, e.g. 2048,4096")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--grad-accum-sweep", type=str, default="2,4,8")
    ap.add_argument("--sweep-epochs", type=int, default=2)
    ap.add_argument("--forward-epochs", type=int, default=10)

    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss-tol", type=float, default=2e-5)
    ap.add_argument("--param-tol", type=float, default=3e-5)
    ap.add_argument("--ck-loss-backend", choices=["c", "c_ptref", "torch"], default="c_ptref")
    ap.add_argument(
        "--train-text",
        type=str,
        default="<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect x='1' y='1' width='14' height='14'/></svg>",
    )

    ap.add_argument("--roundtrip-max-lines", type=int, default=2048)
    ap.add_argument("--roundtrip-sample-limit", type=int, default=16)
    ap.add_argument("--tokenizer-gate-vocab", type=int, default=320)

    ap.set_defaults(runtime_checks=True)
    ap.add_argument("--runtime-checks", dest="runtime_checks", action="store_true")
    ap.add_argument("--no-runtime-checks", dest="runtime_checks", action="store_false")

    ap.set_defaults(backend_xray=True)
    ap.add_argument("--backend-xray", dest="backend_xray", action="store_true")
    ap.add_argument("--no-backend-xray", dest="backend_xray", action="store_false")

    ap.add_argument("--force-regimen", action="store_true", help="Force each regimen case (ignore unchanged skip).")

    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out", type=Path, default=None)
    ap.add_argument("--history-jsonl", type=Path, default=None)
    ap.add_argument("--main-run-dir", type=Path, default=None, help="Optional run-dir to validate current main weights as an extra case.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    if not TRAIN_PIPELINE.exists():
        raise SystemExit(f"Missing script: {TRAIN_PIPELINE}")
    if not PARITY_REGIMEN.exists():
        raise SystemExit(f"Missing script: {PARITY_REGIMEN}")

    python_exec = _pick_python(args.python_exec)
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    layers = _parse_csv_ints(args.layers, name="layers", min_value=1)
    token_budgets = _parse_csv_ints(args.token_budgets, name="token_budgets", min_value=args.seq_len + 1)

    json_out = args.json_out.expanduser().resolve() if args.json_out else (run_root / "training_stabilization_scorecard_latest.json")
    md_out = args.md_out.expanduser().resolve() if args.md_out else (run_root / "training_stabilization_scorecard_latest.md")
    history_jsonl = args.history_jsonl.expanduser().resolve() if args.history_jsonl else (run_root / "training_stabilization_history.jsonl")

    print(f"[stabilization] run_root={run_root}")
    print(f"[stabilization] dataset={dataset_path}")
    print(f"[stabilization] layers={layers} token_budgets={token_budgets} grad_accum={args.grad_accum_sweep}")

    tokenizer_modes = ["ascii_bpe", "bpe"]
    tokenizer_gates: List[Dict[str, Any]] = []
    for mode in tokenizer_modes:
        print(f"[tokenizer-gate] mode={mode}")
        gate = _run_tokenizer_gate(
            python_exec=python_exec,
            run_root=run_root,
            dataset_path=dataset_path,
            mode=mode,
            seq_len=int(args.seq_len),
            lr=float(args.lr),
            seed=int(args.seed),
            d_model=int(args.d_model),
            hidden=int(args.hidden),
            vocab_size=int(args.tokenizer_gate_vocab),
            bpe_vocab_size=int(args.tokenizer_gate_vocab),
            roundtrip_max_lines=int(args.roundtrip_max_lines),
            roundtrip_sample_limit=int(args.roundtrip_sample_limit),
        )
        tokenizer_gates.append(gate)

    matrix_cases: List[Dict[str, Any]] = []
    for layer in layers:
        for token_budget in token_budgets:
            case_id = f"l{layer}_tok{token_budget}"
            print(f"[regimen] case={case_id}")
            case = _run_regimen_case(
                python_exec=python_exec,
                case_dir=run_root / "matrix" / case_id,
                case_id=case_id,
                layer=int(layer),
                token_budget=int(token_budget),
                seq_len=int(args.seq_len),
                grad_accum_sweep=str(args.grad_accum_sweep),
                sweep_epochs=int(args.sweep_epochs),
                forward_epochs=int(args.forward_epochs),
                lr=float(args.lr),
                seed=int(args.seed),
                vocab=int(args.vocab),
                d_model=int(args.d_model),
                hidden=int(args.hidden),
                loss_tol=float(args.loss_tol),
                param_tol=float(args.param_tol),
                ck_loss_backend=str(args.ck_loss_backend),
                train_text=str(args.train_text),
                runtime_checks=bool(args.runtime_checks),
                backend_xray=bool(args.backend_xray),
                force_regimen=bool(args.force_regimen),
            )
            matrix_cases.append(case)

    if args.main_run_dir is not None:
        main_run_dir = args.main_run_dir.expanduser().resolve()
        if not main_run_dir.exists():
            raise SystemExit(f"--main-run-dir not found: {main_run_dir}")
        main_case_id = "main_run"
        print(f"[regimen] case={main_case_id} run_dir={main_run_dir}")
        main_case = _run_regimen_case(
            python_exec=python_exec,
            case_dir=run_root / "matrix" / main_case_id,
            case_id=main_case_id,
            layer=int(layers[-1]),
            token_budget=int(token_budgets[-1]),
            seq_len=int(args.seq_len),
            grad_accum_sweep=str(args.grad_accum_sweep),
            sweep_epochs=int(args.sweep_epochs),
            forward_epochs=int(args.forward_epochs),
            lr=float(args.lr),
            seed=int(args.seed),
            vocab=int(args.vocab),
            d_model=int(args.d_model),
            hidden=int(args.hidden),
            loss_tol=float(args.loss_tol),
            param_tol=float(args.param_tol),
            ck_loss_backend=str(args.ck_loss_backend),
            train_text=str(args.train_text),
            runtime_checks=bool(args.runtime_checks),
            backend_xray=bool(args.backend_xray),
            force_regimen=bool(args.force_regimen),
            run_dir=main_run_dir,
        )
        matrix_cases.append(main_case)

    main_run_quality: Dict[str, Any] = {}
    if args.main_run_dir is not None:
        post_eval = args.main_run_dir.expanduser().resolve() / "post_train_eval.json"
        if post_eval.exists():
            try:
                p = _json_load(post_eval)
                main_run_quality = {
                    "status": str(p.get("status", "")),
                    "valid_svg_rate": p.get("valid_svg_rate"),
                    "closure_success_rate": p.get("closure_success_rate"),
                    "loop_score": p.get("loop_score"),
                    "artifact_json": str(post_eval),
                }
            except Exception:
                main_run_quality = {
                    "status": "error",
                    "artifact_json": str(post_eval),
                }

    total_tokenizer = len(tokenizer_gates)
    tokenizer_pass = sum(1 for g in tokenizer_gates if str(g.get("status")) == "PASS")

    total_cases = len(matrix_cases)
    parity_pass = sum(1 for c in matrix_cases if str(c.get("status")) == "PASS")
    replay_values = [
        (c.get("metrics") or {}).get("replay_determinism_pass")
        for c in matrix_cases
        if isinstance(c.get("metrics"), dict)
    ]
    replay_values = [v for v in replay_values if isinstance(v, bool)]
    replay_pass = sum(1 for v in replay_values if v)

    ck_losses = [
        float((c.get("metrics") or {}).get("final_ck_loss", math.nan))
        for c in matrix_cases
        if isinstance((c.get("metrics") or {}).get("final_ck_loss"), (int, float))
        and math.isfinite(float((c.get("metrics") or {}).get("final_ck_loss")))
    ]
    ck_loss_mean = float(statistics.mean(ck_losses)) if ck_losses else math.nan
    ck_loss_median = float(statistics.median(ck_losses)) if ck_losses else math.nan

    summary = {
        "passed": bool(tokenizer_pass == total_tokenizer and parity_pass == total_cases and total_cases > 0),
        "tokenizer_gate_pass_rate": float(tokenizer_pass / total_tokenizer) if total_tokenizer > 0 else 0.0,
        "parity_pass_rate": float(parity_pass / total_cases) if total_cases > 0 else 0.0,
        "replay_determinism_pass_rate": float(replay_pass / len(replay_values)) if replay_values else None,
        "replay_determinism_cases": int(len(replay_values)),
        "total_tokenizer_gates": int(total_tokenizer),
        "passed_tokenizer_gates": int(tokenizer_pass),
        "total_matrix_cases": int(total_cases),
        "passed_matrix_cases": int(parity_pass),
        "final_ck_loss_mean": None if not math.isfinite(ck_loss_mean) else ck_loss_mean,
        "final_ck_loss_median": None if not math.isfinite(ck_loss_median) else ck_loss_median,
        "main_run_valid_svg_rate": main_run_quality.get("valid_svg_rate"),
        "main_run_closure_success_rate": main_run_quality.get("closure_success_rate"),
    }

    payload = {
        "generated_at": _utc_now_iso(),
        "run_root": str(run_root),
        "dataset": str(dataset_path),
        "config": {
            "python_exec": str(python_exec),
            "layers": [int(v) for v in layers],
            "token_budgets": [int(v) for v in token_budgets],
            "seq_len": int(args.seq_len),
            "grad_accum_sweep": [int(v) for v in _parse_csv_ints(args.grad_accum_sweep, name="grad_accum_sweep", min_value=1)],
            "sweep_epochs": int(args.sweep_epochs),
            "forward_epochs": int(args.forward_epochs),
            "vocab": int(args.vocab),
            "d_model": int(args.d_model),
            "hidden": int(args.hidden),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "loss_tol": float(args.loss_tol),
            "param_tol": float(args.param_tol),
            "ck_loss_backend": str(args.ck_loss_backend),
            "runtime_checks": bool(args.runtime_checks),
            "backend_xray": bool(args.backend_xray),
            "force_regimen": bool(args.force_regimen),
            "main_run_dir": (str(args.main_run_dir.expanduser().resolve()) if args.main_run_dir is not None else None),
        },
        "tokenizer_gates": tokenizer_gates,
        "matrix_cases": matrix_cases,
        "main_run_quality": main_run_quality,
        "summary": summary,
    }

    _json_dump(json_out, payload)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(_render_markdown(payload), encoding="utf-8")

    history_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with history_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "generated_at": payload.get("generated_at"),
            "summary": summary,
            "dataset": str(dataset_path),
            "layers": [int(v) for v in layers],
            "token_budgets": [int(v) for v in token_budgets],
            "seq_len": int(args.seq_len),
        }) + "\n")

    print(f"[done] json={json_out}")
    print(f"[done] md={md_out}")
    print(f"[done] history={history_jsonl}")
    if bool(summary.get("passed", False)):
        print("[result] V7_STABILIZATION=PASS")
        return 0
    print("[result] V7_STABILIZATION=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
