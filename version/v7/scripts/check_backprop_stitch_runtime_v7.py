#!/usr/bin/env python3
"""
check_backprop_stitch_runtime_v7.py

Why this script exists (D1 in regimen):
- Generated-runtime stitch smoke test for backprop integration.
- Verifies runtime wiring/shape mapping is correct when using
  ck_run_v7.py train --backend ck.

Checks:
- Manifest dims override conflicting CLI train dims (wiring sanity).
- First checked parity step has no bad tensor/op drift.
- Check-dump artifacts exist so operators can inspect failures quickly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
CK_RUN = SCRIPT_DIR / "ck_run_v7.py"
TEMPLATES_DIR = SCRIPT_DIR.parent / "templates"
TEMPLATE_ALIASES = {
    "gemma": "gemma3",
    "gemma3": "gemma3",
    "llama": "llama",
    "nanbeige": "nanbeige",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "qwen35": "qwen35",
}


def _default_train_root() -> Path:
    env = os.environ.get("CK_CACHE_DIR")
    if env:
        base = Path(env).expanduser()
        if base.name == "train":
            return base
        if base.name == "models":
            return base / "train"
        return base / "models" / "train"
    return Path.home() / ".cache" / "ck-engine-v7" / "models" / "train"


def _run(cmd: list[str]) -> Tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode), str(proc.stdout)


def _first_checked_row(train_report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows = train_report.get("parity_steps")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and bool(row.get("checked")):
            return row
    return None


def _checked_rows(train_report: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows = train_report.get("parity_steps")
    if not isinstance(rows, list):
        return []
    out: list[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and bool(row.get("checked")):
            out.append(row)
    return out


def _resolve_template_path(template: str, template_file: Optional[Path]) -> Optional[Path]:
    if template_file is not None:
        path = Path(template_file)
        return path if path.exists() else None
    text = str(template or "").strip().lower()
    if not text:
        return None
    normalized = TEMPLATE_ALIASES.get(text, text)
    path = TEMPLATES_DIR / f"{normalized}.json"
    return path if path.exists() else None


def _allow_loss_only_runtime_stitch(seq_len: int, template: str, template_file: Optional[Path]) -> bool:
    if int(seq_len) > 1:
        return False
    path = _resolve_template_path(template, template_file)
    if path is None:
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    contract = doc.get("contract")
    contract = contract if isinstance(contract, dict) else {}
    attention = contract.get("attention_contract")
    attention = attention if isinstance(attention, dict) else {}
    return str(attention.get("attn_variant") or "").strip().lower() == "hybrid_recurrent_attention"


def _row_allows_loss_only_relaxation(row: Dict[str, Any], *, max_first_logits_diff: float) -> bool:
    if bool(row.get("oracle_error")) or bool(row.get("first_bad_tensor")):
        return False
    replay_ok = bool(row.get("replay_ok", False))
    if not replay_ok:
        return False
    logits_raw = row.get("logits_max_abs_diff")
    logits_diff = float(logits_raw) if isinstance(logits_raw, (int, float)) else 0.0
    if logits_diff > float(max_first_logits_diff):
        return False
    replay_weight = float(row.get("replay_weight_max_abs_diff", 0.0) or 0.0)
    replay_weight_tol = float(row.get("replay_weight_threshold", 3e-5) or 3e-5)
    if replay_weight > replay_weight_tol:
        return False
    replay_opt = float(row.get("replay_optimizer_state_max_abs_diff", 0.0) or 0.0)
    replay_opt_tol = float(row.get("replay_optimizer_state_threshold", 3e-5) or 3e-5)
    if replay_opt > replay_opt_tol:
        return False
    replay_accum = float(row.get("replay_accum_snapshot_max_abs_diff", 0.0) or 0.0)
    replay_accum_tol = float(row.get("replay_accum_snapshot_threshold", 3e-5) or 3e-5)
    if replay_accum > replay_accum_tol:
        return False
    return True


def _dims_from_manifest(run_dir: Path) -> Dict[str, int]:
    manifest_path = run_dir / "weights_manifest.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = doc.get("config")
    cfg = cfg if isinstance(cfg, dict) else {}
    out = {
        "vocab": int(cfg.get("vocab_size", 0) or 0),
        "d_model": int(cfg.get("embed_dim", cfg.get("hidden_size", 0)) or 0),
        "hidden": int(cfg.get("hidden_size", cfg.get("intermediate_size", 0)) or 0),
        "num_layers": int(cfg.get("num_layers", 0) or 0),
    }
    return out


def _run_smoke(args: argparse.Namespace, run_dir: Path, report_out: Path) -> Dict[str, Any]:
    init_cmd = [
        sys.executable,
        str(CK_RUN),
        "init",
        "--run",
        str(run_dir),
        "--allow-non-cache-run-dir",
        "--train-seed",
        str(args.seed),
        "--layers",
        str(args.init_layers),
        "--vocab-size",
        str(args.init_vocab),
        "--embed-dim",
        str(args.init_d_model),
        "--hidden-dim",
        str(args.init_hidden),
        "--num-heads",
        str(args.init_heads),
        "--num-kv-heads",
        str(args.init_kv_heads),
        "--context-len",
        str(args.seq_len),
        "--template",
        str(args.template),
        "--init",
        str(args.init_method),
    ]
    if args.template_file is not None:
        init_cmd.extend(["--template-file", str(args.template_file)])
    init_rc, init_out = _run(init_cmd)
    if init_rc != 0:
        raise RuntimeError(f"ck_run_v7.py init failed\n{init_out}")

    train_cmd = [
        sys.executable,
        str(CK_RUN),
        "train",
        "--run",
        str(run_dir),
        "--allow-non-cache-run-dir",
        "--backend",
        "ck",
        "--train-epochs",
        str(args.epochs),
        "--train-seq-len",
        str(args.seq_len),
        "--train-total-tokens",
        str(args.total_tokens),
        "--train-grad-accum",
        str(args.grad_accum),
        "--train-optimizer",
        "adamw",
        "--train-lr",
        str(args.lr),
        "--train-max-grad-norm",
        str(args.max_grad_norm),
        "--enforce-production-safety",
        "--train-unsafe-adamw-lr-threshold",
        str(args.unsafe_adamw_lr_threshold),
        "--train-seed",
        str(args.seed),
        # Intentionally conflicting dims: runtime should override from manifest.
        "--train-vocab",
        str(args.train_vocabulary_request),
        "--train-d-model",
        str(args.train_d_model_request),
        "--train-hidden",
        str(args.train_hidden_request),
        "--prompt",
        str(args.prompt),
        "--parity-on",
        "--parity-every",
        str(args.parity_every),
        "--bruteforce-debug",
        "--dump-on-check",
        "--dump-check-topk",
        str(args.dump_check_topk),
        "--train-loss-tol",
        str(args.loss_tol),
        "--train-param-tol",
        str(args.param_tol),
        "--train-json-out",
        str(report_out),
    ]
    train_rc, train_out = _run(train_cmd)
    if train_rc != 0:
        raise RuntimeError(f"ck_run_v7.py train failed\n{train_out}")

    report = json.loads(report_out.read_text(encoding="utf-8"))
    return {
        "report": report,
        "init_cmd": init_cmd,
        "train_cmd": train_cmd,
        "init_output": init_out,
        "train_output": train_out,
    }


def _evaluate(
    report: Dict[str, Any],
    *,
    manifest_dims: Dict[str, int],
    expect_mismatch: bool,
    max_first_loss_diff: float,
    max_first_logits_diff: float,
    require_check_dumps: bool,
    require_all_checked_clean: bool,
    allow_loss_only_relaxation: bool = False,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["pass_parity"] = bool(report.get("pass_parity", False))

    train_dims = report.get("train_dims")
    train_dims = train_dims if isinstance(train_dims, dict) else {}
    requested = train_dims.get("requested")
    requested = requested if isinstance(requested, dict) else {}
    effective = train_dims.get("effective")
    effective = effective if isinstance(effective, dict) else {}
    mismatches = train_dims.get("mismatches")
    mismatches = mismatches if isinstance(mismatches, dict) else {}

    runtime_dim_mismatch: Dict[str, Dict[str, int]] = {}
    for k in ("vocab", "d_model", "hidden", "num_layers"):
        mv = int(manifest_dims.get(k, 0) or 0)
        ev_raw = effective.get(k)
        ev = int(ev_raw) if isinstance(ev_raw, int) else 0
        if mv > 0 and ev > 0 and mv != ev:
            runtime_dim_mismatch[k] = {"manifest": mv, "runtime_effective": ev}
    checks["manifest_dim_wiring"] = {
        "passed": len(runtime_dim_mismatch) == 0,
        "mismatch": runtime_dim_mismatch,
        "effective": effective,
        "requested": requested,
        "source": train_dims.get("source"),
    }

    mismatch_detected = len(mismatches) > 0
    checks["conflicting_request_detected"] = {
        "passed": mismatch_detected if expect_mismatch else True,
        "expect_mismatch": bool(expect_mismatch),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }

    oracle = report.get("oracle")
    oracle = oracle if isinstance(oracle, dict) else {}
    check_dump_files = oracle.get("check_dump_files")
    check_dump_files = check_dump_files if isinstance(check_dump_files, list) else []
    checks["check_dump_artifacts"] = {
        "passed": (len(check_dump_files) > 0) if require_check_dumps else True,
        "require_check_dumps": bool(require_check_dumps),
        "count": int(len(check_dump_files)),
    }

    first = _first_checked_row(report)
    if first is None:
        checks["first_checked_parity_step"] = {
            "passed": False,
            "detail": "missing",
        }
    else:
        loss_diff_raw = first.get("loss_diff")
        logits_diff_raw = first.get("logits_max_abs_diff")
        loss_diff = float(loss_diff_raw) if isinstance(loss_diff_raw, (int, float)) else 0.0
        logits_diff = float(logits_diff_raw) if isinstance(logits_diff_raw, (int, float)) else 0.0
        relaxed_loss_only = bool(allow_loss_only_relaxation) and _row_allows_loss_only_relaxation(
            first,
            max_first_logits_diff=float(max_first_logits_diff),
        )
        first_ok = True
        if int(first.get("step", 0) or 0) != 1:
            first_ok = False
        if bool(first.get("oracle_error")):
            first_ok = False
        if bool(first.get("first_bad_tensor")):
            first_ok = False
        if (not relaxed_loss_only) and int(first.get("slots_compared", 0) or 0) <= 0:
            first_ok = False
        if (not relaxed_loss_only) and loss_diff > float(max_first_loss_diff):
            first_ok = False
        if logits_diff > float(max_first_logits_diff):
            first_ok = False
        checks["first_checked_parity_step"] = {
            "passed": bool(first_ok),
            "step": first.get("step"),
            "oracle_error": first.get("oracle_error"),
            "first_bad_tensor": first.get("first_bad_tensor"),
            "first_bad_op": first.get("first_bad_op"),
            "first_bad_diff": first.get("first_bad_diff"),
            "slots_compared": first.get("slots_compared"),
            "slots_matched": first.get("slots_matched"),
            "loss_diff": loss_diff,
            "logits_max_abs_diff": logits_diff,
            "thresholds": {
                "max_first_loss_diff": float(max_first_loss_diff),
                "max_first_logits_diff": float(max_first_logits_diff),
            },
            "loss_only_relaxation_applied": bool(relaxed_loss_only),
        }

    checked_rows = _checked_rows(report)
    first_bad_checked: Optional[Dict[str, Any]] = None
    for row in checked_rows:
        loss_diff_raw = row.get("loss_diff")
        logits_diff_raw = row.get("logits_max_abs_diff")
        loss_diff = float(loss_diff_raw) if isinstance(loss_diff_raw, (int, float)) else 0.0
        logits_diff = float(logits_diff_raw) if isinstance(logits_diff_raw, (int, float)) else 0.0
        relaxed_loss_only = bool(allow_loss_only_relaxation) and _row_allows_loss_only_relaxation(
            row,
            max_first_logits_diff=float(max_first_logits_diff),
        )
        row_bad = False
        if bool(row.get("oracle_error")):
            row_bad = True
        if bool(row.get("first_bad_tensor")):
            row_bad = True
        if (not relaxed_loss_only) and int(row.get("slots_compared", 0) or 0) <= 0:
            row_bad = True
        if (not relaxed_loss_only) and loss_diff > float(max_first_loss_diff):
            row_bad = True
        if logits_diff > float(max_first_logits_diff):
            row_bad = True
        if row_bad:
            first_bad_checked = {
                "step": row.get("step"),
                "first_bad_tensor": row.get("first_bad_tensor"),
                "first_bad_op": row.get("first_bad_op"),
                "first_bad_diff": row.get("first_bad_diff"),
                "oracle_error": row.get("oracle_error"),
                "loss_diff": loss_diff,
                "logits_max_abs_diff": logits_diff,
                "slots_compared": row.get("slots_compared"),
                "slots_matched": row.get("slots_matched"),
                "loss_only_relaxation_applied": bool(relaxed_loss_only),
            }
            break
    checks["all_checked_steps_clean"] = {
        "passed": (first_bad_checked is None) if require_all_checked_clean else True,
        "require_all_checked_clean": bool(require_all_checked_clean),
        "checked_count": int(len(checked_rows)),
        "first_bad_checked": first_bad_checked,
        "thresholds": {
            "max_loss_diff": float(max_first_loss_diff),
            "max_logits_diff": float(max_first_logits_diff),
        },
    }

    pass_parity_gate = bool(checks["pass_parity"]) if require_all_checked_clean else True
    checks["pass_parity_gate_applied"] = {
        "passed": bool(pass_parity_gate),
        "required": bool(require_all_checked_clean),
        "raw_pass_parity": bool(checks["pass_parity"]),
    }

    passed = (
        bool(pass_parity_gate)
        and bool(checks["manifest_dim_wiring"]["passed"])
        and bool(checks["conflicting_request_detected"]["passed"])
        and bool(checks["first_checked_parity_step"]["passed"])
        and bool(checks["check_dump_artifacts"]["passed"])
        and bool(checks["all_checked_steps_clean"]["passed"])
    )
    return {"passed": bool(passed), "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description="One-step runtime stitch smoke for v7 backprop plumbing.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--total-tokens", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--unsafe-adamw-lr-threshold", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", type=str, default="Hello!")
    ap.add_argument("--parity-every", type=int, default=1)
    ap.add_argument("--loss-tol", type=float, default=2e-5)
    ap.add_argument("--param-tol", type=float, default=3e-5)
    ap.add_argument("--max-first-loss-diff", type=float, default=1e-5)
    ap.add_argument("--max-first-logits-diff", type=float, default=2e-4)
    ap.add_argument("--dump-check-topk", type=int, default=32)
    ap.add_argument("--no-require-check-dumps", action="store_true")
    ap.add_argument("--no-require-all-checked-clean", action="store_true")

    ap.add_argument("--init-method", type=str, default="xavier_uniform")
    ap.add_argument("--init-layers", type=int, default=2)
    ap.add_argument("--init-vocab", type=int, default=128)
    ap.add_argument("--init-d-model", type=int, default=32)
    ap.add_argument("--init-hidden", type=int, default=64)
    ap.add_argument("--init-heads", type=int, default=4)
    ap.add_argument("--init-kv-heads", type=int, default=2)
    ap.add_argument("--template", type=str, default="qwen3",
                    help="Training graph template for the temp runtime run (default: qwen3)")
    ap.add_argument("--template-file", type=Path, default=None,
                    help="Optional custom template JSON path paired with --template")

    ap.add_argument("--train-vocabulary-request", type=int, default=256)
    ap.add_argument("--train-d-model-request", type=int, default=64)
    ap.add_argument("--train-hidden-request", type=int, default=128)
    ap.add_argument("--no-expect-mismatch", action="store_true", help="Do not require train_dims.mismatches to be non-empty")

    ap.add_argument("--keep-run-dir", type=Path, default=None, help="Optional run dir to keep (otherwise temp)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if args.init_d_model % args.init_heads != 0:
        raise SystemExit("--init-d-model must be divisible by --init-heads")
    if args.init_heads % args.init_kv_heads != 0:
        raise SystemExit("--init-heads must be divisible by --init-kv-heads")
    if int(args.grad_accum) <= 0:
        raise SystemExit("--grad-accum must be > 0")

    if not CK_RUN.exists():
        raise SystemExit(f"Missing script: {CK_RUN}")

    if args.keep_run_dir is not None:
        run_dir = args.keep_run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        report_out = run_dir / "train_e2e_latest.json"
        data = _run_smoke(args, run_dir=run_dir, report_out=report_out)
        manifest_dims = _dims_from_manifest(run_dir)
    else:
        train_root = _default_train_root()
        train_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="v7_backprop_stitch_", dir=str(train_root)) as td:
            run_dir = Path(td) / "run"
            report_out = run_dir / "train_e2e_latest.json"
            data = _run_smoke(args, run_dir=run_dir, report_out=report_out)
            manifest_dims = _dims_from_manifest(run_dir)

    eval_out = _evaluate(
        data["report"],
        manifest_dims=manifest_dims,
        expect_mismatch=(not bool(args.no_expect_mismatch)),
        max_first_loss_diff=float(args.max_first_loss_diff),
        max_first_logits_diff=float(args.max_first_logits_diff),
        require_check_dumps=(not bool(args.no_require_check_dumps)),
        require_all_checked_clean=(not bool(args.no_require_all_checked_clean)),
        allow_loss_only_relaxation=_allow_loss_only_runtime_stitch(
            int(args.seq_len),
            str(args.template),
            args.template_file,
        ),
    )

    payload = {
        "format": "v7-backprop-runtime-stitch-smoke",
        "passed": bool(eval_out["passed"]),
        "config": {
            "epochs": int(args.epochs),
            "seq_len": int(args.seq_len),
            "total_tokens": int(args.total_tokens),
            "grad_accum": int(args.grad_accum),
            "lr": float(args.lr),
            "max_grad_norm": float(args.max_grad_norm),
            "seed": int(args.seed),
            "prompt": str(args.prompt),
            "parity_every": int(args.parity_every),
            "init": {
                "template": str(args.template),
                "template_file": str(args.template_file) if args.template_file is not None else None,
                "layers": int(args.init_layers),
                "vocab": int(args.init_vocab),
                "d_model": int(args.init_d_model),
                "hidden": int(args.init_hidden),
                "heads": int(args.init_heads),
                "kv_heads": int(args.init_kv_heads),
                "method": str(args.init_method),
            },
            "train_requested_dims": {
                "vocab": int(args.train_vocabulary_request),
                "d_model": int(args.train_d_model_request),
                "hidden": int(args.train_hidden_request),
            },
            "expect_mismatch": bool(not args.no_expect_mismatch),
            "thresholds": {
                "max_first_loss_diff": float(args.max_first_loss_diff),
                "max_first_logits_diff": float(args.max_first_logits_diff),
            },
            "dump_check_topk": int(args.dump_check_topk),
        },
        "manifest_dims": manifest_dims,
        "checks": eval_out["checks"],
        "train_report": data["report"],
    }

    print("=" * 96)
    print("v7 BACKPROP RUNTIME STITCH SMOKE")
    print("=" * 96)
    print(f"- passed: {payload['passed']}")
    print(f"- pass_parity: {payload['checks']['pass_parity']}")
    print(
        "- pass_parity_gate_applied: %s (required=%s raw=%s)"
        % (
            payload["checks"]["pass_parity_gate_applied"].get("passed"),
            payload["checks"]["pass_parity_gate_applied"].get("required"),
            payload["checks"]["pass_parity_gate_applied"].get("raw_pass_parity"),
        )
    )
    wiring = payload["checks"]["manifest_dim_wiring"]
    print(f"- manifest_dim_wiring: {wiring['passed']} source={wiring.get('source')}")
    first = payload["checks"]["first_checked_parity_step"]
    print(
        "- first_checked_step: %s passed=%s first_bad_tensor=%s loss_diff=%s logits_diff=%s"
        % (
            first.get("step"),
            first.get("passed"),
            first.get("first_bad_tensor"),
            first.get("loss_diff"),
            first.get("logits_max_abs_diff"),
        )
    )
    mismatch = payload["checks"]["conflicting_request_detected"]
    print(
        "- mismatches_detected: %s (count=%s expect=%s)"
        % (mismatch.get("passed"), mismatch.get("mismatch_count"), mismatch.get("expect_mismatch"))
    )
    dump_chk = payload["checks"]["check_dump_artifacts"]
    print(
        "- check_dump_artifacts: %s (count=%s require=%s)"
        % (dump_chk.get("passed"), dump_chk.get("count"), dump_chk.get("require_check_dumps"))
    )
    all_chk = payload["checks"]["all_checked_steps_clean"]
    print(
        "- all_checked_steps_clean: %s (checked=%s require=%s first_bad=%s)"
        % (
            all_chk.get("passed"),
            all_chk.get("checked_count"),
            all_chk.get("require_all_checked_clean"),
            (all_chk.get("first_bad_checked") or {}).get("step"),
        )
    )
    print("=" * 96)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("JSON:", args.json_out)

    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
