#!/usr/bin/env python3
"""
Canonical v7 CK-vs-PyTorch FP32 training benchmark.

This script intentionally reuses the existing `ck_run_v7.py train --backend both`
path so benchmark numbers stay attached to the numerically validated training
surface instead of drifting into a separate micro-harness.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _repo_root_from_file(path: Path) -> Path:
    return path.resolve().parents[3]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_cmd(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> tuple[int, float, str]:
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wall_s = time.time() - t0
    out = proc.stdout or ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(out, encoding="utf-8")
    return proc.returncode, wall_s, out


def _extract_step_profile(report: dict) -> dict:
    if isinstance(report.get("step_profile"), dict):
        return dict(report["step_profile"])
    if isinstance(report.get("training_step_profile"), dict):
        return dict(report["training_step_profile"])
    if "ck_total_ms" in report and "train_tok_s" in report:
        return dict(report)
    return {}


def _render_markdown(summary: dict) -> str:
    env_policy = summary.get("env_policy") or {}
    runtime = summary.get("runtime") or {}
    workload = summary.get("workload") or {}
    perf = summary.get("performance") or {}
    parity = summary.get("parity") or {}
    artifacts = summary.get("artifacts") or {}

    def _fmt_float(value: object, digits: int = 3) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.{digits}f}"
        return "NA"

    lines = [
        "# v7 FP32 Training Benchmark",
        "",
        f"- Generated: `{summary.get('generated_at', 'unknown')}`",
        f"- Backend: `{runtime.get('backend', 'unknown')}`",
        f"- Template: `{runtime.get('template', 'unknown')}`",
        f"- Run dir: `{artifacts.get('run_dir', '')}`",
        "",
        "## Workload",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| layers | {workload.get('layers', 'NA')} |",
        f"| d_model | {workload.get('d_model', 'NA')} |",
        f"| hidden | {workload.get('hidden', 'NA')} |",
        f"| vocab_size | {workload.get('vocab_size', 'NA')} |",
        f"| seq_len | {workload.get('seq_len', 'NA')} |",
        f"| total_tokens | {workload.get('total_tokens', 'NA')} |",
        f"| grad_accum | {workload.get('grad_accum', 'NA')} |",
        f"| epochs | {workload.get('epochs', 'NA')} |",
        "",
        "## Runtime Policy",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| threads | {env_policy.get('threads', 'NA')} |",
        f"| affinity_cpulist | `{env_policy.get('affinity_cpulist', '') or ''}` |",
        f"| CK_NUM_THREADS | {env_policy.get('CK_NUM_THREADS', 'NA')} |",
        f"| OMP_NUM_THREADS | {env_policy.get('OMP_NUM_THREADS', 'NA')} |",
        f"| MKL_NUM_THREADS | {env_policy.get('MKL_NUM_THREADS', 'NA')} |",
        f"| OPENBLAS_NUM_THREADS | {env_policy.get('OPENBLAS_NUM_THREADS', 'NA')} |",
        f"| NUMEXPR_NUM_THREADS | {env_policy.get('NUMEXPR_NUM_THREADS', 'NA')} |",
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| init_wall_s | {_fmt_float(perf.get('init_wall_s'))} |",
        f"| train_wall_s | {_fmt_float(perf.get('train_wall_s'))} |",
        f"| processed_tokens | {perf.get('processed_tokens', 'NA')} |",
        f"| ck_total_ms | {_fmt_float(perf.get('ck_total_ms'))} |",
        f"| torch_total_ms | {_fmt_float(perf.get('torch_total_ms'))} |",
        f"| ck_avg_step_ms | {_fmt_float(perf.get('ck_avg_step_ms'))} |",
        f"| torch_avg_step_ms | {_fmt_float(perf.get('torch_avg_step_ms'))} |",
        f"| ck_train_tok_s | {_fmt_float(perf.get('ck_train_tok_s'))} |",
        f"| torch_train_tok_s | {_fmt_float(perf.get('torch_train_tok_s'))} |",
        f"| ck_speedup_vs_torch | {_fmt_float(perf.get('ck_speedup_vs_torch'))}x |",
        "",
        "## Parity",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| pass_parity | {parity.get('pass_parity', 'NA')} |",
        f"| max_loss_abs_diff | {_fmt_float(parity.get('max_loss_abs_diff'), 6)} |",
        f"| final_param_max_abs_diff | {_fmt_float(parity.get('final_param_max_abs_diff'), 6)} |",
        "",
        "## Artifacts",
        "",
        f"- Raw train report: `{artifacts.get('train_report_json', '')}`",
        f"- Init log: `{artifacts.get('init_log', '')}`",
        f"- Train log: `{artifacts.get('train_log', '')}`",
    ]
    return "\n".join(lines) + "\n"


def _default_env_threads(threads: int) -> Dict[str, str]:
    val = str(threads)
    return {
        "CK_NUM_THREADS": val,
        "OMP_NUM_THREADS": val,
        "MKL_NUM_THREADS": val,
        "OPENBLAS_NUM_THREADS": val,
        "NUMEXPR_NUM_THREADS": val,
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }


def main() -> int:
    repo_root = _repo_root_from_file(Path(__file__))
    default_report_root = repo_root / "version" / "v7" / ".cache" / "reports" / "train_benchmark_fp32"
    default_run_dir = repo_root / "version" / "v7" / ".cache" / "models" / "train" / "benchmark_fp32_canonical"
    default_cache_dir = repo_root / "version" / "v7" / ".cache" / "models"

    ap = argparse.ArgumentParser(description="Canonical v7 CK-vs-PyTorch FP32 training benchmark.")
    ap.add_argument("--repo-root", default=str(repo_root))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--run-dir", default=str(default_run_dir))
    ap.add_argument("--report-root", default=str(default_report_root))
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--reuse-run", action="store_true", help="Skip init if run_dir already has weights.bump + manifest")
    ap.add_argument("--affinity-cpulist", default=None, help="Optional taskset CPU list, for example 0-7")

    ap.add_argument("--template", default="qwen3")
    ap.add_argument("--init", default="xavier_uniform")
    ap.add_argument("--backend", choices=["both", "ck", "pytorch", "torch"], default="both")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--total-tokens", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prompt", default="Hello!")
    ap.add_argument("--train-bridge-lowering", choices=["legacy", "explicit"], default="legacy")
    ap.add_argument("--train-checkpoint-policy", choices=["none", "recompute_attn"], default="none")

    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    report_root = Path(args.report_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    json_out = Path(args.json_out).resolve() if args.json_out else (report_root / "v7_train_benchmark_fp32_latest.json")
    md_out = Path(args.md_out).resolve() if args.md_out else (report_root / "v7_train_benchmark_fp32_latest.md")
    raw_train_json = report_root / "v7_train_benchmark_fp32_train_report.json"
    init_log = report_root / "v7_train_benchmark_fp32_init.log"
    train_log = report_root / "v7_train_benchmark_fp32_train.log"

    ck_run = repo / "version" / "v7" / "scripts" / "ck_run_v7.py"
    if not ck_run.exists():
        raise FileNotFoundError(f"Missing {ck_run}")

    report_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CK_CACHE_DIR", str(default_cache_dir))
    env.update(_default_env_threads(args.threads))

    taskset_prefix: List[str] = []
    if args.affinity_cpulist:
        if shutil.which("taskset") is None:
            raise RuntimeError("--affinity-cpulist requested but taskset is not available")
        taskset_prefix = ["taskset", "-c", str(args.affinity_cpulist)]

    init_needed = not (args.reuse_run and (run_dir / "weights.bump").exists() and (run_dir / "weights_manifest.json").exists())
    init_wall_s = 0.0
    init_cmd = taskset_prefix + [
        args.python,
        str(ck_run),
        "init",
        "--run", str(run_dir),
        "--init", str(args.init),
        "--train-seed", str(args.seed),
        "--layers", str(args.layers),
        "--vocab-size", str(args.vocab_size),
        "--embed-dim", str(args.d_model),
        "--hidden-dim", str(args.hidden),
        "--num-heads", str(args.num_heads),
        "--num-kv-heads", str(args.num_kv_heads),
        "--context-len", str(args.context_len),
        "--template", str(args.template),
        "--train-bridge-lowering", str(args.train_bridge_lowering),
        "--train-checkpoint-policy", str(args.train_checkpoint_policy),
    ]

    if init_needed:
        rc, init_wall_s, _ = _run_cmd(init_cmd, cwd=repo, env=env, log_path=init_log)
        if rc != 0:
            raise SystemExit(f"init failed; see {init_log}")
    else:
        init_log.write_text("reused existing run_dir; init skipped\n", encoding="utf-8")

    train_cmd = taskset_prefix + [
        args.python,
        str(ck_run),
        "train",
        "--run", str(run_dir),
        "--backend", str(args.backend),
        "--train-epochs", str(args.epochs),
        "--train-seq-len", str(args.seq_len),
        "--train-total-tokens", str(args.total_tokens),
        "--train-grad-accum", str(args.grad_accum),
        "--train-lr", str(args.lr),
        "--train-seed", str(args.seed),
        "--train-vocab", str(args.vocab_size),
        "--train-d-model", str(args.d_model),
        "--train-hidden", str(args.hidden),
        "--prompt", str(args.prompt),
        "--profile-train", "none",
        "--train-bridge-lowering", str(args.train_bridge_lowering),
        "--train-checkpoint-policy", str(args.train_checkpoint_policy),
        "--train-json-out", str(raw_train_json),
    ]

    rc, train_wall_s, out = _run_cmd(train_cmd, cwd=repo, env=env, log_path=train_log)
    if rc != 0:
        raise SystemExit(f"train failed; see {train_log}")

    report = json.loads(raw_train_json.read_text(encoding="utf-8"))
    step_profile = _extract_step_profile(report)
    processed_tokens = int(step_profile.get("processed_tokens", 0) or 0)
    ck_total_ms = float(step_profile.get("ck_total_ms", 0.0) or 0.0)
    torch_total_ms = float(step_profile.get("torch_total_ms", 0.0) or 0.0)
    torch_train_tok_s = (processed_tokens / (torch_total_ms / 1000.0)) if torch_total_ms > 0 else None
    ck_speedup_vs_torch = (torch_total_ms / ck_total_ms) if ck_total_ms > 0 and torch_total_ms > 0 else None

    summary = {
        "generated_at": _utc_now_iso(),
        "runtime": {
            "backend": str(args.backend),
            "template": str(args.template),
            "python": str(args.python),
        },
        "workload": {
            "layers": int(args.layers),
            "d_model": int(args.d_model),
            "hidden": int(args.hidden),
            "vocab_size": int(args.vocab_size),
            "context_len": int(args.context_len),
            "seq_len": int(args.seq_len),
            "total_tokens": int(args.total_tokens),
            "grad_accum": int(args.grad_accum),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "lr": float(args.lr),
            "prompt": str(args.prompt),
            "train_bridge_lowering": str(args.train_bridge_lowering),
            "train_checkpoint_policy": str(args.train_checkpoint_policy),
        },
        "env_policy": {
            "threads": int(args.threads),
            "affinity_cpulist": str(args.affinity_cpulist or ""),
            "CK_CACHE_DIR": str(env.get("CK_CACHE_DIR", "")),
            **_default_env_threads(args.threads),
        },
        "performance": {
            "init_wall_s": init_wall_s,
            "train_wall_s": train_wall_s,
            "processed_tokens": processed_tokens,
            "ck_total_ms": ck_total_ms,
            "torch_total_ms": torch_total_ms,
            "ck_avg_step_ms": float(step_profile.get("ck_avg_step_ms", 0.0) or 0.0),
            "torch_avg_step_ms": float(step_profile.get("torch_avg_step_ms", 0.0) or 0.0),
            "ck_train_tok_s": float(step_profile.get("train_tok_s", 0.0) or 0.0),
            "torch_train_tok_s": torch_train_tok_s,
            "ck_speedup_vs_torch": ck_speedup_vs_torch,
        },
        "parity": {
            "pass_parity": bool(report.get("pass_parity", False)),
            "max_loss_abs_diff": float(report.get("max_loss_abs_diff", 0.0) or 0.0),
            "final_param_max_abs_diff": float(report.get("final_param_max_abs_diff", 0.0) or 0.0),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "train_report_json": str(raw_train_json),
            "init_log": str(init_log),
            "train_log": str(train_log),
            "init_cmd": " ".join(shlex.quote(c) for c in init_cmd),
            "train_cmd": " ".join(shlex.quote(c) for c in train_cmd),
            "train_stdout_tail": "\n".join(out.splitlines()[-20:]),
        },
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_out.write_text(_render_markdown(summary), encoding="utf-8")
    print(f"[bench] wrote {json_out}")
    print(f"[bench] wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
