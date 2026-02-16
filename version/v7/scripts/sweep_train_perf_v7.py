#!/usr/bin/env python3
"""
Sweep v7 CK training performance over model shape, token budget, and thread count.

Resume-safe behavior:
- existing per-run JSON is reused when --resume is enabled (default)
- consolidated outputs are rewritten after every run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    if not vals:
        raise ValueError(f"Expected non-empty int list, got {raw!r}")
    return vals


def repo_root_from_file(path: Path) -> Path:
    # version/v7/scripts/<file> -> repo root
    return path.resolve().parents[3]


def run_cmd(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None, log_path: Optional[Path] = None) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = p.stdout or ""
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(out, encoding="utf-8")
    return p.returncode, out


def extract_metrics(report: dict) -> Dict[str, object]:
    sp = report.get("step_profile") or {}
    return {
        "ck_total_ms": sp.get("ck_total_ms"),
        "ck_avg_step_ms": sp.get("ck_avg_step_ms"),
        "train_tok_s": sp.get("train_tok_s"),
        "processed_tokens": sp.get("processed_tokens"),
        "steps": report.get("steps"),
        "micro_steps": report.get("micro_steps"),
        "final_ck_loss": report.get("final_ck_loss"),
        "pass_parity": report.get("pass_parity"),
    }


def write_outputs(out_dir: Path, rows: List[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "sweep_results.json"
    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_cols = [
        "d_model", "hidden", "total_tokens", "seq_len", "grad_accum", "epochs", "threads",
        "rc", "wall_s", "ck_total_ms", "ck_avg_step_ms", "train_tok_s", "processed_tokens",
        "final_ck_loss", "pass_parity", "report_json", "run_dir", "log_path",
    ]
    csv_path = out_dir / "sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in csv_cols})

    grouped: Dict[Tuple[int, int, int], List[dict]] = {}
    for r in rows:
        if int(r.get("rc", 1)) != 0:
            continue
        if not isinstance(r.get("train_tok_s"), (int, float)):
            continue
        key = (int(r["d_model"]), int(r["hidden"]), int(r["total_tokens"]))
        grouped.setdefault(key, []).append(r)

    best_rows: List[dict] = []
    for key, vals in sorted(grouped.items()):
        best = max(vals, key=lambda x: float(x["train_tok_s"]))
        base = next((v for v in vals if int(v["threads"]) == 1), None)
        speedup = None
        if base and isinstance(base.get("train_tok_s"), (int, float)) and base["train_tok_s"]:
            speedup = float(best["train_tok_s"]) / float(base["train_tok_s"])
        best_rows.append(
            {
                "d_model": key[0],
                "hidden": key[1],
                "total_tokens": key[2],
                "best_threads": int(best["threads"]),
                "best_tok_s": best.get("train_tok_s"),
                "best_ck_total_ms": best.get("ck_total_ms"),
                "tok_s_at_th1": base.get("train_tok_s") if base else None,
                "speedup_vs_th1": speedup,
            }
        )

    best_path = out_dir / "sweep_best_by_config.json"
    best_path.write_text(json.dumps(best_rows, indent=2), encoding="utf-8")

    md = out_dir / "sweep_summary.md"
    lines = [
        "# v7 Train Sweep Summary",
        "",
        "## Best Threads Per Config",
        "",
        "| d_model | hidden | total_tokens | best_threads | best_tok_s | speedup_vs_th1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for b in best_rows:
        best_tok_s = float(b["best_tok_s"]) if b.get("best_tok_s") is not None else 0.0
        spd = float(b["speedup_vs_th1"]) if b.get("speedup_vs_th1") is not None else 0.0
        lines.append(
            f"| {b['d_model']} | {b['hidden']} | {b['total_tokens']} | {b['best_threads']} | {best_tok_s:.2f} | {spd:.2f}x |"
        )
    lines += [
        "",
        f"Raw JSON: `{raw_path}`",
        f"Raw CSV: `{csv_path}`",
        f"Best JSON: `{best_path}`",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep v7 CK train throughput over shape/tokens/threads.")
    ap.add_argument("--out-dir", default="/tmp/v7_sweep")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--sleep-seconds", type=float, default=0.0)

    ap.add_argument("--d-models", default="128,256,384")
    ap.add_argument("--hidden-list", default=None)
    ap.add_argument("--hidden-mult", type=float, default=4.0)
    ap.add_argument("--tokens", default="1024,2048,4096")
    ap.add_argument("--threads", default="1,2,4,8,12")

    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--template", default="qwen3")
    ap.add_argument("--init", default="xavier_uniform")
    ap.add_argument("--prompt", default="Hello!")
    ap.add_argument("--warmup-tokens", type=int, default=64)

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    logs_dir = out_dir / "logs"
    runs_root = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    repo = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_file(Path(__file__))
    ck_run = repo / "version/v7/scripts/ck_run_v7.py"
    if not ck_run.exists():
        raise FileNotFoundError(f"Missing {ck_run}")

    resume = bool(args.resume and not args.no_resume)
    d_models = parse_int_list(args.d_models)
    tokens = parse_int_list(args.tokens)
    threads = parse_int_list(args.threads)

    if args.hidden_list:
        hiddens = parse_int_list(args.hidden_list)
        if len(hiddens) != len(d_models):
            raise ValueError("--hidden-list length must match --d-models")
    else:
        hiddens = [int(d * args.hidden_mult) for d in d_models]

    dim_pairs = list(zip(d_models, hiddens))

    # init + warm
    for d_model, hidden in dim_pairs:
        run_dir = runs_root / f"d{d_model}_h{hidden}"
        if not (run_dir / "weights.bump").exists() or not (run_dir / "weights_manifest.json").exists():
            init_cmd = [
                args.python, str(ck_run), "init",
                "--run", str(run_dir),
                "--init", args.init,
                "--layers", str(args.layers),
                "--vocab-size", str(args.vocab_size),
                "--embed-dim", str(d_model),
                "--hidden-dim", str(hidden),
                "--template", args.template,
            ]
            print(f"[init] d={d_model} h={hidden}")
            rc, _ = run_cmd(init_cmd, cwd=repo, log_path=logs_dir / f"init_d{d_model}_h{hidden}.log")
            if rc != 0:
                raise RuntimeError(f"init failed for d={d_model}, h={hidden}")

        warm_json = out_dir / f"warm_d{d_model}_h{hidden}.json"
        if (not warm_json.exists()) or (not resume):
            env = os.environ.copy()
            env["CK_NUM_THREADS"] = "1"
            warm_cmd = [
                args.python, str(ck_run), "train",
                "--run", str(run_dir),
                "--backend", "ck",
                "--train-epochs", "1",
                "--train-seq-len", str(args.seq_len),
                "--train-total-tokens", str(args.warmup_tokens),
                "--train-grad-accum", str(args.grad_accum),
                "--prompt", args.prompt,
                "--profile-train", "none",
                "--train-json-out", str(warm_json),
            ]
            print(f"[warm] d={d_model} h={hidden}")
            rc, _ = run_cmd(warm_cmd, cwd=repo, env=env, log_path=logs_dir / f"warm_d{d_model}_h{hidden}.log")
            if rc != 0:
                raise RuntimeError(f"warm failed for d={d_model}, h={hidden}")

    rows: List[dict] = []
    total = len(dim_pairs) * len(tokens) * len(threads)
    idx = 0
    executed = 0

    for d_model, hidden in dim_pairs:
        run_dir = runs_root / f"d{d_model}_h{hidden}"
        for tok in tokens:
            for th in threads:
                idx += 1
                out_json = out_dir / f"res_d{d_model}_h{hidden}_tok{tok}_th{th}.json"
                log_path = logs_dir / f"run_d{d_model}_h{hidden}_tok{tok}_th{th}.log"
                train_cmd = [
                    args.python, str(ck_run), "train",
                    "--run", str(run_dir),
                    "--backend", "ck",
                    "--train-epochs", str(args.epochs),
                    "--train-seq-len", str(args.seq_len),
                    "--train-total-tokens", str(tok),
                    "--train-grad-accum", str(args.grad_accum),
                    "--prompt", args.prompt,
                    "--profile-train", "none",
                    "--train-json-out", str(out_json),
                ]
                rec = {
                    "d_model": d_model,
                    "hidden": hidden,
                    "total_tokens": tok,
                    "seq_len": args.seq_len,
                    "grad_accum": args.grad_accum,
                    "epochs": args.epochs,
                    "threads": th,
                    "report_json": str(out_json),
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                    "cmd": " ".join(shlex.quote(c) for c in train_cmd),
                }

                if resume and out_json.exists():
                    try:
                        report = json.loads(out_json.read_text(encoding="utf-8"))
                        rec.update(extract_metrics(report))
                        rec["rc"] = 0
                        rec["wall_s"] = 0.0
                        print(f"[{idx:02d}/{total}] reuse d={d_model} h={hidden} tok={tok} th={th} tok/s={rec.get('train_tok_s')}")
                    except Exception:
                        rec["rc"] = 99
                        rec["error"] = "existing JSON parse failed"
                    rows.append(rec)
                    write_outputs(out_dir, rows)
                    continue

                env = os.environ.copy()
                env["CK_NUM_THREADS"] = str(th)
                t0 = time.time()
                rc, out = run_cmd(train_cmd, cwd=repo, env=env, log_path=log_path)
                wall = time.time() - t0
                rec["rc"] = rc
                rec["wall_s"] = wall

                if rc == 0 and out_json.exists():
                    try:
                        report = json.loads(out_json.read_text(encoding="utf-8"))
                        rec.update(extract_metrics(report))
                    except Exception as e:
                        rec["error"] = f"json parse error: {e}"
                else:
                    rec["error"] = "\n".join(out.splitlines()[-20:]) if out else "failed"

                rows.append(rec)
                executed += 1
                print(f"[{idx:02d}/{total}] run   d={d_model} h={hidden} tok={tok} th={th} rc={rc} tok/s={rec.get('train_tok_s')}")
                write_outputs(out_dir, rows)

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

                if args.max_runs > 0 and executed >= args.max_runs:
                    print(f"Reached --max-runs={args.max_runs}. Partial sweep saved.")
                    print(f"CSV: {out_dir / 'sweep_results.csv'}")
                    return 0

    write_outputs(out_dir, rows)
    print("Sweep complete.")
    print(f"CSV: {out_dir / 'sweep_results.csv'}")
    print(f"Best: {out_dir / 'sweep_best_by_config.json'}")
    print(f"Summary: {out_dir / 'sweep_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
