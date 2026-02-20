#!/usr/bin/env python3
"""
Run a v7 training ablation matrix and emit ranked results.

One command:
- executes a grid over architecture/tokenizer/data-size knobs
- stores per-run reports/logs
- writes ranked JSON/CSV/Markdown summaries
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # version/v7/scripts/<this file> -> repo root
    return Path(__file__).resolve().parents[3]


def _python_exec(repo_root: Path) -> str:
    venv_py = repo_root / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError(f"expected non-empty int list, got: {raw!r}")
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    if not out:
        raise ValueError(f"expected non-empty float list, got: {raw!r}")
    return out


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        x = float(v)
        if math.isfinite(x):
            return x
    return None


def _read_text_rows(path: Path) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            rows.append(s)
    return rows


def _subset_rows(rows: list[str], frac: float) -> list[str]:
    if frac >= 0.999999:
        return rows[:]
    k = max(1, int(round(len(rows) * frac)))
    return rows[: min(k, len(rows))]


def _run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = p.stdout or ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(out, encoding="utf-8")
    return int(p.returncode), out


def _extract_ck_loss(report_json: Path) -> dict[str, Any]:
    try:
        payload = json.loads(report_json.read_text(encoding="utf-8"))
    except Exception:
        return {"steps": 0}
    ck = payload.get("ck_loss")
    if not isinstance(ck, dict):
        return {"steps": 0}
    return {
        "steps": int(ck.get("steps", 0) or 0),
        "first": _to_float(ck.get("first")),
        "final": _to_float(ck.get("final")),
        "min": _to_float(ck.get("min")),
        "min_step": int(ck.get("min_step", 0) or 0),
    }


@dataclass(frozen=True)
class SweepCfg:
    embed_dim: int
    hidden_dim: int
    layers: int
    bpe_vocab_size: int
    dataset_frac: float
    total_tokens: int
    num_heads: int
    num_kv_heads: int

    def run_id(self) -> str:
        frac_pct = int(round(self.dataset_frac * 100))
        return (
            f"ed{self.embed_dim}_hd{self.hidden_dim}_l{self.layers}_"
            f"bpe{self.bpe_vocab_size}_df{frac_pct:03d}_"
            f"tok{self.total_tokens}_h{self.num_heads}_kv{self.num_kv_heads}"
        )


def _validate_cfg(cfg: SweepCfg, tokenizer: str) -> None:
    if cfg.embed_dim <= 0 or cfg.hidden_dim <= 0 or cfg.layers <= 0:
        raise ValueError(f"invalid dims in config: {cfg}")
    if cfg.num_heads <= 0 or cfg.num_kv_heads <= 0:
        raise ValueError(f"invalid heads in config: {cfg}")
    if cfg.embed_dim % cfg.num_heads != 0:
        raise ValueError(f"embed_dim must be divisible by num_heads: {cfg}")
    if cfg.num_heads % cfg.num_kv_heads != 0:
        raise ValueError(f"num_heads must be divisible by num_kv_heads: {cfg}")
    if cfg.bpe_vocab_size < 2:
        raise ValueError(f"bpe_vocab_size must be >= 2: {cfg}")
    if tokenizer == "ascii_bpe" and cfg.bpe_vocab_size < 102:
        raise ValueError(
            f"ascii_bpe requires bpe_vocab_size >= 102 (base ASCII symbols): {cfg}"
        )
    if cfg.total_tokens < 2:
        raise ValueError(f"total_tokens must be >= 2: {cfg}")
    if not (0.0 < cfg.dataset_frac <= 1.0):
        raise ValueError(f"dataset_frac must be in (0,1]: {cfg}")


def _build_matrix(args: argparse.Namespace) -> list[SweepCfg]:
    embed_dims = _parse_int_list(args.embed_dims)
    layers_list = _parse_int_list(args.layers_list)
    bpe_vocabs = _parse_int_list(args.bpe_vocab_sizes)
    dataset_fracs = _parse_float_list(args.dataset_fracs)
    total_tokens_list = _parse_int_list(args.total_tokens_list)
    kv_divisors = _parse_int_list(args.kv_divisors)

    matrix: list[SweepCfg] = []
    for ed in embed_dims:
        hidden = int(round(ed * float(args.hidden_mult)))
        for layers in layers_list:
            for bpe in bpe_vocabs:
                for frac in dataset_fracs:
                    for total_tokens in total_tokens_list:
                        for div in kv_divisors:
                            if div <= 0 or (args.num_heads % div) != 0:
                                raise ValueError(
                                    f"invalid kv divisor {div} for num_heads={args.num_heads}"
                                )
                            num_kv = args.num_heads // div
                            cfg = SweepCfg(
                                embed_dim=ed,
                                hidden_dim=hidden,
                                layers=layers,
                                bpe_vocab_size=bpe,
                                dataset_frac=float(frac),
                                total_tokens=total_tokens,
                                num_heads=args.num_heads,
                                num_kv_heads=num_kv,
                            )
                            _validate_cfg(cfg, args.tokenizer)
                            matrix.append(cfg)
    return matrix


def _rank_key(row: dict[str, Any]) -> tuple[int, float, float]:
    ok = int(row.get("rc", 1) == 0 and isinstance(row.get("loss_final"), (int, float)))
    final = float(row.get("loss_final")) if isinstance(row.get("loss_final"), (int, float)) else float("inf")
    wall = float(row.get("wall_s")) if isinstance(row.get("wall_s"), (int, float)) else float("inf")
    return (0 if ok else 1, final, wall)


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(rows, key=_rank_key)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = int(i)

    raw_path = out_dir / "ablation_results_raw.json"
    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    ranked_path = out_dir / "ablation_results_ranked.json"
    ranked_path.write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    csv_cols = [
        "rank",
        "run_id",
        "rc",
        "status",
        "embed_dim",
        "hidden_dim",
        "layers",
        "bpe_vocab_size",
        "dataset_frac",
        "dataset_lines",
        "total_tokens",
        "num_heads",
        "num_kv_heads",
        "loss_first",
        "loss_final",
        "loss_min",
        "loss_drop",
        "loss_ratio",
        "loss_steps",
        "wall_s",
        "est_train_tok_s",
        "run_dir",
        "report_json",
        "log_path",
    ]
    csv_path = out_dir / "ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for row in ranked:
            w.writerow({k: row.get(k) for k in csv_cols})

    md_path = out_dir / "ablation_summary.md"
    lines = [
        "# v7 Ablation Matrix Summary",
        "",
        "| rank | run_id | final_loss | min_loss | first_loss | loss_drop | wall_s | tok/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked[:20]:
        final = row.get("loss_final")
        min_loss = row.get("loss_min")
        first = row.get("loss_first")
        drop = row.get("loss_drop")
        wall = row.get("wall_s")
        tok_s = row.get("est_train_tok_s")
        lines.append(
            "| {rank} | {run_id} | {final} | {min_loss} | {first} | {drop} | {wall} | {tok_s} |".format(
                rank=row.get("rank"),
                run_id=row.get("run_id"),
                final=f"{float(final):.6f}" if isinstance(final, (int, float)) else "NA",
                min_loss=f"{float(min_loss):.6f}" if isinstance(min_loss, (int, float)) else "NA",
                first=f"{float(first):.6f}" if isinstance(first, (int, float)) else "NA",
                drop=f"{float(drop):.6f}" if isinstance(drop, (int, float)) else "NA",
                wall=f"{float(wall):.2f}" if isinstance(wall, (int, float)) else "NA",
                tok_s=f"{float(tok_s):.2f}" if isinstance(tok_s, (int, float)) else "NA",
            )
        )
    lines += [
        "",
        f"Raw: `{raw_path}`",
        f"Ranked: `{ranked_path}`",
        f"CSV: `{csv_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run and rank a v7 train ablation matrix.")
    ap.add_argument("--run-root", required=True, help="Root dir for ablation runs/reports")
    ap.add_argument("--data", required=True, help="Base training dataset text path")
    ap.add_argument("--python", default=None, help="Python interpreter for child scripts")
    ap.add_argument("--resume", action="store_true", default=True, help="Reuse existing per-run reports")
    ap.add_argument("--no-resume", action="store_true", help="Force rerun even if report exists")
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands only")
    ap.add_argument("--max-runs", type=int, default=0, help="Execute at most N runs (0 = all)")

    ap.add_argument("--tokenizer", choices=["bpe", "ascii_bpe"], default="ascii_bpe")
    ap.set_defaults(require_svg_rows=True)
    ap.add_argument("--require-svg-rows", dest="require_svg_rows", action="store_true")
    ap.add_argument("--no-require-svg-rows", dest="require_svg_rows", action="store_false")

    ap.add_argument("--embed-dims", default="64,96")
    ap.add_argument("--layers-list", default="12,24")
    ap.add_argument("--bpe-vocab-sizes", default="320,512")
    ap.add_argument("--dataset-fracs", default="0.5,1.0")
    ap.add_argument("--total-tokens-list", default="131072")
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--kv-divisors", default="1,2", help="num_kv_heads = num_heads / divisor")
    ap.add_argument("--hidden-mult", type=float, default=2.0)

    ap.add_argument("--init", default="xavier_uniform")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--template", default="qwen3")
    args = ap.parse_args()

    repo_root = _repo_root()
    py = args.python or _python_exec(repo_root)
    pipeline = repo_root / "version" / "v7" / "scripts" / "train_data_pipeline_v7.py"
    if not pipeline.exists():
        raise SystemExit(f"ERROR: missing pipeline script: {pipeline}")

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"ERROR: dataset path not found: {data_path}")

    rows = _read_text_rows(data_path)
    if not rows:
        raise SystemExit(f"ERROR: dataset has no non-empty lines: {data_path}")

    run_root = Path(args.run_root).expanduser().resolve()
    runs_dir = run_root / "runs"
    datasets_dir = run_root / "datasets"
    reports_dir = run_root / "reports"
    logs_dir = run_root / "logs"
    for p in (run_root, runs_dir, datasets_dir, reports_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)

    matrix = _build_matrix(args)
    if not matrix:
        raise SystemExit("ERROR: empty ablation matrix")

    dataset_files: dict[float, tuple[Path, int]] = {}
    for frac in sorted({cfg.dataset_frac for cfg in matrix}):
        frac_pct = int(round(frac * 100))
        out = datasets_dir / f"train_df{frac_pct:03d}.txt"
        subset = _subset_rows(rows, frac)
        out.write_text("\n".join(subset) + "\n", encoding="utf-8")
        dataset_files[frac] = (out, len(subset))

    resume = bool(args.resume and not args.no_resume)
    results: list[dict[str, Any]] = []
    executed = 0
    total = len(matrix)

    for idx, cfg in enumerate(matrix, start=1):
        run_id = cfg.run_id()
        run_dir = runs_dir / run_id
        work_dir = run_root / "work" / run_id
        report_json = reports_dir / f"{run_id}.json"
        log_path = logs_dir / f"{run_id}.log"
        dataset_file, dataset_lines = dataset_files[cfg.dataset_frac]

        row: dict[str, Any] = {
            "run_id": run_id,
            "embed_dim": cfg.embed_dim,
            "hidden_dim": cfg.hidden_dim,
            "layers": cfg.layers,
            "bpe_vocab_size": cfg.bpe_vocab_size,
            "dataset_frac": cfg.dataset_frac,
            "dataset_lines": dataset_lines,
            "dataset_file": str(dataset_file),
            "total_tokens": cfg.total_tokens,
            "num_heads": cfg.num_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "run_dir": str(run_dir),
            "work_dir": str(work_dir),
            "report_json": str(report_json),
            "log_path": str(log_path),
        }

        if resume and report_json.exists():
            ck = _extract_ck_loss(report_json)
            first = _to_float(ck.get("first"))
            final = _to_float(ck.get("final"))
            row.update(
                {
                    "rc": 0,
                    "status": "cached",
                    "wall_s": 0.0,
                    "loss_steps": int(ck.get("steps", 0) or 0),
                    "loss_first": first,
                    "loss_final": final,
                    "loss_min": _to_float(ck.get("min")),
                    "loss_drop": (first - final) if (first is not None and final is not None) else None,
                    "loss_ratio": (final / first) if (first is not None and final is not None and first > 0.0) else None,
                    "est_train_tok_s": None,
                }
            )
            results.append(row)
            _write_outputs(run_root, results)
            print(f"[{idx}/{total}] cache {run_id}")
            continue

        if args.max_runs > 0 and executed >= args.max_runs:
            row.update({"rc": 999, "status": "skipped_max_runs"})
            results.append(row)
            _write_outputs(run_root, results)
            print(f"[{idx}/{total}] skip(max-runs) {run_id}")
            continue

        cmd = [
            py,
            str(pipeline),
            "--run",
            str(run_dir),
            "--init-if-missing",
            "--init",
            str(args.init),
            "--tokenizer",
            str(args.tokenizer),
            "--data",
            str(dataset_file),
            "--vocab-size",
            str(cfg.bpe_vocab_size),
            "--bpe-vocab-size",
            str(cfg.bpe_vocab_size),
            "--layers",
            str(cfg.layers),
            "--embed-dim",
            str(cfg.embed_dim),
            "--hidden-dim",
            str(cfg.hidden_dim),
            "--num-heads",
            str(cfg.num_heads),
            "--num-kv-heads",
            str(cfg.num_kv_heads),
            "--context-len",
            str(args.context_len),
            "--template",
            str(args.template),
            "--epochs",
            str(args.epochs),
            "--seq-len",
            str(args.seq_len),
            "--total-tokens",
            str(cfg.total_tokens),
            "--grad-accum",
            str(args.grad_accum),
            "--lr",
            str(args.lr),
            "--max-grad-norm",
            str(args.max_grad_norm),
            "--seed",
            str(args.seed),
            "--work-dir",
            str(work_dir),
            "--json-out",
            str(report_json),
            "--no-open-visualizer",
        ]
        if args.require_svg_rows:
            cmd.append("--require-svg-rows")
        if str(args.tokenizer) == "ascii_bpe":
            cmd.append("--require-ascii-data")

        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        print(f"[{idx}/{total}] run {run_id}")
        print(f"  cmd: {cmd_str}")

        if args.dry_run:
            row.update({"rc": 0, "status": "dry_run", "wall_s": 0.0})
            results.append(row)
            _write_outputs(run_root, results)
            continue

        executed += 1
        t0 = time.perf_counter()
        rc, _out = _run_cmd(cmd, cwd=repo_root, log_path=log_path)
        wall_s = time.perf_counter() - t0

        ck = _extract_ck_loss(report_json) if report_json.exists() else {"steps": 0}
        first = _to_float(ck.get("first"))
        final = _to_float(ck.get("final"))
        est_tok_s = (cfg.total_tokens * args.epochs) / wall_s if wall_s > 0 else None

        row.update(
            {
                "rc": int(rc),
                "status": "ok" if int(rc) == 0 else "failed",
                "wall_s": float(wall_s),
                "loss_steps": int(ck.get("steps", 0) or 0),
                "loss_first": first,
                "loss_final": final,
                "loss_min": _to_float(ck.get("min")),
                "loss_drop": (first - final) if (first is not None and final is not None) else None,
                "loss_ratio": (final / first) if (first is not None and final is not None and first > 0.0) else None,
                "est_train_tok_s": float(est_tok_s) if est_tok_s is not None else None,
            }
        )
        results.append(row)
        _write_outputs(run_root, results)

    _write_outputs(run_root, results)
    ranked = sorted(results, key=_rank_key)
    print("ablation matrix complete")
    print(f"  runs_total:    {len(results)}")
    print(f"  runs_executed: {executed}")
    print(f"  out_dir:       {run_root}")
    print(f"  ranked_json:   {run_root / 'ablation_results_ranked.json'}")
    print(f"  ranked_csv:    {run_root / 'ablation_results.csv'}")
    print(f"  summary_md:    {run_root / 'ablation_summary.md'}")
    if ranked:
        best = ranked[0]
        print(
            "  best:          "
            f"{best.get('run_id')} final_loss={best.get('loss_final')} status={best.get('status')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
