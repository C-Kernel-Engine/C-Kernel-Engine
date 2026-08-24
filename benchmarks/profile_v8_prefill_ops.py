#!/usr/bin/env python3
"""Profile v8 prefill operator costs for cached GGUF runtimes.

This uses the generated CK_PROFILE hooks, so runtimes must be compiled with
``ck_run_v8.py run ... --profile``. The script does that by default because a
non-profiled libmodel.so can otherwise look valid but emit no per-op CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from bench_v8_decoder_matrix import ANSI_RE, CACHE, CK_CLI, CK_RE, MODELS  # noqa: E402

CK_RUN_V8 = ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"
PREFILL_RE = re.compile(r"prefill\s+(\d+)\s+tok.*?([0-9.]+)\s+ms\s+([0-9.]+)\s+tok/s", re.S)
THREADPOOL_RE = re.compile(
    r"\[CK threadpool profile\]\s+dispatches=(\d+)\s+"
    r"total_ms=([0-9.]+)\s+main_work_ms=([0-9.]+)\s+"
    r"completion_wait_ms=([0-9.]+)"
)


def _run(cmd: list[str], *, env: dict[str, str], timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout


def _compile_profile_runtime(
    model_id: str,
    spec: dict[str, str],
    *,
    run_dir: Path,
    context_len: int,
    timeout: int,
    force: bool,
) -> dict[str, Any]:
    gguf = Path(spec["gguf"])
    if not gguf.is_absolute():
        gguf = CACHE / gguf
    cmd = [
        sys.executable,
        str(CK_RUN_V8),
        "run",
        str(gguf),
        "--run",
        str(run_dir),
        "--generate-only",
        "--profile",
        "--context-len",
        str(context_len),
        "--prompt",
        "Hello",
        "--max-tokens",
        "1",
        "--no-chat-template",
    ]
    if force:
        cmd.append("--force-convert")
        cmd.append("--force-compile")
    env = os.environ.copy()
    env.setdefault("CK_V8_COMPILER", "gcc")
    extra_cflags = env.get("CK_V8_EXTRA_CFLAGS", "-fno-omit-frame-pointer -g")
    if "-DCK_PROFILE" not in extra_cflags:
        extra_cflags = f"{extra_cflags} -DCK_PROFILE=1"
    env["CK_V8_EXTRA_CFLAGS"] = extra_cflags
    rc, out = _run(cmd, env=env, timeout=timeout)
    return {
        "model_id": model_id,
        "command": cmd,
        "returncode": rc,
        "stdout_tail": "\n".join(ANSI_RE.sub("", out).splitlines()[-80:]),
    }


def _profile_run(run_dir: Path, *, prompt: int, decode: int, threads: int, csv_path: Path, json_path: Path, timeout: int) -> dict[str, Any]:
    lib = run_dir / "libmodel.so"
    weights = run_dir / "weights.bump"
    manifest = run_dir / "weights_manifest.map"
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["CK_PROFILE"] = "1"
    env["CK_THREADPOOL_PROFILE"] = "1"
    env["CK_PROFILE_CSV"] = str(csv_path)
    env["CK_PROFILE_JSON"] = str(json_path)
    env["LD_LIBRARY_PATH"] = f"{ROOT / 'build'}:{run_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    csv_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)
    prompt_tokens = ",".join(["100"] * prompt)
    cmd = [
        str(CK_CLI),
        "--lib",
        str(lib),
        "--weights",
        str(weights),
        "--manifest",
        str(manifest),
        "--prompt-tokens",
        prompt_tokens,
        "--max-tokens",
        str(decode),
        "--ignore-eos",
        "--quiet-output",
        "--no-chat-template",
        "--no-stream",
        "--timing",
    ]
    rc, out = _run(cmd, env=env, timeout=timeout)
    clean = ANSI_RE.sub("", out)
    row: dict[str, Any] = {
        "command": [f"<{prompt} ids>" if value == prompt_tokens else value for value in cmd],
        "returncode": rc,
        "stdout_tail": "\n".join(clean.splitlines()[-80:]),
    }
    match = CK_RE.search(clean)
    if match:
        row.update(
            {
                "prompt_tokens": int(match.group(1)),
                "prompt_ms": float(match.group(2)),
                "prompt_tok_s": float(match.group(3)),
                "decode_tokens": int(match.group(4)),
                "decode_ms": float(match.group(5)),
                "decode_tok_s": float(match.group(6)),
            }
        )
    else:
        prefill_match = PREFILL_RE.search(clean)
        if prefill_match:
            row.update(
                {
                    "prompt_tokens": int(prefill_match.group(1)),
                    "prompt_ms": float(prefill_match.group(2)),
                    "prompt_tok_s": float(prefill_match.group(3)),
                }
            )
    threadpool_match = THREADPOOL_RE.search(clean)
    if threadpool_match:
        row["threadpool"] = {
            "dispatches": int(threadpool_match.group(1)),
            "dispatch_total_ms": float(threadpool_match.group(2)),
            "main_work_ms": float(threadpool_match.group(3)),
            "completion_wait_ms": float(threadpool_match.group(4)),
        }
    return row


def _load_profile_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["layer"] = int(row.get("layer") or -1)
            row["time_us"] = float(row.get("time_us") or 0.0)
            row["cpu_time_us"] = float(row.get("cpu_time_us") or 0.0)
            row["token_id"] = int(row.get("token_id") or 0)
            rows.append(row)
        return rows


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...], *, limit: int) -> list[dict[str, Any]]:
    agg: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        bucket = agg.setdefault(
            key,
            {k: row.get(k, "") for k in keys}
            | {"time_us": 0.0, "cpu_time_us": 0.0, "count": 0},
        )
        bucket["time_us"] += float(row["time_us"])
        bucket["cpu_time_us"] += float(row["cpu_time_us"])
        bucket["count"] += 1
    ranked = sorted(agg.values(), key=lambda x: float(x["time_us"]), reverse=True)
    return ranked[:limit]


def _summarize(
    rows: list[dict[str, Any]], *, limit: int, threads: int
) -> dict[str, Any]:
    prefill = [r for r in rows if r.get("mode") == "prefill"]
    total_us = sum(float(r["time_us"]) for r in prefill)
    total_cpu_us = sum(float(r["cpu_time_us"]) for r in prefill)
    by_op = _group(prefill, ("op",), limit=limit)
    by_kernel = _group(prefill, ("kernel", "op"), limit=limit)
    by_layer = _group(prefill, ("layer",), limit=limit)
    by_layer_op = _group(prefill, ("layer", "op"), limit=limit)
    for table in (by_op, by_kernel, by_layer, by_layer_op):
        for row in table:
            row["time_ms"] = row["time_us"] / 1000.0
            row["cpu_time_ms"] = row["cpu_time_us"] / 1000.0
            row["core_equivalents"] = (
                row["cpu_time_us"] / row["time_us"] if row["time_us"] > 0 else 0.0
            )
            row["worker_utilization_pct"] = (
                100.0 * row["core_equivalents"] / threads if threads > 0 else 0.0
            )
            row["pct"] = (100.0 * row["time_us"] / total_us) if total_us > 0 else 0.0
    return {
        "prefill_total_us": total_us,
        "prefill_total_ms": total_us / 1000.0,
        "prefill_cpu_time_us": total_cpu_us,
        "prefill_cpu_time_ms": total_cpu_us / 1000.0,
        "prefill_core_equivalents": total_cpu_us / total_us if total_us > 0 else 0.0,
        "prefill_worker_utilization_pct": (
            100.0 * total_cpu_us / (total_us * threads)
            if total_us > 0 and threads > 0
            else 0.0
        ),
        "prefill_entries": len(prefill),
        "by_op": by_op,
        "by_kernel_op": by_kernel,
        "by_layer": by_layer,
        "by_layer_op": by_layer_op,
    }


def _fmt_table(title: str, rows: list[dict[str, Any]], cols: tuple[str, ...]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  no rows")
        return
    header = "  " + "  ".join(
        f"{c:>16}"
        for c in (*cols, "wall ms", "CPU ms", "cores", "util %", "%", "count")
    )
    print(header)
    for row in rows:
        values = [str(row.get(c, "")) for c in cols]
        print(
            "  "
            + "  ".join(f"{v:>16}" for v in values)
            + f"  {row['time_ms']:16.3f}"
            + f"  {row['cpu_time_ms']:16.3f}"
            + f"  {row['core_equivalents']:16.2f}"
            + f"  {row['worker_utilization_pct']:16.1f}"
            + f"  {row['pct']:16.1f}"
            + f"  {row['count']:16d}"
        )


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen35-0.8b-q4_k_m,qwen2-0.5b-q4_k_m", help="Comma-separated model ids, or 'all'")
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--decode", type=int, default=1)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--reuse-runtime", action="store_true", help="Do not force-recompile the profiled runtime")
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "build" / "v8_profile_runtimes")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "build", help="Directory for raw CSV/JSON profiler artifacts")
    parser.add_argument("--json-out", type=Path, default=ROOT / "build" / "v8_prefill_ops_profile.json")
    args = parser.parse_args()

    selected = list(MODELS) if args.models == "all" else [x.strip() for x in args.models.split(",") if x.strip()]
    build_dir = args.artifact_root.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for model_id in selected:
        if model_id not in MODELS:
            raise KeyError(f"unknown model id: {model_id}")
        spec = MODELS[model_id]
        run_dir = args.runtime_root / _safe_id(model_id)
        print(f"\n== {spec['label']} {spec['quant']} ==", flush=True)

        missing = [str(CK_CLI)]
        gguf = Path(spec["gguf"])
        if not gguf.is_absolute():
            gguf = CACHE / gguf
        missing.append(str(gguf))
        missing = [p for p in missing if not Path(p).exists()]
        if missing:
            row = {"id": model_id, "label": spec["label"], "quant": spec["quant"], "status": "skip", "missing": missing}
            print(f"skip: missing {missing}")
            results.append(row)
            continue

        compile_row = _compile_profile_runtime(
            model_id,
            spec,
            run_dir=run_dir,
            context_len=max(args.context_len, args.prompt + args.decode + 8),
            timeout=args.timeout,
            force=not args.reuse_runtime,
        )
        if compile_row["returncode"] != 0:
            compile_row["status"] = "fail"
            print(compile_row["stdout_tail"])
            results.append({"id": model_id, "label": spec["label"], "quant": spec["quant"], "compile": compile_row, "status": "fail"})
            continue

        csv_path = build_dir / f"v8_prefill_profile_{_safe_id(model_id)}_p{args.prompt}_t{args.threads}.csv"
        profile_json_path = build_dir / f"v8_prefill_profile_{_safe_id(model_id)}_p{args.prompt}_t{args.threads}.raw.json"
        run_row = _profile_run(
            run_dir,
            prompt=args.prompt,
            decode=args.decode,
            threads=args.threads,
            csv_path=csv_path,
            json_path=profile_json_path,
            timeout=args.timeout,
        )
        if run_row["returncode"] != 0 or not csv_path.exists():
            print(run_row["stdout_tail"])
            results.append(
                {
                    "id": model_id,
                    "label": spec["label"],
                    "quant": spec["quant"],
                    "compile": compile_row,
                    "run": run_row,
                    "status": "fail",
                    "csv": str(csv_path),
                }
            )
            continue

        rows = _load_profile_rows(csv_path)
        summary = _summarize(rows, limit=args.limit, threads=args.threads)
        prompt_ms = float(run_row.get("prompt_ms") or 0.0)
        profile_ms = float(summary["prefill_total_ms"])
        uncovered_ms = max(0.0, prompt_ms - profile_ms)
        summary["prompt_ms"] = prompt_ms
        summary["prompt_tok_s"] = run_row.get("prompt_tok_s")
        summary["profile_coverage_pct"] = (100.0 * profile_ms / prompt_ms) if prompt_ms > 0 else 0.0
        summary["unprofiled_or_runtime_ms"] = uncovered_ms

        print(
            f"prompt={prompt_ms:.2f} ms ({run_row.get('prompt_tok_s', 0):.1f} tok/s), "
            f"profiled prefill ops={profile_ms:.2f} ms, "
            f"occupancy={summary['prefill_core_equivalents']:.2f}/{args.threads} cores "
            f"({summary['prefill_worker_utilization_pct']:.1f}%), "
            f"coverage={summary['profile_coverage_pct']:.1f}%"
        )
        _fmt_table("Top prefill ops", summary["by_op"], ("op",))
        _fmt_table("Top prefill kernel/op sites", summary["by_kernel_op"], ("kernel", "op"))
        _fmt_table("Prefill layers", summary["by_layer"], ("layer",))
        _fmt_table("Top prefill layer/op sites", summary["by_layer_op"], ("layer", "op"))

        results.append(
            {
                "id": model_id,
                "label": spec["label"],
                "quant": spec["quant"],
                "run_dir": str(run_dir),
                "csv": str(csv_path),
                "raw_profile_json": str(profile_json_path),
                "compile": compile_row,
                "run": run_row,
                "summary": summary,
                "status": "pass",
            }
        )

    doc = {
        "args": {
            "models": selected,
            "prompt": args.prompt,
            "decode": args.decode,
            "threads": args.threads,
            "context_len": args.context_len,
            "limit": args.limit,
            "reuse_runtime": args.reuse_runtime,
            "runtime_root": str(args.runtime_root),
            "artifact_root": str(build_dir),
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")
    return 0 if all(r.get("status") in {"pass", "skip"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
