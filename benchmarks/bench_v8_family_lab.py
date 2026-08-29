#!/usr/bin/env python3
"""Resume-safe CKE/llama.cpp family, context, quality, and profiler sweep."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import hashlib
import html
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "benchmarks" / "compare_ck_llama_v8.py"
PROFILE_OPS = ROOT / "benchmarks" / "profile_v8_prefill_ops.py"
CK_RUN = ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"
DEFAULT_MODELS = ROOT / "benchmarks" / "fixtures" / "v8_lab_models.json"
DEFAULT_PROMPTS = ROOT / "benchmarks" / "fixtures" / "v8_lab_prompts.json"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
MAX_SEQ_LEN_RE = re.compile(r"^#define\s+MAX_SEQ_LEN\s+(\d+)\s*$", re.M)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], *, timeout: int, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": process.returncode,
        "started_at": started.isoformat(),
        "output_tail": ANSI_RE.sub("", process.stdout)[-8000:],
    }


def host_provenance() -> dict[str, Any]:
    cpu = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(errors="replace"), re.M)
        cpu = match.group(1).strip() if match else ""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "hostname": platform.node(),
        "cpu": cpu,
        "machine": platform.machine(),
        "kernel": platform.release(),
        "logical_cpus": os.cpu_count(),
        "commit": commit,
        "tools": {
            name: shutil.which(name)
            for name in ("perf", "vtune", "advisor", "time")
        },
    }


def selected_model_rows(manifest: Path, keys: list[str]) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload["models"]
    if not keys or keys == ["all"]:
        return rows
    by_key = {row["key"]: row for row in rows}
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(f"unknown model keys: {', '.join(missing)}")
    return [by_key[key] for key in keys]


def generated_context_capacity(run_dir: Path) -> int | None:
    source = run_dir / "model_v8.c"
    if not source.is_file():
        return None
    match = MAX_SEQ_LEN_RE.search(source.read_text(encoding="utf-8", errors="replace"))
    return int(match.group(1)) if match else None


def prepare_runtimes(
    models: list[dict[str, Any]],
    *,
    context: int,
    timeout: int,
    force: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model in models:
        run_dir = Path(str(model["ck_run_dir"])).expanduser()
        command = [
            sys.executable,
            str(CK_RUN),
            "run",
            str(Path(str(model["gguf"])).expanduser()),
            "--run",
            str(run_dir),
            "--generate-only",
            "--context-len",
            str(context),
            "--prompt",
            "Hello",
            "--max-tokens",
            "1",
            "--chat-template",
            "auto",
        ]
        runtime_capacity = generated_context_capacity(run_dir)
        capacity_rebuild = runtime_capacity is not None and runtime_capacity < context
        if force:
            command.extend(["--force-convert", "--force-compile"])
        elif capacity_rebuild:
            command.append("--force-compile")
        result = run(command, timeout=timeout)
        result["model_key"] = model["key"]
        result["run_dir"] = str(run_dir)
        result["previous_context_capacity"] = runtime_capacity
        result["capacity_rebuild"] = capacity_rebuild
        results.append(result)
        if result["returncode"] != 0:
            raise RuntimeError(
                f"runtime preparation failed for {model['key']}: {result['output_tail']}"
            )
        updated_capacity = generated_context_capacity(run_dir)
        if updated_capacity is not None:
            result["context_capacity"] = updated_capacity
            if updated_capacity < context:
                raise RuntimeError(
                    f"runtime preparation left {model['key']} at context capacity "
                    f"{updated_capacity}, below required {context}"
                )
    return results


def quality_features(prompt_key: str, output: str) -> dict[str, Any]:
    stripped = output.strip()
    result: dict[str, Any] = {
        "characters": len(stripped),
        "words": len(re.findall(r"\b\w+\b", stripped)),
        "nonempty": bool(stripped),
    }
    if prompt_key == "structured_json":
        try:
            parsed = json.loads(stripped)
            result["valid_json"] = isinstance(parsed, dict)
        except json.JSONDecodeError:
            result["valid_json"] = False
    if prompt_key == "svg_infographic":
        lowered = stripped.lower()
        result.update(
            {
                "has_svg": "<svg" in lowered and "</svg>" in lowered,
                "has_viewbox": "viewbox=" in lowered,
                "has_accessible_title": "<title" in lowered and "<desc" in lowered,
            }
        )
    if prompt_key == "c_python_sql":
        lowered = stripped.lower()
        result.update(
            {
                "mentions_c": "#include" in stripped or " c " in f" {lowered} ",
                "mentions_python": "python" in lowered or "import " in lowered,
                "mentions_sql": "sql" in lowered or "create table" in lowered,
            }
        )
    return result


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def summarize_reports(run_reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    perf_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for report in run_reports:
        context = int(report["config"]["context"])
        prompt_tokens = int(report["config"]["prompt_tokens"])
        for row in report.get("perf", []):
            entry = {"context": context, "prompt_tokens": prompt_tokens, **row}
            perf_rows.append(entry)
        for row in report.get("prompt_runs", []):
            cke_text = str(row["cke"].get("generated") or "")
            llama_text = str((row.get("llama") or {}).get("generated") or "")
            similarity = None
            if cke_text and llama_text:
                similarity = difflib.SequenceMatcher(
                    None, normalize_text(cke_text), normalize_text(llama_text)
                ).ratio()
            prompt_rows.append(
                {
                    "context": context,
                    "model_key": row["model_key"],
                    "model": row["model"],
                    "prompt_key": row["prompt_key"],
                    "prompt": row["prompt"],
                    "cke": row["cke"],
                    "llama": row.get("llama"),
                    "cke_quality": quality_features(row["prompt_key"], cke_text),
                    "llama_quality": quality_features(row["prompt_key"], llama_text),
                    "text_similarity": similarity,
                }
            )
    return perf_rows, prompt_rows


def slowest_cases(perf_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    worst: dict[str, dict[str, Any]] = {}
    for row in perf_rows:
        ratio = (row.get("ratios") or {}).get("prompt")
        if not isinstance(ratio, (int, float)):
            continue
        previous = worst.get(row["model_key"])
        if previous is None or ratio < previous["ratios"]["prompt"]:
            worst[row["model_key"]] = row
    return sorted(worst.values(), key=lambda row: row["ratios"]["prompt"])


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def profiler_plan(
    cases: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    output: Path,
    threads: int,
    ck_cli: Path,
    llama_root: Path,
) -> list[dict[str, Any]]:
    by_key = {row["key"]: row for row in model_rows}
    plan: list[dict[str, Any]] = []
    for case in cases:
        model = by_key[case["model_key"]]
        prompt = int(case["prompt_tokens"])
        context = max(int(case["context"]), prompt + 8)
        profile_dir = output / "profiles" / model["key"] / f"p{prompt}"
        ops_json = profile_dir / "cke_ops.json"
        profile_model_id = model.get("profile_model_id")
        ops = None if not profile_model_id else [
            sys.executable,
            str(PROFILE_OPS),
            "--models",
            str(profile_model_id),
            "--prompt",
            str(prompt),
            "--decode",
            "4",
            "--threads",
            str(threads),
            "--context-len",
            str(context),
            "--reuse-runtime",
            "--runtime-root",
            str(profile_dir / "runtime"),
            "--artifact-root",
            str(profile_dir),
            "--json-out",
            str(ops_json),
        ]
        run_dir = Path(str(model["ck_run_dir"])).expanduser()
        if profile_model_id:
            profile_runtime_id = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", str(profile_model_id)
            )
            run_dir = profile_dir / "runtime" / profile_runtime_id
        token_ids = ",".join(["100"] * prompt)
        cke_command = [
            str(ck_cli),
            "--lib", str(run_dir / "libmodel.so"),
            "--weights", str(run_dir / "weights.bump"),
            "--manifest", str(run_dir / "weights_manifest.map"),
            "--prompt-tokens", token_ids,
            "--max-tokens", "2",
            "--context", str(context),
            "--temperature", "0",
            "--ignore-eos", "--quiet-output", "--no-chat-template", "--no-stream", "--timing",
        ]
        llama_command = [
            str(llama_root / "build/bin/llama-bench"),
            "-m", str(Path(str(model["gguf"])).expanduser()),
            "-p", str(prompt), "-n", "1", "-t", str(threads), "-ngl", "0", "-r", "1", "-o", "json",
        ]
        perf_events = "cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock"
        system_profiles = [
            {
                "name": "cke_perf_stat",
                "tool": "perf",
                "command": ["perf", "stat", "-x,", "-e", perf_events, "-o", str(profile_dir / "cke-perf-stat.csv"), "--", "env", "CK_THREADPOOL_PROFILE=1", *cke_command],
            },
            {
                "name": "llama_perf_stat",
                "tool": "perf",
                "command": ["perf", "stat", "-x,", "-e", perf_events, "-o", str(profile_dir / "llama-perf-stat.csv"), "--", *llama_command],
            },
            {
                "name": "cke_flamegraph_capture",
                "tool": "perf",
                "command": ["perf", "record", "-F", "199", "-g", "--call-graph", "dwarf", "-o", str(profile_dir / "cke-perf.data"), "--", "env", "CK_THREADPOOL_PROFILE=1", *cke_command],
            },
            {
                "name": "cke_vtune_hotspots",
                "tool": "vtune",
                "command": ["vtune", "-collect", "hotspots", "-result-dir", str(profile_dir / "cke-vtune-hotspots"), "-quiet", "--", "env", "CK_THREADPOOL_PROFILE=1", *cke_command],
            },
            {
                "name": "llama_vtune_hotspots",
                "tool": "vtune",
                "command": ["vtune", "-collect", "hotspots", "-result-dir", str(profile_dir / "llama-vtune-hotspots"), "-quiet", "--", *llama_command],
            },
            {
                "name": "cke_vtune_uarch",
                "tool": "vtune",
                "command": ["vtune", "-collect", "uarch-exploration", "-result-dir", str(profile_dir / "cke-vtune-uarch"), "-quiet", "--", "env", "CK_THREADPOOL_PROFILE=1", *cke_command],
            },
            {
                "name": "cke_vtune_memory",
                "tool": "vtune",
                "command": ["vtune", "-collect", "memory-access", "-result-dir", str(profile_dir / "cke-vtune-memory"), "-quiet", "--", "env", "CK_THREADPOOL_PROFILE=1", *cke_command],
            },
            {
                "name": "cke_advisor_roofline",
                "tool": "advisor",
                "command": ["advisor", "--collect=roofline", "--project-dir", str(profile_dir / "cke-advisor-roofline"), "--", "env", "CK_THREADPOOL_PROFILE=1", *cke_command],
            },
        ]
        plan.append(
            {
                "model_key": model["key"],
                "prompt_tokens": prompt,
                "cke_llama_ratio": case["ratios"]["prompt"],
                "directory": str(profile_dir),
                "operation_profile": ops,
                "cke_command": cke_command,
                "llama_command": llama_command,
                "system_profiles": system_profiles,
                "notes": [
                    "Run perf, VTune hotspots/uarch/memory-access, Advisor roofline, and a flamegraph on the exact native CLI command recorded by the operation profile.",
                    "Profile llama.cpp with the same prompt length, threads, affinity, warmup, and repetitions before attributing the performance delta.",
                    "Raw profiler databases remain local; publish derived tables and redacted links only.",
                ],
            }
        )
    return plan


def discover_model_artifacts(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = (
        "ir_report.html", "xray_summary.json", "config.json",
        "lowered_prefill_call.json", "lowered_decode_call.json", "weights_manifest.map",
    )
    rows: list[dict[str, Any]] = []
    for model in models:
        run_dir = Path(str(model["ck_run_dir"])).expanduser()
        files = {name: str(run_dir / name) for name in names if (run_dir / name).is_file()}
        rows.append({"model_key": model["key"], "run_dir": str(run_dir), "files": files})
    return rows


def collect_operation_profiles(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in plan:
        path = Path(item["directory"]) / "cke_ops.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        if not results:
            continue
        result = results[0]
        run = result.get("run") or {}
        summary = result.get("summary") or {}
        phases = summary.get("phases") or {}
        prefill_phase = phases.get("prefill") or {}
        decode_phase = phases.get("decode") or {}
        prompt_ms = float(run.get("prompt_ms") or 0)
        decode_ms = float(run.get("decode_ms") or 0)
        profiled_ms = float(summary.get("prefill_total_ms") or 0)
        decode_profiled_ms = float(decode_phase.get("total_ms") or 0)
        rows.append({
            "model_key": item["model_key"],
            "prompt_tokens": item["prompt_tokens"],
            "prompt_ms": prompt_ms,
            "profiled_ms": profiled_ms,
            "coverage_pct": 100.0 * profiled_ms / prompt_ms if prompt_ms else None,
            "core_equivalents": prefill_phase.get(
                "core_equivalents", summary.get("prefill_core_equivalents")
            ),
            "worker_utilization_pct": prefill_phase.get(
                "worker_utilization_pct",
                summary.get("prefill_worker_utilization_pct"),
            ),
            "top_operations": (summary.get("by_op") or [])[:8],
            "selected_kernels": (summary.get("by_kernel_op") or [])[:12],
            "decode_ms": decode_ms,
            "decode_profiled_ms": decode_profiled_ms,
            "decode_coverage_pct": (
                100.0 * decode_profiled_ms / decode_ms if decode_ms else None
            ),
            "decode_core_equivalents": decode_phase.get("core_equivalents"),
            "decode_worker_utilization_pct": decode_phase.get(
                "worker_utilization_pct"
            ),
            "decode_top_operations": (decode_phase.get("by_op") or [])[:8],
            "decode_selected_kernels": (
                decode_phase.get("by_kernel_op") or []
            )[:12],
            "threadpool": run.get("threadpool"),
            "source": str(path),
        })
    return rows


def parse_perf_stat_csv(path: Path) -> dict[str, float]:
    """Load stable numeric perf-stat counters from a `-x,` capture."""
    counters: dict[str, float] = {}
    if not path.is_file():
        return counters
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        for row in csv.reader(stream):
            if len(row) < 3 or not row[0] or row[0].startswith("<"):
                continue
            try:
                name = row[2].strip()
                hybrid = re.fullmatch(r"cpu_(?:core|atom)/([^/]+)/", name)
                if hybrid:
                    name = hybrid.group(1)
                counters[name] = counters.get(name, 0.0) + float(row[0])
            except ValueError:
                continue
    return counters


def collect_system_profile_counters(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in plan:
        directory = Path(item["directory"])
        cke = parse_perf_stat_csv(directory / "cke-perf-stat.csv")
        llama = parse_perf_stat_csv(directory / "llama-perf-stat.csv")
        if cke or llama:
            rows.append({
                "model_key": item["model_key"],
                "prompt_tokens": item["prompt_tokens"],
                "cke": cke,
                "llama": llama,
            })
    return rows


def render_html(report: dict[str, Any], path: Path) -> None:
    esc = lambda value: html.escape(str(value))
    perf_rows = report["performance"]
    prompts = report["prompt_comparisons"]
    slow = report["slowest_cases"]
    cards = []
    for row in slow:
        cards.append(
            f"<div class='metric'><strong>{esc(row['model'])}</strong>"
            f"<span>{row['ratios']['prompt']:.2f}x</span>"
            f"<small>{row['prompt_tokens']} tokens</small></div>"
        )
    perf_body = []
    for row in perf_rows:
        llama = row.get("llama") or {}
        cke = row.get("cke") or {}
        ratio = (row.get("ratios") or {}).get("prompt")
        consumption = row.get("consumption") or {}
        consumed = (
            consumption.get("cke_count_verified") is True
            and consumption.get("llama_count_verified") is True
        )
        token_hash = str(consumption.get("requested_token_sha256") or "")
        perf_body.append(
            "<tr>"
            f"<td>{esc(row['model'])}</td><td>{esc(row['quant'])}</td>"
            f"<td>{row['prompt_tokens']}</td>"
            f"<td>{float(cke.get('prompt_ms', 0)):.2f}</td>"
            f"<td>{float(llama.get('prompt_ms', 0)):.2f}</td>"
            f"<td class='{('good' if ratio is not None and ratio >= 1 else 'bad')}'>{esc(f'{ratio:.2f}x' if ratio is not None else '-')}</td>"
            f"<td>{float(cke.get('decode_tok_s', 0)):.1f}</td>"
            f"<td>{float(llama.get('decode_tok_s', 0)):.1f}</td>"
            f"<td class='{('good' if consumed else 'bad')}'>{'yes' if consumed else 'no'}</td>"
            f"<td><code>{esc(token_hash[:12] if token_hash else '-')}</code></td>"
            "</tr>"
        )
    prompt_body = []
    for row in prompts:
        cke_text = str(row["cke"].get("generated") or "")
        llama_text = str((row.get("llama") or {}).get("generated") or "")
        similarity = row.get("text_similarity")
        prompt_body.append(
            "<details><summary>"
            f"{esc(row['model'])} / {esc(row['prompt_key'])}"
            f"<span>{esc(f'{similarity:.2f}' if similarity is not None else '-')} similarity</span>"
            "</summary><div class='outputs'>"
            f"<section><h4>CKE</h4><pre>{esc(cke_text)}</pre><code>{esc(json.dumps(row['cke_quality'], sort_keys=True))}</code></section>"
            f"<section><h4>llama.cpp</h4><pre>{esc(llama_text)}</pre><code>{esc(json.dumps(row['llama_quality'], sort_keys=True))}</code></section>"
            "</div></details>"
        )
    profile_body = []
    for item in report["profiler_plan"]:
        directory = Path(item["directory"])
        evidence_names = (
            "cke-perf-stat.csv", "llama-perf-stat.csv",
            "cke-vtune-hotspots.csv", "llama-vtune-hotspots.csv",
        )
        evidence = " ".join(
            f"<a href='{esc(directory / name)}'>{esc(name)}</a>"
            for name in evidence_names if (directory / name).is_file()
        ) or "not collected"
        profile_body.append(
            "<tr>"
            f"<td>{esc(item['model_key'])}</td><td>{item['prompt_tokens']}</td>"
            f"<td>{item['cke_llama_ratio']:.2f}x</td>"
            f"<td><code>{esc(shell_join(item['operation_profile']) if item['operation_profile'] else 'not registered in profile_v8_prefill_ops.py')}</code></td>"
            f"<td>{evidence}</td>"
            "</tr>"
        )
    operation_body = []
    for item in report.get("operation_profiles", []):
        threadpool = item.get("threadpool") or {}
        completion_wait = threadpool.get("completion_wait_ms")
        phase_rows = [
            {
                "phase": "prefill", "wall_ms": item["prompt_ms"],
                "profiled_ms": item["profiled_ms"],
                "coverage_pct": item.get("coverage_pct"),
                "core_equivalents": item.get("core_equivalents"),
                "worker_utilization_pct": item.get("worker_utilization_pct"),
                "operations": item["top_operations"],
                "kernels": item.get("selected_kernels", []),
            },
            {
                "phase": "decode", "wall_ms": item.get("decode_ms", 0.0),
                "profiled_ms": item.get("decode_profiled_ms", 0.0),
                "coverage_pct": item.get("decode_coverage_pct"),
                "core_equivalents": item.get("decode_core_equivalents"),
                "worker_utilization_pct": item.get(
                    "decode_worker_utilization_pct"
                ),
                "operations": item.get("decode_top_operations", []),
                "kernels": item.get("decode_selected_kernels", []),
            },
        ]
        for phase_row in phase_rows:
            operations = "<br>".join(
                f"<code>{esc(op['op'])}</code> {float(op['time_ms']):.1f} ms "
                f"({float(op['pct']):.1f}%)"
                for op in phase_row["operations"]
            ) or "-"
            kernels = "<br>".join(
                f"<code>{esc(kernel['kernel'])}</code>"
                for kernel in phase_row["kernels"][:6]
            ) or "-"
            coverage = phase_row["coverage_pct"]
            cores = phase_row["core_equivalents"]
            utilization = phase_row["worker_utilization_pct"]
            operation_body.append(
                "<tr>"
                f"<td>{esc(item['model_key'])}</td><td>{phase_row['phase']}</td>"
                f"<td>{item['prompt_tokens']}</td>"
                f"<td>{phase_row['wall_ms']:.1f}</td>"
                f"<td>{phase_row['profiled_ms']:.1f}</td>"
                f"<td>{esc(f'{coverage:.1f}%' if coverage is not None else '-')}</td>"
                f"<td>{esc(f'{float(cores):.2f}' if cores is not None else '-')}</td>"
                f"<td>{esc(f'{float(utilization):.1f}%' if utilization is not None else '-')}</td>"
                f"<td>{esc(f'{float(completion_wait):.2f}' if completion_wait is not None else '-')}</td>"
                f"<td>{operations}</td><td>{kernels}</td></tr>"
            )
    system_body = []
    for item in report.get("system_profile_counters", []):
        cke = item.get("cke") or {}
        llama = item.get("llama") or {}
        names = ("cycles", "instructions", "cache-misses", "LLC-load-misses")
        for name in names:
            if name not in cke and name not in llama:
                continue
            system_body.append(
                "<tr>"
                f"<td>{esc(item['model_key'])}</td><td>{item['prompt_tokens']}</td>"
                f"<td><code>{esc(name)}</code></td>"
                f"<td>{esc(f'{cke[name]:.0f}' if name in cke else '-')}</td>"
                f"<td>{esc(f'{llama[name]:.0f}' if name in llama else '-')}</td>"
                "</tr>"
            )
    artifact_body = []
    for item in report.get("model_artifacts", []):
        links = " ".join(
            f"<a href='{esc(path)}'>{esc(name)}</a>" for name, path in item["files"].items()
        ) or "not generated"
        artifact_body.append(f"<tr><td>{esc(item['model_key'])}</td><td>{links}</td></tr>")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CKE v8 Family Lab Sweep</title><style>
:root{{--ink:#17202a;--muted:#5f6b76;--line:#d7dde3;--paper:#fff;--band:#f4f7f8;--accent:#006b5e;--bad:#a43b35;--good:#137a43}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:14px/1.45 system-ui,sans-serif;background:var(--paper)}}
header{{padding:28px max(24px,5vw);background:#17202a;color:white}}h1{{margin:0 0 6px;font-size:28px;letter-spacing:0}}header p{{margin:0;color:#cfd8dc}}
main{{padding:24px max(24px,5vw) 60px}}h2{{margin:30px 0 10px;font-size:20px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}}
.metric{{border-left:4px solid var(--accent);background:var(--band);padding:12px}}.metric strong,.metric span,.metric small{{display:block}}.metric span{{font-size:23px;font-weight:700}}.metric small{{color:var(--muted)}}
.table{{overflow:auto;border:1px solid var(--line)}}table{{width:100%;border-collapse:collapse;min-width:800px}}th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line)}}th{{background:var(--band)}}.good{{color:var(--good);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}
details{{border:1px solid var(--line);margin:8px 0}}summary{{cursor:pointer;padding:10px;font-weight:650}}summary span{{float:right;color:var(--muted)}}.outputs{{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--line)}}.outputs section{{min-width:0;padding:12px}}.outputs section+section{{border-left:1px solid var(--line)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;background:var(--band);padding:10px}}code{{overflow-wrap:anywhere}}.note{{padding:12px;border-left:4px solid #cc8b00;background:#fff8e5}}@media(max-width:760px){{.outputs{{grid-template-columns:1fr}}.outputs section+section{{border-left:0;border-top:1px solid var(--line)}}}}
</style></head><body><header><h1>CKE v8 Family Lab Sweep</h1><p>{esc(report['generated_at'])} · {esc(report['host']['cpu'])} · {report['config']['threads']} threads</p></header><main>
<div class="note">Human-output similarity is diagnostic, not a correctness score. Performance rows use matched fixed-token workloads; profiler attribution requires matched CKE and llama.cpp captures.</div>
<h2>Slowest prefill case per model</h2><div class="metrics">{''.join(cards)}</div>
<h2>Performance matrix</h2><div class="table"><table><thead><tr><th>Model</th><th>Quant</th><th>Prompt</th><th>CKE ms</th><th>llama ms</th><th>CKE/llama</th><th>CKE decode tok/s</th><th>llama decode tok/s</th><th>Consumed</th><th>Token hash</th></tr></thead><tbody>{''.join(perf_body)}</tbody></table></div>
<h2>Prompt output comparison</h2>{''.join(prompt_body)}
<h2>Operation attribution</h2><div class="table"><table><thead><tr><th>Model</th><th>Phase</th><th>Prompt</th><th>Wall ms</th><th>Profiled ms</th><th>Coverage</th><th>Core equivalents</th><th>Worker utilization</th><th>Completion wait ms</th><th>Top operations</th><th>Selected providers</th></tr></thead><tbody>{''.join(operation_body)}</tbody></table></div>
<h2>Matched hardware counters</h2><div class="table"><table><thead><tr><th>Model</th><th>Prompt</th><th>Counter</th><th>CKE</th><th>llama.cpp</th></tr></thead><tbody>{''.join(system_body)}</tbody></table></div>
<h2>Profiler queue</h2><div class="table"><table><thead><tr><th>Model</th><th>Prompt</th><th>Ratio</th><th>Operation attribution command</th><th>Local evidence</th></tr></thead><tbody>{''.join(profile_body)}</tbody></table></div>
<h2>IR and X-Ray artifacts</h2><div class="table"><table><thead><tr><th>Model</th><th>Local artifacts</th></tr></thead><tbody>{''.join(artifact_body)}</tbody></table></div>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--ck-cli", type=Path, default=ROOT / "build" / "ck-cli-v8")
    parser.add_argument("--llama-root", type=Path, default=ROOT / "llama.cpp")
    parser.add_argument("--contexts", default="128,512,1024,2048")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prompt-max-tokens", type=int, default=256)
    parser.add_argument("--prompt-context", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=Path("/data/cke/results/v8-family-lab/latest"))
    parser.add_argument("--run-profiles", action="store_true", help="Run CKE operation attribution for each model's worst case")
    parser.add_argument("--run-system-profiles", action="store_true", help="Run perf, VTune, and Advisor on each selected worst case")
    parser.add_argument(
        "--system-profile",
        action="append",
        default=[],
        help="Run only the named system profiler entry; repeat for multiple entries",
    )
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=3,
        help="Run requested profilers for the N largest measured model gaps; 0 records the plan without running it",
    )
    parser.add_argument("--prepare-runtimes", action="store_true")
    parser.add_argument("--skip-prompts", action="store_true", help="Run only the matched fixed-token performance matrix")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    contexts = sorted({int(value) for value in args.contexts.split(",") if value.strip()})
    if any(value <= 0 for value in contexts):
        raise ValueError("contexts must be positive")
    if args.profile_limit < 0:
        raise ValueError("profile-limit must be non-negative")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    model_rows = selected_model_rows(args.model_manifest, args.model or ["all"])
    model_keys = [row["key"] for row in model_rows]
    reports: list[dict[str, Any]] = []
    preparation: list[dict[str, Any]] = []
    if args.prepare_runtimes:
        preparation = prepare_runtimes(
            model_rows,
            context=max(max(contexts) + args.decode_tokens + 8, args.prompt_context),
            timeout=args.timeout,
            force=args.force_prepare,
        )
        atomic_json(args.output / "runtime_preparation.json", preparation)

    for prompt_tokens in contexts:
        run_dir = args.output / "runs" / f"p{prompt_tokens}"
        result_json = run_dir / "ck_llama_v8_compare.json"
        if not (args.resume and result_json.exists()):
            command = [
                sys.executable,
                str(COMPARE),
                "--lane",
                "perf",
                "--model-manifest",
                str(args.model_manifest),
                "--prompts-file",
                str(args.prompts_file),
                "--ck-cli",
                str(args.ck_cli),
                "--llama-root",
                str(args.llama_root),
                "--threads",
                str(args.threads),
                "--prompt-tokens",
                str(prompt_tokens),
                "--decode-tokens",
                str(args.decode_tokens),
                "--prompt-max-tokens",
                str(args.prompt_max_tokens),
                "--prompt-engine",
                "both",
                "--context",
                str(prompt_tokens + args.decode_tokens + 8),
                "--repetitions",
                str(args.repetitions),
                "--timeout",
                str(args.timeout),
                "--prompt-timeout",
                str(args.timeout),
                "--out-dir",
                str(run_dir),
            ]
            for key in model_keys:
                command.extend(["--model", key])
            result = run(command, timeout=args.timeout * max(1, len(model_rows)))
            atomic_json(run_dir / "runner.json", result)
            if result["returncode"] != 0:
                print(result["output_tail"], file=sys.stderr)
                return result["returncode"]
        reports.append(json.loads(result_json.read_text(encoding="utf-8")))

    if not args.skip_prompts:
        prompt_dir = args.output / "runs" / "prompts"
        prompt_json = prompt_dir / "ck_llama_v8_compare.json"
        if not (args.resume and prompt_json.exists()):
            command = [
                sys.executable,
                str(COMPARE),
                "--lane",
                "prompts",
                "--model-manifest",
                str(args.model_manifest),
                "--prompts-file",
                str(args.prompts_file),
                "--ck-cli",
                str(args.ck_cli),
                "--llama-root",
                str(args.llama_root),
                "--threads",
                str(args.threads),
                "--prompt-max-tokens",
                str(args.prompt_max_tokens),
                "--prompt-engine",
                "both",
                "--context",
                str(args.prompt_context),
                "--timeout",
                str(args.timeout),
                "--prompt-timeout",
                str(args.timeout),
                "--out-dir",
                str(prompt_dir),
            ]
            for key in model_keys:
                command.extend(["--model", key])
            result = run(command, timeout=args.timeout * max(1, len(model_rows)))
            atomic_json(prompt_dir / "runner.json", result)
            if result["returncode"] != 0:
                print(result["output_tail"], file=sys.stderr)
                return result["returncode"]
        reports.append(json.loads(prompt_json.read_text(encoding="utf-8")))

    performance, prompts = summarize_reports(reports)
    slow = slowest_cases(performance)
    plan = profiler_plan(slow, model_rows, args.output, args.threads, args.ck_cli, args.llama_root)
    selected_plan = plan[:args.profile_limit]
    profile_results: list[dict[str, Any]] = []
    if args.run_profiles:
        for item in selected_plan:
            if not item["operation_profile"]:
                profile_results.append({
                    "model_key": item["model_key"],
                    "prompt_tokens": item["prompt_tokens"],
                    "returncode": None,
                    "status": "skip",
                    "reason": "model is not registered in profile_v8_prefill_ops.py",
                })
                continue
            Path(item["directory"]).mkdir(parents=True, exist_ok=True)
            result = run(item["operation_profile"], timeout=args.timeout)
            result.update({"model_key": item["model_key"], "prompt_tokens": item["prompt_tokens"]})
            profile_results.append(result)
    if args.run_system_profiles:
        available = {name: shutil.which(name) for name in ("perf", "vtune", "advisor")}
        for item in selected_plan:
            Path(item["directory"]).mkdir(parents=True, exist_ok=True)
            for profile in item["system_profiles"]:
                if args.system_profile and profile["name"] not in args.system_profile:
                    continue
                if not available.get(profile["tool"]):
                    profile_results.append({
                        "model_key": item["model_key"], "name": profile["name"],
                        "status": "skip", "reason": f"{profile['tool']} is unavailable",
                    })
                    continue
                result = run(profile["command"], timeout=args.timeout)
                result.update({"model_key": item["model_key"], "name": profile["name"]})
                profile_results.append(result)

    report = {
        "schema": "cke.v8.family-lab-sweep",
        "version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host_provenance(),
        "config": {
            "contexts": contexts,
            "threads": args.threads,
            "decode_tokens": args.decode_tokens,
            "prompt_max_tokens": args.prompt_max_tokens,
            "prompt_context": args.prompt_context,
            "repetitions": args.repetitions,
            "profile_limit": args.profile_limit,
            "prompts_skipped": args.skip_prompts,
            "model_manifest": str(args.model_manifest),
            "prompts_file": str(args.prompts_file),
            "ck_cli": str(args.ck_cli),
            "llama_root": str(args.llama_root),
        },
        "performance": performance,
        "prompt_comparisons": prompts,
        "slowest_cases": slow,
        "profiler_plan": plan,
        "profile_results": profile_results,
        "operation_profiles": collect_operation_profiles(plan),
        "system_profile_counters": collect_system_profile_counters(plan),
        "runtime_preparation": preparation,
        "model_catalog": model_rows,
        "model_artifacts": discover_model_artifacts(model_rows),
    }
    atomic_json(args.output / "family_lab_report.json", report)
    render_html(report, args.output / "index.html")
    print(f"wrote {args.output / 'family_lab_report.json'}")
    print(f"wrote {args.output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
