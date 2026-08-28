#!/usr/bin/env python3
"""Run a resumable, capacity-aware long-context model certification sweep."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import platform
import re
import resource
import signal
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CATALOG = SCRIPT_DIR.parent / "regression" / "long_context_models.json"
DEFAULT_PROMPTS = SCRIPT_DIR.parent / "test_assets" / "long_context_quality_prompts.json"
DEFAULT_OUTPUT = ROOT / "build" / "v8-long-context-certification"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CK_TIMING_RE = re.compile(
    r"prefill\s+(?P<prompt_tokens>\d+)\s+tok.*?"
    r"(?P<prompt_ms>[0-9.]+)\s+ms\s+(?P<prompt_tok_s>[0-9.]+)\s+tok/s.*?"
    r"decode\s+(?P<decode_tokens>\d+)\s+tok\s+"
    r"(?P<decode_ms>[0-9.]+)\s+ms\s+(?P<decode_tok_s>[0-9.]+)\s+tok/s",
    re.S,
)
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
TIME_USER_RE = re.compile(r"User time \(seconds\):\s*([0-9.]+)")
TIME_SYSTEM_RE = re.compile(r"System time \(seconds\):\s*([0-9.]+)")
TIME_CPU_RE = re.compile(r"Percent of CPU this job got:\s*([0-9.]+)%")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_schema(path: Path, expected: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != expected or int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"invalid {expected} document: {path}")
    return payload


def parse_contexts(value: str) -> list[int]:
    contexts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not contexts or any(item <= 0 for item in contexts):
        raise ValueError("contexts must be positive integers")
    return contexts


def validate_quality_prompts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("quality prompt set must contain prompts")
    seen: set[str] = set()
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("quality prompts must be objects")
        prompt_id = str(prompt.get("id", ""))
        if not prompt_id or prompt_id in seen:
            raise ValueError("quality prompt IDs must be non-empty and unique")
        if str(prompt.get("kind", "")) not in {"c_kernel.v1", "kernel_svg.v1"}:
            raise ValueError(f"unsupported quality prompt kind for {prompt_id!r}")
        if int(prompt.get("max_tokens", 0)) <= 0 or not str(prompt.get("text", "")):
            raise ValueError(f"quality prompt {prompt_id!r} is incomplete")
        dependency = str(prompt.get("depends_on", ""))
        if dependency and dependency not in seen:
            raise ValueError(
                f"quality prompt {prompt_id!r} depends on unknown or later {dependency!r}"
            )
        if dependency and "{{dependency_output}}" not in str(prompt["text"]):
            raise ValueError(f"quality prompt {prompt_id!r} does not embed its dependency")
        seen.add(prompt_id)
    return prompts


def resolve_model(row: dict[str, Any], allow_download: bool) -> tuple[str, str]:
    env_name = str(row.get("model_env", ""))
    configured = os.environ.get(env_name, "").strip() if env_name else ""
    if configured:
        return configured, f"environment:{env_name}"
    default = str(row.get("default_model", "")).strip()
    if default and allow_download:
        return default, "catalog_default"
    return "", f"unset:{env_name or 'model'}"


def resolve_gguf(row: dict[str, Any], model: str, runtime_dir: Path) -> Path | None:
    env_name = str(row.get("gguf_env", ""))
    configured = os.environ.get(env_name, "").strip() if env_name else ""
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    model_path = Path(model).expanduser()
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return model_path
    candidates = sorted(runtime_dir.glob("*.gguf"))
    return candidates[0] if candidates else None


def token_sha256(token_id: int, count: int) -> str:
    value = int(token_id).to_bytes(4, "little", signed=True)
    digest = hashlib.sha256()
    for _ in range(count):
        digest.update(value)
    return digest.hexdigest()


def parse_timing(output: str) -> dict[str, float]:
    clean = ANSI_RE.sub("", output).replace("\r", "")
    match = CK_TIMING_RE.search(clean)
    if not match:
        raise ValueError("native CKE timing line is missing")
    return {
        "prompt_tokens": int(match.group("prompt_tokens")),
        "prompt_ms": float(match.group("prompt_ms")),
        "prompt_tok_s": float(match.group("prompt_tok_s")),
        "decode_tokens": int(match.group("decode_tokens")),
        "decode_ms": float(match.group("decode_ms")),
        "decode_tok_s": float(match.group("decode_tok_s")),
    }


def parse_peak_rss(stderr: str) -> int | None:
    match = RSS_RE.search(stderr)
    return int(match.group(1)) if match else None


def parse_resource_usage(stderr: str, elapsed_seconds: float) -> dict[str, float] | None:
    user_match = TIME_USER_RE.search(stderr)
    system_match = TIME_SYSTEM_RE.search(stderr)
    cpu_match = TIME_CPU_RE.search(stderr)
    if not user_match or not system_match:
        return None
    user_seconds = float(user_match.group(1))
    system_seconds = float(system_match.group(1))
    process_seconds = user_seconds + system_seconds
    average_cpu_cores = process_seconds / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    return {
        "user_seconds": user_seconds,
        "system_seconds": system_seconds,
        "process_seconds": process_seconds,
        "wall_seconds": elapsed_seconds,
        "average_cpu_cores": average_cpu_cores,
        "reported_cpu_percent": float(cpu_match.group(1)) if cpu_match else average_cpu_cores * 100.0,
    }


def resource_usage_delta(
    before: resource.struct_rusage,
    after: resource.struct_rusage,
    elapsed_seconds: float,
) -> dict[str, float]:
    user_seconds = max(0.0, float(after.ru_utime - before.ru_utime))
    system_seconds = max(0.0, float(after.ru_stime - before.ru_stime))
    process_seconds = user_seconds + system_seconds
    average_cpu_cores = process_seconds / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    return {
        "user_seconds": user_seconds,
        "system_seconds": system_seconds,
        "process_seconds": process_seconds,
        "wall_seconds": elapsed_seconds,
        "average_cpu_cores": average_cpu_cores,
        "reported_cpu_percent": average_cpu_cores * 100.0,
    }


def resolve_time_binary(env: dict[str, str]) -> str | None:
    configured = env.get("CK_TIME_BIN", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which("time", path=env.get("PATH"))


def discover_physical_cpu_representatives(
    *,
    allowed_cpus: set[int] | None = None,
    sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> list[int]:
    if allowed_cpus is None:
        try:
            allowed_cpus = set(os.sched_getaffinity(0))
        except AttributeError:
            allowed_cpus = set(range(os.cpu_count() or 1))
    representatives: dict[tuple[int, int], int] = {}
    unresolved: list[int] = []
    for cpu in sorted(allowed_cpus):
        topology = sysfs_root / f"cpu{cpu}" / "topology"
        try:
            package = int((topology / "physical_package_id").read_text(encoding="ascii"))
            core = int((topology / "core_id").read_text(encoding="ascii"))
        except (OSError, ValueError):
            unresolved.append(cpu)
            continue
        representatives.setdefault((package, core), cpu)
    if representatives:
        return sorted(representatives.values())
    return unresolved or sorted(allowed_cpus)


def resolve_cpu_plan(
    requested_threads: int,
    cpu_policy: str,
    *,
    allowed_cpus: set[int] | None = None,
    sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> dict[str, Any]:
    if allowed_cpus is None:
        try:
            allowed_cpus = set(os.sched_getaffinity(0))
        except AttributeError:
            allowed_cpus = set(range(os.cpu_count() or 1))
    allowed = sorted(allowed_cpus)
    if not allowed:
        allowed = [0]
    physical = discover_physical_cpu_representatives(
        allowed_cpus=set(allowed), sysfs_root=sysfs_root
    )
    effective_threads = requested_threads if requested_threads > 0 else len(physical)
    effective_threads = max(1, min(effective_threads, len(allowed)))
    affinity: list[int] | None = None
    if cpu_policy == "physical":
        affinity = physical[:effective_threads]
        if len(affinity) < effective_threads:
            affinity.extend(cpu for cpu in allowed if cpu not in affinity)
            affinity = affinity[:effective_threads]
    elif cpu_policy != "inherit":
        raise ValueError(f"unsupported CPU policy: {cpu_policy}")
    return {
        "policy": cpu_policy,
        "requested_threads": requested_threads,
        "effective_threads": effective_threads,
        "allowed_cpus": allowed,
        "physical_representatives": physical,
        "affinity": affinity,
    }


def _proc_tree(root_pid: int) -> set[int]:
    pending = [root_pid]
    found: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        task_root = Path(f"/proc/{pid}/task")
        try:
            tasks = list(task_root.iterdir())
        except OSError:
            continue
        for task in tasks:
            try:
                children = (task / "children").read_text(encoding="ascii").split()
            except OSError:
                continue
            pending.extend(int(child) for child in children)
    return found


def _proc_cpu_ticks_and_rss(root_pid: int) -> tuple[int, int]:
    ticks = 0
    rss_kib = 0
    for pid in _proc_tree(root_pid):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
            fields = stat[stat.rfind(")") + 2:].split()
            ticks += int(fields[11]) + int(fields[12])
            status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
        except (OSError, ValueError, IndexError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss_kib += int(line.split()[1])
                break
    return ticks, rss_kib


def _system_cpu_totals(
    cpu_ids: set[int] | None = None,
    proc_stat: Path = Path("/proc/stat"),
) -> dict[int, tuple[int, int]]:
    """Return per-CPU (busy, total) ticks using the same counters as top."""
    totals: dict[int, tuple[int, int]] = {}
    try:
        lines = proc_stat.read_text(encoding="ascii").splitlines()
    except OSError:
        return totals
    for line in lines:
        fields = line.split()
        if not fields or not re.fullmatch(r"cpu\d+", fields[0]):
            continue
        cpu = int(fields[0][3:])
        if cpu_ids is not None and cpu not in cpu_ids:
            continue
        try:
            counters = [int(value) for value in fields[1:]]
        except ValueError:
            continue
        if len(counters) < 4:
            continue
        total = sum(counters)
        idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
        totals[cpu] = (total - idle, total)
    return totals


def summarize_utilization_samples(
    samples: list[dict[str, Any]], requested_threads: int
) -> dict[str, Any]:
    active = sorted(
        float(sample["active_cores"])
        for sample in samples
        if sample.get("active_cores") is not None
    )
    if not active:
        return {"sample_count": 0, "requested_threads": requested_threads}

    def percentile(fraction: float) -> float:
        position = fraction * (len(active) - 1)
        lower = int(position)
        upper = min(lower + 1, len(active) - 1)
        weight = position - lower
        return active[lower] * (1.0 - weight) + active[upper] * weight

    low_threshold = max(1.0, requested_threads * 0.25)
    longest_low = 0.0
    low_started: float | None = None
    previous_time = 0.0
    for sample in samples:
        current_time = float(sample["elapsed_seconds"])
        value = sample.get("active_cores")
        if value is not None and float(value) < low_threshold:
            if low_started is None:
                low_started = previous_time
        elif low_started is not None:
            longest_low = max(longest_low, current_time - low_started)
            low_started = None
        previous_time = current_time
    if low_started is not None:
        longest_low = max(longest_low, previous_time - low_started)

    per_cpu_values: dict[int, list[float]] = {}
    selected_active: list[float] = []
    all_cores_80_samples = 0
    per_cpu_sample_count = 0
    for sample in samples:
        per_cpu = sample.get("per_cpu_busy_pct")
        if not isinstance(per_cpu, dict) or not per_cpu:
            continue
        values = [float(value) for value in per_cpu.values()]
        for cpu, value in per_cpu.items():
            per_cpu_values.setdefault(int(cpu), []).append(float(value))
        selected_active.append(sum(values) / 100.0)
        all_cores_80_samples += int(all(value >= 80.0 for value in values))
        per_cpu_sample_count += 1
    per_cpu_means = {
        str(cpu): statistics.fmean(values)
        for cpu, values in sorted(per_cpu_values.items()) if values
    }
    result = {
        "sample_count": len(active),
        "requested_threads": requested_threads,
        "mean_active_cores": statistics.fmean(active),
        "p10_active_cores": percentile(0.10),
        "p50_active_cores": percentile(0.50),
        "p90_active_cores": percentile(0.90),
        "minimum_active_cores": active[0],
        "maximum_active_cores": active[-1],
        "low_utilization_threshold_cores": low_threshold,
        "longest_low_utilization_seconds": longest_low,
        "peak_sampled_rss_kib": max(
            (int(sample.get("rss_kib", 0)) for sample in samples), default=0
        ),
    }
    if selected_active and per_cpu_means:
        selected_sorted = sorted(selected_active)

        def selected_percentile(fraction: float) -> float:
            position = fraction * (len(selected_sorted) - 1)
            lower = int(position)
            upper = min(lower + 1, len(selected_sorted) - 1)
            weight = position - lower
            return selected_sorted[lower] * (1.0 - weight) + selected_sorted[upper] * weight

        result.update({
            "per_cpu_sample_count": per_cpu_sample_count,
            "sampled_cpu_ids": [int(cpu) for cpu in per_cpu_means],
            "per_cpu_mean_utilization_pct": per_cpu_means,
            "mean_selected_cpu_active_cores": statistics.fmean(selected_active),
            "p10_selected_cpu_active_cores": selected_percentile(0.10),
            "minimum_per_cpu_mean_utilization_pct": min(per_cpu_means.values()),
            "maximum_per_cpu_mean_utilization_pct": max(per_cpu_means.values()),
            "all_sampled_cpus_ge_80pct_fraction": (
                all_cores_80_samples / per_cpu_sample_count
            ),
        })
    return result


def _sample_process_tree(
    root_pid: int,
    stop: threading.Event,
    samples: list[dict[str, Any]],
    monitored_cpus: set[int] | None = None,
    interval_seconds: float = 0.25,
) -> None:
    ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
    origin = time.monotonic()
    previous_time = origin
    previous_ticks: int | None = None
    previous_cpu_totals: dict[int, tuple[int, int]] | None = None
    while True:
        now = time.monotonic()
        ticks, rss_kib = _proc_cpu_ticks_and_rss(root_pid)
        cpu_totals = _system_cpu_totals(monitored_cpus)
        active_cores = None
        per_cpu_busy_pct: dict[str, float] | None = None
        elapsed = now - previous_time
        if previous_ticks is not None and elapsed > 0.0 and ticks >= previous_ticks:
            active_cores = (ticks - previous_ticks) / ticks_per_second / elapsed
        if previous_cpu_totals is not None:
            per_cpu_busy_pct = {}
            for cpu, (busy, total) in cpu_totals.items():
                previous = previous_cpu_totals.get(cpu)
                if previous is None:
                    continue
                busy_delta = busy - previous[0]
                total_delta = total - previous[1]
                if total_delta > 0 and busy_delta >= 0:
                    per_cpu_busy_pct[str(cpu)] = min(
                        100.0, 100.0 * busy_delta / total_delta
                    )
        sample: dict[str, Any] = {
            "elapsed_seconds": now - origin,
            "active_cores": active_cores,
            "rss_kib": float(rss_kib),
        }
        if per_cpu_busy_pct:
            sample["per_cpu_busy_pct"] = per_cpu_busy_pct
        samples.append(sample)
        previous_time = now
        previous_ticks = ticks
        previous_cpu_totals = cpu_totals
        if stop.wait(interval_seconds):
            break


def run_command(
    command: list[str],
    *,
    timeout: int,
    env: dict[str, str],
    output_dir: Path,
    name: str,
    timed: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_binary = resolve_time_binary(env) if timed else None
    executed = [time_binary, "-v", *command] if time_binary else command
    affinity_text = env.get("CK_BENCH_CPU_AFFINITY", "").strip()
    affinity = (
        {int(value) for value in affinity_text.split(",") if value.strip()}
        if affinity_text else None
    )
    preexec_fn = None
    if affinity and hasattr(os, "sched_setaffinity"):
        preexec_fn = lambda: os.sched_setaffinity(0, affinity)
    started = dt.datetime.now(dt.timezone.utc)
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    utilization_samples: list[dict[str, Any]] = []
    sampler_stop = threading.Event()
    sampler: threading.Thread | None = None
    try:
        process = subprocess.Popen(
            executed,
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=preexec_fn,
        )
        if timed and sys.platform.startswith("linux"):
            monitored_cpus = affinity or set(os.sched_getaffinity(0))
            sampler = threading.Thread(
                target=_sample_process_tree,
                args=(process.pid, sampler_stop, utilization_samples, monitored_cpus),
                name=f"cke-utilization-{name}",
                daemon=True,
            )
            sampler.start()
        stdout, stderr = process.communicate(timeout=timeout)
        returncode = process.returncode
        error = None
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        error = f"timeout after {timeout}s"
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (NameError, ProcessLookupError):
            pass
        final_stdout, final_stderr = process.communicate()
        stdout = final_stdout or stdout
        stderr = final_stderr or stderr
    finally:
        sampler_stop.set()
        if sampler is not None:
            sampler.join(timeout=2.0)
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    utilization_path = output_dir / f"{name}.utilization.json"
    requested_threads = max(1, int(env.get("CK_NUM_THREADS", "1") or "1"))
    utilization = summarize_utilization_samples(
        utilization_samples, requested_threads
    )
    if utilization_samples:
        atomic_json(
            utilization_path,
            {
                "schema": "cke.process_utilization_timeline",
                "schema_version": 1,
                "sample_interval_seconds": 0.25,
                "summary": utilization,
                "samples": utilization_samples,
            },
        )
    elapsed_seconds = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    measured_usage = parse_resource_usage(stderr, elapsed_seconds) if time_binary else None
    usage_source = "gnu_time" if measured_usage is not None else "unavailable"
    if timed and measured_usage is None:
        measured_usage = resource_usage_delta(usage_before, usage_after, elapsed_seconds)
        usage_source = "getrusage_children"
    return {
        "command": command,
        "command_shell": shlex.join(command),
        "returncode": returncode,
        "error": error,
        "started_at": started.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_kib": parse_peak_rss(stderr) if time_binary else None,
        "peak_rss_source": "gnu_time" if time_binary else "unavailable",
        "resource_usage": measured_usage if timed else None,
        "resource_usage_source": usage_source,
        "utilization": utilization if timed else None,
        "utilization_timeline_path": (
            str(utilization_path) if timed and utilization_samples else None
        ),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout": stdout,
        "stderr": stderr,
    }


def provider_summary(runtime_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase, filename in (("prefill", "lowered_prefill.json"), ("decode", "lowered_decode.json")):
        path = runtime_dir / filename
        if not path.is_file():
            result[phase] = {"status": "missing", "path": str(path)}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        operations = [row for row in payload.get("operations", []) if isinstance(row, dict)]
        counts = Counter(str(row.get("kernel", "unknown")) for row in operations)
        selections: Counter[tuple[str, str, str]] = Counter()
        for row in operations:
            contract = row.get("resolved_contract")
            if not isinstance(contract, dict):
                continue
            selections[(
                str(row.get("kernel", "unknown")),
                str(contract.get("contract_id", "")),
                json.dumps(contract.get("selector"), sort_keys=True),
            )] += 1
        result[phase] = {
            "status": "available",
            "operation_count": len(operations),
            "kernel_counts": dict(sorted(counts.items())),
            "contract_selections": [
                {"kernel": kernel, "contract_id": contract_id,
                 "selector": json.loads(selector), "count": count}
                for (kernel, contract_id, selector), count in sorted(selections.items())
            ],
            "path": str(path),
        }
    return result


def host_fingerprint() -> dict[str, Any]:
    cpu_model = ""
    flags: list[str] = []
    try:
        payload = json.loads(subprocess.run(
            ["lscpu", "-J"], text=True, capture_output=True, check=True
        ).stdout)
        fields = {str(row.get("field", "")).rstrip(":"): str(row.get("data", ""))
                  for row in payload.get("lscpu", [])}
        cpu_model = fields.get("Model name", "")
        flags = fields.get("Flags", "").split()
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    relevant = sorted(flag for flag in flags if flag.startswith(("avx", "amx", "fma")))
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "isa_flags": relevant,
    }


def median_timing(samples: list[dict[str, Any]]) -> dict[str, float] | None:
    timings = [sample["timing"] for sample in samples if isinstance(sample.get("timing"), dict)]
    if not timings:
        return None
    return {
        key: statistics.median(float(timing[key]) for timing in timings)
        for key in timings[0]
    }


def parse_llama_bench(stdout: str) -> dict[str, float]:
    rows = json.loads(stdout)
    prompt = next((row for row in rows if int(row.get("n_prompt", 0)) > 0), None)
    decode = next((row for row in rows if int(row.get("n_gen", 0)) > 0), None)
    if not prompt or not decode:
        raise ValueError("llama-bench did not return prompt and decode rows")
    return {
        "prompt_tokens": int(prompt["n_prompt"]),
        "prompt_ms": float(prompt["avg_ns"]) / 1_000_000.0,
        "prompt_tok_s": float(prompt["avg_ts"]),
        "decode_tokens": int(decode["n_gen"]),
        "decode_ms": float(decode["avg_ns"]) / 1_000_000.0,
        "decode_tok_s": float(decode["avg_ts"]),
    }


def extract_generated(stdout: str) -> str:
    text = ANSI_RE.sub("", stdout).replace("\r", "")
    marker = "Type /help for commands, Ctrl+C to stop generation"
    if marker in text:
        text = text.split(marker, 1)[1]
    response_markers = list(re.finditer(r"(?m)^(?:Response|Assistant):[ \t]*", text))
    if response_markers:
        text = text[response_markers[-1].end():]
    text = re.split(r"\nprefill\s+\d+\s+tok\b", text, maxsplit=1)[0]
    text = re.split(r"(?m)^stop:\s", text, maxsplit=1)[0]
    text = re.sub(r"(?im)\n(?:goodbye!?|model unloaded\.?)\s*$", "", text)
    return text.strip()


def extract_c_source(text: str) -> str:
    blocks = re.findall(r"```(?:c|C)\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    start = text.find("#include")
    return text[start:].strip() + "\n" if start >= 0 else ""


def extract_svg_markup(text: str) -> str:
    lowered = text.lower()
    start = lowered.find("<svg")
    end = lowered.rfind("</svg>")
    if start < 0 or end < start:
        return ""
    return text[start:end + len("</svg>")].strip() + "\n"


def materialize_quality_prompt(
    prompt: dict[str, Any],
    dependencies: dict[str, str],
    prefix: str = "",
) -> str:
    text = str(prompt["text"])
    dependency_id = str(prompt.get("depends_on", ""))
    if not dependency_id:
        if not prefix:
            return text
        return f"{prefix.rstrip()}\n\nTask:\n{text}"
    dependency = dependencies.get(dependency_id, "")
    if not dependency:
        raise ValueError(f"quality prompt {prompt['id']!r} requires missing {dependency_id!r}")
    return text.replace("{{dependency_output}}", dependency)


def engineering_output_coherence(text: str) -> dict[str, Any]:
    """Reject corrupted/gibberish output without judging artifact perfection."""
    printable = sum(character.isprintable() or character in "\n\r\t" for character in text)
    ratio = printable / max(len(text), 1)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_+-]*", text)
    replacement_chars = text.count("\ufffd") + len(
        re.findall(r"\\u[fF]{3}[dD]", text)
    )
    four_gram_counts = Counter(
        tuple(words[index:index + 4]) for index in range(max(0, len(words) - 3))
    )
    repeated_four_gram = max(four_gram_counts.values(), default=0)
    reasons = []
    if len(text) < 128:
        reasons.append(f"too_short:{len(text)}<128")
    if len(words) < 8:
        reasons.append(f"too_few_words:{len(words)}<8")
    if ratio < 0.96:
        reasons.append(f"printable_ratio:{ratio:.3f}<0.960")
    if replacement_chars:
        reasons.append(f"replacement_chars:{replacement_chars}>0")
    if repeated_four_gram > 4:
        reasons.append(f"repeated_4gram:{repeated_four_gram}>4")
    return {
        "pass": not reasons,
        "characters": len(text),
        "words": len(words),
        "printable_ratio": ratio,
        "replacement_chars": replacement_chars,
        "repeated_4gram": repeated_four_gram,
        "reasons": reasons,
    }


def code_quality(
    text: str,
    source_path: Path | None = None,
    prompt: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    coherence = engineering_output_coherence(text)
    ratio = float(coherence["printable_ratio"])
    source = extract_c_source(text)
    contract = prompt or {}
    required = [str(value) for value in contract.get("required_fragments", [])]
    forbidden = [str(value) for value in contract.get("forbidden_fragments", [])]
    result: dict[str, Any] = {
        "kind": str(contract.get("kind", "c_kernel.v1")),
        "pass": False,
        "coherence_pass": coherence["pass"],
        "coherence": coherence,
        "characters": len(text),
        "printable_ratio": ratio,
        "source_characters": len(source),
        "required_fragments_present": all(value in source for value in required),
        "missing_fragments": [value for value in required if value not in source],
        "forbidden_fragments_absent": all(value not in source for value in forbidden),
        "forbidden_fragments_found": [value for value in forbidden if value in source],
        "strict_compile": None,
        "compile_scope": "syntax-only; generated code is never executed automatically",
    }
    if source_path is not None and source:
        source_path.write_text(source, encoding="utf-8")
    if source_path is not None and source and env is not None and output_dir is not None:
        compiler = env.get("CC", "cc")
        compiler_path = shutil.which(compiler, path=env.get("PATH"))
        if compiler_path:
            compile_result = run_command(
                [compiler_path, "-std=c11", "-mavx2", "-ffp-contract=off",
                 "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(source_path)],
                timeout=60,
                env=env,
                output_dir=output_dir,
                name=f"{source_path.stem}-compile",
            )
            result["strict_compile"] = compile_result["returncode"] == 0
            result["compiler"] = compiler_path
            result["compile_stdout_path"] = compile_result["stdout_path"]
            result["compile_stderr_path"] = compile_result["stderr_path"]
        else:
            result["compile_unavailable"] = compiler
    checks = (
        len(text) >= 256,
        ratio >= 0.96,
        bool(source),
        result["required_fragments_present"],
        result["forbidden_fragments_absent"],
        result["strict_compile"] is True if source_path is not None else True,
    )
    result["pass"] = all(checks)
    return result


def svg_quality(text: str, prompt: dict[str, Any] | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(SCRIPT_DIR))
    from certify_text_prompt_parity_v8 import evaluate_quality_contract
    contract = prompt or {}
    result = evaluate_quality_contract(text, {
        "kind": "standalone_svg.v1",
        "min_graphic_elements": int(contract.get("min_graphic_elements", 12 if prompt else 8)),
        "required_labels": contract.get("required_labels", []),
        "require_arrow_marker": bool(prompt),
    })
    coherence = engineering_output_coherence(text)
    result["coherence_pass"] = coherence["pass"]
    result["coherence"] = coherence
    return result


def engineering_quality_status(
    returncode: int,
    quality: dict[str, Any],
    prefill_consumption_verified: bool,
    prompt_not_truncated: bool,
) -> str:
    if (
        returncode != 0
        or not bool(quality.get("coherence_pass"))
        or not prefill_consumption_verified
        or not prompt_not_truncated
    ):
        return "FAIL"
    if bool(quality.get("pass")):
        return "PASS"
    return "WARN"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v8 Long-Context Certification",
        "",
        f"Generated: `{report['updated_at']}`",
        "",
        "## Performance",
        "",
        "| Model | Context | Status | CKE prefill | llama.cpp prefill | Relative | CKE decode | Active cores avg/p10 | Longest low interval | Peak RAM | First-logit repeatable | Numerical parity |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report.get("performance", []):
        cke = row.get("cke") or {}
        llama = row.get("llama_cpp") or {}
        cke_timing = cke.get("timing") or {}
        llama_timing = llama.get("timing") or {}
        ratio = row.get("relative_prefill")
        peak = cke.get("peak_rss_kib")
        active_cores = cke.get("average_active_cores")
        p10_active_cores = cke.get("p10_active_cores")
        longest_low = cke.get("longest_low_utilization_seconds")
        lines.append(
            f"| {row['model']} | {row['context_tokens']} | {row['status']} | "
            f"{cke_timing.get('prompt_tok_s', '-')} | {llama_timing.get('prompt_tok_s', '-')} | "
            f"{f'{ratio:.2f}x' if isinstance(ratio, float) else '-'} | "
            f"{cke_timing.get('decode_tok_s', '-')} | "
            f"{f'{active_cores:.1f}' if isinstance(active_cores, float) else '-'} / "
            f"{f'{p10_active_cores:.1f}' if isinstance(p10_active_cores, float) else '-'} | "
            f"{f'{longest_low:.2f} s' if isinstance(longest_low, float) else '-'} | "
            f"{f'{peak / 1048576:.2f} GiB' if isinstance(peak, int) else '-'} | "
            f"{cke.get('first_logits_repeatable', '-')} | "
            f"{(row.get('numerical_parity') or {}).get('status', '-')} |"
        )
    lines += [
        "", "## Engineering Quality", "",
        "The C gate is syntax-only and never executes generated code. Human review and a numerical oracle remain required before any kernel can enter CKE.",
        "",
        "| Model | Task | Status | Total context | Output budget | Prompt tokens | Generated tokens | Stop | Compile/XML | Clean artifact | Raw response |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in report.get("quality", []):
        quality = row.get("quality") or {}
        validation = quality.get("strict_compile")
        if validation is None:
            validation = quality.get("xml_parseable", "-")
        lines.append(
            f"| {row['model']} | {row['prompt_id']} | {row['status']} | "
            f"{row.get('total_context_tokens', '-')} | {row.get('output_budget_tokens', '-')} | "
            f"{(row.get('timing') or {}).get('prompt_tokens', '-')} | "
            f"{(row.get('timing') or {}).get('decode_tokens', '-')} | "
            f"{row.get('native_stop_reason', '-')} | {validation} | "
            f"`{row.get('artifact_path', '-')}` | `{row.get('raw_output_path', '-')}` |"
        )
    lines += [
        "", "## Engineering Quality Performance", "",
        "| Model | Task | Prefill tok/s | Decode tok/s | Process cores | p10 sampled cores | Top-like selected cores | Weakest core mean | >=80% all-core samples | Peak RAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("quality", []):
        timing = row.get("timing") or {}
        usage = row.get("resource_usage") or {}
        utilization = row.get("utilization") or {}
        peak = row.get("peak_rss_kib")
        lines.append(
            f"| {row['model']} | {row['prompt_id']} | "
            f"{timing.get('prompt_tok_s', '-')} | {timing.get('decode_tok_s', '-')} | "
            f"{usage.get('average_cpu_cores', '-')} | "
            f"{utilization.get('p10_active_cores', '-')} | "
            f"{utilization.get('mean_selected_cpu_active_cores', '-')} | "
            f"{utilization.get('minimum_per_cpu_mean_utilization_pct', '-')} | "
            f"{utilization.get('all_sampled_cpus_ge_80pct_fraction', '-')} | "
            f"{f'{peak / 1048576:.2f} GiB' if isinstance(peak, int) else '-'} |"
        )
    lines += ["", "## Skips And Failures", ""]
    for row in report.get("events", []):
        if row.get("status") in {"SKIP", "FAIL"}:
            lines.append(f"- **{row.get('status')}** `{row.get('model_id')}`: {row.get('reason', 'unspecified')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_index(report: dict[str, Any], output_dir: Path) -> None:
    rows = report.get("quality", [])
    quality_dir = output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    def artifact_href(value: Any) -> str:
        if not value:
            return ""
        try:
            return os.path.relpath(Path(str(value)), quality_dir)
        except (OSError, ValueError):
            return str(value)

    cards = []
    for row in rows:
        status = str(row.get("status", "UNKNOWN"))
        artifact = artifact_href(row.get("artifact_path"))
        raw = artifact_href(row.get("raw_output_path"))
        prompt = artifact_href(row.get("prompt_path"))
        links = []
        if artifact:
            links.append(f'<a href="{html.escape(artifact)}">artifact</a>')
        if raw:
            links.append(f'<a href="{html.escape(raw)}">raw response</a>')
        if prompt:
            links.append(f'<a href="{html.escape(prompt)}">prompt</a>')
        preview = ""
        if (
            status in {"PASS", "WARN"}
            and artifact.lower().endswith(".svg")
            and bool((row.get("quality") or {}).get("xml_parseable"))
        ):
            preview = f'<img src="{html.escape(artifact)}" alt="Generated SIMD kernel diagram">'
        artifact_source = ""
        artifact_value = row.get("artifact_path")
        if artifact_value:
            artifact_file = Path(str(artifact_value))
            if artifact_file.is_file():
                artifact_source = artifact_file.read_text(encoding="utf-8", errors="replace")
        source_label = "generated C" if artifact.lower().endswith(".c") else "SVG source"
        source_view = (
            f'<details><summary>View {html.escape(source_label)}</summary>'
            f'<pre>{html.escape(artifact_source)}</pre></details>'
            if artifact_source else ""
        )
        quality = row.get("quality") or {}
        objective = quality.get("strict_compile")
        objective_label = "strict C compile" if objective is not None else "SVG parse"
        if objective is None:
            objective = quality.get("xml_parseable", False)
        usage = row.get("resource_usage") or {}
        active_cores = usage.get("average_cpu_cores")
        utilization = row.get("utilization") or {}
        timing = row.get("timing") or {}
        utilization_fact = (
            f'<span>average active cores: {float(active_cores):.1f}</span>'
            if isinstance(active_cores, (int, float)) else ""
        )
        p10 = utilization.get("p10_active_cores")
        selected = utilization.get("mean_selected_cpu_active_cores")
        weakest = utilization.get("minimum_per_cpu_mean_utilization_pct")
        sustained = utilization.get("all_sampled_cpus_ge_80pct_fraction")
        peak = row.get("peak_rss_kib")
        total_context = row.get("total_context_tokens")
        output_budget = row.get("output_budget_tokens")
        stop_reason = row.get("native_stop_reason")
        performance_facts = "".join((
            f'<span>total context: {int(total_context):,}</span>'
            if isinstance(total_context, int) else "",
            f'<span>output budget: {int(output_budget):,}</span>'
            if isinstance(output_budget, int) else "",
            f'<span>stop: {html.escape(str(stop_reason))}</span>'
            if stop_reason else "",
            f'<span>prefill: {float(timing["prompt_tok_s"]):.1f} tok/s</span>'
            if isinstance(timing.get("prompt_tok_s"), (int, float)) else "",
            f'<span>decode: {float(timing["decode_tok_s"]):.2f} tok/s</span>'
            if isinstance(timing.get("decode_tok_s"), (int, float)) else "",
            f'<span>p10 process cores: {float(p10):.1f}</span>'
            if isinstance(p10, (int, float)) else "",
            f'<span>top-like selected cores: {float(selected):.1f}</span>'
            if isinstance(selected, (int, float)) else "",
            f'<span>weakest core mean: {float(weakest):.1f}%</span>'
            if isinstance(weakest, (int, float)) else "",
            f'<span>all cores >=80%: {100.0 * float(sustained):.1f}% samples</span>'
            if isinstance(sustained, (int, float)) else "",
            f'<span>peak RAM: {float(peak) / 1048576.0:.2f} GiB</span>'
            if isinstance(peak, int) else "",
        ))
        cards.append(
            '<section class="result">'
            f'<header><div><h2>{html.escape(str(row.get("model", "unknown")))}</h2>'
            f'<p>{html.escape(str(row.get("prompt_id", "unknown")))}</p></div>'
            f'<span class="status {status.lower()}">{html.escape(status)}</span></header>'
            f'<div class="facts"><span>{html.escape(objective_label)}: {str(bool(objective)).lower()}</span>'
            f'<span>characters: {int(quality.get("characters", quality.get("answer_characters", 0)))}</span>'
            f'{utilization_fact}{performance_facts}</div>'
            f'<nav>{" | ".join(links) if links else "no artifact"}</nav>{preview}{source_view}</section>'
        )
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CKE Engineering Quality Artifacts</title>
<style>
body{{margin:0;background:#f4f6f8;color:#17202a;font:15px/1.45 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 20px 64px}}
h1{{font-size:30px;margin:0 0 8px}}.intro{{max-width:850px;color:#52606d;margin:0 0 26px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}
.result{{background:white;border:1px solid #d7dee5;border-radius:6px;padding:16px;min-width:0}}header{{display:flex;justify-content:space-between;gap:12px;align-items:start}}
h2{{font-size:18px;margin:0}}header p{{margin:2px 0;color:#647381}}.status{{font-weight:700;font-size:12px;padding:3px 7px;border-radius:4px;background:#e8edf2}}
.status.pass{{background:#dff4e6;color:#176b3a}}.status.warn{{background:#fff0c7;color:#7a5100}}.status.fail{{background:#fde3e1;color:#9f2d27}}.facts{{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0;color:#465461;font-size:13px}}
nav{{margin:10px 0}}a{{color:#075da8}}img{{display:block;width:100%;max-height:520px;object-fit:contain;border:1px solid #e1e6eb;background:white;margin-top:14px}}
details{{margin-top:14px}}summary{{cursor:pointer;font-weight:650}}pre{{overflow:auto;max-height:520px;padding:12px;background:#111820;color:#e7edf3;border-radius:4px;font:12px/1.45 ui-monospace,monospace;white-space:pre}}
</style></head><body><main><h1>CKE Engineering Quality Artifacts</h1>
<p class="intro">Generated C is checked with a strict syntax-only compiler invocation and is never executed automatically. SVG checks prove safe, standalone structure. Human review and numerical oracle testing decide whether a kernel is technically useful.</p>
<div class="grid">{''.join(cards) if cards else '<p>No engineering artifacts have completed yet.</p>'}</div></main></body></html>'''
    (quality_dir / "index.html").write_text(document, encoding="utf-8")


def publish(report: dict[str, Any], output_dir: Path) -> None:
    report["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(output_dir / "summary.json", report)
    write_markdown(report, output_dir / "summary.md")
    write_quality_index(report, output_dir)


def merge_quality_rows(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    replacement_keys = {
        (str(row.get("model_id")), str(row.get("prompt_id"))) for row in rows
    }
    retained = [
        row for row in report.get("quality", [])
        if (str(row.get("model_id")), str(row.get("prompt_id"))) not in replacement_keys
    ]
    report["quality"] = retained + rows


def build_runtime(
    model: str,
    runtime_dir: Path,
    context_tokens: int,
    decode_tokens: int,
    args: argparse.Namespace,
    env: dict[str, str],
    evidence_dir: Path,
    *,
    runtime_context: int | None = None,
) -> dict[str, Any]:
    resolved_runtime_context = (
        int(runtime_context)
        if runtime_context is not None
        else context_tokens + decode_tokens + 8
    )
    command = [
        sys.executable,
        str(SCRIPT_DIR / "ck_run_v8.py"),
        "run",
        model,
        "--run",
        str(runtime_dir),
        "--context-len",
        str(resolved_runtime_context),
        "--prefill-chunk-len",
        str(min(args.prefill_chunk_tokens, context_tokens)),
        "--logits-layout",
        "last",
        "--generate-only",
    ]
    return run_command(
        command, timeout=args.build_timeout, env=env, output_dir=evidence_dir,
        name="build-runtime", timed=False,
    )


def quality_context_plan(
    input_reserve_tokens: int,
    prompts: list[dict[str, Any]],
    total_context_tokens: int,
) -> dict[str, Any]:
    if input_reserve_tokens <= 0:
        raise ValueError("quality input reserve must be positive")
    if total_context_tokens > 0:
        output_budget_tokens = total_context_tokens - input_reserve_tokens - 8
        if output_budget_tokens <= 0:
            raise ValueError(
                "--quality-total-context must exceed --quality-context by at least 9 tokens"
            )
        return {
            "mode": "fixed_total_context",
            "total_context_tokens": total_context_tokens,
            "input_reserve_tokens": input_reserve_tokens,
            "output_budget_tokens": output_budget_tokens,
        }
    legacy_output_budget = max(int(prompt["max_tokens"]) for prompt in prompts)
    return {
        "mode": "legacy_prompt_caps",
        "total_context_tokens": input_reserve_tokens + legacy_output_budget + 8,
        "input_reserve_tokens": input_reserve_tokens,
        "output_budget_tokens": None,
    }


def classify_build_failure(build: dict[str, Any]) -> tuple[str, str]:
    detail = str(build.get("error") or build.get("stderr") or "runtime build failed")
    lowered = detail.lower()
    capacity_markers = (
        "cannot allocate memory",
        "failed to allocate",
        "out of memory",
        "context length exceeds",
        "context-len exceeds",
        "maximum context",
    )
    if int(build.get("returncode", 1)) in {-9, 137} or any(marker in lowered for marker in capacity_markers):
        return "SKIP", f"host/model capacity limit: {detail[-2000:]}"
    return "FAIL", detail[-2000:]


def run_cke_perf(
    runtime_dir: Path,
    context_tokens: int,
    args: argparse.Namespace,
    env: dict[str, str],
    evidence_dir: Path,
) -> dict[str, Any]:
    token_csv = ",".join([str(args.token_id)] * context_tokens)
    samples = []
    for repetition in range(args.repetitions):
        trace_path = evidence_dir / f"cke-{repetition + 1}.trace.json"
        command = [
            str(args.ck_cli),
            "--lib", str(runtime_dir / "libmodel.so"),
            "--weights", str(runtime_dir / "weights.bump"),
            "--manifest", str(runtime_dir / "weights_manifest.map"),
            "--prompt-tokens", token_csv,
            "--max-tokens", str(args.decode_tokens + 1),
            "--context", str(context_tokens + args.decode_tokens + 8),
            "--temperature", "0",
            "--ignore-eos",
            "--quiet-output",
            "--no-chat-template",
            "--no-stream",
            "--timing",
            "--token-trace-json", str(trace_path),
        ]
        result = run_command(
            command, timeout=args.run_timeout, env=env, output_dir=evidence_dir,
            name=f"cke-{repetition + 1}", timed=True,
        )
        sample: dict[str, Any] = {
            "returncode": result["returncode"],
            "elapsed_seconds": result["elapsed_seconds"],
            "peak_rss_kib": result["peak_rss_kib"],
            "resource_usage": result["resource_usage"],
            "utilization": result["utilization"],
            "utilization_timeline_path": result["utilization_timeline_path"],
            "stdout_path": result["stdout_path"],
            "stderr_path": result["stderr_path"],
        }
        if result["returncode"] == 0:
            sample["timing"] = parse_timing(result["stdout"] + result["stderr"])
            sample["trace"] = json.loads(trace_path.read_text(encoding="utf-8"))
        else:
            sample["error"] = result["error"] or result["stderr"][-2000:]
        samples.append(sample)
    successful = [sample for sample in samples if "timing" in sample]
    utilization_rows = [
        sample["utilization"] for sample in successful
        if isinstance(sample.get("utilization"), dict)
        and int(sample["utilization"].get("sample_count", 0)) > 0
    ]
    hashes = [sample["trace"].get("first_logits_fnv1a64") for sample in successful]
    consumed = all(
        int(sample["trace"].get("prompt_tokens", -1)) == context_tokens
        and int(sample["timing"].get("prompt_tokens", -1)) == context_tokens
        for sample in successful
    ) and len(successful) == args.repetitions
    minimum_active_cores = max(1.0, float(args.threads) * args.min_active_core_fraction)
    utilization_verified = bool(successful) and all(
        isinstance(sample.get("resource_usage"), dict)
        and float(sample["resource_usage"].get("average_cpu_cores", 0.0)) >= minimum_active_cores
        for sample in successful
    )
    return {
        "status": (
            "PASS" if consumed and len(set(hashes)) == 1 and hashes[0]
            and utilization_verified else "FAIL"
        ),
        "samples": samples,
        "timing": median_timing(successful),
        "peak_rss_kib": max((sample.get("peak_rss_kib") or 0 for sample in samples), default=0),
        "first_logits_hashes": hashes,
        "first_logits_repeatable": bool(hashes and len(hashes) == args.repetitions and len(set(hashes)) == 1),
        "consumed_token_count_verified": consumed,
        "utilization_verified": utilization_verified,
        "minimum_active_cores": minimum_active_cores,
        "average_active_cores": statistics.median(
            float(sample["resource_usage"]["average_cpu_cores"])
            for sample in successful if isinstance(sample.get("resource_usage"), dict)
        ) if successful and all(isinstance(sample.get("resource_usage"), dict)
                                for sample in successful) else None,
        "p10_active_cores": statistics.median(
            float(row["p10_active_cores"]) for row in utilization_rows
        ) if utilization_rows else None,
        "longest_low_utilization_seconds": max(
            (float(row["longest_low_utilization_seconds"])
             for row in utilization_rows), default=None,
        ),
        "input_token_id": args.token_id,
        "input_sha256": token_sha256(args.token_id, context_tokens),
    }


def run_numerical_parity(
    gguf: Path | None,
    runtime_dir: Path,
    context_tokens: int,
    args: argparse.Namespace,
    env: dict[str, str],
    evidence_dir: Path,
) -> dict[str, Any]:
    if args.no_llama:
        return {"status": "SKIP", "reason": "disabled by --no-llama"}
    if gguf is None or not gguf.is_file():
        return {"status": "SKIP", "reason": "no local GGUF numerical oracle"}
    if context_tokens != args.parity_context:
        return {"status": "SKIP", "reason": f"parity is sampled at {args.parity_context} tokens"}
    report_path = evidence_dir / "numerical-parity.json"
    token_csv = ",".join([str(args.token_id)] * context_tokens)
    parity_env = dict(env)
    parity_env["CK_LLAMA_CPP_ROOT"] = str(args.llama_root)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "compare_multitoken_logits_v8.py"),
        "--model-dir", str(runtime_dir),
        "--gguf", str(gguf),
        "--tokens", token_csv,
        "--max-new-tokens", str(args.parity_new_tokens),
        "--ctx-len", str(context_tokens + args.parity_new_tokens + 8),
        "--threads", str(args.threads),
        "--append-on-divergence", "stop",
        "--json-out", str(report_path),
        "--summary",
    ]
    result = run_command(
        command, timeout=args.run_timeout, env=parity_env, output_dir=evidence_dir,
        name="numerical-parity", timed=True,
    )
    if result["returncode"] != 0 or not report_path.is_file():
        return {"status": "FAIL", "reason": result["error"] or result["stderr"][-2000:]}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    top1 = sum(bool(step.get("top1_match")) for step in steps)
    bit_exact = sum(bool(step.get("bit_exact", float(step.get("max_abs_diff", 1.0)) == 0.0)) for step in steps)
    passed = bool(steps and payload.get("first_divergence") is None and top1 == len(steps))
    return {
        "status": "PASS" if passed else "FAIL",
        "compared_rows": len(steps),
        "top1_exact_rows": top1,
        "bit_exact_rows": bit_exact,
        "first_divergence": payload.get("first_divergence"),
        "report_path": str(report_path),
        "peak_rss_kib": result["peak_rss_kib"],
    }


def run_llama_perf(
    gguf: Path | None,
    context_tokens: int,
    args: argparse.Namespace,
    env: dict[str, str],
    evidence_dir: Path,
) -> dict[str, Any]:
    binary = args.llama_root / "build" / "bin" / "llama-bench"
    if args.no_llama:
        return {"status": "SKIP", "reason": "disabled by --no-llama"}
    if gguf is None or not gguf.is_file():
        return {"status": "SKIP", "reason": "no local GGUF oracle artifact"}
    if not binary.is_file():
        return {"status": "SKIP", "reason": f"llama-bench is missing: {binary}"}
    command = [
        str(binary), "-m", str(gguf), "-p", str(context_tokens),
        "-n", str(args.decode_tokens), "-t", str(args.threads), "-ngl", "0",
        "-r", "1", "-o", "json",
    ]
    result = run_command(
        command, timeout=args.run_timeout, env=env, output_dir=evidence_dir,
        name="llama-bench", timed=True,
    )
    if result["returncode"] != 0:
        return {"status": "FAIL", "reason": result["error"] or result["stderr"][-2000:]}
    try:
        timing = parse_llama_bench(result["stdout"])
    except (ValueError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "reason": str(exc)}
    return {
        "status": "PASS",
        "timing": timing,
        "peak_rss_kib": result["peak_rss_kib"],
        "resource_usage": result["resource_usage"],
        "average_active_cores": (
            float(result["resource_usage"]["average_cpu_cores"])
            if isinstance(result.get("resource_usage"), dict) else None
        ),
        "consumed_token_count_verified": int(timing["prompt_tokens"]) == context_tokens,
        "comparison_scope": "matched token count performance; not a token-ID numerical parity claim",
    }


def run_quality(
    row: dict[str, Any],
    runtime_dir: Path,
    runtime_context: int,
    prompts: list[dict[str, Any]],
    args: argparse.Namespace,
    env: dict[str, str],
    output_dir: Path,
    *,
    context_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    dependencies: dict[str, str] = {}
    quality_prefix = str(getattr(args, "quality_prefix", ""))
    minimum_root_prefill_tokens = max(
        0, int(getattr(args, "quality_min_root_prefill_tokens", 0))
    )
    resolved_context_plan = context_plan or {
        "mode": "legacy_prompt_caps",
        "total_context_tokens": runtime_context,
        "input_reserve_tokens": None,
        "output_budget_tokens": None,
    }
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        result_path = output_dir / f"{prompt_id}.json"
        try:
            prompt_text = materialize_quality_prompt(
                prompt, dependencies, quality_prefix
            )
        except ValueError as exc:
            result = {
                "model_id": row["id"], "model": row["label"], "prompt_id": prompt_id,
                "status": "FAIL", "quality": {"pass": False, "reason": str(exc)},
            }
            atomic_json(result_path, result)
            results.append(result)
            continue
        prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        configured_prompt_cap = int(prompt["max_tokens"])
        fixed_output_budget = resolved_context_plan.get("output_budget_tokens")
        output_budget_tokens = (
            int(fixed_output_budget)
            if isinstance(fixed_output_budget, int)
            else configured_prompt_cap
        )
        command_output_limit = output_budget_tokens + (
            0 if resolved_context_plan.get("mode") == "fixed_total_context" else 1
        )
        run_signature_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "prompt_sha256": prompt_sha256,
                    "runtime_context": runtime_context,
                    "output_budget_tokens": output_budget_tokens,
                    "context_mode": resolved_context_plan.get("mode"),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if args.resume and result_path.is_file():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            artifact_path = Path(str(cached.get("artifact_path", "")))
            if (
                cached.get("run_signature_sha256") == run_signature_sha256
                and artifact_path.is_file()
            ):
                dependencies[prompt_id] = artifact_path.read_text(encoding="utf-8")
                results.append(cached)
                continue
        prompt_path = output_dir / f"{prompt_id}.prompt.txt"
        prompt_path.write_text(prompt_text + "\n", encoding="utf-8")
        trace_path = output_dir / f"{prompt_id}.trace.json"
        command = [
            str(args.ck_cli),
            "--lib", str(runtime_dir / "libmodel.so"),
            "--weights", str(runtime_dir / "weights.bump"),
            "--manifest", str(runtime_dir / "weights_manifest.map"),
            "--prompt", prompt_text,
            "--max-tokens", str(command_output_limit),
            "--context", str(runtime_context),
            "--temperature", "0",
            "--no-stream", "--timing",
            "--token-trace-json", str(trace_path),
        ]
        command_result = run_command(
            command, timeout=args.quality_timeout, env=env, output_dir=output_dir,
            name=prompt_id, timed=True,
        )
        generated = extract_generated(command_result["stdout"])
        raw_output_path = output_dir / f"{prompt_id}.raw.txt"
        raw_output_path.write_text(generated + "\n", encoding="utf-8")
        extension = str(prompt.get("artifact_extension", ".txt"))
        if not re.fullmatch(r"\.[a-z0-9]+", extension):
            raise ValueError(f"invalid artifact extension for {prompt_id}: {extension!r}")
        artifact_path = output_dir / f"{prompt_id}{extension}"
        kind = str(prompt.get("kind", ""))
        if kind == "c_kernel.v1":
            quality = code_quality(
                generated, artifact_path, prompt, env, output_dir,
            )
            artifact_text = extract_c_source(generated)
        elif kind == "kernel_svg.v1":
            quality = svg_quality(generated, prompt)
            artifact_text = extract_svg_markup(generated)
            artifact_path.write_text(artifact_text, encoding="utf-8")
        else:
            raise ValueError(f"unsupported quality prompt kind: {kind!r}")
        dependencies[prompt_id] = artifact_text
        timing = None
        native_trace = None
        if command_result["returncode"] == 0:
            try:
                timing = parse_timing(command_result["stdout"] + command_result["stderr"])
            except ValueError:
                pass
        if trace_path.is_file():
            try:
                native_trace = json.loads(trace_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                native_trace = None
        prefill_requirement_applies = not str(prompt.get("depends_on", ""))
        prefill_tokens = int(timing["prompt_tokens"]) if timing is not None else None
        prompt_not_truncated = "truncat" not in str(command_result.get("stderr", "")).lower()
        prefill_consumption_verified = (
            not prefill_requirement_applies
            or minimum_root_prefill_tokens == 0
            or (
                prefill_tokens is not None
                and prefill_tokens >= minimum_root_prefill_tokens
            )
        )
        result = {
            "model_id": row["id"], "model": row["label"], "prompt_id": prompt_id,
            "status": engineering_quality_status(
                int(command_result["returncode"]), quality,
                prefill_consumption_verified, prompt_not_truncated,
            ),
            "quality": quality, "timing": timing, "peak_rss_kib": command_result["peak_rss_kib"],
            "resource_usage": command_result["resource_usage"],
            "utilization": command_result.get("utilization", {}),
            "utilization_timeline_path": command_result.get("utilization_timeline_path"),
            "prompt_sha256": prompt_sha256,
            "run_signature_sha256": run_signature_sha256,
            "context_plan": resolved_context_plan,
            "total_context_tokens": runtime_context,
            "output_budget_tokens": output_budget_tokens,
            "command_output_limit_tokens": command_output_limit,
            "configured_prompt_cap_tokens": configured_prompt_cap,
            "native_stop_reason": (
                native_trace.get("stop_reason")
                if isinstance(native_trace, dict) else None
            ),
            "minimum_root_prefill_tokens": minimum_root_prefill_tokens,
            "prefill_requirement_applies": prefill_requirement_applies,
            "prefill_consumption_verified": prefill_consumption_verified,
            "prompt_not_truncated": prompt_not_truncated,
            "prompt_path": str(prompt_path), "raw_output_path": str(raw_output_path),
            "artifact_path": str(artifact_path), "output_path": str(artifact_path),
            "trace_path": str(trace_path),
            "stdout_log_path": command_result["stdout_path"],
            "stderr_log_path": command_result["stderr_path"],
            "evaluation_scope": (
                "objective structural checks only; human review and numerical oracle required"
            ),
        }
        atomic_json(result_path, result)
        results.append(result)
    return results


def certify_quality_at_context(
    row: dict[str, Any],
    model: str,
    runtime_dir: Path,
    context_tokens: int,
    prompts: list[dict[str, Any]],
    args: argparse.Namespace,
    env: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    model_id = str(row["id"])
    evidence_dir = args.output_dir / "quality" / model_id
    context_plan = quality_context_plan(
        context_tokens,
        prompts,
        int(getattr(args, "quality_total_context", 0)),
    )
    runtime_context = int(context_plan["total_context_tokens"])
    build_decode_tokens = context_plan.get("output_budget_tokens")
    if not isinstance(build_decode_tokens, int):
        build_decode_tokens = max(int(prompt["max_tokens"]) for prompt in prompts)
    build = build_runtime(
        model, runtime_dir, context_tokens, build_decode_tokens, args, env, evidence_dir,
        runtime_context=runtime_context,
    )
    if build["returncode"] != 0:
        status, reason = classify_build_failure(build)
        return [], {
            "model_id": model_id,
            "status": status,
            "reason": f"quality runtime build failed at {runtime_context} total tokens: {reason}",
        }
    return run_quality(
        row, runtime_dir, runtime_context, prompts, args, env, evidence_dir,
        context_plan=context_plan,
    ), None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--quality-prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contexts", default="2048,8192,32768,65536,131072,262144")
    parser.add_argument("--models", default="all", help="Comma-separated IDs or all")
    parser.add_argument(
        "--threads", type=int, default=0,
        help="Worker count; 0 selects the allowed physical cores (default)",
    )
    parser.add_argument(
        "--cpu-policy", choices=("physical", "inherit"), default="inherit",
        help="Inherit scheduler affinity (default) or pin one worker per physical core",
    )
    parser.add_argument(
        "--min-active-core-fraction", type=float, default=0.5,
        help="Fail CKE performance rows below this fraction of requested average core occupancy",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--parity-context", type=int, default=2048)
    parser.add_argument("--parity-new-tokens", type=int, default=8)
    parser.add_argument("--token-id", type=int, default=100)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=8192)
    parser.add_argument("--build-timeout", type=int, default=14400)
    parser.add_argument("--run-timeout", type=int, default=43200)
    parser.add_argument("--quality-timeout", type=int, default=43200)
    parser.add_argument("--quality-context", type=int, default=8192)
    parser.add_argument(
        "--quality-total-context", type=int, default=0,
        help=(
            "Allocate this exact total context for quality generation. Decode may use "
            "all tokens remaining after --quality-context and an 8-token safety margin; "
            "0 preserves per-prompt output caps."
        ),
    )
    parser.add_argument(
        "--quality-prefix-file", type=Path,
        help=(
            "Prepend this dossier to root quality prompts; dependent prompts "
            "receive only their declared artifact"
        ),
    )
    parser.add_argument(
        "--quality-min-root-prefill-tokens", type=int, default=0,
        help=(
            "Fail a root quality prompt unless native timing reports at least "
            "this many consumed prompt tokens"
        ),
    )
    parser.add_argument("--ck-cli", type=Path, default=ROOT / "build" / "ck-cli-v8")
    parser.add_argument("--llama-root", type=Path, default=ROOT / "llama.cpp")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-llama", action="store_true")
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument(
        "--quality-only", action="store_true",
        help="Generate and validate paired C/SVG artifacts without running the capacity ladder",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if args.repetitions < 2:
        raise ValueError("at least two repetitions are required for first-logit repeatability")
    if args.threads < 0:
        raise ValueError("--threads must be non-negative")
    if args.quality_min_root_prefill_tokens < 0:
        raise ValueError("--quality-min-root-prefill-tokens must be non-negative")
    if args.quality_context <= 0:
        raise ValueError("--quality-context must be positive")
    if args.quality_total_context < 0:
        raise ValueError("--quality-total-context must be non-negative")
    if not 0.0 < args.min_active_core_fraction <= 1.0:
        raise ValueError("--min-active-core-fraction must be in (0, 1]")
    if args.quality_only and args.no_quality:
        raise ValueError("--quality-only and --no-quality cannot be combined")
    contexts = parse_contexts(args.contexts)
    catalog = load_schema(args.catalog, "cke.v8.long_context_model_catalog")
    prompt_payload = load_schema(args.quality_prompts, "cke.v8.long_context_quality_prompts")
    quality_prompts = validate_quality_prompts(prompt_payload)
    quality_context_plan(
        args.quality_context, quality_prompts, args.quality_total_context
    )
    args.quality_prefix = ""
    quality_prefix_sha256 = None
    if args.quality_prefix_file is not None:
        prefix_path = args.quality_prefix_file.expanduser().resolve()
        args.quality_prefix = prefix_path.read_text(encoding="utf-8")
        if not args.quality_prefix.strip():
            raise ValueError("--quality-prefix-file must not be empty")
        quality_prefix_sha256 = hashlib.sha256(
            args.quality_prefix.encode("utf-8")
        ).hexdigest()
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    rows = [row for row in catalog["models"] if selected == {"all"} or row["id"] in selected]
    if not rows:
        raise ValueError("no models selected")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.ck_cli = args.ck_cli.expanduser().resolve()
    args.llama_root = args.llama_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu_plan = resolve_cpu_plan(args.threads, args.cpu_policy)
    args.threads = int(cpu_plan["effective_threads"])
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.threads)
    if cpu_plan["affinity"]:
        env["CK_BENCH_CPU_AFFINITY"] = ",".join(
            str(cpu) for cpu in cpu_plan["affinity"]
        )
    env["OMP_NUM_THREADS"] = "1"
    env["CK_THREADPOOL_PROFILE"] = "1"

    quality_signature = {
        "quality_mode": "engineering_pair_v1",
        "quality_prompt_sha256": hashlib.sha256(
            args.quality_prompts.read_bytes()
        ).hexdigest(),
        "quality_prefix_sha256": quality_prefix_sha256,
        "quality_min_root_prefill_tokens": args.quality_min_root_prefill_tokens,
    }
    report: dict[str, Any] = {
        "schema": "cke.v8.long_context_certification", "schema_version": 1,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host_fingerprint(),
        "config": {"contexts": contexts, "threads": args.threads,
                   "cpu_plan": cpu_plan, "repetitions": args.repetitions,
                   "min_active_core_fraction": args.min_active_core_fraction,
                   "decode_tokens": args.decode_tokens, "token_id": args.token_id,
                   "capacity_workload": "deterministic_fixed_token",
                   "parity_context": args.parity_context,
                   "parity_new_tokens": args.parity_new_tokens,
                   "quality_context": args.quality_context,
                   "quality_input_reserve_tokens": args.quality_context,
                   "quality_total_context_tokens": args.quality_total_context,
                   **quality_signature},
        "performance": [], "quality": [], "events": [],
    }
    summary_path = args.output_dir / "summary.json"
    quality_resume_compatible = True
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("schema") == report["schema"]:
            previous_config = previous.get("config", {})
            quality_resume_compatible = all(
                previous_config.get(key) == value
                for key, value in quality_signature.items()
            )
            report = previous
    report.setdefault("config", {}).update(quality_signature)

    completed = {
        (row.get("model_id"), int(row.get("context_tokens", 0)))
        for row in report.get("performance", [])
        if row.get("status") == "PASS"
    }
    required_quality_ids = {str(prompt["id"]) for prompt in quality_prompts}
    quality_ids_by_model: dict[str, set[str]] = {}
    for result in report.get("quality", []):
        if result.get("status") not in {"PASS", "WARN", "FAIL"}:
            continue
        quality_ids_by_model.setdefault(str(result.get("model_id")), set()).add(
            str(result.get("prompt_id"))
        )
    completed_quality = {
        model_id for model_id, prompt_ids in quality_ids_by_model.items()
        if required_quality_ids <= prompt_ids
    }
    if not quality_resume_compatible:
        completed_quality.clear()
    model_runtime: dict[str, Path] = {}
    model_sources: dict[str, tuple[str, str]] = {}
    for row in rows:
        source = resolve_model(row, args.allow_download)
        model_sources[row["id"]] = source
        model_runtime[row["id"]] = args.output_dir / "runtimes" / row["id"]
        if not source[0]:
            report["events"].append({"model_id": row["id"], "status": "SKIP", "reason": source[1]})
    publish(report, args.output_dir)

    if args.quality_only:
        if not args.ck_cli.is_file():
            raise FileNotFoundError(f"native CLI is missing: {args.ck_cli}")
        quality_only_failed = False
        for row in rows:
            model_id = str(row["id"])
            model = model_sources[model_id][0]
            if not model or model_id in completed_quality:
                continue
            quality_rows, event = certify_quality_at_context(
                row, model, model_runtime[model_id], args.quality_context,
                quality_prompts, args, env,
            )
            merge_quality_rows(report, quality_rows)
            if event:
                report["events"].append(event)
                quality_only_failed = quality_only_failed or event.get("status") == "FAIL"
            publish(report, args.output_dir)
        quality_only_failed = quality_only_failed or any(
            result.get("status") == "FAIL" for result in report.get("quality", [])
        )
        return 1 if quality_only_failed else 0

    for context_tokens in contexts:
        for row in rows:
            model_id = str(row["id"])
            if (model_id, context_tokens) in completed:
                continue
            model, source_reason = model_sources[model_id]
            if not model:
                continue
            runtime_dir = model_runtime[model_id]
            evidence_dir = args.output_dir / "models" / model_id / f"ctx-{context_tokens}"
            print(f"[{model_id} ctx={context_tokens}] build", flush=True)
            build = build_runtime(model, runtime_dir, context_tokens, args.decode_tokens, args, env, evidence_dir)
            if build["returncode"] != 0:
                failure_status, failure_reason = classify_build_failure(build)
                result = {"model_id": model_id, "model": row["label"], "context_tokens": context_tokens,
                          "status": failure_status, "reason": failure_reason}
                report["performance"].append(result)
                report["events"].append(result)
                publish(report, args.output_dir)
                print(
                    f"[{model_id} ctx={context_tokens}] {failure_status} build",
                    flush=True,
                )
                continue
            if not args.ck_cli.is_file():
                raise FileNotFoundError(f"native CLI is missing: {args.ck_cli}")
            gguf = resolve_gguf(row, model, runtime_dir)
            print(
                f"[{model_id} ctx={context_tokens}] CKE x{args.repetitions}",
                flush=True,
            )
            cke = run_cke_perf(runtime_dir, context_tokens, args, env, evidence_dir)
            llama = run_llama_perf(gguf, context_tokens, args, env, evidence_dir)
            parity = run_numerical_parity(
                gguf, runtime_dir, context_tokens, args, env, evidence_dir
            )
            status = "PASS" if (
                cke["status"] == "PASS"
                and llama["status"] in {"PASS", "SKIP"}
                and parity["status"] in {"PASS", "SKIP"}
            ) else "FAIL"
            result = {
                "model_id": model_id, "model": row["label"], "model_source": source_reason,
                "context_tokens": context_tokens, "runtime_context": context_tokens + args.decode_tokens + 8,
                "workload": "deterministic_fixed_token",
                "status": status, "cke": cke, "llama_cpp": llama,
                "numerical_parity": parity,
                "providers": provider_summary(runtime_dir),
            }
            if cke.get("timing") and llama.get("timing"):
                result["relative_prefill"] = (
                    float(cke["timing"]["prompt_tok_s"]) / float(llama["timing"]["prompt_tok_s"])
                )
            report["performance"].append(result)
            if status == "FAIL":
                report["events"].append({"model_id": model_id, "status": "FAIL",
                                         "reason": f"context {context_tokens} certification failed"})
            publish(report, args.output_dir)
            timing = cke.get("timing") if isinstance(cke.get("timing"), dict) else {}
            prompt_tok_s = timing.get("prompt_tok_s", "n/a")
            active_cores = cke.get("average_active_cores", "n/a")
            print(
                f"[{model_id} ctx={context_tokens}] {status} "
                f"prefill_tok_s={prompt_tok_s} active_cores={active_cores}",
                flush=True,
            )

            if (
                not args.no_quality
                and status == "PASS"
                and context_tokens == args.quality_context
                and model_id not in completed_quality
            ):
                quality_rows, event = certify_quality_at_context(
                    row, model, runtime_dir, context_tokens,
                    quality_prompts, args, env,
                )
                merge_quality_rows(report, quality_rows)
                completed_quality.add(model_id)
                if event:
                    report["events"].append(event)
                publish(report, args.output_dir)

    if not args.no_quality:
        passing_contexts: dict[str, int] = {}
        for result in report["performance"]:
            if result.get("status") == "PASS":
                model_id = str(result["model_id"])
                passing_contexts[model_id] = max(passing_contexts.get(model_id, 0), int(result["context_tokens"]))
        for row in rows:
            model_id = str(row["id"])
            if model_id not in passing_contexts or model_id in completed_quality:
                continue
            context_tokens = passing_contexts[model_id]
            runtime_dir = model_runtime[model_id]
            quality_rows, event = certify_quality_at_context(
                row, model_sources[model_id][0], runtime_dir, context_tokens,
                quality_prompts, args, env,
            )
            merge_quality_rows(report, quality_rows)
            completed_quality.add(model_id)
            if event:
                report["events"].append(event)
            publish(report, args.output_dir)

    failed = any(
        row.get("status") == "FAIL"
        for section in ("performance", "quality")
        for row in report.get(section, [])
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
