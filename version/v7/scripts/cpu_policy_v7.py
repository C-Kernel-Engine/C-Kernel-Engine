#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


CPU_SYSFS = Path("/sys/devices/system/cpu")


def _cpu_id(cpu_dir: Path) -> int:
    return int(cpu_dir.name.replace("cpu", ""))


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def parse_cpulist(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            if end < start:
                start, end = end, start
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def format_cpulist(cpus: List[int]) -> str:
    vals = sorted(set(int(v) for v in cpus))
    if not vals:
        return ""
    ranges: List[str] = []
    start = prev = vals[0]
    for cur in vals[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cur
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def detect_fast_core_policy(cpu_base: Path = CPU_SYSFS) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    seen_core_keys = set()
    any_core_type = False
    any_freq = False

    for cpu_dir in sorted(cpu_base.glob("cpu[0-9]*"), key=_cpu_id):
        cpu = _cpu_id(cpu_dir)
        siblings = parse_cpulist(_read_text(cpu_dir / "topology" / "thread_siblings_list") or str(cpu))
        primary_cpu = min(siblings) if siblings else cpu
        core_id = _read_text(cpu_dir / "topology" / "core_id") or str(primary_cpu)
        core_key = f"{core_id}:{primary_cpu}"
        if core_key in seen_core_keys:
            continue
        seen_core_keys.add(core_key)

        core_type = _read_int(cpu_dir / "topology" / "core_type")
        max_freq = _read_int(cpu_dir / "cpufreq" / "cpuinfo_max_freq")
        any_core_type = any_core_type or core_type is not None
        any_freq = any_freq or max_freq is not None
        records.append(
            {
                "core_key": core_key,
                "primary_cpu": primary_cpu,
                "siblings": siblings,
                "core_type": core_type,
                "max_freq": max_freq,
            }
        )

    result: Dict[str, object] = {
        "hybrid_cpu": False,
        "detection": None,
        "fast_primary_cpus": [],
        "fast_logical_cpus": [],
        "fast_core_count": 0,
        "fast_logical_count": 0,
    }
    if not records:
        return result

    fast_records: List[Dict[str, object]] = []
    if any_core_type:
        fast_records = [r for r in records if r.get("core_type") == 0]
        slow_records = [r for r in records if r.get("core_type") not in (None, 0)]
        result["detection"] = "core_type"
        result["hybrid_cpu"] = bool(fast_records and slow_records)
    elif any_freq:
        freqs = [int(r["max_freq"]) for r in records if r.get("max_freq") is not None]
        if freqs:
            fast_freq = max(freqs)
            fast_records = [r for r in records if r.get("max_freq") == fast_freq]
            slow_records = [r for r in records if r.get("max_freq") not in (None, fast_freq)]
            result["detection"] = "max_freq"
            result["hybrid_cpu"] = bool(fast_records and slow_records)

    if not fast_records:
        return result

    fast_primary = sorted(int(r["primary_cpu"]) for r in fast_records)
    fast_logical = sorted(
        {
            int(cpu)
            for r in fast_records
            for cpu in (r.get("siblings") or [])
        }
    )
    result["fast_primary_cpus"] = fast_primary
    result["fast_logical_cpus"] = fast_logical
    result["fast_core_count"] = len(fast_primary)
    result["fast_logical_count"] = len(fast_logical)
    return result


def resolve_dense_cpu_policy(
    requested_threads: Optional[int],
    affinity_cpulist: Optional[str],
    *,
    cpu_policy: str = "auto",
    cpu_base: Path = CPU_SYSFS,
) -> Dict[str, object]:
    requested = int(requested_threads or 0)
    explicit_affinity = str(affinity_cpulist or "").strip()
    topology = detect_fast_core_policy(cpu_base=cpu_base)

    resolved_threads = requested
    resolved_affinity = explicit_affinity
    source = "explicit"

    if explicit_affinity:
        cpus = parse_cpulist(explicit_affinity)
        if cpus and resolved_threads <= 0:
            resolved_threads = len(cpus)
    elif cpu_policy != "unrestricted" and bool(topology.get("hybrid_cpu")):
        fast_primary = list(topology.get("fast_primary_cpus") or [])
        fast_logical = list(topology.get("fast_logical_cpus") or [])
        chosen = fast_primary
        if cpu_policy == "prefer-fast-cores-smt":
            chosen = fast_logical
        if chosen:
            target = len(chosen) if requested <= 0 else min(requested, len(chosen))
            chosen = chosen[:target]
            resolved_threads = len(chosen)
            resolved_affinity = format_cpulist(chosen)
            source = "auto-fast-cores"
    else:
        source = "unrestricted"

    if resolved_threads <= 0:
        resolved_threads = requested if requested > 0 else 1

    return {
        "cpu_policy": cpu_policy,
        "policy_source": source,
        "requested_threads": requested,
        "resolved_threads": resolved_threads,
        "resolved_affinity_cpulist": resolved_affinity,
        "topology": topology,
    }
