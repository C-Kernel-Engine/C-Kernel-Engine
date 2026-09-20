#!/usr/bin/env python3
"""Native v8 orchestration for generated FP32 forward/backward certification.

The generated training implementation is shared with the proven v7 codegen path.
This command owns the v8 evidence contract: explicit parameter mapping, loaded
library provenance, PyTorch comparison, negative controls, and fail-closed JSON.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import socket
import shutil
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
V7_SCRIPTS = ROOT / "version" / "v7" / "scripts"
DEFAULT_REPORT = ROOT / "version" / "v8" / ".cache" / "reports" / "training_certification_latest.json"
DEFAULT_RUN_DIR = ROOT / "version" / "v8" / ".cache" / "training_certification" / "fp32_dense_2layer"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
    }


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in cmd],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _git_identity() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--short"])
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status": status.stdout.splitlines()[:100],
    }


def _compiler_identity() -> dict[str, Any]:
    requested = os.environ.get("CK_V7_COMPILER", "").strip() or os.environ.get("CC", "").strip() or "gcc"
    resolved = shutil.which(requested)
    version = _run([requested, "--version"])
    return {
        "requested": requested,
        "path": str(Path(resolved).resolve()) if resolved else None,
        "version": version.stdout.splitlines()[0] if version.returncode == 0 and version.stdout else None,
    }


def _loaded_library_evidence(path: Path) -> dict[str, Any]:
    identity = _file_identity(path)
    maps = Path("/proc/self/maps")
    matches: list[str] = []
    if maps.exists():
        target = str(path.resolve())
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            if target in line:
                matches.append(line)
    identity["observed_in_process_maps"] = bool(matches)
    identity["process_map_entries"] = matches
    return identity


def _mapped_library_evidence(basename: str) -> dict[str, Any]:
    maps = Path("/proc/self/maps")
    entries: list[str] = []
    mapped_paths: list[str] = []
    if maps.exists():
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 6:
                continue
            raw_path = fields[5].removesuffix(" (deleted)")
            if Path(raw_path).name != basename:
                continue
            entries.append(line)
            mapped_paths.append(str(Path(raw_path).resolve()))
    unique_paths = sorted(set(mapped_paths))
    identities: list[dict[str, Any]] = []
    for raw_path in unique_paths:
        path = Path(raw_path)
        if path.is_file():
            identities.append(_file_identity(path))
        else:
            identities.append({"path": raw_path, "missing": True})
    return {
        "basename": basename,
        "observed_in_process_maps": bool(entries),
        "mapped_paths": unique_paths,
        "process_map_entries": entries,
        "identities": identities,
    }


def _manifest_shapes(run_dir: Path) -> dict[str, list[int]]:
    doc = json.loads((run_dir / "weights_manifest.json").read_text(encoding="utf-8"))
    rows = doc.get("entries")
    if not isinstance(rows, list):
        raise ValueError("weights_manifest.json has no entries[]")
    out: dict[str, list[int]] = {}
    duplicates: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("name"):
            continue
        name = str(row["name"])
        if name in out:
            duplicates.append(name)
        shape = row.get("shape")
        if isinstance(shape, list) and shape:
            out[name] = [int(value) for value in shape]
    if duplicates:
        raise ValueError(f"duplicate manifest tensor names: {sorted(set(duplicates))}")
    return out


def _expected_trainable_parameters(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive this FP32 fixture's trainable set without using generated IR inventories."""
    manifest_path = run_dir / "weights_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("weights manifest has no entries[]")
    expected: list[dict[str, Any]] = []
    excluded_fixture: list[str] = []
    frozen_dtype: list[str] = []
    tied: list[dict[str, str]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping) or not raw_entry.get("name"):
            continue
        name = str(raw_entry["name"])
        if name.startswith("tiny."):
            excluded_fixture.append(name)
            continue
        dtype = str(raw_entry.get("dtype", "")).lower()
        if dtype not in {"fp32", "f32"}:
            frozen_dtype.append(name)
            continue
        alias_of = raw_entry.get("alias_of") or raw_entry.get("tied_to")
        if alias_of:
            tied.append({"name": name, "canonical": str(alias_of)})
            continue
        shape = raw_entry.get("shape")
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"manifest parameter {name!r} has no declared shape")
        normalized_shape = [int(value) for value in shape]
        numel = math.prod(normalized_shape)
        byte_size = int(raw_entry.get("size", numel * 4) or 0)
        if byte_size != numel * 4:
            raise ValueError(f"manifest FP32 byte size mismatch for {name!r}")
        expected.append(
            {
                "name": name,
                "manifest_name": name,
                "shape": normalized_shape,
                "numel": numel,
            }
        )
    if not expected:
        raise ValueError("forward parameter contract declares no trainable parameters")
    names = [str(row["name"]) for row in expected]
    if len(names) != len(set(names)):
        raise ValueError("manifest parameter contract contains duplicate trainable names")
    return expected, {
        "source": str(manifest_path),
        "policy": "all_primary_fp32_manifest_parameters_excluding_tiny_parity_fixture_and_aliases",
        "excluded_fixture": sorted(excluded_fixture),
        "frozen_by_dtype": sorted(frozen_dtype),
        "tied": tied,
    }


def _forward_ir_trainable_names(run_dir: Path) -> list[str]:
    ir1 = json.loads((run_dir / "ir1_train_forward.json").read_text(encoding="utf-8"))
    tensors = ir1.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("IR1 forward parameter contract has no tensors map")
    names: list[str] = []
    for tensor_name, raw_meta in tensors.items():
        if not isinstance(raw_meta, Mapping) or raw_meta.get("kind") != "weight":
            continue
        if raw_meta.get("requires_grad") is not True:
            continue
        name = str(tensor_name)
        if not name.startswith("weight."):
            raise ValueError(f"forward weight tensor lacks weight. prefix: {name}")
        names.append(name[len("weight.") :])
    if len(names) != len(set(names)):
        raise ValueError("forward IR contains duplicate trainable names")
    return names


def _serialized_weight_snapshot(run_dir: Path, summary: Mapping[str, Any]) -> np.ndarray:
    manifest = json.loads((run_dir / "weights_manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("weights_manifest.json has no entries[]")
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in entries:
        if not isinstance(row, Mapping) or not row.get("name"):
            continue
        name = str(row["name"])
        if name in by_name:
            raise ValueError(f"duplicate manifest tensor name: {name}")
        by_name[name] = row

    slots = [
        row for row in summary.get("tensor_slots", [])
        if isinstance(row, Mapping)
        and str(row.get("name", "")).startswith("weight.")
        and str(row.get("section", "")) == "weights"
    ]
    slots.sort(key=lambda row: int(row.get("offset", 0)))
    bump = (run_dir / "weights.bump").read_bytes()
    chunks: list[np.ndarray] = []
    for slot in slots:
        name = str(slot["name"])[len("weight.") :]
        manifest_name = name if name in by_name else f"tiny.{name}"
        entry = by_name.get(manifest_name)
        if entry is None:
            raise ValueError(f"serialized weight {name!r} is absent from manifest")
        if str(entry.get("dtype", "")).lower() not in {"fp32", "f32"}:
            raise ValueError(f"serialized weight {name!r} is not fp32")
        offset = int(entry.get("offset", 0) or 0)
        numel = int(slot.get("numel", 0) or 0)
        end = offset + numel * 4
        if numel <= 0 or offset < 0 or end > len(bump):
            raise ValueError(f"serialized weight {name!r} has an invalid byte span")
        chunks.append(np.frombuffer(bump[offset:end], dtype="<f4").astype(np.float32, copy=True))
    if not chunks:
        raise ValueError("generated runtime has no serialized weight slots")
    return np.concatenate(chunks)


def _validate_parameter_inventory(
    summary: Mapping[str, Any],
    run_dir: Path,
    expected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    names = summary.get("parameter_gradient_order")
    numels = summary.get("parameter_gradient_numel")
    if not isinstance(names, list) or not isinstance(numels, list) or not names:
        raise ValueError("generated runtime has no parameter gradient inventory")
    if len(names) != len(numels):
        raise ValueError("parameter gradient name/numel inventory lengths differ")
    text_names = [str(name) for name in names]
    if len(set(text_names)) != len(text_names):
        raise ValueError("parameter gradient inventory contains duplicate names")

    expected_by_name = {str(row["name"]): row for row in expected}
    forward_names = _forward_ir_trainable_names(run_dir)
    missing_forward = sorted(set(expected_by_name) - set(forward_names))
    unexpected_forward = sorted(set(forward_names) - set(expected_by_name))
    if missing_forward or unexpected_forward:
        raise ValueError(
            "expected/forward IR parameter mismatch: "
            f"missing_expected={missing_forward}, unexpected_forward={unexpected_forward}"
        )
    missing_expected = sorted(set(expected_by_name) - set(text_names))
    unexpected_generated = sorted(set(text_names) - set(expected_by_name))
    if missing_expected or unexpected_generated:
        raise ValueError(
            "expected/generated parameter mismatch: "
            f"missing_expected={missing_expected}, unexpected_generated={unexpected_generated}"
        )

    slot_rows = summary.get("tensor_slots")
    if not isinstance(slot_rows, list):
        raise ValueError("generated runtime summary has no tensor_slots")
    grad_slots = [
        str(row.get("name"))[len("grad.weight.") :]
        for row in slot_rows
        if isinstance(row, Mapping) and str(row.get("name", "")).startswith("grad.weight.")
    ]
    if len(set(grad_slots)) != len(grad_slots):
        raise ValueError("generated runtime contains duplicate grad.weight slots")
    missing = sorted(set(text_names) - set(grad_slots))
    extra = sorted(set(grad_slots) - set(text_names))
    if missing or extra:
        raise ValueError(f"gradient inventory/slot mismatch: missing={missing}, extra={extra}")

    inventory: list[dict[str, Any]] = []
    for name, raw_numel in zip(text_names, numels):
        numel = int(raw_numel)
        expected_row = expected_by_name[name]
        manifest_name = str(expected_row["manifest_name"])
        shape = [int(value) for value in expected_row["shape"]]
        shape_numel = math.prod(shape)
        if numel <= 0 or shape_numel != numel or int(expected_row["numel"]) != numel:
            raise ValueError(
                f"parameter {name!r} shape/numel mismatch: shape={shape}, inventory={numel}"
            )
        inventory.append({"name": name, "manifest_name": manifest_name, "shape": shape, "numel": numel})
    return inventory


def _split_snapshot(flat: np.ndarray, inventory: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    expected = sum(int(row["numel"]) for row in inventory)
    if int(flat.size) != expected:
        raise ValueError(f"gradient snapshot size mismatch: got={flat.size}, expected={expected}")
    out: dict[str, np.ndarray] = {}
    cursor = 0
    for row in inventory:
        name = str(row["name"])
        numel = int(row["numel"])
        if name in out:
            raise ValueError(f"duplicate gradient snapshot name: {name}")
        out[name] = np.asarray(flat[cursor : cursor + numel], dtype=np.float32).copy()
        cursor += numel
    return out


def _activation_snapshot(lib: ctypes.CDLL, summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    numel = int(lib.ck_train_get_activation_snapshot_numel())
    buf = (ctypes.c_float * numel)()
    wrote = int(lib.ck_train_export_activation_snapshot(buf, numel))
    if wrote != numel:
        raise RuntimeError(f"activation snapshot wrote {wrote}, expected {numel}")
    flat = np.ctypeslib.as_array(buf, shape=(numel,)).astype(np.float32, copy=True)
    rows = [
        row for row in summary.get("tensor_slots", [])
        if isinstance(row, Mapping) and str(row.get("section", "")) == "activations"
    ]
    rows.sort(key=lambda row: int(row.get("offset", 0)))
    out: dict[str, np.ndarray] = {}
    cursor = 0
    for row in rows:
        name = str(row.get("name"))
        count = int(row.get("numel", 0))
        if name in out:
            raise ValueError(f"duplicate activation snapshot name: {name}")
        out[name] = flat[cursor : cursor + count].copy()
        cursor += count
    if cursor != numel:
        raise ValueError(f"activation inventory consumed {cursor}, snapshot has {numel}")
    return out


def _compare_tensor(actual: np.ndarray, expected: np.ndarray, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    expected = np.asarray(expected, dtype=np.float32).reshape(-1)
    if actual.shape != expected.shape:
        return {"passed": False, "reason": "shape_mismatch", "actual": list(actual.shape), "expected": list(expected.shape)}
    finite_actual = bool(np.isfinite(actual).all())
    finite_expected = bool(np.isfinite(expected).all())
    if not finite_actual or not finite_expected:
        return {
            "passed": False,
            "reason": "non_finite",
            "actual_finite": finite_actual,
            "expected_finite": finite_expected,
        }
    delta = np.abs(actual - expected)
    allowed = float(abs_tol) + float(rel_tol) * np.abs(expected)
    violating = delta > allowed
    violation_indices = np.flatnonzero(violating)
    max_abs = float(delta.max()) if delta.size else 0.0
    mean_abs = float(delta.mean()) if delta.size else 0.0
    ref_scale = float(np.abs(expected).max()) if expected.size else 0.0
    if delta.size:
        ratios = np.zeros_like(delta)
        np.divide(delta, allowed, out=ratios, where=allowed > 0.0)
        ratios[(allowed == 0.0) & (delta > 0.0)] = np.inf
        worst_index = int(np.argmax(ratios))
        worst_ratio = float(ratios[worst_index])
    else:
        worst_index = None
        worst_ratio = 0.0
    first_violation = int(violation_indices[0]) if violation_indices.size else None
    return {
        "passed": bool(not violation_indices.size),
        "contract": "elementwise_abs_plus_relative",
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "reference_max_abs": ref_scale,
        "abs_tolerance": float(abs_tol),
        "relative_tolerance": float(rel_tol),
        "violations": int(violation_indices.size),
        "first_violating_index": first_violation,
        "worst_index": worst_index,
        "worst_error_ratio": worst_ratio,
        "worst_actual": float(actual[worst_index]) if worst_index is not None else None,
        "worst_expected": float(expected[worst_index]) if worst_index is not None else None,
        "worst_abs_diff": float(delta[worst_index]) if worst_index is not None else None,
        "worst_allowed_error": float(allowed[worst_index]) if worst_index is not None else None,
        "numel": int(actual.size),
    }


def _contract_check(lib: ctypes.CDLL, expected: str) -> dict[str, Any]:
    lib.ck_train_get_runtime_contract_sha256.argtypes = []
    lib.ck_train_get_runtime_contract_sha256.restype = ctypes.c_char_p
    raw = lib.ck_train_get_runtime_contract_sha256()
    actual = raw.decode("ascii") if raw else ""
    return {"passed": bool(actual and actual == expected), "expected": expected, "loaded": actual}


def _build_identity_check(lib: ctypes.CDLL, expected: str) -> dict[str, Any]:
    lib.ck_train_get_build_identity_sha256.argtypes = []
    lib.ck_train_get_build_identity_sha256.restype = ctypes.c_char_p
    raw = lib.ck_train_get_build_identity_sha256()
    actual = raw.decode("ascii") if raw else ""
    return {"passed": bool(actual and actual == expected), "expected": expected, "loaded": actual}


def _probe_loaded_dependencies(
    library_path: Path,
    *,
    expected_contract: str,
    expected_build_identity: str,
    expected_engine_sha256: str,
) -> dict[str, Any]:
    lib = ctypes.CDLL(str(library_path.resolve()), mode=ctypes.RTLD_LOCAL)
    training_contract = _contract_check(lib, expected_contract)
    training_build = _build_identity_check(lib, expected_build_identity)
    training_library = _loaded_library_evidence(library_path)
    engine = _mapped_library_evidence("libckernel_engine.so")
    engine_hashes = [
        str(row.get("sha256"))
        for row in engine["identities"]
        if isinstance(row, Mapping) and row.get("sha256")
    ]
    engine["expected_sha256"] = expected_engine_sha256
    engine["loaded_sha256"] = engine_hashes
    engine["passed"] = bool(
        engine["observed_in_process_maps"]
        and len(engine["mapped_paths"]) == 1
        and engine_hashes == [expected_engine_sha256]
    )
    passed = bool(
        training_contract["passed"]
        and training_build["passed"]
        and training_library["observed_in_process_maps"]
        and engine["passed"]
    )
    return {
        "passed": passed,
        "training_contract": training_contract,
        "training_build_identity": training_build,
        "training_library": training_library,
        "engine_dependency": engine,
    }


def _run_dependency_probe(
    library_path: Path,
    output_path: Path,
    *,
    expected_contract: str,
    expected_build_identity: str,
    expected_engine_sha256: str,
    environment: Mapping[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-loaded-dependencies",
        "--probe-library", str(library_path.resolve()),
        "--probe-json-out", str(output_path.resolve()),
        "--probe-runtime-contract", expected_contract,
        "--probe-build-identity", expected_build_identity,
        "--probe-engine-sha256", expected_engine_sha256,
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=dict(environment) if environment is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if not output_path.exists():
        raise RuntimeError(
            f"dependency probe did not publish {output_path}: rc={completed.returncode} output={completed.stdout[-4000:]}"
        )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("dependency probe output is not an object")
    payload["command"] = cmd
    payload["returncode"] = int(completed.returncode)
    payload["stdout"] = completed.stdout[-4000:]
    return completed, payload


def _configure_runtime(lib: ctypes.CDLL) -> None:
    lib.ck_train_init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.ck_train_init.restype = ctypes.c_int
    lib.ck_train_set_batch_ex.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_train_set_batch_ex.restype = ctypes.c_int
    lib.ck_zero_grad.argtypes = []
    lib.ck_zero_grad.restype = None
    lib.ck_train_forward_step.argtypes = []
    lib.ck_train_forward_step.restype = ctypes.c_int
    lib.ck_train_backward_step.argtypes = []
    lib.ck_train_backward_step.restype = ctypes.c_int
    lib.ck_train_get_loss.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_train_get_loss.restype = ctypes.c_int
    lib.ck_train_get_weight_snapshot_numel.argtypes = []
    lib.ck_train_get_weight_snapshot_numel.restype = ctypes.c_int
    lib.ck_train_export_weight_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.ck_train_export_weight_snapshot.restype = ctypes.c_int
    lib.ck_train_get_activation_snapshot_numel.argtypes = []
    lib.ck_train_get_activation_snapshot_numel.restype = ctypes.c_int
    lib.ck_train_export_activation_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.ck_train_export_activation_snapshot.restype = ctypes.c_int
    lib.ck_train_get_parameter_gradient_snapshot_numel.argtypes = []
    lib.ck_train_get_parameter_gradient_snapshot_numel.restype = ctypes.c_int
    lib.ck_train_export_parameter_gradient_snapshot.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.ck_train_export_parameter_gradient_snapshot.restype = ctypes.c_int


def _execution_identity() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "run_id": str(uuid.uuid4()),
        "started_at": _utc_now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "git_commit": _git_identity().get("commit"),
        "github": {
            "sha": os.environ.get("GITHUB_SHA"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        },
        "thread_environment": {
            "CK_NUM_THREADS": os.environ.get("CK_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "cpu_affinity": affinity,
        "cpu_affinity_count": len(affinity) if affinity is not None else None,
    }


def _compile_stale_library_control(
    ck_run: Any,
    c_source: Path,
    run_dir: Path,
    *,
    seq_len: int,
) -> tuple[Path, dict[str, Any]]:
    """Build a distinct implementation artifact under the same runtime contract."""
    stale_source = run_dir / "generated_train_runtime_stale_control.c"
    stale_library = run_dir / "libtrain_stale_control.so"
    stale_source.write_text(
        c_source.read_text(encoding="utf-8")
        + "\nint ck_train_stale_control_marker(void) { return 1; }\n",
        encoding="utf-8",
    )
    identity = ck_run._compile_train_runtime_with_identity(
        stale_source,
        stale_library,
        lib_ck=ROOT / "build" / "libckernel_engine.so",
        cflags=[],
        defines={"CK_NUM_TOKENS": int(seq_len)},
    )
    return stale_library, identity


def _alternate_engine_control(engine_path: Path, run_dir: Path) -> Path:
    alternate_dir = run_dir / "alternate_engine_control"
    alternate_dir.mkdir(parents=True, exist_ok=True)
    alternate_engine = alternate_dir / engine_path.name
    shutil.copy2(engine_path, alternate_engine)
    with alternate_engine.open("ab") as handle:
        handle.write(b"\nCKE_ALTERNATE_ENGINE_NEGATIVE_CONTROL\n")
    if _sha256(alternate_engine) == _sha256(engine_path):
        raise RuntimeError("alternate engine control did not change engine identity")
    return alternate_engine


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested_coverage = [
        "two_layer_fp32_dense_transformer",
        "equal_head_attention_4q_4kv",
        "generated_forward",
        "generated_backward",
        "per_parameter_gradients",
    ]
    report: dict[str, Any] = {
        "schema": "cke.v8.training_certification.v1",
        "generated_at": _utc_now(),
        "status": "INCOMPLETE",
        "passed": False,
        "execution": _execution_identity(),
        "configuration": {
            "model": "fp32_dense_2layer",
            "layers": 2,
            "vocab": int(args.vocab),
            "d_model": int(args.d_model),
            "hidden": int(args.hidden),
            "heads": 4,
            "kv_heads": 4,
            "sequence_length": int(args.seq_len),
            "seed": int(args.seed),
            "dtype": "fp32",
        },
        "implementation": {
            "orchestrator": "version/v8/scripts/run_training_certification_v8.py",
            "shared_codegen": "version/v7/scripts/codegen_train_runtime_v7.py",
            "shared_runtime_builder": "version/v7/scripts/ck_run_v7.py:_ensure_train_runtime_artifacts",
            "oracle": "version/v7/scripts/oracle_snapshot_torch_v7.py:SnapshotQwenLikeOracle",
        },
        "coverage": {
            "requested": requested_coverage,
            "certified": [],
            "not_certified": [
                "adamw_update_and_accumulation",
                "durable_checkpoint_resume",
                "generated_inference_export",
                "unequal_head_gqa_4q_2kv",
                "non_vector_multiple_feature_tails",
                "bf16",
                "svg_learning_fixture",
            ],
        },
        "checks": {},
        "negative_controls": {},
        "failures": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        python_exec = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable
        init_cmd = [
            python_exec,
            str(V7_SCRIPTS / "ck_run_v7.py"),
            "init",
            "--run", str(args.run_dir),
            "--allow-non-cache-run-dir",
            "--train-seed", str(args.seed),
            "--layers", "2",
            "--vocab-size", str(args.vocab),
            "--embed-dim", str(args.d_model),
            "--hidden-dim", str(args.hidden),
            "--num-heads", "4",
            "--num-kv-heads", "4",
            "--context-len", str(args.seq_len),
            "--template", "qwen3",
            "--generate-ir",
            "--generate-runtime",
            "--train-bridge-lowering", "explicit",
        ]
        init = _run(init_cmd)
        report["commands"] = {"init": init_cmd, "init_rc": int(init.returncode), "init_output": init.stdout[-12000:]}
        if init.returncode != 0:
            raise RuntimeError("tiny model initialization failed")

        sys.path.insert(0, str(V7_SCRIPTS))
        ck_run = _load_module("cke_shared_ck_run_v7", V7_SCRIPTS / "ck_run_v7.py")
        oracle = _load_module("cke_shared_snapshot_oracle_v7", V7_SCRIPTS / "oracle_snapshot_torch_v7.py")
        c_source, library_path = ck_run._ensure_train_runtime_artifacts(
            args.run_dir,
            python_exec,
            False,
            runtime_defines={"CK_NUM_TOKENS": int(args.seq_len)},
            train_tokens=int(args.seq_len),
            bridge_lowering="explicit",
        )
        summary_path = args.run_dir / "generated_train_runtime_summary_v7.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_inventory, parameter_policy = _expected_trainable_parameters(args.run_dir)
        inventory = _validate_parameter_inventory(summary, args.run_dir, expected_inventory)

        build_identity_path = args.run_dir / "generated_train_build_identity_v7.json"
        build_identity = json.loads(build_identity_path.read_text(encoding="utf-8"))
        expected_build_identity = str(build_identity.get("build_identity_sha256", ""))
        if not expected_build_identity:
            raise RuntimeError("generated training build identity is missing")
        build_inputs = build_identity.get("inputs") if isinstance(build_identity.get("inputs"), Mapping) else {}
        expected_engine_sha256 = str(build_inputs.get("kernel_library_sha256", ""))
        engine_path = ROOT / "build" / "libckernel_engine.so"
        if not expected_engine_sha256 or not engine_path.is_file():
            raise RuntimeError("training build identity has no usable engine-library identity")

        dependency_probe_path = args.run_dir / "loaded_dependency_probe.json"
        dependency_process, dependency_probe = _run_dependency_probe(
            library_path,
            dependency_probe_path,
            expected_contract=str(summary.get("runtime_contract_sha256", "")),
            expected_build_identity=expected_build_identity,
            expected_engine_sha256=expected_engine_sha256,
        )
        report["checks"]["loaded_engine_dependency"] = dependency_probe
        if dependency_process.returncode != 0 or not dependency_probe.get("passed"):
            raise RuntimeError("fresh-process loaded engine dependency does not match the build identity")

        lib = ctypes.CDLL(str(library_path.resolve()), mode=ctypes.RTLD_GLOBAL)
        _configure_runtime(lib)
        lib.ck_get_num_threads.argtypes = []
        lib.ck_get_num_threads.restype = ctypes.c_int
        report["execution"]["actual_runtime_threads"] = int(lib.ck_get_num_threads())
        contract = _contract_check(lib, str(summary.get("runtime_contract_sha256", "")))
        loaded_build = _build_identity_check(lib, expected_build_identity)
        loaded_library = _loaded_library_evidence(library_path)
        report["checks"]["loaded_library_contract"] = contract
        report["checks"]["loaded_library_build_identity"] = loaded_build
        report["checks"]["loaded_library_observed"] = {
            "passed": bool(loaded_library["observed_in_process_maps"]),
            "path": loaded_library["path"],
            "process_map_entries": loaded_library["process_map_entries"],
        }
        if not contract["passed"]:
            raise RuntimeError("loaded training library contract does not match generated runtime summary")
        if not loaded_build["passed"]:
            raise RuntimeError("loaded training library build identity does not match build inputs")
        if not loaded_library["observed_in_process_maps"]:
            raise RuntimeError("loaded training library was not observed in /proc/self/maps")

        init_payload = ck_run._build_ck_runtime_init_payload(args.run_dir, summary)
        init_rc = int(lib.ck_train_init(
            init_payload["float_buffer"], init_payload["sizes_buffer"], int(init_payload["num_params"])
        ))
        if init_rc < 0:
            raise RuntimeError(f"ck_train_init failed: {init_rc}")

        weight_count = int(lib.ck_train_get_weight_snapshot_numel())
        weight_buf = (ctypes.c_float * weight_count)()
        weight_wrote = int(lib.ck_train_export_weight_snapshot(weight_buf, weight_count))
        if weight_wrote != weight_count:
            raise RuntimeError(f"weight snapshot wrote {weight_wrote}, expected {weight_count}")
        weight_snapshot = np.ctypeslib.as_array(weight_buf, shape=(weight_count,)).astype(np.float32, copy=True)
        serialized_weights = _serialized_weight_snapshot(args.run_dir, summary)
        report["checks"]["serialized_weights"] = _compare_tensor(
            weight_snapshot, serialized_weights, 0.0, 0.0
        )

        rng = np.random.default_rng(int(args.seed) + 17)
        token_values = rng.integers(0, int(args.vocab), size=int(args.seq_len) + 1, dtype=np.int32)
        x = token_values[:-1].copy()
        y = token_values[1:].copy()
        batch_path = args.run_dir / "certification_batch.json"
        _write_report(
            batch_path,
            {
                "schema": "cke.v8.training_certification_batch.v1",
                "seed": int(args.seed) + 17,
                "input_ids": [int(value) for value in x],
                "targets": [int(value) for value in y],
                "valid_tokens": int(args.seq_len),
            },
        )
        x_buf = (ctypes.c_int32 * len(x))(*[int(value) for value in x])
        y_buf = (ctypes.c_int32 * len(y))(*[int(value) for value in y])
        set_rc = int(lib.ck_train_set_batch_ex(x_buf, y_buf, int(args.seq_len)))
        if set_rc != int(args.seq_len):
            raise RuntimeError(f"ck_train_set_batch_ex returned {set_rc}")
        lib.ck_zero_grad()
        forward_calls = int(lib.ck_train_forward_step())
        if forward_calls <= 0:
            raise RuntimeError(f"generated forward returned {forward_calls}")
        activations = _activation_snapshot(lib, summary)
        logits_rows = [(name, value) for name, value in activations.items() if ".logits." in name]
        if len(logits_rows) != 1:
            raise RuntimeError(f"expected one logits activation, found {[name for name, _ in logits_rows]}")
        logits_name, ck_logits = logits_rows[0]
        backward_calls = int(lib.ck_train_backward_step())
        if backward_calls <= 0:
            raise RuntimeError(f"generated backward returned {backward_calls}")
        ck_loss_value = ctypes.c_float()
        if int(lib.ck_train_get_loss(ctypes.byref(ck_loss_value))) != 0:
            raise RuntimeError("ck_train_get_loss failed")

        grad_count = int(lib.ck_train_get_parameter_gradient_snapshot_numel())
        grad_buf = (ctypes.c_float * grad_count)()
        grad_wrote = int(lib.ck_train_export_parameter_gradient_snapshot(grad_buf, grad_count))
        if grad_wrote != grad_count:
            raise RuntimeError(f"parameter gradient snapshot wrote {grad_wrote}, expected {grad_count}")
        ck_gradients = _split_snapshot(
            np.ctypeslib.as_array(grad_buf, shape=(grad_count,)).astype(np.float32, copy=True), inventory
        )

        torch_loss, torch_logits, torch_gradients = oracle.compute_loss_logits_and_gradients_from_snapshot_array(
            args.run_dir,
            summary,
            weight_snapshot,
            x,
            y,
            parameter_names=[str(row["name"]) for row in inventory],
            valid_tokens=int(args.seq_len),
        )
        report["checks"]["parameter_inventory"] = {
            "passed": True,
            "count": len(inventory),
            "total_gradient_floats": grad_count,
            "parameters": inventory,
            "expected_source": parameter_policy,
            "agreement": {
                "expected": [str(row["name"]) for row in expected_inventory],
                "generated_inventory": [str(row["name"]) for row in inventory],
                "exported_gradients": sorted(ck_gradients),
                "pytorch_gradients": sorted(torch_gradients),
            },
        }
        report["checks"]["generated_execution"] = {
            "passed": True,
            "forward_calls": forward_calls,
            "backward_calls": backward_calls,
            "logits_slot": logits_name,
        }
        report["checks"]["loss"] = {
            "passed": bool(abs(float(ck_loss_value.value) - float(torch_loss)) <= float(args.loss_tol)),
            "cke": float(ck_loss_value.value),
            "pytorch": float(torch_loss),
            "abs_diff": abs(float(ck_loss_value.value) - float(torch_loss)),
            "tolerance": float(args.loss_tol),
        }
        report["checks"]["logits"] = _compare_tensor(
            ck_logits, torch_logits, float(args.logits_abs_tol), float(args.logits_rel_tol)
        )

        gradient_rows: list[dict[str, Any]] = []
        for row in inventory:
            name = str(row["name"])
            result = _compare_tensor(
                ck_gradients[name], torch_gradients[name], float(args.gradient_abs_tol), float(args.gradient_rel_tol)
            )
            result["name"] = name
            result["shape"] = row["shape"]
            gradient_rows.append(result)
        report["checks"]["gradients"] = {
            "passed": all(bool(row.get("passed")) for row in gradient_rows),
            "compared": len(gradient_rows),
            "failed": [row["name"] for row in gradient_rows if not row.get("passed")],
            "tensors": gradient_rows,
        }

        corrupted = {name: value.copy() for name, value in ck_gradients.items()}
        corrupt_name = str(inventory[0]["name"])
        corrupted[corrupt_name][0] += max(1.0, float(args.gradient_abs_tol) * 1000.0)
        corrupt_eval = _compare_tensor(
            corrupted[corrupt_name], torch_gradients[corrupt_name], float(args.gradient_abs_tol), float(args.gradient_rel_tol)
        )
        report["negative_controls"]["corrupted_gradient"] = {
            "passed": not bool(corrupt_eval.get("passed")),
            "injected_parameter": corrupt_name,
            "detector_result": corrupt_eval,
        }
        stale_library_path, stale_identity = _compile_stale_library_control(
            ck_run, c_source, args.run_dir, seq_len=int(args.seq_len)
        )
        stale_lib = ctypes.CDLL(str(stale_library_path.resolve()), mode=ctypes.RTLD_LOCAL)
        stale_contract = _contract_check(stale_lib, contract["expected"])
        stale_eval = _build_identity_check(stale_lib, expected_build_identity)
        stale_library_evidence = _loaded_library_evidence(stale_library_path)
        report["negative_controls"]["stale_library"] = {
            "passed": bool(
                stale_contract.get("passed")
                and not stale_eval.get("passed")
                and stale_library_evidence.get("observed_in_process_maps")
                and stale_identity.get("output_library_sha256") != loaded_library.get("sha256")
            ),
            "same_runtime_contract": stale_contract,
            "build_identity_detector": stale_eval,
            "substituted_library": stale_library_evidence,
            "substituted_build_identity": stale_identity,
        }
        alternate_engine_path = _alternate_engine_control(engine_path, args.run_dir)
        alternate_probe_path = args.run_dir / "alternate_engine_dependency_probe.json"
        alternate_environment = dict(os.environ)
        existing_library_path = alternate_environment.get("LD_LIBRARY_PATH", "")
        alternate_environment["LD_LIBRARY_PATH"] = (
            str(alternate_engine_path.parent)
            + (os.pathsep + existing_library_path if existing_library_path else "")
        )
        alternate_environment.pop("LD_PRELOAD", None)
        alternate_process, alternate_probe = _run_dependency_probe(
            library_path,
            alternate_probe_path,
            expected_contract=contract["expected"],
            expected_build_identity=expected_build_identity,
            expected_engine_sha256=expected_engine_sha256,
            environment=alternate_environment,
        )
        alternate_engine = alternate_probe.get("engine_dependency")
        alternate_engine = alternate_engine if isinstance(alternate_engine, Mapping) else {}
        mapped_paths = [str(path) for path in alternate_engine.get("mapped_paths", [])]
        report["negative_controls"]["alternate_engine"] = {
            "passed": bool(
                alternate_process.returncode != 0
                and not alternate_probe.get("passed")
                and alternate_probe.get("training_contract", {}).get("passed")
                and alternate_probe.get("training_build_identity", {}).get("passed")
                and not alternate_engine.get("passed")
                and str(alternate_engine_path.resolve()) in mapped_paths
            ),
            "injected_engine": _file_identity(alternate_engine_path),
            "expected_engine": _file_identity(engine_path),
            "detector_result": alternate_probe,
        }

        report["provenance"] = {
            "git": _git_identity(),
            "compiler": _compiler_identity(),
            "manifest": _file_identity(args.run_dir / "weights_manifest.json"),
            "weights": _file_identity(args.run_dir / "weights.bump"),
            "batch": _file_identity(batch_path),
            "ir1": _file_identity(args.run_dir / "ir1_train_forward.json"),
            "ir2": _file_identity(args.run_dir / "ir2_train_backward.json"),
            "layout": _file_identity(args.run_dir / "layout_train.json"),
            "exec_plan": _file_identity(args.run_dir / "train_exec_plan.json"),
            "generated_source": _file_identity(c_source),
            "runtime_summary": _file_identity(summary_path),
            "build_identity": build_identity,
            "loaded_library": loaded_library,
            "loaded_engine_dependency": dependency_probe.get("engine_dependency"),
            "runtime_contract_sha256": contract["loaded"],
            "build_identity_sha256": loaded_build["loaded"],
            "backward_ops": summary.get("backward_op_trace", []),
        }
        all_checks = [bool(row.get("passed")) for row in report["checks"].values() if isinstance(row, Mapping)]
        all_controls = [bool(row.get("passed")) for row in report["negative_controls"].values() if isinstance(row, Mapping)]
        report["passed"] = bool(all_checks and all(all_checks) and all_controls and all(all_controls))
        report["status"] = "PASS" if report["passed"] else "FAIL"
        if report["passed"]:
            report["coverage"]["certified"] = list(requested_coverage)
        if not report["passed"]:
            report["failures"] = [
                name for name, row in report["checks"].items()
                if isinstance(row, Mapping) and not bool(row.get("passed"))
            ] + [
                f"negative_control:{name}" for name, row in report["negative_controls"].items()
                if isinstance(row, Mapping) and not bool(row.get("passed"))
            ]
    except Exception as exc:
        report["status"] = "FAIL"
        report["passed"] = False
        report["failures"].append(str(exc))
        report["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        report["completed_at"] = _utc_now()
        report["execution"]["completed_at"] = report["completed_at"]
        _write_report(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Certify generated v8 FP32 forward/backward against PyTorch")
    parser.add_argument("--probe-loaded-dependencies", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-library", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--probe-json-out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--probe-runtime-contract", help=argparse.SUPPRESS)
    parser.add_argument("--probe-build-identity", help=argparse.SUPPRESS)
    parser.add_argument("--probe-engine-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--json-out", dest="report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--loss-tol", type=float, default=2e-5)
    parser.add_argument("--logits-abs-tol", type=float, default=2e-4)
    parser.add_argument("--logits-rel-tol", type=float, default=2e-4)
    parser.add_argument("--gradient-abs-tol", type=float, default=3e-4)
    parser.add_argument("--gradient-rel-tol", type=float, default=3e-3)
    args = parser.parse_args()
    if args.probe_loaded_dependencies:
        required = {
            "probe_library": args.probe_library,
            "probe_json_out": args.probe_json_out,
            "probe_runtime_contract": args.probe_runtime_contract,
            "probe_build_identity": args.probe_build_identity,
            "probe_engine_sha256": args.probe_engine_sha256,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"dependency probe is missing: {', '.join(missing)}")
        probe = _probe_loaded_dependencies(
            args.probe_library.expanduser().resolve(),
            expected_contract=str(args.probe_runtime_contract),
            expected_build_identity=str(args.probe_build_identity),
            expected_engine_sha256=str(args.probe_engine_sha256),
        )
        _write_report(args.probe_json_out.expanduser().resolve(), probe)
        return 0 if probe["passed"] else 1
    args.run_dir = args.run_dir.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    report = run(args)
    print(json.dumps({"status": report["status"], "passed": report["passed"], "report": str(args.report)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
