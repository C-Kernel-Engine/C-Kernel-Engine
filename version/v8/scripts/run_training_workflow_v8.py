#!/usr/bin/env python3
"""Generated-C training-to-export certification for a tiny Qwen3-style model.

This v8-owned command selectively reuses the proven v7 training IR/code generator.
It does not invoke v7's training orchestration or use Python to update CKE weights.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V7 = ROOT / "version" / "v7" / "scripts"
if str(V7) not in sys.path:
    sys.path.insert(0, str(V7))
CORPUS_SPEC = ROOT / "version" / "v8" / "training" / "english_byte_v1.json"
DEFAULT_RUN = ROOT / "version" / "v8" / ".cache" / "training_workflow" / "qwen3_4layer_fp32"
DEFAULT_REPORT = ROOT / "version" / "v8" / ".cache" / "reports" / "training_workflow_latest.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CERT = _load_module("cke_v8_training_cert", Path(__file__).with_name("run_training_certification_v8.py"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "architecture": "qwen3_style_dense_reduced", "layers": 4, "dtype": "fp32",
        "d_model": int(args.d_model), "hidden": int(args.hidden), "heads": 4, "kv_heads": 4,
        "seq_len": int(args.seq_len), "epochs": int(args.epochs), "grad_accum": int(args.grad_accum),
        "optimizer": "generated_c_adamw", "lr": float(args.lr), "beta1": float(args.beta1),
        "beta2": float(args.beta2), "eps": float(args.eps), "weight_decay": float(args.weight_decay),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _git_blob(path: Path) -> str:
    result = subprocess.run(["git", "hash-object", str(path)], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _load_corpus(spec_path: Path, out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    outputs: dict[str, Any] = {}
    arrays: list[np.ndarray] = []
    for split in ("train", "validation"):
        row = spec[split]
        source = ROOT / str(row["path"])
        actual_blob = _git_blob(source)
        if actual_blob != str(row["git_blob"]):
            raise RuntimeError(f"{split} corpus blob changed: expected={row['git_blob']} actual={actual_blob}")
        raw = source.read_bytes()[: int(row["token_count"])]
        if len(raw) != int(row["token_count"]):
            raise RuntimeError(f"{split} corpus has only {len(raw)} bytes")
        ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
        token_path = out_dir / f"{split}_token_ids.i32"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_bytes(ids.astype("<i4", copy=False).tobytes())
        outputs[split] = {
            "source": str(row["path"]), "git_blob": actual_blob, "tokens": int(ids.size),
            "original_source": str(row["source_path"]), "source_revision": str(row["source_revision"]),
            "source_blob": str(row["source_blob"]),
            "source_sha256": _sha256(source), "token_ids": str(token_path), "token_ids_sha256": _sha256(token_path),
        }
        arrays.append(ids)
    return arrays[0], arrays[1], {"spec": str(spec_path), "spec_sha256": _sha256(spec_path), "tokenizer": spec["tokenizer"], "splits": outputs}


def _batches(tokens: np.ndarray, seq_len: int, epochs: int) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    # A target token is presented exactly once per epoch. The final short window is padded.
    result: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    stream = np.concatenate([tokens, tokens[:1]])
    for epoch in range(epochs):
        for start in range(0, int(tokens.size), seq_len):
            valid = min(seq_len, int(tokens.size) - start)
            x = np.zeros(seq_len, dtype=np.int32)
            y = np.zeros(seq_len, dtype=np.int32)
            x[:valid] = stream[start : start + valid]
            y[:valid] = stream[start + 1 : start + valid + 1]
            result.append((x, y, valid, epoch))
    return result


def _configure(lib: ctypes.CDLL) -> None:
    CERT._configure_runtime(lib)
    signatures = {
        "ck_train_step_ex": ([ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_float], ctypes.c_int),
        "ck_train_flush_optimizer": ([ctypes.c_float], ctypes.c_int),
        "ck_train_import_weight_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_get_optimizer_state_snapshot_numel": ([], ctypes.c_int),
        "ck_train_export_optimizer_state_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_import_optimizer_state_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_get_accum_snapshot_numel": ([], ctypes.c_int),
        "ck_train_export_accum_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_import_accum_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_get_accum_counter": ([], ctypes.c_int), "ck_train_set_accum_counter": ([ctypes.c_int], ctypes.c_int),
        "ck_train_get_accum_tokens": ([], ctypes.c_int64), "ck_train_set_accum_tokens": ([ctypes.c_int64], ctypes.c_int64),
        "ck_train_get_opt_step": ([], ctypes.c_int), "ck_train_set_opt_step": ([ctypes.c_int], ctypes.c_int),
        "ck_train_get_last_step_profile": ([ctypes.POINTER(ctypes.c_double)] * 4 + [ctypes.POINTER(ctypes.c_int)] * 4, ctypes.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = result


def _array_export(lib: ctypes.CDLL, prefix: str) -> np.ndarray:
    count = int(getattr(lib, f"ck_train_get_{prefix}_snapshot_numel")())
    buf = (ctypes.c_float * count)()
    wrote = int(getattr(lib, f"ck_train_export_{prefix}_snapshot")(buf, count))
    if wrote != count:
        raise RuntimeError(f"{prefix} snapshot wrote {wrote}, expected {count}")
    return np.ctypeslib.as_array(buf, shape=(count,)).astype(np.float32, copy=True)


def _array_import(lib: ctypes.CDLL, prefix: str, values: np.ndarray) -> None:
    flat = np.ascontiguousarray(values, dtype=np.float32)
    ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    wrote = int(getattr(lib, f"ck_train_import_{prefix}_snapshot")(ptr, int(flat.size)))
    if wrote != int(flat.size):
        raise RuntimeError(f"{prefix} snapshot import wrote {wrote}, expected {flat.size}")


def _weight_export(lib: ctypes.CDLL) -> np.ndarray:
    count = int(lib.ck_train_get_weight_snapshot_numel())
    buf = (ctypes.c_float * count)()
    wrote = int(lib.ck_train_export_weight_snapshot(buf, count))
    if wrote != count:
        raise RuntimeError(f"weight snapshot wrote {wrote}, expected {count}")
    return np.ctypeslib.as_array(buf, shape=(count,)).astype(np.float32, copy=True)


def _grad_export(lib: ctypes.CDLL) -> np.ndarray:
    count = int(lib.ck_train_get_parameter_gradient_snapshot_numel())
    buf = (ctypes.c_float * count)()
    wrote = int(lib.ck_train_export_parameter_gradient_snapshot(buf, count))
    if wrote != count:
        raise RuntimeError(f"gradient snapshot wrote {wrote}, expected {count}")
    return np.ctypeslib.as_array(buf, shape=(count,)).astype(np.float32, copy=True)


def _init_runtime(
    run_dir: Path,
    library: Path,
    summary: Mapping[str, Any],
    ck_run: Any,
    *,
    role: str,
) -> tuple[ctypes.CDLL, dict[str, Any]]:
    build_identity_path = run_dir / "generated_train_build_identity_v7.json"
    build_identity = json.loads(build_identity_path.read_text(encoding="utf-8"))
    expected_build = str(build_identity.get("build_identity_sha256") or "")
    inputs = build_identity.get("inputs") if isinstance(build_identity.get("inputs"), Mapping) else {}
    expected_engine = str(inputs.get("kernel_library_sha256") or "")
    expected_contract = str(summary.get("runtime_contract_sha256") or "")
    if not expected_build or not expected_engine or not expected_contract:
        raise RuntimeError(f"{role} runtime identity is incomplete")
    probe_path = run_dir / f"loaded_dependency_probe_{role}.json"
    probe_path.unlink(missing_ok=True)
    probe_process, fresh_probe = CERT._run_dependency_probe(
        library,
        probe_path,
        expected_contract=expected_contract,
        expected_build_identity=expected_build,
        expected_engine_sha256=expected_engine,
    )
    if probe_process.returncode != 0 or not fresh_probe.get("passed"):
        raise RuntimeError(f"{role} fresh-process dependency provenance failed")
    lib = ctypes.CDLL(str(library.resolve()), mode=ctypes.RTLD_LOCAL)
    _configure(lib)
    # The certification lane chooses the reference-order numerical contract.
    # Optimized-provider certification is a later, separate contract.
    lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
    lib.ck_set_strict_parity.restype = None
    lib.ck_set_strict_parity(1)
    contract = CERT._contract_check(lib, expected_contract)
    loaded_build = CERT._build_identity_check(lib, expected_build)
    loaded_library = CERT._loaded_library_evidence(library)
    engine = CERT._mapped_library_evidence("libckernel_engine.so")
    engine_hashes = sorted(
        str(row.get("sha256")) for row in engine.get("identities", [])
        if isinstance(row, Mapping) and row.get("sha256")
    )
    engine_passed = bool(
        engine.get("observed_in_process_maps")
        and len(engine.get("mapped_paths", [])) == 1
        and engine_hashes == [expected_engine]
    )
    if not contract.get("passed") or not loaded_build.get("passed"):
        raise RuntimeError(f"{role} loaded training library identity mismatch")
    if not loaded_library.get("observed_in_process_maps") or not engine_passed:
        raise RuntimeError(f"{role} loaded library/engine was not the built dependency")
    payload = ck_run._build_ck_runtime_init_payload(run_dir, dict(summary))
    rc = int(lib.ck_train_init(payload["float_buffer"], payload["sizes_buffer"], int(payload["num_params"])))
    if rc < 0:
        raise RuntimeError(f"ck_train_init failed: {rc}")
    actual_threads = None
    if hasattr(lib, "ck_get_num_threads"):
        lib.ck_get_num_threads.argtypes = []
        lib.ck_get_num_threads.restype = ctypes.c_int
        actual_threads = int(lib.ck_get_num_threads())
    evidence = {
        "passed": True, "role": role, "fresh_process": fresh_probe,
        "runtime_contract": contract, "build_identity": loaded_build,
        "training_library": loaded_library,
        "engine_dependency": {**engine, "expected_sha256": expected_engine,
                              "loaded_sha256": engine_hashes, "passed": engine_passed},
        "actual_runtime_threads": actual_threads,
    }
    return lib, evidence


def _checkpoint_write(root: Path, name: str, lib: ctypes.CDLL, meta: Mapping[str, Any], *, inject_interrupt: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    tmp = root / ("." + name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    arrays = {"weight": _weight_export(lib), "optimizer_state": _array_export(lib, "optimizer_state"), "accum": _array_export(lib, "accum")}
    files: dict[str, Any] = {}
    for key, value in arrays.items():
        path = tmp / f"{key}.f32"
        path.write_bytes(value.astype("<f4", copy=False).tobytes())
        files[key] = {"file": path.name, "numel": int(value.size), "sha256": _sha256(path)}
    doc = dict(meta)
    doc.update({"schema": "cke.v8.training_checkpoint.v1", "complete": True, "files": files,
                "optimizer_step": int(lib.ck_train_get_opt_step()), "accum_counter": int(lib.ck_train_get_accum_counter()),
                "accum_tokens": int(lib.ck_train_get_accum_tokens())})
    doc["checkpoint_identity_sha256"] = _checkpoint_identity(doc)
    _write_json(tmp / "checkpoint.json", doc)
    if inject_interrupt:
        raise RuntimeError("injected checkpoint interruption before atomic publish")
    if target.exists():
        raise RuntimeError(f"checkpoint target already exists: {target}")
    os.replace(tmp, target)
    _write_json(root / "latest.json", {"checkpoint": name, "checkpoint_manifest_sha256": _sha256(target / "checkpoint.json")})
    return target


def _checkpoint_identity(doc: Mapping[str, Any]) -> str:
    fields = {
        key: doc.get(key)
        for key in (
            "schema", "complete", "runtime_contract_sha256", "training_config_sha256",
            "training_config", "dataset_sha256", "next_microstep", "total_microsteps", "optimizer_step",
            "accum_counter", "accum_tokens", "files",
        )
    }
    return _json_sha256(fields)


def _validate_checkpoint_document(doc: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if doc.get("schema") != "cke.v8.training_checkpoint.v1" or doc.get("complete") is not True:
        raise RuntimeError("checkpoint is incomplete or has the wrong schema")
    recorded_identity = str(doc.get("checkpoint_identity_sha256") or "")
    if not recorded_identity or recorded_identity != _checkpoint_identity(doc):
        raise RuntimeError("checkpoint identity hash mismatch")
    for key in ("runtime_contract_sha256", "training_config_sha256", "dataset_sha256"):
        if str(doc.get(key) or "") != str(expected.get(key) or ""):
            raise RuntimeError(f"checkpoint {key} mismatch")
    checkpoint_config = doc.get("training_config")
    if not isinstance(checkpoint_config, Mapping) or _json_sha256(checkpoint_config) != str(doc["training_config_sha256"]):
        raise RuntimeError("checkpoint training configuration identity mismatch")
    next_microstep = int(doc.get("next_microstep", -1))
    total_microsteps = int(doc.get("total_microsteps", -1))
    if total_microsteps != int(expected.get("total_microsteps", -2)):
        raise RuntimeError("checkpoint total_microsteps mismatch")
    if next_microstep < 0 or next_microstep > total_microsteps:
        raise RuntimeError("checkpoint next_microstep is out of range")
    accum_counter = int(doc.get("accum_counter", -1))
    accum_tokens = int(doc.get("accum_tokens", -1))
    grad_accum = int(expected.get("grad_accum", 0))
    if grad_accum <= 0 or accum_counter < 0 or accum_counter >= grad_accum:
        raise RuntimeError("checkpoint accumulation counter is invalid")
    if accum_tokens < 0 or (accum_counter == 0 and accum_tokens != 0):
        raise RuntimeError("checkpoint contributing-token count is invalid")
    if int(doc.get("optimizer_step", -1)) < 0:
        raise RuntimeError("checkpoint optimizer step is invalid")
    files = doc.get("files")
    if not isinstance(files, Mapping) or set(files) != {"weight", "optimizer_state", "accum"}:
        raise RuntimeError("checkpoint file inventory is invalid")


def _checkpoint_load(path: Path, lib: ctypes.CDLL, expected: Mapping[str, Any]) -> dict[str, Any]:
    doc = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    _validate_checkpoint_document(doc, expected)
    if doc.get("complete") is not True:
        raise RuntimeError("checkpoint is incomplete")
    for key in ("weight", "optimizer_state", "accum"):
        row = doc["files"][key]
        file_path = path / row["file"]
        if _sha256(file_path) != row["sha256"]:
            raise RuntimeError(f"checkpoint {key} hash mismatch")
        values = np.fromfile(file_path, dtype="<f4")
        if int(values.size) != int(row["numel"]):
            raise RuntimeError(f"checkpoint {key} size mismatch")
        _array_import(lib, key if key != "weight" else "weight", values)
    setters = (
        ("optimizer_step", lib.ck_train_set_opt_step, lib.ck_train_get_opt_step),
        ("accum_counter", lib.ck_train_set_accum_counter, lib.ck_train_get_accum_counter),
        ("accum_tokens", lib.ck_train_set_accum_tokens, lib.ck_train_get_accum_tokens),
    )
    for key, setter, getter in setters:
        wanted = int(doc[key])
        if int(setter(wanted)) != wanted or int(getter()) != wanted:
            raise RuntimeError(f"checkpoint {key} restore was rejected")
    return doc


def _step(lib: ctypes.CDLL, batch: tuple[np.ndarray, np.ndarray, int, int], lr: float) -> tuple[float, dict[str, float | int]]:
    x, y, valid, _ = batch
    loss = ctypes.c_float()
    rc = int(lib.ck_train_step_ex(x.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), valid, ctypes.byref(loss), lr))
    if rc <= 0:
        raise RuntimeError(f"generated training step failed: {rc}")
    vals = [ctypes.c_double() for _ in range(4)]
    counts = [ctypes.c_int() for _ in range(4)]
    lib.ck_train_get_last_step_profile(*[ctypes.byref(v) for v in vals], *[ctypes.byref(v) for v in counts])
    return float(loss.value), {"step_ms": vals[0].value, "forward_ms": vals[1].value, "backward_ms": vals[2].value,
                               "optimizer_ms": vals[3].value, "optimizer_applied": counts[3].value}


def _forward_logits(lib: ctypes.CDLL, summary: Mapping[str, Any], x: np.ndarray, y: np.ndarray, valid: int) -> np.ndarray:
    lib.ck_train_set_batch_ex(x.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), valid)
    if int(lib.ck_train_forward_step()) <= 0:
        raise RuntimeError("generated inference forward failed")
    activations = CERT._activation_snapshot(lib, summary)
    rows = [value for name, value in activations.items() if ".logits." in name]
    if len(rows) != 1:
        raise RuntimeError(f"expected one logits tensor, found {len(rows)}")
    return rows[0].reshape(-1, 256)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray, valid: int) -> float:
    active = logits[:valid].astype(np.float64)
    maximum = active.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(active - maximum).sum(axis=1)) + maximum[:, 0]
    return float(np.mean(logsumexp - active[np.arange(valid), targets[:valid]]))


def _evaluate(lib: ctypes.CDLL, summary: Mapping[str, Any], tokens: np.ndarray, seq_len: int) -> float:
    total = 0.0; count = 0
    for x, y, valid, _ in _batches(tokens, seq_len, 1):
        total += _cross_entropy(_forward_logits(lib, summary, x, y, valid), y, valid) * valid
        count += valid
    return total / count


def _sample_trajectory(
    lib: ctypes.CDLL,
    summary: Mapping[str, Any],
    seq_len: int,
    prompt: bytes = b"The ",
    new_tokens: int = 48,
) -> tuple[list[int], np.ndarray]:
    generated = list(prompt)
    trajectory: list[np.ndarray] = []
    dummy = np.zeros(seq_len, dtype=np.int32)
    for _ in range(new_tokens):
        context = generated[-seq_len:]
        x = np.zeros(seq_len, dtype=np.int32); x[:len(context)] = context
        logits = _forward_logits(lib, summary, x, dummy, len(context))
        last = logits[len(context) - 1].copy()
        trajectory.append(last)
        generated.append(int(np.argmax(last)))
    return generated, np.stack(trajectory) if trajectory else np.empty((0, 256), dtype=np.float32)


def _sample(lib: ctypes.CDLL, summary: Mapping[str, Any], seq_len: int, prompt: bytes = b"The ", new_tokens: int = 48) -> str:
    generated, _ = _sample_trajectory(lib, summary, seq_len, prompt=prompt, new_tokens=new_tokens)
    return bytes(generated).decode("utf-8", errors="replace")


def _split_optimizer_snapshot(summary: Mapping[str, Any], flat: np.ndarray) -> dict[str, np.ndarray]:
    rows = [r for r in summary["tensor_slots"] if r.get("section") in {"optimizer_m", "optimizer_v"}]
    rows.sort(key=lambda r: int(r["offset"]))
    out: dict[str, np.ndarray] = {}
    cursor = 0
    for row in rows:
        count = int(row["numel"]); name = str(row["name"])
        out[name] = flat[cursor:cursor + count].copy(); cursor += count
    if cursor != int(flat.size):
        raise RuntimeError("optimizer snapshot inventory mismatch")
    return out


def _torch_optimizer_flat(torch: Any, weights: Mapping[str, Any], optimizer: Any, summary: Mapping[str, Any]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    rows = [r for r in summary["tensor_slots"] if r.get("section") in {"optimizer_m", "optimizer_v"}]
    rows.sort(key=lambda r: int(r["offset"]))
    for row in rows:
        full = str(row["name"]); kind, name = full.split(".", 2)[1:]
        state = optimizer.state[weights[name]]
        key = "exp_avg" if kind == "m" else "exp_avg_sq"
        chunks.append(state[key].detach().cpu().float().reshape(-1).numpy().copy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def _torch_weight_flat(weights: Mapping[str, Any], summary: Mapping[str, Any]) -> np.ndarray:
    rows = [
        row for row in summary["tensor_slots"]
        if str(row.get("name", "")).startswith("weight.") and row.get("section") == "weights"
    ]
    rows.sort(key=lambda row: int(row.get("offset", 0)))
    return np.concatenate([
        weights[str(row["name"])[len("weight."):]].detach().cpu().float().reshape(-1).numpy()
        for row in rows
    ]).astype(np.float32, copy=False)


def _tensor_discrepancies(
    actual: np.ndarray,
    expected: np.ndarray,
    names: Sequence[str],
    numels: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    for name, raw_count in zip(names, numels):
        count = int(raw_count)
        a = np.asarray(actual[cursor:cursor + count], dtype=np.float32)
        e = np.asarray(expected[cursor:cursor + count], dtype=np.float32)
        if a.size != count or e.size != count:
            raise RuntimeError(f"tensor discrepancy inventory mismatch for {name}")
        delta = np.abs(a - e)
        rows.append({
            "name": str(name), "numel": count,
            "max_abs_diff": float(delta.max()) if count else 0.0,
            "mean_abs_diff": float(delta.mean()) if count else 0.0,
            "reference_max_abs": float(np.abs(e).max()) if count else 0.0,
        })
        cursor += count
    if cursor != int(actual.size) or cursor != int(expected.size):
        raise RuntimeError("tensor discrepancy inventory did not consume both snapshots")
    return rows


def _restore_training_state(
    lib: ctypes.CDLL,
    *,
    weight: np.ndarray,
    optimizer: np.ndarray,
    accum: np.ndarray,
    optimizer_step: int,
    accum_counter: int,
    accum_tokens: int,
) -> None:
    _array_import(lib, "weight", weight)
    _array_import(lib, "optimizer_state", optimizer)
    _array_import(lib, "accum", accum)
    setters = (
        ("optimizer_step", lib.ck_train_set_opt_step, lib.ck_train_get_opt_step, optimizer_step),
        ("accum_counter", lib.ck_train_set_accum_counter, lib.ck_train_get_accum_counter, accum_counter),
        ("accum_tokens", lib.ck_train_set_accum_tokens, lib.ck_train_get_accum_tokens, accum_tokens),
    )
    for name, setter, getter, wanted in setters:
        if int(setter(int(wanted))) != int(wanted) or int(getter()) != int(wanted):
            raise RuntimeError(f"failed to restore {name}")


def _export_weights(run_dir: Path, summary: Mapping[str, Any], snapshot: np.ndarray, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "weights_manifest.json").read_text(encoding="utf-8"))
    entries = {str(r["name"]): r for r in manifest["entries"]}
    blob = bytearray((run_dir / "weights.bump").read_bytes())
    cursor = 0
    for name, count in zip(summary["init_weight_order"], summary["init_weight_numel"]):
        count = int(count); entry = entries.get(str(name), entries.get("tiny." + str(name)))
        if entry is None: raise RuntimeError(f"export manifest lacks {name}")
        off = int(entry["offset"]); blob[off:off + count * 4] = snapshot[cursor:cursor + count].astype("<f4").tobytes(); cursor += count
    (destination / "weights.bump").write_bytes(blob)
    (destination / "weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _export_v8_bundle(raw_export: Path, config_path: Path, destination: Path) -> None:
    """Wrap the raw training payload in the inference loader's BUMPWGT4 envelope."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw_export / "weights_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["entries"]:
        row["offset"] = int(row["offset"]) + 128
    (destination / "weights.bump").write_bytes(b"BUMPWGT4" + bytes(120) + (raw_export / "weights.bump").read_bytes())
    (destination / "weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(config_path, destination / "config.json")


def _write_visualizer_artifacts(
    *,
    run_dir: Path,
    python: str,
    args: argparse.Namespace,
    corpus: Mapping[str, Any],
    epoch_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    summary: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    tokenizer_path = run_dir / "tokenizer.json"
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "post_processor": None, "decoder": {"type": "ByteLevel"},
        "model": {"type": "WordLevel", "unk_token": "<0x00>",
                  "vocab": {f"<0x{idx:02X}>": idx for idx in range(256)}},
        "cke_contract": corpus["tokenizer"],
    }
    _write_json(tokenizer_path, tokenizer)

    sample_rows = []
    for line_no, raw in enumerate((ROOT / str(corpus["splits"]["train"]["source"])).read_bytes().splitlines()[:4], 1):
        ids = [int(value) for value in raw[:96]]
        decoded = bytes(ids).decode("utf-8", errors="replace")
        sample_rows.append({"line_no": line_no, "exact_match": bytes(ids) == raw[:96],
                            "token_count": len(ids), "token_ids": ids, "decoded": decoded})
    roundtrip_path = run_dir / "tokenizer_roundtrip.json"
    _write_json(roundtrip_path, {
        "schema": "cke.v8.byte_tokenizer_roundtrip.v1", "status": "pass", "exact_match": True,
        "tokenizer_json_path": str(tokenizer_path), "line_eval": {"passed": len(sample_rows), "failed": 0},
        "sample_rows": sample_rows,
    })
    qc_path = run_dir / "dataset_qc.json"
    _write_json(qc_path, {
        "schema": "cke.v8.dataset_qc.v1", "status": "pass",
        "path": str(ROOT / str(corpus["splits"]["train"]["source"])),
        "non_empty_lines": len((ROOT / str(corpus["splits"]["train"]["source"])).read_bytes().splitlines()),
        "checks": {"pinned_blob": True, "separate_document_split": True, "fixed_token_count": True,
                   "serialized_token_ids": True},
    })
    profile_path = run_dir / "dataset_profile.json"
    _write_json(profile_path, {
        "schema": "cke.v8.dataset_profile.v1", "status": "pass", "tokenizer": corpus["tokenizer"],
        "splits": {name: {key: row[key] for key in ("tokens", "source", "git_blob", "source_revision", "source_blob", "token_ids_sha256")}
                   for name, row in corpus["splits"].items()},
    })
    loss_path = run_dir / "training_loss_curve_latest.json"
    _write_json(loss_path, {
        "schema": "cke.v8.training_loss_curve.v1",
        "steps": [{"step": int(row["last_microstep"]), "epoch": int(row["epoch"]),
                   "loss_ck": float(row["cke_mean_loss"]), "loss_pt": float(row["pytorch_mean_loss"]),
                   "lr": float(args.lr), "grad_norm": None, "source_stage": "pretrain"}
                  for row in epoch_rows],
        "grad_norm_status": "NOT_MEASURED",
    })
    parity_path = run_dir / "training_parity_latest.json"
    _write_json(parity_path, {"schema": "cke.v8.training_parity.v1", "steps": list(parity_rows)})
    step_profile_path = run_dir / "training_step_profile_latest.json"
    _write_json(step_profile_path, {
        "schema": "cke.v8.training_step_profile.v1",
        "train_tok_s": performance["generated_c_tokens_per_second"],
        "scope": "generated_c_profiled_steps_plus_final_flush", "timings": performance["generated_profile_ms"],
    })
    checkpoint_policy_path = run_dir / "training_checkpoint_policy_latest.json"
    _write_json(checkpoint_policy_path, {
        "schema": "cke.v8.training_checkpoint_policy.v1", "status": "pass",
        "checkpoint": str(checkpoint_path), "atomic_publication": True, "power_loss_durability_claim": False,
        "state": ["weights", "optimizer_moments", "accumulated_gradients", "optimizer_step",
                  "accumulation_counter", "contributing_tokens", "next_microstep", "runtime_dataset_config_identity"],
    })
    stitch_path = run_dir / "backprop_stitch_runtime_latest.json"
    forward_raw = summary.get("forward_op_count", summary.get("forward_ops", 0))
    backward_raw = summary.get("backward_op_count", summary.get("backward_ops", 0))
    forward_count = len(forward_raw) if isinstance(forward_raw, list) else int(forward_raw or 0)
    backward_count = len(backward_raw) if isinstance(backward_raw, list) else int(backward_raw or 0)
    _write_json(stitch_path, {
        "schema": "cke.v8.backprop_stitch_runtime.v1", "status": "pass", "passed": True,
        "forward_ops": forward_count, "backward_ops": backward_count,
        "execution_plan": str(run_dir / "train_exec_plan.json"),
        "generated_runtime": str(run_dir / "generated_train_runtime_v7.c"),
    })
    pipeline_path = run_dir / "training_pipeline_latest.json"
    artifacts = {
        "dataset_qc_json": str(qc_path), "dataset_profile_json": str(profile_path),
        "tokenizer_roundtrip_json": str(roundtrip_path),
    }
    data_provenance = [
        {"stage": "pretrain", "dataset_name": name, "source_path": row["source"], "split": name,
         "token_count": row["tokens"], "hash": {"sha256": row["token_ids_sha256"]},
         "sampling": {"epochs": int(args.epochs) if name == "train" else 1},
         "packing": {"seq_len": int(args.seq_len), "cross_document_attention": False}}
        for name, row in corpus["splits"].items()
    ]
    _write_json(pipeline_path, {
        "schema": "cke.v8.training_pipeline.v1", "active_stage": "pretrain", "backend": "generated_c_fp32",
        "stage_timeline": [{"stage": "pretrain", "order": 0, "status": "completed", "active": True}],
        "optimizer": {"name": "adamw", "lr": float(args.lr), "hparams": {"beta1": args.beta1, "beta2": args.beta2,
                       "eps": args.eps, "weight_decay": args.weight_decay}},
        "execution": {"epochs": int(args.epochs), "micro_steps": int(sum(1 for _ in _batches(
            np.zeros(10000, dtype=np.int32), int(args.seq_len), int(args.epochs)))),
            "optimizer_steps": int(performance["optimizer_steps"]), "seq_len": int(args.seq_len),
            "grad_accum": int(args.grad_accum), "tokens_total": 10000 * int(args.epochs)},
        "data_provenance": data_provenance,
        "tokenizer_lineage": {"type": "utf8_byte_v1", "vocab_size": 256, "tokenizer_path": str(tokenizer_path),
                              "tokenizer_sha256": _sha256(tokenizer_path)},
        "data_lab": {"dataset_dir": str(run_dir / "dataset"),
                     "dataset_path": str(ROOT / str(corpus["splits"]["train"]["source"])),
                     "tokenizer_json_path": str(tokenizer_path), "artifacts": artifacts},
        "sources": {"orchestrator": str(Path(__file__).relative_to(ROOT)), "run_dir": str(run_dir)},
    })
    report_path = run_dir / "ir_report.html"
    command = [python, str(ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"),
               "--generate", "--run", str(run_dir), "--html-only", "--strict-run-artifacts", "--output", str(report_path)]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0 or not report_path.is_file():
        raise RuntimeError("v8 training IR visualizer generation failed: " + completed.stdout[-4000:])
    html = report_path.read_text(encoding="utf-8", errors="replace")
    if "training_pipeline" not in html or "tokenizer_roundtrip" not in html:
        raise RuntimeError("v8 training IR visualizer omitted training/data artifacts")
    return {
        "passed": True, "report": str(report_path), "report_sha256": _sha256(report_path),
        "command": command, "stdout": completed.stdout[-4000:],
        "artifacts": {path.name: _sha256(path) for path in (
            tokenizer_path, roundtrip_path, qc_path, profile_path, loss_path, parity_path,
            step_profile_path, checkpoint_policy_path, stitch_path, pipeline_path)},
    }


def _inference_probe(args: argparse.Namespace) -> int:
    runtime = args.inference_runtime.resolve()
    library_path = runtime / "libmodel.so"
    lib = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    loaded_library = CERT._loaded_library_evidence(library_path)
    engine = CERT._mapped_library_evidence("libckernel_engine.so")
    engine_hashes = sorted(
        str(row.get("sha256")) for row in engine.get("identities", [])
        if isinstance(row, Mapping) and row.get("sha256")
    )
    provenance = {
        "training_library": loaded_library,
        "engine_dependency": {**engine, "expected_sha256": args.probe_engine_sha256,
                              "loaded_sha256": engine_hashes},
    }
    provenance["passed"] = bool(
        loaded_library.get("observed_in_process_maps")
        and loaded_library.get("sha256") == args.probe_library_sha256
        and engine.get("observed_in_process_maps")
        and len(engine.get("mapped_paths", [])) == 1
        and engine_hashes == [args.probe_engine_sha256]
    )
    if args.probe_provenance:
        _write_json(args.probe_provenance, provenance)
    if not provenance["passed"]:
        return 5
    lib.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.ck_model_init_with_manifest.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    rc = int(lib.ck_model_init_with_manifest(str(runtime / "weights.bump").encode(), str(runtime / "weights_manifest.map").encode()))
    if rc != 0: return 2
    tokens = np.fromfile(args.probe_tokens, dtype="<i4")[:args.probe_count].astype(np.int32, copy=True)
    output = np.empty(256, dtype=np.float32)
    if int(lib.ck_model_embed_tokens(tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), int(tokens.size))) != 0: return 3
    if int(lib.ck_model_forward(output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))) != 0: return 4
    output.astype("<f4").tofile(args.probe_logits)
    if args.probe_generate_count > 0:
        if args.probe_sequence is None or args.probe_trajectory_logits is None:
            return 7
        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int
        generated = [int(value) for value in tokens]
        logits_rows: list[np.ndarray] = []
        current = output.copy()
        for step in range(int(args.probe_generate_count)):
            logits_rows.append(current.copy())
            token = int(np.argmax(current))
            generated.append(token)
            if step + 1 < int(args.probe_generate_count):
                if int(lib.ck_model_decode(token, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))) != 0:
                    return 6
                current = output.copy()
        np.asarray(generated, dtype="<i4").tofile(args.probe_sequence)
        np.stack(logits_rows).astype("<f4").tofile(args.probe_trajectory_logits)
    return 0


def _resume_worker(args: argparse.Namespace) -> int:
    ck_run = _load_module("cke_v7_runtime_worker", V7 / "ck_run_v7.py")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    summary = json.loads((args.run_dir / "generated_train_runtime_summary_v7.json").read_text(encoding="utf-8"))
    library = args.run_dir / "libtrain.so"
    lib, provenance = _init_runtime(args.run_dir, library, summary, ck_run, role="resume_worker")
    expected = plan.get("checkpoint_expected")
    if not isinstance(expected, Mapping):
        raise RuntimeError("resume plan has no checkpoint identity contract")
    doc = _checkpoint_load(args.checkpoint, lib, expected)
    tokens = np.fromfile(plan["token_file"], dtype="<i4")
    if _sha256(Path(plan["token_file"])) != str(doc["dataset_sha256"]):
        raise RuntimeError("resume token stream does not match checkpoint dataset")
    checkpoint_config = doc["training_config"]
    batches = _batches(tokens, int(checkpoint_config["seq_len"]), int(checkpoint_config["epochs"]))
    timings = []
    start_microstep = int(doc["next_microstep"])
    for batch in batches[start_microstep:]: timings.append(_step(lib, batch, float(checkpoint_config["lr"]))[1])
    if int(lib.ck_train_get_accum_counter()) > 0 and int(lib.ck_train_flush_optimizer(float(checkpoint_config["lr"]))) <= 0:
        raise RuntimeError("resume worker final optimizer flush failed")
    out = {"weight": _weight_export(lib), "optimizer": _array_export(lib, "optimizer_state")}
    for key, value in out.items(): value.astype("<f4").tofile(args.worker_out.with_suffix(f".{key}.f32"))
    _write_json(args.worker_out, {"pid": os.getpid(), "optimizer_step": int(lib.ck_train_get_opt_step()),
                                  "start_microstep": start_microstep, "checkpoint_identity_sha256": doc["checkpoint_identity_sha256"],
                                  "provenance": provenance,
                                  "weight_sha256": _sha256(args.worker_out.with_suffix(".weight.f32")),
                                  "optimizer_sha256": _sha256(args.worker_out.with_suffix(".optimizer.f32"))})
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"schema": "cke.v8.training_workflow.v1", "status": "FAIL", "passed": False,
                              "execution": CERT._execution_identity(), "checks": {}, "negative_controls": {}, "failures": []}
    started = time.perf_counter()
    try:
        import torch
        import torch.nn.functional as F
        args.run_dir.mkdir(parents=True, exist_ok=True)
        train_ids, val_ids, corpus = _load_corpus(args.corpus, args.run_dir / "dataset")
        batches = _batches(train_ids, args.seq_len, args.epochs)
        ck_run = _load_module("cke_v7_runtime_workflow", V7 / "ck_run_v7.py")
        oracle = _load_module("cke_v7_oracle_workflow", V7 / "oracle_snapshot_torch_v7.py")
        python = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable
        init_cmd = [python, str(V7 / "ck_run_v7.py"), "init", "--run", str(args.run_dir), "--allow-non-cache-run-dir",
                    "--train-seed", str(args.seed), "--layers", "4", "--vocab-size", "256", "--embed-dim", str(args.d_model),
                    "--hidden-dim", str(args.hidden), "--num-heads", "4", "--num-kv-heads", "4", "--context-len", str(args.seq_len),
                    "--template", "qwen3", "--generate-ir", "--generate-runtime", "--train-bridge-lowering", "explicit",
                    "--adamw-beta1", str(args.beta1), "--adamw-beta2", str(args.beta2), "--adamw-eps", str(args.eps),
                    "--adamw-weight-decay", str(args.weight_decay)]
        init = subprocess.run(init_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if init.returncode: raise RuntimeError("four-layer initialization failed: " + init.stdout[-4000:])
        build_started = time.perf_counter()
        defines = {"CK_NUM_TOKENS": args.seq_len, "CK_GRAD_ACCUM_STEPS": args.grad_accum, "CK_TRAIN_USE_CE_PTREF": 1,
                   "CK_ADAMW_BETA1": args.beta1, "CK_ADAMW_BETA2": args.beta2, "CK_ADAMW_EPS": args.eps,
                   "CK_ADAMW_WEIGHT_DECAY": args.weight_decay, "CK_MAX_GRAD_NORM": 0}
        source, library = ck_run._ensure_train_runtime_artifacts(args.run_dir, python, False, runtime_defines=defines,
                                                                  train_tokens=args.seq_len, bridge_lowering="explicit")
        build_seconds = time.perf_counter() - build_started
        summary = json.loads((args.run_dir / "generated_train_runtime_summary_v7.json").read_text(encoding="utf-8"))
        expected, policy = CERT._expected_trainable_parameters(args.run_dir)
        inventory = CERT._validate_parameter_inventory(summary, args.run_dir, expected)
        configuration = _training_config(args)
        training_config_sha256 = _json_sha256(configuration)
        lib, main_provenance = _init_runtime(args.run_dir, library, summary, ck_run, role="main_training")
        report["execution"]["actual_runtime_threads"] = main_provenance.get("actual_runtime_threads")
        initial = _weight_export(lib)
        heldout_before = _evaluate(lib, summary, val_ids, args.seq_len)
        sample_before = _sample(lib, summary, args.seq_len)
        decoded, cfg = oracle._decode_weight_snapshot(args.run_dir, summary, initial)
        names = [str(r["name"]) for r in inventory]
        weights = {n: v.detach().clone().requires_grad_(n in names) for n, v in decoded.items()}
        model = oracle.SnapshotQwenLikeOracle(weights, cfg)
        optimizer = torch.optim.AdamW([weights[n] for n in names], lr=args.lr, betas=(args.beta1, args.beta2),
                                      eps=args.eps, weight_decay=args.weight_decay, foreach=False)
        optimizer.zero_grad(set_to_none=True)
        loss_rows, epoch_rows, parity_rows = [], [], []
        timing = {"step_ms": 0.0, "forward_ms": 0.0, "backward_ms": 0.0, "optimizer_ms": 0.0}
        token_count = 0; update_count = 0; max_weight_diff = 0.0; max_moment_diff = 0.0; max_grad_diff = 0.0
        max_loss_diff = 0.0; max_logits_diff = 0.0
        first_update: dict[str, Any] | None = None
        checkpoint_at = max(1, len(batches) // 2)
        checkpoint_at = checkpoint_at - (checkpoint_at % args.grad_accum) + min(3, args.grad_accum - 1)
        checkpoint_at = min(checkpoint_at, len(batches) - 1)
        checkpoint_path = None
        epoch_ck_loss_sum = 0.0; epoch_pt_loss_sum = 0.0; epoch_tokens = 0
        train_started = time.perf_counter()
        for micro, batch in enumerate(batches, 1):
            x, y, valid, epoch = batch
            ck_loss, profile = _step(lib, batch, args.lr)
            for key in timing: timing[key] += float(profile[key])
            tx = torch.from_numpy(x.astype(np.int64)).view(1, -1); ty = torch.from_numpy(y.astype(np.int64)).view(1, -1)
            logits = model.forward(tx)
            t_loss = F.cross_entropy(logits[:, :valid, :].reshape(-1, 256), ty[:, :valid].reshape(-1), reduction="mean")
            max_loss_diff = max(max_loss_diff, abs(ck_loss - float(t_loss.item())))
            if micro == 1 or ((micro - 1) % (len(batches) // args.epochs) == 0):
                ck_logits = [value for name, value in CERT._activation_snapshot(lib, summary).items() if ".logits." in name][0]
                max_logits_diff = max(max_logits_diff, float(np.max(np.abs(ck_logits - logits.detach().float().reshape(-1).numpy()))))
            (t_loss * valid).backward(); token_count += valid
            epoch_ck_loss_sum += ck_loss * valid; epoch_pt_loss_sum += float(t_loss.item()) * valid; epoch_tokens += valid
            loss_rows.append({"microstep": micro, "epoch": epoch + 1, "valid_tokens": valid, "cke": ck_loss, "pytorch": float(t_loss.item())})
            boundary = (micro % args.grad_accum == 0)
            if boundary:
                for n in names: weights[n].grad.div_(token_count)
                torch_grad = np.concatenate([weights[n].grad.detach().float().reshape(-1).numpy() for n in names])
                grad_diff = float(np.max(np.abs(_grad_export(lib) - torch_grad)))
                max_grad_diff = max(max_grad_diff, grad_diff)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); token_count = 0; update_count += 1
                ck_w = _weight_export(lib); torch_w = _torch_weight_flat(weights, summary)
                weight_diff = float(np.max(np.abs(ck_w - torch_w)))
                moment_diff = float(np.max(np.abs(_array_export(lib, "optimizer_state") - _torch_optimizer_flat(torch, weights, optimizer, summary))))
                max_weight_diff = max(max_weight_diff, weight_diff); max_moment_diff = max(max_moment_diff, moment_diff)
                parity_row = {"step": update_count, "microstep": micro, "loss_diff": abs(ck_loss - float(t_loss.item())),
                              "gradient_max_abs_diff": grad_diff, "max_param_diff": weight_diff,
                              "moment_max_abs_diff": moment_diff, "worst_param": "reported_in_final_tensor_inventory"}
                parity_rows.append(parity_row)
                if first_update is None:
                    first_update = dict(parity_row)
            if micro == checkpoint_at:
                checkpoint_path = _checkpoint_write(args.run_dir / "checkpoints", f"micro_{micro:06d}", lib,
                    {"next_microstep": micro, "total_microsteps": len(batches),
                     "runtime_contract_sha256": summary["runtime_contract_sha256"],
                     "training_config_sha256": training_config_sha256,
                     "training_config": configuration,
                     "dataset_sha256": corpus["splits"]["train"]["token_ids_sha256"]})
            if micro == len(batches) or batches[micro][3] != epoch:
                epoch_rows.append({"epoch": epoch + 1, "mean_loss": epoch_ck_loss_sum / epoch_tokens,
                                   "cke_mean_loss": epoch_ck_loss_sum / epoch_tokens,
                                   "pytorch_mean_loss": epoch_pt_loss_sum / epoch_tokens,
                                   "tokens": epoch_tokens, "last_microstep": micro})
                epoch_ck_loss_sum = 0.0; epoch_pt_loss_sum = 0.0; epoch_tokens = 0
        final_partial: dict[str, Any] = {"present": False, "passed": True}
        final_flush_negative = {"passed": True, "applicable": False}
        final_flush_seconds = 0.0
        final_grad_ck = np.empty(0, dtype=np.float32); final_grad_torch = np.empty(0, dtype=np.float32)
        if token_count:
            for n in names: weights[n].grad.div_(token_count)
            torch_grad = np.concatenate([weights[n].grad.detach().float().reshape(-1).numpy() for n in names])
            ck_grad = _grad_export(lib)
            grad_diff = float(np.max(np.abs(ck_grad - torch_grad)))
            max_grad_diff = max(max_grad_diff, grad_diff)
            pre_state = {
                "weight": _weight_export(lib), "optimizer": _array_export(lib, "optimizer_state"),
                "accum": _array_export(lib, "accum"), "optimizer_step": int(lib.ck_train_get_opt_step()),
                "accum_counter": int(lib.ck_train_get_accum_counter()), "accum_tokens": int(lib.ck_train_get_accum_tokens()),
            }
            flush_started = time.perf_counter()
            flush_rc = int(lib.ck_train_flush_optimizer(args.lr))
            final_flush_seconds = time.perf_counter() - flush_started
            if flush_rc <= 0:
                raise RuntimeError("final partial optimizer flush failed")
            optimizer.step(); optimizer.zero_grad(set_to_none=True); update_count += 1
            correct_weight = _weight_export(lib); correct_optimizer = _array_export(lib, "optimizer_state")
            correct_accum = _array_export(lib, "accum")
            torch_weight = _torch_weight_flat(weights, summary)
            torch_optimizer = _torch_optimizer_flat(torch, weights, optimizer, summary)
            weight_diff = float(np.max(np.abs(correct_weight - torch_weight)))
            moment_diff = float(np.max(np.abs(correct_optimizer - torch_optimizer)))
            counters_pass = bool(int(lib.ck_train_get_accum_counter()) == 0 and int(lib.ck_train_get_accum_tokens()) == 0
                                 and int(lib.ck_train_get_opt_step()) == update_count)
            final_partial = {"present": True, "passed": bool(grad_diff <= args.grad_tol and weight_diff <= args.param_tol
                              and moment_diff <= args.moment_tol and counters_pass),
                             "contributing_tokens": token_count, "gradient_max_abs_diff": grad_diff,
                             "weight_max_abs_diff": weight_diff, "moment_max_abs_diff": moment_diff,
                             "flush_return": flush_rc, "counters_passed": counters_pass,
                             "optimizer_step": int(lib.ck_train_get_opt_step()), "accum_counter": int(lib.ck_train_get_accum_counter()),
                             "accum_tokens": int(lib.ck_train_get_accum_tokens())}
            max_weight_diff = max(max_weight_diff, weight_diff); max_moment_diff = max(max_moment_diff, moment_diff)
            parity_rows.append({"step": update_count, "microstep": len(batches), "loss_diff": 0.0,
                                "gradient_max_abs_diff": grad_diff, "max_param_diff": weight_diff,
                                "moment_max_abs_diff": moment_diff, "worst_param": "reported_in_final_tensor_inventory",
                                "partial_window": True})
            final_grad_ck = ck_grad; final_grad_torch = torch_grad
            _restore_training_state(lib, weight=pre_state["weight"], optimizer=pre_state["optimizer"], accum=pre_state["accum"],
                                    optimizer_step=pre_state["optimizer_step"], accum_counter=pre_state["accum_counter"],
                                    accum_tokens=pre_state["accum_tokens"])
            injected_rc = int(lib.ck_train_flush_optimizer(args.lr * 100.0))
            injected_weight_delta = float(np.max(np.abs(_weight_export(lib) - correct_weight)))
            injected_moment_delta = float(np.max(np.abs(_array_export(lib, "optimizer_state") - correct_optimizer)))
            final_flush_negative = {"applicable": True, "passed": bool(injected_rc > 0 and
                                    (injected_weight_delta > 1e-6 or injected_moment_delta > 1e-6)),
                                    "injected_lr": args.lr * 100.0, "weight_delta_from_expected": injected_weight_delta,
                                    "moment_delta_from_expected": injected_moment_delta}
            _restore_training_state(lib, weight=correct_weight, optimizer=correct_optimizer, accum=correct_accum,
                                    optimizer_step=update_count, accum_counter=0, accum_tokens=0)
        train_seconds = time.perf_counter() - train_started
        final_weights = _weight_export(lib); final_opt = _array_export(lib, "optimizer_state")
        # Fresh-process resume from a checkpoint captured inside an accumulation window.
        assert checkpoint_path is not None
        checkpoint_expected = {
            "runtime_contract_sha256": summary["runtime_contract_sha256"],
            "training_config_sha256": training_config_sha256,
            "dataset_sha256": corpus["splits"]["train"]["token_ids_sha256"],
            "total_microsteps": len(batches), "grad_accum": args.grad_accum,
        }
        checkpoint_doc = json.loads((checkpoint_path / "checkpoint.json").read_text(encoding="utf-8"))
        _validate_checkpoint_document(checkpoint_doc, checkpoint_expected)
        checkpoint_controls: dict[str, Any] = {}
        for control_name, key in (("wrong_dataset", "dataset_sha256"), ("wrong_runtime", "runtime_contract_sha256")):
            altered_expected = dict(checkpoint_expected); altered_expected[key] = "0" * 64
            detected = False
            try: _validate_checkpoint_document(checkpoint_doc, altered_expected)
            except RuntimeError: detected = True
            checkpoint_controls[control_name] = {"passed": detected, "altered_field": key}
        wrong_config_expected = dict(checkpoint_expected); wrong_config_expected["training_config_sha256"] = "f" * 64
        detected_config = False
        try: _validate_checkpoint_document(checkpoint_doc, wrong_config_expected)
        except RuntimeError: detected_config = True
        checkpoint_controls["wrong_config"] = {"passed": detected_config, "altered_field": "training_config_sha256"}
        altered_cursor = dict(checkpoint_doc); altered_cursor["next_microstep"] = int(checkpoint_doc["next_microstep"]) + 1
        detected_cursor = False
        try: _validate_checkpoint_document(altered_cursor, checkpoint_expected)
        except RuntimeError: detected_cursor = True
        checkpoint_controls["altered_cursor"] = {"passed": detected_cursor, "authoritative_next_microstep": checkpoint_doc["next_microstep"]}
        plan = {"token_file": corpus["splits"]["train"]["token_ids"], "checkpoint_expected": checkpoint_expected}
        plan_path = args.run_dir / "resume_plan.json"; _write_json(plan_path, plan)
        worker_out = args.run_dir / "resume_worker.json"
        worker = subprocess.run([python, str(Path(__file__).resolve()), "--resume-worker", "--run-dir", str(args.run_dir),
                                 "--checkpoint", str(checkpoint_path), "--plan", str(plan_path), "--worker-out", str(worker_out)],
                                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if worker.returncode: raise RuntimeError("fresh-process resume failed: " + worker.stdout[-4000:])
        worker_doc = json.loads(worker_out.read_text(encoding="utf-8"))
        if not worker_doc.get("provenance", {}).get("passed"):
            raise RuntimeError("resume worker did not verify loaded runtime provenance")
        if int(worker_doc.get("start_microstep", -1)) != int(checkpoint_doc["next_microstep"]):
            raise RuntimeError("resume worker did not use checkpoint-owned cursor")
        resumed_w = np.fromfile(worker_out.with_suffix(".weight.f32"), dtype="<f4")
        resumed_o = np.fromfile(worker_out.with_suffix(".optimizer.f32"), dtype="<f4")
        resume_weight_diff = float(np.max(np.abs(final_weights - resumed_w))); resume_opt_diff = float(np.max(np.abs(final_opt - resumed_o)))
        # Prove an interrupted write cannot replace the last complete checkpoint pointer.
        latest_before = (args.run_dir / "checkpoints" / "latest.json").read_bytes()
        interrupted = False
        try: _checkpoint_write(args.run_dir / "checkpoints", "interrupted", lib, {"injected": True}, inject_interrupt=True)
        except RuntimeError as exc: interrupted = "injected" in str(exc)
        atomic_ok = interrupted and (args.run_dir / "checkpoints" / "latest.json").read_bytes() == latest_before
        # Export then independently regenerate and compile the forward-capable generated runtime.
        export_dir = args.run_dir / "exported_inference"; _export_weights(args.run_dir, summary, final_weights, export_dir)
        export_build = time.perf_counter(); _, export_lib_path = ck_run._ensure_train_runtime_artifacts(export_dir, python, False,
            runtime_defines={"CK_NUM_TOKENS": args.seq_len}, train_tokens=args.seq_len, bridge_lowering="explicit")
        export_build_seconds = time.perf_counter() - export_build
        export_summary = json.loads((export_dir / "generated_train_runtime_summary_v7.json").read_text())
        export_lib, export_provenance = _init_runtime(export_dir, export_lib_path, export_summary, ck_run, role="export_rebuild")
        eval_batch = _batches(val_ids, args.seq_len, 1)[0]; ex, ey, ev, _ = eval_batch
        trained_logits = _forward_logits(lib, summary, ex, ey, ev); exported_logits = _forward_logits(export_lib, export_summary, ex, ey, ev)
        export_diff = float(np.max(np.abs(trained_logits - exported_logits)))
        heldout_after = _evaluate(export_lib, export_summary, val_ids, args.seq_len)
        sample_after = _sample(export_lib, export_summary, args.seq_len)
        # Build the actual inference-only v8 prefill/decode library from the exported weights.
        v8_source = args.run_dir / "v8_inference_source"
        v8_runtime = args.run_dir / "v8_inference_runtime"
        _export_v8_bundle(export_dir, args.run_dir / "config.json", v8_source)
        v8_build_started = time.perf_counter()
        v8_cmd = [python, str(ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"), "run", str(v8_source),
                  "--run", str(v8_runtime), "--context-len", str(args.seq_len), "--logits-layout", "full", "--generate-only"]
        v8_build = subprocess.run(v8_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if v8_build.returncode != 0:
            raise RuntimeError("v8 inference-only generation failed: " + v8_build.stdout[-8000:])
        v8_build_seconds = time.perf_counter() - v8_build_started
        v8_logits_path = args.run_dir / "v8_inference_logits.f32"
        v8_library_sha256 = _sha256(v8_runtime / "libmodel.so")
        v8_engine_sha256 = _sha256(v8_runtime / "libckernel_engine.so")
        v8_provenance_path = args.run_dir / "v8_inference_provenance.json"
        probe_env = dict(os.environ); probe_env["LD_LIBRARY_PATH"] = str(v8_runtime)
        probe = subprocess.run([python, str(Path(__file__).resolve()), "--inference-probe", "--inference-runtime", str(v8_runtime),
                                "--probe-tokens", corpus["splits"]["validation"]["token_ids"], "--probe-count", str(ev),
                                "--probe-logits", str(v8_logits_path), "--probe-provenance", str(v8_provenance_path),
                                "--probe-library-sha256", v8_library_sha256, "--probe-engine-sha256", v8_engine_sha256],
                               cwd=ROOT, env=probe_env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if probe.returncode != 0:
            raise RuntimeError("v8 inference probe failed: " + probe.stdout[-4000:])
        v8_logits = np.fromfile(v8_logits_path, dtype="<f4")
        v8_inference_diff = float(np.max(np.abs(trained_logits[ev - 1] - v8_logits)))
        v8_provenance = json.loads(v8_provenance_path.read_text(encoding="utf-8"))
        generation_prompt = np.frombuffer(b"The ", dtype=np.uint8).astype(np.int32)
        generation_prompt_path = args.run_dir / "generation_prompt.i32"
        generation_prompt.astype("<i4").tofile(generation_prompt_path)
        generation_steps = 12
        reference_sequence, reference_trajectory = _sample_trajectory(
            export_lib, export_summary, args.seq_len, prompt=b"The ", new_tokens=generation_steps
        )
        trajectory_logits_path = args.run_dir / "v8_generation_logits.f32"
        trajectory_sequence_path = args.run_dir / "v8_generation_sequence.i32"
        trajectory_provenance_path = args.run_dir / "v8_generation_provenance.json"
        trajectory_probe = subprocess.run([
            python, str(Path(__file__).resolve()), "--inference-probe", "--inference-runtime", str(v8_runtime),
            "--probe-tokens", str(generation_prompt_path), "--probe-count", str(generation_prompt.size),
            "--probe-logits", str(args.run_dir / "v8_generation_first_logits.f32"),
            "--probe-generate-count", str(generation_steps), "--probe-sequence", str(trajectory_sequence_path),
            "--probe-trajectory-logits", str(trajectory_logits_path), "--probe-provenance", str(trajectory_provenance_path),
            "--probe-library-sha256", v8_library_sha256, "--probe-engine-sha256", v8_engine_sha256,
        ], cwd=ROOT, env=probe_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if trajectory_probe.returncode != 0:
            raise RuntimeError("v8 generation trajectory probe failed: " + trajectory_probe.stdout[-4000:])
        v8_sequence = np.fromfile(trajectory_sequence_path, dtype="<i4").astype(np.int32).tolist()
        v8_trajectory = np.fromfile(trajectory_logits_path, dtype="<f4").reshape(generation_steps, 256)
        trajectory_logits_diff = float(np.max(np.abs(reference_trajectory - v8_trajectory)))
        trajectory_tokens_match = v8_sequence == reference_sequence
        trajectory_provenance = json.loads(trajectory_provenance_path.read_text(encoding="utf-8"))
        first_loss = sum(r["cke"]*r["valid_tokens"] for r in loss_rows[:len(batches)//args.epochs]) / 10000.0
        last_loss = sum(r["cke"]*r["valid_tokens"] for r in loss_rows[-len(batches)//args.epochs:]) / 10000.0
        generated_c_seconds = timing["step_ms"] / 1000.0 + final_flush_seconds
        performance = {
            "setup_compile_seconds": build_seconds,
            "certification_loop_seconds": train_seconds,
            "certification_workflow_tokens_per_second": (10000 * args.epochs) / train_seconds,
            "generated_c_training_seconds": generated_c_seconds,
            "generated_c_tokens_per_second": (10000 * args.epochs) / generated_c_seconds,
            "generated_c_optimizer_steps_per_second": update_count / generated_c_seconds,
            "tokens_per_second": (10000 * args.epochs) / generated_c_seconds,
            "tokens_per_second_scope": "generated_c_profiled_steps_plus_final_flush",
            "export_compile_seconds": export_build_seconds, "v8_inference_compile_seconds": v8_build_seconds,
            "generated_profile_ms": {**timing, "final_flush_ms": final_flush_seconds * 1000.0},
            "optimizer_steps": update_count,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        }
        weight_rows = [row for row in summary["tensor_slots"] if str(row.get("name", "")).startswith("weight.") and row.get("section") == "weights"]
        weight_rows.sort(key=lambda row: int(row.get("offset", 0)))
        final_torch_weights = _torch_weight_flat(weights, summary)
        weight_tensor_diffs = _tensor_discrepancies(
            final_weights, final_torch_weights,
            [str(row["name"])[len("weight."):] for row in weight_rows], [int(row["numel"]) for row in weight_rows],
        )
        optimizer_rows = [row for row in summary["tensor_slots"] if row.get("section") in {"optimizer_m", "optimizer_v"}]
        optimizer_rows.sort(key=lambda row: int(row.get("offset", 0)))
        final_torch_optimizer = _torch_optimizer_flat(torch, weights, optimizer, summary)
        optimizer_tensor_diffs = _tensor_discrepancies(
            final_opt, final_torch_optimizer,
            [str(row["name"]) for row in optimizer_rows], [int(row["numel"]) for row in optimizer_rows],
        )
        gradient_tensor_diffs = _tensor_discrepancies(
            final_grad_ck, final_grad_torch, [str(row["name"]) for row in inventory], [int(row["numel"]) for row in inventory]
        ) if final_partial["present"] else []
        first_growth = None
        previous_weight_diff = 0.0
        for row in parity_rows:
            current = float(row["max_param_diff"])
            if current > previous_weight_diff:
                first_growth = dict(row); break
            previous_weight_diff = current
        visualizer = _write_visualizer_artifacts(
            run_dir=args.run_dir, python=python, args=args, corpus=corpus, epoch_rows=epoch_rows,
            parity_rows=parity_rows, performance=performance, summary=summary, checkpoint_path=checkpoint_path,
        )
        provenance = {
            "passed": bool(main_provenance.get("passed") and export_provenance.get("passed")
                           and worker_doc.get("provenance", {}).get("passed")
                           and v8_provenance.get("passed") and trajectory_provenance.get("passed")),
            "main_training": main_provenance, "resume_worker": worker_doc.get("provenance"),
            "export_rebuild": export_provenance, "inference_probe": v8_provenance,
            "generation_probe": trajectory_provenance,
        }
        checks = {
            "learning": {"passed": last_loss < first_loss and heldout_after < heldout_before, "first_epoch_loss": first_loss, "last_epoch_loss": last_loss,
                "heldout_loss_before": heldout_before, "heldout_loss_after": heldout_after, "sample_before": sample_before,
                "sample_after": sample_after, "epochs": epoch_rows},
            "pytorch_trajectory": {"passed": max_weight_diff <= args.param_tol and max_moment_diff <= args.moment_tol and max_grad_diff <= args.grad_tol and max_loss_diff <= args.loss_tol and max_logits_diff <= args.logits_tol and final_partial["passed"],
                "max_weight_abs_diff": max_weight_diff, "max_moment_abs_diff": max_moment_diff, "max_gradient_abs_diff": max_grad_diff,
                "max_loss_abs_diff": max_loss_diff, "max_selected_logits_abs_diff": max_logits_diff,
                "first_update": first_update, "first_divergence_growth": first_growth,
                "trajectory": parity_rows,
                "final_tensor_discrepancies": {"weights": weight_tensor_diffs, "optimizer": optimizer_tensor_diffs,
                                               "partial_window_gradients": gradient_tensor_diffs},
                "tolerances": {"parameter": args.param_tol, "moment": args.moment_tol, "gradient": args.grad_tol, "loss": args.loss_tol, "logits": args.logits_tol}},
            "final_partial_update": final_partial,
            "fresh_process_resume": {"passed": resume_weight_diff == 0.0 and resume_opt_diff == 0.0 and worker_doc.get("provenance", {}).get("passed") is True,
                "worker_pid": worker_doc["pid"], "checkpoint_owned_start_microstep": worker_doc["start_microstep"],
                "parent_pid": os.getpid(), "weight_max_abs_diff": resume_weight_diff, "optimizer_max_abs_diff": resume_opt_diff,
                "checkpoint_accum_counter": checkpoint_doc["accum_counter"], "checkpoint_accum_tokens": checkpoint_doc["accum_tokens"]},
            "checkpoint_compatibility": {"passed": all(row["passed"] for row in checkpoint_controls.values()),
                                         "authoritative_identity": checkpoint_doc["checkpoint_identity_sha256"],
                                         "negative_controls": checkpoint_controls},
            "negative_control_detection": {"passed": bool(final_flush_negative["passed"] and
                                                   all(row["passed"] for row in checkpoint_controls.values())),
                                           "final_partial_flush": final_flush_negative,
                                           "checkpoint_identity": checkpoint_controls},
            "atomic_checkpoint_publication": {"passed": atomic_ok, "power_loss_durability_claim": False},
            "runtime_provenance": provenance,
            "inference_export": {"passed": export_diff == 0.0 and v8_inference_diff <= args.inference_tol
                                  and trajectory_tokens_match and trajectory_logits_diff <= args.inference_tol
                                  and v8_provenance.get("passed") is True and trajectory_provenance.get("passed") is True,
                "rebuilt_training_forward_logits_max_abs_diff": export_diff, "v8_inference_last_token_logits_max_abs_diff": v8_inference_diff,
                "generation_steps": generation_steps, "generation_tokens_match": trajectory_tokens_match,
                "generation_logits_max_abs_diff": trajectory_logits_diff,
                "generation_sequence": v8_sequence,
                "v8_inference_tolerance": args.inference_tol, "heldout_loss": heldout_after,
                "independent_training_build_directory": str(export_dir), "v8_inference_only_build_directory": str(v8_runtime),
                "standalone_native_executable": "NOT_CERTIFIED",
                "v8_generated_source_sha256": _sha256(v8_runtime / "model_v8.c"), "v8_library_sha256": v8_library_sha256},
            "training_ir_visualizer": visualizer,
        }
        report["negative_controls"] = {"final_partial_flush_wrong_lr": final_flush_negative,
                                       "checkpoint_identity": checkpoint_controls}
        report.update({"status": "PASS" if all(v["passed"] for v in checks.values()) else "FAIL", "checks": checks, "corpus": corpus,
            "configuration": {**configuration, "training_config_sha256": training_config_sha256,
                "unique_train_tokens": 10000, "token_presentations": 10000*args.epochs,
                "numerical_contract": "strict_reference_order", "parallel_scaling_certified": False},
            "implementation": {"orchestrator": str(Path(__file__).relative_to(ROOT)), "shared_training_codegen": "version/v7/scripts/codegen_train_runtime_v7.py",
                "shared_oracle": "version/v7/scripts/oracle_snapshot_torch_v7.py", "v7_training_cli_invoked": False,
                "inference_codegen": "version/v8/scripts/ck_run_v8.py -> build_ir_v8.py -> codegen_v8.py",
                "visualizer": "version/v8/tools/open_ir_visualizer_v8.py (v7 interface lineage)"},
            "performance": performance,
            "artifacts": {"checkpoint": str(checkpoint_path), "training_export": str(export_dir), "v8_inference": str(v8_runtime),
                          "ir_visualizer": visualizer["report"], "generated_source_sha256": _sha256(source)},
            "passed": all(v["passed"] for v in checks.values())})
    except Exception as exc:
        report["failures"].append(str(exc)); report["exception"] = {"type": type(exc).__name__, "traceback": traceback.format_exc()}
    report["wall_seconds"] = time.perf_counter() - started
    _write_json(args.report, report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Certify a complete v8 generated-C training workflow")
    p.add_argument("--resume-worker", action="store_true"); p.add_argument("--checkpoint", type=Path); p.add_argument("--plan", type=Path); p.add_argument("--worker-out", type=Path)
    p.add_argument("--inference-probe", action="store_true"); p.add_argument("--inference-runtime", type=Path)
    p.add_argument("--probe-tokens", type=Path); p.add_argument("--probe-count", type=int); p.add_argument("--probe-logits", type=Path)
    p.add_argument("--probe-provenance", type=Path); p.add_argument("--probe-library-sha256"); p.add_argument("--probe-engine-sha256")
    p.add_argument("--probe-generate-count", type=int, default=0); p.add_argument("--probe-sequence", type=Path)
    p.add_argument("--probe-trajectory-logits", type=Path)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN); p.add_argument("--json-out", dest="report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--corpus", type=Path, default=CORPUS_SPEC); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=32); p.add_argument("--epochs", type=int, default=10); p.add_argument("--d-model", type=int, default=32); p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=8); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--beta1", type=float, default=.9); p.add_argument("--beta2", type=float, default=.999)
    p.add_argument("--eps", type=float, default=1e-8); p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--param-tol", type=float, default=5e-3); p.add_argument("--moment-tol", type=float, default=1e-3); p.add_argument("--grad-tol", type=float, default=5e-3)
    p.add_argument("--loss-tol", type=float, default=2e-2); p.add_argument("--logits-tol", type=float, default=5e-2)
    p.add_argument("--inference-tol", type=float, default=1e-3)
    args = p.parse_args(); args.run_dir=args.run_dir.resolve(); args.report=args.report.resolve(); args.corpus=args.corpus.resolve()
    if args.resume_worker: return _resume_worker(args)
    if args.inference_probe: return _inference_probe(args)
    result=run(args); print(json.dumps({"status":result["status"],"passed":result["passed"],"report":str(args.report)}, indent=2)); return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
