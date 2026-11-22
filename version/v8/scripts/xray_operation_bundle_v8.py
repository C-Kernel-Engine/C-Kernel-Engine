"""Verified, data-only operands for a single numerical operation replay.

Bundles are local diagnostic artifacts. A bundle never chooses executable code:
the consumer must validate its recorded provider against its own fixed adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


_DTYPES = {
    "fp64": np.dtype("<f8"),
    "fp32": np.dtype("<f4"),
    "fp16": np.dtype("<f2"),
    "bf16_bits": np.dtype("<u2"),
    # Opaque storage bytes. The replay adapter validates packed layouts.
    "u8": np.dtype("u1"),
}
_ROLE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


class BundleError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bundle(
    directory: Path, *, operation: str, provider: str, contract: str,
    runtime: Path, context: dict[str, Any],
    tensors: dict[str, tuple[np.ndarray, str, tuple[str, ...]]],
    embed_runtime: bool = False,
) -> Path:
    if any(not isinstance(value, str) or not value for value in (operation, provider, contract)) or not tensors:
        raise BundleError("operation, provider, contract and tensors are required")
    if not runtime.is_file():
        raise BundleError("runtime library is missing")
    if directory.exists() and any(directory.iterdir()):
        raise BundleError("bundle directory must be empty")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.stat().st_mode & 0o077:
        raise BundleError("bundle directory must be private (mode 0700)")
    entries = {}
    for role, (values, dtype_name, axes) in tensors.items():
        if not _ROLE.fullmatch(role) or dtype_name not in _DTYPES:
            raise BundleError(f"invalid tensor role or dtype: {role}")
        array = np.asarray(values)
        if not array.size or array.ndim != len(axes) or any(not axis for axis in axes):
            raise BundleError(f"{role} has invalid logical axes or extent")
        if array.size > (1 << 27):
            raise BundleError(f"{role} exceeds diagnostic tensor limit")
        if dtype_name != "bf16_bits" and not np.isfinite(array).all():
            raise BundleError(f"{role} contains nonfinite values")
        if array.dtype != _DTYPES[dtype_name]:
            raise BundleError(f"{role} storage dtype does not match {dtype_name}")
        if dtype_name == "bf16_bits" and not np.isfinite(
            (array.astype(np.uint32) << 16).view(np.float32)
        ).all():
            raise BundleError(f"{role} contains nonfinite BF16 values")
        raw = np.ascontiguousarray(array).tobytes()
        name = f"{role}.bin"
        path = directory / name
        path.write_bytes(raw)
        path.chmod(0o600)
        entries[role] = {
            "file": name, "dtype": dtype_name, "shape": list(array.shape),
            "axes": list(axes), "size_bytes": len(raw), "sha256": sha256(path),
        }
    if embed_runtime:
        embedded = directory / "runtime.so"
        shutil.copyfile(runtime, embedded)
        embedded.chmod(0o600)
        runtime_entry = "runtime.so"
    else:
        runtime_entry = str(runtime.resolve())
    manifest = {
        "schema": "cke.xray_operation_bundle", "schema_version": 1,
        "operation": operation, "provider": provider, "contract": contract,
        "runtime": {"path": runtime_entry, "sha256": sha256(runtime)},
        "context": context, "tensors": entries,
    }
    destination = directory / "bundle.json"
    temporary = directory / "bundle.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, destination)
    return destination


def load_bundle(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError("cannot read operation bundle") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "cke.xray_operation_bundle" or manifest.get("schema_version") != 1:
        raise BundleError("unsupported operation bundle schema")
    for key in ("operation", "provider", "contract"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise BundleError(f"missing {key} identity")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("path"), str):
        raise BundleError("missing runtime identity")
    runtime_path = Path(runtime["path"])
    if not runtime_path.is_absolute():
        if runtime_path.as_posix() != "runtime.so":
            raise BundleError("invalid embedded runtime path")
        runtime_path = path.parent / runtime_path
    if not runtime_path.is_file() or sha256(runtime_path) != runtime.get("sha256"):
        raise BundleError("runtime library identity changed")
    if not isinstance(manifest.get("context"), dict):
        raise BundleError("bundle context must be an object")
    specs = manifest.get("tensors")
    if not isinstance(specs, dict) or not specs:
        raise BundleError("bundle has no tensors")
    tensors = {}
    for role, spec in specs.items():
        if not isinstance(role, str) or not _ROLE.fullmatch(role) or not isinstance(spec, dict):
            raise BundleError("invalid tensor entry")
        dtype_name = spec.get("dtype")
        shape = spec.get("shape")
        axes = spec.get("axes")
        if not isinstance(dtype_name, str) or dtype_name not in _DTYPES or not isinstance(shape, list) or not shape or not isinstance(axes, list) or len(axes) != len(shape):
            raise BundleError(f"{role} has invalid dtype or geometry")
        if any(type(dim) is not int or dim <= 0 for dim in shape) or any(not isinstance(axis, str) or not axis for axis in axes):
            raise BundleError(f"{role} has invalid logical axes or extent")
        count = math.prod(shape)
        if count > (1 << 27):
            raise BundleError(f"{role} exceeds diagnostic tensor limit")
        expected = count * _DTYPES[dtype_name].itemsize
        filename = spec.get("file")
        if filename != f"{role}.bin" or spec.get("size_bytes") != expected:
            raise BundleError(f"{role} has inconsistent byte extent")
        tensor_path = path.parent / filename
        if not tensor_path.is_file() or tensor_path.stat().st_size != expected or sha256(tensor_path) != spec.get("sha256"):
            raise BundleError(f"{role} tensor bytes changed or are missing")
        values = np.fromfile(tensor_path, dtype=_DTYPES[dtype_name]).reshape(shape)
        if dtype_name != "bf16_bits" and not np.isfinite(values).all():
            raise BundleError(f"{role} contains nonfinite values")
        tensors[role] = values
    return manifest, tensors
