"""Diagnostic text -> generated Clef backbone -> generated head, no torch executor.

Offline head compilation is explicit and request-shaped in this first milestone.
This is not yet a persistent arbitrary-text serving or training-data endpoint.
"""

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import struct
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from version.v8.decision.capture_clef_features import (
    verify_lock,
    verify_release,
    file_sha,
    compact_inputs,
)
from version.v8.decision.clef_text_frontend import Tokenizer, prepare
from version.v8.decision.package_clef_head import build_package
from version.v8.scripts.native_decoder_features_v8 import DecoderFeatures
from server.decision.native import NativeHead

BUNDLE_FILES = (
    "libmodel.so",
    "model_v8.c",
    "weights.bump",
    "weights_manifest.json",
    "weights_manifest.map",
    "lowered_decode_call.json",
    "config.json",
)


def native_dependencies():
    """Fail before loading 9B weights if required native math symbols are absent."""
    result = {}
    for variable, symbols in (
        ("CK_SLEEF_LIBRARY", ("Sleef_expf16_u10", "Sleef_log1pf16_u10")),
        ("CK_MKL_LIBRARY", ("vsExp", "cblas_sgemm")),
    ):
        path = Path(os.environ.get(variable, "")).expanduser()
        if not path.is_file() or "libtorch" in path.name:
            raise ValueError(
                f"{variable} must identify a standalone native math library"
            )
        library = C.CDLL(str(path.resolve()))
        if not all(hasattr(library, symbol) for symbol in symbols):
            raise ValueError(f"{variable} lacks required provider symbols")
        result[variable] = dict(path=str(path.resolve()), sha256=file_sha(path))
    return result


def bind_bundle(folder, lock_sha):
    folder = Path(folder)
    names = set(BUNDLE_FILES) | {p.name for p in folder.glob("*.so")}
    return dict(
        schema="cke.clef.native-backbone.v1",
        release_lock_sha256=lock_sha,
        files={name: file_sha(folder / name) for name in sorted(names)},
    )


def verify_bundle(folder, descriptor, expected, lock_sha):
    if file_sha(descriptor) != expected:
        raise ValueError("backbone bundle descriptor mismatch")
    doc = json.loads(Path(descriptor).read_text())
    if (
        doc.get("schema") != "cke.clef.native-backbone.v1"
        or doc.get("release_lock_sha256") != lock_sha
        or not set(BUNDLE_FILES) <= set(doc.get("files", {}))
    ):
        raise ValueError("unbound backbone bundle")
    for name, digest in doc["files"].items():
        if (
            name in (".", "..")
            or Path(name).name != name
            or file_sha(Path(folder) / name) != digest
        ):
            raise ValueError("backbone artifact mismatch")
    return doc


def embedding_rows(folder, ids):
    """Gather immutable untied output rows from pinned safetensors; no model math."""
    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = "lm_head.weight"
    if name not in index:
        raise ValueError("released untied output embedding missing")
    file = index[name]
    if Path(file).name != file:
        raise ValueError("invalid embedding shard")
    with (folder / file).open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > 16 * 1024**2:
            raise ValueError("oversized safetensors header")
        header = json.loads(f.read(size))
        meta = header[name]
        if meta["dtype"] not in ("BF16", "F32") or len(meta["shape"]) != 2:
            raise ValueError("unsupported embedding storage")
        vocab, width = meta["shape"]
        unit = 2 if meta["dtype"] == "BF16" else 4
        begin, end = meta["data_offsets"]
        if end - begin != vocab * width * unit or any(
            type(i) is not int or not 0 <= i < vocab for i in ids
        ):
            raise ValueError("invalid embedding extent")
        rows = []
        for i in ids:
            f.seek(8 + size + begin + i * width * unit)
            raw = f.read(width * unit)
            if len(raw) != width * unit:
                raise ValueError("truncated embedding row")
            row = (
                (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(
                    np.float32
                )
                if unit == 2
                else np.frombuffer(raw, dtype="<f4")
            )
            if not np.isfinite(row).all():
                raise ValueError("nonfinite embedding row")
            rows.append(row)
        return np.stack(rows)


def load_float_weights(path):
    """Decode checked safetensors storage without importing a model framework."""
    from safetensors import deserialize

    path = Path(path)
    if path.stat().st_size > 1024**3:
        raise ValueError("head weight file exceeds bound")
    result = {}
    for name, tensor in deserialize(path.read_bytes()):
        dtype = tensor["dtype"]
        if dtype == "BF16":
            data = (
                np.frombuffer(tensor["data"], dtype="<u2").astype(np.uint32) << 16
            ).view(np.float32)
        elif dtype == "F32":
            data = np.frombuffer(tensor["data"], dtype="<f4").copy()
        else:
            raise ValueError("unsupported head weight dtype: " + dtype)
        if not np.isfinite(data).all():
            raise ValueError("nonfinite head weight")
        result[name] = data.reshape(tensor["shape"])
    return result


def execute_offline_head(native, artifact_id):
    """One checked generated entry, outside the HTTP service's 1-second budget.

    This offline diagnostic makes no deadline/cancellation or service-readiness
    claim. NativeHead has already verified the package, spans and feature data.
    """
    with native.slot:
        if native.closed:
            raise RuntimeError("head is closed")
        chunks, _ = native.features[artifact_id]
        for offset, size in native.clear_ranges:
            C.memset(native.address + offset, 0, size)
        for name, data in chunks.items():
            C.memmove(
                native.address + native.manifest["inputs"][name]["offset"],
                data,
                len(data),
            )
        if native.execute(native.arena, native.manifest["arena_bytes"]):
            raise RuntimeError("generated head execution failed")
        out = native.manifest["output"]
        scores = np.ctypeslib.as_array(
            (C.c_float * out["count"]).from_address(native.address + out["offset"])
        ).copy()
        if not np.isfinite(scores).all():
            raise RuntimeError("nonfinite decision logits")
        return scores


def run(
    release, lock, lock_sha, bundle, bundle_descriptor, bundle_sha, request, output
):
    started = time.perf_counter()
    release = Path(release)
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists")
    dependencies = native_dependencies()
    verify_lock(release, lock, lock_sha)
    identity = verify_release(release)
    bundle_identity = verify_bundle(bundle, bundle_descriptor, bundle_sha, lock_sha)
    from server.decision.contract import decode

    record = decode(Path(request).read_bytes())
    encoded = prepare(Tokenizer(release / "tokenizer.json"), record, max_tokens=512)
    backbone = DecoderFeatures(bundle)
    try:
        hidden = backbone.encode(list(encoded.input_ids))
    finally:
        backbone.close()
    native_seconds = time.perf_counter() - started
    unique = sorted(set(encoded.input_ids))
    rows = embedding_rows(release, unique)
    inputs = compact_inputs(encoded, hidden, rows, np.array(unique))
    output.mkdir(parents=True)
    np.savez(output / "native-features.npz", **inputs)
    np.save(output / "embedding.npy", rows)
    raw = load_float_weights(release / "joint_head.safetensors")
    raw["output_embedding.weight"] = rows
    artifacts = [
        dict(
            id=f"question-{i}",
            inputs=inputs,
            question_index=i,
            candidate_ids=list(q.option_ids),
        )
        for i, q in enumerate(encoded.questions)
    ]
    package = output / "head"
    head_config = json.loads((release / "joint_head_config.json").read_text())
    package_sha = build_package(
        package,
        raw,
        artifacts,
        dict(id="clef-native-diagnostic", revision=identity["revision"]),
        heads=head_config["heads"],
        provenance=dict(
            backbone_executor="cke_generated",
            release_lock_sha256=lock_sha,
            backbone_bundle_sha256=bundle_sha,
            request_sha256=file_sha(request),
        ),
    )
    native = NativeHead(package / "package.json", package_sha)
    results = []
    try:
        scores = execute_offline_head(native, "question-0")
        np.save(output / "native-logits.npy", scores)
        offset = 0
        for i, q in enumerate(encoded.questions):
            count = len(q.option_ids)
            results.append(
                dict(
                    question_id=q.question_id,
                    candidate_ids=list(q.option_ids),
                    logits=scores[offset : offset + count].tolist(),
                )
            )
            offset += count
    finally:
        native.close()
    report = dict(
        schema="cke.clef.native-diagnostic.v1",
        execution="PASS",
        numerical_parity="NOT_TESTED",
        application_quality="NOT_TESTED",
        backbone_executor="cke_generated",
        head_executor="cke_generated",
        frontend="pinned_schema_python_and_tokenizers",
        model=identity["model"],
        revision=identity["revision"],
        release_lock_sha256=lock_sha,
        request_sha256=file_sha(request),
        tokens=len(encoded.input_ids),
        input_ids=list(encoded.input_ids),
        feature_sha256=file_sha(output / "native-features.npz"),
        embedding_sha256=file_sha(output / "embedding.npy"),
        backbone_bundle_sha256=bundle_sha,
        head_package_sha256=package_sha,
        native_math_libraries=dependencies,
        results=results,
        elapsed_seconds=time.perf_counter() - started,
        verify_and_backbone_seconds=native_seconds,
        limitations=[
            "request-shaped offline head build",
            "no native backbone training",
            "not a persistent serving endpoint",
        ],
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "lock", "bundle", "bundle-descriptor", "request", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("lock-sha256", "bundle-sha256"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    r = run(
        a.release,
        a.lock,
        a.lock_sha256,
        a.bundle,
        a.bundle_descriptor,
        a.bundle_sha256,
        a.request,
        a.output,
    )
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
