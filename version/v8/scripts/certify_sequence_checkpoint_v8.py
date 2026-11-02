#!/usr/bin/env python3
"""Compare a loaded model's fresh and checkpoint-restored continuations.

This is an opt-in, asset-backed diagnostic. The suffix is decoded one token at
a time on both sides; it does not certify arbitrary chunked-prefill splits.
"""

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from sequence_state_contract_v8 import validate_sequence_reset_regions


STATE_NAMES = ("kv_cache", "recurrent_conv_state", "recurrent_ssm_state")


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_memory(address, size):
    h = hashlib.sha256()
    for offset in range(0, size, 16 << 20):
        h.update(C.string_at(address + offset, min(16 << 20, size - offset)))
    return h.hexdigest()


def declared_state_regions(layout):
    rows = validate_sequence_reset_regions(layout)
    required = {}
    for row in rows:
        name, size = row["name"], row["size"]
        if name not in STATE_NAMES or type(size) is not int or size <= 0 or name in required:
            raise RuntimeError(f"unsupported or unproven sequence-state region: {name}")
        required[name] = size
    if "kv_cache" not in required:
        raise RuntimeError("selected artifact declares no KV sequence state")
    if ("recurrent_conv_state" in required) != ("recurrent_ssm_state" in required):
        raise RuntimeError("incomplete recurrent sequence-state pair")
    return required


def signature(model, logits, required_state):
    if any(not math.isfinite(value) for value in logits):
        raise RuntimeError("nonfinite logits")
    state = {}
    for name, required_bytes in required_state.items():
        pointer = model.ck_model_get_named_activation_ptr(name.encode())
        extent = model.ck_model_get_named_activation_nbytes(name.encode())
        if not pointer or extent != required_bytes:
            raise RuntimeError(f"{name} missing or wrong extent: {extent}, expected {required_bytes}")
        state[name] = {"bytes": extent, "sha256": sha_memory(pointer, extent)}
    return {
        "position": model.ck_model_get_active_tokens(),
        "logits_sha256": hashlib.sha256(bytes(logits)).hexdigest(),
        "state": state,
    }


def configure(model):
    model.ck_model_init.argtypes = [C.c_char_p]
    model.ck_model_init.restype = C.c_int
    model.ck_model_sequence_checkpoint_bind_instance.argtypes = [C.c_uint64, C.c_uint64]
    model.ck_model_sequence_checkpoint_bind_instance.restype = C.c_int
    model.ck_model_sequence_state_default.restype = C.c_uint64
    model.ck_model_sequence_state_requirements.argtypes = [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    model.ck_model_sequence_state_requirements.restype = C.c_int
    model.ck_model_sequence_state_create.argtypes = [C.c_void_p, C.c_size_t,
                                                     C.POINTER(C.c_uint64)]
    model.ck_model_sequence_state_create.restype = C.c_int
    model.ck_model_sequence_state_activate.argtypes = [C.c_uint64]
    model.ck_model_sequence_state_activate.restype = C.c_int
    model.ck_model_sequence_state_destroy.argtypes = [C.c_uint64]
    model.ck_model_sequence_state_destroy.restype = C.c_int
    model.ck_model_sequence_checkpoint_requirements.argtypes = [C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    model.ck_model_sequence_checkpoint_requirements.restype = C.c_int
    model.ck_model_sequence_checkpoint_save.argtypes = [C.c_uint64, C.c_void_p, C.c_size_t]
    model.ck_model_sequence_checkpoint_save.restype = C.c_int
    model.ck_model_sequence_checkpoint_restore.argtypes = [C.c_uint64, C.c_void_p, C.c_size_t]
    model.ck_model_sequence_checkpoint_restore.restype = C.c_int
    model.ck_model_embed_tokens.argtypes = [C.POINTER(C.c_int32), C.c_int]
    model.ck_model_embed_tokens.restype = C.c_int
    model.ck_model_forward.argtypes = [C.POINTER(C.c_float)]
    model.ck_model_forward.restype = C.c_int
    model.ck_model_decode.argtypes = [C.c_int32, C.POINTER(C.c_float)]
    model.ck_model_decode.restype = C.c_int
    model.ck_model_get_active_tokens.restype = C.c_int
    model.ck_model_get_vocab_size.restype = C.c_int
    model.ck_model_get_named_activation_ptr.argtypes = [C.c_char_p]
    model.ck_model_get_named_activation_ptr.restype = C.c_size_t
    model.ck_model_get_named_activation_nbytes.argtypes = [C.c_char_p]
    model.ck_model_get_named_activation_nbytes.restype = C.c_ssize_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True, help="JSON array of exact token IDs")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--suffix-tokens", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ids = json.loads(args.ids.read_text())
    if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
        parser.error("--ids must contain nonempty, nonnegative integer token IDs")
    if args.suffix_tokens < 1 or any(p < 1 or p + args.suffix_tokens > len(ids)
                                      for p in args.prefix_lengths):
        parser.error("prefix lengths require at least one prefix and the requested suffix")
    bundle = args.bundle.resolve()
    model_path = bundle / "libmodel.so"
    engine_path = bundle / "libckernel_engine.so"
    model = C.CDLL(str(model_path), mode=C.RTLD_GLOBAL)
    C.CDLL(str(engine_path), mode=C.RTLD_GLOBAL)
    loaded = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if len(line.split()) >= 6}
    if str(model_path) not in loaded or str(engine_path) not in loaded:
        raise RuntimeError("loaded model/engine library identity mismatch")
    configure(model)
    required_state = declared_state_regions(
        json.loads((bundle / "layout_decode.json").read_text()))
    model.ck_model_get_capabilities.restype = C.c_uint64
    if not model.ck_model_get_capabilities() & (1 << 18):
        raise RuntimeError("loaded model does not advertise sequence checkpoints")
    if model.ck_model_init(str(bundle / "weights.bump").encode()) != 0:
        raise RuntimeError("model initialization failed")
    report = {
        "schema": "cke.sequence_checkpoint_parity.v1",
        "source_commit": os.environ.get("CK_SOURCE_COMMIT"),
        "bundle": str(bundle),
        "ids_sha256": sha_file(args.ids),
        "artifact": {name: sha_file(bundle / name) for name in
                     ("libmodel.so", "libckernel_engine.so", "model_v8.c",
                      "weights_manifest.json", "config.json", "weights.bump")},
        "probe_sha256": sha_file(Path(__file__)),
        "required_state": required_state,
        "loaded_libraries": sorted(path for path in loaded if path in
                                   (str(model_path), str(engine_path))),
        "threads": {name: os.environ.get(name) for name in
                    ("CK_NUM_THREADS", "OMP_NUM_THREADS")},
        "cpu": next((line.split(":", 1)[1].strip()
                     for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), None),
        "prefill_policy": json.loads((bundle / "config.json").read_text()).get("prefill_policy"),
        "context_capacity": json.loads((bundle / "layout_decode.json").read_text())
                            ["config"]["context_length"],
        "cases": [],
    }
    try:
        nonce = os.urandom(16)
        hi, lo = int.from_bytes(nonce[:8], "little"), int.from_bytes(nonce[8:], "little")
        if model.ck_model_sequence_checkpoint_bind_instance(hi, lo) != 0:
            raise RuntimeError("checkpoint instance binding failed")
        handle = model.ck_model_sequence_state_default()
        size, alignment = C.c_size_t(), C.c_size_t()
        if not handle or model.ck_model_sequence_checkpoint_requirements(
                C.byref(size), C.byref(alignment)) != 0:
            raise RuntimeError("checkpoint capability unavailable")
        if not size.value or alignment.value < 8 or alignment.value & (alignment.value - 1):
            raise RuntimeError("invalid checkpoint storage requirements")
        storage = C.create_string_buffer(size.value + alignment.value)
        snapshot = (C.addressof(storage) + alignment.value - 1) & ~(alignment.value - 1)
        vocab = model.ck_model_get_vocab_size()
        if vocab <= 0:
            raise RuntimeError("invalid vocabulary size")
        logits = (C.c_float * vocab)()

        def prefill(tokens):
            token_array = (C.c_int32 * len(tokens))(*tokens)
            if model.ck_model_embed_tokens(token_array, len(tokens)) != 0 or \
                    model.ck_model_forward(logits) != 0:
                raise RuntimeError("prefill failed")

        first_baseline = None
        for prefix_length in args.prefix_lengths:
            prefix = ids[:prefix_length]
            suffix = ids[prefix_length:prefix_length + args.suffix_tokens]
            started = time.monotonic()
            model.ck_model_kv_cache_reset()
            prefill(prefix)
            baseline_prefix = signature(model, logits, required_state)
            baseline = []
            for token in suffix:
                if model.ck_model_decode(token, logits) != 0:
                    raise RuntimeError("fresh decode failed")
                baseline.append(signature(model, logits, required_state))
            if first_baseline is None:
                first_baseline = baseline
            model.ck_model_kv_cache_reset()
            prefill(prefix)
            if model.ck_model_sequence_checkpoint_save(handle, snapshot, size.value) != 0:
                raise RuntimeError("checkpoint save failed")
            model.ck_model_kv_cache_reset()
            if model.ck_model_sequence_checkpoint_restore(handle, snapshot, size.value) != 0:
                raise RuntimeError("checkpoint restore failed")
            restored_prefix = signature(model, logits, required_state)
            # Logits are intentionally absent from the snapshot; compare state only.
            restored_prefix.pop("logits_sha256")
            expected_prefix = dict(baseline_prefix)
            expected_prefix.pop("logits_sha256")
            matches = restored_prefix == expected_prefix
            steps = []
            for token, expected in zip(suffix, baseline):
                if model.ck_model_decode(token, logits) != 0:
                    raise RuntimeError("restored decode failed")
                actual = signature(model, logits, required_state)
                steps.append({"token": token, "exact": actual == expected,
                              "expected": expected, "actual": actual})
                matches &= actual == expected
            report["cases"].append({"prefix_tokens": prefix_length,
                                    "suffix_tokens": len(suffix),
                                    "prefix_state_exact": restored_prefix == expected_prefix,
                                    "steps": steps,
                                    "exact": bool(matches),
                                    "elapsed_seconds": time.monotonic() - started})
            print(f"prefix={prefix_length} exact={matches}", flush=True)

        # One loaded model, two sequence arenas. A's saved prefix must survive
        # B's distinct work, and a rejected restore must not change B.
        a_length = args.prefix_lengths[0]
        b_length = min(a_length + 2, len(ids) - args.suffix_tokens - 2)
        if b_length < 1:
            raise RuntimeError("not enough token IDs for A/B/A isolation")
        arena_bytes, arena_alignment = C.c_size_t(), C.c_size_t()
        if model.ck_model_sequence_state_requirements(
                C.byref(arena_bytes), C.byref(arena_alignment)) != 0:
            raise RuntimeError("second-sequence requirements unavailable")
        arena_storage = C.create_string_buffer(arena_bytes.value + arena_alignment.value)
        arena = (C.addressof(arena_storage) + arena_alignment.value - 1) & \
                ~(arena_alignment.value - 1)
        model.ck_model_kv_cache_reset()
        prefill(ids[:a_length])
        if model.ck_model_sequence_checkpoint_save(handle, snapshot, size.value) != 0:
            raise RuntimeError("A checkpoint save failed")
        other = C.c_uint64()
        if model.ck_model_sequence_state_create(
                arena, arena_bytes.value, C.byref(other)) != 0 or not other.value:
            raise RuntimeError("second-sequence creation failed")
        try:
            if model.ck_model_sequence_state_activate(other.value) != 0:
                raise RuntimeError("B activation failed")
            model.ck_model_kv_cache_reset()
            prefill(ids[2:2 + b_length])
            if model.ck_model_decode(ids[2 + b_length], logits) != 0:
                raise RuntimeError("B continuation failed")
            b_before = signature(model, logits, required_state)
            C.c_uint8.from_address(snapshot).value ^= 1
            rejected = model.ck_model_sequence_checkpoint_restore(
                other.value, snapshot, size.value) != 0
            b_after_reject = signature(model, logits, required_state)
            C.c_uint8.from_address(snapshot).value ^= 1
            if model.ck_model_sequence_state_activate(handle) != 0 or \
                    model.ck_model_sequence_checkpoint_restore(
                        handle, snapshot, size.value) != 0:
                raise RuntimeError("A restore after B failed")
            a_steps = []
            for token, expected in zip(
                    ids[a_length:a_length + args.suffix_tokens], first_baseline):
                if model.ck_model_decode(token, logits) != 0:
                    raise RuntimeError("A survivor continuation failed")
                a_steps.append(signature(model, logits, required_state) == expected)
            if model.ck_model_sequence_state_activate(other.value) != 0:
                raise RuntimeError("B reactivation failed")
            b_after_a = signature(model, logits, required_state)
            b_after_a.pop("logits_sha256")
            b_expected = dict(b_before)
            b_expected.pop("logits_sha256")
            isolation = {
                "rejected_restore": rejected,
                "rejected_restore_kept_b": b_before == b_after_reject,
                "a_continuation_exact": all(a_steps),
                "b_survived_a": b_after_a == b_expected,
                "b_expected": b_expected,
                "b_after_a": b_after_a,
            }
            isolation["exact"] = all(isolation[key] for key in
                                     ("rejected_restore", "rejected_restore_kept_b",
                                      "a_continuation_exact", "b_survived_a"))
            report["isolation"] = isolation
            print(f"A/B/A isolation exact={isolation['exact']}", flush=True)
        finally:
            model.ck_model_sequence_state_destroy(other.value)
    finally:
        model.ck_model_free()
        report["verdict"] = "PASS" if report["cases"] and all(
            case["exact"] for case in report["cases"]) and \
            report.get("isolation", {}).get("exact") else "FAIL"
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
