#!/usr/bin/env python3
"""Certify the generated two-row decode step on one identified KV-only bundle.

This is an artifact-specific local lane, not a generic batching claim. It
compares full logits and complete KV bytes with isolated execution and retains
rejected-step/recovery evidence. The same loaded library serves every case.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from server.serving_bundle import verified_loaded_symbol_backing


CAP_BATCH_TWO_ROWS = 1 << 17
TOKENS_A = (100, 101, 102, 103, 104, 105)
TOKENS_B = (200, 201, 202)
TOKENS_C = (300, 301)


class BatchRow(ctypes.Structure):
    _fields_ = [
        ("sequence_handle", ctypes.c_uint64),
        ("token", ctypes.c_int32),
        ("position", ctypes.c_int32),
        ("row_offset", ctypes.c_uint32),
        ("token_count", ctypes.c_uint32),
        ("logits", ctypes.POINTER(ctypes.c_float)),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _aligned(size: int, alignment: int) -> tuple[ctypes.Array, ctypes.c_void_p]:
    storage = ctypes.create_string_buffer(size + alignment - 1)
    address = (ctypes.addressof(storage) + alignment - 1) & ~(alignment - 1)
    return storage, ctypes.c_void_p(address)


def _bind(model: ctypes.CDLL) -> None:
    model.ck_model_init.argtypes = [ctypes.c_char_p]
    model.ck_model_init.restype = ctypes.c_int
    model.ck_model_free.argtypes = []
    model.ck_model_get_capabilities.restype = ctypes.c_uint64
    model.ck_model_get_vocab_size.restype = ctypes.c_int
    model.ck_model_get_context_window.restype = ctypes.c_int
    model.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    model.ck_model_decode.restype = ctypes.c_int
    model.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    model.ck_model_embed_tokens.restype = ctypes.c_int
    model.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    model.ck_model_forward.restype = ctypes.c_int
    model.ck_model_kv_cache_reset.argtypes = []
    model.ck_model_sequence_state_default.restype = ctypes.c_uint64
    model.ck_model_sequence_state_requirements.argtypes = [
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    model.ck_model_sequence_state_requirements.restype = ctypes.c_int
    model.ck_model_sequence_state_create.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
    model.ck_model_sequence_state_create.restype = ctypes.c_int
    model.ck_model_sequence_state_activate.argtypes = [ctypes.c_uint64]
    model.ck_model_sequence_state_activate.restype = ctypes.c_int
    model.ck_model_sequence_state_destroy.argtypes = [ctypes.c_uint64]
    model.ck_model_sequence_state_destroy.restype = ctypes.c_int
    model.ck_model_get_named_activation_ptr.argtypes = [ctypes.c_char_p]
    model.ck_model_get_named_activation_ptr.restype = ctypes.c_size_t
    model.ck_model_batch_decode_workspace.argtypes = [
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    model.ck_model_batch_decode_workspace.restype = ctypes.c_int
    if hasattr(model, "ck_model_batch_decode_projection_groups"):
        model.ck_model_batch_decode_projection_groups.restype = ctypes.c_int
    model.ck_model_decode_batch2.argtypes = [
        ctypes.POINTER(BatchRow), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    model.ck_model_decode_batch2.restype = ctypes.c_int
    model.ck_model_set_cancel_flag.argtypes = [ctypes.POINTER(ctypes.c_int)]
    model.ck_model_generated_source_sha256.restype = ctypes.c_char_p


def certify(bundle: Path, generated_source: Path) -> dict:
    assets = {name: bundle / name for name in (
        "libmodel.so", "libckernel_engine.so", "libckernel_tokenizer.so",
        "weights.bump", "layout_decode.json", "config.json")}
    if missing := [name for name, path in assets.items() if not path.is_file()]:
        raise ValueError(f"missing bundle assets: {missing}")
    model = ctypes.CDLL(str(assets["libmodel.so"]), mode=ctypes.RTLD_GLOBAL)
    _bind(model)
    if model.ck_model_init(str(assets["weights.bump"]).encode()) != 0:
        raise RuntimeError("generated model initialization failed")
    try:
        projection_groups = (
            int(model.ck_model_batch_decode_projection_groups())
            if hasattr(model, "ck_model_batch_decode_projection_groups") else 1
        )
        if projection_groups not in (1, 2):
            raise AssertionError("invalid generated batch projection-group count")
        loaded = {}
        for name, symbol in (("libmodel.so", "ck_model_decode_batch2"),
                             ("libckernel_engine.so", "gemm_nt_q5_1_q8_1_m2"),
                             ("libckernel_tokenizer.so", "ck_tokenizer_encode")):
            identity = verified_loaded_symbol_backing(model, symbol, assets[name])
            if identity["sha256"] != _sha256(assets[name]):
                raise AssertionError(f"loaded {name} differs from bundle file")
            loaded[name] = identity
        source = generated_source.read_bytes()
        marker = b'\n\nCK_EXPORT const char *ck_model_generated_source_sha256(void) {\n    return "'
        prefix, found, tail = source.rpartition(marker)
        claimed = model.ck_model_generated_source_sha256()
        if (not found or not claimed or
                tail != claimed + b'";\n}\n' or
                hashlib.sha256(prefix).hexdigest().encode() != claimed):
            raise AssertionError("loaded generated source identity mismatch")
        if not int(model.ck_model_get_capabilities()) & CAP_BATCH_TWO_ROWS:
            raise AssertionError("generated model did not advertise two-row decode")
        vocab = int(model.ck_model_get_vocab_size())
        context = int(model.ck_model_get_context_window())
        capabilities = int(model.ck_model_get_capabilities())
        if vocab <= max(*TOKENS_A, *TOKENS_B, *TOKENS_C) or context < 8:
            raise ValueError("bundle cannot execute the pinned token fixture")
        kv_bytes, kv_alignment = ctypes.c_size_t(), ctypes.c_size_t()
        if model.ck_model_sequence_state_requirements(
            ctypes.byref(kv_bytes), ctypes.byref(kv_alignment)) != 0:
            raise RuntimeError("KV requirements unavailable")
        work_bytes, work_alignment = ctypes.c_size_t(), ctypes.c_size_t()
        if model.ck_model_batch_decode_workspace(
            ctypes.byref(work_bytes), ctypes.byref(work_alignment)) != 0:
            raise RuntimeError("batch workspace requirements unavailable")
        if kv_alignment.value != 64 or work_alignment.value != 64:
            raise AssertionError("unexpected planner alignment")
        work_storage, work = _aligned(work_bytes.value, work_alignment.value)
        _ = work_storage
        output = (ctypes.c_float * vocab)()
        reference: dict[str, list[tuple[bytes, str]]] = {}
        isolated_ms = 0.0

        def kv_digest() -> str:
            address = model.ck_model_get_named_activation_ptr(b"kv_cache")
            if not address:
                raise AssertionError("active KV pointer unavailable")
            return hashlib.sha256(ctypes.string_at(address, kv_bytes.value)).hexdigest()

        def decode(token: int) -> bytes:
            begin = time.perf_counter()
            if model.ck_model_decode(token, output) != 0:
                raise RuntimeError(f"isolated decode failed for token {token}")
            nonlocal isolated_ms
            isolated_ms += (time.perf_counter() - begin) * 1000
            if any(not math.isfinite(value) for value in output):
                raise AssertionError("isolated logits contain nonfinite values")
            return bytes(output)

        def prefill(tokens: tuple[int, ...]) -> bytes:
            values = (ctypes.c_int32 * len(tokens))(*tokens)
            if model.ck_model_embed_tokens(values, len(tokens)) != 0:
                raise RuntimeError("generated prefill failed")
            if model.ck_model_forward(output) != 0:
                raise RuntimeError("generated prefill output unavailable")
            return bytes(output)

        token_sets = {"a": TOKENS_A, "b": TOKENS_B, "c": TOKENS_C}
        prefill_lengths = {"a": 2, "b": 1, "c": 0}
        for name, tokens in token_sets.items():
            model.ck_model_kv_cache_reset()
            rows: list[tuple[bytes, str]] = []
            length = prefill_lengths[name]
            if length:
                prefill(tokens[:length])
                # Only the last prefill output is needed as a continuation
                # control; the batch comparisons begin after this point.
                rows.extend([(b"", "")] * (length - 1))
                rows.append((bytes(output), kv_digest()))
            for token in tokens[length:]:
                rows.append((decode(token), kv_digest()))
            reference[name] = rows

        default = model.ck_model_sequence_state_default()
        if not default:
            raise RuntimeError("default handle unavailable")
        arenas: dict[str, ctypes.Array] = {}
        handles: dict[str, int] = {}

        def create(name: str) -> int:
            storage, address = _aligned(kv_bytes.value, kv_alignment.value)
            handle = ctypes.c_uint64()
            if model.ck_model_sequence_state_create(address, kv_bytes, ctypes.byref(handle)) != 0:
                raise RuntimeError(f"could not create sequence {name}")
            arenas[name], handles[name] = storage, handle.value
            return handle.value

        create("a")
        create("b")
        comparisons: list[dict] = []
        batch_ms = 0.0

        def activate(name: str) -> None:
            if model.ck_model_sequence_state_activate(handles[name]) != 0:
                raise RuntimeError(f"could not activate sequence {name}")

        def compare(name: str, index: int, actual: bytes) -> None:
            expected, expected_kv = reference[name][index]
            if len(actual) != len(expected):
                raise AssertionError("logit length changed")
            got = (ctypes.c_float * vocab).from_buffer_copy(actual)
            want = (ctypes.c_float * vocab).from_buffer_copy(expected)
            if any(not math.isfinite(value) for value in got):
                raise AssertionError("batch logits contain nonfinite values")
            max_abs = max(abs(x - y) for x, y in zip(got, want))
            if max_abs > 1e-4:
                raise AssertionError(f"{name}[{index}] logit error {max_abs}")
            activate(name)
            if kv_digest() != expected_kv:
                raise AssertionError(f"{name}[{index}] KV cache diverged")
            comparisons.append({"sequence": name, "step": index,
                                "max_absolute_logit_error": max_abs,
                                "exact_logits": actual == expected,
                                "exact_kv": True})

        def batch(order: tuple[tuple[str, int], tuple[str, int]]) -> list[bytes]:
            nonlocal batch_ms
            destinations = [(ctypes.c_float * vocab)() for _ in range(2)]
            rows = (BatchRow * 2)(*[
                BatchRow(handles[name], token_sets[name][index], index, row, 1,
                         destinations[row])
                for row, (name, index) in enumerate(order)
            ])
            begin = time.perf_counter()
            status = model.ck_model_decode_batch2(rows, 2, work, work_bytes)
            batch_ms += (time.perf_counter() - begin) * 1000
            if status != 0:
                raise RuntimeError(f"generated batch step failed: {status}")
            return [bytes(row) for row in destinations]

        try:
            activate("a")
            prefill(TOKENS_A[:2])
            activate("b")
            prefill(TOKENS_B[:1])
            # The default sequence is a valid participant, but B is an idle
            # live arena for this probe. Its bytes must never be workspace.
            idle_kv = kv_digest()
            idle_outputs = [(ctypes.c_float * vocab)() for _ in range(2)]
            idle_rows = (BatchRow * 2)(
                BatchRow(default, TOKENS_C[0], 2, 0, 1, idle_outputs[0]),
                BatchRow(handles["a"], TOKENS_A[2], 2, 1, 1, idle_outputs[1]),
            )
            idle_address = (ctypes.addressof(arenas["b"]) + kv_alignment.value - 1) & ~(kv_alignment.value - 1)
            if model.ck_model_decode_batch2(idle_rows, 2, ctypes.c_void_p(idle_address),
                                            work_bytes) != -2 or kv_digest() != idle_kv:
                raise AssertionError("nonparticipating live KV arena alias was accepted or modified")
            for name, index, result in zip(
                ("a", "b"), (2, 1), batch((("a", 2), ("b", 1)))):
                compare(name, index, result)
            for name, index, result in zip(
                ("b", "a"), (2, 3), batch((("b", 2), ("a", 3)))):
                compare(name, index, result)

            # B has finished; A continues by the original one-row path.
            activate("a")
            compare("a", 4, decode(TOKENS_A[4]))

            # Retire B, reuse its caller arena for C, then batch A with C.
            activate("a")
            retired = handles["b"]
            if model.ck_model_sequence_state_destroy(retired) != 0:
                raise RuntimeError("could not retire B")
            if model.ck_model_sequence_state_activate(retired) != -1:
                raise AssertionError("stale B handle was accepted")
            old = arenas.pop("b")
            new_handle = ctypes.c_uint64()
            old_addr = (ctypes.addressof(old) + kv_alignment.value - 1) & ~(kv_alignment.value - 1)
            if model.ck_model_sequence_state_create(ctypes.c_void_p(old_addr), kv_bytes,
                                                     ctypes.byref(new_handle)) != 0:
                raise RuntimeError("retired slot could not be reused")
            arenas["c"], handles["c"] = old, new_handle.value
            if handles["c"] == retired:
                raise AssertionError("retired handle generation was reused")

            # Cancellation and capacity rejection must leave A and C ready.
            outputs = [(ctypes.c_float * vocab)(*([17.0] * vocab)) for _ in range(2)]
            rows = (BatchRow * 2)(
                BatchRow(handles["a"], TOKENS_A[5], 5, 0, 1, outputs[0]),
                BatchRow(handles["c"], TOKENS_C[0], 0, 1, 1, outputs[1]),
            )
            cancel = ctypes.c_int(1)
            model.ck_model_set_cancel_flag(ctypes.byref(cancel))
            if model.ck_model_decode_batch2(rows, 2, work, work_bytes) != -2:
                raise AssertionError("pre-step cancellation was not rejected")
            model.ck_model_set_cancel_flag(None)
            rows[0].position = context
            if model.ck_model_decode_batch2(rows, 2, work, work_bytes) != -2:
                raise AssertionError("context-capacity request was not rejected")
            rows[0].position = 5
            rows[0].logits = ctypes.cast(work, ctypes.POINTER(ctypes.c_float))
            if model.ck_model_decode_batch2(rows, 2, work, work_bytes) != -2:
                raise AssertionError("workspace/output alias was accepted")
            rows[0].logits = outputs[0]
            kv_alias = (ctypes.addressof(arenas["a"]) + kv_alignment.value - 1) & ~(kv_alignment.value - 1)
            if model.ck_model_decode_batch2(rows, 2, ctypes.c_void_p(kv_alias),
                                            work_bytes) != -2:
                raise AssertionError("workspace/KV alias was accepted")
            if outputs[0][0] != 17.0 or outputs[1][0] != 17.0:
                raise AssertionError("rejected batch modified output buffers")
            for name, index, result in zip(
                ("a", "c"), (5, 0), batch((("a", 5), ("c", 0)))):
                compare(name, index, result)
        finally:
            model.ck_model_set_cancel_flag(None)
            model.ck_model_sequence_state_activate(default)
            for name in ("a", "b", "c"):
                if name in arenas and name in handles:
                    model.ck_model_sequence_state_destroy(handles[name])
    finally:
        model.ck_model_free()

    return {
        "schema": "cke.generated-batch-decode-v1",
        "status": "pass",
        "scope": ("two_rows_shared_first_layer_qk_and_gateup_kv_only_not_continuous_batching"
                  if projection_groups == 2 else
                  "two_rows_shared_first_layer_qk_kv_only_not_continuous_batching"),
        "shared_projection_groups": projection_groups,
        "capabilities": capabilities,
        "compiled_context_length": context,
        "vocab_size": vocab,
        "full_logit_and_kv_comparisons": comparisons,
        "batch_step_total_ms": round(batch_ms, 3),
        "isolated_decode_total_ms": round(isolated_ms, 3),
        "timing_scope": "diagnostic_nonmatched_workloads_no_speedup_claim",
        "kv_bytes_per_sequence": kv_bytes.value,
        "batch_workspace_bytes": work_bytes.value,
        "artifacts": {name: _sha256(path) for name, path in assets.items()},
        "loaded_libraries": loaded,
        "loaded_generated_source_prefix_sha256": claimed.decode(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--generated-source", type=Path, required=True,
                        help="C source whose prefix digest must match the loaded generated-library symbol")
    args = parser.parse_args()
    result = certify(args.bundle.resolve(), args.generated_source.resolve())
    result["generated_source_sha256"] = _sha256(args.generated_source.resolve())
    body = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(body, encoding="utf-8")
    print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
