#!/usr/bin/env python3
"""Compare ordinary full prefill with a restored prefix plus multi-token suffix.

This opt-in, asset-backed probe uses exact token IDs and one loaded generated
model. It does not turn a passing artifact into a generic append capability.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

from certify_sequence_checkpoint_v8 import (
    configure, declared_state_regions, sha_file, signature,
)


def _difference(before: bytes, after: bytes, dtype: str) -> dict:
    import numpy as np

    left = np.frombuffer(before, dtype=dtype)
    right = np.frombuffer(after, dtype=dtype)
    if left.shape != right.shape:
        raise RuntimeError("state comparison extents differ")
    changed = left.view(np.uint8) != right.view(np.uint8)
    if not changed.any():
        return {"different_bytes": 0, "first_byte": None, "max_abs": 0.0}
    max_abs = 0.0
    for start in range(0, len(left), 1 << 20):
        stop = min(start + (1 << 20), len(left))
        delta = np.abs(left[start:stop].astype(np.float32) -
                       right[start:stop].astype(np.float32))
        max_abs = max(max_abs, float(np.max(delta)))
    return {"different_bytes": int(np.count_nonzero(changed)),
            "first_byte": int(np.argmax(changed)), "max_abs": max_abs}


def certify(bundle: Path, ids: list[int], prefixes: list[int], suffixes: list[int],
            diagnose: bool = False, capture_root: Path | None = None,
            capture_layer: int = 0, on_case=None) -> dict:
    model_path = bundle / "libmodel.so"
    engine_path = bundle / "libckernel_engine.so"
    model = C.CDLL(str(model_path), mode=C.RTLD_GLOBAL)
    C.CDLL(str(engine_path), mode=C.RTLD_GLOBAL)
    loaded = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if len(line.split()) >= 6}
    if str(model_path) not in loaded or str(engine_path) not in loaded:
        raise RuntimeError("loaded model/engine library identity mismatch")
    configure(model)
    model.ck_model_free.restype = None
    model.ck_model_kv_cache_reset.restype = None
    model.ck_model_prefill_append_row_group.restype = C.c_int
    model.ck_model_prefill_append_min_prefix.restype = C.c_int
    model.ck_model_prefill_append_alignment.restype = C.c_int
    required = declared_state_regions(json.loads((bundle / "layout_decode.json").read_text()))
    group = model.ck_model_prefill_append_row_group()
    minimum_prefix = model.ck_model_prefill_append_min_prefix()
    alignment = model.ck_model_prefill_append_alignment()
    if group < 1:
        raise RuntimeError("invalid selected prefill row group")
    if minimum_prefix < 0:
        raise RuntimeError("invalid selected minimum prefix")
    if alignment < 1:
        raise RuntimeError("invalid selected append alignment")
    capacity = json.loads((bundle / "layout_decode.json").read_text())["config"]["context_length"]
    if any(prefix + suffix > capacity for prefix in prefixes for suffix in suffixes):
        raise ValueError("requested test exceeds compiled context capacity")
    if model.ck_model_init(str(bundle / "weights.bump").encode()) != 0:
        raise RuntimeError("generated model initialization failed")
    try:
        nonce = os.urandom(16)
        if model.ck_model_sequence_checkpoint_bind_instance(
                int.from_bytes(nonce[:8], "little"),
                int.from_bytes(nonce[8:], "little")) != 0:
            raise RuntimeError("checkpoint instance binding failed")
        handle = model.ck_model_sequence_state_default()
        size, align = C.c_size_t(), C.c_size_t()
        if not handle or model.ck_model_sequence_checkpoint_requirements(
                C.byref(size), C.byref(align)) != 0:
            raise RuntimeError("checkpoint requirements unavailable")
        if not size.value or not align.value or align.value & (align.value - 1):
            raise RuntimeError("invalid checkpoint storage requirements")
        storage = C.create_string_buffer(size.value + align.value)
        snapshot = (C.addressof(storage) + align.value - 1) & ~(align.value - 1)
        vocab = model.ck_model_get_vocab_size()
        if vocab <= 0:
            raise RuntimeError("invalid vocabulary size")
        logits = (C.c_float * vocab)()

        def embed(token_ids: list[int]) -> int:
            values = (C.c_int32 * len(token_ids))(*token_ids)
            return model.ck_model_embed_tokens(values, len(token_ids))

        def capture() -> dict:
            if model.ck_model_forward(logits) != 0:
                raise RuntimeError("forward failed")
            return signature(model, logits, required)

        def values() -> dict:
            return {"logits": bytes(logits), "state": {
                name: C.string_at(model.ck_model_get_named_activation_ptr(name.encode()), size)
                for name, size in required.items()}}

        cases = []

        def capture_stage(prefix: int, suffix: int, stage: str) -> None:
            if capture_root is None:
                return
            destination = capture_root / f"p{prefix}-s{suffix}" / stage
            destination.mkdir(parents=True, exist_ok=True)
            os.environ["CK_DEBUG_EXPORT_HIDDEN"] = str(destination)
            os.environ["CK_DEBUG_EXPORT_HIDDEN_LAYER"] = str(capture_layer)
            os.environ["CK_DEBUG_EXPORT_HIDDEN_NAMES"] = (
                "layer_input,attn_norm,linear_attn_qkv_mixed,z,alpha,beta,"
                "gate,conv_input,conv_output_raw,conv_output_silu,q_conv,"
                "k_conv,v_conv_predelta,q_conv_predelta,k_conv_predelta,"
                "state_predelta,attn_output,new_state,final_output,"
                "linear_attn_out,after_attn,layer_out,q_proj,k_proj,v_proj,"
                "q_proj_last,k_proj_last,v_proj_last,rope_q,rope_k,"
                "stored_key,stored_value,attn_gate,qk_norm_q,qk_norm_k,"
                "attn_pregate,attn_out,out_proj")

        for prefix in prefixes:
            for suffix in suffixes:
                if prefix <= 0 or suffix <= 1 or prefix + suffix > len(ids):
                    raise ValueError("invalid prefix/suffix geometry")
                full_ids = ids[:prefix + suffix]
                model.ck_model_kv_cache_reset()
                capture_stage(prefix, suffix, "full")
                if embed(full_ids) != 0:
                    raise RuntimeError("ordinary full prefill failed")
                full = capture()
                full_values = values() if diagnose else None
                model.ck_model_kv_cache_reset()
                capture_stage(prefix, suffix, "prefix")
                if embed(ids[:prefix]) != 0:
                    raise RuntimeError("prefix prefill failed")
                prefix_before = capture()
                if model.ck_model_sequence_checkpoint_save(handle, snapshot, size.value) != 0:
                    raise RuntimeError("prefix checkpoint save failed")
                capture_stage(prefix, suffix, "append")
                append_status = embed(ids[prefix:prefix + suffix])
                if prefix % group or prefix < minimum_prefix or prefix % alignment:
                    unchanged = prefix_before == capture()
                    cases.append({"prefix": prefix, "suffix": suffix,
                                  "append_status": append_status,
                                  "unsupported_split_rejected_unchanged": append_status == -3 and unchanged})
                    if on_case:
                        on_case(cases)
                    continue
                if append_status != 0:
                    raise RuntimeError(f"aligned append failed: {append_status}")
                appended = capture()
                appended_values = values() if diagnose else None
                model.ck_model_kv_cache_reset()
                if model.ck_model_sequence_checkpoint_restore(handle, snapshot, size.value) != 0:
                    raise RuntimeError("prefix checkpoint restore failed")
                capture_stage(prefix, suffix, "restored")
                if embed(ids[prefix:prefix + suffix]) != 0:
                    raise RuntimeError("restored append failed")
                restored = capture()
                case = {"prefix": prefix, "suffix": suffix,
                              "append_status": 0, "full": full,
                              "appended": appended, "restored": restored,
                              "full_equals_append": full == appended,
                              "append_equals_restore": appended == restored}
                if diagnose:
                    case["difference"] = {
                        "logits": _difference(full_values["logits"],
                                              appended_values["logits"], "<f4"),
                        "state": {name: _difference(full_values["state"][name],
                                                    appended_values["state"][name],
                                                    "<f2" if name == "kv_cache" else "<f4")
                                  for name in required}}
                cases.append(case)
                if on_case:
                    on_case(cases)
        passed = all(case.get("unsupported_split_rejected_unchanged", False) if
                     case["prefix"] % group or case["prefix"] < minimum_prefix or
                     case["prefix"] % alignment else
                     case["full_equals_append"] and case["append_equals_restore"]
                     for case in cases)
        return {"schema": "cke.checkpoint_prefill_equivalence.v1",
                "verdict": "PASS" if passed else "FAIL", "row_group": group,
                "minimum_prefix": minimum_prefix, "append_alignment": alignment,
                "context_capacity": capacity,
                "bundle": str(bundle),
                "ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                "artifact": {name: sha_file(bundle / name) for name in
                             ("libmodel.so", "libckernel_engine.so", "model_v8.c",
                              "weights_manifest.json", "config.json", "weights.bump")},
                "probe_sha256": sha_file(Path(__file__)),
                "loaded_libraries": sorted(path for path in loaded if path in
                                           (str(model_path), str(engine_path))),
                "threads": {name: os.environ.get(name) for name in
                            ("CK_NUM_THREADS", "OMP_NUM_THREADS")},
                "cpu": next((line.split(":", 1)[1].strip() for line in
                             Path("/proc/cpuinfo").read_text().splitlines()
                             if line.startswith("model name")), None),
                "required_state": required, "cases": cases}
    finally:
        model.ck_model_free()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", required=True)
    parser.add_argument("--suffixes", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true",
                        help="include numerical errors and first differing state byte")
    parser.add_argument("--capture-root", type=Path,
                        help="capture selected hidden boundaries per execution stage")
    parser.add_argument("--capture-layer", type=int, default=0)
    args = parser.parse_args()
    ids = json.loads(args.ids.read_text())
    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
        parser.error("--ids must be a JSON array of nonnegative token IDs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = {"schema": "cke.checkpoint_prefill_equivalence.v1",
               "verdict": "INCOMPLETE", "bundle": str(args.bundle.resolve()),
               "ids_sha256": hashlib.sha256(
                   json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
               "cases": []}
    args.out.write_text(json.dumps(partial, indent=2) + "\n")

    def retain_cases(cases):
        partial["cases"] = cases
        args.out.write_text(json.dumps(partial, indent=2) + "\n")

    try:
        report = certify(args.bundle.resolve(), ids, args.prefixes, args.suffixes,
                         diagnose=args.diagnose, capture_root=args.capture_root,
                         capture_layer=args.capture_layer, on_case=retain_cases)
    except Exception as exc:
        partial["error"] = f"{type(exc).__name__}: {exc}"
        args.out.write_text(json.dumps(partial, indent=2) + "\n")
        raise
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["verdict"], len(report["cases"]), "cases", flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
