#!/usr/bin/env python3
"""Capture repeated prefill logits and complete named mutable state for one bundle.

Run in separate processes for baseline and candidate bundles, then compare JSON
captures. This keeps the process-global generated model and engine libraries apart.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path


STATE_NAMES = (
    "kv_cache", "recurrent_conv_state", "recurrent_ssm_state",
    "ple_token_state", "ple_conv_state", "qsa_index_key_cache",
)


def digest_memory(pointer: int, length: int) -> str:
    digest = hashlib.sha256()
    for offset in range(0, length, 16 << 20):
        digest.update(C.string_at(pointer + offset, min(16 << 20, length - offset)))
    return digest.hexdigest()


def reset_match(captures: list[dict]) -> bool:
    return {k: v for k, v in captures[0].items() if k not in ("label", "reuse")} == {
        k: v for k, v in captures[2].items() if k not in ("label", "reuse")
    }


def capture(run: Path, ids: list[int]) -> dict:
    lib_path = (run / "libmodel.so").resolve()
    lib = C.CDLL(str(lib_path), mode=C.RTLD_GLOBAL)
    engine_path = (run / "libckernel_engine.so").resolve()
    engine = C.CDLL(str(engine_path), mode=C.RTLD_GLOBAL)
    lib.ck_model_init.argtypes = [C.c_char_p]
    lib.ck_model_init.restype = C.c_int
    lib.ck_model_embed_tokens.argtypes = [C.POINTER(C.c_int32), C.c_int]
    lib.ck_model_embed_tokens.restype = C.c_int
    lib.ck_model_forward.argtypes = [C.POINTER(C.c_float)]
    lib.ck_model_forward.restype = C.c_int
    lib.ck_model_get_logits.restype = C.c_void_p
    lib.ck_model_get_logits_stride.restype = C.c_int
    lib.ck_model_get_active_tokens.restype = C.c_int
    lib.ck_model_get_vocab_size.restype = C.c_int
    lib.ck_model_get_named_activation_ptr.argtypes = [C.c_char_p]
    lib.ck_model_get_named_activation_ptr.restype = C.c_size_t
    lib.ck_model_get_named_activation_nbytes.argtypes = [C.c_char_p]
    lib.ck_model_get_named_activation_nbytes.restype = C.c_ssize_t
    has_reuse_counters = hasattr(engine, "ck_moe_q4k_reuse_stats")
    if has_reuse_counters:
        engine.ck_moe_q4k_reuse_stats.argtypes = [C.POINTER(C.c_uint64)] * 6
    if lib.ck_model_init(str(run / "weights.bump").encode()) != 0:
        raise RuntimeError("ck_model_init failed")
    try:
        loaded = sorted({str(Path(line.split()[-1]).resolve())
                         for line in Path("/proc/self/maps").read_text().splitlines()
                         if len(line.split()) >= 6 and
                         any(name in line.split()[-1] for name in
                             ("libmodel.so", "libckernel_engine.so"))})
        for expected in (lib_path, engine_path):
            if str(expected) not in loaded:
                raise RuntimeError(f"expected loaded library missing: {expected}")
        captures = []
        for label, tokens in (("A1", ids[:-1]), ("B", ids[1:]), ("A2", ids[:-1])):
            lib.ck_model_kv_cache_reset()
            if has_reuse_counters:
                engine.ck_moe_q4k_reuse_stats_reset()
            token_array = (C.c_int32 * len(tokens))(*tokens)
            if lib.ck_model_embed_tokens(token_array, len(tokens)) != 0:
                raise RuntimeError(f"{label}: embed failed")
            if lib.ck_model_forward(None) != 0:
                raise RuntimeError(f"{label}: forward failed")
            active = lib.ck_model_get_active_tokens()
            vocab = lib.ck_model_get_vocab_size()
            stride = lib.ck_model_get_logits_stride()
            logits_ptr = lib.ck_model_get_logits()
            if active != len(tokens) or vocab <= 0 or not logits_ptr:
                raise RuntimeError(f"{label}: invalid output extent")
            last = logits_ptr + (active - 1) * stride * 4 if stride > 0 else logits_ptr
            state = {}
            for name in STATE_NAMES:
                ptr = int(lib.ck_model_get_named_activation_ptr(name.encode()))
                length = int(lib.ck_model_get_named_activation_nbytes(name.encode()))
                if ptr == 0 or length <= 0:
                    raise RuntimeError(f"{label}: missing state {name}")
                state[name] = {"bytes": length, "sha256": digest_memory(ptr, length)}
            counters = [C.c_uint64() for _ in range(6)]
            if has_reuse_counters:
                engine.ck_moe_q4k_reuse_stats(*(C.byref(x) for x in counters))
            captures.append({
                "label": label,
                "ids_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                "active_tokens": active,
                "logits_sha256": digest_memory(last, vocab * 4),
                "state": state,
                "reuse": dict(zip(("assignments", "distinct", "pack_calls", "packed_bytes",
                                   "reuse_calls", "fallbacks"), (x.value for x in counters)))
                         if has_reuse_counters else None,
            })
        reset_reproduced = reset_match(captures)
        return {
            "schema": "cke.moe_reuse_state_capture.v1",
            "libmodel_sha256": hashlib.sha256(lib_path.read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
            "generated_source_sha256": hashlib.sha256((run / "model_v8.c").read_bytes()).hexdigest(),
            "weights_manifest_sha256": hashlib.sha256((run / "weights_manifest.json").read_bytes()).hexdigest(),
            "cpu": next((line.split(":", 1)[1].strip() for line in
                         Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")), "unknown"),
            "ck_threads": os.environ.get("CK_NUM_THREADS", "runtime-default"),
            "openmp_threads": os.environ.get("OMP_NUM_THREADS", "runtime-default"),
            "loaded_libraries": loaded,
            "reset_reproduced": reset_reproduced,
            "captures": captures,
        }
    finally:
        lib.ck_model_free()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--prefix-len", type=int, default=517)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args()
    if args.compare:
        baseline, candidate = (json.loads(path.read_text()) for path in args.compare)
        if baseline.get("schema") != candidate.get("schema"):
            parser.error("capture schemas differ")
        cases = []
        inventory_match = len(baseline["captures"]) == len(candidate["captures"]) == 3
        for before, after in zip(baseline["captures"], candidate["captures"]):
            same = (before["label"] == after["label"] and
                    before["ids_sha256"] == after["ids_sha256"] and
                    before["active_tokens"] == after["active_tokens"])
            logits_match = before["logits_sha256"] == after["logits_sha256"]
            state_match = before["state"] == after["state"]
            reuse = after.get("reuse") or {}
            executed = reuse.get("reuse_calls", 0) > 0
            cases.append({"case": before["label"], "input_match": same,
                          "full_logits_match": logits_match,
                          "full_named_state_match": state_match,
                          "reuse_executed": executed,
                          "fallback_calls": reuse.get("fallbacks")})
        result = {"schema": "cke.moe_reuse_state_verdict.v1",
                  "status": "PASS" if inventory_match and reset_match(baseline["captures"]) and
                  reset_match(candidate["captures"]) and all(
                      all(x[key] for key in ("input_match", "full_logits_match",
                                             "full_named_state_match", "reuse_executed"))
                      for x in cases) else "FAIL", "cases": cases,
                  "baseline_libmodel_sha256": baseline["libmodel_sha256"],
                  "candidate_libmodel_sha256": candidate["libmodel_sha256"],
                  "case_inventory_match": inventory_match}
        result["baseline_reset_reproduced"] = reset_match(baseline["captures"])
        result["candidate_reset_reproduced"] = reset_match(candidate["captures"])
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return 0 if result["status"] == "PASS" else 1
    if args.run is None or args.ids is None or args.out is None:
        parser.error("capture requires --run, --ids, and --out")
    source = json.loads(args.ids.read_text())
    if args.prefix_len < 512 or len(source) <= args.prefix_len:
        parser.error("need at least 512 prompt rows plus one shifted B token")
    result = capture(args.run, source[:args.prefix_len + 1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"bundle": str(args.run), "reset_reproduced": result["reset_reproduced"],
                      "captures": [
        {"label": x["label"], "reuse": x["reuse"]} for x in result["captures"]
    ]}))
    return 0 if result["reset_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
