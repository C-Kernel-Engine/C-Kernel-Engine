#!/usr/bin/env python3
"""Certify exact-ID session checkpoint continuation against ordinary prefill.

This optional asset-backed lane uses a separate native session library and
bundle. It does not change an existing server or certify arbitrary splits.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from server import session_v8 as binding
from certify_sequence_checkpoint_v8 import (
    configure, declared_state_regions, sha_file, signature,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--session-library", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=128)
    parser.add_argument("--suffix", type=int, default=9)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    session_library = args.session_library.resolve()
    ids = json.loads(args.ids.read_text())
    if (not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids)
            or args.prefix < 64 or args.prefix % 64 or args.suffix < 2
            or args.prefix + args.suffix > len(ids)):
        parser.error("invalid exact token IDs or prefix/suffix boundary")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "cke.session_multitoken_prefill.v1", "verdict": "INCOMPLETE",
              "bundle": str(bundle), "session_library": str(session_library),
              "prefix": args.prefix, "suffix": args.suffix,
              "ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    binding.SESSION_LIB_PATH = session_library
    layout = json.loads((bundle / "layout_decode.json").read_text())
    context = int(layout["config"]["context_length"])
    session = binding.SessionV8.open(bundle, context_length=context, num_threads=args.threads)
    try:
        model = C.CDLL(str(bundle / "libmodel.so"), mode=C.RTLD_GLOBAL)
        configure(model)
        model.ck_model_kv_cache_reset.restype = None
        model.ck_model_get_logits.restype = C.c_void_p
        model.ck_model_get_logits_stride.restype = C.c_int
        required = declared_state_regions(layout)
        loaded = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                  if len(line.split()) >= 6}
        expected = {str(bundle / "libmodel.so"), str(bundle / "libckernel_engine.so"),
                    str(session_library)}
        if not expected <= loaded:
            raise RuntimeError("loaded library identity mismatch")
        vocab = model.ck_model_get_vocab_size()
        if vocab < 1:
            raise RuntimeError("invalid vocabulary size")

        def current_signature():
            address = model.ck_model_get_logits()
            position = model.ck_model_get_active_tokens()
            stride = model.ck_model_get_logits_stride()
            if not address or position <= 0 or stride < 0:
                raise RuntimeError("invalid active logits")
            if stride:
                address += (position - 1) * stride * C.sizeof(C.c_float)
            return signature(model, (C.c_float * vocab).from_address(address), required)

        full = ids[:args.prefix + args.suffix]
        model.ck_model_kv_cache_reset()
        source = (C.c_int32 * len(full))(*full)
        start = time.monotonic()
        if model.ck_model_embed_tokens(source, len(full)) or model.ck_model_forward(None):
            raise RuntimeError("ordinary full prefill failed")
        full_time = time.monotonic() - start
        baseline = current_signature()

        suffix_ids = ids[args.prefix:args.prefix + args.suffix]
        size, alignment = session.checkpoint_requirements()
        snapshot = binding.SequenceCheckpointV8(size, alignment)
        start = time.monotonic()
        session.prefill_ids(ids[:args.prefix], reserved_tokens=args.suffix + 1)
        prefix_time = time.monotonic() - start
        if model.ck_model_get_active_tokens() != args.prefix:
            raise RuntimeError("session prefix did not reach exact token boundary")
        before_rejection = current_signature()
        rejected = {}
        for name, tokens, reserve in (
            ("unaligned", ids[args.prefix:args.prefix + 9], 1),
            ("capacity", ids[args.prefix:args.prefix + 64], context),
            ("invalid_id", [vocab] + ids[args.prefix + 1:args.prefix + 64], 1),
        ):
            try:
                session.prefill_ids(tokens, reserved_tokens=reserve, continue_state=True)
            except binding.SessionError:
                rejected[name] = before_rejection == current_signature()
            else:
                rejected[name] = False
        if not all(rejected.values()):
            raise RuntimeError(f"rejected append mutated session: {rejected}")
        start = time.monotonic()
        session.checkpoint_save(snapshot)
        save_time = time.monotonic() - start
        flags = (binding.CK_SESSION_REQUEST_RAW_PROMPT |
                 binding.CK_SESSION_REQUEST_CONTINUE_STATE |
                 binding.CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION)

        def continue_suffix():
            start = time.monotonic()
            outcome = session.generate(system="", user="", max_tokens=1,
                                       temperature=0.0, top_p=1.0,
                                       on_token=lambda *_: 0, flags=flags,
                                       prompt_ids=suffix_ids)
            elapsed = time.monotonic() - start
            if outcome["prompt_tokens"] != args.suffix:
                raise RuntimeError("session processed different suffix token count")
            return current_signature(), elapsed

        direct, first_time = continue_suffix()
        other = ids[args.prefix:args.prefix + 64]
        session.prefill_ids(other, reserved_tokens=1)
        start = time.monotonic()
        session.checkpoint_restore(snapshot)
        restore_time = time.monotonic() - start
        restored, second_time = continue_suffix()
        report.update({
            "verdict": "PASS" if baseline == direct == restored else "FAIL",
            "full": baseline, "direct": direct, "restored": restored,
            "full_equals_direct": baseline == direct,
            "direct_equals_restore": direct == restored,
            "required_state": required,
            "rejected_unchanged": rejected,
            "checkpoint_bytes": size, "checkpoint_alignment": alignment,
            "loaded_libraries": sorted(expected),
            "artifact_sha256": {name: sha_file(bundle / name) for name in
                                ("libmodel.so", "libckernel_engine.so", "model_v8.c",
                                 "weights.bump", "weights_manifest.map", "config.json")},
            "session_library_sha256": sha_file(session_library),
            "session_source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "session_source_sha256": {
                name: sha_file(ROOT / name) for name in
                ("include/ck_session_v8.h", "version/v8/src/ck_cli_v8.c",
                 "server/session_v8.py")},
            "probe_sha256": sha_file(Path(__file__)),
            "threads": {"session": args.threads, "CK_NUM_THREADS": os.getenv("CK_NUM_THREADS"),
                        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS")},
            "times_seconds": {"full_prefill": full_time, "prefix": prefix_time,
                              "save": save_time, "first_suffix": first_time,
                              "restore": restore_time, "second_suffix": second_time},
        })
    except Exception as exc:
        report["verdict"] = "INCOMPLETE"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        session.close()
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["verdict"], report["prefix"], report["suffix"], flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
