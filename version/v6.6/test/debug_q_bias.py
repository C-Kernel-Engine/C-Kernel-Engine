#!/usr/bin/env python3
"""
Debug Q bias values and buffer aliasing.
"""

import ctypes
import numpy as np
import json
from pathlib import Path

V66_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V65_DIR = Path.home() / ".cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

EMBED_DIM = 896

# V6.6 weight offsets (from layout_decode.json)
W_LAYER_0_BQ = 149154632  # from abs_offset

# V6.6 activation offsets
A_EMBEDDED_INPUT = 396942152
A_LAYER_INPUT = 400612168
A_Q_SCRATCH = 433380168


def check_buffer_aliasing():
    """Check if any buffers overlap."""
    print("=" * 70)
    print("BUFFER ALIASING CHECK")
    print("=" * 70)

    # Load layout
    layout_path = V66_DIR / "layout_decode.json"
    with open(layout_path) as f:
        layout = json.load(f)

    buffers = layout["memory"]["activations"]["buffers"]

    # Check for overlaps
    print(f"\nActivation buffers ({len(buffers)}):")
    for buf in buffers:
        name = buf["name"]
        offset = buf["abs_offset"]
        size = buf["size"]
        end = offset + size
        print(f"  {name:20s}: {offset:12d} - {end:12d} (size: {size:10d})")

    print("\n\nChecking for overlaps...")
    for i, buf1 in enumerate(buffers):
        for buf2 in buffers[i+1:]:
            start1, end1 = buf1["abs_offset"], buf1["abs_offset"] + buf1["size"]
            start2, end2 = buf2["abs_offset"], buf2["abs_offset"] + buf2["size"]

            # Check overlap
            if start1 < end2 and start2 < end1:
                print(f"  OVERLAP: {buf1['name']} and {buf2['name']}")
                print(f"    {buf1['name']}: {start1} - {end1}")
                print(f"    {buf2['name']}: {start2} - {end2}")


def check_q_bias():
    """Check Q bias values."""
    print("\n" + "=" * 70)
    print("Q BIAS VALUES")
    print("=" * 70)

    # Load v6.6 bias
    bump_path = V66_DIR / "weights.bump"
    with open(bump_path, 'rb') as f:
        # Read BQ (fp32, 896 values = 3584 bytes)
        f.seek(W_LAYER_0_BQ)
        bq_bytes = f.read(3584)
        bq_66 = np.frombuffer(bq_bytes, dtype=np.float32)

    print(f"v6.6 BQ shape: {bq_66.shape}")
    print(f"v6.6 BQ stats: min={bq_66.min():.4f}, max={bq_66.max():.4f}, mean={bq_66.mean():.4f}")
    print(f"v6.6 BQ first 20: {bq_66[:20]}")

    # Load v6.5 bias for comparison
    v65_manifest_path = V65_DIR / "weights_manifest.json"
    if v65_manifest_path.exists():
        with open(v65_manifest_path) as f:
            manifest = json.load(f)

        # Find BQ offset
        entries = manifest.get('entries', [])
        bq_entry = None
        for e in entries:
            if e.get('name') == 'layer.0.bq':
                bq_entry = e
                break

        if bq_entry:
            bump_path = V65_DIR / "weights.bump"
            with open(bump_path, 'rb') as f:
                f.seek(bq_entry['file_offset'])
                bq_bytes = f.read(3584)
                bq_65 = np.frombuffer(bq_bytes, dtype=np.float32)

            print(f"\nv6.5 BQ shape: {bq_65.shape}")
            print(f"v6.5 BQ stats: min={bq_65.min():.4f}, max={bq_65.max():.4f}, mean={bq_65.mean():.4f}")
            print(f"v6.5 BQ first 20: {bq_65[:20]}")

            # Compare
            diff = np.abs(bq_65 - bq_66)
            print(f"\nBQ diff: max={diff.max():.8f}")
            if diff.max() < 1e-6:
                print("BQ values MATCH!")
            else:
                print("BQ values DIFFER!")


def trace_q_projection():
    """Trace Q projection step by step."""
    print("\n" + "=" * 70)
    print("Q PROJECTION TRACE (manually computing)")
    print("=" * 70)

    # Load engine
    engine_path = V66_DIR / "libckernel_engine.so"
    if engine_path.exists():
        ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

    lib_path = V66_DIR / "ck-kernel-inference.so"
    if not lib_path.exists():
        lib_path = V66_DIR / "libmodel.so"

    lib = ctypes.CDLL(str(lib_path))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_uint64

    weights_path = V66_DIR / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"Init failed: {ret}")
        return

    base_ptr = lib.ck_model_get_base_ptr()

    def read_act(offset, size):
        ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(ptr, shape=(size,)).copy()

    # Run to stop_seq=2 (before Q projection)
    import os
    os.environ["CK_STOP_OP"] = "2"
    output = (ctypes.c_float * 151936)()
    lib.ck_model_decode(100, output)
    del os.environ["CK_STOP_OP"]

    layer_input = read_act(A_LAYER_INPUT, EMBED_DIM)
    q_before = read_act(A_Q_SCRATCH, EMBED_DIM)
    print(f"\nBefore Q proj (stop_seq=2):")
    print(f"  layer_input: min={layer_input.min():.4f}, max={layer_input.max():.4f}")
    print(f"  q_scratch:   min={q_before.min():.4f}, max={q_before.max():.4f}")

    # Run to stop_seq=3 (after Q projection, before bias)
    lib.ck_model_free()
    lib.ck_model_init(str(weights_path).encode())
    base_ptr = lib.ck_model_get_base_ptr()

    os.environ["CK_STOP_OP"] = "3"
    lib.ck_model_decode(100, output)
    del os.environ["CK_STOP_OP"]

    q_after_gemv = read_act(A_Q_SCRATCH, EMBED_DIM)
    print(f"\nAfter Q gemv (stop_seq=3):")
    print(f"  q_scratch:   min={q_after_gemv.min():.4f}, max={q_after_gemv.max():.4f}")
    print(f"  first 10:    {q_after_gemv[:10]}")

    # Run to stop_seq=4 (after Q bias add)
    lib.ck_model_free()
    lib.ck_model_init(str(weights_path).encode())
    base_ptr = lib.ck_model_get_base_ptr()

    os.environ["CK_STOP_OP"] = "4"
    lib.ck_model_decode(100, output)
    del os.environ["CK_STOP_OP"]

    q_after_bias = read_act(A_Q_SCRATCH, EMBED_DIM)
    print(f"\nAfter Q bias add (stop_seq=4):")
    print(f"  q_scratch:   min={q_after_bias.min():.4f}, max={q_after_bias.max():.4f}")
    print(f"  first 10:    {q_after_bias[:10]}")

    # Compare with BQ values
    bump_path = V66_DIR / "weights.bump"
    with open(bump_path, 'rb') as f:
        f.seek(W_LAYER_0_BQ)
        bq = np.frombuffer(f.read(3584), dtype=np.float32)

    expected = q_after_gemv + bq
    print(f"\nExpected (q + bq): min={expected.min():.4f}, max={expected.max():.4f}")
    print(f"  first 10:        {expected[:10]}")

    diff = np.abs(q_after_bias - expected)
    print(f"\nDiff between actual and expected: max={diff.max():.6f}")

    if diff.max() > 0.001:
        print("Q bias add is NOT matching expected!")
        # Find where diff is largest
        max_idx = np.argmax(diff)
        print(f"  Max diff at idx {max_idx}:")
        print(f"    q_after_gemv[{max_idx}] = {q_after_gemv[max_idx]:.6f}")
        print(f"    bq[{max_idx}]           = {bq[max_idx]:.6f}")
        print(f"    expected                = {expected[max_idx]:.6f}")
        print(f"    actual                  = {q_after_bias[max_idx]:.6f}")

    lib.ck_model_free()


if __name__ == "__main__":
    check_buffer_aliasing()
    check_q_bias()
    trace_q_projection()
