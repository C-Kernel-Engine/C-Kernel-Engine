#!/usr/bin/env python3
"""
Out-proj parity micro-test (packed layout).

Compares:
  1) CK out_proj output vs numpy reference using dequantized wo.
  2) CK bias_add(bo) output vs numpy reference (wo @ x + bo).
Optionally compares to llama.cpp dump kqv_out-0 if present.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np

Q5_0_BLOCK_SIZE = 22
Q5_0_BLOCK_K = 32


def load_llama_tensor(name: str, dump_dir: Path) -> np.ndarray | None:
    path = dump_dir / f"{name}.bin"
    if not path.exists():
        return None
    return np.fromfile(str(path), dtype=np.float32)


def read_f32(base_ptr: int, offset: int, count: int) -> np.ndarray:
    ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(count,)).copy()


def compare_tensors(name: str, ref: np.ndarray, ck: np.ndarray) -> dict:
    ref_flat = ref.flatten()
    ck_flat = ck.flatten()
    n = min(len(ref_flat), len(ck_flat))
    ref_flat = ref_flat[:n]
    ck_flat = ck_flat[:n]

    diff = np.abs(ref_flat - ck_flat)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    max_abs = max(float(np.max(np.abs(ref_flat))), 1e-9)
    rel_err = max_diff / max_abs

    return {
        "name": name,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_err": rel_err,
        "size": n,
    }


def get_op_sequence_index(ops, op_name: str, layer: int, weight_key: str | None = None) -> int | None:
    for idx, op in enumerate(ops):
        if op.get("op") != op_name or op.get("layer") != layer:
            continue
        if weight_key and weight_key not in (op.get("weights") or {}):
            continue
        return idx
    return None


def find_op(ops, op_name: str, layer: int, weight_key: str | None = None) -> dict | None:
    for op in ops:
        if op.get("op") != op_name or op.get("layer") != layer:
            continue
        if weight_key and weight_key not in (op.get("weights") or {}):
            continue
        return op
    return None


def zero_activations(base_ptr: int, arena: dict, memory: dict) -> None:
    act = memory.get("activations", {})
    act_size = int(act.get("size", 0))
    if act_size <= 0:
        return

    act_base = int(arena.get("activations_base", 0))
    rope_buf = None
    for buf in act.get("buffers", []):
        if buf.get("name") == "rope_cache":
            rope_buf = buf
            break

    def abs_offset(buf: dict) -> int:
        if "abs_offset" in buf:
            return int(buf["abs_offset"])
        return act_base + int(buf.get("offset", 0))

    if not rope_buf:
        ctypes.memset(base_ptr + act_base, 0, act_size)
        return

    rope_off = abs_offset(rope_buf)
    rope_size = int(rope_buf.get("size", 0))
    if rope_size <= 0:
        ctypes.memset(base_ptr + act_base, 0, act_size)
        return

    pre = max(0, rope_off - act_base)
    if pre:
        ctypes.memset(base_ptr + act_base, 0, pre)
    end = act_base + act_size
    post_start = rope_off + rope_size
    if post_start < end:
        ctypes.memset(base_ptr + post_start, 0, end - post_start)


def dequant_q5_0_matrix(data: bytes, rows: int, cols: int) -> np.ndarray:
    if cols % Q5_0_BLOCK_K != 0:
        raise ValueError(f"cols must be multiple of {Q5_0_BLOCK_K} (got {cols})")
    blocks_per_row = cols // Q5_0_BLOCK_K
    expected = rows * blocks_per_row * Q5_0_BLOCK_SIZE
    if len(data) != expected:
        raise ValueError(f"Q5_0 size mismatch: expected {expected}, got {len(data)}")

    out = np.empty((rows, cols), dtype=np.float32)
    mv = memoryview(data)
    for r in range(rows):
        row_base = r * blocks_per_row * Q5_0_BLOCK_SIZE
        for b in range(blocks_per_row):
            off = row_base + b * Q5_0_BLOCK_SIZE
            d = np.frombuffer(mv[off : off + 2], dtype=np.float16)[0].astype(np.float32)
            qh = int.from_bytes(mv[off + 2 : off + 6], "little", signed=False)
            qs = mv[off + 6 : off + 22]
            out_base = b * Q5_0_BLOCK_K
            for j in range(Q5_0_BLOCK_K // 2):
                packed = qs[j]
                lo = packed & 0x0F
                hi = packed >> 4
                xh_0 = ((qh >> (j + 0)) << 4) & 0x10
                xh_1 = ((qh >> (j + 12))) & 0x10
                q0 = (lo | xh_0) - 16
                q1 = (hi | xh_1) - 16
                out[r, out_base + j] = d * q0
                out[r, out_base + j + 16] = d * q1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Out-proj parity micro-test")
    ap.add_argument("--model-dir", required=True, help="CK model directory")
    ap.add_argument("--lowered", default=None, help="Path to lowered_decode.json")
    ap.add_argument("--dump-dir", default="llama_dump", help="llama.cpp dump directory")
    ap.add_argument("--token", type=int, default=9707, help="Token ID")
    ap.add_argument("--layer", type=int, default=0, help="Layer index")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    lowered_path = Path(args.lowered) if args.lowered else model_dir / "lowered_decode.json"
    if not lowered_path.exists():
        print(f"ERROR: lowered IR not found: {lowered_path}")
        return 1

    with open(lowered_path, "r", encoding="utf-8") as f:
        lowered = json.load(f)

    memory = lowered.get("memory", {})
    arena = memory.get("arena", {})
    if arena.get("mode") != "packed":
        print(f"ERROR: layout mode is '{arena.get('mode')}', requires packed layout")
        return 1

    ops = lowered.get("operations", [])
    out_proj = find_op(ops, "out_proj", args.layer, weight_key="wo")
    if not out_proj:
        print("ERROR: out_proj op not found (run with --no-fusion?)")
        return 1

    bias_add = find_op(ops, "bias_add", args.layer, weight_key="bo")
    if not bias_add:
        print("WARNING: bias_add(bo) op not found; bias comparison will be skipped")

    weights = memory.get("weights", {})
    entries = weights.get("entries", [])
    wo_name = out_proj["weights"]["wo"]["name"]
    wo_entry = next((e for e in entries if e.get("name") == wo_name), None)
    if not wo_entry:
        print(f"ERROR: weights entry not found for {wo_name}")
        return 1

    bump_path = model_dir / "weights.bump"
    if not bump_path.exists():
        print(f"ERROR: weights.bump not found in {model_dir}")
        return 1

    with open(bump_path, "rb") as bf:
        bf.seek(int(wo_entry["file_offset"]))
        wo_bytes = bf.read(int(wo_entry["size"]))

    params = out_proj.get("params", {})
    M = int(params.get("_output_dim", params.get("embed_dim", 0)))
    K = int(params.get("_input_dim", params.get("embed_dim", 0)))
    if M <= 0 or K <= 0:
        print("ERROR: invalid dims in out_proj params")
        return 1

    expected = M * (K // Q5_0_BLOCK_K) * Q5_0_BLOCK_SIZE
    if expected != len(wo_bytes):
        blocks_per_row = len(wo_bytes) // (M * Q5_0_BLOCK_SIZE)
        K = blocks_per_row * Q5_0_BLOCK_K
        print(f"WARNING: size mismatch; derived K={K} from bytes")

    W = dequant_q5_0_matrix(wo_bytes, M, K)

    lib_path = model_dir / "libmodel.so"
    if not lib_path.exists():
        print(f"ERROR: libmodel.so not found in {model_dir}")
        return 1

    lib = ctypes.CDLL(str(lib_path))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    lib.ck_model_kv_cache_reset.argtypes = []
    lib.ck_model_kv_cache_reset.restype = None

    if lib.ck_model_init(str(bump_path).encode()) != 0:
        print("ERROR: ck_model_init failed")
        return 1

    base_ptr = lib.ck_model_get_base_ptr()
    if not base_ptr:
        print("ERROR: ck_model_get_base_ptr returned NULL")
        return 1

    out_seq = get_op_sequence_index(ops, "out_proj", args.layer, weight_key="wo")
    bias_seq = get_op_sequence_index(ops, "bias_add", args.layer, weight_key="bo") if bias_add else None
    if out_seq is None:
        print("ERROR: missing out_proj op sequence index")
        return 1

    x_off = int(out_proj["activations"]["x"]["activation_offset"])
    y_off = int(out_proj["outputs"]["y"]["activation_offset"])
    bo_off = int(bias_add["weights"]["bo"]["bump_offset"]) if bias_add else None
    yb_off = int(bias_add["outputs"]["y"]["activation_offset"]) if bias_add else None

    # Run to out_proj
    os.environ["CK_STOP_OP"] = str(out_seq)
    lib.ck_model_kv_cache_reset()
    zero_activations(base_ptr, arena, memory)
    if lib.ck_model_decode(ctypes.c_int32(args.token), None) != 0:
        print("ERROR: ck_model_decode failed")
        return 1

    x = read_f32(base_ptr, x_off, K)
    y_ck = read_f32(base_ptr, y_off, M)
    y_ref = W @ x

    res = compare_tensors("out_proj (W @ x)", y_ref, y_ck)

    # Optional transpose check for orientation mismatch
    y_ref_t = W.T @ x
    res_t = compare_tensors("out_proj (W.T @ x)", y_ref_t, y_ck)

    res_bias = None
    llama_res = None
    if bias_add and bias_seq is not None:
        # Run to bias_add
        os.environ["CK_STOP_OP"] = str(bias_seq)
        lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, arena, memory)
        if lib.ck_model_decode(ctypes.c_int32(args.token), None) != 0:
            print("ERROR: ck_model_decode failed")
            return 1

        y_bias = read_f32(base_ptr, yb_off, M)
        bo = read_f32(base_ptr, bo_off, M)
        y_ref_bias = y_ref + bo
        res_bias = compare_tensors("out_proj + bo", y_ref_bias, y_bias)

        llama = load_llama_tensor("kqv_out-0", Path(args.dump_dir))
        if llama is not None:
            llama_res = compare_tensors("llama kqv_out-0 vs ref", y_ref_bias, llama)

    print("\nOut-proj parity micro-test")
    print(f"Model dir: {model_dir}")
    print(f"Layer:     {args.layer}")
    print(f"Dims:      M={M} K={K}")
    print(f"Seq idx:   out_proj={out_seq} bias_add={bias_seq}")
    print(f"{res['name']:<22} rel_err={res['rel_err']:.2%} max_diff={res['max_diff']:.6f}")
    print(f"{res_t['name']:<22} rel_err={res_t['rel_err']:.2%} max_diff={res_t['max_diff']:.6f}")
    if res_bias:
        print(f"{res_bias['name']:<22} rel_err={res_bias['rel_err']:.2%} max_diff={res_bias['max_diff']:.6f}")
    if llama_res:
        print(f"{'llama vs ref':<22} rel_err={llama_res['rel_err']:.2%} max_diff={llama_res['max_diff']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
