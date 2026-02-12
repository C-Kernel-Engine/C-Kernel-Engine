#!/usr/bin/env python3
"""
check_qk_norm_backward_parity_v7.py

PyTorch parity check for qk_norm_backward (v7 training path).
"""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
try:
    import torch
    _TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


ROOT = Path(__file__).resolve().parents[3]
UNITTEST_DIR = ROOT / "unittest"
ISA_ID_TO_NAME = {
    0: "scalar",
    1: "avx",
    2: "avx2",
    3: "avx_vnni",
}


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _load_lib():
    if str(UNITTEST_DIR) not in sys.path:
        sys.path.insert(0, str(UNITTEST_DIR))
    from lib_loader import load_lib  # noqa: E402

    lib = load_lib("libckernel_engine.so")
    lib.qk_norm_backward.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # d_q_out
        ctypes.POINTER(ctypes.c_float),  # d_k_out
        ctypes.POINTER(ctypes.c_float),  # q_in
        ctypes.POINTER(ctypes.c_float),  # k_in
        ctypes.POINTER(ctypes.c_float),  # q_gamma
        ctypes.POINTER(ctypes.c_float),  # k_gamma
        ctypes.POINTER(ctypes.c_float),  # d_q_in
        ctypes.POINTER(ctypes.c_float),  # d_k_in
        ctypes.POINTER(ctypes.c_float),  # d_q_gamma
        ctypes.POINTER(ctypes.c_float),  # d_k_gamma
        ctypes.c_int,  # num_heads
        ctypes.c_int,  # num_kv_heads
        ctypes.c_int,  # num_tokens
        ctypes.c_int,  # head_dim
        ctypes.c_float,  # eps
    ]
    lib.qk_norm_backward.restype = None
    try:
        lib.qk_norm_backward_last_isa.argtypes = []
        lib.qk_norm_backward_last_isa.restype = ctypes.c_int
        has_last_isa = True
    except AttributeError:
        has_last_isa = False
    lib._has_qk_norm_last_isa = has_last_isa  # type: ignore[attr-defined]
    return lib


def _qk_norm_torch(q: torch.Tensor, k: torch.Tensor, q_gamma: torch.Tensor, k_gamma: torch.Tensor, eps: float):
    q_var = q.pow(2).mean(dim=-1, keepdim=True)
    k_var = k.pow(2).mean(dim=-1, keepdim=True)
    q_rstd = (q_var + eps).rsqrt()
    k_rstd = (k_var + eps).rsqrt()
    q_out = q * q_rstd * q_gamma
    k_out = k * k_rstd * k_gamma
    return q_out, k_out


def run(
    seed: int,
    num_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    head_dim: int,
    eps: float,
    force_isa: str = "auto",
):
    if force_isa != "auto":
        os.environ["CK_QK_NORM_BACKWARD_ISA"] = force_isa
    else:
        os.environ.pop("CK_QK_NORM_BACKWARD_ISA", None)

    np.random.seed(seed)
    torch.manual_seed(seed)

    q_np = np.random.randn(num_heads, num_tokens, head_dim).astype(np.float32)
    k_np = np.random.randn(num_kv_heads, num_tokens, head_dim).astype(np.float32)
    q_gamma_np = np.random.randn(head_dim).astype(np.float32)
    k_gamma_np = np.random.randn(head_dim).astype(np.float32)
    d_q_out_np = np.random.randn(num_heads, num_tokens, head_dim).astype(np.float32)
    d_k_out_np = np.random.randn(num_kv_heads, num_tokens, head_dim).astype(np.float32)

    q_t = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
    k_t = torch.tensor(k_np, dtype=torch.float32, requires_grad=True)
    qg_t = torch.tensor(q_gamma_np, dtype=torch.float32, requires_grad=True)
    kg_t = torch.tensor(k_gamma_np, dtype=torch.float32, requires_grad=True)
    d_q_t = torch.tensor(d_q_out_np, dtype=torch.float32)
    d_k_t = torch.tensor(d_k_out_np, dtype=torch.float32)

    q_out_t, k_out_t = _qk_norm_torch(q_t, k_t, qg_t, kg_t, eps)
    loss_t = (q_out_t * d_q_t).sum() + (k_out_t * d_k_t).sum()
    loss_t.backward()

    d_q_in_ref = q_t.grad.detach().cpu().numpy()
    d_k_in_ref = k_t.grad.detach().cpu().numpy()
    d_q_gamma_ref = qg_t.grad.detach().cpu().numpy()
    d_k_gamma_ref = kg_t.grad.detach().cpu().numpy()

    lib = _load_lib()
    d_q_in = np.zeros_like(q_np, dtype=np.float32)
    d_k_in = np.zeros_like(k_np, dtype=np.float32)
    d_q_gamma = np.zeros_like(q_gamma_np, dtype=np.float32)
    d_k_gamma = np.zeros_like(k_gamma_np, dtype=np.float32)

    lib.qk_norm_backward(
        _ptr(d_q_out_np.reshape(-1)),
        _ptr(d_k_out_np.reshape(-1)),
        _ptr(q_np.reshape(-1)),
        _ptr(k_np.reshape(-1)),
        _ptr(q_gamma_np),
        _ptr(k_gamma_np),
        _ptr(d_q_in.reshape(-1)),
        _ptr(d_k_in.reshape(-1)),
        _ptr(d_q_gamma),
        _ptr(d_k_gamma),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(num_tokens),
        ctypes.c_int(head_dim),
        ctypes.c_float(eps),
    )
    selected_isa = "unknown"
    if getattr(lib, "_has_qk_norm_last_isa", False):
        try:
            selected_id = int(lib.qk_norm_backward_last_isa())
            selected_isa = ISA_ID_TO_NAME.get(selected_id, "unknown")
        except Exception:
            selected_isa = "unknown"

    max_diff_dq = float(np.max(np.abs(d_q_in - d_q_in_ref)))
    max_diff_dk = float(np.max(np.abs(d_k_in - d_k_in_ref)))
    max_diff_dqg = float(np.max(np.abs(d_q_gamma - d_q_gamma_ref)))
    max_diff_dkg = float(np.max(np.abs(d_k_gamma - d_k_gamma_ref)))
    max_diff = max(max_diff_dq, max_diff_dk, max_diff_dqg, max_diff_dkg)
    tol = 1e-4

    return {
        "passed": bool(max_diff <= tol),
        "tolerance": tol,
        "max_diff": max_diff,
        "max_diff_breakdown": {
            "d_q_in": max_diff_dq,
            "d_k_in": max_diff_dk,
            "d_q_gamma": max_diff_dqg,
            "d_k_gamma": max_diff_dkg,
        },
        "config": {
            "seed": seed,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "num_tokens": num_tokens,
            "head_dim": head_dim,
            "eps": eps,
            "requested_isa": force_isa,
            "selected_isa": selected_isa,
            "isa_fallback": bool(force_isa != "auto" and selected_isa != force_isa),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch parity check for qk_norm_backward.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--force-isa",
        type=str,
        default="auto",
        choices=["auto", "scalar", "avx", "avx2", "avx_vnni"],
        help="Force backward rstd SIMD path via CK_QK_NORM_BACKWARD_ISA.",
    )
    parser.add_argument(
        "--isa-matrix",
        action="store_true",
        help="Run parity across scalar/avx/avx2/avx_vnni plus auto.",
    )
    parser.add_argument(
        "--strict-isa",
        action="store_true",
        help="Fail when a requested ISA falls back to a different selected ISA.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("CK_TORCH_SAFE", "1")
    if torch is None:
        print("ERROR: PyTorch not available for qk_norm backward parity.", file=sys.stderr)
        print("DETAIL: %s" % _TORCH_IMPORT_ERROR, file=sys.stderr)
        print("HINT: install torch in active env (e.g. `.venv`).", file=sys.stderr)
        return 2

    isa_list = [args.force_isa]
    if args.isa_matrix:
        isa_list = ["scalar", "avx", "avx2", "avx_vnni", "auto"]

    all_results = []
    overall_pass = True
    fallback_fail = False
    for isa in isa_list:
        result = run(
            seed=int(args.seed),
            num_heads=int(args.num_heads),
            num_kv_heads=int(args.num_kv_heads),
            num_tokens=int(args.num_tokens),
            head_dim=int(args.head_dim),
            eps=float(args.eps),
            force_isa=str(isa),
        )
        all_results.append(result)
        overall_pass = overall_pass and bool(result["passed"])
        if args.strict_isa and bool(result["config"]["isa_fallback"]):
            fallback_fail = True
        print(
            "qk_norm_backward parity[%s->%s]: %s max_diff=%.3e tol=%.1e%s" % (
                isa,
                result["config"]["selected_isa"],
                "PASS" if result["passed"] else "FAIL",
                result["max_diff"],
                result["tolerance"],
                " (fallback)" if result["config"]["isa_fallback"] else "",
            )
        )
        print("breakdown:", result["max_diff_breakdown"])

    if args.strict_isa and fallback_fail:
        overall_pass = False
        print("STRICT ISA CHECK: FAIL (at least one requested ISA fell back)")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = all_results[0] if len(all_results) == 1 else {
            "passed": overall_pass,
            "runs": all_results,
            "matrix": True,
        }
        if isinstance(payload, dict):
            payload["strict_isa"] = bool(args.strict_isa)
            payload["strict_isa_failed"] = bool(args.strict_isa and fallback_fail)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("JSON:", args.json_out)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
