#!/usr/bin/env python3
import ctypes
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_softplus_head_gate_broadcasts_over_channels(tmp_path: Path) -> None:
    lib_path = tmp_path / "libsoftplus_gate.so"
    subprocess.run(
        [
            "cc", "-shared", "-fPIC", "-O2", "-I", str(ROOT / "include"),
            str(ROOT / "src/kernels/hybrid_attention_kernels.c"),
            "-lm", "-o", str(lib_path),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.attn_gate_softplus_mul_forward
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    x = (ctypes.c_float * 12)(*range(1, 13))
    gate_values = [-2.0, 0.0, 3.0, 21.0]
    gate = (ctypes.c_float * 4)(*gate_values)
    out = (ctypes.c_float * 12)()
    fn(x, gate, out, 2, 2, 3)

    expected = []
    for row in range(2):
        for head in range(2):
            value = gate_values[row * 2 + head]
            scale = value if value > 20.0 else math.log1p(math.exp(value))
            for channel in range(3):
                expected.append(float(x[row * 6 + head * 3 + channel]) * scale)
    for got, want in zip(out, expected):
        assert math.isclose(got, want, rel_tol=2e-6, abs_tol=2e-6)
