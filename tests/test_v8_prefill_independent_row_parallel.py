from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


class PrefillIndependentRowParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_num_threads = os.environ.get("CK_NUM_THREADS")
        os.environ["CK_NUM_THREADS"] = "8"
        subprocess.run(
            ["make", "--no-print-directory", "build/libckernel_engine.so"],
            cwd=ROOT,
            check=True,
        )
        cls._tmp = tempfile.TemporaryDirectory(prefix="ck_prefill_rows_")
        cls.library_path = Path(cls._tmp.name) / "libprefill_rows.so"
        subprocess.run(
            [
                os.environ.get("CC", "gcc"),
                "-shared",
                "-fPIC",
                "-O3",
                "-march=native",
                "-Iinclude",
                "-Iversion/v8/src",
                "-o",
                str(cls.library_path),
                "version/v8/src/ck_parallel_prefill_v8.c",
                "-Lbuild",
                "-lckernel_engine",
                "-lm",
                "-lpthread",
                f"-Wl,-rpath,{ROOT / 'build'}",
            ],
            cwd=ROOT,
            check=True,
        )
        ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
        cls.lib = ctypes.CDLL(str(cls.library_path))

        quant_args = [FLOAT_PTR, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        cls.quant_serial = cls.lib.quantize_batch_q8_k_4row_nearest_even
        cls.quant_serial.argtypes = quant_args
        cls.quant_parallel = (
            cls.lib.quantize_batch_q8_k_4row_nearest_even_parallel_dispatch
        )
        cls.quant_parallel.argtypes = quant_args

        rms_args = [
            FLOAT_PTR,
            FLOAT_PTR,
            FLOAT_PTR,
            FLOAT_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        cls.rms_serial = cls.lib.rmsnorm_forward_llama_production
        cls.rms_serial.argtypes = rms_args
        cls.rms_parallel = cls.lib.rmsnorm_forward_llama_production_parallel_dispatch
        cls.rms_parallel.argtypes = rms_args

        recurrent_args = [
            FLOAT_PTR,
            FLOAT_PTR,
            FLOAT_PTR,
            FLOAT_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        cls.recurrent_serial = cls.lib.recurrent_norm_gate_llama_avx2_forward
        cls.recurrent_serial.argtypes = recurrent_args
        cls.recurrent_parallel = (
            cls.lib.recurrent_norm_gate_llama_avx2_parallel_dispatch
        )
        cls.recurrent_parallel.argtypes = recurrent_args

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()
        if cls._old_num_threads is None:
            os.environ.pop("CK_NUM_THREADS", None)
        else:
            os.environ["CK_NUM_THREADS"] = cls._old_num_threads

    def test_q8_k_rows_are_byte_exact(self) -> None:
        rows, width = 257, 512
        rng = np.random.default_rng(11)
        source = rng.standard_normal((rows, width), dtype=np.float32)
        output_bytes = rows * (width // 256) * 292
        serial = np.empty(output_bytes, dtype=np.uint8)
        parallel = np.empty_like(serial)

        self.quant_serial(source.ctypes.data_as(FLOAT_PTR), serial.ctypes.data, rows, width)
        self.quant_parallel(
            source.ctypes.data_as(FLOAT_PTR), parallel.ctypes.data, rows, width
        )
        np.testing.assert_array_equal(parallel, serial)

    def test_rmsnorm_rows_are_bit_exact(self) -> None:
        tokens, d_model, aligned = 257, 510, 512
        rng = np.random.default_rng(13)
        source = rng.standard_normal((tokens, aligned), dtype=np.float32)
        gamma = rng.standard_normal(d_model, dtype=np.float32)
        serial = np.empty_like(source)
        parallel = np.empty_like(source)
        serial_rstd = np.empty(tokens, dtype=np.float32)
        parallel_rstd = np.empty_like(serial_rstd)

        self.rms_serial(
            source.ctypes.data_as(FLOAT_PTR),
            gamma.ctypes.data_as(FLOAT_PTR),
            serial.ctypes.data_as(FLOAT_PTR),
            serial_rstd.ctypes.data_as(FLOAT_PTR),
            tokens,
            d_model,
            aligned,
            ctypes.c_float(1e-6),
        )
        self.rms_parallel(
            source.ctypes.data_as(FLOAT_PTR),
            gamma.ctypes.data_as(FLOAT_PTR),
            parallel.ctypes.data_as(FLOAT_PTR),
            parallel_rstd.ctypes.data_as(FLOAT_PTR),
            tokens,
            d_model,
            aligned,
            ctypes.c_float(1e-6),
        )
        np.testing.assert_array_equal(parallel, serial)
        np.testing.assert_array_equal(parallel_rstd, serial_rstd)

    def test_recurrent_norm_gate_rows_are_bit_exact(self) -> None:
        rows, heads, head_dim = 257, 4, 256
        rng = np.random.default_rng(17)
        source = rng.standard_normal((rows, heads, head_dim), dtype=np.float32)
        gate = rng.standard_normal(source.shape, dtype=np.float32)
        weight = rng.standard_normal(head_dim, dtype=np.float32)
        serial = np.empty_like(source)
        parallel = np.empty_like(source)

        common = (
            source.ctypes.data_as(FLOAT_PTR),
            gate.ctypes.data_as(FLOAT_PTR),
            weight.ctypes.data_as(FLOAT_PTR),
        )
        self.recurrent_serial(
            *common,
            serial.ctypes.data_as(FLOAT_PTR),
            rows,
            heads,
            head_dim,
            ctypes.c_float(1e-6),
        )
        self.recurrent_parallel(
            *common,
            parallel.ctypes.data_as(FLOAT_PTR),
            rows,
            heads,
            head_dim,
            ctypes.c_float(1e-6),
        )
        np.testing.assert_array_equal(parallel, serial)


if __name__ == "__main__":
    unittest.main()
