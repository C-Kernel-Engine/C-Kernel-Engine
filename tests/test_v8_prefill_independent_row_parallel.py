from __future__ import annotations

import ctypes
import importlib.util
import math
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
INT32_PTR = ctypes.POINTER(ctypes.c_int32)


def _compiler_defines_avx2(cflags: str, cc: str | None = None) -> bool:
    macros = subprocess.run(
        [*(shlex.split(cc) if cc else [os.environ.get("CC", "gcc")]),
         *shlex.split(cflags), "-dM", "-E", "-x", "c", "-"],
        input="\n", text=True, capture_output=True, check=True, cwd=ROOT,
    ).stdout
    return "#define __AVX2__ 1" in macros


class PrefillIndependentRowParallelTests(unittest.TestCase):
    def test_gemma4_compiled_avx2_detection_accepts_march_only(self) -> None:
        self.assertTrue(_compiler_defines_avx2("-march=haswell"))
        self.assertFalse(_compiler_defines_avx2("-mavx2 -mno-avx2"))

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
        cls.engine = ctypes.CDLL(
            str(ROOT / "build/libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL
        )
        cls.lib = ctypes.CDLL(str(cls.library_path))

        gemm_args = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR]
        gemm_args += [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        cls.gemm_serial = cls.engine.gemm_naive_parallel
        cls.gemm_serial.argtypes = gemm_args
        cls.gemm_parallel = cls.engine.gemm_nt_fp32_exact_parallel_dispatch
        cls.gemm_parallel.argtypes = gemm_args
        cls.engine.ck_set_strict_parity.argtypes = [ctypes.c_int]

        quant_args = [FLOAT_PTR, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        cls.quant_serial = cls.lib.quantize_batch_q8_k_4row_nearest_even
        cls.quant_serial.argtypes = quant_args
        cls.quant_parallel = (
            cls.lib.quantize_batch_q8_k_4row_nearest_even_parallel_dispatch
        )
        cls.quant_parallel.argtypes = quant_args
        cls.quant_canonical_serial = cls.lib.quantize_batch_q8_k
        cls.quant_canonical_serial.argtypes = quant_args
        cls.quant_canonical_parallel = (
            cls.lib.quantize_batch_q8_k_parallel_dispatch
        )
        cls.quant_canonical_parallel.argtypes = quant_args

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
        cls.rms_generic_serial = cls.lib.rmsnorm_forward
        cls.rms_generic_serial.argtypes = rms_args
        cls.rms_generic_parallel = cls.lib.rmsnorm_forward_parallel_dispatch
        cls.rms_generic_parallel.argtypes = rms_args

        qk_norm_args = [
            FLOAT_PTR,
            FLOAT_PTR,
            FLOAT_PTR,
            FLOAT_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        cls.qk_norm_serial = cls.engine.qk_norm_forward
        cls.qk_norm_serial.argtypes = qk_norm_args
        cls.qk_norm_parallel = cls.lib.qk_norm_forward_parallel_dispatch
        cls.qk_norm_parallel.argtypes = qk_norm_args
        cls.qk_norm_llama_serial = cls.lib.qk_norm_forward_llama_production
        cls.qk_norm_llama_serial.argtypes = qk_norm_args
        cls.qk_norm_llama_parallel = (
            cls.lib.qk_norm_forward_llama_production_parallel_dispatch
        )
        cls.qk_norm_llama_parallel.argtypes = qk_norm_args

        v_norm_args = [
            FLOAT_PTR,
            FLOAT_PTR,
            FLOAT_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        cls.v_norm_serial = cls.engine.gemma4_v_norm_forward
        cls.v_norm_serial.argtypes = v_norm_args
        cls.v_norm_parallel = cls.lib.gemma4_v_norm_forward_parallel_dispatch
        cls.v_norm_parallel.argtypes = v_norm_args

        residual_args = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, ctypes.c_int, ctypes.c_int]
        cls.residual_serial = cls.lib.ck_residual_add_token_major
        cls.residual_serial.argtypes = residual_args
        cls.residual_parallel = (
            cls.lib.ck_residual_add_token_major_parallel_dispatch
        )
        cls.residual_parallel.argtypes = residual_args

        cls.memcpy_parallel = cls.lib.ck_memcpy_parallel_dispatch
        cls.memcpy_parallel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        cls.memcpy_parallel.restype = ctypes.c_void_p

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

        mla_args = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR]
        mla_args += [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            FLOAT_PTR,
            ctypes.c_size_t,
        ]
        cls.mla_serial = cls.engine.deepseek_mla_attention_f32_workspace
        cls.mla_serial.argtypes = mla_args
        cls.mla_parallel = cls.engine.deepseek_mla_attention_f32_parallel_dispatch
        cls.mla_parallel.argtypes = mla_args

        gemma4_prepare_args = [
            FLOAT_PTR,
            FLOAT_PTR,
            INT32_PTR,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16),
            FLOAT_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        cls.gemma4_prepare_q5 = cls.engine.gemma4_per_layer_prepare_forward
        cls.gemma4_prepare_q5.argtypes = gemma4_prepare_args
        cls.gemma4_prepare_bf16 = (
            cls.engine.gemma4_per_layer_prepare_bf16_forward
        )
        cls.gemma4_prepare_bf16.argtypes = gemma4_prepare_args
        gemma4_embed_args = [FLOAT_PTR] * 6
        gemma4_embed_args += [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            INT32_PTR,
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.gemma4_embed = cls.engine.gemma4_per_layer_embed_forward
        cls.gemma4_embed.argtypes = gemma4_embed_args

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

    def test_q8_k_full_grid_midpoint_and_tail(self) -> None:
        rows, width = 6, 2560
        scale = np.float32(0.0690993)
        source = np.zeros((rows, width), dtype=np.float32)
        source[:, 3 * 256 + 1] = np.float32(127) * scale
        source[:, 3 * 256 + 42] = np.float32(-24.5 + 2e-5) * scale
        source[3, 3 * 256 + 42] = np.float32(-24.5 - 2e-5) * scale
        source[5, 3 * 256 + 42] = np.float32(-24.5 - 2e-5) * scale
        row_bytes = (width // 256) * 292
        serial = np.empty(rows * row_bytes, dtype=np.uint8)
        parallel = np.empty_like(serial)

        self.quant_serial(source.ctypes.data_as(FLOAT_PTR), serial.ctypes.data, rows, width)
        self.quant_parallel(source.ctypes.data_as(FLOAT_PTR), parallel.ctypes.data, rows, width)
        np.testing.assert_array_equal(parallel, serial)

        q_offset = 3 * 292 + 4 + 42
        quantized = parallel.view(np.int8).reshape(rows, row_bytes)[:, q_offset]
        # Q8_K stores a signed scale, so these negative inputs have positive codes.
        np.testing.assert_array_equal(quantized, [24, 24, 24, 25, 24, 25])
        self.assertFalse(np.array_equal(parallel[:row_bytes], parallel[3 * row_bytes:4 * row_bytes]))

    def test_canonical_q8_k_rows_are_byte_exact(self) -> None:
        rows, width = 259, 10240
        rng = np.random.default_rng(17)
        source = rng.standard_normal((rows, width), dtype=np.float32)
        output_bytes = rows * (width // 256) * 292
        serial = np.empty(output_bytes, dtype=np.uint8)
        parallel = np.empty_like(serial)

        self.quant_canonical_serial(
            source.ctypes.data_as(FLOAT_PTR), serial.ctypes.data, rows, width
        )
        self.quant_canonical_parallel(
            source.ctypes.data_as(FLOAT_PTR), parallel.ctypes.data, rows, width
        )
        np.testing.assert_array_equal(parallel, serial)

    def test_fp32_exact_gemm_rows_are_bit_exact_in_both_modes(self) -> None:
        rows, outputs, width = 257, 63, 129
        rng = np.random.default_rng(19)
        source = rng.standard_normal((rows, width), dtype=np.float32)
        weights = rng.standard_normal((outputs, width), dtype=np.float32)
        bias = rng.standard_normal(outputs, dtype=np.float32)
        serial = np.empty((rows, outputs), dtype=np.float32)
        parallel = np.empty_like(serial)
        pointers = (
            source.ctypes.data_as(FLOAT_PTR),
            weights.ctypes.data_as(FLOAT_PTR),
            bias.ctypes.data_as(FLOAT_PTR),
        )
        for strict in (0, 1):
            self.engine.ck_set_strict_parity(strict)
            self.gemm_serial(
                *pointers, serial.ctypes.data_as(FLOAT_PTR), rows, outputs, width
            )
            self.gemm_parallel(
                *pointers, parallel.ctypes.data_as(FLOAT_PTR), rows, outputs, width
            )
            np.testing.assert_array_equal(parallel, serial)
        self.engine.ck_set_strict_parity(0)

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

    def test_generic_rmsnorm_rows_are_bit_exact_in_place(self) -> None:
        rows, width = 131, 2560
        rng = np.random.default_rng(43)
        source = rng.standard_normal((rows, width), dtype=np.float32)
        gamma = rng.standard_normal(width, dtype=np.float32)
        serial = source.copy()
        parallel = source.copy()
        serial_rstd = np.empty(rows, dtype=np.float32)
        parallel_rstd = np.empty_like(serial_rstd)

        args = (rows, width, width, ctypes.c_float(1e-6))
        self.rms_generic_serial(
            serial.ctypes.data_as(FLOAT_PTR),
            gamma.ctypes.data_as(FLOAT_PTR),
            serial.ctypes.data_as(FLOAT_PTR),
            serial_rstd.ctypes.data_as(FLOAT_PTR),
            *args,
        )
        self.rms_generic_parallel(
            parallel.ctypes.data_as(FLOAT_PTR),
            gamma.ctypes.data_as(FLOAT_PTR),
            parallel.ctypes.data_as(FLOAT_PTR),
            parallel_rstd.ctypes.data_as(FLOAT_PTR),
            *args,
        )
        np.testing.assert_array_equal(parallel, serial)
        np.testing.assert_array_equal(parallel_rstd, serial_rstd)

    def test_qk_norm_rows_are_bit_exact_in_place(self) -> None:
        heads, kv_heads, tokens, head_dim = 8, 2, 131, 256
        rng = np.random.default_rng(59)
        q_source = rng.standard_normal(
            (heads, tokens, head_dim), dtype=np.float32
        )
        k_source = rng.standard_normal(
            (kv_heads, tokens, head_dim), dtype=np.float32
        )
        q_gamma = rng.standard_normal(head_dim, dtype=np.float32)
        k_gamma = rng.standard_normal(head_dim, dtype=np.float32)
        q_serial, q_parallel = q_source.copy(), q_source.copy()
        k_serial, k_parallel = k_source.copy(), k_source.copy()
        suffix = (
            q_gamma.ctypes.data_as(FLOAT_PTR),
            k_gamma.ctypes.data_as(FLOAT_PTR),
            heads,
            kv_heads,
            tokens,
            head_dim,
            ctypes.c_float(1e-6),
        )

        self.qk_norm_serial(
            q_serial.ctypes.data_as(FLOAT_PTR),
            k_serial.ctypes.data_as(FLOAT_PTR),
            *suffix,
        )
        self.qk_norm_parallel(
            q_parallel.ctypes.data_as(FLOAT_PTR),
            k_parallel.ctypes.data_as(FLOAT_PTR),
            *suffix,
        )
        np.testing.assert_array_equal(q_parallel, q_serial)
        np.testing.assert_array_equal(k_parallel, k_serial)

    def test_qk_norm_llama_rows_are_bit_exact_in_place(self) -> None:
        heads, kv_heads, tokens, head_dim = 8, 2, 131, 256
        rng = np.random.default_rng(67)
        q_source = rng.standard_normal(
            (heads, tokens, head_dim), dtype=np.float32
        )
        k_source = rng.standard_normal(
            (kv_heads, tokens, head_dim), dtype=np.float32
        )
        q_gamma = rng.standard_normal(head_dim, dtype=np.float32)
        k_gamma = rng.standard_normal(head_dim, dtype=np.float32)
        q_serial, q_parallel = q_source.copy(), q_source.copy()
        k_serial, k_parallel = k_source.copy(), k_source.copy()
        suffix = (
            q_gamma.ctypes.data_as(FLOAT_PTR),
            k_gamma.ctypes.data_as(FLOAT_PTR),
            heads,
            kv_heads,
            tokens,
            head_dim,
            ctypes.c_float(1e-6),
        )

        self.qk_norm_llama_serial(
            q_serial.ctypes.data_as(FLOAT_PTR),
            k_serial.ctypes.data_as(FLOAT_PTR),
            *suffix,
        )
        self.qk_norm_llama_parallel(
            q_parallel.ctypes.data_as(FLOAT_PTR),
            k_parallel.ctypes.data_as(FLOAT_PTR),
            *suffix,
        )
        np.testing.assert_array_equal(q_parallel, q_serial)
        np.testing.assert_array_equal(k_parallel, k_serial)

    def test_qk_norm_llama_parallel_boundaries_are_repeatable(self) -> None:
        rng = np.random.default_rng(71)
        for heads, kv_heads, tokens, head_dim in (
            (1, 1, 1, 64),
            (3, 2, 5, 96),
            (4, 3, 7, 128),
            (8, 4, 17, 256),
        ):
            with self.subTest(
                heads=heads,
                kv_heads=kv_heads,
                tokens=tokens,
                head_dim=head_dim,
            ):
                q_source = rng.standard_normal(
                    (heads, tokens, head_dim), dtype=np.float32
                )
                k_source = rng.standard_normal(
                    (kv_heads, tokens, head_dim), dtype=np.float32
                )
                q_gamma = rng.standard_normal(head_dim, dtype=np.float32)
                k_gamma = rng.standard_normal(head_dim, dtype=np.float32)
                q_expected, k_expected = q_source.copy(), k_source.copy()
                suffix = (
                    q_gamma.ctypes.data_as(FLOAT_PTR),
                    k_gamma.ctypes.data_as(FLOAT_PTR),
                    heads,
                    kv_heads,
                    tokens,
                    head_dim,
                    ctypes.c_float(1e-6),
                )
                self.qk_norm_llama_serial(
                    q_expected.ctypes.data_as(FLOAT_PTR),
                    k_expected.ctypes.data_as(FLOAT_PTR),
                    *suffix,
                )
                for _ in range(4):
                    q_actual, k_actual = q_source.copy(), k_source.copy()
                    self.qk_norm_llama_parallel(
                        q_actual.ctypes.data_as(FLOAT_PTR),
                        k_actual.ctypes.data_as(FLOAT_PTR),
                        *suffix,
                    )
                    np.testing.assert_array_equal(q_actual, q_expected)
                    np.testing.assert_array_equal(k_actual, k_expected)

    def test_v_norm_rows_are_bit_exact_in_place(self) -> None:
        kv_heads, head_dim = 2, 256
        rng = np.random.default_rng(61)
        gamma = np.ones(head_dim, dtype=np.float32)
        for tokens in (1, 5, 17, 131):
            with self.subTest(tokens=tokens):
                source = rng.standard_normal(
                    (tokens, kv_heads, head_dim), dtype=np.float32
                )
                serial, parallel = source.copy(), source.copy()
                weighted = np.empty_like(source)
                serial_rstd = np.empty(tokens * kv_heads, dtype=np.float32)
                parallel_rstd = np.empty_like(serial_rstd)
                weighted_rstd = np.empty_like(serial_rstd)
                suffix = (tokens, kv_heads, head_dim, ctypes.c_float(1e-6))

                self.v_norm_serial(
                    serial.ctypes.data_as(FLOAT_PTR),
                    serial.ctypes.data_as(FLOAT_PTR),
                    serial_rstd.ctypes.data_as(FLOAT_PTR),
                    *suffix,
                )
                self.v_norm_parallel(
                    parallel.ctypes.data_as(FLOAT_PTR),
                    parallel.ctypes.data_as(FLOAT_PTR),
                    parallel_rstd.ctypes.data_as(FLOAT_PTR),
                    *suffix,
                )
                self.rms_serial(
                    source.ctypes.data_as(FLOAT_PTR),
                    gamma.ctypes.data_as(FLOAT_PTR),
                    weighted.ctypes.data_as(FLOAT_PTR),
                    weighted_rstd.ctypes.data_as(FLOAT_PTR),
                    tokens * kv_heads, head_dim, head_dim, ctypes.c_float(1e-6),
                )
                np.testing.assert_array_equal(parallel.view(np.uint32), serial.view(np.uint32))
                np.testing.assert_array_equal(serial.view(np.uint32), weighted.view(np.uint32))
                np.testing.assert_array_equal(parallel_rstd.view(np.uint32), serial_rstd.view(np.uint32))
                np.testing.assert_array_equal(serial_rstd.view(np.uint32), weighted_rstd.view(np.uint32))

    def test_residual_add_rows_are_bit_exact_in_place(self) -> None:
        rows, width = 131, 2560
        rng = np.random.default_rng(47)
        left = rng.standard_normal((rows, width), dtype=np.float32)
        right = rng.standard_normal((rows, width), dtype=np.float32)
        serial = left.copy()
        parallel = left.copy()

        self.residual_serial(
            serial.ctypes.data_as(FLOAT_PTR),
            right.ctypes.data_as(FLOAT_PTR),
            serial.ctypes.data_as(FLOAT_PTR),
            rows,
            width,
        )
        self.residual_parallel(
            parallel.ctypes.data_as(FLOAT_PTR),
            right.ctypes.data_as(FLOAT_PTR),
            parallel.ctypes.data_as(FLOAT_PTR),
            rows,
            width,
        )
        np.testing.assert_array_equal(parallel, serial)

    def test_large_memcpy_is_byte_exact_and_preserves_guards(self) -> None:
        guard = 257
        size = (7 << 20) + 123
        rng = np.random.default_rng(53)
        source = rng.integers(0, 256, size=size, dtype=np.uint8)
        destination = np.full(size + 2 * guard, 0xA5, dtype=np.uint8)
        destination_address = destination.ctypes.data + guard

        returned = self.memcpy_parallel(
            destination_address, source.ctypes.data, source.nbytes
        )

        self.assertEqual(returned, destination_address)
        np.testing.assert_array_equal(destination[:guard], 0xA5)
        np.testing.assert_array_equal(
            destination[guard : guard + size], source
        )
        np.testing.assert_array_equal(destination[guard + size :], 0xA5)

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

    def test_dynamic_mla_token_rows_are_bit_exact(self) -> None:
        tokens, heads, kv_heads, qk_dim, v_dim = 17, 4, 2, 8, 6
        rng = np.random.default_rng(23)
        query = rng.standard_normal((tokens, heads, qk_dim), dtype=np.float32)
        key = rng.standard_normal((tokens, kv_heads, qk_dim), dtype=np.float32)
        value = rng.standard_normal((tokens, kv_heads, v_dim), dtype=np.float32)
        serial = np.empty((tokens, heads, v_dim), dtype=np.float32)
        parallel = np.empty_like(serial)
        serial_scores = np.empty(tokens, dtype=np.float32)
        parallel_scores = np.empty((heads, tokens), dtype=np.float32)
        common = (
            query.ctypes.data_as(FLOAT_PTR),
            key.ctypes.data_as(FLOAT_PTR),
            value.ctypes.data_as(FLOAT_PTR),
        )
        scale = ctypes.c_float(qk_dim ** -0.5)

        self.mla_serial(
            *common,
            serial.ctypes.data_as(FLOAT_PTR),
            heads,
            kv_heads,
            tokens,
            qk_dim,
            v_dim,
            scale,
            serial_scores.ctypes.data_as(FLOAT_PTR),
            serial_scores.nbytes,
        )
        self.mla_parallel(
            *common,
            parallel.ctypes.data_as(FLOAT_PTR),
            heads,
            kv_heads,
            tokens,
            qk_dim,
            v_dim,
            scale,
            parallel_scores.ctypes.data_as(FLOAT_PTR),
            parallel_scores.nbytes,
        )
        np.testing.assert_array_equal(parallel, serial)

    def test_gemma4_prepare_token_rows_are_bit_exact(self) -> None:
        tokens, layers, embed_dim, per_layer_dim, vocab = 11, 3, 8, 256, 7
        rng = np.random.default_rng(29)
        hidden = rng.standard_normal((tokens, embed_dim), dtype=np.float32)
        token_ids = rng.integers(0, vocab, size=tokens, dtype=np.int32)
        model_proj = rng.integers(
            0, 1 << 16, size=(layers, per_layer_dim, embed_dim), dtype=np.uint16
        )
        proj_norm = rng.standard_normal(per_layer_dim, dtype=np.float32)
        bf16_embeddings = rng.integers(
            0, 1 << 16, size=(vocab, layers, per_layer_dim), dtype=np.uint16
        )
        q5_block_bytes = 2 + 2 + 12 + 32 + 128
        q5_embeddings = rng.integers(
            0, 256, size=vocab * layers * q5_block_bytes, dtype=np.uint8
        )
        q5_blocks = q5_embeddings.reshape(vocab * layers, q5_block_bytes)
        q5_blocks[:, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)
        q5_blocks[:, 2:4] = np.array([0x00, 0x38], dtype=np.uint8)

        common = (
            hidden.ctypes.data_as(FLOAT_PTR),
            token_ids.ctypes.data_as(INT32_PTR),
        )
        suffix = (
            model_proj.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            proj_norm.ctypes.data_as(FLOAT_PTR),
            tokens,
            layers,
            embed_dim,
            per_layer_dim,
            vocab,
            ctypes.c_float(1.0e-6),
        )
        for function, embeddings in (
            (self.gemma4_prepare_q5, q5_embeddings),
            (self.gemma4_prepare_bf16, bf16_embeddings),
        ):
            serial = np.empty((tokens, layers, per_layer_dim), dtype=np.float32)
            parallel = np.empty_like(serial)
            os.environ["CK_DISABLE_GEMMA4_PREPARE_PARALLEL"] = "1"
            try:
                function(
                    serial.ctypes.data_as(FLOAT_PTR),
                    *common,
                    embeddings.ctypes.data_as(ctypes.c_void_p),
                    *suffix,
                )
            finally:
                os.environ.pop("CK_DISABLE_GEMMA4_PREPARE_PARALLEL", None)
            function(
                parallel.ctypes.data_as(FLOAT_PTR),
                *common,
                embeddings.ctypes.data_as(ctypes.c_void_p),
                *suffix,
            )
            np.testing.assert_array_equal(parallel.view(np.uint32), serial.view(np.uint32))

    def test_gemma4_prepare_rounds_bf16_projection_input(self) -> None:
        # Both inputs are exactly halfway between adjacent BF16 values.
        hidden = np.asarray([[1.00390625, -1.01171875]], dtype=np.float32)
        token_ids = np.zeros(1, dtype=np.int32)
        weights = np.tile(
            np.asarray([[0x3F80, 0xBF80], [0x3F00, 0x3E80]], dtype=np.uint16),
            (128, 1),
        ).reshape(1, 256, 2)
        norm = np.ones(256, dtype=np.float32)
        bf16_tokens = np.zeros((1, 1, 256), dtype=np.uint16)
        q5_tokens = np.zeros(2 + 2 + 12 + 32 + 128, dtype=np.uint8)

        bits = hidden.view(np.uint32)
        rounded_bits = (
            bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
        ) & np.uint32(0xFFFF0000)
        rounded = rounded_bits.view(np.float32)[0]
        np.testing.assert_array_equal(rounded, np.asarray([1.0, -1.015625], dtype=np.float32))
        float_weights = (weights.astype(np.uint32) << 16).view(np.float32).reshape(256, 2)
        projection = (float_weights @ rounded) / np.float32(math.sqrt(2.0))
        scale = np.float32(
            1.0 / math.sqrt(float(np.mean(projection.astype(np.float64) ** 2)) + 1.0e-6)
        )
        expected = projection * scale * np.float32(1.0 / math.sqrt(2.0))
        unrounded = (float_weights @ hidden[0]) / np.float32(math.sqrt(2.0))
        old_scale = np.float32(
            1.0 / math.sqrt(float(np.mean(unrounded.astype(np.float64) ** 2)) + 1.0e-6)
        )
        old = unrounded * old_scale * np.float32(1.0 / math.sqrt(2.0))
        self.assertGreater(float(np.max(np.abs(expected - old))), 1.0e-4)

        for function, tokens in (
            (self.gemma4_prepare_q5, q5_tokens),
            (self.gemma4_prepare_bf16, bf16_tokens),
        ):
            out = np.full((1, 1, 256), np.nan, dtype=np.float32)
            function(
                out.ctypes.data_as(FLOAT_PTR),
                hidden.ctypes.data_as(FLOAT_PTR),
                token_ids.ctypes.data_as(INT32_PTR),
                tokens.ctypes.data_as(ctypes.c_void_p),
                weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                norm.ctypes.data_as(FLOAT_PTR),
                1, 1, 2, 256, 1, ctypes.c_float(1.0e-6),
            )
            np.testing.assert_allclose(out[0, 0], expected, rtol=0, atol=2.0e-7)

    def test_gemma4_prepare_matches_bf16_avx2_reduction(self) -> None:
        build_flags = (ROOT / "build/.ck_build_flags").read_text().splitlines()
        cflags = next(line[7:] for line in build_flags if line.startswith("CFLAGS="))
        cc = next(line[3:] for line in build_flags if line.startswith("CC="))
        if not _compiler_defines_avx2(cflags, cc):
            self.skipTest("AVX2 BF16 projection contract requires AVX2")
        width = 2560
        rng = np.random.default_rng(137)
        hidden = rng.standard_normal((1, width), dtype=np.float32)
        weight_values = rng.standard_normal((2, width), dtype=np.float32)

        def to_bf16(values: np.ndarray) -> np.ndarray:
            bits = values.view(np.uint32)
            return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

        weight = np.zeros((1, 256, width), dtype=np.uint16)
        weight[0, :2] = to_bf16(weight_values)
        rounded = (to_bf16(hidden).astype(np.uint32) << 16).view(np.float32)[0]
        float_weights = (weight[0, :2].astype(np.uint32) << 16).view(np.float32)
        reference_dot = np.zeros(2, dtype=np.float32)
        scalar_dot = np.zeros(2, dtype=np.float32)
        for output in range(2):
            accumulators = np.zeros((4, 8), dtype=np.float32)
            for start in range(0, width, 32):
                for group in range(4):
                    left = float_weights[output, start + group * 8:start + (group + 1) * 8]
                    right = rounded[start + group * 8:start + (group + 1) * 8]
                    accumulators[group] += left * right
            lanes = (accumulators[0] + accumulators[2]) + (accumulators[1] + accumulators[3])
            halves = lanes[:4] + lanes[4:]
            reference_dot[output] = np.float32((halves[0] + halves[2]) + (halves[1] + halves[3]))
            for channel in range(width):
                scalar_dot[output] = np.float32(
                    scalar_dot[output] + float_weights[output, channel] * rounded[channel]
                )

        def normalize(dot: np.ndarray) -> np.ndarray:
            projection = dot * np.float32(1.0 / math.sqrt(width))
            scale = np.float32(1.0 / math.sqrt(float(np.sum(
                projection.astype(np.float64) ** 2, dtype=np.float64) / 256.0
            ) + 1.0e-6))
            return projection * scale * np.float32(1.0 / math.sqrt(2.0))

        expected = normalize(reference_dot)
        self.assertGreater(float(np.max(np.abs(expected - normalize(scalar_dot)))), 4.0e-6)
        out = np.empty((1, 1, 256), dtype=np.float32)
        token_ids = np.zeros(1, dtype=np.int32)
        tokens = np.zeros((1, 1, 256), dtype=np.uint16)
        gamma = np.ones(256, dtype=np.float32)
        self.gemma4_prepare_bf16(
            out.ctypes.data_as(FLOAT_PTR), hidden.ctypes.data_as(FLOAT_PTR),
            token_ids.ctypes.data_as(INT32_PTR), tokens.ctypes.data_as(ctypes.c_void_p),
            weight.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            gamma.ctypes.data_as(FLOAT_PTR), 1, 1, width, 256, 1,
            ctypes.c_float(1.0e-6),
        )
        np.testing.assert_allclose(out[0, 0, :2], expected, rtol=0, atol=3.0e-6)
        np.testing.assert_array_equal(out[0, 0, 2:], 0.0)

    def test_gemma4_prepare_bf16_text_only_token_stream(self) -> None:
        hidden = np.zeros((2, 2), dtype=np.float32)
        token_ids = np.asarray([1, 0], dtype=np.int32)
        embeddings = np.zeros((2, 1, 256), dtype=np.uint16)
        embeddings[1, 0, 0] = 0x3F80  # 1.0 BF16
        embeddings[0, 0, 1] = 0xBF00  # -0.5 BF16
        weights = np.zeros((1, 256, 2), dtype=np.uint16)
        norm = np.ones(256, dtype=np.float32)
        out = np.full((2, 1, 256), np.nan, dtype=np.float32)
        self.gemma4_prepare_bf16(
            out.ctypes.data_as(FLOAT_PTR),
            hidden.ctypes.data_as(FLOAT_PTR),
            token_ids.ctypes.data_as(INT32_PTR),
            embeddings.ctypes.data_as(ctypes.c_void_p),
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            norm.ctypes.data_as(FLOAT_PTR),
            2, 1, 2, 256, 2, ctypes.c_float(1.0e-6),
        )
        expected = np.zeros_like(out)
        expected[0, 0, 0] = np.float32(16.0 * 0.7071067811865475)
        expected[1, 0, 1] = np.float32(-8.0 * 0.7071067811865475)
        np.testing.assert_allclose(out, expected, rtol=0, atol=1.0e-6)

    def test_gemma4_embed_token_rows_are_bit_exact(self) -> None:
        tokens, layers, layer = 13, 4, 2
        embed_dim, per_layer_dim = 8, 256
        rng = np.random.default_rng(31)
        hidden = rng.standard_normal((tokens, embed_dim), dtype=np.float32)
        per_layer_input = rng.standard_normal(
            (tokens, layers, per_layer_dim), dtype=np.float32
        )
        inp_gate = rng.standard_normal(
            (per_layer_dim, embed_dim), dtype=np.float32
        )
        projection = rng.standard_normal(
            (embed_dim, per_layer_dim), dtype=np.float32
        )
        post_norm = rng.standard_normal(embed_dim, dtype=np.float32)
        out_scale = np.asarray([0.875], dtype=np.float32)

        serial = hidden.copy()
        parallel = hidden.copy()
        common = (
            per_layer_input.ctypes.data_as(FLOAT_PTR),
            inp_gate.ctypes.data_as(FLOAT_PTR),
            projection.ctypes.data_as(FLOAT_PTR),
            post_norm.ctypes.data_as(FLOAT_PTR),
            out_scale.ctypes.data_as(FLOAT_PTR),
            tokens,
            layer,
            layers,
            embed_dim,
            per_layer_dim,
            ctypes.c_float(1.0e-6),
            None, 0, 0,
        )
        os.environ["CK_DISABLE_GEMMA4_EMBED_PARALLEL"] = "1"
        try:
            self.gemma4_embed(serial.ctypes.data_as(FLOAT_PTR), *common)
        finally:
            os.environ.pop("CK_DISABLE_GEMMA4_EMBED_PARALLEL", None)
        self.gemma4_embed(parallel.ctypes.data_as(FLOAT_PTR), *common)
        np.testing.assert_array_equal(parallel.view(np.uint32), serial.view(np.uint32))

    def test_gemma4_embed_segment_tail_uses_single_row_contract(self) -> None:
        tokens, width = 3, 256
        rng = np.random.default_rng(20261014)
        hidden = rng.standard_normal((tokens, width), dtype=np.float32) * 0.1
        stream = rng.standard_normal((tokens, 1, width), dtype=np.float32)
        gate = rng.standard_normal((width, width), dtype=np.float32) * 0.1
        projection = rng.standard_normal((width, width), dtype=np.float32) * 0.1
        norm = np.ones(width, dtype=np.float32)
        scale = np.ones(1, dtype=np.float32)
        segments = np.asarray([2, 1], dtype=np.int32)

        def run(source: np.ndarray, inp: np.ndarray, lengths: np.ndarray | None,
                singleton_tail: int = 0) -> np.ndarray:
            result = source.copy()
            self.gemma4_embed(
                result.ctypes.data_as(FLOAT_PTR),
                inp.ctypes.data_as(FLOAT_PTR),
                gate.ctypes.data_as(FLOAT_PTR),
                projection.ctypes.data_as(FLOAT_PTR),
                norm.ctypes.data_as(FLOAT_PTR),
                scale.ctypes.data_as(FLOAT_PTR),
                len(result), 0, 1, width, width, ctypes.c_float(1.0e-6),
                None if lengths is None else lengths.ctypes.data_as(INT32_PTR),
                0 if lengths is None else len(lengths), singleton_tail,
            )
            return result

        segmented = run(hidden, stream, segments, 1)
        grouped = run(hidden, stream, None)
        singleton = run(hidden[2:], stream[2:], None)
        np.testing.assert_array_equal(segmented[:2].view(np.uint32),
                                      grouped[:2].view(np.uint32))
        np.testing.assert_array_equal(segmented[2:].view(np.uint32),
                                      singleton.view(np.uint32))
        with_empty = run(hidden, stream, np.asarray([0, 2, 1], dtype=np.int32), 1)
        np.testing.assert_array_equal(with_empty.view(np.uint32),
                                      segmented.view(np.uint32))

        trailing_text = run(hidden, stream, np.asarray([1, 0, 2], dtype=np.int32), 1)
        for row in range(tokens):
            one = run(hidden[row:row + 1], stream[row:row + 1], None)
            np.testing.assert_array_equal(trailing_text[row].view(np.uint32),
                                          one[0].view(np.uint32))
        np.testing.assert_array_equal(run(hidden, stream, None).view(np.uint32),
                                      grouped.view(np.uint32))

        bad = hidden.copy()
        invalid = np.asarray([2, 2], dtype=np.int32)
        self.gemma4_embed(
            bad.ctypes.data_as(FLOAT_PTR), stream.ctypes.data_as(FLOAT_PTR),
            gate.ctypes.data_as(FLOAT_PTR), projection.ctypes.data_as(FLOAT_PTR),
            norm.ctypes.data_as(FLOAT_PTR), scale.ctypes.data_as(FLOAT_PTR),
            tokens, 0, 1, width, width, ctypes.c_float(1.0e-6),
            invalid.ctypes.data_as(INT32_PTR), len(invalid), 1,
        )
        np.testing.assert_array_equal(bad.view(np.uint32), hidden.view(np.uint32))

    def test_gemma4_benchmark_binding_calls_current_abi(self) -> None:
        path = ROOT / "benchmarks/bench_final_logits_q4k_q8k.py"
        spec = importlib.util.spec_from_file_location("gemma4_final_logit_bench", path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bind = module._bind_gemma4_embed(self.engine)
        self.assertEqual(len(bind.argtypes), 15)

        width, branch_width = 8, 256
        hidden = np.ones(width, dtype=np.float32)
        stream = np.ones(branch_width, dtype=np.float32)
        gate = np.zeros((branch_width, width), dtype=np.float32)
        gate[0, 0] = 1.0
        projection = np.zeros((width, branch_width), dtype=np.float32)
        projection[0, 0] = 1.0
        norm = np.ones(width, dtype=np.float32)
        scale = np.ones(1, dtype=np.float32)
        module._call_gemma4_embed_one(
            bind, hidden, stream, gate, projection, norm, scale,
            1, width, branch_width, 1.0e-6,
        )
        self.assertGreater(hidden[0], 1.0)

    def test_gemma4_embed_uses_llama_rmsnorm_reduction(self) -> None:
        width = 2560
        branch_width = 256
        hidden = np.random.default_rng(47).standard_normal((1, width), dtype=np.float32)
        hidden[0, 0] = 10.0
        stream = np.zeros((1, 1, branch_width), dtype=np.float32)
        stream[0, 0, 0] = 0.1
        gate = np.zeros((branch_width, width), dtype=np.float32)
        gate[0, 0] = 1.0
        projection = np.zeros((width, branch_width), dtype=np.float32)
        projection[:, 0] = (np.random.default_rng(458).standard_normal(
            width, dtype=np.float32) * np.float32(20))
        norm = np.ones(width, dtype=np.float32)
        scale = np.ones(1, dtype=np.float32)
        branch = projection[:, 0].copy()
        expected_norm = np.empty(width, dtype=np.float32)
        rmsnorm = self.engine.rmsnorm_forward_llama_production
        rmsnorm.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        rmsnorm(branch.ctypes.data_as(FLOAT_PTR), norm.ctypes.data_as(FLOAT_PTR),
                expected_norm.ctypes.data_as(FLOAT_PTR), None,
                1, width, width, ctypes.c_float(1.0e-6))
        old_scale = np.float32(1.0 / math.sqrt(float(np.float32(
            sum(float(x) * float(x) for x in branch) / width
        ) + np.float32(1.0e-6))))
        self.assertFalse(np.array_equal(expected_norm.view(np.uint32),
                                        (branch * old_scale).view(np.uint32)))
        expected = hidden + expected_norm
        actual = hidden.copy()
        self.gemma4_embed(
            actual.ctypes.data_as(FLOAT_PTR), stream.ctypes.data_as(FLOAT_PTR),
            gate.ctypes.data_as(FLOAT_PTR), projection.ctypes.data_as(FLOAT_PTR),
            norm.ctypes.data_as(FLOAT_PTR), scale.ctypes.data_as(FLOAT_PTR),
            1, 0, 1, width, branch_width, ctypes.c_float(1.0e-6), None, 0, 0,
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_gemma4_embed_matches_half_rounded_gelu_oracle(self) -> None:
        hidden = np.asarray([[0.13, 0.7, -0.1, 0.2]], dtype=np.float32)
        per_layer_input = np.zeros((1, 1, 256), dtype=np.float32)
        per_layer_input[0, 0, :2] = 1.0
        inp_gate = np.zeros((256, 4), dtype=np.float32)
        inp_gate[0, 0] = inp_gate[1, 1] = 1.0
        projection = np.zeros((4, 256), dtype=np.float32)
        projection[0, 0] = projection[1, 1] = 1.0
        post_norm = np.ones(4, dtype=np.float32)
        out_scale = np.ones(1, dtype=np.float32)

        def gelu(x: float, half_rounded: bool) -> float:
            if half_rounded:
                x = float(np.float16(x))
            y = 0.5 * x * (1.0 + math.tanh(0.7978845608028654 * x *
                                             (1.0 + 0.044715 * x * x)))
            return float(np.float16(y)) if half_rounded else y

        def expected(half_rounded: bool) -> np.ndarray:
            branch = np.asarray(
                [gelu(0.13, half_rounded), gelu(0.7, half_rounded), 0.0, 0.0],
                dtype=np.float32,
            )
            scale = np.float32(1.0 / np.sqrt(np.float32(
                np.sum(branch.astype(np.float64) ** 2) / 4.0 + 1.0e-6
            )))
            return hidden[0] + branch * scale

        half_oracle = expected(True)
        full_oracle = expected(False)
        self.assertGreater(np.max(np.abs(half_oracle - full_oracle)), 1.0e-4)
        actual = hidden.copy()
        self.gemma4_embed(
            actual.ctypes.data_as(FLOAT_PTR),
            per_layer_input.ctypes.data_as(FLOAT_PTR),
            inp_gate.ctypes.data_as(FLOAT_PTR),
            projection.ctypes.data_as(FLOAT_PTR),
            post_norm.ctypes.data_as(FLOAT_PTR),
            out_scale.ctypes.data_as(FLOAT_PTR),
            1, 0, 1, 4, 256, ctypes.c_float(1.0e-6), None, 0, 0,
        )
        np.testing.assert_allclose(actual[0], half_oracle, atol=1.0e-6, rtol=0)

    def test_gemma4_embed_matches_pinned_cpu_gelu_table_boundaries(self) -> None:
        # llama.cpp 19e28a27702117d8f2eb16b825b9a308111f67d9,
        # ggml-cpu built with GGML_GELU_FP16. Values read from its GELU table.
        values = np.asarray(
            [-10.01, -10.0, -2.0, -0.7, -0.125, 0.0, 0.13,
             0.7, 0.700244140625, 1.00048828125, 1.5, 9.99, 10.0, 10.01],
            dtype=np.float32,
        )
        table = np.asarray(
            [0.0, 0.0, -0.04541015625, -0.16943359375,
             -0.0562744140625, 0.0, 0.07171630859375,
             0.53076171875, 0.53076171875, 0.84130859375,
             1.3994140625, 9.9921875, 10.0, 10.01],
            dtype=np.float32,
        )
        dim = len(values)
        hidden = values.reshape(1, dim).copy()
        per_layer_input = np.zeros((1, 1, 256), dtype=np.float32)
        per_layer_input[0, 0, :dim] = 1.0
        inp_gate = np.zeros((256, dim), dtype=np.float32)
        projection = np.zeros((dim, 256), dtype=np.float32)
        for i in range(dim):
            inp_gate[i, i] = projection[i, i] = 1.0
        post_norm = np.ones(dim, dtype=np.float32)
        out_scale = np.ones(1, dtype=np.float32)
        scale = np.float32(1.0 / math.sqrt(float(np.mean(table.astype(np.float64) ** 2)) + 1.0e-6))
        expected = hidden[0] + table * scale
        actual = hidden.copy()
        self.gemma4_embed(
            actual.ctypes.data_as(FLOAT_PTR),
            per_layer_input.ctypes.data_as(FLOAT_PTR),
            inp_gate.ctypes.data_as(FLOAT_PTR),
            projection.ctypes.data_as(FLOAT_PTR),
            post_norm.ctypes.data_as(FLOAT_PTR),
            out_scale.ctypes.data_as(FLOAT_PTR),
            1, 0, 1, dim, 256, ctypes.c_float(1.0e-6), None, 0, 0,
        )
        np.testing.assert_allclose(actual[0], expected, rtol=0, atol=3.0e-6)


if __name__ == "__main__":
    unittest.main()
