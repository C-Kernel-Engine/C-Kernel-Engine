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
INT32_PTR = ctypes.POINTER(ctypes.c_int32)


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

    def test_v_norm_rows_are_bit_exact_in_place(self) -> None:
        tokens, kv_heads, head_dim = 131, 2, 256
        rng = np.random.default_rng(61)
        source = rng.standard_normal(
            (tokens, kv_heads, head_dim), dtype=np.float32
        )
        serial, parallel = source.copy(), source.copy()
        serial_rstd = np.empty(tokens * kv_heads, dtype=np.float32)
        parallel_rstd = np.empty_like(serial_rstd)
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
        np.testing.assert_array_equal(parallel, serial)
        np.testing.assert_array_equal(parallel_rstd, serial_rstd)

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
        )
        os.environ["CK_DISABLE_GEMMA4_EMBED_PARALLEL"] = "1"
        try:
            self.gemma4_embed(serial.ctypes.data_as(FLOAT_PTR), *common)
        finally:
            os.environ.pop("CK_DISABLE_GEMMA4_EMBED_PARALLEL", None)
        self.gemma4_embed(parallel.ctypes.data_as(FLOAT_PTR), *common)
        np.testing.assert_array_equal(parallel.view(np.uint32), serial.view(np.uint32))


if __name__ == "__main__":
    unittest.main()
