import ctypes
import json
import math
import os
import struct
import subprocess
import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from version.v8.scripts import replay_attention_from_dumps_v8 as replay


ROOT = Path(__file__).resolve().parents[1]


class DirectLayoutAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="cke-direct-attention-")
        cls._lib_path = Path(cls._tmp.name) / "libattention.so"
        subprocess.run([
            "gcc", "-std=gnu11", "-O2", "-shared", "-fPIC", "-mavx2", "-mfma", "-mf16c",
            "-I", str(ROOT / "include"),
            str(ROOT / "src" / "kernels" / "attention_kernels.c"),
            str(ROOT / "src" / "kernels" / "attention_kernels_sliding.c"),
            str(ROOT / "src" / "kernels" / "attention_flash_true.c"),
            str(ROOT / "src" / "kernels" / "softmax_kernels.c"),
            str(ROOT / "src" / "kernels" / "gemm_kernels_bf16.c"),
            str(ROOT / "src" / "ckernel_strict.c"),
            str(ROOT / "src" / "ck_threadpool.c"),
            "-lm", "-lpthread", "-o", str(cls._lib_path),
        ], check=True)
        cls._lib = ctypes.CDLL(str(cls._lib_path))
        signature = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        cls._lib.attention_forward_causal_head_major_gqa_flash_strided.argtypes = signature
        cls._lib.attention_forward_causal_head_major_gqa_flash_strided_token_output.argtypes = signature
        cls._lib.attention_forward_causal_head_major_gqa_flash_strided_gemma4.argtypes = signature
        cls._lib.attention_forward_causal_head_major_gqa_flash_strided_gemma4_token_output.argtypes = signature
        sliding_signature = signature + [ctypes.c_int]
        cls._lib.attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4.argtypes = sliding_signature
        cls._lib.attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4_token_output.argtypes = sliding_signature
        mixed_signature = signature + [ctypes.c_int, ctypes.c_int]
        cls._lib.attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4.argtypes = mixed_signature
        cls._lib.attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4_token_output.argtypes = mixed_signature
        prefill_workspace_signature = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract_workspace.argtypes = (
            prefill_workspace_signature
        )
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract_workspace.restype = ctypes.c_int
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gemma4_workspace.argtypes = (
            prefill_workspace_signature
        )
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gemma4_workspace.restype = ctypes.c_int
        gemma_sliding_prefill_signature = prefill_workspace_signature[:11] + [
            ctypes.c_int,
        ] + prefill_workspace_signature[11:]
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_sliding_gemma4_workspace.argtypes = (
            gemma_sliding_prefill_signature
        )
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_sliding_gemma4_workspace.restype = ctypes.c_int
        cls._lib.attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_sliding_gemma4_workspace.argtypes = (
            gemma_sliding_prefill_signature + [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        )
        cls._lib.attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_sliding_gemma4_workspace.restype = ctypes.c_int
        gemma_decode_signature = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        cls._lib.attention_forward_decode_head_major_gqa_f16cache_gemma4_contract.argtypes = (
            gemma_decode_signature
        )
        cls._lib.attention_forward_decode_head_major_gqa_f16cache_gemma4_contract.restype = ctypes.c_int
        cls._lib.attention_forward_decode_head_major_gqa_f16cache_regular_noscale.argtypes = (
            gemma_decode_signature
        )
        cls._lib.attention_forward_decode_head_major_gqa_f16cache_regular_noscale.restype = ctypes.c_int
        cls._lib.attention_forward_decode_head_major_gqa_f16cache_sliding_gemma4_contract.argtypes = (
            gemma_decode_signature[:10] + [ctypes.c_int] + gemma_decode_signature[10:]
        )
        cls._lib.attention_forward_decode_head_major_gqa_f16cache_sliding_gemma4_contract.restype = ctypes.c_int
        qtile_schedule_signature = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_qtile64_schedule.argtypes = (
            qtile_schedule_signature
        )
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_qtile64_schedule.restype = ctypes.c_int
        reuse_signature = qtile_schedule_signature[:-1] + [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
        ]
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gqa_reuse_config.argtypes = (
            reuse_signature
        )
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gqa_reuse_config.restype = ctypes.c_int
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gqa_reuse_workspace_bytes.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gqa_reuse_workspace_bytes.restype = ctypes.c_size_t
        auto_signature = qtile_schedule_signature[:-1] + [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace.argtypes = (
            auto_signature
        )
        cls._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace.restype = ctypes.c_int
        cls._lib.attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_contract_workspace.argtypes = (
            prefill_workspace_signature
            + [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        )
        cls._lib.attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_contract_workspace.restype = ctypes.c_int

    def test_gemma4_f16cache_prefill_honors_unit_scale_and_sliding_range(self):
        heads, kv_heads, tokens, dim = 2, 1, 5, 8
        q = array("f", (0.1 + math.sin(i * 0.17) for i in range(heads * tokens * dim)))

        def fp16_bits(value):
            return int.from_bytes(struct.pack("<e", value), "little")

        k = array("H", (fp16_bits(math.cos(i * 0.11)) for i in range(kv_heads * tokens * dim)))
        v = array("H", (fp16_bits(math.sin(i * 0.13 + 0.4)) for i in range(kv_heads * tokens * dim)))
        workspace = array("f", [0.0]) * (2 * heads * dim)

        def float_pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        def half_pointer(values):
            return (ctypes.c_uint16 * len(values)).from_buffer(values)

        def run(function, *extra):
            output = array("f", [0.0]) * (heads * tokens * dim)
            status = function(
                float_pointer(q), half_pointer(k), half_pointer(v),
                float_pointer(output), heads, kv_heads, tokens, 0, tokens,
                dim, dim, *extra, 2, float_pointer(workspace),
                len(workspace) * ctypes.sizeof(ctypes.c_float),
            )
            self.assertEqual(status, 0)
            return output

        full = run(
            self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gemma4_workspace
        )
        unbounded_sliding = run(
            self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_sliding_gemma4_workspace,
            tokens,
        )
        bounded_sliding = run(
            self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_sliding_gemma4_workspace,
            2,
        )
        scaled = run(
            self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract_workspace
        )

        self.assertEqual(full.tobytes(), unbounded_sliding.tobytes())
        self.assertNotEqual(full.tobytes(), bounded_sliding.tobytes())
        self.assertNotEqual(full.tobytes(), scaled.tobytes())

    def test_gemma4_long_prefill_uses_unit_scale_tiled_reduction(self):
        heads, kv_heads, past, tokens, dim, capacity = 4, 2, 5, 72, 16, 80
        q = array("f", (math.sin(i * 0.013) for i in range(heads * tokens * dim)))
        scaled_q = array("f", (value * math.sqrt(dim) for value in q))

        def half_bits(value):
            return int.from_bytes(struct.pack("<e", value), "little")

        k = array("H", (half_bits(math.cos(i * 0.011)) for i in range(kv_heads * capacity * dim)))
        v = array("H", (half_bits(math.sin(i * 0.017)) for i in range(kv_heads * capacity * dim)))
        workspace = array("f", [0.0]) * (2 * heads * dim)

        def fp(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        def hp(values):
            return (ctypes.c_uint16 * len(values)).from_buffer(values)

        def run(function, query, sliding, reduction):
            output = array("f", [0.0]) * (heads * tokens * dim)
            args = (
                fp(query), hp(k), hp(v), fp(output), heads, kv_heads,
                tokens, past, capacity, dim, dim,
            )
            if sliding is not None:
                args += (sliding,)
            status = function(
                *args, reduction, fp(workspace), len(workspace) * 4,
            )
            self.assertEqual(status, 0)
            return output

        gemma = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_sliding_gemma4_workspace
        generic = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract_workspace
        tiled = run(gemma, q, capacity, 3)
        expected_tiled = run(generic, scaled_q, None, 3)
        self.assertEqual(tiled.tobytes(), expected_tiled.tobytes())
        self.assertNotEqual(tiled.tobytes(), run(gemma, q, capacity, 2).tobytes())
        self.assertEqual(
            run(gemma, q, 16, 3).tobytes(),
            run(gemma, q, 16, 2).tobytes(),
        )

    def test_gemma4_segmented_prefill_restarts_tiled_policy(self):
        heads, kv_heads, tokens, dim, capacity = 4, 2, 91, 16, 100
        segments = (5, 72, 14)
        q = array("f", (math.sin(i * 0.013) for i in range(heads * tokens * dim)))

        def half_bits(value):
            return int.from_bytes(struct.pack("<e", value), "little")

        k = array("H", (half_bits(math.cos(i * 0.011)) for i in range(kv_heads * capacity * dim)))
        v = array("H", (half_bits(math.sin(i * 0.017)) for i in range(kv_heads * capacity * dim)))
        workspace = array("f", [0.0]) * (2 * heads * dim)
        actual = array("f", [0.0]) * (heads * tokens * dim)
        expected = array("f", [0.0]) * (heads * tokens * dim)
        unsegmented = array("f", [0.0]) * (heads * tokens * dim)

        def fp(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        def hp(values):
            return (ctypes.c_uint16 * len(values)).from_buffer(values)

        fn = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_sliding_gemma4_workspace
        segmented_fn = self._lib.attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_sliding_gemma4_workspace
        args = (fp(q), hp(k), hp(v), fp(actual), heads, kv_heads,
                tokens, 0, capacity, dim, dim, capacity, 3,
                fp(workspace), len(workspace) * 4)
        plan = (ctypes.c_int * len(segments))(*segments)
        self.assertEqual(segmented_fn(*args, plan, len(segments)), 0)

        offset = 0
        for rows in segments:
            query = array("f")
            for head in range(heads):
                start = (head * tokens + offset) * dim
                query.extend(q[start:start + rows * dim])
            result = array("f", [0.0]) * (heads * rows * dim)
            self.assertEqual(fn(
                fp(query), hp(k), hp(v), fp(result), heads, kv_heads,
                rows, offset, capacity, dim, dim, capacity, 3,
                fp(workspace), len(workspace) * 4,
            ), 0)
            for head in range(heads):
                dst = (head * tokens + offset) * dim
                src = head * rows * dim
                expected[dst:dst + rows * dim] = result[src:src + rows * dim]
            offset += rows
        self.assertEqual(actual.tobytes(), expected.tobytes())
        regular_tail = array("f", [0.0]) * len(actual)
        regular_args = list(args)
        regular_args[3] = fp(regular_tail)
        regular_args[12] = 6  # Tiled visual segment, regular short-query tail.
        self.assertEqual(segmented_fn(*regular_args, plan, len(segments)), 0)
        tail_q = array("f")
        for head in range(heads):
            start = (head * tokens + segments[0] + segments[1]) * dim
            tail_q.extend(q[start:start + dim])
        direct_tail = array("f", [0.0]) * (heads * dim)
        self.assertEqual(
            self._lib.attention_forward_decode_head_major_gqa_f16cache_regular_noscale(
                fp(tail_q), hp(k), hp(v), fp(direct_tail), heads, kv_heads,
                segments[0] + segments[1] + 1, capacity, dim, dim, capacity,
            ), 0,
        )
        for head in range(heads):
            src = (head * tokens + segments[0] + segments[1]) * dim
            self.assertEqual(
                regular_tail[src:src + dim].tobytes(),
                direct_tail[head * dim:(head + 1) * dim].tobytes(),
            )
        self.assertNotEqual(regular_tail.tobytes(), actual.tobytes())
        self.assertEqual(fn(
            fp(q), hp(k), hp(v), fp(unsegmented), heads, kv_heads,
            tokens, 0, capacity, dim, dim, capacity, 3,
            fp(workspace), len(workspace) * 4,
        ), 0)
        self.assertNotEqual(actual.tobytes(), unsegmented.tobytes())

        zero_plan = (ctypes.c_int * 3)(0, tokens, 0)
        zero_output = array("f", [0.0]) * len(actual)
        self.assertEqual(segmented_fn(
            fp(q), hp(k), hp(v), fp(zero_output), heads, kv_heads,
            tokens, 0, capacity, dim, dim, capacity, 3,
            fp(workspace), len(workspace) * 4, zero_plan, 3,
        ), 0)
        self.assertEqual(zero_output.tobytes(), unsegmented.tobytes())

        for invalid in ((5, 72, 13), (5, -1, 87), (5, 92, -6)):
            sentinel = array("f", [123.0]) * len(actual)
            bad_plan = (ctypes.c_int * 3)(*invalid)
            self.assertNotEqual(segmented_fn(
                fp(q), hp(k), hp(v), fp(sentinel), heads, kv_heads,
                tokens, 0, capacity, dim, dim, capacity, 3,
                fp(workspace), len(workspace) * 4, bad_plan, 3,
            ), 0)
            self.assertEqual(sentinel[0], 123.0)
            self.assertEqual(sentinel[-1], 123.0)

    def test_f16cache_regular_noscale_matches_materialized_softmax(self):
        heads, kv_heads, tokens, capacity, dim, stride = 4, 2, 79, 83, 40, 48

        def half(value):
            return struct.unpack("<e", struct.pack("<e", value))[0]

        def half_bits(value):
            return int.from_bytes(struct.pack("<e", value), "little")

        def half_from_bits(bits):
            return struct.unpack("<e", struct.pack("<H", bits))[0]

        q = array("f", (
            math.sin(i * 0.17) * 1.3 + math.cos(i * 0.037) * 0.4
            for i in range(heads * stride)
        ))
        k = array("H", (
            half_bits(math.cos(i * 0.031) * 0.5 + math.sin(i * 0.071) * 0.2)
            for i in range(kv_heads * capacity * stride)
        ))
        v = array("H", (
            half_bits(math.sin(i * 0.021) * 1.8)
            for i in range(kv_heads * capacity * stride)
        ))
        guarded = array("f", [99.0] * (heads * stride + 2))
        output = (ctypes.c_float * (heads * stride)).from_buffer(
            guarded, ctypes.sizeof(ctypes.c_float)
        )
        args = (
            (ctypes.c_float * len(q)).from_buffer(q),
            (ctypes.c_uint16 * len(k)).from_buffer(k),
            (ctypes.c_uint16 * len(v)).from_buffer(v),
            output, heads, kv_heads, tokens, capacity, dim, stride, 0,
        )
        fn = self._lib.attention_forward_decode_head_major_gqa_f16cache_regular_noscale
        self.assertEqual(fn(*args), 0)
        self.assertEqual(guarded[0], 99.0)
        self.assertEqual(guarded[-1], 99.0)

        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            query = [half(q[head * stride + d]) for d in range(dim)]
            scores = []
            for token in range(tokens):
                base = (kv_head * capacity + token) * stride
                scores.append(sum(query[d] * half_from_bits(k[base + d])
                                  for d in range(dim)))
            peak = max(scores)
            exps = [math.exp(score - peak) for score in scores]
            total = sum(exps)
            for d in range(dim):
                expected = sum(
                    half(exps[token] / total) *
                    half_from_bits(v[(kv_head * capacity + token) * stride + d])
                    for token in range(tokens)
                )
                self.assertAlmostEqual(guarded[1 + head * stride + d], expected, delta=2e-4)
            self.assertEqual(list(guarded[1 + head * stride + dim:1 + (head + 1) * stride]),
                             [0.0] * (stride - dim))

        invalid = list(args)
        invalid[6] = capacity + 1
        before = guarded.tobytes()
        self.assertNotEqual(fn(*invalid), 0)
        self.assertEqual(guarded.tobytes(), before)
        invalid = list(args)
        invalid[8] = 4097
        self.assertNotEqual(fn(*invalid), 0)
        self.assertEqual(guarded.tobytes(), before)
        window = 11
        sliding = array("f", [0.0] * (heads * stride))
        sliding_args = list(args)
        sliding_args[3] = (ctypes.c_float * len(sliding)).from_buffer(sliding)
        sliding_args[-1] = window
        self.assertEqual(fn(*sliding_args), 0)
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            query = [half(q[head * stride + d]) for d in range(dim)]
            scores = []
            for token in range(tokens - window, tokens):
                base = (kv_head * capacity + token) * stride
                scores.append(sum(query[d] * half_from_bits(k[base + d])
                                  for d in range(dim)))
            peak = max(scores)
            exps = [math.exp(score - peak) for score in scores]
            total = sum(exps)
            for d in range(dim):
                expected = sum(
                    half(exps[index] / total) *
                    half_from_bits(v[(kv_head * capacity + tokens - window + index) * stride + d])
                    for index in range(window)
                )
                self.assertAlmostEqual(sliding[head * stride + d], expected, delta=2e-4)

    def test_gemma4_tiled_fp16_q_probability_contract(self):
        tokens, dim = 64, 8
        q = np.zeros((1, tokens, dim), dtype=np.float32)
        q[0, :, :] = np.float32(0.3841)
        q[0, -1, :] = np.array(
            [1.0003, -0.7003, 0.5003, -0.3003, 0.2503, -0.1253, 0.7503, -0.6253],
            dtype=np.float32,
        )
        k = np.array(
            [[math.sin((t + 1) * (d + 2) * 0.13) * 0.7 for d in range(dim)]
             for t in range(tokens)], dtype=np.float16,
        )
        v = np.array(
            [[math.cos((t + 3) * (d + 1) * 0.17) * 3.0 for d in range(dim)]
             for t in range(tokens)], dtype=np.float16,
        )
        workspace = np.zeros(2 * dim, dtype=np.float32)

        def execute(reduction):
            output = np.zeros_like(q)
            status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gemma4_workspace(
                q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                k.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                v.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                1, 1, tokens, 0, tokens, dim, dim, reduction,
                workspace.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), workspace.nbytes,
            )
            self.assertEqual(status, 0)
            return output[0, -1]

        candidate = execute(8)
        baseline = execute(6)
        q_half = q[0, -1].astype(np.float16).astype(np.float32)
        scores = k.astype(np.float32) @ q_half
        weights = np.exp(scores - scores.max())
        probabilities = weights / weights.sum()
        expected = probabilities.astype(np.float16).astype(np.float32) @ v.astype(np.float32)
        np.testing.assert_allclose(candidate, expected, rtol=2e-4, atol=2e-4)
        self.assertGreater(float(np.max(np.abs(candidate - baseline))), 1e-5)
        q_only = probabilities @ v.astype(np.float32)
        raw_scores = k.astype(np.float32) @ q[0, -1]
        raw_weights = np.exp(raw_scores - raw_scores.max())
        prob_only = (
            (raw_weights / raw_weights.sum()).astype(np.float16).astype(np.float32)
            @ v.astype(np.float32)
        )
        self.assertGreater(float(np.max(np.abs(expected - q_only))), 1e-5)
        self.assertGreater(float(np.max(np.abs(expected - prob_only))), 1e-5)

    def test_gemma4_fp16_q_probability_segmented_tails(self):
        heads, tokens, dim = 2, 70, 8
        q = np.array(
            [math.sin(i * 0.17) * 0.8 + 0.0003 for i in range(heads * tokens * dim)],
            dtype=np.float32,
        ).reshape(heads, tokens, dim)
        k = np.array(
            [math.cos(i * 0.11) * 0.7 for i in range(tokens * dim)],
            dtype=np.float16,
        )
        v = np.array(
            [math.sin(i * 0.19) * 2.0 for i in range(tokens * dim)],
            dtype=np.float16,
        )
        plan = (ctypes.c_int * 3)(3, 64, 3)
        workspace = np.zeros(2 * heads * dim, dtype=np.float32)
        fp = ctypes.POINTER(ctypes.c_float)
        hp = ctypes.POINTER(ctypes.c_uint16)

        def execute(reduction, workspace_bytes=None):
            output = np.full_like(q, 123.0)
            status = self._lib.attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_sliding_gemma4_workspace(
                q.ctypes.data_as(fp), k.view(np.uint16).ctypes.data_as(hp),
                v.view(np.uint16).ctypes.data_as(hp), output.ctypes.data_as(fp),
                heads, 1, tokens, 0, tokens, dim, dim, 0, reduction,
                workspace.ctypes.data_as(fp),
                workspace.nbytes if workspace_bytes is None else workspace_bytes,
                plan, len(plan),
            )
            return status, output

        status, baseline = execute(6)
        self.assertEqual(status, 0)
        status, candidate = execute(8)
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(candidate[:, :3], baseline[:, :3])
        np.testing.assert_array_equal(candidate[:, 67:], baseline[:, 67:])
        self.assertGreater(float(np.max(np.abs(candidate[:, 3:67] - baseline[:, 3:67]))), 1e-5)
        status, untouched = execute(8, workspace.nbytes - 1)
        self.assertNotEqual(status, 0)
        np.testing.assert_array_equal(untouched, np.full_like(q, 123.0))

    def test_gemma4_f16cache_matches_independent_rounding_oracle(self):
        heads, kv_heads = 6, 2
        past_tokens, q_tokens, capacity = 3, 5, 11
        head_dim, aligned_dim, sliding_window = 40, 48, 4
        reduction = 2  # CK_ATTN_REDUCTION_F16_ONLINE_SINGLE_RANGE

        def f32(value):
            return ctypes.c_float(value).value

        def f16(value):
            return struct.unpack("<e", struct.pack("<e", value))[0]

        libm = ctypes.CDLL("libm.so.6")
        libm.expf.argtypes = [ctypes.c_float]
        libm.expf.restype = ctypes.c_float
        libm.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        libm.fmaf.restype = ctypes.c_float

        q = array("f", (
            f32(math.sin(i * 0.19) * 0.7 + math.cos(i * 0.07) * 0.2)
            for i in range(heads * q_tokens * aligned_dim)
        ))

        def fp16_bits(value):
            return int.from_bytes(struct.pack("<e", value), "little")

        k = array("H", (
            fp16_bits(math.cos(i * 0.11) * 0.6 + math.sin(i * 0.03) * 0.1)
            for i in range(kv_heads * capacity * aligned_dim)
        ))
        v = array("H", (
            fp16_bits(math.sin(i * 0.13 + 0.4) * 0.8)
            for i in range(kv_heads * capacity * aligned_dim)
        ))

        def float_pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        def half_pointer(values):
            return (ctypes.c_uint16 * len(values)).from_buffer(values)

        def half_value(values, index):
            return struct.unpack("<e", struct.pack("<H", values[index]))[0]

        def avx2_f16_dot(q_half, cache_values, row):
            accumulators = [[f32(0.0)] * 8 for _ in range(4)]
            vector_end = head_dim & ~31
            for block in range(0, vector_end, 32):
                for group in range(4):
                    for lane in range(8):
                        d = block + group * 8 + lane
                        accumulators[group][lane] = libm.fmaf(
                            q_half[d], half_value(cache_values, row + d),
                            accumulators[group][lane],
                        )
            merged = [
                f32(f32(accumulators[0][lane] + accumulators[2][lane]) +
                    f32(accumulators[1][lane] + accumulators[3][lane]))
                for lane in range(8)
            ]
            pairs = [f32(merged[lane] + merged[lane + 4]) for lane in range(4)]
            vector_result = f32(
                f32(pairs[0] + pairs[1]) + f32(pairs[2] + pairs[3])
            )
            result = float(vector_result)
            for d in range(vector_end, head_dim):
                product = f32(q_half[d] * half_value(cache_values, row + d))
                result += float(product)
            return f32(result)

        def oracle_token(q_values, token, kv_tokens, window):
            result = array("f", [0.0]) * (heads * aligned_dim)
            kv_start = max(0, kv_tokens - window) if window else 0
            for head in range(heads):
                kv_head = head * kv_heads // heads
                q_base = (head * len(q_values) // heads) + token * aligned_dim
                q_half = [f16(q_values[q_base + d]) for d in range(aligned_dim)]
                accumulator = [f16(0.0)] * aligned_dim
                maximum = -math.inf
                total = f32(0.0)
                cache_base = kv_head * capacity * aligned_dim
                for position in range(kv_start, kv_tokens):
                    row = cache_base + position * aligned_dim
                    dot = avx2_f16_dot(q_half, k, row)
                    old_maximum = maximum
                    max_scale = f32(1.0)
                    value_scale = f32(1.0)
                    if dot > maximum:
                        maximum = dot
                        max_scale = (
                            libm.expf(f32(old_maximum - maximum))
                            if math.isfinite(old_maximum) else f32(0.0)
                        )
                        for d in range(head_dim):
                            accumulator[d] = f16(f32(accumulator[d] * max_scale))
                    else:
                        value_scale = libm.expf(f32(dot - maximum))
                    for d in range(head_dim):
                        product = f32(half_value(v, row + d) * value_scale)
                        accumulator[d] = f16(f32(accumulator[d] + product))
                    total = libm.fmaf(total, max_scale, value_scale)
                inverse = f32(1.0 / total)
                for d in range(head_dim):
                    result[head * aligned_dim + d] = f32(accumulator[d] * inverse)
            return result

        def prefill(q_values, count, past, window):
            output = array("f", [0.0]) * (heads * count * aligned_dim)
            workspace = array("f", [0.0]) * (2 * heads * aligned_dim)
            common = (
                float_pointer(q_values), half_pointer(k), half_pointer(v),
                float_pointer(output), heads, kv_heads, count, past, capacity,
                head_dim, aligned_dim,
            )
            if window:
                status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_sliding_gemma4_workspace(
                    *common, window, reduction, float_pointer(workspace),
                    len(workspace) * ctypes.sizeof(ctypes.c_float),
                )
            else:
                status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gemma4_workspace(
                    *common, reduction, float_pointer(workspace),
                    len(workspace) * ctypes.sizeof(ctypes.c_float),
                )
            self.assertEqual(status, 0)
            return output

        def expected_prefill(window):
            expected = array("f", [0.0]) * (heads * q_tokens * aligned_dim)
            for token in range(q_tokens):
                token_output = oracle_token(q, token, past_tokens + token + 1, window)
                for head in range(heads):
                    src = head * aligned_dim
                    dst = (head * q_tokens + token) * aligned_dim
                    expected[dst:dst + aligned_dim] = token_output[src:src + aligned_dim]
            return expected

        for window in (0, sliding_window):
            actual = prefill(q, q_tokens, past_tokens, window)
            expected = expected_prefill(window)
            self.assertEqual(actual.tobytes(), expected.tobytes())

            split = 2
            q_first = array("f")
            q_second = array("f")
            for head in range(heads):
                base = head * q_tokens * aligned_dim
                q_first.extend(q[base:base + split * aligned_dim])
                q_second.extend(q[base + split * aligned_dim:base + q_tokens * aligned_dim])
            first = prefill(q_first, split, past_tokens, window)
            second = prefill(q_second, q_tokens - split, past_tokens + split, window)
            segmented = array("f", [0.0]) * len(actual)
            for head in range(heads):
                dst = head * q_tokens * aligned_dim
                first_src = head * split * aligned_dim
                second_src = head * (q_tokens - split) * aligned_dim
                segmented[dst:dst + split * aligned_dim] = first[
                    first_src:first_src + split * aligned_dim
                ]
                segmented[dst + split * aligned_dim:dst + q_tokens * aligned_dim] = second[
                    second_src:second_src + (q_tokens - split) * aligned_dim
                ]
            self.assertEqual(actual.tobytes(), segmented.tobytes())

            final_q = array("f")
            for head in range(heads):
                src = (head * q_tokens + q_tokens - 1) * aligned_dim
                final_q.extend(q[src:src + aligned_dim])
            decode = array("f", [0.0]) * (heads * aligned_dim)
            common = (
                float_pointer(final_q), half_pointer(k), half_pointer(v),
                float_pointer(decode), heads, kv_heads, past_tokens + q_tokens,
                capacity, head_dim, aligned_dim,
            )
            if window:
                status = self._lib.attention_forward_decode_head_major_gqa_f16cache_sliding_gemma4_contract(
                    *common, window, reduction,
                )
            else:
                status = self._lib.attention_forward_decode_head_major_gqa_f16cache_gemma4_contract(
                    *common, reduction,
                )
            self.assertEqual(status, 0)
            expected_final = array("f")
            for head in range(heads):
                src = (head * q_tokens + q_tokens - 1) * aligned_dim
                expected_final.extend(actual[src:src + aligned_dim])
            self.assertEqual(decode.tobytes(), expected_final.tobytes())

    def test_gemma4_prefill_rejects_overflowing_extent_before_workspace_use(self):
        q = array("f", [1.0])
        cache = array("H", [0])
        output = array("f", [123.0])
        workspace = array("f", [456.0, 789.0])

        def float_pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        def half_pointer(values):
            return (ctypes.c_uint16 * len(values)).from_buffer(values)

        status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gemma4_workspace(
            float_pointer(q), half_pointer(cache), half_pointer(cache),
            float_pointer(output), 1, 1, 2, 2_147_483_646, 2_147_483_647,
            1, 1, 2, float_pointer(workspace),
            len(workspace) * ctypes.sizeof(ctypes.c_float),
        )
        self.assertEqual(status, -1)
        self.assertEqual(output.tolist(), [123.0])
        self.assertEqual(workspace.tolist(), [456.0, 789.0])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_token_major_output_is_bit_exact_with_transposed_head_major_output(self):
        heads, kv_heads, tokens, dim = 4, 2, 11, 16
        count = heads * tokens * dim
        kv_count = kv_heads * tokens * dim
        q = array("f", (math.sin(index * 0.013) for index in range(count)))
        k = array("f", (math.cos(index * 0.017) for index in range(kv_count)))
        v = array("f", (math.sin(index * 0.019 + 0.3) for index in range(kv_count)))
        head_output = array("f", [0.0]) * count
        token_output = array("f", [0.0]) * count

        def pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        args = (pointer(q), pointer(k), pointer(v))
        self._lib.attention_forward_causal_head_major_gqa_flash_strided(
            *args, pointer(head_output), heads, kv_heads, tokens, dim, dim, tokens
        )
        self._lib.attention_forward_causal_head_major_gqa_flash_strided_token_output(
            *args, pointer(token_output), heads, kv_heads, tokens, dim, dim, tokens
        )

        expected = array("f", [0.0]) * count
        for token in range(tokens):
            for head in range(heads):
                src = (head * tokens + token) * dim
                dst = (token * heads + head) * dim
                expected[dst:dst + dim] = head_output[src:src + dim]
        self.assertEqual(expected.tobytes(), token_output.tobytes())

    def test_qwen2_short_prefill_threading_is_bit_exact_with_serial(self):
        heads, kv_heads, tokens, dim = 14, 2, 128, 64
        count = heads * tokens * dim
        kv_count = kv_heads * tokens * dim
        q = array("f", (math.sin(index * 0.0013) for index in range(count)))
        k = array("f", (math.cos(index * 0.0017) for index in range(kv_count)))
        v = array("f", (math.sin(index * 0.0019 + 0.3) for index in range(kv_count)))
        serial = array("f", [0.0]) * count
        threaded = array("f", [0.0]) * count

        def pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        args = (pointer(q), pointer(k), pointer(v))
        old_disable = os.environ.get("CK_DISABLE_ATTENTION_THREADPOOL")
        old_threads = os.environ.get("CK_NUM_THREADS")
        try:
            os.environ["CK_DISABLE_ATTENTION_THREADPOOL"] = "1"
            self._lib.attention_forward_causal_head_major_gqa_flash_strided(
                *args, pointer(serial), heads, kv_heads, tokens, dim, dim, tokens
            )
            os.environ.pop("CK_DISABLE_ATTENTION_THREADPOOL", None)
            os.environ["CK_NUM_THREADS"] = "4"
            self._lib.attention_forward_causal_head_major_gqa_flash_strided(
                *args, pointer(threaded), heads, kv_heads, tokens, dim, dim, tokens
            )
        finally:
            if old_disable is None:
                os.environ.pop("CK_DISABLE_ATTENTION_THREADPOOL", None)
            else:
                os.environ["CK_DISABLE_ATTENTION_THREADPOOL"] = old_disable
            if old_threads is None:
                os.environ.pop("CK_NUM_THREADS", None)
            else:
                os.environ["CK_NUM_THREADS"] = old_threads

        self.assertEqual(serial.tobytes(), threaded.tobytes())

    def test_gemma4_direct_token_outputs_are_bit_exact(self):
        heads, kv_heads, tokens, dim = 4, 2, 13, 16
        count = heads * tokens * dim
        kv_count = kv_heads * tokens * dim
        q = array("f", (math.sin(index * 0.011) for index in range(count)))
        k = array("f", (math.cos(index * 0.017) for index in range(kv_count)))
        v = array("f", (math.sin(index * 0.023 + 0.2) for index in range(kv_count)))

        def pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        def expected_token_output(head_output):
            token_output = array("f", [0.0]) * count
            for token in range(tokens):
                for head in range(heads):
                    src = (head * tokens + token) * dim
                    dst = (token * heads + head) * dim
                    token_output[dst:dst + dim] = head_output[src:src + dim]
            return token_output

        common = (pointer(q), pointer(k), pointer(v))
        old_threads = os.environ.get("CK_NUM_THREADS")
        os.environ["CK_NUM_THREADS"] = "4"
        try:
            for head_fn, token_fn, tail in (
                (
                    self._lib.attention_forward_causal_head_major_gqa_flash_strided_gemma4,
                    self._lib.attention_forward_causal_head_major_gqa_flash_strided_gemma4_token_output,
                    (),
                ),
                (
                    self._lib.attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4,
                    self._lib.attention_forward_causal_head_major_gqa_flash_strided_sliding_gemma4_token_output,
                    (5,),
                ),
            ):
                head_output = array("f", [0.0]) * count
                token_output = array("f", [0.0]) * count
                geometry = (heads, kv_heads, tokens, dim, dim, tokens, *tail)
                head_fn(*common, pointer(head_output), *geometry)
                token_fn(*common, pointer(token_output), *geometry)
                self.assertEqual(expected_token_output(head_output).tobytes(), token_output.tobytes())
        finally:
            if old_threads is None:
                os.environ.pop("CK_NUM_THREADS", None)
            else:
                os.environ["CK_NUM_THREADS"] = old_threads

    def test_mixed_visual_attention_preserves_direct_output_layout(self):
        heads, kv_heads, tokens, dim = 4, 2, 9, 64
        count = heads * tokens * dim
        kv_count = kv_heads * tokens * dim
        q = array("f", (math.sin(index * 0.011) for index in range(count)))
        k = array("f", (math.cos(index * 0.023) for index in range(kv_count)))
        v = array("f", (math.sin(index * 0.029 + 0.2) for index in range(kv_count)))
        head_output = array("f", [0.0]) * count
        token_output = array("f", [0.0]) * count

        def pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        common = (
            pointer(q), pointer(k), pointer(v), heads, kv_heads, tokens,
            dim, dim, tokens, 2, 4,
        )
        self._lib.attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4(
            common[0], common[1], common[2], pointer(head_output), *common[3:]
        )
        self._lib.attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4_token_output(
            common[0], common[1], common[2], pointer(token_output), *common[3:]
        )

        expected = array("f", [0.0]) * count
        for token in range(tokens):
            for head in range(heads):
                src = (head * tokens + token) * dim
                dst = (token * heads + head) * dim
                expected[dst:dst + dim] = head_output[src:src + dim]
        self.assertEqual(expected.tobytes(), token_output.tobytes())

    def test_gqa_reuse_workspace_is_exact_and_fails_closed(self):
        heads, kv_heads, query_tokens, past_tokens, dim = 4, 2, 128, 64, 16
        capacity = query_tokens + past_tokens
        q = array("f", (
            math.sin(index * 0.007 + 0.1)
            for index in range(heads * query_tokens * dim)
        ))

        def fp16_bits(value):
            import struct
            return int.from_bytes(struct.pack("<e", value), "little")

        k = array("H", (
            fp16_bits(math.cos(index * 0.011))
            for index in range(kv_heads * capacity * dim)
        ))
        v = array("H", (
            fp16_bits(math.sin(index * 0.013 + 0.2))
            for index in range(kv_heads * capacity * dim)
        ))
        baseline = array("f", [0.0]) * (heads * query_tokens * dim)
        reused = array("f", [7.0]) * (heads * query_tokens * dim)
        fallback = array("f", [0.0]) * (heads * query_tokens * dim)
        token_workspace = array("f", [0.0]) * (2 * heads * dim)

        def float_pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        def half_pointer(values):
            return (ctypes.c_uint16 * len(values)).from_buffer(values)

        old_threads = os.environ.get("CK_NUM_THREADS")
        os.environ["CK_NUM_THREADS"] = "4"
        try:
            status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_qtile64_schedule(
                float_pointer(q), half_pointer(k), half_pointer(v),
                float_pointer(baseline), heads, kv_heads, query_tokens,
                past_tokens, capacity, dim, dim, 2,
            )
            self.assertEqual(status, 0)

            required = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gqa_reuse_workspace_bytes(
                heads, kv_heads, dim, 4, 64, 2,
            )
            self.assertGreater(required, 0)
            raw_workspace = bytearray(required + 63)
            raw_view = (ctypes.c_ubyte * len(raw_workspace)).from_buffer(raw_workspace)
            workspace = ctypes.c_void_p((ctypes.addressof(raw_view) + 63) & ~63)

            status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gqa_reuse_config(
                float_pointer(q), half_pointer(k), half_pointer(v),
                float_pointer(reused), heads, kv_heads, query_tokens,
                past_tokens, capacity, dim, dim, 64, 2,
                workspace, required - 1,
            )
            self.assertEqual(status, -3)
            self.assertEqual(reused.tobytes(), (array("f", [7.0]) * len(reused)).tobytes())

            status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gqa_reuse_config(
                float_pointer(q), half_pointer(k), half_pointer(v),
                float_pointer(reused), heads, kv_heads, query_tokens,
                past_tokens, capacity, dim, dim, 64, 2,
                workspace, required,
            )
            self.assertEqual(status, 0)
            self.assertEqual(reused.tobytes(), baseline.tobytes())

            status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace(
                float_pointer(q), half_pointer(k), half_pointer(v),
                float_pointer(fallback), heads, kv_heads, query_tokens,
                past_tokens, capacity, dim, dim, 3,
                float_pointer(token_workspace),
                len(token_workspace) * ctypes.sizeof(ctypes.c_float),
                workspace, required,
                24, 4, 256, 4096, 8192, 16, 128, 4,
            )
            self.assertEqual(status, 0)
            self.assertEqual(fallback.tobytes(), baseline.tobytes())
        finally:
            if old_threads is None:
                os.environ.pop("CK_NUM_THREADS", None)
            else:
                os.environ["CK_NUM_THREADS"] = old_threads

    def test_segmented_prefill_matches_independent_segment_calls_bit_exactly(self):
        heads, kv_heads, tokens, dim = 4, 2, 75, 16
        segments = [9, 64, 2]
        q = array("f", (
            math.sin(index * 0.007 + 0.1)
            for index in range(heads * tokens * dim)
        ))

        def fp16_bits(value):
            import struct
            return int.from_bytes(struct.pack("<e", value), "little")

        k = array("H", (
            fp16_bits(math.cos(index * 0.011))
            for index in range(kv_heads * tokens * dim)
        ))
        v = array("H", (
            fp16_bits(math.sin(index * 0.013 + 0.2))
            for index in range(kv_heads * tokens * dim)
        ))
        expected = array("f", [0.0]) * (heads * tokens * dim)
        actual = array("f", [0.0]) * (heads * tokens * dim)
        workspace = array("f", [0.0]) * (2 * heads * dim)

        def float_pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        def half_pointer(values):
            return (ctypes.c_uint16 * len(values)).from_buffer(values)

        row_offset = 0
        for rows in segments:
            q_segment = array("f", [0.0]) * (heads * rows * dim)
            for head in range(heads):
                source = (head * tokens + row_offset) * dim
                destination = head * rows * dim
                q_segment[destination:destination + rows * dim] = q[
                    source:source + rows * dim
                ]
            out_segment = array("f", [0.0]) * (heads * rows * dim)
            status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract_workspace(
                float_pointer(q_segment), half_pointer(k), half_pointer(v),
                float_pointer(out_segment), heads, kv_heads, rows, row_offset,
                tokens, dim, dim, 3, float_pointer(workspace),
                len(workspace) * ctypes.sizeof(ctypes.c_float),
            )
            self.assertEqual(status, 0)
            for head in range(heads):
                source = head * rows * dim
                destination = (head * tokens + row_offset) * dim
                expected[destination:destination + rows * dim] = out_segment[
                    source:source + rows * dim
                ]
            row_offset += rows

        segment_array = (ctypes.c_int * len(segments))(*segments)
        status = self._lib.attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_contract_workspace(
            float_pointer(q), half_pointer(k), half_pointer(v),
            float_pointer(actual), heads, kv_heads, tokens, 0, tokens, dim, dim,
            3, float_pointer(workspace),
            len(workspace) * ctypes.sizeof(ctypes.c_float),
            segment_array, len(segments),
        )
        self.assertEqual(status, 0)
        self.assertEqual(expected.tobytes(), actual.tobytes())

    def test_qtile64_parallel_schedules_are_bit_exact(self):
        heads, kv_heads, query_tokens, past_tokens, dim = 24, 4, 130, 17, 16
        capacity = query_tokens + past_tokens
        q = array("f", (
            math.sin(index * 0.007 + 0.1)
            for index in range(heads * query_tokens * dim)
        ))

        def fp16_bits(value):
            import struct
            return int.from_bytes(struct.pack("<e", value), "little")

        k = array("H", (
            fp16_bits(math.cos(index * 0.011))
            for index in range(kv_heads * capacity * dim)
        ))
        v = array("H", (
            fp16_bits(math.sin(index * 0.013 + 0.2))
            for index in range(kv_heads * capacity * dim)
        ))

        def float_pointer(values):
            return (ctypes.c_float * len(values)).from_buffer(values)

        def half_pointer(values):
            return (ctypes.c_uint16 * len(values)).from_buffer(values)

        outputs = []
        for schedule in range(4):
            output = array("f", [0.0]) * (heads * query_tokens * dim)
            status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_qtile64_schedule(
                float_pointer(q), half_pointer(k), half_pointer(v),
                float_pointer(output), heads, kv_heads, query_tokens,
                past_tokens, capacity, dim, dim, schedule,
            )
            self.assertEqual(status, 0)
            outputs.append(output.tobytes())

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])
        self.assertEqual(outputs[0], outputs[3])

    def test_resolved_segmented_replay_requires_fidelity_and_contract_identity(self):
        heads, kv_heads, tokens, dim, capacity = 2, 1, 70, 8, 72
        segments = [5, 65]
        q = np.sin(np.arange(heads * tokens * dim, dtype=np.float32) * 0.013).reshape(heads, tokens, dim)
        k = np.cos(np.arange(kv_heads * tokens * dim, dtype=np.float32) * 0.011).reshape(kv_heads, tokens, dim)
        v = np.sin(np.arange(kv_heads * tokens * dim, dtype=np.float32) * 0.017 + 0.2).reshape(kv_heads, tokens, dim)
        k_cache = np.zeros((kv_heads, capacity, dim), dtype=np.float16)
        v_cache = np.zeros_like(k_cache)
        k_cache[:, :tokens, :] = k
        v_cache[:, :tokens, :] = v
        output = np.zeros_like(q)
        workspace = np.zeros(256, dtype=np.float32)
        plan = (ctypes.c_int * len(segments))(*segments)
        fp = ctypes.POINTER(ctypes.c_float)
        hp = ctypes.POINTER(ctypes.c_uint16)
        status = self._lib.attention_forward_causal_head_major_gqa_prefill_segmented_f16cache_sliding_gemma4_workspace(
            q.ctypes.data_as(fp), k_cache.ctypes.data_as(hp), v_cache.ctypes.data_as(hp),
            output.ctypes.data_as(fp), heads, kv_heads, tokens, 0, capacity, dim, dim,
            capacity, 6, workspace.ctypes.data_as(fp), workspace.nbytes, plan, len(segments),
        )
        self.assertEqual(status, 0)

        names = ("q", "k_cache", "v_cache", "output", "num_heads", "num_kv_heads",
                 "q_tokens", "past_tokens", "cache_capacity", "head_dim", "aligned_head_dim",
                 "sliding_window", "reduction", "token_workspace", "token_workspace_bytes")
        expressions = ("q", "k", "v", "out", str(heads), str(kv_heads), str(capacity),
                       "model->pos", str(capacity), str(dim), str(dim), str(capacity),
                       "CK_ATTN_REDUCTION_F16_FLASH_AUTO_QTILE64_REGULAR_TAIL_NOSCALE",
                       "workspace", str(workspace.nbytes))
        kernel_id = "attention_forward_causal_head_major_gqa_prefill_append_f16cache_sliding_gemma4"
        function = kernel_id + "_workspace"
        call_ir = {"operations": [{
            "layer": 0, "section": "body", "op": "attn_sliding", "function": function,
            "args": [{"name": name, "expr": expr} for name, expr in zip(names, expressions)],
            "call_abi": {"kernel_id": kernel_id},
            "required_contract": {"tensor.kv.dtype": "fp16", "tensor.layout": "head_major"},
            "resolved_contract": {
                "kernel_id": kernel_id, "function": function,
                "selector": expressions[12],
                "implementation": {"segmented_query_provider": {
                    "function": replay._SEGMENTED_F16_SLIDING, "base_args": list(names),
                }},
            },
        }]}
        resolved = replay._resolved_segmented_call(call_ir, 0)
        parity_report = {"dump_compare": {
            "ck_dump_path": str(self._lib_path),
            "prefill_segments": [
                {"name": "text_before", "physical_start": 0, "rows": 5},
                {"name": "visual", "physical_start": 5, "rows": 65},
            ],
        }}
        self.assertEqual(replay._segments_from_parity_report(parity_report, self._lib_path), segments)
        parity_report["dump_compare"]["prefill_segments"][1]["physical_start"] = 6
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            replay._segments_from_parity_report(parity_report, self._lib_path)
        parity_report["dump_compare"]["prefill_segments"][1]["physical_start"] = 5
        parity_report["dump_compare"]["ck_dump_path"] = str(self._lib_path.parent / "stale-dump.bin")
        with self.assertRaisesRegex(ValueError, "does not identify"):
            replay._segments_from_parity_report(parity_report, self._lib_path)

        call_ir_path = self._lib_path.parent / "call_prefill.json"
        manifest_path = self._lib_path.with_name(self._lib_path.name + ".build.json")
        parity_report["decoder_runtime"] = {"so_path": str(self._lib_path.parent / "libdecoder.so")}
        manifest_path.write_text(json.dumps({"sha256": replay._sha256(self._lib_path)}), encoding="utf-8")
        replay._validate_replay_artifacts(parity_report, call_ir_path, self._lib_path)
        manifest_path.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            replay._validate_replay_artifacts(parity_report, call_ir_path, self._lib_path)
        parity_report["decoder_runtime"]["so_path"] = str(self._lib_path.parent.parent / "stale/libdecoder.so")
        with self.assertRaisesRegex(ValueError, "not from"):
            replay._validate_replay_artifacts(parity_report, call_ir_path, self._lib_path)

        def dump(name, data, dtype="fp32"):
            return SimpleNamespace(layer_id=0, op_name=name, data=data, dtype=dtype, token_id=0)

        def token_major(values):
            return values.transpose(1, 0, 2).copy().reshape(-1)

        dumps = [
            dump("Qcur_rope", token_major(q)),
            dump("Kcur_rope", token_major(k)),
            dump("Vcur_normed", token_major(v)),
            dump("kqv_out", token_major(output)),
            dump("Kcache_post_store", k_cache[:, :tokens, :].transpose(1, 0, 2).astype(np.float32), "fp16"),
            dump("Vcache_post_store", v_cache[:, :tokens, :].transpose(1, 0, 2).astype(np.float32), "fp16"),
        ]
        result = replay.replay_resolved_segmented_prefill(
            dumps, 0, resolved, segments, self._lib_path, independent_fp64=True,
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["cache_capture_is_direct"])
        self.assertTrue(result["cache_reconstruction"]["k_bitwise_equal"])
        self.assertTrue(result["cache_reconstruction"]["v_bitwise_equal"])
        self.assertEqual(result["metrics"]["max_abs"], 0.0)
        self.assertEqual(result["attribution"]["identical_input_oracle"], "not_tested")
        self.assertIsNone(result["attribution"]["first_proven_local_divergence"])
        self.assertEqual(result["independent_fp64"]["vs_captured_cke"],
                         result["independent_fp64"]["vs_replayed_cke"])
        self.assertLess(result["independent_fp64"]["vs_captured_cke"]["rmse"], 0.01)
        bundle_path = replay._export_resolved_bundle(
            self._lib_path.parent / "operation-bundle", dumps, 0, resolved,
            call_ir["operations"][0], segments, self._lib_path,
            {"dump": "0" * 64, "call_ir": "1" * 64, "parity_report": "2" * 64},
            "disabled",
        )
        bundle_dumps, bundle_layer, bundle_call, bundle_segments, bundle_engine = \
            replay._load_resolved_bundle(bundle_path)
        bundle_result = replay.replay_resolved_segmented_prefill(
            bundle_dumps, bundle_layer, bundle_call, bundle_segments, bundle_engine,
        )
        self.assertEqual(bundle_result["status"], "pass")
        self.assertEqual(bundle_result["metrics"]["max_abs"], 0.0)
        cache_path = bundle_path.parent / "Kcache_post_store.bin"
        changed_bytes = bytearray(cache_path.read_bytes())
        changed_bytes[0] ^= 1
        cache_path.write_bytes(changed_bytes)
        with self.assertRaisesRegex(ValueError, "tensor bytes changed"):
            replay._load_resolved_bundle(bundle_path)
        cache_path.write_bytes(dumps[4].data.astype(np.float16).tobytes())
        manifest = json.loads(bundle_path.read_text())
        manifest["provider"] = "stale_provider"
        bundle_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "provider or numerical contract differs"):
            replay._load_resolved_bundle(bundle_path)
        with self.assertRaisesRegex(ValueError, "expected one"):
            replay.replay_resolved_segmented_prefill(dumps[:4], 0, resolved, segments, self._lib_path)
        self.assertEqual(
            replay.replay_resolved_segmented_prefill(
                dumps[:4], 0, resolved, segments, self._lib_path, require_direct_cache=False,
            )["status"], "pass",
        )
        wrong_dtype = list(dumps)
        wrong_dtype[4] = dump("Kcache_post_store", dumps[4].data, "fp32")
        with self.assertRaisesRegex(ValueError, "must be FP16"):
            replay.replay_resolved_segmented_prefill(wrong_dtype, 0, resolved, segments, self._lib_path)
        changed_cache = list(dumps)
        changed_k = dumps[4].data.copy()
        changed_k[0, 0, 0] += 0.25
        changed_cache[4] = dump("Kcache_post_store", changed_k, "fp16")
        self.assertFalse(
            replay.replay_resolved_segmented_prefill(
                changed_cache, 0, resolved, segments, self._lib_path,
            )["cache_reconstruction"]["k_bitwise_equal"]
        )
        self.assertEqual(
            replay.replay_resolved_segmented_prefill(dumps, 0, resolved, [35, 35], self._lib_path)["status"],
            "fail_replay_fidelity",
        )

        corrupted = list(dumps)
        changed = dumps[3].data.copy()
        changed[0] += 0.25
        corrupted[3] = dump("kqv_out", changed)
        self.assertEqual(
            replay.replay_resolved_segmented_prefill(corrupted, 0, resolved, segments, self._lib_path)["status"],
            "fail_replay_fidelity",
        )
        with self.assertRaisesRegex(ValueError, "expected one"):
            replay.replay_resolved_segmented_prefill(dumps + dumps[0:1], 0, resolved, segments, self._lib_path)
        with self.assertRaisesRegex(ValueError, "segment lengths"):
            replay.replay_resolved_segmented_prefill(dumps, 0, resolved, [5, 0, 65], self._lib_path)
        call_ir["operations"][0]["resolved_contract"]["selector"] = "CK_ATTN_REDUCTION_F16_REGULAR_NOSCALE"
        with self.assertRaisesRegex(ValueError, "contradictory"):
            replay._resolved_segmented_call(call_ir, 0)
        call_ir["operations"][0]["resolved_contract"]["selector"] = expressions[12]
        call_ir["operations"][0]["required_contract"]["tensor.kv.dtype"] = "bf16"
        with self.assertRaisesRegex(ValueError, "storage/layout"):
            replay._resolved_segmented_call(call_ir, 0)


    def test_z_flash_consumer_short_append_matches_fresh_tiled_prefix(self):
        """The shared Flash map must keep the stored-KV tail's reduction."""
        heads, kv_heads, dim, total, prefix = 2, 1, 16, 65, 64
        q = array("f", (math.sin(i * 0.137 + 0.31)
                        for i in range(heads * total * dim)))
        def half(value):
            return int.from_bytes(struct.pack("<e", value), "little")
        k = array("H", (half(math.cos(i * 0.113 + 0.19))
                        for i in range(kv_heads * total * dim)))
        v = array("H", (half(math.sin(i * 0.171 + 0.27))
                        for i in range(kv_heads * total * dim)))
        tail_q = array("f", (value for head in range(heads)
                             for value in q[(head * total + prefix) * dim:
                                            (head * total + prefix + 1) * dim]))
        token_workspace = array("f", [0.0]) * (2 * heads * dim)
        def fp(values):
            return (ctypes.c_float * len(values)).from_buffer(values)
        def hp(values):
            return (ctypes.c_uint16 * len(values)).from_buffer(values)
        def run(query, rows, past, reduction):
            out = array("f", [0.0]) * (heads * rows * dim)
            status = self._lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace(
                fp(query), hp(k), hp(v), fp(out), heads, kv_heads, rows,
                past, total, dim, dim, reduction, fp(token_workspace),
                len(token_workspace) * 4, None, 0,
                24, 4, 256, 4096, 8192, 16, 128, 4,
            )
            self.assertEqual(status, 0)
            return out
        full = run(q, total, 0, 7)
        staged = run(tail_q, 1, prefix, 7)
        expected = array("f", (value for head in range(heads)
                               for value in full[(head * total + prefix) * dim:
                                                 (head * total + prefix + 1) * dim]))
        self.assertEqual(staged.tobytes(), expected.tobytes())
        # The old selector's short-query online path is a real negative
        # control, not just a changed map string.
        self.assertNotEqual(run(tail_q, 1, prefix, 3).tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()
