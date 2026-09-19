import ctypes
import math
import os
import struct
import subprocess
import tempfile
import unittest
from array import array
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
