import ctypes
import math
import os
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
