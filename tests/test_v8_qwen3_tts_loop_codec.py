"""Generated Qwen code loop to codec handoff; synthetic EOS is diagnostic."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_checked_frame_loop import Loop
from tests.test_v8_qwen3_tts_codec_waveform import FINAL_WEIGHT, synthetic_arrays
from tests.test_v8_qwen3_tts_codec_preconv import fill_ids
from tests.test_v8_qwen3_tts_codec_semantic import generated_entry
from tests.test_v8_qwen3_tts_first_frame_waveform import CODEC_SOURCES
from tests.test_v8_qwen3_tts_prompt_first_block import compile_entry, view
from tests.test_v8_qwen3_tts_second_generated_frame import prepare_inputs
from tests.test_v8_qwen3_tts_talker_prefill import weight_shapes
from tests.test_v8_qwen3_tts_two_phase_speech import SOURCES, segment_function
from version.v8.tts.build_qwen3_tts_loop_codec_circuit import (
    AVX2_CODEC_CONV, OUTPUT, SCALAR_CODEC_CONV, build_circuit,
)
from version.v8.tts.build_qwen3_tts_second_generated_frame_circuit import (
    build_circuit as generated_frames,
)

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = dict(
    vocabulary=12, codebook_width=4, output_width=7, acoustic_codebooks=3,
    latent_width=3, hidden_width=4, heads=2, head_dim=2, sliding_window=3,
    intermediate_width=6, layers=2, depthwise_kernel=7, expansion=2,
    decoder_width=16, ingress_kernel=7, decoder_upsample_rate=2,
    second_decoder_upsample_rate=2, third_decoder_upsample_rate=2,
    fourth_decoder_upsample_rate=2,
)


class FrameCodec(ctypes.Structure):
    _fields_ = [
        ("generation", ctypes.POINTER(Loop)), ("codec", ctypes.c_void_p),
        ("codec_frame_major_offset", ctypes.c_size_t),
        ("expected_codebooks", ctypes.c_size_t),
        ("expected_code_frames", ctypes.c_size_t),
        ("waveform_offset", ctypes.c_size_t),
        ("waveform_valid_samples_offset", ctypes.c_size_t),
        ("expected_samples", ctypes.c_size_t),
        ("output_samples", ctypes.POINTER(ctypes.c_float)),
        ("output_capacity", ctypes.c_size_t),
        ("valid_samples", ctypes.POINTER(ctypes.c_size_t)),
    ]


class Qwen3TTSLoopCodecTest(unittest.TestCase):
    def test_declared_handoff_and_geometry(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2) + "\n")
        self.assertEqual(len(graph["native_entry"]["segments"]), 3)
        self.assertEqual(graph["activation_buffers"]["loop_frame_major_codes"]["shape"],
                         [80])
        self.assertEqual(graph["activation_buffers"]["packed_loop_acoustic_ids_14"]["shape"],
                         [5])
        ops = graph["block_types"]["pack_loop_codec_ids"]["body"]["ops"]
        self.assertEqual(len(ops), 80)
        self.assertEqual(ops[-1]["params"]["call_constants"]["source_offset"], 79)
        self.assertFalse(graph["contract"]["runtime_invariants"]["complete_text_to_wav"])
        with self.assertRaises(ValueError):
            build_circuit(frames=1)
        with self.assertRaises(ValueError):
            build_circuit(frames=2, continuation_steps=2,
                          codec_kwargs={**GEOMETRY, "acoustic_codebooks": 2})
        with self.assertRaises(ValueError):
            build_circuit(codec_conv_provider="unknown")
        optimized = build_circuit(frames=2, max_generated_frames=3,
                                  talker_layers=1, continuation_steps=2,
                                  codec_kwargs=GEOMETRY,
                                  codec_conv_provider="avx2_fma")
        selected = [op["kernel"] for block in optimized["sequence"]
                    for op in optimized["block_types"][block].get("body", {}).get("ops", [])
                    if op["op"] == "audio_conv1d_dilated_checked"]
        self.assertTrue(selected)
        self.assertEqual(set(selected), {AVX2_CODEC_CONV})
        self.assertEqual(optimized["contract"]["runtime_invariants"][
            "codec_conv_provider"], AVX2_CODEC_CONV)
        self.assertNotIn(SCALAR_CODEC_CONV, selected)

    def test_generated_loop_to_codec_and_recovery(self):
        arrays = synthetic_arrays(np.random.default_rng(20261029))
        arrays["codec.acoustic.2.effective_embedding"] = (
            arrays["codec.acoustic.0.effective_embedding"].copy())
        graph = build_circuit(frames=2, max_generated_frames=3,
                              talker_layers=1, continuation_steps=2,
                              codec_kwargs=GEOMETRY)
        # This deliberately forces a zero-weight synthetic talker to stop at
        # frame two. It is not evidence of natural EOS or model-level speech.
        selector = graph["block_types"]["second_talker_predictor_bridge"]["body"]["ops"][1]
        selector["params"]["call_constants"]["stop_id"] = 0
        base = generated_frames(talker_layers=1, continuation_steps=2,
                                max_generated_frames=3)
        shapes = weight_shapes(base)
        shapes.update({name: value.shape for name, value in arrays.items()})
        sources = tuple(dict.fromkeys((*SOURCES, *CODEC_SOURCES,
                                       "src/runtime/checked_frame_loop.c",
                                       "src/runtime/checked_frame_codec.c")))
        with tempfile.TemporaryDirectory() as temp:
            _, acts, weights, layout, calls, library = compile_entry(
                Path(temp), graph, shapes, extra_sources=sources)
            self.assertEqual(len(calls["entry"]["segments"]), 3)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, value in arrays.items():
                item = weights[name]
                arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
            prepare_inputs(arena, acts)
            first = ["talker_first_code_id", "pred_predictor_codebook0_selected_id",
                     "pred_step1_selected_id", "pred_step2_selected_id"]
            later = ["second_talker_first_code_id",
                     "second_predictor_codebook0_selected_id",
                     "second_step1_selected_id", "second_step2_selected_id"]
            first_offsets = (ctypes.c_size_t * 4)(
                *(acts[name]["abs_offset"] for name in first))
            later_offsets = (ctypes.c_size_t * 4)(
                *(acts[name]["abs_offset"] for name in later))
            staging = np.full(12, -901, np.int32)
            codes = np.full(12, -901, np.int32)
            frames = ctypes.c_size_t(99)
            loop = Loop(
                ctypes.cast(segment_function(library, graph, 0), ctypes.c_void_p),
                ctypes.cast(segment_function(library, graph, 1), ctypes.c_void_p),
                arena, size, acts["talker_step_index_input"]["abs_offset"],
                acts["pred_predictor_valid_code_count"]["abs_offset"],
                acts["second_predictor_valid_code_count"]["abs_offset"],
                first_offsets, later_offsets, 4, 3,
                staging.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 12,
                codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 12,
                ctypes.pointer(frames))
            samples = np.full(128, -901, np.float32)
            valid = ctypes.c_size_t(99)
            request = FrameCodec(
                ctypes.pointer(loop),
                ctypes.cast(segment_function(library, graph, 2), ctypes.c_void_p),
                acts["loop_frame_major_codes"]["abs_offset"], 4, 2,
                acts["codec_waveform_output"]["abs_offset"],
                acts["waveform_valid_samples"]["abs_offset"], 128,
                samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 128,
                ctypes.pointer(valid))
            run = ctypes.CDLL(str(library)).ck_run_checked_frame_codec
            run.argtypes = [ctypes.POINTER(FrameCodec)]
            run.restype = ctypes.c_int
            self.assertEqual(run(ctypes.byref(request)), 0)
            self.assertEqual((frames.value, valid.value), (2, 128))
            self.assertTrue(np.isfinite(samples).all())
            self.assertTrue(np.any(samples != -901))
            matrix = codes[:8].reshape(2, 4)
            for codebook in range(4):
                name = ("packed_loop_semantic_ids" if codebook == 0 else
                        f"packed_loop_acoustic_ids_{codebook - 1}")
                np.testing.assert_array_equal(
                    view(arena, acts, name, np.int32, (2,)), matrix[:, codebook])
            direct_folder = Path(temp) / "direct_codec"
            direct_folder.mkdir()
            direct_fn, direct_arena, direct_acts, _, _ = generated_entry(
                direct_folder, arrays, 2,
                acoustic_codebooks=3, codec_waveform=True,
                decoder_upsample_rate=2, second_decoder_upsample_rate=2,
                third_decoder_upsample_rate=2, fourth_decoder_upsample_rate=2,
                convnext_expansion=2, transformer_layers=2,
                qkv_heads=2, qkv_head_dim=2, sliding_window=3)
            fill_ids(direct_arena, direct_acts, matrix)
            self.assertEqual(direct_fn(direct_arena, len(direct_arena)), 0)
            np.testing.assert_allclose(
                samples, view(direct_arena, direct_acts, "waveform_output",
                              np.float32, (1, 128)).reshape(-1),
                rtol=0, atol=0)
            baseline = samples.copy()
            samples.fill(-901)
            frames.value = 77
            request.valid_samples = ctypes.pointer(frames)
            self.assertEqual(run(ctypes.byref(request)), -1)
            self.assertEqual(frames.value, 77)
            np.testing.assert_array_equal(samples, -901)
            request.valid_samples = ctypes.pointer(valid)
            valid.value = 55
            loop.valid_frames = ctypes.cast(
                samples.ctypes.data, ctypes.POINTER(ctypes.c_size_t))
            self.assertEqual(run(ctypes.byref(request)), -1)
            self.assertEqual(valid.value, 55)
            np.testing.assert_array_equal(samples, -901)
            loop.valid_frames = ctypes.pointer(frames)
            request.valid_samples = ctypes.cast(
                samples.ctypes.data, ctypes.POINTER(ctypes.c_size_t))
            self.assertEqual(run(ctypes.byref(request)), -1)
            np.testing.assert_array_equal(samples, -901)
            request.valid_samples = ctypes.pointer(valid)
            request.expected_codebooks = 5
            self.assertEqual(run(ctypes.byref(request)), -1)
            np.testing.assert_array_equal(samples, -901)
            request.expected_codebooks = 4
            request.output_capacity = 127
            self.assertEqual(run(ctypes.byref(request)), -1)
            self.assertEqual(valid.value, 0)
            np.testing.assert_array_equal(samples, -901)
            request.output_capacity = 128
            request.expected_code_frames = 3
            valid.value = 99
            self.assertEqual(run(ctypes.byref(request)), -4)
            self.assertEqual(valid.value, 0)
            np.testing.assert_array_equal(samples, -901)
            request.expected_code_frames = 2
            weight = np.ndarray(arrays[FINAL_WEIGHT].shape, np.float32,
                                buffer=arena, offset=weights[FINAL_WEIGHT]["abs_offset"])
            original = weight.copy()
            weight.fill(np.nan)
            valid.value = 99
            self.assertEqual(run(ctypes.byref(request)), -5)
            self.assertEqual(valid.value, 0)
            np.testing.assert_array_equal(samples, -901)
            weight[:] = original
            self.assertEqual(run(ctypes.byref(request)), 0)
            self.assertEqual(valid.value, 128)
            np.testing.assert_array_equal(samples, baseline)
            # Distinct values in every codebook/frame expose swapped packing
            # edges even when synthetic talker logits happen to tie.
            view(arena, acts, "loop_frame_major_codes", np.int32, (8,))[:] = [
                0, 1, 2, 3, 4, 5, 6, 7]
            codec = segment_function(library, graph, 2)
            self.assertEqual(codec(arena, size), 0)
            for codebook in range(4):
                name = ("packed_loop_semantic_ids" if codebook == 0 else
                        f"packed_loop_acoustic_ids_{codebook - 1}")
                np.testing.assert_array_equal(
                    view(arena, acts, name, np.int32, (2,)),
                    [codebook, 4 + codebook])


if __name__ == "__main__":
    unittest.main()
