"""Checked prompt length drives Qwen continuation across native segments."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_prompt_first_block import compile_entry, view
from tests.test_v8_qwen3_tts_talker_prefill import weight_shapes
from version.v8.tts.build_qwen3_tts_prompt_first_block_circuit import prompt_plan
from version.v8.tts.build_qwen3_tts_talker_predictor_frame_circuit import bridge_plan
from version.v8.tts.build_qwen3_tts_runtime_second_frame_circuit import build_circuit
from version.v8.tts.build_qwen3_tts_runtime_loop_codec_circuit import (
    build_circuit as build_runtime_loop,
)
from version.v8.tts.qwen3_tts_pinned_session import (
    PinnedQwenSession, _compile, _prompt_ids, _view, _weight_shapes,
)


ROOT = Path(__file__).resolve().parents[1]
EXTRA_SOURCES = (
    "src/kernels/select_next_id_checked.c",
    "src/kernels/repetition_penalty_checked.c",
    "src/kernels/copy_i32_span_checked.c",
    "src/kernels/equal_i32_scalar_checked.c",
)
SHORT_FIXTURE = (ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"
                 / "qwen3_tts_hi_continuation.npz")
LONG_FIXTURE = (SHORT_FIXTURE.parent / "qwen3_tts_talker_continuation.npz")
SPEECH_FIXTURE = SHORT_FIXTURE.parent / "qwen3_tts_thank_you_speech.npz"
SEVEN_TOKEN_SPEECH = SHORT_FIXTURE.parent / "qwen3_tts_customvoice_reference.npz"
EIGHT_TOKEN_SPEECH = SHORT_FIXTURE.parent / "qwen3_tts_eight_token_speech.npz"


def prepare(arena, acts, layout, arena_bytes, count):
    base = layout["memory"]["arena"]["activations_base"]
    ctypes.memset(ctypes.addressof(arena) + base, 0, arena_bytes - base)
    view(arena, acts, "text_ids", np.int32, (13,))[:] = 0
    view(arena, acts, "codec_ids", np.int32, (7,))[:] = 0
    for name, indices in prompt_plan(count).items():
        view(arena, acts, name, np.int32, (len(indices),))[:] = indices
    for name, indices in bridge_plan(count + 11).items():
        view(arena, acts, "bridge_" + name, np.int32, (2,))[:] = indices
    for name in ("pred_feedback_step_index_input", "feedback_left_indices",
                 "talker_step_index_input", "second_feedback_step_index_input"):
        view(arena, acts, name, np.int32, (1,))[0] = 0
    view(arena, acts, "feedback_right_indices", np.int32, (1,))[0] = 3
    view(arena, acts, "second_bridge_left_indices", np.int32, (2,))[:] = [0, -1]
    view(arena, acts, "second_bridge_right_indices", np.int32, (2,))[:] = [-1, 0]
    view(arena, acts, "request_context_tokens", np.int32, (1,))[0] = count


class Qwen3TTSRuntimeSecondFrameTest(unittest.TestCase):
    def test_short_reference_identity(self):
        manifest = json.loads(SHORT_FIXTURE.with_suffix(".json").read_text())
        baseline = json.loads((SHORT_FIXTURE.parent /
                               "qwen3_tts_talker_continuation.json").read_text())
        self.assertEqual(hashlib.sha256(SHORT_FIXTURE.read_bytes()).hexdigest(),
                         manifest["fixture_sha256"])
        self.assertEqual(manifest["request"]["text"], "Hi.")
        for key in ("model_revision", "model_files_sha256",
                    "reference_revision", "reference_files_sha256"):
            self.assertEqual(manifest[key], baseline[key])
        with np.load(SHORT_FIXTURE) as short:
            self.assertEqual(short["step0_input"].shape, (13, 1024))
            self.assertEqual(short["step1_logits"].shape, (1, 3072))
            self.assertTrue(all(np.isfinite(short[name]).all()
                                for name in short.files))

    def test_three_token_speech_reference_identity(self):
        manifest = json.loads(SPEECH_FIXTURE.with_suffix(".json").read_text())
        baseline = json.loads((SHORT_FIXTURE.parent /
                               "qwen3_tts_talker_continuation.json").read_text())
        self.assertEqual(hashlib.sha256(SPEECH_FIXTURE.read_bytes()).hexdigest(),
                         manifest["fixture_sha256"])
        for key in ("model_revision", "model_files_sha256",
                    "reference_revision", "reference_files_sha256"):
            self.assertEqual(manifest[key], baseline[key])
        self.assertEqual(manifest["request"]["text"], "Thank you.")
        self.assertFalse(manifest["request"]["do_sample"])
        self.assertFalse(manifest["request"]["subtalker_dosample"])
        self.assertEqual(manifest["reference_runtime"]["torch"], "2.12.1+cpu")
        self.assertEqual(manifest["reference_runtime"]["transformers"], "4.57.3")
        self.assertEqual(manifest["reference_runtime"]["attention"], "eager")
        self.assertEqual(manifest["reference_runtime"]["dtype"], "float32")
        with np.load(SPEECH_FIXTURE) as reference:
            self.assertEqual(reference["context_ids"].shape, (1, 3))
            self.assertEqual(reference["codec_ids"].shape, (20, 16))
            self.assertEqual(reference["waveform"].shape, (20 * 1920,))
            self.assertTrue(np.isfinite(reference["waveform"]).all())

    def test_eight_token_speech_reference_identity(self):
        manifest = json.loads(EIGHT_TOKEN_SPEECH.with_suffix(".json").read_text())
        baseline = json.loads(SPEECH_FIXTURE.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(EIGHT_TOKEN_SPEECH.read_bytes()).hexdigest(),
                         manifest["fixture_sha256"])
        for key in ("model_revision", "model_files_sha256",
                    "reference_revision", "reference_files_sha256"):
            self.assertEqual(manifest[key], baseline[key])
        self.assertEqual(manifest["request"]["text"], "Yes, I can hear you now.")
        self.assertFalse(manifest["request"]["do_sample"])
        self.assertFalse(manifest["request"]["subtalker_dosample"])
        self.assertEqual(manifest["reference_runtime"]["torch"], "2.12.1+cpu")
        self.assertEqual(manifest["reference_runtime"]["transformers"], "4.57.3")
        with np.load(EIGHT_TOKEN_SPEECH) as reference:
            self.assertEqual(reference["context_ids"].shape, (1, 8))
            self.assertEqual(reference["codec_ids"].shape, (39, 16))
            self.assertEqual(reference["waveform"].shape, (39 * 1920,))
            self.assertTrue(np.isfinite(reference["waveform"]).all())

    def test_runtime_prompt_rejects_bad_lengths_before_model_execution(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                if text.startswith("<|im_start|>"):
                    return [1, 2, 3]
                return list(range(int(text)))

        config = {"tts_pad_token_id": 4, "tts_bos_token_id": 5,
                  "tts_eos_token_id": 6,
                  "talker_config": {"codec_think_id": 7, "codec_think_bos_id": 8,
                                    "codec_language_id": {"english": 9},
                                    "codec_think_eos_id": 10,
                                    "spk_id": {"ryan": 11},
                                    "codec_pad_id": 12, "codec_bos_id": 13}}
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            (model_dir / "config.json").write_text(json.dumps(config))
            for count in (1, 3, 7, 16):
                text_ids, _ = _prompt_ids(model_dir, str(count), Tokenizer(), None)
                self.assertEqual(len(text_ids), count + 6)
            for count in (0, 17):
                with self.assertRaisesRegex(ValueError, "1..16 tokens"):
                    _prompt_ids(model_dir, str(count), Tokenizer(), None)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                _prompt_ids(model_dir, "\n", Tokenizer(), None)

    def test_canonical_runtime_edges(self):
        graph = build_circuit()
        committed = ROOT / "version/v8/circuits/qwen3_tts_runtime_second_frame_bounded.json"
        self.assertEqual(json.loads(committed.read_text()), graph)
        self.assertEqual(len(graph["native_entry"]["segments"]), 2)
        header = graph["block_types"]["talker_code_feedback"]["header"]
        position = next(op for op in header if op["id"] == "check_talker_decode_position")
        self.assertEqual(position["runtime_scalar_bindings"], {
            "extent_affine_source": "talker_step_index",
            "extent_affine_offset": "frame_prompt_rows"})
        self.assertNotIn("extent_affine_offset", position["params"]["call_constants"])
        self.assertLess([op["id"] for op in header].index("verify_prefill_frame_prompt_rows"),
                        [op["id"] for op in header].index("check_talker_decode_position"))
        for invalid in (0, -1, 118, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_circuit(context_capacity=invalid)

    def test_runtime_loop_artifact_matches_authoring(self):
        graph = build_runtime_loop()
        committed = ROOT / "version/v8/circuits/qwen3_tts_runtime_loop_codec_bounded.json"
        self.assertEqual(json.loads(committed.read_text()), graph)
        self.assertEqual(len(graph["native_entry"]["segments"]), 3)
        self.assertEqual(graph["contract"]["runtime_invariants"]["context_capacity"], 16)

    def test_generated_segments_change_position_and_reject_stale_context(self):
        graph = build_circuit(talker_layers=1, continuation_steps=2,
                              max_generated_frames=4)
        with tempfile.TemporaryDirectory() as temp:
            _, acts, _, layout, _, library = compile_entry(
                Path(temp), graph, weight_shapes(graph), extra_sources=EXTRA_SOURCES)
            native = ctypes.CDLL(str(library))
            prefill = getattr(native, graph["native_entry"]["segments"][0]["function"])
            frame = getattr(native, graph["native_entry"]["segments"][1]["function"])
            for fn in (prefill, frame):
                fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
                fn.restype = ctypes.c_int
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for count in (2, 7, 2):
                with self.subTest(context_tokens=count):
                    prepare(arena, acts, layout, size, count)
                    self.assertEqual(prefill(arena, size), 0)
                    rows = count + 11
                    self.assertEqual(int(view(arena, acts, "published_prompt_rows",
                                              np.int32, (1,))[0]), rows)
                    for kind in ("k", "v"):
                        cache = view(arena, acts, f"talker_layer0_cache_{kind}",
                                     np.float32, (8, 21, 128))
                        cache[:, rows:] = -777
                    self.assertEqual(frame(arena, size), 0)
                    for kind in ("k", "v"):
                        cache = view(arena, acts, f"talker_layer0_cache_{kind}",
                                     np.float32, (8, 21, 128))
                        self.assertTrue(np.all(cache[:, rows] != -777))
                        np.testing.assert_array_equal(cache[:, rows + 1:], -777)
                    view(arena, acts, "talker_step_index_input", np.int32, (1,))[0] = 1
                    self.assertEqual(frame(arena, size), 0)
                    for kind in ("k", "v"):
                        cache = view(arena, acts, f"talker_layer0_cache_{kind}",
                                     np.float32, (8, 21, 128))
                        self.assertTrue(np.all(cache[:, rows + 1] != -777))
                        np.testing.assert_array_equal(cache[:, rows + 2:], -777)
            prepare(arena, acts, layout, size, 2)
            self.assertEqual(prefill(arena, size), 0)
            view(arena, acts, "request_context_tokens", np.int32, (1,))[0] = 7
            self.assertNotEqual(frame(arena, size), 0)
            self.assertEqual(int(view(arena, acts, "second_predictor_valid_code_count",
                                      np.int32, (1,))[0]), 0)
            prepare(arena, acts, layout, size, 2)
            self.assertEqual(prefill(arena, size), 0)
            self.assertEqual(frame(arena, size), 0)
            for invalid in (0, 8):
                prepare(arena, acts, layout, size, 2)
                view(arena, acts, "request_context_tokens", np.int32, (1,))[0] = invalid
                published = view(arena, acts, "published_prompt_rows", np.int32, (1,))
                published[0] = -901
                self.assertNotEqual(prefill(arena, size), 0)
                self.assertEqual(int(published[0]), -901)
                prepare(arena, acts, layout, size, 2)
                self.assertEqual(prefill(arena, size), 0)
                self.assertEqual(frame(arena, size), 0)

    def test_sixteen_token_capacity_stitches_through_generated_frame(self):
        graph = build_circuit(context_capacity=16, talker_layers=1,
                              continuation_steps=2, max_generated_frames=4)
        with tempfile.TemporaryDirectory() as temp:
            _, acts, _, layout, _, library = compile_entry(
                Path(temp), graph, weight_shapes(graph), extra_sources=EXTRA_SOURCES)
            native = ctypes.CDLL(str(library))
            prefill = getattr(native, graph["native_entry"]["segments"][0]["function"])
            frame = getattr(native, graph["native_entry"]["segments"][1]["function"])
            for fn in (prefill, frame):
                fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
                fn.restype = ctypes.c_int
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for count in (3, 16, 3):
                with self.subTest(context_tokens=count):
                    prepare(arena, acts, layout, size, count)
                    self.assertEqual(prefill(arena, size), 0)
                    rows = count + 11
                    self.assertEqual(int(view(arena, acts, "published_prompt_rows",
                                              np.int32, (1,))[0]), rows)
                    for kind in ("k", "v"):
                        cache = view(arena, acts, f"talker_layer0_cache_{kind}",
                                     np.float32, (8, 30, 128))
                        cache[:, rows:] = -777
                    self.assertEqual(frame(arena, size), 0)
                    for kind in ("k", "v"):
                        cache = view(arena, acts, f"talker_layer0_cache_{kind}",
                                     np.float32, (8, 30, 128))
                        self.assertTrue(np.all(cache[:, rows] != -777))
                        np.testing.assert_array_equal(cache[:, rows + 1:], -777)
            prepare(arena, acts, layout, size, 3)
            view(arena, acts, "request_context_tokens", np.int32, (1,))[0] = 17
            published = view(arena, acts, "published_prompt_rows", np.int32, (1,))
            published[0] = -901
            self.assertNotEqual(prefill(arena, size), 0)
            self.assertEqual(int(published[0]), -901)
            prepare(arena, acts, layout, size, 3)
            self.assertEqual(prefill(arena, size), 0)
            self.assertEqual(frame(arena, size), 0)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned Qwen assets unavailable; runtime continuation NOT_TESTED")
    def test_full_weight_short_long_short_continuation(self):
        from safetensors import safe_open
        from transformers import AutoTokenizer

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        graph = build_circuit()
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        with tempfile.TemporaryDirectory() as temp, \
                safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as main, \
                safe_open(model_dir / "speech_tokenizer/model.safetensors",
                          framework="pt", device="cpu") as codec, \
                np.load(SHORT_FIXTURE) as short, np.load(LONG_FIXTURE) as long:
            shapes = _weight_shapes(graph, main, codec)
            acts, weights, layout, _, library = _compile(Path(temp), graph, shapes)
            size = layout["memory"]["arena"]["total_size"]
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for name, item in weights.items():
                value = np.ascontiguousarray(main.get_tensor(name).float().numpy())
                self.assertEqual(value.shape, shapes[name], name)
                arena[item["abs_offset"]:item["abs_offset"] + value.nbytes] = value.tobytes()
            native = ctypes.CDLL(str(library))
            prefill = getattr(native, graph["native_entry"]["segments"][0]["function"])
            frame = getattr(native, graph["native_entry"]["segments"][1]["function"])
            for fn in (prefill, frame):
                fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
                fn.restype = ctypes.c_int
            first_short = None
            for text, count, oracle in (("Hi.", 2, short),
                                        ("Hello, this is CKE.", 7, long),
                                        ("Hi.", 2, short)):
                with self.subTest(text=text):
                    prepare(arena, acts, layout, size, count)
                    text_ids, codec_ids = _prompt_ids(
                        model_dir, text, tokenizer, context_tokens=count)
                    _view(arena, acts, "text_ids", np.int32,
                          (len(text_ids),))[:] = text_ids
                    _view(arena, acts, "codec_ids", np.int32, (7,))[:] = codec_ids
                    self.assertEqual(prefill(arena, size), 0)
                    rows = count + 11
                    prefill_logits = _view(arena, acts, "talker_logits", np.float32,
                                           (18, 3072))[rows - 1]
                    expected_logits = (oracle["step0_logits_last"] if count == 2
                                       else oracle["step0_logits"][0, -1])
                    self.assertLessEqual(float(np.max(np.abs(
                        prefill_logits - expected_logits))), 1.5e-4)
                    for layer in range(28):
                        for kind in ("k", "v"):
                            cache = _view(arena, acts, f"talker_layer{layer}_cache_{kind}",
                                          np.float32, (8, 81, 128))
                            reference = oracle[f"step0_layer{layer}_{kind}_cache"]
                            if count == 7:
                                reference = reference[0]
                            self.assertLessEqual(float(np.max(np.abs(
                                cache[:, :rows] - reference))), 8e-5)
                    self.assertEqual(frame(arena, size), 0)
                    next_input = _view(arena, acts, "feedback_next_input", np.float32,
                                       (1, 1024))
                    expected_input = (oracle["step1_input"] if count == 2
                                      else oracle["step1_input"][0])
                    self.assertLessEqual(float(np.max(np.abs(
                        next_input - expected_input))), 4e-5)
                    logits = _view(arena, acts, "cont_talker_logits", np.float32,
                                   (1, 3072))
                    expected = (oracle["step1_logits"] if count == 2
                                else oracle["step1_logits"][0])
                    self.assertLessEqual(float(np.max(np.abs(logits - expected))), 2e-4)
                    self.assertEqual(int(np.argmax(logits)), int(np.argmax(expected)))
                    for layer in range(28):
                        for kind in ("k", "v"):
                            cache = _view(arena, acts, f"talker_layer{layer}_cache_{kind}",
                                          np.float32, (8, 81, 128))
                            reference = oracle[f"step1_layer{layer}_{kind}_new"]
                            if count == 7:
                                reference = reference[0]
                            np.testing.assert_allclose(cache[:, rows:rows + 1],
                                                       reference, atol=4e-5, rtol=4e-5)
                    if count == 2:
                        if first_short is None:
                            first_short = logits.copy()
                        else:
                            np.testing.assert_array_equal(logits, first_short)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned Qwen assets unavailable; varied-length WAV NOT_TESTED")
    def test_three_lengths_generated_wavs_match_direct_references(self):
        with tempfile.TemporaryDirectory() as temp:
            session = PinnedQwenSession(
                Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"]), Path(temp),
                provider=os.environ.get("CKE_QWEN3_TTS_CODEC_CONV_PROVIDER", "scalar"),
                runtime_prompts=True)
            with self.assertRaisesRegex(ValueError, "1..16 tokens"):
                session.synthesize("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen")
            for text, count, frames, path, code_key in (
                ("Thank you.", 3, 20, SPEECH_FIXTURE, "codec_ids"),
                ("Hello, this is CKE.", 7, 46, SEVEN_TOKEN_SPEECH, "codec_tokens"),
                ("Yes, I can hear you now.", 8, 39, EIGHT_TOKEN_SPEECH, "codec_ids"),
            ):
                with self.subTest(text=text), np.load(path) as reference:
                    samples, metric = session.synthesize(text)
                    self.assertEqual(metric["text_tokens"], count)
                    self.assertEqual(metric["prompt_rows"], count + 11)
                    self.assertEqual(metric["frames"], frames)
                    self.assertEqual(metric["samples"], frames * 1920)
                    self.assertEqual(metric["load_count"], 1)
                    expected_codes = np.ascontiguousarray(
                        reference[code_key].astype(np.int32))
                    self.assertEqual(metric["code_sha256"], hashlib.sha256(
                        expected_codes.tobytes()).hexdigest())
                    expected = reference["waveform"]
                    self.assertEqual(samples.shape, expected.shape)
                    self.assertTrue(np.isfinite(samples).all())
                    actual_pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype(np.int32)
                    expected_pcm = np.rint(np.clip(expected, -1, 1) * 32767).astype(np.int32)
                    error = np.abs(actual_pcm - expected_pcm)
                    print("QWEN3_TTS_RUNTIME_PROMPT_ORACLE " + json.dumps({
                        "text_tokens": count,
                        "frames": metric["frames"],
                        "code_sha256": metric["code_sha256"],
                        "pcm_max_lsb": int(error.max()),
                        "pcm_differing_samples": int(np.count_nonzero(error)),
                    }, sort_keys=True), flush=True)
                    self.assertLessEqual(int(error.max()), 1)
