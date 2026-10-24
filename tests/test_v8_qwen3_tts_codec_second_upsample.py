"""Generated Qwen3-TTS codec through the second causal factor-2 upsample."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_qwen3_tts_codec_semantic import MODEL, generated_entry, view
from tests.test_v8_qwen3_tts_codec_preconv import fill_ids
from tests.test_v8_qwen3_tts_codec_attention import synthetic_weights
from tests.test_v8_qwen3_tts_codec_pretransformer import (
    add_synthetic_weights, pinned_pretransformer_weights,
)
from tests.test_v8_qwen3_tts_codec_convnext_complete import (
    WEIGHT_NAMES, UPSAMPLE_WEIGHT, UPSAMPLE_BIAS, DEPTHWISE_WEIGHT,
    DEPTHWISE_BIAS, add_synthetic_convnext_weights,
)
from version.v8.tts.build_qwen3_tts_codec_second_upsample_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_second_upsample.npz")
WEIGHT = "codec.decoder.upsample.1.0.conv.weight"
BIAS = "codec.decoder.upsample.1.0.conv.bias"


class Qwen3TTSCodecSecondUpsampleTest(unittest.TestCase):
    def test_authored_circuit_and_fixture_integrity(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_second_upsample_bounded.json"
        graph = build_circuit()
        self.assertEqual(json.loads(path.read_text()), graph)
        op = graph["block_types"]["codec_semantic"]["body"]["ops"][-1]
        self.assertEqual(op["id"], "second_upsample")
        self.assertEqual(op["graph_slots"]["inputs"]["input"], "convnext_output")
        self.assertEqual(op["weight_refs"], {"weight": WEIGHT, "bias": BIAS})
        self.assertEqual(op["params"]["call_constants"]["dense_deconv_input_frames"], 92)
        self.assertEqual(op["params"]["call_constants"]["dense_deconv_output_frames"], 184)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        prior = MODEL.with_name("qwen3_tts_codec_convnext_complete.npz")
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),
                         manifest["source_convnext_sha256"])
        prior_manifest = json.loads(prior.with_suffix(".json").read_text())
        for identity in ("model_revision", "reference_revision", "codec_source_sha256"):
            self.assertEqual(manifest[identity], prior_manifest[identity], identity)
        self.assertLessEqual(manifest["source_convnext_max_abs"], 1e-7)
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(set(oracle.files),
                             {"convnext_output", "second_upsample"})
            self.assertEqual(oracle["convnext_output"].shape, (1024, 92))
            self.assertEqual(oracle["second_upsample"].shape, (1024, 184))
            self.assertTrue(np.isfinite(oracle["second_upsample"]).all())

    def test_generated_second_upsample_and_recovery(self):
        rng = np.random.default_rng(20261024)
        arrays = synthetic_weights(rng, vocab=12)
        add_synthetic_weights(arrays, rng, 2)
        add_synthetic_convnext_weights(arrays, rng)
        arrays[WEIGHT] = (rng.normal(size=(3, 3, 2)) * .1).astype(np.float32)
        arrays[BIAS] = (rng.normal(size=3) * .1).astype(np.float32)
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    second_decoder_upsample=True, convnext_expansion=2,
                    transformer_layers=2,
                    qkv_heads=2, qkv_head_dim=2, sliding_window=3)
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                source = view(arena, acts, "convnext_output", np.float32,
                              (3, frames * 2))
                output = view(arena, acts, "second_upsample", np.float32,
                              (3, frames * 4))
                expected = np.broadcast_to(
                    arrays[BIAS][:, None], output.shape).copy()
                for frame in range(frames * 2):
                    for tap in range(2):
                        expected[:, frame * 2 + tap] += (
                            source[:, frame] @ arrays[WEIGHT][:, :, tap])
                np.testing.assert_allclose(output, expected, atol=2e-6, rtol=2e-6)
                baseline = output.copy()
                entry = next(item for item in calls["memory"]["weights"]["entries"]
                             if item["name"] == WEIGHT)
                bound = np.ndarray(arrays[WEIGHT].shape, dtype=np.float32,
                                   buffer=arena, offset=entry["abs_offset"])
                saved = bound.copy()
                bound[0, 0, 0] = np.nan
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -123)
                bound[:] = saved
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                np.testing.assert_array_equal(output, -123)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; direct-model second upsample NOT_TESTED")
    def test_pinned_connected_second_upsample(self):
        from safetensors import safe_open
        import torch

        torch.set_num_threads(4)
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = pinned_pretransformer_weights(model_dir)
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in (UPSAMPLE_WEIGHT, UPSAMPLE_BIAS,
                         DEPTHWISE_WEIGHT, DEPTHWISE_BIAS,
                         *WEIGHT_NAMES, WEIGHT, BIAS):
                arrays[name] = np.ascontiguousarray(
                    source.get_tensor(name.removeprefix("codec.")).float().numpy())
        with np.load(MODEL) as fixture, np.load(CHECKPOINT) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            with tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, _, library = generated_entry(
                    Path(temp), arrays, len(codes), acoustic_codebooks=15,
                    second_decoder_upsample=True)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                errors, rmses, worst = {}, {}, {}
                for name in oracle.files:
                    expected = oracle[name]
                    actual = view(arena, acts, name, np.float32, expected.shape)
                    diff = np.abs(actual - expected)
                    errors[name] = float(np.max(diff))
                    rmses[name] = float(np.sqrt(np.mean(np.square(diff))))
                    index = np.unravel_index(int(np.argmax(diff)), diff.shape)
                    worst[name] = {"index": [int(i) for i in index],
                                   "actual": float(actual[index]),
                                   "reference": float(expected[index])}
                    self.assertTrue(np.isfinite(actual).all(), name)
                actual_input = view(arena, acts, "convnext_output",
                                    np.float32, oracle["convnext_output"].shape)
                actual_output = view(arena, acts, "second_upsample",
                                     np.float32, oracle["second_upsample"].shape)
                with torch.inference_mode():
                    same_input = torch.nn.functional.conv_transpose1d(
                        torch.from_numpy(actual_input.copy()[None]),
                        torch.from_numpy(arrays[WEIGHT]),
                        torch.from_numpy(arrays[BIAS]), stride=2)[0].numpy()
                    with torch.backends.mkldnn.flags(enabled=False):
                        same_input_no_mkldnn = torch.nn.functional.conv_transpose1d(
                            torch.from_numpy(actual_input.copy()[None]),
                            torch.from_numpy(arrays[WEIGHT]),
                            torch.from_numpy(arrays[BIAS]), stride=2)[0].numpy()
                local_error = float(np.max(np.abs(actual_output - same_input)))
                no_mkldnn_error = float(np.max(np.abs(
                    actual_output - same_input_no_mkldnn)))
                pointer = ctypes.POINTER(ctypes.c_float)
                provider = ctypes.CDLL(str(library)).audio_conv_transpose1d_dense_channel_major_f32_checked
                provider.argtypes = [
                    pointer, ctypes.c_size_t, ctypes.c_size_t,
                    pointer, ctypes.c_size_t, pointer, ctypes.c_size_t,
                    pointer, ctypes.c_size_t, ctypes.c_size_t,
                    pointer, ctypes.c_size_t,
                    *([ctypes.c_size_t] * 8)]
                provider.restype = ctypes.c_int
                reference_input = np.ascontiguousarray(
                    oracle["convnext_output"], dtype=np.float32)
                reference_input_output = np.empty_like(actual_output)
                scratch = np.empty_like(actual_output)
                def ptr(value):
                    return value.ctypes.data_as(pointer)
                self.assertEqual(provider(
                    ptr(reference_input), reference_input.size,
                    reference_input.shape[1],
                    ptr(arrays[WEIGHT]), arrays[WEIGHT].size,
                    ptr(arrays[BIAS]), arrays[BIAS].size,
                    ptr(reference_input_output), reference_input_output.size,
                    reference_input_output.shape[1],
                    ptr(scratch), scratch.size,
                    1024, 1024, 92, 2, 2, 0, 0, 184), 0)
                reference_input_local_error = float(np.max(np.abs(
                    reference_input_output - oracle["second_upsample"])))
                propagated_error = float(np.max(np.abs(
                    actual_output - reference_input_output)))
                print("QWEN3_TTS_CODEC_SECOND_UPSAMPLE " + json.dumps({
                    "max_abs_errors": errors, "rmse": rmses, "worst": worst,
                    "same_input_local_max_abs": local_error,
                    "same_input_no_mkldnn_max_abs": no_mkldnn_error,
                    "reference_input_local_max_abs": reference_input_local_error,
                    "upstream_propagated_max_abs": propagated_error,
                    "torch_num_threads": torch.get_num_threads(),
                    "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
                    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                    "scope": "reference codec codes through second upsample; second ConvNeXt and waveform NOT_TESTED",
                }, sort_keys=True))
                self.assertLessEqual(errors["convnext_output"], 5e-6)
                self.assertLessEqual(errors["second_upsample"], 8e-5)
                self.assertLessEqual(local_error, 3e-5)
                self.assertLessEqual(reference_input_local_error, 3e-5)


if __name__ == "__main__":
    unittest.main()
