"""Generated reference-code-fed Qwen codec through final bounded PCM samples."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from tests.test_v8_qwen3_tts_codec_semantic import MODEL, generated_entry, view
from tests.test_v8_qwen3_tts_codec_preconv import fill_ids
from tests import test_v8_qwen3_tts_codec_third_decoder_block as third
from tests import test_v8_qwen3_tts_codec_second_decoder_block as second
from tests import test_v8_qwen3_tts_codec_decoder_upsample as first
from tests.test_v8_qwen3_tts_codec_decoder_ingress import CONV_WEIGHT, CONV_BIAS
from tests.test_v8_qwen3_tts_codec_snake_beta import ALPHA, BETA
from tests.test_v8_qwen3_tts_codec_first_residual_pool import weight_names as residual_weights
from version.v8.tts.build_qwen3_tts_codec_waveform_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL.with_name("qwen3_tts_codec_waveform.npz")
PRIOR = MODEL.with_name("qwen3_tts_codec_third_decoder_block.npz")
RETAINED_WAV = MODEL.with_name("qwen3_tts_codec_generated_46.wav")
RETAINED_WAV_SHA256 = "77d03382bd6120c6fd5e94a84af147732f4d205913136f104f4f8e9ef49b773f"
FINAL_ALPHA = "codec.decoder.decoder.5.alpha"
FINAL_BETA = "codec.decoder.decoder.5.beta"
FINAL_WEIGHT = "codec.decoder.decoder.6.conv.weight"
FINAL_BIAS = "codec.decoder.decoder.6.conv.bias"
FOURTH_ALPHA = "codec.decoder.decoder.4.block.0.alpha"
FOURTH_BETA = "codec.decoder.decoder.4.block.0.beta"
FOURTH_WEIGHT = "codec.decoder.decoder.4.block.1.conv.weight"
FOURTH_BIAS = "codec.decoder.decoder.4.block.1.conv.bias"
SHORT_CHECKPOINT_LIMITS = {
    "decoder_third_residual_2_output": 2e-4,
    "decoder_fourth_snake_beta_output": 2e-4,
    "decoder_fourth_upsample_output": 1e-5,
    "decoder_fourth_residual_0_output": 3e-5,
    "decoder_fourth_residual_1_output": 4e-4,
    "decoder_fourth_residual_2_output": 6e-4,
    "decoder_final_snake_beta_output": 1e-3,
    "decoder_waveform_preclip_output": 3e-6,
    "waveform_output": 3e-6,
}


def new_weight_names():
    return (FOURTH_ALPHA, FOURTH_BETA, FOURTH_WEIGHT, FOURTH_BIAS,
            *(name for unit in range(3)
              for name in residual_weights(unit, decoder_block=4).values()),
            FINAL_ALPHA, FINAL_BETA, FINAL_WEIGHT, FINAL_BIAS)


def synthetic_arrays(rng):
    arrays = third.Qwen3TTSCodecThirdDecoderBlockTest.synthetic_arrays(rng)
    arrays[CONV_WEIGHT] = (rng.normal(size=(16, 3, 7)) * .03).astype(np.float32)
    arrays[CONV_BIAS] = (rng.normal(size=16) * .03).astype(np.float32)
    arrays[ALPHA] = rng.uniform(-.2, .2, 16).astype(np.float32)
    arrays[BETA] = rng.uniform(-.2, .2, 16).astype(np.float32)
    for block, in_channels, out_channels in ((1, 16, 8), (2, 8, 4),
                                              (3, 4, 2), (4, 2, 1)):
        snake_a = f"codec.decoder.decoder.{block}.block.0.alpha"
        snake_b = f"codec.decoder.decoder.{block}.block.0.beta"
        weight = f"codec.decoder.decoder.{block}.block.1.conv.weight"
        bias = f"codec.decoder.decoder.{block}.block.1.conv.bias"
        arrays[snake_a] = rng.uniform(-.2, .2, in_channels).astype(np.float32)
        arrays[snake_b] = rng.uniform(-.2, .2, in_channels).astype(np.float32)
        arrays[weight] = (rng.normal(size=(in_channels, out_channels, 4)) * .03).astype(np.float32)
        arrays[bias] = (rng.normal(size=out_channels) * .03).astype(np.float32)
        for unit in range(3):
            names = residual_weights(unit, decoder_block=block)
            for part in ("a1", "b1", "a2", "b2"):
                arrays[names[part]] = rng.uniform(-.2, .2, out_channels).astype(np.float32)
            for part, kernel in (("w1", 7), ("w2", 1)):
                arrays[names[part]] = (rng.normal(size=(out_channels, out_channels, kernel)) * .03).astype(np.float32)
            for part in ("bias1", "bias2"):
                arrays[names[part]] = (rng.normal(size=out_channels) * .03).astype(np.float32)
    arrays[FINAL_ALPHA] = rng.uniform(-.2, .2, 1).astype(np.float32)
    arrays[FINAL_BETA] = rng.uniform(-.2, .2, 1).astype(np.float32)
    arrays[FINAL_WEIGHT] = (rng.normal(size=(1, 1, 7)) * .03).astype(np.float32)
    arrays[FINAL_BIAS] = np.array([.1], dtype=np.float32)
    return arrays


class Qwen3TTSCodecWaveformTest(unittest.TestCase):
    def test_circuit_and_reference_fixture_integrity(self):
        graph = build_circuit()
        self.assertEqual(json.loads((ROOT / "version/v8/circuits/qwen3_tts_codec_waveform_bounded.json").read_text()), graph)
        ops = graph["block_types"]["codec_semantic"]["body"]["ops"]
        self.assertEqual(ops[-3]["op"], "audio_zero_pad1d_left_checked")
        self.assertEqual(ops[-2]["weight_refs"]["weight"], FINAL_WEIGHT)
        self.assertEqual(ops[-1]["op"], "clip_strided_checked")
        self.assertEqual(ops[-1]["graph_slots"]["inputs"]["input"],
                         "decoder_waveform_preclip_output")
        self.assertEqual(ops[-1]["params"]["maximum"], 1.0)
        self.assertEqual(graph["activation_buffers"]["waveform_output"]["shape"], [1, 88320])
        for invalid in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                build_circuit(fourth_decoder_upsample_rate=invalid)
        with self.assertRaises(ValueError):
            build_circuit(decoder_width=24)
        manifest = json.loads(CHECKPOINT.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), manifest["npz_sha256"])
        self.assertEqual(hashlib.sha256(PRIOR.read_bytes()).hexdigest(),
                         manifest["source_third_decoder_block_sha256"])
        previous = json.loads(PRIOR.with_suffix(".json").read_text())
        for key in ("model_revision", "reference_revision", "codec_source_sha256",
                    "source_codec_fixture_sha256", "short_codec_ids_sha256",
                    "full_codec_ids_sha256"):
            self.assertEqual(manifest[key], previous[key])
        self.assertLessEqual(manifest["source_third_decoder_short_max_abs"], 1e-7)
        self.assertLessEqual(manifest["source_third_decoder_full_max_abs"], 1e-7)
        self.assertEqual((manifest["short_samples"], manifest["full_samples"]), (15360, 88320))
        self.assertEqual(manifest["sample_rate_hz"], 24000)
        self.assertEqual(hashlib.sha256(RETAINED_WAV.read_bytes()).hexdigest(),
                         RETAINED_WAV_SHA256)
        with wave.open(str(RETAINED_WAV), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(),
                              wav.getframerate(), wav.getnframes()),
                             (1, 2, 24000, 88320))
        with np.load(CHECKPOINT) as oracle:
            self.assertEqual(oracle["waveform_output"].shape, (1, 15360))
            self.assertEqual(oracle["full_waveform_output"].shape, (1, 88320))
            self.assertTrue(all(np.isfinite(oracle[name]).all() for name in oracle.files))
            with np.load(MODEL) as baseline:
                self.assertLessEqual(float(np.max(np.abs(
                    oracle["full_waveform_output"].reshape(-1) -
                    baseline["waveform"]))), 1e-6)

    def test_synthetic_generated_pcm_and_recovery(self):
        arrays = synthetic_arrays(np.random.default_rng(20261025))
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2,
                    codec_waveform=True, decoder_upsample_rate=2,
                    second_decoder_upsample_rate=2,
                    third_decoder_upsample_rate=2,
                    fourth_decoder_upsample_rate=2,
                    convnext_expansion=2, transformer_layers=2,
                    qkv_heads=2, qkv_head_dim=2, sliding_window=3)
                self.assertEqual(calls["operations"][-1]["function"], "clip_strided_f32_checked")
                codes = np.stack((np.arange(1, frames + 1),
                                  np.arange(2, frames + 2),
                                  np.arange(3, frames + 3)), axis=1).astype(np.int32)
                fill_ids(arena, acts, codes)
                self.assertEqual(fn(arena, len(arena)), 0)
                length = frames * 64
                preclip = view(arena, acts, "decoder_waveform_preclip_output", np.float32, (1, length))
                output = view(arena, acts, "waveform_output", np.float32, (1, length))
                np.testing.assert_array_equal(output, np.clip(preclip, -1, 1))
                baseline = output.copy()
                entries = {item["name"]: item for item in calls["memory"]["weights"]["entries"]}
                weight = np.ndarray(arrays[FINAL_WEIGHT].shape, np.float32, buffer=arena,
                                    offset=entries[FINAL_WEIGHT]["abs_offset"])
                old = weight.copy()
                weight.fill(np.nan)
                output.fill(-77)
                self.assertNotEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, -77)
                weight[:] = old
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                np.testing.assert_array_equal(output, -77)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_array_equal(output, baseline)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned weights unavailable; connected PCM NOT_TESTED")
    def test_pinned_connected_pcm(self):
        from safetensors import safe_open
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(4)

        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        arrays = second.pinned_second_block_weights(model_dir)
        names = (third.SNAKE_ALPHA, third.SNAKE_BETA,
                 third.UPSAMPLE_WEIGHT, third.UPSAMPLE_BIAS,
                 *(name for unit in range(3) for name in third.weight_names(unit).values()),
                 *new_weight_names())
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for name in names:
                arrays[name] = np.ascontiguousarray(source.get_tensor(
                    name.removeprefix("codec.")).float().numpy())
        with np.load(MODEL) as source, np.load(CHECKPOINT) as oracle:
            codes = source["codec_tokens"].astype(np.int32)
            lengths = (8,) if os.environ.get("CKE_QWEN3_TTS_CODEC_SHORT_ONLY") == "1" else (8, 46)
            generated_waveforms = {}
            for frames in lengths:
                with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                    fn, arena, acts, calls, library = generated_entry(
                        Path(temp), arrays, frames, acoustic_codebooks=15,
                        codec_waveform=True)
                    fill_ids(arena, acts, codes[:frames])
                    self.assertEqual(fn(arena, len(arena)), 0)
                    expected = oracle["waveform_output" if frames == 8 else "full_waveform_output"]
                    actual = view(arena, acts, "waveform_output", np.float32, expected.shape)
                    generated_waveforms[frames] = actual.copy()
                    difference = actual.astype(np.float64) - expected.astype(np.float64)
                    index = np.unravel_index(int(np.argmax(np.abs(difference))), difference.shape)
                    metric = {"code_frames": frames, "samples": actual.size,
                              "sample_rate_hz": 24000,
                              "max_abs": float(np.max(np.abs(difference))),
                              "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                              "worst_index": list(map(int, index)),
                              "actual": float(actual[index]),
                              "reference": float(expected[index]),
                              "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
                    print("QWEN3_TTS_CODEC_PCM " + json.dumps(metric, sort_keys=True))
                    if frames == 8:
                        checkpoint_failures = []
                        for name, limit in SHORT_CHECKPOINT_LIMITS.items():
                            reference_stage = oracle[name]
                            actual_stage = view(arena, acts, name, np.float32,
                                                reference_stage.shape)
                            delta = (actual_stage.astype(np.float64) -
                                     reference_stage.astype(np.float64))
                            maximum = float(np.max(np.abs(delta)))
                            print("QWEN3_TTS_CODEC_CHECKPOINT " + json.dumps({
                                "name": name,
                                "max_abs": maximum,
                                "rmse": float(np.sqrt(np.mean(np.square(delta)))),
                                "limit": limit,
                            }, sort_keys=True))
                            self.assertTrue(np.isfinite(actual_stage).all())
                            if maximum > limit:
                                checkpoint_failures.append((name, maximum, limit))
                        names = residual_weights(1, decoder_block=4)

                        def replay_residual_one(value):
                            tensor = torch.from_numpy(value.copy()[None])

                            def snake(x, alpha_name, beta_name):
                                alpha = torch.from_numpy(arrays[names[alpha_name]])[None, :, None]
                                beta = torch.from_numpy(arrays[names[beta_name]])[None, :, None]
                                return x + torch.sin(x * torch.exp(alpha)).pow(2) / (
                                    torch.exp(beta) + 1e-9)

                            with torch.inference_mode():
                                result = snake(tensor, "a1", "b1")
                                result = F.conv1d(
                                    F.pad(result, (18, 0)),
                                    torch.from_numpy(arrays[names["w1"]]),
                                    torch.from_numpy(arrays[names["bias1"]]),
                                    dilation=3)
                                result = snake(result, "a2", "b2")
                                result = F.conv1d(
                                    result, torch.from_numpy(arrays[names["w2"]]),
                                    torch.from_numpy(arrays[names["bias2"]]))
                                return (result + tensor)[0].numpy()

                        previous = view(arena, acts, "decoder_fourth_residual_0_output",
                                        np.float32, oracle["decoder_fourth_residual_0_output"].shape)
                        observed = view(arena, acts, "decoder_fourth_residual_1_output",
                                        np.float32, oracle["decoder_fourth_residual_1_output"].shape)
                        local = replay_residual_one(previous)
                        reference_replay = replay_residual_one(
                            oracle["decoder_fourth_residual_0_output"])
                        local_error = float(np.max(np.abs(observed - local)))
                        reference_error = float(np.max(np.abs(
                            oracle["decoder_fourth_residual_1_output"] - reference_replay)))
                        print("QWEN3_TTS_CODEC_RESIDUAL_ONE_REPLAY " + json.dumps({
                            "same_input_local_max_abs": local_error,
                            "reference_replay_max_abs": reference_error,
                            "input_propagation_max_abs": float(np.max(np.abs(
                                local - reference_replay))),
                            "connected_input_max_abs": float(np.max(np.abs(
                                previous - oracle["decoder_fourth_residual_0_output"]))),
                        }, sort_keys=True))
                        self.assertLessEqual(local_error, 2e-5)
                        self.assertLessEqual(reference_error, 1e-5)
                        self.assertEqual(checkpoint_failures, [])
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertLessEqual(float(np.max(np.abs(actual))), 1.0)
                    self.assertEqual(actual.size, frames * 1920)
                    destination = os.environ.get("CKE_QWEN3_TTS_CODEC_WAV")
                    if destination and frames == 46:
                        # Container writing is host I/O; generated C owns all
                        # neural operations and the bounded FP32 PCM output.
                        pcm16 = np.rint(np.clip(actual[0], -1, 1) * 32767).astype("<i2")
                        with wave.open(destination, "wb") as wav:
                            wav.setnchannels(1)
                            wav.setsampwidth(2)
                            wav.setframerate(24000)
                            wav.writeframes(pcm16.tobytes())
                        print("QWEN3_TTS_CODEC_WAV " + json.dumps({
                            "path": destination,
                            "sha256": hashlib.sha256(Path(destination).read_bytes()).hexdigest(),
                            "scope": "reference-code-fed diagnostic WAV",
                        }, sort_keys=True))
                    # Connected PCM parity remains a separate verdict from
                    # successful execution and WAV serialization.
                    self.assertLessEqual(metric["max_abs"], 3e-6)
            if 46 in generated_waveforms:
                # The capacity graph may publish only a shorter valid prefix
                # when future codec frames cannot influence earlier samples.
                np.testing.assert_array_equal(
                    generated_waveforms[8], generated_waveforms[46][:, :8 * 1920])


if __name__ == "__main__":
    unittest.main()
