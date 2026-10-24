"""Generated first-codebook codec prefix with independent arithmetic checks."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import build_ir_v8
from build_qwen3_tts_codec_semantic_circuit import build_circuit
from build_qwen3_tts_codec_rvq_circuit import build_circuit as build_rvq_circuit
from build_qwen3_tts_codec_preconv_circuit import build_circuit as build_preconv_circuit
from build_qwen3_tts_codec_qkv_circuit import build_circuit as build_qkv_circuit
from build_qwen3_tts_codec_attention_circuit import build_circuit as build_attention_circuit
from build_qwen3_tts_codec_first_block_circuit import build_circuit as build_first_block_circuit
from build_qwen3_tts_codec_pretransformer_circuit import build_circuit as build_pretransformer_circuit
from build_qwen3_tts_codec_first_upsample_circuit import build_circuit as build_first_upsample_circuit
from build_qwen3_tts_codec_second_upsample_circuit import build_circuit as build_second_upsample_circuit
from build_qwen3_tts_codec_depthwise_circuit import build_circuit as build_depthwise_circuit
from build_qwen3_tts_codec_convnext_complete_circuit import build_circuit as build_convnext_complete_circuit

TABLE = "codec.semantic.effective_embedding"
PROJECTION = "codec.semantic.output_projection.weight"
ACOUSTIC_PROJECTION = "codec.acoustic.output_projection.weight"
MODEL = (ROOT / "tests/fixtures/tts/qwen3_tts_0b6_customvoice"
         / "qwen3_tts_customvoice_reference.npz")


def generated_entry(folder: Path, arrays: dict[str, np.ndarray], frames: int,
                    *, acoustic_codebooks: int = 0, pre_conv: bool = False,
                    pre_transformer_qkv: bool = False,
                    qkv_heads: int = 16, qkv_head_dim: int = 64,
                    pre_transformer_attention: bool = False,
                    sliding_window: int = 72,
                    pre_transformer_first_block: bool = False,
                    pre_transformer_complete: bool = False,
                    transformer_layers: int = 8,
                    first_decoder_upsample: bool = False,
                    first_decoder_depthwise: bool = False,
                    depthwise_kernel: int = 7,
                    first_convnext_complete: bool = False,
                    second_decoder_upsample: bool = False,
                    convnext_expansion: int = 4):
    table, projection = arrays[TABLE], arrays[PROJECTION]
    dimensions = {"frames": frames, "vocabulary": table.shape[0],
                  "codebook_width": table.shape[1],
                  "output_width": projection.shape[0]}
    if (pre_conv or pre_transformer_qkv or pre_transformer_attention or
            pre_transformer_first_block or pre_transformer_complete or
            first_decoder_upsample or first_decoder_depthwise or first_convnext_complete or second_decoder_upsample):
        dimensions["latent_width"] = arrays["codec.decoder.pre_conv.conv.weight"].shape[0]
    if (pre_transformer_qkv or pre_transformer_attention or
            pre_transformer_first_block or pre_transformer_complete or
            first_decoder_upsample or first_decoder_depthwise or first_convnext_complete or second_decoder_upsample):
        dimensions.update({
            "hidden_width": arrays[
                "codec.decoder.pre_transformer.input_proj.weight"].shape[0],
            "heads": qkv_heads, "head_dim": qkv_head_dim,
        })
        block_dimensions = {
            "acoustic_codebooks": acoustic_codebooks,
            "sliding_window": sliding_window,
            "intermediate_width": arrays[
                "codec.decoder.pre_transformer.layers.0.mlp.gate_proj.weight"].shape[0],
            **dimensions,
        } if (first_decoder_upsample or first_decoder_depthwise or first_convnext_complete or second_decoder_upsample or pre_transformer_complete or
              pre_transformer_first_block) else None
        if second_decoder_upsample:
            graph = build_second_upsample_circuit(
                layers=transformer_layers, depthwise_kernel=depthwise_kernel,
                expansion=convnext_expansion, **block_dimensions)
        elif first_convnext_complete:
            graph = build_convnext_complete_circuit(
                layers=transformer_layers, depthwise_kernel=depthwise_kernel,
                expansion=convnext_expansion, **block_dimensions)
        elif first_decoder_depthwise:
            graph = build_depthwise_circuit(
                layers=transformer_layers, depthwise_kernel=depthwise_kernel,
                **block_dimensions)
        elif first_decoder_upsample:
            graph = build_first_upsample_circuit(
                layers=transformer_layers, **block_dimensions)
        elif pre_transformer_complete:
            graph = build_pretransformer_circuit(
                layers=transformer_layers, **block_dimensions)
        elif pre_transformer_first_block:
            graph = build_first_block_circuit(**block_dimensions)
        elif pre_transformer_attention:
            graph = build_attention_circuit(
                acoustic_codebooks=acoustic_codebooks,
                sliding_window=sliding_window, **dimensions)
        else:
            graph = build_qkv_circuit(
                acoustic_codebooks=acoustic_codebooks, **dimensions)
    elif pre_conv:
        graph = build_preconv_circuit(acoustic_codebooks=acoustic_codebooks,
                                      **dimensions)
    else:
        graph = (build_rvq_circuit(acoustic_codebooks=acoustic_codebooks,
                                   **dimensions) if acoustic_codebooks else
                 build_circuit(**dimensions))
    if acoustic_codebooks and acoustic_codebooks != 15:
        # Built-in circuits merge canonical checkpoint declarations by name.
        # Give smaller synthetic compositions their own explicit identity.
        graph["name"] += f"_synthetic_{acoustic_codebooks}"
    circuit = folder / "circuit.json"
    circuit.write_text(json.dumps(graph))
    source = {"config": {
        "model": graph["name"], "arch": graph["name"],
        "num_layers": 1, "embed_dim": table.shape[1], "num_heads": 1,
        "num_kv_heads": 1, "head_dim": table.shape[1],
        "intermediate_size": table.shape[1] * 2,
        "context_length": frames, "max_seq_len": frames,
        "vocab_size": table.shape[0],
        "activation_buffer_dtypes": {"semantic_ids": "i32"}},
        "entries": [], "quant_summary": {}, "template": graph}
    offset = 4096
    for name in arrays:
        value = arrays[name]
        offset = (offset + 63) & ~63
        source["entries"].append({
            "name": name, "shape": list(value.shape), "dtype": "F32",
            "size": value.nbytes, "file_offset": offset,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()})
        offset += value.nbytes
    registry = build_ir_v8.load_kernel_registry()
    with contextlib.redirect_stdout(io.StringIO()):
        ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode="prefill")
        lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
        layout = build_ir_v8.generate_memory_layout(
            lower1, source, registry, mode="prefill", context_len=frames)
        lower2 = build_ir_v8.generate_ir_lower_2(
            lower1, layout, source, registry, mode="prefill")
        call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
    if call_ir["errors"]:
        raise AssertionError(call_ir["errors"])
    (folder / "layout.json").write_text(json.dumps(layout))
    (folder / "call.json").write_text(json.dumps(call_ir))
    c_path, so_path = folder / "generated.c", folder / "generated.so"
    subprocess.run([
        sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
        "--ir", str(folder / "call.json"),
        "--layout", str(folder / "layout.json"), "--output", str(c_path),
    ], check=True, capture_output=True, text=True)
    extra_sources = ([str(ROOT / "src/kernels/audio_zero_pad_checked.c"),
                      str(ROOT / "src/kernels/audio_conv1d_checked.c")]
                     if pre_conv or pre_transformer_qkv or pre_transformer_attention or
                     pre_transformer_first_block or pre_transformer_complete or
                     first_decoder_upsample or first_decoder_depthwise or first_convnext_complete or second_decoder_upsample else [])
    subprocess.run([
        "cc", "-std=gnu11", "-O2", "-w", "-fopenmp",
        "-shared", "-fPIC", "-ffunction-sections", "-fdata-sections",
        "-Wl,--gc-sections", "-Wl,--no-undefined", str(c_path),
        str(ROOT / "src/kernels/audio_text_embedding.c"),
        str(ROOT / "src/kernels/strided_unary_checked.c"),
        str(ROOT / "src/kernels/linear_rows_checked.c"),
        str(ROOT / "src/kernels/transformer_rows_checked.c"),
        *extra_sources,
        str(ROOT / "src/kernels/rope_kernels.c"),
        str(ROOT / "src/kernels/attention_kernels.c"),
        str(ROOT / "src/kernels/softmax_kernels.c"),
        str(ROOT / "src/kernels/attention_flash_true.c"),
        *([str(ROOT / "src/kernels/audio_conv_transpose_dense_checked.c")]
          if first_decoder_upsample or first_decoder_depthwise or first_convnext_complete or second_decoder_upsample else []),
        *([str(ROOT / "src/kernels/audio_conv1d_depthwise_checked.c")]
          if first_decoder_depthwise or first_convnext_complete or second_decoder_upsample else []),
        *([str(ROOT / "src/kernels/layernorm_rows_checked.c"),
           str(ROOT / "src/kernels/layernorm_kernels.c"),
           str(ROOT / "src/kernels/gelu_rows_erf_checked.c"),
           str(ROOT / "src/kernels/gelu_kernels.c")]
          if first_convnext_complete or second_decoder_upsample else []),
        str(ROOT / "src/ck_threadpool.c"),
        str(ROOT / "src/ckernel_strict.c"),
        "-I", str(ROOT / "include"), "-lm", "-ldl", "-lpthread",
        "-o", str(so_path),
    ], check=True, capture_output=True, text=True)
    symbol = ("ck_qwen3_tts_codec_second_upsample_bounded" if second_decoder_upsample else
              "ck_qwen3_tts_codec_convnext_complete_bounded" if first_convnext_complete else
              "ck_qwen3_tts_codec_depthwise_bounded" if first_decoder_depthwise else
              "ck_qwen3_tts_codec_first_upsample_bounded" if first_decoder_upsample else
              "ck_qwen3_tts_codec_pretransformer_bounded" if pre_transformer_complete else
              "ck_qwen3_tts_codec_first_block_bounded" if pre_transformer_first_block else
              "ck_qwen3_tts_codec_attention_bounded" if pre_transformer_attention else
              "ck_qwen3_tts_codec_qkv_bounded" if pre_transformer_qkv else
              "ck_qwen3_tts_codec_preconv_bounded" if pre_conv else
              "ck_qwen3_tts_codec_rvq_bounded" if acoustic_codebooks else
              "ck_qwen3_tts_codec_semantic_bounded")
    fn = getattr(ctypes.CDLL(str(so_path)), symbol)
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    weights = {e["name"]: e for e in layout["memory"]["weights"]["entries"]}
    acts = {e["name"]: e for e in layout["memory"]["activations"]["buffers"]}
    size = layout["memory"]["arena"]["total_size"]
    raw = (ctypes.c_uint8 * (size + 63))()
    arena = (ctypes.c_uint8 * size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)
    for name, entry in weights.items():
        value = arrays[name]
        arena[entry["abs_offset"]:entry["abs_offset"] + value.nbytes] = value.tobytes()
    return fn, arena, acts, call_ir, so_path


def view(arena, acts, name, dtype, shape):
    return np.ndarray(shape, dtype=dtype, buffer=arena,
                      offset=acts[name]["abs_offset"])


class Qwen3TTSCodecSemanticTest(unittest.TestCase):
    def test_pinned_checkpoint_identity(self):
        folder = MODEL.parent
        manifest = json.loads((folder / "qwen3_tts_codec_semantic.json").read_text())
        checkpoint = folder / "qwen3_tts_codec_semantic.npz"
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         manifest["npz_sha256"])
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                         manifest["source_codes_sha256"])
        self.assertEqual(manifest["model_revision"],
                         "85e237c12c027371202489a0ec509ded67b5e4b5")
        self.assertEqual(manifest["reference_revision"],
                         "022e286b98fbec7e1e916cb940cdf532cd9f488e")
        with np.load(MODEL) as codes, np.load(checkpoint) as values:
            np.testing.assert_array_equal(values["semantic_ids"],
                                          codes["codec_tokens"][:, 0])
            self.assertEqual(values["semantic_raw"].shape,
                             (manifest["frames"], 256))
            self.assertEqual(values["semantic_projected"].shape,
                             (manifest["frames"], 512))
            self.assertEqual(values["rvq_output"].shape,
                             (manifest["frames"], 512))
            self.assertTrue(np.isfinite(values["semantic_projected"]).all())

    def test_authored_circuit_is_current(self):
        path = ROOT / "version/v8/circuits/qwen3_tts_codec_semantic_bounded.json"
        self.assertEqual(json.loads(path.read_text()), build_circuit())
        with self.assertRaises(ValueError):
            build_circuit(frames=0)
        with self.assertRaises(ValueError):
            build_circuit(frames=4097)
        rvq_path = ROOT / "version/v8/circuits/qwen3_tts_codec_rvq_bounded.json"
        self.assertEqual(json.loads(rvq_path.read_text()), build_rvq_circuit())
        with self.assertRaises(ValueError):
            build_rvq_circuit(acoustic_codebooks=0)

    def test_generated_nonzero_rejection_and_recovery(self):
        rng = np.random.default_rng(20261023)
        arrays = {
            TABLE: rng.normal(size=(11, 4)).astype(np.float32),
            PROJECTION: (rng.normal(size=(7, 4)) * .1).astype(np.float32),
        }
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, call_ir, _ = generated_entry(Path(temp), arrays, frames)
                self.assertEqual([op["function"] for op in call_ir["operations"]], [
                    "audio_text_embedding_channel_major_f32",
                    "transpose_strided_f32_checked", "linear_rows_checked_f32"])
                ids = view(arena, acts, "semantic_ids", np.int32, (frames,))
                ids[:] = np.arange(frames) + 1
                output = view(arena, acts, "semantic_projected", np.float32,
                              (frames, 7))
                expected = (arrays[TABLE][ids].astype(np.float64)
                            @ arrays[PROJECTION].T.astype(np.float64)).astype(np.float32)
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_allclose(output, expected, atol=3e-7, rtol=3e-7)
                weight_entry = next(e for e in call_ir["memory"]["weights"]["entries"]
                                    if e["name"] == PROJECTION)
                weight = np.ndarray(arrays[PROJECTION].shape, dtype=np.float32,
                                    buffer=arena, offset=weight_entry["abs_offset"])
                original = float(weight[0, 0])
                weight[0, 0] = original + 1.0
                self.assertEqual(fn(arena, len(arena)), 0)
                self.assertGreater(float(np.max(np.abs(output - expected))), 1e-3)
                weight[0, 0] = original
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_allclose(output, expected, atol=3e-7, rtol=3e-7)
                ids[-1] = 11
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                self.assertTrue(np.all(output == -123))
                ids[-1] = frames
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_allclose(output, expected, atol=3e-7, rtol=3e-7)
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena) - 1), 0)
                self.assertTrue(np.all(output == -123))

    def test_generated_split_rvq_order_and_late_rejection(self):
        rng = np.random.default_rng(23)
        arrays = {TABLE: rng.normal(size=(11, 4)).astype(np.float32),
                  PROJECTION: (rng.normal(size=(7, 4)) * .1).astype(np.float32),
                  ACOUSTIC_PROJECTION: (rng.normal(size=(7, 4)) * .1).astype(np.float32)}
        for index in range(2):
            arrays[f"codec.acoustic.{index}.effective_embedding"] = (
                rng.normal(size=(11, 4)).astype(np.float32))
        for frames in (1, 5):
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                fn, arena, acts, calls, _ = generated_entry(
                    Path(temp), arrays, frames, acoustic_codebooks=2)
                self.assertEqual(len(calls["operations"]), 9)
                ids = np.arange(frames, dtype=np.int32) + 1
                view(arena, acts, "semantic_ids", np.int32, (frames,))[:] = ids
                for index in range(2):
                    view(arena, acts, f"acoustic_ids_{index}", np.int32,
                         (frames,))[:] = ids + index + 1
                semantic = (arrays[TABLE][ids].astype(np.float64)
                            @ arrays[PROJECTION].T.astype(np.float64)).astype(np.float32)
                acoustic = arrays["codec.acoustic.0.effective_embedding"][ids + 1].copy()
                acoustic += arrays["codec.acoustic.1.effective_embedding"][ids + 2]
                acoustic = (acoustic.astype(np.float64)
                            @ arrays[ACOUSTIC_PROJECTION].T.astype(np.float64)).astype(np.float32)
                expected = semantic + acoustic
                output = view(arena, acts, "rvq_output", np.float32, (frames, 7))
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_allclose(output, expected, atol=3e-7, rtol=3e-7)
                late_ids = view(arena, acts, "acoustic_ids_1", np.int32, (frames,))
                late_ids[-1] = 11
                output[:] = -123
                self.assertNotEqual(fn(arena, len(arena)), 0)
                self.assertTrue(np.all(output == -123))
                late_ids[-1] = ids[-1] + 2
                self.assertEqual(fn(arena, len(arena)), 0)
                np.testing.assert_allclose(output, expected, atol=3e-7, rtol=3e-7)

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned codec weights unavailable; model parity NOT_TESTED")
    def test_pinned_semantic_projection(self):
        from safetensors import safe_open
        from capture_qwen3_tts_customvoice_oracle import verify_model_dir
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            root = "decoder.quantizer.rvq_first."
            usage = source.get_tensor(root + "vq.layers.0._codebook.cluster_usage").float().numpy()
            summed = source.get_tensor(root + "vq.layers.0._codebook.embedding_sum").float().numpy()
            table = (summed / np.maximum(usage, 1e-5)[:, None]).astype(np.float32)
            weight = source.get_tensor(root + "output_proj.weight").float().numpy().reshape(512, 256)
        arrays = {TABLE: np.ascontiguousarray(table),
                  PROJECTION: np.ascontiguousarray(weight)}
        with np.load(MODEL) as fixture:
            codes = fixture["codec_tokens"][:, 0].astype(np.int32)
        # Independent PyTorch decoder checkpoint is retained separately from
        # this generated circuit; it is never fed into the native entry.
        oracle_path = MODEL.with_name("qwen3_tts_codec_semantic.npz")
        with np.load(oracle_path) as oracle:
            reference = oracle["semantic_projected"]
            np.testing.assert_array_equal(oracle["semantic_ids"], codes)
            for frames in (3, len(codes)):
                with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                    fn, arena, acts, _, so_path = generated_entry(Path(temp), arrays, frames)
                    view(arena, acts, "semantic_ids", np.int32, (frames,))[:] = codes[:frames]
                    self.assertEqual(fn(arena, len(arena)), 0)
                    raw = view(arena, acts, "semantic_token", np.float32,
                               (frames, 256))
                    raw_error = float(np.max(np.abs(raw - oracle["semantic_raw"][:frames])))
                    self.assertLessEqual(raw_error, 2e-6)
                    output = view(arena, acts, "semantic_projected", np.float32,
                                  (frames, 512))
                    error = float(np.max(np.abs(output - reference[:frames])))
                    self.assertLessEqual(error, 3e-5)
                    print("QWEN3_TTS_CODEC_SEMANTIC " + json.dumps({
                        "frames": frames, "raw_max_abs_error": raw_error,
                        "projected_max_abs_error": error,
                        "library_sha256": hashlib.sha256(so_path.read_bytes()).hexdigest(),
                        "scope": "first semantic codebook only; rest of codec NOT_TESTED",
                    }, sort_keys=True))

    @unittest.skipUnless(os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"),
                         "pinned codec weights unavailable; full RVQ parity NOT_TESTED")
    def test_pinned_full_split_rvq(self):
        from safetensors import safe_open
        from capture_qwen3_tts_customvoice_oracle import verify_model_dir
        model_dir = Path(os.environ["CKE_QWEN3_TTS_MODEL_DIR"])
        verify_model_dir(model_dir)
        arrays = {}
        with safe_open(model_dir / "speech_tokenizer/model.safetensors",
                       framework="pt", device="cpu") as source:
            for group, count in (("rvq_first", 1), ("rvq_rest", 15)):
                root = f"decoder.quantizer.{group}."
                for index in range(count):
                    layer = f"{root}vq.layers.{index}._codebook."
                    usage = source.get_tensor(layer + "cluster_usage").float().numpy()
                    summed = source.get_tensor(layer + "embedding_sum").float().numpy()
                    table = np.ascontiguousarray(
                        (summed / np.maximum(usage, 1e-5)[:, None]).astype(np.float32))
                    name = (TABLE if group == "rvq_first" else
                            f"codec.acoustic.{index}.effective_embedding")
                    arrays[name] = table
                projected = np.ascontiguousarray(
                    source.get_tensor(root + "output_proj.weight")
                    .float().numpy().reshape(512, 256))
                arrays[PROJECTION if group == "rvq_first" else
                       ACOUSTIC_PROJECTION] = projected
        with np.load(MODEL) as fixture, np.load(
                MODEL.with_name("qwen3_tts_codec_semantic.npz")) as oracle:
            codes = fixture["codec_tokens"].astype(np.int32)
            reference = oracle["rvq_output"]
            for frames in (3, len(codes)):
                with self.subTest(frames=frames), tempfile.TemporaryDirectory() as temp:
                    fn, arena, acts, calls, so_path = generated_entry(
                        Path(temp), arrays, frames, acoustic_codebooks=15)
                    self.assertEqual(len(calls["operations"]), 35)
                    view(arena, acts, "semantic_ids", np.int32,
                         (frames,))[:] = codes[:frames, 0]
                    for index in range(15):
                        view(arena, acts, f"acoustic_ids_{index}", np.int32,
                             (frames,))[:] = codes[:frames, index + 1]
                    self.assertEqual(fn(arena, len(arena)), 0)
                    output = view(arena, acts, "rvq_output", np.float32,
                                  (frames, 512))
                    difference = np.abs(output - reference[:frames])
                    error = float(np.max(difference))
                    self.assertLessEqual(error, 1e-4)
                    print("QWEN3_TTS_CODEC_RVQ " + json.dumps({
                        "frames": frames, "max_abs_error": error,
                        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                        "library_sha256": hashlib.sha256(so_path.read_bytes()).hexdigest(),
                        "scope": "all 16 codebooks through split projections; pre-conv and waveform NOT_TESTED",
                    }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
