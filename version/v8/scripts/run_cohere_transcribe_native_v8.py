#!/usr/bin/env python3
"""Run Cohere Transcribe with CKE kernels and a thin Python session driver.

This is the native bring-up path: GGUF supplies immutable model tensors and
tokenizer metadata, while all dense, normalization, activation, convolution,
attention, residual, and argmax compute is dispatched to CKE shared libraries.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
F32P = ctypes.POINTER(ctypes.c_float)


def _load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parakeet = _load_sibling("cke_parakeet_native", "run_parakeet_native_v8.py")
converter = _load_sibling("cke_gguf_converter", "convert_gguf_to_bump_v8.py")
f32 = parakeet.f32
ptr = parakeet.ptr


class GGUFWeights:
    """Read F16/F32 Cohere tensors directly from a validated GGUF artifact."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        with self.path.open("rb") as handle:
            reader = converter.GGUFReader(handle)
            if reader._read_exact(4) != b"GGUF":
                raise ValueError(f"{self.path}: invalid GGUF magic")
            version = reader.u32()
            if version < 2:
                raise ValueError(f"{self.path}: GGUF v{version} is unsupported")
            tensor_count, metadata_count = reader.u64(), reader.u64()
            self.metadata: dict[str, object] = {}
            for _ in range(metadata_count):
                key = reader.key_str()
                self.metadata[key] = converter._gguf_read_value(reader, reader.u32())
            self.tensors: dict[str, tuple[tuple[int, ...], int, int]] = {}
            for _ in range(tensor_count):
                name = reader.key_str()
                dimensions = tuple(int(reader.u64()) for _ in range(reader.u32()))
                ggml_type, offset = reader.u32(), reader.u64()
                if name in self.tensors:
                    raise ValueError(f"{self.path}: duplicate tensor {name}")
                self.tensors[name] = (dimensions, ggml_type, offset)
            alignment = int(self.metadata.get("general.alignment", 32))
            self.data_start = converter.align_up(reader.tell(), alignment)
        if self.metadata.get("general.architecture") != "cohere-transcribe":
            raise ValueError(f"{self.path}: expected cohere-transcribe GGUF")
        report = converter.gguf_ck_tensor_inventory_report(
            "cohere-transcribe",
            self.metadata,
            {name: None for name in self.tensors},
        )
        if not report or report.get("status") != "complete":
            raise ValueError(f"{self.path}: incomplete Cohere tensor inventory: {report}")
        self.cache: dict[str, np.ndarray] = {}

    def get(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        dimensions, ggml_type, offset = self.tensors[name]
        if ggml_type == converter.GGML_TYPE_F32:
            dtype = np.dtype("<f4")
        elif ggml_type == converter.GGML_TYPE_F16:
            dtype = np.dtype("<f2")
        else:
            raise ValueError(
                f"{name}: native correctness path requires F16/F32, got "
                f"{converter.ggml_type_name(ggml_type)}"
            )
        count = math.prod(dimensions)
        mapped = np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=self.data_start + offset,
            shape=(count,),
        )
        conventional_shape = tuple(reversed(dimensions))
        value = np.ascontiguousarray(mapped.astype(np.float32).reshape(conventional_shape))
        self.cache[name] = value
        return value

    def token_id(self, text: str) -> int:
        tokens = self.metadata.get("tokenizer.ggml.tokens")
        if not isinstance(tokens, list):
            raise ValueError("GGUF has no tokenizer.ggml.tokens")
        if not hasattr(self, "_token_ids"):
            self._token_ids = {str(token): index for index, token in enumerate(tokens)}
        return int(self._token_ids.get(text, -1))

    def decode(self, token_ids: list[int]) -> str:
        tokens = self.metadata.get("tokenizer.ggml.tokens")
        assert isinstance(tokens, list)
        pieces = [str(tokens[token]) for token in token_ids]
        return "".join(pieces).replace("▁", " ").strip()


class BumpWeights:
    """Read the Cohere mixed-FP16/FP32 BUMP correctness bundle."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.path = self.directory / "weights.bump"
        with self.path.open("rb") as handle:
            if handle.read(8) != b"BUMPWGT5":
                raise ValueError(f"{self.path}: expected BUMPWGT5")
        manifest = json.loads((self.directory / "weights_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("source_tensor_coverage", {}).get("pass"):
            raise ValueError("BUMP manifest does not account for every source tensor")
        entries = manifest.get("entries")
        if not isinstance(entries, list) or len(entries) != 2104:
            raise ValueError("Cohere BUMP manifest must contain exactly 2,104 tensors")
        self.entries = {str(entry["name"]): entry for entry in entries}
        self.tensors = self.entries
        self.metadata = json.loads((self.directory / "config.json").read_text(encoding="utf-8"))
        self.cache: dict[str, np.ndarray] = {}

    def get(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        entry = self.entries[name]
        dtype_name = str(entry["dtype"])
        if dtype_name not in {"fp16", "fp32"}:
            raise ValueError(f"{name}: unsupported Cohere BUMP dtype {dtype_name}")
        storage = np.dtype("<f2" if dtype_name == "fp16" else "<f4")
        shape = tuple(int(value) for value in entry["shape"])
        mapped = np.memmap(
            self.path,
            mode="r",
            dtype=storage,
            offset=int(entry["file_offset"]),
            shape=shape,
        )
        value = np.ascontiguousarray(mapped.astype(np.float32))
        self.cache[name] = value
        return value

    token_id = GGUFWeights.token_id
    decode = GGUFWeights.decode


class CohereKernels(parakeet.Kernels):
    def __init__(self, engine: Path, audio: Path):
        super().__init__(engine, audio)
        signature = [
            F32P, F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
        self.engine.attention_forward_query_key_head_major_f32.argtypes = signature
        self.engine.attention_forward_query_key_head_major_f32.restype = ctypes.c_int
        self.engine.attention_forward_query_key_head_major_f32_decode_heads.argtypes = signature
        self.engine.attention_forward_query_key_head_major_f32_decode_heads.restype = ctypes.c_int

    def attention(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray, *, decode: bool,
    ) -> np.ndarray:
        query, key, value = f32(query), f32(key), f32(value)
        heads, query_tokens, head_dim = query.shape
        if key.shape != value.shape or key.shape[0] != heads or key.shape[2] != head_dim:
            raise ValueError("attention shape mismatch")
        key_tokens = key.shape[1]
        output = np.empty_like(query)
        scratch = np.empty(
            (heads, key_tokens) if decode else (query_tokens, key_tokens),
            dtype=np.float32,
        )
        function = (
            self.engine.attention_forward_query_key_head_major_f32_decode_heads
            if decode else self.engine.attention_forward_query_key_head_major_f32
        )
        status = function(
            ptr(query), ptr(key), ptr(value), ptr(output), ptr(scratch),
            heads, query_tokens, key_tokens, head_dim,
            ctypes.c_float(head_dim ** -0.5),
        )
        if status != 0:
            raise RuntimeError(f"attention failed with status {status}")
        return output


class CohereSession:
    def __init__(self, model: Path, kernels: CohereKernels):
        self.weights = BumpWeights(model) if model.is_dir() else GGUFWeights(model)
        self.k = kernels
        self.encoder_layers = int(self.weights.metadata["cohere_transcribe.encoder.n_layers"])
        self.encoder_width = int(self.weights.metadata["cohere_transcribe.encoder.d_model"])
        self.encoder_heads = int(self.weights.metadata["cohere_transcribe.encoder.n_heads"])
        self.decoder_layers = int(self.weights.metadata["cohere_transcribe.decoder.n_layers"])
        self.decoder_width = int(self.weights.metadata["cohere_transcribe.decoder.d_model"])
        self.decoder_heads = int(self.weights.metadata["cohere_transcribe.decoder.n_heads"])
        self.max_context = int(self.weights.metadata["cohere_transcribe.decoder.max_ctx"])

    def w(self, name: str) -> np.ndarray:
        return self.weights.get(name)

    def _conv2d(self, value: np.ndarray, prefix: str, groups: int, stride: int, padding: int) -> np.ndarray:
        weight, bias = self.w(prefix + ".weight"), self.w(prefix + ".bias")
        output_channels, input_per_group, kh, kw = weight.shape
        input_channels, height, width = value.shape
        if input_per_group * groups != input_channels:
            raise ValueError(f"{prefix}: grouped convolution mismatch")
        oh = (height + 2 * padding - kh) // stride + 1
        ow = (width + 2 * padding - kw) // stride + 1
        output = np.empty((output_channels, oh, ow), np.float32)
        status = self.k.audio.audio_conv2d_whc_grouped_f32(
            ptr(f32(value)), ptr(weight), ptr(bias), ptr(output), width, height,
            input_channels, output_channels, kw, kh, stride, stride, padding,
            padding, groups, ow, oh,
        )
        if status != 0:
            raise RuntimeError(f"{prefix}: Conv2D failed with status {status}")
        return output

    def _conv1d(self, value: np.ndarray, prefix: str, groups: int, padding: int) -> np.ndarray:
        weight = self.w(prefix + ".weight")
        if weight.ndim == 3 and weight.shape[-1] == 1:
            weight = np.ascontiguousarray(weight[:, :, 0])
        if weight.ndim == 2:
            weight = weight[:, :, None]
        bias = self.w(prefix + ".bias")
        output_channels, input_per_group, kernel = weight.shape
        input_channels, frames = value.shape
        output_frames = frames + 2 * padding - kernel + 1
        output = np.empty((output_channels, output_frames), np.float32)
        status = self.k.audio.audio_conv1d_channel_major_grouped_f32(
            ptr(f32(value)), ptr(f32(weight)), ptr(bias), ptr(output), input_channels,
            output_channels, frames, kernel, 1, padding, groups, output_frames,
        )
        if status != 0:
            raise RuntimeError(f"{prefix}: Conv1D failed with status {status}")
        return output

    def frontend(self, wav_path: Path) -> np.ndarray:
        wav = np.frombuffer(wav_path.read_bytes(), dtype=np.uint8).copy()
        info = parakeet.WavInfo()
        if self.k.audio.audio_wav_parse_memory(
            wav.ctypes.data_as(parakeet.U8P), wav.nbytes, ctypes.byref(info),
        ) != 0:
            raise RuntimeError("WAV parse failed")
        if info.bits_per_sample != 16 or info.sample_rate != 16000:
            raise ValueError("current Cohere correctness path requires 16 kHz PCM16 WAV")
        samples = np.empty(info.frames, np.float32)
        decoded = self.k.audio.audio_wav_decode_pcm16_mono_f32(
            wav.ctypes.data_as(parakeet.U8P), wav.nbytes, ctypes.byref(info), ptr(samples), samples.size,
        )
        if decoded != info.frames:
            raise RuntimeError("WAV decode was incomplete")
        emphasized = np.empty_like(samples)
        if self.k.audio.audio_preemphasis_f32(ptr(samples), ptr(emphasized), samples.size, ctypes.c_float(0.97)) != 0:
            raise RuntimeError("pre-emphasis failed")
        n_fft, hop, frames, bins = 512, 160, samples.size // 160 + 1, 257
        unused = np.empty(n_fft, np.float32)
        cosine = np.empty((bins, n_fft), np.float32)
        sine = np.empty_like(cosine)
        if self.k.audio.audio_stft_precompute_tables_f32(n_fft, ptr(unused), ptr(cosine), ptr(sine)) != 0:
            raise RuntimeError("STFT table preparation failed")
        power = np.empty((frames, bins), np.float32)
        window = self.w("fe.window")
        if self.k.audio.audio_stft_power_centered_window_f32(
            ptr(emphasized), emphasized.size, ptr(window), window.size,
            ptr(cosine), ptr(sine), n_fft, hop, 0, ptr(power), frames,
        ) != 0:
            raise RuntimeError("centered STFT failed")
        log_mel = np.empty((frames, 128), np.float32)
        if self.k.audio.audio_log_mel_time_major_f32(
            ptr(power), ptr(np.ascontiguousarray(self.w("fe.mel_fb").reshape(128, bins))), ptr(log_mel), frames, bins, 128,
            ctypes.c_float(2.0 ** -24),
        ) != 0:
            raise RuntimeError("log-mel projection failed")
        normalized = np.empty_like(log_mel)
        if self.k.audio.audio_feature_normalize_per_feature_f32(
            ptr(log_mel), ptr(normalized), 128, frames, ctypes.c_float(1.0e-5),
        ) != 0:
            raise RuntimeError("feature normalization failed")
        return normalized

    def subsampling(self, features: np.ndarray) -> np.ndarray:
        hidden = f32(features)[None, :, :]
        for index, groups in ((0, 1), (2, 256), (3, 1), (5, 256), (6, 1)):
            stride = 2 if index in {0, 2, 5} else 1
            hidden = self._conv2d(hidden, f"enc.pre.conv.{index}", groups, stride, 1 if stride == 2 else 0)
            if index in {0, 3, 6}:
                self.k.relu_inplace(hidden)
        tokens = f32(hidden.transpose(1, 0, 2).reshape(hidden.shape[1], -1))
        return self.k.gemm(tokens, self.w("enc.pre.out.weight"), self.w("enc.pre.out.bias"))

    def _norm(self, value: np.ndarray, prefix: str) -> np.ndarray:
        return self.k.layer_norm(value, self.w(prefix + ".weight"), self.w(prefix + ".bias"), 1.0e-5)

    def _ff(self, value: np.ndarray, prefix: str, activation: str) -> np.ndarray:
        separator = "." if prefix + ".up.weight" in self.weights.tensors else "_"
        hidden = self.k.gemm(value, self.w(prefix + separator + "up.weight"), self.w(prefix + separator + "up.bias"))
        if activation == "silu":
            hidden = self.k.silu(hidden)
        else:
            self.k.relu_inplace(hidden)
        return self.k.gemm(hidden, self.w(prefix + separator + "down.weight"), self.w(prefix + separator + "down.bias"))

    def _encoder_attention(self, value: np.ndarray, positions: np.ndarray, prefix: str) -> np.ndarray:
        frames, width = value.shape
        heads, head_dim = self.encoder_heads, width // self.encoder_heads
        query = self.k.gemm(value, self.w(prefix + ".q.weight"), self.w(prefix + ".q.bias"))
        key = self.k.gemm(value, self.w(prefix + ".k.weight"), self.w(prefix + ".k.bias"))
        val = self.k.gemm(value, self.w(prefix + ".v.weight"), self.w(prefix + ".v.bias"))
        relative = self.k.gemm(positions, self.w(prefix + ".pos.weight"))
        output = np.empty_like(value)
        scratch = np.empty((heads, frames), np.float32)
        status = self.k.audio.audio_conformer_relative_attention_f32(
            ptr(query), ptr(key), ptr(val), ptr(relative),
            ptr(self.w(prefix + ".pos_bias_u")), ptr(self.w(prefix + ".pos_bias_v")),
            ptr(output), frames, heads, head_dim, ctypes.c_float(head_dim ** -0.5),
            ptr(scratch), scratch.nbytes,
        )
        if status != 0:
            raise RuntimeError(f"{prefix}: relative attention failed with status {status}")
        return self.k.gemm(output, self.w(prefix + ".out.weight"), self.w(prefix + ".out.bias"))

    def encoder_block(self, value: np.ndarray, positions: np.ndarray, layer: int) -> np.ndarray:
        prefix = f"enc.blk.{layer}"
        branch = self._ff(self._norm(value, prefix + ".ff1.norm"), prefix + ".ff1", "silu")
        value = self.k.residual_add(value, branch, 0.5)
        branch = self._encoder_attention(self._norm(value, prefix + ".attn.norm"), positions, prefix + ".attn")
        value = self.k.residual_add(value, branch)
        normalized = self._norm(value, prefix + ".conv.norm")
        channel = f32(normalized.T)
        hidden = self._conv1d(channel, prefix + ".conv.pw1", 1, 0)
        gated = np.empty((self.encoder_width, hidden.shape[1]), np.float32)
        if self.k.audio.audio_glu_split_channel_major_f32(
            ptr(hidden), ptr(gated), gated.shape[0], gated.shape[1],
        ) != 0:
            raise RuntimeError("encoder GLU failed")
        hidden = self._conv1d(gated, prefix + ".conv.dw", self.encoder_width, 4)
        batch = np.empty_like(hidden)
        if self.k.audio.audio_batch_norm_inference_channel_major_f32(
            ptr(hidden), ptr(self.w(prefix + ".conv.bn.mean")), ptr(self.w(prefix + ".conv.bn.var")),
            ptr(self.w(prefix + ".conv.bn.weight")), ptr(self.w(prefix + ".conv.bn.bias")),
            ptr(batch), self.encoder_width, batch.shape[1], ctypes.c_float(1.0e-5),
        ) != 0:
            raise RuntimeError("encoder BatchNorm failed")
        hidden = self.k.silu(f32(batch.T)).T.copy()
        branch = f32(self._conv1d(hidden, prefix + ".conv.pw2", 1, 0).T)
        value = self.k.residual_add(value, branch)
        branch = self._ff(self._norm(value, prefix + ".ff2.norm"), prefix + ".ff2", "silu")
        value = self.k.residual_add(value, branch, 0.5)
        return self._norm(value, prefix + ".out_norm")

    def encode(
        self,
        features: np.ndarray,
        stop_after_layer: int | None = None,
        layer_references: dict[int, np.ndarray] | None = None,
        xray: list[dict[str, object]] | None = None,
    ) -> np.ndarray:
        value = self.subsampling(features)
        positions = np.empty((2 * value.shape[0] - 1, self.encoder_width), np.float32)
        if self.k.audio.audio_relative_sinusoidal_position_f32(
            ptr(positions), value.shape[0], self.encoder_width,
        ) != 0:
            raise RuntimeError("relative position generation failed")
        for layer in range(self.encoder_layers):
            started = time.perf_counter()
            value = self.encoder_block(value, positions, layer)
            if layer_references is not None and layer in layer_references and xray is not None:
                xray.append({"layer": layer, **compare(value, layer_references[layer])})
            print(f"encoder layer {layer:02d}: {time.perf_counter() - started:.3f}s", file=sys.stderr, flush=True)
            if layer == stop_after_layer:
                return value
        return self.k.gemm(value, self.w("enc.proj.weight"), self.w("enc.proj.bias"))

    @staticmethod
    def _heads(value: np.ndarray, heads: int) -> np.ndarray:
        return np.ascontiguousarray(value.reshape(value.shape[0], heads, -1).transpose(1, 0, 2))

    @staticmethod
    def _tokens(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(value.transpose(1, 0, 2).reshape(value.shape[1], -1))

    @staticmethod
    def _f16_cache_round(value: np.ndarray) -> np.ndarray:
        """Match Cohere's persistent F16 self/cross-attention cache contract."""
        return np.ascontiguousarray(value.astype(np.float16).astype(np.float32))

    def _prompt(self, language: str) -> list[int]:
        names = [
            "▁", "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
            f"<|{language}|>", f"<|{language}|>", "<|pnc|>", "<|noitn|>",
            "<|notimestamp|>", "<|nodiarize|>",
        ]
        ids = [self.weights.token_id(name) for name in names]
        if any(token < 0 for token in ids):
            raise ValueError(f"tokenizer lacks required prompt token: {dict(zip(names, ids))}")
        return ids

    def decode(self, encoder: np.ndarray, language: str, max_new_tokens: int) -> tuple[list[int], str]:
        cross: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in range(self.decoder_layers):
            prefix = f"dec.blk.{layer}"
            key = self.k.gemm(encoder, self.w(prefix + ".cross_k.weight"), self.w(prefix + ".cross_k.bias"))
            val = self.k.gemm(encoder, self.w(prefix + ".cross_v.weight"), self.w(prefix + ".cross_v.bias"))
            cross.append((
                self._f16_cache_round(self._heads(key, self.decoder_heads)),
                self._f16_cache_round(self._heads(val, self.decoder_heads)),
            ))
        self_keys = [np.empty((self.decoder_heads, 0, self.decoder_width // self.decoder_heads), np.float32) for _ in range(self.decoder_layers)]
        self_values = [value.copy() for value in self_keys]
        generated: list[int] = []
        sequence = self._prompt(language)
        if len(sequence) + max_new_tokens > self.max_context:
            raise ValueError(
                f"prompt plus max_new_tokens exceeds decoder context: "
                f"{len(sequence)} + {max_new_tokens} > {self.max_context}"
            )
        eos = self.weights.token_id("<|endoftext|>")
        if eos < 0:
            raise ValueError("tokenizer lacks <|endoftext|>")
        for position, token in enumerate(sequence + [0] * max_new_tokens):
            if position >= len(sequence):
                token = generated[-1]
            value = self.w("dec.emb.weight")[token] + self.w("dec.pos.weight")[position]
            value = self._norm(value[None], "dec.emb_ln")
            for layer in range(self.decoder_layers):
                prefix = f"dec.blk.{layer}"
                residual = value
                normalized = self._norm(value, prefix + ".attn_ln")
                query = self.k.gemm(normalized, self.w(prefix + ".attn_q.weight"), self.w(prefix + ".attn_q.bias"))
                key = self.k.gemm(normalized, self.w(prefix + ".attn_k.weight"), self.w(prefix + ".attn_k.bias"))
                val = self.k.gemm(normalized, self.w(prefix + ".attn_v.weight"), self.w(prefix + ".attn_v.bias"))
                query_h, key_h, val_h = (self._heads(item, self.decoder_heads) for item in (query, key, val))
                self_keys[layer] = np.concatenate((self_keys[layer], self._f16_cache_round(key_h)), axis=1)
                self_values[layer] = np.concatenate((self_values[layer], self._f16_cache_round(val_h)), axis=1)
                attended = self._tokens(self.k.attention(query_h, self_keys[layer], self_values[layer], decode=True))
                value = self.k.residual_add(residual, self.k.gemm(attended, self.w(prefix + ".attn_o.weight"), self.w(prefix + ".attn_o.bias")))
                residual = value
                normalized = self._norm(value, prefix + ".cross_ln")
                query = self.k.gemm(normalized, self.w(prefix + ".cross_q.weight"), self.w(prefix + ".cross_q.bias"))
                attended = self._tokens(self.k.attention(self._heads(query, self.decoder_heads), *cross[layer], decode=True))
                value = self.k.residual_add(residual, self.k.gemm(attended, self.w(prefix + ".cross_o.weight"), self.w(prefix + ".cross_o.bias")))
                branch = self._ff(self._norm(value, prefix + ".ffn_ln"), prefix + ".ffn", "relu")
                value = self.k.residual_add(value, branch)
            logits = self.k.gemm(self._norm(value, "dec.out_ln"), self.w("dec.head.weight"), self.w("dec.head.bias"))[0]
            if position + 1 < len(sequence):
                continue
            selected = self.k.argmax_first(logits)
            if selected == eos:
                break
            generated.append(selected)
        else:
            raise RuntimeError("decoder did not emit EOS within max_new_tokens")
        return generated, self.weights.decode(generated)


def compare(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    if actual.shape != reference.shape:
        return {"shape_match": False, "actual_shape": list(actual.shape), "reference_shape": list(reference.shape)}
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "shape_match": True,
        "finite": bool(np.isfinite(actual).all()),
        "max_abs": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "bit_exact": bool(np.array_equal(actual.view(np.uint32), reference.view(np.uint32))),
    }


def run(args: argparse.Namespace, report: dict[str, object]) -> None:
    started = time.perf_counter()
    kernels = CohereKernels(args.engine, args.audio_lib)
    session = CohereSession(args.model, kernels)
    features = session.frontend(args.audio)
    report.update({
        "model": (
            {
                "format": "BUMPWGT5",
                "weights": parakeet.file_identity(args.model / "weights.bump"),
                "manifest": parakeet.file_identity(args.model / "weights_manifest.json"),
                "config": parakeet.file_identity(args.model / "config.json"),
            }
            if args.model.is_dir()
            else {"format": "GGUF", **parakeet.file_identity(args.model)}
        ),
        "input": parakeet.file_identity(args.audio),
        "runtime": kernels.provenance(),
        "thread_policy": kernels.thread_policy(),
        "frontend_shape": list(features.shape),
        "finite_frontend": bool(np.isfinite(features).all()),
    })
    references: dict[str, np.ndarray] = {}
    if args.reference_manifest:
        manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
        checkpoints = manifest.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError("reference manifest has no checkpoints")
        for checkpoint in checkpoints:
            references[str(checkpoint["checkpoint_id"])] = np.fromfile(
                checkpoint["tensor_path"], dtype="<f4",
            ).reshape(checkpoint["logical_shape"])
        required = {"audio.frontend.log_mel.output"} | {
            f"audio.encoder.layer.{layer}.output"
            for layer in range(session.encoder_layers)
        }
        missing = sorted(required - references.keys())
        if missing:
            raise ValueError(f"reference manifest is incomplete; missing={missing}")
        report["frontend_comparison"] = compare(
            features, references["audio.frontend.log_mel.output"]
        )
    layer_references = {
        layer: references[f"audio.encoder.layer.{layer}.output"]
        for layer in range(session.encoder_layers)
        if f"audio.encoder.layer.{layer}.output" in references
    }
    encoder_xray: list[dict[str, object]] = []
    encoded = session.encode(features, args.stop_after_layer, layer_references, encoder_xray)
    if encoder_xray:
        report["encoder_xray"] = encoder_xray
    report["encoder_shape"] = list(encoded.shape)
    report["finite_encoder"] = bool(np.isfinite(encoded).all())
    if args.dump_encoder:
        args.dump_encoder.parent.mkdir(parents=True, exist_ok=True)
        encoded.astype("<f4", copy=False).tofile(args.dump_encoder)
    checkpoint_id = (
        f"audio.encoder.layer.{args.stop_after_layer}.output"
        if args.stop_after_layer is not None
        else "audio.decoder.cross_attention.context"
    )
    if checkpoint_id in references:
        report["encoder_comparison"] = compare(encoded, references[checkpoint_id])
    if args.decode and args.stop_after_layer is None:
        token_ids, transcript = session.decode(encoded, args.language, args.max_new_tokens)
        eos_id = session.weights.token_id("<|endoftext|>")
        report["decode"] = {
            "token_ids": token_ids,
            "emitted_token_ids": token_ids + [eos_id],
            "transcript": transcript,
            "eos": True,
        }
        if args.reference_summary:
            reference_summary = json.loads(args.reference_summary.read_text(encoding="utf-8"))
            reference_ids = [int(value) for value in reference_summary.get("generated_token_ids", [])]
            if not reference_ids:
                raise ValueError("reference summary has no generated_token_ids")
            report["trajectory_comparison"] = {
                "reference_token_ids": reference_ids,
                "exact": reference_ids == token_ids + [eos_id],
            }
    elif args.reference_summary:
        raise ValueError("--reference-summary requires --decode without --stop-after-layer")
    report["wall_seconds"] = time.perf_counter() - started
    report["peak_rss_kib"] = parakeet.resource.getrusage(parakeet.resource.RUSAGE_SELF).ru_maxrss
    comparisons: list[dict[str, object]] = list(encoder_xray)
    if args.reference_manifest:
        comparisons.append(report["frontend_comparison"])
        expected_xray = (
            args.stop_after_layer + 1
            if args.stop_after_layer is not None
            else session.encoder_layers
        )
        report["reference_coverage"] = {
            "expected_encoder_checkpoints": expected_xray,
            "compared_encoder_checkpoints": len(encoder_xray),
            "complete": len(encoder_xray) == expected_xray,
        }
    comparisons_ok = all(
        bool(item.get("shape_match"))
        and bool(item.get("finite"))
        and float(item.get("rmse", math.inf)) <= 1.0e-3
        for item in comparisons
    )
    coverage = report.get("reference_coverage")
    coverage_ok = coverage is None or bool(coverage.get("complete"))
    trajectory = report.get("trajectory_comparison")
    trajectory_ok = trajectory is None or bool(trajectory.get("exact"))
    report["status"] = "PASS" if (
        report["finite_frontend"]
        and report["finite_encoder"]
        and comparisons_ok
        and coverage_ok
        and trajectory_ok
    ) else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ROOT / "build/libckernel_engine.so")
    parser.add_argument("--audio-lib", type=Path, default=ROOT / "build/libckernel_audio.so")
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-encoder", type=Path)
    args = parser.parse_args()
    report: dict[str, object] = {
        "schema": "cke.v8.cohere_transcribe_native",
        "schema_version": 1,
        "status": "ERROR",
    }
    try:
        run(args, report)
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
