#!/usr/bin/env python3
"""Run Parakeet TDT with CKE FP32 kernels and a thin Python session driver."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


F32P = ctypes.POINTER(ctypes.c_float)
U8P = ctypes.POINTER(ctypes.c_uint8)


class WavInfo(ctypes.Structure):
    _fields_ = [
        ("format_tag", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("bits_per_sample", ctypes.c_int),
        ("frames", ctypes.c_int),
        ("data_offset", ctypes.c_size_t),
        ("data_bytes", ctypes.c_size_t),
    ]


def f32(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32)


def ptr(value: np.ndarray) -> F32P:
    return value.ctypes.data_as(F32P)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Kernels:
    def __init__(self, engine: Path, audio: Path):
        self.engine = ctypes.CDLL(str(engine.resolve()))
        self.audio = ctypes.CDLL(str(audio.resolve()))
        self.engine.gemm_nt_f32_llama_production_parallel_dispatch.argtypes = [
            F32P, F32P, F32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.engine.layernorm_naive_serial_matched_precision.argtypes = [
            F32P, F32P, F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
        self.engine.recurrent_silu_forward.argtypes = [
            F32P, F32P, ctypes.c_int, ctypes.c_int,
        ]
        self.engine.relu_forward_inplace.argtypes = [F32P, ctypes.c_size_t]
        self.audio.audio_conv2d_whc_grouped_f32.argtypes = [
            F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.audio.audio_conv1d_channel_major_f32.argtypes = [
            F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.audio.audio_conv1d_channel_major_grouped_f32.argtypes = [
            F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.audio.audio_glu_split_channel_major_f32.argtypes = [
            F32P, F32P, ctypes.c_int, ctypes.c_int,
        ]
        self.audio.audio_batch_norm_inference_channel_major_f32.argtypes = [
            F32P, F32P, F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
        self.audio.audio_relative_sinusoidal_position_f32.argtypes = [
            F32P, ctypes.c_int, ctypes.c_int,
        ]
        self.audio.audio_conformer_relative_attention_f32.argtypes = [
            F32P, F32P, F32P, F32P, F32P, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            F32P, ctypes.c_size_t,
        ]
        self.audio.audio_lstm_step_f32.argtypes = [
            F32P, F32P, F32P, F32P, F32P, F32P, F32P, F32P, F32P,
            ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
        ]
        self.audio.audio_wav_parse_memory.argtypes = [
            U8P, ctypes.c_size_t, ctypes.POINTER(WavInfo),
        ]
        self.audio.audio_wav_decode_pcm16_mono_f32.argtypes = [
            U8P, ctypes.c_size_t, ctypes.POINTER(WavInfo), F32P, ctypes.c_int,
        ]
        self.audio.audio_preemphasis_f32.argtypes = [
            F32P, F32P, ctypes.c_int, ctypes.c_float,
        ]
        self.audio.audio_stft_precompute_tables_f32.argtypes = [
            ctypes.c_int, F32P, F32P, F32P,
        ]
        self.audio.audio_stft_power_centered_window_f32.argtypes = [
            F32P, ctypes.c_int, F32P, ctypes.c_int, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int,
        ]
        self.audio.audio_whisper_mel_filters_slaney_f32.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, F32P,
        ]
        self.audio.audio_log_mel_time_major_f32.argtypes = [
            F32P, F32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float,
        ]
        self.audio.audio_feature_normalize_per_feature_f32.argtypes = [
            F32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]

    def provenance(self) -> dict[str, object]:
        loaded = Path("/proc/self/maps").read_text(encoding="utf-8")
        result = {}
        for name, library in (("engine", self.engine), ("audio", self.audio)):
            path = Path(str(library._name)).resolve()
            result[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "present_in_process_maps": str(path) in loaded,
            }
        return result

    def gemm(self, value: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        value = f32(value)
        weight = f32(weight)
        rows, input_size = value.shape
        output_size = weight.shape[0]
        if weight.shape != (output_size, input_size):
            raise ValueError(f"GEMM shape mismatch: {value.shape} x {weight.shape}")
        result = np.empty((rows, output_size), dtype=np.float32)
        bias_pointer = ptr(f32(bias)) if bias is not None else None
        self.engine.gemm_nt_f32_llama_production_parallel_dispatch(
            ptr(value), ptr(weight), bias_pointer, ptr(result), rows, output_size, input_size,
        )
        return result

    def layer_norm(self, value: np.ndarray, weight: np.ndarray, bias: np.ndarray, epsilon: float) -> np.ndarray:
        value = f32(value)
        result = np.empty_like(value)
        self.engine.layernorm_naive_serial_matched_precision(
            ptr(value), ptr(f32(weight)), ptr(f32(bias)), ptr(result), None, None,
            value.shape[0], value.shape[1], epsilon,
        )
        return result

    def silu(self, value: np.ndarray) -> np.ndarray:
        value = f32(value)
        result = np.empty_like(value)
        self.engine.recurrent_silu_forward(ptr(value), ptr(result), value.shape[0], value.shape[1])
        return result

    def relu_inplace(self, value: np.ndarray) -> None:
        self.engine.relu_forward_inplace(ptr(value), value.size)


class SafetensorWeights:
    def __init__(self, model_dir: Path):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required only for source-checkpoint bring-up; "
                "convert the model to BUMP for normal inference"
            ) from exc
        self.handle = safe_open(model_dir / "model.safetensors", framework="numpy")
        self.handle.__enter__()

    def close(self) -> None:
        self.handle.__exit__(None, None, None)

    def keys(self) -> list[str]:
        return list(self.handle.keys())

    def get_tensor(self, name: str) -> np.ndarray:
        return self.handle.get_tensor(name)

    def provenance(self) -> dict[str, object]:
        path = Path(str(self.handle.filename())).resolve() if hasattr(self.handle, "filename") else None
        return {"format": "safetensors", "path": str(path) if path else None}


class BumpWeights:
    def __init__(self, model_dir: Path):
        manifest_path = model_dir / "weights_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{manifest_path}: missing nonempty entries")
        self.path = model_dir / "weights.bump"
        with self.path.open("rb") as handle:
            if handle.read(8) != b"BUMPWGT5":
                raise ValueError(f"{self.path}: expected BUMPWGT5 header")
        self.entries = {str(entry["name"]): entry for entry in entries}
        if len(self.entries) != len(entries):
            raise ValueError(f"{manifest_path}: duplicate tensor names")
        self.manifest_path = manifest_path
        self.cache: dict[str, np.ndarray] = {}

    def close(self) -> None:
        return None

    def keys(self) -> list[str]:
        return list(self.entries)

    def get_tensor(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        entry = self.entries[name]
        if entry.get("dtype") != "fp32":
            raise ValueError(f"{name}: Parakeet native FP32 path cannot load {entry.get('dtype')}")
        shape = tuple(int(value) for value in entry["shape"])
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
        if int(entry["size"]) != expected:
            raise ValueError(f"{name}: manifest size {entry['size']} != shape size {expected}")
        value = np.memmap(
            self.path, mode="r", dtype=np.float32,
            offset=int(entry["file_offset"]), shape=shape, order="C",
        )
        self.cache[name] = value
        return value

    def provenance(self) -> dict[str, object]:
        return {
            "format": "BUMPWGT5",
            "weights": {"path": str(self.path.resolve()), "bytes": self.path.stat().st_size, "sha256": sha256_file(self.path)},
            "manifest": {"path": str(self.manifest_path.resolve()), "bytes": self.manifest_path.stat().st_size, "sha256": sha256_file(self.manifest_path)},
            "mapped_inference_tensors": sum(
                1 for entry in self.entries.values()
                if entry.get("dtype") == "fp32" and not str(entry.get("name", "")).startswith("tokenizer.")
            ),
        }


class ParakeetSession:
    def __init__(self, model_dir: Path, kernels: Kernels):
        self.model_dir = model_dir
        self.k = kernels
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.encoder_config = self.config["encoder_config"]
        self.weights = (
            BumpWeights(model_dir) if (model_dir / "weights.bump").is_file()
            else SafetensorWeights(model_dir)
        )

    def close(self) -> None:
        self.weights.close()

    def w(self, name: str) -> np.ndarray:
        return f32(self.weights.get_tensor(name))

    def frontend(self, wav_path: Path) -> tuple[np.ndarray, int, dict[str, object]]:
        wav = np.frombuffer(wav_path.read_bytes(), dtype=np.uint8).copy()
        info = WavInfo()
        status = self.k.audio.audio_wav_parse_memory(
            wav.ctypes.data_as(U8P), wav.nbytes, ctypes.byref(info),
        )
        if status != 0:
            raise RuntimeError(f"WAV parse failed with status {status}")
        if info.sample_rate != 16000 or info.channels != 1 or info.bits_per_sample != 16:
            raise ValueError(
                "Parakeet frontend currently requires mono 16-bit PCM WAV at 16000 Hz; "
                f"got {info.channels} channel(s), {info.bits_per_sample}-bit, {info.sample_rate} Hz"
            )
        samples = np.empty(info.frames, dtype=np.float32)
        decoded = self.k.audio.audio_wav_decode_pcm16_mono_f32(
            wav.ctypes.data_as(U8P), wav.nbytes, ctypes.byref(info), ptr(samples), samples.size,
        )
        if decoded != info.frames:
            raise RuntimeError(f"WAV decode returned {decoded} frames, expected {info.frames}")
        emphasized = np.empty_like(samples)
        status = self.k.audio.audio_preemphasis_f32(
            ptr(samples), ptr(emphasized), samples.size, ctypes.c_float(0.97),
        )
        if status != 0:
            raise RuntimeError(f"preemphasis failed with status {status}")

        n_fft = 512
        window_length = 400
        hop_length = 160
        bins = n_fft // 2 + 1
        frames = samples.size // hop_length + 1
        live_frames = samples.size // hop_length
        unused_window = np.empty(n_fft, dtype=np.float32)
        cosine = np.empty((bins, n_fft), dtype=np.float32)
        sine = np.empty_like(cosine)
        status = self.k.audio.audio_stft_precompute_tables_f32(
            n_fft, ptr(unused_window), ptr(cosine), ptr(sine),
        )
        if status != 0:
            raise RuntimeError(f"STFT table preparation failed with status {status}")
        # torch.hann_window(400, periodic=False), matching the pinned feature extractor.
        window = f32(
            0.5 - 0.5 * np.cos(
                2.0 * np.pi * np.arange(window_length, dtype=np.float64) /
                (window_length - 1)
            )
        )
        power = np.empty((frames, bins), dtype=np.float32)
        status = self.k.audio.audio_stft_power_centered_window_f32(
            ptr(emphasized), emphasized.size, ptr(window), window_length,
            ptr(cosine), ptr(sine), n_fft, hop_length, 0, ptr(power), frames,
        )
        if status != 0:
            raise RuntimeError(f"centered STFT failed with status {status}")
        mel_filters = np.empty((128, bins), dtype=np.float32)
        status = self.k.audio.audio_whisper_mel_filters_slaney_f32(
            info.sample_rate, n_fft, 128, ptr(mel_filters),
        )
        if status != 0:
            raise RuntimeError(f"Slaney mel filter preparation failed with status {status}")
        log_mel = np.empty((frames, 128), dtype=np.float32)
        status = self.k.audio.audio_log_mel_time_major_f32(
            ptr(power), ptr(mel_filters), ptr(log_mel), frames, bins, 128,
            ctypes.c_float(2.0**-24),
        )
        if status != 0:
            raise RuntimeError(f"log-mel projection failed with status {status}")
        features = np.zeros_like(log_mel)
        status = self.k.audio.audio_feature_normalize_per_feature_f32(
            ptr(log_mel), ptr(features), 128, live_frames, ctypes.c_float(1.0e-5),
        )
        if status != 0:
            raise RuntimeError(f"feature normalization failed with status {status}")
        return features, live_frames, {
            "samples": samples,
            "preemphasis": emphasized,
            "power": power,
            "log_mel": log_mel,
        }

    def conv2d(self, value: np.ndarray, prefix: str, groups: int, stride: int, padding: int) -> np.ndarray:
        weight = self.w(prefix + ".weight")
        bias = self.w(prefix + ".bias")
        output_channels, input_per_group, kh, kw = weight.shape
        input_channels, height, width = value.shape
        if input_per_group * groups != input_channels:
            raise ValueError(f"{prefix}: invalid grouped convolution weights")
        oh = (height + 2 * padding - kh) // stride + 1
        ow = (width + 2 * padding - kw) // stride + 1
        result = np.empty((output_channels, oh, ow), dtype=np.float32)
        status = self.k.audio.audio_conv2d_whc_grouped_f32(
            ptr(f32(value)), ptr(weight), ptr(bias), ptr(result), width, height,
            input_channels, output_channels, kw, kh, stride, stride, padding,
            padding, groups, ow, oh,
        )
        if status != 0:
            raise RuntimeError(f"{prefix}: Conv2D failed with status {status}")
        return result

    def conv1d(self, value: np.ndarray, prefix: str, groups: int, padding: int) -> np.ndarray:
        weight = self.w(prefix + ".weight")
        bias_name = prefix + ".bias"
        bias = self.w(bias_name) if bias_name in self.weights.keys() else None
        output_channels, input_per_group, kernel = weight.shape
        input_channels, frames = value.shape
        if input_per_group * groups != input_channels:
            raise ValueError(f"{prefix}: invalid grouped convolution weights")
        output_frames = frames + 2 * padding - kernel + 1
        result = np.empty((output_channels, output_frames), dtype=np.float32)
        status = self.k.audio.audio_conv1d_channel_major_grouped_f32(
            ptr(f32(value)), ptr(weight), ptr(bias) if bias is not None else None,
            ptr(result), input_channels, output_channels, frames, kernel, 1,
            padding, groups, output_frames,
        )
        if status != 0:
            raise RuntimeError(f"{prefix}: Conv1D failed with status {status}")
        return result

    def subsampling(self, features: np.ndarray, live_frames: int) -> np.ndarray:
        hidden = f32(features)[None, :, :]
        length = live_frames
        stages = ((0, 1), (2, 256), (3, 1), (5, 256), (6, 1))
        for index, groups in stages:
            stride = 2 if index in {0, 2, 5} else 1
            hidden = self.conv2d(hidden, f"encoder.subsampling.layers.{index}", groups, stride, 1 if stride == 2 else 0)
            if stride == 2:
                length = (length + 2 - 3) // 2 + 1
            hidden[:, length:, :] = 0.0
            if index in {0, 3, 6}:
                self.k.relu_inplace(hidden)
        tokens = f32(hidden.transpose(1, 0, 2).reshape(hidden.shape[1], -1))
        return self.k.gemm(
            tokens, self.w("encoder.subsampling.linear.weight"),
            self.w("encoder.subsampling.linear.bias"),
        )

    def relative_positions(self, frames: int) -> np.ndarray:
        channels = int(self.encoder_config["hidden_size"])
        result = np.empty((2 * frames - 1, channels), dtype=np.float32)
        status = self.k.audio.audio_relative_sinusoidal_position_f32(ptr(result), frames, channels)
        if status != 0:
            raise RuntimeError(f"relative position provider failed with status {status}")
        return result

    def feed_forward(self, value: np.ndarray, prefix: str) -> np.ndarray:
        hidden = self.k.gemm(value, self.w(prefix + ".linear1.weight"))
        hidden = self.k.silu(hidden)
        return self.k.gemm(hidden, self.w(prefix + ".linear2.weight"))

    def layer_norm(self, value: np.ndarray, prefix: str) -> np.ndarray:
        return self.k.layer_norm(
            value, self.w(prefix + ".weight"), self.w(prefix + ".bias"), 1.0e-5,
        )

    def attention(self, value: np.ndarray, positions: np.ndarray, prefix: str) -> np.ndarray:
        frames, channels = value.shape
        heads = int(self.encoder_config["num_attention_heads"])
        head_dim = channels // heads
        query = self.k.gemm(value, self.w(prefix + ".q_proj.weight"))
        key = self.k.gemm(value, self.w(prefix + ".k_proj.weight"))
        val = self.k.gemm(value, self.w(prefix + ".v_proj.weight"))
        relative = self.k.gemm(positions, self.w(prefix + ".relative_k_proj.weight"))
        attended = np.empty_like(value)
        scores = np.empty(frames, dtype=np.float32)
        status = self.k.audio.audio_conformer_relative_attention_f32(
            ptr(query), ptr(key), ptr(val), ptr(relative),
            ptr(self.w(prefix + ".bias_u")), ptr(self.w(prefix + ".bias_v")),
            ptr(attended), frames, heads, head_dim, head_dim**-0.5,
            ptr(scores), scores.nbytes,
        )
        if status != 0:
            raise RuntimeError(f"relative attention failed with status {status}")
        return self.k.gemm(attended, self.w(prefix + ".o_proj.weight"))

    def convolution(self, value: np.ndarray, prefix: str) -> np.ndarray:
        channel_major = f32(value.T)
        hidden = self.conv1d(channel_major, prefix + ".pointwise_conv1", 1, 0)
        glu = np.empty((hidden.shape[0] // 2, hidden.shape[1]), dtype=np.float32)
        status = self.k.audio.audio_glu_split_channel_major_f32(
            ptr(hidden), ptr(glu), glu.shape[0], glu.shape[1],
        )
        if status != 0:
            raise RuntimeError(f"GLU failed with status {status}")
        hidden = self.conv1d(glu, prefix + ".depthwise_conv", glu.shape[0], 4)
        normalized = np.empty_like(hidden)
        status = self.k.audio.audio_batch_norm_inference_channel_major_f32(
            ptr(hidden), ptr(self.w(prefix + ".norm.running_mean")),
            ptr(self.w(prefix + ".norm.running_var")), ptr(self.w(prefix + ".norm.weight")),
            ptr(self.w(prefix + ".norm.bias")), ptr(normalized), normalized.shape[0],
            normalized.shape[1], 1.0e-5,
        )
        if status != 0:
            raise RuntimeError(f"BatchNorm failed with status {status}")
        activated = self.k.silu(f32(normalized.T)).T.copy()
        hidden = self.conv1d(f32(activated), prefix + ".pointwise_conv2", 1, 0)
        return f32(hidden.T)

    def encoder_block(self, value: np.ndarray, positions: np.ndarray, layer: int) -> np.ndarray:
        prefix = f"encoder.layers.{layer}"
        normalized = self.layer_norm(value, prefix + ".norm_feed_forward1")
        value = f32(value + np.float32(0.5) * self.feed_forward(normalized, prefix + ".feed_forward1"))
        normalized = self.layer_norm(value, prefix + ".norm_self_att")
        value = f32(value + self.attention(normalized, positions, prefix + ".self_attn"))
        normalized = self.layer_norm(value, prefix + ".norm_conv")
        value = f32(value + self.convolution(normalized, prefix + ".conv"))
        normalized = self.layer_norm(value, prefix + ".norm_feed_forward2")
        value = f32(value + np.float32(0.5) * self.feed_forward(normalized, prefix + ".feed_forward2"))
        return self.layer_norm(value, prefix + ".norm_out")

    def encode(self, features: np.ndarray, live_frames: int, stop_after_layer: int | None = None) -> np.ndarray:
        hidden = self.subsampling(features, live_frames)
        if stop_after_layer == -1:
            return hidden
        positions = self.relative_positions(hidden.shape[0])
        for layer in range(int(self.encoder_config["num_hidden_layers"])):
            started = time.perf_counter()
            hidden = self.encoder_block(hidden, positions, layer)
            print(
                f"layer {layer:02d}: {time.perf_counter() - started:.3f}s",
                file=sys.stderr, flush=True,
            )
            if stop_after_layer == layer:
                return hidden
        return self.k.gemm(
            hidden, self.w("encoder_projector.weight"), self.w("encoder_projector.bias"),
        )

    def decoder_step(
        self,
        token: int,
        hidden_state: np.ndarray,
        cell_state: np.ndarray,
    ) -> np.ndarray:
        hidden_size = int(self.config["decoder_hidden_size"])
        value = self.w("decoder.embedding.weight")[token].copy()
        gates = np.empty(4 * hidden_size, dtype=np.float32)
        for layer in range(int(self.config["num_decoder_layers"])):
            output = np.empty(hidden_size, dtype=np.float32)
            status = self.k.audio.audio_lstm_step_f32(
                ptr(value), ptr(self.w(f"decoder.lstm.weight_ih_l{layer}")),
                ptr(self.w(f"decoder.lstm.weight_hh_l{layer}")),
                ptr(self.w(f"decoder.lstm.bias_ih_l{layer}")),
                ptr(self.w(f"decoder.lstm.bias_hh_l{layer}")),
                ptr(hidden_state[layer]), ptr(cell_state[layer]), ptr(output),
                ptr(gates), gates.nbytes, hidden_size, hidden_size,
            )
            if status != 0:
                raise RuntimeError(f"LSTM layer {layer} failed with status {status}")
            value = output
        return self.k.gemm(
            value[None], self.w("decoder.decoder_projector.weight"),
            self.w("decoder.decoder_projector.bias"),
        )[0]

    def joint(self, encoder_row: np.ndarray, decoder_row: np.ndarray) -> np.ndarray:
        value = f32(encoder_row + decoder_row)[None]
        self.k.relu_inplace(value)
        return self.k.gemm(
            value, self.w("joint.head.weight"), self.w("joint.head.bias"),
        )[0]

    def decode(self, encoder: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_size = int(self.config["decoder_hidden_size"])
        blank = int(self.config["blank_token_id"])
        vocab_size = int(self.config["vocab_size"])
        durations_table = [int(value) for value in self.config["durations"]]
        hidden_state = np.zeros(
            (int(self.config["num_decoder_layers"]), hidden_size), dtype=np.float32,
        )
        cell_state = np.zeros_like(hidden_state)
        decoder_cache: np.ndarray | None = None
        token = blank
        frame = 0
        sequences = [blank]
        durations = [0]
        first_logits = None
        max_steps = int(self.config["max_symbols_per_step"]) * encoder.shape[0]
        while frame < encoder.shape[0] and len(sequences) <= max_steps:
            if decoder_cache is None or token != blank:
                decoder_cache = self.decoder_step(token, hidden_state, cell_state)
            logits = self.joint(encoder[frame], decoder_cache)
            if first_logits is None:
                first_logits = logits.copy()
            token = int(np.argmax(logits[:vocab_size]))
            duration_index = int(np.argmax(logits[vocab_size:]))
            duration = durations_table[duration_index]
            if token == blank and duration == 0:
                duration = 1
            sequences.append(token)
            durations.append(duration)
            frame += duration
        if frame < encoder.shape[0]:
            raise RuntimeError("TDT decoder exceeded its encoder-derived step bound")
        assert first_logits is not None
        return (
            np.asarray(sequences, dtype=np.int64),
            np.asarray(durations, dtype=np.int64),
            first_logits,
        )


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | bool]:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "finite": bool(np.isfinite(actual).all()),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--engine", default=Path("build/libckernel_engine.so"), type=Path)
    parser.add_argument("--audio-lib", default=Path("build/libckernel_audio.so"), type=Path)
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.fixture is None and args.audio is None:
        parser.error("at least one of --fixture or --audio is required")
    fixture_context = np.load(args.fixture, allow_pickle=False) if args.fixture else None
    fixture = fixture_context
    try:
        run_started = time.perf_counter()
        kernels = Kernels(args.engine, args.audio_lib)
        session = ParakeetSession(args.model, kernels)
        frontend_report = None
        frontend_seconds = None
        if args.audio is not None:
            frontend_started = time.perf_counter()
            features, live_frames, _frontend_intermediates = session.frontend(args.audio)
            frontend_seconds = time.perf_counter() - frontend_started
            if fixture is not None:
                frontend_report = compare(features, f32(fixture["frontend.input_features"][0]))
        else:
            assert fixture is not None
            features = f32(fixture["frontend.input_features"][0])
            live_frames = int(fixture["frontend.attention_mask"][0].sum())
        expected = None
        if fixture is not None:
            if args.stop_after_layer == -1:
                expected = fixture["encoder.subsampling"][0]
            elif args.stop_after_layer == 0:
                expected = fixture["encoder.layer.0"][0]
            elif args.stop_after_layer == 23:
                expected = fixture["encoder.layer.23"][0]
            elif args.stop_after_layer is None:
                expected = fixture["encoder.projected"][0]
        try:
            started = time.perf_counter()
            actual = session.encode(features, live_frames, args.stop_after_layer)
            encoder_seconds = time.perf_counter() - started
            report = {
                "schema_version": 1,
                "kind": "cke.parakeet.native_e2e",
                "status": "incomplete",
                "model": {
                    "id": "nvidia/parakeet-tdt-0.6b-v3",
                    "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
                    "dtype": "fp32",
                },
                "reference": {
                    "implementation": "huggingface/transformers",
                    "revision": "66799f45cea7513712580c6170cdaa4438df702a",
                },
                "host": {"platform": platform.platform(), "python": platform.python_version()},
                "runtime": kernels.provenance(),
                "weights": session.weights.provenance(),
                "inputs": {
                    "audio": {"path": str(args.audio.resolve()), "sha256": sha256_file(args.audio)} if args.audio else None,
                    "fixture": {"path": str(args.fixture.resolve()), "sha256": sha256_file(args.fixture)} if args.fixture else None,
                },
                "shape": list(actual.shape),
                "frontend_elapsed_seconds": frontend_seconds,
                "encoder_elapsed_seconds": encoder_seconds,
                "frontend": frontend_report,
                "comparison": compare(actual, expected) if expected is not None else None,
            }
            if args.decode:
                if args.stop_after_layer is not None:
                    raise ValueError("--decode requires the complete projected encoder")
                sequences, durations, first_logits = session.decode(actual)
                expected_sequences = fixture["decode.sequences"][0] if fixture is not None else None
                expected_durations = fixture["decode.durations"][0] if fixture is not None else None
                expected_logits = fixture["joint.first_logits"][0, 0, 0] if fixture is not None else None
                report["decode"] = {
                    "steps": int(sequences.size),
                    "sequences_equal": bool(np.array_equal(sequences, expected_sequences)) if expected_sequences is not None else None,
                    "durations_equal": bool(np.array_equal(durations, expected_durations)) if expected_durations is not None else None,
                    "first_divergent_sequence": next(
                        (i for i, (a, b) in enumerate(zip(sequences, expected_sequences)) if a != b),
                        None if sequences.size == expected_sequences.size else min(sequences.size, expected_sequences.size),
                    ) if expected_sequences is not None else None,
                    "first_divergent_duration": next(
                        (i for i, (a, b) in enumerate(zip(durations, expected_durations)) if a != b),
                        None if durations.size == expected_durations.size else min(durations.size, expected_durations.size),
                    ) if expected_durations is not None else None,
                    "first_logits": compare(first_logits, expected_logits) if expected_logits is not None else None,
                    "sequences": sequences.tolist(),
                    "durations": durations.tolist(),
                }
                try:
                    from tokenizers import Tokenizer
                    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
                    report["decode"]["transcript"] = tokenizer.decode(
                        sequences.tolist(), skip_special_tokens=True,
                    )
                except ImportError:
                    report["decode"]["transcript"] = None
            checks = {
                "runtime_libraries_loaded": all(
                    bool(item["present_in_process_maps"])
                    for item in report["runtime"].values()
                ),
                "finite_frontend": frontend_report is None or bool(frontend_report["finite"]),
                "finite_encoder": bool(report["comparison"]["finite"]) if report["comparison"] else bool(np.isfinite(actual).all()),
                "frontend_rmse_at_most_5e_4": frontend_report is None or float(frontend_report["rmse"]) <= 5.0e-4,
                "encoder_rmse_at_most_2e_5": report["comparison"] is None or float(report["comparison"]["rmse"]) <= 2.0e-5,
            }
            if args.decode:
                checks.update({
                    "finite_first_logits": bool(report["decode"]["first_logits"]["finite"]) if report["decode"]["first_logits"] else bool(np.isfinite(first_logits).all()),
                    "sequence_trajectory_exact": report["decode"]["sequences_equal"] is not False,
                    "duration_trajectory_exact": report["decode"]["durations_equal"] is not False,
                    "nonempty_transcript": bool(report["decode"]["transcript"]),
                })
            report["checks"] = checks
            report["status"] = "pass" if all(checks.values()) else "fail"
            report["scope"] = (
                "one 7.435-second mono 16 kHz PCM WAV trajectory; exact token and "
                "duration parity does not certify silence, multilingual, resampling, "
                "long-audio chunking, or diarization"
            )
            report["total_elapsed_seconds"] = time.perf_counter() - run_started
            serialized = json.dumps(report, indent=2) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized, encoding="utf-8")
            print(serialized, end="")
            return 0 if report["status"] == "pass" else 1
        finally:
            session.close()
    finally:
        if fixture_context is not None:
            fixture_context.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
