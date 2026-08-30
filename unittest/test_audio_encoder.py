#!/usr/bin/env python3
"""Oracle coverage for reusable audio-transformer primitive kernels."""

from __future__ import annotations

import ctypes
import math
import os
import struct

import numpy as np

# This file validates scalar C providers. Pin the PyTorch oracle before import
# so a runner's AVX2/AVX-512 SLEEF dispatch cannot silently change reference
# arithmetic while the C provider remains scalar.
os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")

import torch
import torch.nn.functional as F

from lib_loader import load_lib


lib = load_lib("libckernel_audio.so", "libckernel_engine.so")
attention_lib = load_lib("libckernel_attention.so", "libckernel_engine.so")
gelu_lib = load_lib("libckernel_gelu.so", "libckernel_engine.so")
_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_I16_P = ctypes.POINTER(ctypes.c_int16)
_U8_P = ctypes.POINTER(ctypes.c_uint8)


class CKAudioWavInfo(ctypes.Structure):
    _fields_ = [
        ("format_tag", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("bits_per_sample", ctypes.c_int),
        ("frames", ctypes.c_int),
        ("data_offset", ctypes.c_size_t),
        ("data_bytes", ctypes.c_size_t),
    ]


def _fptr(array: np.ndarray) -> _FLOAT_P:
    return array.ctypes.data_as(_FLOAT_P)


def _i16ptr(array: np.ndarray) -> _I16_P:
    return array.ctypes.data_as(_I16_P)


lib.audio_pcm_s16_to_mono_f32.argtypes = [
    _I16_P, ctypes.c_int, ctypes.c_int, _FLOAT_P,
]
lib.audio_pcm_s16_to_mono_f32.restype = ctypes.c_int
lib.audio_resampled_frame_count.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.audio_resampled_frame_count.restype = ctypes.c_int
lib.audio_resample_linear_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, ctypes.c_int, _FLOAT_P, ctypes.c_int, ctypes.c_int,
]
lib.audio_resample_linear_f32.restype = ctypes.c_int
lib.audio_stft_precompute_tables_f32.argtypes = [
    ctypes.c_int, _FLOAT_P, _FLOAT_P, _FLOAT_P,
]
lib.audio_stft_precompute_tables_f32.restype = ctypes.c_int
lib.audio_stft_power_precomputed_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, _FLOAT_P, ctypes.c_int,
]
lib.audio_stft_power_precomputed_f32.restype = ctypes.c_int
lib.audio_stft_power_centered_window_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, ctypes.c_int, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, _FLOAT_P, ctypes.c_int,
]
lib.audio_stft_power_centered_window_f32.restype = ctypes.c_int
lib.audio_log_mel_time_major_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, _FLOAT_P, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_float,
]
lib.audio_log_mel_time_major_f32.restype = ctypes.c_int
lib.audio_whisper_stft_power_reference_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, ctypes.c_int,
]
lib.audio_whisper_stft_power_reference_f32.restype = ctypes.c_int
lib.audio_conv1d_channel_major_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.audio_conv1d_channel_major_f32.restype = ctypes.c_int
lib.audio_conv2d_whc_grouped_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.audio_conv2d_whc_grouped_f32.restype = ctypes.c_int
lib.audio_glu_split_channel_major_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, ctypes.c_int, ctypes.c_int,
]
lib.audio_glu_split_channel_major_f32.restype = ctypes.c_int
lib.audio_relative_shift_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, ctypes.c_int, ctypes.c_int,
]
lib.audio_relative_shift_f32.restype = ctypes.c_int
lib.audio_transpose_channel_to_token_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, ctypes.c_int, ctypes.c_int,
]
lib.audio_transpose_channel_to_token_f32.restype = ctypes.c_int
attention_lib.attention_forward_query_key_head_major_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
attention_lib.attention_forward_query_key_head_major_f32.restype = ctypes.c_int
attention_lib.attention_forward_query_key_head_major_f32_packed_k.argtypes = [
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
attention_lib.attention_forward_query_key_head_major_f32_packed_k.restype = ctypes.c_int
attention_lib.attention_forward_query_key_head_major_tiled_f16kv_fp32.argtypes = [
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
attention_lib.attention_forward_query_key_head_major_tiled_f16kv_fp32.restype = ctypes.c_int
lib.audio_wav_parse_memory.argtypes = [
    _U8_P, ctypes.c_size_t, ctypes.POINTER(CKAudioWavInfo),
]
lib.audio_wav_parse_memory.restype = ctypes.c_int
lib.audio_wav_decode_pcm16_mono_f32.argtypes = [
    _U8_P, ctypes.c_size_t, ctypes.POINTER(CKAudioWavInfo), _FLOAT_P, ctypes.c_int,
]
lib.audio_wav_decode_pcm16_mono_f32.restype = ctypes.c_int
lib.audio_wav_decode_memory_pcm16_mono_f32.argtypes = [
    _U8_P, ctypes.c_size_t, _FLOAT_P, ctypes.c_int,
    ctypes.POINTER(CKAudioWavInfo),
]
lib.audio_wav_decode_memory_pcm16_mono_f32.restype = ctypes.c_int
lib.audio_wav_decode_memory_pcm16_mono_window_f32.argtypes = [
    _U8_P, ctypes.c_size_t, ctypes.c_int, _FLOAT_P, ctypes.c_int,
    ctypes.POINTER(CKAudioWavInfo),
]
lib.audio_wav_decode_memory_pcm16_mono_window_f32.restype = ctypes.c_int
lib.audio_resample_windowed_sinc_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, ctypes.c_int, _FLOAT_P, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
]
lib.audio_resample_windowed_sinc_f32.restype = ctypes.c_int
lib.audio_pad_or_truncate_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, ctypes.c_int,
]
lib.audio_pad_or_truncate_f32.restype = ctypes.c_int
lib.audio_preemphasis_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, ctypes.c_int, ctypes.c_float,
]
lib.audio_preemphasis_f32.restype = ctypes.c_int
lib.audio_feature_normalize_per_feature_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
lib.audio_feature_normalize_per_feature_f32.restype = ctypes.c_int
lib.audio_stft_power_fft400_f32.argtypes = [
    _FLOAT_P, ctypes.c_int, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, _FLOAT_P, ctypes.c_int, _FLOAT_P,
]
lib.audio_stft_power_fft400_f32.restype = ctypes.c_int
lib.audio_whisper_mel_filters_slaney_f32.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, _FLOAT_P,
]
lib.audio_whisper_mel_filters_slaney_f32.restype = ctypes.c_int
lib.audio_whisper_log_mel_from_power_reference_f32.argtypes = [
    _FLOAT_P, _FLOAT_P, ctypes.c_int, ctypes.c_int, _FLOAT_P,
]
lib.audio_whisper_log_mel_from_power_reference_f32.restype = ctypes.c_int
lib.audio_whisper_log_mel_window_wav_pcm16_f32.argtypes = [
    _U8_P, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
    _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
    ctypes.c_int, ctypes.c_int, _FLOAT_P,
]
lib.audio_whisper_log_mel_window_wav_pcm16_f32.restype = ctypes.c_int
gelu_lib.gelu_erf_fp64_f32_inplace.argtypes = [_FLOAT_P, ctypes.c_size_t]
gelu_lib.gelu_erf_fp64_f32_inplace.restype = None


def check_wav_pcm16() -> None:
    pcm = np.array(
        [[-32768, -32768], [32767, 32767], [-1000, 1000], [1234, 5678]],
        dtype="<i2",
    )
    fmt = struct.pack("<HHIIHH", 1, 2, 48000, 48000 * 4, 4, 16)
    junk = b"abc"
    chunks = (
        b"JUNK" + struct.pack("<I", len(junk)) + junk + b"\0"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", pcm.nbytes) + pcm.tobytes()
    )
    wav = np.frombuffer(
        b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks,
        dtype=np.uint8,
    )
    info = CKAudioWavInfo()
    assert lib.audio_wav_parse_memory(
        wav.ctypes.data_as(_U8_P), wav.size, ctypes.byref(info)
    ) == 0
    assert (info.channels, info.sample_rate, info.bits_per_sample, info.frames) == (
        2, 48000, 16, 4,
    )
    actual = np.empty(info.frames, dtype=np.float32)
    assert lib.audio_wav_decode_pcm16_mono_f32(
        wav.ctypes.data_as(_U8_P), wav.size, ctypes.byref(info),
        _fptr(actual), actual.size,
    ) == info.frames
    expected = np.array(
        [-1.0, 32767.0 / 32768.0, 0.0, 3456.0 / 32768.0], dtype=np.float32,
    )
    assert np.array_equal(actual, expected)
    fused_actual = np.empty(info.frames, dtype=np.float32)
    fused_info = CKAudioWavInfo()
    assert lib.audio_wav_decode_memory_pcm16_mono_f32(
        wav.ctypes.data_as(_U8_P), wav.size, _fptr(fused_actual),
        fused_actual.size, ctypes.byref(fused_info),
    ) == info.frames
    assert np.array_equal(fused_actual, expected)
    assert (
        fused_info.channels,
        fused_info.sample_rate,
        fused_info.frames,
    ) == (2, 48000, 4)
    window = np.empty(2, dtype=np.float32)
    window_info = CKAudioWavInfo()
    assert lib.audio_wav_decode_memory_pcm16_mono_window_f32(
        wav.ctypes.data_as(_U8_P), wav.size, 2, _fptr(window),
        window.size, ctypes.byref(window_info),
    ) == 2
    assert np.array_equal(window, expected[2:])
    assert window_info.frames == 4
    assert lib.audio_wav_decode_memory_pcm16_mono_window_f32(
        wav.ctypes.data_as(_U8_P), wav.size, 4, _fptr(window),
        window.size, ctypes.byref(window_info),
    ) == -7
    truncated = wav[:-1].copy()
    assert lib.audio_wav_parse_memory(
        truncated.ctypes.data_as(_U8_P), truncated.size, ctypes.byref(info)
    ) == -3
    print("audio_wav_pcm16_chunked_decode max_diff=0 tol=0 [PASS]")


def check_pcm() -> None:
    stereo = np.array(
        [[-32768, -32768], [32767, 32767], [-1000, 1000], [1234, 5678]],
        dtype=np.int16,
    )
    actual = np.empty(stereo.shape[0], dtype=np.float32)
    assert lib.audio_pcm_s16_to_mono_f32(
        _i16ptr(stereo), stereo.shape[0], stereo.shape[1], _fptr(actual)
    ) == 0
    expected = np.array([-1.0, 32767.0 / 32768.0, 0.0, 3456.0 / 32768.0], dtype=np.float32)
    assert np.array_equal(actual, expected)
    print("audio_pcm_s16_stereo_to_mono max_diff=0 tol=0 [PASS]")


def check_pad_or_truncate() -> None:
    source = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    padded = np.full(5, 99.0, dtype=np.float32)
    assert lib.audio_pad_or_truncate_f32(
        _fptr(source), source.size, _fptr(padded), padded.size
    ) == source.size
    assert np.array_equal(padded, np.array([1.0, -2.0, 3.0, 0.0, 0.0], dtype=np.float32))
    truncated = np.empty(2, dtype=np.float32)
    assert lib.audio_pad_or_truncate_f32(
        _fptr(source), source.size, _fptr(truncated), truncated.size
    ) == truncated.size
    assert np.array_equal(truncated, source[:2])
    print("audio_pad_or_truncate max_diff=0 tol=0 [PASS]")


def check_preemphasis() -> None:
    source = np.array([0.25, -0.5, 1.0, 0.125, -0.75], dtype=np.float32)
    coefficient = np.float32(0.97)
    expected = source.copy()
    expected[1:] = source[1:] - coefficient * source[:-1]

    actual = np.empty_like(source)
    assert lib.audio_preemphasis_f32(
        _fptr(source), _fptr(actual), source.size, float(coefficient),
    ) == 0
    assert np.array_equal(actual, expected)

    inplace = source.copy()
    assert lib.audio_preemphasis_f32(
        _fptr(inplace), _fptr(inplace), inplace.size, float(coefficient),
    ) == 0
    assert np.array_equal(inplace, expected)
    print("audio_preemphasis max_diff=0 tol=0 [PASS]")


def check_per_feature_normalization() -> None:
    source = np.array(
        [[1.0, 5.0, -2.0], [2.0, 5.0, 0.0], [4.0, 5.0, 6.0], [9.0, 5.0, 8.0]],
        dtype=np.float32,
    )
    epsilon = np.float32(1.0e-5)
    mean = source.astype(np.float64).mean(axis=0)
    centered = source.astype(np.float64) - mean
    variance = np.sum(centered * centered, axis=0) / (source.shape[0] - 1)
    std = np.sqrt(variance.astype(np.float32), dtype=np.float32) + epsilon
    expected = (centered.astype(np.float32) * (np.float32(1.0) / std)).astype(np.float32)

    actual = np.empty_like(source)
    assert lib.audio_feature_normalize_per_feature_f32(
        _fptr(source), _fptr(actual), source.shape[1], source.shape[0], float(epsilon),
    ) == 0
    assert np.array_equal(actual, expected)

    single = np.array([[3.0, -4.0]], dtype=np.float32)
    assert lib.audio_feature_normalize_per_feature_f32(
        _fptr(single), _fptr(single), 2, 1, float(epsilon),
    ) == 0
    assert np.array_equal(single, np.zeros_like(single))
    print("audio_per_feature_normalization max_diff=0 tol=0 [PASS]")


def check_resample() -> None:
    source = np.random.default_rng(20260720).normal(0.0, 0.2, 97).astype(np.float32)
    output_frames = lib.audio_resampled_frame_count(source.size, 48000, 16000)
    assert output_frames == 33
    actual = np.empty(output_frames, dtype=np.float32)
    assert lib.audio_resample_linear_f32(
        _fptr(source), source.size, 48000, _fptr(actual), output_frames, 16000
    ) == 0
    expected = source[np.arange(output_frames, dtype=np.int64) * 3]
    assert np.array_equal(actual, expected)
    invalid = np.empty(output_frames + 1, dtype=np.float32)
    assert lib.audio_resample_linear_f32(
        _fptr(source), source.size, 48000, _fptr(invalid), invalid.size, 16000
    ) == -2
    print("audio_resample_linear_48k_to_16k max_diff=0 tol=0 [PASS]")

    source = np.random.default_rng(44100).normal(0.0, 0.2, 127).astype(np.float32)
    output_frames = lib.audio_resampled_frame_count(source.size, 44100, 16000)
    actual = np.empty(output_frames, dtype=np.float32)
    assert lib.audio_resample_linear_f32(
        _fptr(source), source.size, 44100, _fptr(actual), output_frames, 16000
    ) == 0
    expected = np.empty_like(actual)
    for output_index in range(output_frames):
        numerator = output_index * 44100
        left = numerator // 16000
        remainder = numerator % 16000
        right = min(left + 1, source.size - 1)
        fraction = np.float32(remainder) / np.float32(16000)
        expected[output_index] = np.float32(
            source[left] + fraction * np.float32(source[right] - source[left])
        )
    max_diff = float(np.max(np.abs(actual - expected)))
    assert max_diff <= 1.0e-7, max_diff
    print(
        f"audio_resample_linear_44k1_to_16k max_diff={max_diff:.8e} "
        "tol=1.0e-07 [PASS]"
    )


def _windowed_sinc_reference(
    source: np.ndarray, input_rate: int, output_rate: int, radius: int,
) -> np.ndarray:
    output_frames = 1 + ((source.size - 1) * output_rate) // input_rate
    output = np.empty(output_frames, dtype=np.float32)
    cutoff = min(1.0, output_rate / input_rate)
    for frame in range(output_frames):
        coordinate = frame * input_rate / output_rate
        center = math.floor(coordinate)
        weighted = 0.0
        weight_sum = 0.0
        for tap in range(center - radius + 1, center + radius + 1):
            if tap < 0 or tap >= source.size:
                continue
            distance = coordinate - tap
            scaled = cutoff * distance
            sinc = 1.0 if abs(scaled) < 1.0e-12 else math.sin(math.pi * scaled) / (math.pi * scaled)
            window_x = distance / radius
            if abs(window_x) >= 1.0:
                continue
            weight = cutoff * sinc * (0.5 * (1.0 + math.cos(math.pi * window_x)))
            weighted += float(source[tap]) * weight
            weight_sum += weight
        output[frame] = weighted / weight_sum if weight_sum else 0.0
    return output


def check_bandlimited_resample() -> None:
    input_rate = 44100
    output_rate = 16000
    radius = 16
    source = np.random.default_rng(16000).normal(0.0, 0.2, 257).astype(np.float32)
    expected = _windowed_sinc_reference(source, input_rate, output_rate, radius)
    actual = np.empty_like(expected)
    assert lib.audio_resample_windowed_sinc_f32(
        _fptr(source), source.size, input_rate, _fptr(actual), actual.size,
        output_rate, radius,
    ) == 0
    max_diff = float(np.max(np.abs(actual - expected)))
    assert max_diff <= 3.0e-8, max_diff

    time = np.arange(4800, dtype=np.float64) / 48000.0
    alias_source = np.sin(2.0 * math.pi * 12000.0 * time).astype(np.float32)
    alias_frames = lib.audio_resampled_frame_count(alias_source.size, 48000, 16000)
    linear = np.empty(alias_frames, dtype=np.float32)
    filtered = np.empty(alias_frames, dtype=np.float32)
    assert lib.audio_resample_linear_f32(
        _fptr(alias_source), alias_source.size, 48000, _fptr(linear), alias_frames, 16000
    ) == 0
    assert lib.audio_resample_windowed_sinc_f32(
        _fptr(alias_source), alias_source.size, 48000, _fptr(filtered), alias_frames,
        16000, radius,
    ) == 0
    trim = radius
    linear_rms = float(np.sqrt(np.mean(linear[trim:-trim] ** 2)))
    filtered_rms = float(np.sqrt(np.mean(filtered[trim:-trim] ** 2)))
    assert filtered_rms < linear_rms * 0.05, (linear_rms, filtered_rms)
    print(
        f"audio_resample_windowed_sinc max_diff={max_diff:.8e} tol=3.0e-08 [PASS] "
        f"alias_rejection={linear_rms / max(filtered_rms, 1.0e-20):.2f}x"
    )


def check_precomputed_stft() -> None:
    n_fft = 400
    hop = 160
    bins = n_fft // 2 + 1
    samples = np.random.default_rng(73).normal(0.0, 0.1, 3200).astype(np.float32)
    frames = samples.size // hop
    window = np.empty(n_fft, dtype=np.float32)
    cos_table = np.empty((bins, n_fft), dtype=np.float32)
    sin_table = np.empty_like(cos_table)
    assert lib.audio_stft_precompute_tables_f32(
        n_fft, _fptr(window), _fptr(cos_table), _fptr(sin_table)
    ) == 0
    direct = np.empty((frames, bins), dtype=np.float32)
    table = np.empty_like(direct)
    assert lib.audio_whisper_stft_power_reference_f32(
        _fptr(samples), samples.size, _fptr(direct), frames
    ) == 0
    assert lib.audio_stft_power_precomputed_f32(
        _fptr(samples), samples.size, _fptr(window), _fptr(cos_table),
        _fptr(sin_table), n_fft, hop, _fptr(table), frames
    ) == 0
    assert np.array_equal(table, direct)
    print("audio_stft_precomputed_vs_direct max_diff=0 tol=0 [PASS]")

    fft_power = np.empty_like(direct)
    fft_scratch = np.empty(n_fft * 2, dtype=np.float32)
    assert lib.audio_stft_power_fft400_f32(
        _fptr(samples), samples.size, _fptr(window), _fptr(cos_table),
        _fptr(sin_table), hop, _fptr(fft_power), frames, _fptr(fft_scratch),
    ) == 0
    max_diff = float(np.max(np.abs(fft_power - direct)))
    rmse = float(np.sqrt(np.mean((fft_power - direct) ** 2)))
    assert max_diff <= 4.0e-4, max_diff
    assert rmse <= 5.0e-5, rmse
    print(
        f"audio_stft_fft400_vs_direct max_diff={max_diff:.8e} tol=4.0e-04 [PASS] "
        f"rmse={rmse:.8e} rmse_tol=5.0e-05"
    )


def check_centered_window_stft_and_log_mel() -> None:
    n_fft = 16
    window_length = 10
    hop = 4
    bins = n_fft // 2 + 1
    samples = np.random.default_rng(512400).normal(0.0, 0.2, 33).astype(np.float32)
    window = np.hanning(window_length + 1)[:-1].astype(np.float32)
    frames = samples.size // hop + 1
    table_window = np.empty(n_fft, dtype=np.float32)
    cos_table = np.empty((bins, n_fft), dtype=np.float32)
    sin_table = np.empty_like(cos_table)
    assert lib.audio_stft_precompute_tables_f32(
        n_fft, _fptr(table_window), _fptr(cos_table), _fptr(sin_table)
    ) == 0

    expected = np.empty((frames, bins), dtype=np.float32)
    window_start = (n_fft - window_length) // 2
    for frame in range(frames):
        for frequency in range(bins):
            real = np.float32(0.0)
            imag = np.float32(0.0)
            for sample in range(window_length):
                fft_sample = window_start + sample
                source = frame * hop + fft_sample - n_fft // 2
                if source < 0 or source >= samples.size:
                    continue
                value = np.float32(samples[source] * window[sample])
                real = np.float32(real + np.float32(value * cos_table[frequency, fft_sample]))
                imag = np.float32(imag + np.float32(value * sin_table[frequency, fft_sample]))
            expected[frame, frequency] = np.float32(real * real + imag * imag)

    actual = np.empty_like(expected)
    assert lib.audio_stft_power_centered_window_f32(
        _fptr(samples), samples.size, _fptr(window), window_length,
        _fptr(cos_table), _fptr(sin_table), n_fft, hop, 0,
        _fptr(actual), frames,
    ) == 0
    max_diff = float(np.max(np.abs(actual - expected)))
    assert max_diff <= 2.0e-6, max_diff

    filters = np.random.default_rng(128257).uniform(0.0, 0.2, (3, bins)).astype(np.float32)
    epsilon = np.float32(2.0 ** -24)
    actual_mel = np.empty((frames, filters.shape[0]), dtype=np.float32)
    assert lib.audio_log_mel_time_major_f32(
        _fptr(actual), _fptr(filters), _fptr(actual_mel), frames, bins,
        filters.shape[0], float(epsilon),
    ) == 0
    expected_mel = np.log(actual @ filters.T + epsilon).astype(np.float32)
    mel_diff = float(np.max(np.abs(actual_mel - expected_mel)))
    assert mel_diff <= 1.0e-6, mel_diff
    print(
        f"audio_centered_window_stft max_diff={max_diff:.8e} tol=2.0e-06 [PASS] "
        f"log_mel_max_diff={mel_diff:.8e} tol=1.0e-06"
    )


def check_grouped_conv2d() -> None:
    rng = np.random.default_rng(20260829)

    def run_case(channels_in: int, channels_out: int, groups: int) -> float:
        width, height = 7, 6
        kernel_width = kernel_height = 3
        stride_width = stride_height = 2
        padding_width = padding_height = 1
        output_width = (width + 2 * padding_width - kernel_width) // stride_width + 1
        output_height = (height + 2 * padding_height - kernel_height) // stride_height + 1
        input_value = rng.normal(0.0, 0.2, (channels_in, height, width)).astype(np.float32)
        weight = rng.normal(
            0.0, 0.1,
            (channels_out, channels_in // groups, kernel_height, kernel_width),
        ).astype(np.float32)
        bias = rng.normal(0.0, 0.05, channels_out).astype(np.float32)
        actual = np.empty((channels_out, output_height, output_width), dtype=np.float32)
        assert lib.audio_conv2d_whc_grouped_f32(
            _fptr(input_value), _fptr(weight), _fptr(bias), _fptr(actual),
            width, height, channels_in, channels_out, kernel_width, kernel_height,
            stride_width, stride_height, padding_width, padding_height, groups,
            output_width, output_height,
        ) == 0
        expected = F.conv2d(
            torch.from_numpy(input_value[None]), torch.from_numpy(weight),
            torch.from_numpy(bias), stride=(stride_height, stride_width),
            padding=(padding_height, padding_width), groups=groups,
        ).numpy()[0]
        difference = np.abs(actual - expected)
        maximum = float(np.max(difference))
        assert maximum <= 3.0e-7, maximum
        return maximum

    regular = run_case(3, 5, 1)
    depthwise = run_case(4, 4, 4)
    print(
        "audio_grouped_conv2d "
        f"regular_max_diff={regular:.8e} depthwise_max_diff={depthwise:.8e} "
        "tol=3.0e-07 [PASS]"
    )


def check_split_glu() -> None:
    rng = np.random.default_rng(20260830)
    channels, frames = 1280, 37
    packed = rng.normal(0.0, 0.7, (2 * channels, frames)).astype(np.float32)
    actual = np.empty((channels, frames), dtype=np.float32)
    assert lib.audio_glu_split_channel_major_f32(
        _fptr(packed), _fptr(actual), channels, frames,
    ) == 0
    value = torch.from_numpy(packed[:channels])
    gate = torch.from_numpy(packed[channels:])
    expected = (value * torch.sigmoid(gate)).numpy()
    difference = np.abs(actual - expected)
    maximum = float(np.max(difference))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    assert maximum <= 1.2e-7, maximum
    assert rmse <= 2.0e-8, rmse
    print(
        "audio_split_glu "
        f"max_diff={maximum:.8e} tol=1.2e-07 [PASS] "
        f"rmse={rmse:.8e} rmse_tol=2.0e-08"
    )


def check_relative_shift() -> None:
    heads, frames = 3, 11
    raw_frames = 2 * frames - 1
    raw = np.arange(heads * frames * raw_frames, dtype=np.float32).reshape(
        heads, frames, raw_frames,
    )
    actual = np.empty((heads, frames, frames), dtype=np.float32)
    assert lib.audio_relative_shift_f32(
        _fptr(raw), _fptr(actual), heads, frames,
    ) == 0
    expected = np.empty_like(actual)
    for query in range(frames):
        for key in range(frames):
            expected[:, query, key] = raw[:, query, frames - 1 + key - query]
    assert np.array_equal(actual, expected)
    print("audio_relative_shift exact_index_mapping [PASS]")


def check_global_log_mel_window() -> None:
    sample_rate = 16000
    samples = (
        np.sin(np.arange(1920, dtype=np.float64) * (2.0 * np.pi * 440.0 / sample_rate))
        * 12000.0
    ).astype("<i2")
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    chunks = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", samples.nbytes) + samples.tobytes()
    )
    wav = np.frombuffer(
        b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks,
        dtype=np.uint8,
    )
    decoded = samples.astype(np.float32) / np.float32(32768.0)
    window = np.empty(400, dtype=np.float32)
    cos_table = np.empty((201, 400), dtype=np.float32)
    sin_table = np.empty((201, 400), dtype=np.float32)
    filters = np.empty((4, 201), dtype=np.float32)
    assert lib.audio_stft_precompute_tables_f32(
        400, _fptr(window), _fptr(cos_table), _fptr(sin_table)
    ) == 0
    assert lib.audio_whisper_mel_filters_slaney_f32(
        sample_rate, 400, filters.shape[0], _fptr(filters)
    ) == 0
    power = np.empty((12, 201), dtype=np.float32)
    scratch = np.empty(800, dtype=np.float32)
    assert lib.audio_stft_power_fft400_f32(
        _fptr(decoded), decoded.size, _fptr(window), _fptr(cos_table),
        _fptr(sin_table), 160, _fptr(power), power.shape[0], _fptr(scratch)
    ) == 0
    complete = np.empty((4, 12), dtype=np.float32)
    assert lib.audio_whisper_log_mel_from_power_reference_f32(
        _fptr(power), _fptr(filters), complete.shape[0], complete.shape[1],
        _fptr(complete)
    ) == 0

    actual = np.full((4, 12), 99.0, dtype=np.float32)
    valid = lib.audio_whisper_log_mel_window_wav_pcm16_f32(
        wav.ctypes.data_as(_U8_P), wav.size, 4 * 160, sample_rate,
        _fptr(window), _fptr(cos_table), _fptr(sin_table), _fptr(filters),
        actual.shape[0], actual.shape[1], _fptr(actual)
    )
    expected = np.zeros_like(actual)
    expected[:, :8] = complete[:, 4:]
    assert valid == 8
    assert np.array_equal(actual, expected)
    print("audio_global_log_mel_window max_diff=0 tol=0 tail_zero=1 [PASS]")


def _check_conv(name: str, cin: int, cout: int, frames: int, stride: int) -> None:
    rng = np.random.default_rng(cin * 1000 + cout + frames + stride)
    source = rng.normal(0.0, 0.15, (cin, frames)).astype(np.float32)
    weight = rng.normal(0.0, 0.08, (cout, cin, 3)).astype(np.float32)
    bias = rng.normal(0.0, 0.03, cout).astype(np.float32)
    output_frames = (frames + 2 - 3) // stride + 1
    actual = np.empty((cout, output_frames), dtype=np.float32)
    assert lib.audio_conv1d_channel_major_f32(
        _fptr(source), _fptr(weight), _fptr(bias), _fptr(actual),
        cin, cout, frames, 3, stride, 1, output_frames,
    ) == 0
    expected = F.conv1d(
        torch.from_numpy(source)[None], torch.from_numpy(weight),
        torch.from_numpy(bias), stride=stride, padding=1,
    )[0].numpy()
    max_diff = float(np.max(np.abs(actual - expected)))
    rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
    assert max_diff <= 2.0e-5, (name, max_diff)
    assert rmse <= 2.0e-6, (name, rmse)
    print(
        f"{name} max_diff={max_diff:.8e} tol=2.0e-05 [PASS] "
        f"rmse={rmse:.8e} rmse_tol=2.0e-06"
    )


def check_conv_stride2_production_equivalence() -> None:
    cin, cout, frames, output_frames = 512, 512, 3000, 1500
    rng = np.random.default_rng(2026080102)
    source = rng.normal(0.0, 0.15, (cin, frames)).astype(np.float32)
    weight = rng.normal(0.0, 0.08, (cout, cin, 3)).astype(np.float32)
    bias = rng.normal(0.0, 0.03, cout).astype(np.float32)
    baseline = np.empty((cout, output_frames), dtype=np.float32)
    optimized = np.empty_like(baseline)
    previous = os.environ.get("CK_DISABLE_AUDIO_CONV_STRIDE2_CONTIGUOUS")
    try:
        os.environ["CK_DISABLE_AUDIO_CONV_STRIDE2_CONTIGUOUS"] = "1"
        assert lib.audio_conv1d_channel_major_f32(
            _fptr(source), _fptr(weight), _fptr(bias), _fptr(baseline),
            cin, cout, frames, 3, 2, 1, output_frames,
        ) == 0
        os.environ.pop("CK_DISABLE_AUDIO_CONV_STRIDE2_CONTIGUOUS", None)
        assert lib.audio_conv1d_channel_major_f32(
            _fptr(source), _fptr(weight), _fptr(bias), _fptr(optimized),
            cin, cout, frames, 3, 2, 1, output_frames,
        ) == 0
    finally:
        if previous is None:
            os.environ.pop("CK_DISABLE_AUDIO_CONV_STRIDE2_CONTIGUOUS", None)
        else:
            os.environ["CK_DISABLE_AUDIO_CONV_STRIDE2_CONTIGUOUS"] = previous
    assert np.array_equal(optimized, baseline)
    print(
        "audio_conv1d_whisper_stem2_production "
        f"compared={optimized.size} max_diff=0 tol=0 [PASS]"
    )


def check_transpose() -> None:
    source = np.arange(7 * 13, dtype=np.float32).reshape(7, 13)
    actual = np.empty((13, 7), dtype=np.float32)
    assert lib.audio_transpose_channel_to_token_f32(
        _fptr(source), _fptr(actual), 7, 13
    ) == 0
    assert np.array_equal(actual, source.T)
    print("audio_channel_to_token_transpose max_diff=0 tol=0 [PASS]")


def check_pytorch_erf_gelu() -> None:
    edge = np.array(
        [
            -20.0, -10.0, -5.0, -3.0, -1.0, -0.5, -0.0,
            0.0, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0,
        ],
        dtype=np.float32,
    )
    random = np.random.default_rng(20260725).normal(0.0, 2.5, 16384).astype(np.float32)
    source = np.concatenate((edge, random))
    actual = source.copy()
    gelu_lib.gelu_erf_fp64_f32_inplace(_fptr(actual), actual.size)
    inv_sqrt_2 = 0.707106781186547524400844362104849039
    expected = np.asarray(
        [
            np.float32(
                0.5 * float(value)
                * (1.0 + math.erf(float(value) * inv_sqrt_2))
            )
            for value in source
        ],
        dtype=np.float32,
    )
    assert np.array_equal(actual, expected)
    print(
        "audio_erf_gelu_fp64_scalar max_diff=0 tol=0 [PASS]",
        flush=True,
    )

    pytorch = F.gelu(torch.from_numpy(source), approximate="none").numpy()
    observed = np.abs(actual - pytorch)
    print(
        "audio_pytorch_erf_gelu_observed "
        f"max_diff={float(np.max(observed)):.8e} "
        f"rmse={float(np.sqrt(np.mean(observed * observed))):.8e} "
        f"torch={torch.__version__} "
        f"cpu={torch.backends.cpu.get_cpu_capability()}",
        flush=True,
    )


def _check_cross_attention(name: str, heads: int, query_tokens: int, key_tokens: int, dim: int) -> None:
    rng = np.random.default_rng(heads * 100000 + query_tokens * 1000 + key_tokens + dim)
    query = rng.normal(0.0, 0.12, (heads, query_tokens, dim)).astype(np.float32)
    key = rng.normal(0.0, 0.12, (heads, key_tokens, dim)).astype(np.float32)
    value = rng.normal(0.0, 0.12, (heads, key_tokens, dim)).astype(np.float32)
    actual = np.empty_like(query)
    packed_actual = np.empty_like(query)
    scratch = np.empty((query_tokens, key_tokens), dtype=np.float32)
    key_transpose_scratch = np.empty((heads, dim, key_tokens), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(dim))
    assert attention_lib.attention_forward_query_key_head_major_f32(
        _fptr(query), _fptr(key), _fptr(value), _fptr(actual), _fptr(scratch),
        heads, query_tokens, key_tokens, dim, float(scale),
    ) == 0
    assert attention_lib.attention_forward_query_key_head_major_f32_packed_k(
        _fptr(query), _fptr(key), _fptr(value), _fptr(packed_actual), _fptr(scratch),
        _fptr(key_transpose_scratch),
        heads, query_tokens, key_tokens, dim, float(scale),
    ) == 0
    assert np.array_equal(packed_actual, actual), name
    tq = torch.from_numpy(query)
    tk = torch.from_numpy(key)
    tv = torch.from_numpy(value)
    expected = (torch.softmax((tq @ tk.transpose(-1, -2)) * float(scale), dim=-1) @ tv).numpy()
    max_diff = float(np.max(np.abs(actual - expected)))
    rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
    assert max_diff <= 2.0e-6, (name, max_diff)
    assert rmse <= 3.0e-7, (name, rmse)
    print(
        f"{name} max_diff={max_diff:.8e} tol=2.0e-06 [PASS] "
        f"rmse={rmse:.8e} rmse_tol=3.0e-07"
    )


def check_tiled_f16kv_encoder_attention() -> None:
    rng = np.random.default_rng(20260801)
    heads, tokens, dim = 2, 257, 64
    query = rng.normal(0.0, 0.12, (heads, tokens, dim)).astype(np.float32)
    key = rng.normal(0.0, 0.12, query.shape).astype(np.float32)
    value = rng.normal(0.0, 0.12, query.shape).astype(np.float32)
    actual = np.empty_like(query)
    scale = np.float32(1.0 / math.sqrt(dim))
    assert attention_lib.attention_forward_query_key_head_major_tiled_f16kv_fp32(
        _fptr(query), _fptr(key), _fptr(value), _fptr(actual),
        heads, tokens, tokens, dim, float(scale),
    ) == 0

    tq = torch.from_numpy(query)
    tk = torch.from_numpy(key.astype(np.float16).astype(np.float32))
    tv = torch.from_numpy(value.astype(np.float16).astype(np.float32))
    expected = (
        torch.softmax((tq @ tk.transpose(-1, -2)) * float(scale), dim=-1) @ tv
    ).numpy()
    diff = actual - expected
    max_diff = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    assert max_diff <= 2.0e-5, max_diff
    assert rmse <= 2.0e-6, rmse
    print(
        "audio_encoder_tiled_f16kv_attention "
        f"max_diff={max_diff:.8e} tol=2.0e-05 [PASS] "
        f"rmse={rmse:.8e} rmse_tol=2.0e-06"
    )


def main() -> None:
    capability = torch.backends.cpu.get_cpu_capability()
    assert capability in {"DEFAULT", "NO AVX"}, capability
    torch.set_num_threads(1)
    check_wav_pcm16()
    check_pcm()
    check_pad_or_truncate()
    check_preemphasis()
    check_per_feature_normalization()
    check_resample()
    check_bandlimited_resample()
    check_precomputed_stft()
    check_centered_window_stft_and_log_mel()
    check_grouped_conv2d()
    check_split_glu()
    check_relative_shift()
    check_global_log_mel_window()
    _check_conv("audio_conv1d_whisper_stem1", 80, 384, 16, 1)
    _check_conv("audio_conv1d_whisper_stem2", 384, 384, 16, 2)
    check_conv_stride2_production_equivalence()
    check_pytorch_erf_gelu()
    check_transpose()
    _check_cross_attention("audio_encoder_self_attention_equal", 6, 11, 11, 64)
    _check_cross_attention("audio_cross_attention_unequal_small", 3, 5, 17, 8)
    _check_cross_attention("audio_cross_attention_whisper_decode", 6, 1, 1500, 64)
    check_tiled_f16kv_encoder_attention()
    print("ALL TESTS PASSED (24/24)")


if __name__ == "__main__":
    main()
