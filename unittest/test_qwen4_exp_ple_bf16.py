#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int32)
I64P = ctypes.POINTER(ctypes.c_int64)
U16P = ctypes.POINTER(ctypes.c_uint16)

LIB.qwen4_ple_ngram_embed_bf16.argtypes = [
    I32P, U16P, I64P, I64P, I64P, F32P, F32P, F32P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
LIB.qwen4_ple_gate_conv_inject_bf16.argtypes = [
    F32P, F32P, F32P, F32P, F32P, F32P, U16P, F32P,
    F32P, F32P, F32P, F32P, F32P, F32P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float,
]


def fptr(value: np.ndarray) -> F32P:
    return value.ctypes.data_as(F32P)


def i32ptr(value: np.ndarray) -> I32P:
    return value.ctypes.data_as(I32P)


def i64ptr(value: np.ndarray) -> I64P:
    return value.ctypes.data_as(I64P)


def u16ptr(value: np.ndarray) -> U16P:
    return value.ctypes.data_as(U16P)


def bf16_bits(value: torch.Tensor) -> np.ndarray:
    return value.to(torch.bfloat16).view(torch.uint16).cpu().numpy().copy()


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).float()


class TestQwen4ExpPleBf16(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def test_ngram_embedding_is_chunk_stable_and_respects_eos(self) -> None:
        ngram_size, heads_per_ngram, head_dim, eos = 3, 2, 3, 2
        multipliers = np.asarray([11, 13, 17], dtype=np.int64)
        vocab_sizes = np.asarray([5, 7, 11, 13], dtype=np.int64)
        offsets = np.asarray([0, 5, 12, 23], dtype=np.int64)
        table = torch.arange(36 * head_dim, dtype=torch.float32).reshape(36, head_dim) / 32.0
        table_bits = bf16_bits(table)
        tokens = np.asarray([7, 8, 9, eos, 10, 11, 12], dtype=np.int32)
        embed_dim = (ngram_size - 1) * heads_per_ngram * head_dim

        def run_chunk(chunk: np.ndarray, state: np.ndarray, position: int) -> np.ndarray:
            output = np.zeros((len(chunk), embed_dim), dtype=np.float32)
            state_out = np.zeros_like(state)
            LIB.qwen4_ple_ngram_embed_bf16(
                i32ptr(chunk), u16ptr(table_bits), i64ptr(multipliers),
                i64ptr(offsets), i64ptr(vocab_sizes), fptr(output), fptr(state), fptr(state_out),
                len(chunk), ngram_size, heads_per_ngram, head_dim, eos, position,
            )
            state[:] = state_out
            return output

        one_state = np.full(ngram_size - 1, eos, dtype=np.float32)
        one_shot = run_chunk(tokens, one_state, 0)
        split_state = np.full(ngram_size - 1, eos, dtype=np.float32)
        chunked = np.concatenate([
            run_chunk(tokens[:3].copy(), split_state, 0),
            run_chunk(tokens[3:].copy(), split_state, 3),
        ])
        np.testing.assert_array_equal(chunked, one_shot)
        np.testing.assert_array_equal(split_state, tokens[-2:].astype(np.float32))

    def test_ngram_embedding_initializes_short_decode_history_with_eos(self) -> None:
        ngram_size, heads_per_ngram, head_dim, eos = 3, 2, 3, 2
        multipliers = np.asarray([11, 13, 17], dtype=np.int64)
        vocab_sizes = np.asarray([5, 7, 11, 13], dtype=np.int64)
        offsets = np.asarray([0, 5, 12, 23], dtype=np.int64)
        table = torch.arange(36 * head_dim, dtype=torch.float32).reshape(36, head_dim) / 32.0
        table_bits = bf16_bits(table)
        embed_dim = (ngram_size - 1) * heads_per_ngram * head_dim

        def run(chunk: np.ndarray, state: np.ndarray, position: int) -> np.ndarray:
            output = np.zeros((len(chunk), embed_dim), dtype=np.float32)
            LIB.qwen4_ple_ngram_embed_bf16(
                i32ptr(chunk), u16ptr(table_bits), i64ptr(multipliers),
                i64ptr(offsets), i64ptr(vocab_sizes), fptr(output), fptr(state), fptr(state),
                len(chunk), ngram_size, heads_per_ngram, head_dim, eos, position,
            )
            return output

        decode_state = np.zeros(ngram_size - 1, dtype=np.float32)
        first = run(np.asarray([7], dtype=np.int32), decode_state, 0)
        np.testing.assert_array_equal(decode_state, np.asarray([eos, 7], dtype=np.float32))
        second = run(np.asarray([8], dtype=np.int32), decode_state, 1)

        reference_state = np.full(ngram_size - 1, eos, dtype=np.float32)
        reference = run(np.asarray([7, 8], dtype=np.int32), reference_state, 0)
        np.testing.assert_array_equal(np.concatenate([first, second]), reference)

    def test_gate_and_dilated_conv_are_chunk_stable(self) -> None:
        rows, streams, hidden, kernel, dilation = 7, 4, 8, 3, 3
        channels = streams * hidden
        history = (kernel - 1) * dilation
        hyper = bf16(torch.randn(rows, channels)).numpy().copy()
        key = bf16(torch.randn(rows, channels)).numpy().copy()
        value = bf16(torch.randn(rows, hidden)).numpy().copy()
        norm_key = (1.0 + 0.1 * torch.randn(channels)).numpy().copy()
        norm_query = (1.0 + 0.1 * torch.randn(channels)).numpy().copy()
        norm_conv = (1.0 + 0.1 * torch.randn(channels)).numpy().copy()
        conv = bf16_bits(0.05 * torch.randn(channels, kernel))

        def run_chunk(start: int, end: int, state: np.ndarray) -> np.ndarray:
            count = end - start
            output = np.zeros((count, channels), dtype=np.float32)
            scratches = [np.zeros((count, channels), dtype=np.float32) for _ in range(4)]
            state_out = np.zeros_like(state)
            LIB.qwen4_ple_gate_conv_inject_bf16(
                fptr(hyper[start:end]), fptr(key[start:end]), fptr(value[start:end]),
                fptr(norm_key), fptr(norm_query), fptr(norm_conv), u16ptr(conv),
                fptr(output), *(fptr(item) for item in scratches), fptr(state), fptr(state_out),
                count, streams, hidden, kernel, dilation, ctypes.c_float(1e-6),
            )
            state[:] = state_out
            return output

        one_state = np.zeros((history, channels), dtype=np.float32)
        one_shot = run_chunk(0, rows, one_state)
        split_state = np.zeros((history, channels), dtype=np.float32)
        chunked = np.concatenate([
            run_chunk(0, 2, split_state),
            run_chunk(2, rows, split_state),
        ])
        np.testing.assert_array_equal(chunked, one_shot)
        np.testing.assert_array_equal(split_state, one_state)

    def _assert_gate_and_dilated_conv_match_pytorch_bf16(self, hidden: int) -> None:
        rows, streams, kernel, dilation = 5, 4, 3, 3
        channels = streams * hidden
        history = (kernel - 1) * dilation
        hyper_t = torch.randn(rows, channels).to(torch.bfloat16)
        key_t = torch.randn(rows, channels).to(torch.bfloat16)
        value_t = torch.randn(rows, hidden).to(torch.bfloat16)
        norm_key_t = (1.0 + 0.1 * torch.randn(channels).to(torch.bfloat16)).float()
        norm_query_t = (1.0 + 0.1 * torch.randn(channels).to(torch.bfloat16)).float()
        norm_conv_t = (1.0 + 0.1 * torch.randn(channels).to(torch.bfloat16)).float()
        conv_t = (0.05 * torch.randn(channels, 1, kernel)).to(torch.bfloat16)

        def grouped_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            grouped = x.float().reshape(rows, streams, hidden)
            normalized = grouped * torch.rsqrt(
                grouped.pow(2).mean(-1, keepdim=True) + 1.0e-6
            )
            return (normalized.flatten(-2) * weight.float()).to(torch.bfloat16)

        key_norm_expected = grouped_rmsnorm(key_t, norm_key_t)
        query_norm_expected = grouped_rmsnorm(hyper_t, norm_query_t)
        gate = (
            key_norm_expected.reshape(rows, streams, hidden)
            * query_norm_expected.reshape(rows, streams, hidden)
        ).sum(dim=-1, keepdim=True) / hidden**0.5
        gate = gate.abs().clamp_min(1.0e-6).sqrt() * gate.sign()
        gated_expected = (torch.sigmoid(gate) * value_t.unsqueeze(1)).flatten(-2)
        conv_norm_expected = grouped_rmsnorm(gated_expected, norm_conv_t)
        padded = F.pad(conv_norm_expected.transpose(0, 1).unsqueeze(0), (history, 0))
        conv_expected = F.silu(
            F.conv1d(padded, conv_t, dilation=dilation, groups=channels)
        ).squeeze(0).transpose(0, 1)
        ple_expected = gated_expected + conv_expected
        output_expected = (hyper_t + ple_expected).float().numpy()

        hyper = hyper_t.float().numpy().copy()
        key = key_t.float().numpy().copy()
        value = value_t.float().numpy().copy()
        norm_key = norm_key_t.numpy().copy()
        norm_query = norm_query_t.numpy().copy()
        norm_conv = norm_conv_t.numpy().copy()
        conv = conv_t.view(torch.uint16).numpy().reshape(channels, kernel).copy()
        output = np.zeros((rows, channels), dtype=np.float32)
        scratches = [np.zeros((rows, channels), dtype=np.float32) for _ in range(4)]
        state_in = np.zeros((history, channels), dtype=np.float32)
        state_out = np.zeros_like(state_in)
        LIB.qwen4_ple_gate_conv_inject_bf16(
            fptr(hyper), fptr(key), fptr(value),
            fptr(norm_key), fptr(norm_query), fptr(norm_conv), u16ptr(conv),
            fptr(output), *(fptr(item) for item in scratches),
            fptr(state_in), fptr(state_out),
            rows, streams, hidden, kernel, dilation, ctypes.c_float(1.0e-6),
        )

        boundaries = (
            ("key_norm", scratches[0], key_norm_expected.float().numpy()),
            ("query_norm", scratches[1], query_norm_expected.float().numpy()),
            ("gated", scratches[2], gated_expected.float().numpy()),
            ("conv_norm", scratches[3], conv_norm_expected.float().numpy()),
            ("output", output, output_expected),
        )
        for name, actual, expected in boundaries:
            differing = int(np.count_nonzero(actual != expected))
            self.assertEqual(
                differing,
                0,
                msg=(
                    f"{name}: differing={differing}/{expected.size} "
                    f"max_abs={float(np.max(np.abs(actual - expected))):.9g}"
                ),
            )

    def test_gate_and_dilated_conv_match_pytorch_bf16(self) -> None:
        for hidden in (64, 256):
            with self.subTest(hidden=hidden):
                self._assert_gate_and_dilated_conv_match_pytorch_bf16(hidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
