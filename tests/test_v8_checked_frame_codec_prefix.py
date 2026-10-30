"""Checked capacity-codec handoff with a shorter runtime-valid prefix."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from tests.test_v8_checked_frame_loop import Loop
from tests.test_v8_qwen3_tts_loop_codec import FrameCodec

ROOT = Path(__file__).resolve().parents[1]


class PrefixCodec(ctypes.Structure):
    _fields_ = [("base", FrameCodec), ("samples_per_code_frame", ctypes.c_size_t)]


class CheckedFrameCodecPrefixTest(unittest.TestCase):
    def test_prefix_rejection_failure_and_recovery(self):
        source = r'''
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
static int mode;
static int codec_calls;
void set_mode(int value) { mode = value; codec_calls = 0; }
int get_codec_calls(void) { return codec_calls; }
int prefill(uint8_t *arena, size_t bytes) {
    if (bytes != 128) return -1;
    int32_t one = 1, a = 11, b = 12;
    memcpy(arena + 4, &one, 4);
    memcpy(arena + 16, &a, 4);
    memcpy(arena + 20, &b, 4);
    return 0;
}
int frame(uint8_t *arena, size_t bytes) {
    if (bytes != 128 || mode == 2) return -1;
    int32_t step;
    memcpy(&step, arena, 4);
    if (mode == 4 && step == 0) return 1;
    if (step == 1) return 1;
    int32_t one = 1, a = 21, b = 22;
    memcpy(arena + 8, &one, 4);
    memcpy(arena + 24, &a, 4);
    memcpy(arena + 28, &b, 4);
    return 0;
}
int codec(uint8_t *arena, size_t bytes) {
    if (bytes != 128) return -1;
    ++codec_calls;
    if (mode == 1) return -1;
    int32_t *codes = (int32_t *)(void *)(arena + 32);
    float *wave = (float *)(void *)(arena + 64);
    for (int frame = 0; frame < 3; ++frame)
        for (int sample = 0; sample < 4; ++sample)
            wave[frame * 4 + sample] = (float)codes[frame * 2];
    if (mode == 3) wave[0] = NAN;
    int32_t twelve = 12;
    memcpy(arena + 120, &twelve, 4);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / "fake.c"
            fake.write_text(source)
            library = Path(temp) / "prefix.so"
            subprocess.run([
                "cc", "-std=c11", "-O2", "-shared", "-fPIC",
                "-I", str(ROOT / "include"),
                str(ROOT / "src/runtime/checked_frame_loop.c"),
                str(ROOT / "src/runtime/checked_frame_codec.c"),
                str(fake), "-lm", "-o", str(library),
            ], check=True, capture_output=True)
            native = ctypes.CDLL(str(library))
            native.set_mode.argtypes = [ctypes.c_int]
            native.get_codec_calls.restype = ctypes.c_int
            run = native.ck_run_checked_frame_codec_prefix
            run.argtypes = [ctypes.POINTER(PrefixCodec)]
            run.restype = ctypes.c_int
            arena = (ctypes.c_uint8 * 128)()
            first = (ctypes.c_size_t * 2)(16, 20)
            later = (ctypes.c_size_t * 2)(24, 28)
            staging = np.full(6, -901, np.int32)
            codes = np.full(6, -901, np.int32)
            frames = ctypes.c_size_t(99)
            loop = Loop(ctypes.cast(native.prefill, ctypes.c_void_p),
                        ctypes.cast(native.frame, ctypes.c_void_p),
                        arena, 128, 0, 4, 8, first, later, 2, 3,
                        staging.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 6,
                        codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 6,
                        ctypes.pointer(frames))
            output = np.full(13, -901, np.float32)
            valid = ctypes.c_size_t(99)
            base = FrameCodec(
                ctypes.pointer(loop), ctypes.cast(native.codec, ctypes.c_void_p),
                32, 2, 3, 64, 120, 12,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 13,
                ctypes.pointer(valid))
            request = PrefixCodec(base, 4)
            self.assertEqual(run(ctypes.byref(request)), 0)
            self.assertEqual((frames.value, valid.value), (2, 8))
            np.testing.assert_array_equal(output[:8], [11] * 4 + [21] * 4)
            np.testing.assert_array_equal(output[8:], [-901] * 5)
            self.assertEqual(np.frombuffer(arena, np.int32, 6, 32).tolist(),
                             [11, 12, 21, 22, 0, 0])
            output.fill(-901)
            native.set_mode(4)
            self.assertEqual(run(ctypes.byref(request)), 0)
            self.assertEqual((frames.value, valid.value), (1, 4))
            np.testing.assert_array_equal(output[:4], [11] * 4)
            np.testing.assert_array_equal(output[4:], -901)
            output.fill(-901)
            request.samples_per_code_frame = 0
            valid.value = 99
            self.assertEqual(run(ctypes.byref(request)), -1)
            self.assertEqual(valid.value, 0)
            np.testing.assert_array_equal(output, -901)
            request.samples_per_code_frame = 5
            valid.value = 99
            native.set_mode(0)
            self.assertEqual(run(ctypes.byref(request)), -1)
            self.assertEqual(native.get_codec_calls(), 0)
            self.assertEqual(valid.value, 0)
            np.testing.assert_array_equal(output, -901)
            request.samples_per_code_frame = 4
            request.base.output_capacity = 11
            self.assertEqual(run(ctypes.byref(request)), -1)
            np.testing.assert_array_equal(output, -901)
            request.base.output_capacity = 13
            request.base.expected_code_frames = 4
            self.assertEqual(run(ctypes.byref(request)), -1)
            np.testing.assert_array_equal(output, -901)
            request.base.expected_code_frames = 3
            for mode, expected in ((1, -5), (2, -2), (3, -6)):
                native.set_mode(mode)
                valid.value = 99
                self.assertEqual(run(ctypes.byref(request)), expected)
                self.assertEqual(valid.value, 0)
                np.testing.assert_array_equal(output, -901)
            native.set_mode(0)
            self.assertEqual(run(ctypes.byref(request)), 0)
            self.assertEqual(valid.value, 8)
            np.testing.assert_array_equal(output[:8], [11] * 4 + [21] * 4)
