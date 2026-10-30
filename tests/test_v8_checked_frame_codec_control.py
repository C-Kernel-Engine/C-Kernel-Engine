"""Cancellation and recovery at checked frame/codec boundaries."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

import numpy as np

from tests.test_v8_checked_frame_loop import Loop
from tests.test_v8_qwen3_tts_loop_codec import FrameCodec
from tests.test_v8_checked_frame_codec_prefix import PrefixCodec


ROOT = Path(__file__).resolve().parents[1]


class ControlledPrefix(ctypes.Structure):
    _fields_ = [("base", PrefixCodec), ("should_cancel", ctypes.c_void_p),
                ("cancel_context", ctypes.c_void_p)]


class CheckedFrameCodecControlTest(unittest.TestCase):
    def test_cancel_at_each_boundary_then_recover(self):
        source = r'''
#include <stdint.h>
#include <string.h>
static int cancel_at, checks, codec_calls;
void set_cancel_at(int value) { cancel_at = value; checks = codec_calls = 0; }
int get_codec_calls(void) { return codec_calls; }
int cancel(void *context) {
    (void)context;
    ++checks;
    return cancel_at && checks >= cancel_at;
}
int prefill(uint8_t *arena, size_t bytes) {
    if (bytes != 128) return -1;
    int32_t one = 1, a = 11, b = 12;
    memcpy(arena + 4, &one, 4);
    memcpy(arena + 16, &a, 4);
    memcpy(arena + 20, &b, 4);
    return 0;
}
int frame(uint8_t *arena, size_t bytes) {
    if (bytes != 128) return -1;
    int32_t step;
    memcpy(&step, arena, 4);
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
    float one = 1.0f;
    for (int i = 0; i < 12; ++i) memcpy(arena + 64 + 4*i, &one, 4);
    int32_t twelve = 12;
    memcpy(arena + 120, &twelve, 4);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / "fake.c"
            fake.write_text(source)
            library = Path(temp) / "controlled.so"
            subprocess.run([
                "cc", "-std=c11", "-O2", "-shared", "-fPIC",
                "-I", str(ROOT / "include"),
                str(ROOT / "src/runtime/checked_frame_loop.c"),
                str(ROOT / "src/runtime/checked_frame_codec.c"),
                str(fake), "-lm", "-o", str(library),
            ], check=True, capture_output=True)
            native = ctypes.CDLL(str(library))
            native.set_cancel_at.argtypes = [ctypes.c_int]
            native.get_codec_calls.restype = ctypes.c_int
            run = native.ck_run_checked_frame_codec_prefix_control
            run.argtypes = [ctypes.POINTER(ControlledPrefix)]
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
            request = ControlledPrefix(
                PrefixCodec(base, 4), ctypes.cast(native.cancel, ctypes.c_void_p), None)
            for boundary in range(1, 10):
                with self.subTest(boundary=boundary):
                    output.fill(-901)
                    frames.value = valid.value = 99
                    native.set_cancel_at(boundary)
                    self.assertEqual(run(ctypes.byref(request)), -7)
                    self.assertEqual((frames.value, valid.value), (0, 0))
                    np.testing.assert_array_equal(output, -901)
                    self.assertEqual(native.get_codec_calls(), int(boundary >= 8))
            request.should_cancel = None
            valid.value = 99
            self.assertEqual(run(ctypes.byref(request)), -1)
            self.assertEqual(valid.value, 0)
            ready = threading.Event()
            release = threading.Event()
            stop = threading.Event()
            callback_calls = 0

            @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
            def async_cancel(_context):
                nonlocal callback_calls
                callback_calls += 1
                if callback_calls == 3:
                    ready.set()
                    release.wait(5)
                return int(stop.is_set())

            request.should_cancel = ctypes.cast(async_cancel, ctypes.c_void_p)
            result = {}
            worker = threading.Thread(
                target=lambda: result.setdefault("status", run(ctypes.byref(request))))
            output.fill(-901)
            frames.value = valid.value = 99
            worker.start()
            try:
                self.assertTrue(ready.wait(5))
                stop.set()
            finally:
                release.set()
                worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result["status"], -7)
            self.assertEqual((frames.value, valid.value), (0, 0))
            np.testing.assert_array_equal(output, -901)
            request.should_cancel = ctypes.cast(native.cancel, ctypes.c_void_p)
            native.set_cancel_at(0)
            self.assertEqual(run(ctypes.byref(request)), 0)
            self.assertEqual((frames.value, valid.value), (2, 8))
            np.testing.assert_array_equal(output[:8], 1)
            np.testing.assert_array_equal(output[8:], -901)
