"""Time the pinned generated Qwen3-TTS entry by segment.

Run from the repository root with CKE_QWEN3_TTS_MODEL_DIR set. Set
CKE_QWEN3_TTS_CODEC_CONV_PROVIDER=avx2_fma to profile the optional map.
The underlying test still checks code IDs and waveform against the oracle.
"""

import ctypes
import json
import os
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import tests.test_v8_qwen3_tts_full_generated_waveform as target


_segment = target.segment_function
_compile = target.compile_entry
_callbacks = []
_segments = {index: {"calls": 0, "seconds": 0.0} for index in range(3)}
_compile_seconds = 0.0


def _timed_compile(*args, **kwargs):
    global _compile_seconds
    start = time.perf_counter()
    result = _compile(*args, **kwargs)
    _compile_seconds += time.perf_counter() - start
    return result


def _timed_segment(library, graph, index):
    native = _segment(library, graph, index)
    signature = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_uint8),
                                 ctypes.c_size_t)

    def call(arena, size):
        start = time.perf_counter()
        status = native(arena, size)
        _segments[index]["seconds"] += time.perf_counter() - start
        _segments[index]["calls"] += 1
        return status

    callback = signature(call)
    _callbacks.append((callback, native))
    return callback


def main():
    if not os.environ.get("CKE_QWEN3_TTS_MODEL_DIR"):
        raise SystemExit("CKE_QWEN3_TTS_MODEL_DIR is required")
    target.compile_entry = _timed_compile
    target.segment_function = _timed_segment
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.test_v8_qwen3_tts_full_generated_waveform."
        "Qwen3TTSFullGeneratedWaveformTest.test_pinned_natural_eos_to_waveform")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("QWEN3_TTS_STAGE_PROFILE " + json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "codec_conv_provider": os.environ.get(
            "CKE_QWEN3_TTS_CODEC_CONV_PROVIDER", "scalar"),
        "total_seconds": time.perf_counter() - start,
        "compile_seconds": _compile_seconds,
        "segments": _segments,
    }, sort_keys=True), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
