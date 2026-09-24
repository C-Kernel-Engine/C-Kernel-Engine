"""Independent PyTorch repeat-interleave oracle for checked duration expansion."""

import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tts" / "duration_expand_torch.json"


class AudioDurationExpandOracleTest(unittest.TestCase):
    def test_committed_torch_fixture(self):
        fixture = json.loads(FIXTURE.read_text())
        features = fixture["features"]
        durations = fixture["durations"]
        expected = fixture["output"]
        channels, tokens = len(features), len(durations)
        frames = sum(durations)
        input_stride, output_stride = fixture["input_stride"], fixture["output_stride"]
        source = (ctypes.c_float * (channels * input_stride))(*(
            v for row in features for v in row + [777.0] * (input_stride - tokens)))
        duration_buf = (ctypes.c_int32 * tokens)(*durations)
        output = (ctypes.c_float * (channels * output_stride))(*([-999.0] * (channels * output_stride)))

        with tempfile.TemporaryDirectory() as temp_dir:
            library = Path(temp_dir) / "libduration_expand.so"
            subprocess.run([
                "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                "-shared", "-fPIC", "-I", str(ROOT / "include"),
                str(ROOT / "src" / "kernels" / "audio_duration_expand.c"),
                "-o", str(library),
            ], check=True)
            native = ctypes.CDLL(str(library))
            fn = native.audio_duration_expand_channel_major_f32
            fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                           ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t]
            fn.restype = ctypes.c_int
            args = (source, len(source), channels, tokens, input_stride,
                    duration_buf, frames, output, len(output), output_stride)
            self.assertEqual(fn(*args), 0)
            for channel, row in enumerate(expected):
                self.assertEqual(list(output[channel * output_stride:][:frames]), row)
                self.assertEqual(list(output[channel * output_stride + frames:
                                             (channel + 1) * output_stride]),
                                 [-999.0] * (output_stride - frames))
            self.assertEqual(fn(*args), 0)
            self.assertEqual(fn(*args[:6], frames - 1, *args[7:]), -2)

    def test_live_torch_oracle(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"live PyTorch oracle dependency unavailable: {exc}")
        fixture = json.loads(FIXTURE.read_text())
        x = torch.tensor(fixture["features"], dtype=torch.float32)
        durations = torch.tensor(fixture["durations"], dtype=torch.int64)
        expected = torch.repeat_interleave(x, durations, dim=1).tolist()
        self.assertEqual(expected, fixture["output"])


if __name__ == "__main__":
    unittest.main()
