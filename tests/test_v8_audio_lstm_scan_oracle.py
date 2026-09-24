"""Independent PyTorch fixtures and checked native bidirectional LSTM scan."""

import ctypes
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tts/bidirectional_lstm_torch.json"
FLOAT = ctypes.c_float
POINTER = ctypes.POINTER(FLOAT)


def flat(value):
    if isinstance(value, list):
        return [item for part in value for item in flat(part)]
    return [value]


class AudioLstmScanOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        library = Path(cls.temp.name) / "libaudio_lstm_scan.so"
        exports = Path(cls.temp.name) / "exports.map"
        exports.write_text("{ global: audio_lstm_bidirectional_scan_f32; local: *; };\n")
        subprocess.run([
            "cc", "-std=c11", "-O0", "-Wall", "-Wextra", "-Werror",
            "-ffunction-sections", "-fdata-sections",
            "-shared", "-fPIC", "-I", str(ROOT / "include"),
            str(ROOT / "src/kernels/audio_kernels.c"),
            str(ROOT / "src/kernels/audio_lstm_scan.c"),
            "-o", str(library), "-Wl,--gc-sections",
            f"-Wl,--version-script={exports}", "-lm",
        ], check=True)
        cls.fn = ctypes.CDLL(str(library)).audio_lstm_bidirectional_scan_f32
        cls.fn.argtypes = [POINTER, ctypes.c_size_t] * 9 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_size_t, ctypes.c_size_t,
        ]
        cls.fn.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def buffers(self, case):
        tokens, width, hidden = (case[key] for key in
                                 ("tokens", "input_size", "hidden_size"))
        input_stride, output_stride = width + 2, 2 * hidden + 3
        source = [777.0] * (tokens * input_stride)
        for index, row in enumerate(case["input"]):
            source[index * input_stride:index * input_stride + width] = row
        output = [-999.0] * (tokens * output_stride)
        args = []
        for values in (source, case["weight_ih"], case["weight_hh"],
                       case["bias_ih"], case["bias_hh"], output,
                       [13.0] * (2 * hidden), [17.0] * (2 * hidden),
                       [19.0] * (4 * hidden)):
            data = flat(values)
            args.extend(((FLOAT * len(data))(*data), len(data)))
        # The final capacity is measured in bytes.
        args[-1] *= ctypes.sizeof(FLOAT)
        return args + [tokens, width, hidden, input_stride, output_stride]

    def test_committed_torch_fixture(self):
        fixture = json.loads(FIXTURE.read_text())
        worst = (0.0, None)
        for case in fixture["cases"]:
            with self.subTest(shape=(case["tokens"], case["input_size"],
                                     case["hidden_size"])):
                args = self.buffers(case)
                for _ in range(2):
                    self.assertEqual(self.fn(*args), 0)
                    for token, row in enumerate(case["output"]):
                        start = token * args[-1]
                        for channel, expected in enumerate(row):
                            actual = args[10][start + channel]
                            self.assertTrue(math.isfinite(actual), (token, channel))
                            error = abs(actual - expected)
                            if error > worst[0]:
                                worst = (error, (token, channel, actual, expected))
                            self.assertLessEqual(error, 2e-6,
                                                 (token, channel, actual, expected))
                        self.assertEqual(list(args[10][start + len(row):
                                                       start + args[-1]]),
                                         [-999.0] * (args[-1] - len(row)))
        self.assertLessEqual(worst[0], 2e-6, worst[1])

    def test_rejection_preserves_output(self):
        case = json.loads(FIXTURE.read_text())["cases"][1]
        baseline = self.buffers(case)
        for capacity_index in (1, 3, 5, 7, 9, 11, 13, 15, 17):
            with self.subTest(capacity_index=capacity_index):
                args = self.buffers(case)
                args[capacity_index] = 0
                self.assertEqual(self.fn(*args), -3)
                self.assertEqual(list(args[10]), list(baseline[10]))
        for geometry_index, bad_value in ((18, 0), (19, 0), (20, 0),
                                          (21, 0), (22, 0),
                                          (21, ctypes.c_size_t(-1).value),
                                          (22, ctypes.c_size_t(-1).value)):
            with self.subTest(geometry_index=geometry_index, value=bad_value):
                args = self.buffers(case)
                args[geometry_index] = bad_value
                self.assertEqual(self.fn(*args), -2)
                self.assertEqual(list(args[10]), list(baseline[10]))

    def test_exact_capacity_and_null_pointer(self):
        case = json.loads(FIXTURE.read_text())["cases"][1]
        args = self.buffers(case)
        args[1] = (case["tokens"] - 1) * args[-2] + case["input_size"]
        args[11] = (case["tokens"] - 1) * args[-1] + 2 * case["hidden_size"]
        self.assertEqual(self.fn(*args), 0)
        args = self.buffers(case)
        args[0] = None
        self.assertEqual(self.fn(*args), -1)
        self.assertEqual(list(args[10]), [-999.0] * len(args[10]))

    def test_live_torch_oracle(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"live PyTorch oracle dependency unavailable: {exc}")
        fixture = json.loads(FIXTURE.read_text())
        for case in fixture["cases"]:
            model = torch.nn.LSTM(case["input_size"], case["hidden_size"],
                                  bidirectional=True, batch_first=True).eval()
            with torch.no_grad():
                for direction, suffix in enumerate(("", "_reverse")):
                    for field in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                        getattr(model, f"{field}_l0{suffix}").copy_(
                            torch.tensor(case[field][direction], dtype=torch.float32))
                output, _ = model(torch.tensor([case["input"]], dtype=torch.float32))
            for actual, expected in zip(flat(output[0].tolist()),
                                        flat(case["output"])):
                self.assertLessEqual(abs(actual - expected), 1e-7)

    def test_live_larger_shape(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"live PyTorch oracle dependency unavailable: {exc}")
        tokens, input_size, hidden_size = 16, 128, 128
        torch.manual_seed(2917)
        model = torch.nn.LSTM(input_size, hidden_size, bidirectional=True,
                              batch_first=True).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.02)
            source = torch.randn(1, tokens, input_size) * 0.1
            output, _ = model(source)
        case = {
            "tokens": tokens, "input_size": input_size, "hidden_size": hidden_size,
            "input": source[0].tolist(), "output": output[0].tolist(),
        }
        for field in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            case[field] = [getattr(model, f"{field}_l0{suffix}").tolist()
                           for suffix in ("", "_reverse")]
        args = self.buffers(case)
        self.assertEqual(self.fn(*args), 0)
        for token, row in enumerate(case["output"]):
            for channel, expected in enumerate(row):
                actual = args[10][token * args[-1] + channel]
                self.assertTrue(math.isfinite(actual))
                self.assertLessEqual(abs(actual - expected), 2e-6,
                                     (token, channel, actual, expected))


if __name__ == "__main__":
    unittest.main()
