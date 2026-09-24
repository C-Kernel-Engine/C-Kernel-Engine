"""Executable generated-C fixture for bounded producer -> expansion -> consumer."""

import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/scripts/codegen_checked_calls_v8.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("codegen_checked_calls_v8", SCRIPT)
codegen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codegen)


def _call(kernel_id, expressions, *, op_id, produced=None, consumed=None):
    kernel = json.loads((ROOT / "version/v8/kernel_maps" / f"{kernel_id}.json").read_text())
    params = kernel["call_abi"]["params"]
    assert len(params) == len(expressions)
    return {
        "function": kernel["impl"]["function"],
        "template_op_id": op_id,
        "call_abi": {"kernel_id": kernel_id, "version": 1},
        "args": [{"name": param["name"], "source": param["source"],
                  "expr": expr} for param, expr in zip(params, expressions)],
        "produces_runtime_lengths": produced or {},
        "consumes_runtime_lengths": consumed or [],
        "returns_status": True,
        "errors": [],
    }


def _fixture(*, include_istft=False):
    document = {
        "entry": {
            "function": "ck_test_expansion_graph",
            "params": [
                {"c_type": "const int32_t *", "name": "durations"},
                {"c_type": "size_t", "name": "duration_elements"},
                {"c_type": "size_t", "name": "phoneme_count"},
                {"c_type": "const float *", "name": "features"},
                {"c_type": "float *", "name": "expanded"},
                {"c_type": "float *", "name": "consumed"},
                {"c_type": "int32_t *", "name": "out_frames"},
            ],
            "runtime_length_outputs": {"expanded_frames": "out_frames"},
        },
        "runtime_extent_contract": {
            "runtime_lengths": {"expanded_frames": {
                "producer": "extent", "result": "valid_extent",
                "capacity": 6, "allow_zero": False,
            }},
            "runtime_views": {
                "expanded": {"buffer": "expanded", "length": "expanded_frames",
                             "channels": 2, "physical_stride": 8,
                             "element_bytes": 4, "required_bytes": 56,
                             "buffer_bytes": 64},
            },
            "runtime_constants": {},
        },
        "operations": [
            _call("runtime_extent_sum_i32", [
                "durations", "duration_elements", "phoneme_count", "1", "6",
                "0", "6", "&runtime_extents.expanded_frames",
            ], op_id="extent", produced={"expanded_frames": "valid_extent"}),
            _call("audio_duration_expand_channel_major_f32", [
                "features", "8", "2", "phoneme_count", "4", "durations",
                "(size_t)runtime_extents.expanded_frames", "expanded", "16", "8",
            ], op_id="expand", consumed=["expanded_frames"]),
            _call("runtime_copy_valid_f32", [
                "expanded", "16", "2", "(size_t)runtime_extents.expanded_frames",
                "8", "consumed", "16", "8",
            ], op_id="consume", consumed=["expanded_frames"]),
        ],
    }
    if include_istft:
        document["entry"]["params"].extend([
            {"c_type": "const float *", "name": "magnitude"},
            {"c_type": "const float *", "name": "phase"},
            {"c_type": "float *", "name": "waveform"},
            {"c_type": "float *", "name": "istft_scratch"},
        ])
        document["operations"].append(_call("audio_istft_mag_phase_f32", [
            "magnitude", "phase", "77", "7", "20", "5",
            "waveform", "30", "istft_scratch", "120",
        ], op_id="reconstruct"))
    return document


class CheckedCallCodegenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        source = Path(cls.temp.name) / "generated.c"
        source.write_text(codegen.emit_checked_calls(_fixture(), ROOT))
        library = Path(cls.temp.name) / "generated.so"
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", str(source),
            str(ROOT / "src/kernels/runtime_extent.c"),
            str(ROOT / "src/kernels/audio_duration_expand.c"),
            "-I", str(ROOT / "include"), "-o", str(library),
        ], check=True)
        cls.function = ctypes.CDLL(str(library)).ck_test_expansion_graph
        cls.function.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                                 ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_int32)]
        cls.function.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, durations, *, count=None):
        values = (ctypes.c_int32 * len(durations))(*durations)
        features = (ctypes.c_float * 8)(10, 20, 30, -777, 40, 50, 60, -777)
        expanded = (ctypes.c_float * 16)(*([-99] * 16))
        consumed = (ctypes.c_float * 16)(*([-88] * 16))
        out_frames = ctypes.c_int32(0xC0FFEE)
        status = self.function(values, len(durations),
                               len(durations) if count is None else count,
                               features, expanded, consumed, ctypes.byref(out_frames))
        return status, list(expanded), list(consumed), out_frames.value

    def test_generated_graph_oracle_and_padding(self):
        for durations in ([1, 2, 1], [2, 2, 2], [3, 1, 1], [1, 1, 1]):
            with self.subTest(durations=durations):
                status, expanded, consumed, out_frames = self.invoke(durations)
                self.assertEqual(status, 0)
                self.assertEqual(out_frames, sum(durations))
                expected = [-99.0] * 16
                final = [-88.0] * 16
                for channel in range(2):
                    values = [10, 20, 30] if channel == 0 else [40, 50, 60]
                    row = [value for value, repeat in zip(values, durations)
                           for _ in range(repeat)]
                    expected[channel * 8:channel * 8 + len(row)] = row
                    final[channel * 8:channel * 8 + len(row)] = row
                self.assertEqual(expanded, expected)
                self.assertEqual(consumed, final)

    def test_failure_stops_downstream_and_leaves_output_unchanged(self):
        for durations, count, error in [
            ([], None, -1),
            ([3, 3, 3], None, -2),
            ([0, 1, 1], None, -1),
            ([1, -1, 1], None, -1),
            ([1, 1, 1], 9, -2),
        ]:
            with self.subTest(durations=durations, count=count):
                status, expanded, consumed, out_frames = self.invoke(durations, count=count)
                self.assertEqual(status, error)
                self.assertEqual(out_frames, 0xC0FFEE)
                self.assertEqual(expanded, [-99.0] * 16)
                self.assertEqual(consumed, [-88.0] * 16)

    def test_missing_length_result_rejected_before_model_writes(self):
        values = (ctypes.c_int32 * 3)(1, 2, 1)
        features = (ctypes.c_float * 8)(10, 20, 30, -777, 40, 50, 60, -777)
        expanded = (ctypes.c_float * 16)(*([-99] * 16))
        consumed = (ctypes.c_float * 16)(*([-88] * 16))
        self.assertEqual(self.function(values, 3, 3, features, expanded, consumed,
                                       None), -1)
        self.assertEqual(list(expanded), [-99.0] * 16)
        self.assertEqual(list(consumed), [-88.0] * 16)

    def test_codegen_rejects_unresolved_and_misordered_calls(self):
        for mutate in (
            lambda fixture: fixture["operations"][1]["args"].reverse(),
            lambda fixture: fixture["operations"][0].update(errors=["missing buffer"]),
            lambda fixture: fixture["operations"][0].update(function="unresolved"),
            lambda fixture: fixture["operations"][0]["args"][-1].update(expr="elsewhere"),
        ):
            fixture = _fixture()
            mutate(fixture)
            with self.assertRaises(codegen.CheckedCallCodegenError):
                codegen.emit_checked_calls(fixture, ROOT)

    def test_generated_graph_invokes_istft_and_matches_committed_torch_oracle(self):
        fixture = json.loads((ROOT / "tests/fixtures/tts/istft_torch20_hop5.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "generated.c"
            source.write_text(codegen.emit_checked_calls(_fixture(include_istft=True), ROOT))
            library = Path(temporary) / "generated.so"
            subprocess.run([
                "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
                "-shared", "-fPIC", str(source),
                str(ROOT / "src/kernels/runtime_extent.c"),
                str(ROOT / "src/kernels/audio_duration_expand.c"),
                str(ROOT / "src/kernels/audio_istft_mag_phase.c"),
                "-I", str(ROOT / "include"), "-lm", "-o", str(library),
            ], check=True)
            function = ctypes.CDLL(str(library)).ck_test_expansion_graph
            fptr = ctypes.POINTER(ctypes.c_float)
            function.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                                 ctypes.c_size_t, fptr, fptr, fptr,
                                 ctypes.POINTER(ctypes.c_int32), fptr, fptr, fptr, fptr]
            function.restype = ctypes.c_int
            durations = (ctypes.c_int32 * 3)(1, 2, 1)
            features = (ctypes.c_float * 8)(10, 20, 30, -777, 40, 50, 60, -777)
            expanded = (ctypes.c_float * 16)(*([-99] * 16))
            consumed = (ctypes.c_float * 16)(*([-88] * 16))
            magnitude = (ctypes.c_float * 77)(*[
                value for row in fixture["magnitude"] for value in row])
            phase = (ctypes.c_float * 77)(*[
                value for row in fixture["phase"] for value in row])
            waveform = (ctypes.c_float * 30)(*([-999] * 30))
            scratch = (ctypes.c_float * 120)()
            out_frames = ctypes.c_int32(-1)
            status = function(durations, 3, 3, features, expanded, consumed,
                              ctypes.byref(out_frames),
                              magnitude, phase, waveform, scratch)
            self.assertEqual(status, 0)
            self.assertEqual(out_frames.value, 4)
            errors = [abs(actual - expected) for actual, expected in
                      zip(waveform, fixture["waveform"])]
            worst = max(range(len(errors)), key=errors.__getitem__)
            self.assertLessEqual(errors[worst], fixture["atol"],
                                 f"sample {worst} error {errors[worst]}")


if __name__ == "__main__":
    unittest.main()
