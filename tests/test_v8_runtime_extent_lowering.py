"""Full v8 circuit -> planner -> call IR -> generated-C bounded graph fixture."""

import contextlib
import ctypes
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import build_ir_v8
import codegen_checked_calls_v8


def manifest():
    path = ROOT / "tests/fixtures/tts/runtime_extent_circuit.json"
    return path, {
        "config": {
            "model": "runtime_extent_synthetic", "arch": "runtime_extent_synthetic",
            "num_layers": 1, "embed_dim": 4, "num_heads": 1,
            "num_kv_heads": 1, "head_dim": 4, "intermediate_size": 8,
            "context_length": 3, "max_seq_len": 3, "vocab_size": 8,
            "activation_buffer_dtypes": {
                "runtime_values": "i32", "runtime_valid_extent": "i32",
            },
        },
        "entries": [], "quant_summary": {},
        "template": json.loads(path.read_text(encoding="utf-8")),
    }


def lower_fixture():
    path, source = manifest()
    registry = build_ir_v8.load_kernel_registry()
    with contextlib.redirect_stdout(io.StringIO()):
        ir1 = build_ir_v8.build_ir1_direct(source, path, mode="prefill")
        lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
        layout = build_ir_v8.generate_memory_layout(
            lower1, source, registry, mode="prefill", context_len=3)
        lower2 = build_ir_v8.generate_ir_lower_2(
            lower1, layout, source, registry, mode="prefill")
        call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
    return source, layout, call_ir


class RuntimeExtentLoweringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.layout, cls.call_ir = lower_fixture()
        cls.temp = tempfile.TemporaryDirectory()
        generated = Path(cls.temp.name) / "generated.c"
        call_path = Path(cls.temp.name) / "call.json"
        layout_path = Path(cls.temp.name) / "layout.json"
        call_path.write_text(json.dumps(cls.call_ir), encoding="utf-8")
        layout_path.write_text(json.dumps(cls.layout), encoding="utf-8")
        subprocess.run([
            sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
            "--ir", str(call_path), "--layout", str(layout_path),
            "--output", str(generated),
        ], check=True, capture_output=True, text=True)
        library = Path(cls.temp.name) / "generated.so"
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", str(generated),
            str(ROOT / "src/kernels/runtime_extent.c"),
            str(ROOT / "src/kernels/audio_duration_expand.c"),
            str(ROOT / "src/kernels/audio_istft_mag_phase.c"),
            "-I", str(ROOT / "include"), "-lm", "-o", str(library),
        ], check=True)
        cls.function = ctypes.CDLL(str(library)).ck_test_planned_graph
        cls.function.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int32)]
        cls.function.restype = ctypes.c_int
        cls.buffers = {item["name"]: item for item in
                       cls.layout["memory"]["activations"]["buffers"]}
        cls.arena_size = cls.layout["memory"]["arena"]["total_size"]
        cls.istft_fixture = json.loads((ROOT / "tests/fixtures/tts/istft_torch20_hop5.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _array(self, arena, name, element_type, length):
        offset = self.buffers[name]["abs_offset"]
        return (element_type * length).from_buffer(arena, offset)

    def _arena(self, durations):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        offset = (-ctypes.addressof(raw)) & 63
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(raw, offset)
        self._array(arena, "runtime_values", ctypes.c_int32, 3)[:] = durations
        self._array(arena, "audio_features", ctypes.c_float, 8)[:] = (
            10, 20, 30, -777, 40, 50, 60, -777)
        self._array(arena, "audio_expanded", ctypes.c_float, 16)[:] = [-99] * 16
        self._array(arena, "runtime_valid_copy", ctypes.c_float, 16)[:] = [-88] * 16
        self._array(arena, "audio_waveform", ctypes.c_float, 30)[:] = [-999] * 30
        self._array(arena, "audio_magnitude", ctypes.c_float, 77)[:] = [
            item for row in self.istft_fixture["magnitude"] for item in row]
        self._array(arena, "audio_phase", ctypes.c_float, 77)[:] = [
            item for row in self.istft_fixture["phase"] for item in row]
        return arena

    def test_full_lowered_graph_oracle_and_physical_padding(self):
        self.assertEqual([op["function"] for op in self.call_ir["operations"]], [
            "ck_runtime_sum_i32_checked",
            "audio_duration_expand_channel_major_f32",
            "ck_runtime_copy_valid_f32",
            "audio_istft_mag_phase_f32",
        ])
        self.assertTrue(all(not op["errors"] for op in self.call_ir["operations"]))
        for durations in ((1, 2, 1), (2, 2, 2)):
            arena = self._arena(durations)
            out_frames = ctypes.c_int32(-1)
            self.assertEqual(self.function(arena, len(arena), ctypes.byref(out_frames)), 0)
            self.assertEqual(out_frames.value, sum(durations))
            expanded = list(self._array(arena, "audio_expanded", ctypes.c_float, 16))
            copied = list(self._array(arena, "runtime_valid_copy", ctypes.c_float, 16))
            for channel in range(2):
                values = (10, 20, 30) if channel == 0 else (40, 50, 60)
                expected = [value for value, count in zip(values, durations)
                            for _ in range(count)]
                self.assertEqual(expanded[channel * 8:channel * 8 + len(expected)], expected)
                self.assertEqual(copied[channel * 8:channel * 8 + len(expected)], expected)
                self.assertEqual(expanded[channel * 8 + len(expected):(channel + 1) * 8],
                                 [-99] * (8 - len(expected)))
                self.assertEqual(copied[channel * 8 + len(expected):(channel + 1) * 8],
                                 [-88] * (8 - len(expected)))
            waveform = self._array(arena, "audio_waveform", ctypes.c_float, 30)
            errors = [abs(actual - expected) for actual, expected in
                      zip(waveform, self.istft_fixture["waveform"])]
            worst = max(range(30), key=errors.__getitem__)
            self.assertLessEqual(errors[worst], self.istft_fixture["atol"])

    def test_over_capacity_stops_all_downstream_ops(self):
        arena = self._arena((3, 3, 3))
        out_frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(out_frames)), -2)
        self.assertEqual(out_frames.value, -1)
        self.assertEqual(list(self._array(arena, "audio_expanded", ctypes.c_float, 16)),
                         [-99] * 16)
        self.assertEqual(list(self._array(arena, "runtime_valid_copy", ctypes.c_float, 16)),
                         [-88] * 16)
        self.assertEqual(list(self._array(arena, "audio_waveform", ctypes.c_float, 30)),
                         [-999] * 30)

    def test_repeated_requests_bind_new_valid_extent_on_same_arena(self):
        arena = self._arena((2, 2, 2))
        out_frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(out_frames)), 0)
        self.assertEqual(out_frames.value, 6)
        self._array(arena, "runtime_values", ctypes.c_int32, 3)[:] = (1, 1, 1)
        out_frames.value = -1
        self.assertEqual(self.function(arena, len(arena), ctypes.byref(out_frames)), 0)
        self.assertEqual(out_frames.value, 3)
        copied = list(self._array(arena, "runtime_valid_copy", ctypes.c_float, 16))
        self.assertEqual(copied[:3], [10, 20, 30])
        self.assertEqual(copied[8:11], [40, 50, 60])

    def test_undersized_planned_arena_is_rejected_before_write(self):
        arena = self._arena((1, 2, 1))
        out_frames = ctypes.c_int32(-1)
        self.assertEqual(self.function(arena, len(arena) - 1,
                                       ctypes.byref(out_frames)), -2)
        self.assertEqual(out_frames.value, -1)
        self.assertEqual(list(self._array(arena, "audio_expanded", ctypes.c_float, 16)),
                         [-99] * 16)

    def test_misaligned_arena_is_rejected_before_provider_write(self):
        arena = self._arena((1, 2, 1))
        raw = (ctypes.c_uint8 * (self.arena_size + 1))()
        ctypes.memmove(ctypes.addressof(raw) + 1, arena, self.arena_size)
        misaligned = (ctypes.c_uint8 * self.arena_size).from_buffer(raw, 1)
        if ctypes.addressof(misaligned) % 64 == 0:
            misaligned = (ctypes.c_uint8 * self.arena_size).from_buffer(raw, 0)
        out_frames = ctypes.c_int32(-1)
        before = bytes(misaligned)
        self.assertEqual(self.function(misaligned, self.arena_size,
                                       ctypes.byref(out_frames)), -2)
        self.assertEqual(out_frames.value, -1)
        self.assertEqual(bytes(misaligned), before)

    def test_declared_stronger_arena_alignment_is_emitted(self):
        call_ir = json.loads(json.dumps(self.call_ir))
        call_ir["memory"]["arena"]["alignment"] = 128
        source = codegen_checked_calls_v8.emit_checked_calls(call_ir, ROOT)
        self.assertIn("(uintptr_t)arena & 127u", source)
        call_ir["memory"]["arena"]["alignment"] = 96
        with self.assertRaisesRegex(codegen_checked_calls_v8.CheckedCallCodegenError,
                                    "alignment must be a positive power of two"):
            codegen_checked_calls_v8.emit_checked_calls(call_ir, ROOT)

    def test_native_entry_and_extent_contract_survive_lowering(self):
        self.assertEqual(self.call_ir["entry"], self.source["template"]["native_entry"])
        self.assertEqual(self.call_ir["runtime_extent_contract"],
                         self.source["config"]["runtime_extent_contract"])

    def _run_codegen_command(self, call_ir, *extra_args):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            call_path = root / "call.json"
            layout_path = root / "layout.json"
            output_path = root / "generated.c"
            companion_path = root / "companion.json"
            call_path.write_text(json.dumps(call_ir), encoding="utf-8")
            layout_path.write_text(json.dumps(self.layout), encoding="utf-8")
            companion_path.write_text("{}", encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
                "--ir", str(call_path), "--layout", str(layout_path),
                "--output", str(output_path),
            ]
            command.extend(str(companion_path) if arg == "COMPANION" else arg
                           for arg in extra_args)
            result = subprocess.run(command, capture_output=True, text=True)
            return result, output_path.exists()

    def test_normal_codegen_rejects_missing_circuit_entry(self):
        call_ir = json.loads(json.dumps(self.call_ir))
        call_ir.pop("entry")
        result, emitted = self._run_codegen_command(call_ir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bounded call IR is missing its circuit-declared native entry",
                      result.stderr)
        self.assertFalse(emitted)

    def test_normal_codegen_rejects_decoder_companion_artifacts(self):
        for flag in ("--prefill", "--prefill-layout", "--init",
                     "--generation-config"):
            with self.subTest(flag=flag):
                result, emitted = self._run_codegen_command(
                    self.call_ir, flag, "COMPANION")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("bounded native entry does not accept decoder companion artifacts",
                              result.stderr)
                self.assertFalse(emitted)


if __name__ == "__main__":
    unittest.main()
