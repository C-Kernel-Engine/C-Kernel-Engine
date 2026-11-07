"""Two-op Kokoro embedding to ALBERT input projection through normal v8 codegen."""
import contextlib
import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import build_ir_v8
import build_xray_checkpoint_manifest_v8 as xray_builder
import xray_numerical_parity_v8 as xray

EXPORTER = ROOT / "version/v8/tts/export_kokoro_bump.py"
SPEC = importlib.util.spec_from_file_location("kokoro_projection_exporter", EXPORTER)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)

WEIGHTS = {
    "word": "phoneme_encoder.embeddings.word_embeddings.weight",
    "position": "phoneme_encoder.embeddings.position_embeddings.weight",
    "token_type": "phoneme_encoder.embeddings.token_type_embeddings.weight",
    "gamma": "phoneme_encoder.embeddings.LayerNorm.weight",
    "beta": "phoneme_encoder.embeddings.LayerNorm.bias",
    "projection_weight": "phoneme_encoder.encoder.embedding_hidden_mapping_in.weight",
    "projection_bias": "phoneme_encoder.encoder.embedding_hidden_mapping_in.bias",
}


class KokoroGeneratedProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        embed_path = ROOT / "tests/fixtures/tts/bert_embedding_pinned.npz"
        projection_path = ROOT / "tests/fixtures/tts/kokoro_projection_pinned.npz"
        cls.embed = dict(np.load(embed_path))
        cls.projection = dict(np.load(projection_path))
        cls.meta = json.loads(projection_path.with_suffix(".json").read_text())
        if hashlib.sha256(projection_path.read_bytes()).hexdigest() != cls.meta["fixture_sha256"]:
            raise RuntimeError("projection fixture hash mismatch")
        cls.temp = tempfile.TemporaryDirectory()
        temp = Path(cls.temp.name)
        tensors = {WEIGHTS[key]: cls.embed[key] for key in
                   ("word", "position", "token_type", "gamma", "beta")}
        tensors[WEIGHTS["projection_weight"]] = cls.projection["weight"]
        tensors[WEIGHTS["projection_bias"]] = cls.projection["bias"]
        origins = {name: {"source_name": name, "transform": "identity"} for name in tensors}
        bundle = exporter.write_bundle(temp, tensors, origins, {
            "n_token": 178, "hidden_dim": 512,
            "plbert": {"intermediate_size": 2048, "max_position_embeddings": 512,
                       "num_attention_heads": 12}},
            {"source": "pinned embedding and PyTorch projection fixtures"})
        exporter.verify_bundle(temp)
        cls.bump = (temp / "weights.bump").read_bytes()
        cls.entries = {item["name"]: item for item in bundle["entries"]}
        circuit = ROOT / "tests/fixtures/tts/kokoro_embedding_projection_generated_circuit.json"
        template = json.loads(circuit.read_text())
        source = {
            "config": {
                "model": template["name"], "arch": template["name"],
                "num_layers": 1, "embed_dim": 128, "num_heads": 1,
                "num_kv_heads": 1, "head_dim": 128, "intermediate_size": 256,
                "context_length": 36, "max_seq_len": 36, "vocab_size": 178,
                "T": 36, "C": 128, "epsilon": 1e-12,
                "activation_buffer_dtypes": {"word_ids": "i32", "type_ids": "i32"},
            },
            "entries": bundle["entries"], "quant_summary": {}, "template": template,
        }
        registry = build_ir_v8.load_kernel_registry()
        with contextlib.redirect_stdout(io.StringIO()):
            ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode="prefill")
            lower1 = build_ir_v8.generate_ir_lower_1(ir1, registry, source, "prefill")
            layout = build_ir_v8.generate_memory_layout(
                lower1, source, registry, mode="prefill", context_len=36)
            lower2 = build_ir_v8.generate_ir_lower_2(
                lower1, layout, source, registry, mode="prefill")
            call_ir = build_ir_v8.generate_ir_lower_3(lower2, mode="prefill")
        cls.layout, cls.call_ir = layout, call_ir
        (temp / "layout.json").write_text(json.dumps(layout))
        (temp / "call.json").write_text(json.dumps(call_ir))
        generated = temp / "generated.c"
        subprocess.run([
            sys.executable, str(ROOT / "version/v8/scripts/codegen_v8.py"),
            "--ir", str(temp / "call.json"), "--layout", str(temp / "layout.json"),
            "--output", str(generated),
        ], check=True, capture_output=True, text=True)
        cls.generated_c = generated.read_text()
        library = temp / "generated.so"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-shared", "-fPIC", str(generated),
            str(ROOT / "src/kernels/embedding_three_table_layer_norm.c"),
            str(ROOT / "src/kernels/linear_rows_checked.c"),
            "-I", str(ROOT / "include"), "-lm", "-o", str(library),
        ], check=True, capture_output=True, text=True)
        cls.library = library
        cls.loaded = ctypes.CDLL(str(library))
        cls.fn = cls.loaded.ck_kokoro_embedding_projection_boundary
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        cls.fn.restype = ctypes.c_int
        cls.weights = {item["name"]: item for item in layout["memory"]["weights"]["entries"]}
        cls.activations = {item["name"]: item for item in
                           layout["memory"]["activations"]["buffers"]}
        cls.arena_size = layout["memory"]["arena"]["total_size"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_checked_projection_group_tail_and_rejection(self):
        """Independent ordered FP64 reference, including distinct SIMD lanes."""
        function = self.loaded.linear_rows_checked_f32
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                             ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.c_size_t, ctypes.c_size_t]
        function.restype = ctypes.c_int
        rng = np.random.default_rng(173)
        for rows in (1, 3, 4, 5, 7, 8, 9):
            for width in (1, 7, 31):
                for biased in (False, True):
                    with self.subTest(rows=rows, width=width, bias=biased):
                        channels = 11
                        x = rng.standard_normal((rows, width + 3)).astype(np.float32)
                        w = rng.standard_normal((channels, width + 2)).astype(np.float32)
                        # Mixed magnitudes and cancellation retain scalar reduction order.
                        x[:, :width:3] *= np.float32(1e12)
                        w[:, :width:3] *= np.float32(1e-12)
                        bias = rng.standard_normal(channels).astype(np.float32)
                        output = np.full((rows, channels + 4), -777, dtype=np.float32)
                        expected = np.zeros((rows, channels), dtype=np.float64)
                        if biased:
                            expected[:] = bias
                        for c in range(width):
                            expected += (x[:, c].astype(np.float64)[:, None] *
                                         w[:, c].astype(np.float64)[None, :])
                        args = [x.ctypes.data, x.size, width + 3,
                                w.ctypes.data, w.size, width + 2,
                                bias.ctypes.data if biased else None,
                                channels if biased else 0,
                                output.ctypes.data, output.size, channels + 4,
                                rows, width, channels]
                        self.assertEqual(function(*args), 0)
                        np.testing.assert_array_equal(output[:, :channels].view(np.uint32),
                                                      expected.astype(np.float32).view(np.uint32))
                        self.assertTrue(np.all(output[:, channels:] == -777))
                        output.fill(-777)
                        x[-1, width - 1] = np.nan
                        self.assertEqual(function(*args), -1)
                        self.assertTrue(np.all(output == -777))
                        x[-1, width - 1] = 1
                        args[9] = (rows - 1) * (channels + 4) + channels - 1
                        self.assertEqual(function(*args), -1)
                        self.assertTrue(np.all(output == -777))

    def vector(self, arena, name, dtype, count):
        return np.ndarray((count,), dtype=dtype, buffer=arena,
                          offset=self.activations[name]["abs_offset"])

    def arena(self):
        raw = (ctypes.c_uint8 * (self.arena_size + 63))()
        offset = (-ctypes.addressof(raw)) & 63
        arena = (ctypes.c_uint8 * self.arena_size).from_buffer(raw, offset)
        for name, planned in self.weights.items():
            entry = self.entries[name]
            payload = self.bump[entry["file_offset"]:entry["file_offset"] + entry["size"]]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
            arena[planned["abs_offset"]:planned["abs_offset"] + len(payload)] = payload
        self.vector(arena, "word_ids", np.int32, 36)[:] = self.embed["ids"]
        self.vector(arena, "type_ids", np.int32, 36)[:] = self.embed["type_ids"]
        self.vector(arena, "embedding_output", np.float32, 4608)[:] = -777.
        self.vector(arena, "projection_output", np.float32, 27648)[:] = -777.
        return arena

    def test_generated_projection_matches_both_oracle_checkpoints(self):
        self.assertEqual(self.call_ir["errors"], [])
        self.assertEqual([op["function"] for op in self.call_ir["operations"]], [
            "embedding_three_table_layer_norm_f32", "linear_rows_checked_f32"])
        self.assertLess(self.generated_c.index("embedding_three_table_layer_norm_f32("),
                        self.generated_c.rindex("linear_rows_checked_f32("))
        arena = self.arena()
        self.assertEqual(self.fn(arena, len(arena)), 0)
        boundaries = [
            ("embedding_output", self.embed["expected"],
             "kokoro.phoneme_encoder.embeddings"),
            ("projection_output", self.projection["expected"],
             "kokoro.phoneme_encoder.input_projection"),
        ]
        report = {"torch": {"tensors": {}}, "comparisons": {}}
        for name, expected, _checkpoint in boundaries:
            actual = self.vector(arena, name, np.float32, expected.size).reshape(expected.shape)
            self.assertTrue(np.isfinite(actual).all())
            error = np.abs(actual - expected)
            self.assertLessEqual(float(np.max(error)), 3e-5)
            candidate_path = Path(self.temp.name) / f"{name}-candidate.f32"
            oracle_path = Path(self.temp.name) / f"{name}-oracle.f32"
            actual.tofile(candidate_path)
            expected.tofile(oracle_path)
            selector = f"{name}@0" if name == "projection_output" else name
            report["torch"]["tensors"][selector] = {"path": str(oracle_path),
                                                   "shape": list(expected.shape)}
            report["comparisons"][selector] = {
                "ck_path": str(candidate_path), "shape": list(expected.shape),
                "physical_shape": list(expected.shape),
                "capacity_shape": list(expected.shape), "valid_shape": list(expected.shape),
                "physical_strides": [expected.shape[1], 1]}
        runtime = xray_builder.capture_runtime_library_identity(
            self.loaded, "ck_kokoro_embedding_projection_boundary")
        candidate = xray_builder.build_manifest(
            backend="ck", call_ir=self.call_ir, tensor_report=report,
            model="kokoro_embedding_projection_generated_boundary", source="generated_native",
            phase="prefill", loaded_library=self.library, runtime_library=runtime)
        oracle = xray_builder.build_manifest(
            backend="pytorch", call_ir=self.call_ir, tensor_report=report,
            model="kokoro_embedding_projection_generated_boundary", source="pinned_capture",
            phase="prefill")
        order = [checkpoint for _name, _expected, checkpoint in boundaries]
        profile = {
            "schema": "cke.parity_profile", "schema_version": 1, "name": "kokoro_projection",
            "backend": "pytorch", "contract_schema_version": 1,
            "required_match_fields": ["checkpoint_id", "producer", "logical_layout",
                                      "axis_names", "resolved_contract_id", "kernel_id",
                                      "function"],
            "observed_storage": {"default": "fp32", "checkpoints": {}},
            "dtype_thresholds": {"fp32": {"cosine_min": 0.99999,
                                           "rmse_max": 3e-5, "relative_rmse_max": 3e-5,
                                           "max_abs_max": 3e-5, "finite_required": True}},
            "checkpoint_order": order, "interval_expansions": {}, "backend_mappings": {},
        }
        comparison = xray.compare_manifests(candidate, oracle, profile)
        self.assertEqual(comparison["status"], "pass", comparison)
        self.assertEqual(len(comparison["comparisons"]), 2)
        self.assertEqual(candidate["run"]["artifact_library"]["sha256"],
                         candidate["run"]["runtime_library"]["sha256"])
        print("TTS_GENERATED_PROJECTION_EVIDENCE " + json.dumps({
            "status": "PASS", "oracle": self.meta["oracle"],
            "checkpoints": order,
            "max_abs_errors": {item["checkpoint_id"]: item["metrics"]["max_abs"]
                               for item in comparison["comparisons"]},
            "generated_library_sha256": hashlib.sha256(
                self.library.read_bytes()).hexdigest(),
            "reproduce": "python3 -m unittest tests.test_v8_kokoro_generated_projection",
            "complete_encoder": "NOT_TESTED", "generated_waveform": "NOT_TESTED",
        }, sort_keys=True))

    def test_rejected_ids_stop_projection_and_preserve_output(self):
        arena = self.arena()
        self.vector(arena, "word_ids", np.int32, 36)[0] = 178
        self.assertNotEqual(self.fn(arena, len(arena)), 0)
        self.assertTrue(np.all(self.vector(arena, "embedding_output", np.float32, 4608) == -777.))
        self.assertTrue(np.all(self.vector(arena, "projection_output", np.float32, 27648) == -777.))

    def test_projection_failure_preserves_projection_output(self):
        arena = self.arena()
        item = self.weights[WEIGHTS["projection_weight"]]
        np.ndarray((1,), dtype=np.float32, buffer=arena,
                   offset=item["abs_offset"])[0] = np.nan
        self.assertNotEqual(self.fn(arena, len(arena)), 0)
        self.assertTrue(np.isfinite(self.vector(arena, "embedding_output", np.float32, 4608)).all())
        self.assertTrue(np.all(self.vector(arena, "projection_output", np.float32, 27648) == -777.))

    def test_repeated_requests_and_arena_bounds(self):
        arena = self.arena()
        self.assertEqual(self.fn(arena, len(arena)), 0)
        first = self.vector(arena, "projection_output", np.float32, 27648).copy()
        self.vector(arena, "word_ids", np.int32, 36)[:] = 0
        self.assertEqual(self.fn(arena, len(arena)), 0)
        self.assertFalse(np.array_equal(first,
                          self.vector(arena, "projection_output", np.float32, 27648)))
        self.vector(arena, "projection_output", np.float32, 27648)[:] = -777.
        self.assertEqual(self.fn(arena, len(arena) - 1), -2)
        self.assertTrue(np.all(self.vector(arena, "projection_output", np.float32, 27648) == -777.))

    def test_exported_bump_projection_when_available(self):
        bundle_dir = os.environ.get("CKE_KOKORO_BUMP_DIR")
        if not bundle_dir:
            self.skipTest("CKE_KOKORO_BUMP_DIR unset; exported bundle NOT_TESTED")
        bundle_dir = Path(bundle_dir)
        manifest_path = bundle_dir / "weights_manifest.json"
        self.assertEqual(hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                         self.meta["bundle_manifest_sha256"])
        entries = {entry["name"]: entry for entry in
                   exporter.verify_bundle(bundle_dir)["entries"]}
        arena = self.arena()
        with (bundle_dir / "weights.bump").open("rb") as stream:
            for name, planned in self.weights.items():
                entry = entries[name]
                self.assertEqual(entry["size"], planned["size"])
                stream.seek(entry["file_offset"])
                payload = stream.read(entry["size"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                start = planned["abs_offset"]
                arena[start:start + len(payload)] = payload
        self.assertEqual(self.fn(arena, len(arena)), 0)
        actual = self.vector(arena, "projection_output", np.float32, 27648)
        self.assertTrue(np.isfinite(actual).all())
        self.assertLessEqual(float(np.max(np.abs(
            actual - self.projection["expected"].ravel()))), 3e-5)


if __name__ == "__main__":
    unittest.main()
