"""Generated Kokoro duration expansion through the shared prosody BiLSTM."""
import ctypes
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests import test_v8_kokoro_generated_albert_layer as xray_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_prosody_shared_circuit as author


class KokoroGeneratedProsodySharedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence = os.environ.get('CKE_KOKORO_PROSODY_EVIDENCE_DIR')
        if evidence:
            root = Path(evidence).resolve() / 'prosody_shared'
            root.mkdir(parents=True, exist_ok=True)
            cls.temp = SimpleNamespace(name=str(root), cleanup=lambda: None)
        else:
            cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.root = root
        path = ROOT / 'tests/fixtures/tts/kokoro_prosody_shared_pinned.npz'
        metadata = json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['fixture_sha256']:
            raise RuntimeError('prosody fixture hash mismatch')
        with np.load(path) as archive:
            cls.prosody = {name: archive[name].copy() for name in archive.files}
        for name, value in cls.prosody.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != metadata['array_sha256'][name]:
                raise RuntimeError(f'prosody array hash mismatch: {name}')
        text_path = ROOT / 'tests/fixtures/tts/kokoro_text_encoder_pinned.npz'
        with np.load(text_path) as archive:
            text = {name: archive[name].copy() for name in archive.files}
        embedding_path = ROOT / 'tests/fixtures/tts/kokoro_text_embedding_pinned.npz'
        with np.load(embedding_path) as archive:
            weights = {'acoustic_text_encoder.embedding.weight': archive['table'].copy()}
        for index in range(3):
            for kind in ('weight', 'bias'):
                weights[f'acoustic_text_encoder.conv{index}.{kind}'] = text[f'conv{index}_{kind}']
            weights[f'acoustic_text_encoder.norm{index}.weight'] = text[f'norm{index}_gamma']
            weights[f'acoustic_text_encoder.norm{index}.bias'] = text[f'norm{index}_beta']
        for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
            weights[f'acoustic_text_encoder.lstm.{kind}'] = text[f'lstm_{kind}']
            weights[f'duration_prosody.shared_scan.{kind}'] = cls.prosody[kind]
        circuit = materialize_stage_circuit(root, author)
        fixture = prepare_duration_fixture(root, circuit, weights)
        for key, value in vars(fixture).items():
            setattr(cls, key, value)
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = compile_native_graph(
            root, cls.source, circuit)
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32)]
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, dtype=np.float32):
        item = self.buffers[name]
        return np.ndarray((item['size'] // np.dtype(dtype).itemsize,), dtype,
                          buffer=arena, offset=item['abs_offset'])

    def arena(self):
        arena = populated_duration_arena(self.layout, self.entries, self.bump,
                                         self.encoder, self.duration)
        for name in ('duration_expanded', 'prosody_input_token',
                     'prosody_shared_features'):
            self.view(arena, name)[:] = -91.
        return arena

    def test_generated_shared_scan_matches_pinned_oracle(self):
        self.assertEqual(self.calls['errors'], [])
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        duration = self.view(arena, 'duration_expanded').reshape(640, 128)
        durations = self.view(arena, 'runtime_values', np.int32)[:36]
        expected_input = np.repeat(self.duration['predictor_output'], durations,
                                   axis=0)
        np.testing.assert_allclose(duration[:, :103].T, expected_input,
                                   rtol=0, atol=3e-5)
        transposed = self.view(arena, 'prosody_input_token').reshape(128, 640)
        np.testing.assert_allclose(transposed[:103], expected_input,
                                   rtol=0, atol=3e-5)
        self.assertTrue(np.all(transposed[103:] == -91.))
        result = self.view(arena, 'prosody_shared_features').reshape(128, 512)
        self.assertTrue(np.isfinite(result[:103]).all())
        error = np.abs(result[:103] - self.prosody['output'])
        worst = np.unravel_index(np.argmax(error), error.shape)
        self.assertLessEqual(float(error[worst]), 1e-5,
                             (worst, float(error[worst])))
        self.assertTrue(np.all(result[103:] == -91.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.prosody-shared.generated-v1',
            'name': 'generated duration expansion and shared prosody BiLSTM',
            'provider': 'generated_prosody_shared', 'dtype': 'fp32',
            'direction': 'inference', 'oracle': 'pinned-pytorch',
            'backend_version': '2.8.0+cpu', 'status': 'pass',
            'max_diff': float(error[worst]),
            'worst_index': [int(index) for index in worst],
            'tolerance': 1e-5,
            'configuration': '36 phonemes; 103 valid of 128 frames; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id(),
        }))

    def test_repeated_request_resets_state_and_preserves_padding(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        first = self.view(arena, 'prosody_shared_features').reshape(128, 512)[:103].copy()
        self.view(arena, 'prosody_shared_features')[:] = -13.
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        result = self.view(arena, 'prosody_shared_features').reshape(128, 512)
        np.testing.assert_array_equal(result[:103], first)
        self.assertTrue(np.all(result[103:] == -13.))

    def test_one_deployment_accepts_two_other_checked_frame_lengths(self):
        arena = self.arena()
        head = {item['name']: item for item in
                self.layout['memory']['weights']['entries']}
        weight = head['duration_prosody.duration_head.weight']
        bias = head['duration_prosody.duration_head.bias']
        np.ndarray((weight['size'] // 4,), np.float32, buffer=arena,
                   offset=weight['abs_offset'])[:] = 0.
        biases = np.ndarray((bias['size'] // 4,), np.float32, buffer=arena,
                            offset=bias['abs_offset'])
        frames = ctypes.c_int32(-999)
        for value, expected in ((-3.2, 72), (-10., 36)):
            biases[:] = value
            self.view(arena, 'prosody_input_token')[:] = -91.
            self.view(arena, 'prosody_shared_features')[:] = -91.
            frames.value = -999
            self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
            self.assertEqual(frames.value, expected)
            self.assertTrue(np.isfinite(self.view(arena,
                'prosody_shared_features').reshape(128, 512)[:expected]).all())
            self.assertTrue(np.all(self.view(arena,
                'prosody_input_token').reshape(128, 640)[expected:] == -91.))
            self.assertTrue(np.all(self.view(arena,
                'prosody_shared_features').reshape(128, 512)[expected:] == -91.))

    def test_failed_duration_stops_prosody_without_publishing_length(self):
        arena = self.arena()
        for name, value in (('duration_prosody.duration_head.weight', 0.),
                            ('duration_prosody.duration_head.bias', 1.)):
            weight = next(item for item in self.layout['memory']['weights']['entries']
                          if item['name'] == name)
            np.ndarray((weight['size'] // 4,), np.float32, buffer=arena,
                       offset=weight['abs_offset'])[:] = value
        frames = ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, -999)
        self.assertTrue(np.all(self.view(arena, 'prosody_input_token') == -91.))
        self.assertTrue(np.all(self.view(arena, 'prosody_shared_features') == -91.))

    def test_xray_records_selected_contracts_and_valid_frame_regions(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        expected_input = np.repeat(self.duration['predictor_output'],
                                   self.duration['durations'], axis=0)
        expected = {
            'prosody_input_token': (expected_input, (103, 640), (128, 640), 3e-5),
            'prosody_shared_features': (self.prosody['output'],
                                        (103, 512), (128, 512), 1e-5),
        }
        tensor_report = {'torch': {'tensors': {}}, 'comparisons': {}}
        points = {}
        for name, (oracle, valid, physical, _limit) in expected.items():
            point = next(item for op in self.calls['operations']
                         for item in op.get('semantic_checkpoints', [])
                         if item['tensor'] == name)
            self.assertNotEqual(point['resolved_contract_id'], 'unresolved')
            points[name] = point
            selector = name if point['layer'] < 0 else f'{name}@{point["layer"]}'
            native_path = self.root / f'{name}-native.f32'
            oracle_path = self.root / f'{name}-pytorch.f32'
            self.view(arena, name).tofile(native_path)
            oracle.tofile(oracle_path)
            tensor_report['torch']['tensors'][selector] = {
                'path': str(oracle_path), 'shape': list(valid)}
            tensor_report['comparisons'][selector] = {
                'ck_path': str(native_path), 'shape': list(valid),
                'physical_shape': list(physical),
                'capacity_shape': list(physical), 'valid_shape': list(valid),
                'physical_strides': [physical[1], 1]}
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_prosody_shared_bounded')
        subject = builder.build_manifest(
            backend='ck', call_ir=self.calls, tensor_report=tensor_report,
            model='kokoro_prosody_shared_bounded', source='generated_native',
            phase='prefill', loaded_library=self.library, runtime_library=runtime)
        oracle_manifest = builder.build_manifest(
            backend='pytorch', call_ir=self.calls, tensor_report=tensor_report,
            model='kokoro_prosody_shared_bounded', source='pinned_capture',
            phase='prefill')
        for name, (_reference, _valid, _physical, limit) in expected.items():
            point = points[name]
            profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
                'name': 'kokoro_prosody_shared', 'backend': 'pytorch',
                'contract_schema_version': 1,
                'required_match_fields': ['checkpoint_id', 'producer',
                    'logical_layout', 'axis_names', 'resolved_contract_id',
                    'kernel_id', 'function'],
                'observed_storage': {'default': 'fp32', 'checkpoints': {}},
                'dtype_thresholds': {'fp32': {'cosine_min': 0.99999,
                    'rmse_max': limit, 'relative_rmse_max': limit,
                    'max_abs_max': limit, 'finite_required': True}},
                'checkpoint_order': [point['id']], 'interval_expansions': {},
                'backend_mappings': {}}
            one = lambda manifest: {**manifest, 'checkpoints': [item for item in
                manifest['checkpoints'] if item['checkpoint_id'] == point['id']]}
            report = xray_support.xray.compare_manifests(
                one(subject), one(oracle_manifest), profile)
            self.assertEqual(report['status'], 'pass', (name, report))
            (self.root / f'{name}-xray.json').write_text(
                json.dumps(report, indent=2))
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        (self.root / 'native-checkpoints.json').write_text(
            json.dumps(subject, indent=2))

    def test_wrong_edge_rejected_and_wrong_weight_detected_by_oracle(self):
        graph = copy.deepcopy(self.source['template'])
        graph['block_types']['prosody_shared']['body']['ops'][0][
            'graph_slots']['inputs']['input'] = 'duration_expanded'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'wrong-edge.json'
            path.write_text(json.dumps(graph))
            source = copy.deepcopy(self.source)
            source['template'] = graph
            with self.assertRaisesRegex(RuntimeError,
                                        'HARD CALL CONSTANT FAULT'):
                compile_native_graph(root, source, path)
        arena = self.arena()
        weight = next(item for item in self.layout['memory']['weights']['entries']
                      if item['name'] == 'duration_prosody.shared_scan.weight_ih')
        np.ndarray((weight['size'] // 4,), np.float32, buffer=arena,
                   offset=weight['abs_offset'])[:] = 0.
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        result = self.view(arena, 'prosody_shared_features').reshape(128, 512)[:103]
        self.assertGreater(float(np.max(np.abs(
            result - self.prosody['output']))), 1e-3)


if __name__ == '__main__':
    unittest.main()
