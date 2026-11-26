"""Generated Kokoro prefix through both first F0/noise AdaIN operations."""
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
import build_kokoro_prosody_norm_circuit as author


class KokoroGeneratedProsodyNormTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence = os.environ.get('CKE_KOKORO_PROSODY_EVIDENCE_DIR')
        if evidence:
            root = Path(evidence).resolve() / 'prosody_norm'
            root.mkdir(parents=True, exist_ok=True)
            cls.temp = SimpleNamespace(name=str(root), cleanup=lambda: None)
        else:
            cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        norm_path = ROOT / 'tests/fixtures/tts/kokoro_prosody_norm_pinned.npz'
        cls.norm_meta = json.loads(norm_path.with_suffix('.json').read_text())
        if hashlib.sha256(norm_path.read_bytes()).hexdigest() != \
                cls.norm_meta['fixture_sha256']:
            raise RuntimeError('prosody norm fixture hash mismatch')
        with np.load(norm_path) as archive:
            cls.norm = {name: archive[name].copy() for name in archive.files}
        for name, value in cls.norm.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != \
                    cls.norm_meta['array_sha256'][name]:
                raise RuntimeError(f'prosody norm tensor hash mismatch: {name}')
        shared_path = ROOT / 'tests/fixtures/tts/kokoro_prosody_shared_pinned.npz'
        with np.load(shared_path) as archive:
            cls.shared = {name: archive[name].copy() for name in archive.files}
        text_path = ROOT / 'tests/fixtures/tts/kokoro_text_encoder_pinned.npz'
        with np.load(text_path) as archive:
            text = {name: archive[name].copy() for name in archive.files}
        embedding_path = ROOT / 'tests/fixtures/tts/kokoro_text_embedding_pinned.npz'
        with np.load(embedding_path) as archive:
            weights = {'acoustic_text_encoder.embedding.weight':
                       archive['table'].copy()}
        for index in range(3):
            for kind in ('weight', 'bias'):
                weights[f'acoustic_text_encoder.conv{index}.{kind}'] = \
                    text[f'conv{index}_{kind}']
            weights[f'acoustic_text_encoder.norm{index}.weight'] = \
                text[f'norm{index}_gamma']
            weights[f'acoustic_text_encoder.norm{index}.bias'] = \
                text[f'norm{index}_beta']
        for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
            weights[f'acoustic_text_encoder.lstm.{kind}'] = text[f'lstm_{kind}']
            weights[f'duration_prosody.shared_scan.{kind}'] = cls.shared[kind]
        for branch in ('F0', 'N'):
            prefix = f'duration_prosody.{branch}.0.norm1'
            for kind in ('fc_weight', 'fc_bias', 'norm_weight', 'norm_bias'):
                weights[f'{prefix}.{kind.replace("_", ".")}'] = \
                    cls.norm[f'{branch}_{kind}']
        circuit = materialize_stage_circuit(cls.root, author)
        fixture = prepare_duration_fixture(cls.root, circuit, weights)
        for key, value in vars(fixture).items():
            setattr(cls, key, value)
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = \
            compile_native_graph(cls.root, cls.source, circuit)
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32)]
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weight_layout = {item['name']: item for item in
                             cls.layout['memory']['weights']['entries']}

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
        for name in ('prosody_shared_features', 'prosody_shared_channel',
                     'f0_norm0_style', 'n_norm0_style',
                     'f0_norm0_output', 'n_norm0_output'):
            self.view(arena, name)[:] = -91.
        return arena

    def test_generated_branches_match_independent_reference_operations(self):
        self.assertEqual(self.calls['errors'], [])
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        shared = self.view(arena, 'prosody_shared_features').reshape(128, 512)
        self.assertTrue(np.isfinite(shared[:103]).all())
        self.assertLessEqual(float(np.max(np.abs(shared[:103] -
                                                 self.shared['output']))), 1e-5)
        transposed = self.view(arena, 'prosody_shared_channel').reshape(512, 128)
        np.testing.assert_array_equal(transposed[:, :103], shared[:103].T)
        self.assertTrue(np.all(transposed[:, 103:] == -91.))
        worst = 0.
        for branch, stem in (('F0', 'f0'), ('N', 'n')):
            actual_style = self.view(arena, f'{stem}_norm0_style')
            self.assertTrue(np.isfinite(actual_style).all())
            style_error = np.max(np.abs(actual_style -
                                        self.norm[f'{branch}_style_affine']))
            self.assertLessEqual(float(style_error), 5e-5)
            output = self.view(arena,
                               f'{stem}_norm0_output').reshape(512, 128)
            self.assertTrue(np.isfinite(output[:, :103]).all())
            error = np.abs(output[:, :103] - self.norm[f'{branch}_output'])
            point = np.unravel_index(np.argmax(error), error.shape)
            self.assertLessEqual(float(error[point]), 5e-5,
                                 (branch, point, float(error[point])))
            worst = max(worst, float(error[point]))
            self.assertTrue(np.all(output[:, 103:] == -91.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.prosody-first-adain.generated-v1',
            'name': 'generated shared-LSTM to first F0/noise AdaIN stages',
            'provider': 'generated_prosody_norm_prefix', 'dtype': 'fp32',
            'direction': 'inference',
            'oracle': 'pinned-shared-pytorch28-plus-pytorch213-operations',
            'backend_version': self.norm_meta['oracle_version'],
            'status': 'pass', 'max_diff': worst, 'tolerance': 5e-5,
            'configuration': '36 phonemes; 103 valid of 128 frames; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_repeated_requests_padding_and_upstream_rejection(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        first = {name: self.view(arena, name).copy() for name in
                 ('f0_norm0_output', 'n_norm0_output')}
        for name in first:
            self.view(arena, name)[:] = -13.
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        for name in first:
            result = self.view(arena, name).reshape(512, 128)
            np.testing.assert_array_equal(result[:, :103],
                                          first[name].reshape(512, 128)[:, :103])
            self.assertTrue(np.all(result[:, 103:] == -13.))
        # A rejected duration must stop both branches before publishing A.
        arena = self.arena()
        for name, value in (('duration_prosody.duration_head.weight', 0.),
                            ('duration_prosody.duration_head.bias', 1.)):
            item = self.weight_layout[name]
            np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                       offset=item['abs_offset'])[:] = value
        frames.value = -999
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, -999)
        for name in ('f0_norm0_output', 'n_norm0_output'):
            self.assertTrue(np.all(self.view(arena, name) == -91.))

    def test_other_frame_lengths_match_independent_scan_and_norm(self):
        arena = self.arena()
        weight = self.weight_layout['duration_prosody.duration_head.weight']
        bias = self.weight_layout['duration_prosody.duration_head.bias']
        np.ndarray((weight['size'] // 4,), np.float32, buffer=arena,
                   offset=weight['abs_offset'])[:] = 0.
        biases = np.ndarray((bias['size'] // 4,), np.float32, buffer=arena,
                            offset=bias['abs_offset'])
        frames = ctypes.c_int32(-999)
        worst = 0.
        for value, expected in ((-3.2, 72), (-10., 36)):
            with self.subTest(frames=expected):
                biases[:] = value
                for name in ('prosody_shared_features',
                             'prosody_shared_channel',
                             'f0_norm0_output', 'n_norm0_output'):
                    self.view(arena, name)[:] = -91.
                frames.value = -999
                self.assertEqual(self.fn(arena, len(arena),
                                         ctypes.byref(frames)), 0)
                self.assertEqual(frames.value, expected)
                shared = self.view(arena,
                    'prosody_shared_features').reshape(128, 512)
                self.assertTrue(np.isfinite(shared[:expected]).all())
                self.assertLessEqual(float(np.max(np.abs(shared[:expected] -
                    self.norm[f'length{expected}_shared']))), 2e-5)
                for branch, stem in (('F0', 'f0'), ('N', 'n')):
                    result = self.view(arena,
                        f'{stem}_norm0_output').reshape(512, 128)
                    self.assertTrue(np.isfinite(result[:, :expected]).all())
                    error = np.abs(result[:, :expected] -
                        self.norm[f'length{expected}_{branch}_output'])
                    point = np.unravel_index(np.argmax(error), error.shape)
                    self.assertLessEqual(float(error[point]), 5e-5,
                        (expected, branch, point, float(error[point])))
                    worst = max(worst, float(error[point]))
                    self.assertTrue(np.all(result[:, expected:] == -91.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.prosody-first-adain.altered-lengths-v1',
            'name': 'independent 36/72-frame scan and first F0/noise AdaIN',
            'provider': 'generated_prosody_norm_prefix', 'dtype': 'fp32',
            'direction': 'inference',
            'oracle': 'pinned-predictor-features-plus-pytorch213-operations',
            'backend_version': self.norm_meta['oracle_version'],
            'status': 'pass', 'max_diff': worst, 'tolerance': 5e-5,
            'configuration': '36 phonemes; 36 and 72 valid of 128 frames',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_wrong_weight_detected_and_xray_sees_valid_regions(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        tensor_report = {'torch': {'tensors': {}}, 'comparisons': {}}
        points = {}
        for branch, stem in (('F0', 'f0'), ('N', 'n')):
            name = f'{stem}_norm0_output'
            point = next(item for operation in self.calls['operations']
                         for item in operation.get('semantic_checkpoints', [])
                         if item['tensor'] == name)
            self.assertEqual(point['resolved_contract_id'],
                'audio_adain_instance_norm_biased_channel_fp64_stats_fp32_affine')
            points[name] = point
            selector = name if point['layer'] < 0 else f'{name}@{point["layer"]}'
            native_path = self.root / f'{name}-native.f32'
            oracle_path = self.root / f'{name}-pytorch.f32'
            self.view(arena, name).tofile(native_path)
            self.norm[f'{branch}_output'].tofile(oracle_path)
            tensor_report['torch']['tensors'][selector] = {
                'path': str(oracle_path), 'shape': [512, 103]}
            tensor_report['comparisons'][selector] = {
                'ck_path': str(native_path), 'shape': [512, 103],
                'physical_shape': [512, 128],
                'capacity_shape': [512, 128],
                'valid_shape': [512, 103],
                'physical_strides': [128, 1]}
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_prosody_norm_bounded')
        subject = builder.build_manifest(
            backend='ck', call_ir=self.calls, tensor_report=tensor_report,
            model='kokoro_prosody_norm_bounded', source='generated_native',
            phase='prefill', loaded_library=self.library,
            runtime_library=runtime)
        oracle_manifest = builder.build_manifest(
            backend='pytorch', call_ir=self.calls,
            tensor_report=tensor_report,
            model='kokoro_prosody_norm_bounded',
            source='pinned_shared_plus_independent_ops', phase='prefill')
        for name, point in points.items():
            profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
                'name': 'kokoro_prosody_norm', 'backend': 'pytorch',
                'contract_schema_version': 1,
                'required_match_fields': ['checkpoint_id', 'producer',
                    'logical_layout', 'axis_names', 'resolved_contract_id',
                    'kernel_id', 'function'],
                'observed_storage': {'default': 'fp32', 'checkpoints': {}},
                'dtype_thresholds': {'fp32': {'cosine_min': .99999,
                    'rmse_max': 5e-5, 'relative_rmse_max': 5e-5,
                    'max_abs_max': 5e-5, 'finite_required': True}},
                'checkpoint_order': [point['id']],
                'interval_expansions': {}, 'backend_mappings': {}}
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
        # Same-shape F0/N weights are a realistic wrong-binding control.
        f0 = self.weight_layout['duration_prosody.F0.0.norm1.fc.weight']
        n = self.weight_layout['duration_prosody.N.0.norm1.fc.weight']
        arena[f0['abs_offset']:f0['abs_offset'] + f0['size']] = \
            arena[n['abs_offset']:n['abs_offset'] + n['size']]
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        changed = self.view(arena, 'f0_norm0_output').reshape(512, 128)[:, :103]
        self.assertGreater(float(np.max(np.abs(changed -
            self.norm['F0_output']))), 5e-5)

    def test_wrong_style_edge_is_detected_by_independent_output(self):
        graph = copy.deepcopy(self.source['template'])
        f0 = graph['block_types']['prosody_norm']['footer'][0]
        f0['graph_slots']['inputs']['style_affine'] = 'n_norm0_style'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'wrong-style-edge.json'
            path.write_text(json.dumps(graph))
            source = copy.deepcopy(self.source)
            source['template'] = graph
            layout, calls, _library, _loaded, function = \
                compile_native_graph(root, source, path)
            self.assertEqual(calls['errors'], [])
            function.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                                 ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int32)]
            arena = populated_duration_arena(layout, self.entries, self.bump,
                self.encoder, self.duration)
            frames = ctypes.c_int32(-999)
            self.assertEqual(function(arena, len(arena),
                                      ctypes.byref(frames)), 0)
            self.assertEqual(frames.value, 103)
            buffers = {item['name']: item for item in
                       layout['memory']['activations']['buffers']}
            item = buffers['f0_norm0_output']
            result = np.ndarray((512, 128), np.float32, buffer=arena,
                                offset=item['abs_offset'])[:, :103]
            self.assertGreater(float(np.max(np.abs(result -
                self.norm['F0_output']))), 5e-5)

    def test_failed_first_branch_stops_second_without_publishing_extent(self):
        arena = self.arena()
        item = self.weight_layout['duration_prosody.F0.0.norm1.fc.weight']
        np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[0] = np.nan
        frames = ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, -999)
        self.assertTrue(np.all(self.view(arena, 'f0_norm0_output') == -91.))
        self.assertTrue(np.all(self.view(arena, 'n_norm0_output') == -91.))


if __name__ == '__main__':
    unittest.main()
