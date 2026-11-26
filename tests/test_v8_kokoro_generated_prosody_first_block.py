"""Generated Kokoro prefix through both first complete F0/noise residual blocks."""
import ctypes
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests import test_v8_kokoro_generated_albert_layer as xray_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_prosody_first_block_circuit as author


class KokoroGeneratedProsodyFirstBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        paths = {key: ROOT / f'tests/fixtures/tts/kokoro_{key}_pinned.npz'
                 for key in ('prosody_first_block', 'prosody_norm',
                             'prosody_shared', 'text_encoder', 'text_embedding')}
        for key, path in paths.items():
            with np.load(path) as archive:
                setattr(cls, key, {name: archive[name].copy()
                                   for name in archive.files})
        meta = json.loads(paths['prosody_first_block'].with_suffix('.json').read_text())
        if hashlib.sha256(paths['prosody_first_block'].read_bytes()).hexdigest() != \
                meta['fixture_sha256']:
            raise RuntimeError('first block fixture hash mismatch')
        for name, value in cls.prosody_first_block.items():
            if hashlib.sha256(value.tobytes()).hexdigest() != meta['array_sha256'][name]:
                raise RuntimeError(f'first block tensor hash mismatch: {name}')
        cls.first_block_meta = meta
        weights = {'acoustic_text_encoder.embedding.weight':
                   cls.text_embedding['table'].copy()}
        for index in range(3):
            for kind in ('weight', 'bias'):
                weights[f'acoustic_text_encoder.conv{index}.{kind}'] = \
                    cls.text_encoder[f'conv{index}_{kind}']
            weights[f'acoustic_text_encoder.norm{index}.weight'] = \
                cls.text_encoder[f'norm{index}_gamma']
            weights[f'acoustic_text_encoder.norm{index}.bias'] = \
                cls.text_encoder[f'norm{index}_beta']
        for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
            weights[f'acoustic_text_encoder.lstm.{kind}'] = \
                cls.text_encoder[f'lstm_{kind}']
            weights[f'duration_prosody.shared_scan.{kind}'] = \
                cls.prosody_shared[kind]
        for branch in ('F0', 'N'):
            prefix = f'duration_prosody.{branch}.0'
            for norm_name in ('norm1', 'norm2'):
                for kind in ('fc_weight', 'fc_bias', 'norm_weight', 'norm_bias'):
                    canonical = f'{prefix}.{norm_name}.{kind.replace("_", ".")}'
                    if norm_name == 'norm1':
                        weights[canonical] = cls.prosody_norm[f'{branch}_{kind}']
                    else:
                        weights[canonical] = cls.prosody_first_block[
                            f'{branch}_norm2_{kind}']
            for index in (1, 2):
                for kind in ('weight', 'bias'):
                    weights[f'{prefix}.conv{index}.{kind}'] = \
                        cls.prosody_first_block[f'{branch}_conv{index}_{kind}']
        circuit = materialize_stage_circuit(cls.root, author)
        fixture = prepare_duration_fixture(cls.root, circuit, weights)
        for name, value in vars(fixture).items():
            setattr(cls, name, value)
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

    def view(self, arena, name):
        item = self.buffers[name]
        return np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                          offset=item['abs_offset'])

    def arena(self):
        arena = populated_duration_arena(self.layout, self.entries, self.bump,
                                         self.encoder, self.duration)
        for branch in ('f0', 'n'):
            for stage in ('act0', 'conv0', 'norm1_output', 'act1',
                          'conv1', 'block0_output'):
                self.view(arena, f'{branch}_{stage}')[:] = -91.
        return arena

    def test_connected_first_blocks_match_independent_pytorch(self):
        self.assertEqual(self.calls['errors'], [])
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        worst = (0., None)
        for branch, stem in (('F0', 'f0'), ('N', 'n')):
            for actual_stage, oracle_stage in (
                    ('act0', 'act0'), ('conv0', 'conv0'),
                    ('norm1_output', 'norm1'), ('act1', 'act1'),
                    ('conv1', 'conv1'), ('block0_output', 'block0')):
                actual = self.view(arena, f'{stem}_{actual_stage}').reshape(512, 128)
                expected = self.prosody_first_block[
                    f'length103_{branch}_{oracle_stage}']
                error = np.abs(actual[:, :103] - expected)
                self.assertTrue(np.isfinite(actual[:, :103]).all())
                point = np.unravel_index(np.argmax(error), error.shape)
                if float(error[point]) > worst[0]:
                    worst = (float(error[point]),
                             (branch, actual_stage, *(int(x) for x in point)))
                self.assertLessEqual(float(error[point]), 5e-5,
                    (branch, actual_stage, point, float(error[point])))
                self.assertTrue(np.all(actual[:, 103:] == -91.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.prosody-first-residual-blocks.generated-v1',
            'name': 'generated shared prosody through first F0/noise residual blocks',
            'provider': 'generated_prosody_first_block_prefix', 'dtype': 'fp32',
            'direction': 'inference',
            'oracle': 'pinned-shared-pytorch28-plus-pytorch213-operations',
            'backend_version': self.first_block_meta['oracle_version'],
            'status': 'pass', 'max_diff': worst[0],
            'worst_index': worst[1], 'tolerance': 5e-5,
            'configuration': '36 phonemes; 103 valid of 128 frames; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_xray_selected_provider_and_wrong_branch_weight(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        points = {}
        tensor_report = {'torch': {'tensors': {}}, 'comparisons': {}}
        for branch, stem in (('F0', 'f0'), ('N', 'n')):
            name = f'{stem}_block0_output'
            point = next(item for operation in self.calls['operations']
                for item in operation.get('semantic_checkpoints', [])
                if item['tensor'] == name)
            self.assertEqual(point['resolved_contract_id'],
                'audio_scaled_sum_strided_sum_then_scale_fp32')
            points[name] = point
            selector = name if point['layer'] < 0 else f'{name}@{point["layer"]}'
            native_path = self.root / f'{name}-native.f32'
            oracle_path = self.root / f'{name}-pytorch.f32'
            self.view(arena, name).tofile(native_path)
            self.prosody_first_block[f'length103_{branch}_block0'].tofile(
                oracle_path)
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
            self.loaded, 'ck_kokoro_prosody_first_block_bounded')
        subject = builder.build_manifest(
            backend='ck', call_ir=self.calls, tensor_report=tensor_report,
            model='kokoro_prosody_first_block_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=runtime)
        oracle = builder.build_manifest(
            backend='pytorch', call_ir=self.calls, tensor_report=tensor_report,
            model='kokoro_prosody_first_block_bounded',
            source='pinned_shared_plus_independent_ops', phase='prefill')
        for name, point in points.items():
            profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
                'name': 'kokoro_prosody_first_block', 'backend': 'pytorch',
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
                one(subject), one(oracle), profile)
            self.assertEqual(report['status'], 'pass', (name, report))
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        f0 = self.weight_layout['duration_prosody.F0.0.conv2.weight']
        n = self.weight_layout['duration_prosody.N.0.conv2.weight']
        arena[f0['abs_offset']:f0['abs_offset'] + f0['size']] = \
            arena[n['abs_offset']:n['abs_offset'] + n['size']]
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        changed = self.view(arena, 'f0_block0_output').reshape(512, 128)[:, :103]
        self.assertGreater(float(np.max(np.abs(changed -
            self.prosody_first_block['length103_F0_block0']))), 5e-5)

    def test_altered_lengths_match_reference_and_preserve_padding(self):
        arena = self.arena()
        weight = self.weight_layout['duration_prosody.duration_head.weight']
        bias = self.weight_layout['duration_prosody.duration_head.bias']
        np.ndarray((weight['size'] // 4,), np.float32, buffer=arena,
                   offset=weight['abs_offset'])[:] = 0.
        biases = np.ndarray((bias['size'] // 4,), np.float32, buffer=arena,
                            offset=bias['abs_offset'])
        frames = ctypes.c_int32(-999)
        for value, expected in ((-3.2, 72), (-10., 36)):
            with self.subTest(frames=expected):
                biases[:] = value
                for branch in ('f0', 'n'):
                    self.view(arena, f'{branch}_block0_output')[:] = -91.
                frames.value = -999
                self.assertEqual(self.fn(arena, len(arena),
                                         ctypes.byref(frames)), 0)
                self.assertEqual(frames.value, expected)
                for branch, stem in (('F0', 'f0'), ('N', 'n')):
                    actual = self.view(arena,
                        f'{stem}_block0_output').reshape(512, 128)
                    oracle = self.prosody_first_block[
                        f'length{expected}_{branch}_block0']
                    self.assertTrue(np.isfinite(actual[:, :expected]).all())
                    self.assertLessEqual(float(np.max(np.abs(
                        actual[:, :expected] - oracle))), 5e-5)
                    self.assertTrue(np.all(actual[:, expected:] == -91.))

    def test_repeated_execution_and_failure_propagation(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        first = self.view(arena, 'f0_block0_output').copy()
        self.view(arena, 'f0_block0_output')[:] = -13.
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        result = self.view(arena, 'f0_block0_output').reshape(512, 128)
        np.testing.assert_array_equal(result[:, :103], first.reshape(512, 128)[:, :103])
        self.assertTrue(np.all(result[:, 103:] == -13.))
        arena = self.arena()
        entry = self.weight_layout['duration_prosody.F0.0.conv1.weight']
        np.ndarray((entry['size'] // 4,), np.float32, buffer=arena,
                   offset=entry['abs_offset'])[0] = np.nan
        frames.value = -999
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, -999)
        self.assertTrue(np.all(self.view(arena, 'f0_block0_output') == -91.))
        self.assertTrue(np.all(self.view(arena, 'n_block0_output') == -91.))

    def test_wrong_residual_edge_is_detected(self):
        graph = copy.deepcopy(self.source['template'])
        operations = graph['block_types']['prosody_first_block']['body']['ops']
        join = next(item for item in operations
                    if item['id'] == 'f0_block0_output')
        join['graph_slots']['inputs']['left'] = 'f0_norm0_output'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'wrong-residual-edge.json'
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
            item = buffers['f0_block0_output']
            result = np.ndarray((512, 128), np.float32, buffer=arena,
                                offset=item['abs_offset'])[:, :103]
            self.assertGreater(float(np.max(np.abs(result -
                self.prosody_first_block['length103_F0_block0']))), 5e-5)


if __name__ == '__main__':
    unittest.main()
