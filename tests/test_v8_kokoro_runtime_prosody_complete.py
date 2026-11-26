"""Complete F0/noise curves from variable-length phonemes in generated C."""
import ctypes
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import prepare_duration_fixture
from tests.test_v8_kokoro_generated_prosody_complete import (
    complete_prosody_weights, verified_fixture)
from tests import test_v8_kokoro_generated_albert_layer as xray_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_prosody_complete_circuit import (
    build_circuit, DOUBLED_FRAMES, FRAMES)

CAPACITY = 64
PAD = -91.0


class KokoroRuntimeProsodyCompleteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.root = root
        graph = build_circuit()
        circuit = root / 'circuit.json'
        circuit.write_text(json.dumps(graph))
        weights, cls.direct, cls.direct_meta = complete_prosody_weights()
        cls.fixture = prepare_duration_fixture(root, circuit, weights)
        if (cls.direct_meta['model_pin'] !=
                cls.fixture.duration_meta['model_pin']['revision'] or
            cls.direct_meta['asset_sha256'] !=
                cls.fixture.duration_meta['asset_hashes'] or
            cls.direct_meta['voice'] != cls.fixture.duration_meta['voice']):
            raise RuntimeError('direct prosody capture and upstream assets differ')
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.fixture.source['config']['context_length'] = CAPACITY
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(root, cls.fixture.source, circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}
        cls.ids = {36: cls.fixture.encoder['word_ids']}
        for count in (2, 8, 64):
            cls.ids[count] = verified_fixture(f'encoder_{count}')[0]['word_ids']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, shape, dtype=np.float32):
        return np.ndarray(shape, dtype, buffer=arena,
                          offset=self.buffers[name]['abs_offset'])

    def arena(self):
        size = self.layout['memory']['arena']['total_size']
        raw = (ctypes.c_uint8 * (size + 63))()
        arena = (ctypes.c_uint8 * size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for planned in self.weights.values():
            entry = self.fixture.entries[planned['name']]
            start = entry['file_offset']
            arena[planned['abs_offset']:planned['abs_offset'] + entry['size']] = (
                self.fixture.bump[start:start + entry['size']])
        self.view(arena, 'type_ids', (CAPACITY,), np.int32)[:] = 0
        self.view(arena, 'predictor_style', (128,))[:] = (
            self.fixture.duration['predictor_style'])
        return arena

    def execute(self, arena, count):
        ids = self.view(arena, 'word_ids', (CAPACITY,), np.int32)
        ids[:] = 178
        if count in self.ids:
            ids[:count] = self.ids[count]
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        outputs = {stem: self.view(arena, f'{stem}_output', (DOUBLED_FRAMES,))
                   for stem in ('f0', 'n')}
        for output in outputs.values():
            output[:] = PAD
        extents = [ctypes.c_int32(-1) for _ in range(4)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in extents))
        return status, [value.value for value in extents], outputs

    def test_full_branches_match_pinned_model_and_reuse_arena(self):
        self.assertEqual(self.calls['errors'], [])
        for name in ('prosody_double_frames', 'f0_block1_pool',
                     'n_block1_pool', 'f0_block0_output',
                     'n_block0_output', 'f0_output', 'n_output'):
            self.assertTrue(any(op.get('template_op_id') == name for op in
                                self.calls['operations']), name)
        arena = self.arena()
        first_short = None
        for count in (2, 36, 64, 2):
            with self.subTest(count=count):
                status, extents, outputs = self.execute(arena, count)
                self.assertEqual(status, 0)
                frames, doubled = extents[2:]
                self.assertEqual(extents[:2], [count, count * 768])
                self.assertEqual(doubled, 2 * frames)
                self.assertGreater(frames, 0)
                self.assertLessEqual(doubled, DOUBLED_FRAMES)
                for branch, stem in (('F0', 'f0'), ('N', 'n')):
                    output = outputs[stem]
                    self.assertTrue(np.isfinite(output[:doubled]).all())
                    self.assertTrue(np.all(output[doubled:] == PAD))
                    if count == 36:
                        self.assertEqual((frames, doubled), (103, 206))
                        first = self.view(arena, f'{stem}_block0_output',
                                          (512, FRAMES))[:, :frames]
                        first_ref = self.direct[f'predictor_{branch}_0']
                        first_error = np.abs(first - first_ref)
                        first_worst = np.unravel_index(np.argmax(first_error),
                                                       first_error.shape)
                        self.assertTrue(np.isfinite(first).all())
                        self.assertLessEqual(float(first_error[first_worst]),
                                             1e-4, (branch, first_worst))
                        print('CKE_NUMERICAL_CASE ' + json.dumps({
                            'case_id': f'kokoro.runtime-{stem}-block0.36',
                            'name': f'runtime Kokoro {branch} first residual block',
                            'provider': 'generated_kokoro_runtime_prosody_complete',
                            'dtype': 'fp32', 'direction': 'inference',
                            'oracle': 'pinned PyTorch 2.8 direct full-model capture',
                            'status': 'pass', 'gate': 'composition',
                            'blocking': True,
                            'max_diff': float(first_error[first_worst]),
                            'rmse': float(np.sqrt(np.mean(
                                first_error.astype(np.float64) ** 2))),
                            'worst_index': [int(index) for index in first_worst],
                            'actual_at_worst': float(first[first_worst]),
                            'reference_at_worst': float(first_ref[first_worst]),
                            'tolerance': 1e-4,
                            'configuration': 'capacity=64 tokens, valid=36, A=103',
                            'reproduction_command': 'python3 -m unittest '
                                'tests.test_v8_kokoro_runtime_prosody_complete',
                        }, sort_keys=True))
                        expected = self.direct[f'predictor_{branch}_proj'][0]
                        error = np.abs(output[:doubled] - expected)
                        worst = int(np.argmax(error))
                        maximum = float(error[worst])
                        ceiling = 5e-4 if branch == 'F0' else 1e-4
                        self.assertLessEqual(maximum, ceiling,
                                             (branch, worst, maximum))
                        print('CKE_NUMERICAL_CASE ' + json.dumps({
                            'case_id': f'kokoro.runtime-{stem}-complete.36',
                            'name': f'runtime Kokoro {branch} complete prosody',
                            'provider': 'generated_kokoro_runtime_prosody_complete',
                            'dtype': 'fp32', 'direction': 'inference',
                            'oracle': 'pinned PyTorch 2.8 direct full-model capture',
                            'status': 'pass', 'gate': 'composition',
                            'blocking': True, 'max_diff': maximum,
                            'rmse': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                            'worst_index': [worst],
                            'actual_at_worst': float(output[worst]),
                            'reference_at_worst': float(expected[worst]),
                            'tolerance': ceiling,
                            'configuration': 'capacity=64 tokens, valid=36, A=103, 2A=206',
                            'reproduction_command': 'python3 -m unittest '
                                'tests.test_v8_kokoro_runtime_prosody_complete',
                        }, sort_keys=True))
                if count == 2 and first_short is None:
                    first_short = (doubled, {key: value[:doubled].copy()
                                             for key, value in outputs.items()})
                elif count == 2:
                    self.assertEqual(doubled, first_short[0])
                    for key, value in outputs.items():
                        np.testing.assert_array_equal(value[:doubled],
                                                      first_short[1][key])

    def test_failed_branch_does_not_publish_or_write_final_outputs(self):
        arena = self.arena()
        status, extents, outputs = self.execute(arena, 8)
        self.assertEqual(status, 0)
        baseline = {key: value[:extents[3]].copy()
                    for key, value in outputs.items()}
        name = 'duration_prosody.F0.1.conv1.weight'
        planned = self.weights[name]
        weight = np.ndarray((planned['size'] // 4,), np.float32,
                            buffer=arena, offset=planned['abs_offset'])
        original = float(weight[0])
        weight[0] = np.nan
        status, extents, outputs = self.execute(arena, 8)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1, -1, -1, -1])
        for output in outputs.values():
            self.assertTrue(np.all(output == PAD))
        weight[0] = original
        status, extents, outputs = self.execute(arena, 8)
        self.assertEqual(status, 0)
        for key, value in outputs.items():
            np.testing.assert_array_equal(value[:extents[3]], baseline[key])
        status, extents, outputs = self.execute(arena, CAPACITY + 1)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1, -1, -1, -1])
        for output in outputs.values():
            self.assertTrue(np.all(output == PAD))

    def test_xray_uses_selected_provider_and_valid_extent(self):
        arena = self.arena()
        status, extents, outputs = self.execute(arena, 36)
        self.assertEqual(status, 0)
        self.assertEqual(extents[2:], [103, 206])
        tensor_report = {'torch': {'tensors': {}}, 'comparisons': {}}
        points = {}
        for branch, stem in (('F0', 'f0'), ('N', 'n')):
            name = f'{stem}_output'
            point = next(item for operation in self.calls['operations']
                for item in operation.get('semantic_checkpoints', [])
                if item['tensor'] == name)
            self.assertEqual(point['resolved_contract_id'],
                             'audio_conv1d_checked_scalar_fma_fp32')
            selector = name if point['layer'] < 0 else \
                f'{name}@{point["layer"]}'
            native = self.root / f'{name}-runtime-native.f32'
            oracle = self.root / f'{name}-runtime-pytorch.f32'
            outputs[stem].tofile(native)
            self.direct[f'predictor_{branch}_proj'].tofile(oracle)
            tensor_report['torch']['tensors'][selector] = {
                'path': str(oracle), 'shape': [1, 206]}
            tensor_report['comparisons'][selector] = {
                'ck_path': str(native), 'shape': [1, 206],
                'physical_shape': [1, DOUBLED_FRAMES],
                'capacity_shape': [1, DOUBLED_FRAMES],
                'valid_shape': [1, 206],
                'physical_strides': [DOUBLED_FRAMES, 1]}
            points[name] = point
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_prosody_complete_bounded')
        subject = builder.build_manifest(backend='ck', call_ir=self.calls,
            tensor_report=tensor_report,
            model='kokoro_runtime_prosody_complete_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=runtime)
        oracle = builder.build_manifest(backend='pytorch', call_ir=self.calls,
            tensor_report=tensor_report,
            model='kokoro_runtime_prosody_complete_bounded',
            source='pinned_full_kmodel', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        for name, point in points.items():
            ceiling = 5e-4 if name == 'f0_output' else 1e-4
            profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
                'name': 'kokoro_runtime_complete_prosody',
                'backend': 'pytorch', 'contract_schema_version': 1,
                'required_match_fields': ['checkpoint_id', 'producer',
                    'logical_layout', 'axis_names', 'resolved_contract_id',
                    'kernel_id', 'function'],
                'observed_storage': {'default': 'fp32', 'checkpoints': {}},
                'dtype_thresholds': {'fp32': {'cosine_min': .99999,
                    'rmse_max': ceiling, 'relative_rmse_max': ceiling,
                    'max_abs_max': ceiling, 'finite_required': True}},
                'checkpoint_order': [point['id']],
                'interval_expansions': {}, 'backend_mappings': {}}
            only = lambda manifest: {**manifest, 'checkpoints': [entry for entry
                in manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
            report = xray_support.xray.compare_manifests(
                only(subject), only(oracle), profile)
            self.assertEqual(report['status'], 'pass', (name, report))


if __name__ == '__main__':
    unittest.main()
