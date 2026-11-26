"""Variable-length generated prosody feeding checked decoder ingress."""
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
from build_kokoro_runtime_decoder_ingress_circuit import (
    build_circuit, FRAMES, DOUBLED_FRAMES)

CAPACITY = 64
PAD = -91.0
CHANNELS = {'decoder_f0_downsample': 1, 'decoder_n_downsample': 1,
            'decoder_text_f0': 513, 'decoder_joined': 514,
            'decoder_asr_res': 64}
TOLERANCES = {'decoder_f0_downsample': 2e-5,
              'decoder_n_downsample': 1e-5,
              'decoder_text_f0': 2e-5,
              'decoder_joined': 2e-5,
              'decoder_asr_res': 3e-6}


class KokoroRuntimeDecoderIngressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.root = root
        graph = build_circuit()
        circuit = root / 'circuit.json'
        circuit.write_text(json.dumps(graph))
        weights, _, prosody_meta = complete_prosody_weights()
        cls.direct, cls.direct_meta = verified_fixture('decoder_ingress')
        for name, value in cls.direct.items():
            if name.startswith('waveform_decoder.'):
                weights[name] = value
        cls.fixture = prepare_duration_fixture(root, circuit, weights)
        meta = cls.fixture.duration_meta
        if (cls.direct_meta['model_pin'] != meta['model_pin']['revision'] or
            cls.direct_meta['asset_sha256'] != meta['asset_hashes'] or
            cls.direct_meta['code_pin'] != prosody_meta['code_pin'] or
            prosody_meta['voice'] != meta['voice']):
            raise RuntimeError('decoder capture and upstream assets differ')
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
        for count in (2, 64):
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
        for name, channels in CHANNELS.items():
            self.view(arena, name, (channels, FRAMES))[:] = PAD
        extents = [ctypes.c_int32(-1) for _ in range(4)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in extents))
        return status, [value.value for value in extents]

    def test_connected_parity_and_repeated_variable_lengths(self):
        self.assertEqual(self.calls['errors'], [])
        for name in CHANNELS:
            self.assertTrue(any(op.get('template_op_id') == name for op in
                                self.calls['operations']), name)
        arena = self.arena()
        first_short = None
        for count in (2, 36, 64, 2):
            with self.subTest(count=count):
                status, extents = self.execute(arena, count)
                self.assertEqual(status, 0)
                frames, doubled = extents[2:]
                self.assertEqual(extents[:2], [count, count * 768])
                self.assertGreater(frames, 0)
                self.assertLessEqual(frames, FRAMES)
                self.assertEqual(doubled, 2 * frames)
                self.assertLessEqual(doubled, DOUBLED_FRAMES)
                for name, channels in CHANNELS.items():
                    actual = self.view(arena, name, (channels, FRAMES))
                    self.assertTrue(np.isfinite(actual[:, :frames]).all())
                    self.assertTrue(np.all(actual[:, frames:] == PAD), name)
                joined = self.view(arena, 'decoder_joined', (514, FRAMES))
                text_f0 = self.view(arena, 'decoder_text_f0', (513, FRAMES))
                f0 = self.view(arena, 'decoder_f0_downsample', (1, FRAMES))
                noise = self.view(arena, 'decoder_n_downsample', (1, FRAMES))
                np.testing.assert_array_equal(text_f0[512, :frames], f0[0, :frames])
                np.testing.assert_array_equal(joined[:513, :frames],
                                              text_f0[:, :frames])
                np.testing.assert_array_equal(joined[513, :frames],
                                              noise[0, :frames])
                if count == 36:
                    self.assertEqual((frames, doubled), (103, 206))
                    expected = {
                        'decoder_f0_downsample': self.direct['decoder_F0_conv'][0],
                        'decoder_n_downsample': self.direct['decoder_N_conv'][0],
                        'decoder_asr_res': self.direct['decoder_asr_res'][0],
                    }
                    expected['decoder_text_f0'] = np.concatenate((
                        self.direct['decoder_input_0'][0],
                        expected['decoder_f0_downsample']), axis=0)
                    expected['decoder_joined'] = np.concatenate((
                        expected['decoder_text_f0'],
                        expected['decoder_n_downsample']), axis=0)
                    for name, reference in expected.items():
                        actual = self.view(arena, name,
                                           (CHANNELS[name], FRAMES))[:, :frames]
                        error = np.abs(actual - reference)
                        worst = np.unravel_index(np.argmax(error), error.shape)
                        maximum = float(error[worst])
                        self.assertLessEqual(maximum, TOLERANCES[name],
                                             (name, worst, maximum))
                        print('CKE_NUMERICAL_CASE ' + json.dumps({
                            'case_id': f'kokoro.runtime-decoder-ingress.{name}.36',
                            'name': f'runtime Kokoro {name} versus full-model hook',
                            'provider': 'generated_kokoro_runtime_decoder_ingress',
                            'dtype': 'fp32', 'direction': 'inference',
                            'oracle': 'pinned PyTorch 2.8 direct full-model capture',
                            'status': 'pass', 'gate': 'composition',
                            'blocking': True, 'max_diff': maximum,
                            'rmse': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                            'worst_index': [int(i) for i in worst],
                            'actual_at_worst': float(actual[worst]),
                            'reference_at_worst': float(reference[worst]),
                            'tolerance': TOLERANCES[name],
                            'configuration': 'capacity=64 tokens, valid=36, A=103',
                            'reproduction_command': 'python3 -m unittest '
                                'tests.test_v8_kokoro_runtime_decoder_ingress',
                        }, sort_keys=True))
                if count == 2 and first_short is None:
                    first_short = (frames, joined[:, :frames].copy())
                elif count == 2:
                    self.assertEqual(frames, first_short[0])
                    np.testing.assert_array_equal(joined[:, :frames], first_short[1])

    def test_rejection_preserves_outputs_and_recovers(self):
        arena = self.arena()
        self.assertEqual(self.execute(arena, 2)[0], 0)
        baseline = self.view(arena, 'decoder_joined', (514, FRAMES)).copy()
        item = self.weights['waveform_decoder.F0_conv.weight']
        weight = np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                            offset=item['abs_offset'])
        original = float(weight[0])
        weight[0] = np.nan
        status, extents = self.execute(arena, 2)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1] * 4)
        for name in ('decoder_f0_downsample', 'decoder_text_f0',
                     'decoder_joined', 'decoder_asr_res'):
            self.assertTrue(np.all(self.view(arena, name,
                (CHANNELS[name], FRAMES)) == PAD), name)
        weight[0] = original
        self.assertEqual(self.execute(arena, 2)[0], 0)
        np.testing.assert_array_equal(
            self.view(arena, 'decoder_joined', (514, FRAMES)), baseline)
        status, extents = self.execute(arena, CAPACITY + 1)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1] * 4)
        self.assertTrue(np.all(self.view(arena, 'decoder_joined',
                                          (514, FRAMES)) == PAD))

    def test_wrong_join_weight_is_detected(self):
        arena = self.arena()
        item = self.weights['waveform_decoder.F0_conv.weight']
        np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[:] = 0
        status, extents = self.execute(arena, 36)
        self.assertEqual(status, 0)
        self.assertEqual(extents[2:], [103, 206])
        actual = self.view(arena, 'decoder_f0_downsample',
                           (1, FRAMES))[:, :103]
        self.assertGreater(float(np.max(np.abs(
            actual - self.direct['decoder_F0_conv'][0]))), 1e-3)

    def test_xray_join_uses_selected_contract_and_valid_extent(self):
        arena = self.arena()
        status, extents = self.execute(arena, 36)
        self.assertEqual(status, 0)
        self.assertEqual(extents[2:], [103, 206])
        name = 'decoder_joined'
        point = next(point for operation in self.calls['operations']
            for point in operation.get('semantic_checkpoints', [])
            if point['tensor'] == name)
        self.assertEqual(point['resolved_contract_id'],
                         'audio_concat_channels_exact_copy_fp32')
        reference = np.concatenate((self.direct['decoder_input_0'][0],
            self.direct['decoder_F0_conv'][0],
            self.direct['decoder_N_conv'][0]), axis=0)
        native = self.root / 'runtime-decoder-joined-native.f32'
        oracle = self.root / 'runtime-decoder-joined-pytorch.f32'
        self.view(arena, name, (514, FRAMES)).tofile(native)
        reference.tofile(oracle)
        selector = name if point['layer'] < 0 else f'{name}@{point["layer"]}'
        tensor_report = {'torch': {'tensors': {selector: {
            'path': str(oracle), 'shape': [514, 103]}}},
            'comparisons': {selector: {'ck_path': str(native),
                'shape': [514, 103], 'physical_shape': [514, FRAMES],
                'capacity_shape': [514, FRAMES], 'valid_shape': [514, 103],
                'physical_strides': [FRAMES, 1]}}}
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_decoder_ingress_bounded')
        subject = builder.build_manifest(backend='ck', call_ir=self.calls,
            tensor_report=tensor_report,
            model='kokoro_runtime_decoder_ingress_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=runtime)
        expected = builder.build_manifest(backend='pytorch', call_ir=self.calls,
            tensor_report=tensor_report,
            model='kokoro_runtime_decoder_ingress_bounded',
            source='pinned_full_kmodel', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
            'name': 'kokoro_runtime_decoder_ingress', 'backend': 'pytorch',
            'contract_schema_version': 1,
            'required_match_fields': ['checkpoint_id', 'producer',
                'logical_layout', 'axis_names', 'resolved_contract_id',
                'kernel_id', 'function'],
            'observed_storage': {'default': 'fp32', 'checkpoints': {}},
            'dtype_thresholds': {'fp32': {'cosine_min': .99999,
                'rmse_max': 2e-5, 'relative_rmse_max': 2e-5,
                'max_abs_max': 2e-5, 'finite_required': True}},
            'checkpoint_order': [point['id']],
            'interval_expansions': {}, 'backend_mappings': {}}
        only = lambda manifest: {**manifest, 'checkpoints': [entry for entry in
            manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
        report = xray_support.xray.compare_manifests(
            only(subject), only(expected), profile)
        self.assertEqual(report['status'], 'pass', report)


if __name__ == '__main__':
    unittest.main()
