"""Variable-length generated first Kokoro decoder residual block."""
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
CAPACITY = 64
PAD = -91.0
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_decoder_encode_circuit import (
    build_circuit, FRAMES)

CHECKPOINTS = {
    'decoder_encode_norm0': ('decoder_encode_norm1', 514, 6e-5),
    'decoder_encode_conv0': ('decoder_encode_conv1', 1024, 1e-5),
    'decoder_encode_norm1': ('decoder_encode_norm2', 1024, 2e-5),
    'decoder_encode_conv1': ('decoder_encode_conv2', 1024, 1e-5),
    'decoder_encode_shortcut': ('decoder_encode_conv1x1', 1024, 1e-5),
    'decoder_encode_output': ('decoder_encode', 1024, 1e-5),
}


class KokoroRuntimeDecoderEncodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        graph = build_circuit()
        circuit = cls.root / 'circuit.json'
        circuit.write_text(json.dumps(graph))
        weights, _, prosody_meta = complete_prosody_weights()
        cls.ingress, ingress_meta = verified_fixture('decoder_ingress')
        cls.direct, cls.direct_meta = verified_fixture('decoder_encode')
        for meta in (ingress_meta, cls.direct_meta):
            if any(meta[key] != prosody_meta[key] for key in
                   ('model_pin', 'code_pin', 'asset_sha256')):
                raise RuntimeError('decoder capture and upstream assets differ')
        for capture in (cls.ingress, cls.direct):
            for name, value in capture.items():
                if name.startswith('waveform_decoder.'):
                    weights[name] = value
        cls.fixture = prepare_duration_fixture(cls.root, circuit, weights)
        meta = cls.fixture.duration_meta
        if (cls.direct_meta['model_pin'] != meta['model_pin']['revision'] or
            cls.direct_meta['asset_sha256'] != meta['asset_hashes'] or
            prosody_meta['voice'] != meta['voice']):
            raise RuntimeError('decoder capture and upstream bundle differ')
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.fixture.source['config']['context_length'] = CAPACITY
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(cls.root, cls.fixture.source, circuit))
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
        self.view(arena, 'decoder_style', (128,))[:] = (
            self.direct['decoder_style'].ravel())
        return arena

    def execute(self, arena, count):
        ids = self.view(arena, 'word_ids', (CAPACITY,), np.int32)
        ids[:] = 178
        if count in self.ids:
            ids[:count] = self.ids[count]
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        for name, channels in (
                ('decoder_f0_downsample', 1), ('decoder_n_downsample', 1),
                ('decoder_text_f0', 513), ('decoder_joined', 514),
                ('decoder_asr_res', 64)):
            self.view(arena, name, (channels, FRAMES))[:] = PAD
        for name, (_, channels, _) in CHECKPOINTS.items():
            self.view(arena, name, (channels, FRAMES))[:] = PAD
        extents = [ctypes.c_int32(-1) for _ in range(4)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(value) for value in extents))
        return status, [value.value for value in extents]

    def test_direct_parity_and_variable_requests(self):
        self.assertEqual(self.calls['errors'], [])
        for name in CHECKPOINTS:
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
                for name, (reference_name, channels, ceiling) in CHECKPOINTS.items():
                    actual = self.view(arena, name, (channels, FRAMES))
                    self.assertTrue(np.isfinite(actual[:, :frames]).all(), name)
                    self.assertTrue(np.all(actual[:, frames:] == PAD), name)
                    if count != 36:
                        continue
                    self.assertEqual((frames, doubled), (103, 206))
                    reference = self.direct[reference_name][0]
                    error = np.abs(actual[:, :frames] - reference)
                    worst = np.unravel_index(np.argmax(error), error.shape)
                    maximum = float(error[worst])
                    self.assertLessEqual(maximum, ceiling, (name, worst, maximum))
                    print('CKE_NUMERICAL_CASE ' + json.dumps({
                        'case_id': f'kokoro.runtime-decoder-encode.{name}.36',
                        'name': f'runtime Kokoro {name} versus full-model hook',
                        'provider': 'generated_kokoro_runtime_decoder_encode',
                        'dtype': 'fp32', 'direction': 'inference',
                        'oracle': 'pinned PyTorch 2.8 direct full-model capture',
                        'status': 'pass', 'gate': 'composition', 'blocking': True,
                        'max_diff': maximum,
                        'rmse': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                        'worst_index': [int(i) for i in worst],
                        'actual_at_worst': float(actual[worst]),
                        'reference_at_worst': float(reference[worst]),
                        'tolerance': ceiling,
                        'configuration': 'capacity=64 tokens, valid=36, A=103',
                        'reproduction_command': 'python3 -m unittest '
                            'tests.test_v8_kokoro_runtime_decoder_encode',
                    }, sort_keys=True))
                output = self.view(arena, 'decoder_encode_output', (1024, FRAMES))
                if count == 2 and first_short is None:
                    first_short = (frames, output[:, :frames].copy())
                elif count == 2:
                    self.assertEqual(frames, first_short[0])
                    np.testing.assert_array_equal(output[:, :frames],
                                                  first_short[1])

    def test_failure_preserves_final_output_and_recovers(self):
        arena = self.arena()
        status, extents = self.execute(arena, 2)
        self.assertEqual(status, 0)
        baseline = self.view(arena, 'decoder_encode_output',
                             (1024, FRAMES)).copy()
        style = self.view(arena, 'decoder_style', (128,))
        original = float(style[0])
        style[0] = np.nan
        status, extents = self.execute(arena, 2)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1] * 4)
        self.assertTrue(np.all(self.view(arena, 'decoder_encode_output',
                                          (1024, FRAMES)) == PAD))
        style[0] = original
        self.assertEqual(self.execute(arena, 2)[0], 0)
        np.testing.assert_array_equal(
            self.view(arena, 'decoder_encode_output', (1024, FRAMES)), baseline)
        status, extents = self.execute(arena, CAPACITY + 1)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1] * 4)
        self.assertTrue(np.all(self.view(arena, 'decoder_encode_output',
                                          (1024, FRAMES)) == PAD))

    def test_wrong_style_weight_is_detected(self):
        arena = self.arena()
        item = self.weights['waveform_decoder.encode.norm1.fc.weight']
        np.ndarray((item['size'] // 4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[:] = 0
        status, extents = self.execute(arena, 36)
        self.assertEqual(status, 0)
        self.assertEqual(extents[2:], [103, 206])
        actual = self.view(arena, 'decoder_encode_output',
                           (1024, FRAMES))[:, :103]
        self.assertGreater(float(np.max(np.abs(
            actual - self.direct['decoder_encode'][0]))), 1e-3)

    def test_xray_final_uses_valid_extent(self):
        arena = self.arena()
        status, extents = self.execute(arena, 36)
        self.assertEqual(status, 0)
        self.assertEqual(extents[2:], [103, 206])
        name = 'decoder_encode_output'
        point = next(point for operation in self.calls['operations']
            for point in operation.get('semantic_checkpoints', [])
            if point['tensor'] == name)
        self.assertEqual(point['resolved_contract_id'],
                         'audio_scaled_sum_strided_sum_then_scale_fp32')
        native = self.root / 'runtime-decoder-encode-native.f32'
        oracle = self.root / 'runtime-decoder-encode-pytorch.f32'
        self.view(arena, name, (1024, FRAMES)).tofile(native)
        self.direct['decoder_encode'][0].tofile(oracle)
        selector = name if point['layer'] < 0 else f'{name}@{point["layer"]}'
        tensor_report = {'torch': {'tensors': {selector: {
            'path': str(oracle), 'shape': [1024, 103]}}},
            'comparisons': {selector: {'ck_path': str(native),
                'shape': [1024, 103], 'physical_shape': [1024, FRAMES],
                'capacity_shape': [1024, FRAMES], 'valid_shape': [1024, 103],
                'physical_strides': [FRAMES, 1]}}}
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_runtime_decoder_encode_bounded')
        subject = builder.build_manifest(backend='ck', call_ir=self.calls,
            tensor_report=tensor_report,
            model='kokoro_runtime_decoder_encode_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=runtime)
        expected = builder.build_manifest(backend='pytorch', call_ir=self.calls,
            tensor_report=tensor_report,
            model='kokoro_runtime_decoder_encode_bounded',
            source='pinned_full_kmodel', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
            'name': 'kokoro_runtime_decoder_encode', 'backend': 'pytorch',
            'contract_schema_version': 1,
            'required_match_fields': ['checkpoint_id', 'producer',
                'logical_layout', 'axis_names', 'resolved_contract_id',
                'kernel_id', 'function'],
            'observed_storage': {'default': 'fp32', 'checkpoints': {}},
            'dtype_thresholds': {'fp32': {'cosine_min': .99999,
                'rmse_max': 1e-5, 'relative_rmse_max': 1e-5,
                'max_abs_max': 1e-5, 'finite_required': True}},
            'checkpoint_order': [point['id']],
            'interval_expansions': {}, 'backend_mappings': {}}
        only = lambda manifest: {**manifest, 'checkpoints': [entry for entry in
            manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
        report = xray_support.xray.compare_manifests(
            only(subject), only(expected), profile)
        self.assertEqual(report['status'], 'pass', report)


if __name__ == '__main__':
    unittest.main()
