"""Connected generated Kokoro decoder ingress versus direct pinned hooks."""

import ctypes
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
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_decoder_encode_circuit as author


class KokoroGeneratedDecoderEncodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        fixtures = {}
        for key in ('prosody_first_block', 'prosody_norm', 'prosody_shared',
                    'text_encoder', 'text_embedding',
                    'prosody_second_block_weights'):
            fixtures[key], _ = verified_fixture(key)
        cls.direct, cls.direct_meta = verified_fixture('decoder_ingress')
        cls.encode, cls.encode_meta = verified_fixture('decoder_encode')
        weights = {'acoustic_text_encoder.embedding.weight':
                   fixtures['text_embedding']['table'].copy()}
        text = fixtures['text_encoder']
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
            weights[f'duration_prosody.shared_scan.{kind}'] = \
                fixtures['prosody_shared'][kind]
        for branch in ('F0', 'N'):
            prefix = f'duration_prosody.{branch}.0'
            for norm_name in ('norm1', 'norm2'):
                for kind in ('fc_weight', 'fc_bias', 'norm_weight', 'norm_bias'):
                    canonical = f'{prefix}.{norm_name}.{kind.replace("_", ".")}'
                    source = ('prosody_norm' if norm_name == 'norm1' else
                              'prosody_first_block')
                    source_name = (f'{branch}_{kind}' if norm_name == 'norm1' else
                                   f'{branch}_norm2_{kind}')
                    weights[canonical] = fixtures[source][source_name]
            for index in (1, 2):
                for kind in ('weight', 'bias'):
                    weights[f'{prefix}.conv{index}.{kind}'] = \
                        fixtures['prosody_first_block'][f'{branch}_conv{index}_{kind}']
        weights.update(fixtures['prosody_second_block_weights'])
        for name, value in cls.direct.items():
            if name.startswith('waveform_decoder.'):
                weights[name] = value
        for name, value in cls.encode.items():
            if name.startswith('waveform_decoder.'):
                weights[name] = value
        circuit = materialize_stage_circuit(cls.root, author)
        fixture = prepare_duration_fixture(cls.root, circuit, weights)
        for name, value in vars(fixture).items():
            setattr(cls, name, value)
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = \
            compile_native_graph(cls.root, cls.source, circuit)
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
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
        self.view(arena, 'decoder_style')[:] = self.encode['decoder_style'].ravel()
        for name in ('decoder_encode_norm0', 'decoder_encode_conv0',
                     'decoder_encode_norm1', 'decoder_encode_conv1',
                     'decoder_encode_shortcut', 'decoder_encode_output'):
            self.view(arena, name)[:] = -91.
        return arena

    def test_direct_model_parity(self):
        self.assertEqual(self.calls['errors'], [])
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        self.assertEqual((frames.value, doubled.value), (103, 206))
        refs = {
            'decoder_encode_norm0': 'decoder_encode_norm1',
            'decoder_encode_conv0': 'decoder_encode_conv1',
            'decoder_encode_norm1': 'decoder_encode_norm2',
            'decoder_encode_conv1': 'decoder_encode_conv2',
            'decoder_encode_shortcut': 'decoder_encode_conv1x1',
            'decoder_encode_output': 'decoder_encode',
        }
        tolerances = {'decoder_encode_norm0': 6e-5,
                      'decoder_encode_conv0': 1e-5,
                      'decoder_encode_norm1': 2e-5,
                      'decoder_encode_conv1': 1e-5,
                      'decoder_encode_shortcut': 1e-5,
                      'decoder_encode_output': 1e-5}
        errors = {}
        for name, ref in refs.items():
            actual = self.view(arena, name).reshape(1024 if name not in
                ('decoder_encode_norm0',) else 514, 128)
            expected = self.encode[ref][0]
            self.assertTrue(np.isfinite(actual[:, :103]).all(), name)
            error = np.abs(actual[:, :103] - expected)
            point = np.unravel_index(np.argmax(error), error.shape)
            errors[name] = float(error[point])
            self.assertLessEqual(errors[name], tolerances[name],
                                 (name, point, errors[name]))
            self.assertTrue(np.all(actual[:, 103:] == -91.), name)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.decoder-encode.generated-v1',
            'name': 'connected generated decoder encode versus direct pinned hooks',
            'provider': 'generated_decoder_encode', 'dtype': 'fp32',
            'direction': 'inference', 'oracle': 'pinned-full-kmodel-pytorch28',
            'backend_version': self.encode_meta['environment']['torch'],
            'status': 'pass', 'max_diff': max(errors.values()),
            'tolerance': max(tolerances.values()),
            'checkpoint_max_abs': errors,
            'checkpoint_tolerances': tolerances,
            'configuration': '36 phonemes; A=103/128; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_upstream_failure_stops_decoder(self):
        arena = self.arena()
        item = self.weight_layout['duration_prosody.duration_head.weight']
        np.ndarray((item['size']//4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[:] = 0.
        item = self.weight_layout['duration_prosody.duration_head.bias']
        np.ndarray((item['size']//4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[:] = 1.
        frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                    ctypes.byref(doubled)), 0)
        self.assertEqual((frames.value, doubled.value), (-999, -999))
        self.assertTrue(np.all(self.view(arena, 'decoder_encode_output') == -91.))

    def test_repeated_lengths_preserve_padding(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        baseline = self.view(arena, 'decoder_encode_output').copy()
        self.view(arena, 'decoder_encode_output')[:] = -91.
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        np.testing.assert_array_equal(self.view(arena, 'decoder_encode_output'),
                                      baseline)
        item = self.weight_layout['duration_prosody.duration_head.weight']
        np.ndarray((item['size']//4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[:] = 0.
        item = self.weight_layout['duration_prosody.duration_head.bias']
        biases = np.ndarray((item['size']//4,), np.float32, buffer=arena,
                            offset=item['abs_offset'])
        for bias_value, valid in ((-3.2, 72), (-10., 36)):
            biases[:] = bias_value
            self.view(arena, 'decoder_encode_output')[:] = -91.
            self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                     ctypes.byref(doubled)), 0)
            self.assertEqual((frames.value, doubled.value), (valid, 2*valid))
            output = self.view(arena, 'decoder_encode_output').reshape(1024, 128)
            self.assertTrue(np.isfinite(output[:, :valid]).all())
            self.assertTrue(np.all(output[:, valid:] == -91.))

    def test_style_and_shortcut_weights_are_observed(self):
        for weight_name in ('waveform_decoder.encode.norm1.fc.weight',
                            'waveform_decoder.encode.conv1x1.weight'):
            arena = self.arena()
            item = self.weight_layout[weight_name]
            np.ndarray((item['size']//4,), np.float32, buffer=arena,
                       offset=item['abs_offset'])[:] = 0.
            frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
            self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                     ctypes.byref(doubled)), 0)
            actual = self.view(arena, 'decoder_encode_output').reshape(1024, 128)[:, :103]
            self.assertGreater(float(np.max(np.abs(
                actual - self.encode['decoder_encode'][0]))), 1e-3,
                weight_name)

    def test_invalid_style_stops_before_decoder_output(self):
        arena = self.arena()
        self.view(arena, 'decoder_style')[0] = np.nan
        frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                    ctypes.byref(doubled)), 0)
        self.assertEqual((frames.value, doubled.value), (-999, -999))
        self.assertTrue(np.all(self.view(arena, 'decoder_encode_output') == -91.))

    def test_xray_final_uses_selected_contract_and_valid_frames(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        name = 'decoder_encode_output'
        point = next(point for operation in self.calls['operations']
            for point in operation.get('semantic_checkpoints', [])
            if point['tensor'] == name)
        self.assertEqual(point['resolved_contract_id'],
                         'audio_scaled_sum_strided_sum_then_scale_fp32')
        native = self.root / 'decoder-encode-native.f32'
        oracle = self.root / 'decoder-encode-pytorch.f32'
        self.view(arena, name).tofile(native)
        self.encode['decoder_encode'][0].tofile(oracle)
        selector = name if point['layer'] < 0 else f'{name}@{point["layer"]}'
        tensor_report = {'torch': {'tensors': {selector: {
            'path': str(oracle), 'shape': [1024, 103]}}},
            'comparisons': {selector: {'ck_path': str(native),
                'shape': [1024, 103], 'physical_shape': [1024, 128],
                'capacity_shape': [1024, 128], 'valid_shape': [1024, 103],
                'physical_strides': [128, 1]}}}
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_decoder_encode_bounded')
        subject = builder.build_manifest(backend='ck', call_ir=self.calls,
            tensor_report=tensor_report, model='kokoro_decoder_encode_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=runtime)
        expected = builder.build_manifest(backend='pytorch', call_ir=self.calls,
            tensor_report=tensor_report, model='kokoro_decoder_encode_bounded',
            source='pinned_full_kmodel', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
            'name': 'kokoro_decoder_encode', 'backend': 'pytorch',
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
