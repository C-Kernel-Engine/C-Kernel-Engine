"""Connected Kokoro source residual prefix and direct-input oracle isolation."""

import copy
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_oracle_circuit_support import compile_oracle_graph
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests.test_v8_kokoro_generated_prosody_complete import (
    complete_prosody_weights)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_source_residual_prefix_circuit as connected_author
import build_kokoro_source_residual_oracle_circuit as oracle_author


class KokoroSourceResidualPrefixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        for label in ('source', 'residual'):
            path = ROOT / 'tests/fixtures/tts' / (
                'kokoro_source_pinned.npz' if label == 'source' else
                'kokoro_source_residual_prefix_pinned.npz')
            meta = json.loads(path.with_suffix('.json').read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta['fixture_sha256']:
                raise RuntimeError(f'{label} fixture checksum mismatch')
            with np.load(path) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            for name, array in arrays.items():
                if hashlib.sha256(array.tobytes()).hexdigest() != meta['array_sha256'][name]:
                    raise RuntimeError(f'{label}.{name} checksum mismatch')
            setattr(cls, 'source_capture' if label == 'source' else label,
                    arrays)
            setattr(cls, label + '_meta', meta)
        weights, _, prosody_meta = complete_prosody_weights()
        for meta in (cls.source_meta, cls.residual_meta):
            if (meta['model_pin'] != prosody_meta['model_pin'] or
                meta['code_pin'] != prosody_meta['code_pin'] or
                meta['asset_sha256'] != prosody_meta['asset_sha256']):
                raise RuntimeError('source residual oracle identities disagree')
        for key, name in (
                ('linear_weight', 'waveform_decoder.generator.m_source.l_linear.weight'),
                ('linear_bias', 'waveform_decoder.generator.m_source.l_linear.bias'),
                ('source_conv_weight', 'waveform_decoder.generator.noise_convs.0.weight'),
                ('source_conv_bias', 'waveform_decoder.generator.noise_convs.0.bias')):
            weights[name] = cls.source_capture[key]
        prefix = 'waveform_decoder.generator.noise_res.0'
        for key, name in (
                ('style_weight', f'{prefix}.adain1.0.fc.weight'),
                ('style_bias', f'{prefix}.adain1.0.fc.bias'),
                ('norm_weight', f'{prefix}.adain1.0.norm.weight'),
                ('norm_bias', f'{prefix}.adain1.0.norm.bias'),
                ('alpha', f'{prefix}.alpha1.0.channel'),
                ('conv_weight', f'{prefix}.convs1.0.weight'),
                ('conv_bias', f'{prefix}.convs1.0.bias')):
            weights[name] = cls.residual[key]
        fixture = prepare_duration_fixture(root, connected_author.OUTPUT, weights)
        for name, value in vars(fixture).items():
            setattr(cls, name, value)
        connected_dir = root / 'connected'
        connected_dir.mkdir()
        (cls.connected_layout, cls.connected_calls, _, cls.connected_loaded,
            cls.connected_fn) = compile_native_graph(
                connected_dir, fixture.source, connected_author.OUTPUT)
        cls.connected_fn.argtypes = [ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t] + [ctypes.POINTER(ctypes.c_int32)] * 5
        diagnostic = copy.deepcopy(fixture.source)
        diagnostic['template'] = oracle_author.build_circuit()
        diagnostic['config']['model'] = diagnostic['template']['name']
        diagnostic['config']['arch'] = diagnostic['template']['name']
        diagnostic['config']['activation_buffer_dtypes'].update({
            'source_conv_frame_count': 'i32',
            'source_conv_extent_value': 'i32'})
        block = diagnostic['template']['block_types']['source_residual_prefix']
        operations = (block['header'] + block['body']['ops'] + block['footer'])
        referenced = {name for op in operations
                      for name in op.get('weight_refs', {}).values()}
        diagnostic['entries'] = [entry for entry in diagnostic['entries']
                                 if entry['name'] in referenced]
        oracle_dir = root / 'oracle-input'
        oracle_dir.mkdir()
        (cls.oracle_layout, cls.oracle_calls, _, cls.oracle_loaded,
            cls.oracle_fn) = compile_oracle_graph(oracle_dir, diagnostic)
        cls.oracle_fn.argtypes = [ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def view(layout, arena, name, dtype=np.float32):
        item = next(item for item in layout['memory']['activations']['buffers']
                    if item['name'] == name)
        return np.ndarray((item['size'] // np.dtype(dtype).itemsize,), dtype,
                          buffer=arena, offset=item['abs_offset'])

    @classmethod
    def arena(cls, layout):
        if layout is cls.oracle_layout:
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for item in layout['memory']['weights']['entries']:
                entry = cls.entries[item['name']]
                start = entry['file_offset']
                data = cls.bump[start:start + entry['size']]
                arena[item['abs_offset']:item['abs_offset'] + len(data)] = data
            for item in layout['memory']['activations']['buffers']:
                name = item['name']
                dtype = np.int32 if name in (
                    'source_conv_frame_count', 'source_conv_extent_value') \
                    else np.float32
                np.ndarray((item['size'] // 4,), dtype, buffer=arena,
                           offset=item['abs_offset'])[:] = \
                    -999 if dtype == np.int32 else -777.
            return arena
        return populated_duration_arena(layout, cls.entries, cls.bump,
                                        cls.encoder, cls.duration)

    @classmethod
    def oracle_arena(cls):
        arena = cls.arena(cls.oracle_layout)
        cls.view(cls.oracle_layout, arena, 'decoder_style')[:] = \
            cls.residual['decoder_style'].reshape(-1)
        cls.view(cls.oracle_layout, arena, 'source_conv0').reshape(256, 2560)[:, :2060] = \
            cls.residual['first_source_conv'][0]
        return arena

    def test_circuit_regeneration_and_selected_providers(self):
        self.assertEqual(json.loads(connected_author.OUTPUT.read_text()),
                         connected_author.build_circuit())
        selected = [item['function'] for item in
                    self.connected_calls['operations'][-4:]]
        self.assertEqual(selected, ['linear_rows_checked_f32',
            'audio_adain_instance_norm_f32',
            'audio_snake_strided_f32_checked',
            'audio_conv1d_dilated_checked_channel_major_f32'])
        self.assertEqual(self.connected_calls['errors'], [])
        self.assertEqual(self.oracle_calls['errors'], [])

    def test_reference_input_numerical_isolation_and_repeated_lengths(self):
        arena = self.oracle_arena()
        layout = self.oracle_layout
        durations = self.view(layout, arena, 'source_conv_frame_count', np.int32)
        first = None
        for frames in (2060, 1960, 2060):
            durations[0] = frames
            for name in ('source_res0_norm0', 'source_res0_snake0',
                         'source_res0_conv0'):
                self.view(layout, arena, name)[:] = -91.
            published = ctypes.c_int32(-999)
            self.assertEqual(self.oracle_fn(arena, len(arena),
                                            ctypes.byref(published)), 0)
            self.assertEqual(published.value, frames)
            for name in ('source_res0_norm0', 'source_res0_snake0',
                         'source_res0_conv0'):
                tensor = self.view(layout, arena, name).reshape(256, 2560)
                self.assertTrue(np.isfinite(tensor[:, :frames]).all())
                self.assertTrue(np.all(tensor[:, frames:] == -91.))
            if frames == 2060:
                outputs = {
                    'style0': self.view(layout, arena, 'source_res0_style0').copy(),
                    'norm0': self.view(layout, arena, 'source_res0_norm0').reshape(256, 2560)[:, :frames].copy(),
                    'snake0': self.view(layout, arena, 'source_res0_snake0').reshape(256, 2560)[:, :frames].copy(),
                    'conv0': self.view(layout, arena, 'source_res0_conv0').reshape(256, 2560)[:, :frames].copy()}
                if first is None:
                    first = outputs
                else:
                    for key in outputs:
                        np.testing.assert_array_equal(outputs[key], first[key])
        for name, actual in first.items():
            expected = self.residual[name].reshape(actual.shape)
            error = np.abs(actual - expected)
            worst = np.unravel_index(np.argmax(error), error.shape)
            maximum = float(error[worst])
            ceiling = {'style0': 2e-5, 'norm0': 5e-5,
                       'snake0': 5e-5, 'conv0': 2e-4}[name]
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.source-residual.reference-input.{name}',
                'name': 'pinned source residual prefix with direct source-conv input',
                'provider': 'generated-checked-native-graph',
                'oracle': 'direct-pinned-full-model-pytorch28',
                'status': 'pass' if maximum <= ceiling else 'fail',
                'gate': 'diagnostic', 'max_diff': maximum,
                'tolerance': ceiling, 'worst_index': list(map(int, worst)),
                'actual': float(actual[worst]),
                'reference': float(expected[worst]),
                'configuration': '36 phonemes; 2060 source-conv frames; oracle-supplied source-conv input',
                'reproduction_command': 'python3 -m unittest ' + self.id()}))
            self.assertLessEqual(maximum, ceiling, (name, worst))

    def test_rejected_alpha_does_not_publish_output(self):
        arena = self.oracle_arena()
        layout = self.oracle_layout
        self.view(layout, arena, 'source_conv_frame_count', np.int32)[0] = 2060
        weight = next(item for item in layout['memory']['weights']['entries']
            if item['name'].endswith('.alpha1.0.channel'))
        alpha = np.ndarray((256,), np.float32, buffer=arena,
                           offset=weight['abs_offset'])
        saved = alpha[-1]
        alpha[-1] = 0
        output = self.view(layout, arena, 'source_res0_conv0')
        output[:] = -91.
        published = ctypes.c_int32(-999)
        self.assertNotEqual(self.oracle_fn(arena, len(arena),
                                           ctypes.byref(published)), 0)
        self.assertEqual(published.value, -999)
        self.assertTrue(np.all(output == -91.))
        alpha[-1] = saved
        self.assertEqual(self.oracle_fn(arena, len(arena),
                                        ctypes.byref(published)), 0)

    def test_connected_execution_keeps_numerical_failure_explicit(self):
        arena = self.arena(self.connected_layout)
        layout = self.connected_layout
        self.view(layout, arena, 'decoder_style')[:] = \
            self.residual['decoder_style'].reshape(-1)
        self.view(layout, arena, 'source_gaussian').reshape(76800, 9)[:61800] = \
            self.source_capture['gaussian'].reshape(61800, 9)
        self.view(layout, arena, 'source_stft_window')[:] = self.source_capture['window']
        taps = np.arange(20, dtype=np.float64)
        angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
        self.view(layout, arena, 'source_stft_cos').reshape(11, 20)[:] = np.cos(angles)
        self.view(layout, arena, 'source_stft_sin').reshape(11, 20)[:] = np.sin(angles)
        output = self.view(layout, arena, 'source_res0_conv0')
        output[:] = -91.
        published = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(item) for item in published)), 0)
        self.assertEqual(published[-1].value, 2060)
        actual = output.reshape(256, 2560)[:, :2060]
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.all(output.reshape(256, 2560)[:, 2060:] == -91.))
        expected = self.residual['conv0'][0]
        maximum = float(np.max(np.abs(actual - expected)))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.source-residual.connected.raw-phase-v1',
            'name': 'connected generated source residual prefix',
            'provider': 'generated-checked-native-graph',
            'oracle': 'direct-pinned-full-model-pytorch28',
            'status': 'fail' if maximum > 2e-4 else 'pass',
            'gate': 'diagnostic', 'blocking': False,
            'reason': 'upstream source STFT raw-phase compatibility unresolved',
            'max_diff': maximum, 'tolerance': 2e-4,
            'configuration': '36 phonemes; captured Gaussian; CKE-generated F0 and source',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.assertGreater(maximum, 2e-4)


if __name__ == '__main__':
    unittest.main()
