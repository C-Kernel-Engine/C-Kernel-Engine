"""Pinned second source convolution through normal v8 generated execution."""

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
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests.test_v8_kokoro_generator_stage0_join import (
    join_weights_and_references)
from tests.test_v8_kokoro_generator_stage0_pool import stage0_pool_weights
from tests import test_v8_kokoro_source_residual_first_pair as fixture_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_generator_stage1_source_conv_circuit as connected_author
import build_kokoro_generator_stage1_source_conv_oracle_circuit as direct_author


class KokoroGeneratorStage1SourceConvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.refs = join_weights_and_references()
        first, first_meta = fixture_support.load_fixture(
            'kokoro_generator_main_resblock0_pinned')
        cls.tail, tail_meta = fixture_support.load_fixture(
            'kokoro_generator_stage0_pool_pinned')
        cls.ingress, ingress_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_ingress_pinned')
        cls.source, cls.meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_source_conv_pinned')
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            if any(item[field] != first_meta[field] for item in (
                    tail_meta, ingress_meta, cls.meta, cls.refs.join_meta)):
                raise RuntimeError(f'stage-one source fixture {field} differs')
        previous = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_ingress_pinned.npz'
        if hashlib.sha256(previous.read_bytes()).hexdigest() != \
                cls.meta['stage1_fixture_sha256']:
            raise RuntimeError('stage-one source input fixture hash differs')
        weights = stage0_pool_weights(cls.refs, first, cls.tail)
        weights['waveform_decoder.generator.ups.1.weight'] = cls.ingress['weight']
        weights['waveform_decoder.generator.ups.1.bias'] = cls.ingress['bias']
        weights['waveform_decoder.generator.noise_convs.1.weight'] = cls.source['weight']
        weights['waveform_decoder.generator.noise_convs.1.bias'] = cls.source['bias']
        circuit = materialize_stage_circuit(root, connected_author)
        fixture = prepare_duration_fixture(root, circuit, weights)
        cls.encoder, cls.duration = fixture.encoder, fixture.duration
        cls.entries, cls.bump = fixture.entries, fixture.bump
        connected = root / 'connected'
        connected.mkdir()
        cls.connected_layout, cls.connected_calls, _, _, cls.connected_fn = \
            compile_native_graph(connected, fixture.source, circuit)
        cls.connected_fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 7
        diagnostic = json.loads(json.dumps(fixture.source))
        diagnostic['template'] = direct_author.build_circuit()
        diagnostic['config']['model'] = diagnostic['template']['name']
        diagnostic['config']['arch'] = diagnostic['template']['name']
        diagnostic['config']['activation_buffer_dtypes'].update({
            'source_frame_count': 'i32', 'source_extent_value': 'i32'})
        referenced = {name
            for block in diagnostic['template']['block_types'].values()
            for op in block['header'] + block['body']['ops'] + block['footer']
            for name in op.get('weight_refs', {}).values()}
        diagnostic['entries'] = [
            item for item in diagnostic['entries'] if item['name'] in referenced]
        direct = root / 'direct'
        direct.mkdir()
        cls.direct_layout, cls.direct_calls, _, _, cls.direct_fn = \
            compile_oracle_graph(direct, diagnostic)
        cls.direct_fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def view(layout, arena, name, dtype=np.float32):
        item = next(item for item in layout['memory']['activations']['buffers']
                    if item['name'] == name)
        return np.ndarray((item['size'] // np.dtype(dtype).itemsize,),
                          dtype, buffer=arena, offset=item['abs_offset'])

    @classmethod
    def arena(cls, layout):
        if layout is cls.connected_layout:
            arena = populated_duration_arena(
                layout, cls.entries, cls.bump, cls.encoder, cls.duration)
            cls.view(layout, arena, 'decoder_style')[:] = \
                cls.refs.encode['decoder_style'].ravel()
            cls.view(layout, arena, 'source_gaussian').reshape(
                76800, 9)[:61800] = cls.refs.source['gaussian'].reshape(61800, 9)
            cls.view(layout, arena, 'source_stft_window')[:] = \
                cls.refs.source['window']
            taps = np.arange(20, dtype=np.float64)
            angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
            cls.view(layout, arena, 'source_stft_cos').reshape(11, 20)[:] = \
                np.cos(angles)
            cls.view(layout, arena, 'source_stft_sin').reshape(11, 20)[:] = \
                np.sin(angles)
        else:
            size = layout['memory']['arena']['total_size']
            raw = (ctypes.c_uint8 * (size + 63))()
            arena = (ctypes.c_uint8 * size).from_buffer(
                raw, (-ctypes.addressof(raw)) & 63)
            for item in layout['memory']['weights']['entries']:
                entry = cls.entries[item['name']]
                start = entry['file_offset']
                arena[item['abs_offset']:item['abs_offset'] + entry['size']] = \
                    cls.bump[start:start + entry['size']]
            cls.view(layout, arena, 'source_frame_count', np.int32)[0] = 12361
            channels = cls.view(layout, arena, 'source_stft_channels').reshape(
                22, 15361)
            channels[:] = -91.
            channels[:11, :12361] = cls.source['stft_input'][0]
            channels[11:, :12361] = cls.source['stft_phase'][0]
        cls.view(layout, arena, 'generator_stage1_source_conv')[:] = -91.
        return arena

    def test_circuit_and_selected_provider(self):
        self.assertEqual(self.connected_calls['errors'], [])
        self.assertEqual(self.direct_calls['errors'], [])
        for calls in (self.connected_calls, self.direct_calls):
            self.assertEqual(calls['operations'][-1]['function'],
                'audio_conv1d_checked_channel_major_f32')

    def test_direct_model_input_parity_recovery_and_capacity(self):
        layout = self.direct_layout
        arena = self.arena(layout)
        output = self.view(layout, arena,
            'generator_stage1_source_conv').reshape(128, 15361)
        length = ctypes.c_int32(-999)
        self.assertEqual(self.direct_fn(arena, len(arena), ctypes.byref(length)), 0)
        self.assertEqual(length.value, 12361)
        error = np.abs(output[:, :12361] - self.source['output'][0])
        index = np.unravel_index(np.argmax(error), error.shape)
        self.assertTrue(np.isfinite(output[:, :12361]).all())
        self.assertTrue(np.all(output[:, 12361:] == -91.))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-stage1-source-conv.direct-v1',
            'name': 'Kokoro second source convolution direct-input parity',
            'provider': 'generated_generator_stage1_source_conv',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-PyTorch-2.8-full-model-hook',
            'status': 'pass' if float(error[index]) <= 2e-5 else 'fail',
            'max_diff': float(error[index]), 'worst_index': list(map(int, index)),
            'actual': float(output[index]),
            'reference': float(self.source['output'][0][index]),
            'tolerance': 2e-5,
            'configuration': '36 phonemes, one voice, 12361 frames',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_generator_stage1_source_conv.'
                'KokoroGeneratorStage1SourceConvTest.' + self._testMethodName}))
        self.assertLessEqual(float(error[index]), 2e-5)
        baseline = output.copy()
        count = self.view(layout, arena, 'source_frame_count', np.int32)
        output[:] = -91.
        count[0] = 12001
        self.assertEqual(self.direct_fn(arena, len(arena), ctypes.byref(length)), 0)
        self.assertEqual(length.value, 12001)
        self.assertTrue(np.all(output[:, 12001:] == -91.))
        output[:] = -91.
        count[0] = 12361
        self.assertEqual(self.direct_fn(arena, len(arena), ctypes.byref(length)), 0)
        np.testing.assert_array_equal(output, baseline)
        count[0] = 15362
        length = ctypes.c_int32(-999)
        self.assertNotEqual(self.direct_fn(arena, len(arena), ctypes.byref(length)), 0)
        self.assertEqual(length.value, -999)
        np.testing.assert_array_equal(output, baseline)
        count[0] = 12361
        weight_item = next(item for item in
            layout['memory']['weights']['entries']
            if item['name'] == 'waveform_decoder.generator.noise_convs.1.weight')
        weight = np.ndarray((weight_item['size'] // 4,), np.float32,
            buffer=arena, offset=weight_item['abs_offset'])
        original = float(weight[0])
        weight[0] = np.nan
        length = ctypes.c_int32(-999)
        self.assertNotEqual(self.direct_fn(arena, len(arena), ctypes.byref(length)), 0)
        self.assertEqual(length.value, -999)
        np.testing.assert_array_equal(output, baseline)
        weight[0] = original
        self.assertEqual(self.direct_fn(arena, len(arena), ctypes.byref(length)), 0)
        np.testing.assert_array_equal(output, baseline)

    def test_connected_source_numerical_diagnostic(self):
        arena = self.arena(self.connected_layout)
        length = [ctypes.c_int32(-999) for _ in range(7)]
        self.assertEqual(self.connected_fn(arena, len(arena), *(
            ctypes.byref(value) for value in length)), 0)
        self.assertEqual(tuple(value.value for value in length),
                         (103, 206, 2060, 61800, 12361, 12360, 12361))
        output = self.view(self.connected_layout, arena,
            'generator_stage1_source_conv').reshape(128, 15361)
        self.assertTrue(np.isfinite(output[:, :12361]).all())
        self.assertTrue(np.all(output[:, 12361:] == -91.))
        error = np.abs(output[:, :12361] - self.source['output'][0])
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-stage1-source-conv.connected-v1',
            'name': 'Kokoro connected second source convolution diagnostic',
            'provider': 'generated_generator_stage1_source_conv',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-PyTorch-2.8-full-model-hook',
            'status': 'fail' if float(error.max()) > 2e-4 else 'pass',
            'gate': 'diagnostic', 'blocking': False,
            'reason': 'known upstream source/STFT numerical mismatch',
            'max_diff': float(error.max()), 'tolerance': 2e-4,
            'configuration': '36 phonemes, one voice, captured Gaussian',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_generator_stage1_source_conv.'
                'KokoroGeneratorStage1SourceConvTest.' + self._testMethodName}))


if __name__ == '__main__':
    unittest.main()
