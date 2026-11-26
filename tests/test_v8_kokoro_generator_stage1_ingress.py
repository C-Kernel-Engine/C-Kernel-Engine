"""Generated Kokoro second upsample and reflection with pinned model evidence."""

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
from tests.test_v8_kokoro_generator_stage0_join import join_weights_and_references
from tests.test_v8_kokoro_generator_stage0_pool import stage0_pool_weights
from tests import test_v8_kokoro_source_residual_first_pair as fixture_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_generator_stage1_ingress_circuit as connected_author
import build_kokoro_generator_stage1_ingress_oracle_circuit as direct_author


class KokoroGeneratorStage1IngressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.refs = join_weights_and_references()
        first, first_meta = fixture_support.load_fixture(
            'kokoro_generator_main_resblock0_pinned')
        cls.tail, tail_meta = fixture_support.load_fixture(
            'kokoro_generator_stage0_pool_pinned')
        cls.stage1, cls.meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_ingress_pinned')
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            if any(item[field] != first_meta[field] for item in (
                    tail_meta, cls.meta, cls.refs.join_meta)):
                raise RuntimeError(f'stage-one fixture {field} identities differ')
        previous = ROOT / 'tests/fixtures/tts/kokoro_generator_stage0_pool_pinned.npz'
        if hashlib.sha256(previous.read_bytes()).hexdigest() != \
                cls.meta['stage0_fixture_sha256']:
            raise RuntimeError('stage-one input fixture hash differs')
        weights = stage0_pool_weights(cls.refs, first, cls.tail)
        weights['waveform_decoder.generator.ups.1.weight'] = cls.stage1['weight']
        weights['waveform_decoder.generator.ups.1.bias'] = cls.stage1['bias']
        circuit = materialize_stage_circuit(root, connected_author)
        fixture = prepare_duration_fixture(root, circuit, weights)
        cls.encoder, cls.duration = fixture.encoder, fixture.duration
        cls.entries, cls.bump = fixture.entries, fixture.bump
        connected = root / 'connected'
        connected.mkdir()
        (cls.connected_layout, cls.connected_calls, _,
         _, cls.connected_fn) = compile_native_graph(
            connected, fixture.source, circuit)
        cls.connected_fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 7
        diagnostic = json.loads(json.dumps(fixture.source))
        diagnostic['template'] = direct_author.build_circuit()
        diagnostic['config']['model'] = diagnostic['template']['name']
        diagnostic['config']['arch'] = diagnostic['template']['name']
        diagnostic['config']['activation_buffer_dtypes'].update({
            'stage0_frame_count': 'i32', 'stage0_extent_value': 'i32',
            'generator_stage1_deconv_extent_value': 'i32',
            'generator_stage1_pad_extent_value': 'i32'})
        referenced = {name
            for block in diagnostic['template']['block_types'].values()
            for op in block['header'] + block['body']['ops'] + block['footer']
            for name in op.get('weight_refs', {}).values()}
        diagnostic['entries'] = [
            item for item in diagnostic['entries'] if item['name'] in referenced]
        direct = root / 'direct'
        direct.mkdir()
        (cls.direct_layout, cls.direct_calls, _,
         _, cls.direct_fn) = compile_oracle_graph(direct, diagnostic)
        cls.direct_fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 3

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
            for item in layout['memory']['activations']['buffers']:
                dtype = np.int32 if item['name'] in (
                    'stage0_frame_count', 'stage0_extent_value',
                    'generator_stage1_deconv_extent_value',
                    'generator_stage1_pad_extent_value') else np.float32
                np.ndarray((item['size'] // 4,), dtype, buffer=arena,
                           offset=item['abs_offset'])[:] = \
                    -999 if dtype == np.int32 else -91.
            cls.view(layout, arena, 'generator_stage0_residual_mean').reshape(
                256, 2560)[:, :2060] = cls.tail['stage0_mean'][0]
        for name in ('generator_stage1_activation',
                     'generator_stage1_upsample',
                     'generator_stage1_reflection'):
            cls.view(layout, arena, name)[:] = -91.
        return arena

    def test_declarations_and_selected_providers(self):
        self.assertEqual(self.connected_calls['errors'], [])
        self.assertEqual(self.direct_calls['errors'], [])
        self.assertEqual(
            [op['function'] for op in self.connected_calls['operations'][-5:]],
            [op['function'] for op in self.direct_calls['operations'][-5:]])
        self.assertEqual(
            self.connected_calls['operations'][-1]['function'],
            'audio_reflect_pad1d_left_channel_major_f32_checked')
        final = self.connected_calls['operations'][-1]
        self.assertEqual(final['call_abi']['kernel_id'],
                         'audio_reflect_pad1d_left_channel_major_f32_checked')
        checkpoint = next(point for point in final['semantic_checkpoints']
                          if point['tensor'] == 'generator_stage1_reflection')
        self.assertEqual(checkpoint['resolved_contract_id'],
                         'audio_reflect_pad1d_left_exact_fp32')

    def test_direct_stage0_input_parity_lengths_and_recovery(self):
        layout = self.direct_layout
        arena = self.arena(layout)
        frame_count = self.view(layout, arena, 'stage0_frame_count', np.int32)
        input_data = self.view(layout, arena,
                               'generator_stage0_residual_mean').reshape(256, 2560)
        output = self.view(layout, arena,
                           'generator_stage1_reflection').reshape(128, 15361)
        first = None
        for frames in (2060, 1960, 2060):
            frame_count[0] = frames
            for name in ('generator_stage1_activation',
                         'generator_stage1_upsample',
                         'generator_stage1_reflection'):
                self.view(layout, arena, name)[:] = -91.
            lengths = [ctypes.c_int32(-999) for _ in range(3)]
            self.assertEqual(self.direct_fn(arena, len(arena), *(
                ctypes.byref(value) for value in lengths)), 0)
            self.assertEqual(tuple(value.value for value in lengths),
                             (frames, frames * 6, frames * 6 + 1))
            activation = self.view(layout, arena,
                'generator_stage1_activation').reshape(256, 2560)
            deconv = self.view(layout, arena,
                'generator_stage1_upsample').reshape(128, 15360)
            self.assertTrue(np.all(activation[:, frames:] == -91.))
            self.assertTrue(np.all(deconv[:, frames * 6:] == -91.))
            self.assertTrue(np.all(output[:, frames * 6 + 1:] == -91.))
            np.testing.assert_array_equal(
                output[:, 0], deconv[:, 1])
            np.testing.assert_array_equal(
                output[:, 1:frames * 6 + 1], deconv[:, :frames * 6])
            if frames == 2060:
                first_run = first is None
                if first is None:
                    first = output.copy()
                else:
                    np.testing.assert_array_equal(first, output)
                for name, actual, reference in (
                    ('activation', activation[:, :frames],
                     self.stage1['activation'][0]),
                    ('deconv', deconv[:, :frames * 6],
                     self.stage1['deconv'][0]),
                    ('reflection', output[:, :frames * 6 + 1],
                     self.stage1['reflection'][0]),
                ):
                    error = np.abs(actual - reference)
                    index = np.unravel_index(np.argmax(error), error.shape)
                    self.assertTrue(np.isfinite(actual).all(), name)
                    tolerance = 0.0 if name == 'activation' else 2e-5
                    if first_run:
                        print('CKE_NUMERICAL_CASE ' + json.dumps({
                            'case_id': f'kokoro.generator-stage1-ingress.{name}.direct-v1',
                            'name': f'Kokoro stage-one ingress {name} direct-input parity',
                            'provider': 'generated_generator_stage1_ingress',
                            'dtype': 'fp32', 'direction': 'inference',
                            'oracle': 'pinned-PyTorch-2.8-full-model-hooks',
                            'status': 'pass' if float(error[index]) <= tolerance else 'fail',
                            'max_diff': float(error[index]),
                            'worst_index': list(map(int, index)),
                            'actual': float(actual[index]),
                            'reference': float(reference[index]),
                            'tolerance': tolerance,
                            'configuration': '36 phonemes, one voice, 2060 input frames',
                            'reproduction_command': 'python3 -m unittest ' + self.id()}))
                    self.assertLessEqual(float(error[index]), tolerance, name)
        before = output.copy()
        frame_count[0] = 2561
        lengths = [ctypes.c_int32(-999) for _ in range(3)]
        self.assertNotEqual(self.direct_fn(arena, len(arena), *(
            ctypes.byref(value) for value in lengths)), 0)
        self.assertEqual(tuple(value.value for value in lengths), (-999,) * 3)
        np.testing.assert_array_equal(output, before)
        frame_count[0] = 2060
        original = float(input_data[0, 0])
        input_data[0, 0] = np.nan
        self.assertNotEqual(self.direct_fn(arena, len(arena), *(
            ctypes.byref(value) for value in lengths)), 0)
        np.testing.assert_array_equal(output, before)
        input_data[0, 0] = original
        weight_item = next(item for item in
            layout['memory']['weights']['entries']
            if item['name'] == 'waveform_decoder.generator.ups.1.weight')
        weight = np.ndarray((weight_item['size'] // 4,), np.float32,
            buffer=arena, offset=weight_item['abs_offset'])
        original_weight = float(weight[0])
        weight[0] = np.nan
        self.assertNotEqual(self.direct_fn(arena, len(arena), *(
            ctypes.byref(value) for value in lengths)), 0)
        self.assertEqual(tuple(value.value for value in lengths), (-999,) * 3)
        np.testing.assert_array_equal(output, before)
        weight[0] = original_weight
        self.assertEqual(self.direct_fn(arena, len(arena), *(
            ctypes.byref(value) for value in lengths)), 0)
        np.testing.assert_array_equal(output, before)

    def test_connected_stage1_extent_and_known_source_mismatch(self):
        arena = self.arena(self.connected_layout)
        lengths = [ctypes.c_int32(-999) for _ in range(7)]
        self.assertEqual(self.connected_fn(arena, len(arena), *(
            ctypes.byref(value) for value in lengths)), 0)
        self.assertEqual(tuple(value.value for value in lengths),
                         (103, 206, 2060, 61800, 12361, 12360, 12361))
        output = self.view(self.connected_layout, arena,
            'generator_stage1_reflection').reshape(128, 15361)
        self.assertTrue(np.isfinite(output[:, :12361]).all())
        error = np.abs(output[:, :12361] - self.stage1['reflection'][0])
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-stage1-ingress.connected-v1',
            'name': 'Kokoro connected second upsample numerical diagnostic',
            'provider': 'generated_generator_stage1_ingress',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-PyTorch-2.8-full-model-hooks',
            'status': 'fail' if float(error.max()) > 2e-4 else 'pass',
            'gate': 'diagnostic', 'blocking': False,
            'reason': 'inherited upstream mismatch; F0 and source-STFT are established contributors',
            'max_diff': float(error.max()), 'tolerance': 2e-4,
            'configuration': '36 phonemes, one voice, captured Gaussian',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        before = output.copy()
        weight_item = next(item for item in
            self.connected_layout['memory']['weights']['entries']
            if item['name'] == 'waveform_decoder.generator.ups.1.weight')
        weight = np.ndarray((weight_item['size'] // 4,), np.float32,
            buffer=arena, offset=weight_item['abs_offset'])
        original = float(weight[0])
        weight[0] = np.nan
        rejected = [ctypes.c_int32(-999) for _ in range(7)]
        self.assertNotEqual(self.connected_fn(arena, len(arena), *(
            ctypes.byref(value) for value in rejected)), 0)
        self.assertEqual(tuple(value.value for value in rejected), (-999,) * 7)
        np.testing.assert_array_equal(output, before)
        weight[0] = original
        recovered = [ctypes.c_int32(-999) for _ in range(7)]
        self.assertEqual(self.connected_fn(arena, len(arena), *(
            ctypes.byref(value) for value in recovered)), 0)
        self.assertEqual(tuple(value.value for value in recovered),
                         (103, 206, 2060, 61800, 12361, 12360, 12361))
        np.testing.assert_array_equal(output, before)


if __name__ == '__main__':
    unittest.main()
