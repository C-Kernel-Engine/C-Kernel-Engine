"""First main generator residual block through connected and direct-input graphs."""

import ctypes
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
from tests.test_v8_kokoro_generator_stage0_join import join_weights_and_references
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests import test_v8_kokoro_source_residual_first_pair as fixture_support
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_generator_main_resblock0_circuit as connected_author
import build_kokoro_generator_main_resblock0_oracle_circuit as oracle_author

CHECKPOINT_TOLERANCES = {
    'pair0_conv_c1': 5e-6, 'pair0_conv_c2': 1e-5,
    'pair0_output': 1e-5,
    'pair1_conv_c1': 6e-6, 'pair1_conv_c2': 1.5e-5,
    'pair1_output': 2e-5,
    'pair2_conv_c1': 1.5e-5, 'pair2_conv_c2': 5e-5,
    'pair2_output': 5e-5,
}


class KokoroGeneratorMainResblock0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        refs = join_weights_and_references()
        cls.main, cls.main_meta = fixture_support.load_fixture(
            'kokoro_generator_main_resblock0_pinned')
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            if cls.main_meta[field] != refs.join_meta[field]:
                raise RuntimeError(f'main generator block {field} identity differs')
        np.testing.assert_array_equal(cls.main['input_join'], refs.join['join'])
        cls.refs = refs
        weights = refs.weights.copy()
        prefix = 'waveform_decoder.generator.resblocks.0'
        for pair in range(3):
            for side in (1, 2):
                for key, name in (
                        ('style_weight', f'{prefix}.adain{side}.{pair}.fc.weight'),
                        ('style_bias', f'{prefix}.adain{side}.{pair}.fc.bias'),
                        ('norm_weight', f'{prefix}.adain{side}.{pair}.norm.weight'),
                        ('norm_bias', f'{prefix}.adain{side}.{pair}.norm.bias'),
                        ('alpha', f'{prefix}.alpha{side}.{pair}.channel'),
                        ('conv_weight', f'{prefix}.convs{side}.{pair}.weight'),
                        ('conv_bias', f'{prefix}.convs{side}.{pair}.bias')):
                    weights[name] = cls.main[f'pair{pair}_{key}_c{side}']
        fixture = prepare_duration_fixture(root, connected_author.OUTPUT, weights)
        cls.encoder = fixture.encoder
        cls.duration = fixture.duration
        cls.entries = fixture.entries
        cls.bump = fixture.bump
        connected_dir = root / 'connected'
        connected_dir.mkdir()
        (cls.connected_layout, cls.connected_calls, cls.connected_library,
         cls.connected_loaded,
         cls.connected_fn) = compile_native_graph(
            connected_dir, fixture.source, connected_author.OUTPUT)
        cls.connected_fn.argtypes = [ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t] + [ctypes.POINTER(ctypes.c_int32)] * 5
        diagnostic = json.loads(json.dumps(fixture.source))
        diagnostic['template'] = oracle_author.build_circuit()
        diagnostic['config']['model'] = diagnostic['template']['name']
        diagnostic['config']['arch'] = diagnostic['template']['name']
        diagnostic['config']['activation_buffer_dtypes'].update({
            'main_res_frame_count': 'i32', 'main_res_extent_value': 'i32'})
        referenced = {name for block in diagnostic['template']['block_types'].values()
            for op in block['header'] + block['body']['ops'] + block['footer']
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
        if layout is cls.connected_layout:
            arena = populated_duration_arena(layout, cls.entries, cls.bump,
                                             cls.encoder, cls.duration)
            cls.view(layout, arena, 'decoder_style')[:] = \
                cls.refs.encode['decoder_style'].ravel()
            cls.view(layout, arena, 'source_gaussian').reshape(76800, 9)[:61800] = \
                cls.refs.source['gaussian'].reshape(61800, 9)
            cls.view(layout, arena, 'source_stft_window')[:] = \
                cls.refs.source['window']
            taps = np.arange(20, dtype=np.float64)
            angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
            cls.view(layout, arena, 'source_stft_cos').reshape(11, 20)[:] = \
                np.cos(angles)
            cls.view(layout, arena, 'source_stft_sin').reshape(11, 20)[:] = \
                np.sin(angles)
            return arena
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
            dtype = np.int32 if item['name'] in (
                'main_res_frame_count', 'main_res_extent_value') else np.float32
            np.ndarray((item['size'] // 4,), dtype, buffer=arena,
                       offset=item['abs_offset'])[:] = \
                -999 if dtype == np.int32 else -91.
        cls.view(layout, arena, 'decoder_style')[:] = \
            cls.refs.encode['decoder_style'].ravel()
        cls.view(layout, arena, 'generator_stage0_join').reshape(
            256, 2560)[:, :2060] = cls.main['input_join'][0]
        return arena

    def test_circuit_and_selected_providers(self):
        self.assertEqual(json.loads(connected_author.OUTPUT.read_text()),
                         connected_author.build_circuit())
        self.assertEqual(self.connected_calls['errors'], [])
        self.assertEqual(self.oracle_calls['errors'], [])
        selected = [op['function'] for op in self.connected_calls['operations'][-27:]]
        self.assertEqual(selected.count(
            'audio_conv1d_dilated_checked_channel_major_f32'), 6)
        self.assertEqual(selected.count('audio_scaled_sum_strided_f32_checked'), 3)
        self.assertEqual(selected.count('audio_snake_strided_f32_checked'), 6)

    def test_direct_join_input_block_parity_and_repeated_lengths(self):
        layout = self.oracle_layout
        arena = self.arena(layout)
        frame_count = self.view(layout, arena, 'main_res_frame_count', np.int32)
        checkpoints = [(f'pair{pair}_{stage}',
                        f'main_res_pair{pair}_' + (
                            'output' if stage == 'output' else
                            '_'.join(reversed(stage.split('_')))))
                       for pair in range(3)
                       for stage in ('conv_c1', 'conv_c2', 'output')]
        observed = {}
        first = None
        for frames in (2060, 1960, 2060):
            frame_count[0] = frames
            for _, name in checkpoints:
                self.view(layout, arena, name)[:] = -91.
            published = ctypes.c_int32(-999)
            self.assertEqual(self.oracle_fn(arena, len(arena),
                                            ctypes.byref(published)), 0)
            self.assertEqual(published.value, frames)
            for key, name in checkpoints:
                actual = self.view(layout, arena, name).reshape(256, 2560)
                self.assertTrue(np.isfinite(actual[:, :frames]).all())
                self.assertTrue(np.all(actual[:, frames:] == -91.))
                if frames != 2060:
                    continue
                expected = self.main[key][0]
                error = np.abs(actual[:, :frames] - expected)
                worst = np.unravel_index(np.argmax(error), error.shape)
                observed[key] = {'max_abs': float(error[worst]),
                    'rmse': float(np.sqrt(np.mean((
                        actual[:, :frames].astype(np.float64) -
                        expected.astype(np.float64)) ** 2))),
                    'worst_index': list(map(int, worst)),
                    'actual': float(actual[worst]),
                    'reference': float(expected[worst])}
                self.assertLessEqual(float(error[worst]),
                                     CHECKPOINT_TOLERANCES[key],
                                     (key, worst, float(error[worst])))
            final = self.view(layout, arena, 'main_res_pair2_output').copy()
            if first is None:
                first = final
            elif frames == 2060:
                np.testing.assert_array_equal(first, final)
        final = self.view(layout, arena, 'main_res_pair2_output').reshape(256, 2560)
        error = np.abs(final[:, :2060] - self.main['pair2_output'][0])
        point = next(point for operation in self.connected_calls['operations']
            for point in operation.get('semantic_checkpoints', [])
            if point['tensor'] == 'main_res_pair2_output')
        self.assertEqual(point['resolved_contract_id'],
                         'audio_scaled_sum_strided_sum_then_scale_fp32')
        native_path = Path(self.temp.name) / 'main-resblock0-native.f32'
        oracle_path = Path(self.temp.name) / 'main-resblock0-oracle.f32'
        self.view(layout, arena, 'main_res_pair2_output').tofile(native_path)
        self.main['pair2_output'][0].tofile(oracle_path)
        selector = 'main_res_pair2_output' if point['layer'] < 0 else \
            f'main_res_pair2_output@{point["layer"]}'
        tensor_report = {'torch': {'tensors': {selector: {
            'path': str(oracle_path), 'shape': [256, 2060]}}},
            'comparisons': {selector: {'ck_path': str(native_path),
                'shape': [256, 2060], 'physical_shape': [256, 2560],
                'capacity_shape': [256, 2560], 'valid_shape': [256, 2060],
                'physical_strides': [2560, 1]}}}
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.connected_loaded, 'ck_kokoro_generator_main_resblock0_bounded')
        subject = builder.build_manifest(backend='ck',
            call_ir=self.connected_calls, tensor_report=tensor_report,
            model='kokoro_generator_main_resblock0_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.connected_library, runtime_library=runtime)
        oracle = builder.build_manifest(backend='pytorch',
            call_ir=self.connected_calls, tensor_report=tensor_report,
            model='kokoro_generator_main_resblock0_bounded',
            source='pinned_full_kmodel', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
            'name': 'kokoro_generator_main_resblock0', 'backend': 'pytorch',
            'contract_schema_version': 1,
            'required_match_fields': ['checkpoint_id', 'producer',
                'logical_layout', 'axis_names', 'resolved_contract_id',
                'kernel_id', 'function'],
            'observed_storage': {'default': 'fp32', 'checkpoints': {}},
            'dtype_thresholds': {'fp32': {'cosine_min': .99999,
                'rmse_max': CHECKPOINT_TOLERANCES['pair2_output'],
                'relative_rmse_max': CHECKPOINT_TOLERANCES['pair2_output'],
                'max_abs_max': CHECKPOINT_TOLERANCES['pair2_output'],
                'finite_required': True}},
            'checkpoint_order': [point['id']],
            'interval_expansions': {}, 'backend_mappings': {}}
        only = lambda manifest: {**manifest, 'checkpoints': [entry for entry in
            manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
        report = xray_support.xray.compare_manifests(
            only(subject), only(oracle), profile)
        self.assertEqual(report['status'], 'pass', report)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-main-resblock0.direct-input-v1',
            'name': 'first main generator block with pinned join input',
            'provider': 'generated_generator_main_resblock0',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'direct-pinned-full-model-pytorch28',
            'status': 'pass', 'max_diff': float(error.max()),
            'tolerance': CHECKPOINT_TOLERANCES['pair2_output'],
            'checkpoint_tolerances': CHECKPOINT_TOLERANCES,
            'checkpoints': observed,
            'configuration': '36 phonemes; 2060 valid frames of 2560; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        frame_count[0] = 2561
        before = final.copy()
        published = ctypes.c_int32(-999)
        self.assertNotEqual(self.oracle_fn(arena, len(arena),
                                           ctypes.byref(published)), 0)
        self.assertEqual(published.value, -999)
        np.testing.assert_array_equal(before, final)
        frame_count[0] = 2060
        self.assertEqual(self.oracle_fn(arena, len(arena),
                                       ctypes.byref(published)), 0)
        np.testing.assert_array_equal(before, final)
        bias_entry = next(item for item in layout['memory']['weights']['entries']
            if item['name'] ==
                'waveform_decoder.generator.resblocks.0.convs2.2.bias')
        bias = np.ndarray((bias_entry['size'] // 4,), np.float32,
                          buffer=arena, offset=bias_entry['abs_offset'])
        saved_bias = float(bias[0])
        bias[0] = saved_bias + 0.5
        self.assertEqual(self.oracle_fn(arena, len(arena),
                                       ctypes.byref(published)), 0)
        changed = self.view(layout, arena, 'main_res_pair2_output').reshape(
            256, 2560)
        self.assertGreater(float(np.max(np.abs(
            changed[:, :2060] - self.main['pair2_output'][0]))), 0.1)
        bias[0] = saved_bias
        self.assertEqual(self.oracle_fn(arena, len(arena),
                                       ctypes.byref(published)), 0)
        np.testing.assert_array_equal(before, final)

    def test_connected_entry_and_rejected_extent(self):
        layout = self.connected_layout
        arena = self.arena(layout)
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        self.view(layout, arena, 'main_res_pair2_output')[:] = -91.
        self.assertEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(x) for x in outputs)), 0)
        names = [item['name'] for item in
                 connected_author.build_circuit()['native_entry']['params'][2:]]
        self.assertEqual(dict(zip(names, (x.value for x in outputs))), {
            'out_frames': 103, 'out_upsampled_frames': 206,
            'out_generator_frames': 2060, 'out_source_samples': 61800,
            'out_stft_frames': 12361})
        final = self.view(layout, arena, 'main_res_pair2_output').reshape(256, 2560)
        self.assertTrue(np.isfinite(final[:, :2060]).all())
        self.assertTrue(np.all(final[:, 2060:] == -91.))
        error = np.abs(final[:, :2060] - self.main['pair2_output'][0])
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-main-resblock0.connected-v1',
            'name': 'connected phonemes through first main generator residual block',
            'provider': 'generated_generator_main_resblock0',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'direct-pinned-full-model-pytorch28',
            'status': 'fail' if float(error.max()) > 2e-4 else 'pass',
            'gate': 'diagnostic', 'blocking': False,
            'reason': 'upstream source and raw-phase numerical compatibility unresolved',
            'max_diff': float(error.max()), 'tolerance': 2e-4,
            'configuration': '36 phonemes; 2060 valid frames; captured Gaussian',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        baseline = final.copy()
        alternate, meta = verified_fixture('generator_stage0_your')
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            self.assertEqual(meta[field], self.main_meta[field])
        self.view(layout, arena, 'word_ids', np.int32)[:] = alternate['word_ids']
        self.view(layout, arena, 'predictor_style')[:] = \
            alternate['predictor_style'].ravel()
        self.view(layout, arena, 'decoder_style')[:] = \
            alternate['decoder_style'].ravel()
        final[:] = -91.
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(x) for x in outputs)), 0)
        self.assertEqual(tuple(x.value for x in outputs),
                         (98, 196, 1960, 58800, 11761))
        self.assertTrue(np.isfinite(final[:, :1960]).all())
        self.assertTrue(np.all(final[:, 1960:] == -91.))
        self.view(layout, arena, 'word_ids', np.int32)[:] = \
            self.encoder['word_ids']
        self.view(layout, arena, 'predictor_style')[:] = \
            self.duration['predictor_style'].ravel()
        self.view(layout, arena, 'decoder_style')[:] = \
            self.refs.encode['decoder_style'].ravel()
        final[:] = -91.
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(x) for x in outputs)), 0)
        np.testing.assert_array_equal(baseline, final)
        before = final.copy()
        self.view(layout, arena, 'source_stft_window')[0] = np.nan
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertNotEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(x) for x in outputs)), 0)
        np.testing.assert_array_equal(before, final)
        self.assertEqual(tuple(x.value for x in outputs), (-999,) * 5)
        self.view(layout, arena, 'source_stft_window')[0] = \
            self.refs.source['window'][0]
        self.assertEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(x) for x in outputs)), 0)
        before = final.copy()
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertNotEqual(self.connected_fn(arena, len(arena) - 1,
            *(ctypes.byref(x) for x in outputs)), 0)
        np.testing.assert_array_equal(before, final)
        self.assertEqual(tuple(x.value for x in outputs), (-999,) * 5)
        alpha_entry = next(item for item in layout['memory']['weights']['entries']
            if item['name'] ==
                'waveform_decoder.generator.resblocks.0.alpha1.0.channel')
        alpha = np.ndarray((alpha_entry['size'] // 4,), np.float32,
                           buffer=arena, offset=alpha_entry['abs_offset'])
        saved_alpha = float(alpha[0])
        alpha[0] = 0.
        before = final.copy()
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertNotEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(x) for x in outputs)), 0)
        np.testing.assert_array_equal(before, final)
        self.assertEqual(tuple(x.value for x in outputs), (-999,) * 5)
        alpha[0] = saved_alpha
        self.assertEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(x) for x in outputs)), 0)
        np.testing.assert_array_equal(before, final)


if __name__ == '__main__':
    unittest.main()
