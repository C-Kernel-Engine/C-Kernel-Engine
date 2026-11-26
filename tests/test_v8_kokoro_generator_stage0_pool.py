"""All three first-stage Kokoro generator blocks through generated C."""

import ctypes
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
from tests import test_v8_kokoro_source_residual_first_pair as fixture_support
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_generator_stage0_pool_circuit as connected_author
import build_kokoro_generator_stage0_pool_oracle_circuit as oracle_author

CHECKPOINT_TOLERANCES = {
    f'main_resblock{block}_pair{pair}_{stage}': (
        1.2e-4 if pair == 2 and stage in ('c2_conv', 'output') else 5e-5)
    for block in (1, 2) for pair in range(3)
    for stage in ('c1_conv', 'c2_conv', 'output')
}


def stage0_pool_weights(refs, first, tail):
    """Effective weights for the connected first-stage generator graph."""
    weights = refs.weights.copy()
    for block, arrays in ((0, first), (1, tail), (2, tail)):
        prefix = f'waveform_decoder.generator.resblocks.{block}'
        fixture_prefix = '' if block == 0 else f'block{block}_'
        for pair in range(3):
            for side in (1, 2):
                for key, name in (
                    ('style_weight', f'{prefix}.adain{side}.{pair}.fc.weight'),
                    ('style_bias', f'{prefix}.adain{side}.{pair}.fc.bias'),
                    ('norm_weight', f'{prefix}.adain{side}.{pair}.norm.weight'),
                    ('norm_bias', f'{prefix}.adain{side}.{pair}.norm.bias'),
                    ('alpha', f'{prefix}.alpha{side}.{pair}.channel'),
                    ('conv_weight', f'{prefix}.convs{side}.{pair}.weight'),
                    ('conv_bias', f'{prefix}.convs{side}.{pair}.bias'),
                ):
                    weights[name] = arrays[
                        f'{fixture_prefix}pair{pair}_{key}_c{side}']
    return weights


class KokoroGeneratorStage0PoolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.refs = join_weights_and_references()
        cls.first, cls.first_meta = fixture_support.load_fixture(
            'kokoro_generator_main_resblock0_pinned')
        cls.tail, cls.tail_meta = fixture_support.load_fixture(
            'kokoro_generator_stage0_pool_pinned')
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            if (cls.first_meta[field] != cls.tail_meta[field] or
                cls.tail_meta[field] != cls.refs.join_meta[field]):
                raise RuntimeError(f'generator fixture {field} identity differs')
        np.testing.assert_array_equal(cls.tail['input_join'],
                                      cls.first['input_join'])
        weights = stage0_pool_weights(cls.refs, cls.first, cls.tail)
        circuit = materialize_stage_circuit(root, connected_author)
        fixture = prepare_duration_fixture(root, circuit, weights)
        cls.encoder, cls.duration = fixture.encoder, fixture.duration
        cls.entries, cls.bump = fixture.entries, fixture.bump
        connected_dir = root / 'connected'
        connected_dir.mkdir()
        (cls.connected_layout, cls.connected_calls, _, _,
         cls.connected_fn) = compile_native_graph(
            connected_dir, fixture.source, circuit)
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
        (cls.oracle_layout, cls.oracle_calls, cls.oracle_library,
         cls.oracle_loaded,
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
            256, 2560)[:, :2060] = cls.tail['input_join'][0]
        return arena

    def test_circuit_and_selected_providers(self):
        self.assertEqual(self.connected_calls['errors'], [])
        self.assertEqual(self.oracle_calls['errors'], [])
        selected = [op['function'] for op in self.connected_calls['operations'][-56:]]
        self.assertEqual(selected.count(
            'audio_conv1d_dilated_checked_channel_major_f32'), 12)
        self.assertEqual(selected.count('audio_scaled_sum_strided_f32_checked'), 8)
        self.assertEqual(selected.count('audio_snake_strided_f32_checked'), 12)

    def test_direct_join_input_pool_parity_and_repeated_lengths(self):
        layout = self.oracle_layout
        arena = self.arena(layout)
        frame_count = self.view(layout, arena, 'main_res_frame_count', np.int32)
        final_name = 'generator_stage0_residual_mean'
        first = None
        observed = {}
        for frames in (2060, 1960, 2060):
            frame_count[0] = frames
            for name in ('main_res_pair2_output', final_name, *(
                    f'main_resblock{block}_pair{pair}_{stage}'
                    for block in (1, 2) for pair in range(3)
                    for stage in ('c1_conv', 'c2_conv', 'output'))):
                self.view(layout, arena, name)[:] = -91.
            published = ctypes.c_int32(-999)
            self.assertEqual(self.oracle_fn(arena, len(arena),
                                            ctypes.byref(published)), 0)
            self.assertEqual(published.value, frames)
            for block in (1, 2):
                for pair in range(3):
                    for stage, key in (('c1_conv', 'conv_c1'),
                                       ('c2_conv', 'conv_c2'),
                                       ('output', 'output')):
                        name = f'main_resblock{block}_pair{pair}_{stage}'
                        actual = self.view(layout, arena, name).reshape(256, 2560)
                        self.assertTrue(np.isfinite(actual[:, :frames]).all())
                        self.assertTrue(np.all(actual[:, frames:] == -91.),
                                        (name, frames))
                        if frames == 2060:
                            expected = self.tail[
                                f'block{block}_pair{pair}_{key}'][0]
                            delta = np.abs(actual[:, :frames] - expected)
                            worst = np.unravel_index(np.argmax(delta), delta.shape)
                            observed[name] = {'max_abs': float(delta[worst]),
                                'worst_index': list(map(int, worst)),
                                'actual': float(actual[worst]),
                                'reference': float(expected[worst])}
                            limit = CHECKPOINT_TOLERANCES[name]
                            self.assertLessEqual(float(delta[worst]), limit,
                                                 (name, worst))
            final = self.view(layout, arena, final_name).reshape(256, 2560)
            self.assertTrue(np.isfinite(final[:, :frames]).all())
            self.assertTrue(np.all(final[:, frames:] == -91.))
            if frames == 2060:
                error = np.abs(final[:, :frames] - self.tail['stage0_mean'][0])
                self.assertLessEqual(float(error.max()), 5e-5)
                sum01 = self.view(layout, arena,
                    'generator_stage0_residual_sum01').reshape(256, 2560)
                b0 = self.view(layout, arena, 'main_res_pair2_output').reshape(256, 2560)
                b1 = self.view(layout, arena,
                    'main_resblock1_pair2_output').reshape(256, 2560)
                b2 = self.view(layout, arena,
                    'main_resblock2_pair2_output').reshape(256, 2560)
                np.testing.assert_array_equal(sum01[:, :frames],
                    b0[:, :frames] + b1[:, :frames])
                np.testing.assert_allclose(final[:, :frames],
                    (sum01[:, :frames] + b2[:, :frames]) * np.float32(1. / 3.),
                    rtol=0, atol=1e-6)
            if first is None:
                first = final.copy()
            elif frames == 2060:
                np.testing.assert_array_equal(first, final)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-stage0-pool.direct-input-v1',
            'name': 'Kokoro first-stage generator pool with direct reference join',
            'provider': 'generated_generator_stage0_pool',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-full-model-block-hooks-and-explicit-mean',
            'status': 'pass', 'max_diff': float(error.max()),
            'tolerance': 5e-5, 'checkpoints': observed,
            'checkpoint_tolerances': CHECKPOINT_TOLERANCES,
            'configuration': '36 phonemes; 2060 valid frames',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        point = next(point for operation in self.oracle_calls['operations']
            for point in operation.get('semantic_checkpoints', [])
            if point['tensor'] == final_name)
        self.assertEqual(point['resolved_contract_id'],
                         'audio_scaled_sum_strided_sum_then_scale_fp32')
        native_path = Path(self.temp.name) / 'stage0-pool-native.f32'
        oracle_path = Path(self.temp.name) / 'stage0-pool-oracle.f32'
        self.view(layout, arena, final_name).tofile(native_path)
        self.tail['stage0_mean'][0].tofile(oracle_path)
        selector = final_name if point['layer'] < 0 else \
            f'{final_name}@{point["layer"]}'
        tensor_report = {'torch': {'tensors': {selector: {
            'path': str(oracle_path), 'shape': [256, 2060]}}},
            'comparisons': {selector: {'ck_path': str(native_path),
                'shape': [256, 2060], 'physical_shape': [256, 2560],
                'capacity_shape': [256, 2560], 'valid_shape': [256, 2060],
                'physical_strides': [2560, 1]}}}
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.oracle_loaded,
            'ck_kokoro_generator_stage0_pool_oracle_input_bounded')
        subject = builder.build_manifest(backend='ck',
            call_ir=self.oracle_calls, tensor_report=tensor_report,
            model='kokoro_generator_stage0_pool_oracle_input_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.oracle_library, runtime_library=runtime)
        oracle = builder.build_manifest(backend='pytorch',
            call_ir=self.oracle_calls, tensor_report=tensor_report,
            model='kokoro_generator_stage0_pool_oracle_input_bounded',
            source='pinned_full_kmodel_block_outputs', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
            'name': 'kokoro_generator_stage0_pool', 'backend': 'pytorch',
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
        only = lambda manifest: {**manifest, 'checkpoints': [entry for entry in
            manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
        report = xray_support.xray.compare_manifests(
            only(subject), only(oracle), profile)
        self.assertEqual(report['status'], 'pass', report)
        before = final.copy()
        frame_count[0] = 2561
        published = ctypes.c_int32(-999)
        self.assertNotEqual(self.oracle_fn(arena, len(arena),
                                           ctypes.byref(published)), 0)
        self.assertEqual(published.value, -999)
        np.testing.assert_array_equal(before, final)
        frame_count[0] = 2060
        bias_entry = next(item for item in layout['memory']['weights']['entries']
            if item['name'] ==
                'waveform_decoder.generator.resblocks.2.convs2.2.bias')
        bias = np.ndarray((bias_entry['size'] // 4,), np.float32,
                          buffer=arena, offset=bias_entry['abs_offset'])
        saved_bias = float(bias[0])
        bias[0] = saved_bias + 0.5
        self.assertEqual(self.oracle_fn(arena, len(arena),
                                       ctypes.byref(published)), 0)
        self.assertGreater(float(np.max(np.abs(final[:, :2060] -
            self.tail['stage0_mean'][0]))), 0.05)
        bias[0] = saved_bias
        self.assertEqual(self.oracle_fn(arena, len(arena),
                                       ctypes.byref(published)), 0)
        np.testing.assert_array_equal(before, final)

    def test_connected_entry_reports_existing_source_mismatch(self):
        layout = self.connected_layout
        arena = self.arena(layout)
        name = 'generator_stage0_residual_mean'
        self.view(layout, arena, name)[:] = -91.
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(value) for value in outputs)), 0)
        self.assertEqual(tuple(value.value for value in outputs),
                         (103, 206, 2060, 61800, 12361))
        actual = self.view(layout, arena, name).reshape(256, 2560)
        self.assertTrue(np.isfinite(actual[:, :2060]).all())
        self.assertTrue(np.all(actual[:, 2060:] == -91.))
        difference = np.abs(actual[:, :2060] - self.tail['stage0_mean'][0])
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-stage0-pool.connected-v1',
            'name': 'Kokoro connected first-stage generator pool source parity',
            'provider': 'generated_generator_stage0_pool',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-full-model-block-hooks-and-explicit-mean',
            'status': 'fail' if float(difference.max()) > 2e-4 else 'pass',
            'gate': 'diagnostic', 'blocking': False,
            'reason': 'inherited generated F0 and raw-phase sensitivity',
            'max_diff': float(difference.max()), 'tolerance': 2e-4,
            'configuration': '36 phonemes; 2060 valid frames; captured Gaussian',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        before = actual.copy()
        self.view(layout, arena, 'source_stft_window')[0] = np.nan
        outputs = [ctypes.c_int32(-999) for _ in range(5)]
        self.assertNotEqual(self.connected_fn(arena, len(arena),
            *(ctypes.byref(value) for value in outputs)), 0)
        self.assertEqual(tuple(value.value for value in outputs), (-999,) * 5)
        np.testing.assert_array_equal(before, actual)


if __name__ == '__main__':
    unittest.main()
