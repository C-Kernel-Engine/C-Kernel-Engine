"""One generated Kokoro entry joins the first main and source paths."""

import ctypes
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests.test_v8_kokoro_generated_decoder_complete import (
    checkpoint_metrics, require_matching_reference_identity)
from tests.v8_kokoro_decoder_weight_support import decoder_weights_and_references
from tests import test_v8_kokoro_source_residual_first_pair as pair_support
from tests import test_v8_kokoro_generated_albert_layer as xray_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_generator_stage0_join_circuit as author


def join_weights_and_references():
    """Load the identity-checked pinned inputs shared with later generator tests."""
    (weights, _, direct_meta, encode, encode_meta,
     _, decode_meta) = decoder_weights_and_references()
    main, main_meta = verified_fixture('generator_stage0')
    source, source_meta = pair_support.load_fixture('kokoro_source_pinned')
    prefix, prefix_meta = pair_support.load_fixture(
        'kokoro_source_residual_prefix_pinned')
    first_pair, first_pair_meta = pair_support.load_fixture(
        'kokoro_source_residual_first_pair_pinned')
    block, block_meta = pair_support.load_fixture(
        'kokoro_source_residual_block0_pinned')
    join, join_meta = pair_support.load_fixture(
        'kokoro_generator_stage0_join_pinned')
    for meta in (encode_meta, *decode_meta.values(), main_meta):
        require_matching_reference_identity(direct_meta, meta)
    for meta in (source_meta, prefix_meta, first_pair_meta,
                 block_meta, join_meta):
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            if meta[field] != main_meta[field]:
                raise RuntimeError(f'generator join {field} identities disagree')
    for name, value in main.items():
        if name.startswith('waveform_decoder.'):
            weights[name] = value
    for key, name in (
            ('linear_weight', 'waveform_decoder.generator.m_source.l_linear.weight'),
            ('linear_bias', 'waveform_decoder.generator.m_source.l_linear.bias'),
            ('source_conv_weight', 'waveform_decoder.generator.noise_convs.0.weight'),
            ('source_conv_bias', 'waveform_decoder.generator.noise_convs.0.bias')):
        weights[name] = source[key]
    weight_prefix = 'waveform_decoder.generator.noise_res.0'
    for arrays, side in ((prefix, 1), (first_pair, 2)):
        for key, name in (
                ('style_weight', f'{weight_prefix}.adain{side}.0.fc.weight'),
                ('style_bias', f'{weight_prefix}.adain{side}.0.fc.bias'),
                ('norm_weight', f'{weight_prefix}.adain{side}.0.norm.weight'),
                ('norm_bias', f'{weight_prefix}.adain{side}.0.norm.bias'),
                ('alpha', f'{weight_prefix}.alpha{side}.0.channel'),
                ('conv_weight', f'{weight_prefix}.convs{side}.0.weight'),
                ('conv_bias', f'{weight_prefix}.convs{side}.0.bias')):
            weights[name] = arrays[key]
    for pair in (1, 2):
        for side in (1, 2):
            for key, name in (
                    ('style_weight', f'{weight_prefix}.adain{side}.{pair}.fc.weight'),
                    ('style_bias', f'{weight_prefix}.adain{side}.{pair}.fc.bias'),
                    ('norm_weight', f'{weight_prefix}.adain{side}.{pair}.norm.weight'),
                    ('norm_bias', f'{weight_prefix}.adain{side}.{pair}.norm.bias'),
                    ('alpha', f'{weight_prefix}.alpha{side}.{pair}.channel'),
                    ('conv_weight', f'{weight_prefix}.convs{side}.{pair}.weight'),
                    ('conv_bias', f'{weight_prefix}.convs{side}.{pair}.bias')):
                weights[name] = block[f'pair{pair}_{key}_c{side}']
    return SimpleNamespace(weights=weights, main=main, source=source,
        prefix=prefix, first_pair=first_pair, block=block, join=join,
        join_meta=join_meta, encode=encode)


class KokoroGeneratorStage0JoinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        refs = join_weights_and_references()
        for name, value in vars(refs).items():
            if name != 'weights':
                setattr(cls, name, value)
        circuit = materialize_stage_circuit(root, author)
        fixture = prepare_duration_fixture(root, circuit, refs.weights)
        cls.encoder = fixture.encoder
        cls.duration = fixture.duration
        cls.entries = fixture.entries
        cls.bump = fixture.bump
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = \
            compile_native_graph(root, fixture.source, circuit)
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + \
            [ctypes.POINTER(ctypes.c_int32)] * 5
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weight_layout = {item['name']: item for item in
                             cls.layout['memory']['weights']['entries']}
        cls.decoder_style = refs.encode['decoder_style'].ravel()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def view(self, arena, name, dtype=np.float32):
        item = self.buffers[name]
        return np.ndarray((item['size'] // np.dtype(dtype).itemsize,), dtype,
                          buffer=arena, offset=item['abs_offset'])

    def arena(self):
        arena = populated_duration_arena(self.layout, self.entries, self.bump,
                                         self.encoder, self.duration)
        self.view(arena, 'decoder_style')[:] = self.decoder_style
        self.view(arena, 'source_gaussian').reshape(76800, 9)[:61800] = \
            self.source['gaussian'].reshape(61800, 9)
        self.view(arena, 'source_stft_window')[:] = self.source['window']
        taps = np.arange(20, dtype=np.float64)
        angles = -2 * np.pi * np.arange(11)[:, None] * taps / 20
        self.view(arena, 'source_stft_cos').reshape(11, 20)[:] = np.cos(angles)
        self.view(arena, 'source_stft_sin').reshape(11, 20)[:] = np.sin(angles)
        return arena

    def run_entry(self, arena):
        lengths = [ctypes.c_int32(-999) for _ in range(5)]
        status = self.fn(arena, len(arena),
                         *(ctypes.byref(item) for item in lengths))
        return status, tuple(item.value for item in lengths)

    def test_circuit_selection_and_shared_runtime_length(self):
        self.assertEqual(self.calls['errors'], [])
        operations = self.calls['operations']
        self.assertEqual(operations[-1]['function'],
                         'audio_scaled_sum_strided_f32_checked')
        graph = author.build_circuit()
        self.assertNotIn('source_conv_frames', graph['runtime_lengths'])
        self.assertEqual(graph['block_types']['generator_stage0']['header'][0]
                         ['id'], 'generator_stage0_extent')
        self.assertNotIn('source_conv_extent', [op['id'] for op in
            graph['block_types']['source_stft_conv']['header']])
        self.assertEqual(graph['block_types']['generator_stage0_join']
            ['body']['ops'][0]['runtime_scalar_bindings']['sum_columns'],
            'generator_stage0_frames')

    def test_connected_join_and_known_upstream_numerical_failure(self):
        arena = self.arena()
        output = self.view(arena, 'generator_stage0_join').reshape(256, 2560)
        output[:] = -91.
        status, lengths = self.run_entry(arena)
        self.assertEqual((status, lengths), (0, (103, 206, 2060, 61800, 12361)))
        main = self.view(arena, 'generator_stage0_upsample').reshape(256, 2560)
        source = self.view(arena, 'source_res_pair2_output').reshape(256, 2560)
        self.assertLessEqual(checkpoint_metrics(main[:, :2060],
            self.main['decoder_generator_ups_0'][0])['max_abs'], 5e-5)
        np.testing.assert_array_equal(output[:, :2060],
            (main[:, :2060] + source[:, :2060]).astype(np.float32))
        self.assertTrue(np.isfinite(output[:, :2060]).all())
        self.assertTrue(np.all(output[:, 2060:] == -91.))
        expected = self.join['join'][0]
        metrics = checkpoint_metrics(output[:, :2060], expected)
        point = next(point for operation in self.calls['operations']
            for point in operation.get('semantic_checkpoints', [])
            if point['tensor'] == 'generator_stage0_join')
        self.assertEqual(point['resolved_contract_id'],
                         'audio_scaled_sum_strided_sum_then_scale_fp32')
        native_path = Path(self.temp.name) / 'generator-join-native.f32'
        oracle_path = Path(self.temp.name) / 'generator-join-oracle.f32'
        self.view(arena, 'generator_stage0_join').tofile(native_path)
        expected.tofile(oracle_path)
        selector = 'generator_stage0_join' if point['layer'] < 0 else \
            f'generator_stage0_join@{point["layer"]}'
        tensor_report = {'torch': {'tensors': {selector: {
            'path': str(oracle_path), 'shape': [256, 2060]}}},
            'comparisons': {selector: {'ck_path': str(native_path),
                'shape': [256, 2060], 'physical_shape': [256, 2560],
                'capacity_shape': [256, 2560], 'valid_shape': [256, 2060],
                'physical_strides': [2560, 1]}}}
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_generator_stage0_join_bounded')
        subject = builder.build_manifest(backend='ck', call_ir=self.calls,
            tensor_report=tensor_report,
            model='kokoro_generator_stage0_join_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=runtime)
        oracle_manifest = builder.build_manifest(backend='pytorch',
            call_ir=self.calls, tensor_report=tensor_report,
            model='kokoro_generator_stage0_join_bounded',
            source='pinned_full_kmodel', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
            'name': 'kokoro_generator_stage0_join', 'backend': 'pytorch',
            'contract_schema_version': 1,
            'required_match_fields': ['checkpoint_id', 'producer',
                'logical_layout', 'axis_names', 'resolved_contract_id',
                'kernel_id', 'function'],
            'observed_storage': {'default': 'fp32', 'checkpoints': {}},
            'dtype_thresholds': {'fp32': {'cosine_min': .99999,
                'rmse_max': 2e-4, 'relative_rmse_max': 2e-4,
                'max_abs_max': 2e-4, 'finite_required': True}},
            'checkpoint_order': [point['id']],
            'interval_expansions': {}, 'backend_mappings': {}}
        only = lambda manifest: {**manifest, 'checkpoints': [entry for entry in
            manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
        xray_report = xray_support.xray.compare_manifests(
            only(subject), only(oracle_manifest), profile)
        self.assertEqual(xray_report['status'], 'fail', xray_report)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.generator-stage0-join.connected.raw-phase-v1',
            'name': 'generated first main/source join versus direct full model',
            'provider': 'generated-checked-native-graph',
            'oracle': 'direct-pinned-full-model-pytorch28',
            'status': 'fail' if metrics['max_abs'] > 2e-4 else 'pass',
            'gate': 'diagnostic', 'blocking': False,
            'reason': 'source/STFT raw-phase compatibility unresolved',
            'max_diff': metrics['max_abs'], 'tolerance': 2e-4,
            'worst_index': metrics['worst_index'], 'rmse': metrics['rmse'],
            'configuration': '36 phonemes; af_heart; captured Gaussian; 2060 frames',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))
        self.assertGreater(metrics['max_abs'], 2e-4)

    def test_changed_length_failure_and_recovery(self):
        arena = self.arena()
        output = self.view(arena, 'generator_stage0_join').reshape(256, 2560)
        output[:] = -91.
        baseline, baseline_lengths = self.run_entry(arena)
        self.assertEqual((baseline, baseline_lengths),
                         (0, (103, 206, 2060, 61800, 12361)))
        first = output[:, :2060].copy()
        alternate, meta = verified_fixture('generator_stage0_your')
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            self.assertEqual(meta[field], self.join_meta[field])
        self.view(arena, 'word_ids', np.int32)[:] = alternate['word_ids']
        self.view(arena, 'predictor_style')[:] = alternate['predictor_style'].ravel()
        self.view(arena, 'decoder_style')[:] = alternate['decoder_style'].ravel()
        output[:] = -91.
        status, lengths = self.run_entry(arena)
        self.assertEqual((status, lengths), (0, (98, 196, 1960, 58800, 11761)))
        self.assertTrue(np.isfinite(output[:, :1960]).all())
        self.assertTrue(np.all(output[:, 1960:] == -91.))
        self.view(arena, 'word_ids', np.int32)[:] = self.encoder['word_ids']
        self.view(arena, 'predictor_style')[:] = \
            self.duration['predictor_style'].ravel()
        self.view(arena, 'decoder_style')[:] = self.decoder_style
        output[:] = -91.
        self.assertEqual(self.run_entry(arena),
                         (0, (103, 206, 2060, 61800, 12361)))
        np.testing.assert_array_equal(output[:, :2060], first)
        self.view(arena, 'source_stft_window')[0] = np.nan
        output[:] = -91.
        status, lengths = self.run_entry(arena)
        self.assertNotEqual(status, 0)
        self.assertEqual(lengths, (-999,) * 5)
        self.assertTrue(np.all(output == -91.))


if __name__ == '__main__':
    unittest.main()
