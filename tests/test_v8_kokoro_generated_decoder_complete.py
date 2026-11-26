"""Complete generated Kokoro acoustic decoder versus direct pinned hooks."""

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
from tests.v8_kokoro_decoder_weight_support import decoder_weights_and_references


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_decoder_complete_circuit as author


DECODER_CONTRACT_NAME = 'kokoro.decoder-complete.pinned-fp32-v1'
# The last block's AdaIN discrepancy reaches 8.12e-5 on one alternate
# utterance. Its following convolution accumulates that input difference to
# 1.20e-4; the residual output is 8.41e-5. These are checkpoint-specific
# propagation bounds, not a claim of identical PyTorch reduction order.
DECODER_BLOCK_CHECKPOINT_TOLERANCES = {
    'input': 3e-5, 'norm0': 6e-5, 'conv0': 3e-5,
    'norm1': 8e-5, 'conv1': 8e-5, 'shortcut': 6e-5,
    'output': 6e-5,
}
DECODER_CHECKPOINT_TOLERANCES = {
    f'decoder_decode{index}_{checkpoint}': ceiling
    for index in range(4)
    for checkpoint, ceiling in DECODER_BLOCK_CHECKPOINT_TOLERANCES.items()
}
DECODER_CHECKPOINT_TOLERANCES.update({
    'decoder_decode3_pool': 3e-5,
    'decoder_decode3_shortcut_upsample': 3e-5,
    'decoder_decode3_norm1': 1e-4,
    'decoder_decode3_conv1': 2e-4,
    'decoder_decode3_output': 1e-4,
})


def decoder_checkpoint_tolerance(name):
    return DECODER_CHECKPOINT_TOLERANCES[name]


def require_matching_reference_identity(reference, candidate):
    for field in ('model_pin', 'code_pin', 'asset_sha256'):
        if field not in reference or field not in candidate:
            raise ValueError(f'missing reference identity: {field}')
        if reference[field] != candidate[field]:
            raise ValueError(f'reference identity mismatch: {field}')


def checkpoint_metrics(actual, expected):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('nonfinite checkpoint samples')
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    magnitude = np.abs(difference)
    point = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    return {
        'max_abs': float(magnitude[point]),
        'worst_index': [int(value) for value in point],
        'actual': float(actual[point]),
        'reference': float(expected[point]),
        'rmse': float(np.sqrt(np.mean(difference * difference))),
    }


class KokoroGeneratedDecoderCompleteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (weights, cls.direct, cls.direct_meta, cls.encode, cls.encode_meta,
         cls.decode, cls.decode_meta) = decoder_weights_and_references()
        for meta in (cls.encode_meta, *cls.decode_meta.values()):
            require_matching_reference_identity(cls.direct_meta, meta)
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
        for index in range(4):
            self.view(arena, f'decoder_decode{index}_output')[:] = -91.
        return arena

    def test_direct_full_model_checkpoints(self):
        self.assertEqual(self.calls['errors'], [])
        arena = self.arena()
        frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        self.assertEqual((frames.value, doubled.value), (103, 206))
        metrics = {}
        tolerances = {}
        for index in range(4):
            prefix = f'decoder_decode{index}'
            direct = self.decode[index]
            capture = f'decoder_decode_{index}'
            names = {
                f'{prefix}_input': f'{capture}_input_0',
                f'{prefix}_norm0': f'{capture}_norm1',
                f'{prefix}_conv0': f'{capture}_conv1',
                f'{prefix}_norm1': f'{capture}_norm2',
                f'{prefix}_conv1': f'{capture}_conv2',
                f'{prefix}_shortcut': f'{capture}_conv1x1',
                f'{prefix}_output': capture,
            }
            if index == 3:
                names[f'{prefix}_pool'] = f'{capture}_pool'
                names[f'{prefix}_shortcut_upsample'] = f'{capture}_upsample'
            for name, ref in names.items():
                expected = direct[ref][0]
                channels, valid = expected.shape
                capacity = 256 if valid == 206 else 128
                actual = self.view(arena, name).reshape(channels, capacity)
                metrics[name] = checkpoint_metrics(actual[:, :valid], expected)
                ceiling = decoder_checkpoint_tolerance(name)
                tolerances[name] = ceiling
                self.assertLessEqual(metrics[name]['max_abs'], ceiling,
                                     (name, metrics[name], ceiling))
                if name.endswith('_output'):
                    self.assertTrue(np.all(actual[:, valid:] == -91.), name)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.decoder-complete.generated-v1',
            'name': 'connected generated acoustic decoder versus direct pinned hooks',
            'provider': 'generated_decoder_complete', 'dtype': 'fp32',
            'direction': 'inference', 'oracle': 'pinned-full-kmodel-pytorch28',
            'backend_version': self.decode_meta[0]['environment']['torch'],
            'status': 'pass',
            'numerical_contract': DECODER_CONTRACT_NAME,
            'max_diff': max(item['max_abs'] for item in metrics.values()),
            'tolerance': max(tolerances.values()),
            'checkpoint_max_abs': {name: item['max_abs']
                                   for name, item in metrics.items()},
            'checkpoint_metrics': metrics,
            'checkpoint_tolerances': tolerances,
            'configuration': '36 phonemes; A=103/128; 2A=206/256; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_additional_real_utterances(self):
        arena = self.arena()
        for label in ('your', 'today'):
            with self.subTest(utterance=label):
                reference, meta = verified_fixture(
                    f'decoder_utterance_{label}')
                require_matching_reference_identity(self.direct_meta, meta)
                for index in range(4):
                    self.view(arena, f'decoder_decode{index}_output')[:] = -91.
                ids = self.buffers['word_ids']
                np.ndarray((36,), np.int32, buffer=arena,
                           offset=ids['abs_offset'])[:] = reference['word_ids']
                self.view(arena, 'predictor_style')[:] = \
                    reference['predictor_style'].ravel()
                self.view(arena, 'decoder_style')[:] = \
                    reference['decoder_style'].ravel()
                frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
                self.assertEqual(self.fn(arena, len(arena),
                    ctypes.byref(frames), ctypes.byref(doubled)), 0)
                self.assertEqual((frames.value, doubled.value),
                                 (meta['valid_frames'], meta['upsampled_frames']))
                metrics = {}
                tolerances = {}
                names = {f'decoder_decode{i}_output': f'decoder_decode_{i}'
                         for i in range(4)}
                names.update({f'decoder_decode3_{native}':
                    f'decoder_decode_3_{captured}' for native, captured in (
                    ('norm0', 'norm1'), ('pool', 'pool'),
                    ('conv0', 'conv1'), ('norm1', 'norm2'),
                    ('conv1', 'conv2'),
                    ('shortcut_upsample', 'upsample'),
                    ('shortcut', 'conv1x1'))})
                for name, ref in names.items():
                    expected = reference[ref][0]
                    channels, valid = expected.shape
                    capacity = 256 if valid == doubled.value else 128
                    actual = self.view(arena, name).reshape(channels, capacity)
                    metrics[name] = checkpoint_metrics(actual[:, :valid],
                                                        expected)
                    ceiling = decoder_checkpoint_tolerance(name)
                    tolerances[name] = ceiling
                    self.assertLessEqual(metrics[name]['max_abs'], ceiling,
                                         (label, name, metrics[name], ceiling))
                    if name.endswith('_output'):
                        self.assertTrue(np.all(actual[:, valid:] == -91.),
                                        (label, name))
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': f'kokoro.decoder-complete.{label}-utterance-v1',
                    'name': 'generated decoder versus distinct direct-model utterance',
                    'provider': 'generated_decoder_complete', 'dtype': 'fp32',
                    'direction': 'inference',
                    'oracle': 'pinned-full-kmodel-pytorch28',
                    'backend_version': meta['environment']['torch'],
                    'status': 'pass',
                    'numerical_contract': DECODER_CONTRACT_NAME,
                    'max_diff': max(item['max_abs'] for item in metrics.values()),
                    'tolerance': max(tolerances.values()),
                    'checkpoint_max_abs': {name: item['max_abs']
                                           for name, item in metrics.items()},
                    'checkpoint_metrics': metrics,
                    'checkpoint_tolerances': tolerances,
                    'configuration': f'{meta["input_text"]}; A={frames.value}/128',
                    'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_rejected_duration_stops_decoder_without_publishing_length(self):
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
        self.assertTrue(np.all(self.view(arena, 'decoder_decode3_output') == -91.))

    def test_reference_identity_rejects_mixed_experiments(self):
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            candidate = dict(self.direct_meta)
            candidate[field] = 'different experiment'
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, field):
                require_matching_reference_identity(self.direct_meta, candidate)

    def test_invalid_upsample_weight_stops_final_output(self):
        arena = self.arena()
        item = self.weight_layout['waveform_decoder.decode.3.pool.weight']
        np.ndarray((item['size']//4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[0] = np.nan
        frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                    ctypes.byref(doubled)), 0)
        self.assertEqual((frames.value, doubled.value), (-999, -999))
        self.assertTrue(np.all(self.view(arena, 'decoder_decode3_output') == -91.))

    def test_wrong_final_block_weight_is_detected(self):
        arena = self.arena()
        item = self.weight_layout['waveform_decoder.decode.3.conv2.weight']
        np.ndarray((item['size']//4,), np.float32, buffer=arena,
                   offset=item['abs_offset'])[:] = 0.
        frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        actual = self.view(arena, 'decoder_decode3_output').reshape(512, 256)
        expected = self.decode[3]['decoder_decode_3'][0]
        self.assertGreater(float(np.max(np.abs(actual[:, :206] - expected))),
                           1e-3)

    def test_xray_final_uses_selected_contract_and_valid_frames(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999); doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        name = 'decoder_decode3_output'
        point = next(point for operation in self.calls['operations']
            for point in operation.get('semantic_checkpoints', [])
            if point['tensor'] == name)
        self.assertEqual(point['resolved_contract_id'],
                         'audio_scaled_sum_strided_sum_then_scale_fp32')
        native = self.root / 'decoder-complete-native.f32'
        oracle = self.root / 'decoder-complete-pytorch.f32'
        self.view(arena, name).tofile(native)
        self.decode[3]['decoder_decode_3'][0].tofile(oracle)
        selector = name if point['layer'] < 0 else f'{name}@{point["layer"]}'
        tensor_report = {'torch': {'tensors': {selector: {
            'path': str(oracle), 'shape': [512, 206]}}},
            'comparisons': {selector: {'ck_path': str(native),
                'shape': [512, 206], 'physical_shape': [512, 256],
                'capacity_shape': [512, 256], 'valid_shape': [512, 206],
                'physical_strides': [256, 1]}}}
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_decoder_complete_bounded')
        subject = builder.build_manifest(backend='ck', call_ir=self.calls,
            tensor_report=tensor_report, model='kokoro_decoder_complete_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=runtime)
        expected = builder.build_manifest(backend='pytorch', call_ir=self.calls,
            tensor_report=tensor_report, model='kokoro_decoder_complete_bounded',
            source='pinned_full_kmodel', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
            'name': 'kokoro_decoder_complete', 'backend': 'pytorch',
            'contract_schema_version': 1,
            'required_match_fields': ['checkpoint_id', 'producer',
                'logical_layout', 'axis_names', 'resolved_contract_id',
                'kernel_id', 'function'],
            'observed_storage': {'default': 'fp32', 'checkpoints': {}},
            'dtype_thresholds': {'fp32': {'cosine_min': .99999,
                'rmse_max': decoder_checkpoint_tolerance(name),
                'relative_rmse_max': 6e-5,
                'max_abs_max': decoder_checkpoint_tolerance(name),
                'finite_required': True}},
            'checkpoint_order': [point['id']],
            'interval_expansions': {}, 'backend_mappings': {}}
        only = lambda manifest: {**manifest, 'checkpoints': [entry for entry in
            manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
        report = xray_support.xray.compare_manifests(
            only(subject), only(expected), profile)
        self.assertEqual(report['status'], 'pass', report)
