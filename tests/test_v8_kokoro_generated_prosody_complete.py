"""Complete generated Kokoro F0/noise curves versus pinned model hooks."""
import ctypes
import copy
import hashlib
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
from tests import test_v8_kokoro_generated_albert_layer as xray_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_prosody_complete_circuit as author


def verified_fixture(name):
    path = ROOT / f'tests/fixtures/tts/kokoro_{name}_pinned.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['fixture_sha256']:
        raise RuntimeError(f'fixture mismatch: {name}')
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, value in arrays.items():
        if ('array_sha256' in meta and
                hashlib.sha256(value.tobytes()).hexdigest() !=
                meta['array_sha256'][key]):
            raise RuntimeError(f'fixture tensor mismatch: {name}.{key}')
    return arrays, meta


def complete_prosody_weights():
    """Return the pinned effective tensors shared by connected prosody tests."""
    fixtures = {}
    for key in ('prosody_first_block', 'prosody_norm', 'prosody_shared',
                'text_encoder', 'text_embedding',
                'prosody_second_block_weights'):
        fixtures[key], _ = verified_fixture(key)
    direct, direct_meta = verified_fixture('prosody_branch_model')
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
    return weights, direct, direct_meta


class KokoroGeneratedProsodyCompleteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        weights, cls.direct, cls.direct_meta = complete_prosody_weights()
        cls.projection_weights = {
            branch: (weights[f'duration_prosody.{branch}_proj.weight'].copy(),
                     weights[f'duration_prosody.{branch}_proj.bias'].copy())
            for branch in ('F0', 'N')}
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

    def view(self, arena, name, dtype=np.float32):
        item = self.buffers[name]
        return np.ndarray((item['size'] // np.dtype(dtype).itemsize,), dtype,
                          buffer=arena, offset=item['abs_offset'])

    def arena(self):
        arena = populated_duration_arena(self.layout, self.entries, self.bump,
                                         self.encoder, self.duration)
        for branch in ('f0', 'n'):
            for name in ('block1_pool', 'block1_conv0',
                         'block1_shortcut_upsample', 'block1_shortcut_conv',
                         'block1_output', 'block2_output', 'output'):
                self.view(arena, f'{branch}_{name}')[:] = -91.
        return arena

    def fine_f0_fixture(self):
        arrays, meta = verified_fixture('prosody_f0_fine')
        for key in ('model_pin', 'code_pin', 'voice', 'asset_sha256'):
            self.assertEqual(meta[key], self.direct_meta[key], key)
        manifest = json.loads((ROOT / 'version/v8/tts/reference/fixture_manifest.json').read_text())
        self.assertEqual(meta['capture_waveform_sha256'],
                         manifest['tensors']['waveform_f32']['sha256'])
        return arrays, meta

    def test_direct_full_model_f0_inner_checkpoints_localize_drift(self):
        direct, meta = self.fine_f0_fixture()
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        self.assertEqual((frames.value, doubled.value), (103, 206))
        checkpoints = (
            ('phoneme_features', 'bert_encoder', 'token'),
            ('predictor_features', 'predictor_text_encoder', 'token'),
            ('prosody_input_token', 'predictor_shared_input', 'token'),
            ('prosody_shared_features', 'predictor_shared_0', 'token'),
            ('f0_norm0_output', 'predictor_F0_0_norm1', 'channel'),
            ('f0_conv0', 'predictor_F0_0_conv1', 'channel'),
            ('f0_norm1_output', 'predictor_F0_0_norm2', 'channel'),
            ('f0_conv1', 'predictor_F0_0_conv2', 'channel'),
            ('f0_block1_norm0_output', 'predictor_F0_1_norm1', 'channel'),
            ('f0_block1_pool', 'predictor_F0_1_pool', 'channel'),
            ('f0_block1_conv0', 'predictor_F0_1_conv1', 'channel'),
            ('f0_block1_norm1_output', 'predictor_F0_1_norm2', 'channel'),
            ('f0_block1_conv1', 'predictor_F0_1_conv2', 'channel'),
            ('f0_block1_shortcut_upsample', 'predictor_F0_1_upsample', 'channel'),
            ('f0_block1_shortcut_conv', 'predictor_F0_1_conv1x1', 'channel'),
            ('f0_block2_norm0_output', 'predictor_F0_2_norm1', 'channel'),
            ('f0_block2_conv0', 'predictor_F0_2_conv1', 'channel'),
            ('f0_block2_norm1_output', 'predictor_F0_2_norm2', 'channel'),
            ('f0_block2_conv1', 'predictor_F0_2_conv2', 'channel'),
        )
        results = {}
        for name, reference_name, layout in checkpoints:
            reference = direct[reference_name]
            item = self.buffers[name]
            if layout == 'token':
                rows, width = reference.shape
                physical_rows, stride = self.source['template'][
                    'activation_buffers'][name]['shape']
                self.assertEqual(item['size'], physical_rows * stride * 4)
                self.assertGreaterEqual(physical_rows, rows)
                self.assertGreaterEqual(stride, width)
                actual = np.ndarray((physical_rows, stride), np.float32,
                                    buffer=arena, offset=item['abs_offset'])[:rows, :width]
            else:
                channels, length = reference.shape
                shape = self.source['template']['activation_buffers'][name]['shape']
                self.assertEqual(shape[0], channels)
                stride = shape[1]
                self.assertEqual(item['size'], channels * stride * 4)
                self.assertGreaterEqual(stride, length)
                actual = np.ndarray((channels, stride), np.float32,
                                    buffer=arena, offset=item['abs_offset'])[:, :length]
            self.assertTrue(np.isfinite(actual).all(), name)
            error = np.abs(actual - reference)
            index = np.unravel_index(np.argmax(error), error.shape)
            maximum = float(error[index])
            self.assertLessEqual(maximum, 5e-5, (name, index, maximum))
            results[name] = {
                'max_abs': maximum,
                'rmse': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                'worst_index': [int(value) for value in index],
                'actual': float(actual[index]),
                'reference': float(reference[index]),
            }
        eager_meta = json.loads((ROOT / 'tests/fixtures/tts/'
            'kokoro_encoder_pinned.json').read_text())
        self.assertEqual(eager_meta['attention_implementation'], 'eager')
        eager_encoder = self.encoder['phoneme_features']
        dispatch_error = np.abs(eager_encoder - direct['bert_encoder'])
        dispatch_index = np.unravel_index(np.argmax(dispatch_error),
                                          dispatch_error.shape)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.encoder.eager-vs-full-sdpa-reference-v1',
            'name': 'pinned PyTorch eager encoder fixture versus full-model SDPA hook',
            'provider': 'oracle_backend_comparison',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pytorch-2.8-eager-versus-pytorch-2.8-sdpa',
            'backend_version': meta['environment']['torch'],
            'status': 'fail' if float(dispatch_error[dispatch_index]) > 1e-5
                      else 'pass',
            'gate': 'diagnostic', 'comparison_kind': 'oracle_reproducibility',
            'blocking': False,
            'reason': 'different declared attention dispatch; no CKE kernel '
                      'verdict follows from this reference comparison',
            'max_diff': float(dispatch_error[dispatch_index]),
            'worst_index': [int(value) for value in dispatch_index],
            'tolerance': 1e-5,
            'configuration': '36 all-valid tokens; af_heart; pinned PyTorch 2.8',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_generated_prosody_complete.'
                'KokoroGeneratedProsodyCompleteTest.'
                'test_direct_full_model_f0_inner_checkpoints_localize_drift',
        }))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.F0.direct-full-model-inner-checkpoints-v1',
            'name': 'connected F0 inner checkpoints against direct pinned KModel hooks',
            'provider': 'generated_prosody_complete',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-full-kmodel-pytorch28',
            'backend_version': meta['environment']['torch'],
            'status': 'fail' if results['f0_norm0_output']['max_abs'] > 1e-5
                      else 'pass',
            'gate': 'diagnostic', 'blocking': False,
            'reason': 'connected upstream drift; local same-input arithmetic is '
                      'reported by the separate pinned PyTorch replay',
            'max_diff': results['f0_norm0_output']['max_abs'],
            'tolerance': 1e-5,
            'checkpoints': results,
            'configuration': '36 phonemes; A=103; 2A=206; af_heart',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_generated_prosody_complete.'
                'KokoroGeneratedProsodyCompleteTest.'
                'test_direct_full_model_f0_inner_checkpoints_localize_drift',
        }))

    def test_pinned_pytorch_same_input_replay_separates_upstream_and_local_error(self):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest('pinned PyTorch 2.8.0+cpu is unavailable')
        if torch.__version__ != '2.8.0+cpu':
            self.skipTest(f'pinned PyTorch 2.8.0+cpu required; found {torch.__version__}')
        direct, _ = self.fine_f0_fixture()
        shared, _ = verified_fixture('prosody_shared')
        norm, _ = verified_fixture('prosody_norm')
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        self.assertEqual((frames.value, doubled.value), (103, 206))
        native_input = self.view(arena, 'prosody_input_token').reshape(
            128, 640)[:103].copy()
        native_shared = self.view(arena, 'prosody_shared_features').reshape(
            128, 512)[:103].copy()
        native_norm = self.view(arena, 'f0_norm0_output').reshape(
            512, 128)[:, :103].copy()
        style = self.view(arena, 'predictor_style')[:128].copy()
        np.testing.assert_array_equal(style, self.duration['predictor_style'])
        model = torch.nn.LSTM(640, 256, batch_first=True,
                              bidirectional=True).eval()
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.no_grad():
                for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
                    for direction, suffix in enumerate(('', '_reverse')):
                        getattr(model, f'{kind}_l0{suffix}').copy_(
                            torch.from_numpy(shared[kind][direction]))

                def replay_scan(value):
                    model.flatten_parameters()
                    return model(torch.from_numpy(value.copy())[None])[0][0].numpy()

                def replay_norm(value):
                    affine = functional.linear(
                        torch.from_numpy(style.copy())[None],
                        torch.from_numpy(norm['F0_fc_weight']),
                        torch.from_numpy(norm['F0_fc_bias']))[0]
                    normalized = functional.instance_norm(
                        torch.from_numpy(value.T.copy())[None],
                        weight=torch.from_numpy(norm['F0_norm_weight']),
                        bias=torch.from_numpy(norm['F0_norm_bias']),
                        use_input_stats=True, eps=1e-5)[0]
                    return ((1 + affine[:512, None]) * normalized +
                            affine[512:, None]).numpy()

                direct_shared = direct['predictor_shared_0']
                replayed_reference = replay_scan(direct['predictor_shared_input'])
                replayed_generated = replay_scan(native_input)
                replayed_norm = replay_norm(native_shared)
        finally:
            torch.set_num_threads(previous_threads)
        errors = {
            'reference_scan_replay': float(np.max(np.abs(
                replayed_reference - direct_shared))),
            'connected_scan_input': float(np.max(np.abs(
                native_input - direct['predictor_shared_input']))),
            'local_scan': float(np.max(np.abs(
                native_shared - replayed_generated))),
            'connected_shared_output': float(np.max(np.abs(
                native_shared - direct_shared))),
            'local_first_adain': float(np.max(np.abs(
                native_norm - replayed_norm))),
            'connected_first_adain': float(np.max(np.abs(
                native_norm - direct['predictor_F0_0_norm1']))),
        }
        self.assertLessEqual(errors['reference_scan_replay'], 1e-5)
        self.assertLessEqual(errors['local_scan'], 3e-6)
        self.assertLessEqual(errors['local_first_adain'], 2e-6)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.F0.pinned-same-input-replay-v1',
            'name': 'pinned PyTorch same-input shared scan and first AdaIN replay',
            'provider': 'generated_prosody_complete',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'live-pytorch-2.8.0+cpu-and-direct-full-model-hooks',
            'backend_version': torch.__version__,
            'status': 'pass', 'gate': 'diagnostic', 'blocking': False,
            'max_diff': max(errors['local_scan'], errors['local_first_adain']),
            'tolerance': 3e-6,
            'error_decomposition': errors,
            'configuration': '36 phonemes; A=103; af_heart; one torch thread',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_generated_prosody_complete.'
                'KokoroGeneratedProsodyCompleteTest.'
                'test_pinned_pytorch_same_input_replay_separates_upstream_and_local_error',
        }))

    def test_second_blocks_match_direct_full_model_hooks(self):
        self.assertEqual(self.calls['errors'], [])
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        doubled = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames),
                                 ctypes.byref(doubled)), 0)
        self.assertEqual((frames.value, doubled.value), (103, 206))
        worst = (0., None)
        f0_stage_errors = []
        for branch, stem in (('F0', 'f0'), ('N', 'n')):
            first_block = self.view(arena, f'{stem}_block0_output').reshape(512, 128)
            first_expected = self.direct[f'predictor_{branch}_0']
            first_error = np.abs(first_block[:, :103] - first_expected)
            if branch == 'F0':
                f0_stage_errors.append(('block0', first_error,
                                        first_block[:, :103], first_expected,
                                        1e-4))
            self.assertTrue(np.isfinite(first_block[:, :103]).all())
            self.assertLessEqual(float(np.max(first_error)), 1e-4)
            self.assertTrue(np.all(first_block[:, 103:] == -777.))
            actual = self.view(arena, f'{stem}_block1_output').reshape(256, 256)
            expected = self.direct[f'predictor_{branch}_1']
            self.assertTrue(np.isfinite(actual[:, :206]).all())
            error = np.abs(actual[:, :206] - expected)
            if branch == 'F0':
                f0_stage_errors.append(('block1', error,
                                        actual[:, :206], expected, 1e-4))
            point = np.unravel_index(np.argmax(error), error.shape)
            if float(error[point]) > worst[0]:
                worst = (float(error[point]),
                         (branch, *(int(x) for x in point)))
            self.assertLessEqual(float(error[point]), 1e-4,
                                 (branch, point, float(error[point])))
            self.assertTrue(np.all(actual[:, 206:] == -91.))
            third = self.view(arena, f'{stem}_block2_output').reshape(256, 256)
            third_expected = self.direct[f'predictor_{branch}_2']
            third_error = np.abs(third[:, :206] - third_expected)
            if branch == 'F0':
                f0_stage_errors.append(('block2', third_error,
                                        third[:, :206], third_expected, 1e-4))
            third_point = np.unravel_index(np.argmax(third_error),
                                           third_error.shape)
            self.assertTrue(np.isfinite(third[:, :206]).all())
            self.assertLessEqual(float(third_error[third_point]), 1e-4,
                (branch, 'third block', third_point,
                 float(third_error[third_point])))
            self.assertTrue(np.all(third[:, 206:] == -91.))
            final = self.view(arena, f'{stem}_output').reshape(1, 256)
            final_expected = self.direct[f'predictor_{branch}_proj']
            self.assertTrue(np.isfinite(final[:, :206]).all())
            final_error = np.abs(final[:, :206] - final_expected)
            if branch == 'F0':
                f0_stage_errors.append(('projection', final_error,
                                        final[:, :206], final_expected, 5e-4))
            final_point = np.unravel_index(np.argmax(final_error),
                                           final_error.shape)
            projection_weight, projection_bias = self.projection_weights[branch]
            fp64_actual = (projection_weight.reshape(256).astype(np.float64)
                           @ third[:, :206].astype(np.float64) +
                           float(projection_bias[0]))
            fp64_expected = (projection_weight.reshape(256).astype(np.float64)
                             @ third_expected.astype(np.float64) +
                             float(projection_bias[0]))
            # F0's 256-channel projection amplifies the upstream block error.
            # Keep the three contributions separate from a final comparison:
            # generated block drift, native reduction, and oracle reduction.
            self.assertLessEqual(float(np.max(np.abs(
                fp64_actual - fp64_expected))), 4e-4)
            self.assertLessEqual(float(np.max(np.abs(
                final[0, :206] - fp64_actual))), 2e-4)
            self.assertLessEqual(float(np.max(np.abs(
                final_expected[0] - fp64_expected))), 2e-4)
            if float(final_error[final_point]) > worst[0]:
                worst = (float(final_error[final_point]),
                         (branch, 'projection',
                          *(int(x) for x in final_point)))
            self.assertLessEqual(float(final_error[final_point]),
                                 5e-4 if branch == 'F0' else 1e-4,
                (branch, 'projection', final_point,
                 float(final_error[final_point])))
            self.assertTrue(np.all(final[:, 206:] == -91.))
        for stage, error, actual, reference, limit in f0_stage_errors:
            point = np.unravel_index(np.argmax(error), error.shape)
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.complete-prosody.F0-{stage}.drift-v1',
                'name': f'Kokoro generated F0 {stage} full-model drift',
                'provider': 'generated_prosody_complete',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned-full-kmodel-pytorch28',
                'backend_version': self.direct_meta['environment']['torch'],
                'status': 'pass' if float(error[point]) <= limit else 'fail',
                'gate': 'diagnostic', 'blocking': False,
                'reason': 'connected checkpoint; separate identical-input '
                          'stage replay isolates local arithmetic drift',
                'max_diff': float(error[point]),
                'worst_index': list(map(int, point)),
                'actual': float(actual[point]),
                'reference': float(reference[point]),
                'rmse': float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                'tolerance': limit,
                'configuration': '36 phonemes; A=103; 2A=206; af_heart',
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_kokoro_generated_prosody_complete.'
                    'KokoroGeneratedProsodyCompleteTest.'
                    'test_second_blocks_match_direct_full_model_hooks'}))
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.complete-prosody.generated-v1',
            'name': 'generated complete F0/noise outputs versus direct pinned KModel hooks',
            'provider': 'generated_prosody_complete',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pinned-full-kmodel-pytorch28',
            'backend_version': self.direct_meta['environment']['torch'],
            'status': 'pass', 'max_diff': worst[0], 'worst_index': worst[1],
            'tolerance': 5e-4,
            'checkpoint_tolerances': {'first_block': 1e-4,
                                      'second_block': 1e-4,
                                      'third_block': 1e-4,
                                      'F0_final': 5e-4,
                                      'N_final': 1e-4},
            'configuration': '36 phonemes; A=103 of 128; 2A=206 of 256; af_heart',
            'reproduction_command': 'python3 -m unittest ' + self.id()}))

    def test_identical_reference_inputs_localize_f0_stage_drift(self):
        shared, shared_meta = verified_fixture('prosody_shared')
        self.assertEqual(shared_meta['model_pin'], self.direct_meta['model_pin'])
        graph = copy.deepcopy(author.build_circuit())
        graph['name'] = 'kokoro_f0_stage_reference_input_replay'
        graph['native_entry']['function'] = 'ck_kokoro_f0_stage_reference_input_replay'
        references = {
            'shared': (shared['output'].T, 512, 128, 103),
            'block0': (self.direct['predictor_F0_0'], 512, 128, 103),
            'block1': (self.direct['predictor_F0_1'], 256, 256, 206),
            'block2': (self.direct['predictor_F0_2'], 256, 256, 206),
        }
        for name, (_values, channels, capacity, _length) in references.items():
            buffer = f'reference_f0_{name}'
            graph['activation_buffers'][buffer] = {
                'shape': [channels, capacity]}
            graph['activation_bindings'][buffer] = buffer

        def replace(section, operation, port, reference):
            block = graph['block_types'][section]
            entries = block['footer'] if section == 'prosody_norm' else \
                block['body']['ops']
            item = next(value for value in entries if value['id'] == operation)
            item['graph_slots']['inputs'][port] = \
                f'external:reference_f0_{reference}'

        for section, operation, port, reference in (
                ('prosody_norm', 'f0_norm0_output', 'input', 'shared'),
                ('prosody_first_block', 'f0_block0_output', 'left', 'shared'),
                ('prosody_second_block', 'f0_block1_norm0_output', 'input', 'block0'),
                ('prosody_second_block', 'f0_block1_shortcut_upsample',
                 'input', 'block0'),
                ('prosody_final', 'f0_block2_norm0_output', 'input', 'block1'),
                ('prosody_final', 'f0_block2_output', 'left', 'block1'),
                ('prosody_final', 'f0_output', 'input', 'block2')):
            replace(section, operation, port, reference)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'f0-stage-reference-input-replay.json'
            path.write_text(json.dumps(graph))
            source = copy.deepcopy(self.source)
            source['template'] = graph
            source['config']['model'] = graph['name']
            source['config']['arch'] = graph['name']
            layout, calls, _library, _loaded, function = \
                compile_native_graph(root, source, path)
            self.assertEqual(calls['errors'], [])
            function.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                                 ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int32),
                                 ctypes.POINTER(ctypes.c_int32)]
            buffers = {item['name']: item for item in
                       layout['memory']['activations']['buffers']}
            stages = (
                ('block0', 'f0_block0_output', 'predictor_F0_0', 512, 128, 103),
                ('block1', 'f0_block1_output', 'predictor_F0_1', 256, 256, 206),
                ('block2', 'f0_block2_output', 'predictor_F0_2', 256, 256, 206),
                ('projection', 'f0_output', 'predictor_F0_proj', 1, 256, 206),
            )

            def execute(zero_references=False):
                arena = populated_duration_arena(layout, self.entries,
                    self.bump, self.encoder, self.duration)
                for name, (values, channels, capacity, length) in references.items():
                    item = buffers[f'reference_f0_{name}']
                    supplied = np.ndarray((channels, capacity), np.float32,
                                          buffer=arena, offset=item['abs_offset'])
                    supplied[:] = -91.
                    supplied[:, :length] = 0. if zero_references else values
                frames = ctypes.c_int32(-999)
                doubled = ctypes.c_int32(-999)
                self.assertEqual(function(arena, len(arena),
                    ctypes.byref(frames), ctypes.byref(doubled)), 0)
                self.assertEqual((frames.value, doubled.value), (103, 206))
                outputs = {}
                for stage, buffer, _oracle, channels, capacity, length in stages:
                    item = buffers[buffer]
                    actual = np.ndarray((channels, capacity), np.float32,
                                        buffer=arena, offset=item['abs_offset'])
                    self.assertTrue(np.isfinite(actual[:, :length]).all())
                    outputs[stage] = actual[:, :length].copy()
                return outputs

            baseline = execute()
            for stage, _buffer, oracle, _channels, _capacity, _length in stages:
                actual = baseline[stage]
                reference = self.direct[oracle]
                error = np.abs(actual - reference)
                point = np.unravel_index(np.argmax(error), error.shape)
                worst = float(error[point])
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': f'kokoro.F0.{stage}.identical-input-replay-v1',
                    'name': f'Kokoro F0 {stage} identical-input replay',
                    'provider': 'generated_prosody_complete',
                    'dtype': 'fp32', 'direction': 'inference',
                    'oracle': 'pinned-full-kmodel-pytorch28',
                    'status': 'pass' if worst <= 5e-5 else 'fail',
                    'gate': 'diagnostic', 'blocking': False,
                    'max_diff': worst,
                    'worst_index': list(map(int, point)),
                    'actual': float(actual[point]),
                    'reference': float(reference[point]),
                    'tolerance': 5e-5,
                    'configuration': '36 phonemes; af_heart; captured upstream input at each F0 stage',
                    'reproduction_command': 'python3 -m unittest '
                        'tests.test_v8_kokoro_generated_prosody_complete.'
                        'KokoroGeneratedProsodyCompleteTest.'
                        'test_identical_reference_inputs_localize_f0_stage_drift'}))

            changed = execute(zero_references=True)
            for input_stage, output_stage in (
                    ('shared', 'block0'), ('block0', 'block1'),
                    ('block1', 'block2'), ('block2', 'projection')):
                self.assertGreater(float(np.max(np.abs(
                    changed[output_stage] - baseline[output_stage]))), 1e-4,
                    f'{input_stage} is disconnected from {output_stage}')


    def test_repeated_lengths_and_rejection_preserve_outputs(self):
        arena = self.arena()
        first = ctypes.c_int32(-999); second = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                 ctypes.byref(second)), 0)
        reference = self.view(arena, 'f0_block1_output').copy()
        self.view(arena, 'f0_block1_output')[:] = -91.
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                 ctypes.byref(second)), 0)
        np.testing.assert_array_equal(
            self.view(arena, 'f0_block1_output').reshape(256, 256)[:, :206],
            reference.reshape(256, 256)[:, :206])
        weight = self.weight_layout['duration_prosody.duration_head.weight']
        bias = self.weight_layout['duration_prosody.duration_head.bias']
        np.ndarray((weight['size']//4,), np.float32, buffer=arena,
                   offset=weight['abs_offset'])[:] = 0.
        biases = np.ndarray((bias['size']//4,), np.float32, buffer=arena,
                            offset=bias['abs_offset'])
        for bias_value, expected in ((-3.2, 72), (-10., 36)):
            biases[:] = bias_value
            for branch in ('f0', 'n'):
                self.view(arena, f'{branch}_output')[:] = -91.
            self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                     ctypes.byref(second)), 0)
            self.assertEqual((first.value, second.value),
                             (expected, 2 * expected))
            for branch in ('f0', 'n'):
                output = self.view(arena, f'{branch}_output').reshape(1, 256)
                self.assertTrue(np.isfinite(output[:, :2 * expected]).all())
                self.assertTrue(np.all(output[:, 2 * expected:] == -91.))
        # A corrupt upstream duration fails before 2A becomes valid or a
        # downstream output is published.
        self.view(arena, 'f0_block1_output')[:] = -91.
        biases[:] = 1.
        first.value = second.value = -999
        self.assertNotEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                    ctypes.byref(second)), 0)
        self.assertEqual((first.value, second.value), (-999, -999))
        self.assertTrue(np.all(self.view(arena, 'f0_block1_output') == -91.))

    def test_wrong_final_branch_weight_is_detected(self):
        arena = self.arena()
        source = self.weight_layout['duration_prosody.N_proj.weight']
        target = self.weight_layout['duration_prosody.F0_proj.weight']
        self.assertEqual(source['size'], target['size'])
        arena[target['abs_offset']:target['abs_offset'] + target['size']] = \
            arena[source['abs_offset']:source['abs_offset'] + source['size']]
        first = ctypes.c_int32(-999); second = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                 ctypes.byref(second)), 0)
        actual = self.view(arena, 'f0_output').reshape(1, 256)[:, :206]
        self.assertGreater(float(np.max(np.abs(actual -
            self.direct['predictor_F0_proj']))), 1e-3)

    def test_xray_final_curves_resolve_selected_provider(self):
        arena = self.arena()
        first = ctypes.c_int32(-999); second = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(first),
                                 ctypes.byref(second)), 0)
        tensor_report = {'torch': {'tensors': {}}, 'comparisons': {}}
        points = {}
        for branch, stem in (('F0', 'f0'), ('N', 'n')):
            name = f'{stem}_output'
            point = next(item for operation in self.calls['operations']
                for item in operation.get('semantic_checkpoints', [])
                if item['tensor'] == name)
            self.assertEqual(point['resolved_contract_id'],
                             'audio_conv1d_checked_scalar_fma_fp32')
            selector = name if point['layer'] < 0 else \
                f'{name}@{point["layer"]}'
            native = self.root / f'{name}-native.f32'
            oracle = self.root / f'{name}-pytorch.f32'
            self.view(arena, name).tofile(native)
            self.direct[f'predictor_{branch}_proj'].tofile(oracle)
            tensor_report['torch']['tensors'][selector] = {
                'path': str(oracle), 'shape': [1, 206]}
            tensor_report['comparisons'][selector] = {
                'ck_path': str(native), 'shape': [1, 206],
                'physical_shape': [1, 256], 'capacity_shape': [1, 256],
                'valid_shape': [1, 206], 'physical_strides': [256, 1]}
            points[name] = point
        builder = xray_support.xray_builder
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_prosody_complete_bounded')
        subject = builder.build_manifest(backend='ck', call_ir=self.calls,
            tensor_report=tensor_report, model='kokoro_prosody_complete_bounded',
            source='generated_native', phase='prefill',
            loaded_library=self.library, runtime_library=runtime)
        oracle = builder.build_manifest(backend='pytorch', call_ir=self.calls,
            tensor_report=tensor_report, model='kokoro_prosody_complete_bounded',
            source='pinned_full_kmodel', phase='prefill')
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        for name, point in points.items():
            ceiling = 5e-4 if name == 'f0_output' else 1e-4
            profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
                'name': 'kokoro_complete_prosody', 'backend': 'pytorch',
                'contract_schema_version': 1,
                'required_match_fields': ['checkpoint_id', 'producer',
                    'logical_layout', 'axis_names', 'resolved_contract_id',
                    'kernel_id', 'function'],
                'observed_storage': {'default': 'fp32', 'checkpoints': {}},
                'dtype_thresholds': {'fp32': {'cosine_min': .99999,
                    'rmse_max': ceiling, 'relative_rmse_max': ceiling,
                    'max_abs_max': ceiling, 'finite_required': True}},
                'checkpoint_order': [point['id']],
                'interval_expansions': {}, 'backend_mappings': {}}
            one = lambda manifest: {**manifest, 'checkpoints': [entry for entry in
                manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
            report = xray_support.xray.compare_manifests(
                one(subject), one(oracle), profile)
            self.assertEqual(report['status'], 'pass', (name, report))


if __name__ == '__main__':
    unittest.main()
