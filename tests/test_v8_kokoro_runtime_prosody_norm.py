"""Connected runtime-length Kokoro shared scan and first F0/noise AdaIN."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_duration_fixture_support import prepare_duration_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
from build_kokoro_runtime_prosody_norm_circuit import build_circuit

CAPACITY = 64
FRAMES = CAPACITY * 50


def pinned(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['fixture_sha256']:
        raise RuntimeError(f'fixture hash mismatch: {path}')
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, expected in meta.get('array_sha256', {}).items():
        if hashlib.sha256(arrays[key].tobytes()).hexdigest() != expected:
            raise RuntimeError(f'array hash mismatch: {path}:{key}')
    return arrays, meta


class KokoroRuntimeProsodyNormTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        graph = build_circuit()
        circuit = root / 'circuit.json'
        circuit.write_text(json.dumps(graph))
        base = ROOT / 'tests/fixtures/tts'
        text, text_meta = pinned(base / 'kokoro_text_encoder_pinned.npz')
        norm, norm_meta = pinned(base / 'kokoro_prosody_norm_pinned.npz')
        shared, shared_meta = pinned(base / 'kokoro_prosody_shared_pinned.npz')
        if (shared_meta['model_pin'] != text_meta['model_pin']['revision'] or
            norm_meta['model_pin'] != text_meta['model_pin']['revision']):
            raise RuntimeError('text and prosody fixtures use different models')
        cls.norm = norm
        cls.norm_meta = norm_meta
        embedding, _ = pinned(base / 'kokoro_text_embedding_pinned.npz')
        weights = {'acoustic_text_encoder.embedding.weight': embedding['table']}
        for index in range(3):
            for kind in ('weight', 'bias'):
                weights[f'acoustic_text_encoder.conv{index}.{kind}'] = text[
                    f'conv{index}_{kind}']
            weights[f'acoustic_text_encoder.norm{index}.weight'] = text[
                f'norm{index}_gamma']
            weights[f'acoustic_text_encoder.norm{index}.bias'] = text[
                f'norm{index}_beta']
        for kind in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
            weights[f'acoustic_text_encoder.lstm.{kind}'] = text[f'lstm_{kind}']
            weights[f'duration_prosody.shared_scan.{kind}'] = shared[kind]
        for branch in ('F0', 'N'):
            prefix = f'duration_prosody.{branch}.0.norm1'
            for kind in ('fc_weight', 'fc_bias', 'norm_weight', 'norm_bias'):
                weights[f'{prefix}.{kind.replace("_", ".")}'] = norm[
                    f'{branch}_{kind}']
        cls.fixture = prepare_duration_fixture(root, circuit, weights)
        cls.fixture.source['config']['activation_buffer_dtypes'].update({
            'request_length': 'i32', 'valid_length': 'i32',
            'valid_residual_elements': 'i32'})
        cls.fixture.source['config']['context_length'] = CAPACITY
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = (
            compile_native_graph(root, cls.fixture.source, circuit))
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}
        cls.ids = {36: cls.fixture.encoder['word_ids']}
        for count in (2, 8, 64):
            cls.ids[count] = pinned(
                base / f'kokoro_encoder_{count}_pinned.npz')[0]['word_ids']

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
        return arena

    def execute(self, arena, count):
        ids = self.view(arena, 'word_ids', (CAPACITY,), np.int32)
        ids[:] = 178
        ids[:count] = self.ids[count]
        self.view(arena, 'request_length', (1,), np.int32)[0] = count
        outputs = {name: self.view(arena, name, (512, FRAMES)) for name in
                   ('f0_norm0_output', 'n_norm0_output')}
        for output in outputs.values():
            output[:] = -91
        results = [ctypes.c_int32(-1) for _ in range(3)]
        status = self.fn(arena, len(arena), *(ctypes.byref(x) for x in results))
        return status, [x.value for x in results], outputs

    def test_connected_branches_and_changing_lengths(self):
        self.assertEqual(self.calls['errors'], [])
        for name in ('prosody_shared_channel_transpose',
                     'f0_norm0_output', 'n_norm0_output'):
            op = next(item for item in self.calls['operations']
                      if item.get('template_op_id') == name)
            self.assertIn('expanded_frames', op['consumes_runtime_lengths'])
        arena = self.arena()
        first_short = None
        for count in (2, 8, 36, 64, 2):
            with self.subTest(count=count):
                status, extents, outputs = self.execute(arena, count)
                self.assertEqual(status, 0)
                frames = extents[2]
                self.assertEqual(extents[:2], [count, count * 768])
                self.assertGreater(frames, 0)
                self.assertLessEqual(frames, FRAMES)
                for branch, stem in (('F0', 'f0'), ('N', 'n')):
                    output = outputs[f'{stem}_norm0_output']
                    self.assertTrue(np.isfinite(output[:, :frames]).all())
                    self.assertTrue(np.all(output[:, frames:] == -91))
                    if count == 36:
                        self.assertEqual(frames, 103)
                        style = self.view(arena, f'{stem}_norm0_style', (1024,))
                        self.assertTrue(np.isfinite(style).all())
                        np.testing.assert_allclose(
                            style, self.norm[f'{branch}_style_affine'],
                            rtol=0, atol=5e-5)
                        error = np.abs(output[:, :frames] -
                                       self.norm[f'{branch}_output'])
                        worst = np.unravel_index(np.argmax(error), error.shape)
                        rmse = float(np.sqrt(np.mean(error.astype(np.float64)**2)))
                        self.assertLessEqual(float(error[worst]), 5e-5,
                                             (branch, worst, float(error[worst])))
                        print('CKE_NUMERICAL_CASE ' + json.dumps({
                            'case_id': f'kokoro.runtime-{stem}-first-adain.36',
                            'name': f'runtime Kokoro {branch} first AdaIN',
                            'provider': 'generated_kokoro_runtime_prosody_norm',
                            'dtype': 'fp32', 'direction': 'inference',
                            'oracle': 'pinned shared PyTorch 2.8 plus PyTorch 2.13 operation capture',
                            'status': 'pass', 'gate': 'composition',
                            'blocking': True,
                            'max_diff': float(error[worst]),
                            'rmse': rmse,
                            'worst_index': [int(value) for value in worst],
                            'actual_at_worst': float(output[worst]),
                            'reference_at_worst': float(
                                self.norm[f'{branch}_output'][worst]),
                            'tolerance': 5e-5,
                            'configuration': 'capacity=64 tokens, valid=36, frames=103',
                            'reproduction_command': 'python3 -m unittest '
                                'tests.test_v8_kokoro_runtime_prosody_norm',
                        }, sort_keys=True))
                if count == 2 and first_short is None:
                    first_short = (frames, {key: value[:, :frames].copy()
                                            for key, value in outputs.items()})
                elif count == 2:
                    self.assertEqual(frames, first_short[0])
                    for key, value in outputs.items():
                        np.testing.assert_array_equal(value[:, :frames],
                                                      first_short[1][key])

    def test_upstream_failure_stops_both_branches_and_recovers(self):
        arena = self.arena()
        status, baseline_extent, baseline = self.execute(arena, 8)
        self.assertEqual(status, 0)
        baseline = {key: value[:, :baseline_extent[2]].copy()
                    for key, value in baseline.items()}
        style = self.view(arena, 'predictor_style', (128,))
        original = float(style[0])
        style[0] = np.nan
        status, extents, outputs = self.execute(arena, 8)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1, -1, -1])
        for output in outputs.values():
            self.assertTrue(np.all(output == -91))
        style[0] = original
        status, extents, outputs = self.execute(arena, 8)
        self.assertEqual(status, 0)
        for key, value in outputs.items():
            np.testing.assert_array_equal(value[:, :extents[2]], baseline[key])

    def test_f0_style_failure_stops_noise_branch_and_recovers(self):
        arena = self.arena()
        status, baseline_extent, outputs = self.execute(arena, 8)
        self.assertEqual(status, 0)
        baseline = {key: value[:, :baseline_extent[2]].copy()
                    for key, value in outputs.items()}
        name = 'duration_prosody.F0.0.norm1.fc.weight'
        planned = self.weights[name]
        reference = self.fixture.entries[name]
        weight = np.ndarray((planned['size'] // 4,), np.float32,
                            buffer=arena, offset=planned['abs_offset'])
        weight[0] = np.nan
        status, extents, outputs = self.execute(arena, 8)
        self.assertNotEqual(status, 0)
        self.assertEqual(extents, [-1, -1, -1])
        for output in outputs.values():
            self.assertTrue(np.all(output == -91))
        start = reference['file_offset']
        arena[planned['abs_offset']:planned['abs_offset'] + reference['size']] = (
            self.fixture.bump[start:start + reference['size']])
        status, extents, outputs = self.execute(arena, 8)
        self.assertEqual(status, 0)
        for key, value in outputs.items():
            np.testing.assert_array_equal(value[:, :extents[2]], baseline[key])


if __name__ == '__main__':
    unittest.main()
