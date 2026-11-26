"""Generated Kokoro duration features and oracle-fed text features to valid frames."""
import ctypes
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_checked_graph_test_support import build_ir_v8
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests import test_v8_kokoro_generated_albert_layer as first_layer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_expansion_circuit as author


class KokoroGeneratedExpansionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence = os.environ.get('CKE_KOKORO_EXPANSION_EVIDENCE_DIR')
        if evidence:
            root = Path(evidence).resolve() / 'expansion'
            root.mkdir(parents=True, exist_ok=True)
            cls.temp = SimpleNamespace(name=str(root), cleanup=lambda: None)
        else:
            cls.temp = tempfile.TemporaryDirectory()
            root = Path(cls.temp.name)
        circuit = materialize_stage_circuit(root, author)
        fixture = prepare_duration_fixture(root, circuit)
        for key, value in vars(fixture).items():
            setattr(cls, key, value)
        cls.layout, cls.calls, cls.library, cls.loaded, cls.fn = compile_native_graph(
            root, cls.source, circuit)
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32)]
        cls.buffers = {item['name']: item for item in
                       cls.layout['memory']['activations']['buffers']}
        cls.weights = {item['name']: item for item in
                       cls.layout['memory']['weights']['entries']}
        cls.root = root
        fixture_dir = ROOT / 'tests/fixtures/tts'
        cls.reference_meta = json.loads((fixture_dir / 'duration_two_stream_reference.json').read_text())
        with np.load(fixture_dir / 'duration_two_stream_reference.npz') as archive:
            cls.reference = {name: archive[name].copy() for name in archive.files}
        for name, digest in cls.reference_meta['arrays_sha256'].items():
            if hashlib.sha256(cls.reference[name].tobytes()).hexdigest() != digest:
                raise AssertionError('reference fixture hash mismatch: ' + name)

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
        self.view(arena, 'text_features')[:] = self.reference['text_features'].ravel()
        self.view(arena, 'duration_expanded')[:] = -99
        self.view(arena, 'text_expanded')[:] = -98
        return arena

    def test_connected_generated_expansions_match_pinned_reference(self):
        self.assertEqual(len(self.calls['operations']), 162)
        duration_op, text_op = self.calls['operations'][-2:]
        self.assertEqual([op['function'] for op in (duration_op, text_op)], [
            'audio_duration_expand_token_major_f32',
            'audio_duration_expand_channel_major_f32'])
        self.assertEqual(duration_op['errors'], [])
        self.assertEqual(text_op['errors'], [])
        for op in (duration_op, text_op):
            args = {arg['name']: arg['expr'] for arg in op['args']}
            self.assertEqual(args['expanded_frames'], 'runtime_extents.expanded_frames')
            self.assertEqual(args['output_stride'], '128')
        self.assertIn('A_PREDICTOR_FEATURES',
                      next(a['expr'] for a in duration_op['args'] if a['name'] == 'features'))
        self.assertIn('A_TEXT_FEATURES',
                      next(a['expr'] for a in text_op['args'] if a['name'] == 'features'))
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        self.assertEqual(frames.value, 103)
        np.testing.assert_array_equal(self.view(arena, 'runtime_values', np.int32)[:36],
                                      self.duration['durations'])
        generated_features = self.view(arena, 'predictor_features').reshape(36, 640)
        generated_duration = self.view(arena, 'duration_expanded').reshape(640, 128)
        expected_exact_copy = np.repeat(generated_features.T, self.duration['durations'], axis=1)
        np.testing.assert_array_equal(generated_duration[:, :103], expected_exact_copy)
        error = np.abs(generated_duration[:, :103].astype(np.float64) -
                       self.reference['duration_expected'].astype(np.float64))
        self.assertTrue(np.isfinite(generated_duration[:, :103]).all())
        self.assertLessEqual(float(error.max()), 3e-5)
        self.assertTrue(np.all(generated_duration[:, 103:] == -99))
        generated_text = self.view(arena, 'text_expanded').reshape(640, 128)
        np.testing.assert_array_equal(generated_text[:512, :103],
                                      self.reference['text_expected'])
        self.assertTrue(np.all(generated_text[:512, 103:] == -98))
        self.assertTrue(np.all(generated_text[512:] == -98))
        report = {'status': 'PASS', 'scope': 'generated duration features; oracle-fed text features',
                  'frames': frames.value, 'duration_max_abs': float(error.max()),
                  'text_max_abs': 0.0, 'output_stride': 128,
                  'library_sha256': hashlib.sha256(self.library.read_bytes()).hexdigest(),
                  'reproduce': 'python3 -m unittest tests.test_v8_kokoro_generated_expansion'}
        (self.root / 'expansion-evidence.json').write_text(json.dumps(report, indent=2))
        for stream, maximum, tolerance, provider, feed in (
                ('duration', float(error.max()), 3e-5,
                 'audio_duration_expand_token_major_f32', 'generated'),
                ('text', 0.0, 0.0,
                 'audio_duration_expand_channel_major_f32', 'pinned-oracle')):
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.expansion.{stream}.connected-vs-pytorch28',
                'name': f'Kokoro {stream} feature expansion',
                'status': 'pass', 'max_diff': maximum, 'tolerance': tolerance,
                'provider': provider, 'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned-pytorch', 'backend_version': '2.8.0+cpu',
                'configuration': f'tokens=36; valid_frames=103; stride=128; input={feed}',
                'reproduction_command':
                    'python3 -m unittest tests.test_v8_kokoro_generated_expansion',
            }, sort_keys=True))

    def test_failed_producers_do_not_publish_extent_or_touch_expansions(self):
        for failure in ('nan_style', 'excess_duration', 'small_arena'):
            with self.subTest(failure=failure):
                arena = self.arena()
                if failure == 'nan_style':
                    self.view(arena, 'predictor_style')[0] = np.nan
                elif failure == 'excess_duration':
                    bias = self.weights['duration_prosody.duration_head.bias']
                    np.ndarray((50,), np.float32, buffer=arena,
                               offset=bias['abs_offset'])[:] = 1000.
                frames = ctypes.c_int32(-999)
                size = len(arena) - 1 if failure == 'small_arena' else len(arena)
                self.assertNotEqual(self.fn(arena, size, ctypes.byref(frames)), 0)
                self.assertEqual(frames.value, -999)
                self.assertTrue(np.all(self.view(arena, 'duration_expanded') == -99))
                self.assertTrue(np.all(self.view(arena, 'text_expanded') == -98))

    def test_xray_captures_valid_extent_and_physical_stride(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
        builder, xray = first_layer.xray_builder, first_layer.xray
        tensor_report = {'torch': {'tensors': {}}, 'comparisons': {}}
        for tensor, oracle, channels in (
                ('duration_expanded', 'duration_expected', 640),
                ('text_expanded', 'text_expected', 512)):
            point = next(point for op in self.calls['operations']
                         for point in op.get('semantic_checkpoints', [])
                         if point['tensor'] == tensor)
            selector = tensor if point['layer'] < 0 else f'{tensor}@{point["layer"]}'
            native_path = self.root / f'{tensor}-native.f32'
            oracle_path = self.root / f'{tensor}-pytorch.f32'
            self.view(arena, tensor).tofile(native_path)
            self.reference[oracle].tofile(oracle_path)
            tensor_report['torch']['tensors'][selector] = {
                'path': str(oracle_path), 'shape': [channels, frames.value]}
            tensor_report['comparisons'][selector] = {
                'ck_path': str(native_path), 'shape': [channels, frames.value],
                'physical_shape': [640, 128], 'capacity_shape': [channels, 128],
                'valid_shape': [channels, frames.value], 'physical_strides': [128, 1]}
        runtime = builder.capture_runtime_library_identity(
            self.loaded, 'ck_kokoro_duration_expansion')
        subject = builder.build_manifest(
            backend='ck', call_ir=self.calls, tensor_report=tensor_report,
            model='kokoro_duration_expansion_bounded', source='generated_native',
            phase='prefill', loaded_library=self.library, runtime_library=runtime)
        oracle = builder.build_manifest(
            backend='pytorch', call_ir=self.calls, tensor_report=tensor_report,
            model='kokoro_duration_expansion_bounded', source='pinned_capture',
            phase='prefill')
        for tensor, limit in (('duration_expanded', 3e-5), ('text_expanded', 0.0)):
            point = next(point for op in self.calls['operations']
                         for point in op.get('semantic_checkpoints', [])
                         if point['tensor'] == tensor)
            # Selected call-site provider maps carry distinct numerical
            # capabilities despite sharing the semantic expansion operation.
            self.assertEqual(point['resolved_contract_id'],
                'audio_duration_expand_token_major_checked_strided_copy_fp32'
                if tensor == 'duration_expanded' else
                'audio_duration_expand_checked_strided_copy_fp32')
            profile = {'schema': 'cke.parity_profile', 'schema_version': 1,
                'name': 'kokoro_connected_expansion', 'backend': 'pytorch',
                'contract_schema_version': 1,
                'required_match_fields': ['checkpoint_id', 'producer',
                    'logical_layout', 'axis_names', 'resolved_contract_id',
                    'kernel_id', 'function'],
                'observed_storage': {'default': 'fp32', 'checkpoints': {}},
                'dtype_thresholds': {'fp32': {'cosine_min': 0.99999,
                    'rmse_max': 1e-5 if limit else 0.0,
                    'relative_rmse_max': 1e-5 if limit else 0.0,
                    'max_abs_max': limit, 'finite_required': True}},
                'checkpoint_order': [point['id']], 'interval_expansions': {},
                'backend_mappings': {}}
            one = lambda manifest: {**manifest, 'checkpoints': [entry for entry
                in manifest['checkpoints'] if entry['checkpoint_id'] == point['id']]}
            result = xray.compare_manifests(one(subject), one(oracle), profile)
            self.assertEqual(result['status'], 'pass', (tensor, result))
            (self.root / f'{tensor}-xray.json').write_text(json.dumps(result, indent=2))
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        (self.root / 'native-checkpoints.json').write_text(json.dumps(subject, indent=2))
        (self.root / 'pytorch-checkpoints.json').write_text(json.dumps(oracle, indent=2))

    def test_repeated_execution_rewrites_valid_region_only(self):
        arena = self.arena()
        frames = ctypes.c_int32(-999)
        for repeat in range(2):
            self.assertEqual(self.fn(arena, len(arena), ctypes.byref(frames)), 0)
            self.assertEqual(frames.value, 103)
            self.assertTrue(np.all(self.view(arena, 'duration_expanded').reshape(640, 128)[:, 103:] == -99))
            self.view(arena, 'duration_expanded').reshape(640, 128)[:, :103] = -99
            self.view(arena, 'text_expanded').reshape(640, 128)[:512, :103] = -98
            frames.value = -999

    def test_compiler_rejects_false_expansion_storage_claims(self):
        registry = build_ir_v8.load_kernel_registry()
        for operation, key, value in (
                ('expand_generated_duration', 'input_elements', 36 * 640 + 1),
                ('expand_generated_duration', 'input_stride', 641),
                ('expand_generated_duration', 'phoneme_count', 37),
                ('expand_oracle_text', 'output_elements', 640 * 128 + 1),
                ('expand_oracle_text', 'output_stride', 129)):
            with self.subTest(operation=operation, key=key):
                source = copy.deepcopy(self.source)
                ops = source['template']['block_types']['duration_expansion']['body']['ops']
                target = next(op for op in ops if op['id'] == operation)
                target['params']['call_constants'][key] = value
                circuit = self.root / f'bad-{operation}-{key}.json'
                circuit.write_text(json.dumps(source['template']))
                ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode='prefill')
                with self.assertRaises((RuntimeError, ValueError)):
                    lower1 = build_ir_v8.generate_ir_lower_1(
                        ir1, registry, source, 'prefill')
                    layout = build_ir_v8.generate_memory_layout(
                        lower1, source, registry, mode='prefill', context_len=36)
                    lower2 = build_ir_v8.generate_ir_lower_2(
                        lower1, layout, source, registry, mode='prefill')
                    build_ir_v8.generate_ir_lower_3(lower2, mode='prefill')

    def test_swapped_feature_edges_fail_physical_stride_contract(self):
        registry = build_ir_v8.load_kernel_registry()
        for operation, wrong_source in (
                ('expand_generated_duration', 'external:text_features'),
                ('expand_oracle_text', 'predictor_features')):
            with self.subTest(operation=operation):
                source = copy.deepcopy(self.source)
                ops = source['template']['block_types']['duration_expansion']['body']['ops']
                target = next(op for op in ops if op['id'] == operation)
                target['graph_slots']['inputs']['features'] = wrong_source
                circuit = self.root / f'wrong-edge-{operation}.json'
                circuit.write_text(json.dumps(source['template']))
                ir1 = build_ir_v8.build_ir1_direct(source, circuit, mode='prefill')
                with self.assertRaises((RuntimeError, ValueError)):
                    lower1 = build_ir_v8.generate_ir_lower_1(
                        ir1, registry, source, 'prefill')
                    layout = build_ir_v8.generate_memory_layout(
                        lower1, source, registry, mode='prefill', context_len=36)
                    lower2 = build_ir_v8.generate_ir_lower_2(
                        lower1, layout, source, registry, mode='prefill')
                    build_ir_v8.generate_ir_lower_3(lower2, mode='prefill')
