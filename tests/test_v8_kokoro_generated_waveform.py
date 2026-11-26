"""Complete generated Kokoro waveform execution and diagnostic parity."""

import ctypes
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import wave

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_oracle_circuit_support import compile_oracle_graph
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests.test_v8_kokoro_generated_prosody_complete import verified_fixture
from tests.test_v8_kokoro_generator_stage0_join import (
    join_weights_and_references)
from tests.test_v8_kokoro_generator_stage0_pool import stage0_pool_weights
from tests import test_v8_kokoro_source_residual_first_pair as fixture_support


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_waveform_circuit as waveform_author
import build_kokoro_waveform_oracle_circuit as direct_author


class KokoroGeneratedWaveformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.refs = join_weights_and_references()
        first, first_meta = fixture_support.load_fixture(
            'kokoro_generator_main_resblock0_pinned')
        tail, tail_meta = fixture_support.load_fixture(
            'kokoro_generator_stage0_pool_pinned')
        ingress, ingress_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_ingress_pinned')
        source, source_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_source_conv_pinned')
        join, join_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_join_pinned')
        pool, pool_meta = fixture_support.load_fixture(
            'kokoro_generator_stage1_pool_pinned')
        cls.pool = pool
        cls.spectral, spectral_meta = fixture_support.load_fixture(
            'kokoro_spectral_head_pinned')
        cls.spectral_meta = spectral_meta
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            if any(item[field] != first_meta[field] for item in (
                    tail_meta, ingress_meta, source_meta, join_meta,
                    pool_meta, spectral_meta, cls.refs.join_meta)):
                raise RuntimeError(f'waveform fixture {field} differs')
        previous = ROOT / 'tests/fixtures/tts/kokoro_generator_stage1_pool_pinned.npz'
        if hashlib.sha256(previous.read_bytes()).hexdigest() != \
                spectral_meta['previous_fixture_sha256']:
            raise RuntimeError('waveform fixture upstream hash differs')
        weights = stage0_pool_weights(cls.refs, first, tail)
        weights['waveform_decoder.generator.ups.1.weight'] = ingress['weight']
        weights['waveform_decoder.generator.ups.1.bias'] = ingress['bias']
        weights['waveform_decoder.generator.noise_convs.1.weight'] = source['weight']
        weights['waveform_decoder.generator.noise_convs.1.bias'] = source['bias']
        weights.update({key: value for data in (join, pool)
                        for key, value in data.items()
                        if key.startswith('waveform_decoder.')})
        weights['waveform_decoder.generator.conv_post.weight'] = \
            cls.spectral['weight']
        weights['waveform_decoder.generator.conv_post.bias'] = \
            cls.spectral['bias']
        fixture = prepare_duration_fixture(root, waveform_author.OUTPUT, weights)
        cls.waveform_source = fixture.source
        cls.encoder, cls.duration = fixture.encoder, fixture.duration
        cls.entries, cls.bump = fixture.entries, fixture.bump
        native = root / 'waveform'
        native.mkdir()
        cls.native_dir = native
        try:
            cls.layout, cls.calls, _, _, cls.fn = compile_native_graph(
                native, fixture.source, waveform_author.OUTPUT)
        except subprocess.CalledProcessError as exc:
            calls = native / 'call.json'
            errors = json.loads(calls.read_text()).get('errors') if calls.exists() else None
            raise RuntimeError(f'{exc.stderr or exc.stdout}\ncall errors: {errors}') from exc
        cls.fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] + [
            ctypes.POINTER(ctypes.c_int32)] * 7
        diagnostic = copy.deepcopy(fixture.source)
        diagnostic['template'] = direct_author.build_circuit()
        diagnostic['config']['model'] = diagnostic['template']['name']
        diagnostic['config']['arch'] = diagnostic['template']['name']
        diagnostic['config']['activation_buffer_dtypes'].update({
            name: 'i32' for name in (
                'head_frame_count', 'head_sample_count',
                'head_frame_extent_value', 'head_sample_extent_value')})
        referenced = {name
            for block in diagnostic['template']['block_types'].values()
            for op in block['header'] + block['body']['ops'] + block['footer']
            for name in op.get('weight_refs', {}).values()}
        diagnostic['entries'] = [
            item for item in diagnostic['entries']
            if item['name'] in referenced]
        direct = root / 'direct'
        direct.mkdir()
        cls.direct_layout, cls.direct_calls, _, _, cls.direct_fn = \
            compile_oracle_graph(direct, diagnostic)
        cls.direct_fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]

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
    def arena(cls, layout=None):
        layout = cls.layout if layout is None else layout
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
        cls.view(layout, arena, 'generator_waveform')[:] = -91.
        return arena

    def conv_variant(self, variant, workers=4):
        if variant not in ('hwfma', 'avx2_fma', 'avx2_parallel'):
            self.fail(f'unsupported explicit Conv1D candidate: {variant}')
        candidate = copy.deepcopy(self.waveform_source)
        names = {
            'audio_conv1d_checked_channel_major_f32':
                f'audio_conv1d_checked_channel_major_f32_{variant}',
            'audio_conv1d_dilated_checked_channel_major_f32':
                f'audio_conv1d_dilated_checked_channel_major_f32_{variant}'}
        selected = 0
        for block in candidate['template']['block_types'].values():
            for op in block['header'] + block['body']['ops'] + block['footer']:
                if op.get('kernel') in names:
                    op['kernel'] = names[op['kernel']]
                    if variant == 'avx2_parallel':
                        op.setdefault('params', {})['conv_worker_pool'] = 'worker_pool'
                        op['params']['conv_worker_count'] = workers
                    selected += 1
        if variant == 'avx2_parallel':
            candidate['template']['native_entry']['params'].append(
                {'c_type': 'void *', 'name': 'worker_pool'})
        self.assertGreater(selected, 0)
        candidate['template']['required_numerical_contracts'][
            'audio_conv1d_checked']['phases']['prefill']['contract_id'] = {
                'hwfma': 'audio_conv1d_checked_x86_hwfma_fp32',
                'avx2_fma': 'audio_conv1d_checked_x86_avx2_fma_fp32',
                'avx2_parallel': 'audio_conv1d_checked_x86_avx2_parallel_fp32',
            }[variant]
        return candidate, selected

    @unittest.skipUnless(os.environ.get('CKE_TTS_RUN_HW_FMA_REPLAY'),
                         'asset-backed full-waveform hardware-FMA replay not requested')
    def test_full_waveform_hardware_fma_provider_replay(self):
        """Compare an explicit FP32 FMA candidate with the full scalar graph."""
        candidate_variant = os.environ.get('CKE_TTS_CONV_PROVIDER', 'hwfma')
        candidate, selected = self.conv_variant(candidate_variant, int(
            os.environ.get('CKE_TTS_CONV_WORKERS', '4')))
        names = {
            'audio_conv1d_checked_channel_major_f32':
                f'audio_conv1d_checked_channel_major_f32_{candidate_variant}',
            'audio_conv1d_dilated_checked_channel_major_f32':
                f'audio_conv1d_dilated_checked_channel_major_f32_{candidate_variant}'}
        candidate_root = Path(self.temp.name) / 'hwfma'
        candidate_root.mkdir()
        extra_exports = (('ck_threadpool_create', 'ck_threadpool_destroy')
                         if candidate_variant == 'avx2_parallel' else ())
        layout, calls, _, loaded, function = compile_native_graph(
            candidate_root, candidate, waveform_author.OUTPUT,
            extra_exports=extra_exports)
        self.assertEqual(calls['errors'], [])
        functions = [item['function'] for item in calls['operations']]
        self.assertEqual(sum(name in functions for name in names.values()), 2)
        self.assertEqual(sum(function_name in names.values()
                             for function_name in functions), selected)
        function.argtypes = self.fn.argtypes + ([ctypes.c_void_p]
            if candidate_variant == 'avx2_parallel' else [])
        worker_pool = None
        if candidate_variant == 'avx2_parallel':
            loaded.ck_threadpool_create.argtypes = [ctypes.c_int]
            loaded.ck_threadpool_create.restype = ctypes.c_void_p
            loaded.ck_threadpool_destroy.argtypes = [ctypes.c_void_p]
            worker_pool = loaded.ck_threadpool_create(int(
                os.environ.get('CKE_TTS_CONV_WORKERS', '4')))
            self.assertTrue(worker_pool)
        baseline_arena = self.arena()
        candidate_arena = self.arena(layout)
        repeats = int(os.environ.get('CKE_TTS_HW_FMA_TIMING_REPEATS', '2'))
        self.assertGreaterEqual(repeats, 2)
        timings = {'scalar': [], 'candidate': []}
        baseline_lengths = candidate_lengths = None
        try:
            for repetition in range(repeats):
                variants = [('scalar', self.fn, baseline_arena),
                            ('candidate', function, candidate_arena)]
                if repetition % 2:
                    variants.reverse()
                for variant, entry, arena in variants:
                    lengths = [ctypes.c_int32(-999) for _ in range(7)]
                    started = time.perf_counter()
                    arguments = [arena, len(arena), *(
                        ctypes.byref(value) for value in lengths)]
                    if variant == 'candidate' and worker_pool is not None:
                        arguments.append(worker_pool)
                    status = entry(*arguments)
                    timings[variant].append(time.perf_counter() - started)
                    self.assertEqual(status, 0, variant)
                    if variant == 'scalar':
                        baseline_lengths = lengths
                    else:
                        candidate_lengths = lengths
            self.assertEqual([value.value for value in baseline_lengths],
                             [value.value for value in candidate_lengths])
            for name in ('text_conv0', 'f0_conv0', 'source_conv0',
                         'generator_stage1_source_conv', 'generator_conv_post',
                         'generator_waveform'):
                baseline = self.view(self.layout, baseline_arena, name)
                optimized = self.view(layout, candidate_arena, name)
                np.testing.assert_array_equal(baseline, optimized, err_msg=name)
        finally:
            if worker_pool is not None:
                loaded.ck_threadpool_destroy(worker_pool)
        print('CKE_CONV_CANDIDATE_FULL_WAVEFORM ' + json.dumps({
            'selected_conv_operations': selected,
            'candidate_provider': candidate_variant,
            'valid_samples': candidate_lengths[3].value,
            'waveform_sha256': hashlib.sha256(self.view(
                layout, candidate_arena, 'generator_waveform')[:61800]
                .tobytes()).hexdigest(),
            'baseline_library_sha256': json.loads((self.native_dir /
                'build-identity.json').read_text())['library_sha256'],
            'candidate_library_sha256': json.loads((candidate_root /
                'build-identity.json').read_text())['library_sha256'],
            'complete_entry_seconds': timings,
            'warm_speedup': float(np.median(timings['scalar'][1:]) /
                                  np.median(timings['candidate'][1:])),
            'timing_scope': 'generated native entry only; prepared arena and '
                'weights, captured Gaussian, no WAV serialization'}))

    @unittest.skipUnless(os.environ.get('CKE_TTS_RUN_PARALLEL_SCALING'),
                         'asset-backed serial AVX2 versus parallel scaling not requested')
    def test_full_waveform_parallel_scaling(self):
        """Compare the selected provider with serial AVX2 in one run."""
        variants = {}
        for name in ('avx2_fma', 'avx2_parallel'):
            source, selected = self.conv_variant(name, 4)
            self.assertEqual(selected, 88)
            root = Path(self.temp.name) / f'scaling_{name}'
            root.mkdir()
            exports = ['audio_conv1d_checked_avx2_fma_available']
            if name == 'avx2_parallel':
                exports += ['ck_threadpool_create', 'ck_threadpool_destroy']
            layout, calls, library, loaded, function = compile_native_graph(
                root, source, waveform_author.OUTPUT, extra_exports=exports)
            self.assertEqual(calls['errors'], [])
            loaded.audio_conv1d_checked_avx2_fma_available.restype = ctypes.c_int
            if not loaded.audio_conv1d_checked_avx2_fma_available():
                self.skipTest('AVX2/FMA unavailable on this host')
            function.argtypes = self.fn.argtypes + ([ctypes.c_void_p]
                if name == 'avx2_parallel' else [])
            identity = json.loads((root / 'build-identity.json').read_text())
            mapped = str(library.resolve()) in Path('/proc/self/maps').read_text()
            self.assertTrue(mapped, f'compiled library not found in runtime maps: {library}')
            variants[name] = (layout, loaded, function, identity)
        parallel_lib = variants['avx2_parallel'][1]
        parallel_lib.ck_threadpool_create.argtypes = [ctypes.c_int]
        parallel_lib.ck_threadpool_create.restype = ctypes.c_void_p
        parallel_lib.ck_threadpool_destroy.argtypes = [ctypes.c_void_p]
        pools = {count: parallel_lib.ck_threadpool_create(count)
                 for count in (1, 2, 4)}
        self.assertTrue(all(pools.values()))
        arenas = {'serial_avx2': self.arena(variants['avx2_fma'][0])}
        arenas.update({f'parallel_{count}': self.arena(variants['avx2_parallel'][0])
                       for count in pools})
        timings = {name: [] for name in arenas}
        outputs = {}
        repeats = int(os.environ.get('CKE_TTS_PARALLEL_TIMING_REPEATS', '5'))
        self.assertGreaterEqual(repeats, 1)
        order = tuple(arenas)
        started_load = os.getloadavg()
        try:
            for iteration in range(repeats + 1):
                sequence = list(order[iteration % len(order):] +
                                order[:iteration % len(order)])
                if iteration % 2:
                    sequence.reverse()
                for name in sequence:
                    variant = 'avx2_fma' if name == 'serial_avx2' else 'avx2_parallel'
                    layout, _, function, _ = variants[variant]
                    arena = arenas[name]
                    lengths = [ctypes.c_int32(-999) for _ in range(7)]
                    arguments = [arena, len(arena), *(
                        ctypes.byref(value) for value in lengths)]
                    if variant == 'avx2_parallel':
                        arguments.append(pools[int(name.rsplit('_', 1)[1])])
                    start = time.perf_counter()
                    status = function(*arguments)
                    elapsed = time.perf_counter() - start
                    self.assertEqual(status, 0, name)
                    if iteration:
                        timings[name].append(elapsed)
                    outputs[name] = ([value.value for value in lengths],
                        {tensor: self.view(layout, arena, tensor).copy()
                         for tensor in ('text_conv0', 'f0_conv0',
                             'source_conv0', 'generator_stage1_source_conv',
                             'generator_conv_post', 'generator_waveform')})
            serial_lengths, serial_tensors = outputs['serial_avx2']
            self.assertEqual(serial_lengths[3], 61800)
            for name, (lengths, tensors) in outputs.items():
                self.assertEqual(lengths, serial_lengths, name)
                for tensor, expected in serial_tensors.items():
                    np.testing.assert_array_equal(tensors[tensor], expected,
                                                  err_msg=f'{name}: {tensor}')
        finally:
            for pool in pools.values():
                if pool:
                    parallel_lib.ck_threadpool_destroy(pool)
        source_revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        report = {
            'status': 'pass', 'source_revision': source_revision,
            'worktree_dirty': dirty,
            'host_cpu': next((line.partition(':')[2].strip()
                for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor()),
            'affinity_cpus': sorted(os.sched_getaffinity(0)),
            'load_average_before': started_load,
            'load_average_after': os.getloadavg(),
            'compiler': variants['avx2_fma'][3]['compiler'],
            'builds': {name: {key: identity[key] for key in (
                'library_sha256', 'generated_source_sha256', 'source_sha256',
                'compile_command', 'embedded_runtime_build_identity')}
                for name, (_, _, _, identity) in variants.items()},
            'loaded_library_mapping_verified': True,
            'selected_conv_operations': 88,
            'warmup_runs_per_variant': 1,
            'measured_runs_per_variant': repeats,
            'seconds': timings,
            'speedup_vs_serial_avx2_at_four_workers': float(
                np.median(timings['serial_avx2']) /
                np.median(timings['parallel_4'])),
            'audio_seconds': 61800 / 24000,
            'waveform_sha256': hashlib.sha256(
                serial_tensors['generator_waveform'][:61800].tobytes()).hexdigest(),
            'reproduction_command': 'CKE_TTS_RUN_PARALLEL_SCALING=1 '
                'CKE_TTS_PARALLEL_TIMING_REPEATS=5 python3 -m unittest '
                'tests.test_v8_kokoro_generated_waveform.'
                'KokoroGeneratedWaveformTest.test_full_waveform_parallel_scaling -q',
            'scope': 'warm generated native entry only; prepared weights and '
                'arena, captured Gaussian, no loading/WAV/playback timing',
        }
        report_path = os.environ.get('CKE_TTS_PARALLEL_TIMING_REPORT')
        if report_path:
            Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
        print('CKE_CONV_PARALLEL_SCALING ' + json.dumps(report))

    @classmethod
    def direct_arena(cls):
        layout = cls.direct_layout
        size = layout['memory']['arena']['total_size']
        raw = (ctypes.c_uint8 * (size + 63))()
        arena = (ctypes.c_uint8 * size).from_buffer(
            raw, (-ctypes.addressof(raw)) & 63)
        for item in layout['memory']['weights']['entries']:
            entry = cls.entries[item['name']]
            start = entry['file_offset']
            arena[item['abs_offset']:item['abs_offset'] + entry['size']] = \
                cls.bump[start:start + entry['size']]
        features = cls.view(layout, arena,
            'generator_stage1_final_activation').reshape(128, 15361)
        features[:] = -91.
        features[:, :12361] = cls.pool['final_activation'][0]
        cls.view(layout, arena, 'head_frame_count', np.int32)[0] = 12361
        cls.view(layout, arena, 'head_sample_count', np.int32)[0] = 61800
        for name in ('generator_conv_post', 'generator_magnitude',
                     'generator_phase', 'generator_waveform'):
            cls.view(layout, arena, name)[:] = -91.
        return arena

    def test_direct_generated_tail_matches_full_model(self):
        self.assertEqual(self.direct_calls['errors'], [])
        arena = self.direct_arena()
        frames = ctypes.c_int32(-999)
        samples = ctypes.c_int32(-999)
        status = self.direct_fn(arena, len(arena),
            ctypes.byref(frames), ctypes.byref(samples))
        if status:
            print('DIRECT_WAVEFORM_DIAGNOSTIC', {
                'status': status,
                'conv_first': float(self.view(self.direct_layout, arena,
                    'generator_conv_post')[0]),
                'magnitude_first': float(self.view(self.direct_layout, arena,
                    'generator_magnitude')[0]),
                'phase_first': float(self.view(self.direct_layout, arena,
                    'generator_phase')[0]),
                'waveform_first': float(self.view(self.direct_layout, arena,
                    'generator_waveform')[0]),
                'frame_input': int(self.view(self.direct_layout, arena,
                    'head_frame_count', np.int32)[0]),
                'sample_input': int(self.view(self.direct_layout, arena,
                    'head_sample_count', np.int32)[0]),
            })
        self.assertEqual(status, 0)
        self.assertEqual((frames.value, samples.value), (12361, 61800))
        layout = self.direct_layout
        cases = (
            ('conv_post', self.view(layout, arena,
                'generator_conv_post').reshape(22, 15361)[:, :12361],
             self.spectral['conv_post'][0], 5e-5),
            ('magnitude', self.view(layout, arena,
                'generator_magnitude').reshape(15361, 11)[:12361],
             self.spectral['magnitude'][0].T, 5e-5),
            ('phase', self.view(layout, arena,
                'generator_phase').reshape(15361, 11)[:12361],
             self.spectral['phase'][0].T, 1e-5),
            ('waveform', self.view(layout, arena,
                'generator_waveform')[:61800],
             self.spectral['waveform'][0, 0], 1e-5),
        )
        for name, actual, reference, limit in cases:
            self.assertTrue(np.isfinite(actual).all(), name)
            error = np.abs(actual - reference)
            point = np.unravel_index(np.argmax(error), error.shape)
            print('CKE_NUMERICAL_CASE ' + json.dumps({
                'case_id': f'kokoro.waveform-tail.{name}.direct-v1',
                'name': f'Kokoro generated waveform tail {name} direct parity',
                'provider': 'generated_kokoro_waveform_tail',
                'dtype': 'fp32', 'direction': 'inference',
                'oracle': 'pinned-PyTorch-2.8-full-model-hook',
                'status': 'pass' if float(error[point]) <= limit else 'fail',
                'max_diff': float(error[point]), 'tolerance': limit,
                'worst_index': list(map(int, point)),
                'actual': float(actual[point]),
                'reference': float(reference[point]),
                'configuration': 'oracle-fed final generator features, 12361 frames',
                'reproduction_command': 'python3 -m unittest '
                    'tests.test_v8_kokoro_generated_waveform.'
                    'KokoroGeneratedWaveformTest.test_direct_generated_tail_matches_full_model'}))
            self.assertLessEqual(float(error[point]), limit, name)
        self.assertTrue(np.all(self.view(layout, arena,
            'generator_conv_post').reshape(22, 15361)[:, 12361:] == -91.))
        for name in ('generator_magnitude', 'generator_phase'):
            self.assertTrue(np.all(self.view(layout, arena,
                name).reshape(15361, 11)[12361:] == -91.))
        self.assertTrue(np.all(self.view(layout, arena,
            'generator_waveform')[61800:] == -91.))

    def test_connected_waveform_is_finite_and_reports_numerical_status(self):
        self.assertEqual(json.loads(waveform_author.OUTPUT.read_text()),
                         waveform_author.build_circuit())
        self.assertEqual(self.calls['errors'], [])
        arena = self.arena()
        lengths = [ctypes.c_int32(-999) for _ in range(7)]
        self.assertEqual(self.fn(arena, len(arena), *(
            ctypes.byref(value) for value in lengths)), 0)
        self.assertEqual(tuple(value.value for value in lengths),
                         (103, 206, 2060, 61800, 12361, 12360, 12361))
        output = self.view(self.layout, arena, 'generator_waveform')
        self.assertTrue(np.isfinite(output[:61800]).all())
        self.assertTrue(np.all(output[61800:] == -91.))
        reference = self.spectral['waveform'][0, 0]
        error = np.abs(output[:61800] - reference)
        point = int(np.argmax(error))
        limit = 2e-4
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.waveform.connected-diagnostic-v1',
            'name': 'Kokoro complete generated waveform diagnostic',
            'provider': 'generated_kokoro_waveform', 'dtype': 'fp32',
            'direction': 'inference',
            'oracle': 'pinned-PyTorch-2.8-full-model-waveform',
            'status': 'fail' if float(error[point]) > limit else 'pass',
            'gate': 'diagnostic', 'blocking': False,
            'reason': 'known connected F0 and source/STFT mismatch',
            'max_diff': float(error[point]), 'tolerance': limit,
            'worst_index': point, 'actual': float(output[point]),
            'reference': float(reference[point]),
            'configuration': '36 phonemes, af_heart, captured Gaussian, 61800 samples',
            'reproduction_command': 'python3 -m unittest '
                'tests.test_v8_kokoro_generated_waveform'}))

    @unittest.skipUnless(os.environ.get('CKE_TTS_RUN_SEEDED_REPLAY'),
                         'asset-backed seeded full-graph replay not requested')
    def test_seeded_native_noise_reuses_one_generated_entry(self):
        """Exercise A→B→A in one process without captured Gaussian input."""
        library = Path(self.temp.name) / 'audio_noise_counter.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra',
            '-Werror', '-pedantic', '-shared', '-fPIC',
            '-I', str(ROOT / 'include'),
            str(ROOT / 'src/kernels/audio_noise_counter.c'),
            '-lm', '-o', str(library)], check=True, capture_output=True)
        noise = ctypes.CDLL(str(library)).audio_noise_counter_gaussian_f32
        noise.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                          ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64]
        noise.restype = ctypes.c_int

        other, other_meta = verified_fixture('decoder_utterance_your')
        for field in ('model_pin', 'code_pin', 'asset_sha256'):
            self.assertEqual(other_meta[field], self.spectral_meta[field])
        self.assertEqual(other_meta['valid_frames'], 98)
        arena = self.arena()
        gaussian = self.view(self.layout, arena, 'source_gaussian')
        waveform = self.view(self.layout, arena, 'generator_waveform')
        ids = self.view(self.layout, arena, 'word_ids', np.int32)
        predictor_style = self.view(self.layout, arena, 'predictor_style')
        decoder_style = self.view(self.layout, arena, 'decoder_style')
        baseline = (ids.copy(), predictor_style.copy(), decoder_style.copy())
        alternate = (other['word_ids'].ravel(),
                     other['predictor_style'].ravel(),
                     other['decoder_style'].ravel())
        outputs = []
        execution_seconds = []
        sample_counts = []
        for request_index, (inputs, seed, expected_count) in enumerate((
                (baseline, 7, 61800), (alternate, 8, 58800),
                (baseline, 7, 61800))):
            if request_index == 1:
                # Reject a malformed request in the same arena before B. The
                # B must still execute, and the repeated A must match exactly.
                # This position-zero ID is rejected before model execution;
                # it does not test recovery after a partial graph failure.
                ids[:] = baseline[0]
                ids[0] = -1
                waveform.fill(-91.)
                rejected_lengths = [ctypes.c_int32(-1) for _ in range(7)]
                self.assertNotEqual(self.fn(arena, len(arena), *(
                    ctypes.byref(value) for value in rejected_lengths)), 0)
                self.assertEqual([value.value for value in rejected_lengths],
                                 [-1] * 7)
                self.assertTrue(np.all(waveform == -91.))
            ids[:] = inputs[0]
            predictor_style[:] = inputs[1]
            decoder_style[:] = inputs[2]
            gaussian.fill(-91.)
            self.assertEqual(noise(gaussian.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)), gaussian.size,
                gaussian.size, seed, 0), 0)
            waveform.fill(-91.)
            lengths = [ctypes.c_int32(-1) for _ in range(7)]
            started = time.perf_counter()
            self.assertEqual(self.fn(arena, len(arena), *(
                ctypes.byref(value) for value in lengths)), 0)
            execution_seconds.append(time.perf_counter() - started)
            count = lengths[3].value
            self.assertEqual(count, expected_count)
            self.assertTrue(np.isfinite(waveform[:count]).all())
            self.assertTrue(np.all(waveform[count:] == -91.))
            outputs.append(waveform[:count].copy())
            sample_counts.append(count)
        np.testing.assert_array_equal(outputs[0], outputs[2])
        self.assertGreater(float(np.max(np.abs(
            outputs[0][:sample_counts[1]] - outputs[1]))), 1e-6)
        peaks = [float(np.max(np.abs(output))) for output in outputs]
        rms_values = [float(np.sqrt(np.mean(
            output.astype(np.float64) ** 2))) for output in outputs]
        for peak, rms in zip(peaks, rms_values):
            self.assertLess(peak, 1.)
            self.assertGreater(rms, 1e-5)
        print('CKE_NUMERICAL_CASE ' + json.dumps({
            'case_id': 'kokoro.waveform.native-noise-replay-v1',
            'name': 'Kokoro generated waveform seeded A-B-A replay',
            'provider': 'audio_noise_counter_gaussian_f32+generated_kokoro_waveform',
            'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'deterministic-same-seed-and-different-seed-control',
            'status': 'pass', 'gate': 'session', 'blocking': True,
            'max_diff': 0, 'tolerance': 0,
            'execution_seconds': execution_seconds,
            'sample_counts': sample_counts,
            'audio_seconds': [count / 24000 for count in sample_counts],
            'peak_abs': peaks, 'rms': rms_values,
            'configuration': 'one process; one arena and generated entry; '
                'two distinct 36-phoneme af_heart utterances; '
                '103,98,103 frames; seeds 7,8,7; native Gaussian; '
                'position-zero invalid ID rejected before B; B executes '
                'with valid bounds; repeated A recovers exactly',
            'numerical_parity': 'NOT_TESTED against pinned PyTorch noise',
            'reproduction_command': 'CKE_TTS_RUN_SEEDED_REPLAY=1 '
                'python3 -m unittest tests.test_v8_kokoro_generated_waveform.'
                'KokoroGeneratedWaveformTest.'
                'test_seeded_native_noise_reuses_one_generated_entry'}))

    def _run_standalone_native_wav_outside_checkout(self):
        retained = os.environ.get('CKE_TTS_RETAIN_DIR')
        root = Path(retained) if retained else Path(self.temp.name) / 'standalone'
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.native_dir / 'generated.so', root / 'generated.so')
        shutil.copy2(self.native_dir / 'build-identity.json',
                     root / 'build-identity.json')
        source = ROOT / 'version/v8/tts/native_kokoro_wav_host.c'
        shutil.copy2(source, root / 'host.c')
        shutil.copy2(ROOT / 'src/kernels/audio_noise_counter.c',
                     root / 'audio_noise_counter.c')
        shutil.copy2(ROOT / 'src/kernels/checked_float_buffer.h',
                     root / 'checked_float_buffer.h')
        shutil.copy2(ROOT / 'include/ckernel_tts.h', root / 'ckernel_tts.h')
        binary = root / 'native-kokoro-wav'
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra',
            '-Werror', '-pedantic', '-I', str(root), str(root / 'host.c'),
            str(root / 'audio_noise_counter.c'), '-L', str(root),
            '-l:generated.so', '-Wl,-rpath,$ORIGIN', '-lm', '-o', str(binary)],
            check=True)
        arena = self.arena()
        output_item = next(item for item in
            self.layout['memory']['activations']['buffers']
            if item['name'] == 'generator_waveform')
        package = root / 'prepared-arena.cktts'
        page = 65536
        segments = 0
        with package.open('wb') as stream:
            stream.write(b'CKTTS001')
            stream.write(struct.pack('<QQQI', len(arena),
                output_item['abs_offset'], output_item['size'] // 4, 0))
            for offset in range(0, len(arena), page):
                chunk = bytes(arena[offset:offset + page])
                if any(chunk):
                    stream.write(struct.pack('<QQ', offset, len(chunk)))
                    stream.write(chunk)
                    segments += 1
            stream.seek(32)
            stream.write(struct.pack('<I', segments))
        wav_path = root / 'kokoro-generated-diagnostic.wav'
        subprocess.run([str(binary), str(package), str(wav_path)],
                       cwd=root, check=True, capture_output=True, text=True)
        with wave.open(str(wav_path), 'rb') as stream:
            self.assertEqual(stream.getnchannels(), 1)
            self.assertEqual(stream.getsampwidth(), 2)
            self.assertEqual(stream.getframerate(), 24000)
            self.assertEqual(stream.getnframes(), 61800)
            pcm = stream.readframes(61800)
        self.assertTrue(any(pcm))
        gaussian_item = next(item for item in
            self.layout['memory']['activations']['buffers']
            if item['name'] == 'source_gaussian')
        native_arena = self.arena()
        self.view(self.layout, native_arena, 'source_gaussian')[:] = 0.
        native_package = root / 'prepared-native-noise.cktts'
        native_segments = 0
        with native_package.open('wb') as stream:
            stream.write(b'CKTTS002')
            stream.write(struct.pack('<QQQQQI', len(native_arena),
                output_item['abs_offset'], output_item['size'] // 4,
                gaussian_item['abs_offset'], gaussian_item['size'] // 4, 0))
            for offset in range(0, len(native_arena), page):
                chunk = bytes(native_arena[offset:offset + page])
                if any(chunk):
                    stream.write(struct.pack('<QQ', offset, len(chunk)))
                    stream.write(chunk)
                    native_segments += 1
            stream.seek(48)
            stream.write(struct.pack('<I', native_segments))
        native_path = root / 'kokoro-native-seed-7.wav'
        subprocess.run([str(binary), str(native_package), str(native_path),
            '7'], cwd=root, check=True, capture_output=True, text=True)
        with wave.open(str(native_path), 'rb') as stream:
            self.assertEqual(stream.getnframes(), 61800)
            self.assertEqual(stream.getframerate(), 24000)
            self.assertTrue(any(stream.readframes(61800)))
        native_wav = native_path.read_bytes()
        self.assertNotEqual(native_wav, wav_path.read_bytes())
        (root / 'reproduction.json').write_text(json.dumps({
            'scope': 'experimental fixed-phoneme diagnostic WAV',
            'model_pin': self.spectral_meta['model_pin'],
            'code_pin': self.spectral_meta['code_pin'],
            'library_sha256': hashlib.sha256(
                (root / 'generated.so').read_bytes()).hexdigest(),
            'package_sha256': hashlib.sha256(package.read_bytes()).hexdigest(),
            'wav_sha256': hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            'sample_rate': 24000, 'channels': 1, 'samples': 61800,
            'segments': segments,
            'command': './native-kokoro-wav prepared-arena.cktts '
                'kokoro-generated-diagnostic.wav',
            'numerical_status': 'FAIL, connected diagnostic',
            'captured_gaussian': True,
            'native_noise': {
                'package_sha256': hashlib.sha256(
                    native_package.read_bytes()).hexdigest(),
                'seed_7_wav_sha256': hashlib.sha256(native_wav).hexdigest(),
                'seed_replay': 'covered by counter-kernel oracle suite',
                'command': './native-kokoro-wav prepared-native-noise.cktts '
                    'kokoro-native-seed-7.wav 7',
                'numerical_status': 'NOT_TESTED against pinned PyTorch noise',
            },
        }, indent=2) + '\n')
        invalid_package = root / 'invalid.cktts'
        with package.open('rb') as stream:
            invalid_package.write_bytes(stream.read(20))
        failed = subprocess.run([str(binary), str(invalid_package),
            str(root / 'invalid.wav')], cwd=root)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((root / 'invalid.wav').exists())
        for invalid in ('-1', '18446744073709551616', 'bad'):
            path = root / 'invalid-native.wav'
            failed = subprocess.run([str(binary), str(native_package),
                str(path), invalid], cwd=root)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(path.exists())

if os.environ.get('CKE_TTS_RUN_STANDALONE'):
    KokoroGeneratedWaveformTest.test_standalone_native_wav_outside_checkout = (
        KokoroGeneratedWaveformTest._run_standalone_native_wav_outside_checkout)


if __name__ == '__main__':
    unittest.main()
