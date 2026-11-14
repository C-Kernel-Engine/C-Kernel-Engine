"""Native generated flow arithmetic, not Ming model or waveform certification."""
import ctypes
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from version.v8.tts.build_ming3_flow_step_circuit import build_circuit, OUTPUT
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/ming3_flow/publisher_fp32.json'


def compile_step(folder, n, graph=None):
    graph = build_circuit(n) if graph is None else graph
    path = folder/'circuit.json'; path.write_text(json.dumps(graph))
    source = {'config': {'model': graph['name'], 'arch': graph['name'],
        'num_layers': 1, 'embed_dim': 2, 'num_heads': 1, 'num_kv_heads': 1,
        'head_dim': 2, 'intermediate_size': 2, 'context_length': n,
        'max_seq_len': n, 'vocab_size': 1}, 'entries': [], 'quant_summary': {}, 'template': graph}
    layout, calls, library, loaded, fn = compile_native_graph(folder, source, path)
    size = layout['memory']['arena']['total_size']
    raw = (ctypes.c_uint8*(size+63))()
    arena = (ctypes.c_uint8*size).from_buffer(raw, (-ctypes.addressof(raw)) & 63)
    buffers = {row['name']: row for row in layout['memory']['activations']['buffers']}
    def view(name):
        count = 1 if name in ('neg_one','cfg','dt','noise_amplitude') else n
        return np.ndarray((count,), np.float32, buffer=arena, offset=buffers[name]['abs_offset'])
    view('neg_one')[:] = -1
    return fn, arena, view, calls, loaded


def reference(x, p, q, noise, cfg, dt, amplitude):
    difference = np.subtract(p, q, dtype=np.float32)
    guidance = np.multiply(difference, cfg, dtype=np.float32)
    velocity = np.add(p, guidance, dtype=np.float32)
    delta = np.multiply(velocity, dt, dtype=np.float32)
    euler = np.add(x, delta, dtype=np.float32)
    stochastic = np.multiply(noise, amplitude, dtype=np.float32)
    return np.add(euler, stochastic, dtype=np.float32)


class MingFlowStepTests(unittest.TestCase):
    def test_authoring_bounds_and_checked_in_circuit(self):
        self.assertEqual(json.loads(OUTPUT.read_text()), build_circuit())
        for value in (0, -1, True, 1.5, 65537):
            with self.subTest(value=value), self.assertRaises(ValueError): build_circuit(value)
        self.assertFalse(build_circuit()['contract']['runtime_invariants']['complete_synthesis'])

    def test_pinned_publisher_sampler_trajectory_and_negative_control(self):
        from version.v8.tts.capture_ming3_flow_reference import SOURCE_SHA256
        from version.v8.tts.build_ming3_flow_step_circuit import SOURCE_REVISION, CHECKPOINT_REVISION
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture['source_sha256'], SOURCE_SHA256)
        self.assertEqual(fixture['source_revision'], SOURCE_REVISION)
        self.assertEqual(fixture['checkpoint_revision'], CHECKPOINT_REVISION)
        with tempfile.TemporaryDirectory() as tmp:
            fn, arena, view, calls, loaded = compile_step(Path(tmp), fixture['elements'])
            self.assertEqual([op['function'] for op in calls['operations']],
                ['ck_scale_vector_f32', 'audio_scaled_sum_strided_f32_checked']*4)
            previous = None
            for row in fixture['steps']:
                if previous is not None: np.testing.assert_array_equal(previous, row['latent'])
                for name in ('latent','prediction','null_prediction','noise','cfg','dt','noise_amplitude'):
                    view(name)[:] = row[name]
                self.assertEqual(fn(arena,len(arena)), 0)
                np.testing.assert_array_equal(view('output'), np.array(row['expected'],np.float32))
                previous = view('output').copy()
            view('cfg')[:] = 0
            self.assertEqual(fn(arena,len(arena)), 0)
            self.assertFalse(np.array_equal(view('output'), np.array(row['expected'],np.float32)))

    def test_wrong_composition_edge_fails_reference_gate(self):
        fixture = json.loads(FIXTURE.read_text()); row = fixture['steps'][0]
        graph = build_circuit(fixture['elements'])
        ops = graph['block_types']['flow_step']['body']['ops']
        ops[3]['graph_slots']['inputs']['left'] = 'external:null_prediction'
        with tempfile.TemporaryDirectory() as tmp:
            fn, arena, view, calls, loaded = compile_step(Path(tmp), fixture['elements'], graph)
            for name in ('latent','prediction','null_prediction','noise','cfg','dt','noise_amplitude'):
                view(name)[:] = row[name]
            self.assertEqual(fn(arena,len(arena)), 0)
            self.assertFalse(np.array_equal(view('output'), np.array(row['expected'],np.float32)))

    def test_shapes_rejection_recovery_and_independent_arenas(self):
        rng = np.random.default_rng(751)
        for n in (1, 63, 64, 65, 256):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as tmp:
                fn, arena, view, calls, loaded = compile_step(Path(tmp),n)
                values = rng.normal(size=(4,n)).astype('f4')
                for name, value in zip(('latent','prediction','null_prediction','noise'),values): view(name)[:]=value
                view('cfg')[:]=2;view('dt')[:]=.0625;view('noise_amplitude')[:]=.05
                expected=reference(*values,np.float32(2),np.float32(.0625),np.float32(.05))
                self.assertEqual(fn(arena,len(arena)),0)
                np.testing.assert_array_equal(view('output'),expected)
                # A different arena and deliberately late input failure cannot alter A.
                other=(ctypes.c_uint8*(len(arena)+63))()
                b=(ctypes.c_uint8*len(arena)).from_buffer(other,(-ctypes.addressof(other)) & 63)
                ctypes.memmove(b,arena,len(arena))
                offset = view('noise').ctypes.data - ctypes.addressof(arena)
                np.ndarray((n,),np.float32,buffer=b,offset=offset)[-1]=np.inf
                self.assertNotEqual(fn(b,len(b)),0)
                np.testing.assert_array_equal(view('output'),expected)
                view('output')[:]=-777
                before=bytes(arena)
                self.assertNotEqual(fn(arena,1),0)
                self.assertEqual(bytes(arena),before)
                view('noise')[-1]=np.nan
                self.assertNotEqual(fn(arena,len(arena)),0)
                np.testing.assert_array_equal(view('output'),np.full(n,-777,np.float32))
                view('noise')[:]=values[3]
                self.assertEqual(fn(arena,len(arena)),0)
                np.testing.assert_array_equal(view('output'),expected)
                view('noise_amplitude')[:]=0
                self.assertEqual(fn(arena,len(arena)),0)
                np.testing.assert_array_equal(view('output'),reference(*values,np.float32(2),np.float32(.0625),np.float32(0)))

if __name__ == '__main__': unittest.main()
