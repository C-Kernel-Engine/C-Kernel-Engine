"""Independent operation captures and rejection checks for native math adapters."""
import ctypes
import hashlib
import json
import platform
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/tts/kokoro_albert_norm_activation_pinned.npz'
PTR=ctypes.POINTER(ctypes.c_float)
SZ=ctypes.c_size_t

def ptr(a): return a.ctypes.data_as(PTR)

class NormalizationCheckedOracleTest(unittest.TestCase):
    @staticmethod
    def build_library(library, *flags):
        sources = ['src/kernels/layernorm_rows_checked.c', 'src/kernels/gelu_rows_tanh_checked.c',
                   'src/kernels/layernorm_kernels.c', 'src/kernels/gelu_kernels.c',
                   'src/ck_threadpool.c', 'src/ckernel_strict.c']
        subprocess.run(['cc', '-std=c11', '-O2', '-ffp-contract=off', *flags,
                        '-shared', '-fPIC', '-I', str(ROOT/'include'),
                        *[str(ROOT/s) for s in sources], '-lm', '-lpthread', '-ldl',
                        '-o', str(library)], check=True)
        return ctypes.CDLL(str(library))

    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.lib=cls.build_library(Path(cls.temp.name)/'checked.so')
        cls.norm=cls.lib.layernorm_rows_checked_f32
        cls.norm.argtypes=[PTR,SZ,SZ,PTR,SZ,PTR,SZ,PTR,SZ,SZ,PTR,SZ,SZ,SZ,ctypes.c_float]
        cls.norm.restype=ctypes.c_int
        cls.gelu=cls.lib.gelu_rows_tanh_checked_f32
        cls.gelu.argtypes=[PTR,SZ,SZ,PTR,SZ,SZ,PTR,SZ,SZ,SZ];cls.gelu.restype=ctypes.c_int
        cls.query=cls.lib.layernorm_rows_checked_workspace
        cls.query.argtypes=[SZ,SZ,ctypes.POINTER(SZ)];cls.query.restype=ctypes.c_int
        cls.meta=json.loads(FIXTURE.with_suffix('.json').read_text())
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest()!=cls.meta['fixture_sha256']:
            raise RuntimeError('pinned independent operation fixture hash mismatch')
        cls.fixture=dict(np.load(FIXTURE))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def evidence(self, provider, stage, actual, expected, tolerance, backend):
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        error = np.abs(actual - expected) if finite else None
        maximum = float(error.max()) if finite else None
        record = {
            'case_id': provider + '.' + stage + '.' + backend + '.' + str(list(actual.shape)),
            'name': stage, 'configuration': str(list(actual.shape)),
            'provider': provider, 'dtype': 'fp32', 'direction': 'inference',
            'oracle': 'pytorch', 'backend_version': backend, 'shape': list(actual.shape),
            'execution_settings': {'native': 'serial', 'native_build_flags': '-std=c11 -O2 -ffp-contract=off', 'dispatch_identity': 'NOT_VERIFIED'},
            'consumers': ['kokoro', 'generic_encoder'], 'evidence_kind': 'numerical',
            'max_diff': maximum, 'tolerance': tolerance,
            'worst_sample': list(map(int,np.unravel_index(error.argmax(), error.shape))) if finite else None,
            'status': 'pass' if finite and maximum <= tolerance else 'fail',
            'reproduction_command': 'python3 -m unittest ' + self.id(),
        }
        print('CKE_NUMERICAL_CASE ' + json.dumps(record, sort_keys=True))
        self.assertEqual(record['status'], 'pass', record)

    def norm_args(self,x,g,b,y,scratch):
        return [ptr(x),x.size,x.shape[1],ptr(g),g.size,ptr(b),b.size,
                ptr(y),y.size,y.shape[1],ptr(scratch),scratch.size,x.shape[0],g.size,1e-12]

    def test_pinned_layernorm_and_repeated_padded_rows(self):
        # The established FP32 LayerNorm family uses a 1e-5 absolute oracle bound.
        # This adapter preserves its reduction, not the AdaLN fixture-reproduction contract.
        for name,base in [('attention_norm','attention__LayerNorm'),('layer_output','full_layer_layer_norm')]:
            with self.subTest(stage=name):
                x=self.fixture[name+'_input'];g=self.fixture['weight__'+base+'__weight'];b=self.fixture['weight__'+base+'__bias']
                padded=np.full((36,773),np.nan,'f');padded[:,:768]=x
                y=np.full((36,775),-777.,'f');scratch=np.empty(36*770,'f')
                args=self.norm_args(padded,g,b,y,scratch)
                for _ in range(2): self.assertEqual(self.norm(*args),0)
                self.assertTrue(np.isfinite(y).all());self.assertTrue((y[:,768:]==-777.).all())
                error=np.abs(y[:,:768]-self.fixture[name+'_expected'])
                self.evidence('layernorm_rows_checked_f32', name, y[:,:768],
                              self.fixture[name+'_expected'], 1e-5, self.meta['dependencies']['torch'])

    @unittest.skipUnless(platform.machine() == 'x86_64', 'compares x86 compiler targets')
    def test_layernorm_variance_order_is_identical_across_compiler_targets(self):
        source = self.fixture['attention_norm_input']
        gamma = self.fixture['weight__attention__LayerNorm__weight']
        beta = self.fixture['weight__attention__LayerNorm__bias']
        scratch = np.empty(source.shape[0] * (source.shape[1] + 2), dtype=np.float32)
        outputs = []
        for name, flag in [('baseline', '-march=x86-64'), ('native', '-march=native')]:
            library = self.build_library(Path(self.temp.name) / f'{name}.so', flag)
            normalize = library.layernorm_rows_checked_f32
            normalize.argtypes = self.norm.argtypes
            normalize.restype = self.norm.restype
            output = np.empty_like(source)
            self.assertEqual(normalize(*self.norm_args(source, gamma, beta, output, scratch)), 0)
            outputs.append(output)
        np.testing.assert_array_equal(outputs[0].view(np.uint32), outputs[1].view(np.uint32))

    def test_pinned_tanh_gelu(self):
        x=self.fixture['ffn_linear_expected'];y=np.full((36,2051),-777.,'f');scratch=np.empty(x.size,'f')
        args=[ptr(x),x.size,2048,ptr(y),y.size,2051,ptr(scratch),scratch.size,36,2048]
        self.assertEqual(self.gelu(*args),0);self.assertTrue(np.isfinite(y).all())
        self.assertTrue((y[:,2048:]==-777.).all())
        error=np.abs(y[:,:2048]-self.fixture['ffn_projection_input'])
        self.evidence('gelu_rows_tanh_checked_f32', 'pinned-tanh', y[:,:2048],
                      self.fixture['ffn_projection_input'], 1e-6, self.meta['dependencies']['torch'])

    def test_query_rejects_hostile_geometry_without_writing(self):
        for rows,channels in [(0,7),(1,0),(1,2**31-1),(2**63,8),((1<<64)-1,3)]:
            with self.subTest(rows=rows,channels=channels):
                result=SZ(123)
                self.assertNotEqual(self.query(rows,channels,ctypes.byref(result)),0)
                self.assertEqual(result.value,123)

    def test_bounds_nonfinite_and_alias_rejections_preserve_output(self):
        x=np.ones((2,7),'f');g=np.ones(7,'f');b=np.zeros(7,'f');y=np.full((2,9),-777.,'f');scratch=np.empty(18,'f')
        original=self.norm_args(x,g,b,y,scratch)
        for index,value in [(1,13),(4,6),(6,6),(8,15),(11,17),(2,6),(9,6),(14,0.),(14,float('nan'))]:
            with self.subTest(argument=index):
                args=list(original);args[index]=value;before=y.copy()
                self.assertNotEqual(self.norm(*args),0);np.testing.assert_array_equal(y,before)
        for value in [np.nan,np.inf,-np.inf]:
            x[-1,-1]=value;before=y.copy()
            self.assertNotEqual(self.norm(*original),0);np.testing.assert_array_equal(y,before)
        x.fill(1.);args=list(original);args[10]=ptr(x);args[11]=18
        self.assertNotEqual(self.norm(*args),0);np.testing.assert_array_equal(x,np.ones_like(x))
        args=[ptr(x),x.size,7,ptr(y),y.size,9,ptr(scratch),14,2,7]
        for index,value in [(1,13),(4,15),(7,13),(2,6),(5,6)]:
            altered=list(args);altered[index]=value;before=y.copy()
            self.assertNotEqual(self.gelu(*altered),0);np.testing.assert_array_equal(y,before)

    def test_exact_legacy_arithmetic_and_thread_setting_independence(self):
        legacy_norm=self.lib.layernorm_naive_serial_matched_precision
        legacy_norm.argtypes=[PTR,PTR,PTR,PTR,PTR,PTR,ctypes.c_int,ctypes.c_int,ctypes.c_float]
        legacy_gelu=self.lib.gelu_exact_inplace
        legacy_gelu.argtypes=[PTR,SZ]
        thread_setting=self.lib.ck_set_num_threads
        thread_setting.argtypes=[ctypes.c_int]
        rng=np.random.default_rng(120)
        for channels in [1,3,7,17,768]:
            x=rng.normal(size=(3,channels)).astype('f');g=np.ones(channels,'f');b=np.zeros(channels,'f')
            expected=np.empty_like(x);mean=np.empty(1,'f');rstd=np.empty(1,'f')
            for row in range(3):legacy_norm(ptr(x[row]),ptr(g),ptr(b),ptr(expected[row]),ptr(mean),ptr(rstd),1,channels,1e-12)
            try:
                for threads in [1,4]:
                    thread_setting(threads)
                    y=np.empty_like(x);scratch=np.empty(3*(channels+2),'f')
                    self.assertEqual(self.norm(*self.norm_args(x,g,b,y,scratch)),0)
                    np.testing.assert_array_equal(y,expected)
                    ge=x.copy()
                    for row in range(3):legacy_gelu(ptr(ge[row]),channels)
                    self.assertEqual(self.gelu(ptr(x),x.size,channels,ptr(y),y.size,channels,ptr(scratch),scratch.size,3,channels),0)
                    np.testing.assert_array_equal(y,ge)
            finally:thread_setting(1)

    def test_gelu_hostile_nonfinite_alignment_and_alias(self):
        x=np.ones((2,7),'f');y=np.full((2,9),-777.,'f');scratch=np.empty(14,'f')
        args=[ptr(x),x.size,7,ptr(y),y.size,9,ptr(scratch),14,2,7]
        for index,value in [(8,0),(9,0),(8,2**63),(9,(1<<64)-1),
                            (0,ctypes.cast(x.ctypes.data+1,PTR)),(3,ptr(x)),(6,ptr(y))]:
            altered=list(args);altered[index]=value;before=y.copy();before_x=x.copy()
            self.assertNotEqual(self.gelu(*altered),0)
            np.testing.assert_array_equal(y,before);np.testing.assert_array_equal(x,before_x)
        for value in [np.nan,np.inf,-np.inf]:
            x[-1,-1]=value;before=y.copy()
            self.assertNotEqual(self.gelu(*args),0);np.testing.assert_array_equal(y,before)
        x.fill(0.)
        self.assertEqual(self.gelu(*args),0);np.testing.assert_array_equal(y[:,:7],np.zeros_like(x))
        # Constant input has zero centered variance and must return beta.
        g=np.ones(7,'f');b=np.arange(7,dtype='f');ns=np.empty(18,'f')
        self.assertEqual(self.norm(*self.norm_args(x,g,b,y,ns)),0)
        np.testing.assert_array_equal(y[:,:7],np.tile(b,(2,1)))

    def test_large_finite_values_and_nonfinite_result_preservation(self):
        maximum=np.finfo(np.float32).max
        x=np.array([[maximum,-maximum,0.]],dtype='f');y=np.full_like(x,-777.);scratch=np.empty(3,'f')
        self.assertEqual(self.gelu(ptr(x),3,3,ptr(y),3,3,ptr(scratch),3,1,3),0)
        self.assertTrue(np.isfinite(y).all())
        np.testing.assert_array_equal(y,np.array([[maximum,0.,0.]],dtype='f'))
        x=np.arange(14,dtype='f').reshape(2,7);g=np.full(7,maximum,'f');b=np.zeros(7,'f')
        y=np.full((2,9),-777.,'f');scratch=np.empty(18,'f');before=y.copy()
        self.assertEqual(self.norm(*self.norm_args(x,g,b,y,scratch)),-4)
        np.testing.assert_array_equal(y,before)

    def test_live_pytorch_small_dimensions_and_tails(self):
        try: import torch
        except ImportError:
            for provider in ('layernorm_rows_checked_f32', 'gelu_rows_tanh_checked_f32'):
                print('CKE_NUMERICAL_CASE ' + json.dumps({
                    'case_id': provider + '.native-vs-live.unavailable', 'name': 'native-vs-live',
                    'configuration': 'PyTorch unavailable', 'provider': provider,
                    'dtype': 'fp32', 'direction': 'inference', 'oracle': 'pytorch',
                    'evidence_kind': 'numerical', 'status': 'not_tested',
                    'max_diff': None, 'tolerance': None,
                    'reason': 'live PyTorch unavailable',
                    'reproduction_command': 'python3 -m unittest ' + self.id(),
                }))
            self.skipTest('live PyTorch unavailable; NOT_TESTED')
        rng=np.random.default_rng(83)
        for channels in [1,3,7,17,768]:
            with self.subTest(channels=channels):
                x=rng.normal(size=(3,channels)).astype('f');g=rng.normal(size=channels).astype('f');b=rng.normal(size=channels).astype('f')
                y=np.empty_like(x);scratch=np.empty(3*(channels+2),'f')
                self.assertEqual(self.norm(*self.norm_args(x,g,b,y,scratch)),0)
                expected=torch.nn.functional.layer_norm(torch.from_numpy(x),(channels,),torch.from_numpy(g),torch.from_numpy(b),1e-12).numpy()
                self.evidence('layernorm_rows_checked_f32', 'native-vs-live', y, expected, 1e-5, torch.__version__)
                args=[ptr(x),x.size,channels,ptr(y),y.size,channels,ptr(scratch),scratch.size,3,channels]
                self.assertEqual(self.gelu(*args),0)
                expected=(.5*torch.from_numpy(x)*(1+torch.tanh((2/np.pi)**.5*(torch.from_numpy(x)+.044715*torch.from_numpy(x)**3)))).numpy()
                self.evidence('gelu_rows_tanh_checked_f32', 'native-vs-live', y, expected, 1e-6, torch.__version__)

if __name__=='__main__': unittest.main()
