"""Checked positional features: native arithmetic, rejection and recovery."""
import ctypes
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
P = ctypes.POINTER(ctypes.c_float)
S = ctypes.c_size_t

class SinusoidalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / 'features.so'
        subprocess.run(['cc', '-O3', '-std=c11', '-Wall', '-Wextra', '-Werror',
                        '-ffp-contract=off', '-shared', '-fPIC', '-I', str(ROOT/'include'),
                        str(ROOT/'src/kernels/sinusoidal_rows_checked.c'), '-lm', '-o', str(path)],
                       check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(path))
        cls.fn = cls.lib.sinusoidal_features_f32_checked
        cls.fn.argtypes = [P,S,P,S,ctypes.c_float,P,S,S,S]
        cls.fn.restype = ctypes.c_int
        cls.tables = cls.lib.sinusoidal_tables_f32_checked
        cls.tables.argtypes = [P,S,P,S,ctypes.c_float,P,S,P,S,S,S]
        cls.tables.restype = ctypes.c_int
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    def call(self, p, f, y, scale=1000., **kw):
        return self.fn(p.ctypes.data_as(P), kw.get('positions', p.size),
                       f.ctypes.data_as(P), kw.get('frequencies', f.size), scale,
                       y.ctypes.data_as(P), kw.get('capacity', y.size),
                       kw.get('rows', p.size), kw.get('columns', f.size))
    def test_nonzero_shapes_and_separate_tables(self):
        for rows, columns in ((1,2), (3,32), (7,128)):
            p = np.linspace(-.7, 1.3, rows, dtype='f4')
            f = np.geomspace(1, .0001, columns).astype('f4')
            phase = ((p[:,None]*np.float32(1000))*f).astype('f8')
            expected = np.concatenate([np.sin(phase), np.cos(phase)], axis=1).astype('f4')
            y = np.empty((rows,2*columns), 'f4')
            self.assertEqual(self.call(p,f,y), 0)
            np.testing.assert_allclose(y, expected, atol=6e-8, rtol=1e-7)
            s = np.empty((rows,columns), 'f4'); c = np.empty_like(s)
            self.assertEqual(self.tables(p.ctypes.data_as(P), p.size, f.ctypes.data_as(P), f.size,
                1000., s.ctypes.data_as(P), s.size, c.ctypes.data_as(P), c.size, rows, columns), 0)
            np.testing.assert_array_equal(y, np.concatenate([s,c],axis=1))
            before=s.copy()
            self.assertNotEqual(self.tables(p.ctypes.data_as(P),p.size,f.ctypes.data_as(P),f.size,
                1.,s.ctypes.data_as(P),s.size,s.ctypes.data_as(P),s.size,rows,columns),0)
            np.testing.assert_array_equal(s,before)
    def test_rejection_without_mutation_and_recovery(self):
        p=np.array([0,.4,.9], 'f4'); f=np.array([1,.01], 'f4'); y=np.full((3,4), -99., 'f4')
        for kw in ({'positions':2}, {'frequencies':1}, {'capacity':11}, {'rows':0},
                   {'columns':0}, {'columns':2**63}, {'scale':float('nan')}):
            self.assertNotEqual(self.call(p,f,y,**kw),0)
            self.assertTrue((y==-99).all())
        for a in (p,f):
            old=a[-1]; a[-1]=np.nan
            self.assertNotEqual(self.call(p,f,y),0); self.assertTrue((y==-99).all()); a[-1]=old
        p[-1]=np.finfo('f4').max
        self.assertNotEqual(self.call(p,f,y),0); self.assertTrue((y==-99).all())
        p[-1]=.9
        self.assertNotEqual(self.call(y.ravel()[:3],f,y),0); self.assertTrue((y==-99).all())
        self.assertEqual(self.call(p,f,y),0); self.assertTrue(np.isfinite(y).all())

if __name__ == '__main__':
    unittest.main()
