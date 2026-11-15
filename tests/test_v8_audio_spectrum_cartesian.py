"""Independent clamped log-spectrum conversion and checked failure boundaries."""
import ctypes, subprocess, tempfile, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ctypes.POINTER(ctypes.c_float);S=ctypes.c_size_t
class SpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();p=Path(cls.tmp.name)/'spectrum.so'
        subprocess.run(['cc','-std=c11','-O3','-Wall','-Wextra','-Werror','-ffp-contract=off','-shared','-fPIC',
            str(ROOT/'src/kernels/audio_spectrum_cartesian_checked.c'),'-lm','-o',str(p)],check=True,capture_output=True)
        cls.lib=ctypes.CDLL(str(p));cls.fn=cls.lib.audio_spectrum_cartesian_f32_checked
        cls.fn.argtypes=[P,S,P,S,P,S,S,S,ctypes.c_float];cls.fn.restype=ctypes.c_int
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def call(self,x,r,i,**kw):
        return self.fn(x.ctypes.data_as(P),kw.get('input_elements',x.size),r.ctypes.data_as(P),kw.get('real_elements',r.size),
            i.ctypes.data_as(P),kw.get('imag_elements',i.size),kw.get('frames',r.shape[0]),kw.get('bins',r.shape[1]),kw.get('limit',100.))
    def test_reference_clamping_and_shapes(self):
        rng=np.random.default_rng(731)
        for frames,bins in ((1,2),(3,5),(7,1765)):
            x=rng.normal(size=(frames,2*bins)).astype('f4');x[0,0]=1000;x[0,1]=-1000
            r=np.empty((frames,bins),'f4');i=np.empty_like(r)
            self.assertEqual(self.call(x,r,i),0)
            magnitude=np.exp(np.minimum(x[:,:bins].astype('f8'),np.log(100)))
            np.testing.assert_allclose(r,magnitude*np.cos(x[:,bins:].astype('f8')),atol=1e-6,rtol=2e-7)
            np.testing.assert_allclose(i,magnitude*np.sin(x[:,bins:].astype('f8')),atol=1e-6,rtol=2e-7)
    def test_rejection_no_partial_write_and_recovery(self):
        x=np.ones((3,10),'f4');r=np.full((3,5),-99.,'f4');i=r.copy()
        for kw in ({'input_elements':29},{'real_elements':14},{'imag_elements':14},
                   {'frames':0},{'bins':0},{'frames':2**63},{'limit':0},{'limit':float('inf')}):
            self.assertNotEqual(self.call(x,r,i,**kw),0);self.assertTrue((r==-99).all() and (i==-99).all())
        for bad in (np.nan,np.inf,-np.inf):
            x[-1,-1]=bad;self.assertNotEqual(self.call(x,r,i),0)
            self.assertTrue((r==-99).all() and (i==-99).all())
        x[-1,-1]=1
        self.assertNotEqual(self.call(x,r,r),0);self.assertTrue((r==-99).all())
        old=x.copy();self.assertNotEqual(self.call(x,x[:,:5],i),0);np.testing.assert_array_equal(x,old)
        self.assertEqual(self.call(x,r,i),0);self.assertTrue(np.isfinite(r).all() and np.isfinite(i).all())
if __name__=='__main__':unittest.main()
