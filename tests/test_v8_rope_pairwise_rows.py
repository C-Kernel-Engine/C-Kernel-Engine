"""Explicit-table pairwise rotation and optional BF16 intermediate stores."""
import ctypes,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ctypes.POINTER(ctypes.c_float);S=ctypes.c_size_t

def bf16(x):
    a=np.asarray(x,dtype='f4');u=a.view('u4')
    return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view('f4')

class PairwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();path=Path(cls.tmp.name)/'rotation.so'
        subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-ffp-contract=off','-shared','-fPIC',
            '-I',str(ROOT/'include'),str(ROOT/'src/kernels/rope_pairwise_rows_checked.c'),'-lm','-o',str(path)],check=True,capture_output=True)
        cls.lib=ctypes.CDLL(str(path));cls.functions=[]
        for name in ('rope_pairwise_rows_f32_checked','rope_pairwise_rows_bf16_storage_checked'):
            fn=getattr(cls.lib,name);fn.argtypes=[P,S,P,P,S,P,S,S,S,S];fn.restype=ctypes.c_int;cls.functions.append(fn)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def call(self,fn,x,c,s,y,**kw):
        return fn(x.ctypes.data_as(P),kw.get('input_elements',x.size),c.ctypes.data_as(P),s.ctypes.data_as(P),
            kw.get('table_elements',c.size),y.ctypes.data_as(P),kw.get('output_elements',y.size),
            kw.get('tokens',x.shape[0]),x.shape[1],kw.get('head_dim',x.shape[2]))
    def test_reference_modes_shapes(self):
        rng=np.random.default_rng(731)
        for shape in ((1,1,2),(3,2,8),(7,16,64)):
            raw=rng.normal(size=shape).astype('f4');phase=rng.normal(size=(shape[0],shape[2]//2)).astype('f4')
            for mode,fn in enumerate(self.functions):
                cast=bf16 if mode else lambda x:x
                x=cast(raw);c=cast(np.cos(phase));s=cast(np.sin(phase));expected=np.empty_like(x)
                expected[:,:,::2]=cast(cast(x[:,:,::2]*c[:,None])-cast(x[:,:,1::2]*s[:,None]))
                expected[:,:,1::2]=cast(cast(x[:,:,1::2]*c[:,None])+cast(x[:,:,::2]*s[:,None]))
                y=np.empty_like(x);self.assertEqual(self.call(fn,x,c,s,y),0);np.testing.assert_array_equal(y,expected)
    def test_rejection_before_write_and_recovery(self):
        x=np.ones((3,2,8),'f4');c=np.ones((3,4),'f4');s=np.zeros_like(c);y=np.full_like(x,-999.)
        for fn in self.functions:
            for kw in ({'input_elements':x.size-1},{'table_elements':c.size-1},{'output_elements':y.size-1},
                       {'tokens':0},{'head_dim':7},{'tokens':2**63}):
                self.assertNotEqual(self.call(fn,x,c,s,y,**kw),0);self.assertTrue((y==-999).all())
            for operand in (x,c,s):
                old=operand.flat[-1];operand.flat[-1]=np.nan
                self.assertNotEqual(self.call(fn,x,c,s,y),0);self.assertTrue((y==-999).all());operand.flat[-1]=old
            before=x.copy();self.assertNotEqual(self.call(fn,x,c,s,x),0);np.testing.assert_array_equal(x,before)
            self.assertEqual(self.call(fn,x,c,s,y),0);np.testing.assert_array_equal(y,x);y.fill(-999.)
        x[-1,-1,-1]=.123456
        self.assertNotEqual(self.call(self.functions[1],x,c,s,y),0);self.assertTrue((y==-999).all())

if __name__=='__main__':unittest.main()
