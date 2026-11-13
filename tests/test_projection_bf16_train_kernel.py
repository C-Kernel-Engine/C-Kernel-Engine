"""Independent arithmetic and checked-buffer contract of mixed BF16 projection."""
import ctypes as C
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ProjectionKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise unittest.SkipTest('independent PyTorch unavailable: '+str(exc))
        cls.np=np; cls.torch=torch
        cls.tmp=tempfile.TemporaryDirectory(prefix='bf16-projection-')
        cls.libs=[]
        for optimization in ('-O0','-O3'):
            path=Path(cls.tmp.name)/(optimization+'.so')
            subprocess.run([os.environ.get('CC','cc'),optimization,'-shared','-fPIC','-I'+str(ROOT/'include'),str(ROOT/'src/kernels/projection_bf16_train.c'),'-lm','-o',str(path)],check=True)
            lib=C.CDLL(str(path))
            for phase in ('forward','backward'):
                fn=getattr(lib,'ck_projection_bf16_train_'+phase)
                fn.argtypes=[v for _ in range(6) for v in (C.c_void_p,C.c_size_t)]+[C.c_size_t]*3
                fn.restype=C.c_int
            cls.libs.append(lib)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def call(self,lib,phase,arrays,shape,override=None):
        args=[]
        for a in arrays: args.extend([a.ctypes.data,a.size])
        args.extend(shape)
        for k,v in (override or {}).items(): args[k]=v
        return getattr(lib,'ck_projection_bf16_train_'+phase)(*args)

    def data(self,shape):
        np=self.np; r,k,o=shape; rng=np.random.default_rng(91)
        x=rng.normal(0,.3,r*k).astype('f');w=rng.normal(0,.3,o*k).astype('f')
        return [x,w,np.full(r*o+3,93.,'f'),np.full(r*k+3,0x1234,'uint16'),np.full(o*k+3,0x1234,'uint16'),np.zeros(r*o+3,'f')]

    def test_rectangles_and_fp32_gradient_oracle(self):
        np=self.np;torch=self.torch
        for shape in ((1,1,1),(3,5,7),(4,17,3)):
            r,k,o=shape;results=[]
            for lib in self.libs:
                a=self.data(shape);self.assertEqual(self.call(lib,'forward',a,shape),0)
                x=torch.tensor(a[0].reshape(r,k)).bfloat16().float().requires_grad_()
                w=torch.tensor(a[1].reshape(o,k)).bfloat16().float().requires_grad_()
                y=x@w.T
                # Contract oracle: separate FP32 products and additions in the
                # declared reduction order, independent of BLAS dispatch.
                xn=x.detach().numpy();wn=w.detach().numpy()
                ordered=np.zeros((r,o),dtype='f')
                for j in range(k): ordered += xn[:,j,None]*wn[None,:,j]
                np.testing.assert_array_equal(a[2][:r*o].reshape(r,o),ordered)
                np.testing.assert_allclose(a[2][:r*o].reshape(r,o),y.detach().numpy(),atol=2e-7,rtol=2e-6)
                np.testing.assert_array_equal(a[3][:r*k],x.detach().bfloat16().view(torch.int16).numpy().view('uint16').ravel())
                dy=np.random.default_rng(11).normal(0,.2,r*o).astype('f')
                y.backward(torch.tensor(dy.reshape(r,o)))
                b=[dy,a[3],a[4],np.full(r*k+3,93.,'f'),np.full(o*k+3,93.,'f'),np.zeros((r+o)*k+3,'f')]
                self.assertEqual(self.call(lib,'backward',b,shape),0)
                d=dy.reshape(r,o)
                dx=np.zeros((r,k),dtype='f');dw=np.zeros((o,k),dtype='f')
                for j in range(o): dx += d[:,j,None]*wn[None,j,:]
                for j in range(r): dw += d[j,:,None]*xn[None,j,:]
                np.testing.assert_array_equal(b[3][:r*k].reshape(r,k),dx)
                np.testing.assert_array_equal(b[4][:o*k].reshape(o,k),dw)
                np.testing.assert_allclose(b[3][:r*k].reshape(r,k),x.grad.numpy(),atol=2e-7,rtol=2e-6)
                np.testing.assert_allclose(b[4][:o*k].reshape(o,k),w.grad.numpy(),atol=2e-7,rtol=2e-6)
                for array,n in ((a[2],r*o),(b[3],r*k),(b[4],o*k)):np.testing.assert_array_equal(array[n:],93.)
                for array,n in ((a[3],r*k),(a[4],o*k)):np.testing.assert_array_equal(array[n:],0x1234)
                results.append([z.copy() for z in (a[2],a[3],a[4],b[3],b[4])])
            for left,right in zip(*results):np.testing.assert_array_equal(left,right)

    def test_rne_ties_and_exact_capacities(self):
        np=self.np;torch=self.torch
        # Half-way mantissas with even/odd upper bits, signed zero and subnormals.
        bits=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000,0,0x80000000,0x00008000,0x00018000],dtype='uint32')
        x=bits.view('f').copy(); shape=(1,len(x),1)
        expected=torch.tensor(x).bfloat16().view(torch.int16).numpy().view('uint16')
        for lib in self.libs:
            arrays=[x,np.ones(len(x),'f'),np.zeros(1,'f'),np.zeros(len(x),'uint16'),np.zeros(len(x),'uint16'),np.zeros(1,'f')]
            self.assertEqual(self.call(lib,'forward',arrays,shape),0)
            np.testing.assert_array_equal(arrays[3],expected)
            for index in (1,3,5,7,9,11):
                before=[a.copy() for a in arrays[2:5]]
                self.assertLess(self.call(lib,'forward',arrays,shape,{index:arrays[index//2].size-1}),0)
                for a,b in zip(arrays[2:5],before):np.testing.assert_array_equal(a,b)

    def test_reject_then_recover_and_unchanged_outputs(self):
        np=self.np;shape=(3,5,7)
        for lib in self.libs:
            baseline=self.data(shape);self.assertEqual(self.call(lib,'forward',baseline,shape),0)
            for case in ('nan','round_overflow','sum_overflow','null','alignment','overlap','capacity','zero','size_overflow'):
                with self.subTest(case=case):
                    a=self.data(shape);over={}
                    if case=='nan':a[0][0]=np.nan
                    if case=='round_overflow':a[0][0]=np.finfo('f').max
                    if case=='sum_overflow':a[0][:]=1e30;a[1][:]=1e30
                    if case=='null':over[0]=None
                    if case=='alignment':over[6]=a[3].ctypes.data+1
                    if case=='overlap':over[4]=a[0].ctypes.data+4
                    if case=='capacity':over[7]=14
                    if case=='zero':over[12]=0
                    if case=='size_overflow':over[12]=2**64-1
                    before=[z.copy() for z in a[2:5]]
                    self.assertLess(self.call(lib,'forward',a,shape,over),0)
                    for left,right in zip(a[2:5],before):np.testing.assert_array_equal(left,right)
                    clean=self.data(shape);a[0][:]=clean[0];a[1][:]=clean[1]
                    self.assertEqual(self.call(lib,'forward',a,shape),0)
                    for left,right in zip(a[2:5],baseline[2:5]):np.testing.assert_array_equal(left,right)
            # Backward rejection, including corrupted saved state and partial alias.
            for case in ('nan','saved_nonfinite','overflow','capacity','overlap','null'):
                a=baseline;dy=np.ones(21,'f');b=[dy,a[3].copy(),a[4].copy(),np.full(15,93.,'f'),np.full(35,93.,'f'),np.zeros(50,'f')];over={}
                if case=='nan':b[0][0]=np.nan
                if case=='saved_nonfinite':b[1][0]=0x7f80
                if case=='overflow':b[0][:]=1e30;b[1][:]=0x7180
                if case=='capacity':over[11]=49
                if case=='overlap':over[8]=b[3].ctypes.data+4
                if case=='null':over[2]=None
                self.assertLess(self.call(lib,'backward',b,shape,over),0)
                np.testing.assert_array_equal(b[3],93.);np.testing.assert_array_equal(b[4],93.)
                b[0][:]=1.;b[1][:]=a[3];b[2][:]=a[4]
                self.assertEqual(self.call(lib,'backward',b,shape),0)

if __name__=='__main__':unittest.main()
