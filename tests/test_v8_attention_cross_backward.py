"""Independent ordered and autograd oracles for checked noncausal FP32 VJP."""
import ctypes as C
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FP = C.POINTER(C.c_float)

def ptr(a):
    return a.ctypes.data_as(FP)

def ordered(q, k, v, dy):
    """Scalar oracle: preserve declared FP32 stores and ascending reductions."""
    Q, H, D = q.shape
    S = len(k)
    dq, dk, dv = [np.zeros_like(x) for x in (q, k, v)]
    def serial_sum(values):
        result = 0.0
        for value in values:
            result += value
        return result
    dot = lambda a, b: serial_sum(float(x)*float(y) for x, y in zip(a, b))
    scale = 1.0 / math.sqrt(D)
    for t in range(Q):
        for h in range(H):
            scores = [np.float32(dot(q[t,h], k[s,h])*scale) for s in range(S)]
            exps = [np.float32(math.exp(float(x)-float(max(scores)))) for x in scores]
            p = [float(x)/serial_sum(map(float, exps)) for x in exps]
            dp = [dot(dy[t,h], v[s,h]) for s in range(S)]
            center = serial_sum(x*y for x,y in zip(p,dp))
            ds = [np.float32(x*(y-center)*scale) for x,y in zip(p,dp)]
            for d in range(D):
                dq[t,h,d] = serial_sum(float(ds[s])*float(k[s,h,d]) for s in range(S))
                for s in range(S):
                    dk[s,h,d] = float(dk[s,h,d])+float(ds[s])*float(q[t,h,d])
                    dv[s,h,d] = float(dv[s,h,d])+p[s]*float(dy[t,h,d])
    return dq, dk, dv

class AttentionBackward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.libs = []
        for opt in ('O0', 'O3'):
            path = Path(cls.tmp.name)/('attention_'+opt+'.so')
            subprocess.run(['cc', '-std=c11', '-'+opt, '-Wall', '-Wextra', '-Werror',
                '-ffp-contract=off', '-shared', '-fPIC', '-I'+str(ROOT/'include'),
                str(ROOT/'src/kernels/attention_cross_token_major_checked.c'), '-lm', '-o', str(path)], check=True)
            lib = C.CDLL(str(path))
            lib.ck_attention_cross_backward_f32.argtypes = [x for _ in range(8) for x in (FP,C.c_size_t)]+[C.c_size_t]*4
            lib.ck_attention_cross_token_major_checked_f32.argtypes = [x for _ in range(5) for x in (FP,C.c_size_t)]+[C.c_size_t]*4
            cls.libs.append(lib)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def operands(self, shape):
        Q,S,H,D = shape
        rng = np.random.default_rng(923)
        return [rng.normal(0,.4,size).astype('f') for size in ((Q,H,D),(S,H,D),(S,H,D),(Q,H,D))]

    def args(self, arrays):
        q,k,v,dy = arrays
        Q,H,D = q.shape; S = len(k)
        out = [np.full(x.size+3, 91., 'f') for x in (q,k,v)]
        scratch = np.empty(q.size+2*k.size+2*S, 'f')
        all_arrays = arrays+out+[scratch]
        args = [y for a in all_arrays for y in (ptr(a),a.size)]+[Q,S,H,D]
        return args, out, all_arrays

    def test_ordered_oracle_and_optimization_levels(self):
        for shape in ((1,1,1,1),(2,5,2,3),(5,2,3,7),(2,7,2,64),(3,4,1,128)):
            arrays = self.operands(shape)
            expected = ordered(*arrays)
            originals = [a.copy() for a in arrays]
            for lib in self.libs:
                args,out,_ = self.args(arrays)
                self.assertEqual(lib.ck_attention_cross_backward_f32(*args),0)
                for actual, ref in zip(out,expected):
                    np.testing.assert_allclose(actual[:-3].reshape(ref.shape),ref,atol=2e-8,rtol=2e-7)
                    np.testing.assert_array_equal(actual[-3:],91)
            for a,b in zip(arrays,originals): np.testing.assert_array_equal(a,b)

    def test_pytorch_autograd_and_finite_differences(self):
        try: import torch
        except ImportError as exc: self.skipTest('PyTorch oracle unavailable: '+str(exc))
        torch.set_num_threads(1)
        for shape in ((1,1,1,1),(2,5,2,3),(5,2,3,7),(2,7,2,64)):
            arrays = self.operands(shape)
            tensors = [torch.tensor(a,requires_grad=True) for a in arrays[:3]]
            q,k,v = [t.transpose(0,1) for t in tensors]
            y = (torch.softmax(q @ k.transpose(-1,-2)/math.sqrt(shape[-1]),-1) @ v).transpose(0,1)
            y.backward(torch.tensor(arrays[3]))
            args,out,_ = self.args(arrays)
            self.assertEqual(self.libs[1].ck_attention_cross_backward_f32(*args),0)
            for actual,t in zip(out,tensors):
                np.testing.assert_allclose(actual[:-3].reshape(t.shape),t.grad.numpy(),atol=2e-6,rtol=2e-5)
            # Central differences of native forward, independent of backward code.
            Q,S,H,D = shape
            def loss():
                dst=np.empty_like(arrays[0]); scratch=np.empty(S+dst.size,'f')
                args=[z for a in arrays[:3]+[dst,scratch] for z in (ptr(a),a.size)]+[Q,S,H,D]
                self.assertEqual(self.libs[1].ck_attention_cross_token_major_checked_f32(*args),0)
                return float(np.sum(dst.astype('d')*arrays[3]))
            for a,gradient in zip(arrays[:3],out):
                old=a.flat[0]; delta=.002
                a.flat[0]=old+delta; plus=loss()
                a.flat[0]=old-delta; minus=loss()
                a.flat[0]=old
                self.assertAlmostEqual((plus-minus)/(2*delta),float(gradient[0]),delta=4e-5)

    def test_rejection_atomicity_and_recovery(self):
        arrays=self.operands((2,3,2,3))
        # Every pointer/capacity; exact workspace; every dimensional bound.
        cases=[]
        for i in range(8):
            cases.extend([('null',i),('short',i),('unaligned',i)])
        for i in range(4): cases.extend([('zero',i),('overflow',i),('nan',i)])
        for i in range(4,8):
            for j in range(i): cases.append(('overlap',(i,j)))
        for kind,index in cases:
            with self.subTest(kind=kind,index=index):
                inputs=[a.copy() for a in arrays]
                args,out,keep=self.args(inputs)
                if kind=='null': args[index*2]=None
                elif kind=='short': args[index*2+1]=(inputs[index].size if index<4 else (out[index-4].size-3 if index<7 else keep[-1].size))-1
                elif kind=='unaligned': args[index*2]=C.cast(keep[index].ctypes.data+1,FP)
                elif kind=='zero': args[16+index]=0
                elif kind=='overflow': args[16+index]=2**64-1
                elif kind=='nan': inputs[index].flat[0]=np.nan
                else:
                    i,j=index
                    args[i*2]=C.cast(keep[j].ctypes.data+4,FP)
                self.assertLess(self.libs[1].ck_attention_cross_backward_f32(*args),0)
                for a in out: np.testing.assert_array_equal(a,91)
                args,out,_=self.args(arrays)
                self.assertEqual(self.libs[1].ck_attention_cross_backward_f32(*args),0)
                for a,b in zip(out,ordered(*arrays)): np.testing.assert_allclose(a[:-3],b.ravel(),atol=2e-8,rtol=2e-7)

    def test_late_overflow_leaves_all_gradients_unchanged(self):
        arrays=[np.zeros((2,1,1),'f'),np.zeros((1,1,1),'f'),np.ones((1,1,1),'f'),np.full((2,1,1),np.finfo('f').max,'f')]
        args,out,_=self.args(arrays)
        self.assertEqual(self.libs[1].ck_attention_cross_backward_f32(*args),-4)
        for a in out: np.testing.assert_array_equal(a,91)

    def test_shared_readers_and_repeat(self):
        a=self.operands((3,3,2,3)); a[1]=a[0]; a[2]=a[0]
        for inputs in (a,self.operands((1,2,1,7)),a):
            args,out,_=self.args(inputs)
            self.assertEqual(self.libs[1].ck_attention_cross_backward_f32(*args),0)
            for x,y in zip(out,ordered(*inputs)): np.testing.assert_allclose(x[:-3],y.ravel(),atol=2e-8,rtol=2e-7)

if __name__=='__main__': unittest.main(verbosity=2)
