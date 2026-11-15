"""Explicit-window/crop inverse transform and BF16 fold-order regression."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ctypes.POINTER(ctypes.c_float); D=ctypes.POINTER(ctypes.c_double); S=ctypes.c_size_t

def bf16(x):
    a=np.array(x,dtype='f4');u=a.view('u4');u=(u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)
    return u.view('f4')

def reference(real,imag,window,hop,left,right,bfloat=False,reverse=True):
    n=len(window);frames=len(real);size=(frames-1)*hop+n
    wave=np.fft.irfft(real.astype('f8')+1j*imag.astype('f8'),n=n,axis=1)
    total=np.zeros(size);env=np.zeros(size)
    for t in range(frames):total[t*hop:t*hop+n]+=wave[t]*window
    order=range(frames-1,-1,-1) if reverse else range(frames)
    for t in order:
        region=slice(t*hop,t*hop+n)
        if bfloat:env[region]=bf16(env[region]+bf16(window*window))
        else:env[region]+=window.astype('f8')**2
    end=size-right
    return (total[left:end]/env[left:end]).astype('f4')

class ComplexIstftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();path=Path(cls.tmp.name)/'istft.so'
        subprocess.run(['cc','-O3','-std=c11','-Wall','-Wextra','-Werror','-ffp-contract=off',
            '-shared','-fPIC','-I',str(ROOT/'include'),str(ROOT/'src/kernels/audio_istft_complex_window_checked.c'),
            '-lm','-o',str(path)],check=True,capture_output=True)
        cls.lib=ctypes.CDLL(str(path));cls.functions=[]
        for name in ('audio_istft_complex_window_f32_checked','audio_istft_complex_window_bf16_envelope_f32_checked'):
            fn=getattr(cls.lib,name);fn.restype=ctypes.c_int
            fn.argtypes=[P,P,S,P,*([S]*6),P,S,D,S];cls.functions.append(fn)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def call(self,fn,a,b,w,hop,left,right,out,scratch=None,**kwargs):
        n=len(w);padded=(len(a)-1)*hop+n
        if scratch is None:scratch=np.zeros(2*padded+n,'f8')
        return fn(a.ctypes.data_as(P),b.ctypes.data_as(P),kwargs.get('spectral_elements',a.size),
            w.ctypes.data_as(P),kwargs.get('window_elements',w.size),len(a),n,hop,left,right,
            out.ctypes.data_as(P),kwargs.get('output_elements',out.size),scratch.ctypes.data_as(D),
            kwargs.get('scratch_elements',scratch.size))
    def test_independent_fft_and_order_negative_control(self):
        rng=np.random.default_rng(731);found=False
        for n,hop,frames in ((20,5,1),(20,5,7),(32,8,13),(3528,882,4)):
            a=rng.normal(size=(frames,n//2+1)).astype('f4');b=rng.normal(size=a.shape).astype('f4')
            w=bf16(.5-.5*np.cos(2*np.pi*np.arange(n)/n));crop=(n-hop)//2
            for mode,fn in enumerate(self.functions):
                expected=reference(a,b,w,hop,crop,crop,bool(mode))
                out=np.full(expected.size+7,-999.,'f4')
                self.assertEqual(self.call(fn,a,b,w,hop,crop,crop,out),0)
                np.testing.assert_allclose(out[:len(expected)],expected,atol=2e-7,rtol=2e-6)
                self.assertTrue((out[len(expected):]==-999).all())
                if mode:
                    wrong=reference(a,b,w,hop,crop,crop,True,False)
                    found |= bool(np.max(abs(wrong-expected))>1e-5)
        self.assertTrue(found,'wrong BF16 fold order must fail numerical gate')
    def test_retained_publisher_bf16_fold(self):
        f=json.loads((ROOT/'tests/fixtures/ming3_flow/publisher_istft_bf16.json').read_text())
        a=np.array(f['real'],'f4');b=np.array(f['imag'],'f4');w=np.array(f['window'],'f4')
        expected=np.array(f['expected'],'f4');out=np.zeros_like(expected)
        self.assertEqual(self.call(self.functions[1],a,b,w,f['hop'],f['left_crop'],f['right_crop'],out),0)
        np.testing.assert_allclose(out,expected,rtol=2e-6,atol=2e-7)
        wrong=reference(a,b,w,f['hop'],f['left_crop'],f['right_crop'],True,False)
        self.assertGreater(float(abs(wrong-expected).max()),1e-5)

    def test_generated_waveform_and_scratch_width(self):
        from version.v8.tts.build_ming3_waveform_circuit import build_circuit,OUTPUT
        from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph
        self.assertEqual(json.loads(OUTPUT.read_text()),build_circuit())
        for frames in (1,7):
            graph=build_circuit(frames,20,5)
            source={'config':{'model':graph['name'],'arch':graph['name'],'num_layers':1,'embed_dim':2,
                'num_heads':1,'num_kv_heads':1,'head_dim':2,'intermediate_size':2,
                'context_length':frames,'max_seq_len':frames,'vocab_size':1},'entries':[],
                'quant_summary':{},'template':graph}
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'circuit.json';path.write_text(json.dumps(graph))
                layout,calls,library,loaded,fn=compile_native_graph(Path(d),source,path)
                size=layout['memory']['arena']['total_size'];raw=(ctypes.c_uint8*(size+63))()
                arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
                buffers={b['name']:b for b in layout['memory']['activations']['buffers']}
                def view(name):return np.ndarray(tuple(graph['activation_buffers'][name]['shape']),np.float32,buffer=arena,offset=buffers[name]['abs_offset'])
                rng=np.random.default_rng(731)
                a=view('real');b=view('imag');w=view('window');y=view('output')
                a[:]=rng.normal(size=a.shape);b[:]=rng.normal(size=b.shape)
                w[:]=bf16(.5-.5*np.cos(2*np.pi*np.arange(20)/20))
                expected=reference(a,b,w,5,7,7,True)
                self.assertEqual(fn(arena,size),0)
                np.testing.assert_allclose(y,expected,atol=2e-7,rtol=2e-6)
                before=y.copy();a[-1,-1]=np.nan
                self.assertNotEqual(fn(arena,size),0);np.testing.assert_array_equal(y,before)

    def test_capacities_overlap_recovery_and_bad_normalization(self):
        n=20;frames=7;hop=5;crop=7
        a=np.ones((frames,n//2+1),'f4');b=np.zeros_like(a)
        w=bf16(.5-.5*np.cos(2*np.pi*np.arange(n)/n));size=(frames-1)*hop+n-2*crop
        y=np.full(size,-999.,'f4');scratch=np.zeros(2*((frames-1)*hop+n)+n,'f8')
        for fn in self.functions:
            for kw in ({'spectral_elements':a.size-1},{'window_elements':n-1},
                       {'output_elements':size-1},{'scratch_elements':scratch.size-1}):
                self.assertNotEqual(self.call(fn,a,b,w,hop,crop,crop,y,**kw),0)
                self.assertTrue((y==-999).all())
            for operand in (a,b,w):
                old=operand.flat[-1];operand.flat[-1]=np.nan
                self.assertNotEqual(self.call(fn,a,b,w,hop,crop,crop,y),0)
                self.assertTrue((y==-999).all());operand.flat[-1]=old
            self.assertNotEqual(self.call(fn,a,b,np.zeros_like(w),hop,crop,crop,y),0)
            self.assertTrue((y==-999).all())
            before=a.copy();self.assertNotEqual(self.call(fn,a,b,w,hop,crop,crop,a.ravel()),0)
            np.testing.assert_array_equal(a,before)
            aliased=scratch.view('f4')[:size];aliased.fill(-999.)
            self.assertNotEqual(self.call(fn,a,b,w,hop,crop,crop,aliased,scratch),0)
            self.assertTrue((aliased==-999).all())
            self.assertEqual(self.call(fn,a,b,w,hop,crop,crop,y),0);self.assertTrue(np.isfinite(y).all());y.fill(-999)
        huge=np.full_like(w,np.float32(2.0**100))
        self.assertNotEqual(self.call(self.functions[1],a,b,huge,hop,crop,crop,y),0)
        self.assertTrue((y==-999).all())
        w[-1]=.123456
        self.assertNotEqual(self.call(self.functions[1],a,b,w,hop,crop,crop,y),0)
        self.assertTrue((y==-999).all())

if __name__=='__main__':unittest.main()
