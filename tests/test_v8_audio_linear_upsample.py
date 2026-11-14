"""Checked half-pixel interpolation: arithmetic, rejection, generated binding."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from version.v8.tts.build_ming3_flow_step_circuit import build_circuit
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph

ROOT = Path(__file__).resolve().parents[1]
PTR = ctypes.POINTER(ctypes.c_float)

def oracle(x, factor):
    out = np.empty((len(x)*factor, x.shape[1]), np.float32)
    for row in range(len(out)):
        pos = max(0., (row+.5)/factor-.5)
        left = int(pos); right = min(left+1,len(x)-1)
        w = np.float32(pos-left)
        out[row] = np.float32(1-w)*x[left] + w*x[right]
    return out

class LinearUpsampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.libs=[]; cls.functions=[]
        for opt in ('-O0','-O3'):
            path=Path(cls.tmp.name)/(opt+'.so')
            subprocess.run(['cc','-std=c11',opt,'-Wall','-Wextra','-Werror','-pedantic',
                '-ffp-contract=off','-shared','-fPIC',str(ROOT/'src/kernels/audio_upsample_linear_checked.c'),
                '-lm','-o',str(path)],check=True,capture_output=True)
            lib=ctypes.CDLL(str(path)); cls.libs.append(lib)
            fn=lib.audio_upsample_linear_frames_f32_checked
            fn.argtypes=[PTR,ctypes.c_size_t,PTR,*([ctypes.c_size_t]*4)]
            fn.restype=ctypes.c_int; cls.functions.append(fn)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def call(self,fn,x,y,factor,ni=None,no=None,frames=None):
        return fn(x.ctypes.data_as(PTR),x.size if ni is None else ni,
            y.ctypes.data_as(PTR),y.size if no is None else no,
            len(x) if frames is None else frames,x.shape[1],factor)
    def test_shapes_and_padding(self):
        rng=np.random.default_rng(731)
        for frames,channels in ((1,1),(2,7),(3,64),(31,896)):
            x=rng.normal(size=(frames,channels)).astype('f4')
            for factor in (1,2,3,4,7):
                expected=oracle(x,factor)
                for fn in self.functions:
                    y=np.full(expected.size+13,-999.,np.float32)
                    self.assertEqual(self.call(fn,x,y,factor),0)
                    np.testing.assert_array_equal(y[:expected.size],expected.ravel())
                    self.assertTrue((y[expected.size:]==-999).all())
    def test_captured_publisher_values_and_wrong_mode_control(self):
        fixture=json.loads((ROOT/'tests/fixtures/ming3_flow/publisher_linear_upsample.json').read_text())
        x=np.array(fixture['input'],dtype='f4'); expected=np.array(fixture['expected'],dtype='f4')
        for fn in self.functions:
            y=np.zeros_like(expected)
            self.assertEqual(self.call(fn,x,y,fixture['factor']),0)
            np.testing.assert_array_equal(y,expected)
            self.assertFalse(np.array_equal(np.repeat(x,fixture['factor'],axis=0),expected))

    def test_rejection_and_recovery(self):
        x=np.arange(21,dtype='f4').reshape(3,7); y=np.full(84,-999.,np.float32)
        for fn in self.functions:
            for kwargs in ({'ni':20},{'no':83},{'frames':0},{'frames':2**63}):
                self.assertNotEqual(self.call(fn,x,y,4,**kwargs),0)
                self.assertTrue((y==-999).all())
            for factor in (0,65537):
                self.assertNotEqual(self.call(fn,x,y,factor),0)
                self.assertTrue((y==-999).all())
            for bad in (np.nan,np.inf,-np.inf):
                x[-1,-1]=bad
                self.assertNotEqual(self.call(fn,x,y,4),0)
                self.assertTrue((y==-999).all())
            x[-1,-1]=20
            before=x.copy()
            self.assertNotEqual(self.call(fn,x,x,1),0)
            np.testing.assert_array_equal(x,before)
            self.assertEqual(self.call(fn,x,y,4),0)
            np.testing.assert_array_equal(y,oracle(x,4).ravel()); y.fill(-999)
    def test_generated_binding(self):
        frames,channels,factor=3,7,4; n=frames*channels
        graph=build_circuit(n)
        graph['name']='linear_upsample_fixture'
        graph['activation_buffers']={'input':{'shape':[frames,channels]},'output':{'shape':[frames*factor,channels]}}
        graph['activation_bindings']={'input':'input','output':'output'}
        graph.pop('semantic_checkpoints')
        graph['block_types']['flow_step']['body']['ops']=[{'id':'upsample','op':'audio_upsample_linear_checked',
            'kernel':'audio_upsample_linear_frames_f32_checked','returns_status':True,
            'graph_slots':{'inputs':{'input':'external:input'},'outputs':{'output':'output'}},
            'params':{'T':frames,'C':channels,'factor':factor,'call_constants':{
                'linear_input_elements':n,'linear_output_elements':n*factor,
                'linear_frames':frames,'linear_channels':channels,'linear_factor':factor}}}]
        source={'config':{'model':graph['name'],'arch':graph['name'],'num_layers':1,'embed_dim':2,
            'num_heads':1,'num_kv_heads':1,'head_dim':2,'intermediate_size':2,
            'context_length':frames,'max_seq_len':frames,'vocab_size':1},'entries':[],
            'quant_summary':{},'template':graph}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'circuit.json';path.write_text(json.dumps(graph))
            layout,calls,library,loaded,fn=compile_native_graph(Path(d),source,path)
            size=layout['memory']['arena']['total_size']; raw=(ctypes.c_uint8*(size+63))()
            arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
            buffers={b['name']:b for b in layout['memory']['activations']['buffers']}
            x=np.ndarray((frames,channels),np.float32,buffer=arena,offset=buffers['input']['abs_offset'])
            y=np.ndarray((frames*factor,channels),np.float32,buffer=arena,offset=buffers['output']['abs_offset'])
            x[:]=np.arange(n,dtype='f4').reshape(x.shape)/7
            self.assertEqual(fn(arena,size),0);np.testing.assert_array_equal(y,oracle(x,factor))
            x[-1,-1]=np.nan; before=y.copy()
            self.assertNotEqual(fn(arena,size),0);np.testing.assert_array_equal(y,before)

if __name__=='__main__': unittest.main()
