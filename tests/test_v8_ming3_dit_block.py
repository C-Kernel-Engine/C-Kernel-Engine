"""Nonzero FP32 DiT composition; distinct from publisher BF16 parity."""
import ctypes,json
from pathlib import Path
import tempfile,unittest
import numpy as np
from version.v8.tts.build_ming3_dit_block_circuit import build_circuit,weight_shapes,OUTPUT
from version.v8.scripts.compile_checked_bundle_v8 import compile_native_graph
ROOT=Path(__file__).resolve().parents[1]

from version.v8.tts.replay_ming3_dit_block import compile_block

def reference(x,cos,sin,w,heads):
    stages={}
    def norm(x,name):
        y=(x.astype('f8')/np.sqrt(np.mean(x.astype('f8')**2,-1,keepdims=True)+1e-6)*w[name+'.weight']).astype('f4');stages[name]=y;return y
    def linear(x,name,stage):
        y=(x.astype('f8')@w[name+'.weight'].astype('f8').T+w[name+'.bias'].astype('f8')).astype('f4');stages[stage]=y;return y
    def rotate(x):
        y=x.reshape(len(x),heads,-1).copy();a=y[:,:,::2].copy();b=y[:,:,1::2].copy()
        y[:,:,::2]=a*cos[:,None]-b*sin[:,None];y[:,:,1::2]=b*cos[:,None]+a*sin[:,None];return y.reshape(x.shape)
    normalized=norm(x,'norm1');q=linear(normalized,'attn.to_q','q');k=linear(normalized,'attn.to_k','k');v=linear(normalized,'attn.to_v','v')
    q=rotate(q);k=rotate(k);stages['q_rot']=q;stages['k_rot']=k
    d=x.shape[-1]//heads
    qh=q.reshape(len(x),heads,d).transpose(1,0,2).astype('f8');kh=k.reshape(len(x),heads,d).transpose(1,0,2).astype('f8');vh=v.reshape(len(x),heads,d).transpose(1,0,2).astype('f8')
    score=((qh@kh.transpose(0,2,1))/np.sqrt(d)).astype('f4').astype('f8');prob=np.exp(score-score.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
    context=(prob@vh).transpose(1,0,2).reshape(x.shape).astype('f4');stages['attention']=context
    z=(x+linear(context,'attn.to_out.0','attention_projection')).astype('f4');stages['residual1']=z
    up=linear(norm(z,'norm2'),'mlp.ff.0.0','up')
    active=(.5*up.astype('f8')*(1+np.tanh(np.sqrt(2/np.pi)*(up.astype('f8')+.044715*up.astype('f8')**3)))).astype('f4');stages['activated']=active
    result=(z+linear(active,'mlp.ff.2','down')).astype('f4');stages['output']=result
    return stages

class MingDiTTests(unittest.TestCase):
    def test_freshness_bounds(self):
        self.assertEqual(json.loads(OUTPUT.read_text()),build_circuit())
        for kwargs in ({'tokens':0},{'width':7},{'heads':0},{'tokens':True},{'mult':9}):
            with self.assertRaises(ValueError):build_circuit(**kwargs)
    def test_nonzero_generated_composition_and_corruption(self):
        rng=np.random.default_rng(731)
        for tokens in (2,3,7):
            graph=build_circuit(tokens,16,2,2)
            w={name:rng.normal(0,.12,size=shape).astype('f4') for name,shape in weight_shapes(graph).items()}
            for name in ('norm1.weight','norm2.weight'):w[name]+=1
            x=rng.normal(size=(tokens,16)).astype('f4')
            phase=np.arange(tokens)[:,None]*np.array([1,.1,.01,.001])[None,:];cos=np.cos(phase).astype('f4');sin=np.sin(phase).astype('f4')
            expected=reference(x,cos,sin,w,2)
            with tempfile.TemporaryDirectory() as temp:
                fn,arena,view,loaded=compile_block(Path(temp),graph,w)
                view('input')[:]=x;view('cosine')[:]=cos;view('sine')[:]=sin
                self.assertEqual(fn(arena,len(arena)),0)
                for key,value in expected.items():
                    np.testing.assert_allclose(view(key),value,atol=2e-6,rtol=2e-6,err_msg=key)
                layout=json.loads((Path(temp)/'layout.json').read_text())
                planned=next(row for row in layout['memory']['weights']['entries'] if row['name']=='attn.to_q.weight')
                damaged=np.ndarray(w['attn.to_q.weight'].shape,np.float32,buffer=arena,offset=planned['abs_offset'])
                original=float(damaged[0,0]);damaged[0,0]+=1
                self.assertEqual(fn(arena,len(arena)),0)
                self.assertGreater(float(abs(view('q')-expected['q']).max()),1e-4)
                damaged[0,0]=original
                view('cosine')[1,0]+=.5
                self.assertEqual(fn(arena,len(arena)),0)
                self.assertGreater(float(abs(view('q_rot')-expected['q_rot']).max()),1e-4)
                view('cosine')[:]=cos;before=view('output').copy();view('sine')[-1,-1]=np.nan
                self.assertNotEqual(fn(arena,len(arena)),0);np.testing.assert_array_equal(view('output'),before)
                view('sine')[:]=sin;self.assertEqual(fn(arena,len(arena)),0)
                np.testing.assert_allclose(view('output'),expected['output'],atol=2e-6,rtol=2e-6)

if __name__=='__main__':unittest.main()
