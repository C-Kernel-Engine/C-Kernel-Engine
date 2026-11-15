"""Dynamic backbone compared with an independent full-prefix NumPy oracle."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from version.v8.tts.build_ming3_backbone_circuit import build_circuit
from version.v8.tts.build_ming3_dit_block_circuit import weight_shapes
from version.v8.tts.replay_ming3_dit_block import compile_block
from version.v8.tts.native_guided_flow import ENTRY

def reference(x,w,heads,kv_heads,depth):
    t,width=x.shape;d=width//heads
    def linear(x,p):return (x.astype('f8')@w[p+'.weight'].astype('f8').T+w.get(p+'.bias',0)).astype('f4')
    def norm(x,p):return (x.astype('f8')/np.sqrt((x.astype('f8')**2).mean(-1,keepdims=True)+1e-6)*w[p+'.weight']).astype('f4')
    for layer in range(depth):
        p=f'model.layers.{layer}';n=norm(x,p+'.input_layernorm')
        q=linear(n,p+'.self_attn.q_proj').reshape(t,heads,d).transpose(1,0,2).astype('f8')
        k=linear(n,p+'.self_attn.k_proj').reshape(t,kv_heads,d).transpose(1,0,2).astype('f8')
        v=linear(n,p+'.self_attn.v_proj').reshape(t,kv_heads,d).transpose(1,0,2).astype('f8')
        phase=np.arange(t)[:,None]/1000000.**(np.arange(d//2)*2/d)
        for z in (q,k):
            left=z[:,:,:d//2].copy();right=z[:,:,d//2:].copy()
            z[:,:,:d//2]=left*np.cos(phase)-right*np.sin(phase)
            z[:,:,d//2:]=right*np.cos(phase)+left*np.sin(phase)
        attention=np.zeros((t,heads,d),'f4')
        for row in range(t):
            for h in range(heads):
                kh=h//(heads//kv_heads);scores=k[kh,:row+1]@q[h,row]/np.sqrt(d)
                probs=np.exp(scores-scores.max());probs/=probs.sum()
                attention[row,h]=probs@v[kh,:row+1]
        x=(x+linear(attention.reshape(t,width),p+'.self_attn.o_proj')).astype('f4')
        n=norm(x,p+'.post_attention_layernorm');gate=linear(n,p+'.mlp.gate_proj').astype('f8')
        activated=(gate/(1+np.exp(-gate))*linear(n,p+'.mlp.up_proj')).astype('f4')
        x=(x+linear(activated,p+'.mlp.down_proj')).astype('f4')
    hidden=norm(x,'model.norm');return hidden,linear(hidden,'stop_head')

class BackboneTests(unittest.TestCase):
    def test_prefix_append_independent_math_and_recovery(self):
        g=build_circuit(capacity=8,width=8,heads=2,kv_heads=1,depth=2,inner=12,vocab=16)
        rng=np.random.default_rng(731);w={k:rng.normal(0,.2,s).astype('f4') for k,s in weight_shapes(g).items()}
        for k in w:
            if 'norm' in k:w[k]+=1
        with tempfile.TemporaryDirectory() as d:
            fn,a,v,lib=compile_block(Path(d),g,w,openmp=True,c_standard='gnu11',provider_warnings='report')
            prefix=ENTRY(('ck_ming3_backbone_prefix',lib));ids=np.array([1,7,3],np.int32)
            v('prompt_ids',np.int32)[:3]=ids;v('request_rows',np.int32)[:]=3
            self.assertEqual(fn(a,len(a)),0);x=w['model.embed_tokens.weight'][ids].copy()
            for count in (3,4,5,6,7,8):
                if count>3:
                    row=rng.normal(size=(1,8)).astype('f4');x=np.concatenate([x,row]);v('embeddings')[count-1]=row[0]
                    v('request_rows',np.int32)[:]=count;self.assertEqual(prefix(a,len(a)),0)
                hidden,stop=reference(x,w,2,1,2)
                np.testing.assert_allclose(v('final_norm')[:count],hidden,atol=1e-5,rtol=1e-5)
                np.testing.assert_allclose(v('stop_logits')[:count],stop,atol=1e-5,rtol=1e-5)
            before=v('stop_logits').copy()
            for count in (0,9,-1):
                v('request_rows',np.int32)[:]=count;self.assertNotEqual(prefix(a,len(a)),0)
                np.testing.assert_array_equal(v('stop_logits'),before)
            v('request_rows',np.int32)[:]=3;v('prompt_ids',np.int32)[1]=16
            self.assertNotEqual(fn(a,len(a)),0)
            v('prompt_ids',np.int32)[:3]=ids;self.assertEqual(fn(a,len(a)),0)
            np.testing.assert_allclose(v('final_norm')[:3],reference(w['model.embed_tokens.weight'][ids],w,2,1,2)[0],atol=1e-5,rtol=1e-5)

if __name__=="__main__":
    import unittest
    unittest.main()
