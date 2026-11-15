"""Generated audio-latent feedback, independent nonzero reference and isolation."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from version.v8.tts.build_ming3_aggregator_circuit import build_circuit,OUTPUT
from version.v8.tts.build_ming3_dit_block_circuit import weight_shapes
from version.v8.tts.replay_ming3_dit_block import compile_block
from tests.test_v8_ming3_dit_block import reference as block_reference

def reference(x,w,g):
    c=g['contract']['runtime_invariants'];heads=c['heads']
    def linear(x,p):return (x.astype('f8')@w[p+'.weight'].astype('f8').T+w[p+'.bias']).astype('f4')
    x=np.concatenate([w['word_embedder.weight'],linear(x,'x_embedder')])
    phase=w['rotary.positions'][:,None]*w['rotary_embed.inv_freq'][None,:]
    for layer in range(c['depth']):
        prefix=f'blocks.{layer}.';local={k[len(prefix):]:v for k,v in w.items() if k.startswith(prefix)}
        x=block_reference(x,np.cos(phase),np.sin(phase),local,heads)['output']
    x=(x.astype('f8')/np.sqrt(np.mean(x.astype('f8')**2,-1,keepdims=True)+1e-6)*w['final_layer.norm_final.weight']).astype('f4')
    return linear(x,'final_layer.linear')

class AggregatorTests(unittest.TestCase):
    def test_freshness(self):self.assertEqual(json.loads(OUTPUT.read_text()),build_circuit())
    def test_nonzero_feedback_class_selection_rejection_and_repeat(self):
        rng=np.random.default_rng(731)
        for patches,depth in ((1,1),(3,2),(1,8)):
            g=build_circuit(patches=patches,width=8,heads=2,depth=depth,latent=4,output_width=6,mult=2)
            w={k:rng.normal(0,.2,size=shape).astype('f4') for k,shape in weight_shapes(g).items()}
            w['rotary.positions']=np.arange(patches+1,dtype='f4');w['rotary_embed.inv_freq']=np.array([1,.01],'f4')
            for key in w:
                if 'norm' in key:w[key]+=1
            x=rng.normal(size=(patches,4)).astype('f4');expected=reference(x,w,g)
            with tempfile.TemporaryDirectory() as folder:
                fn,a,view,loaded=compile_block(Path(folder),g,w)
                view('latent')[:]=x;view('class_id',np.int32)[:]=0
                self.assertEqual(fn(a,len(a)),0)
                np.testing.assert_allclose(view('all_rows'),expected,atol=4e-6,rtol=4e-6)
                np.testing.assert_array_equal(view('output'),view('all_rows')[:1])
                self.assertGreater(abs(view('output')-view('all_rows')[-1:]).max(),1e-4)
                good=view('output').copy();view('latent')[:]=x+3
                self.assertEqual(fn(a,len(a)),0);self.assertGreater(abs(view('output')-good).max(),1e-5)
                before=view('output').copy();view('class_id',np.int32)[:]=1
                self.assertNotEqual(fn(a,len(a)),0);np.testing.assert_array_equal(view('output'),before)
                view('class_id',np.int32)[:]=0;view('latent')[:]=x
                self.assertEqual(fn(a,len(a)),0);np.testing.assert_array_equal(view('output'),good)
                view('latent')[-1,-1]=np.nan
                self.assertNotEqual(fn(a,len(a)),0);np.testing.assert_array_equal(view('output'),good)

if __name__=='__main__':unittest.main()
