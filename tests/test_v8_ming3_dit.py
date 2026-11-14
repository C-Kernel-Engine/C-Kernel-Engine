"""Connected learned DiT layers with native positional features."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from version.v8.tts.build_ming3_dit_circuit import build_circuit,constant_weights,OUTPUT
from version.v8.tts.build_ming3_dit_block_circuit import weight_shapes
from version.v8.tts.replay_ming3_dit_block import compile_block
from tests.test_v8_ming3_dit_block import reference as block_reference

def reference(inputs,w,depth,heads):
    def linear(x,p):return (x.astype('f8')@w[p+'.weight'].astype('f8').T+w[p+'.bias']).astype('f4')
    phase=(inputs['time']*np.float32(1000))*w['timestep.inverse_frequency']
    features=np.concatenate([np.sin(phase.astype('f8')),np.cos(phase.astype('f8'))]).astype('f4')[None,:]
    rotary=w['rotary.positions'][:,None]*w['rotary_embed.inv_freq'][None,:]
    cosine=np.cos(rotary.astype('f8')).astype('f4');sine=np.sin(rotary.astype('f8')).astype('f4')
    t=linear(features,'t_embedder.time_mlp.0')
    t=(t.astype('f8')/(1+np.exp(-t.astype('f8')))).astype('f4')
    t=linear(t,'t_embedder.time_mlp.2')
    c=linear(inputs['condition'],'c_embedder.cond_embedder')
    x=np.concatenate([(t+c).astype('f4'),linear(inputs['history'],'x_embedder'),linear(inputs['latent'],'x_embedder')])
    for layer in range(depth):
        prefix=f'blocks.{layer}.';local={key[len(prefix):]:value for key,value in w.items() if key.startswith(prefix)}
        x=block_reference(x,cosine,sine,local,heads)['output']
    norm=(x.astype('f8')/np.sqrt(np.mean(x.astype('f8')**2,-1,keepdims=True)+1e-6)*w['final_layer.norm_final.weight']).astype('f4')
    return linear(norm,'final_layer.linear')

class MingDiTTests(unittest.TestCase):
    def test_freshness(self):self.assertEqual(json.loads(OUTPUT.read_text()),build_circuit())
    def test_nonzero_layers_conditioning_rejection_and_repeat(self):
        rng=np.random.default_rng(731)
        for depth in (1,2,8):
            graph=build_circuit(width=8,heads=2,depth=depth,latent=4,condition=6,time_width=8,mult=2)
            w={key:rng.normal(0,.1,size=shape).astype('f4') for key,shape in weight_shapes(graph).items()}
            w.update(constant_weights(8))
            w['rotary_embed.inv_freq']=np.array([1,.01],dtype='f4')
            for key in w:
                if 'norm' in key and key.endswith('.weight'):w[key]+=1
            inputs={key:rng.normal(size=graph['activation_buffers'][key]['shape']).astype('f4') for key in ('latent','history','condition','time')}
            expected=reference(inputs,w,depth,2)
            with tempfile.TemporaryDirectory() as folder:
                fn,arena,view,loaded=compile_block(Path(folder),graph,w)
                for key,value in inputs.items():view(key)[:]=value
                self.assertEqual(fn(arena,len(arena)),0)
                np.testing.assert_allclose(view('output'),expected,atol=3e-6,rtol=3e-6)
                original=view('output').copy()
                view('time')[:]+=np.float32(.25)
                self.assertEqual(fn(arena,len(arena)),0)
                self.assertGreater(float(abs(view('output')-original).max()),1e-5)
                view('time')[:]=inputs['time']
                view('condition')[:]+=10
                self.assertEqual(fn(arena,len(arena)),0)
                self.assertGreater(float(abs(view('output')-original).max()),1e-5)
                view('condition')[:]=inputs['condition'];view('time')[0]=np.nan;before=view('output').copy()
                self.assertNotEqual(fn(arena,len(arena)),0);np.testing.assert_array_equal(view('output'),before)
                view('time')[:]=inputs['time'];self.assertEqual(fn(arena,len(arena)),0)
                np.testing.assert_array_equal(view('output'),original)
                self.assertNotEqual(fn(arena,len(arena)-1),0);np.testing.assert_array_equal(view('output'),original)

if __name__=='__main__':unittest.main()
