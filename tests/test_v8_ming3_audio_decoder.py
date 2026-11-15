"""Nonzero generated latents-to-waveform, with independent mathematical reference."""
import json, tempfile, unittest
from pathlib import Path
import numpy as np
from version.v8.tts.build_ming3_audio_decoder_circuit import build_circuit, OUTPUT
from version.v8.tts.build_ming3_dit_block_circuit import weight_shapes
from version.v8.tts.replay_ming3_dit_block import compile_block
from tests.test_v8_audio_linear_upsample import oracle as interpolate

def reference(x,w,g,window,wrong_mask=False):
    c=g['contract']['runtime_invariants'];h=c['heads'];kh=c['kv_heads'];d=c['width']//h
    def linear(x,p):
        return (x.astype('f8')@w[p+'.weight'].astype('f8').T+w.get(p+'.bias',0)).astype('f4')
    def norm(x,p):
        return (x.astype('f8')/np.sqrt(np.mean(x.astype('f8')**2,-1,keepdims=True)+1e-6)*w[p+'.weight']).astype('f4')
    projection=linear(x,'fc1');x=interpolate(projection,c['factor']);t=len(x)
    for layer in range(c['depth']):
        p=f'decoder.layers.{layer}';n=norm(x,p+'.input_layernorm')
        q=linear(n,p+'.self_attn.q_proj').reshape(t,h,d).transpose(1,0,2).astype('f8')
        k=linear(n,p+'.self_attn.k_proj').reshape(t,kh,d).transpose(1,0,2).astype('f8')
        v=linear(n,p+'.self_attn.v_proj').reshape(t,kh,d).transpose(1,0,2).astype('f8')
        phase=np.arange(t)[:,None]/1000000.**(np.arange(d//2)*2/d)
        for a in (q,k):
            l=a[:,:,:d//2].copy();r=a[:,:,d//2:].copy()
            a[:,:,:d//2]=l*np.cos(phase)-r*np.sin(phase)
            a[:,:,d//2:]=r*np.cos(phase)+l*np.sin(phase)
        attention=np.empty((h,t,d),'f4')
        for head in range(h):
            for row in range(t):
                begin=0 if wrong_mask else max(0,row+1-c['window'])
                scores=(k[head//(h//kh),begin:row+1]@q[head,row])/np.sqrt(d)
                probs=np.exp(scores-scores.max());probs/=probs.sum()
                attention[head,row]=probs@v[head//(h//kh),begin:row+1]
        x=(x+linear(attention.transpose(1,0,2).reshape(t,-1),p+'.self_attn.o_proj')).astype('f4')
        n=norm(x,p+'.post_attention_layernorm');gate=linear(n,p+'.mlp.gate_proj').astype('f8')
        activated=(gate/(1+np.exp(-gate))*linear(n,p+'.mlp.up_proj')).astype('f4')
        x=(x+linear(activated,p+'.mlp.down_proj')).astype('f4')
    final=norm(x,'decoder.norm');spectra=linear(final,'head.out');b=spectra.shape[1]//2
    mag=np.minimum(np.exp(spectra[:,:b].astype('f8')),100)
    z=mag*np.exp(1j*spectra[:,b:].astype('f8'))
    frames=np.fft.irfft(z,n=c['n_fft'],axis=1)
    padded=(t-1)*c['hop']+c['n_fft'];y=np.zeros(padded);denom=np.zeros(padded)
    for i,frame in enumerate(frames):
        start=i*c['hop'];y[start:start+c['n_fft']]+=frame*window
        denom[start:start+c['n_fft']]+=window.astype('f8')**2
    crop=(c['n_fft']-c['hop'])//2
    return {'projection':projection,'final_norm':final,'spectral_projection':spectra,
            'output':(y[crop:padded-crop]/denom[crop:padded-crop]).astype('f4')}

class DecoderTests(unittest.TestCase):
    def test_freshness_and_rejection(self):
        self.assertEqual(json.loads(OUTPUT.read_text()),build_circuit())
        for kwargs in ({'frames':0},{'frames':513},{'depth':25},{'heads':3},{'kv_heads':3},{'n_fft':2},{'window':0}):
            with self.assertRaises(ValueError):build_circuit(**kwargs)
    def test_connected_lengths_repeat_rejection_and_mask(self):
        rng=np.random.default_rng(731)
        for frames,depth in ((2,1),(5,2),(2,24)):
            g=build_circuit(frames=frames,width=8,heads=2,kv_heads=1,depth=depth,inner=12,latent=4,n_fft=8,window=3)
            w={k:rng.normal(0,.18,size=shape).astype('f4') for k,shape in weight_shapes(g).items()}
            for key in w:
                if 'norm' in key:w[key]+=1
            x=rng.normal(size=(frames,4)).astype('f4');window=np.hanning(9)[:-1].astype('f4')
            expected=reference(x,w,g,window)
            with tempfile.TemporaryDirectory() as folder:
                fn,a,view,loaded=compile_block(Path(folder),g,w,openmp=True,c_standard='gnu11',provider_warnings='report')
                view('latent')[:]=x;view('window')[:]=window
                self.assertEqual(fn(a,len(a)),0)
                for key,value in expected.items():np.testing.assert_allclose(view(key),value,atol=1e-5,rtol=1e-5,err_msg=key)
                good=view('output').copy();view('latent')[:]=x+1
                self.assertEqual(fn(a,len(a)),0);self.assertGreater(abs(view('output')-good).max(),1e-5)
                view('latent')[:]=x;view('latent')[-1,-1]=np.nan;before=view('output').copy()
                self.assertNotEqual(fn(a,len(a)),0);np.testing.assert_array_equal(view('output'),before)
                view('latent')[:]=x;self.assertEqual(fn(a,len(a)),0);np.testing.assert_array_equal(view('output'),good)
                self.assertNotEqual(fn(a,len(a)-1),0);np.testing.assert_array_equal(view('output'),good)
                wrong=reference(x,w,g,window,wrong_mask=True)
                self.assertGreater(abs(wrong['spectral_projection']-expected['spectral_projection']).max(),1e-5)

if __name__=='__main__':unittest.main()
