"""Resident dynamic lengths share the fixed decoder's independent oracle."""
import tempfile,unittest,ctypes as C
from pathlib import Path
import numpy as np
from version.v8.tts.build_ming3_runtime_audio_decoder_circuit import build_circuit
from version.v8.tts.build_ming3_dit_block_circuit import weight_shapes
from version.v8.tts.replay_ming3_dit_block import compile_block
from tests.test_v8_ming3_audio_decoder import reference

class RuntimeDecoderTests(unittest.TestCase):
    def test_resident_lengths_capacity_and_recovery(self):
        g=build_circuit(capacity=5,width=8,heads=2,kv_heads=1,depth=2,inner=12,latent=4,n_fft=8,window=3)
        rng=np.random.default_rng(731)
        w={k:rng.normal(0,.1,size=s).astype('f4') for k,s in weight_shapes(g).items()}
        for k in w:
            if 'norm' in k:w[k]+=1
        window=np.hanning(9)[:-1].astype('f4');inputs={n:rng.normal(size=(n,4)).astype('f4') for n in (1,2,5)}
        with tempfile.TemporaryDirectory() as d:
            fn,a,v,lib=compile_block(Path(d),g,w,openmp=True,c_standard='gnu11',provider_warnings='report')
            v('window')[:]=window;first=None;count=C.c_int32(-1)
            for frames in (1,5,2,1):
                x=inputs[frames];v('latent')[:]=np.nan;v('latent')[:frames]=x
                v('request_frames',np.int32)[:]=frames
                expected=reference(x,w,g,window)['output']
                self.assertEqual(fn(a,len(a),C.byref(count)),0)
                self.assertEqual(count.value,len(expected))
                np.testing.assert_allclose(v('output')[:len(expected)],expected,atol=1e-5,rtol=1e-5)
                if frames==1:
                    if first is None:first=v('output')[:len(expected)].copy()
                    else:np.testing.assert_array_equal(v('output')[:len(expected)],first)
            before=v('output').copy()
            for invalid in (0,-1,6,2147483647):
                v('request_frames',np.int32)[:]=invalid
                self.assertNotEqual(fn(a,len(a),C.byref(count)),0)
                np.testing.assert_array_equal(v('output'),before)
            v('request_frames',np.int32)[:]=1
            self.assertEqual(fn(a,len(a),C.byref(count)),0)

if __name__=="__main__":
    import unittest
    unittest.main()
