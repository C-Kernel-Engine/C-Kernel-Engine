"""Cached continuation compared with uninterrupted prefill, including logical KV."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from version.v8.tts.build_ming3_cached_backbone_circuit import build_circuit as cached
from version.v8.tts.build_ming3_backbone_circuit import build_circuit as full
from version.v8.tts.build_ming3_dit_block_circuit import weight_shapes
from version.v8.tts.replay_ming3_dit_block import compile_block
from version.v8.tts.native_guided_flow import ENTRY

class CachedBackboneTests(unittest.TestCase):
    def test_append_full_prefix_and_cache_reset(self):
        kwargs=dict(capacity=9,width=16,heads=4,kv_heads=2,depth=2,inner=12,vocab=16)
        g=cached(**kwargs);rng=np.random.default_rng(49)
        w={k:rng.normal(0,.2,s).astype('f4') for k,s in weight_shapes(g).items()}
        for k in w:
            if 'norm' in k:w[k]+=1
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'cached').mkdir();(p/'full').mkdir()
            _,a,v,l=compile_block(p/'cached',g,w,openmp=True,c_standard='gnu11',provider_warnings='report')
            _,fa,fv,fl=compile_block(p/'full',full(**kwargs),w,openmp=True,c_standard='gnu11',provider_warnings='report')
            embed=ENTRY(('ck_ming3_backbone_embed',l));prefix=ENTRY(('ck_ming3_backbone_prefix',l));decode=ENTRY(('ck_ming3_backbone_decode',l))
            fe=ENTRY(('ck_ming3_backbone_embed',fl));fp=ENTRY(('ck_ming3_backbone_prefix',fl))
            sequences=[]
            for initial in (3,5,3):
                ids=list(range(initial));v('prompt_ids',np.int32)[:initial]=ids;fv('prompt_ids',np.int32)[:initial]=ids
                v('request_rows',np.int32)[:]=initial;fv('request_rows',np.int32)[:]=initial
                self.assertEqual(embed(a,len(a)),0);self.assertEqual(fe(fa,len(fa)),0)
                self.assertEqual(prefix(a,len(a)),0);self.assertEqual(fp(fa,len(fa)),0)
                history=[]
                for pos in range(initial,9):
                    row=np.sin(np.arange(16,dtype='f4')+pos)
                    v('decode_embeddings')[:]=row;v('decode_position_input',np.int32)[:]=pos
                    fv('embeddings')[pos]=row;fv('request_rows',np.int32)[:]=pos+1
                    self.assertEqual(decode(a,len(a)),0);self.assertEqual(fp(fa,len(fa)),0)
                    np.testing.assert_allclose(v('decode_final_norm')[0],fv('final_norm')[pos],atol=1e-5,rtol=1e-5)
                    np.testing.assert_allclose(v('decode_stop_logits')[0],fv('stop_logits')[pos],atol=1e-5,rtol=1e-5)
                    for layer in range(2):
                        for kind in ('k','v'):
                            actual=v(f'layer{layer}_cache_{kind}')[:,:pos+1]
                            # Full-prefix layouts are packed using valid rows, not capacity strides.
                            expected=fv(f'layer{layer}_{kind}_head').reshape(-1)[:2*(pos+1)*4].reshape(2,pos+1,4)
                            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
                    history.append(v('decode_stop_logits').copy())
                sequences.append(history)
            for arow,brow in zip(sequences[0],sequences[2]):np.testing.assert_array_equal(arow,brow)
            before=v('layer0_cache_k').copy();v('decode_position_input',np.int32)[:]=9
            self.assertNotEqual(decode(a,len(a)),0);np.testing.assert_array_equal(v('layer0_cache_k'),before)
            # A dropped-history fault changes the intended numerical gate.
            v('decode_position_input',np.int32)[:]=8;v('layer0_cache_k')[:,:8]=0;v('layer0_cache_v')[:,:8]=0
            self.assertEqual(decode(a,len(a)),0)
            self.assertGreater(abs(v('decode_stop_logits')[0]-fv('stop_logits')[8]).max(),1e-5)

from tests import test_v8_ming3_native_latent_loop as loop_tests
class CachedNativeLoopTests(loop_tests.LatentLoopTests):
    backbone_builder=staticmethod(cached)
    def make_loop(self):
        from version.v8.tts.native_guided_flow import offsets
        r,a=super().make_loop();b=offsets(self.root/'backbone')
        r.decode=ENTRY(('ck_ming3_backbone_decode',self.backbone[3]))
        r.decode_input=b['decode_embeddings'];r.decode_hidden=b['decode_final_norm']
        r.decode_stop_logits=b['decode_stop_logits'];r.decode_position=b['decode_position_input']
        return r,a

if __name__=='__main__':unittest.main()
