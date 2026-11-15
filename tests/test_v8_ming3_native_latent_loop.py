"""Connected generated backbone/flow/aggregator; no oracle activations."""
import ctypes as C,unittest
import numpy as np
from tests import test_v8_ming3_native_flow as flow_fixture
from version.v8.tts.build_ming3_backbone_circuit import build_circuit as backbone
from version.v8.tts.build_ming3_aggregator_circuit import build_circuit as aggregator
from version.v8.tts.build_ming3_dit_block_circuit import weight_shapes
from version.v8.tts.build_ming3_sampler_update_circuit import build_circuit as update_graph
from version.v8.tts.replay_ming3_dit_block import compile_block
from version.v8.tts.native_guided_flow import ENTRY,U,P,S,Request,CANCEL,offsets
from version.v8.tts.native_latent_loop import LatentLoop

class LatentLoopTests(unittest.TestCase):
    request=flow_fixture.FlowTests.request
    expected=flow_fixture.FlowTests.expected
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    backbone_builder=staticmethod(backbone)
    @classmethod
    def setUpClass(cls):
        flow_fixture.FlowTests.setUpClass.__func__(cls)
        # Flow conditioning width six matches this backbone and aggregator output.
        cls.bg=cls.backbone_builder(capacity=12,width=6,heads=1,kv_heads=1,depth=1,inner=12,vocab=16)
        cls.ag=aggregator(width=8,heads=2,depth=1,latent=4,output_width=6,mult=2)
        rng=np.random.default_rng(9)
        for name,g in [('backbone',cls.bg),('aggregator',cls.ag)]:
            folder=cls.root/name;folder.mkdir()
            weights={k:rng.normal(0,.15,shape).astype('f4') for k,shape in weight_shapes(g).items()}
            for k in weights:
                if 'norm' in k:weights[k]+=1
            if name=='backbone':
                weights['stop_head.weight'][:]=0;weights['stop_head.bias'][:]=[-1,1]
            else:
                weights['rotary.positions']=np.arange(2,dtype='f4')
                weights['rotary_embed.inv_freq']=np.array([1,.01],'f4')
            setattr(cls,name+'_weights',weights)
            setattr(cls,name,compile_block(folder,g,weights,openmp=True,c_standard='gnu11',provider_warnings='report'))
        folder=cls.root/'loop';folder.mkdir()
        cls.loop_bundle=compile_block(folder,update_graph(4),{},
            runtime_sources=['src/runtime/checked_guided_flow.c','src/runtime/checked_latent_loop.c'],
            extra_exports=['ck_run_checked_latent_loop'])
        cls.loop=cls.loop_bundle[3].ck_run_checked_latent_loop
        cls.loop.argtypes=[C.POINTER(LatentLoop)];cls.loop.restype=C.c_int
    def make_loop(self):
        flow,arrays=self.request()
        b=offsets(self.root/'backbone');a=offsets(self.root/'aggregator')
        bb=self.backbone;ab=self.aggregator
        bb[2]('prompt_ids',np.int32)[:]=1
        staging=np.zeros((5,4),'f4');output=np.full((5,4),-999.,'f4');valid=S(99)
        request=LatentLoop(ENTRY(('ck_ming3_backbone_embed',bb[3])),ENTRY(('ck_ming3_backbone_prefix',bb[3])),
            C.cast(ab[0],ENTRY),C.cast(bb[1],U),C.cast(ab[1],U),len(bb[1]),len(ab[1]),
            *[b[k] for k in ('request_rows','embeddings','final_norm','stop_logits')],
            *[a[k] for k in ('latent','class_id','output')],6,4,12,3,5,5,C.pointer(flow),
            staging.ctypes.data_as(P),output.ctypes.data_as(P),staging.size,output.size,C.pointer(valid))
        request.policy_version=1
        return request,(flow,arrays,staging,output,valid)
    def test_connected_repeat_budget_and_cancel(self):
        r,a=self.make_loop();self.assertEqual(self.loop(C.byref(r)),0)
        self.assertEqual(a[-1].value,5);baseline=a[-2].copy();self.assertTrue(np.isfinite(baseline).all())
        from tests.test_v8_ming3_backbone import reference as backbone_reference
        from tests.test_v8_ming3_aggregator import reference as aggregator_reference
        x=self.backbone_weights['model.embed_tokens.weight'][[1,1,1]].copy()
        history=np.zeros(4,'f4');expected=[]
        for step in range(5):
            hidden,_=backbone_reference(x,self.backbone_weights,1,1,1)
            _,arrays=self.request(step);arrays[0][:]=hidden[-1];arrays[1][:]=history
            latent=self.expected(arrays,step);expected.append(latent)
            feedback=aggregator_reference(latent[None,:],self.aggregator_weights,self.ag)[:1]
            x=np.concatenate([x,feedback]);history=latent
        np.testing.assert_allclose(baseline,expected,atol=3e-5,rtol=3e-5)
        # Distinct prompt affects generated conditioning and speech latents.
        r,b=self.make_loop();self.backbone[2]('prompt_ids',np.int32)[:]=2
        self.assertEqual(self.loop(C.byref(r)),0);self.assertGreater(abs(b[-2]-baseline).max(),1e-5)
        r,a=self.make_loop();self.assertEqual(self.loop(C.byref(r)),0)
        np.testing.assert_array_equal(a[-2],baseline)
        r,a=self.make_loop();r.capacity=3
        self.assertEqual(self.loop(C.byref(r)),2);self.assertEqual(a[-1].value,0)
        self.assertTrue((a[-2]==-999).all())
        for boundary in (1,5,40,100):
            r,a=self.make_loop();calls=[0]
            def cancel(context):calls[0]+=1;return int(calls[0]>=boundary)
            cb=CANCEL(cancel);a[0].should_cancel=cb
            self.assertEqual(self.loop(C.byref(r)),-7);self.assertTrue((a[-2]==-999).all())
            a[0].should_cancel=CANCEL();self.assertEqual(self.loop(C.byref(r)),0)
            np.testing.assert_array_equal(a[-2],baseline)
    def test_invalid_rejected_and_native_failure_recovers(self):
        for field,value in [('policy_version',0),('policy_version',2),('prompt_rows',13),('min_frames',6),('output_elements',19)]:
            r,a=self.make_loop();setattr(r,field,value)
            self.assertEqual(self.loop(C.byref(r)),-1);self.assertTrue((a[-2]==-999).all())
        r,a=self.make_loop();bad=ENTRY(lambda arena,size:-33);r.aggregate=bad
        self.assertEqual(self.loop(C.byref(r)),-2);self.assertTrue((a[-2]==-999).all())
        r.aggregate=C.cast(self.aggregator[0],ENTRY)
        self.assertEqual(self.loop(C.byref(r)),0)

if __name__=="__main__":
    import unittest
    unittest.main()
