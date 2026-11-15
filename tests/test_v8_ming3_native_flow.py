"""Native guided sampling calls real generated DiT/update entries, with recovery."""
import ctypes as C,tempfile,unittest
from pathlib import Path
import numpy as np
from version.v8.tts.build_ming3_dit_circuit import build_circuit,constant_weights
from version.v8.tts.build_ming3_sampler_update_circuit import build_circuit as update_graph
from version.v8.tts.build_ming3_dit_block_circuit import weight_shapes
from version.v8.tts.replay_ming3_dit_block import compile_block
from version.v8.tts.native_guided_flow import Model,Update,Request,ENTRY,NOISE,CANCEL,P,U,offsets
from version.v8.tts.ming3_execution_profile import sampling_times as schedule
from tests.test_v8_ming3_dit import reference as model_reference
from tests.test_v8_audio_noise_counter_oracle import oracle as noise_reference

class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        (cls.root/'model').mkdir();(cls.root/'update').mkdir()
        cls.g=build_circuit(width=8,heads=2,depth=1,latent=4,condition=6,time_width=8,mult=2)
        rng=np.random.default_rng(731)
        cls.w={k:rng.normal(0,.15,size=shape).astype('f4') for k,shape in weight_shapes(cls.g).items()}
        cls.w.update(constant_weights(8));cls.w['rotary_embed.inv_freq']=np.array([1,.01],'f4')
        for k in cls.w:
            if 'norm' in k:cls.w[k]+=1
        cls.model=compile_block(cls.root/'model',cls.g,cls.w)
        cls.update=compile_block(cls.root/'update',update_graph(4),{},runtime_sources=['src/runtime/checked_guided_flow.c'],
            extra_exports=['ck_run_checked_guided_flow','audio_noise_counter_gaussian_f32'])
        cls.driver=cls.update[3].ck_run_checked_guided_flow;cls.driver.argtypes=[C.POINTER(Request)];cls.driver.restype=C.c_int
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def request(self,stream=0):
        m=offsets(self.root/'model');u=offsets(self.root/'update')
        model=Model(ENTRY(('ck_ming3_dit_fp32_bounded',self.model[3])),C.cast(self.model[1],U),len(self.model[1]),
            m['latent'],m['time'],m['condition'],m['history'],m['output']+2*4*4)
        update=Update(ENTRY(('ck_ming3_sampler_update',self.update[3])),C.cast(self.update[1],U),len(self.update[1]),
            *[u[k] for k in ('latent','prediction','null_prediction','noise','neg_one','cfg','dt','noise_amplitude','output')])
        cond=np.linspace(-.8,.7,6,dtype='f4');history=np.linspace(-.2,.4,4,dtype='f4');times=schedule()
        workspace=np.zeros(44,'f4');out=np.full(4,-999.,'f4')
        r=Request(model,update,NOISE(('audio_noise_counter_gaussian_f32',self.update[3])),
            cond.ctypes.data_as(P),history.ctypes.data_as(P),times.ctypes.data_as(P),4,6,4,10,2.,.25,.1,731,stream,
            workspace.ctypes.data_as(P),workspace.size,out.ctypes.data_as(P),out.size,CANCEL(),None)
        return r,(cond,history,times,workspace,out)
    def expected(self,arrays,stream=0,cfg=2.):
        cond,hist,times,_,_=arrays;noise=noise_reference(44,731,stream).reshape(11,4);y=noise[0].copy()
        for step in range(10):
            inp={'latent':y[None,:],'history':hist[None,:],'condition':cond[None,:],'time':times[step:step+1]}
            pred=model_reference(inp,self.w,1,2)[-1]
            inp['condition']=np.zeros_like(inp['condition']);null=model_reference(inp,self.w,1,2)[-1]
            dt=np.float32(times[step+1]-times[step]);velocity=(pred+(pred-null)*np.float32(cfg)).astype('f4')
            amp=np.float32(np.float32(.25)*np.sqrt(np.float32(.1)))*np.sqrt(dt)
            y=((y+velocity*dt).astype('f4')+(noise[step+1]*amp).astype('f4')).astype('f4')
        return y
    def test_generated_sampling_math_and_repeat(self):
        first=None
        for stream in (0,1,0):
            r,arrays=self.request(stream);self.assertEqual(self.driver(C.byref(r)),0)
            np.testing.assert_allclose(arrays[-1],self.expected(arrays,stream),atol=2e-5,rtol=2e-5)
            if first is None:first=arrays[-1].copy()
            elif stream==0:np.testing.assert_array_equal(first,arrays[-1])
            else:self.assertGreater(abs(first-arrays[-1]).max(),.01)
            self.assertGreater(abs(arrays[-1]-self.expected(arrays,stream,cfg=0)).max(),1e-5)
    def test_distinct_concurrent_native_arenas(self):
        from concurrent.futures import ThreadPoolExecutor
        def run(stream):
            r,a=self.request(stream)
            def clone(source):
                raw=(C.c_uint8*(len(source)+63))()
                aligned=(C.c_uint8*len(source)).from_buffer(raw,(-C.addressof(raw))&63)
                C.memmove(C.addressof(aligned),C.addressof(source),len(source));return aligned
            model=clone(self.model[1]);update=clone(self.update[1])
            r.model.arena=C.cast(model,U);r.update.arena=C.cast(update,U)
            status=self.driver(C.byref(r))
            self.assertEqual(status,0)
            np.testing.assert_allclose(a[-1],self.expected(a,stream),atol=2e-5,rtol=2e-5)
            return a[-1].copy()
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,[0,1,0,1]))
        np.testing.assert_array_equal(results[0],results[2]);np.testing.assert_array_equal(results[1],results[3])

    def test_host_callback_exception_is_not_swallowed(self):
        from version.v8.tts.native_guided_flow import NativeGuidedFlow
        with tempfile.TemporaryDirectory() as d:
            runner=NativeGuidedFlow(self.model,self.g,self.root/'model',Path(d),times=schedule())
            _,arrays=self.request()
            def broken():raise ValueError('cancel callback failed')
            with self.assertRaisesRegex(ValueError,'cancel callback failed'):
                runner.run(arrays[0],arrays[1],cfg=2.,sigma=.25,temperature=.1,should_cancel=broken)
            output=runner.run(arrays[0],arrays[1],cfg=2.,sigma=.25,temperature=.1)
            np.testing.assert_allclose(output,self.expected(arrays),atol=2e-5,rtol=2e-5)

    def test_rejection_cancellation_failure_and_recovery(self):
        for field,value in [('workspace_elements',43),('steps',0),('steps',65),('output_elements',3),('temperature',float('nan'))]:
            r,a=self.request();setattr(r,field,value);self.assertNotEqual(self.driver(C.byref(r)),0)
            self.assertTrue((a[-1]==-999).all())
        r,a=self.request();r.model.prediction=r.model.latent
        self.assertEqual(self.driver(C.byref(r)),-1);self.assertTrue((a[-1]==-999).all())
        for boundary in (1,2,3,4,5,31,32):
            r,a=self.request();calls=[0]
            def cancel(context):calls[0]+=1;return int(calls[0]>=boundary)
            cb=CANCEL(cancel);r.should_cancel=cb
            status=self.driver(C.byref(r))
            self.assertEqual(status,-7);self.assertTrue((a[-1]==-999).all())
            r.should_cancel=CANCEL();self.assertEqual(self.driver(C.byref(r)),0)
            np.testing.assert_allclose(a[-1],self.expected(a),atol=2e-5,rtol=2e-5)
        r,a=self.request();original=r.model.entry
        failed=ENTRY(lambda arena,size:-99);r.model.entry=failed
        self.assertEqual(self.driver(C.byref(r)),-2);self.assertTrue((a[-1]==-999).all())
        r.model.entry=ENTRY(('ck_ming3_dit_fp32_bounded',self.model[3]));self.assertEqual(self.driver(C.byref(r)),0)

if __name__=='__main__':unittest.main()
