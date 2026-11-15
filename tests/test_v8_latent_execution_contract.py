"""Bind the declared runtime pattern to real compiled components; fail closed."""
import copy,ctypes as C,json
from pathlib import Path
import numpy as np
from tests import test_v8_ming3_cached_backbone as fixtures
from version.v8.scripts.checked_execution_exports_v8 import load_exports,resolve_exports
from version.v8.scripts.latent_execution_contract_v8 import resolve_latent_execution
from version.v8.tts.ming3_execution_profile import load_profile
from version.v8.tts.build_ming3_runtime_audio_decoder_circuit import build_circuit as decoder
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph
from version.v8.tts.native_latent_loop import bind_latent_request
from version.v8.tts.native_guided_flow import CANCEL

class ExecutionContractTests(fixtures.CachedNativeLoopTests):
    @classmethod
    def setUpClass(cls):
        super().setUpClass();folder=cls.root/'decoder';folder.mkdir()
        graph=decoder(capacity=5,width=8,heads=2,kv_heads=1,depth=1,inner=12,latent=4,n_fft=8,window=3)
        rng=np.random.default_rng(8);weights={k:rng.normal(0,.1,s).astype('f4') for k,s in declared_f32_weight_shapes(graph).items()}
        cls.decoder=compile_declared_f32_graph(folder,graph,weights,openmp=True,c_standard='gnu11',provider_warnings='report')
        cls.bundles={'backbone':cls.backbone,'aggregator':cls.aggregator,'dit':cls.model,'update':cls.loop_bundle,'decoder':cls.decoder}
        dirs={'backbone':'backbone','aggregator':'aggregator','dit':'model','update':'loop','decoder':'decoder'}
        cls.exports={name:load_exports(cls.root/dirs[name],bundle[3]) for name,bundle in cls.bundles.items()}
    def test_resolved_binding_executes_without_host_geometry(self):
        descriptor=load_profile();plan=resolve_latent_execution(descriptor,self.exports)
        self.assertEqual((plan['width'],plan['latent'],plan['noise_elements']),(6,4,44))
        request,latents,valid,keepalive=bind_latent_request(plan,self.bundles,[1,1,1],731,CANCEL())
        self.assertEqual(self.loop(C.byref(request)),0);self.assertEqual(valid.value,5)
        self.assertTrue(np.isfinite(latents).all())
    def test_session_uses_named_decoder_and_native_sample_count(self):
        from version.v8.tts.run_ming3_native_v8 import NativeMingSession,CancelledSynthesis
        session=object.__new__(NativeMingSession)
        session.plan=resolve_latent_execution(load_profile(),self.exports)
        session.bundles=self.bundles;session.driver=self.loop
        first,measure,latent=session._synthesize([1,1,1],731,None)
        self.assertEqual(measure['samples'],len(first))
        self.assertEqual(measure['samples'],self.exports['decoder']['ports']['waveform']['elements'])
        self.assertTrue(np.isfinite(first).all())
        with self.assertRaises(CancelledSynthesis):session._synthesize([1,1,1],732,lambda:True)
        repeat,_,again=session._synthesize([1,1,1],731,None)
        np.testing.assert_array_equal(first,repeat);np.testing.assert_array_equal(latent,again)

    def test_mismatches_reject_before_native_execution(self):
        descriptor=load_profile();before=bytes(self.backbone[1])
        for mutation in ('dimension','selection','noise','count','stop','schedule','entry'):
            d=copy.deepcopy(descriptor);e=copy.deepcopy(self.exports)
            if mutation=='dimension':e['dit']['ports']['condition']['shape']=[1,7]
            elif mutation=='selection':e['dit']['ports']['prediction']['slice']['start']=1
            elif mutation=='noise':e['update']['provider_entries']['noise']['operation_interface']='wrong.v1'
            elif mutation=='count':e['decoder']['runtime_length_outputs']['valid_samples']['capacity']+=1
            elif mutation=='stop':d['policy']['stop']='stop0_gt_continue1'
            elif mutation=='entry':d['components']['backbone']['entries']['prefill']='embed'
            elif mutation=='schedule':d['sampling']['times'][3]=float('nan')
            with self.assertRaises(ValueError,msg=mutation):resolve_latent_execution(d,e)
            self.assertEqual(bytes(self.backbone[1]),before)
        plan=resolve_latent_execution(descriptor,self.exports)
        wrong=dict(self.bundles);wrong['backbone']=self.aggregator
        with self.assertRaises(ValueError):bind_latent_request(plan,wrong,[1],731,CANCEL())
    def test_compiler_rejects_invalid_named_tensor_exports(self):
        folder=self.root/'model'
        graph=json.loads((folder/'circuit.json').read_text())
        layout=json.loads((folder/'layout.json').read_text());calls=json.loads((folder/'call.json').read_text())
        root=Path(__file__).resolve().parents[1]
        providers={op['call_abi']['kernel_id']:json.loads((root/'version/v8/kernel_maps'/
            (op['call_abi']['kernel_id']+'.json')).read_text()) for op in calls['operations']}
        for mutation in ('shape','slice','entry'):
            g=copy.deepcopy(graph)
            if mutation=='shape':g['execution_exports']['ports']['condition']['shape']=[1,999]
            elif mutation=='slice':g['execution_exports']['ports']['prediction']['slice']['start']=999
            else:g['execution_exports']['entries']['run']='not_a_generated_function'
            with self.assertRaises(ValueError,msg=mutation):resolve_exports(g,layout,calls,providers)

    def test_stale_compiled_export_file_is_rejected(self):
        path=self.root/'model'/'execution-bindings.json';old=path.read_text();document=json.loads(old)
        document['ports']['prediction']['offset']+=4
        try:
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError,'bindings differ'):load_exports(path.parent,self.model[3])
        finally:path.write_text(old)

if __name__=='__main__':
    import unittest
    unittest.main()
