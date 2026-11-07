"""Projection/attention/residual training via normal backward lowering and AdamW.

Synthetic nonzero features, not full Clef head or backbone training.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from tests import test_v8_selective_training as base
from tests.v8_training_test_helpers import certify_training
ROOT=Path(__file__).resolve().parents[1]
REGISTRY=deepcopy(base.REGISTRY)
REGISTRY['kernels'].extend(json.loads((ROOT/'version/v8/kernel_maps'/f'{n}.json').read_text()) for n in ('attention_cross_token_major_checked_f32','attention_cross_backward_checked_f32'))
RULES=json.loads((ROOT/'version/v8/scripts/grad_rules_v8.json').read_text())

def fixture(frozen_key=False):
    g=base.fixture(); g['ops']=g['ops'][:2]
    for n in ('weight.bridge','act.bridge','weight.score','act.score'): g['tensors'].pop(n,None)
    g['config']['train_batch_policy']='fixed'
    def ref(tid,shape,kind='activation',trainable=True):
        import math
        item=dict(tensor=tid,shape=shape,numel=math.prod(shape),dtype='fp32',kind=kind,requires_grad=trainable)
        g['tensors'][tid]=dict(item,persistent=kind=='weight');return item
    def projection(name,x,trainable=True):
        w=dict(ref('weight.'+name,[4,4],'weight',trainable),name=name); y=ref('act.'+name,[4,4])
        g['ops'].append(dict(op_id=len(g['ops']),op='logits' if name=='score' else 'projection',kernel_id='gemm_blocked_serial',phase='forward',section='body',layer=0,requires_grad=True,grad_rule='matmul_backward',dataflow=dict(inputs={'input':x},outputs={'y':y}),weights={'W':w},save_for_backward={'input':x,'W':w}))
        return y
    x=g['ops'][1]['dataflow']['outputs']['y']
    q=projection('query',x);k=projection('key',x,not frozen_key);v=projection('value',x)
    y=ref('act.attention',[4,4]); scratch=ref('aux.attention',[20],'aux',False)
    inputs=dict(query=q,key=k,value=v)
    g['ops'].append(dict(op_id=len(g['ops']),op='attention_cross',kernel_id='attention_cross_token_major_checked_f32',phase='forward',section='body',layer=0,requires_grad=True,map_call=True,
        params=dict(call_constants=dict(query_tokens=4,memory_tokens=4,heads=2,head_dim=2,query_elements=16,key_elements=16,value_elements=16,output_elements=16,scratch_elements=20,backward_scratch_elements=56)),
        dataflow=dict(inputs=inputs,outputs=dict(output=y,scratch=scratch)),weights={},save_for_backward=inputs,grad_rule='attention_cross_backward'))
    joined=ref('act.join',[4,4])
    g['ops'].append(dict(op_id=len(g['ops']),op='residual_add',kernel_id='ck_residual_add_token_major',phase='forward',section='body',layer=0,requires_grad=True,grad_rule='residual_add_backward',dataflow=dict(inputs=dict(a=x,b=y),outputs=dict(out=joined)),weights={}))
    projection('score',joined)
    return g

def lower(g):
    return base.backward.synthesize_ir2_backward(g,REGISTRY,base.BINDINGS,RULES,strict=True,allow_partial=False,checkpoint_policy='none')

def generate(g,registry=REGISTRY):
    ir=lower(g); manifest={'config':ir['config']}; layout=base.planner.build_layout(ir,manifest,64,strict=True)
    return base.codegen.generate_c(ir,registry,manifest=manifest,layout=layout)

class AttentionTrainingContracts(unittest.TestCase):
    def test_generated_map_calls_and_scratch(self):
        g=fixture(); old=deepcopy(g); source,report=generate(g)
        self.assertEqual(g,old);self.assertEqual(report['skipped_ops'],[])
        self.assertIn('#if CK_NUM_TOKENS != 4',source)
        self.assertIn('if (ck_attention_cross_backward_f32(',source)
        self.assertIn('if (ck_attention_cross_token_major_checked_f32(',source)
        ir=lower(g)
        scratch=[t for t in ir['tensors'].values() if t.get('numel')==56]
        self.assertEqual(len(scratch),1)
        self.assertNotIn('key',generate(fixture(True))[1]['parameter_gradient_order'])

    def test_invalid_bounds_and_contracts_reject_before_codegen(self):
        for key,value in (('backward_scratch_elements',0),('backward_scratch_elements',None),('backward_scratch_elements',2**63),('query_elements',17),('query_elements',15),('query_tokens',5),('heads',3)):
            with self.subTest(key=key,value=value):
                g=fixture();g['ops'][5]['params']['call_constants'][key]=value
                with self.assertRaises((ValueError,RuntimeError)):generate(g)
        g=fixture();g['config'].pop('train_batch_policy')
        with self.assertRaisesRegex(ValueError,'fixed training batch'):generate(g)
        for status in (None,'positive_success'):
            r=deepcopy(REGISTRY);r['kernels'][-1]['training_contract']['return_status']=status
            with self.assertRaises(ValueError):generate(fixture(),r)

class AttentionTrainingNative(unittest.TestCase):
    def test_gradients_adamw_resume_export_and_wrong_edge(self):
        try:
            import numpy as np
            import torch
        except ImportError as exc:self.skipTest('PyTorch oracle unavailable: '+str(exc))
        rng=np.random.default_rng(837)
        values=dict(features=np.eye(4,dtype='f'),**{n:rng.normal(0,.3,(4,4)).astype('f') for n in ('up','query','key','value','score')})
        def forward(p):
            x=p['features']@p['up'].T
            q,k,v=[(x@p[n].T).reshape(4,2,2).transpose(0,1) for n in ('query','key','value')]
            y=(torch.softmax(q@k.transpose(-1,-2)/(2**.5),-1)@v).transpose(0,1).reshape(4,4)
            return (x+y)@p['score'].T
        for frozen in (False,True):
            with self.subTest(frozen_key=frozen):
                trainable=set(values)-{'features'}-({'key'} if frozen else set())
                certify_training(self,lower(fixture(frozen)),REGISTRY,values,trainable,forward)
        wrong=fixture();wrong['ops'][5]['save_for_backward']=dict(wrong['ops'][5]['save_for_backward'])
        wrong['ops'][5]['save_for_backward']['key']=wrong['ops'][5]['save_for_backward']['value']
        with self.assertRaisesRegex(AssertionError,'Not equal to tolerance'):
            certify_training(self,lower(wrong),REGISTRY,values,set(values)-{'features'},forward)

class AttentionTrainingRecovery(unittest.TestCase):
    def test_checked_failure_discards_partial_window_then_matches_clean_run(self):
        import ctypes as C
        import numpy as np
        import subprocess
        import tempfile
        from tests.v8_training_test_helpers import ROOT as root
        engine=root/'build/libckernel_engine.so'
        if not engine.exists():self.skipTest('native engine is not built')
        registry=deepcopy(REGISTRY)
        mapping=registry['kernels'][-1]
        original=mapping['impl']['c_declaration']
        mapping['impl']['function']='test_attention_backward'
        mapping['impl']['c_declaration']=original.replace('ck_attention_cross_backward_f32','test_attention_backward')
        source,report=generate(fixture(),registry)
        # Test-only failure injection AFTER score backward, before attention VJP.
        params=[p['name'] for p in mapping['call_abi']['params']]
        wrapper='#include <stddef.h>\n'+original+'\nstatic int fail_next;\nvoid test_fail_next(void) { fail_next=1; }\n'
        wrapper+=mapping['impl']['c_declaration'][:-1]+' { if (fail_next) { fail_next=0; return -77; } return ck_attention_cross_backward_f32('+', '.join(params)+'); }\n'
        fp=C.POINTER(C.c_float);ip=C.POINTER(C.c_int32)
        ptr=lambda a:a.ctypes.data_as(fp)
        rng=np.random.default_rng(837)
        values={n:rng.normal(0,.3,(4,4)).astype('f') for n in report['init_weight_order']}
        values['features']=np.eye(4,dtype='f')
        packed=np.concatenate([values[n].ravel() for n in report['init_weight_order']])
        sizes=np.array(report['init_weight_numel'],dtype='int32');ids=np.arange(4,dtype='int32');idptr=ids.ctypes.data_as(ip)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'train.c').write_text(source);(path/'fault.c').write_text(wrapper)
            subprocess.run(['cc','-O2','-shared','-fPIC','-DCK_NUM_TOKENS=4','-DCK_GRAD_ACCUM_STEPS=2','-I'+str(root/'include'),str(path/'train.c'),str(path/'fault.c'),str(engine),'-Wl,-rpath,'+str(engine.parent),'-lm','-o',str(path/'train.so')],check=True)
            lib=C.CDLL(str(path/'train.so'))
            lib.ck_train_init.argtypes=[fp,ip,C.c_int];lib.ck_train_step_ex.argtypes=[ip,ip,C.c_int,fp,C.c_float]
            def init(data=packed):self.assertEqual(lib.ck_train_init(ptr(data),sizes.ctypes.data_as(ip),len(sizes)),len(sizes))
            def snapshot(kind):
                count=report[kind+'_snapshot_floats'];a=np.empty(count,'f')
                fn=getattr(lib,'ck_train_export_'+kind+'_snapshot');fn.argtypes=[fp,C.c_int]
                self.assertEqual(fn(ptr(a),count),count);return a
            def step(a=idptr,b=idptr,n=4):
                loss=C.c_float(987)
                rc=lib.ck_train_step_ex(a,b,n,C.byref(loss),.03)
                return rc,loss.value
            try:
                init();initial=snapshot('weight'); moments=snapshot('optimizer_state')
                for a,b,n in ((None,idptr,4),(idptr,None,4),(idptr,idptr,3),(idptr,idptr,5)):
                    rc,loss=step(a,b,n);self.assertLess(rc,0);self.assertEqual(loss,987)
                for bad in (-1,4):
                    arr=ids.copy();arr[1]=bad
                    for a,b in ((arr.ctypes.data_as(ip),idptr),(idptr,arr.ctypes.data_as(ip))):
                        rc,loss=step(a,b);self.assertLess(rc,0);self.assertEqual(loss,987)
                np.testing.assert_array_equal(snapshot('weight'),initial)
                self.assertGreater(step()[0],0);self.assertEqual(lib.ck_train_get_opt_step(),0)
                self.assertGreater(np.max(np.abs(snapshot('parameter_gradient'))),0)
                lib.test_fail_next();rc,loss=step();self.assertLess(rc,0);self.assertEqual(loss,987)
                self.assertEqual(lib.ck_train_get_opt_step(),0)
                np.testing.assert_array_equal(snapshot('weight'),initial)
                np.testing.assert_array_equal(snapshot('optimizer_state'),moments)
                np.testing.assert_array_equal(snapshot('parameter_gradient'),0)
                self.assertGreater(step()[0],0);self.assertEqual(lib.ck_train_get_opt_step(),0)
                self.assertGreater(step()[0],0);self.assertEqual(lib.ck_train_get_opt_step(),1)
                recovered=[snapshot(k) for k in ('weight','optimizer_state')]
                lib.ck_train_free();init()
                self.assertGreater(step()[0],0);self.assertGreater(step()[0],0)
                for actual,kind in zip(recovered,('weight','optimizer_state')):np.testing.assert_array_equal(actual,snapshot(kind))
                # Real nonfinite rejection in forward also cannot update weights.
                lib.ck_train_free();bad=packed.copy();bad[0]=np.nan;init(bad)
                before=snapshot('weight');rc,loss=step();self.assertLess(rc,0);self.assertEqual(loss,987)
                self.assertEqual(lib.ck_train_get_opt_step(),0)
                self.assertEqual(snapshot('weight').tobytes(),before.tobytes())
                np.testing.assert_array_equal(snapshot('optimizer_state'),0)
                lib.ck_train_free();init();self.assertGreater(step()[0],0)
            finally:lib.ck_train_free()

if __name__=='__main__':unittest.main(verbosity=2)
