"""Automatic span/grouped pooling backward with typed per-batch metadata.

Shared generated trainer, nonzero synthetic features, not full Clef training.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from tests import test_v8_gelu_training as gelu
from tests import test_v8_selective_training as base
from tests.v8_training_test_helpers import certify_training
ROOT=Path(__file__).resolve().parents[1]
NAMES=('token_span_mean_checked_f32','token_span_mean_backward_checked_f32',
       'grouped_softmax_weighted_rows_checked_f32','grouped_weighted_rows_backward_checked_f32')
REGISTRY=deepcopy(gelu.REGISTRY)
REGISTRY['kernels'].extend(json.loads((ROOT/'version/v8/kernel_maps'/f'{n}.json').read_text()) for n in NAMES)
RULES=json.loads((ROOT/'version/v8/scripts/grad_rules_v8.json').read_text())
SPANS=((0,1),(0,2),(1,2),(2,3),(2,4),(3,4))
COUNTS=(2,1,2,1)


def fixture():
    g=gelu.fixture()
    g['config']['training_integer_inputs']=['input.token_spans','input.group_counts']
    def ref(tid,shape,kind='activation',dtype='fp32',grad=True):
        import math
        r=dict(tensor=tid,shape=shape,numel=math.prod(shape),dtype=dtype,kind=kind,requires_grad=grad,persistent=False)
        g['tensors'][tid]=dict(r);return r
    spans=ref('input.token_spans',[6,2],'input','i32',False)
    counts=ref('input.group_counts',[4],'input','i32',False)
    x=g['ops'][1]['dataflow']['outputs']['y']
    pooled=ref('act.span',[6,4])
    span=dict(op_id=2,op='token_span_mean',kernel_id=NAMES[0],phase='forward',section='body',layer=0,
              requires_grad=True,map_call=True,weights={},grad_rule='token_span_mean_checked_backward',
              dataflow=dict(inputs=dict(input=x,spans=spans),outputs=dict(output=pooled)),save_for_backward=dict(spans=spans),
              params=dict(call_constants=dict(input_elements=16,input_stride=4,token_capacity=4,valid_tokens=4,channels=4,
                span_elements=12,span_count=6,output_elements=24,output_stride=4,d_output_elements=24,d_output_stride=4,
                d_input_elements=16,d_input_stride=4,scratch_elements=16)))
    activation=g['ops'][2]
    activation['dataflow']['inputs']['input']=pooled
    activation['save_for_backward']['input']=pooled
    activation['dataflow']['outputs']['output']=ref('act.gelu',[6,4])
    activation['dataflow']['outputs']['scratch']=ref('aux.gelu',[24],grad=False)
    c=activation['params']['call_constants']
    for k in c:
        if k.endswith('elements'):c[k]=24
        elif k.endswith('rows'):c[k]=6
    y=ref('act.group',[4,4])
    inputs=dict(options=activation['dataflow']['outputs']['output'],fields=x,group_counts=counts)
    grouped=dict(op_id=4,op='grouped_weighted',kernel_id=NAMES[2],phase='forward',section='body',layer=0,
                 requires_grad=True,map_call=True,weights={},grad_rule='grouped_weighted_checked_backward',
                 dataflow=dict(inputs=inputs,outputs=dict(output=y,scratch=ref('aux.group',[84],grad=False))),save_for_backward=inputs,
                 params=dict(call_constants=dict(option_elements=24,option_stride=4,field_elements=16,field_stride=4,
                    count_elements=4,output_elements=16,output_stride=4,scratch_elements=84,groups=4,option_rows=6,channels=4,
                    d_output_elements=16,d_output_stride=4,d_options_elements=24,d_options_stride=4,d_fields_elements=16,d_fields_stride=4)))
    g['ops'][3]['dataflow']['inputs']['input']=y
    g['ops'][3]['save_for_backward']['input']=y
    g['ops'].insert(2,span);g['ops'].insert(4,grouped)
    for i,o in enumerate(g['ops']):o['op_id']=i
    return g


def lower(g):
    return base.backward.synthesize_ir2_backward(g,REGISTRY,base.BINDINGS,RULES,strict=True,allow_partial=False,checkpoint_policy='none')


def generate(g,registry=REGISTRY):
    ir=lower(g);manifest=dict(config=ir['config']);layout=base.planner.build_layout(ir,manifest,64,strict=True)
    return base.codegen.generate_c(ir,registry,manifest=manifest,layout=layout)


class PoolingContracts(unittest.TestCase):
    def test_named_integer_inputs_and_automatic_branch_gradients(self):
        g=fixture();original=deepcopy(g);source,report=generate(g)
        self.assertEqual(g,original);self.assertEqual(report['skipped_ops'],[])
        self.assertEqual(report['integer_input_count'],16)
        self.assertEqual([i['tensor'] for i in report['integer_inputs']],g['config']['training_integer_inputs'])
        self.assertEqual(report['parameter_gradient_order'],['score','up'])
        self.assertIn('ck_token_span_mean_backward_f32(',source)
        self.assertIn('ck_grouped_weighted_rows_backward_f32(',source)
        self.assertNotIn('memcpy(input_token_spans, token_ids',source)
        ir=lower(g)
        contributions=[o for o in ir['backward'] if o.get('op')=='grad_accumulate' and any(v.get('tensor')=='grad.act.act.up' for v in o.get('dataflow',{}).get('outputs',{}).values())]
        self.assertEqual(len(contributions),2)
        self.assertFalse(any('grad.input.token_spans' in t or 'grad.input.group_counts' in t for t in ir['tensors']))

    def test_invalid_input_registry_and_integer_dtype_fail_closed(self):
        for mutation in ('missing','duplicate','unknown','trainable','dtype','reserved','dynamic'):
            g=fixture()
            if mutation=='missing':g['config'].pop('training_integer_inputs')
            elif mutation=='duplicate':g['config']['training_integer_inputs']*=2
            elif mutation=='unknown':g['config']['training_integer_inputs'].append('input.unknown')
            elif mutation=='trainable':g['tensors']['input.token_spans']['requires_grad']=True
            elif mutation=='dtype':g['tensors']['input.token_spans']['dtype']='fp32'
            elif mutation=='reserved':g['config']['training_integer_inputs'].append('input.token_ids')
            else:g['config'].pop('train_batch_policy')
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,RuntimeError)):generate(g)

    def test_shapes_strides_capacities_and_valid_extent_rejected(self):
        for op,key,value in ((2,'span_elements',11),(2,'input_stride',3),(2,'valid_tokens',3),(4,'count_elements',3),(4,'option_stride',5)):
            g=fixture();g['ops'][op]['params']['call_constants'][key]=value
            with self.subTest(key=key),self.assertRaises((ValueError,RuntimeError)):generate(g)
        for tid in ('input.token_spans','input.group_counts'):
            g=fixture();g['tensors'][tid]['shape']=[1,g['tensors'][tid]['numel']]
            with self.assertRaises(ValueError):generate(g)

    def test_integer_contract_and_order_affect_runtime_identity(self):
        _,a=generate(fixture());g=fixture();g['config']['training_integer_inputs'].reverse();_,b=generate(g)
        self.assertNotEqual(a['runtime_contract_sha256'],b['runtime_contract_sha256'])
        for i in (0,1,2,3):
            reg=deepcopy(REGISTRY);m=reg['kernels'][-4+i]
            ptr=next(p for p in m['call_abi']['params'] if p['name'] in ('spans','group_counts'))
            m['impl']['c_declaration']=m['impl']['c_declaration'].replace('const int32_t *','int32_t *')
            with self.assertRaises(ValueError):generate(fixture(),reg)


class PoolingNative(unittest.TestCase):
    def test_gradients_updates_overlap_branch_join_and_reload(self):
        try:
            import numpy as np
            import torch
        except ImportError as exc:self.skipTest('independent PyTorch unavailable: '+str(exc))
        rng=np.random.default_rng(837)
        values=dict(features=np.eye(4,dtype='f'),bridge=np.eye(4,dtype='f')+rng.normal(0,.1,(4,4)).astype('f'),
                    up=rng.normal(0,.2,(4,4)).astype('f'),score=rng.normal(0,.2,(4,4)).astype('f'))
        def forward(p):
            x=p['features']@p['up'].T
            options=torch.nn.functional.gelu(torch.stack([x[a:b].mean(0) for a,b in SPANS]),approximate='none')
            result=[];start=0
            for q,n in enumerate(COUNTS):
                o=options[start:start+n];prob=torch.softmax(o@x[q]/2,dim=0)
                result.append((prob[:,None]*o).sum(0));start+=n
            return torch.stack(result)@p['bridge'].T@p['score'].T
        metadata=np.concatenate([np.array(SPANS,'i4').ravel(),np.array(COUNTS,'i4')])
        certify_training(self,lower(fixture()),REGISTRY,values,{'up','score'},forward,integer_inputs=metadata)
        wrong=fixture()
        wrong['ops'][4]['save_for_backward']=dict(wrong['ops'][4]['save_for_backward'])
        wrong['ops'][4]['save_for_backward']['fields']=wrong['ops'][4]['dataflow']['outputs']['output']
        with self.assertRaises(AssertionError):
            certify_training(self,lower(wrong),REGISTRY,values,{'up','score'},forward,integer_inputs=metadata)

class PoolingRecovery(unittest.TestCase):
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
        mapping['impl']['function']='test_grouped_backward'
        mapping['impl']['c_declaration']=original.replace('ck_grouped_weighted_rows_backward_f32','test_grouped_backward')
        source,report=generate(fixture(),registry)
        # Test-only failure injection AFTER score backward, before grouped VJP.
        params=[p['name'] for p in mapping['call_abi']['params']]
        wrapper='#include <stddef.h>\n#include <stdint.h>\n'+original+'\nstatic int fail_next;\nvoid test_fail_next(void) { fail_next=1; }\n'
        wrapper+=mapping['impl']['c_declaration'][:-1]+' { if (fail_next) { fail_next=0; return -77; } return ck_grouped_weighted_rows_backward_f32('+', '.join(params)+'); }\n'
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
            lib.ck_train_step_with_i32_ex.argtypes=[ip,ip,C.c_int,ip,C.c_size_t,fp,C.c_float]
            lib.ck_train_set_batch_with_i32_ex.argtypes=[ip,ip,C.c_int,ip,C.c_size_t]
            metadata=np.concatenate([np.array(SPANS,'i4').ravel(),np.array(COUNTS,'i4')])
            def init(data=packed):self.assertEqual(lib.ck_train_init(ptr(data),sizes.ctypes.data_as(ip),len(sizes)),len(sizes))
            def snapshot(kind):
                count=report[kind+'_snapshot_floats'];a=np.empty(count,'f')
                fn=getattr(lib,'ck_train_export_'+kind+'_snapshot');fn.argtypes=[fp,C.c_int]
                self.assertEqual(fn(ptr(a),count),count);return a
            def step(a=idptr,b=idptr,n=4,extra=None,count=None):
                if extra is None:extra=metadata
                if count is None:count=extra.size
                loss=C.c_float(987)
                rc=lib.ck_train_step_with_i32_ex(a,b,n,extra.ctypes.data_as(ip),count,C.byref(loss),.03)
                return rc,loss.value
            try:
                init();initial=snapshot('weight'); moments=snapshot('optimizer_state')
                for a,b,n in ((None,idptr,4),(idptr,None,4),(idptr,idptr,3),(idptr,idptr,5)):
                    rc,loss=step(a,b,n);self.assertLess(rc,0);self.assertEqual(loss,987)
                for bad in (-1,4):
                    arr=ids.copy();arr[1]=bad
                    for a,b in ((arr.ctypes.data_as(ip),idptr),(idptr,arr.ctypes.data_as(ip))):
                        rc,loss=step(a,b);self.assertLess(rc,0);self.assertEqual(loss,987)
                for extra,count in ((metadata,15),(metadata,17),(np.zeros(0,'i4'),0)):
                    rc,loss=step(extra=extra,count=count);self.assertLess(rc,0);self.assertEqual(loss,987)
                loss=C.c_float(987)
                self.assertLess(lib.ck_train_step_ex(idptr,idptr,4,C.byref(loss),.03),0)
                self.assertEqual(loss.value,987)
                self.assertLess(lib.ck_train_step_with_i32_ex(idptr,idptr,4,None,16,C.byref(loss),.03),0)
                misaligned=np.zeros(65,'u1')[1:].view('i4')
                self.assertLess(step(extra=misaligned)[0],0)
                for index,value in ((0,-1),(1,0),(1,5),(12,0),(12,3)):
                    badmeta=metadata.copy();badmeta[index]=value
                    rc,loss=step(extra=badmeta);self.assertLess(rc,0);self.assertEqual(loss,987)
                    self.assertEqual(lib.ck_train_get_opt_step(),0)
                    np.testing.assert_array_equal(snapshot('optimizer_state'),moments)
                np.testing.assert_array_equal(snapshot('weight'),initial)
                self.assertGreater(step()[0],0);self.assertEqual(lib.ck_train_get_opt_step(),0)
                self.assertGreater(np.max(np.abs(snapshot('parameter_gradient'))),0)
                partial=snapshot('parameter_gradient')
                rc,loss=step(count=15);self.assertLess(rc,0);self.assertEqual(loss,987)
                np.testing.assert_array_equal(snapshot('parameter_gradient'),partial)
                np.testing.assert_array_equal(snapshot('weight'),initial)
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
                lib.ck_train_free();init()
                def inference(extra):
                    self.assertGreater(lib.ck_train_set_batch_with_i32_ex(idptr,idptr,4,extra.ctypes.data_as(ip),extra.size),0)
                    self.assertGreater(lib.ck_train_forward_step(),0)
                    return snapshot('activation')
                baseline=inference(metadata)
                other=metadata.copy();other[:2]=[1,3];other[12:]=[1,2,1,2]
                changed=inference(other);self.assertFalse(np.array_equal(changed,baseline))
                np.testing.assert_array_equal(inference(metadata),baseline)
                copied=metadata.copy()
                self.assertGreater(lib.ck_train_set_batch_with_i32_ex(idptr,idptr,4,copied.ctypes.data_as(ip),copied.size),0)
                copied[:]=-77
                self.assertGreater(lib.ck_train_forward_step(),0)
                np.testing.assert_array_equal(snapshot('activation'),baseline)
                # Legacy entry may not reuse previous integer metadata silently.
                self.assertLess(lib.ck_train_step_ex(idptr,idptr,4,C.byref(C.c_float()),.03),0)
                np.testing.assert_array_equal(snapshot('weight'),initial)
                lib.ck_train_free();init()
                np.testing.assert_array_equal(inference(other),changed)
                guarded=np.concatenate([metadata,np.array([1234567],'i4')])
                np.testing.assert_array_equal(inference(guarded[:16]),baseline)
                self.assertEqual(guarded[-1],1234567)
                np.testing.assert_array_equal(guarded[:16],metadata)
            finally:lib.ck_train_free()



if __name__=='__main__':unittest.main(verbosity=2)
