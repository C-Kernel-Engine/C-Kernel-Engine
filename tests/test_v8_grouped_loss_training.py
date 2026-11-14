"""Grouped decision loss and indexed scorer through automatic native training."""
import ctypes as C
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from tests import test_v8_selective_training as base
from tests import test_v8_pooling_training as pooling
from tests.v8_training_test_helpers import certify_training
ROOT=Path(__file__).resolve().parents[1]
FP=C.POINTER(C.c_float);IP=C.POINTER(C.c_int32)
REGISTRY=deepcopy(pooling.REGISTRY)
REGISTRY['kernels'].extend(json.loads((ROOT/'version/v8/kernel_maps'/f'{n}.json').read_text()) for n in
    ('indexed_pair_features_checked_f32','indexed_pair_features_backward_checked_f32','grouped_cross_entropy_checked_f32'))
RULES=json.loads((ROOT/'version/v8/scripts/grad_rules_v8.json').read_text())


class GroupedLossKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.functions=[]
        for opt in ('O0','O3'):
            path=Path(cls.tmp.name)/(opt+'.so')
            subprocess.run(['cc','-std=c11','-'+opt,'-Wall','-Wextra','-Werror','-ffp-contract=off','-shared','-fPIC','-I'+str(ROOT/'include'),str(ROOT/'src/kernels/grouped_cross_entropy.c'),'-lm','-o',str(path)],check=True)
            lib=C.CDLL(str(path));fn=lib.ck_grouped_cross_entropy_f32
            fn.argtypes=[FP,C.c_size_t,IP,C.c_size_t,IP,C.c_size_t,FP,C.c_size_t,FP,C.c_size_t,FP,C.c_size_t,C.c_size_t,C.c_size_t]
            cls.functions.append(fn)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def arrays(self,z=(.2,.7,-.4,1.2),counts=(1,3),targets=(0,2)):
        return dict(logits=np.array(z,'f'),counts=np.array(counts,'i4'),targets=np.array(targets,'i4'),gradient=np.full(len(z)+2,77,'f'),loss=np.full(3,77,'f'),scratch=np.empty(len(z)+1,'f'))
    def args(self,a):
        p=lambda n:a[n].ctypes.data_as(IP if n in ('counts','targets') else FP)
        return [p('logits'),a['logits'].size,p('counts'),a['counts'].size,p('targets'),a['targets'].size,
                p('gradient'),a['logits'].size,p('loss'),1,p('scratch'),a['scratch'].size,len(a['counts']),len(a['logits'])]
    def test_ordered_oracle_padding_and_optimization(self):
        rng=np.random.default_rng(116)
        for counts in ((1,),(1,3),(2,1,4),(9,2)):
            z=rng.normal(0,2,sum(counts)).astype('f');labels=np.array([n-1 for n in counts],'i4');a=self.arrays(z,counts,labels)
            expected=[];loss=0.;start=0
            for n,y in zip(counts,labels):
                row=z[start:start+n].astype('d');m=float(max(row));exp=[math.exp(float(x)-m) for x in row];den=sum(exp)
                loss+=((m-float(row[y]))+math.log(den))/len(counts)
                expected.extend([(v/den-(i==y))/len(counts) for i,v in enumerate(exp)]);start+=n
            previous=None
            for fn in self.functions:
                self.assertEqual(fn(*self.args(a)),0)
                np.testing.assert_allclose(a['gradient'][:len(z)],np.array(expected,'f'),rtol=1e-6,atol=1e-8)
                self.assertAlmostEqual(float(a['loss'][0]),loss,delta=2e-7)
                np.testing.assert_array_equal(a['gradient'][len(z):],77);np.testing.assert_array_equal(a['loss'][1:],77)
                if previous is not None:self.assertEqual(previous,a['gradient'].tobytes()+a['loss'].tobytes())
                previous=a['gradient'].tobytes()+a['loss'].tobytes()
    def test_pytorch_and_finite_difference(self):
        try:import torch
        except ImportError as exc:self.skipTest('independent PyTorch unavailable: '+str(exc))
        a=self.arrays();z=torch.tensor(a['logits'],requires_grad=True)
        loss=(torch.nn.functional.cross_entropy(z[:1][None],torch.tensor([0]))+torch.nn.functional.cross_entropy(z[1:][None],torch.tensor([2])))/2
        loss.backward();self.assertEqual(self.functions[-1](*self.args(a)),0)
        np.testing.assert_allclose(a['gradient'][:4],z.grad.numpy(),atol=1e-7,rtol=1e-6)
        self.assertAlmostEqual(a['loss'][0],loss.item(),delta=1e-7)
        for i in range(4):
            vals=[]
            for delta in (-1e-3,1e-3):
                b=self.arrays();b['logits'][i]+=delta;self.assertEqual(self.functions[-1](*self.args(b)),0);vals.append(float(b['loss'][0]))
            self.assertAlmostEqual((vals[1]-vals[0])/2e-3,float(z.grad[i]),delta=3e-5)
    def test_rejects_invalid_metadata_and_nonfinite_without_publishing(self):
        for name,index,value in [('counts',0,0),('counts',1,2),('counts',1,4),('targets',0,1),('targets',1,-1),('logits',2,np.nan),('logits',2,np.inf)]:
            a=self.arrays();a[name][index]=value
            self.assertLess(self.functions[-1](*self.args(a)),0)
            np.testing.assert_array_equal(a['loss'],77);np.testing.assert_array_equal(a['gradient'],77)
        a=self.arrays((-np.finfo('f').max,np.finfo('f').max),(2,),(0,))
        self.assertLess(self.functions[-1](*self.args(a)),0);np.testing.assert_array_equal(a['loss'],77);np.testing.assert_array_equal(a['gradient'],77)
        a=self.arrays((1e30,1e30),(2,),(0,));self.assertEqual(self.functions[-1](*self.args(a)),0)
        self.assertAlmostEqual(a['loss'][0],math.log(2),delta=1e-7)
    def test_capacity_null_alignment_overlap_and_reuse(self):
        a=self.arrays();baseline=self.args(a)
        for i in (0,2,4,6,8,10):
            args=list(baseline);args[i]=None;self.assertLess(self.functions[-1](*args),0)
        for i in (1,3,5,7,9,11):
            args=list(baseline);args[i]-=1;self.assertLess(self.functions[-1](*args),0)
        storage=np.zeros(80,'u1');args=list(baseline);args[0]=C.cast(storage.ctypes.data+1,FP)
        self.assertLess(self.functions[-1](*args),0)
        for writable in (6,8,10):
            for earlier in (0,2,4,6,8):
                if earlier>=writable:continue
                args=list(baseline);args[writable]=C.cast(C.cast(args[earlier],C.c_void_p).value,FP)
                self.assertLess(self.functions[-1](*args),0)
        args=list(baseline);args[8]=C.cast(a['gradient'].ctypes.data+4,FP);self.assertLess(self.functions[-1](*args),0)
        np.testing.assert_array_equal(a['loss'],77);np.testing.assert_array_equal(a['gradient'],77)
        self.assertEqual(self.functions[-1](*baseline),0)
        clean=self.arrays();self.assertEqual(self.functions[-1](*self.args(clean)),0)
        np.testing.assert_array_equal(a['loss'],clean['loss']);np.testing.assert_array_equal(a['gradient'],clean['gradient'])


def fixture():
    g=base.fixture();g['ops']=g['ops'][:2]
    for n in ('weight.bridge','act.bridge','weight.score','act.score'):g['tensors'].pop(n,None)
    g['config'].update(train_batch_policy='fixed',training_integer_inputs=['input.spans','input.owners','input.counts','input.labels'])
    def ref(tid,shape,kind='activation',dtype='fp32',grad=True):
        r=dict(tensor=tid,shape=shape,numel=math.prod(shape),dtype=dtype,kind=kind,requires_grad=grad,persistent=kind=='weight');g['tensors'][tid]=dict(r);return r
    def op(kind,kernel,inputs,outputs,constants,rule,saved):
        g['ops'].append(dict(op_id=len(g['ops']),op=kind,kernel_id=kernel,phase='forward',section='body',layer=0,requires_grad=True,map_call=True,weights={},params=dict(call_constants=constants),dataflow=dict(inputs=inputs,outputs=outputs),grad_rule=rule,save_for_backward=saved))
    x=g['ops'][1]['dataflow']['outputs']['y'];spans=ref('input.spans',[2,2],'input','i32',False);owners=ref('input.owners',[4],'input','i32',False)
    ref('input.counts',[2],'input','i32',False);ref('input.labels',[2],'input','i32',False)
    q=ref('act.questions',[2,4])
    op('span','token_span_mean_checked_f32',dict(input=x,spans=spans),dict(output=q),dict(input_elements=16,input_stride=4,token_capacity=4,valid_tokens=4,channels=4,span_elements=4,span_count=2,output_elements=8,output_stride=4,d_output_elements=8,d_output_stride=4,d_input_elements=16,d_input_stride=4,scratch_elements=16),'token_span_mean_checked_backward',dict(spans=spans))
    pair=ref('act.pair',[4,16]);inputs=dict(left=q,right=x,indices=owners)
    op('pair','indexed_pair_features_checked_f32',inputs,dict(output=pair,scratch=ref('aux.pair',[64],grad=False)),dict(left_elements=8,right_elements=16,indices_elements=4,output_elements=64,scratch_elements=64,groups=2,rows=4,channels=4,d_output_elements=64,d_left_elements=8,d_right_elements=16),'indexed_pair_features_checked_backward',inputs)
    def project(name,source,inp,out):
        w=dict(ref('weight.'+name,[out,inp],'weight'),name=name)
        # A shared scalar score offset is unidentifiable under grouped CE.
        b=dict(ref('weight.'+name+'_bias',[out],'weight',grad=name!='score'),name=name+'_bias')
        y=ref('act.'+name,[4,out]);g['ops'].append(dict(op_id=len(g['ops']),op='projection',kernel_id='gemm_blocked_serial',phase='forward',section='body',layer=0,requires_grad=True,grad_rule='matmul_backward',dataflow=dict(inputs={'input':source},outputs={'y':y}),weights={'W':w,'bias':b},save_for_backward={'input':source,'W':w}));return y
    h=project('hidden',pair,16,4);a=ref('act.gelu',[4,4])
    op('gelu','gelu_rows_erf_checked_f32',dict(input=h),dict(output=a,scratch=ref('aux.gelu',[16],grad=False)),dict(gelu_input_elements=16,gelu_input_stride=4,gelu_output_elements=16,gelu_output_stride=4,gelu_scratch_elements=16,gelu_rows=4,gelu_channels=4,input_elements=16,input_stride=4,d_output_elements=16,d_output_stride=4,d_input_elements=16,d_input_stride=4,scratch_elements=16,rows=4,channels=4),'gelu_erf_checked_backward',dict(input=h))
    project('score',a,4,1)
    g['loss']=dict(kernel_id='grouped_cross_entropy_checked_f32',inputs=dict(logits='act.score',counts='input.counts',targets='input.labels'),params=dict(call_constants=dict(logits_elements=4,count_elements=2,target_elements=2,gradient_elements=4,loss_elements=1,scratch_elements=5,groups=2,options=4)),scratch=dict(scratch=5))
    return g


def lower(g,registry=REGISTRY):
    return base.backward.synthesize_ir2_backward(g,registry,base.BINDINGS,RULES,strict=True,allow_partial=False,checkpoint_policy='none')

def generate(g,registry=REGISTRY):
    ir=lower(g,registry);manifest=dict(config=ir['config']);layout=base.planner.build_layout(ir,manifest,64,strict=True)
    return base.codegen.generate_c(ir,registry,manifest=manifest,layout=layout)

class DeclaredLossContracts(unittest.TestCase):
    def test_map_loads_with_all_existing_consumers(self):
        import sys
        sys.path.insert(0,str(ROOT/'version/v8/scripts'))
        import resolve_numerical_execution_contracts_v8 as resolver
        result=resolver.load_kernel_capabilities()
        self.assertEqual(result['kernels']['grouped_cross_entropy_checked_f32']['interface_call_abi'],'validated')

    def test_declared_loss_seeds_non_decoder_scores(self):
        g=fixture();original=deepcopy(g);source,report=generate(g)
        self.assertEqual(g,original);self.assertEqual(report['skipped_ops'],[])
        self.assertIn('ck_grouped_cross_entropy_f32(',source)
        self.assertIn('g_loss_scalar[0] = aux_loss[0]',source)
        ir=lower(g);seed=next(o for o in ir['backward'] if o.get('role')=='loss_seed')
        self.assertEqual(seed['dataflow']['inputs']['logits']['tensor'],'act.score')
        self.assertEqual(ir['tensors'][seed['dataflow']['outputs']['d_logits']['tensor']]['numel'],4)
        self.assertFalse(any(o['op']=='logits' for o in g['ops']))
        self.assertEqual(report['evaluation_loss_ops'],1)
        evaluation=source.split('int ck_train_eval_step(float *loss_out) {',1)[1].split('\n}',1)[0]
        self.assertIn('ck_train_forward_step()',evaluation)
        self.assertIn('ck_train_eval_loss_step()',evaluation)
        self.assertNotIn('ck_train_backward_step()',evaluation)
        self.assertNotIn('ck_train_optimizer_step(',evaluation)
    def test_invalid_declared_loss_fails_closed(self):
        for mutation in ('unknown','missing','root','workspace','dtype','count','modes','no_seed','collision'):
            g=fixture();r=deepcopy(REGISTRY)
            if mutation=='unknown':g['loss']['kernel_id']='unknown'
            elif mutation=='missing':g['loss']['inputs'].pop('targets')
            elif mutation=='root':g['loss']['inputs']['logits']='weight.up'
            elif mutation=='workspace':g['loss']['scratch']['scratch']=0
            elif mutation=='dtype':g['tensors']['input.labels']['dtype']='fp32'
            elif mutation=='count':g['loss']['params']['call_constants']['logits_elements']=5
            elif mutation=='modes':r['kernels'][-1]['modes']['backward']=False
            elif mutation=='collision':g['tensors']['aux.loss']=dict(g['tensors']['act.score'])
            else:r['kernels'][-1]['training_contract'].pop('loss_seed')
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,RuntimeError)):generate(g,r)
    def test_default_decoder_loss_remains_supported(self):
        ir=base.lower(base.fixture());manifest=dict(config=ir['config'])
        layout=base.planner.build_layout(ir,manifest,64,strict=True)
        source,report=base.codegen.generate_c(ir,base.REGISTRY,manifest=manifest,layout=layout)
        self.assertEqual(report['skipped_ops'],[]);self.assertIn('ck_train_ce_dispatch(',source)
        self.assertEqual(report['evaluation_loss_ops'],1)
        self.assertIn('int ck_train_eval_step(float *loss_out)',source)

class GroupedLossRecovery(unittest.TestCase):
    def test_invalid_group_labels_discard_partial_window_and_recover(self):
        engine=ROOT/'build/libckernel_engine.so'
        if not engine.exists():self.skipTest('native engine is not built')
        source,report=generate(fixture())
        rng=np.random.default_rng(92)
        sizes=np.array(report['init_weight_numel'],'i4')
        packed=rng.normal(0,.2,int(sizes.sum())).astype('f')
        ids=np.arange(4,dtype='i4');idptr=ids.ctypes.data_as(IP)
        metadata=np.array([0,2,0,4,0,1,1,1,1,3,0,2],'i4')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'train.c').write_text(source)
            subprocess.run(['cc','-O2','-shared','-fPIC','-DCK_NUM_TOKENS=4',
                '-DCK_GRAD_ACCUM_STEPS=2','-I'+str(ROOT/'include'),str(path/'train.c'),
                str(engine),'-Wl,-rpath,'+str(engine.parent),'-lm','-o',str(path/'train.so')],check=True)
            lib=C.CDLL(str(path/'train.so'))
            lib.ck_train_init.argtypes=[FP,IP,C.c_int]
            lib.ck_train_step_with_i32_ex.argtypes=[IP,IP,C.c_int,IP,C.c_size_t,FP,C.c_float]
            def init():
                self.assertEqual(lib.ck_train_init(packed.ctypes.data_as(FP),sizes.ctypes.data_as(IP),len(sizes)),len(sizes))
            def snap(kind):
                n=report[kind+'_snapshot_floats'];a=np.empty(n,'f')
                fn=getattr(lib,'ck_train_export_'+kind+'_snapshot');fn.argtypes=[FP,C.c_int]
                self.assertEqual(fn(a.ctypes.data_as(FP),n),n);return a
            def step(meta):
                loss=C.c_float(987)
                rc=lib.ck_train_step_with_i32_ex(idptr,idptr,4,meta.ctypes.data_as(IP),meta.size,C.byref(loss),.03)
                return rc,loss.value
            try:
                init();initial=snap('weight');moments=snap('optimizer_state')
                # Real loss-provider rejection, after the complete forward graph.
                for index,value in ((8,0),(9,2),(10,1),(11,-1),(11,3)):
                    with self.subTest(index=index,value=value):
                        self.assertGreater(step(metadata)[0],0)
                        self.assertEqual(lib.ck_train_get_opt_step(),0)
                        self.assertGreater(np.max(np.abs(snap('parameter_gradient'))),0)
                        bad=metadata.copy();bad[index]=value
                        rc,loss=step(bad);self.assertLess(rc,0);self.assertEqual(loss,987)
                        np.testing.assert_array_equal(snap('parameter_gradient'),0)
                        np.testing.assert_array_equal(snap('weight'),initial)
                        np.testing.assert_array_equal(snap('optimizer_state'),moments)
                        self.assertEqual(lib.ck_train_get_opt_step(),0)
                # Recover numerically, not merely by returning a success code.
                self.assertGreater(step(metadata)[0],0);self.assertGreater(step(metadata)[0],0)
                recovered=[snap(k) for k in ('weight','optimizer_state')]
                lib.ck_train_free();init()
                self.assertGreater(step(metadata)[0],0);self.assertGreater(step(metadata)[0],0)
                for actual,kind in zip(recovered,('weight','optimizer_state')):
                    np.testing.assert_array_equal(actual,snap(kind))
                # Different question partitions and labels must affect training;
                # repeat A after B from fresh state to detect stale metadata.
                def fresh(meta):
                    lib.ck_train_free();init()
                    rc,loss=step(meta);self.assertGreater(rc,0)
                    return loss,snap('parameter_gradient')
                a,ga=fresh(metadata)
                bmeta=metadata.copy();bmeta[4:]=[0,0,1,1,2,2,1,0]
                b,gb=fresh(bmeta)
                self.assertFalse(np.array_equal(ga,gb));self.assertNotEqual(a,b)
                again,ga2=fresh(metadata);self.assertEqual(a,again)
                np.testing.assert_array_equal(ga,ga2)
            finally:lib.ck_train_free()


class ScoringLossNative(unittest.TestCase):
    def test_all_gradients_biases_updates_and_export(self):
        try:import torch
        except ImportError as exc:self.skipTest('independent PyTorch unavailable: '+str(exc))
        rng=np.random.default_rng(837)
        values=dict(features=np.eye(4,dtype='f'),up=rng.normal(0,.2,(4,4)).astype('f'),hidden=rng.normal(0,.2,(4,16)).astype('f'),hidden_bias=rng.normal(0,.1,4).astype('f'),score=rng.normal(0,.2,(1,4)).astype('f'),score_bias=np.array([.1],'f'))
        def forward(p):
            x=p['features']@p['up'].T;q=torch.stack([x[:2].mean(0),x.mean(0)]);left=q[[0,1,1,1]]
            pair=torch.cat([left,x,left*x,torch.abs(left-x)],-1)
            h=torch.nn.functional.gelu(pair@p['hidden'].T+p['hidden_bias'],approximate='none')
            return h@p['score'].T+p['score_bias']
        def criterion(z):
            z=z.flatten();return (torch.nn.functional.cross_entropy(z[:1][None],torch.tensor([0]))+torch.nn.functional.cross_entropy(z[1:][None],torch.tensor([2])))/2
        metadata=np.array([0,2,0,4,0,1,1,1,1,3,0,2],'i4')
        certify_training(self,lower(fixture()),REGISTRY,values,set(values)-{'features','score_bias'},forward,integer_inputs=metadata,loss_function=criterion,check_evaluation=True)

if __name__=='__main__':unittest.main(verbosity=2)
