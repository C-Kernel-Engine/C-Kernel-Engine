"""Automatic training of cosine/prior/residual scoring, not the full Clef head."""
import ctypes as C
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from tests import test_v8_grouped_loss_training as base
from tests.v8_training_test_helpers import certify_training
ROOT=Path(__file__).resolve().parents[1]
FP=C.POINTER(C.c_float)
STEMS=('scalar_exp_clamped','scalar_sigmoid','scale_vector')
MAPS={stem:json.loads((ROOT/'version/v8/kernel_maps'/f'{stem}_backward_checked_f32.json').read_text()) for stem in STEMS}
REGISTRY=deepcopy(base.REGISTRY)
NAMES=['indexed_cosine_rows_checked_f32','indexed_cosine_rows_backward_checked_f32','indexed_normalized_dot_backward_checked_f32','add_rows_checked_f32']
NAMES += [stem+suffix for stem in STEMS for suffix in ('_checked_f32','_backward_checked_f32')]
REGISTRY['kernels'].extend(json.loads((ROOT/'version/v8/kernel_maps'/f'{n}.json').read_text()) for n in NAMES)
RULES=json.loads((ROOT/'version/v8/scripts/grad_rules_v8.json').read_text())


def setUpModule():
    # Initialize native threading before unittest writes a case-status line, so
    # the runtime's stderr notice cannot hide native parity from nightly parsing.
    global _engine
    path=ROOT/'build/libckernel_engine.so'
    if path.exists():
        _engine=C.CDLL(str(path));_engine.ck_set_num_threads.argtypes=[C.c_int]
        _engine.ck_set_num_threads(1)


class ScalarVJPs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.libs=[]
        for opt in ('O0','O3'):
            path=Path(cls.temp.name)/(opt+'.so')
            subprocess.run(['cc','-std=c11','-'+opt,'-Wall','-Wextra','-Werror','-ffp-contract=off','-shared','-fPIC','-I'+str(ROOT/'include'),str(ROOT/'src/kernels/decision_scoring_backward.c'),'-lm','-o',str(path)],check=True)
            lib=C.CDLL(str(path));cls.libs.append(lib)
            for m in MAPS.values():
                fn=getattr(lib,m['impl']['function'])
                fn.argtypes=[FP if '*' in p.get('cast','') else C.c_float if p['name']=='maximum' else C.c_size_t for p in m['call_abi']['params']]
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def arrays(self,stem,n=7):
        rng=np.random.default_rng(81)
        a=dict(input=rng.normal(0,.4,n if stem=='scale_vector' else 1).astype('f'),d_output=rng.normal(0,.5,n if stem=='scale_vector' else 1).astype('f'))
        a['d_input']=np.full(a['input'].size+2,77,'f')
        if stem=='scale_vector':a.update(gain=np.array([.7],'f'),d_gain=np.full(3,77,'f'),scratch=np.empty(n+1,'f'))
        return a
    def args(self,stem,a,maximum=2.):
        scalar=dict(rows=a['input'].size,maximum=maximum)
        return [a[p['name']].ctypes.data_as(FP) if p['name'] in a else a[p['name'][:-9]].size if p['name'].endswith('_elements') else scalar[p['name']] for p in MAPS[stem]['call_abi']['params']]
    def run_op(self,stem,a,maximum=2.,lib=None):
        return getattr(lib or self.libs[-1],MAPS[stem]['impl']['function'])(*self.args(stem,a,maximum))
    def test_ordered_oracle_padding_and_optimization(self):
        for stem in STEMS:
            for n in (1,7,33):
                a=self.arrays(stem,n)
                if stem=='scale_vector':expected=a['d_output'].astype('d')*float(a['gain'][0]);gain=sum(float(x)*float(y) for x,y in zip(a['input'],a['d_output']))
                elif stem=='scalar_exp_clamped':expected=a['d_output'].astype('d')*np.exp(a['input']).astype('f')
                else:
                    x=float(a['input'][0]);e=np.float32(math.exp(-abs(x)));y=np.float32(1)/np.float32(1+e) if x>=0 else e/np.float32(1+e)
                    expected=a['d_output'].astype('d')*float(y)*(1-float(y))
                previous=None
                for lib in self.libs:
                    self.assertEqual(self.run_op(stem,a,lib=lib),0)
                    np.testing.assert_allclose(a['d_input'][:-2],expected,atol=2e-7,rtol=2e-7)
                    np.testing.assert_array_equal(a['d_input'][-2:],77)
                    if stem=='scale_vector':
                        np.testing.assert_allclose(a['d_gain'][0],gain,atol=2e-7,rtol=2e-7);np.testing.assert_array_equal(a['d_gain'][1:],77)
                    result=a['d_input'].tobytes()+(a['d_gain'].tobytes() if stem=='scale_vector' else b'')
                    if previous is not None:self.assertEqual(previous,result)
                    previous=result
    def test_gain_reduction_is_fp64_and_buffers_are_bounded(self):
        a=self.arrays('scale_vector',3);a['input'][:]=[1e8,1,-1e8];a['d_output'][:]=1
        self.assertEqual(self.run_op('scale_vector',a),0);self.assertEqual(a['d_gain'][0],1.)
        fn=getattr(self.libs[-1],MAPS['scale_vector']['impl']['function'])
        for name,minimum in (('input',3),('gain',1),('d_output',3),('d_input',3),('d_gain',1),('scratch',4)):
            a=self.arrays('scale_vector',3);args=self.args('scale_vector',a)
            ix=next(i for i,p in enumerate(MAPS['scale_vector']['call_abi']['params']) if p['name']==name+'_elements')
            args[ix]=minimum-1;self.assertLess(fn(*args),0)
            np.testing.assert_array_equal(a['d_input'],77);np.testing.assert_array_equal(a['d_gain'],77)
            args[ix]=minimum;self.assertEqual(fn(*args),0)
        a=self.arrays('scale_vector',7)
        for output in ('d_input','d_gain','scratch'):
            args=self.args('scale_vector',a)
            ix=next(i for i,p in enumerate(MAPS['scale_vector']['call_abi']['params']) if p['name']==output)
            args[ix]=C.cast(a['input'].ctypes.data+4,FP);self.assertLess(fn(*args),0)
        args=self.args('scale_vector',a)
        ix=next(i for i,p in enumerate(MAPS['scale_vector']['call_abi']['params']) if p['name']=='d_gain')
        args[ix]=C.cast(a['d_input'].ctypes.data+4,FP);self.assertLess(fn(*args),0)
        np.testing.assert_array_equal(a['d_input'],77);np.testing.assert_array_equal(a['d_gain'],77)
        a=self.arrays('scalar_exp_clamped');a['input'][0]=2;a['d_output'][0]=np.finfo('f').max
        self.assertLess(self.run_op('scalar_exp_clamped',a),0);np.testing.assert_array_equal(a['d_input'],77)

    def test_pytorch_boundaries_and_finite_differences(self):
        try:import torch
        except ImportError as exc:self.skipTest('independent PyTorch unavailable: '+str(exc))
        for stem in STEMS:
            for value in (-100.,-.7,0.,2.,2.001,100.):
                a=self.arrays(stem);a['input'][0]=value
                x=torch.tensor(a['input'],requires_grad=True);dy=torch.tensor(a['d_output'])
                if stem=='scale_vector':
                    gain=torch.tensor(a['gain'],requires_grad=True);y=x*gain
                else:y=torch.exp(torch.clamp(x,max=2.)) if stem=='scalar_exp_clamped' else torch.sigmoid(x)
                (y*dy).sum().backward();self.assertEqual(self.run_op(stem,a),0)
                np.testing.assert_allclose(a['d_input'][:-2],x.grad.numpy(),atol=2e-7,rtol=2e-6)
                if stem=='scale_vector':np.testing.assert_allclose(a['d_gain'][:1],gain.grad.numpy(),atol=2e-6,rtol=2e-6)
                if value in (-.7,0.):
                    xx=x.detach().double()
                    def f(z):
                        yy=z*gain.detach().double() if stem=='scale_vector' else torch.exp(torch.clamp(z,max=2.)) if stem=='scalar_exp_clamped' else torch.sigmoid(z)
                        return (yy*dy.double()).sum().item()
                    lo=xx.clone();hi=xx.clone();lo[0]-=1e-5;hi[0]+=1e-5
                    self.assertAlmostEqual((f(hi)-f(lo))/2e-5,float(a['d_input'][0]),delta=2e-7)
    def test_rejection_atomicity_and_reuse(self):
        for stem in STEMS:
            a=self.arrays(stem);args=self.args(stem,a);fn=getattr(self.libs[-1],MAPS[stem]['impl']['function'])
            inputs=['input','d_output']+(['gain'] if stem=='scale_vector' else [])
            for name in inputs:
                old=a[name][0];a[name][0]=np.nan;self.assertLess(fn(*args),0);a[name][0]=old
            for i,p in enumerate(MAPS[stem]['call_abi']['params']):
                bad=list(args)
                if '*' in p.get('cast',''):bad[i]=None
                elif p['name'].endswith('_elements'):bad[i]=0
                elif p['name']=='maximum':bad[i]=float('nan')
                else:bad[i]=0
                self.assertLess(fn(*bad),0)
            for out in ('d_input','d_gain') if stem=='scale_vector' else ('d_input',):
                ix=next(i for i,p in enumerate(MAPS[stem]['call_abi']['params']) if p['name']==out)
                for name in inputs:
                    bad=list(args);bad[ix]=a[name].ctypes.data_as(FP);self.assertLess(fn(*bad),0)
                bad=list(args);bad[ix]=C.cast(a[out].ctypes.data+1,FP);self.assertLess(fn(*bad),0)
                np.testing.assert_array_equal(a[out],77)
            self.assertEqual(fn(*args),0);clean=self.arrays(stem);self.assertEqual(self.run_op(stem,clean),0)
            np.testing.assert_array_equal(a['d_input'],clean['d_input'])
            if stem=='scale_vector':np.testing.assert_array_equal(a['d_gain'],clean['d_gain'])
        # Finite operands can overflow either output; neither may be published.
        for mode in ('input','gain'):
            a=self.arrays('scale_vector',3);a['d_output'][:]=np.finfo('f').max
            a[mode][:]=2
            self.assertLess(self.run_op('scale_vector',a),0)
            np.testing.assert_array_equal(a['d_input'],77);np.testing.assert_array_equal(a['d_gain'],77)
        a=self.arrays('scalar_exp_clamped');a['input'][0]=100
        self.assertLess(self.run_op('scalar_exp_clamped',a,maximum=100),0);np.testing.assert_array_equal(a['d_input'],77)


def fixture():
    g=base.fixture()
    # Preserve the residual scorer, then attach the other scoring branches.
    old=g['ops'][-1]['dataflow']['outputs']['y'];old['tensor']='act.residual'
    g['tensors']['act.residual']=dict(g['tensors'].pop('act.score'),tensor='act.residual')
    def ref(tid,shape,kind='activation',grad=True):
        r=dict(tensor=tid,shape=shape,numel=math.prod(shape),dtype='fp32',kind=kind,requires_grad=grad,persistent=kind=='weight');g['tensors'][tid]=dict(r);return r
    def op(name,kernel,inputs,shape,constants,rule,weights=None,params=None,scratch=0):
        y=ref('act.'+name,shape);outputs=dict(output=y)
        if scratch:outputs['scratch']=ref('aux.'+name,[scratch],grad=False)
        g['ops'].append(dict(op_id=len(g['ops']),op=name,kernel_id=kernel,phase='forward',section='body',layer=0,requires_grad=True,map_call=True,weights=weights or {},params=dict(params or {},call_constants=constants),dataflow=dict(inputs=inputs,outputs=outputs),save_for_backward=inputs,grad_rule=rule));return y
    x=g['ops'][1]['dataflow']['outputs']['y'];q=g['ops'][2]['dataflow']['outputs']['output'];owners=g['ops'][3]['dataflow']['inputs']['indices']
    def cosine(name,rule,eps):
        return op(name,'indexed_cosine_rows_checked_f32',dict(left=q,right=x,indices=owners),[4,1],dict(left_elements=8,right_elements=16,indices_elements=4,output_elements=4,scratch_elements=40,groups=2,rows=4,channels=4,d_output_elements=4,d_left_elements=8,d_right_elements=16),rule,params=dict(epsilon=eps),scratch=40)
    prior=cosine('prior','indexed_normalized_dot_checked_backward',1e-12)
    joint=cosine('joint','indexed_cosine_rows_checked_backward',1e-8)
    def scalar(name,stem):
        w=dict(ref('weight.'+name,[1],'weight'),name=name)
        return op(name,stem+'_checked_f32',{},[1],dict(input_elements=1,output_elements=1,d_output_elements=1,d_input_elements=1),stem+'_checked_backward',weights=dict(input=w),params=dict(maximum=math.log(100.)) if stem=='scalar_exp_clamped' else {})
    ps=scalar('prior_scale','scalar_exp_clamped');js=scalar('joint_scale','scalar_exp_clamped');gate=scalar('gate','scalar_sigmoid')
    def scale(name,a,b):return op(name,'scale_vector_checked_f32',dict(input=a,gain=b),[4,1],dict(input_elements=4,gain_elements=1,output_elements=4,scratch_elements=5,rows=4,d_output_elements=4,d_input_elements=4,d_gain_elements=1,backward_scratch_elements=5),'scale_vector_checked_backward',scratch=5)
    def add(name,a,b):return op(name,'add_rows_checked_f32',dict(left=a,right=b),[4,1],dict(left_elements=4,right_elements=4,output_elements=4,elements=4),'add_rows_checked_backward')
    left=scale('prior_scores',prior,ps);right=add('joint_scores',scale('joint_scaled',joint,js),old)
    add('score',left,scale('gated',right,gate))
    return g

def lower(g):return base.base.backward.synthesize_ir2_backward(g,REGISTRY,base.base.BINDINGS,RULES,strict=True,allow_partial=False,checkpoint_policy='none')
def generate(g):
    ir=lower(g);manifest=dict(config=ir['config']);layout=base.base.planner.build_layout(ir,manifest,64,strict=True)
    return base.base.codegen.generate_c(ir,REGISTRY,manifest=manifest,layout=layout)

class ScoringContracts(unittest.TestCase):
    def test_all_derivatives_and_scalar_parameters_are_resolved(self):
        g=fixture();original=deepcopy(g);source,report=generate(g);self.assertEqual(g,original)
        self.assertEqual(report['skipped_ops'],[])
        for name in ('gate','prior_scale','joint_scale'):self.assertIn(name,report['parameter_gradient_order'])
        for stem in STEMS:self.assertIn(MAPS[stem]['impl']['function']+'(',source)
        self.assertIn('ck_indexed_normalized_dot_backward_f32(',source);self.assertIn('ck_indexed_cosine_rows_backward_f32(',source)
    def test_frozen_gate_has_no_optimizer_state_without_cutting_branch_gradients(self):
        g=fixture();gate=next(o for o in g['ops'] if o['op']=='gate')
        gate['weights']['input']['requires_grad']=False;g['tensors']['weight.gate']['requires_grad']=False
        ir=lower(g);source,report=generate(g)
        self.assertNotIn('gate',report['parameter_gradient_order'])
        self.assertNotIn('grad.weight.gate',ir['tensors'])
        self.assertIn('joint_scale',report['parameter_gradient_order'])
        derivative=next(o for o in ir['backward'] if o.get('forward_ref')==gate['op_id'] and o.get('role')=='core')
        self.assertEqual(derivative['dataflow']['outputs']['d_input']['kind'],'discarded_grad_weight')
        self.assertIn('ck_scale_vector_backward_f32(',source)

    def test_malformed_shapes_and_capacities_are_rejected(self):
        for name,key,value in (('prior','left_elements',7),('gated','gain_elements',2),('score','elements',5)):
            g=fixture();o=next(o for o in g['ops'] if o['op']==name);o['params']['call_constants'][key]=value
            with self.subTest(name=name),self.assertRaises(ValueError):generate(g)

class ScoringRecovery(unittest.TestCase):
    def test_partial_backward_failure_preserves_optimizer_and_recovers(self):
        engine=ROOT/'build/libckernel_engine.so'
        if not engine.exists():self.skipTest('native engine is not built')
        registry=deepcopy(REGISTRY)
        m=next(m for m in registry['kernels'] if m['id']=='scalar_sigmoid_backward_checked_f32')
        original=m['impl']['c_declaration'];fun=m['impl']['function']
        m['impl']['function']='test_sigmoid_backward'
        m['impl']['c_declaration']=original.replace(fun,'test_sigmoid_backward')
        ir=lower(fixture());manifest=dict(config=ir['config'])
        layout=base.base.planner.build_layout(ir,manifest,64,strict=True)
        source,report=base.base.codegen.generate_c(ir,registry,manifest=manifest,layout=layout)
        wrapper='#include <stddef.h>\n'+original+'\nstatic int fail; void fail_next(void){fail=1;}\n'
        wrapper+=m['impl']['c_declaration'][:-1]+' {if(fail){fail=0;return -77;} return '+fun+'('+', '.join(p['name'] for p in m['call_abi']['params'])+');}\n'
        rng=np.random.default_rng(837);sizes=np.array(report['init_weight_numel'],'i4')
        packed=rng.normal(0,.2,int(sizes.sum())).astype('f')
        ip=C.POINTER(C.c_int32);ids=np.arange(4,dtype='i4');ptr=ids.ctypes.data_as(ip)
        metadata=np.array([0,2,0,4,0,1,1,1,1,3,0,2],'i4')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'train.c').write_text(source);(path/'fault.c').write_text(wrapper)
            subprocess.run(['cc','-O2','-shared','-fPIC','-DCK_NUM_TOKENS=4','-DCK_GRAD_ACCUM_STEPS=2',
                '-I'+str(ROOT/'include'),str(path/'train.c'),str(path/'fault.c'),str(engine),
                '-Wl,-rpath,'+str(engine.parent),'-lm','-o',str(path/'train.so')],check=True)
            lib=C.CDLL(str(path/'train.so'));lib.ck_train_init.argtypes=[FP,ip,C.c_int]
            lib.ck_train_step_with_i32_ex.argtypes=[ip,ip,C.c_int,ip,C.c_size_t,FP,C.c_float]
            def init():self.assertEqual(lib.ck_train_init(packed.ctypes.data_as(FP),sizes.ctypes.data_as(ip),len(sizes)),len(sizes))
            def snap(kind):
                n=report[kind+'_snapshot_floats'];a=np.empty(n,'f')
                f=getattr(lib,'ck_train_export_'+kind+'_snapshot');f.argtypes=[FP,C.c_int]
                self.assertEqual(f(a.ctypes.data_as(FP),n),n);return a
            def step():
                loss=C.c_float(987)
                rc=lib.ck_train_step_with_i32_ex(ptr,ptr,4,metadata.ctypes.data_as(ip),metadata.size,C.byref(loss),.03)
                return rc,loss.value
            try:
                init();initial=snap('weight');moments=snap('optimizer_state')
                self.assertGreater(step()[0],0);self.assertGreater(np.max(np.abs(snap('parameter_gradient'))),0)
                lib.fail_next();rc,loss=step();self.assertLess(rc,0);self.assertEqual(loss,987)
                np.testing.assert_array_equal(snap('weight'),initial);np.testing.assert_array_equal(snap('optimizer_state'),moments)
                np.testing.assert_array_equal(snap('parameter_gradient'),0);self.assertEqual(lib.ck_train_get_opt_step(),0)
                self.assertGreater(step()[0],0);self.assertGreater(step()[0],0)
                recovered=[snap(k) for k in ('weight','optimizer_state')]
                lib.ck_train_free();init();self.assertGreater(step()[0],0);self.assertGreater(step()[0],0)
                for actual,kind in zip(recovered,('weight','optimizer_state')):np.testing.assert_array_equal(actual,snap(kind))
            finally:lib.ck_train_free()


class ScoringNative(unittest.TestCase):
    def test_learned_gates_cosine_branches_and_residual_train(self):
        try:import torch
        except ImportError as exc:self.skipTest('independent PyTorch unavailable: '+str(exc))
        rng=np.random.default_rng(837)
        values=dict(features=np.eye(4,dtype='f'),up=rng.normal(0,.2,(4,4)).astype('f'),hidden=rng.normal(0,.2,(4,16)).astype('f'),hidden_bias=rng.normal(0,.1,4).astype('f'),score=rng.normal(0,.2,(1,4)).astype('f'),score_bias=np.array([.1],'f'),prior_scale=np.array([.2],'f'),joint_scale=np.array([-.1],'f'),gate=np.array([.3],'f'))
        def forward(p):
            x=p['features']@p['up'].T;q=torch.stack([x[:2].mean(0),x.mean(0)]);left=q[[0,1,1,1]]
            pair=torch.cat([left,x,left*x,torch.abs(left-x)],-1)
            h=torch.nn.functional.gelu(pair@p['hidden'].T+p['hidden_bias'],approximate='none')
            residual=(h@p['score'].T+p['score_bias']).flatten()
            prior=(torch.nn.functional.normalize(left,dim=-1,eps=1e-12)*torch.nn.functional.normalize(x,dim=-1,eps=1e-12)).sum(-1)
            joint=torch.nn.functional.cosine_similarity(left,x,dim=-1,eps=1e-8)
            return (prior*torch.exp(p['prior_scale'].clamp(max=math.log(100.)))+(joint*torch.exp(p['joint_scale'].clamp(max=math.log(100.)))+residual)*torch.sigmoid(p['gate']))[:,None]
        def criterion(z):
            z=z.flatten();return (torch.nn.functional.cross_entropy(z[:1][None],torch.tensor([0]))+torch.nn.functional.cross_entropy(z[1:][None],torch.tensor([2])))/2
        metadata=np.array([0,2,0,4,0,1,1,1,1,3,0,2],'i4')
        # Ensure the fixture meaningfully exercises every newly trainable scalar.
        params={n:torch.tensor(v,requires_grad=n not in ('features','score_bias')) for n,v in values.items()}
        criterion(forward(params)).backward()
        for n in ('prior_scale','joint_scale','gate'):
            self.assertGreater(float(params[n].grad.abs().max()),1e-4,n)
            double={k:v.detach().double() for k,v in params.items()}
            def evaluate(delta):
                perturbed=dict(double);perturbed[n]=double[n]+delta
                return criterion(forward(perturbed)).item()
            self.assertAlmostEqual(float(params[n].grad[0]),(evaluate(1e-5)-evaluate(-1e-5))/2e-5,delta=2e-7)
        certify_training(self,lower(fixture()),REGISTRY,values,set(values)-{'features','score_bias'},forward,integer_inputs=metadata,loss_function=criterion)

if __name__=='__main__':unittest.main(verbosity=2)
