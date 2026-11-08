"""Indexed scoring VJPs: independent references and generated branch joins."""
import ctypes as C
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FP=C.POINTER(C.c_float); IP=C.POINTER(C.c_int32)
NAMES=('indexed_pair_features','indexed_normalized_dot','indexed_cosine_rows')

def serial_sum(values):
    result=0.
    for x in values:result+=float(x)
    return result

def oracle(name,a,b,indices,dy,epsilon):
    g,c=a.shape;r=len(b);local=np.zeros_like(b);right=np.zeros_like(b)
    for row,owner in enumerate(indices):
        x=a[owner];y=b[row]
        if name=='indexed_pair_features':
            for col in range(c):
                u,v=float(x[col]),float(y[col]);sign=float(u>v)-float(u<v)
                d=dy[row]
                local[row,col]=(float(d[col])+float(d[2*c+col])*v)+float(d[3*c+col])*sign
                right[row,col]=(float(d[c+col])+float(d[2*c+col])*u)-float(d[3*c+col])*sign
        else:
            na=np.float32(math.sqrt(serial_sum(float(z)*float(z) for z in x)))
            nb=np.float32(math.sqrt(serial_sum(float(z)*float(z) for z in y)))
            fa=max(na,np.float32(epsilon));fb=max(nb,np.float32(epsilon));xn=x/fa;yn=y/fb
            dot=serial_sum(np.float32(u*v) for u,v in zip(xn,yn))
            for col in range(c):
                da=float(yn[col])/float(fa);db=float(xn[col])/float(fb)
                if na>0 and (name=='indexed_cosine_rows' or na>=epsilon):da-=dot*(float(x[col])/float(na))/float(fa)
                if nb>0 and (name=='indexed_cosine_rows' or nb>=epsilon):db-=dot*(float(y[col])/float(nb))/float(fb)
                local[row,col]=float(dy[row])*da;right[row,col]=float(dy[row])*db
    left=np.array([[serial_sum(local[row,col] for row in range(r) if indices[row]==owner) for col in range(c)] for owner in range(g)],dtype='f')
    return left,right

def torch_output(torch,name,a,b,indices,eps):
    x=a[indices]
    if name=='indexed_pair_features':return torch.cat((x,b,x*b,torch.abs(x-b)),dim=-1)
    if name=='indexed_normalized_dot':return (torch.nn.functional.normalize(x,dim=-1,eps=eps)*torch.nn.functional.normalize(b,dim=-1,eps=eps)).sum(-1)
    return torch.nn.functional.cosine_similarity(x,b,dim=-1,eps=eps)

class ScoringBackward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.libs=[];cls.maps={}
        for opt in ('O0','O3'):
            path=Path(cls.temp.name)/(opt+'.so')
            subprocess.run(['cc','-std=c11','-'+opt,'-Wall','-Wextra','-Werror','-ffp-contract=off','-shared','-fPIC','-I'+str(ROOT/'include'),str(ROOT/'src/kernels/decision_scoring_backward.c'),'-lm','-o',str(path)],check=True)
            lib=C.CDLL(str(path));cls.libs.append(lib)
            for name in NAMES:
                m=json.loads((ROOT/f'version/v8/kernel_maps/{name}_backward_checked_f32.json').read_text());cls.maps[name]=m
                fn=getattr(lib,m['impl']['function']);fn.argtypes=[IP if 'int32_t' in p.get('cast','') else FP if '*' in p.get('cast','') else C.c_float if p['name']=='epsilon' else C.c_size_t for p in m['call_abi']['params']]
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def operands(self,name,g=3,r=5,c=7,seed=819):
        rng=np.random.default_rng(seed)
        a=rng.normal(0,.4,(g,c)).astype('f');b=rng.normal(0,.5,(r,c)).astype('f')
        ids=np.arange(r,dtype='i4')%max(g-1,1) # repeated, one unreferenced owner
        dy=rng.normal(0,.3,(r,4*c) if name==NAMES[0] else r).astype('f')
        return dict(left=a,right=b,indices=ids,d_output=dy,d_left=np.full(g*c+3,77,'f'),d_right=np.full(r*c+3,77,'f'),scratch=np.empty(g*c+2*r*c,'f'))

    def arguments(self,name,arrays,epsilon=1e-8):
        scalars=dict(groups=len(arrays['left']),rows=len(arrays['right']),channels=arrays['left'].shape[1],epsilon=epsilon)
        result=[]
        for p in self.maps[name]['call_abi']['params']:
            n=p['name']
            if n in arrays:result.append(arrays[n].ctypes.data_as(IP if n=='indices' else FP))
            elif n.endswith('_elements'):result.append(arrays[n[:-9]].size)
            else:result.append(scalars[n])
        return result

    def run_op(self,name,arrays,epsilon=1e-8,lib=None):
        return getattr(lib or self.libs[-1],self.maps[name]['impl']['function'])(*self.arguments(name,arrays,epsilon))

    def test_ordered_oracle_tails_padding_and_optimization(self):
        for name in NAMES:
            for dims in ((1,1,1),(3,5,7),(4,9,64),(2,3,129)):
                with self.subTest(name=name,dims=dims):
                    a=self.operands(name,*dims);before={k:v.copy() for k,v in a.items() if k in ('left','right','indices','d_output')}
                    expected=oracle(name,a['left'],a['right'],a['indices'],a['d_output'],1e-8)
                    baseline=None
                    for lib in self.libs:
                        self.assertEqual(self.run_op(name,a,lib=lib),0)
                        out=[a['d_left'][:-3],a['d_right'][:-3]]
                        for x,y in zip(out,expected):np.testing.assert_allclose(x,y.ravel(),atol=2e-7,rtol=3e-7)
                        if baseline is not None:
                            for x,y in zip(out,baseline):np.testing.assert_array_equal(x,y)
                        baseline=[x.copy() for x in out]
                        for n in ('d_left','d_right'):np.testing.assert_array_equal(a[n][-3:],77)
                    for k,v in before.items():np.testing.assert_array_equal(a[k],v)

    def test_pytorch_autograd_and_finite_differences(self):
        try:import torch
        except ImportError as exc:self.skipTest('PyTorch oracle unavailable: '+str(exc))
        torch.set_num_threads(1)
        for name in NAMES:
            max_error=0.
            for dims in ((1,1,1),(3,5,7),(2,4,129)):
                a=self.operands(name,*dims);ta=torch.tensor(a['left'],requires_grad=True);tb=torch.tensor(a['right'],requires_grad=True)
                y=torch_output(torch,name,ta,tb,a['indices'],1e-8);y.backward(torch.tensor(a['d_output']))
                self.assertEqual(self.run_op(name,a),0)
                for actual,expected in ((a['d_left'][:-3],ta.grad.numpy()),(a['d_right'][:-3],tb.grad.numpy())):
                    max_error=max(max_error,float(np.max(np.abs(actual-expected.ravel()))))
                    np.testing.assert_allclose(actual,expected.ravel(),atol=2e-6,rtol=2e-5)
                # Independent double reference central differences away from kinks.
                for key,out in (('left','d_left'),('right','d_right')):
                    tensors={k:torch.tensor(a[k],dtype=torch.float64) for k in ('left','right')}
                    def value(delta):
                        p={k:v.clone() for k,v in tensors.items()};p[key][0,0]+=delta
                        return float((torch_output(torch,name,p['left'],p['right'],a['indices'],1e-8)*torch.tensor(a['d_output'],dtype=torch.float64)).sum())
                    self.assertAlmostEqual(float(a[out][0]),(value(1e-5)-value(-1e-5))/2e-5,delta=2e-6)

            print(f'{name} FP32 gradient max_abs={max_error:.9g}; PyTorch {torch.__version__}')

    def test_distinct_clamp_semantics_zero_and_boundary(self):
        try:import torch
        except ImportError as exc:self.skipTest('PyTorch oracle unavailable: '+str(exc))
        observed={}
        for name in NAMES[1:]:
            for magnitude in (0.,.25,np.nextafter(np.float32(1),np.float32(0)),1.,np.nextafter(np.float32(1),np.float32(2)),2.):
                for side in ('left','right'):
                    a=self.operands(name,1,1,2);a[side][:]=[magnitude,0];a['right' if side=='left' else 'left'][:]=[1,1];a['d_output'][:]=1
                    ta=torch.tensor(a['left'],requires_grad=True);tb=torch.tensor(a['right'],requires_grad=True)
                    torch_output(torch,name,ta,tb,a['indices'],1.).sum().backward()
                    self.assertEqual(self.run_op(name,a,1.),0)
                    for x,y in ((a['d_left'][:-3],ta.grad.numpy()),(a['d_right'][:-3],tb.grad.numpy())):np.testing.assert_allclose(x,y.ravel(),atol=2e-7,rtol=2e-6)
                    if magnitude==.25 and side=='left':observed[name]=a['d_left'][0]
        self.assertGreater(abs(float(observed[NAMES[1]]-observed[NAMES[2]])),.1)
        # abs derivative at equality must be zero, not signbit-based.
        a=self.operands(NAMES[0],1,2,3);a['left'][:]=0;a['right'][:]=-0.;a['d_output'][:]=0;a['d_output'][:,9:]=1
        self.assertEqual(self.run_op(NAMES[0],a),0)
        np.testing.assert_array_equal(a['d_left'][:-3],0);np.testing.assert_array_equal(a['d_right'][:-3],0)

    def test_rejection_is_atomic_and_recovery_matches_fresh(self):
        for name in NAMES:
            for kind in ('null','short','unaligned','overlap','nonfinite','index','dimension','zero','epsilon'):
                if kind=='epsilon' and name==NAMES[0]:continue
                cases=range(7) if kind in ('null','short','unaligned') else range(3) if kind in ('nonfinite','dimension','zero') else range(2) if kind=='index' else range(5) if kind=='epsilon' else range(1)
                for index in cases:
                    with self.subTest(name=name,kind=kind,index=index):
                        a=self.operands(name);args=self.arguments(name,a);names=['left','right','indices','d_output','d_left','d_right','scratch']
                        if kind=='null':args[2*index]=None
                        elif kind=='short':args[2*index+1]=(a[names[index]].size-(3 if index in (4,5) else 0))-1
                        elif kind=='unaligned':args[2*index]=C.cast(a[names[index]].ctypes.data+1,IP if index==2 else FP)
                        elif kind=='overlap':args[10]=C.cast(a['d_left'].ctypes.data+4,FP)
                        elif kind=='nonfinite':a[('left','right','d_output')[index]].flat[0]=np.nan
                        elif kind=='index':a['indices'][-1]=(-1,3)[index]
                        elif kind=='dimension':args[14+index]=2**64-1
                        elif kind=='zero':args[14+index]=0
                        else:args[-1]=(0.,-1.,float('nan'),float('inf'),-float('inf'))[index]
                        self.assertLess(getattr(self.libs[-1],self.maps[name]['impl']['function'])(*args),0)
                        for key in ('d_left','d_right'):np.testing.assert_array_equal(a[key],77)
                        clean=self.operands(name);a.update({k:clean[k] for k in ('left','right','indices','d_output')})
                        self.assertEqual(self.run_op(name,a),0);self.assertEqual(self.run_op(name,clean),0)
                        for key in ('d_left','d_right'):np.testing.assert_array_equal(a[key],clean[key])

    def test_exact_capacity_leaves_trailing_output_guards(self):
        for name in NAMES:
            a=self.operands(name);args=self.arguments(name,a)
            args[9]-=3;args[11]-=3
            self.assertEqual(getattr(self.libs[-1],self.maps[name]['impl']['function'])(*args),0)
            for key in ('d_left','d_right'):np.testing.assert_array_equal(a[key][-3:],77)

    def test_all_writable_regions_reject_partial_aliases(self):
        names=['left','right','indices','d_output','d_left','d_right','scratch']
        for name in NAMES:
            for dest in range(4,7):
                for src in range(dest):
                    a=self.operands(name);args=self.arguments(name,a)
                    args[2*dest]=C.cast(a[names[src]].ctypes.data+4,FP)
                    self.assertLess(getattr(self.libs[-1],self.maps[name]['impl']['function'])(*args),0)
                    for key in ('d_left','d_right'):np.testing.assert_array_equal(a[key],77)

    def test_repeated_owner_uses_declared_fp64_reduction(self):
        a=self.operands(NAMES[0],2,3,1)
        a['left'][:]=0;a['right'][:]=0;a['indices'][:]=0;a['d_output'][:]=0
        a['d_output'][:,0]=[1e8,1,-1e8]
        self.assertEqual(self.run_op(NAMES[0],a),0)
        np.testing.assert_array_equal(a['d_left'][:-3],[1,0])

    def test_late_scatter_overflow_and_shared_readers(self):
        a=self.operands(NAMES[0],1,2,1);a['left'][:]=0;a['right'][:]=0;a['d_output'][:]=0;a['d_output'][:,0]=np.finfo('f').max
        self.assertLess(self.run_op(NAMES[0],a),0)
        for k in ('d_left','d_right'):np.testing.assert_array_equal(a[k],77)
        for name in NAMES:
            a=self.operands(name,3,3,5);a['right']=a['left'];self.assertEqual(self.run_op(name,a),0)

    def test_generated_scoring_branches_accumulate_and_recover(self):
        from tests.test_v8_token_span_mean_checked import _circuit
        from tests.v8_checked_graph_test_support import compile_native_graph
        try:import torch
        except ImportError as exc:self.skipTest('PyTorch oracle unavailable: '+str(exc))
        g,r,c=3,5,3
        shapes=dict(left=[g,c],right=[r,c],indices=[r],features=[r,4*c],activated=[r,4*c],normalized=[r],cosine=[r],
                    dy_pair=[r,4*c],dy_norm=[r],dy_cos=[r],d_features=[r,4*c],d_left=[g,c],d_right=[r,c])
        for side,n in (('left',g),('right',r)):
            for suffix in ('pair','norm','cos','partial'):shapes['d_'+side+'_'+suffix]=[n,c]
        graph=_circuit();graph['name']='indexed_scoring_training_component';graph['contract']['runtime_invariants']['inference_only']=False
        graph['activation_buffers']={k:dict(shape=v) for k,v in shapes.items()};graph['activation_bindings']={k:k for k in shapes}
        def op(kid,ident,ins,outs,constants,extra=None):
            m=json.loads((ROOT/'version/v8/kernel_maps'/f'{kid}.json').read_text())
            return dict(id=ident,op=m['op'],kernel=kid,returns_status=True,
                graph_slots=dict(inputs={k:'external:'+v for k,v in ins.items()},outputs={k:'external:'+v for k,v in outs.items()}),
                params=dict({k:constants[v] for k,v in m['call_constant_contract']['shape_symbols'].items()},call_constants=constants,**(extra or {})))
        common=dict(left_elements=g*c,right_elements=r*c,indices_elements=r,groups=g,rows=r,channels=c)
        fops=[];bops=[]
        fops.append(op('indexed_pair_features_checked_f32','pair',dict(left='left',right='right',indices='indices'),dict(output='features'),dict(common,output_elements=4*r*c,scratch_elements=4*r*c)))
        fops.append(op('gelu_rows_erf_checked_f32','gelu',dict(input='features'),dict(output='activated'),dict(gelu_input_elements=4*r*c,gelu_input_stride=4*c,gelu_output_elements=4*r*c,gelu_output_stride=4*c,gelu_scratch_elements=4*r*c,gelu_rows=r,gelu_channels=4*c)))
        bops.append(op('gelu_erf_backward_checked_f32','gelu_backward',dict(input='features',d_output='dy_pair'),dict(d_input='d_features'),dict(input_elements=4*r*c,input_stride=4*c,d_output_elements=4*r*c,d_output_stride=4*c,d_input_elements=4*r*c,d_input_stride=4*c,scratch_elements=4*r*c,rows=r,channels=4*c)))
        for name,suffix,dy in zip(NAMES,('pair','norm','cos'),('d_features','dy_norm','dy_cos')):
            extra={} if suffix=='pair' else {'epsilon':.7}
            bops.append(op(name+'_backward_checked_f32',suffix+'_backward',dict(left='left',right='right',indices='indices',d_output=dy),dict(d_left='d_left_'+suffix,d_right='d_right_'+suffix),dict(common,d_output_elements=4*r*c if suffix=='pair' else r,d_left_elements=g*c,d_right_elements=r*c,scratch_elements=g*c+2*r*c),extra))
            if suffix!='pair':fops.append(op('indexed_cosine_rows_checked_f32',suffix,dict(left='left',right='right',indices='indices'),dict(output='normalized' if suffix=='norm' else 'cosine'),dict(common,output_elements=r,scratch_elements=r),extra))
        for side,n in (('left',g),('right',r)):
            for lhs,rhs,out in [('pair','norm','partial'),('partial','cos','')]:
                bops.append(op('add_rows_checked_f32','join_'+side+out,dict(left='d_'+side+'_'+lhs,right='d_'+side+'_'+rhs),dict(output='d_'+side+('_'+out if out else '')),dict(left_elements=n*c,right_elements=n*c,output_elements=n*c,elements=n*c)))
        config=dict(model=graph['name'],arch=graph['name'],num_layers=1,embed_dim=c,num_heads=1,num_kv_heads=1,head_dim=c,intermediate_size=4*c,context_length=r,max_seq_len=r,vocab_size=16,activation_buffer_dtypes={'indices':'i32'})
        def build(path,ops,mode):
            path.mkdir();cg=deepcopy(graph);cg['native_entry']['function']='ck_scoring_'+mode;cg['block_types']['pool']['body']['ops']=ops
            src=dict(config=config,entries=[],quant_summary={},template=cg)
            layout,calls,_,lib,execute=compile_native_graph(path,src,cg,mode=mode)
            self.assertEqual(calls['errors'],[])
            size=layout['memory']['arena']['total_size'];raw=(C.c_uint8*(size+63))();arena=(C.c_uint8*size).from_buffer(raw,(-C.addressof(raw))&63)
            entries={x['name']:x for x in layout['memory']['activations']['buffers']}
            arrays={k:np.ndarray(v,dtype='i4' if k=='indices' else 'f4',buffer=arena,offset=entries[k]['abs_offset']) for k,v in shapes.items()}
            return lib,execute,size,arena,arrays,calls
        with tempfile.TemporaryDirectory() as tmp:
            _,fexecute,fs,fa,f,fcalls=build(Path(tmp)/'forward',fops,'prefill')
            _,execute,size,arena,a,calls=build(Path(tmp)/'backward',bops,'backward')
            self.assertEqual(len(calls['operations']),8)
            self.assertEqual([x['call_abi']['kernel_id'] for x in calls['operations'][1:4]],[n+'_backward_checked_f32' for n in NAMES])
            baseline=None
            for seed in (71,92,71):
                inp=self.operands(NAMES[0],g,r,c,seed)
                for k in ('left','right','indices'):f[k][:]=inp[k]
                self.assertEqual(fexecute(fa,fs),0)
                for k in ('left','right','indices','features'):a[k][:]=f[k]
                rng=np.random.default_rng(seed);a['dy_pair'][:]=inp['d_output'];a['dy_norm'][:]=rng.normal(size=r);a['dy_cos'][:]=rng.normal(size=r)
                ta=torch.tensor(a['left'],requires_grad=True);tb=torch.tensor(a['right'],requires_grad=True)
                pair=torch.nn.functional.gelu(torch_output(torch,NAMES[0],ta,tb,a['indices'],.7),approximate='none')
                norm=torch_output(torch,NAMES[1],ta,tb,a['indices'],.7);cos=torch_output(torch,NAMES[2],ta,tb,a['indices'],.7)
                for actual,expected in ((f['activated'],pair),(f['normalized'],norm),(f['cosine'],cos)):np.testing.assert_allclose(actual,expected.detach().numpy(),atol=2e-6,rtol=2e-5)
                loss=(pair*torch.tensor(a['dy_pair'])).sum()+(norm*torch.tensor(a['dy_norm'])).sum()+(cos*torch.tensor(a['dy_cos'])).sum();loss.backward()
                a['d_left'][:]=77;a['d_right'][:]=77
                self.assertNotEqual(execute(arena,size-1),0)
                saved=a['dy_cos'].copy();a['dy_cos'][0]=np.nan
                self.assertNotEqual(execute(arena,size),0)
                for key in ('d_left','d_right'):np.testing.assert_array_equal(a[key],77)
                a['dy_cos'][:]=saved;self.assertEqual(execute(arena,size),0)
                for actual,expected in ((a['d_left'],ta.grad.numpy()),(a['d_right'],tb.grad.numpy())):np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-5)
                result=[a[k].copy() for k in ('d_left','d_right')]
                if baseline is None:baseline=result
                elif seed==71:
                    for x,y in zip(result,baseline):np.testing.assert_array_equal(x,y)
            # Miswire one branch join: correct shapes cannot hide lost gradients.
            wrong=deepcopy(bops);wrong[-1]['graph_slots']['inputs']['right']='external:d_right_pair'
            _,we,ws,wa,w,_=build(Path(tmp)/'wrong',wrong,'backward')
            for k in ('left','right','indices','features','dy_pair','dy_norm','dy_cos'):w[k][:]=a[k]
            self.assertEqual(we(wa,ws),0)
            self.assertGreater(float(np.max(np.abs(w['d_right']-a['d_right']))),.01)

if __name__=='__main__':unittest.main(verbosity=2)
