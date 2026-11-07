"""Checked VJPs and a generated pooling/GELU/grouped-pooling gradient chain.

This explicitly authored backward circuit is not automatic differentiation or
complete Clef head training. PyTorch is an optional independent oracle only.
"""
import ctypes
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int32)
S = ctypes.c_size_t


def torch_or_skip(test):
    try:
        import torch
        return torch
    except ImportError:
        test.skipTest('independent PyTorch autograd unavailable')


def rows(rng, n, c, padding=2):
    a = np.full((n, c + padding), np.nan, dtype='f4')
    a[:, :c] = rng.normal(scale=.7, size=(n, c))
    return a


def grouped_forward(options, fields, counts):
    from tests.test_v8_grouped_weighted_rows import ordered_reference
    return ordered_reference(options, fields, counts)


def grouped_backward(options, fields, counts, dy):
    """Independent scalar evaluation of the declared rounding/reduction order."""
    dx = np.empty_like(options); df = np.empty_like(fields); first = 0
    for q, n in enumerate(counts):
        x = options[first:first+n]; c = x.shape[1]
        z = [np.float32(sum(float(a)*float(b) for a,b in zip(row,fields[q])) / math.sqrt(c)) for row in x]
        ex = [np.float32(math.exp(float(v)-float(max(z)))) for v in z]
        p = [np.float32(float(v)/sum(float(e) for e in ex)) for v in ex]
        dp = [np.float32(sum(float(a)*float(b) for a,b in zip(row,dy[q]))) for row in x]
        weighted = sum(float(a)*float(b) for a,b in zip(p,dp))
        dz = [np.float32(float(a)*(float(b)-weighted)/math.sqrt(c)) for a,b in zip(p,dp)]
        for i in range(n):
            for k in range(c):
                dx[first+i,k] = np.float32(np.float32(p[i]*dy[q,k])+np.float32(dz[i]*fields[q,k]))
        for k in range(c):
            df[q,k] = np.float32(sum(float(dz[i])*float(x[i,k]) for i in range(n)))
        first += n
    return dx,df


class HeadBackwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.libs=[]
        for opt in ('-O0','-O3'):
            path=Path(cls.temp.name)/(opt+'.so')
            subprocess.run(['cc','-std=c11',opt,'-Wall','-Wextra','-Werror','-shared','-fPIC',
                '-ffp-contract=off','-I',str(ROOT/'include'),str(ROOT/'src/kernels/head_backward.c'),
                str(ROOT/'src/kernels/grouped_weighted_rows.c'),'-lm','-o',str(path)],check=True)
            lib=ctypes.CDLL(str(path))
            lib.ck_token_span_mean_backward_f32.argtypes=[FP,S,S,IP,S,S,FP,S,S,FP,S,S,S,S]
            lib.ck_gelu_erf_backward_f32.argtypes=[FP,S,S,FP,S,S,FP,S,S,FP,S,S,S]
            lib.ck_grouped_weighted_rows_backward_f32.argtypes=[FP,S,S,FP,S,S,IP,S,FP,S,S,FP,S,S,FP,S,S,FP,S,S,S,S]
            cls.libs.append(lib)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def case(self, kind, counts=(3,1,2), c=5):
        rng=np.random.default_rng(401); g=len(counts); o=sum(counts)
        def pack(a): return [a.ctypes.data_as(FP), a.size, a.shape[1]]
        if kind=='span':
            dy=rows(rng,3,c); spans=np.array([[0,3],[1,5],[2,4]],dtype='i4')
            dx=np.full((7,c+3),-91,dtype='f4'); scratch=np.empty(6*c,dtype='f4')
            args=pack(dy)+[spans.ctypes.data_as(IP),spans.size,3]+pack(dx)+[scratch.ctypes.data_as(FP),scratch.size,7,6,c]
            return 'ck_token_span_mean_backward_f32',args,[dx],dict(dy=dy,spans=spans,scratch=scratch,c=c)
        if kind=='gelu':
            x=rows(rng,4,c); dy=rows(rng,4,c,3);dx=np.full((4,c+1),-91,dtype='f4');scratch=np.empty(4*c,dtype='f4')
            args=pack(x)+pack(dy)+pack(dx)+[scratch.ctypes.data_as(FP),scratch.size,4,c]
            return 'ck_gelu_erf_backward_f32',args,[dx],dict(x=x,dy=dy,scratch=scratch,c=c)
        x=rows(rng,o,c); f=rows(rng,g,c,3); dy=rows(rng,g,c,1); cnt=np.array(counts,dtype='i4')
        dx=np.full((o,c+4),-91,dtype='f4');df=np.full((g,c+2),-91,dtype='f4')
        scratch=np.empty(2*o+3*g*c+o*c,dtype='f4')
        args=pack(x)+pack(f)+[cnt.ctypes.data_as(IP),g]+pack(dy)+pack(dx)+pack(df)+[scratch.ctypes.data_as(FP),scratch.size,g,o,c]
        return 'ck_grouped_weighted_rows_backward_f32',args,[dx,df],dict(x=x,f=f,dy=dy,counts=cnt,scratch=scratch,c=c)

    def assert_oracle(self,kind,case):
        _,_,out,v=case;c=v['c'];dy=v['dy'][:,:c]
        if kind=='span':
            expected=np.zeros((6,c),dtype='f4')
            for t in range(6):
                for k in range(c):
                    expected[t,k]=sum(float(dy[i,k])/(b-a) for i,(a,b) in enumerate(v['spans']) if a<=t<b)
            np.testing.assert_array_equal(out[0][:6,:c],expected)
            np.testing.assert_array_equal(out[0][6],-91)
        elif kind=='gelu':
            expected=np.array([[float(dy[r,k])*(.5*(1+math.erf(float(v['x'][r,k])/math.sqrt(2)))+float(v['x'][r,k])*math.exp(-.5*float(v['x'][r,k])**2)/math.sqrt(2*math.pi)) for k in range(c)] for r in range(4)],dtype='f4')
            np.testing.assert_allclose(out[0][:,:c],expected,atol=1e-7,rtol=1e-7)
        else:
            for a,e in zip(out,grouped_backward(v['x'][:,:c],v['f'][:,:c],v['counts'],dy)):
                np.testing.assert_array_equal(a[:,:c],e)
        for a in out:np.testing.assert_array_equal(a[:,c:],-91)

    def test_backward_contract_resolution_and_phase_rejection(self):
        from tests.test_v8_numerical_execution_contracts import resolver
        contracts=resolver.load_json(resolver.DEFAULT_CONTRACTS)
        kernels=resolver.load_kernel_capabilities(contracts=contracts)
        for name in ('token_span_mean_backward','gelu_erf_backward','grouped_weighted_rows_backward'):
            op=name+'_checked'
            circuit={'name':'gradient_contract','required_numerical_contracts':{op:{
                'op':op,'template_ops':[op],
                'phases':{'backward':{'contract_id':name+'_ordered_fp64_f32_v1','validation':'validated','evidence':__file__}},
                'checkpoint':{'id':name,'producer':op,'logical_layout':'row_major','axis_names':['row','channel']}}}}
            plan=resolver.resolve_contract(circuit,contracts,kernels,op,'backward')
            self.assertEqual(plan['kernel']['id'],op+'_f32')
            for phase in ('prefill','decode'):
                with self.assertRaises(resolver.ContractError):resolver.resolve_contract(circuit,contracts,kernels,op,phase)

    def test_ordered_oracles_and_strided_padding(self):
        for lib in self.libs:
            for kind in ('span','gelu','grouped'):
                with self.subTest(kind=kind,lib=lib._name):
                    case=self.case(kind);name,args,_,_=case
                    self.assertEqual(getattr(lib,name)(*args),0);self.assert_oracle(kind,case)

    def test_reject_before_parity_capacity_alignment_overlap_and_reuse(self):
        for kind, pointers, capacities,strides,scalars in (
            ('span',(0,3,6,9),(1,4,7,10),(2,8),(5,11,12,13)),
            ('gelu',(0,3,6,9),(1,4,7,10),(2,5,8),(11,12)),
            ('grouped',(0,3,6,8,11,14,17),(1,4,7,9,12,15,18),(2,5,10,13,16),(19,20,21))):
            case=self.case(kind);name,args,out,v=case;fn=getattr(self.libs[-1],name)
            mutations=[(i,None) for i in pointers]+[(i,0) for i in capacities+strides+scalars]
            mutations += [(i,2**63) for i in scalars]
            # Minimal physical row spans (last padding is not accessed).
            for i in capacities:
                minimum=args[i]
                if i+1 in strides:
                    minimum=args[i]-args[i+1]+v['c']
                mutations.append((i,minimum-1))
            for i in pointers:
                ptr=args[i];typ=IP if kind=='span' and i==3 or kind=='grouped' and i==6 else FP
                mutations.append((i,ctypes.cast(ctypes.cast(ptr,ctypes.c_void_p).value+1,typ)))
            writable=(6,9) if kind!='grouped' else (11,14,17)
            for i in writable:
                mutations.append((i,ctypes.cast(ctypes.cast(args[0],ctypes.c_void_p).value+4,FP)))
            mutations.append((writable[-1],ctypes.cast(ctypes.cast(args[writable[0]],ctypes.c_void_p).value+4,FP)))
            if kind=='grouped':
                mutations.append((14,ctypes.cast(ctypes.cast(args[11],ctypes.c_void_p).value+4,FP)))
            for i,value in mutations:
                with self.subTest(kind=kind,arg=i,value=str(value)):
                    altered=list(args);altered[i]=value
                    self.assertNotEqual(fn(*altered),0)
                    for a in out:np.testing.assert_array_equal(a,-91)
            exact=list(args)
            for i in capacities:
                if i+1 in strides:exact[i]=args[i]-args[i+1]+v['c']
            self.assertEqual(fn(*exact),0);self.assert_oracle(kind,case)

    def test_late_failure_leaves_all_gradients_unchanged(self):
        for kind in ('span','gelu','grouped'):
            case=self.case(kind);name,args,out,v=case;fn=getattr(self.libs[-1],name)
            saved=v['dy'].copy();v['dy'][-1,0]=np.nan
            self.assertNotEqual(fn(*args),0)
            for a in out:np.testing.assert_array_equal(a,-91)
            v['dy'][:]=saved
            key='spans' if kind=='span' else 'counts' if kind=='grouped' else 'x'
            original=v[key].copy();v[key].flat[-1 if kind!='gelu' else 0]=-1 if kind!='gelu' else np.inf
            self.assertNotEqual(fn(*args),0)
            for a in out:np.testing.assert_array_equal(a,-91)
            v[key][:]=original
            self.assertEqual(fn(*args),0);self.assert_oracle(kind,case)

    def test_finite_gradient_overflow_rejection_and_recovery(self):
        for kind in ('span','gelu','grouped'):
            case=self.case(kind);name,args,out,v=case;fn=getattr(self.libs[-1],name)
            saved={k:a.copy() for k,a in v.items() if isinstance(a,np.ndarray)}
            v['dy'][:,:v['c']]=np.finfo('f4').max
            if kind=='span':v['spans'][:]=[0,1]
            elif kind=='gelu':v['x'][:,:v['c']]=1
            else:v['x'][:,:v['c']]=2;v['f'][:,:v['c']]=0
            self.assertNotEqual(fn(*args),0)
            for a in out:np.testing.assert_array_equal(a,-91)
            for k,a in saved.items():v[k][:]=a
            self.assertEqual(fn(*args),0);self.assert_oracle(kind,case)

    def test_group_geometries_and_singleton_score_gradient(self):
        for counts,c in (([1],1),([1,4,2],9),([2,1,1,3],17)):
            case=self.case('grouped',counts,c);name,args,out,_=case
            self.assertEqual(getattr(self.libs[-1],name)(*args),0);self.assert_oracle('grouped',case)
            for q,n in enumerate(counts):
                if n==1:np.testing.assert_array_equal(out[1][q,:c],0)

    def test_independent_autograd_all_input_gradients(self):
        torch=torch_or_skip(self)
        for kind in ('span','gelu','grouped'):
            case=self.case(kind);name,args,out,v=case;c=v['c']
            dy=torch.tensor(v['dy'][:,:c])
            if kind=='span':
                x=torch.tensor(np.random.default_rng(3).normal(size=(6,c)),dtype=torch.float32,requires_grad=True)
                y=torch.stack([x[a:b].mean(0) for a,b in v['spans']]);leaves=[x]
            elif kind=='gelu':
                x=torch.tensor(v['x'][:,:c],requires_grad=True);y=torch.nn.functional.gelu(x,approximate='none');leaves=[x]
            else:
                x=torch.tensor(v['x'][:,:c],requires_grad=True);f=torch.tensor(v['f'][:,:c],requires_grad=True)
                y=[];first=0
                for q,n in enumerate(v['counts']):
                    r=x[first:first+n];y.append((torch.softmax(r@f[q]/math.sqrt(c),0)[:,None]*r).sum(0));first+=n
                y=torch.stack(y);leaves=[x,f]
            (y*dy).sum().backward();self.assertEqual(getattr(self.libs[-1],name)(*args),0)
            for index,(a,t) in enumerate(zip(out,leaves)):
                actual=a[:len(t),:c];expected=t.grad.numpy()
                error=actual.astype('f8')-expected.astype('f8')
                print(f'{kind} gradient {index}: max_abs={np.max(np.abs(error)):.9g} rmse={np.sqrt(np.mean(error**2)):.9g}')
                np.testing.assert_allclose(actual,expected,atol=4e-7,rtol=3e-6)

    def test_grouped_finite_differences_both_gradient_routes(self):
        case=self.case('grouped',c=3);name,args,out,v=case
        self.assertEqual(getattr(self.libs[-1],name)(*args),0)
        def loss():return float(np.sum(grouped_forward(v['x'][:,:3],v['f'][:,:3],v['counts']).astype('f8')*v['dy'][:,:3]))
        for key,grad in zip(('x','f'),out):
            for index in ((0,0),(len(v[key])-1,2)):
                old=v[key][index];h=.002
                v[key][index]=old+h;plus=loss();v[key][index]=old-h;minus=loss();v[key][index]=old
                self.assertAlmostEqual(float(grad[index]),(plus-minus)/(2*h),delta=1e-4)

    def test_generated_backward_chain_and_rejection_recovery(self):
        from tests.test_v8_token_span_mean_checked import _circuit
        from tests.v8_checked_graph_test_support import compile_native_graph
        c=5;t=7;o=3;g=2
        graph=_circuit();graph['name']='head_backward_synthetic';graph['contract']['runtime_invariants']['inference_only']=False
        graph['native_entry']['function']='ck_test_head_backward'
        shapes={'tokens':[t,c],'summary':[g,c],'pooled':[o,c],'options':[o,c],'fields':[g,c],'counts':[g],'spans':[o,2],'dy':[g,c],
                'd_options':[o,c],'d_fields':[g,c],'d_pooled':[o,c],'d_tokens':[t,c]}
        graph['activation_buffers']={k:{'shape':v} for k,v in shapes.items()};graph['activation_bindings']={k:k for k in shapes}
        specs=[('grouped_weighted_rows_backward',{'options':'options','fields':'fields','group_counts':'counts','d_output':'dy'}, {'d_options':'d_options','d_fields':'d_fields'},
            dict(option_elements=o*c,option_stride=c,field_elements=g*c,field_stride=c,count_elements=g,d_output_elements=g*c,d_output_stride=c,d_options_elements=o*c,d_options_stride=c,d_fields_elements=g*c,d_fields_stride=c,scratch_elements=2*o+3*g*c+o*c,groups=g,option_rows=o,channels=c)),
            ('gelu_erf_backward',{'input':'pooled','d_output':'d_options'},{'d_input':'d_pooled'},dict(input_elements=o*c,input_stride=c,d_output_elements=o*c,d_output_stride=c,d_input_elements=o*c,d_input_stride=c,scratch_elements=o*c,rows=o,channels=c)),
            ('token_span_mean_backward',{'d_output':'d_pooled','spans':'spans'},{'d_input':'d_tokens'},dict(d_output_elements=o*c,d_output_stride=c,span_elements=o*2,span_count=o,d_input_elements=t*c,d_input_stride=c,scratch_elements=(t-1)*c,token_capacity=t,valid_tokens=t-1,channels=c))]
        ops=[]
        for name,ins,outs,constants in specs:
            kid=name+'_checked_f32';m=json.loads((ROOT/'version/v8/kernel_maps'/f'{kid}.json').read_text())
            ops.append({'id':name,'op':m['op'],'kernel':kid,'returns_status':True,
                'graph_slots':{'inputs':{k:'external:'+v for k,v in ins.items()},'outputs':{k:'external:'+v for k,v in outs.items()}},
                'params':{**{k:constants[v] for k,v in m['call_constant_contract']['shape_symbols'].items()},'call_constants':constants}})
        graph['block_types']['pool']['body']['ops']=ops
        source={'config':{'model':graph['name'],'arch':graph['name'],'num_layers':1,'embed_dim':c,'num_heads':1,'num_kv_heads':1,'head_dim':c,'intermediate_size':c,'context_length':t,'max_seq_len':t,'vocab_size':16,'activation_buffer_dtypes':{'counts':'i32','spans':'i32'}},'entries':[],'quant_summary':{},'template':graph}
        # Forward and VJP are separately phase-checked native entries. Python
        # only supplies inputs/invokes entries; it does not compute intermediates.
        forward=deepcopy(graph);forward['native_entry']['function']='ck_test_head_forward'
        fspecs=[('token_span_mean_checked_f32','token_span_mean_checked',
            {'input':'tokens','spans':'spans'},{'output':'pooled'},
            dict(input_elements=t*c,input_stride=c,token_capacity=t,valid_tokens=t-1,channels=c,span_elements=2*o,span_count=o,output_elements=o*c,output_stride=c)),
            ('gelu_rows_erf_checked_f32','gelu',{'input':'pooled'},{'output':'options'},
            dict(gelu_input_elements=o*c,gelu_input_stride=c,gelu_output_elements=o*c,gelu_output_stride=c,gelu_scratch_elements=o*c,gelu_rows=o,gelu_channels=c)),
            ('grouped_softmax_weighted_rows_checked_f32','grouped_softmax_weighted_rows_checked',
            {'options':'options','fields':'fields','group_counts':'counts'},{'output':'summary'},
            dict(option_elements=o*c,option_stride=c,field_elements=g*c,field_stride=c,count_elements=g,output_elements=g*c,output_stride=c,scratch_elements=o+g*c,groups=g,option_rows=o,channels=c))]
        fops=[]
        for kid,op,ins,outs,constants in fspecs:
            m=json.loads((ROOT/'version/v8/kernel_maps'/f'{kid}.json').read_text())
            fops.append({'id':op,'op':op,'kernel':kid,'returns_status':True,
                'graph_slots':{'inputs':{k:'external:'+v for k,v in ins.items()},'outputs':{k:'external:'+v for k,v in outs.items()}},
                'params':{**{k:constants[v] for k,v in m['call_constant_contract']['shape_symbols'].items()},'call_constants':constants}})
        forward['block_types']['pool']['body']['ops']=fops
        def load(temp,graph,mode):
            temp.mkdir();src=deepcopy(source);src['template']=graph
            layout,calls,_,lib,execute=compile_native_graph(temp,src,graph,mode=mode)
            self.assertEqual(calls['errors'],[])
            size=layout['memory']['arena']['total_size'];raw=(ctypes.c_uint8*(size+63))();arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
            entries={x['name']:x for x in layout['memory']['activations']['buffers']}
            a={k:np.ndarray(v,dtype='i4' if k in ('spans','counts') else 'f4',buffer=arena,offset=entries[k]['abs_offset']) for k,v in shapes.items()}
            return calls,lib,execute,size,arena,a
        with tempfile.TemporaryDirectory() as temp:
            calls,lib,execute,size,arena,a=load(Path(temp)/'backward',graph,'backward')
            fcalls,flib,fexecute,fsize,farena,fa=load(Path(temp)/'forward',forward,'prefill')
            self.assertEqual([op['function'] for op in calls['operations']],['ck_'+n+'_f32' for n,*_ in specs])
            from version.v8.scripts import codegen_checked_calls_v8 as emitter
            for phase in ('init','prefill','decode','training','bogus'):
                broken=deepcopy(calls);broken['mode']=phase
                with self.assertRaises(emitter.CheckedCallCodegenError):emitter.emit_checked_calls(broken,ROOT)
            broken=deepcopy(fcalls);broken['mode']='backward'
            with self.assertRaises(emitter.CheckedCallCodegenError):emitter.emit_checked_calls(broken,ROOT)
            spans=np.array([[0,3],[1,5],[2,6]],dtype='i4');baselines=[]
            for seed in (71,92,71):
                rng=np.random.default_rng(seed);x=rng.normal(size=(t-1,c)).astype('f4');f=rng.normal(size=(g,c)).astype('f4');dy=rng.normal(size=(g,c)).astype('f4')
                fa['tokens'][:t-1]=x;fa['tokens'][-1]=np.nan;fa['fields'][:]=f;fa['spans'][:]=spans;fa['counts'][:]=[2,1]
                self.assertEqual(fexecute(farena,fsize),0)
                from tests.test_v8_token_span_mean_checked import sequential_fp64_span_mean
                pooled=sequential_fp64_span_mean(x,spans)
                options=np.array([[float(v)*.5*(1+math.erf(float(v)/math.sqrt(2))) for v in row] for row in pooled],dtype='f4')
                np.testing.assert_array_equal(fa['pooled'],pooled)
                np.testing.assert_allclose(fa['options'],options,atol=1e-7,rtol=1e-7)
                np.testing.assert_allclose(fa['summary'],grouped_forward(options,f,[2,1]),atol=1e-7,rtol=1e-7)
                # Only native-produced intermediates enter the generated VJP.
                for k in ('pooled','options','fields','spans','counts'):a[k][:]=fa[k]
                a['dy'][:]=dy
                a['d_tokens'][:]=-91;a['d_fields'][:]=-91;a['counts'][1]=0
                self.assertNotEqual(execute(arena,size),0);np.testing.assert_array_equal(a['d_tokens'],-91);np.testing.assert_array_equal(a['d_fields'],-91)
                a['counts'][1]=1
                self.assertNotEqual(execute(arena,size-1),0);np.testing.assert_array_equal(a['d_tokens'],-91)
                # A last-operation failure must propagate and leave the final
                # gradient untouched; earlier intermediate VJPs may have run.
                a['spans'][-1,1]=t
                self.assertNotEqual(execute(arena,size),0);np.testing.assert_array_equal(a['d_tokens'],-91)
                a['spans'][:]=spans
                self.assertEqual(execute(arena,size),0)
                dop,df=grouped_backward(fa['options'],f,[2,1],dy)
                dp=np.array([[float(dop[r,k])*(.5*(1+math.erf(float(pooled[r,k])/math.sqrt(2)))+float(pooled[r,k])*math.exp(-.5*float(pooled[r,k])**2)/math.sqrt(2*math.pi)) for k in range(c)] for r in range(o)],dtype='f4')
                dx=np.array([[sum(float(dp[i,k])/(e-b) for i,(b,e) in enumerate(spans) if b<=r<e) for k in range(c)] for r in range(t-1)],dtype='f4')
                np.testing.assert_array_equal(a['d_options'],dop)
                np.testing.assert_allclose(a['d_pooled'],dp,atol=1e-7,rtol=1e-7)
                np.testing.assert_allclose(a['d_tokens'][:t-1],dx,atol=1e-7,rtol=1e-7)
                np.testing.assert_array_equal(a['d_fields'],df)
                np.testing.assert_array_equal(a['d_tokens'][-1],-91)
                baselines.append((a['d_tokens'].copy(),a['d_fields'].copy()))
            for first,last in zip(baselines[0],baselines[2]):np.testing.assert_array_equal(first,last)


if __name__=='__main__':unittest.main()
