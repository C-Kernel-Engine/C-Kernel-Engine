"""BF16-storage VJPs: explicit FP32-compute semantics, not full BF16 training."""
import ctypes
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from tests import test_v8_head_backward as fp

ROOT=Path(__file__).resolve().parents[1]
BP=ctypes.POINTER(ctypes.c_uint16)
FP=ctypes.POINTER(ctypes.c_float)
IP=ctypes.POINTER(ctypes.c_int32)
S=ctypes.c_size_t
SENTINEL=0x1234
NAMES={'span':'token_span_mean_backward','gelu':'gelu_erf_backward','grouped':'grouped_weighted_rows_backward'}


def encode(x):
    """Independent IEEE RNE conversion; used only with finite oracle values."""
    u=np.asarray(x,dtype='f4').view('u4')
    return ((u.astype('u8')+0x7fff+((u>>16)&1))>>16).astype('u2')


def decode(x):return (np.asarray(x,dtype='u2').astype('u4')<<16).view('f4')


def expected(kind,v,constants):
    c=constants['channels'];dy=decode(v['d_output'][:,:c])
    if kind=='span':
        spans=v['spans'];out=np.zeros((constants['valid_tokens'],c),dtype='f4')
        for t in range(len(out)):
            for k in range(c):out[t,k]=sum(float(dy[i,k])/(b-a) for i,(a,b) in enumerate(spans) if a<=t<b)
        return [encode(out)]
    if kind=='gelu':
        x=decode(v['input'][:,:c])
        out=np.array([[float(dy[r,k])*(.5*(1+math.erf(float(x[r,k])/math.sqrt(2)))+float(x[r,k])*math.exp(-.5*float(x[r,k])**2)/math.sqrt(2*math.pi)) for k in range(c)] for r in range(len(x))],dtype='f4')
        return [encode(out)]
    return [encode(a) for a in fp.grouped_backward(decode(v['options'][:,:c]),decode(v['fields'][:,:c]),v['group_counts'],dy)]


class HeadBackwardBF16Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.libs=[];cls.maps={}
        for opt in ('-O0','-O3'):
            so=Path(cls.temp.name)/(opt+'.so')
            subprocess.run(['cc','-std=c11',opt,'-Wall','-Wextra','-Werror','-shared','-fPIC','-ffp-contract=off',
                '-I',str(ROOT/'include'),*[str(ROOT/'src/kernels'/n) for n in ('head_backward_bf16.c','head_backward.c','grouped_weighted_rows.c')],'-lm','-o',str(so)],check=True)
            lib=ctypes.CDLL(str(so))
            for kind,name in NAMES.items():
                m=json.loads((ROOT/'version/v8/kernel_maps'/f'{name}_checked_bf16.json').read_text());cls.maps[kind]=m
                fn=getattr(lib,m['impl']['function']);fn.restype=ctypes.c_int
                fn.argtypes=[BP if 'uint16_t' in p.get('cast','') else IP if 'int32_t' in p.get('cast','') else FP if 'float' in p.get('cast','') else S for p in m['call_abi']['params']]
            cls.libs.append(lib)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def case(self,kind,counts=(3,1,2),c=5):
        _,oldargs,oldout,old=fp.HeadBackwardTest.case(self,kind,counts,c)
        m=self.maps[kind];params=m['call_abi']['params']
        constants={p['name']:arg for p,arg in zip(params,oldargs) if 'cast' not in p}
        aliases={'input':'x','options':'x','fields':'f','d_output':'dy','group_counts':'counts','spans':'spans'}
        v={p['name']:old[aliases[p['name']]].copy() if p['dtype']=='i32' else encode(old[aliases[p['name']]]) for p in m['inputs']}
        for p,a in zip(m['outputs'],oldout):v[p['name']]=np.full(a.shape,SENTINEL,dtype='u2')
        if kind=='span':n=(constants['span_count']+constants['token_capacity']+constants['valid_tokens'])*c
        elif kind=='gelu':n=4*constants['rows']*c
        else:n=3*constants['option_rows']*c+6*constants['groups']*c+2*constants['option_rows']
        constants['scratch_elements']=n;v['scratch']=np.empty(n,dtype='f4')
        return v,constants

    def args(self,kind,v,k):
        result=[]
        for p in self.maps[kind]['call_abi']['params']:
            name=p['name']
            if 'cast' not in p:result.append(k[name])
            else:result.append(v[name].ctypes.data_as(FP if name=='scratch' else IP if v[name].dtype==np.int32 else BP))
        return result

    def assert_outputs(self,kind,v,k,reference):
        c=k['channels']
        for port,e in zip(self.maps[kind]['outputs'],reference):
            a=v[port['name']]
            np.testing.assert_array_equal(a[:len(e),:c],e)
            np.testing.assert_array_equal(a[:,c:],SENTINEL)
            np.testing.assert_array_equal(a[len(e):],SENTINEL)

    def test_ordered_fp32_compute_and_bf16_storage(self):
        for lib in self.libs:
            for kind in NAMES:
                v,k=self.case(kind);ref=expected(kind,v,k)
                self.assertEqual(getattr(lib,self.maps[kind]['impl']['function'])(*self.args(kind,v,k)),0)
                self.assert_outputs(kind,v,k,ref)

    def test_rejection_capacities_mixed_width_aliases_and_recovery(self):
        for kind,m in self.maps.items():
            v,k=self.case(kind);args=self.args(kind,v,k);fn=getattr(self.libs[-1],m['impl']['function'])
            mutations=[]
            for i,p in enumerate(m['call_abi']['params']):
                name=p['name']
                if 'cast' in p:
                    typ=FP if name=='scratch' else IP if v[name].dtype==np.int32 else BP
                    mutations.extend([(i,None),(i,ctypes.cast(v[name].ctypes.data+1,typ))])
                    if typ in (FP,IP):mutations.append((i,ctypes.cast(v[name].ctypes.data+2,typ)))
                else:
                    mutations.append((i,0))
                    if name.endswith('_elements'):
                        # For row buffers, reject one less than the minimal span.
                        stem=name[:-9];stride=k.get(stem+'_stride')
                        minimum=k[name]-(stride-k['channels'] if stride is not None else 0)
                        mutations.append((i,minimum-1))
                    else:mutations.append((i,2**63))
            first=next(p['name'] for p in m['inputs'] if p['dtype']=='bf16')
            dest=m['outputs'][0]['name']
            for i,p in enumerate(m['call_abi']['params']):
                if p['name'] in [x['name'] for x in m['outputs']]+['scratch']:
                    typ=FP if p['name']=='scratch' else BP
                    for source in (first,dest):
                        if source!=p['name']:mutations.append((i,ctypes.cast(v[source].ctypes.data+4,typ)))
            for i,value in mutations:
                with self.subTest(kind=kind,arg=m['call_abi']['params'][i]['name'],value=str(value)):
                    a=list(args);a[i]=value;self.assertNotEqual(fn(*a),0)
                    for p in m['outputs']:np.testing.assert_array_equal(v[p['name']],SENTINEL)
            self.assertEqual(fn(*args),0);self.assert_outputs(kind,v,k,expected(kind,v,k))

    def test_exact_capacities_and_two_byte_alignment(self):
        for kind,m in self.maps.items():
            v,k=self.case(kind);backings=[]
            for p in m['inputs']+m['outputs']:
                name=p['name']
                if p['dtype']!='bf16':continue
                original=v[name];backing=np.full(original.size+2,SENTINEL,dtype='u2')
                v[name]=backing[1:-1].reshape(original.shape);v[name][:]=original
                self.assertEqual(v[name].ctypes.data%4,2);backings.append(backing)
            for key in list(k):
                if key.endswith('_elements'):
                    stride=k.get(key[:-9]+'_stride')
                    if stride is not None:k[key]-=stride-k['channels']
            self.assertEqual(getattr(self.libs[-1],m['impl']['function'])(*self.args(kind,v,k)),0)
            self.assert_outputs(kind,v,k,expected(kind,v,k))
            for a in backings:np.testing.assert_array_equal(a[[0,-1]],SENTINEL)

    def test_nonfinite_finite_overflow_and_recovery(self):
        for kind,m in self.maps.items():
            v,k=self.case(kind);fn=getattr(self.libs[-1],m['impl']['function']);saved={n:a.copy() for n,a in v.items()}
            for bad in (0x7f80,0xff80,0x7fc1):
                v['d_output'][-1,0]=bad
                self.assertNotEqual(fn(*self.args(kind,v,k)),0)
                for p in m['outputs']:np.testing.assert_array_equal(v[p['name']],SENTINEL)
            v['d_output'][:]=saved['d_output'];v['d_output'][:,:k['channels']]=0x7f7f
            if kind=='span':v['spans'][:]=[0,1]
            elif kind=='gelu':v['input'][:,:k['channels']]=encode(1.)
            else:v['options'][:,:k['channels']]=encode(2.);v['fields'][:,:k['channels']]=0
            self.assertNotEqual(fn(*self.args(kind,v,k)),0)
            for p in m['outputs']:np.testing.assert_array_equal(v[p['name']],SENTINEL)
            for n,a in saved.items():v[n][:]=a
            self.assertEqual(fn(*self.args(kind,v,k)),0);self.assert_outputs(kind,v,k,expected(kind,v,k))

    def test_rne_ties_subnormal_and_bf16_only_overflow(self):
        v,k=self.case('span',c=3);fn=self.libs[-1].ck_token_span_mean_backward_bf16
        v['spans'][:]=[0,1];v['d_output'][:,:3]=0
        v['d_output'][0,:3]=encode([1,1+1/128,2**-133]);v['d_output'][1,:3]=encode([1/256,1/256,2**-133])
        self.assertEqual(fn(*self.args('span',v,k)),0)
        np.testing.assert_array_equal(v['d_input'][0,:3],encode([1,1+2/128,2**-132]))
        v['d_input'][:]=SENTINEL;v['d_output'][:,:3]=0
        v['d_output'][0,0]=0x7f7f;v['d_output'][1,0]=encode(2.**119)
        # The sum is finite FP32, but ties to BF16 infinity: must reject before publication.
        self.assertTrue(np.isfinite(decode(v['d_output'][:,:3]).astype('f8').sum(0).astype('f4')).all())
        self.assertNotEqual(fn(*self.args('span',v,k)),0)
        np.testing.assert_array_equal(v['d_input'],SENTINEL)
        v['d_output'][:,:3]=encode(.25)
        self.assertEqual(fn(*self.args('span',v,k)),0);self.assert_outputs('span',v,k,expected('span',v,k))

    def test_second_gradient_bf16_overflow_is_atomic(self):
        v,k=self.case('grouped',counts=(8,),c=1)
        v['options'][:,0]=encode([1.4140625,-1.4140625]*3+[1.421875,-1.421875])
        v['fields'][:,0]=0;v['d_output'][:,0]=0x7eff
        dx,df=fp.grouped_backward(decode(v['options'][:,:1]),decode(v['fields'][:,:1]),[8],decode(v['d_output'][:,:1]))
        self.assertTrue(np.isfinite(df).all())
        self.assertTrue(np.isfinite(decode(encode(dx))).all())
        self.assertTrue(np.isinf(decode(encode(df))).all())
        fn=self.libs[-1].ck_grouped_weighted_rows_backward_bf16
        self.assertNotEqual(fn(*self.args('grouped',v,k)),0)
        np.testing.assert_array_equal(v['d_options'],SENTINEL)
        np.testing.assert_array_equal(v['d_fields'],SENTINEL)
        v['d_output'][:,0]=encode(.25)
        self.assertEqual(fn(*self.args('grouped',v,k)),0)
        self.assert_outputs('grouped',v,k,expected('grouped',v,k))

    def test_autograd_on_identical_dequantized_operands(self):
        torch=fp.torch_or_skip(self)
        for kind,m in self.maps.items():
            v,k=self.case(kind);c=k['channels'];dy=torch.tensor(decode(v['d_output'][:,:c]))
            if kind=='span':
                x=torch.ones((k['valid_tokens'],c),requires_grad=True);y=torch.stack([x[a:b].mean(0) for a,b in v['spans']]);leaves=[x]
            elif kind=='gelu':
                x=torch.tensor(decode(v['input'][:,:c]),requires_grad=True);y=torch.nn.functional.gelu(x,approximate='none');leaves=[x]
            else:
                x=torch.tensor(decode(v['options'][:,:c]),requires_grad=True);f=torch.tensor(decode(v['fields'][:,:c]),requires_grad=True);y=[];first=0
                for q,n in enumerate(v['group_counts']):
                    r=x[first:first+n];y.append((torch.softmax(r@f[q]/math.sqrt(c),0)[:,None]*r).sum(0));first+=n
                y=torch.stack(y);leaves=[x,f]
            (y*dy).sum().backward()
            self.assertEqual(getattr(self.libs[-1],m['impl']['function'])(*self.args(kind,v,k)),0)
            for port,leaf in zip(m['outputs'],leaves):
                actual=decode(v[port['name']][:len(leaf),:c]);ref=leaf.grad.to(torch.bfloat16).float().numpy()
                np.testing.assert_array_equal(actual,ref)

    def test_group_geometries_singletons_and_fp32_resolution(self):
        from tests.test_v8_numerical_execution_contracts import resolver
        contracts=resolver.load_json(resolver.DEFAULT_CONTRACTS);kernels=resolver.load_kernel_capabilities(contracts=contracts)
        for kind,name in NAMES.items():
            m=self.maps[kind];op=m['op']
            circuit={'name':'bf16_gradient','required_numerical_contracts':{op:{'op':op,'operation_interface':m['operation_interface'],'template_ops':[op],
                'phases':{'backward':{'contract_id':m['numerical_capabilities'][0]['contract_id'],'validation':'validated','evidence':__file__}},
                'checkpoint':{'id':kind,'producer':op,'logical_layout':'row_major','axis_names':['row','channel']}}}}
            plan=resolver.resolve_contract(circuit,contracts,kernels,op,'backward');self.assertEqual(plan['kernel']['id'],name+'_checked_bf16')
        for counts,c in (([1],1),([1,4,2],9),([2,1,1,3],17)):
            v,k=self.case('grouped',counts,c)
            self.assertEqual(self.libs[-1].ck_grouped_weighted_rows_backward_bf16(*self.args('grouped',v,k)),0)
            self.assert_outputs('grouped',v,k,expected('grouped',v,k))

    def test_generated_bf16_backward_chain(self):
        from tests.test_v8_token_span_mean_checked import _circuit
        from tests.v8_checked_graph_test_support import compile_native_graph
        graph=_circuit();graph['name']='bf16_vjp_chain';graph['native_entry']['function']='ck_bf16_vjp_chain';graph['contract']['runtime_invariants']['inference_only']=False
        shapes={'input':[3,5],'options':[3,5],'fields':[2,5],'group_counts':[2],'spans':[3,2],'d_output':[2,5],'d_options':[3,5],'d_fields':[2,5],'d_pooled':[3,5],'d_input':[7,5]}
        graph['activation_buffers']={n:{'shape':s} for n,s in shapes.items()};graph['activation_bindings']={n:n for n in shapes}
        specs=[('grouped',{'options':'options','fields':'fields','group_counts':'group_counts','d_output':'d_output'},{'d_options':'d_options','d_fields':'d_fields'},dict(option_elements=15,option_stride=5,field_elements=10,field_stride=5,count_elements=2,d_output_elements=10,d_output_stride=5,d_options_elements=15,d_options_stride=5,d_fields_elements=10,d_fields_stride=5,scratch_elements=111,groups=2,option_rows=3,channels=5)),
            ('gelu',{'input':'input','d_output':'d_options'},{'d_input':'d_pooled'},dict(input_elements=15,input_stride=5,d_output_elements=15,d_output_stride=5,d_input_elements=15,d_input_stride=5,scratch_elements=60,rows=3,channels=5)),
            ('span',{'d_output':'d_pooled','spans':'spans'},{'d_input':'d_input'},dict(d_output_elements=15,d_output_stride=5,span_elements=6,span_count=3,d_input_elements=35,d_input_stride=5,scratch_elements=80,token_capacity=7,valid_tokens=6,channels=5))]
        ops=[]
        for kind,ins,outs,k in specs:
            m=self.maps[kind];ops.append({'id':kind,'op':m['op'],'kernel':m['id'],'returns_status':True,'graph_slots':{'inputs':{n:'external:'+v for n,v in ins.items()},'outputs':{n:'external:'+v for n,v in outs.items()}},'params':{**{n:k[v] for n,v in m['call_constant_contract']['shape_symbols'].items()},'call_constants':k}})
        graph['block_types']['pool']['body']['ops']=ops
        dtypes={n:'i32' if n in ('spans','group_counts') else 'bf16' for n in shapes}
        source={'config':{'model':graph['name'],'arch':graph['name'],'num_layers':1,'embed_dim':5,'num_heads':1,'num_kv_heads':1,'head_dim':5,'intermediate_size':5,'context_length':7,'max_seq_len':7,'vocab_size':16,'activation_buffer_dtypes':dtypes},'entries':[],'quant_summary':{},'template':graph}
        with tempfile.TemporaryDirectory() as temp:
            layout,calls,_,lib,execute=compile_native_graph(Path(temp),source,graph,mode='backward')
            self.assertEqual([o['function'] for o in calls['operations']],[self.maps[k]['impl']['function'] for k,*_ in specs])
            size=layout['memory']['arena']['total_size'];raw=(ctypes.c_uint8*(size+63))();arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
            entries={x['name']:x for x in layout['memory']['activations']['buffers']}
            for name,shape in shapes.items():
                self.assertEqual(entries[name]['dtype'],dtypes[name])
                self.assertEqual(entries[name]['size'],int(np.prod(shape))*(4 if dtypes[name]=='i32' else 2))
            a={n:np.ndarray(s,dtype='i4' if dtypes[n]=='i32' else 'u2',buffer=arena,offset=entries[n]['abs_offset']) for n,s in shapes.items()}
            results=[]
            for seed in (19,71,19):
                rng=np.random.default_rng(seed)
                a['spans'][:]=[[0,3],[1,5],[2,6]]
                tokens=decode(encode(rng.normal(scale=.5,size=(6,5))))
                a['input'][:]=encode(np.stack([tokens[b:e].astype('f8').mean(0) for b,e in a['spans']]))
                a['options'][:]=encode([[float(x)*.5*(1+math.erf(float(x)/math.sqrt(2))) for x in row] for row in decode(a['input'])])
                for n in ('fields','d_output'):a[n][:]=encode(rng.normal(scale=.5,size=shapes[n]))
                a['group_counts'][:]=[2,1];a['d_input'][:]=SENTINEL
                a['spans'][-1,1]=7
                self.assertNotEqual(execute(arena,size),0);np.testing.assert_array_equal(a['d_input'],SENTINEL)
                a['spans'][-1,1]=6
                self.assertNotEqual(execute(arena,size-1),0);np.testing.assert_array_equal(a['d_input'],SENTINEL)
                self.assertEqual(execute(arena,size),0)
                refs={n:v.copy() for n,v in a.items()}
                for kind,ins,outs,k in specs:
                    v={n:refs[r] for n,r in ins.items()}
                    for (n,dest),e in zip(outs.items(),expected(kind,v,k)):
                        np.testing.assert_array_equal(a[dest][:len(e)],e);refs[dest][:len(e)]=e
                np.testing.assert_array_equal(a['d_input'][-1],SENTINEL);results.append(a['d_input'].copy())
            np.testing.assert_array_equal(results[0],results[2])


if __name__=='__main__':unittest.main()
