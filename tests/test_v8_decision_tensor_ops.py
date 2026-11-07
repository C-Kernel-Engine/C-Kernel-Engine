"""Model-neutral checked row/scalar providers needed for decision-head scoring."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OPS=['broadcast_add_rows','indexed_pair_features','indexed_cosine_rows',
     'scalar_exp_clamped','scalar_sigmoid','scale_vector']
FP=ctypes.POINTER(ctypes.c_float)
IP=ctypes.POINTER(ctypes.c_int32)

class DecisionTensorOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();path=Path(cls.temp.name)/'ops.so'
        subprocess.run(['cc','-std=c11','-O3','-Wall','-Wextra','-Werror','-shared','-fPIC',
            '-ffp-contract=off','-I',str(ROOT/'include'),str(ROOT/'src/kernels/decision_tensor_ops.c'),
            '-lm','-o',str(path)],check=True)
        cls.lib=ctypes.CDLL(str(path));cls.maps={}
        for name in OPS:
            m=json.loads((ROOT/f'version/v8/kernel_maps/{name}_checked_f32.json').read_text())
            fn=getattr(cls.lib,m['impl']['function']);types=[]
            for p in m['call_abi']['params']:
                types.append(IP if 'int32_t' in p.get('cast','') else FP if '*' in p.get('cast','')
                             else ctypes.c_float if p['name'] in ('epsilon','maximum') else ctypes.c_size_t)
            fn.argtypes=types;fn.restype=ctypes.c_int;cls.maps[name]=(m,fn)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def run_op(self,name,arrays,**constants):
        m,fn=self.maps[name];args=[]
        for p in m['call_abi']['params']:
            n=p['name']
            if n in constants:args.append(constants[n])
            elif n in arrays:args.append(arrays[n].ctypes.data_as(IP if arrays[n].dtype==np.int32 else FP))
            elif n.endswith('_elements'):args.append(arrays[n[:-9]].size)
            else:raise AssertionError('missing argument '+n)
        return fn(*args)

    def test_broadcast_scale_and_padding(self):
        left=np.arange(12,dtype='f4').reshape(3,4)/5
        right=np.linspace(-.3,.7,4,dtype='f4')
        arrays=dict(left=left,right=right,output=np.full(15,-13,dtype='f4'),scratch=np.zeros(15,dtype='f4'))
        self.assertEqual(self.run_op('broadcast_add_rows',arrays,rows=3,channels=4),0)
        np.testing.assert_array_equal(arrays['output'][:12].reshape(3,4),left+right)
        np.testing.assert_array_equal(arrays['output'][12:],-13)
        arrays=dict(input=left.ravel(),gain=np.array([-.7],dtype='f4'),output=np.full(15,-13,dtype='f4'),scratch=np.zeros(12,dtype='f4'))
        self.assertEqual(self.run_op('scale_vector',arrays,rows=12),0)
        np.testing.assert_array_equal(arrays['output'][:12],left.ravel()*np.float32(-.7))
        np.testing.assert_array_equal(arrays['output'][12:],-13)

    def pair_arrays(self,features=False):
        rng=np.random.default_rng(410);n=60 if features else 3
        return dict(left=rng.normal(size=(2,5)).astype('f4'),right=rng.normal(size=(3,5)).astype('f4'),
            indices=np.array([1,0,1],dtype='i4'),output=np.full(n+2,-13,dtype='f4'),scratch=np.zeros(n,dtype='f4'))

    def test_pair_features_and_cosine_against_torch(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:self.skipTest('independent PyTorch comparison unavailable')
        for features in (False,True):
            arrays=self.pair_arrays(features);name='indexed_pair_features' if features else 'indexed_cosine_rows'
            extra={} if features else {'epsilon':1e-8}
            self.assertEqual(self.run_op(name,arrays,groups=2,rows=3,channels=5,**extra),0)
            left=torch.from_numpy(arrays['left'][arrays['indices']]);right=torch.from_numpy(arrays['right'])
            expected=torch.cat((left,right,left*right,(left-right).abs()),-1) if features else F.cosine_similarity(left,right,dim=-1,eps=1e-8)
            np.testing.assert_allclose(arrays['output'][:-2],expected.numpy().ravel(),atol=2e-7,rtol=0)
            np.testing.assert_array_equal(arrays['output'][-2:],-13)

    def test_cosine_zero_small_norms_and_epsilon(self):
        arrays=self.pair_arrays()
        arrays['left'][:]=0;arrays['right'][:]=1e-14
        for eps in (1e-12,1e-8):
            self.assertEqual(self.run_op('indexed_cosine_rows',arrays,groups=2,rows=3,channels=5,epsilon=eps),0)
            np.testing.assert_array_equal(arrays['output'][:3],0)
        arrays['left'][:]=1e-14;arrays['right'][:]=2e-14
        for eps in (1e-12,1e-8):
            self.assertEqual(self.run_op('indexed_cosine_rows',arrays,groups=2,rows=3,channels=5,epsilon=eps),0)
            expected=np.float32(5*float(np.float32(1e-14)/np.float32(eps))*float(np.float32(2e-14)/np.float32(eps)))
            np.testing.assert_allclose(arrays['output'][:3],expected,rtol=2e-7,atol=0)
        arrays['output'][:]=-13
        for eps in (0,-1,float('nan'),float('inf')):
            self.assertNotEqual(self.run_op('indexed_cosine_rows',arrays,groups=2,rows=3,channels=5,epsilon=eps),0)
            np.testing.assert_array_equal(arrays['output'],-13)

    def test_scalar_transforms_bounds_and_recovery(self):
        a=dict(input=np.array([0],dtype='f4'),output=np.full(3,-13,dtype='f4'))
        for x in (-100.,-.4,0.,.5,100.):
            a['input'][0]=x
            self.assertEqual(self.run_op('scalar_sigmoid',a),0)
            expected=1/(1+np.exp(-x))
            self.assertAlmostEqual(float(a['output'][0]),expected,places=7)
            self.assertEqual(self.run_op('scalar_exp_clamped',a,maximum=np.log(100.)),0)
            np.testing.assert_allclose(a['output'][0],np.exp(min(x,np.float32(np.log(100.)))),rtol=2e-7,atol=np.nextafter(np.float32(0),np.float32(1)))
            np.testing.assert_array_equal(a['output'][1:],-13)
        a['output'][:]=-13;a['input'][0]=1000
        self.assertNotEqual(self.run_op('scalar_exp_clamped',a,maximum=1000.),0)
        np.testing.assert_array_equal(a['output'],-13)
        for name in ('scalar_sigmoid','scalar_exp_clamped'):
            kw={'maximum':1.} if name=='scalar_exp_clamped' else {}
            for val in (np.nan,np.inf,-np.inf):
                a['input'][0]=val
                self.assertNotEqual(self.run_op(name,a,**kw),0)
                np.testing.assert_array_equal(a['output'],-13)
            a['input'][0]=.25
            self.assertNotEqual(self.run_op(name,a,output=a['input'].ctypes.data_as(FP),**kw),0)
            self.assertEqual(a['input'][0],.25)
            self.assertEqual(self.run_op(name,a,**kw),0);a['output'][:]=-13

    def test_rejection_before_success_all_provider_buffers(self):
        for name in OPS:
            if name.startswith('indexed_'):
                a=self.pair_arrays(name=='indexed_pair_features');kw=dict(groups=2,rows=3,channels=5)
                if 'cosine' in name:kw['epsilon']=1e-8
            elif name=='broadcast_add_rows':
                a=dict(left=np.ones(15,dtype='f4'),right=np.ones(5,dtype='f4'),output=np.full(15,-13,dtype='f4'),scratch=np.zeros(15,dtype='f4'));kw=dict(rows=3,channels=5)
            elif name=='scale_vector':
                a=dict(input=np.ones(15,dtype='f4'),gain=np.ones(1,dtype='f4'),output=np.full(15,-13,dtype='f4'),scratch=np.zeros(15,dtype='f4'));kw=dict(rows=15)
            else:
                a=dict(input=np.ones(1,dtype='f4'),output=np.full(1,-13,dtype='f4'));kw={'maximum':1.} if 'exp' in name else {}
            for key in a:
                with self.subTest(name=name,buffer=key):
                    self.assertNotEqual(self.run_op(name,a,**dict(kw,**{key:None})),0)
                    self.assertNotEqual(self.run_op(name,a,**dict(kw,**{key+'_elements':0})),0)
                    ptr=ctypes.cast(a[key].ctypes.data+1,IP if key=='indices' else FP)
                    self.assertNotEqual(self.run_op(name,a,**dict(kw,**{key:ptr})),0)
                    np.testing.assert_array_equal(a['output'],-13)
            for key in a:
                if key in ('output','scratch','indices'):continue
                saved=a[key].copy();a[key].flat[0]=np.nan
                self.assertNotEqual(self.run_op(name,a,**kw),0)
                np.testing.assert_array_equal(a['output'],-13);a[key][:]=saved
            for dim in ('groups','rows','channels'):
                if dim in kw:
                    self.assertNotEqual(self.run_op(name,a,**dict(kw,**{dim:0})),0)
                    self.assertNotEqual(self.run_op(name,a,**dict(kw,**{dim:2**64-1})),0)
                    np.testing.assert_array_equal(a['output'],-13)
            if 'scratch' in a:
                self.assertNotEqual(self.run_op(name,a,**dict(kw,scratch_elements=a['scratch'].size-1)),0)
                self.assertNotEqual(self.run_op(name,a,**dict(kw,scratch=ctypes.cast(a['output'].ctypes.data+4,FP))),0)
                np.testing.assert_array_equal(a['output'],-13)
            if 'indices' in a:
                a['indices'][1]=2
                self.assertNotEqual(self.run_op(name,a,**kw),0)
                np.testing.assert_array_equal(a['output'],-13);a['indices'][1]=0
            self.assertEqual(self.run_op(name,a,**kw),0)

    def test_late_finite_overflow_is_atomic(self):
        a=self.pair_arrays(True);a['left'][1,:]=np.finfo('f4').max;a['right'][2,:]=np.finfo('f4').max
        self.assertNotEqual(self.run_op('indexed_pair_features',a,groups=2,rows=3,channels=5),0)
        np.testing.assert_array_equal(a['output'],-13)
        a=dict(input=np.array([1,np.finfo('f4').max],dtype='f4'),gain=np.array([2],dtype='f4'),output=np.full(2,-13,dtype='f4'),scratch=np.zeros(2,dtype='f4'))
        self.assertNotEqual(self.run_op('scale_vector',a,rows=2),0)
        np.testing.assert_array_equal(a['output'],-13)
        a['gain'][0]=.5
        self.assertEqual(self.run_op('scale_vector',a,rows=2),0)
        np.testing.assert_array_equal(a['output'],a['input']*np.float32(.5))

    def test_model_neutral_generated_composition(self):
        from tests.v8_checked_graph_test_support import compile_native_graph
        from tests.test_v8_checked_op_constants import write_synthetic_weights
        shapes={'left':[2,3],'right':[3,3],'offset':[3],'indices':[3],
                'shifted':[2,3],'features':[3,12],'cosine':[3],'exp':[1],'sigmoid':[1],'scaled':[3]}
        operations=[]
        def operation(name,kind,inputs,output,params,weight=False):
            op={'id':name,'op':kind+'_checked','kernel':kind+'_checked_f32','returns_status':True,
                'graph_slots':{'inputs':inputs,'outputs':{'output':output}},'params':params}
            if weight:op['weight_refs']={'input':name+'.weight'}
            operations.append(op)
        operation('shift','broadcast_add_rows',{'left':'external:left','right':'external:offset'},'shifted',
            {'R':2,'C':3,'X':6,'call_constants':{'left_elements':6,'right_elements':3,'output_elements':6,'scratch_elements':6,'rows':2,'channels':3}})
        for name,kind,n in [('features','indexed_pair_features',36),('cosine','indexed_cosine_rows',3)]:
            params={'G':2,'R':3,'C':3,'X':n,'call_constants':{'left_elements':6,'right_elements':9,'indices_elements':3,
                'output_elements':n,'scratch_elements':n,'groups':2,'rows':3,'channels':3}}
            if name=='cosine':params['epsilon']=1e-8
            operation(name,kind,{'left':'shifted','right':'external:right','indices':'external:indices'},name,params)
        for name,kind in [('exp','scalar_exp_clamped'),('sigmoid','scalar_sigmoid')]:
            params={'call_constants':{'input_elements':1,'output_elements':1}}
            if name=='exp':params['maximum']=1.
            operation(name,kind,{},name,params,True)
        operation('scale','scale_vector',{'input':'cosine','gain':'exp'},'scaled',
            {'R':3,'X':3,'call_constants':{'input_elements':3,'gain_elements':1,'output_elements':3,'scratch_elements':3,'rows':3}})
        graph={'version':3,'name':'row_scalar_composition','family':'bounded_graph','checked_native_entry':True,
            'contract':{'runtime_invariants':{'inference_only':True}},
            'activation_buffers':{k:{'shape':v} for k,v in shapes.items()},'activation_bindings':{k:k for k in shapes},
            'native_entry':{'function':'ck_row_scalar_test','params':[{'c_type':'uint8_t *','name':'arena'},{'c_type':'size_t','name':'arena_bytes'}],
                            'arena':{'pointer':'arena','bytes':'arena_bytes'}},
            'sequence':['ops'],'block_types':{'ops':{'sequence':['header','body','footer'],'header':[],
                'body':{'type':'dense','ops':operations},'footer':[]}},'runtime_constants':{},'runtime_lengths':{}}
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);weights={'exp.weight':np.array([.2],dtype='f4'),'sigmoid.weight':np.array([-.3],dtype='f4')}
            bundle=write_synthetic_weights(folder,weights)
            source={'config':{'model':graph['name'],'arch':graph['name'],'num_layers':1,'embed_dim':3,
                'num_heads':1,'num_kv_heads':1,'head_dim':3,'intermediate_size':3,'context_length':3,'max_seq_len':3,
                'vocab_size':4,'activation_buffer_dtypes':{'indices':'i32'}},'entries':bundle['entries'],'quant_summary':{},'template':graph}
            layout,calls,_,_,execute=compile_native_graph(folder,source,graph)
            self.assertEqual(calls['errors'],[]);self.assertEqual(len(calls['operations']),6)
            size=layout['memory']['arena']['total_size'];raw=(ctypes.c_uint8*(size+63))()
            arena=(ctypes.c_uint8*size).from_buffer(raw,(-ctypes.addressof(raw))&63)
            for row in layout['memory']['weights']['entries']:
                if row['name'] in weights:np.ndarray((1,),dtype='f4',buffer=arena,offset=row['abs_offset'])[:]=weights[row['name']]
            entries={r['name']:r for r in layout['memory']['activations']['buffers']}
            arrays={k:np.ndarray(v,dtype='i4' if k=='indices' else 'f4',buffer=arena,offset=entries[k]['abs_offset']) for k,v in shapes.items()}
            arrays['left'][:]=[[1,2,3],[3,1,2]];arrays['offset'][:]=[.2,-.3,.4]
            arrays['right'][:]=[[1,0,1],[0,1,1],[1,1,0]];arrays['indices'][:]=[1,0,1]
            self.assertEqual(execute(arena,size),0)
            paired=(arrays['left']+arrays['offset'])[arrays['indices']];right=arrays['right']
            expected=(paired/np.linalg.norm(paired,axis=1)[:,None]*right/np.linalg.norm(right,axis=1)[:,None]).sum(-1)
            np.testing.assert_allclose(arrays['scaled'],expected*np.exp(np.float32(.2)),atol=3e-7,rtol=0)
            np.testing.assert_allclose(arrays['features'],np.concatenate((paired,right,paired*right,np.abs(paired-right)),axis=1),atol=0,rtol=0)
            np.testing.assert_allclose(arrays['sigmoid'],1/(1+np.exp(.3)),atol=1e-7,rtol=0)
            baseline=arrays['scaled'].copy();arrays['indices'][1]=2;arrays['scaled'][:]=-13
            self.assertNotEqual(execute(arena,size),0);np.testing.assert_array_equal(arrays['scaled'],-13)
            arrays['indices'][1]=0;self.assertEqual(execute(arena,size),0)
            np.testing.assert_array_equal(arrays['scaled'],baseline)

if __name__=='__main__':unittest.main()
