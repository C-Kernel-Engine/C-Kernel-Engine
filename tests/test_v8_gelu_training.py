"""Exact-erf GELU automatic backward and native optimizer composition.

Model-neutral prerequisite, not complete Clef training or a registered job recipe.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from tests import test_v8_selective_training as base
from tests.v8_training_test_helpers import certify_training

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = deepcopy(base.REGISTRY)
REGISTRY['kernels'].extend(json.loads((ROOT/'version/v8/kernel_maps'/f'{name}.json').read_text())
                          for name in ('gelu_rows_erf_checked_f32', 'gelu_erf_backward_checked_f32'))
RULES = json.loads((ROOT/'version/v8/scripts/grad_rules_v8.json').read_text())


def fixture():
    graph = base.fixture()
    graph['config']['train_batch_policy'] = 'fixed'
    source = graph['ops'][1]['dataflow']['outputs']['y']
    def ref(name, shape):
        import math
        out = dict(tensor=name, shape=shape, numel=math.prod(shape), dtype='fp32',
                   kind='activation', requires_grad=False, persistent=False)
        graph['tensors'][name] = dict(out)
        return out
    output = ref('act.gelu', [4, 4])
    output['requires_grad'] = graph['tensors']['act.gelu']['requires_grad'] = True
    scratch = ref('aux.gelu', [16])
    constants = dict(gelu_input_elements=16, gelu_input_stride=4, gelu_output_elements=16,
                     gelu_output_stride=4, gelu_scratch_elements=16, gelu_rows=4, gelu_channels=4,
                     input_elements=16, input_stride=4, d_output_elements=16, d_output_stride=4,
                     d_input_elements=16, d_input_stride=4, scratch_elements=16, rows=4, channels=4)
    op = dict(op_id=2, op='gelu_erf', kernel_id='gelu_rows_erf_checked_f32', phase='forward',
              section='body', layer=0, requires_grad=True, map_call=True,
              params=dict(call_constants=constants), weights={},
              dataflow=dict(inputs=dict(input=source), outputs=dict(output=output, scratch=scratch)),
              save_for_backward=dict(input=source), grad_rule='gelu_erf_checked_backward')
    graph['ops'][2]['dataflow']['inputs']['input'] = output
    graph['ops'][2]['save_for_backward']['input'] = output
    graph['ops'].insert(2, op)
    for i, op in enumerate(graph['ops']):
        op['op_id'] = i
    return graph


def lower(graph):
    return base.backward.synthesize_ir2_backward(graph, REGISTRY, base.BINDINGS, RULES,
                 strict=True, allow_partial=False, checkpoint_policy='none')


def generate(graph, registry=REGISTRY):
    ir = lower(graph)
    manifest = dict(config=ir['config'])
    layout = base.planner.build_layout(ir, manifest, 64, strict=True)
    return base.codegen.generate_c(ir, registry, manifest=manifest, layout=layout)


class GeluTrainingContracts(unittest.TestCase):
    def test_automatic_derivative_saved_tensor_and_optimizer(self):
        graph = fixture(); original = deepcopy(graph)
        source, report = generate(graph)
        self.assertEqual(graph, original)
        self.assertEqual(report['skipped_ops'], [])
        self.assertEqual(report['parameter_gradient_order'], ['score', 'up'])
        self.assertIn('gelu_rows_erf_checked_f32(', source)
        self.assertIn('ck_gelu_erf_backward_f32(act_up,', source)
        self.assertNotIn('gelu_backward_exact(', source)
        ir = lower(graph)
        op = next(o for o in ir['backward'] if o.get('grad_rule') == 'gelu_erf_checked_backward' and o.get('role') == 'core')
        self.assertEqual(ir['tensors'][op['dataflow']['outputs']['scratch']['tensor']]['numel'], 16)

    def test_bad_forward_and_backward_geometry_rejected(self):
        for key in ('gelu_input_stride','gelu_output_stride','input_stride','d_input_stride','d_output_stride'):
            for value in (3,5):
                with self.subTest(key=key, value=value):
                    graph = fixture(); graph['ops'][2]['params']['call_constants'][key] = value
                    with self.assertRaisesRegex(ValueError,'stride'):
                        generate(graph)
        for key in ('gelu_input_elements','d_input_elements','d_output_elements','input_elements'):
            for value in (15,17):
                with self.subTest(key=key,value=value):
                    graph=fixture(); graph['ops'][2]['params']['call_constants'][key]=value
                    with self.assertRaises(ValueError): generate(graph)

    def test_missing_saved_operand_and_bad_workspace_rejected(self):
        graph=fixture(); graph['ops'][2]['save_for_backward']={}
        with self.assertRaises((ValueError,RuntimeError)): generate(graph)
        for value in (0,15,-1,True,1.5,2**63):
            graph=fixture(); graph['ops'][2]['params']['call_constants']['scratch_elements']=value
            with self.assertRaises(ValueError): generate(graph)
        graph=fixture(); graph['config'].pop('train_batch_policy')
        with self.assertRaisesRegex(ValueError,'fixed training batch'): generate(graph)

    def test_invalid_map_or_alias_rejected(self):
        for change in ('status','dtype','alias','missing_map'):
            graph=fixture(); registry=deepcopy(REGISTRY)
            if change=='status': registry['kernels'][-1]['training_contract'].pop('return_status')
            elif change=='dtype': graph['tensors']['act.up']['dtype']='bf16'
            elif change=='alias': graph['ops'][2]['dataflow']['outputs']['scratch']=graph['ops'][2]['dataflow']['outputs']['output']
            else: registry['kernels'].pop()
            with self.subTest(change=change), self.assertRaises((ValueError, RuntimeError)): generate(graph,registry)

    def test_derivative_read_names_are_complete_and_unambiguous(self):
        for names in (['input'], ['input', 'input'], ['input', 'bad-name'], None):
            rules = deepcopy(RULES)
            rules['rules']['gelu_erf_checked_backward']['backward_ops'][0]['input_names'] = names
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, 'input_names'):
                base.backward.synthesize_ir2_backward(fixture(), REGISTRY, base.BINDINGS, rules,
                    strict=True, allow_partial=False, checkpoint_policy='none')

    def test_contract_identity_changes_with_derivative(self):
        _, first=generate(fixture()); registry=deepcopy(REGISTRY)
        registry['kernels'][-1]['numerical_capabilities'][0]['contract_id']+='-changed'
        _, second=generate(fixture(),registry)
        self.assertNotEqual(first['runtime_contract_sha256'],second['runtime_contract_sha256'])


class GeluNativeTraining(unittest.TestCase):
    def test_gradients_learning_frozen_weights_resume_and_reload(self):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest('independent PyTorch unavailable: '+str(exc))
        rng=np.random.default_rng(837)
        values=dict(features=np.eye(4,dtype='f'),
                    bridge=np.eye(4,dtype='f')+rng.normal(0,.1,(4,4)).astype('f'),
                    up=rng.normal(0,.2,(4,4)).astype('f'),
                    score=rng.normal(0,.2,(4,4)).astype('f'))
        def forward(p):
            return torch.nn.functional.gelu(p['features']@p['up'].T, approximate='none')@p['bridge'].T@p['score'].T
        certify_training(self,lower(fixture()),REGISTRY,values,{'up','score'},forward)
        wrong = fixture()
        wrong['ops'][2]['save_for_backward']['input'] = wrong['ops'][2]['dataflow']['outputs']['output']
        with self.assertRaisesRegex(AssertionError, 'Not equal to tolerance'):
            certify_training(self, lower(wrong), REGISTRY, values, {'up', 'score'}, forward)


class GeluTrainingRecovery(unittest.TestCase):
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
        mapping['impl']['function']='test_gelu_backward'
        mapping['impl']['c_declaration']=original.replace('ck_gelu_erf_backward_f32','test_gelu_backward')
        source,report=generate(fixture(),registry)
        # Test-only failure injection AFTER score backward, before GELU VJP.
        params=[p['name'] for p in mapping['call_abi']['params']]
        wrapper='#include <stddef.h>\n'+original+'\nstatic int fail_next;\nvoid test_fail_next(void) { fail_next=1; }\n'
        wrapper+=mapping['impl']['c_declaration'][:-1]+' { if (fail_next) { fail_next=0; return -77; } return ck_gelu_erf_backward_f32('+', '.join(params)+'); }\n'
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


if __name__=='__main__': unittest.main(verbosity=2)
