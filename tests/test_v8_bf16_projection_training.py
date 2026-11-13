"""One real mixed-storage graph, not BF16-complete Clef certification."""
import json
import unittest
from copy import deepcopy
from tests import test_v8_selective_training as base
from version.v8.training.bf16_projection import use_bf16_projection
from version.v8.decision.build_clef_training_ir import training_registry


def fixture(rectangular=False):
    ir = base.fixture()
    if rectangular:
        shapes = {'weight.features':[4,5], 'act.features':[4,5],
                  'weight.up':[7,5], 'act.up':[4,7], 'weight.bridge':[7,7],
                  'act.bridge':[4,7], 'weight.score':[4,7]}
        def update(ref):
            tid=ref.get('tensor')
            if tid in shapes:
                shape=shapes[tid]; ref.update(shape=shape, numel=shape[0]*shape[1])
        for tid,shape in shapes.items(): ir['tensors'][tid].update(shape=shape,numel=shape[0]*shape[1])
        for op in ir['ops']:
            for pool in (op['dataflow']['inputs'],op['dataflow']['outputs'],op.get('weights',{}),op.get('save_for_backward',{})):
                for ref in pool.values(): update(ref)
        ir['config']['hidden_size']=5
    ir = use_bf16_projection(ir, 1)
    return use_bf16_projection(ir, 2)  # frozen bridge must still pass dX upstream


def lower(ir):
    return base.backward.synthesize_ir2_backward(ir, training_registry(), base.BINDINGS,
        json.loads((base.ROOT/'version/v8/scripts/grad_rules_v8.json').read_text()),
        strict=True, allow_partial=False, checkpoint_policy='none')


class BF16ProjectionTraining(unittest.TestCase):
    def test_canonical_map_contracts(self):
        import sys
        sys.path.insert(0, str(base.ROOT/'version/v8/scripts'))
        import resolve_numerical_execution_contracts_v8 as resolver
        catalog=resolver.load_kernel_capabilities()
        for phase in ('forward','backward'):
            self.assertIn('projection_bf16_train_'+phase,catalog['kernels'])

    def test_typed_codegen_and_master_ownership(self):
        ir = lower(fixture())
        layout = base.planner.build_layout(ir, {'config':ir['config']}, 64, strict=True)
        source, report = base.codegen.generate_c(ir, training_registry(), manifest={'config':ir['config']}, layout=layout)
        self.assertEqual(report['skipped_ops'], [])
        self.assertEqual(report['parameter_gradient_order'], ['score','up'])
        self.assertIn('static uint16_t *', source)
        self.assertNotIn('grad.weight.bridge', ir['tensors'])
        self.assertEqual(ir['tensors']['weight.up']['dtype'],'fp32')
        self.assertEqual(ir['tensors']['grad.weight.up']['dtype'],'fp32')

    def test_no_implicit_storage_admission(self):
        ir = lower(fixture())
        for change in ('persistent','gradient','legacy','layout'):
            with self.subTest(change=change):
                bad=deepcopy(ir)
                tid='aux.projection_1.saved_input'
                if change=='persistent': bad['tensors'][tid]['persistent']=True
                if change=='gradient': bad['tensors'][tid]['requires_grad']=True
                if change=='legacy': bad['forward'][1]['map_call']=False
                layout=base.planner.build_layout(bad, {'config':bad['config']},64,strict=True)
                if change=='layout':
                    next(t for t in layout['tensors'] if t['id']==tid)['dtype']='fp32'
                with self.assertRaises((ValueError,RuntimeError)):
                    base.codegen.generate_c(bad,training_registry(),manifest={'config':bad['config']},layout=layout)

    def test_native_trajectory_resume_and_reload(self):
        self.certify(False)

    def test_rectangular_native_trajectory(self):
        self.certify(True)

    def certify(self, rectangular):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest('independent PyTorch unavailable: '+str(exc))
        from tests.v8_training_test_helpers import certify_training
        class Projection(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x,w):
                xr=x.to(torch.bfloat16).float(); wr=w.to(torch.bfloat16).float()
                ctx.save_for_backward(xr,wr)
                return xr @ wr.T
            @staticmethod
            def backward(ctx,dy):
                x,w=ctx.saved_tensors
                return dy @ w, dy.T @ x
        rng=np.random.default_rng(837)
        k,n=(5,7) if rectangular else (4,4)
        values={'features':np.eye(4,k,dtype='f'),
                'bridge':np.eye(n,dtype='f')+rng.normal(0,.1,(n,n)).astype('f'),
                'up':rng.normal(0,.2,(n,k)).astype('f'),
                'score':rng.normal(0,.2,(4,n)).astype('f')}
        def forward(p):
            return Projection.apply(Projection.apply(p['features'],p['up']),p['bridge']) @ p['score'].T
        certify_training(self,lower(fixture(rectangular)),training_registry(),values,{'up','score'},forward,
                         finite_difference=False,precision_label='BF16 operands / FP32 master-gradient',fresh_process=True,checked_runtime=True)

if __name__=='__main__': unittest.main()
