#!/usr/bin/env python3
"""Generic frozen-parameter contract in the shared v8 training backend.

This small linear decision fixture is not the released Clef head. All CKE
forward, backward and AdamW arithmetic is emitted by the existing trainer.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'version/v7/scripts'))
import lower_ir2_backward_v7 as backward
import generate_train_layout_v7 as planner
import codegen_train_runtime_v7 as codegen


def load(name):
    return json.loads((ROOT / name).read_text())


REGISTRY = load('version/v7/kernel_maps/KERNEL_REGISTRY.json')
BINDINGS = load('version/v7/kernel_maps/kernel_bindings.json')
RULES = load('version/v7/scripts/grad_rules_v7.json')


def fixture():
    """Four fixed feature vectors, trainable projection, frozen bridge, logits."""
    tensors, ops = {}, []

    def ref(tid, shape, kind='activation', trainable=True):
        n = 1
        for d in shape:
            n *= d
        item = dict(tensor=tid, shape=shape, numel=n, dtype='fp32',
                    kind=kind, requires_grad=trainable)
        tensors[tid] = {k: v for k, v in item.items() if k != 'tensor'}
        tensors[tid]['persistent'] = kind == 'weight'
        return item

    ids = ref('input.token_ids', [4], 'input', False)
    ids['dtype'] = tensors['input.token_ids']['dtype'] = 'i32'
    target = ref('input.targets', [4], 'input', False)
    target['dtype'] = tensors['input.targets']['dtype'] = 'i32'
    table = dict(ref('weight.features', [4, 4], 'weight', False), name='features')
    x = ref('act.features', [4, 4], trainable=False)
    ops.append(dict(op_id=0, op='embedding', kernel_id='dense_embedding_lookup',
                    phase='forward', section='header', layer=-1, requires_grad=False,
                    dataflow=dict(inputs={'token_ids': ids}, outputs={'output': x}),
                    weights={'W': table}))
    for i, name in enumerate(('up', 'bridge', 'score'), 1):
        w = dict(ref('weight.' + name, [4, 4], 'weight', name != 'bridge'), name=name)
        y = ref('act.' + name, [4, 4])
        ops.append(dict(op_id=i, op='logits' if name == 'score' else 'projection',
                        kernel_id='gemm_blocked_serial', phase='forward', section='body',
                        layer=i, requires_grad=True, grad_rule='matmul_backward',
                        dataflow=dict(inputs={'input': x}, outputs={'y': y}),
                        weights={'W': w}, save_for_backward={'input': x, 'W': w}))
        x = y
    return dict(format='ir1-train-v7', version=1, num_layers=0,
                config=dict(hidden_size=4, intermediate_size=4, num_heads=1,
                            num_kv_heads=1, head_dim=4, vocab_size=4, train_tokens=4),
                ops=ops, tensors=tensors)


def lower(ir):
    return backward.synthesize_ir2_backward(ir, REGISTRY, BINDINGS, RULES,
                                            strict=True, allow_partial=False, checkpoint_policy="none")


class SelectiveTrainingContractTests(unittest.TestCase):
    def test_frozen_bridge_keeps_activation_gradient_without_optimizer_state(self):
        ir = lower(fixture())
        self.assertIn('grad.weight.up', ir['tensors'])
        self.assertIn('grad.weight.score', ir['tensors'])
        self.assertNotIn('grad.weight.bridge', ir['tensors'])
        self.assertNotIn('grad.weight.features', ir['tensors'])
        bridge = next(o for o in ir['backward'] if o.get('forward_ref') == 2 and o.get('role') == 'core')
        self.assertEqual(bridge['dataflow']['outputs']['d_weight']['kind'], 'discarded_grad_weight')
        self.assertEqual(bridge['dataflow']['outputs']['d_input']['kind'], 'tmp_grad_activation')
        layout = planner.build_layout(ir, {'config': ir['config']}, 64, strict=True)
        _, summary = codegen.generate_c(ir, REGISTRY, manifest={'config': ir['config']}, layout=layout)
        self.assertEqual(summary['parameter_gradient_order'], ['score', 'up'])
        self.assertEqual(summary['optimizer_state_snapshot_floats'], 64)

    def test_conflicting_and_non_boolean_declarations_rejected(self):
        for value in (True, 'false', 0, None):
            with self.subTest(value=value):
                ir = fixture()
                ir['ops'][2]['weights']['W']['requires_grad'] = value
                with self.assertRaises(ValueError):
                    lower(ir)

    def test_shared_weight_conflicting_uses_rejected(self):
        ir = fixture()
        ir['ops'][3]['weights']['W'] = dict(ir['ops'][2]['weights']['W'], requires_grad=True)
        with self.assertRaises(ValueError):
            lower(ir)

    def test_legacy_undeclared_weights_remain_trainable(self):
        ir = fixture()
        del ir['ops'][2]['weights']['W']['requires_grad']
        del ir['tensors']['weight.bridge']['requires_grad']
        self.assertIn('grad.weight.bridge', lower(ir)['tensors'])

    def test_tensor_only_freeze(self):
        ir = fixture()
        del ir['ops'][2]['weights']['W']['requires_grad']
        self.assertNotIn('grad.weight.bridge', lower(ir)['tensors'])

    def test_reference_only_freeze_and_shared_frozen_uses(self):
        ir = fixture()
        del ir['tensors']['weight.bridge']['requires_grad']
        ir['ops'][3]['weights']['W'] = dict(ir['ops'][2]['weights']['W'])
        lowered = lower(ir)
        self.assertNotIn('grad.weight.bridge', lowered['tensors'])
        self.assertIn('grad.weight.up', lowered['tensors'])
        discarded = [o for o in lowered['backward']
                     if o.get('role') == 'core' and o.get('forward_ref') in (2, 3)]
        self.assertEqual(len(discarded), 2)
        for op in discarded:
            self.assertEqual(op['dataflow']['outputs']['d_weight']['kind'], 'discarded_grad_weight')

    def test_frozen_bias_has_scratch_but_no_published_gradient(self):
        ir = fixture()
        bias = dict(tensor='weight.bridge_bias', name='bridge_bias', dtype='fp32',
                    kind='weight', requires_grad=False, shape=[4], numel=4)
        ir['tensors']['weight.bridge_bias'] = dict(bias)
        ir['ops'][2]['weights']['bias'] = bias
        lowered = lower(ir)
        self.assertNotIn('grad.weight.bridge_bias', lowered['tensors'])
        bridge = next(o for o in lowered['backward'] if o.get('forward_ref') == 2 and o.get('role') == 'core')
        self.assertEqual(bridge['dataflow']['outputs']['d_bias']['kind'], 'discarded_grad_weight')


class SelectiveTrainingNativeTests(unittest.TestCase):
    def test_native_gradients_updates_and_learning(self):
        import numpy as np
        from tests.v8_training_test_helpers import certify_training
        rng = np.random.default_rng(837)
        values = {'features': np.eye(4, dtype=np.float32),
                  'bridge': np.eye(4, dtype=np.float32) + rng.normal(0, .1, (4, 4)).astype('f'),
                  'up': rng.normal(0, .2, (4, 4)).astype('f'),
                  'score': rng.normal(0, .2, (4, 4)).astype('f')}
        certify_training(self, lower(fixture()), REGISTRY, values, {'up', 'score'},
                         lambda p: p['features'] @ p['up'].T @ p['bridge'].T @ p['score'].T, check_evaluation=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
