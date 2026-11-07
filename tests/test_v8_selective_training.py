#!/usr/bin/env python3
"""Generic frozen-parameter contract in the shared v8 training backend.

This small linear decision fixture is not the released Clef head. All CKE
forward, backward and AdamW arithmetic is emitted by the existing trainer.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
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
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest('independent PyTorch gradient comparison unavailable: ' + str(exc))
        sys.path.insert(0, str(ROOT / 'version/v8/scripts'))
        import run_training_workflow_v8 as workflow
        engine = ROOT / 'build/libckernel_engine.so'
        if not engine.exists():
            self.skipTest('build/libckernel_engine.so is required for native training')
        torch.set_num_threads(1)
        ir = lower(fixture())
        manifest = {'config': ir['config']}
        layout = planner.build_layout(ir, manifest, 64, strict=True)
        source, summary = codegen.generate_c(ir, REGISTRY, manifest=manifest, layout=layout)
        self.assertEqual(summary['skipped_ops'], [])
        with tempfile.TemporaryDirectory(prefix='cke-selective-training-') as tmp:
            path = Path(tmp)
            (path / 'train.c').write_text(source)
            subprocess.run([os.environ.get('CC', 'cc'), '-DCK_NUM_TOKENS=4', '-O2', '-shared', '-fPIC', '-I' + str(ROOT / 'include'),
                            str(path / 'train.c'), str(engine), '-Wl,-rpath,' + str(engine.parent),
                            '-lm', '-o', str(path / 'train.so')], check=True)
            lib = ctypes.CDLL(str(path / 'train.so'))
            fp = ctypes.POINTER(ctypes.c_float)
            ip = ctypes.POINTER(ctypes.c_int32)
            ptr = lambda a: a.ctypes.data_as(fp)
            lib.ck_train_init.argtypes = [fp, ip, ctypes.c_int]
            lib.ck_train_step_ex.argtypes = [ip, ip, ctypes.c_int, fp, ctypes.c_float]
            workflow._configure(lib)
            lib.ck_set_strict_parity(1)
            rng = np.random.default_rng(837)
            values = {'features': np.eye(4, dtype=np.float32),
                      'bridge': np.eye(4, dtype=np.float32) + rng.normal(0, .1, (4, 4)).astype('f'),
                      'up': rng.normal(0, .2, (4, 4)).astype('f'),
                      'score': rng.normal(0, .2, (4, 4)).astype('f')}
            ordered = summary['init_weight_order']
            packed = np.concatenate([values[n].ravel() for n in ordered])
            sizes = np.array(summary['init_weight_numel'], dtype='int32')
            self.assertEqual(lib.ck_train_init(ptr(packed), sizes.ctypes.data_as(ip), len(sizes)), len(sizes))
            params = {n: torch.tensor(v, requires_grad=n in ('up', 'score')) for n, v in values.items()}
            opt = torch.optim.AdamW([params['score'], params['up']], lr=.03, weight_decay=.01, foreach=False)
            ids = np.arange(4, dtype='int32')
            labels = torch.arange(4)
            losses = []
            errors = dict(gradient=0.0, weight=0.0, moment=0.0)
            config = dict(dtype='fp32', optimizer='AdamW', lr=.03, weight_decay=.01,
                          beta1=.9, beta2=.999, eps=1e-8, grad_accum=1)
            metadata = dict(runtime_contract_sha256=summary['runtime_contract_sha256'],
                            training_config=config, training_config_sha256=workflow._json_sha256(config),
                            dataset_sha256=workflow._sha256_bytes(ids.tobytes() + values['features'].tobytes()),
                            next_microstep=20, total_microsteps=40, grad_accum=1)

            def snapshot(kind, count):
                out = np.empty(count, dtype='float32')
                fn = getattr(lib, 'ck_train_export_' + kind + '_snapshot')
                fn.argtypes = [fp, ctypes.c_int]
                self.assertEqual(fn(ptr(out), count), count)
                return out

            try:
                for step in range(40):
                    opt.zero_grad()
                    logits = params['features'] @ params['up'].T @ params['bridge'].T @ params['score'].T
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    loss.backward()
                    self.assertGreater(float(params['up'].grad.abs().max()), 1e-5)
                    native_loss = ctypes.c_float()
                    self.assertGreater(lib.ck_train_step_ex(ids.ctypes.data_as(ip), ids.ctypes.data_as(ip),
                                                           4, ctypes.byref(native_loss), .03), 0)
                    self.assertAlmostEqual(native_loss.value, loss.item(), delta=3e-5, msg=str(step))
                    actual = snapshot('weight', summary['weight_snapshot_floats'])
                    for n in ('features', 'bridge'):
                        offset = sum(summary['init_weight_numel'][:ordered.index(n)])
                        self.assertEqual(actual[offset:offset+16].tobytes(), values[n].tobytes())
                    self.assertEqual(summary['parameter_gradient_order'], ['score', 'up'])
                    grads = snapshot('parameter_gradient', summary['parameter_gradient_snapshot_floats'])
                    expected = np.concatenate([params[n].grad.numpy().ravel() for n in summary['parameter_gradient_order']])
                    errors['gradient'] = max(errors['gradient'], float(np.max(np.abs(grads - expected))))
                    np.testing.assert_allclose(grads, expected, atol=3e-5, rtol=3e-4)
                    if step == 0:
                        # Independent central difference at an upstream parameter:
                        # detaching the frozen middle would lose this derivative.
                        double = {n: p.detach().double().clone() for n, p in params.items()}
                        def evaluate(delta):
                            w = double['up'].clone()
                            w[0, 0] += delta
                            z = double['features'] @ w.T @ double['bridge'].T @ double['score'].T
                            return torch.nn.functional.cross_entropy(z, labels).item()
                        finite_difference = (evaluate(1e-5) - evaluate(-1e-5)) / 2e-5
                        self.assertAlmostEqual(float(grads[16]), finite_difference, delta=1e-6)
                    opt.step()
                    moments = snapshot('optimizer_state', summary['optimizer_state_snapshot_floats'])
                    expected_moments = np.concatenate([
                        opt.state[params[n]][kind].numpy().ravel()
                        for kind in ('exp_avg', 'exp_avg_sq') for n in ('score', 'up')])
                    errors['moment'] = max(errors['moment'], float(np.max(np.abs(moments - expected_moments))))
                    np.testing.assert_allclose(moments, expected_moments, atol=3e-6, rtol=3e-4)
                    actual = snapshot('weight', summary['weight_snapshot_floats'])
                    expected = np.concatenate([params[n].detach().numpy().ravel() for n in ordered])
                    errors['weight'] = max(errors['weight'], float(np.max(np.abs(actual - expected))))
                    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
                    losses.append(native_loss.value)
                    if step == 19:
                        checkpoint = workflow._checkpoint_write(path, 'step20', lib, metadata)
                self.assertLess(losses[-1], losses[0] * .2)
                final_weights = snapshot('weight', summary['weight_snapshot_floats'])
                final_moments = snapshot('optimizer_state', summary['optimizer_state_snapshot_floats'])
                # Reset the native runtime before restoring; no Python weight updates.
                lib.ck_train_free()
                self.assertEqual(lib.ck_train_init(ptr(packed), sizes.ctypes.data_as(ip), len(sizes)), len(sizes))
                with self.assertRaisesRegex(RuntimeError, 'dataset_sha256 mismatch'):
                    workflow._checkpoint_load(checkpoint, lib, dict(metadata, dataset_sha256='wrong'))
                workflow._checkpoint_load(checkpoint, lib, metadata)
                for step in range(20, 40):
                    native_loss = ctypes.c_float()
                    self.assertGreater(lib.ck_train_step_ex(ids.ctypes.data_as(ip), ids.ctypes.data_as(ip),
                                                           4, ctypes.byref(native_loss), .03), 0)
                    self.assertEqual(native_loss.value, losses[step])
                np.testing.assert_array_equal(snapshot('weight', len(final_weights)), final_weights)
                np.testing.assert_array_equal(snapshot('optimizer_state', len(final_moments)), final_moments)
                # Reload the trained weights for forward-only execution.
                lib.ck_train_free()
                self.assertEqual(lib.ck_train_init(ptr(final_weights), sizes.ctypes.data_as(ip), len(sizes)), len(sizes))
                lib.ck_train_set_batch_ex(ids.ctypes.data_as(ip), ids.ctypes.data_as(ip), 4)
                self.assertGreater(lib.ck_train_forward_step(), 0)
                activations = workflow.CERT._activation_snapshot(lib, summary)
                expected_logits = params['features'] @ params['up'].T @ params['bridge'].T @ params['score'].T
                np.testing.assert_allclose(activations['act.score'].reshape(4, 4),
                                           expected_logits.detach().numpy(), atol=3e-5, rtol=3e-4)
                self.assertEqual(lib.ck_train_get_opt_step(), 0)
                print('selective FP32 native AdamW: loss %.8f -> %.8f; 40 gradient/update comparisons' % (losses[0], losses[-1]))
                print(json.dumps(dict(seed=837, torch_version=torch.__version__, max_abs_errors=errors,
                                      runtime_contract_sha256=summary['runtime_contract_sha256'],
                                      generated_c_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                      engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest(),
                                      native_training='PASS', full_clef_training='NOT_TESTED'), sort_keys=True))
            finally:
                lib.ck_train_free()


if __name__ == '__main__':
    unittest.main(verbosity=2)
