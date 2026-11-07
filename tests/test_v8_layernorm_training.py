"""Cached-statistics LayerNorm in a generated FP32 training composition.

Uses existing native arithmetic, automatic backward lowering and native AdamW.
This is a complete normalization/projection training test, not full Clef training.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests import test_v8_selective_training as base
from tests.v8_training_test_helpers import certify_training

MAPS = [json.loads((ROOT / 'version/v8/kernel_maps' / (name + '.json')).read_text())
        for name in ('layernorm_train_forward_f32', 'layernorm_train_backward_f32')]
REGISTRY = deepcopy(base.REGISTRY)
REGISTRY['kernels'].extend(MAPS)
RULES = json.loads((ROOT / 'version/v8/scripts/grad_rules_v8.json').read_text())


def fixture(epsilon=1e-5, frozen_affine=False):
    graph = base.fixture()
    del graph['tensors']['weight.bridge']
    def ref(tid, shape, kind='activation', requires_grad=True):
        n = 1
        for dim in shape:
            n *= dim
        x = dict(tensor=tid, dtype='fp32', shape=shape, numel=n,
                 kind=kind, requires_grad=requires_grad, persistent=kind == 'weight')
        graph['tensors'][tid] = {k: v for k, v in x.items() if k != 'tensor'}
        return x
    gamma = dict(ref('weight.gamma', [4], 'weight', not frozen_affine), name='gamma')
    beta = dict(ref('weight.beta', [4], 'weight', not frozen_affine), name='beta')
    mean = ref('act.norm_mean', [4], requires_grad=False)
    rstd = ref('act.norm_rstd', [4], requires_grad=False)
    source = graph['ops'][2]['dataflow']['inputs']['input']
    output = graph['ops'][2]['dataflow']['outputs']['y']
    graph['ops'][2] = dict(op_id=2, op='layernorm', kernel_id='layernorm_train_forward_f32',
        phase='forward', section='body', layer=0, requires_grad=True, map_call=True,
        params=dict(epsilon=epsilon, call_constants=dict(rows=4, channels=4)),
        dataflow=dict(inputs={'input': source}, outputs={'output': output, 'mean_cache': mean, 'rstd_cache': rstd}),
        weights=dict(gamma=gamma, beta=beta), grad_rule='layernorm_cached_backward',
        save_for_backward=dict(input=source, gamma=gamma, mean_cache=mean, rstd_cache=rstd))
    # Deliberately different enclosing model epsilon: the authored operation wins.
    graph['config']['rms_norm_eps'] = .7
    return graph


def lower(graph):
    return base.backward.synthesize_ir2_backward(graph, REGISTRY, base.BINDINGS, RULES,
                     strict=True, allow_partial=False, checkpoint_policy='none')


def generate(graph, registry=REGISTRY):
    ir = lower(graph)
    manifest = {'config': ir['config']}
    layout = base.planner.build_layout(ir, manifest, 64, strict=True)
    return base.codegen.generate_c(ir, registry, manifest=manifest, layout=layout)


class LayerNormTrainingContracts(unittest.TestCase):
    def test_cached_stats_and_operation_epsilon_reach_native_calls(self):
        source, report = generate(fixture(.03))
        self.assertEqual(report['skipped_ops'], [])
        call = next(line for line in source.splitlines() if 'layernorm_naive_serial_matched_precision(act_' in line)
        self.assertIn('act_norm_mean, act_norm_rstd, 4, 4, 3.000000000e-02f', call)
        self.assertIn('layernorm_backward_kernel(grad_act_act_bridge, act_up, weight_gamma, act_norm_mean, act_norm_rstd', source)
        self.assertEqual(report['parameter_gradient_order'], ['beta', 'gamma', 'score', 'up'])

    def test_missing_invalid_and_nonlocal_scalars_rejected(self):
        for value in (None, 0., -1., float('nan'), float('inf'), '1e-5', 1e-100):
            with self.subTest(epsilon=value):
                graph = fixture(value)
                with self.assertRaises(ValueError):
                    generate(graph)
        for value in (0, -1, 5, 2**31, 4.0, True):
            with self.subTest(rows=value):
                graph = fixture()
                graph['ops'][2]['params']['call_constants']['rows'] = value
                with self.assertRaises(ValueError):
                    generate(graph)

    def test_saved_stats_capacity_layout_and_dtype_rejected(self):
        for change in ('short', 'layout', 'dtype'):
            with self.subTest(change=change):
                graph = fixture()
                if change == 'short':
                    graph['tensors']['act.norm_mean']['numel'] = 3
                elif change == 'layout':
                    graph['tensors']['act.up']['shape'] = [2, 8]
                else:
                    graph['tensors']['act.norm_rstd']['dtype'] = 'bf16'
                with self.assertRaises((ValueError, RuntimeError)):
                    generate(graph)

    def test_missing_binding_abi_phase_and_span_rejected(self):
        for change in ('order', 'span', 'mode', 'missing', 'status'):
            with self.subTest(change=change):
                registry = deepcopy(REGISTRY)
                forward = registry['kernels'][-2]
                if change == 'order': forward['call_abi']['params'].reverse()
                elif change == 'span': forward['training_contract']['buffer_spans'].pop()
                elif change == 'mode': forward['modes']['training'] = False
                elif change == 'missing': forward['call_abi']['params'][0]['source'] = 'activation:missing'
                else: forward['impl']['c_declaration'] = forward['impl']['c_declaration'].replace('void ', 'int ', 1)
                with self.assertRaises(ValueError):
                    generate(fixture(), registry)

    def test_alias_rejected_and_input_graph_not_mutated(self):
        graph = fixture()
        original = deepcopy(graph)
        generate(graph)
        self.assertEqual(graph, original)
        graph['ops'][2]['dataflow']['outputs']['mean_cache'] = graph['ops'][2]['dataflow']['outputs']['rstd_cache']
        with self.assertRaises(ValueError):
            generate(graph)

    def test_map_contract_identity_changes_checkpoint_identity(self):
        _, baseline = generate(fixture())
        registry = deepcopy(REGISTRY)
        registry['kernels'][-1]['numerical_capabilities'][0]['contract_id'] += '_different'
        _, changed = generate(fixture(), registry)
        self.assertNotEqual(baseline['runtime_contract_sha256'], changed['runtime_contract_sha256'])

    def test_missing_provider_is_not_silently_skipped(self):
        for change in ('map', 'declaration'):
            registry = deepcopy(REGISTRY)
            if change == 'map': registry['kernels'].pop()
            else: del registry['kernels'][-1]['impl']['c_declaration']
            with self.assertRaises(ValueError):
                generate(fixture(), registry)


class LayerNormNativeTraining(unittest.TestCase):
    def test_saved_stats_vjp_at_tail_and_head_widths(self):
        import ctypes
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest('independent PyTorch unavailable: ' + str(exc))
        engine = ROOT / 'build/libckernel_engine.so'
        if not engine.exists(): self.skipTest('native engine is not built')
        lib = ctypes.CDLL(str(engine))
        fp = ctypes.POINTER(ctypes.c_float)
        lib.layernorm_naive_serial_matched_precision.argtypes = [fp]*6 + [ctypes.c_int]*2 + [ctypes.c_float]
        lib.layernorm_backward_kernel.argtypes = [fp]*8 + [ctypes.c_int]*3
        rng = np.random.default_rng(938)
        ptr = lambda a: a.ctypes.data_as(fp)
        for rows, width in ((1, 1), (3, 7), (2, 512), (3, 513), (2, 4096)):
            with self.subTest(rows=rows, width=width):
                x = rng.normal(0, .5, (rows, width)).astype('f')
                dy = rng.normal(0, .2, x.shape).astype('f')
                gamma = rng.uniform(.5, 1.5, width).astype('f')
                beta = rng.normal(0, .1, width).astype('f')
                y = np.empty_like(x); mean = np.empty(rows, 'f'); rstd = np.empty(rows, 'f')
                lib.layernorm_naive_serial_matched_precision(*map(ptr, (x, gamma, beta, y, mean, rstd)), rows, width, 1e-5)
                dx = np.empty_like(x); dg = np.zeros_like(gamma); db = np.zeros_like(beta)
                lib.layernorm_backward_kernel(*map(ptr, (dy, x, gamma, mean, rstd, dx, dg, db)), rows, width, width)
                tx, tg, tb = [torch.tensor(a, requires_grad=True) for a in (x, gamma, beta)]
                ty = torch.nn.functional.layer_norm(tx, (width,), tg, tb, 1e-5)
                ty.backward(torch.tensor(dy))
                for actual, expected in ((y, ty), (dx, tx.grad), (dg, tg.grad), (db, tb.grad)):
                    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=3e-5, rtol=3e-4)

    def test_affine_and_upstream_gradients_updates_resume_and_reload(self):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest('independent PyTorch unavailable: ' + str(exc))
        for eps, frozen in ((1e-5, False), (.03, True)):
            with self.subTest(epsilon=eps, frozen_affine=frozen):
                rng = np.random.default_rng(837)
                values = dict(features=np.eye(4, dtype='f'), gamma=rng.uniform(.7, 1.3, 4).astype('f'),
                              beta=rng.normal(0, .1, 4).astype('f'), up=rng.normal(0, .2, (4, 4)).astype('f'),
                              score=rng.normal(0, .2, (4, 4)).astype('f'))
                trainable = {'up', 'score'} if frozen else {'up', 'score', 'gamma', 'beta'}
                def forward(p):
                    h = torch.nn.functional.layer_norm(p['features'] @ p['up'].T, (4,), p['gamma'], p['beta'], eps)
                    return h @ p['score'].T
                certify_training(self, lower(fixture(eps, frozen)), REGISTRY, values, trainable, forward)
                if not frozen:
                    wrong = fixture(eps)
                    wrong['ops'][2]['save_for_backward']['mean_cache'] = wrong['ops'][2]['save_for_backward']['rstd_cache']
                    with self.assertRaisesRegex(AssertionError, 'Not equal to tolerance'):
                        certify_training(self, lower(wrong), REGISTRY, values, trainable, forward)


if __name__ == '__main__':
    unittest.main(verbosity=2)
