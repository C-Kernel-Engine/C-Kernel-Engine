"""Shared test-only certification; CKE arithmetic and updates stay native."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v7/scripts'))
import codegen_train_runtime_v7 as codegen
import generate_train_layout_v7 as planner


def certify_training(self, ir, registry, values, trainable, forward, *, integer_inputs=None, loss_function=None):
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
    manifest = {'config': ir['config']}
    layout = planner.build_layout(ir, manifest, 64, strict=True)
    source, summary = codegen.generate_c(ir, registry, manifest=manifest, layout=layout)
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
        ordered = summary['init_weight_order']
        packed = np.concatenate([values[n].ravel() for n in ordered])
        sizes = np.array(summary['init_weight_numel'], dtype='int32')
        self.assertEqual(lib.ck_train_init(ptr(packed), sizes.ctypes.data_as(ip), len(sizes)), len(sizes))
        params = {n: torch.tensor(v, requires_grad=n in trainable) for n, v in values.items()}
        opt = torch.optim.AdamW([params[n] for n in sorted(trainable)], lr=.03, weight_decay=.01, foreach=False)
        metadata_inputs = np.ascontiguousarray(integer_inputs, dtype='int32') if integer_inputs is not None else None
        if metadata_inputs is None:
            native_step = lib.ck_train_step_ex
            native_batch = lib.ck_train_set_batch_ex
        else:
            self.assertEqual(metadata_inputs.size, summary['integer_input_count'])
            lib.ck_train_step_with_i32_ex.argtypes = [ip, ip, ctypes.c_int, ip, ctypes.c_size_t, fp, ctypes.c_float]
            lib.ck_train_set_batch_with_i32_ex.argtypes = [ip, ip, ctypes.c_int, ip, ctypes.c_size_t]
            def native_step(ids, targets, count, loss, lr):
                return lib.ck_train_step_with_i32_ex(ids, targets, count, metadata_inputs.ctypes.data_as(ip), metadata_inputs.size, loss, lr)
            def native_batch(ids, targets, count):
                return lib.ck_train_set_batch_with_i32_ex(ids, targets, count, metadata_inputs.ctypes.data_as(ip), metadata_inputs.size)
        ids = np.arange(4, dtype='int32')
        labels = torch.arange(4)
        criterion = loss_function or (lambda logits: torch.nn.functional.cross_entropy(logits, labels))
        losses = []
        errors = dict(gradient=0.0, weight=0.0, moment=0.0)
        config = dict(dtype='fp32', optimizer='AdamW', lr=.03, weight_decay=.01,
                      beta1=.9, beta2=.999, eps=1e-8, grad_accum=1)
        metadata = dict(runtime_contract_sha256=summary['runtime_contract_sha256'],
                        training_config=config, training_config_sha256=workflow._json_sha256(config),
                        dataset_sha256=workflow._sha256_bytes(ids.tobytes() + values['features'].tobytes() + (metadata_inputs.tobytes() if metadata_inputs is not None else b'')),
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
                logits = forward(params)
                loss = criterion(logits)
                loss.backward()
                # Grouped objectives can converge to a zero gradient; preserve
                # the existing decoder assertion for its original trajectory.
                if loss_function is None or step == 0:
                    self.assertGreater(float(params['up'].grad.abs().max()), 1e-5)
                native_loss = ctypes.c_float()
                self.assertGreater(native_step(ids.ctypes.data_as(ip), ids.ctypes.data_as(ip),
                                                       4, ctypes.byref(native_loss), .03), 0)
                self.assertAlmostEqual(native_loss.value, loss.item(), delta=3e-5, msg=str(step))
                actual = snapshot('weight', summary['weight_snapshot_floats'])
                for n in set(values) - set(trainable):
                    offset = sum(summary['init_weight_numel'][:ordered.index(n)])
                    self.assertEqual(actual[offset:offset+values[n].size].tobytes(), values[n].tobytes())
                self.assertEqual(summary['parameter_gradient_order'], sorted(trainable))
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
                        z = forward(dict(double, up=w))
                        return criterion(z).item()
                    finite_difference = (evaluate(1e-5) - evaluate(-1e-5)) / 2e-5
                    self.assertAlmostEqual(float(grads[sum(values[n].size for n in sorted(trainable)[:sorted(trainable).index('up')])]), finite_difference, delta=1e-6)
                opt.step()
                moments = snapshot('optimizer_state', summary['optimizer_state_snapshot_floats'])
                expected_moments = np.concatenate([
                    opt.state[params[n]][kind].numpy().ravel()
                    for kind in ('exp_avg', 'exp_avg_sq') for n in sorted(trainable)])
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
                self.assertGreater(native_step(ids.ctypes.data_as(ip), ids.ctypes.data_as(ip),
                                                       4, ctypes.byref(native_loss), .03), 0)
                self.assertEqual(native_loss.value, losses[step])
            np.testing.assert_array_equal(snapshot('weight', len(final_weights)), final_weights)
            np.testing.assert_array_equal(snapshot('optimizer_state', len(final_moments)), final_moments)
            # Reload the trained weights for forward-only execution.
            lib.ck_train_free()
            self.assertEqual(lib.ck_train_init(ptr(final_weights), sizes.ctypes.data_as(ip), len(sizes)), len(sizes))
            self.assertGreater(native_batch(ids.ctypes.data_as(ip), ids.ctypes.data_as(ip), 4), 0)
            self.assertGreater(lib.ck_train_forward_step(), 0)
            activations = workflow.CERT._activation_snapshot(lib, summary)
            expected_logits = forward(params)
            np.testing.assert_allclose(activations['act.score'].reshape(tuple(expected_logits.shape)),
                                       expected_logits.detach().numpy(), atol=3e-5, rtol=3e-4)
            self.assertEqual(lib.ck_train_get_opt_step(), 0)
            print('FP32 native AdamW: loss %.8f -> %.8f; 40 gradient/update comparisons' % (losses[0], losses[-1]))
            print(json.dumps(dict(seed=837, torch_version=torch.__version__, max_abs_errors=errors,
                                  runtime_contract_sha256=summary['runtime_contract_sha256'],
                                  generated_c_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                  engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest(),
                                  native_training='PASS', full_clef_training='NOT_TESTED'), sort_keys=True))
        finally:
            lib.ck_train_free()
