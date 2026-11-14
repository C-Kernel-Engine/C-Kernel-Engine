"""Declared frozen native prefix with generated head derivatives."""

import json
import unittest
from tests.test_v8_circuit_training_lowering import fixture, ROOT, backend
from version.v8.scripts.compose_decoder_features_v8 import compose_decoder_features
from version.v8.scripts.lower_circuit_training_v8 import (
    lower_circuit_training,
    training_registry,
)


def composed_fixture(stop_gradient=True):
    graph, weights, args = fixture()
    ops = graph["block_types"]["component"]["body"]["ops"]
    ops.pop(0)
    ops[0]["graph_slots"]["inputs"]["input"] = "external:embedded"
    graph.update(
        checked_native_entry=True,
        activation_bindings={},
        native_entry=dict(
            params=[
                dict(name="arena", c_type="uint8_t *"),
                dict(name="arena_bytes", c_type="size_t"),
            ]
        ),
    )
    graph = compose_decoder_features(
        graph, feature_input="embedded", stop_gradient=stop_gradient
    )
    del weights["features"]
    args["frozen"].remove("features")
    del args["inputs"]["ids"]
    del args["bindings"]["ids"]
    args["inputs"]["backbone_ids"] = dict(shape=[4], dtype="i32")
    args["bindings"]["backbone_ids"] = "input.backbone_ids"
    return graph, weights, args


def generated(with_lora=False):
    graph, weights, args = composed_fixture()
    ir = lower_circuit_training(graph, weights, **args)
    if with_lora:
        from version.v8.training.lora import attach_lora, initialize_lora

        ir, spec = attach_lora(ir, target="bridge", rank=2, alpha=3, name="prefix")
        weights.update(initialize_lora(spec, seed=41))
    registry = training_registry()
    rules = json.loads((ROOT / "version/v8/scripts/grad_rules_v8.json").read_text())
    lowered = backend.backward.synthesize_ir2_backward(
        ir,
        registry,
        backend.BINDINGS,
        rules,
        strict=True,
        allow_partial=False,
        checkpoint_policy="none",
    )
    layout = backend.planner.build_layout(
        lowered, {"config": lowered["config"]}, 64, strict=True
    )
    source, summary = backend.codegen.generate_c(
        lowered, registry, manifest={"config": lowered["config"]}, layout=layout
    )
    return source, summary, lowered, weights


class FrozenComponentTraining(unittest.TestCase):
    def test_explicit_frozen_boundary_and_native_binding(self):
        source, summary, ir, _ = generated()
        self.assertEqual(summary["skipped_ops"], [])
        self.assertEqual(summary["parameter_gradient_order"], ["score", "up"])
        self.assertNotIn("grad.act.embedded", ir["tensors"])
        self.assertIn("ck_train_set_native_context_ex", source)
        self.assertIn(
            "decoder_features_sequence_checked_f32(g_native_contexts[0]", source
        )
        self.assertEqual(
            summary["native_contexts"],
            [dict(pointer="decoder_context", bytes="decoder_context_bytes")],
        )

    def test_invalid_native_context_configuration_rejected(self):
        from copy import deepcopy

        _, _, ir, _ = generated()
        for replacement in (
            None,
            [],
            [{"pointer": "undeclared", "bytes": "decoder_context_bytes"}],
            [{"pointer": "decoder_context", "bytes": "decoder_context"}],
            [
                {
                    "pointer": "decoder_context",
                    "bytes": "decoder_context_bytes",
                    "extra": 1,
                }
            ],
        ):
            bad = deepcopy(ir)
            bad["config"]["training_native_contexts"] = replacement
            with self.subTest(contexts=replacement), self.assertRaises(ValueError):
                layout = backend.planner.build_layout(
                    bad, {"config": bad["config"]}, 64, strict=True
                )
                backend.codegen.generate_c(
                    bad,
                    training_registry(),
                    manifest={"config": bad["config"]},
                    layout=layout,
                )

    def test_implicit_stop_gradient_rejected(self):
        graph, weights, args = composed_fixture(False)
        with self.assertRaisesRegex(ValueError, "explicit nondifferentiable boundary"):
            lower_circuit_training(graph, weights, **args)

    def test_wrong_context_type_rejected(self):
        graph, weights, args = composed_fixture()
        graph["native_entry"]["params"][-2]["c_type"] = "float *"
        with self.assertRaises(ValueError):
            lower_circuit_training(graph, weights, **args)

    def test_native_training_failure_recovery_and_oracle(self, with_lora=False):
        if not (ROOT / "build/libckernel_engine.so").is_file():
            self.skipTest("native engine unavailable")
        import ctypes as C
        import subprocess
        import tempfile
        import numpy as np

        try:
            import torch
        except ImportError as exc:
            self.skipTest("PyTorch oracle unavailable: " + str(exc))
        from tests.test_v8_decoder_feature_sequence import FAKE
        from version.v8.scripts import run_training_workflow_v8 as workflow

        source, summary, ir, weights = generated(with_lora)
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            path = Path(tmp)
            (path / "train.c").write_text(source)
            (path / "fake.c").write_text(FAKE)
            subprocess.run(
                [
                    "cc",
                    "-O2",
                    "-shared",
                    "-fPIC",
                    "-DCK_NUM_TOKENS=4",
                    "-Werror=incompatible-pointer-types",
                    "-I" + str(ROOT / "include"),
                    str(path / "train.c"),
                    str(path / "fake.c"),
                    str(ROOT / "src/runtime/decoder_features_sequence.c"),
                    str(ROOT / "src/kernels/linear_rows_checked.c"),
                    str(ROOT / "src/kernels/transformer_rows_checked.c"),
                    str(ROOT / "build/libckernel_engine.so"),
                    "-Wl,-rpath," + str(ROOT / "build"),
                    "-lm",
                    "-o",
                    str(path / "train.so"),
                ],
                check=True,
                capture_output=True,
            )
            lib = C.CDLL(str(path / "train.so"))
            workflow._configure(lib)
            fp, ip = C.POINTER(C.c_float), C.POINTER(C.c_int32)
            ptr = lambda a: a.ctypes.data_as(fp)
            ints = np.array([0, 1, 2, 3, 2, 2, 0, 1], dtype="i4")
            ids = np.arange(4, dtype="i4")
            packed = np.concatenate(
                [weights[n].ravel() for n in summary["init_weight_order"]]
            )
            sizes = np.array(summary["init_weight_numel"], dtype="i4")
            lib.ck_train_init.argtypes = [fp, ip, C.c_int]
            self.assertEqual(
                lib.ck_train_init(ptr(packed), sizes.ctypes.data_as(ip), len(sizes)),
                len(sizes),
            )
            lib.ck_train_step_with_i32_ex.argtypes = [
                ip,
                ip,
                C.c_int,
                ip,
                C.c_size_t,
                fp,
                C.c_float,
            ]
            lib.ck_train_set_native_context_ex.argtypes = [
                C.c_int,
                C.c_void_p,
                C.c_size_t,
            ]
            lib.test_context.restype = C.c_void_p
            lib.test_context_bytes.restype = C.c_size_t

            def step():
                loss = C.c_float()
                result = lib.ck_train_step_with_i32_ex(
                    ids.ctypes.data_as(ip),
                    ids.ctypes.data_as(ip),
                    4,
                    ints.ctypes.data_as(ip),
                    len(ints),
                    C.byref(loss),
                    0.03,
                )
                return result, loss.value

            def snapshot(kind, count):
                a = np.empty(count, dtype="f4")
                fn = getattr(lib, "ck_train_export_" + kind + "_snapshot")
                fn.argtypes = [fp, C.c_int]
                self.assertEqual(fn(ptr(a), count), count)
                return a

            def unchanged_rejection():
                before = snapshot("weight", len(packed))
                moments = workflow._array_export(lib, "optimizer_state")
                count = lib.ck_train_get_opt_step()
                self.assertLess(step()[0], 0)
                np.testing.assert_array_equal(snapshot("weight", len(packed)), before)
                self.assertEqual(lib.ck_train_get_opt_step(), count)
                np.testing.assert_array_equal(
                    workflow._array_export(lib, "optimizer_state"), moments
                )

            try:
                unchanged_rejection()
                for index, context, count in (
                    (1, lib.test_context(), lib.test_context_bytes()),
                    (0, None, lib.test_context_bytes()),
                    (0, lib.test_context(), 0),
                    (0, lib.test_context(), 1048577),
                    (0, C.c_void_p(-1), 8),
                ):
                    self.assertLess(
                        lib.ck_train_set_native_context_ex(index, context, count), 0
                    )
                    unchanged_rejection()
                self.assertEqual(
                    lib.ck_train_set_native_context_ex(
                        0, lib.test_context(), lib.test_context_bytes()
                    ),
                    1,
                )
                for mode in (1, 2, 3, 4, 5, 6):
                    lib.test_mode(mode)
                    unchanged_rejection()
                lib.test_mode(0)
                original_ids = ints[:4].copy()
                ints[0] = 10
                unchanged_rejection()
                ints[:4] = original_ids
                torch.set_num_threads(1)
                params = {
                    n: torch.tensor(
                        v, requires_grad=n in summary["parameter_gradient_order"]
                    )
                    for n, v in weights.items()
                }
                optimizer = torch.optim.AdamW(
                    [params[n] for n in summary["parameter_gradient_order"]],
                    lr=0.03,
                    weight_decay=0.01,
                    foreach=False,
                )
                features = (
                    torch.arange(4).float()[:, None]
                    * torch.arange(1, 5).float()[None, :]
                    + 0.1 * torch.arange(1, 5).float()[:, None]
                )
                for iteration in range(5):
                    if iteration == 3:
                        import hashlib

                        training_config = dict(
                            dtype="fp32",
                            optimizer="AdamW",
                            lr=0.03,
                            weight_decay=0.01,
                            beta1=0.9,
                            beta2=0.999,
                            eps=1e-8,
                            grad_accum=1,
                        )
                        checkpoint_metadata = dict(
                            runtime_contract_sha256=summary["runtime_contract_sha256"],
                            training_config=training_config,
                            training_config_sha256=workflow._json_sha256(
                                training_config
                            ),
                            dataset_sha256=hashlib.sha256(ints.tobytes()).hexdigest(),
                            next_microstep=3,
                            total_microsteps=5,
                            grad_accum=1,
                        )
                        checkpoint = workflow._checkpoint_write(
                            path, "native-prefix", lib, checkpoint_metadata
                        )
                        self.assertEqual(
                            lib.ck_train_init(
                                ptr(packed), sizes.ctypes.data_as(ip), len(sizes)
                            ),
                            len(sizes),
                        )
                        workflow._checkpoint_load(checkpoint, lib, checkpoint_metadata)
                        self.assertEqual(lib.ck_train_get_opt_step(), 3)
                        # Borrowed pointers are deliberately not checkpoint state.
                        unchanged_rejection()
                        self.assertEqual(
                            lib.ck_train_set_native_context_ex(
                                0, lib.test_context(), lib.test_context_bytes()
                            ),
                            1,
                        )
                    if iteration == 2:
                        lib.test_mode(1)
                        unchanged_rejection()
                        lib.test_mode(0)
                    optimizer.zero_grad()
                    projection = features @ params["up"].T
                    middle = projection @ params["bridge"].T
                    if with_lora:
                        middle = (
                            middle
                            + (
                                projection
                                @ params["lora.prefix.a"].T
                                @ params["lora.prefix.b"].T
                            )
                            * params["lora.prefix.scale"]
                        )
                    logits = middle @ params["score"].T
                    expected = torch.nn.functional.cross_entropy(
                        logits.reshape(2, 2), torch.arange(2)
                    )
                    expected.backward()
                    if with_lora:
                        if iteration == 0:
                            self.assertEqual(
                                float(params["lora.prefix.a"].grad.abs().max()), 0.0
                            )
                        else:
                            self.assertGreater(
                                float(params["lora.prefix.a"].grad.abs().max()), 0.0
                            )
                        self.assertGreater(
                            float(params["lora.prefix.b"].grad.abs().max()), 0.0
                        )
                    result, loss = step()
                    self.assertGreater(result, 0)
                    self.assertAlmostEqual(loss, expected.item(), delta=3e-5)
                    grads = snapshot(
                        "parameter_gradient",
                        summary["parameter_gradient_snapshot_floats"],
                    )
                    np.testing.assert_allclose(
                        grads,
                        np.concatenate(
                            [
                                params[n].grad.numpy().ravel()
                                for n in summary["parameter_gradient_order"]
                            ]
                        ),
                        atol=3e-5,
                        rtol=3e-4,
                    )
                    optimizer.step()
                    actual = snapshot("weight", len(packed))
                    np.testing.assert_allclose(
                        actual,
                        np.concatenate(
                            [
                                params[n].detach().numpy().ravel()
                                for n in summary["init_weight_order"]
                            ]
                        ),
                        atol=3e-5,
                        rtol=3e-4,
                    )
                    self.assertEqual(lib.test_active(), 0)
                self.assertLess(lib.ck_train_backward_step(), 0)
                lib.ck_train_optimizer_step.argtypes = [C.c_float]
                self.assertLess(lib.ck_train_optimizer_step(0.03), 0)
                before = snapshot("weight", len(packed))
                lib.ck_train_eval_step.argtypes = [fp]
                evaluation = C.c_float()
                self.assertGreater(lib.ck_train_eval_step(C.byref(evaluation)), 0)
                self.assertLess(lib.ck_train_optimizer_step(0.03), 0)
                np.testing.assert_array_equal(snapshot("weight", len(packed)), before)
                lib.ck_train_clear_native_contexts()
                unchanged_rejection()
            finally:
                lib.ck_train_free()

    def test_lora_after_frozen_native_component(self):
        self.test_native_training_failure_recovery_and_oracle(with_lora=True)

    def test_relocated_c_only_training(self):
        if not (ROOT / "build/libckernel_engine.so").is_file():
            self.skipTest("native engine unavailable")
        import os
        import shutil
        import subprocess
        import tempfile
        import numpy as np
        from pathlib import Path
        from tests.test_v8_decoder_feature_sequence import FAKE

        source, summary, _, weights = generated()
        packed = np.concatenate(
            [weights[n].ravel() for n in summary["init_weight_order"]]
        )
        sizes = ",".join(map(str, summary["init_weight_numel"]))
        host = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
int ck_train_init(const float *,const int32_t *,int);
int ck_train_set_native_context_ex(int,const void *,size_t);
int ck_train_step_with_i32_ex(const int32_t *,const int32_t *,int,const int32_t *,size_t,float *,float);
int ck_train_get_opt_step(void);
void ck_train_free(void);
void *test_context(void);
size_t test_context_bytes(void);
int test_active(void);
void test_mode(int);
int main(void) {
 float weights[WEIGHT_COUNT],loss=0,first=0;
 int32_t sizes[]={WEIGHT_SIZES}, ids[]={0,1,2,3}, inputs[]={0,1,2,3,2,2,0,1};
 FILE *f=fopen("weights.bin","rb");
 if(!f || fread(weights,sizeof(float),WEIGHT_COUNT,f)!=WEIGHT_COUNT) return 1;
 fclose(f);
 if(ck_train_init(weights,sizes,3)!=3) return 2;
 if(ck_train_step_with_i32_ex(ids,ids,4,inputs,8,&loss,.03f)>=0) return 3;
 if(ck_train_set_native_context_ex(0,test_context(),test_context_bytes())!=1) return 4;
 for(int i=0;i<5;i++) {
   if(i==2) {
     test_mode(1);
     if(ck_train_step_with_i32_ex(ids,ids,4,inputs,8,&loss,.03f)>=0 || ck_train_get_opt_step()!=2) return 5;
     test_mode(0);
   }
   if(ck_train_step_with_i32_ex(ids,ids,4,inputs,8,&loss,.03f)<=0 || test_active()!=0) return 6;
   if(i==0) first=loss;
 }
 if(!isfinite(loss) || !isfinite(first) || ck_train_get_opt_step()!=5) { fprintf(stderr,"first=%g last=%g step=%d\n",first,loss,ck_train_get_opt_step()); return 7; }
 ck_train_free();
 return 0;
}
""".replace("WEIGHT_COUNT", str(len(packed))).replace("WEIGHT_SIZES", sizes)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            moved = root / "relocated"
            build.mkdir()
            (build / "train.c").write_text(source)
            (build / "fake.c").write_text(FAKE)
            (build / "host.c").write_text(host)
            packed.tofile(build / "weights.bin")
            shutil.copy2(
                ROOT / "build/libckernel_engine.so", build / "libckernel_engine.so"
            )
            subprocess.run(
                [
                    "cc",
                    "-O2",
                    "-shared",
                    "-fPIC",
                    "-DCK_NUM_TOKENS=4",
                    "-I" + str(ROOT / "include"),
                    str(build / "train.c"),
                    str(build / "fake.c"),
                    str(ROOT / "src/runtime/decoder_features_sequence.c"),
                    str(ROOT / "src/kernels/linear_rows_checked.c"),
                    "-L" + str(build),
                    "-lckernel_engine",
                    "-Wl,-rpath,$ORIGIN",
                    "-lm",
                    "-o",
                    str(build / "train.so"),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "cc",
                    str(build / "host.c"),
                    "-L" + str(build),
                    "-l:train.so",
                    "-Wl,-rpath,$ORIGIN",
                    "-o",
                    str(build / "host"),
                ],
                check=True,
                capture_output=True,
            )
            moved.mkdir()
            for name in ("host", "train.so", "libckernel_engine.so", "weights.bin"):
                shutil.copy2(build / name, moved / name)
            shutil.rmtree(build)
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("LD_LIBRARY_PATH", "PYTHONPATH")
            }
            subprocess.run(
                [str(moved / "host")],
                cwd=moved,
                env=env,
                check=True,
                capture_output=True,
            )
            (moved / "libckernel_engine.so").unlink()
            self.assertNotEqual(
                subprocess.run(
                    [str(moved / "host")], cwd=moved, env=env, capture_output=True
                ).returncode,
                0,
            )
