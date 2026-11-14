"""Relocated generated forward/backward/AdamW executes in a C-only process."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from tests import test_v8_selective_training as backend
from tests.test_v8_training_float_inputs import graph
from version.v8.scripts import run_training_workflow_v8 as workflow

ROOT = Path(__file__).resolve().parents[1]

HOST = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int ck_train_init(const float *, const int32_t *, int);
int ck_train_set_float_inputs_ex(const float *, size_t);
int ck_train_set_batch_ex(const int32_t *, const int32_t *, int);
int ck_train_eval_step(float *);
int ck_train_step_ex(const int32_t *, const int32_t *, int, float *, float);
int ck_train_get_opt_step(void);
int ck_train_get_weight_snapshot_numel(void);
int ck_train_export_weight_snapshot(float *, int);
void ck_train_free(void);
void ck_set_strict_parity(int);
int main(void) {
    float weights[CK_WEIGHT_ELEMENTS], features[16], loss = -1, before = -1;
    int32_t sizes[CK_WEIGHT_COUNT] = {CK_WEIGHT_SIZES}, ids[4] = {0,1,2,3};
    FILE *input = fopen("request.bin", "rb");
    if (!input || fread(weights, sizeof(float), CK_WEIGHT_ELEMENTS, input) != CK_WEIGHT_ELEMENTS ||
        fread(features, sizeof(float), 16, input) != 16) return 1;
    fclose(input);
    if (ck_train_init(weights, sizes, CK_WEIGHT_COUNT) != CK_WEIGHT_COUNT) return 2;
    ck_set_strict_parity(1);
    if (ck_train_set_float_inputs_ex(features, 16) != 16 ||
        ck_train_set_batch_ex(ids, ids, 4) <= 0 ||
        ck_train_eval_step(&before) <= 0) return 3;
    for (int i = 0; i < 40; ++i)
        if (ck_train_step_ex(ids, ids, 4, &loss, 0.03f) <= 0) return 4;
    if (ck_train_eval_step(&loss) <= 0 || !isfinite(loss) ||
        loss >= before || ck_train_get_opt_step() != 40) return 5;
    if (ck_train_get_weight_snapshot_numel() != CK_WEIGHT_ELEMENTS ||
        ck_train_export_weight_snapshot(weights, CK_WEIGHT_ELEMENTS) != CK_WEIGHT_ELEMENTS) return 6;
    FILE *out = fopen("result.bin", "wb");
    if (!out || fwrite(&before, sizeof(float), 1, out) != 1 ||
        fwrite(&loss, sizeof(float), 1, out) != 1 ||
        fwrite(weights, sizeof(float), CK_WEIGHT_ELEMENTS, out) != CK_WEIGHT_ELEMENTS) return 7;
    fclose(out);
    ck_train_free();
    return 0;
}
"""


class PortableTrainingRuntime(unittest.TestCase):

    def test_provider_dependency_resolution_rejects_missing_contracts(self):
        compiler = workflow._load_module(
            "cke_portable_dependency_test", ROOT / "version/v7/scripts/ck_run_v7.py"
        )
        summary = {
            "declared_kernel_providers": {"external": "missing", "base": "present"}
        }
        registry = {
            "kernels": [
                {
                    "id": "external",
                    "impl": {"sources": ["src/kernels/linear_rows_checked.c"]},
                }
            ]
        }
        with patch.object(
            compiler.subprocess, "check_output", return_value="0000 T present"
        ):
            self.assertEqual(
                compiler._resolve_training_provider_sources(
                    summary, registry, Path("engine")
                ),
                [ROOT / "src/kernels/linear_rows_checked.c"],
            )
            for bad_summary, bad_registry in (
                ({}, registry),
                (summary, {"kernels": []}),
                (
                    summary,
                    {
                        "kernels": [
                            {"id": "external", "impl": {"sources": ["../outside.c"]}}
                        ]
                    },
                ),
            ):
                with self.subTest(registry=bad_registry), self.assertRaises(ValueError):
                    compiler._resolve_training_provider_sources(
                        bad_summary, bad_registry, Path("engine")
                    )

    def test_relocated_c_host_trains_without_checkout_or_python(self):
        self.relocated_c_host()

    def test_relocated_lora_trains_without_checkout_or_python(self):
        self.relocated_c_host(adapters=True)

    def relocated_c_host(self, adapters=False):
        engine = ROOT / "build/libckernel_engine.so"
        if not engine.is_file():
            self.skipTest("native engine unavailable")
        authored = graph()
        registry = backend.REGISTRY
        if adapters:
            from version.v8.training.lora import attach_lora_adapters
            from version.v8.training.lora_weights import (
                prepare_lora_weights,
                resolve_lora_parameters,
            )
            from tests.test_v8_lora_training import lower, training_registry

            authored, _ = attach_lora_adapters(
                authored,
                adapters=[
                    dict(target="up", rank=2, alpha=2, name="first"),
                    dict(target="bridge", rank=1, alpha=2, name="second"),
                ],
            )
            registry = training_registry()
            ir = lower(authored)
        else:
            ir = backend.lower(authored)
        layout = backend.planner.build_layout(
            ir, {"config": ir["config"]}, 64, strict=True
        )
        source, summary = backend.codegen.generate_c(
            ir, registry, manifest={"config": ir["config"]}, layout=layout
        )
        if not adapters:
            self.assertEqual(summary["init_weight_numel"], [16, 16, 16])
        compiler = workflow._load_module(
            "cke_portable_training_test", ROOT / "version/v7/scripts/ck_run_v7.py"
        )
        rng = np.random.default_rng(731)
        initial = rng.normal(0, 0.4, 48).astype("f4")
        features = rng.normal(0, 0.4, 16).astype("f4")
        if adapters:
            schema = resolve_lora_parameters(authored)
            values = {
                name: (np.eye(4) + rng.normal(0, 0.1, meta["shape"])).astype("f4")
                for name, meta in schema.items()
                if not name.startswith("lora.")
            }
            values, _ = prepare_lora_weights(
                authored, values, seeds={"first": 41, "second": 42}
            )
            initial = np.concatenate(
                [values[n].ravel() for n in summary["init_weight_order"]]
            )
            features = np.eye(4, dtype="f4").ravel()
        host_source = (
            f"#define CK_WEIGHT_ELEMENTS {len(initial)}\n"
            f"#define CK_WEIGHT_COUNT {len(summary['init_weight_order'])}\n"
            "#define CK_WEIGHT_SIZES "
            + ",".join(map(str, summary["init_weight_numel"]))
            + "\n"
            + HOST
        )
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build"
            bundle = Path(tmp) / "bundle"
            build.mkdir()
            bundle.mkdir()
            (build / "train.c").write_text(source)
            # A preexisting convenience symlink must not escape into the deployed bundle.
            (bundle / "libckernel_engine.so").symlink_to(engine)
            identity = compiler._compile_train_runtime_with_identity(
                build / "train.c",
                bundle / "train.so",
                lib_ck=engine,
                cflags=(),
                defines={"CK_NUM_TOKENS": 4},
                relocatable=True,
                extra_sources=compiler._resolve_training_provider_sources(
                    summary, registry, engine
                ),
            )
            self.assertFalse((bundle / "libckernel_engine.so").is_symlink())
            self.assertEqual(identity["inputs"]["rpath"], "$ORIGIN")
            self.assertEqual(
                identity["inputs"]["kernel_library_sha256"],
                workflow._sha256(bundle / "libckernel_engine.so"),
            )
            (build / "host.c").write_text(host_source)
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(build / "host.c"),
                    "-L" + str(bundle),
                    "-l:train.so",
                    "-lckernel_engine",
                    "-Wl,-rpath,$ORIGIN",
                    "-lm",
                    "-o",
                    str(bundle / "host"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            (bundle / "request.bin").write_bytes(initial.tobytes() + features.tobytes())
            # No Python, compiler, module paths or loader overrides in the deployed run.
            env = {
                "PATH": "/usr/bin:/bin",
                "CK_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
            subprocess.run(
                ["./host"],
                cwd=bundle,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            baseline = (bundle / "result.bin").read_bytes()
            relocated = Path(tmp) / "relocated"
            shutil.move(bundle, relocated)
            shutil.rmtree(build)
            (relocated / "result.bin").unlink()
            for name in ("host", "train.so"):
                dynamic = subprocess.check_output(
                    ["readelf", "-d", str(relocated / name)], text=True
                )
                self.assertNotIn(str(ROOT), dynamic)
                self.assertNotIn(str(bundle), dynamic)
                self.assertIn("$ORIGIN", dynamic)
            dependencies = subprocess.check_output(
                ["ldd", "./train.so"], cwd=relocated, env=env, text=True
            )
            self.assertIn(str(relocated / "libckernel_engine.so"), dependencies)
            subprocess.run(
                ["./host"],
                cwd=relocated,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual((relocated / "result.bin").read_bytes(), baseline)
            result = np.frombuffer(baseline, dtype="f4")
            self.assertLess(result[1], result[0])
            for name in (
                ["bridge", "up", "lora.first.scale", "lora.second.scale"]
                if adapters
                else ["bridge"]
            ):
                index = summary["init_weight_order"].index(name)
                begin = sum(summary["init_weight_numel"][:index])
                end = begin + summary["init_weight_numel"][index]
                np.testing.assert_array_equal(
                    result[2 + begin : 2 + end], initial[begin:end]
                )
            # Removing the bundled dependency must fail rather than find a checkout copy.
            (relocated / "libckernel_engine.so").unlink()
            failed = subprocess.run(
                ["./host"], cwd=relocated, env=env, capture_output=True, text=True
            )
            self.assertNotEqual(failed.returncode, 0)

    def test_complete_small_head_runs_relocated_in_c(self):
        try:
            import torch  # Only the development fixture builder uses PyTorch.
        except ImportError as exc:
            self.skipTest("fixture builder unavailable: " + str(exc))
        if not (ROOT / "build/libckernel_engine.so").is_file():
            self.skipTest("native engine unavailable")
        from tests.test_v8_clef_training_recipe import fixture
        from version.v8.decision.training_package import load_package
        from version.v8.decision.train_clef_head import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, pin, _ = fixture(root)
            doc, raw, records = load_package(package, pin)
            folder = root / "head"
            folder.mkdir()
            rt = Runtime(folder, doc, raw, records[0]["arrays"])
            try:
                arrays = records[0]["arrays"]
                initial = rt.snapshot().copy()
                metadata = rt.batch(arrays)
                sizes = rt.sizes.copy()
                tokens = rt.tokens
                (folder / "request.bin").write_bytes(
                    initial.tobytes() + arrays["hidden"].tobytes() + metadata.tobytes()
                )
                for _ in range(3):
                    rt.step(arrays, 0.001)
                expected = rt.snapshot().copy()
                expected_loss, _ = rt.evaluate(arrays)
            finally:
                rt.close()
            identity_path = folder / "build-identity.json"
            identity_bytes = identity_path.read_bytes()
            identity_path.write_bytes(identity_bytes + b" ")
            with self.assertRaisesRegex(ValueError, "runtime identity mismatch"):
                Runtime(folder, doc, raw, arrays, resume=True)
            identity_path.write_bytes(identity_bytes)
            n, nf, ni, ns = (
                initial.size,
                arrays["hidden"].size,
                metadata.size,
                sizes.size,
            )
            host = f"""
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
int ck_train_init(const float *, const int32_t *, int);
int ck_train_set_float_inputs_ex(const float *, size_t);
int ck_train_step_with_i32_ex(const int32_t *, const int32_t *, int,
                             const int32_t *, size_t, float *, float);
int ck_train_eval_step(float *);
int ck_train_export_weight_snapshot(float *, int);
int ck_train_get_opt_step(void);
void ck_train_free(void);
void ck_set_strict_parity(int);
int main(void) {{
    float weights[{n}], features[{nf}], loss = -1;
    int32_t metadata[{ni}], ids[{tokens}] = {{0}};
    int32_t sizes[{ns}] = {{{','.join(map(str, sizes.tolist()))}}};
    FILE *f = fopen("request.bin", "rb");
    if (!f || fread(weights, 4, {n}, f) != {n} ||
        fread(features, 4, {nf}, f) != {nf} ||
        fread(metadata, 4, {ni}, f) != {ni}) return 1;
    fclose(f);
    if (ck_train_init(weights, sizes, {ns}) != {ns}) return 2;
    ck_set_strict_parity(1);
    if (ck_train_set_float_inputs_ex(features, {nf}) != {nf}) return 3;
    for (int i = 0; i < 3; ++i)
        if (ck_train_step_with_i32_ex(ids, ids, {tokens}, metadata, {ni}, &loss, 0.001f) <= 0)
            return 4;
    if (ck_train_eval_step(&loss) <= 0 || ck_train_get_opt_step() != 3 ||
        ck_train_export_weight_snapshot(weights, {n}) != {n}) return 5;
    f = fopen("result.bin", "wb");
    if (!f || fwrite(&loss, 4, 1, f) != 1 || fwrite(weights, 4, {n}, f) != {n}) return 6;
    fclose(f);
    ck_train_free();
    return 0;
}}
"""
            source = root / "host.c"
            source.write_text(host)
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(source),
                    "-L" + str(folder),
                    "-l:train.so",
                    "-lckernel_engine",
                    "-Wl,-rpath,$ORIGIN",
                    "-o",
                    str(folder / "host"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            deployed = root / "standalone"
            deployed.mkdir()
            for name in ("host", "train.so", "libckernel_engine.so", "request.bin"):
                shutil.copyfile(folder / name, deployed / name)
            (deployed / "host").chmod(0o700)
            shutil.rmtree(folder)
            source.unlink()
            subprocess.run(
                ["./host"],
                cwd=deployed,
                env={
                    "PATH": "/usr/bin:/bin",
                    "CK_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
                check=True,
                capture_output=True,
                text=True,
            )
            result = np.fromfile(deployed / "result.bin", dtype="f4")
            self.assertEqual(float(result[0]), expected_loss)
            np.testing.assert_array_equal(result[1:], expected)


if __name__ == "__main__":
    unittest.main()
