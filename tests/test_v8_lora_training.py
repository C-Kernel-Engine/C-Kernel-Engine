"""FP32 adapter authoring and independent native training certification."""

import copy
import unittest

from tests.test_v8_selective_training import fixture, BINDINGS, load, backward
from version.v8.training.lora import attach_lora, initialize_lora
from version.v8.decision.build_clef_training_ir import training_registry


def build():
    return attach_lora(fixture(), target="bridge", rank=2, alpha=3, name="demo")


def lower(ir):
    return backward.synthesize_ir2_backward(
        ir,
        training_registry(),
        BINDINGS,
        load("version/v8/scripts/grad_rules_v8.json"),
        strict=True,
        allow_partial=False,
        checkpoint_policy="none",
    )


class LoRAContractTests(unittest.TestCase):
    def test_immutable_graph_and_freeze(self):
        original = fixture()
        saved = copy.deepcopy(original)
        ir, spec = attach_lora(original, target="bridge", rank=2, alpha=3, name="demo")
        self.assertEqual(original, saved)
        self.assertEqual(spec["parameter_count"], 16)
        self.assertEqual(len(ir["ops"]), len(original["ops"]) + 4)
        trained = lower(ir)
        for name in ("bridge", "features", "lora.demo.scale"):
            self.assertNotIn("grad.weight." + name, trained["tensors"])
        for name in ("up", "score", "lora.demo.a", "lora.demo.b"):
            self.assertIn("grad.weight." + name, trained["tensors"])

    def test_initialization_and_rejection(self):
        import numpy as np

        ir, spec = build()
        a = initialize_lora(spec, seed=41)
        b = initialize_lora(spec, seed=41)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertTrue(np.any(a["lora.demo.a"]))
        self.assertFalse(np.any(a["lora.demo.b"]))
        self.assertEqual(a["lora.demo.scale"][0], 1.5)
        for override in (
            {"rank": 0},
            {"rank": True},
            {"rank": 5},
            {"alpha": float("nan")},
            {"alpha": float("inf")},
            {"alpha": 0},
            {"target": "absent"},
            {"name": "bad/name"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                attach_lora(
                    fixture(),
                    **(dict(target="bridge", rank=2, alpha=3, name="demo") | override)
                )
        with self.assertRaises(ValueError):
            attach_lora(ir, target="bridge", rank=2, alpha=3, name="demo")
        for mutation in ("dtype", "stride", "shared"):
            bad = fixture()
            if mutation == "dtype":
                bad["tensors"]["weight.bridge"]["dtype"] = "bf16"
            if mutation == "stride":
                bad["ops"][2]["params"] = {"call_constants": {"linear_input_stride": 8}}
            if mutation == "shared":
                bad["ops"][3]["weights"] = copy.deepcopy(bad["ops"][2]["weights"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                attach_lora(bad, target="bridge", rank=2, alpha=3, name="demo")

    def test_invalid_initialization_spec(self):
        _, spec = build()
        for field, value in (
            ("rank", 0),
            ("parameter_count", 3),
            ("scale", 0),
            ("alpha", float("nan")),
            ("parameters", {}),
            ("dtype", "bf16"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                initialize_lora(dict(spec, **{field: value}), seed=41)
        for seed in (-1, True, 2**32):
            with self.assertRaises(ValueError):
                initialize_lora(spec, seed=seed)

    def test_duplicate_operation_ids_and_scale_range(self):
        original = fixture()
        original["ops"][2]["op_id"] = 0
        with self.assertRaises(ValueError):
            attach_lora(original, target="bridge", rank=2, alpha=3, name="demo")
        for alpha in (1e-45, 1e40):
            with self.assertRaises(ValueError):
                attach_lora(
                    fixture(), target="bridge", rank=2, alpha=alpha, name="demo"
                )


class LoRANativeTests(unittest.TestCase):
    def test_native_adamw_resume_and_export(self):
        self.certify(False)

    def test_rectangular_projection(self):
        self.certify(True)

    def test_adapter_only_training(self):
        self.certify(False, adapter_only=True)

    def certify(self, rectangular, adapter_only=False):
        import numpy as np
        from tests.v8_training_test_helpers import certify_training

        original = fixture()
        if rectangular:
            shapes = {
                "weight.up": [6, 4],
                "act.up": [4, 6],
                "weight.bridge": [3, 6],
                "act.bridge": [4, 3],
                "weight.score": [4, 3],
            }

            def reshape(item):
                if isinstance(item, dict):
                    if item.get("tensor") in shapes:
                        shape = shapes[item["tensor"]]
                        item.update(shape=shape, numel=shape[0] * shape[1])
                    for v in item.values():
                        reshape(v)
                elif isinstance(item, list):
                    for v in item:
                        reshape(v)

            reshape(original)
            for name, shape in shapes.items():
                original["tensors"][name].update(shape=shape, numel=shape[0] * shape[1])
        ir, spec = attach_lora(original, target="bridge", rank=2, alpha=3, name="demo")
        if adapter_only:

            def freeze_other(item):
                if isinstance(item, dict):
                    if item.get("tensor") in ("weight.up", "weight.score"):
                        item["requires_grad"] = False
                    for value in item.values():
                        freeze_other(value)
                elif isinstance(item, list):
                    for value in item:
                        freeze_other(value)

            freeze_other(ir)
            for name in ("up", "score"):
                ir["tensors"]["weight." + name]["requires_grad"] = False
        rng = np.random.default_rng(837)
        values = {
            "features": np.eye(4, dtype="f"),
            "bridge": np.eye(4, dtype="f") + rng.normal(0, 0.1, (4, 4)).astype("f"),
            "up": rng.normal(0, 0.2, (4, 4)).astype("f"),
            "score": rng.normal(0, 0.2, (4, 4)).astype("f"),
        }
        if adapter_only:
            values["up"] = np.eye(4, dtype="f")
            values["score"] = np.eye(4, dtype="f")
        if rectangular:
            for name in ("up", "bridge", "score"):
                values[name] = rng.normal(
                    0, 0.2, original["tensors"]["weight." + name]["shape"]
                ).astype("f")
        values.update(initialize_lora(spec, seed=41))

        def forward(p):
            x = p["features"] @ p["up"].T
            y = (
                x @ p["bridge"].T
                + (x @ p["lora.demo.a"].T @ p["lora.demo.b"].T) * p["lora.demo.scale"]
            )
            return y @ p["score"].T

        def check_gradients(step, p):
            if step == 0:
                self.assertEqual(float(p["lora.demo.a"].grad.abs().max()), 0.0)
                self.assertGreater(float(p["lora.demo.b"].grad.abs().max()), 0.0)
            if step == 1:
                self.assertGreater(float(p["lora.demo.a"].grad.abs().max()), 0.0)

        certify_training(
            self,
            lower(ir),
            training_registry(),
            values,
            (
                {"lora.demo.a", "lora.demo.b"}
                if adapter_only
                else {"up", "score", "lora.demo.a", "lora.demo.b"}
            ),
            forward,
            gradient_check=check_gradients,
            probe_parameter="lora.demo.b" if adapter_only else "up",
            fresh_process=True,
            check_evaluation=True,
            checked_runtime=True,
            extra_sources=(
                "src/kernels/linear_rows_checked.c",
                "src/kernels/transformer_rows_checked.c",
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
