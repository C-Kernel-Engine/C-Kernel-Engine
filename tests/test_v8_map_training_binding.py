"""Map-owned direct VJP bindings reject incomplete or incompatible semantics."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from version.v8.scripts.map_training_binding_v8 import resolve_training_binding

ROOT = Path(__file__).resolve().parents[1]


class MapTrainingBindingTests(unittest.TestCase):
    def setUp(self):
        self.provider = json.loads(
            (
                ROOT / "version/v8/kernel_maps/linear_rows_checked_no_bias_f32.json"
            ).read_text()
        )
        self.rules = json.loads(
            (ROOT / "version/v8/scripts/grad_rules_v8.json").read_text()
        )["rules"]
        self.args = dict(
            inputs={"input": dict(tensor="input.features", dtype="fp32")},
            weights={"weight": dict(tensor="weight.projection", dtype="fp32")},
            output=dict(tensor="act.result", dtype="fp32"),
            constants=dict(
                linear_rows=3, linear_input_channels=4, linear_output_channels=5
            ),
        )

    def test_binding_uses_semantics_not_provider_spelling(self):
        before = deepcopy(self.args)
        rule, constants, saved = resolve_training_binding(
            self.provider, self.rules, **self.args
        )
        self.assertEqual(rule, "linear_rows_train_backward_nobias")
        self.assertEqual(
            (constants["T"], constants["aligned_in"], constants["aligned_out"]),
            (3, 4, 5),
        )
        self.assertEqual(saved["weight"]["tensor"], "weight.projection")
        renamed = deepcopy(self.provider)
        renamed["id"] = "another_compatible_projection_provider"
        self.assertEqual(
            resolve_training_binding(renamed, self.rules, **self.args),
            (rule, constants, saved),
        )
        self.assertEqual(self.args, before)

    def test_invalid_map_bindings_fail_before_backward(self):
        for field, value in (
            ("schema", "unknown"),
            ("operation_interface", "wrong.interface"),
            ("gradient_rule", "missing_rule"),
            ("gradient_rule", []),
            ("saved_operands", {}),
            ("saved_operands", {"input": "input:input", "weight": "weight:missing"}),
            ("constant_bindings", {"T": "absent"}),
            ("constant_bindings", {"T": True}),
            ("constant_bindings", {"T": -1}),
        ):
            bad = deepcopy(self.provider)
            bad["training_contract"]["differentiation"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                resolve_training_binding(bad, self.rules, **self.args)
        for defect in ("dtype", "port", "output"):
            args = deepcopy(self.args)
            if defect == "dtype":
                args["weights"]["weight"]["dtype"] = "bf16"
            if defect == "port":
                args["weights"]["bias"] = dict(dtype="fp32")
            if defect == "output":
                args["output"]["dtype"] = "bf16"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                resolve_training_binding(self.provider, self.rules, **args)

    def test_bias_and_addition_ports_are_explicit(self):
        biased = json.loads(
            (ROOT / "version/v8/kernel_maps/linear_rows_checked_f32.json").read_text()
        )
        with self.assertRaisesRegex(ValueError, "ports"):
            resolve_training_binding(biased, self.rules, **self.args)
        args = deepcopy(self.args)
        args["weights"]["bias"] = dict(tensor="weight.bias", dtype="fp32")
        rule, _, saved = resolve_training_binding(biased, self.rules, **args)
        self.assertEqual(rule, "linear_rows_train_backward")
        self.assertEqual(set(saved), {"input", "weight"})
        addition = json.loads(
            (ROOT / "version/v8/kernel_maps/add_rows_checked_f32.json").read_text()
        )
        args["inputs"] = {
            p["name"]: dict(tensor="act." + p["name"], dtype="fp32")
            for p in addition["inputs"]
        }
        args["weights"] = {}
        _, _, saved = resolve_training_binding(addition, self.rules, **args)
        self.assertEqual(saved, {})

    def test_absent_contract_is_not_inferred(self):
        missing = deepcopy(self.provider)
        del missing["training_contract"]["differentiation"]
        self.assertIsNone(resolve_training_binding(missing, self.rules, **self.args))


if __name__ == "__main__":
    unittest.main()
