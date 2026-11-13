"""Unsupported training precision must fail before C emission or optimizer omission."""

from copy import deepcopy
import unittest
from tests import test_v8_selective_training as base


class TrainingStorageContract(unittest.TestCase):
    def emit(self, ir):
        manifest = {"config": ir["config"]}
        layout = base.planner.build_layout(ir, manifest, 64, strict=True)
        return base.codegen.generate_c(
            ir, base.REGISTRY, manifest=manifest, layout=layout
        )

    def test_fp32_and_integer_metadata_still_generate(self):
        _, report = self.emit(base.lower(base.fixture()))
        self.assertEqual(report["parameter_gradient_order"], ["score", "up"])

    def test_unsupported_storage_in_all_tensor_roles(self):
        original = base.lower(base.fixture())
        roles = ("weight.up", "weight.bridge", "act.up", "grad.weight.up")
        for name in roles:
            for dtype in ("bf16", "fp16", "fp64", "int8", "nvfp4", "unknown"):
                with self.subTest(name=name, dtype=dtype):
                    ir = deepcopy(original)
                    ir["tensors"][name]["dtype"] = dtype
                    # Direct codegen without a layout must reject dtype first, too.
                    with self.assertRaisesRegex(
                        ValueError, "unsupported generated-training storage dtype"
                    ):
                        base.codegen.generate_c(ir, base.REGISTRY)

    def test_config_cannot_claim_bf16_for_fp32_tensors(self):
        ir = base.lower(base.fixture())
        ir["config"]["dtype"] = "bf16"
        with self.assertRaisesRegex(
            ValueError, "unsupported generated-training storage dtype in IR config"
        ):
            base.codegen.generate_c(ir, base.REGISTRY)
        ir["config"]["dtype"] = "fp32"
        with self.assertRaisesRegex(
            ValueError,
            "unsupported generated-training storage dtype in manifest config",
        ):
            base.codegen.generate_c(
                ir, base.REGISTRY, manifest={"config": {"dtype": "bf16"}}
            )

    def test_bf16_weight_cannot_silently_drop_from_optimizer_with_supplied_layout(self):
        ir = base.lower(base.fixture())
        manifest = {"config": ir["config"]}
        layout = base.planner.build_layout(ir, manifest, 64, strict=True)
        ir["tensors"]["weight.up"]["dtype"] = "bf16"
        # Before admission checks this emitted float* storage, returned no
        # skipped_ops, and silently removed "up" from parameter_gradient_order.
        with self.assertRaisesRegex(
            ValueError, "unsupported generated-training storage dtype"
        ):
            base.codegen.generate_c(ir, base.REGISTRY, manifest=manifest, layout=layout)

    def test_reference_cannot_hide_unsupported_dtype(self):
        ir = base.lower(base.fixture())
        op = next(o for o in ir["forward"] if o["op"] == "projection")
        op["weights"]["W"]["dtype"] = "bf16"
        with self.assertRaisesRegex(
            ValueError, "unsupported generated-training storage dtype"
        ):
            self.emit(ir)

    def test_reference_and_table_must_agree(self):
        ir = base.lower(base.fixture())
        op = next(o for o in ir["forward"] if o["op"] == "projection")
        op["dataflow"]["inputs"]["input"]["dtype"] = "i32"
        with self.assertRaisesRegex(
            ValueError, "conflicting generated-training storage dtype"
        ):
            self.emit(ir)

    def test_producer_reference_dtype_must_agree(self):
        ir = base.lower(base.fixture())
        op = next(o for o in ir["forward"] if o["op"] == "projection")
        op["dataflow"]["inputs"]["input"] = dict(
            from_op=0, from_output="output", dtype="i32"
        )
        with self.assertRaisesRegex(
            ValueError, "conflicting generated-training storage dtype"
        ):
            self.emit(ir)

    def test_head_authoring_does_not_reinterpret_packed_weights(self):
        try:
            import numpy as np
            from version.v8.decision.build_clef_training_ir import build_training_ir
        except ImportError as exc:
            self.skipTest("numpy unavailable: " + str(exc))
        for dtype in ("uint16", "float64", "int8"):
            with self.subTest(dtype=dtype):
                raw = {"question_projection.weight": np.ones((1, 1), dtype=dtype)}
                with self.assertRaisesRegex(ValueError, "explicit FP32 weight arrays"):
                    build_training_ir(
                        raw,
                        np.ones((1, 1), dtype="f4"),
                        questions=1,
                        options=1,
                        heads=1,
                        feedforward=1,
                    )

    def test_integer_weight_is_not_silently_omitted(self):
        ir = base.lower(base.fixture())
        ir["tensors"]["weight.up"]["dtype"] = "i32"
        with self.assertRaisesRegex(ValueError, "cannot be a weight or gradient"):
            base.codegen.generate_c(ir, base.REGISTRY)


if __name__ == "__main__":
    unittest.main()
