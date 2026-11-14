"""BF16 compute in a frozen base projection; FP32 adapters and optimizer.

The oracle implements the declared saved-BF16-operand, FP32-gradient contract.
It is not a comparison with default PyTorch BF16 autocast or quantized training.
"""

import unittest
from tests import test_v8_lora_training as lora_tests
from version.v8.training.lora import attach_lora
from version.v8.training.bf16_projection import use_bf16_projection


class BF16BaseLoRA(unittest.TestCase):
    def test_frozen_base_and_fp32_adapter_ownership(self):
        ir, spec = attach_lora(
            lora_tests.fixture(), target="bridge", rank=2, alpha=3, name="demo"
        )
        base = next(
            op
            for op in ir["ops"]
            if any(
                ref.get("tensor") == "weight.bridge"
                for ref in op.get("weights", {}).values()
            )
        )
        adapted = use_bf16_projection(ir, base["op_id"])
        trained = lora_tests.lower(adapted)
        self.assertNotIn("grad.weight.bridge", trained["tensors"])
        self.assertNotIn("grad.weight.lora.demo.scale", trained["tensors"])
        for suffix in ("a", "b"):
            self.assertEqual(
                trained["tensors"]["weight.lora.demo." + suffix]["dtype"], "fp32"
            )
            self.assertEqual(
                trained["tensors"]["grad.weight.lora.demo." + suffix]["dtype"], "fp32"
            )
        saved = [
            t
            for name, t in trained["tensors"].items()
            if name.startswith("aux.projection_") and t["dtype"] == "bf16"
        ]
        self.assertEqual(len(saved), 2)
        self.assertTrue(
            all(not t["persistent"] and not t["requires_grad"] for t in saved)
        )
        from tests import test_v8_selective_training as backend

        layout = backend.planner.build_layout(
            trained, {"config": trained["config"]}, 64, strict=True
        )
        _, summary = backend.codegen.generate_c(
            trained,
            lora_tests.training_registry(),
            manifest={"config": trained["config"]},
            layout=layout,
        )
        providers = summary["declared_kernel_providers"]
        for phase in ("forward", "backward"):
            identity = "projection_bf16_train_" + phase
            self.assertEqual(providers[identity], "ck_" + identity)
        self.assertEqual(spec["dtype"], "fp32")
        self.assertNotEqual(base["kernel_id"], "projection_bf16_train_forward")

    def test_bf16_frozen_base_with_fp32_adapters(self):
        lora_tests.LoRANativeTests.certify(self, False, bf16_base=True)

    def test_bf16_frozen_base_adapter_only(self):
        lora_tests.LoRANativeTests.certify(
            self, False, adapter_only=True, bf16_base=True
        )

    def test_bf16_rectangular_frozen_base(self):
        lora_tests.LoRANativeTests.certify(self, True, bf16_base=True)

    def test_two_bf16_bases_with_distinct_adapter_ranks(self):
        lora_tests.LoRANativeTests.certify_multiple_adapters(self, bf16_base=True)
