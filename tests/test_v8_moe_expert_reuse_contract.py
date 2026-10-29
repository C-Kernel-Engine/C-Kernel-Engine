"""The map-selected reuse path is limited to supported prefill formats."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
from build_ir_v8 import _kernel_scratch_size_bytes, resolve_swiglu_moe_provider


class MoEExpertReuseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(
            (ROOT / "version/v8/kernel_maps/KERNEL_REGISTRY.json").read_text()
        )
        cls.by_id = {item["id"]: item for item in cls.registry["kernels"]}

    def test_prefill_selects_reuse_and_decode_retains_existing_provider(self) -> None:
        for kind in ("q5_0", "q8_0"):
            with self.subTest(kind=kind):
                quant = {
                    "moe_expert_gate": "q4_k",
                    "moe_expert_up": "q4_k",
                    "moe_expert_down": kind,
                }
                for phase, suffix in (("prefill", "_reuse"), ("decode", "")):
                    selected = resolve_swiglu_moe_provider(
                        self.registry, kernel_op="moe_swiglu_expert_mlp",
                        layer_quant=quant, weight_prefix="moe_expert",
                        mode=phase, prefer_q8_activation=False,
                    )
                    self.assertEqual(
                        selected,
                        f"moe_swiglu_expert_forward_q4k_{kind}{suffix}",
                    )

    def test_reuse_workspace_is_bounded_for_flash_next_shape(self) -> None:
        dims = {"R": 2323, "H": 2560, "I": 640, "E": 512, "K": 10}
        for kind in ("q5_0", "q8_0"):
            with self.subTest(kind=kind):
                provider = self.by_id[
                    f"moe_swiglu_expert_forward_q4k_{kind}_reuse"
                ]
                self.assertEqual(
                    _kernel_scratch_size_bytes(provider["scratch"][0], dims, {}),
                    130045376,
                )
                self.assertEqual(provider["constraints"]["alignment"]["I"], 32)
                self.assertEqual(provider["scratch"][0]["lifetime"], "kernel_call")


if __name__ == "__main__":
    unittest.main()
