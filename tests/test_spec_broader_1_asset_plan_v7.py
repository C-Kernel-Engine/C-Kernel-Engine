import unittest

from version.v7.scripts.build_spec_broader_1_asset_plan_v7 import build_plan


class SpecBroader1AssetPlanTests(unittest.TestCase):
    def test_build_plan_selects_only_missing_assets(self) -> None:
        audit = {
            "asset_count": 8,
            "covered_count": 2,
            "missing_count": 6,
            "coverage_rate": 0.25,
            "covered_assets": [
                "compute-bandwidth-chasm.svg",
                "memory-layout-map.svg",
            ],
            "missing_assets": [
                "cpu-gpu-analysis.svg",
                "performance-balance.svg",
                "theory-of-constraints.svg",
                "scale-economics.svg",
                "v7-cross-entropy-parity-map.svg",
                "bf16_format.svg",
                "quantization_grouping.svg",
                "quantization_overview.svg",
                "ir-v66-edge-case-matrix.svg",
                "qwen_layer_dataflow.svg",
                "ir-dataflow-stitching.svg",
                "tokenizer-architecture.svg",
                "architecture-overview.svg",
                "rdma-observer-architecture.svg",
                "activation-memory-infographic.svg",
                "memory-reality-infographic.svg",
                "power-delivery-infographic.svg",
                "c-kernel-engine-overview.svg",
                "training-intuition-map.svg",
                "v6_plan.svg",
                "v6_plan_inkscape.svg",
                "ir-v66-gate-ladder.svg",
                "ir-v66-runtime-modes.svg",
            ],
        }
        plan = build_plan(audit)

        self.assertEqual(plan["baseline"]["current_gold_covered_assets"], 2)
        self.assertEqual(plan["first_wave"]["family_count"], 6)
        self.assertEqual(plan["first_wave"]["new_asset_count"], 23)
        self.assertEqual(plan["first_wave"]["post_wave_target_gold_asset_count"], 25)
        family_names = {family["family"] for family in plan["first_wave"]["families"]}
        self.assertIn("comparison_span_chart", family_names)
        self.assertIn("architecture_map", family_names)

    def test_build_plan_requires_assets_to_be_missing(self) -> None:
        audit = {
            "asset_count": 1,
            "covered_count": 1,
            "missing_count": 0,
            "coverage_rate": 1.0,
            "covered_assets": ["compute-bandwidth-chasm.svg"],
            "missing_assets": [],
        }
        with self.assertRaises(ValueError):
            build_plan(audit)


if __name__ == "__main__":
    unittest.main()
