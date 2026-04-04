import unittest

from version.v7.scripts.build_spec_broader_1_family_packs_v7 import build_outputs


class SpecBroader1FamilyPacksTests(unittest.TestCase):
    def test_build_outputs_marks_readiness_buckets(self) -> None:
        plan = {
            "branch": "spec_broader_1",
            "first_wave": {
                "families": [
                    {
                        "family": "comparison_span_chart",
                        "priority": "P0",
                        "lineage": "existing",
                        "why_now": "x",
                        "dsl_additions": ["a"],
                        "compiler_additions": ["b"],
                        "assets": [{"asset": "cpu-gpu-analysis.svg", "proposed_form": "dual_bar_analysis", "priority": "P0", "needs": ["dsl_extension"], "notes": ""}],
                    },
                    {
                        "family": "table_matrix",
                        "priority": "P0",
                        "lineage": "new",
                        "why_now": "x",
                        "dsl_additions": ["a"],
                        "compiler_additions": ["b"],
                        "assets": [{"asset": "bf16_format.svg", "proposed_form": "format_matrix", "priority": "P0", "needs": ["dsl_extension"], "notes": ""}],
                    },
                    {
                        "family": "architecture_map",
                        "priority": "P0",
                        "lineage": "new",
                        "why_now": "x",
                        "dsl_additions": ["a"],
                        "compiler_additions": ["b"],
                        "assets": [{"asset": "qwen_layer_dataflow.svg", "proposed_form": "layer_dataflow_stack", "priority": "P0", "needs": ["dsl_extension"], "notes": ""}],
                    },
                ]
            },
        }
        packs, queue = build_outputs(plan)

        self.assertEqual(len(packs), 3)
        buckets = {row["family"]: row["queue_bucket"] for row in queue["queue"]}
        self.assertEqual(buckets["comparison_span_chart"], "author_now")
        self.assertEqual(buckets["table_matrix"], "author_with_precursor")
        self.assertEqual(buckets["architecture_map"], "blocked_on_new_family_compiler")

    def test_queue_summary_counts(self) -> None:
        plan = {
            "branch": "spec_broader_1",
            "first_wave": {
                "families": [
                    {
                        "family": "poster_stack",
                        "priority": "P1",
                        "lineage": "existing",
                        "why_now": "x",
                        "dsl_additions": [],
                        "compiler_additions": [],
                        "assets": [],
                    },
                    {
                        "family": "timeline_flow",
                        "priority": "P1",
                        "lineage": "existing",
                        "why_now": "x",
                        "dsl_additions": [],
                        "compiler_additions": [],
                        "assets": [],
                    },
                ]
            },
        }
        _, queue = build_outputs(plan)
        self.assertEqual(queue["summary"]["author_now"], 2)
        self.assertEqual(queue["summary"]["author_with_precursor"], 0)
        self.assertEqual(queue["summary"]["blocked_on_new_family_compiler"], 0)


if __name__ == "__main__":
    unittest.main()
