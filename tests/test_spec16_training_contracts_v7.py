#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts" / "dataset"))

import build_spec16_probe_contract_v7 as probe_contract  # type: ignore
import generate_svg_structured_spec16_v7 as spec16_generator  # type: ignore
import materialize_spec16_scene_bundle_v7 as materializer  # type: ignore


class _FakeBase:
    @staticmethod
    def _row_from_catalog(prompt: str, output_tokens: str) -> str:
        return f"{prompt} {output_tokens}".strip()


def _bundle_output(family: str, form_token: str) -> str:
    return f"[bundle] [family:{family}] [form:{form_token}] [/bundle]"


def _catalog_row(
    *,
    family: str,
    form_token: str,
    case_id: str,
    source_asset: str,
    prompt_surface: str,
    prompt: str,
    output_tokens: str,
    split: str = "train",
) -> dict[str, str | bool]:
    return {
        "prompt": prompt,
        "output_tokens": output_tokens,
        "split": split,
        "layout": family,
        "family": family,
        "case_id": case_id,
        "form_token": form_token,
        "theme": "infra_dark",
        "tone": "blue",
        "density": "balanced",
        "background": "grid",
        "source_asset": source_asset,
        "prompt_surface": prompt_surface,
        "training_prompt": True,
    }


class Spec16TrainingContractsTests(unittest.TestCase):
    def test_hidden_probe_picker_balances_hidden_holdout_across_families(self) -> None:
        prompts: list[str] = []
        catalog: dict[str, dict[str, str | bool]] = {}
        families = [
            ("memory_map", "layer_stack", 4),
            ("timeline", "phase_flow", 2),
            ("system_diagram", "build_path", 2),
        ]
        for family, form_token, case_count in families:
            for case_idx in range(case_count):
                case_id = f"{family}_case_{case_idx}"
                for surface in ("hidden_compose", "hidden_stop"):
                    prompt = f"[task:hidden:{family}:{case_idx}:{surface}] [OUT]"
                    prompts.append(prompt)
                    catalog[prompt] = _catalog_row(
                        family=family,
                        form_token=form_token,
                        case_id=case_id,
                        source_asset=f"{family}-{case_idx}.svg",
                        prompt_surface=surface,
                        prompt=prompt,
                        output_tokens=_bundle_output(family, form_token),
                        split="probe_hidden_holdout",
                    )

        selected = probe_contract._pick_balanced_prompts(prompts, catalog, 6, hidden=True)
        family_counts = Counter(str(catalog[prompt]["family"]) for prompt in selected)

        self.assertEqual(len(selected), 6)
        self.assertEqual(
            family_counts,
            Counter(
                {
                    "memory_map": 2,
                    "timeline": 2,
                    "system_diagram": 2,
                }
            ),
        )

    def test_midtrain_stage_rows_are_deduped_family_balanced_and_repair_focused(self) -> None:
        catalog_rows: list[dict[str, str | bool]] = []
        families = [
            ("memory_map", "layer_stack"),
            ("timeline", "phase_flow"),
            ("system_diagram", "build_path"),
        ]
        stage_surfaces = [
            "tag_canonical",
            "bridge_create",
            "bridge_count_guard",
            "bridge_bundle_only",
            "bridge_style_lock",
            "repair_family_form_lock",
            "repair_bundle_singletons",
            "repair_topology_lock",
            "repair_clean_stop",
            "repair_control_stop",
            "train_hidden_compose",
            "train_hidden_stop",
            "train_hidden_style_bundle",
            "train_hidden_clean_stop",
        ]
        for family, form_token in families:
            case_id = f"{family}_base"
            output_tokens = _bundle_output(family, form_token)
            for surface in stage_surfaces:
                catalog_rows.append(
                    _catalog_row(
                        family=family,
                        form_token=form_token,
                        case_id=case_id,
                        source_asset=f"{family}.svg",
                        prompt_surface=surface,
                        prompt=f"[task:{family}:{surface}] [OUT]",
                        output_tokens=output_tokens,
                    )
                )
            catalog_rows.append(
                _catalog_row(
                    family=family,
                    form_token=form_token,
                    case_id=case_id,
                    source_asset=f"{family}.svg",
                    prompt_surface="tag_canonical",
                    prompt=f"[task:{family}:tag_canonical] [OUT]",
                    output_tokens=output_tokens,
                )
            )

        selected_rows = materializer._build_stage_catalog_rows(
            catalog_rows,
            stage="midtrain",
            split="train",
            canonical_repeat=1,
            bridge_repeat=1,
            base=_FakeBase(),
        )
        summary = materializer._summarize_stage_rows(selected_rows)
        prompt_surfaces = {str(row.get("prompt_surface") or "") for row in selected_rows}
        first_family_block = {str(row.get("family") or "") for row in selected_rows[:3]}

        self.assertEqual(summary["duplicate_unique_rows"], 0)
        self.assertEqual(summary["duplicate_rows_total"], 0)
        self.assertEqual(
            summary["family_counts"],
            {
                "memory_map": 13,
                "system_diagram": 13,
                "timeline": 13,
            },
        )
        self.assertEqual(first_family_block, {"memory_map", "timeline", "system_diagram"})
        self.assertNotIn("bridge_create", prompt_surfaces)
        self.assertIn("bridge_count_guard", prompt_surfaces)
        self.assertIn("repair_clean_stop", prompt_surfaces)
        self.assertIn("repair_control_stop", prompt_surfaces)
        self.assertIn("train_hidden_compose", prompt_surfaces)
        self.assertIn("train_hidden_clean_stop", prompt_surfaces)
        self.assertIn("bridge_style_lock", prompt_surfaces)
        self.assertIn("repair_topology_lock", prompt_surfaces)
        self.assertIn("train_hidden_style_bundle", prompt_surfaces)

    def test_train_split_includes_contrast_aug_rows(self) -> None:
        catalog_rows = [
            _catalog_row(
                family="memory_map",
                form_token="layer_stack",
                case_id="memory_map_base",
                source_asset="memory.svg",
                prompt_surface="tag_canonical",
                prompt="[task:memory_map:tag_canonical:contrast] [OUT]",
                output_tokens=_bundle_output("memory_map", "layer_stack"),
                split="contrast_aug",
            )
        ]

        selected_rows = materializer._build_stage_catalog_rows(
            catalog_rows,
            stage="midtrain",
            split="train",
            canonical_repeat=1,
            bridge_repeat=1,
            base=_FakeBase(),
        )

        self.assertEqual(len(selected_rows), 1)
        self.assertEqual(str(selected_rows[0].get("split") or ""), "contrast_aug")

    def test_pretrain_stage_keeps_broad_create_surface(self) -> None:
        catalog_rows = [
            _catalog_row(
                family="memory_map",
                form_token="layer_stack",
                case_id="memory_map_base",
                source_asset="memory.svg",
                prompt_surface="tag_canonical",
                prompt="[task:memory_map:tag_canonical] [OUT]",
                output_tokens=_bundle_output("memory_map", "layer_stack"),
            ),
            _catalog_row(
                family="memory_map",
                form_token="layer_stack",
                case_id="memory_map_base",
                source_asset="memory.svg",
                prompt_surface="bridge_create",
                prompt="[task:memory_map:bridge_create] [OUT]",
                output_tokens=_bundle_output("memory_map", "layer_stack"),
            ),
            _catalog_row(
                family="memory_map",
                form_token="layer_stack",
                case_id="memory_map_base",
                source_asset="memory.svg",
                prompt_surface="train_hidden_compose",
                prompt="[task:memory_map:train_hidden_compose] [OUT]",
                output_tokens=_bundle_output("memory_map", "layer_stack"),
            ),
        ]

        selected_rows = materializer._build_stage_catalog_rows(
            catalog_rows,
            stage="pretrain",
            split="train",
            canonical_repeat=1,
            bridge_repeat=1,
            base=_FakeBase(),
        )
        prompt_surfaces = [str(row.get("prompt_surface") or "") for row in selected_rows]

        self.assertIn("bridge_create", prompt_surfaces)
        self.assertIn("train_hidden_compose", prompt_surfaces)

    def test_contrast_case_generators_expand_unique_training_variants(self) -> None:
        base_cases = spec16_generator._cases()
        form_cases = spec16_generator._form_contrast_cases(base_cases)
        memory_style_cases = spec16_generator._memory_style_contrast_cases(base_cases)
        memory_single_axis_cases = spec16_generator._memory_single_axis_style_cases(base_cases)
        memory_frontier_cases = spec16_generator._memory_cross_form_frontier_cases(base_cases)
        system_repair_cases = spec16_generator._system_cross_form_repair_cases(base_cases)

        self.assertEqual(len(form_cases), 14)
        self.assertEqual(len(memory_style_cases), 14)
        self.assertEqual(len(memory_single_axis_cases), 33)
        self.assertEqual(len(memory_frontier_cases), 12)
        self.assertEqual(len(system_repair_cases), 6)
        self.assertTrue(all(case.split == "contrast_aug" for case in form_cases))
        self.assertTrue(all(case.split == "contrast_aug" for case in memory_style_cases))
        self.assertTrue(all(case.split == "contrast_aug" for case in memory_single_axis_cases))
        self.assertTrue(all(case.split == "cross_form" for case in memory_frontier_cases))
        self.assertTrue(all(case.split == "cross_form" for case in system_repair_cases))
        self.assertEqual(
            Counter(case.family for case in form_cases),
            Counter(
                {
                    "memory_map": 6,
                    "timeline": 2,
                    "system_diagram": 6,
                }
            ),
        )
        self.assertEqual(
            Counter(case.family for case in memory_style_cases),
            Counter({"memory_map": 14}),
        )
        self.assertEqual(
            Counter(case.family for case in memory_single_axis_cases),
            Counter({"memory_map": 33}),
        )
        self.assertEqual(
            Counter(case.family for case in memory_frontier_cases),
            Counter({"memory_map": 12}),
        )
        self.assertEqual(
            Counter(case.family for case in system_repair_cases),
            Counter({"system_diagram": 6}),
        )


if __name__ == "__main__":
    unittest.main()
