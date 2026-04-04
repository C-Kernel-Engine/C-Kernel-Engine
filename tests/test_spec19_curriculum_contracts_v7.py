#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts"))
sys.path.insert(0, str(ROOT / "version" / "v7" / "scripts" / "dataset"))

import generate_svg_structured_spec19_v7 as spec19_generator  # type: ignore
import materialize_spec19_scene_bundle_v7 as materializer  # type: ignore


class _FakeBase:
    shutil = shutil

    @staticmethod
    def _row_from_catalog(prompt: str, output_tokens: str) -> str:
        return f"{prompt} {output_tokens}".strip()

    @staticmethod
    def _copy_tree(src: Path, dst: Path) -> None:
        shutil.copytree(src, dst)

    @staticmethod
    def _write_lines(path: Path, rows: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(str(row).strip() for row in rows if str(row).strip())
        if payload:
            payload += "\n"
        path.write_text(payload, encoding="utf-8")


def _bundle_output(family: str, form_token: str) -> str:
    return f"[bundle] [family:{family}] [form:{form_token}] [/bundle]"


def _parse_prompt_tags(prompt: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for token in str(prompt or "").split():
        text = token.strip()
        if not text.startswith("[") or not text.endswith("]"):
            continue
        inner = text[1:-1]
        if ":" not in inner:
            continue
        key, value = inner.split(":", 1)
        tags[key] = value
    return tags


def _catalog_row(
    *,
    family: str,
    form_token: str,
    case_id: str,
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
        "profile_id": case_id,
        "form_token": form_token,
        "theme": "signal_glow",
        "tone": "blue",
        "density": "balanced",
        "background": "mesh",
        "source_asset": f"{case_id}.svg",
        "prompt_surface": prompt_surface,
        "training_prompt": True,
    }


class Spec19CurriculumContractsTests(unittest.TestCase):
    def test_profile_cases_cover_three_families_and_nine_profiles(self) -> None:
        cases = spec19_generator._build_profile_cases()
        families = Counter(case.bundle.family for case in cases)

        self.assertEqual(len(cases), 9)
        self.assertEqual(
            families,
            Counter(
                {
                    "memory_map": 3,
                    "timeline": 3,
                    "system_diagram": 3,
                }
            ),
        )

    def test_prompt_rows_emit_declared_routing_first_scaffolds(self) -> None:
        case = spec19_generator._build_profile_cases()[0]
        surface_ids = {surface for _, surface, _, _ in spec19_generator._prompt_rows(case)}

        self.assertIn("routebook_direct", surface_ids)
        self.assertIn("routebook_direct_hint", surface_ids)
        self.assertIn("form_minimal_pair", surface_ids)
        self.assertIn("family_minimal_pair", surface_ids)
        self.assertIn("routebook_paraphrase", surface_ids)
        self.assertIn("style_topology_bridge", surface_ids)
        self.assertIn("recombination_bridge", surface_ids)

    def test_routing_surfaces_emit_multiple_training_variants(self) -> None:
        case = spec19_generator._build_profile_cases()[0]
        counts = Counter(
            surface
            for _prompt, surface, split, training_prompt in spec19_generator._prompt_rows(case)
            if split == "train" and training_prompt
        )

        self.assertGreaterEqual(counts["routebook_direct"], 4)
        self.assertGreaterEqual(counts["routebook_direct_hint"], 5)
        self.assertGreaterEqual(counts["form_minimal_pair"], 5)
        self.assertGreaterEqual(counts["family_minimal_pair"], 5)
        self.assertGreaterEqual(counts["routebook_paraphrase"], 6)
        self.assertGreaterEqual(counts["style_topology_bridge"], 5)
        self.assertGreaterEqual(counts["recombination_bridge"], 4)

    def test_form_minimal_pair_prompts_encode_sibling_form_boundary(self) -> None:
        cases = spec19_generator._build_profile_cases()
        lookup = {case.profile_id: case for case in cases}

        for case in cases:
            prompt_rows = {
                surface: prompt
                for prompt, surface, _, _ in spec19_generator._prompt_rows(case, lookup)
            }
            prompt = prompt_rows["form_minimal_pair"]
            tags = _parse_prompt_tags(prompt)
            sibling = lookup[spec19_generator.FORM_CONTRAST_PARTNER_IDS[case.profile_id]]

            self.assertEqual(tags["topic"], case.prompt_topic)
            self.assertEqual(tags["contrast_topic"], sibling.prompt_topic)
            self.assertEqual(tags["contrast_form"], sibling.bundle.form)
            self.assertEqual(tags["contrast_emphasis"], sibling.emphasis)
            self.assertIn("decision_hint", tags)
            self.assertNotEqual(tags["contrast_form"], case.bundle.form)

    def test_family_minimal_pair_prompts_encode_cross_family_boundary(self) -> None:
        cases = spec19_generator._build_profile_cases()
        lookup = {case.profile_id: case for case in cases}

        for case in cases:
            prompt_rows = {
                surface: prompt
                for prompt, surface, _, _ in spec19_generator._prompt_rows(case, lookup)
            }
            prompt = prompt_rows["family_minimal_pair"]
            tags = _parse_prompt_tags(prompt)
            sibling = lookup[spec19_generator.FAMILY_CONTRAST_PARTNER_IDS[case.profile_id]]

            self.assertEqual(tags["topic"], case.prompt_topic)
            self.assertEqual(tags["contrast_topic"], sibling.prompt_topic)
            self.assertEqual(tags["contrast_goal"], sibling.goal)
            self.assertEqual(tags["contrast_family"], sibling.bundle.family)
            self.assertIn("decision_hint", tags)
            self.assertNotEqual(tags["contrast_family"], case.bundle.family)

    def test_stage_rows_follow_blueprint_surfaces_without_duplicate_backing(self) -> None:
        catalog_rows: list[dict[str, str | bool]] = []
        families = [
            ("memory_map", "layer_stack"),
            ("timeline", "stage_sequence"),
            ("system_diagram", "build_path"),
        ]
        surfaces = [
            "explicit_bundle_anchor",
            "explicit_permuted_anchor",
            "clean_stop_anchor",
            "routebook_direct",
            "routebook_direct_hint",
            "form_minimal_pair",
            "family_minimal_pair",
            "routebook_paraphrase",
            "style_topology_bridge",
            "recombination_bridge",
        ]
        for family, form_token in families:
            for surface in surfaces:
                prompt = f"[task:{family}:{surface}] [OUT]"
                catalog_rows.append(
                    _catalog_row(
                        family=family,
                        form_token=form_token,
                        case_id=f"{family}_{surface}",
                        prompt_surface=surface,
                        prompt=prompt,
                        output_tokens=_bundle_output(family, form_token),
                    )
                )

        selected_rows = materializer._build_stage_catalog_rows(
            catalog_rows,
            stage="midtrain",
            base=_FakeBase(),
            weight_quantum=5,
        )
        summary = materializer._summarize_stage_rows(selected_rows)
        prompt_surfaces = {str(row.get("prompt_surface") or "") for row in selected_rows}
        family_counts = summary["family_counts"]

        self.assertEqual(summary["duplicate_unique_rows"], 0)
        self.assertEqual(summary["duplicate_rows_total"], 0)
        self.assertEqual(set(family_counts), {"memory_map", "timeline", "system_diagram"})
        self.assertTrue(all(int(count) > 0 for count in family_counts.values()))
        self.assertIn("explicit_bundle_anchor", prompt_surfaces)
        self.assertIn("routebook_direct", prompt_surfaces)
        self.assertIn("routebook_direct_hint", prompt_surfaces)
        self.assertIn("form_minimal_pair", prompt_surfaces)
        self.assertIn("family_minimal_pair", prompt_surfaces)
        self.assertIn("routebook_paraphrase", prompt_surfaces)
        self.assertIn("style_topology_bridge", prompt_surfaces)
        self.assertIn("recombination_bridge", prompt_surfaces)

    def test_frozen_tokenizer_copy_strips_placeholder_special_token_but_keeps_vocab_slot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spec19_tok_") as tmp:
            root = Path(tmp)
            freeze_run = root / "freeze_run"
            tokenizer_dir = freeze_run / "tokenizer_bin"
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_json = freeze_run / "tokenizer.json"
            tokenizer_json.write_text(
                json.dumps(
                    {
                        "added_tokens": [
                            {"id": 15, "content": "[bundle]", "special": True},
                            {"id": 17, "content": "[/bundle]", "special": True},
                            {"id": 18, "content": "[bundle]...[/bundle]", "special": True},
                        ],
                        "model": {
                            "vocab": {
                                "[bundle]": 15,
                                "[/bundle]": 17,
                                "[bundle]...[/bundle]": 18,
                            },
                            "merges": [],
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (tokenizer_dir / "tokenizer_meta.json").write_text("{}", encoding="utf-8")

            out = materializer._copy_frozen_tokenizer(
                freeze_run,
                root / "out_tokenizer",
                "spec19_scene_bundle",
                base=_FakeBase(),
            )

            staged_doc = json.loads(Path(out["tokenizer_json"]).read_text(encoding="utf-8"))
            reserved = (root / "out_tokenizer" / "spec19_scene_bundle_reserved_control_tokens.txt").read_text(encoding="utf-8").splitlines()

            self.assertEqual(staged_doc["model"]["vocab"]["[bundle]...[/bundle]"], 18)
            self.assertNotIn("[bundle]...[/bundle]", [row["content"] for row in staged_doc["added_tokens"]])
            self.assertNotIn("[bundle]...[/bundle]", reserved)
            self.assertIn("[bundle]", reserved)
            self.assertIn("[/bundle]", reserved)


if __name__ == "__main__":
    unittest.main()
