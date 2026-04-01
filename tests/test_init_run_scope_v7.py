#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "version" / "v7" / "scripts" / "init_run_scope_v7.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("init_run_scope_v7", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class InitRunScopeV7Tests(unittest.TestCase):
    def test_research_priors_and_lessons_are_preserved_and_rendered(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory(prefix="ck_run_scope_") as tmp:
            run_dir = Path(tmp) / "spec16_scene_bundle_l3_d192_h384_ctx768_r8"
            run_dir.mkdir(parents=True, exist_ok=True)
            args = types.SimpleNamespace(
                notes_file=None,
                notes=None,
                title="Spec16 R8",
                spec="spec16",
                rung="r8",
                family="visual_scene_bundle",
                objective="Repair exactness.",
                hypothesis="More coverage should recover exactness.",
                prompt_contract="Prompt contract.",
                output_contract="Output contract.",
                in_scope=[],
                out_of_scope=[],
                success_gate=[],
                guardrail=[],
                follow_up=[],
                research_prior=[
                    "Chinchilla (Hoffmann et al., 2022): do not undertrain on tokens.",
                    "phi-1 (Gunasekar et al., 2023): prefer clean, dense data.",
                ],
                lesson_learned=[
                    "Spec16 r7 lesson: dedupe without replacement collapsed exactness.",
                ],
                read_first=[],
                context_file=[],
            )
            scope = mod._normalize_scope_payload(
                run_dir=run_dir,
                existing_scope=None,
                template_scope=None,
                args=args,
            )

            self.assertEqual(
                scope["research_priors"],
                [
                    "Chinchilla (Hoffmann et al., 2022): do not undertrain on tokens.",
                    "phi-1 (Gunasekar et al., 2023): prefer clean, dense data.",
                ],
            )
            self.assertEqual(
                scope["lessons_learned"],
                [
                    "Spec16 r7 lesson: dedupe without replacement collapsed exactness.",
                ],
            )

            run_scope_md = mod._render_scope_markdown(scope)
            agent_md = mod._render_agent_markdown(scope)
            training_md = mod._render_training_markdown(scope)

            self.assertIn("## Research Priors", run_scope_md)
            self.assertIn("## Lessons Learned", run_scope_md)
            self.assertIn("Chinchilla (Hoffmann et al., 2022)", run_scope_md)
            self.assertIn("Spec16 r7 lesson", run_scope_md)
            self.assertIn("## Research Priors", agent_md)
            self.assertIn("## Lessons Learned", agent_md)
            self.assertIn("## Research Priors", training_md)
            self.assertIn("## Lessons Learned", training_md)


if __name__ == "__main__":
    unittest.main()
