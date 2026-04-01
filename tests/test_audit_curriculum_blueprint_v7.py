#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "version" / "v7" / "scripts" / "audit_curriculum_blueprint_v7.py"
BLUEPRINT_PATH = ROOT / "version" / "v7" / "reports" / "spec17_curriculum_blueprint.json"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class CurriculumBlueprintAuditTests(unittest.TestCase):
    def test_spec17_blueprint_passes_audit(self) -> None:
        mod = _load_module(MODULE_PATH, "audit_curriculum_blueprint_v7")
        payload = mod.audit_blueprint(BLUEPRINT_PATH)

        self.assertEqual(payload["verdict"], "pass")
        self.assertEqual(payload["summary"]["family_count"], 3)
        self.assertEqual(payload["summary"]["intent_profile_count"], 9)
        self.assertEqual(payload["summary"]["surface_count"], 9)

    def test_missing_form_coverage_fails(self) -> None:
        mod = _load_module(MODULE_PATH, "audit_curriculum_blueprint_v7_broken")
        blueprint_doc = json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))
        blueprint_doc["intent_profiles"] = [
            row for row in blueprint_doc["intent_profiles"] if str(row.get("id")) != "registry_selection_flow"
        ]

        payload = mod.audit_blueprint_doc(blueprint_doc, blueprint_path="/tmp/broken_blueprint.json")

        self.assertEqual(payload["verdict"], "fail")
        messages = "\n".join(row["message"] for row in payload["findings"])
        self.assertIn("missing intent-profile coverage", messages)


if __name__ == "__main__":
    unittest.main()
