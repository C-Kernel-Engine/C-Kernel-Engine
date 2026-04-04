#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v7" / "scripts" / "training_launcher_guard_v7.py"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class TrainingLauncherGuardV7Tests(unittest.TestCase):
    def test_guard_accepts_complete_policy_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_guard_ok_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                run_dir / "training_plan.json",
                {
                    "run_scope": {"spec": "spec17"},
                    "run_policy": {"mode": "canary"},
                    "token_budget": {
                        "recommended_pretrain_total_tokens": 90,
                        "selected_pretrain_total_tokens": 30,
                        "recommended_midtrain_total_tokens": 120,
                        "selected_midtrain_total_tokens": 40,
                        "canary_token_fraction": "1/3",
                    },
                },
            )
            _write_json(run_dir / "run_scope.json", {"spec": "spec17"})
            preflight = run_dir / "preflight.json"
            _write_json(preflight, {"canary": {"per_split": 4}})
            blueprint = run_dir / "blueprint.json"
            _write_json(blueprint, {"spec": "spec17"})
            audit = run_dir / "blueprint_audit.json"
            _write_json(audit, {"verdict": "pass"})
            decision = run_dir / "decision.json"
            _write_json(decision, {"training_allowed": True})

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run",
                    str(run_dir),
                    "--preflight",
                    str(preflight),
                    "--blueprint",
                    str(blueprint),
                    "--blueprint-audit",
                    str(audit),
                    "--decision-artifact",
                    str(decision),
                    "--require-run-scope",
                    "--require-run-policy",
                    "--require-token-budget",
                    "--require-canary-metadata",
                    "--allow-non-cache-run-dir",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["reasons"], [])

    def test_guard_blocks_missing_canary_metadata_and_blocking_decision(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_guard_block_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                run_dir / "training_plan.json",
                {
                    "run_scope": {"spec": "spec16"},
                    "run_policy": {"mode": "pilot"},
                    "token_budget": {
                        "recommended_pretrain_total_tokens": 90,
                        "selected_pretrain_total_tokens": 30,
                    },
                },
            )
            _write_json(run_dir / "run_scope.json", {"spec": "spec16"})
            preflight = run_dir / "preflight.json"
            _write_json(preflight, {"stages": {}})
            decision = run_dir / "decision.json"
            _write_json(decision, {"training_allowed": False, "default_action": "decode_repair", "reasons": ["blocked"]})

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run",
                    str(run_dir),
                    "--preflight",
                    str(preflight),
                    "--decision-artifact",
                    str(decision),
                    "--require-run-scope",
                    "--require-run-policy",
                    "--require-token-budget",
                    "--require-canary-metadata",
                    "--allow-non-cache-run-dir",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(proc.stdout)
            joined = " ".join(payload["reasons"])
            self.assertIn("token_budget is missing recommended_midtrain_total_tokens", joined)
            self.assertIn("preflight JSON is missing canary metadata", joined)
            self.assertIn("training blocked by decision artifact", joined)

    def test_guard_blocks_repo_local_run_dir_by_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_guard_repo_local_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(run_dir / "training_plan.json", {})

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run",
                    str(run_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(proc.stdout)
            joined = " ".join(payload["reasons"])
            self.assertIn("canonical cache train root", joined)


if __name__ == "__main__":
    unittest.main()
