#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version" / "v7" / "scripts" / "stage_custom_tokenizer_to_run_v7.py"


class StageCustomTokenizerToRunV7Tests(unittest.TestCase):
    def test_stager_sanitizes_placeholder_special_token_and_rebuilds_reserved_list(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ck_stage_tok_") as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / ".ck_build").mkdir(parents=True, exist_ok=True)
            (run_dir / "train_init_config.json").write_text("{}", encoding="utf-8")

            tokenizer_json = root / "tokenizer.json"
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

            tokenizer_bin = root / "tokenizer_bin"
            tokenizer_bin.mkdir(parents=True, exist_ok=True)
            (tokenizer_bin / "tokenizer_meta.json").write_text("{}", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run",
                    str(run_dir),
                    "--tokenizer-json",
                    str(tokenizer_json),
                    "--tokenizer-bin",
                    str(tokenizer_bin),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)

            staged_doc = json.loads((run_dir / "tokenizer.json").read_text(encoding="utf-8"))
            staged_sidecar = json.loads((run_dir / "tokenizer_sidecar.json").read_text(encoding="utf-8"))
            reserved = (run_dir / "reserved_control_tokens.txt").read_text(encoding="utf-8").splitlines()
            ck_build_doc = json.loads((run_dir / ".ck_build" / "tokenizer.json").read_text(encoding="utf-8"))

            self.assertEqual(staged_doc["model"]["vocab"]["[bundle]...[/bundle]"], 18)
            self.assertNotIn("[bundle]...[/bundle]", [row["content"] for row in staged_doc["added_tokens"]])
            self.assertNotIn("[bundle]...[/bundle]", reserved)
            self.assertEqual(staged_sidecar["sanitized_removed_special_tokens"], ["[bundle]...[/bundle]"])
            self.assertEqual(staged_sidecar["visible_special_tokens"], reserved)
            self.assertIn("[bundle]", reserved)
            self.assertIn("[/bundle]", reserved)
            self.assertNotIn("[bundle]...[/bundle]", [row["content"] for row in ck_build_doc["added_tokens"]])


if __name__ == "__main__":
    unittest.main()
