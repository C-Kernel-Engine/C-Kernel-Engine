from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_source_integrity.py"
SPEC = importlib.util.spec_from_file_location("check_source_integrity", SCRIPT)
assert SPEC and SPEC.loader
integrity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(integrity)


def _load_nightly_runner():
    path = ROOT / "scripts" / "nightly_runner.py"
    spec = importlib.util.spec_from_file_location("nightly_runner_source_integrity", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    ):
        env.pop(name, None)
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class SourceIntegrityTests(unittest.TestCase):
    def test_repository_sources_are_clean(self) -> None:
        report = integrity.audit(ROOT)
        self.assertEqual(report["status"], "pass", integrity.format_failure(report))
        self.assertGreater(report["scanned_files"], 100)

    def test_marker_is_rejected_with_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src" / "broken.c"
            source.parent.mkdir(parents=True)
            source.write_text("int ok;\n" + "<" * 7 + " HEAD\nint broken;\n", encoding="utf-8")
            report = integrity.audit(root)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["conflict_markers"][0]["path"], "src/broken.c")
        self.assertEqual(report["conflict_markers"][0]["line"], 2)

    def test_real_unresolved_git_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _git(root, "init", "-q", "-b", "main")
            _git(root, "config", "user.email", "integrity@example.invalid")
            _git(root, "config", "user.name", "Integrity Test")
            source = root / "src" / "value.c"
            source.parent.mkdir(parents=True)
            source.write_text("int value = 0;\n", encoding="utf-8")
            _git(root, "add", "src/value.c")
            _git(root, "commit", "-qm", "base")
            _git(root, "checkout", "-qb", "other")
            source.write_text("int value = 1;\n", encoding="utf-8")
            _git(root, "commit", "-qam", "other")
            _git(root, "checkout", "-q", "main")
            source.write_text("int value = 2;\n", encoding="utf-8")
            _git(root, "commit", "-qam", "master")
            merge = _git(root, "merge", "other", check=False)
            self.assertNotEqual(merge.returncode, 0)
            report = integrity.audit(root)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["unmerged_paths"], ["src/value.c"])

    def test_nightly_registers_blocking_source_integrity_row(self) -> None:
        nightly = _load_nightly_runner()
        row = nightly.MAKE_TARGETS["source_integrity"]
        self.assertEqual(row["name"], "Repository Source Integrity")
        self.assertEqual(row["target"], "test-source-integrity")
        self.assertEqual(row["category"], "inference")


if __name__ == "__main__":
    unittest.main()
