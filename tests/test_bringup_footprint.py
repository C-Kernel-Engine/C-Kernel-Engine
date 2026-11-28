import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPORT = Path(__file__).resolve().parents[1] / "scripts/report_bringup_footprint.py"


class BringupFootprintTests(unittest.TestCase):
    def test_counts_only_topic_changes_including_binary_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

            def commit():
                git("add", ".")
                git("-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "fixture")
                return git("rev-parse", "HEAD")

            policy = root / "version/v8/dsl_policy.json"
            policy.parent.mkdir(parents=True)
            policy.write_text(json.dumps({"compiler_files": {"compiler.py": {}},
                                          "compiler_functions": {}}))
            git("init", "-q")
            ancestor = commit()
            (root / "unrelated").touch()
            base = commit()
            git("checkout", "-q", "--detach", ancestor)
            (root / "compiler.py").write_text("one\ntwo\n")
            fixtures = root / "tests/fixtures"
            fixtures.mkdir(parents=True)
            (fixtures / "sample").write_bytes(b"a\0b")
            head = commit()
            rows = json.loads(subprocess.check_output(
                [sys.executable, REPORT, base, head, "--json"], cwd=root))
            self.assertEqual(rows, [
                dict(path="compiler.py", area="Policy-scoped compiler", added=2, removed=0),
                dict(path="tests/fixtures/sample", area="Fixtures", added=None, removed=None),
            ])
