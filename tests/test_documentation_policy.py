import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("policy", Path(__file__).resolve().parents[1] / "scripts/check_documentation_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class DocumentationPolicyTests(unittest.TestCase):
    def test_existing_debt_is_visible(self):
        report = policy.audit({"docs/old.md"}, {"docs/old.md"})
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["legacy_debt_count"], 1)

    def test_new_markdown_fails_even_with_html(self):
        self.assertEqual(policy.audit({"docs/new.md", "docs/site/a.html"}, set(), {"docs/site/a.html"})["status"], "fail")

    def test_rename_cannot_hide_behind_same_count(self):
        self.assertEqual(policy.audit({"docs/new.md"}, {"docs/old.md"})["status"], "fail")

    def test_extensions_and_root_notes(self):
        for path in ("notes.MD", "docs/a.mdx", "docs/a.markdown"):
            self.assertEqual(policy.audit({path}, set())["status"], "fail")

    def test_exact_exceptions_only(self):
        self.assertEqual(policy.audit({"README.md"}, set())["status"], "pass")
        self.assertEqual(policy.audit({"docs/README.md"}, set())["status"], "fail")

    def test_legacy_edit_needs_site_update(self):
        paths = {"docs/old.md", "docs/site/a.html"}
        self.assertEqual(policy.audit(paths, paths, {"docs/old.md"})["status"], "fail")
        self.assertEqual(policy.audit(paths, paths, paths)["status"], "pass")

    def test_deletion_is_allowed(self):
        self.assertEqual(policy.audit(set(), {"docs/old.md"}, {"docs/old.md"})["status"], "pass")
