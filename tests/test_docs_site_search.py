import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = REPO_ROOT / "docs" / "site"
GENERATOR_PATH = SITE_DIR / "scripts" / "build_search_index.py"

spec = importlib.util.spec_from_file_location("build_search_index", GENERATOR_PATH)
generator = importlib.util.module_from_spec(spec)
sys.modules.setdefault("build_search_index", generator)
spec.loader.exec_module(generator)

INDEX_SIZE_CAP_BYTES = 1_000_000


def built_content_pages():
    metadata = json.loads((SITE_DIR / "page_metadata.json").read_text(encoding="utf-8"))
    page_meta = metadata.get("pages", {})
    pages = []
    for html_path in sorted(SITE_DIR.glob("*.html")):
        if any(part in generator.EXCLUDED_DIRS for part in html_path.parts):
            continue
        if generator._is_excluded(html_path.name, page_meta):
            continue
        pages.append(html_path.name)
    return pages


class SearchIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = generator.build_index(SITE_DIR)
        cls.index_pages = {page["url"]: page for page in cls.index["pages"]}

    def test_index_covers_all_content_pages(self):
        expected = set(built_content_pages())
        indexed = set(self.index_pages)
        self.assertEqual(expected, indexed)

    def test_excluded_pages_are_absent(self):
        for name in ("search.html", "test-report.html", "test-viewer.html"):
            self.assertNotIn(name, self.index_pages)

    def test_every_indexed_anchor_exists_in_its_page(self):
        missing = []
        for url, page in self.index_pages.items():
            html = (SITE_DIR / url).read_text(encoding="utf-8")
            for section in page["sections"]:
                anchor = section["anchor"]
                if anchor and f'id="{anchor}"' not in html:
                    missing.append(f"{url}#{anchor}")
        self.assertEqual([], missing)

    def test_excerpts_are_bounded(self):
        for page in self.index["pages"]:
            self.assertLessEqual(len(page["sections"]), generator.MAX_SECTIONS_PER_PAGE)
            for section in page["sections"]:
                self.assertLessEqual(len(section["text"]), generator.MAX_EXCERPT_CHARS)

    def test_index_size_stays_under_cap(self):
        payload = json.dumps(self.index, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        self.assertLessEqual(len(payload.encode("utf-8")), INDEX_SIZE_CAP_BYTES)

    def test_index_is_deterministic(self):
        first = generator.build_index(SITE_DIR)
        second = generator.build_index(SITE_DIR)
        self.assertEqual(
            json.dumps(first, ensure_ascii=True, sort_keys=True),
            json.dumps(second, ensure_ascii=True, sort_keys=True),
        )


class SearchPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search_html = (SITE_DIR / "search.html").read_text(encoding="utf-8")

    def test_required_markers_present(self):
        for marker in (
            'data-test="search-form"',
            'data-test="search-input"',
            'data-test="search-status"',
            'data-test="search-results"',
            "'data-test', 'search-result'",
        ):
            self.assertIn(marker, self.search_html)

    def test_header_search_box_present(self):
        self.assertIn('id="site-search-input"', self.search_html)
        self.assertIn('aria-label="Search documentation"', self.search_html)
        self.assertIn('action="search.html"', self.search_html)

    def test_no_innerhtml_with_index_content(self):
        self.assertNotIn("innerHTML", self.search_html)

    def test_index_fetch_is_relative(self):
        self.assertIn("fetch('search-index.json')", self.search_html)
        self.assertNotIn("fetch('/search-index.json')", self.search_html)
        self.assertNotIn("fetch(\"/search-index.json\")", self.search_html)


class BuildIntegrationTests(unittest.TestCase):
    def test_build_script_regenerates_search_index(self):
        # Copy the site inputs into a scratch tree so the build does not
        # touch the checked-out generated pages.
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "site"
            scratch.mkdir()
            for name in ("build.sh", "page_metadata.json"):
                shutil.copy2(SITE_DIR / name, scratch / name)
            for name in ("_pages", "_partials", "scripts"):
                shutil.copytree(SITE_DIR / name, scratch / name)

            env = dict(os.environ)
            env["CK_SITE_SKIP_SPEC_TRAINING_REFRESH"] = "1"
            result = subprocess.run(
                ["bash", "build.sh"],
                cwd=scratch,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertIn("Search index:", result.stdout)

            index_path = scratch / "search-index.json"
            self.assertTrue(index_path.exists())
            index = json.loads(index_path.read_text(encoding="utf-8"))
            scratch_pages = {
                p.name
                for p in scratch.glob("*.html")
                if not generator._is_excluded(
                    p.name,
                    json.loads((scratch / "page_metadata.json").read_text(encoding="utf-8")).get("pages", {}),
                )
            }
            self.assertEqual(scratch_pages, {page["url"] for page in index["pages"]})

            search_page = (scratch / "search.html").read_text(encoding="utf-8")
            self.assertIn('data-test="search-results"', search_page)
            self.assertIn('id="site-search-input"', search_page)


if __name__ == "__main__":
    unittest.main()
