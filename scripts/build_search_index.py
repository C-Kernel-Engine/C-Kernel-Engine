#!/usr/bin/env python3
"""Build the client-side search index for the CKE documentation site.

Parses the assembled top-level HTML pages in docs/site/ (run this after
build.sh has combined the partials and pages) and emits search-index.json
next to them. The index is consumed by search.html, which runs entirely in
the browser with no server-side component.

Python 3 standard library only. The output is deterministic: pages are
sorted by URL, sections stay in document order, keys are sorted, and no
timestamps are embedded, so rebuilds do not churn the file.

Usage:
    python3 build_search_index.py [--site-dir DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

# Pages are excluded when page_metadata.json marks them noindex or keeps
# them out of the sitemap, or when they are listed here. Test report /
# viewer artifacts and the search page itself carry no prose worth indexing.
EXCLUDED_PAGES = frozenset(
    {
        "search.html",
        "test-report.html",
        "test-viewer.html",
    }
)

# Directories that must never be indexed even if HTML leaks into them.
EXCLUDED_DIRS = ("doxygen", "nightly-results")

# Bound each section excerpt so the index stays a few hundred KB.
MAX_EXCERPT_CHARS = 220
MAX_SECTIONS_PER_PAGE = 48

_WHITESPACE_RE = re.compile(r"\s+")
_TITLE_SUFFIX_RE = re.compile(r"\s*\|\s*C-Kernel-Engine\s*$")

# Tags whose subtree never contributes searchable prose.
_SKIP_TAGS = frozenset({"script", "style", "svg", "noscript", "template", "nav"})


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


class _PageParser(HTMLParser):
    """Extract the page title plus h2/h3-anchored sections from <main>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.sections: list[dict[str, str]] = []
        self._in_title = False
        self._in_main = False
        self._skip_depth = 0
        self._current: dict[str, str] | None = None
        self._text_parts: list[str] = []

    # -- helpers ---------------------------------------------------------
    def _flush_section(self) -> None:
        if self._current is None:
            return
        text = _collapse(" ".join(self._text_parts))
        self._current["text"] = text[:MAX_EXCERPT_CHARS]
        if len(self.sections) < MAX_SECTIONS_PER_PAGE:
            self.sections.append(self._current)
        self._current = None
        self._text_parts = []

    def _start_section(self, anchor: str, level: str) -> None:
        self._flush_section()
        self._current = {"anchor": anchor, "level": level, "heading": "", "text": ""}
        self._text_parts = []

    # -- HTMLParser hooks -------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "title":
            self._in_title = True
            return
        if tag == "main" and not self._in_main:
            self._in_main = True
            # Intro prose before the first heading belongs to the page top.
            self._start_section("", "intro")
            return
        if not self._in_main:
            return
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in ("h2", "h3"):
            anchor = attr.get("id") or ""
            self._start_section(anchor, tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if not self._in_main:
            return
        if tag == "main":
            self._flush_section()
            self._in_main = False
            return
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if not self._in_main or self._skip_depth > 0:
            return
        if self._current is None:
            return
        text = _collapse(data)
        if not text:
            return
        # Heading data arrives before any body data for the section.
        if (
            self._current["level"] in ("h2", "h3")
            and not self._current["heading"]
            and not self._text_parts
        ):
            self._current["heading"] = text
            return
        self._text_parts.append(text)


def _parse_page(path: Path) -> tuple[str, list[dict[str, str]]]:
    parser = _PageParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    parser._flush_section()
    title = _TITLE_SUFFIX_RE.sub("", _collapse(" ".join(parser.title_parts)))
    sections = []
    for section in parser.sections:
        if not section["text"] and not section["heading"]:
            continue
        sections.append(
            {
                "anchor": section["anchor"],
                "heading": section["heading"],
                "text": section["text"],
            }
        )
    return title, sections


def _load_metadata(site_dir: Path) -> dict:
    metadata_path = site_dir / "page_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _is_excluded(name: str, page_meta: dict) -> bool:
    if name in EXCLUDED_PAGES:
        return True
    meta = page_meta.get(name, {})
    if "noindex" in str(meta.get("robots", "")):
        return True
    if meta.get("sitemap") is False:
        return True
    return False


def build_index(site_dir: Path) -> dict:
    metadata = _load_metadata(site_dir)
    page_meta = metadata.get("pages", {})
    defaults = metadata.get("defaults", {})

    pages = []
    for html_path in sorted(site_dir.glob("*.html")):
        name = html_path.name
        if any(part in EXCLUDED_DIRS for part in html_path.parts):
            continue
        if _is_excluded(name, page_meta):
            continue
        title, sections = _parse_page(html_path)
        meta = page_meta.get(name, {})
        description = meta.get("description", defaults.get("description", ""))
        if not title:
            title = meta.get("title", name)
        pages.append(
            {
                "url": name,
                "title": title,
                "description": description,
                "sections": sections,
            }
        )

    return {
        "format": "cke-docs-search-index/v1",
        "pages": pages,
    }


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=script_dir.parent,
        help="Directory containing the assembled site (default: docs/site).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <site-dir>/search-index.json).",
    )
    args = parser.parse_args(argv)

    site_dir = args.site_dir.resolve()
    output = args.output or (site_dir / "search-index.json")

    index = build_index(site_dir)
    payload = json.dumps(index, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    output.write_text(payload + "\n", encoding="utf-8")

    size_kb = output.stat().st_size / 1024
    print(
        f"  Search index: {len(index['pages'])} pages indexed, "
        f"{output.name} ({size_kb:.1f} KB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
