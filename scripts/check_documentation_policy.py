#!/usr/bin/env python3
"""Report legacy Markdown debt and reject new hidden documentation."""
import argparse
import json
import subprocess
from pathlib import Path

# Fixed pre-policy inventory, not a moving main branch or count-only allowance.
BASELINE = "c1702bb561751a0dd74faa3b35ef46f65a8c02d9"
EXCEPTIONS = {
    "README.md", "LICENSING.md", "CONTRIBUTORS.md", "LICENSE.md",
    "SECURITY.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "CHANGELOG.md",
    ".github/PULL_REQUEST_TEMPLATE.md", ".github/pull_request_template.md",
}


def markdown(path):
    return Path(path).suffix.lower() in {".md", ".markdown", ".mdx"}


def audit(current, baseline, changed=()):
    legacy = sorted(p for p in current if markdown(p) and p not in EXCEPTIONS)
    added = sorted(p for p in legacy if p not in baseline)
    edited = sorted(p for p in changed if p in current and p in legacy and p in baseline)
    html = sorted(p for p in changed if p in current and p.startswith("docs/site/") and p.endswith(".html"))
    failures = [f"New Markdown documentation: {p}; update the HTML site instead." for p in added]
    if edited and not html:
        failures += [f"Legacy Markdown edited without an HTML site update: {p}" for p in edited]
    return {"schema": "cke.documentation_policy.v1", "status": "fail" if failures else "pass",
            "legacy_debt_count": len(legacy), "legacy_debt": legacy,
            "new_markdown": added, "edited_legacy": edited,
            "html_updates": html, "failures": failures,
            "note": "Existing debt is not certified documentation. HTML relevance and readability require review."}


def git_paths(root, *args):
    return set(subprocess.check_output(["git", "-C", str(root), *args]).decode().split("\0")) - {""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="PR base SHA for changed-document checks")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        current = git_paths(root, "ls-files", "-z")
        baseline = git_paths(root, "ls-tree", "-r", "--name-only", "-z", BASELINE)
        changed = git_paths(root, "diff", "--name-only", "-z", args.base, "HEAD") if args.base else set()
        report = audit(current, baseline, changed)
        report["baseline_commit"] = BASELINE
    except (subprocess.CalledProcessError, OSError) as exc:
        report = {"schema": "cke.documentation_policy.v1", "status": "error", "failures": [str(exc)]}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
