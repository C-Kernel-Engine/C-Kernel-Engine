#!/usr/bin/env python3
"""
Zero-dependency JS unit-test harness for visualizer pure functions.

Extracts pure JS functions from the IR visualizer and dataset viewer
generator source, runs test vectors through Node.js, and reports pass/fail.

Usage:
    python3 version/v7/scripts/test_visualizer_js_units_v7.py
    python3 version/v7/scripts/test_visualizer_js_units_v7.py --json-out report.json
    python3 version/v7/scripts/test_visualizer_js_units_v7.py --quiet

No npm packages required — just Python 3.8+ and Node.js.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
IR_VIZ = ROOT / "version" / "v7" / "tools" / "ir_visualizer.html"
DS_GEN = ROOT / "version" / "v7" / "scripts" / "dataset" / "build_svg_dataset_visualizer_v7.py"

# ── colours ──────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
TICK = "✓"
CROSS = "✗"

# ── result tracking ──────────────────────────────────────────────────
results: list[dict[str, Any]] = []
fail_count = 0


def record(name: str, passed: bool, detail: str = "") -> None:
    global fail_count
    results.append({"name": name, "passed": passed, "detail": detail})
    if not passed:
        fail_count += 1


# ── JS extraction ────────────────────────────────────────────────────

def extract_js_block(source_path: Path, *, python_embedded: bool = False) -> str:
    """Extract JS from an HTML file or a Python file with embedded JS strings."""
    text = source_path.read_text(encoding="utf-8", errors="replace")
    if python_embedded:
        # Dataset viewer: JS lives inside Python string literals (_HTML_SUFFIX)
        # Grab everything between <script> and </script> in the Python source.
        blocks = re.findall(r"<script[^>]*>(.*?)</script>", text, re.DOTALL)
        return "\n".join(blocks)
    else:
        blocks = re.findall(r"<script[^>]*>(.*?)</script>", text, re.DOTALL)
        return "\n".join(blocks)


def extract_function(js: str, fname: str) -> str | None:
    """Extract a named function body from JS source (handles indentation)."""
    # Try: function fname(...) { ... }
    pattern = rf"(function\s+{re.escape(fname)}\s*\([^)]*\)\s*\{{)"
    m = re.search(pattern, js)
    if m:
        return _extract_braced_block(js, m.start())
    # Try: const fname = (...) => { ... }
    pattern2 = rf"(const\s+{re.escape(fname)}\s*=\s*(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>\s*\{{)"
    m2 = re.search(pattern2, js)
    if m2:
        return _extract_braced_block(js, m2.start())
    return None


def _extract_braced_block(js: str, start: int) -> str:
    """Given a position at the start of a block, extract through matching }.
    
    Handles: strings, template literals with ${...} nesting, regex literals,
    single-line and block comments.
    """
    depth = 0       # tracks only real { } braces at function level
    i = start
    # Context stack: tracks string/template nesting.
    # Values: '"', "'", '`', '${' (template expression)
    ctx_stack: list[str] = []
    while i < len(js):
        ch = js[i]
        ctx = ctx_stack[-1] if ctx_stack else None

        # ── inside single/double quoted string ──
        if ctx in ('"', "'"):
            if ch == '\\':
                i += 2
                continue
            if ch == ctx:
                ctx_stack.pop()
            i += 1
            continue

        # ── inside template literal ──
        if ctx == '`':
            if ch == '\\':
                i += 2
                continue
            if ch == '`':
                ctx_stack.pop()
                i += 1
                continue
            if ch == '$' and i + 1 < len(js) and js[i + 1] == '{':
                ctx_stack.append('${')
                i += 2
                continue
            i += 1
            continue

        # ── inside template expression ${...} ──
        if ctx == '${':
            if ch in ('"', "'", '`'):
                ctx_stack.append(ch)
                i += 1
                continue
            if ch == '\\':
                i += 2
                continue
            if ch == '{':
                # Nested brace inside template expression — push another ${ to track
                ctx_stack.append('${')
                i += 1
                continue
            if ch == '}':
                # Close this template expression level
                ctx_stack.pop()
                i += 1
                continue
            if ch == '/' and i + 1 < len(js):
                nch = js[i + 1]
                if nch == '/':
                    eol = js.find('\n', i)
                    i = eol if eol >= 0 else len(js)
                    continue
                if nch == '*':
                    end = js.find('*/', i + 2)
                    i = end + 2 if end >= 0 else len(js)
                    continue
                # Regex in template expression
                prev_pos = i - 1
                while prev_pos >= start and js[prev_pos] in ' \t\n\r':
                    prev_pos -= 1
                prev_ch = js[prev_pos] if prev_pos >= start else '('
                if prev_ch in '=(!&|,;:?[{+->~%^*/\n':
                    i = _skip_regex(js, i)
                    continue
            i += 1
            continue

        # ── top-level (or plain brace context) ──
        if ch in ('"', "'", '`'):
            ctx_stack.append(ch)
            i += 1
            continue

        if ch == '/' and i + 1 < len(js):
            nch = js[i + 1]
            if nch == '/':
                eol = js.find('\n', i)
                i = eol if eol >= 0 else len(js)
                continue
            if nch == '*':
                end = js.find('*/', i + 2)
                i = end + 2 if end >= 0 else len(js)
                continue
            # Regex literal heuristic
            prev_pos = i - 1
            while prev_pos >= start and js[prev_pos] in ' \t\n\r':
                prev_pos -= 1
            prev_ch = js[prev_pos] if prev_pos >= start else '('
            if prev_ch in '=(!&|,;:?[{+->~%^*/\n':
                i = _skip_regex(js, i)
                continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return js[start:i + 1]

        i += 1
    return js[start:]  # unterminated — return what we have


def _skip_regex(js: str, i: int) -> int:
    """Skip a regex literal starting at position i (the opening /)."""
    j = i + 1
    while j < len(js):
        rc = js[j]
        if rc == '\\':
            j += 2
            continue
        if rc == '[':
            j += 1
            while j < len(js) and js[j] != ']':
                if js[j] == '\\':
                    j += 1
                j += 1
            j += 1
            continue
        if rc == '/':
            j += 1
            while j < len(js) and js[j].isalpha():
                j += 1
            return j
        j += 1
    return j


# ── Node.js runner ───────────────────────────────────────────────────

def run_js(code: str, *, timeout: int = 10) -> tuple[bool, str]:
    """Run JS code via Node.js, return (success, stdout_or_stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(code)
        f.flush()
        tmp = f.name
    try:
        r = subprocess.run(
            ["node", tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0:
            return True, r.stdout.strip()
        return False, (r.stderr or r.stdout).strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError:
        return False, "node not found"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── Test suites ──────────────────────────────────────────────────────

def build_test_harness(functions_js: str, test_cases: list[dict]) -> str:
    """Build a self-contained .mjs that defines functions, runs tests, and prints JSON results."""
    lines = [
        "// Auto-generated test harness",
        "const __results = [];",
        "",
        "// ── Function definitions ──",
        textwrap.dedent(functions_js),
        "",
        "// ── Test runner ──",
        "function assertApprox(a, b, tol = 1e-6) {",
        "  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < tol;",
        "  return false;",
        "}",
        "function assertDeepEq(a, b) {",
        "  return JSON.stringify(a) === JSON.stringify(b);",
        "}",
        "function test(name, fn) {",
        "  try {",
        "    const result = fn();",
        "    if (result === true || result === undefined) {",
        "      __results.push({ name, passed: true });",
        "    } else {",
        "      __results.push({ name, passed: false, detail: 'returned ' + JSON.stringify(result) });",
        "    }",
        "  } catch (e) {",
        "    __results.push({ name, passed: false, detail: String(e) });",
        "  }",
        "}",
        "",
    ]
    for tc in test_cases:
        lines.append(f"test({json.dumps(tc['name'])}, () => {{ {tc['body']} }});")
    lines.append("")
    lines.append("console.log(JSON.stringify(__results));")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# IR VISUALIZER TESTS
# ──────────────────────────────────────────────────────────────────────

def ir_viz_tests(js: str) -> tuple[str, list[dict]]:
    """Extract IR visualizer functions and build test cases."""
    fns_needed = [
        "formatBytes", "normalizeShapeInput", "formatShapeDisplay",
        "normalizeMode", "escapeHtml", "quoteShell",
        "normalizePathString", "pathDirname", "extractGgufStem",
        "relativePathFromTo",
    ]
    extracted = []
    for fn in fns_needed:
        body = extract_function(js, fn)
        if body:
            extracted.append(body)
        else:
            record(f"ir:extract:{fn}", False, f"could not extract {fn} from IR visualizer")
            return "", []

    for fn in fns_needed:
        record(f"ir:extract:{fn}", True)

    functions_js = "\n\n".join(extracted)

    test_cases = [
        # formatBytes
        {"name": "ir:formatBytes:bytes", "body": "return assertDeepEq(formatBytes(512), '512 B');"},
        {"name": "ir:formatBytes:KB", "body": "return assertDeepEq(formatBytes(2048), '2.00 KB');"},
        {"name": "ir:formatBytes:MB", "body": "return assertDeepEq(formatBytes(3 * 1024 * 1024), '3.00 MB');"},
        {"name": "ir:formatBytes:GB", "body": "return assertDeepEq(formatBytes(2.5 * 1024 ** 3), '2.50 GB');"},

        # normalizeShapeInput
        {"name": "ir:normalizeShape:null", "body": "return assertDeepEq(normalizeShapeInput(null), []);"},
        {"name": "ir:normalizeShape:array", "body": "return assertDeepEq(normalizeShapeInput([2,3,4]), ['2','3','4']);"},
        {"name": "ir:normalizeShape:string", "body": "return assertDeepEq(normalizeShapeInput('2x3x4'), ['2','3','4']);"},
        {"name": "ir:normalizeShape:bracketed", "body": "return assertDeepEq(normalizeShapeInput('[2,3,4]'), ['2','3','4']);"},
        {"name": "ir:normalizeShape:number", "body": "return assertDeepEq(normalizeShapeInput(42), ['42']);"},
        {"name": "ir:normalizeShape:empty", "body": "return assertDeepEq(normalizeShapeInput(''), []);"},
        {"name": "ir:normalizeShape:object_shape", "body": "return assertDeepEq(normalizeShapeInput({shape:[1,2]}), ['1','2']);"},

        # formatShapeDisplay
        {"name": "ir:formatShapeDisplay:multi", "body": "return assertDeepEq(formatShapeDisplay([2,3,4]), '2 × 3 × 4');"},
        {"name": "ir:formatShapeDisplay:single", "body": "return assertDeepEq(formatShapeDisplay([5]), '[5]');"},
        {"name": "ir:formatShapeDisplay:empty", "body": "return assertDeepEq(formatShapeDisplay(null), '-');"},

        # normalizeMode
        {"name": "ir:normalizeMode:prefill", "body": "return assertDeepEq(normalizeMode('prefill'), 'prefill');"},
        {"name": "ir:normalizeMode:decode", "body": "return assertDeepEq(normalizeMode('decode'), 'decode');"},
        {"name": "ir:normalizeMode:other", "body": "return assertDeepEq(normalizeMode('xyz'), 'decode');"},

        # escapeHtml
        {"name": "ir:escapeHtml:basic", "body": 'return assertDeepEq(escapeHtml("<b>hi</b>"), "&lt;b&gt;hi&lt;/b&gt;");'},
        {"name": "ir:escapeHtml:quotes", "body": "return assertDeepEq(escapeHtml('a\"b'), 'a&quot;b');"},
        {"name": "ir:escapeHtml:null", "body": "return assertDeepEq(escapeHtml(null), '');"},

        # quoteShell
        {"name": "ir:quoteShell:safe", "body": "return assertDeepEq(quoteShell('hello'), 'hello');"},
        {"name": "ir:quoteShell:spaces", "body": 'return assertDeepEq(quoteShell("hello world"), \'"hello world"\');'},
        {"name": "ir:quoteShell:empty", "body": "return assertDeepEq(quoteShell(''), '');"},

        # normalizePathString
        {"name": "ir:normalizePath:backslash", "body": "return assertDeepEq(normalizePathString('a\\\\b\\\\c'), 'a/b/c');"},
        {"name": "ir:normalizePath:trailing", "body": "return assertDeepEq(normalizePathString('/a/b/'), '/a/b');"},

        # pathDirname
        {"name": "ir:pathDirname:basic", "body": "return assertDeepEq(pathDirname('/a/b/c'), '/a/b');"},
        {"name": "ir:pathDirname:root", "body": "return assertDeepEq(pathDirname('/a'), '/');"},
        {"name": "ir:pathDirname:empty", "body": "return assertDeepEq(pathDirname(''), '');"},

        # extractGgufStem
        {"name": "ir:extractGgufStem:path", "body": "return assertDeepEq(extractGgufStem('/models/qwen.gguf'), 'qwen');"},
        {"name": "ir:extractGgufStem:url", "body": "return assertDeepEq(extractGgufStem('https://x.co/m.gguf?dl=1'), 'm');"},
        {"name": "ir:extractGgufStem:non_gguf", "body": "return assertDeepEq(extractGgufStem('model.bin'), '');"},

        # relativePathFromTo
        {"name": "ir:relPath:sibling", "body": "return assertDeepEq(relativePathFromTo('/a/b', '/a/c'), '../c');"},
        {"name": "ir:relPath:child", "body": "return assertDeepEq(relativePathFromTo('/a/b', '/a/b/c'), 'c');"},
        {"name": "ir:relPath:same", "body": "return assertDeepEq(relativePathFromTo('/a/b', '/a/b'), '.');"},
        {"name": "ir:relPath:nonabs", "body": "return assertDeepEq(relativePathFromTo('a/b', '/c'), null);"},
    ]

    return functions_js, test_cases


# ──────────────────────────────────────────────────────────────────────
# DATASET VIEWER TESTS
# ──────────────────────────────────────────────────────────────────────

def dataset_viewer_tests(js: str) -> tuple[str, list[dict]]:
    """Extract dataset viewer functions and build test cases."""
    fns_needed = [
        "attnColor", "embColor", "embNormalise", "cosineSim",
        "attnEntropy", "avgMatrices",
    ]
    extracted = []
    for fn in fns_needed:
        body = extract_function(js, fn)
        if body:
            extracted.append(body)
        else:
            record(f"ds:extract:{fn}", False, f"could not extract {fn} from dataset viewer")
            return "", []

    for fn in fns_needed:
        record(f"ds:extract:{fn}", True)

    functions_js = "\n\n".join(extracted)

    test_cases = [
        # attnColor — orange channel (default)
        {"name": "ds:attnColor:zero", "body": "return assertDeepEq(attnColor(0, 'orange'), [0, 0, 0]);"},
        {"name": "ds:attnColor:one_orange", "body": "return assertDeepEq(attnColor(1, 'orange'), [255, 180, 0]);"},
        {"name": "ds:attnColor:clamp_neg", "body": "return assertDeepEq(attnColor(-1, 'orange'), [0, 0, 0]);"},
        {"name": "ds:attnColor:clamp_over", "body": "return assertDeepEq(attnColor(2, 'orange'), [255, 180, 0]);"},
        {"name": "ds:attnColor:heatmap_0", "body": "const r = attnColor(0, 'heatmap'); return assertDeepEq(r, [7, 100, 248]);"},
        {"name": "ds:attnColor:heatmap_1", "body": "const r = attnColor(1, 'heatmap'); return assertDeepEq(r, [255, 160, 0]);"},
        {"name": "ds:attnColor:heatmap_mid", "body": "const r = attnColor(0.5, 'heatmap'); return assertDeepEq(r, [240, 240, 240]);"},
        {"name": "ds:attnColor:blue", "body": "return assertDeepEq(attnColor(1, 'blue'), [7, 173, 248]);"},
        {"name": "ds:attnColor:green", "body": "return assertDeepEq(attnColor(1, 'green'), [71, 180, 117]);"},

        # embColor
        {"name": "ds:embColor:zero", "body": "const r = embColor(0); return assertDeepEq(r, [7, 173, 248]);"},
        {"name": "ds:embColor:one", "body": "const r = embColor(1); return assertDeepEq(r, [255, 180, 0]);"},
        {"name": "ds:embColor:half", "body": "const r = embColor(0.5); return assertDeepEq(r, [195, 200, 208]);"},

        # cosineSim
        {"name": "ds:cosineSim:identical", "body": "return assertApprox(cosineSim([1,0,0],[1,0,0]), 1.0);"},
        {"name": "ds:cosineSim:orthogonal", "body": "return assertApprox(cosineSim([1,0],[0,1]), 0.0);"},
        {"name": "ds:cosineSim:opposite", "body": "return assertApprox(cosineSim([1,0],[-1,0]), -1.0);"},
        {"name": "ds:cosineSim:scaled", "body": "return assertApprox(cosineSim([2,0],[4,0]), 1.0);"},

        # attnEntropy
        {"name": "ds:attnEntropy:uniform_2", "body": "return assertApprox(attnEntropy([0.5, 0.5]), 1.0);"},
        {"name": "ds:attnEntropy:uniform_4", "body": "return assertApprox(attnEntropy([0.25, 0.25, 0.25, 0.25]), 2.0);"},
        {"name": "ds:attnEntropy:peaked", "body": "return assertApprox(attnEntropy([1, 0, 0]), 0.0);"},
        {"name": "ds:attnEntropy:with_zeros", "body": "return assertApprox(attnEntropy([0.5, 0.5, 0, 0]), 1.0);"},

        # avgMatrices
        {"name": "ds:avgMatrices:single", "body": """
            const m = [[[1,2],[3,4]]];
            const r = avgMatrices(m);
            return assertDeepEq(r, [[1,2],[3,4]]);
        """},
        {"name": "ds:avgMatrices:two", "body": """
            const m = [[[0,0],[0,0]], [[2,4],[6,8]]];
            const r = avgMatrices(m);
            return assertDeepEq(r, [[1,2],[3,4]]);
        """},
        {"name": "ds:avgMatrices:empty", "body": """
            const r = avgMatrices(null);
            return assertDeepEq(r, [[]]);
        """},

        # embNormalise — global mode
        {"name": "ds:embNormalise:global_range", "body": """
            const {norm, vmin, vmax} = embNormalise([[0, 10], [5, 5]], 'global');
            return assertApprox(vmin, 0) && assertApprox(vmax, 10)
                && assertApprox(norm[0][0], 0) && assertApprox(norm[0][1], 1);
        """},
        {"name": "ds:embNormalise:null_safe", "body": """
            const r = embNormalise(null, 'global');
            return assertDeepEq(r.norm, []) && assertApprox(r.vmin, 0) && assertApprox(r.vmax, 0);
        """},
        {"name": "ds:embNormalise:col_returns", "body": """
            const {norm, note} = embNormalise([[1, 2], [3, 4]], 'col');
            return norm.length === 2 && norm[0].length === 2 && note.includes('col');
        """},
        {"name": "ds:embNormalise:row_returns", "body": """
            const {norm, note} = embNormalise([[1, 2], [3, 4]], 'row');
            return norm.length === 2 && norm[0].length === 2 && note.includes('row');
        """},
    ]

    return functions_js, test_cases


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="JS unit tests for visualizer pure functions")
    ap.add_argument("--json-out", type=str, help="Write JSON report to file")
    ap.add_argument("--quiet", action="store_true", help="Only print failures and summary")
    args = ap.parse_args()

    # Check Node.js available
    try:
        r = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            print(f"{RED}  {CROSS} node not available{RESET}")
            return 1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"{RED}  {CROSS} node not found — install Node.js to run JS unit tests{RESET}")
        return 1

    # ── Extract source JS ──
    if not IR_VIZ.exists():
        print(f"{RED}  {CROSS} IR visualizer not found: {IR_VIZ}{RESET}")
        return 1
    if not DS_GEN.exists():
        print(f"{RED}  {CROSS} Dataset viewer generator not found: {DS_GEN}{RESET}")
        return 1

    ir_js = extract_js_block(IR_VIZ)
    ds_js = extract_js_block(DS_GEN, python_embedded=True)

    # ── Build and run IR visualizer tests ──
    print(f"\n{'━' * 3} IR Visualizer JS Units {'━' * 3}")
    ir_fns, ir_cases = ir_viz_tests(ir_js)
    if ir_fns and ir_cases:
        harness = build_test_harness(ir_fns, ir_cases)
        ok, output = run_js(harness)
        if ok:
            try:
                jr = json.loads(output)
                for t in jr:
                    record(t["name"], t["passed"], t.get("detail", ""))
            except json.JSONDecodeError:
                record("ir:harness_parse", False, f"invalid JSON: {output[:200]}")
        else:
            record("ir:harness_run", False, output[:300])

    # ── Build and run dataset viewer tests ──
    print(f"\n{'━' * 3} Dataset Viewer JS Units {'━' * 3}")
    ds_fns, ds_cases = dataset_viewer_tests(ds_js)
    if ds_fns and ds_cases:
        harness = build_test_harness(ds_fns, ds_cases)
        ok, output = run_js(harness)
        if ok:
            try:
                jr = json.loads(output)
                for t in jr:
                    record(t["name"], t["passed"], t.get("detail", ""))
            except json.JSONDecodeError:
                record("ds:harness_parse", False, f"invalid JSON: {output[:200]}")
        else:
            record("ds:harness_run", False, output[:300])

    # ── Print results ──
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total = len(results)

    for r in results:
        if r["passed"] and not args.quiet:
            print(f"  {GREEN}{TICK}{RESET} {r['name']}")
        elif not r["passed"]:
            detail = f" — {r['detail']}" if r.get("detail") else ""
            print(f"  {RED}{CROSS}{RESET} {r['name']}{detail}")

    # ── Summary ──
    print()
    bar = "═" * 60
    if failed == 0:
        print(f"{bar}\n  {GREEN}{TICK} All {total} JS unit tests passed{RESET}\n{bar}")
    else:
        print(f"{bar}\n  {RED}{CROSS} {failed}/{total} JS unit tests FAILED{RESET}\n{bar}")

    # ── JSON output ──
    if args.json_out:
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "tool": "test_visualizer_js_units_v7",
            "total": total,
            "passed": passed,
            "failed": failed,
            "tests": results,
        }
        outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  JSON report: {outpath}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
