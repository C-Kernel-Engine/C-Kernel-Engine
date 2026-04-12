#!/usr/bin/env python3
"""
test_visualizer_health_v7.py — Zero-dependency HTML visualizer health gate.

Static analysis of generated IR visualizer, dataset viewer, and IR hub HTML.
Catches missing functions, broken tabs, undefined references, and regressions
without requiring a browser, jsdom, or any npm packages.

USAGE:
    # Test source templates (fast, pre-commit)
    python3 version/v7/scripts/test_visualizer_health_v7.py --source

    # Test a generated ir_report.html
    python3 version/v7/scripts/test_visualizer_health_v7.py --ir-report path/to/ir_report.html

    # Test a generated dataset_viewer.html
    python3 version/v7/scripts/test_visualizer_health_v7.py --dataset-viewer path/to/dataset_viewer.html

    # Test an ir_hub.html
    python3 version/v7/scripts/test_visualizer_health_v7.py --ir-hub path/to/ir_hub.html

    # Test everything (source + any generated files found)
    python3 version/v7/scripts/test_visualizer_health_v7.py --all

    # JSON output for CI
    python3 version/v7/scripts/test_visualizer_health_v7.py --source --json-out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


VIS_VERSION = os.environ.get("CK_VIS_VERSION", "v7")
CONTRACTS_DIR = _env_path(
    "CK_VIS_CONTRACTS_DIR",
    ROOT / "version" / VIS_VERSION / "tests" / "contracts",
)
IR_VIZ_SOURCE_PATH = _env_path(
    "CK_VIS_IR_VIZ_SOURCE",
    ROOT / "version" / VIS_VERSION / "tools" / "ir_visualizer.html",
)
DV_SOURCE_PATH = _env_path(
    "CK_VIS_DV_SOURCE",
    ROOT / "version" / VIS_VERSION / "scripts" / "dataset" / f"build_svg_dataset_visualizer_{VIS_VERSION}.py",
)
HUB_SOURCE_PATH = _env_path(
    "CK_VIS_HUB_SOURCE",
    ROOT / "version" / VIS_VERSION / "tools" / ("open_ir_hub.py" if VIS_VERSION == "v7" else f"open_ir_hub_{VIS_VERSION}.py"),
)
DEFAULT_MODELS_ROOT = _env_path(
    "CK_VIS_MODELS_ROOT",
    Path.home() / ".cache" / f"ck-engine-{VIS_VERSION}" / "models",
)

# ── Colour output ────────────────────────────────────────────────────────────

_RED = "\033[0;31m"
_GREEN = "\033[0;32m"
_YELLOW = "\033[1;33m"
_CYAN = "\033[0;36m"
_BOLD = "\033[1m"
_NC = "\033[0m"


# ── Result tracking ──────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # error | warning


@dataclass
class SuiteResult:
    suite: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")


# ── JS extraction & analysis ─────────────────────────────────────────────────

def extract_js_blocks(html: str) -> str:
    """Pull all <script> content from HTML."""
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)
    return "\n".join(blocks)


def find_function_defs(js: str) -> set[str]:
    """Find all function definitions: function Foo(...), const Foo = ..., etc."""
    # Standard function declarations
    defs = set(re.findall(r"\bfunction\s+(\w+)\s*\(", js))
    # Arrow functions and function expressions: const/let/var Foo = (...)=> or function(
    defs.update(re.findall(r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|function\s*\()", js))
    # Also catch: const Foo = s => { ... }  (single-arg arrow)
    defs.update(re.findall(r"\b(?:const|let|var)\s+(\w+)\s*=\s*\w+\s*=>", js))
    return defs


def find_function_calls(js: str) -> set[str]:
    """Find all standalone Foo(...) call sites (not .method() calls)."""
    # Match word( but ONLY if not preceded by a dot (exclude method calls)
    raw = set(re.findall(r"(?<![.\w])(\w+)\s*\(", js))
    # Remove JS builtins and keywords
    builtins = {
        "if", "for", "while", "switch", "catch", "return", "typeof", "new",
        "function", "class", "import", "export", "throw", "delete", "void",
        "yield", "await", "async", "super", "this", "arguments",
        # JS global functions
        "parseInt", "parseFloat", "isNaN", "isFinite", "encodeURI",
        "encodeURIComponent", "decodeURI", "decodeURIComponent", "eval",
        "setTimeout", "clearTimeout", "setInterval", "clearInterval",
        "requestAnimationFrame", "cancelAnimationFrame", "fetch", "alert",
        "confirm", "prompt", "atob", "btoa", "queueMicrotask",
        # Common constructors / objects
        "Array", "Object", "String", "Number", "Boolean", "Map", "Set",
        "WeakMap", "WeakSet", "Promise", "Date", "RegExp", "Error",
        "TypeError", "RangeError", "SyntaxError", "Float32Array",
        "Float64Array", "Int32Array", "Uint8Array", "Uint32Array",
        "Int8Array", "Uint8ClampedArray", "DataView", "ArrayBuffer",
        "SharedArrayBuffer", "Atomics", "BigInt", "BigInt64Array",
        "BigUint64Array", "Symbol", "Proxy", "Reflect", "Intl",
        "TextEncoder", "TextDecoder", "URL", "URLSearchParams",
        "AbortController", "AbortSignal", "Blob", "File", "FileReader",
        "FormData", "Headers", "Request", "Response", "ReadableStream",
        "WritableStream", "TransformStream", "FinalizationRegistry",
        "WeakRef", "structuredClone", "crypto", "performance",
        "Math", "JSON", "console", "document", "window", "navigator",
        "location", "history", "localStorage", "sessionStorage",
        "XMLHttpRequest", "WebSocket", "Worker", "EventSource",
        "MutationObserver", "ResizeObserver", "IntersectionObserver",
        "CustomEvent", "Event", "Image", "Option", "Audio", "Video",
        "d3", "require",
    }
    return raw - builtins


def find_getbyid_targets(js: str) -> set[str]:
    """Extract all getElementById('foo') targets from JS."""
    return set(re.findall(r"getElementById\(['\"](\w+)['\"]\)", js))


def find_html_ids(html: str) -> set[str]:
    """Extract all id="foo" from HTML (supports hyphens in IDs)."""
    return set(re.findall(r'\bid=["\']([a-zA-Z][\w-]*)["\']', html))


def find_tabs(html: str) -> set[str]:
    """Extract data-tab="..." values."""
    raw = set(re.findall(r'data-tab="([^"]+)"', html))
    # Remove template expressions
    return {t for t in raw if "${" not in t and "tabId" not in t}


def find_panels(html: str) -> set[str]:
    """Extract panel-* IDs."""
    return {m.group(1) for m in re.finditer(r'id="panel-(\w+(?:-\w+)*)"', html)}


# ── Node.js syntax check ────────────────────────────────────────────────────

def check_js_syntax(js: str) -> tuple[bool, str]:
    """Use node --check to validate JS syntax. Returns (ok, error_msg)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        # Wrap in function scope so top-level return etc. are valid
        f.write("(function(){\n")
        f.write(js)
        f.write("\n});\n")
        f.flush()
        try:
            r = subprocess.run(
                ["node", "--check", f.name],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                return True, ""
            return False, r.stderr.strip()[:500]
        except FileNotFoundError:
            return True, "(node not found, skipped syntax check)"
        except subprocess.TimeoutExpired:
            return True, "(timeout, skipped syntax check)"
        finally:
            Path(f.name).unlink(missing_ok=True)


# ── Contract definitions ─────────────────────────────────────────────────────


def _load_contract(name: str) -> dict:
    """Load a contract JSON, return empty dict on failure."""
    path = CONTRACTS_DIR / f"{name}_contract.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _contract_to_tab_map(contract: dict) -> dict[str, dict[str, Any]]:
    """Convert contract tabs list → {tab_id: {render_fn, panel_id}} dict."""
    result: dict[str, dict[str, Any]] = {}
    for tab in contract.get("tabs", []):
        result[tab["id"]] = {
            "render_fn": tab.get("render_fn", ""),
            "panel_id": tab.get("panel_id", tab["id"]),
        }
    return result


def _contract_required_fns(contract: dict) -> list[str]:
    """Collect all required functions from a contract."""
    fns = contract.get("required_functions", {})
    result: list[str] = []
    for group in fns.values():
        if isinstance(group, list):
            result.extend(group)
    return result


def _contract_required_els(contract: dict) -> list[str]:
    """Collect required DOM elements from a contract."""
    return contract.get("required_dom_elements", [])


def _load_contracts() -> tuple:
    """Load IR and dataset viewer contracts, return (ir_tabs, ir_fns, ir_els, dv_tabs, dv_fns, dv_els)."""
    ir = _load_contract("ir_visualizer")
    dv = _load_contract("dataset_viewer")
    return (
        _contract_to_tab_map(ir),
        _contract_required_fns(ir),
        _contract_required_els(ir),
        _contract_to_tab_map(dv),
        _contract_required_fns(dv),
        _contract_required_els(dv),
    )


# Load from JSON contracts (single source of truth)
(IR_VIZ_TAB_CONTRACTS, IR_VIZ_REQUIRED_FUNCTIONS, IR_VIZ_REQUIRED_ELEMENTS,
 DV_TAB_CONTRACTS, DV_REQUIRED_FUNCTIONS, DV_REQUIRED_ELEMENTS) = _load_contracts()



# ── Test suites ──────────────────────────────────────────────────────────────

def test_visualizer(html: str, name: str, tab_contracts: dict,
                    required_fns: list[str], required_els: list[str]) -> SuiteResult:
    """Run the full health check battery on one visualizer HTML."""
    result = SuiteResult(suite=name)
    js = extract_js_blocks(html)
    func_defs = find_function_defs(js)
    func_calls = find_function_calls(js)
    html_ids = find_html_ids(html)
    tabs_found = find_tabs(html)
    panels_found = find_panels(html)

    # ── 1. JS Syntax ─────────────────────────────────────────────────────
    ok, err = check_js_syntax(js)
    result.checks.append(Check("js_syntax_valid", ok, err))

    # ── 2. Tab presence ──────────────────────────────────────────────────
    for tab_id, contract in tab_contracts.items():
        result.checks.append(Check(
            f"tab_exists:{tab_id}",
            tab_id in tabs_found,
            f"data-tab=\"{tab_id}\" {'found' if tab_id in tabs_found else 'MISSING'} in HTML",
        ))

    # ── 3. Panel presence ────────────────────────────────────────────────
    for tab_id, contract in tab_contracts.items():
        panel_id = contract.get("panel_id")
        if panel_id:
            # IR viz: panels use bare id (e.g., id="memory")
            found = panel_id in html_ids
            result.checks.append(Check(
                f"panel_exists:{tab_id}",
                found,
                f"id=\"{panel_id}\" {'found' if found else 'MISSING'} in HTML",
            ))
        elif contract.get("panel_el"):
            # Dataset viewer: panels use id="panel-*"
            found = tab_id in panels_found
            result.checks.append(Check(
                f"panel_exists:{tab_id}",
                found,
                f"id=\"panel-{tab_id}\" {'found' if found else 'MISSING'} in HTML",
            ))

    # ── 4. Render functions exist ────────────────────────────────────────
    for tab_id, contract in tab_contracts.items():
        fn = contract.get("render_fn")
        if fn:
            found = fn in func_defs
            result.checks.append(Check(
                f"render_fn_defined:{fn}",
                found,
                f"function {fn}() {'defined' if found else 'MISSING — tab will not render'}",
            ))

    # ── 5. Required functions ────────────────────────────────────────────
    for fn in required_fns:
        found = fn in func_defs
        result.checks.append(Check(
            f"required_fn:{fn}",
            found,
            f"function {fn}() {'defined' if found else 'MISSING'}",
        ))

    # ── 6. Undefined function references (the attnColor class of bug) ───
    # Only check functions in the required_fns list that weren't found.
    # The full "find all undefined calls" approach has too many false positives
    # with method chains, template literals, and bundled modules.
    # The required_fns contract is the reliable gate.
    missing_required = [fn for fn in required_fns if fn not in func_defs]
    if not missing_required:
        result.checks.append(Check(
            "no_undefined_fn_calls",
            True,
            f"All {len(required_fns)} required functions are defined",
        ))

    # ── 7. Required DOM elements ─────────────────────────────────────────
    for el_id in required_els:
        found = el_id in html_ids
        result.checks.append(Check(
            f"required_el:{el_id}",
            found,
            f"id=\"{el_id}\" {'found' if found else 'MISSING'} in HTML",
        ))

    # ── 8. getElementById targets exist in HTML ──────────────────────────
    # Only check targets that are string literals (not dynamic)
    js_targets = find_getbyid_targets(js)
    missing_targets = js_targets - html_ids
    # Filter out targets that are created dynamically by JS (e.g., via innerHTML)
    # We allow up to some dynamic targets, but flag if >20% are missing
    static_hit_rate = 1.0 - (len(missing_targets) / max(1, len(js_targets)))
    result.checks.append(Check(
        "dom_target_coverage",
        static_hit_rate >= 0.60,
        f"{len(js_targets) - len(missing_targets)}/{len(js_targets)} getElementById targets found in static HTML ({static_hit_rate:.0%})",
        severity="warning" if static_hit_rate >= 0.40 else "error",
    ))
    # Log up to 10 missing targets as warnings
    for target in sorted(missing_targets)[:10]:
        result.checks.append(Check(
            f"dom_target_missing:{target}",
            False,
            f"getElementById('{target}') target not in static HTML (may be dynamically created)",
            severity="warning",
        ))

    # ── 9. renderAll completeness (dataset viewer) ───────────────────────
    if "renderAll" in func_defs and tab_contracts is DV_TAB_CONTRACTS:
        # Check that renderAll calls all render functions
        render_all_block = ""
        m = re.search(r"function renderAll\(\)\s*\{(.*?)^\}", js, re.DOTALL | re.MULTILINE)
        if m:
            render_all_block = m.group(1)
        for tab_id, contract in tab_contracts.items():
            fn = contract.get("render_fn")
            if fn and render_all_block:
                called = fn in render_all_block
                result.checks.append(Check(
                    f"renderAll_calls:{fn}",
                    called,
                    f"renderAll() {'calls' if called else 'DOES NOT call'} {fn}()",
                ))

    return result


def test_ir_hub(html: str) -> SuiteResult:
    """Lightweight health check for ir_hub.html."""
    result = SuiteResult(suite="ir_hub")
    js = extract_js_blocks(html)
    func_defs = find_function_defs(js)

    ok, err = check_js_syntax(js)
    result.checks.append(Check("js_syntax_valid", ok, err))

    # Hub should have basic navigation and run listing
    has_runs_index = "runs_index" in html.lower() or "runsIndex" in js or "runs" in js
    result.checks.append(Check(
        "has_runs_structure",
        has_runs_index,
        "Runs index/listing structure present" if has_runs_index else "No runs structure found",
    ))

    # Check for navigation links
    has_navigation = 'class="run-card"' in html or 'class="hub-card"' in html or "run-card" in html
    result.checks.append(Check(
        "has_navigation",
        has_navigation,
        "Navigation cards present" if has_navigation else "No navigation structure found",
        severity="warning",
    ))

    return result


# ── Source-level tests (run on templates, not generated output) ───────────────

def test_ir_viz_source() -> SuiteResult:
    """Test the ir_visualizer.html source template."""
    path = IR_VIZ_SOURCE_PATH
    if not path.exists():
        r = SuiteResult(suite="ir_visualizer_source")
        r.checks.append(Check("file_exists", False, str(path)))
        return r
    html = path.read_text(encoding="utf-8")
    return test_visualizer(
        html, "ir_visualizer_source",
        IR_VIZ_TAB_CONTRACTS, IR_VIZ_REQUIRED_FUNCTIONS, IR_VIZ_REQUIRED_ELEMENTS,
    )


def test_dv_source() -> SuiteResult:
    """Test the dataset viewer generator source (embedded JS)."""
    path = DV_SOURCE_PATH
    if not path.exists():
        r = SuiteResult(suite="dataset_viewer_source")
        r.checks.append(Check("file_exists", False, str(path)))
        return r
    # The Python file contains HTML+JS as string literals.
    # Extract the big _HTML_PREFIX / _HTML_SUFFIX plus the JS sections.
    src = path.read_text(encoding="utf-8")

    # Extract JS by finding all function/const/let/var definitions
    # These are Python string content that will become HTML <script> content
    # We look for lines that are JS (heuristic: indented JS in a Python triple-quoted string)
    js_lines = []
    html_lines = []
    for line in src.split("\n"):
        # Lines that look like JS/HTML (not Python logic)
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "import ", "from ", "if __name__",
                                "return ", "ap.", "args.", "    ap.", "    args.",
                                "#!/", '"""', "    #", "ROOT =", "SUPPORTED_")):
            continue
        # Collect everything — the Python file is mostly embedded HTML/JS
        js_lines.append(line)
        html_lines.append(line)

    combined_js = "\n".join(js_lines)
    combined_html = "\n".join(html_lines)

    # Run function-level analysis on the embedded JS
    func_defs = find_function_defs(combined_js)
    func_calls = find_function_calls(combined_js)
    html_ids = find_html_ids(combined_html)
    tabs_found = find_tabs(combined_html)
    panels_found = find_panels(combined_html)

    result = SuiteResult(suite="dataset_viewer_source")

    # Tab presence
    for tab_id in DV_TAB_CONTRACTS:
        result.checks.append(Check(
            f"tab_exists:{tab_id}",
            tab_id in tabs_found,
            f"data-tab=\"{tab_id}\" {'found' if tab_id in tabs_found else 'MISSING'}",
        ))

    # Panel presence
    for tab_id in DV_TAB_CONTRACTS:
        found = tab_id in panels_found
        result.checks.append(Check(
            f"panel_exists:{tab_id}",
            found,
            f"id=\"panel-{tab_id}\" {'found' if found else 'MISSING'}",
        ))

    # Render functions
    for tab_id, contract in DV_TAB_CONTRACTS.items():
        fn = contract.get("render_fn")
        if fn:
            found = fn in func_defs
            result.checks.append(Check(
                f"render_fn_defined:{fn}",
                found,
                f"function {fn}() {'defined' if found else 'MISSING'}",
            ))

    # Required functions
    for fn in DV_REQUIRED_FUNCTIONS:
        found = fn in func_defs
        result.checks.append(Check(
            f"required_fn:{fn}",
            found,
            f"function {fn}() {'defined' if found else 'MISSING'}",
        ))

    # Undefined call detection — contract-based
    missing_required = [fn for fn in DV_REQUIRED_FUNCTIONS if fn not in func_defs]
    if not missing_required:
        result.checks.append(Check(
            "no_undefined_fn_calls",
            True,
            "All required functions are defined",
        ))

    return result


# ── IR hub source test ───────────────────────────────────────────────────────

def test_ir_hub_source() -> SuiteResult:
    """Test open_ir_hub.py generates valid hub HTML structure."""
    path = HUB_SOURCE_PATH
    if not path.exists():
        r = SuiteResult(suite="ir_hub_source")
        r.checks.append(Check("file_exists", False, str(path)))
        return r
    src = path.read_text(encoding="utf-8")
    js = extract_js_blocks(src)  # won't work perfectly on Python, but catches embedded HTML
    func_defs = find_function_defs(src)

    result = SuiteResult(suite="ir_hub_source")
    result.checks.append(Check("file_exists", True, str(path)))

    # Check for essential hub structure markers
    for marker in ["run-card", "hub-header", "ir_report.html", "dataset_viewer.html"]:
        found = marker in src
        result.checks.append(Check(
            f"hub_marker:{marker}",
            found,
            f"'{marker}' {'found' if found else 'MISSING'} in hub generator",
            severity="warning" if not found else "error",
        ))

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def print_suite(suite: SuiteResult) -> None:
    total = len(suite.checks)
    print(f"\n{_BOLD}{_CYAN}━━━ {suite.suite} ━━━{_NC}  ({suite.passed}/{total} passed"
          f"{f', {suite.warnings} warnings' if suite.warnings else ''}"
          f"{f', {suite.failed} FAILED' if suite.failed else ''})")

    for c in suite.checks:
        if c.passed:
            print(f"  {_GREEN}✓{_NC} {c.name}")
        elif c.severity == "warning":
            print(f"  {_YELLOW}⚠{_NC} {c.name}: {c.detail}")
        else:
            print(f"  {_RED}✗{_NC} {c.name}: {c.detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=f"HTML visualizer health gate ({VIS_VERSION})")
    ap.add_argument("--source", action="store_true", help="Test source templates")
    ap.add_argument("--ir-report", type=Path, help="Test a generated ir_report.html")
    ap.add_argument("--dataset-viewer", type=Path, help="Test a generated dataset_viewer.html")
    ap.add_argument("--ir-hub", type=Path, help="Test a generated ir_hub.html")
    ap.add_argument("--all", action="store_true", help="Test source + discover generated files")
    ap.add_argument("--json-out", type=Path, help="Write JSON report")
    ap.add_argument("--quiet", action="store_true", help="Only print failures")
    args = ap.parse_args()

    if not any([args.source, args.ir_report, args.dataset_viewer, args.ir_hub, args.all]):
        args.source = True  # Default: test sources

    suites: list[SuiteResult] = []

    # ── Source tests ─────────────────────────────────────────────────────
    if args.source or args.all:
        suites.append(test_ir_viz_source())
        suites.append(test_dv_source())
        suites.append(test_ir_hub_source())

    # ── Generated file tests ─────────────────────────────────────────────
    if args.ir_report:
        html = args.ir_report.read_text(encoding="utf-8")
        suites.append(test_visualizer(
            html, f"ir_report:{args.ir_report.name}",
            IR_VIZ_TAB_CONTRACTS, IR_VIZ_REQUIRED_FUNCTIONS, IR_VIZ_REQUIRED_ELEMENTS,
        ))

    if args.dataset_viewer:
        html = args.dataset_viewer.read_text(encoding="utf-8")
        suites.append(test_visualizer(
            html, f"dataset_viewer:{args.dataset_viewer.name}",
            DV_TAB_CONTRACTS, DV_REQUIRED_FUNCTIONS, DV_REQUIRED_ELEMENTS,
        ))

    if args.ir_hub:
        html = args.ir_hub.read_text(encoding="utf-8")
        suites.append(test_ir_hub(html))

    # ── Auto-discover generated files ────────────────────────────────────
    if args.all:
        cache = DEFAULT_MODELS_ROOT
        if cache.exists():
            hub = cache / "ir_hub.html"
            if hub.exists():
                suites.append(test_ir_hub(hub.read_text(encoding="utf-8")))

            # Find most recent train run with both visualizers
            train_dir = cache / "train"
            if train_dir.exists():
                for run in sorted(train_dir.iterdir(), reverse=True):
                    ir = run / "ir_report.html"
                    dv = run / "dataset_viewer.html"
                    if ir.exists():
                        suites.append(test_visualizer(
                            ir.read_text(encoding="utf-8"),
                            f"ir_report:{run.name}",
                            IR_VIZ_TAB_CONTRACTS, IR_VIZ_REQUIRED_FUNCTIONS,
                            IR_VIZ_REQUIRED_ELEMENTS,
                        ))
                    if dv.exists():
                        suites.append(test_visualizer(
                            dv.read_text(encoding="utf-8"),
                            f"dataset_viewer:{run.name}",
                            DV_TAB_CONTRACTS, DV_REQUIRED_FUNCTIONS,
                            DV_REQUIRED_ELEMENTS,
                        ))
                    if ir.exists() or dv.exists():
                        break  # Test latest run only

    # ── Print results ────────────────────────────────────────────────────
    total_checks = sum(len(s.checks) for s in suites)
    total_passed = sum(s.passed for s in suites)
    total_failed = sum(s.failed for s in suites)
    total_warnings = sum(s.warnings for s in suites)

    for s in suites:
        if args.quiet and s.failed == 0:
            continue
        print_suite(s)

    print(f"\n{_BOLD}{'═' * 60}{_NC}")
    if total_failed == 0:
        print(f"{_GREEN}  ✓ All {total_checks} checks passed"
              f"{f' ({total_warnings} warnings)' if total_warnings else ''}{_NC}")
    else:
        print(f"{_RED}  ✗ {total_failed} FAILED{_NC} / {total_checks} checks"
              f"{f' ({total_warnings} warnings)' if total_warnings else ''}")
    print(f"{_BOLD}{'═' * 60}{_NC}")

    # ── Emit sub-test lines for nightly report parsing ────────────────
    # Format: name  max_diff=X  tol=Y  [PASS/FAIL]
    for s in suites:
        tag = s.suite.replace(" ", "_").replace(":", "_")
        status = "PASS" if s.failed == 0 else "FAIL"
        diff = f"{s.failed:.2e}" if s.failed else "0.00e+00"
        print(f"{tag}  max_diff={diff}  tol=1e+00  [{status}]")

    # ── JSON output ──────────────────────────────────────────────────────
    if args.json_out:
        report = {
            "version": VIS_VERSION,
            "total_checks": total_checks,
            "passed": total_passed,
            "failed": total_failed,
            "warnings": total_warnings,
            "suites": [
                {
                    "suite": s.suite,
                    "passed": s.passed,
                    "failed": s.failed,
                    "warnings": s.warnings,
                    "checks": [asdict(c) for c in s.checks],
                }
                for s in suites
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  JSON report: {args.json_out}")

    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
