#!/usr/bin/env python3
"""Fail when cleaned generic DSL functions contain model-specific literals."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = REPO_ROOT / "version" / "v8" / "dsl_policy.json"


class DSLPolicyError(RuntimeError):
    pass


def _function_nodes(
    tree: ast.AST, requested: set[str]
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in requested:
                continue
            if node.name in nodes:
                raise DSLPolicyError(f"ambiguous compiler function name: {node.name}")
            nodes[node.name] = node
    return nodes


def scan_source(source: str, functions: list[str], forbidden: list[str], *, path: str) -> list[dict[str, Any]]:
    tree = ast.parse(source, filename=path)
    available = _function_nodes(tree, set(functions))
    findings: list[dict[str, Any]] = []
    forbidden_lc = tuple(item.lower() for item in forbidden)
    for function in functions:
        node = available.get(function)
        if node is None:
            raise DSLPolicyError(f"policy function not found: {path}:{function}")
        for child in ast.walk(node):
            if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
                continue
            value = child.value.lower()
            matched = sorted({item for item in forbidden_lc if item in value})
            if matched:
                findings.append(
                    {
                        "path": path,
                        "function": function,
                        "line": child.lineno,
                        "literals": matched,
                        "value": child.value,
                    }
                )
    return findings


def _scope_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    stack = list(reversed(function.body))
    while stack:
        node = stack.pop()
        nodes.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return nodes


def _reads_dispatch_key(node: ast.AST, keys: set[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and isinstance(child.slice, ast.Constant):
            if child.slice.value in keys:
                return True
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr != "get" or not child.args:
            continue
        key = child.args[0]
        if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in keys:
            return True
    return False


def scan_model_dispatch_source(
    source: str,
    dispatch_keys: list[str],
    *,
    path: str,
    exclude_functions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Reject family-based branches while permitting exact op/kernel dispatch."""
    tree = ast.parse(source, filename=path)
    excluded = set(exclude_functions or [])
    keys = set(dispatch_keys)
    findings: list[dict[str, Any]] = []
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in excluded
    ]
    for function in functions:
        nodes = _scope_nodes(function)
        tainted: set[str] = set()
        changed = True
        while changed:
            changed = False
            for node in nodes:
                if isinstance(node, ast.Assign):
                    targets = node.targets
                    value = node.value
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    targets = [node.target]
                    value = node.value
                else:
                    continue
                depends = _reads_dispatch_key(value, keys) or any(
                    isinstance(child, ast.Name) and child.id in tainted for child in ast.walk(value)
                )
                if not depends:
                    continue
                for target in targets:
                    for child in ast.walk(target):
                        if isinstance(child, ast.Name) and child.id not in tainted:
                            tainted.add(child.id)
                            changed = True

        for node in nodes:
            if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                test = node.test
            elif isinstance(node, ast.Match):
                test = node.subject
            elif isinstance(node, ast.Subscript):
                test = node.slice
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "get" and node.args):
                test = node.args[0]
            else:
                continue
            depends = _reads_dispatch_key(test, keys) or any(
                isinstance(child, ast.Name) and child.id in tainted for child in ast.walk(test)
            )
            if depends:
                findings.append(
                    {
                        "path": path,
                        "function": function.name,
                        "line": node.lineno,
                        "kind": "model_dispatch",
                        "value": ast.unparse(test),
                    }
                )
    return findings


def count_model_literal_sites(source: str, forbidden: list[str], *, path: str) -> dict[str, Any]:
    """Count existing specialization without treating comments/docstrings as code."""
    tree = ast.parse(source, filename=path)
    forbidden_lc = tuple(item.lower() for item in forbidden)
    lines: set[int] = set()
    by_function: dict[str, set[int]] = {}
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    documentation_nodes = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, str)
            or node in documentation_nodes
        ):
            continue
        if not any(item in node.value.lower() for item in forbidden_lc):
            continue
        owner = "<module>"
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = parent.name
                break
            parent = parents.get(parent)
        by_function.setdefault(owner, set()).add(node.lineno)
        lines.add(node.lineno)
    return {
        "path": path,
        "sites": len(lines),
        "functions": {
            name: len(function_lines)
            for name, function_lines in sorted(by_function.items())
        },
    }


def audit(policy_path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema") != "cke.v8_dsl_policy" or policy.get("schema_version") != 1:
        raise DSLPolicyError("unsupported DSL policy schema")
    forbidden = policy.get("forbidden_model_literals")
    compiler_functions = policy.get("compiler_functions")
    compiler_files = policy.get("compiler_files", {})
    literal_site_limits = policy.get("model_literal_site_limits", {})
    dispatch_keys = policy.get("forbidden_dispatch_keys", [])
    if not isinstance(forbidden, list) or not forbidden or not all(isinstance(item, str) and item for item in forbidden):
        raise DSLPolicyError("forbidden_model_literals must be a non-empty string list")
    if not isinstance(compiler_functions, dict) or not compiler_functions:
        raise DSLPolicyError("compiler_functions must be a non-empty object")

    findings: list[dict[str, Any]] = []
    checked = 0
    for relative_path, functions in compiler_functions.items():
        if not isinstance(relative_path, str) or not isinstance(functions, list) or not functions:
            raise DSLPolicyError("each compiler policy entry requires a path and non-empty function list")
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise DSLPolicyError(f"compiler file not found: {relative_path}")
        findings.extend(
            scan_source(path.read_text(encoding="utf-8"), functions, forbidden, path=relative_path)
        )
        checked += len(functions)
    if not isinstance(compiler_files, dict):
        raise DSLPolicyError("compiler_files must be an object")
    if not isinstance(dispatch_keys, list) or not all(isinstance(item, str) and item for item in dispatch_keys):
        raise DSLPolicyError("forbidden_dispatch_keys must be a string list")
    for relative_path, file_policy in compiler_files.items():
        if not isinstance(file_policy, dict):
            raise DSLPolicyError("each compiler_files entry must be an object")
        exclude = file_policy.get("exclude_functions", [])
        if not isinstance(exclude, list) or not all(isinstance(item, str) and item for item in exclude):
            raise DSLPolicyError("exclude_functions must be a string list")
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise DSLPolicyError(f"compiler file not found: {relative_path}")
        findings.extend(
            scan_model_dispatch_source(
                path.read_text(encoding="utf-8"),
                dispatch_keys,
                path=relative_path,
                exclude_functions=exclude,
            )
        )
    if not isinstance(literal_site_limits, dict):
        raise DSLPolicyError("model_literal_site_limits must be an object")
    inventory: dict[str, Any] = {}
    total_sites = 0
    for relative_path, limit in literal_site_limits.items():
        if not isinstance(relative_path, str) or not isinstance(limit, int) or limit < 0:
            raise DSLPolicyError("model literal site limits require paths and non-negative integers")
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise DSLPolicyError(f"compiler file not found: {relative_path}")
        row = count_model_literal_sites(
            path.read_text(encoding="utf-8"), forbidden, path=relative_path
        )
        inventory[relative_path] = row
        total_sites += int(row["sites"])
        if row["sites"] > limit:
            findings.append(
                {
                    "path": relative_path,
                    "kind": "model_literal_site_limit",
                    "sites": row["sites"],
                    "limit": limit,
                }
            )
    return {
        "status": "fail" if findings else "pass",
        "checked_functions": checked,
        "model_literal_sites": total_sites,
        "model_literal_inventory": inventory,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.policy)
    except (DSLPolicyError, OSError, ValueError, SyntaxError) as exc:
        report = {"status": "error", "error": str(exc)}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
