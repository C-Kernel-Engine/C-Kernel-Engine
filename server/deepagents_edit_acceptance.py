#!/usr/bin/env python3
"""Artifact-bound Nemotron Deep Agents edit/test acceptance on disposable files.

This is an optional live harness test, not a numerical or general coding certificate.
The Deep Agents dependencies are imported only for a live run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


SCHEMA = "cke.deepagents_edit_acceptance.v1"
VALIDATOR_OUTPUT = "PASS: add(2, 3) == 5\n"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def loaded_identity(endpoint: str) -> dict:
    with urlopen(endpoint.rstrip("/") + "/cke/loaded-identity", timeout=15) as response:
        value = json.load(response)
    if value.get("schema") != "cke.loaded_serving_identity.v1":
        raise ValueError("loaded identity schema mismatch")
    return value


def validate_fixture(root: Path) -> dict:
    """Run the trusted assertion and report only after its whole script returns."""
    program = (
        "import contextlib, io, json, runpy, sys\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "sys.path.insert(0, str(root))\n"
        "capture = io.StringIO()\n"
        "try:\n"
        "    with contextlib.redirect_stdout(capture):\n"
        "        runpy.run_path(str(root / 'test_module.py'), run_name='__main__')\n"
        "except BaseException as exc:\n"
        "    print(json.dumps({'assertion_completed': False, 'error_type': type(exc).__name__}))\n"
        "    sys.exit(1)\n"
        "print(json.dumps({'assertion_completed': True, 'output': capture.getvalue()}))\n"
    )
    process = subprocess.run([sys.executable, "-I", "-c", program, str(root)],
                             cwd=root, capture_output=True, text=True, timeout=15)
    try:
        result = json.loads(process.stdout)
        if not isinstance(result, dict):
            raise ValueError("validator returned a non-object")
    except (ValueError, TypeError):
        result = {"assertion_completed": False, "error_type": "invalid_validator_report"}
    return {"exit_code": process.returncode,
            "assertion_completed": result.get("assertion_completed") is True,
            "output": result.get("output", ""),
            "error_type": result.get("error_type"),
            "stderr": process.stderr}


def assess(report: dict, expected_value: str, *, expect_read_recovery: bool = False) -> list[str]:
    """Check objective work and chronological, correlated tool events."""
    errors: list[str] = []
    before, after = report.get("identity_before"), report.get("identity_after")
    if not isinstance(before, dict) or not before.get("server_instance_id") or before != after:
        errors.append("loaded_identity_changed_or_missing")
    calls = report.get("calls")
    messages = report.get("messages")
    if not isinstance(calls, list) or not isinstance(messages, list):
        return errors + ["missing_trace"]
    names = [item.get("name") for item in calls if isinstance(item, dict)]
    if expect_read_recovery:
        failures = [(index, item) for index, item in enumerate(calls)
                    if isinstance(item, dict) and item.get("name") == "read_fixture"
                    and item.get("outcome") == "injected_failure"]
        if (len(failures) != 1 or failures[0][0] != 0
                or not any(isinstance(item, dict)
                           and item.get("name") == "read_fixture"
                           and item.get("arguments") == failures[0][1].get("arguments")
                           and isinstance(item.get("outcome"), str)
                           and item["outcome"] not in {"injected_failure", "rejected"}
                           for item in calls[1:])):
            errors.append("read_failure_not_recovered")
    if not {"read_fixture", "edit_fixture", "run_validator"} <= set(names):
        errors.append("required_tool_missing")
    if ("read_fixture" in names and "edit_fixture" in names
            and names.index("read_fixture") > names.index("edit_fixture")):
        errors.append("edit_before_read")
    if "edit_fixture" in names and "run_validator" in names and names.index("run_validator") < names.index("edit_fixture"):
        errors.append("validator_before_edit")
    emitted_calls = [call for message in messages
                     if isinstance(message, dict) for call in (message.get("tool_calls") or [])
                     if isinstance(call, dict)]
    if expect_read_recovery:
        failed_id = emitted_calls[0].get("id") if emitted_calls else None
        if not isinstance(failed_id, str) or not any(
                isinstance(message, dict)
                and message.get("type") == "ToolMessage"
                and message.get("tool_call_id") == failed_id
                and "ERROR: temporary read failure" in str(message.get("content", ""))
                for message in messages):
            errors.append("read_failure_result_missing")
    if [call.get("name") for call in emitted_calls] != names:
        errors.append("tool_execution_mismatch")
    for call, execution in zip(emitted_calls, calls):
        if (not isinstance(execution, dict)
                or not isinstance(call.get("args"), dict)
                or not isinstance(execution.get("arguments"), dict)
                or call.get("args") != execution.get("arguments")):
            errors.append("tool_arguments_mismatch")
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            errors.append("malformed_message")
            break
        if message.get("type") == "AIMessage":
            if pending:
                errors.append("unresolved_tool_call_before_next_turn")
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict) or not isinstance(tool_call.get("id"), str) or not tool_call["id"]:
                    errors.append("malformed_tool_call")
                    continue
                call_id = tool_call["id"]
                if call_id in seen:
                    errors.append("reused_tool_call_id")
                seen.add(call_id)
                pending.add(call_id)
        elif message.get("tool_calls"):
            errors.append("tool_call_outside_assistant_message")
        if message.get("type") == "ToolMessage":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                errors.append("orphan_or_duplicate_tool_result")
            else:
                pending.remove(call_id)
        elif message.get("tool_call_id"):
            errors.append("tool_result_outside_tool_message")
    if pending:
        errors.append("unresolved_tool_call")
    if messages:
        final = messages[-1]
        if (not isinstance(final, dict) or final.get("type") != "AIMessage"
                or final.get("tool_calls") or expected_value not in str(final.get("content", ""))):
            errors.append("final_answer_missing_value")
        elif ("pass" not in str(final.get("content", "")).lower()
              or final.get("finish_reason") not in ("stop", "eos")):
            errors.append("final_answer_not_terminal_or_missing_test_result")
    else:
        errors.append("missing_final_answer")
    validation = report.get("validation") or {}
    if ((validation.get("exit_code") != 0
            or validation.get("assertion_completed") is not True
            or validation.get("stdout") != VALIDATOR_OUTPUT) or not any(
            isinstance(item, dict) and item.get("name") == "run_validator"
            and isinstance(item.get("outcome"), dict)
            and item["outcome"].get("exit_code") == 0
            and item["outcome"].get("assertion_completed") is True
            and item["outcome"].get("output") == VALIDATOR_OUTPUT for item in calls)):
        errors.append("validator_not_passed_by_agent")
    if report.get("validator_sha256_after") != report.get("validator_sha256_before"):
        errors.append("validator_modified")
    return sorted(set(errors))


def run(args: argparse.Namespace) -> dict:
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    value = "cke-" + secrets.token_hex(6)
    (root / "note.txt").write_text(value + "\n")
    (root / "module.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_module.py").write_text(
        "from module import add\nassert add(2, 3) == 5\n"
        "print('PASS: add(2, 3) == 5')\n")
    validator_hash = digest((root / "test_module.py").read_bytes())
    report = {
        "schema": SCHEMA, "status": "INCOMPLETE", "endpoint": args.endpoint,
        "model": args.model, "expected_serving_identity": args.expected_serving_identity,
        "expected_session_library_sha256": args.expected_session_library_sha256,
        "fixture_sha256": {name: digest((root / name).read_bytes())
                           for name in ("note.txt", "module.py", "test_module.py")},
        "validator_sha256_before": validator_hash, "calls": [], "messages": [], "errors": [],
        "inject_first_read_error": args.inject_first_read_error,
    }
    report_path = root / "report.json"

    def save() -> None:
        temporary = root / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temporary.replace(report_path)

    save()
    try:
        from deepagents import create_deep_agent
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        before = loaded_identity(args.endpoint)
        report["identity_before"] = before
        if (before.get("model") != args.model
                or before.get("serving_identity") != args.expected_serving_identity
                or before.get("session_library_sha256") != args.expected_session_library_sha256
                or not before.get("server_instance_id")):
            raise ValueError("loaded artifact identity mismatch")

        def record(name: str, arguments: dict, outcome: object) -> None:
            report["calls"].append({"name": name, "arguments": arguments, "outcome": outcome})
            save()

        @tool
        def read_fixture(path: str) -> str:
            """Read note.txt or module.py from the disposable coding workspace."""
            if path not in ("note.txt", "module.py"):
                record("read_fixture", {"path": path}, "rejected")
                return "ERROR: path is not allowed"
            if args.inject_first_read_error and not report["calls"]:
                record("read_fixture", {"path": path}, "injected_failure")
                return "ERROR: temporary read failure; retry the same path."
            content = (root / path).read_text()
            record("read_fixture", {"path": path}, digest(content.encode()))
            return content

        @tool
        def edit_fixture(path: str, old: str, new: str) -> str:
            """Replace one exact piece of module.py in the disposable coding workspace."""
            if path != "module.py" or not old or old == new:
                record("edit_fixture", {"path": path, "old": old, "new": new}, "rejected")
                return "ERROR: invalid edit"
            original = (root / path).read_text()
            if original.count(old) != 1:
                record("edit_fixture", {"path": path, "old": old, "new": new}, "rejected")
                return "ERROR: old text must occur exactly once"
            (root / path).write_text(original.replace(old, new, 1))
            record("edit_fixture", {"path": path, "old": old, "new": new},
                   digest((root / path).read_bytes()))
            return "Edited module.py"

        @tool
        def run_validator() -> str:
            """Run the deterministic Python test in the disposable coding workspace."""
            outcome = validate_fixture(root)
            record("run_validator", {}, outcome)
            return json.dumps(outcome)

        prompt = ("/no_think\nRead note.txt and module.py using read_fixture. "
                  "If a tool reports a temporary read failure, retry that read before editing. "
                  "The add function is wrong. Use edit_fixture to make add(2, 3) return 5. "
                  "Call run_validator, then report the exact note value and test result. "
                  "Use only the three supplied restricted tools. Do not claim an edit or test without executing it.")
        report["prompt_sha256"] = digest(prompt.encode())
        model = ChatOpenAI(model=args.model, base_url=args.endpoint, api_key="cke-local",
                           temperature=0, max_tokens=args.output_tokens,
                           timeout=args.request_timeout)
        agent = create_deep_agent(
            model=model, tools=[read_fixture, edit_fixture, run_validator], subagents=[],
            system_prompt="/no_think\nUse the restricted supplied tools for the disposable task. "
                          "Do not delegate or use unrelated filesystem tools.")
        started = time.monotonic()
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                              config={"recursion_limit": 14})
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["identity_after"] = loaded_identity(args.endpoint)
        report["messages"] = [{
            "type": type(message).__name__, "content": str(getattr(message, "content", ""))[:5000],
            "tool_calls": getattr(message, "tool_calls", None) or [],
            "tool_call_id": getattr(message, "tool_call_id", None),
            "finish_reason": (getattr(message, "response_metadata", {}) or {}).get("finish_reason"),
            "usage_metadata": getattr(message, "usage_metadata", None),
        } for message in result.get("messages") or []]
        valid = validate_fixture(root)
        report["validation"] = {"exit_code": valid["exit_code"],
                                "assertion_completed": valid["assertion_completed"],
                                "stdout": valid["output"], "stderr": valid["stderr"],
                                "module_sha256": digest((root / "module.py").read_bytes())}
        report["validator_sha256_after"] = digest((root / "test_module.py").read_bytes())
        report["errors"] = assess(report, value,
                                  expect_read_recovery=args.inject_first_read_error)
        report["status"] = "PASS" if not report["errors"] else "FAIL"
    except Exception as exc:
        report["status"] = "FAIL"
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-serving-identity", required=True)
    parser.add_argument("--expected-session-library-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--inject-first-read-error", action="store_true",
                        help="return one deterministic transient read error and require recovery")
    args = parser.parse_args()
    if args.output_tokens < 1 or args.request_timeout < 1:
        parser.error("positive token and timeout limits are required")
    report = run(args)
    print(json.dumps({"status": report["status"], "errors": report["errors"],
                      "report": str(args.run_dir / "report.json")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
