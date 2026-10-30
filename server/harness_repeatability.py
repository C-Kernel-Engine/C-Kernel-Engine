#!/usr/bin/env python3
"""Run existing local serving acceptance suites repeatedly with retained evidence.

The configuration names exact deployed artifacts and client executables. This
coordinator does not select a model template, execute harness tools, or retry a
possibly interrupted task. Every invocation gets a new disposable directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


SCHEMA = "cke.harness_repeatability.v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[a-z][a-z0-9_-]{0,47}\Z")
ADAPTERS = {"qwen_code", "antigravity_gemma"}
QWEN_CODE_VERSIONS = frozenset({"0.24.6", "0.25.1"})
QWEN_CODE_PROFILES = frozenset({"bare", "normal"})
HTTP_CASES = {
    "plain-chat": {"chat-completions", "responses-stream", "multi-turn",
                   "output-limit", "undeclared-tools"},
    "budget-boundary": {"context-boundary"},
    "lifecycle": {"long-input-timed-cancel", "decode", "stream-disconnect",
                  "client-read-timeout"},
}


def _validate(config: dict) -> dict:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"configuration must use {SCHEMA}")
    attempts = config.get("attempts")
    if type(attempts) is not int or not 1 <= attempts <= 20:
        raise ValueError("attempts must be an integer from 1 through 20")
    clients = config.get("clients")
    if not isinstance(clients, list) or not clients:
        raise ValueError("clients must be a nonempty list")
    names: set[str] = set()
    endpoints: set[tuple[str, int]] = set()
    for client in clients:
        if not isinstance(client, dict):
            raise ValueError("each client must be an object")
        name = client.get("name")
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names:
            raise ValueError("client names must be distinct safe directory names")
        names.add(name)
        if client.get("adapter") not in ADAPTERS:
            raise ValueError(f"{name}: unsupported client adapter")
        endpoint = client.get("endpoint")
        if not isinstance(endpoint, str):
            raise ValueError(f"{name}: endpoint is required")
        parsed = urlparse(endpoint)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
                or parsed.path != "/v1" or parsed.query or parsed.fragment
                or parsed.username or parsed.password):
            raise ValueError(f"{name}: endpoint must be an explicit localhost HTTP /v1 URL")
        endpoint_key = ("127.0.0.1", parsed.port or 80)
        if endpoint_key in endpoints:
            raise ValueError(f"{name}: duplicate endpoint would bypass timeout isolation")
        endpoints.add(endpoint_key)
        if not isinstance(client.get("model"), str) or not client["model"]:
            raise ValueError(f"{name}: model is required")
        for field in ("serving_identity", "session_library_sha256"):
            if not isinstance(client.get(field), str) or not SHA256.fullmatch(client[field]):
                raise ValueError(f"{name}: {field} must be a SHA-256 digest")
        if client["adapter"] == "qwen_code":
            if not isinstance(client.get("qwen_bin"), str) or not client["qwen_bin"]:
                raise ValueError(f"{name}: qwen_bin is required")
            version = client.get("qwen_version")
            if not isinstance(version, str) or version not in QWEN_CODE_VERSIONS:
                raise ValueError(f"{name}: unsupported pinned Qwen Code version")
            profile = client.get("qwen_profile", "bare")
            if not isinstance(profile, str) or profile not in QWEN_CODE_PROFILES:
                raise ValueError(f"{name}: qwen_profile must be bare or normal")
            settings = client.get("settings")
            if settings is not None and (not isinstance(settings, str) or not settings):
                raise ValueError(f"{name}: settings must be a nonempty path")
            deadline = client.get("task_timeout_seconds", 1800)
            if type(deadline) is not int or not 1 <= deadline <= 86400:
                raise ValueError(f"{name}: invalid task timeout")
            output_tokens = client.get("max_output_tokens", 768)
            if type(output_tokens) is not int or not 1 <= output_tokens <= 32768:
                raise ValueError(f"{name}: invalid max_output_tokens")
        else:
            if not isinstance(client.get("python"), str) or not client["python"]:
                raise ValueError(f"{name}: Python with pinned Antigravity SDK is required")
            deadline = client.get("case_timeout_seconds", 300)
            if type(deadline) not in (int, float) or not 1 <= deadline <= 3600:
                raise ValueError(f"{name}: invalid case timeout")
        prefill = client.get("prefill_lines", 100)
        if type(prefill) is not int or not 1 <= prefill <= 10000:
            raise ValueError(f"{name}: invalid prefill_lines")
    return config


def _identity_status(client: dict, report: dict) -> str:
    if report.get("schema") == "cke.http_lifecycle_acceptance.v1":
        before, after = report.get("identity_before"), report.get("identity_after")
    elif client["adapter"] == "qwen_code":
        observed = report.get("identity") or {}
        if not isinstance(observed, dict):
            return "missing"
        before, after = observed.get("before"), observed.get("after")
    else:
        before, after = report.get("identity_before"), report.get("identity_after")
    if not isinstance(before, dict):
        return "missing"
    instance = before.get("server_instance_id")
    if not isinstance(instance, str) or not instance:
        return "missing"
    for identity in (before, after):
        if not isinstance(identity, dict):
            return "incomplete"
        if (identity.get("serving_identity") != client["serving_identity"]
                or identity.get("session_library_sha256") != client["session_library_sha256"]
                or identity.get("model") != client["model"]):
            return "fail"
    return "pass" if before == after else "fail"


def _server_instance(client: dict, report: dict | None) -> str | None:
    if not isinstance(report, dict):
        return None
    if report.get("schema") == "cke.http_lifecycle_acceptance.v1":
        before = report.get("identity_before")
    elif client["adapter"] == "qwen_code":
        observed = report.get("identity") or {}
        before = observed.get("before") if isinstance(observed, dict) else None
    else:
        before = report.get("identity_before")
    return before.get("server_instance_id") if isinstance(before, dict) else None


def _report_matches_step(client: dict, kind: str, report: object) -> bool:
    if not isinstance(report, dict) or report.get("model") != client["model"]:
        return False
    if report.get("endpoint") != client["endpoint"]:
        return False
    if kind == "task":
        expected = ("cke.serving_harness_acceptance.v1" if client["adapter"] == "qwen_code"
                    else "cke.gemma_antigravity_acceptance.v1")
        return report.get("schema") == expected
    if (report.get("schema") != "cke.http_lifecycle_acceptance.v1"
            or report.get("suite") != kind):
        return False
    cases = report.get("cases")
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        return False
    labels = [case.get("label") for case in cases]
    expected_labels = set(HTTP_CASES[kind])
    identity = report.get("identity_before")
    if (kind == "plain-chat" and isinstance(identity, dict)
            and identity.get("output_protocol") == "none"):
        expected_labels.add("chat-after-rejection")
    return len(labels) == len(expected_labels) and set(labels) == expected_labels


def _command(client: dict, kind: str, directory: Path) -> tuple[list[str], Path, int]:
    scripts = Path(__file__).resolve().parent
    base = ["--endpoint", client["endpoint"], "--model", client["model"],
            "--expected-serving-identity", client["serving_identity"],
            "--expected-session-library-sha256", client["session_library_sha256"]]
    if kind in {"plain-chat", "lifecycle", "budget-boundary"}:
        report = directory / "report.json"
        command = [sys.executable, str(scripts / "http_lifecycle_acceptance.py"),
                   *base, "--suite", kind, "--prefill-lines",
                   str(client.get("prefill_lines", 100)), "--output", str(report)]
        return command, report, 900
    if client["adapter"] == "qwen_code":
        report = directory / "workspace" / "report.json"
        command = [sys.executable, str(scripts / "qwen_harness_acceptance.py"),
                   *base, "--run-dir", str(directory / "workspace"),
                   "--qwen-bin", client["qwen_bin"],
                   "--expected-qwen-version", client["qwen_version"],
                   "--qwen-profile", client.get("qwen_profile", "bare"),
                   "--timeout", str(client.get("task_timeout_seconds", 1800)),
                   "--max-output-tokens", str(client.get("max_output_tokens", 768))]
        if client.get("settings"):
            command += ["--settings", client["settings"]]
        return command, report, client.get("task_timeout_seconds", 1800) + 90
    report = directory / "report.json"
    command = [client["python"], str(scripts / "gemma_antigravity_acceptance.py"),
               *base, "--workspace", str(directory / "workspace"),
               "--output", str(report), "--case-timeout-seconds",
               str(client.get("case_timeout_seconds", 300))]
    return command, report, int(4 * client.get("case_timeout_seconds", 300) + 90)


def _run_step(client: dict, kind: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    command, report_path, deadline = _command(client, kind, directory)
    started = time.time()
    timed_out = False
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
        except OSError as exc:
            stderr.write(f"runner launch failed: {exc}\n".encode())
            exit_code = 127
        else:
            try:
                exit_code = process.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                exit_code = 124
    try:
        report_bytes = report_path.read_bytes() if report_path.is_file() else None
        report = json.loads(report_bytes) if report_bytes is not None else None
    except (OSError, json.JSONDecodeError):
        report_bytes = None
        report = None
    report_valid = _report_matches_step(client, kind, report)
    if not report_valid:
        report = None
    row = {"client": client["name"], "kind": kind, "report": str(report_path),
           "report_sha256": hashlib.sha256(report_bytes).hexdigest() if report_bytes is not None else None,
           "report_integrity_status": "pass" if report_bytes is not None else "missing",
           "started_at_unix": started, "elapsed_seconds": round(time.time() - started, 3),
           "exit_code": exit_code, "timed_out": timed_out,
           "report_schema_status": "pass" if report_valid else "missing_or_invalid",
           "loaded_identity_status": _identity_status(client, report) if report else "missing",
           "server_instance_id": _server_instance(client, report)}
    if kind == "task" and client["adapter"] == "qwen_code":
        row["task_status"] = ("incomplete" if timed_out else
                              (report or {}).get("result", {}).get("status", "missing"))
        if timed_out:
            row["incomplete_reason"] = "coordinator_time_budget_exceeded"
        elif row["task_status"] == "incomplete":
            row["incomplete_reason"] = (report or {}).get("result", {}).get(
                "incomplete_reason", "client_incomplete")
        row["client_certification_status"] = (report or {}).get("certification", {}).get("status", "missing")
        row["tool_call_count"] = len((report or {}).get("result", {}).get("tool_call_ids", []))
    elif kind == "task":
        cases = (report or {}).get("cases", [])
        row["task_status"] = (report or {}).get("status", "missing")
        row["cases"] = [{"case": case.get("case"), "status": case.get("status")}
                        for case in cases if isinstance(case, dict)]
        row["tool_call_count"] = sum(len(case.get("tool_calls", [])) for case in cases
                                     if isinstance(case, dict))
    else:
        row["http_suite_status"] = (report or {}).get("status", "missing")
        row["cases"] = (report or {}).get("cases", [])
    row["status"] = ("pass" if exit_code == 0 and not timed_out and report is not None
                     and row["loaded_identity_status"] == "pass"
                     and (row.get("task_status", row.get("http_suite_status")) == "pass")
                     and row.get("client_certification_status", "pass") == "pass"
                     else "fail")
    return row


def _save(summary: dict, root: Path) -> None:
    temporary = root / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(root / "summary.json")


def _complete(config: dict, rows: list[dict]) -> bool:
    expected = Counter()
    for client in config["clients"]:
        for kind in ("plain-chat", "budget-boundary", "lifecycle"):
            expected[(client["name"], kind)] = 1
        expected[(client["name"], "task")] = config["attempts"]
    actual = Counter((row.get("client"), row.get("kind")) for row in rows)
    return actual == expected and all(row.get("status") == "pass" for row in rows)


def run(config: dict, root: Path) -> dict:
    config = _validate(config)
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = {"schema": SCHEMA, "started_at_unix": time.time(),
               "source_commit": _source_commit(), "steps": [], "status": "incomplete"}
    _save(summary, root)
    instances: dict[str, str] = {}
    for client in config["clients"]:
        for kind in ("plain-chat", "budget-boundary", "lifecycle",
                     *(["task"] * config["attempts"])):
            count = sum(row["client"] == client["name"] and row["kind"] == kind
                        for row in summary["steps"]) + 1
            directory = root / client["name"] / f"{kind}-{count:02d}"
            row = _run_step(client, kind, directory)
            instance = row.get("server_instance_id")
            previous = instances.get(client["name"])
            row["server_instance_continuity_status"] = (
                "pass" if isinstance(instance, str) and instance
                and (previous is None or previous == instance) else "fail")
            if row["server_instance_continuity_status"] != "pass":
                row["status"] = "fail"
            if previous is None and isinstance(instance, str) and instance:
                instances[client["name"]] = instance
            summary["steps"].append(row)
            _save(summary, root)
            if row["timed_out"]:
                # Native completion after a client timeout is not proven. Do not
                # automatically submit another request to this endpoint.
                break
    summary["finished_at_unix"] = time.time()
    summary["status"] = "pass" if _complete(config, summary["steps"]) else "fail"
    _save(summary, root)
    return summary


def reassess(root: Path) -> dict:
    """Re-evaluate retained reports without repeating any client/tool action."""
    config = _validate(json.loads((root / "config.json").read_text()))
    original_bytes = (root / "summary.json").read_bytes()
    original = json.loads(original_bytes)
    if original.get("schema") != SCHEMA or original.get("status") == "incomplete":
        raise ValueError("matrix must have a completed summary before reassessment")
    clients = {client["name"]: client for client in config["clients"]}
    rows = []
    instances: dict[str, str] = {}
    for prior in original.get("steps", []):
        if not isinstance(prior, dict) or prior.get("client") not in clients:
            raise ValueError("matrix contains an unknown client step")
        client = clients[prior["client"]]
        report_path = Path(prior.get("report", "")).resolve()
        if not report_path.is_relative_to(root.resolve()):
            raise ValueError("matrix report path escapes its output directory")
        try:
            report_bytes = report_path.read_bytes()
            report = json.loads(report_bytes)
        except (OSError, json.JSONDecodeError):
            report_bytes = None
            report = None
        kind = prior.get("kind")
        if kind not in (*HTTP_CASES, "task"):
            raise ValueError("matrix contains an unknown gate")
        valid = _report_matches_step(client, kind, report)
        row = dict(prior)
        expected_digest = prior.get("report_sha256")
        if expected_digest is None:
            row["report_integrity_status"] = "unverified"
        elif (report_bytes is not None and isinstance(expected_digest, str)
              and hashlib.sha256(report_bytes).hexdigest() == expected_digest):
            row["report_integrity_status"] = "pass"
        else:
            row["report_integrity_status"] = "fail"
        row["report_schema_status"] = "pass" if valid else "missing_or_invalid"
        row["loaded_identity_status"] = (_identity_status(client, report) if valid else "missing")
        row["server_instance_id"] = _server_instance(client, report if valid else None)
        instance = row["server_instance_id"]
        previous = instances.get(client["name"])
        row["server_instance_continuity_status"] = (
            "pass" if isinstance(instance, str) and instance
            and (previous is None or previous == instance) else "fail")
        if previous is None and isinstance(instance, str) and instance:
            instances[client["name"]] = instance
        if kind == "task" and client["adapter"] == "qwen_code":
            row["task_status"] = (report or {}).get("result", {}).get("status", "missing")
            row["client_certification_status"] = (report or {}).get("certification", {}).get("status", "missing")
        elif kind == "task":
            row["task_status"] = (report or {}).get("status", "missing")
        else:
            row["http_suite_status"] = (report or {}).get("status", "missing")
        row["status"] = ("pass" if row.get("exit_code") == 0 and not row.get("timed_out")
                         and valid and row["report_integrity_status"] == "pass"
                         and row["loaded_identity_status"] == "pass"
                         and row["server_instance_continuity_status"] == "pass"
                         and row.get("task_status", row.get("http_suite_status")) == "pass"
                         and row.get("client_certification_status", "pass") == "pass"
                         else "fail")
        rows.append(row)
    result = {"schema": SCHEMA, "source_summary_sha256": hashlib.sha256(original_bytes).hexdigest(),
              "reassessed_at_unix": time.time(), "reassessment_source_commit": _source_commit(),
              "steps": rows,
              "status": "pass" if _complete(config, rows) else "fail"}
    with (root / "reassessment.json").open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    return result


def _source_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                              capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reassess-dir", type=Path,
                        help="Re-evaluate completed retained reports without invoking clients")
    args = parser.parse_args()
    try:
        if args.reassess_dir is not None:
            if args.config is not None or args.output_dir is not None:
                parser.error("reassessment uses only --reassess-dir")
            path = args.reassess_dir.resolve()
            summary = reassess(path)
            result_path = path / "reassessment.json"
        else:
            if args.config is None or args.output_dir is None:
                parser.error("new runs require --config and --output-dir")
            config = json.loads(args.config.read_text())
            summary = run(config, args.output_dir.resolve())
            result_path = args.output_dir.resolve() / "summary.json"
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"summary": str(result_path),
                      "status": summary["status"], "steps": len(summary["steps"])}))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
