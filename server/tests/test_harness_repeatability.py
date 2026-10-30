"""The repeatability coordinator retains failures without inventing certification."""
import hashlib
import json
from pathlib import Path

import pytest

from server import harness_repeatability as matrix


HASH = "a" * 64


def _config():
    return {"schema": matrix.SCHEMA, "attempts": 2, "clients": [{
        "name": "qwen-control", "adapter": "qwen_code",
        "endpoint": "http://127.0.0.1:18103/v1", "model": "qwen-test",
        "serving_identity": HASH, "session_library_sha256": HASH,
        "qwen_bin": "qwen", "qwen_version": "0.24.6",
    }]}


def test_config_rejects_remote_endpoints_and_unpinned_artifacts():
    config = _config()
    config["clients"][0]["endpoint"] = "https://cloud.example/v1"
    with pytest.raises(ValueError, match="localhost"):
        matrix._validate(config)
    config = _config()
    config["clients"][0]["serving_identity"] = "unverified"
    with pytest.raises(ValueError, match="serving_identity"):
        matrix._validate(config)


def test_config_rejects_two_clients_on_same_endpoint():
    config = _config()
    second = dict(config["clients"][0], name="same-port",
                  endpoint="http://localhost:18103/v1")
    config["clients"].append(second)
    with pytest.raises(ValueError, match="duplicate endpoint"):
        matrix._validate(config)


def test_qwen_child_workspace_is_separate_from_retained_logs(tmp_path):
    command, report, _ = matrix._command(_config()["clients"][0], "task", tmp_path)
    assert command[command.index("--run-dir") + 1] == str(tmp_path / "workspace")
    assert report == tmp_path / "workspace" / "report.json"


def test_qwen_code_versions_and_profiles_are_explicit(tmp_path):
    legacy = _config()["clients"][0]
    assert matrix._validate(_config())
    legacy_command, _, _ = matrix._command(legacy, "task", tmp_path)
    assert legacy_command[legacy_command.index("--qwen-profile") + 1] == "bare"
    assert legacy_command[legacy_command.index("--expected-qwen-version") + 1] == "0.24.6"

    current = dict(legacy, qwen_version="0.25.1", qwen_profile="normal",
                   task_timeout_seconds=14400, max_output_tokens=2048)
    config = _config()
    config["clients"] = [current]
    assert matrix._validate(config)
    command, _, deadline = matrix._command(current, "task", tmp_path)
    assert command[command.index("--qwen-profile") + 1] == "normal"
    assert command[command.index("--expected-qwen-version") + 1] == "0.25.1"
    assert command[command.index("--max-output-tokens") + 1] == "2048"
    assert deadline == 14490
    current["qwen_version"] = "0.25.2"
    with pytest.raises(ValueError, match="unsupported pinned"):
        matrix._validate(config)
    current["qwen_version"] = {"unhashable": True}
    with pytest.raises(ValueError, match="unsupported pinned"):
        matrix._validate(config)
    current["qwen_version"] = "0.25.1"
    current["qwen_profile"] = {"unhashable": True}
    with pytest.raises(ValueError, match="qwen_profile"):
        matrix._validate(config)
    current["qwen_profile"] = "normal"
    current["settings"] = {"path": "settings.json"}
    with pytest.raises(ValueError, match="settings"):
        matrix._validate(config)


def test_matrix_preserves_every_attempt_and_separates_statuses(tmp_path, monkeypatch):
    calls = []

    def fake_step(client, kind, directory: Path):
        directory.mkdir(parents=True)
        calls.append((kind, directory.name))
        return {"client": client["name"], "kind": kind, "timed_out": False,
                "status": "fail" if directory.name == "task-01" else "pass",
                "task_status": "fail" if directory.name == "task-01" else "pass",
                "loaded_identity_status": "pass"}

    monkeypatch.setattr(matrix, "_run_step", fake_step)
    result = matrix.run(_config(), tmp_path / "matrix")
    assert result["status"] == "fail"
    assert calls == [("plain-chat", "plain-chat-01"),
                     ("budget-boundary", "budget-boundary-01"),
                     ("lifecycle", "lifecycle-01"),
                     ("task", "task-01"), ("task", "task-02")]
    assert (tmp_path / "matrix" / "summary.json").is_file()


def test_timeout_stops_reusing_that_endpoint(tmp_path, monkeypatch):
    calls = []

    def fake_step(client, kind, directory: Path):
        calls.append(kind)
        return {"client": client["name"], "kind": kind,
                "timed_out": kind == "task", "status": "fail" if kind == "task" else "pass"}

    monkeypatch.setattr(matrix, "_run_step", fake_step)
    result = matrix.run(_config(), tmp_path / "matrix")
    assert calls == ["plain-chat", "budget-boundary", "lifecycle", "task"]
    assert result["status"] == "fail"


def test_identity_is_independent_of_task_success():
    client = _config()["clients"][0]
    identity = {"model": "qwen-test", "serving_identity": HASH,
                "session_library_sha256": HASH, "server_instance_id": "first"}
    report = {"identity": {"before": identity, "after": identity},
              "result": {"status": "fail"}}
    assert matrix._identity_status(client, report) == "pass"
    report["identity"]["after"] = {**identity, "serving_identity": "b" * 64}
    assert matrix._identity_status(client, report) == "fail"


def test_http_report_uses_its_own_identity_fields():
    client = _config()["clients"][0]
    identity = {"model": "qwen-test", "serving_identity": HASH,
                "session_library_sha256": HASH, "server_instance_id": "first"}
    report = {"schema": "cke.http_lifecycle_acceptance.v1",
              "identity_before": identity, "identity_after": identity,
              "status": "pass"}
    assert matrix._identity_status(client, report) == "pass"


def test_http_report_must_match_requested_suite_endpoint_and_cases():
    client = _config()["clients"][0]
    report = {"schema": "cke.http_lifecycle_acceptance.v1", "model": "qwen-test",
              "endpoint": client["endpoint"], "suite": "lifecycle",
              "identity_before": {"output_protocol": "qwen_xml"},
              "cases": [{"label": label} for label in matrix.HTTP_CASES["lifecycle"]]}
    assert matrix._report_matches_step(client, "lifecycle", report)
    assert not matrix._report_matches_step(client, "plain-chat", report)
    report["endpoint"] = "http://127.0.0.1:18104/v1"
    assert not matrix._report_matches_step(client, "lifecycle", report)
    report["endpoint"] = client["endpoint"]
    report["cases"][-1]["label"] = "unexpected-case"
    assert not matrix._report_matches_step(client, "lifecycle", report)


def test_successful_child_exit_cannot_certify_wrong_report_schema(tmp_path, monkeypatch):
    import sys

    def fake_command(client, kind, directory):
        report = directory / "report.json"
        source = ("import json,sys; open(sys.argv[1], 'w').write(json.dumps("
                  "{'schema':'wrong','model':'qwen-test','status':'pass'}))")
        return [sys.executable, "-c", source, str(report)], report, 10

    monkeypatch.setattr(matrix, "_command", fake_command)
    row = matrix._run_step(_config()["clients"][0], "task", tmp_path / "step")
    assert row["exit_code"] == 0
    assert row["status"] == "fail"
    assert row["report_schema_status"] == "missing_or_invalid"


def test_coordinator_time_budget_retains_incomplete_task(tmp_path, monkeypatch):
    import subprocess

    class SlowChild:
        pid = 12345
        waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("qwen", timeout)
            return 0

    monkeypatch.setattr(matrix, "_command", lambda _client, _kind, directory: (
        ["qwen"], directory / "report.json", 1))
    monkeypatch.setattr(matrix.subprocess, "Popen", lambda *_args, **_kwargs: SlowChild())
    monkeypatch.setattr(matrix.os, "killpg", lambda *_args: None)
    row = matrix._run_step(_config()["clients"][0], "task", tmp_path / "step")
    assert row["status"] == "fail"
    assert row["timed_out"] is True
    assert row["task_status"] == "incomplete"
    assert row["incomplete_reason"] == "coordinator_time_budget_exceeded"
    assert row["termination_reason"] == "coordinator_time_budget_exceeded"


def test_coordinator_timeout_preserves_child_protocol_failure(tmp_path, monkeypatch):
    import subprocess

    client = _config()["clients"][0]
    identity = {"model": client["model"], "serving_identity": HASH,
                "session_library_sha256": HASH, "server_instance_id": "first"}
    report = {"schema": "cke.serving_harness_acceptance.v1",
              "model": client["model"], "endpoint": client["endpoint"],
              "identity": {"before": identity, "after": identity},
              "result": {"status": "fail", "errors": ["mixed_stream_sessions"],
                         "observed_failures": ["mixed_stream_sessions"]},
              "certification": {"status": "fail"}}

    class SlowChild:
        pid = 12345
        waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("qwen", timeout)
            return 0

    def fake_command(_client, _kind, directory):
        path = directory / "report.json"
        path.write_text(json.dumps(report))
        return ["qwen"], path, 1

    monkeypatch.setattr(matrix, "_command", fake_command)
    monkeypatch.setattr(matrix.subprocess, "Popen", lambda *_args, **_kwargs: SlowChild())
    monkeypatch.setattr(matrix.os, "killpg", lambda *_args: None)
    row = matrix._run_step(client, "task", tmp_path / "failed-step")
    assert row["status"] == "fail"
    assert row["task_status"] == "fail"
    assert row["termination_reason"] == "coordinator_time_budget_exceeded"
    assert row["observed_failures"] == ["mixed_stream_sessions"]
    assert "incomplete_reason" not in row


def test_swapped_http_suite_report_fails_and_retains_its_hash(tmp_path, monkeypatch):
    import sys

    client = _config()["clients"][0]
    identity = {"model": client["model"], "serving_identity": HASH,
                "session_library_sha256": HASH, "server_instance_id": "first"}
    report = {"schema": "cke.http_lifecycle_acceptance.v1", "model": client["model"],
              "endpoint": client["endpoint"], "suite": "plain-chat", "status": "pass",
              "identity_before": identity, "identity_after": identity,
              "cases": [{"label": label} for label in matrix.HTTP_CASES["plain-chat"]]}
    report_text = json.dumps(report)

    def fake_command(_client, _kind, directory):
        path = directory / "report.json"
        return ([sys.executable, "-c",
                 "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])",
                 str(path), report_text], path, 10)

    monkeypatch.setattr(matrix, "_command", fake_command)
    row = matrix._run_step(client, "lifecycle", tmp_path / "step")
    assert row["exit_code"] == 0
    assert row["report_schema_status"] == "missing_or_invalid"
    assert row["report_sha256"] == hashlib.sha256(report_text.encode()).hexdigest()
    assert row["status"] == "fail"


def test_reassessment_uses_retained_reports_without_replaying_tasks(tmp_path):
    root = tmp_path / "matrix"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_config()))
    identity = {"model": "qwen-test", "serving_identity": HASH,
                "session_library_sha256": HASH, "server_instance_id": "first"}
    report_path = root / "http-report.json"
    report_path.write_text(json.dumps({
        "schema": "cke.http_lifecycle_acceptance.v1", "model": "qwen-test",
        "endpoint": "http://127.0.0.1:18103/v1", "suite": "plain-chat",
        "cases": [{"label": label} for label in matrix.HTTP_CASES["plain-chat"]],
        "identity_before": identity, "identity_after": identity, "status": "pass"}))
    original = {"schema": matrix.SCHEMA, "status": "fail", "steps": [{
        "client": "qwen-control", "kind": "plain-chat", "report": str(report_path),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "exit_code": 0, "timed_out": False, "status": "fail"}]}
    (root / "summary.json").write_text(json.dumps(original))
    result = matrix.reassess(root)
    assert result["steps"][0]["status"] == "pass"
    assert result["steps"][0]["loaded_identity_status"] == "pass"
    assert result["steps"][0]["report_integrity_status"] == "pass"
    assert result["status"] == "fail"  # Other required steps are still missing.
    assert json.loads((root / "summary.json").read_text()) == original
    assert (root / "reassessment.json").is_file()


@pytest.mark.parametrize("change", ["mutated", "missing_hash"])
def test_reassessment_cannot_certify_unbound_child_report(tmp_path, change):
    root = tmp_path / "matrix"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_config()))
    identity = {"model": "qwen-test", "serving_identity": HASH,
                "session_library_sha256": HASH, "server_instance_id": "first"}
    report_path = root / "report.json"
    report_path.write_text(json.dumps({
        "schema": "cke.http_lifecycle_acceptance.v1", "model": "qwen-test",
        "endpoint": "http://127.0.0.1:18103/v1", "suite": "budget-boundary",
        "cases": [{"label": "context-boundary"}],
        "identity_before": identity, "identity_after": identity, "status": "pass"}))
    step = {"client": "qwen-control", "kind": "budget-boundary",
            "report": str(report_path), "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "exit_code": 0, "timed_out": False, "status": "fail"}
    if change == "mutated":
        report_path.write_text(report_path.read_text() + " ")
    else:
        del step["report_sha256"]
    (root / "summary.json").write_text(json.dumps({
        "schema": matrix.SCHEMA, "status": "fail", "steps": [step]}))
    result = matrix.reassess(root)
    assert result["steps"][0]["report_integrity_status"] == (
        "fail" if change == "mutated" else "unverified")
    assert result["steps"][0]["status"] == "fail"


def test_matrix_rejects_server_restart_between_passed_steps(tmp_path, monkeypatch):
    def fake_step(client, kind, directory):
        return {"client": client["name"], "kind": kind, "status": "pass",
                "timed_out": False,
                "server_instance_id": "second" if directory.name == "task-02" else "first"}

    monkeypatch.setattr(matrix, "_run_step", fake_step)
    result = matrix.run(_config(), tmp_path / "matrix")
    assert result["status"] == "fail"
    assert result["steps"][-1]["server_instance_continuity_status"] == "fail"


def test_matrix_requires_each_distinct_http_gate():
    config = _config()
    rows = [{"client": "qwen-control", "kind": kind, "status": "pass"}
            for kind in ("plain-chat", "budget-boundary", "lifecycle", "task", "task")]
    assert matrix._complete(config, rows)
    rows[2]["kind"] = "plain-chat"
    assert not matrix._complete(config, rows)
