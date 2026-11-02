"""A local harness night cannot turn missing execution into PASS."""
import json
import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts import local_harness_nightly as lane


def _config(path):
    path.write_text(json.dumps({
        "schema": "cke.harness_repeatability.v1", "attempts": 1,
        "clients": [{
            "name": "qwen-control", "adapter": "qwen_code",
            "endpoint": "http://127.0.0.1:19099/v1", "model": "qwen-test",
            "serving_identity": "a" * 64, "session_library_sha256": "b" * 64,
            "qwen_bin": "qwen", "qwen_version": "0.24.6",
        }],
    }))


def test_missing_config_is_not_tested(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    report = lane.run(tmp_path / "missing.json", tmp_path / "state")
    assert report["status"] == "NOT_TESTED"
    assert json.loads((tmp_path / "state/latest.json").read_text())["status"] == "NOT_TESTED"


def test_optional_prefix_lane_missing_assets_is_not_tested(tmp_path):
    config = tmp_path / "prefix.json"
    config.write_text(json.dumps({
        "schema": "cke.local_prefix_cert.v1", "bundle": str(tmp_path / "bundle"),
        "session_library": str(tmp_path / "missing.so"),
        "ids": str(tmp_path / "missing.ids.json"),
        "cases": [{"prefix": 128, "suffix": 9}],
        "artifact_sha256": {}, "session_sha256": "b" * 64,
        "ids_sha256": "c" * 64,
    }))
    row = lane._prefix_certification(config, tmp_path, paused=False)
    assert row["status"] == "NOT_TESTED"
    assert row["reason"] == "prefix assets unavailable"


def test_optional_prefix_lane_requires_pinned_pass_report(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("libmodel.so", "libckernel_engine.so", "layout_decode.json",
                 "weights.bump", "model_v8.c", "weights_manifest.map", "config.json"):
        (bundle / name).write_bytes(name.encode())
    session = tmp_path / "session.so"
    ids = tmp_path / "ids.json"
    session.write_bytes(b"session")
    ids.write_text("[1, 2, 3]\n")
    config = tmp_path / "prefix.json"
    config.write_text(json.dumps({
        "schema": "cke.local_prefix_cert.v1", "bundle": str(bundle),
        "session_library": str(session), "ids": str(ids),
        "cases": [{"prefix": 128, "suffix": 9}], "threads": 2,
        "artifact_sha256": {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                            for name in ("libmodel.so", "libckernel_engine.so",
                                         "model_v8.c", "weights.bump",
                                         "weights_manifest.map", "config.json")},
        "session_sha256": hashlib.sha256(session.read_bytes()).hexdigest(),
        "ids_sha256": hashlib.sha256(b"[1,2,3]").hexdigest(),
    }))

    def fake_invoke(command, **_kwargs):
        output = Path(command[command.index("--out") + 1])
        output.write_text(json.dumps({
            "verdict": "PASS",
            "artifact_sha256": config_doc["artifact_sha256"],
            "session_library_sha256": config_doc["session_sha256"],
            "ids_sha256": config_doc["ids_sha256"],
        }))
        return 0

    config_doc = json.loads(config.read_text())
    monkeypatch.setattr(lane, "_invoke", fake_invoke)
    row = lane._prefix_certification(config, tmp_path, paused=False)
    assert row["status"] == "PASS" and row["cases"][0]["verdict"] == "PASS"
    (bundle / "weights.bump").write_bytes(b"replacement")
    row = lane._prefix_certification(config, tmp_path / "replaced-weights", paused=False)
    assert row["status"] == "FAIL" and "pinned weights.bump" in row["reason"]
    (bundle / "weights.bump").write_bytes(b"weights.bump")
    def wrong_report(command, **kwargs):
        fake_invoke(command, **kwargs)
        output = Path(command[command.index("--out") + 1])
        observed = json.loads(output.read_text())
        observed["artifact_sha256"]["weights.bump"] = "0" * 64
        output.write_text(json.dumps(observed))
        return 0

    monkeypatch.setattr(lane, "_invoke", wrong_report)
    row = lane._prefix_certification(config, tmp_path / "wrong-report", paused=False)
    assert row["status"] == "FAIL" and row["reason"] == "prefix parity case failed"
    monkeypatch.setattr(lane, "_invoke", lambda *args, **kwargs: 124)
    row = lane._prefix_certification(config, tmp_path / "timeout", paused=False)
    assert row["status"] == "INCOMPLETE" and "time budget" in row["reason"]
    config_doc["session_sha256"] = "0" * 64
    config.write_text(json.dumps(config_doc))
    row = lane._prefix_certification(config, tmp_path / "other", paused=False)
    assert row["status"] == "FAIL" and "pinned" in row["reason"]


def test_malformed_current_prefix_config_replaces_previous_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    state = tmp_path / "state"
    state.mkdir()
    (state / "latest.json").write_text(json.dumps({"status": "PASS", "run_id": "old"}))
    prefix = tmp_path / "prefix.json"
    prefix.write_text("{")
    report = lane.run(tmp_path / "missing-harness.json", state,
                      prefix_cert_config=prefix)
    assert report["status"] == "FAIL"
    assert report["failure_stage"] == "prefix_certification"
    assert json.loads((state / "latest.json").read_text())["status"] == "FAIL"


def test_incomplete_prefix_lane_cannot_leave_current_run_not_tested(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    monkeypatch.setattr(lane, "_prefix_certification", lambda *_args: {
        "status": "INCOMPLETE", "reason": "time budget expired", "cases": []})
    state = tmp_path / "state"
    report = lane.run(tmp_path / "missing-harness.json", state,
                      prefix_cert_config=tmp_path / "prefix.json")
    assert report["status"] == "INCOMPLETE"
    assert json.loads((state / "latest.json").read_text())["status"] == "INCOMPLETE"


def test_unavailable_endpoint_cannot_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    config = tmp_path / "config.json"
    _config(config)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("NOT_TESTED", "connection refused", None))
    report = lane.run(config, tmp_path / "state")
    assert report["status"] == "NOT_TESTED"
    assert report["clients"][0]["status"] == "NOT_TESTED"


def test_wrong_identity_fails_before_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    config = tmp_path / "config.json"
    _config(config)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("FAIL", "wrong identity", "instance"))
    monkeypatch.setattr(lane, "_invoke", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("tools must not run")))
    report = lane.run(config, tmp_path / "state")
    assert report["status"] == "FAIL"
    assert report["clients"][0]["reason"] == "wrong identity"


def test_invalid_identity_schema_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_identity", lambda _: (_ for _ in ()).throw(
        ValueError("unsupported schema")))
    client = {"endpoint": "http://127.0.0.1:19099/v1"}
    status, reason, instance = lane._preflight(client)
    assert status == "FAIL" and "unsupported schema" in reason and instance is None


def test_child_exit_alone_cannot_pass_without_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    config = tmp_path / "config.json"
    _config(config)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("READY", "", "instance"))
    monkeypatch.setattr(lane, "_invoke", lambda *args, **kwargs: 0)
    report = lane.run(config, tmp_path / "state")
    assert report["status"] == "FAIL"
    assert report["clients"][0]["reason"] == "repeatability or reassessment failed"


def test_restarted_instance_cannot_pass_reassessment(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    config = tmp_path / "config.json"
    _config(config)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("READY", "", "original"))

    def fake_invoke(command, **_kwargs):
        flag = "--output-dir" if "--output-dir" in command else "--reassess-dir"
        target = Path(command[command.index(flag) + 1])
        target.mkdir(parents=True, exist_ok=True)
        if flag == "--output-dir":
            (target / "summary.json").write_text(json.dumps({
                "status": "pass", "source_commit": "c" * 40,
                "steps": [{"server_instance_id": "restarted"}] * 4,
            }))
        else:
            (target / "reassessment.json").write_text(json.dumps({
                "status": "pass", "reassessment_source_commit": "c" * 40,
                "steps": [{"server_instance_id": "restarted"}] * 4,
            }))
        return 0

    monkeypatch.setattr(lane, "_invoke", fake_invoke)
    report = lane.run(config, tmp_path / "state")
    assert report["status"] == "FAIL"
    assert report["clients"][0]["server_instance_id"] == "original"


def test_pause_retains_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    config = tmp_path / "config.json"
    _config(config)
    state = tmp_path / "state"
    state.mkdir()
    (state / "PAUSE").touch()
    report = lane.run(config, state)
    assert report["status"] == "NOT_TESTED"
    assert report["clients"][0]["status"] == "NOT_TESTED"


def test_missing_http_dependency_fails_before_model_work(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    monkeypatch.setattr(lane, "_http_dependencies_ready", lambda: False)
    monkeypatch.setattr(lane, "_preflight", lambda _: (_ for _ in ()).throw(
        AssertionError("the endpoint must not be probed")))
    monkeypatch.setattr(lane, "_invoke", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("the harness must not be started")))
    config = tmp_path / "config.json"
    _config(config)
    report = lane.run(config, tmp_path / "state")
    assert report["status"] == "FAIL"
    assert report["clients"][0]["status"] == "FAIL"
    assert "httpx" in report["reason"]


def test_failed_configuration_replaces_previous_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    state = tmp_path / "state"
    state.mkdir()
    (state / "latest.json").write_text(json.dumps({"status": "PASS", "run_id": "yesterday"}))
    config = tmp_path / "malformed.json"
    config.write_text("{")
    report = lane.run(config, state)
    latest = json.loads((state / "latest.json").read_text())
    assert report["status"] == latest["status"] == "FAIL"
    assert latest["run_id"] != "yesterday"
    assert latest["failure_stage"] == "configuration"
    assert json.loads((state / "runs" / latest["run_id"] / "summary.json").read_text()) == latest


def test_child_launch_error_preserves_partial_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("READY", "", "instance"))
    monkeypatch.setattr(lane, "_invoke", lambda *a, **k: (_ for _ in ()).throw(
        OSError("launcher unavailable")))
    config = tmp_path / "config.json"
    _config(config)
    report = lane.run(config, tmp_path / "state")
    assert report["status"] == "FAIL"
    assert report["failure_stage"] == "launch:qwen-control"
    assert report["clients"][0]["status"] == "FAIL"
    assert "launcher unavailable" in report["clients"][0]["reason"]


def test_malformed_reassessment_replaces_previous_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "c" * 40)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("READY", "", "instance"))
    state = tmp_path / "state"
    state.mkdir()
    (state / "latest.json").write_text(json.dumps({"status": "PASS", "run_id": "yesterday"}))
    config = tmp_path / "config.json"
    _config(config)

    def fake_invoke(command, **_kwargs):
        flag = "--output-dir" if "--output-dir" in command else "--reassess-dir"
        target = Path(command[command.index(flag) + 1])
        target.mkdir(parents=True, exist_ok=True)
        if flag == "--output-dir":
            (target / "summary.json").write_text('{"status":"pass"}')
        else:
            (target / "reassessment.json").write_text("{")
        return 0

    monkeypatch.setattr(lane, "_invoke", fake_invoke)
    report = lane.run(config, state)
    latest = json.loads((state / "latest.json").read_text())
    assert report["status"] == latest["status"] == "FAIL"
    assert latest["run_id"] != "yesterday"
    assert latest["failure_stage"] == "reassessment:qwen-control"
    assert latest["clients"][0]["status"] == "FAIL"


def test_refreshed_source_is_recorded_before_endpoint_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(lane, "_refresh_main", lambda: "b" * 40)
    monkeypatch.setattr(lane, "_preflight", lambda _: ("NOT_TESTED", "offline", None))
    config = tmp_path / "config.json"
    _config(config)
    report = lane.run(config, tmp_path / "state", refresh_main=True)
    assert report["status"] == "NOT_TESTED"
    assert report["source_commit"] == "b" * 40
    assert report["source_commit_before_refresh"] == "a" * 40
    assert report["source_refresh"] == "fetched_origin_main"


def test_refresh_failure_replaces_prior_pass_without_probing(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(lane, "_refresh_main", lambda: (_ for _ in ()).throw(
        ValueError("dirty worktree")))
    monkeypatch.setattr(lane, "_preflight", lambda _: (_ for _ in ()).throw(
        AssertionError("must not probe after refresh failure")))
    state = tmp_path / "state"
    state.mkdir()
    (state / "latest.json").write_text(json.dumps({"status": "PASS", "run_id": "yesterday"}))
    config = tmp_path / "config.json"
    _config(config)
    report = lane.run(config, state, refresh_main=True)
    latest = json.loads((state / "latest.json").read_text())
    assert report["status"] == latest["status"] == "FAIL"
    assert latest["failure_stage"] == "source_refresh"
    assert latest["run_id"] != "yesterday"


def test_pause_skips_source_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(lane, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(lane, "_refresh_main", lambda: (_ for _ in ()).throw(
        AssertionError("paused run must not fetch")))
    state = tmp_path / "state"
    state.mkdir()
    (state / "PAUSE").touch()
    config = tmp_path / "config.json"
    _config(config)
    report = lane.run(config, state, refresh_main=True)
    assert report["status"] == "NOT_TESTED"
    assert report["source_refresh"] == "skipped_paused"


def test_refresh_requires_clean_detached_checkout_and_fetches_main(tmp_path, monkeypatch):
    source = tmp_path / "source"
    origin = tmp_path / "origin.git"
    worker = tmp_path / "worker"

    def git(*args, cwd=None):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()

    git("init", "-b", "main", str(source))
    git("config", "user.email", "test@example.invalid", cwd=source)
    git("config", "user.name", "Test", cwd=source)
    (source / "fixture.txt").write_text("old\n")
    git("add", "fixture.txt", cwd=source)
    git("commit", "-m", "old", cwd=source)
    git("clone", "--bare", str(source), str(origin))
    git("clone", str(origin), str(worker))
    monkeypatch.setattr(lane, "ROOT", worker)
    with pytest.raises(ValueError, match="detached"):
        lane._refresh_main()
    git("checkout", "--detach", cwd=worker)
    (worker / "fixture.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="clean"):
        lane._refresh_main()
    git("checkout", "--", "fixture.txt", cwd=worker)
    (source / "fixture.txt").write_text("new\n")
    git("add", "fixture.txt", cwd=source)
    git("commit", "-m", "new", cwd=source)
    git("push", str(origin), "main", cwd=source)
    assert lane._refresh_main() == git("rev-parse", "HEAD", cwd=source)
    assert (worker / "fixture.txt").read_text() == "new\n"
