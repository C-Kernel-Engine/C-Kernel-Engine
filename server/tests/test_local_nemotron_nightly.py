"""The optional Nemotron run must not turn missing execution into a green result."""
import json

from scripts import local_nemotron_nightly as lane


def config(tmp_path):
    value = {
        "schema": lane.SCHEMA, "bundle": str(tmp_path / "bundle"),
        "service_checkout": str(tmp_path / "checkout"),
        "service_python": str(tmp_path / "service-python"),
        "client_python": str(tmp_path / "client-python"),
        "model": "nemotron-test", "serving_identity": "a" * 64,
        "session_library_sha256": "b" * 64, "source_commit": "c" * 40,
        "client_freeze_sha256": "e" * 64,
        "artifact_sha256": {name: "d" * 64 for name in lane.ARTIFACTS},
        "port": 18319, "threads": 16, "context_tokens": 20480,
        "output_tokens": 512, "memory_max_gib": 36,
        "memory_reserve_gib": 8, "task_timeout_seconds": 900,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return path, value


def test_previous_pass_is_replaced_by_current_config_failure(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "latest.json").write_text('{"status":"PASS","run_id":"old"}')
    path = tmp_path / "broken.json"
    path.write_text("{")
    result = lane.run(path, state)
    latest = json.loads((state / "latest.json").read_text())
    assert result["status"] == latest["status"] == "FAIL"
    assert latest["run_id"] != "old"


def test_pause_and_missing_assets_do_not_start_server(tmp_path, monkeypatch):
    path, _ = config(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "PAUSE").touch()
    monkeypatch.setattr(lane, "_start", lambda *_: 1 / 0)
    assert lane.run(path, state)["status"] == "NOT_TESTED"
    (state / "PAUSE").unlink()
    result = lane.run(path, state)
    assert result["status"] == "NOT_TESTED"
    assert "assets unavailable" in result["reason"]


def test_memory_preflight_preserves_running_models(tmp_path, monkeypatch):
    path, _ = config(tmp_path)
    monkeypatch.setattr(lane, "_assets", lambda _: ("READY", ""))
    monkeypatch.setattr(lane, "_available_bytes", lambda: 43 * lane.GIB)
    monkeypatch.setattr(lane, "_start", lambda *_: 1 / 0)
    result = lane.run(path, tmp_path / "state")
    assert result["status"] == "NOT_TESTED"
    assert result["required_memory_bytes"] == 44 * lane.GIB


def test_three_steps_use_one_instance_and_service_stops(tmp_path, monkeypatch):
    path, value = config(tmp_path)
    instance = "isolated-instance"
    identity = {"model": value["model"],
                "serving_identity": value["serving_identity"],
                "session_library_sha256": value["session_library_sha256"],
                "context_capacity": value["context_tokens"],
                "server_instance_id": instance}
    events = []
    monkeypatch.setattr(lane, "_assets", lambda _: ("READY", ""))
    monkeypatch.setattr(lane, "_available_bytes", lambda: 100 * lane.GIB)
    monkeypatch.setattr(lane, "_start", lambda *_: events.append("start"))
    monkeypatch.setattr(lane, "_wait_ready", lambda *_: identity)

    def invoke(_config, _directory, label, recover):
        current = json.loads((tmp_path / "state/latest.json").read_text())
        assert current["status"] == "INCOMPLETE"
        events.append((label, recover))
        return {"label": label, "status": "PASS", "server_instance_id": instance}

    monkeypatch.setattr(lane, "_invoke", invoke)
    monkeypatch.setattr(lane, "_systemctl", lambda *args: type("Result", (), {
        "returncode": 0, "stdout": "MemoryPeak=123\n"})())
    result = lane.run(path, tmp_path / "state")
    assert result["status"] == "PASS"
    assert result["service_stop_status"] == "PASS"
    assert events == ["start", ("fresh", False), ("tool-recovery", True),
                      ("fresh-after-recovery", False)]


def test_failed_recovery_stops_before_followup_and_stops_service(tmp_path, monkeypatch):
    path, value = config(tmp_path)
    identity = {"model": value["model"], "serving_identity": value["serving_identity"],
                "session_library_sha256": value["session_library_sha256"],
                "context_capacity": value["context_tokens"],
                "server_instance_id": "same"}
    steps = []
    stops = []
    monkeypatch.setattr(lane, "_assets", lambda _: ("READY", ""))
    monkeypatch.setattr(lane, "_available_bytes", lambda: 100 * lane.GIB)
    monkeypatch.setattr(lane, "_start", lambda *_: None)
    monkeypatch.setattr(lane, "_wait_ready", lambda *_: identity)

    def invoke(_config, _directory, label, _recover):
        steps.append(label)
        return {"label": label, "status": "FAIL" if label == "tool-recovery" else "PASS",
                "server_instance_id": "same"}

    def ctl(*args):
        if args[0] == "stop":
            stops.append(args[1])
        return type("Result", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(lane, "_invoke", invoke)
    monkeypatch.setattr(lane, "_systemctl", ctl)
    result = lane.run(path, tmp_path / "state")
    assert result["status"] == "FAIL"
    assert steps == ["fresh", "tool-recovery"]
    assert len(stops) == 1
