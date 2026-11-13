"""Opt-in real generated-worker HTTP acceptance, using the existing benchmark pins."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest

from scripts.whisper_service_benchmark import identity, write
from server.whisper_http import _WhisperBackend, create_app


def test_pinned_whisper_http_lifecycle(monkeypatch):
    config_path = os.environ.get("CK_WHISPER_SERVICE_ACCEPTANCE_CONFIG")
    output = os.environ.get("CK_WHISPER_SERVICE_ACCEPTANCE_OUTPUT")
    if not config_path:
        if output:
            write(output, {"status": "SKIP", "reason": "no real-worker configuration supplied"})
        pytest.skip("set CK_WHISPER_SERVICE_ACCEPTANCE_CONFIG for pinned real workers")
    report = {"status": "INCOMPLETE", "attempts": []}

    def publish():
        if output:
            write(output, report)

    publish()
    try:
        path = Path(config_path)
        config = json.loads(path.read_text())
        report["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        fixtures = []
        for case in config["cases"]:
            with Path(case["wav"]).open("rb") as handle:
                audio = handle.read(1048577)
            assert len(audio) <= 1048576
            assert hashlib.sha256(audio).hexdigest() == case["sha256"]
            fixtures.append((audio, case["expected_text"]))
        assert len(fixtures) >= 2, "use distinguishable A and B recordings"
        assert fixtures[0][1] != fixtures[1][1]
        for key, value in config["worker_environment"].items():
            assert key in {"CK_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
            monkeypatch.setenv(key, value)
        budget = config.get("max_output_tokens", 256)
        factory = lambda: _WhisperBackend(Path(config["encoder_run_dir"]),
                                          Path(config["decoder_run_dir"]), 180, None, budget)
        with TestClient(create_app(factory, preload=True, max_output_tokens=budget)) as client:
            def wait(predicate):
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    health = client.get("/health").json()
                    assert not health["startup_error"], health["startup_error"]
                    if predicate(health):
                        return health
                    time.sleep(0.02)
                raise AssertionError("native lifecycle condition timed out")

            def submit(index, request_id):
                return client.post("/audio/transcriptions", content=fixtures[index][0],
                                   headers={"Content-Type": "audio/wav", "X-CKE-Request-ID": request_id})

            def check(response, index, request_id):
                assert response.status_code == 200
                row = response.json()
                assert row["id"] == request_id
                assert row["service_instance_id"] == initial_health["service_instance_id"]
                assert row["text"] == fixtures[index][1]
                assert row["complete"] is True and row["stop"] == "eos"
                assert identity(row["loaded_artifact"]) == config["identity"]
                report["attempts"].append(row)
                publish()
                return row

            initial_health = wait(lambda h: h["ready"])
            assert isinstance(initial_health["service_instance_id"], str) and initial_health["service_instance_id"]
            first = check(submit(0, "artifact-A"), 0, "artifact-A")
            check(submit(1, "artifact-B"), 1, "artifact-B")
            repeated = check(submit(0, "artifact-A-repeat"), 0, "artifact-A-repeat")
            assert repeated["loaded_artifact"] == first["loaded_artifact"]
            assert repeated["worker_generation"] == first["worker_generation"]
            assert client.post("/audio/transcriptions", content=b"invalid",
                               headers={"Content-Type": "audio/wav"}).status_code == 400
            with ThreadPoolExecutor(2) as pool:
                active = pool.submit(submit, 0, "artifact-cancel")
                wait(lambda h: h["active"] == "artifact-cancel" and h["running_native"])
                queued = pool.submit(submit, 1, "artifact-queued")
                wait(lambda h: h["queued"] == 1)
                assert client.post("/audio/transcriptions/artifact-queued/cancel").status_code == 200
                assert queued.result(timeout=10).status_code == 409
                assert client.post("/audio/transcriptions/artifact-cancel/cancel").status_code == 200
                assert active.result(timeout=30).status_code == 409
            recovered = check(submit(0, "artifact-recovery"), 0, "artifact-recovery")
            assert recovered["worker_generation"] > first["worker_generation"]
            assert recovered["service_instance_id"] == first["service_instance_id"]
        report["status"] = "PASS"
    except FileNotFoundError as error:
        report.update(status="SKIP", reason=f"asset unavailable: {error}")
        pytest.skip(report["reason"])
    except BaseException as error:
        report.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        publish()


@pytest.mark.parametrize("failure", ["unconfigured", "missing", "malformed"])
def test_unavailable_attempt_replaces_previous_pass(monkeypatch, tmp_path, failure):
    output = tmp_path / "latest.json"
    write(output, {"status": "PASS", "old": True})
    monkeypatch.setenv("CK_WHISPER_SERVICE_ACCEPTANCE_OUTPUT", str(output))
    config = tmp_path / "config.json"
    if failure == "unconfigured":
        monkeypatch.delenv("CK_WHISPER_SERVICE_ACCEPTANCE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CK_WHISPER_SERVICE_ACCEPTANCE_CONFIG", str(config))
        if failure == "malformed":
            config.write_text("{bad JSON")
    expected = json.JSONDecodeError if failure == "malformed" else pytest.skip.Exception
    with pytest.raises(expected):
        test_pinned_whisper_http_lifecycle(monkeypatch)
    report = json.loads(output.read_text())
    assert report["status"] == ("FAIL" if failure == "malformed" else "SKIP")
    assert "old" not in report
