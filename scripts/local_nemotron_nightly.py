#!/usr/bin/env python3
"""Run pinned Nemotron Deep Agents acceptance in a bounded isolated service.

This optional local lane starts its own loopback server, runs fresh/recovery/fresh
tasks on one loaded instance, then stops that exact service. It never retries an
interrupted task or treats a missing artifact as a pass.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


SCHEMA = "cke.local_nemotron_nightly.v1"
ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
ARTIFACTS = ("weights.bump", "libmodel.so", "libckernel_engine.so",
             "libckernel_tokenizer.so")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config(raw: object) -> dict:
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError(f"configuration must use {SCHEMA}")
    for name in ("bundle", "service_checkout", "service_python", "client_python",
                 "model", "serving_identity", "session_library_sha256",
                 "source_commit", "client_freeze_sha256"):
        if not isinstance(raw.get(name), str) or not raw[name]:
            raise ValueError(f"missing {name}")
    for name in ("bundle", "service_checkout", "service_python", "client_python"):
        if not Path(raw[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    hashes = raw.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACTS):
        raise ValueError("artifact_sha256 must pin every loaded bundle library and weight file")
    for name, value in [*hashes.items(),
                        ("session_library", raw["session_library_sha256"]),
                        ("client_freeze", raw["client_freeze_sha256"])]:
        if (not isinstance(value, str) or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"invalid {name} SHA-256")
    if (len(raw["source_commit"]) != 40 or
            any(char not in "0123456789abcdef" for char in raw["source_commit"])):
        raise ValueError("source_commit must be a full commit hash")
    limits = {"port": (1024, 65535), "threads": (1, 256),
              "context_tokens": (1024, 262144), "output_tokens": (64, 32768),
              "memory_max_gib": (8, 512), "memory_reserve_gib": (1, 128),
              "task_timeout_seconds": (60, 86400)}
    for name, (lower, upper) in limits.items():
        value = raw.get(name)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer from {lower} to {upper}")
    return raw


def _assets(config: dict) -> tuple[str, str]:
    checkout = Path(config["service_checkout"])
    bundle = Path(config["bundle"])
    paths = {name: bundle / name for name in ARTIFACTS}
    paths["session_library"] = checkout / "build/libck_session_v8.so"
    paths["service_python"] = Path(config["service_python"])
    paths["client_python"] = Path(config["client_python"])
    paths["server_source"] = checkout / "version/v8/scripts/ck_serve_v8.py"
    paths["client_source"] = ROOT / "server/deepagents_edit_acceptance.py"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return "NOT_TESTED", "assets unavailable: " + ", ".join(missing)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=checkout, text=True).strip()
    if commit != config["source_commit"]:
        return "FAIL", "service checkout does not match pinned source commit"
    for name in ARTIFACTS:
        if _sha256(paths[name]) != config["artifact_sha256"][name]:
            return "FAIL", f"pinned {name} differs"
    if _sha256(paths["session_library"]) != config["session_library_sha256"]:
        return "FAIL", "pinned session library differs"
    freeze = subprocess.check_output([config["client_python"], "-m", "pip",
                                      "freeze", "--all"], timeout=30)
    if hashlib.sha256(freeze).hexdigest() != config["client_freeze_sha256"]:
        return "FAIL", "pinned client Python environment differs"
    return "READY", ""


def _available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable is unavailable")


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True,
                          text=True, check=False, timeout=30)


def _start(config: dict, unit: str) -> None:
    checkout = Path(config["service_checkout"])
    build = checkout / "build"
    command = ["systemd-run", "--user", f"--unit={unit}",
               "--description=Isolated pinned Nemotron acceptance",
               f"--property=MemoryMax={config['memory_max_gib']}G",
               "--property=MemorySwapMax=0", "--property=Nice=10",
               f"--working-directory={checkout}",
               f"--setenv=CK_NUM_THREADS={config['threads']}",
               f"--setenv=LD_LIBRARY_PATH={build}:{config['bundle']}",
               config["service_python"], "version/v8/scripts/ck_serve_v8.py",
               config["bundle"], "--run", config["bundle"], "--no-build",
               "--host", "127.0.0.1", "--port", str(config["port"]),
               "--model-name", config["model"], "--request-output-cap",
               str(config["output_tokens"]), "--max-tokens",
               str(config["output_tokens"]), "--context-len",
               str(config["context_tokens"]), "--no-viz"]
    subprocess.run(command, check=True, timeout=30, capture_output=True, text=True)


def _identity(config: dict) -> dict:
    url = f"http://127.0.0.1:{config['port']}/v1/cke/loaded-identity"
    with urlopen(url, timeout=5) as response:
        value = json.load(response)
    if not isinstance(value, dict) or value.get("schema") != "cke.loaded_serving_identity.v1":
        raise ValueError("loaded identity schema mismatch")
    return value


def _same_identity(config: dict, observed: dict) -> bool:
    return all((observed.get(name) == expected for name, expected in (
        ("model", config["model"]),
        ("serving_identity", config["serving_identity"]),
        ("session_library_sha256", config["session_library_sha256"]),
        ("context_capacity", config["context_tokens"])))) and bool(
            observed.get("server_instance_id"))


def _wait_ready(config: dict, unit: str) -> dict:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            identity = _identity(config)
        except (OSError, URLError, HTTPError, ValueError):
            identity = None
        if identity is not None:
            if not _same_identity(config, identity):
                raise ValueError("loaded model identity differs from pinned configuration")
            return identity
        if _systemctl("is-active", unit).stdout.strip() != "active":
            raise RuntimeError("isolated model service stopped before readiness")
        time.sleep(2)
    raise TimeoutError("isolated model service did not become ready")


def _invoke(config: dict, run_dir: Path, label: str, recover: bool) -> dict:
    directory = run_dir / label
    command = [config["client_python"],
               str(ROOT / "server/deepagents_edit_acceptance.py"),
               "--endpoint", f"http://127.0.0.1:{config['port']}/v1",
               "--model", config["model"], "--expected-serving-identity",
               config["serving_identity"], "--expected-session-library-sha256",
               config["session_library_sha256"], "--run-dir", str(directory),
               "--output-tokens", str(config["output_tokens"]),
               "--request-timeout", str(config["task_timeout_seconds"])]
    if recover:
        command.append("--inject-first-read-error")
    started = time.monotonic()
    with (run_dir / f"{label}.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            exit_code = child.wait(timeout=config["task_timeout_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return {"label": label, "status": "INCOMPLETE",
                    "reason": "task time budget expired; native state unverified"}
    path = directory / "report.json"
    try:
        raw = path.read_bytes()
        report = json.loads(raw)
    except (OSError, ValueError):
        return {"label": label, "status": "FAIL", "reason": "missing or malformed child report",
                "exit_code": exit_code}
    before, after = report.get("identity_before"), report.get("identity_after")
    calls = report.get("calls") or []
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from server.deepagents_edit_acceptance import assess
        expected_value = (directory / "note.txt").read_text().strip()
        reassessment_errors = assess(report, expected_value,
                                     expect_read_recovery=recover)
    except (OSError, ValueError, TypeError) as exc:
        reassessment_errors = [f"reassessment unavailable: {type(exc).__name__}"]
    valid = (exit_code == 0 and report.get("schema") == "cke.deepagents_edit_acceptance.v1"
             and report.get("status") == "PASS" and before == after
             and isinstance(before, dict) and _same_identity(config, before)
             and report.get("inject_first_read_error") is recover
             and isinstance(calls, list) and not reassessment_errors)
    return {"label": label, "status": "PASS" if valid else "FAIL",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "exit_code": exit_code, "report": str(path),
            "report_sha256": hashlib.sha256(raw).hexdigest(),
            "server_instance_id": before.get("server_instance_id") if isinstance(before, dict) else None,
            "tool_sequence": [call.get("name") for call in calls if isinstance(call, dict)],
            "errors": report.get("errors"),
            "reassessment_errors": reassessment_errors}


def run(config_path: Path, state_dir: Path) -> dict:
    started = datetime.now(timezone.utc)
    run_dir = state_dir / "runs" / started.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "started_at": started.isoformat(),
              "run_id": run_dir.name, "status": "INCOMPLETE", "steps": []}

    def publish() -> None:
        _write(run_dir / "summary.json", report)
        _write(state_dir / "latest.json", report)

    publish()  # A failed current attempt cannot leave yesterday's PASS as latest.
    unit = "cke-nemotron-nightly-" + started.strftime("%Y%m%d%H%M%S%f")
    started_service = False
    try:
        report["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if (state_dir / "PAUSE").exists():
            report.update(status="NOT_TESTED", reason="paused before service startup")
            return report
        if not config_path.is_file():
            report.update(status="NOT_TESTED", reason="configuration unavailable")
            return report
        raw = config_path.read_bytes()
        report["config_sha256"] = hashlib.sha256(raw).hexdigest()
        config = _config(json.loads(raw))
        status, reason = _assets(config)
        if status != "READY":
            report.update(status=status, reason=reason)
            return report
        available = _available_bytes()
        required = (config["memory_max_gib"] + config["memory_reserve_gib"]) * GIB
        report.update(available_memory_bytes=available, required_memory_bytes=required)
        if available < required:
            report.update(status="NOT_TESTED", reason="insufficient host memory for isolated service")
            return report
        report["unit"] = unit
        _start(config, unit)
        started_service = True
        identity = _wait_ready(config, unit)
        report["loaded_identity"] = identity
        publish()
        for label, recover in (("fresh", False), ("tool-recovery", True),
                               ("fresh-after-recovery", False)):
            row = _invoke(config, run_dir, label, recover)
            report["steps"].append(row)
            publish()
            if row["status"] != "PASS" or row["server_instance_id"] != identity["server_instance_id"]:
                report.update(status=row["status"] if row["status"] == "INCOMPLETE" else "FAIL",
                              reason=f"{label} did not pass on the loaded instance")
                return report
        report["status"] = "PASS"
        return report
    except Exception as exc:
        report.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}")
        return report
    finally:
        if started_service:
            memory = _systemctl("show", unit, "-p", "MemoryPeak", "-p", "MemoryMax")
            report["memory_properties"] = memory.stdout.strip()
            stopped = _systemctl("stop", unit)
            report["service_stop_status"] = "PASS" if stopped.returncode == 0 else "FAIL"
            if stopped.returncode != 0:
                report.update(status="FAIL", reason="isolated service did not stop cleanly")
            try:
                journal = subprocess.run(["journalctl", "--user", "-u", unit,
                                          "--no-pager"], capture_output=True, timeout=30)
                (run_dir / "service.log").write_bytes(journal.stdout)
            except (OSError, subprocess.SubprocessError):
                pass
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        publish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "LOCK").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another Nemotron local run owns this state directory")
            return 2
        report = run(args.config, args.state_dir)
    print(json.dumps({"status": report["status"], "run_id": report["run_id"],
                      "reason": report.get("reason")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
