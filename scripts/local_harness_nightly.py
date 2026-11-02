#!/usr/bin/env python3
"""Run the existing Qwen/Gemma repeatability matrix on available local endpoints.

This opt-in client-side nightly never launches models or retries tool tasks. A
missing endpoint is NOT_TESTED; a wrong loaded identity is FAIL. Configure one
exact artifact per client in a private cke.harness_repeatability.v1 JSON file.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
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


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cke.local_harness_nightly.v1"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _identity(endpoint: str) -> dict:
    with urlopen(endpoint.rstrip("/") + "/cke/loaded-identity", timeout=5) as response:
        data = json.load(response)
    if not isinstance(data, dict) or data.get("schema") != "cke.loaded_serving_identity.v1":
        raise ValueError("endpoint returned an unsupported identity schema")
    return data


def _preflight(client: dict) -> tuple[str, str, str | None]:
    try:
        observed = _identity(client["endpoint"])
    except HTTPError as exc:
        return "FAIL", f"loaded identity endpoint returned HTTP {exc.code}", None
    except ValueError as exc:
        return "FAIL", f"loaded identity is invalid: {exc}", None
    except (OSError, URLError, TimeoutError) as exc:
        return "NOT_TESTED", f"endpoint unavailable: {type(exc).__name__}", None
    instance = observed.get("server_instance_id")
    if not isinstance(instance, str) or not instance:
        return "FAIL", "loaded server instance is missing", None
    for expected, actual in (
        (client["model"], observed.get("model")),
        (client["serving_identity"], observed.get("serving_identity")),
        (client["session_library_sha256"], observed.get("session_library_sha256")),
    ):
        if expected != actual:
            return "FAIL", "loaded artifact identity differs from configuration", instance
    return "READY", "", instance


def _invoke(command: list[str], *, cwd: Path, log: Path, deadline: int | None) -> int:
    with log.open("wb") as output:
        child = subprocess.Popen(command, cwd=cwd, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return 124


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def _refresh_main() -> str:
    """Advance an otherwise untouched dedicated checkout to fetched main."""
    if subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=ROOT,
                      capture_output=True, check=False).returncode == 0:
        raise ValueError("source refresh requires a detached worktree")
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True).strip():
        raise ValueError("source refresh requires a clean worktree")
    subprocess.run(
        ["git", "fetch", "--quiet", "--no-tags", "origin",
         "+refs/heads/main:refs/remotes/origin/main"],
        cwd=ROOT, check=True, timeout=90,
    )
    target = subprocess.check_output(
        ["git", "rev-parse", "refs/remotes/origin/main"], cwd=ROOT, text=True,
    ).strip()
    subprocess.run(["git", "checkout", "--detach", target], cwd=ROOT,
                   check=True, timeout=30, capture_output=True)
    if _source_commit() != target:
        raise RuntimeError("source refresh did not load fetched main")
    return target


def _http_dependencies_ready() -> bool:
    """The coordinator launches HTTP probes with this same Python interpreter."""
    try:
        importlib.import_module("httpx")
    except ImportError:
        return False
    return True


def _prefix_certification(config_path: Path, run_dir: Path,
                          paused: bool) -> dict:
    """Optional pinned native prefix lane; missing assets remain NOT_TESTED."""
    row = {"status": "INCOMPLETE", "config": str(config_path), "cases": []}
    if paused:
        row.update(status="NOT_TESTED", reason="paused before startup")
        return row
    if not config_path.is_file():
        row.update(status="NOT_TESTED", reason="prefix certification config unavailable")
        return row
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or config.get("schema") != "cke.local_prefix_cert.v1":
        raise ValueError("invalid local prefix certification schema")
    paths = {}
    for name in ("bundle", "session_library", "ids"):
        value = config.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing prefix certification {name}")
        paths[name] = Path(value).expanduser().resolve()
    cases = config.get("cases")
    threads = config.get("threads", 16)
    max_seconds = config.get("max_seconds", 7200)
    if (not isinstance(cases, list) or not 1 <= len(cases) <= 16 or
            type(threads) is not int or not 1 <= threads <= 256 or
            type(max_seconds) is not int or not 60 <= max_seconds <= 86400 or
            any(not isinstance(case, dict) or
                type(case.get("prefix")) is not int or case["prefix"] < 1 or
                type(case.get("suffix")) is not int or case["suffix"] < 2
                for case in cases)):
        raise ValueError("invalid prefix certification cases, threads or time budget")
    required = (paths["bundle"] / "libmodel.so",
                paths["bundle"] / "libckernel_engine.so",
                paths["bundle"] / "layout_decode.json",
                paths["bundle"] / "weights.bump",
                paths["bundle"] / "model_v8.c",
                paths["bundle"] / "weights_manifest.map",
                paths["bundle"] / "config.json",
                paths["session_library"], paths["ids"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        row.update(status="NOT_TESTED", reason="prefix assets unavailable",
                   missing=missing)
        return row
    bundle_hashes = config.get("artifact_sha256")
    artifact_names = ("libmodel.so", "libckernel_engine.so", "model_v8.c",
                      "weights.bump", "weights_manifest.map", "config.json")
    if not isinstance(bundle_hashes, dict) or set(bundle_hashes) != set(artifact_names):
        raise ValueError("missing pinned bundle artifact hashes")
    pinned = [(name, paths["bundle"] / name, bundle_hashes[name])
              for name in artifact_names]
    pinned.append(("session_library", paths["session_library"],
                   config.get("session_sha256")))
    for name, path, expected in pinned:
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError(f"missing pinned {name} sha256")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            row.update(status="FAIL", reason=f"pinned {name} differs")
            return row
    ids = json.loads(paths["ids"].read_text())
    if not isinstance(ids, list) or any(type(value) is not int or value < 0
                                        for value in ids):
        raise ValueError("invalid pinned exact token IDs")
    expected_ids = config.get("ids_sha256")
    if (not isinstance(expected_ids, str) or len(expected_ids) != 64 or
            any(char not in "0123456789abcdef" for char in expected_ids)):
        raise ValueError("missing pinned ids sha256")
    observed_ids = hashlib.sha256(json.dumps(
        ids, separators=(",", ":")).encode()).hexdigest()
    if observed_ids != expected_ids:
        row.update(status="FAIL", reason="pinned ids differ")
        return row
    lane = run_dir / "prefix_certification"
    lane.mkdir(parents=True)
    probe = ROOT / "version/v8/scripts/certify_session_multitoken_prefill_v8.py"
    if not probe.is_file():
        row.update(status="NOT_TESTED", reason="prefix probe unavailable")
        return row
    for index, case in enumerate(cases):
        output = lane / f"case-{index}.json"
        code = _invoke([
            sys.executable, str(probe), "--bundle", str(paths["bundle"]),
            "--session-library", str(paths["session_library"]),
            "--ids", str(paths["ids"]), "--prefix", str(case["prefix"]),
            "--suffix", str(case["suffix"]), "--threads", str(threads),
            "--out", str(output),
        ], cwd=ROOT, log=lane / f"case-{index}.log", deadline=max_seconds)
        try:
            observed = json.loads(output.read_text())
        except (OSError, ValueError):
            observed = {"verdict": "INCOMPLETE"}
        verdict = observed.get("verdict")
        row["cases"].append({"prefix": case["prefix"], "suffix": case["suffix"],
                             "verdict": verdict, "exit_code": code,
                             "report": str(output)})
        if code == 124:
            row.update(status="INCOMPLETE", reason="prefix case time budget expired")
            return row
        reported_artifacts = observed.get("artifact_sha256") or {}
        reported_session = observed.get("session_library_sha256")
        reported_ids = observed.get("ids_sha256")
        if (code != 0 or verdict != "PASS" or
                any(reported_artifacts.get(name) != bundle_hashes[name]
                    for name in artifact_names) or
                reported_session != config["session_sha256"] or
                reported_ids != config["ids_sha256"]):
            row.update(status="FAIL", reason="prefix parity case failed")
            return row
    row["status"] = "PASS"
    return row


def run(config_path: Path, state_dir: Path, *, refresh_main: bool = False,
        prefix_cert_config: Path | None = None) -> dict:
    state_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    run_dir = state_dir / "runs" / (started.strftime("%Y%m%dT%H%M%S%fZ"))
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": SCHEMA, "source_commit": None, "run_id": run_dir.name,
        "started_at": started.isoformat(), "status": "INCOMPLETE",
        "clients": [],
    }
    def publish() -> None:
        _write(state_dir / "latest.json", report)
        _write(run_dir / "summary.json", report)

    # Replace a previous PASS before reading configuration or starting a child.
    publish()
    stage = "source_commit"
    try:
        report["source_commit"] = _source_commit()
        if refresh_main:
            if (state_dir / "PAUSE").exists():
                report["source_refresh"] = "skipped_paused"
            else:
                stage = "source_refresh"
                report["source_commit_before_refresh"] = report["source_commit"]
                report["source_commit"] = _refresh_main()
                report["source_refresh"] = "fetched_origin_main"
                publish()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from server.harness_repeatability import _validate

        stage = "configuration"
        if not config_path.is_file():
            report["reason"] = "configuration unavailable"
            report["status"] = "NOT_TESTED"
        else:
            raw = config_path.read_bytes()
            report["config_sha256"] = hashlib.sha256(raw).hexdigest()
            config = _validate(json.loads(raw))
            if (state_dir / "PAUSE").exists():
                report["status"] = "NOT_TESTED"
                report["reason"] = "local harness nightly paused before startup"
                report["clients"] = [{"name": item["name"], "status": "NOT_TESTED"}
                                     for item in config["clients"]]
            elif not _http_dependencies_ready():
                report["status"] = "FAIL"
                report["reason"] = "client Python lacks httpx required by the HTTP acceptance suites"
                report["clients"] = [{"name": item["name"], "status": "FAIL",
                                      "reason": report["reason"]}
                                     for item in config["clients"]]
            else:
                for client in config["clients"]:
                    name = client["name"]
                    stage = f"preflight:{name}"
                    status, reason, instance = _preflight(client)
                    row = {"name": name, "status": status, "reason": reason,
                           "endpoint": client["endpoint"], "model": client["model"],
                           "serving_identity": client["serving_identity"],
                           "session_library_sha256": client["session_library_sha256"],
                           "server_instance_id": instance}
                    report["clients"].append(row)
                    publish()
                    if status == "READY":
                        row["status"] = "INCOMPLETE"
                        publish()
                        directory = run_dir / name
                        child_config = directory / "config.json"
                        _write(child_config, {
                            "schema": config["schema"], "attempts": config["attempts"],
                            "clients": [client],
                        })
                        outcome = directory / "matrix"
                        stage = f"launch:{name}"
                        code = _invoke([
                            sys.executable, str(ROOT / "server/harness_repeatability.py"),
                            "--config", str(child_config), "--output-dir", str(outcome),
                        ], cwd=ROOT, log=directory / "run.log", deadline=None)
                        summary_path = outcome / "summary.json"
                        if code == 0 and summary_path.is_file():
                            stage = f"reassessment:{name}"
                            reassess_code = _invoke([
                                sys.executable, str(ROOT / "server/harness_repeatability.py"),
                                "--reassess-dir", str(outcome),
                            ], cwd=ROOT, log=directory / "reassess.log", deadline=120)
                            reassess_path = outcome / "reassessment.json"
                            if reassess_code == 0 and reassess_path.is_file():
                                reassessed = json.loads(reassess_path.read_text())
                                original = json.loads(summary_path.read_text())
                                if (reassessed.get("status") == "pass"
                                        and original.get("status") == "pass"
                                        and original.get("source_commit") == report["source_commit"]
                                        and reassessed.get("reassessment_source_commit") == report["source_commit"]
                                        and len(original.get("steps", [])) == 3 + config["attempts"]
                                        and len(reassessed.get("steps", [])) == 3 + config["attempts"]
                                        and all(step.get("server_instance_id") == instance
                                                for step in original["steps"])
                                        and all(step.get("server_instance_id") == instance
                                                for step in reassessed.get("steps", []))):
                                    row.update(status="PASS", reason="", steps=len(original.get("steps", [])))
                        if row["status"] != "PASS":
                            row.update(status="FAIL", reason="repeatability or reassessment failed",
                                       exit_code=code)
                    publish()
                statuses = {item["status"] for item in report["clients"]}
                report["status"] = ("FAIL" if "FAIL" in statuses else
                                    "PASS" if statuses == {"PASS"} else
                                    "NOT_TESTED" if statuses == {"NOT_TESTED"} else "INCOMPLETE")
        if prefix_cert_config is not None:
            stage = "prefix_certification"
            report["prefix_certification"] = {
                "status": "INCOMPLETE", "config": str(prefix_cert_config),
                "cases": []}
            publish()
            report["prefix_certification"] = _prefix_certification(
                prefix_cert_config, run_dir, (state_dir / "PAUSE").exists())
            publish()
            prefix_status = report["prefix_certification"]["status"]
            if prefix_status == "FAIL":
                report["status"] = "FAIL"
            elif "INCOMPLETE" in (prefix_status, report["status"]):
                report["status"] = "INCOMPLETE"
            elif prefix_status != report["status"]:
                report["status"] = "INCOMPLETE"
    except Exception as exc:
        report.update(status="FAIL", reason=f"{stage} failed: {type(exc).__name__}: {exc}",
                      error_type=type(exc).__name__, failure_stage=stage)
        for row in report["clients"]:
            if row["status"] in {"READY", "INCOMPLETE"}:
                row.update(status="FAIL", reason=report["reason"])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    publish()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--refresh-main", action="store_true",
                        help="fetch origin/main in a clean detached worktree before testing")
    parser.add_argument("--prefix-cert-config", type=Path,
                        help="optional pinned exact-prefix native certification matrix")
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "LOCK").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another local harness nightly owns this state directory")
            return 2
        try:
            result = run(args.config, args.state_dir, refresh_main=args.refresh_main,
                         prefix_cert_config=args.prefix_cert_config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"local harness nightly configuration or report error: {exc}")
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
