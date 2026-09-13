#!/usr/bin/env python3
"""Run registered CKE nightly profiles opportunistically on an idle local host.

This coordinator owns scheduling only. Test selection, execution, and evidence remain
the responsibility of scripts/nightly_runner.py.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts" / "idle_nightly_cases.json"
GIB = 1024**3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def physical_cpu_ids() -> list[int]:
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    selected: dict[tuple[str, str], int] = {}
    for cpu in sorted(allowed):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text(encoding="utf-8").strip()
            core = (topology / "core_id").read_text(encoding="utf-8").strip()
        except OSError:
            return sorted(allowed)
        selected.setdefault((package, core), cpu)
    return sorted(selected.values()) or sorted(allowed)


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if fields:
            values[key] = int(fields[0]) * 1024
    return values


def swap_counters() -> tuple[int, int]:
    values = {"pswpin": 0, "pswpout": 0}
    try:
        lines = Path("/proc/vmstat").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0, 0
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] in values:
            values[fields[0]] = int(fields[1])
    return values["pswpin"], values["pswpout"]


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_fingerprint(
    case: dict[str, Any], commit: str, execution_identity: dict[str, Any] | None = None
) -> str:
    material = {
        "schema": 2,
        "commit": commit,
        "case": case,
        "execution_identity": execution_identity or {},
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def expand_command(command: list[Any], values: dict[str, str]) -> list[str]:
    if not command or not all(isinstance(part, str) and part for part in command):
        raise ValueError("case command must be a nonempty array of nonempty strings")
    return [part.format_map(values) for part in command]


@dataclass
class HostSnapshot:
    load_1m: float
    physical_cores: int
    available_memory_bytes: int
    free_disk_bytes: int
    free_swap_bytes: int
    swap_in_pages: int
    swap_out_pages: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def snapshot(repo: Path, cpu_ids: list[int]) -> HostSnapshot:
    memory = meminfo()
    usage = shutil.disk_usage(repo)
    swap_in, swap_out = swap_counters()
    return HostSnapshot(
        load_1m=os.getloadavg()[0],
        physical_cores=len(cpu_ids),
        available_memory_bytes=memory.get("MemAvailable", 0),
        free_disk_bytes=usage.free,
        free_swap_bytes=memory.get("SwapFree", 0),
        swap_in_pages=swap_in,
        swap_out_pages=swap_out,
    )


def admission_reasons(
    case: dict[str, Any],
    host: HostSnapshot,
    *,
    pause_file: Path,
    max_load_per_core: float,
    ignore_load: bool = False,
) -> list[str]:
    resources = dict(case.get("resources") or {})
    reasons: list[str] = []
    if pause_file.exists():
        reasons.append(f"reservation present: {pause_file}")
    if host.physical_cores <= 0:
        reasons.append("physical core inventory unavailable")
    elif not ignore_load and host.load_1m / host.physical_cores > max_load_per_core:
        reasons.append(
            f"load per physical core {host.load_1m / host.physical_cores:.2f} exceeds {max_load_per_core:.2f}"
        )
    minimum_memory = float(resources.get("min_available_memory_gib", 0)) * GIB
    if host.available_memory_bytes < minimum_memory:
        reasons.append(
            f"available memory {host.available_memory_bytes / GIB:.1f} GiB is below {minimum_memory / GIB:.1f} GiB"
        )
    minimum_disk = float(resources.get("min_free_disk_gib", 0)) * GIB
    if host.free_disk_bytes < minimum_disk:
        reasons.append(
            f"free disk {host.free_disk_bytes / GIB:.1f} GiB is below {minimum_disk / GIB:.1f} GiB"
        )
    minimum_swap = float(resources.get("min_free_swap_gib", 0)) * GIB
    if host.free_swap_bytes < minimum_swap:
        reasons.append(
            f"free swap {host.free_swap_bytes / GIB:.1f} GiB is below {minimum_swap / GIB:.1f} GiB"
        )
    return reasons


class Coordinator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.source_repo = args.repo.resolve()
        self.repo = self.source_repo
        self.config_path = args.config.resolve()
        try:
            self.config_relative = self.config_path.relative_to(self.source_repo)
        except ValueError:
            self.config_relative = None
        self.config = load_json(self.config_path)
        self._validate_config()
        self.defaults = dict(self.config.get("defaults") or {})

        self.state_dir = args.state_dir.expanduser().resolve()
        self.pause_file = (args.pause_file or self.state_dir / "PAUSE").expanduser().resolve()
        self.ledger_path = self.state_dir / "ledger.json"
        self.lock_path = self.state_dir / "coordinator.lock"
        self.stop_reason: str | None = None
        self.child: subprocess.Popen[bytes] | None = None
        self.cpu_ids = physical_cpu_ids()

    def _validate_config(self) -> None:
        if self.config.get("schema") != "cke.idle-nightly.v1":
            raise ValueError("unsupported idle-nightly configuration schema")

    def _ledger(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            return {"schema": "cke.idle-nightly-ledger.v1", "cases": {}}
        ledger = load_json(self.ledger_path)
        if ledger.get("schema") != "cke.idle-nightly-ledger.v1":
            raise ValueError(f"unsupported ledger schema: {self.ledger_path}")
        ledger.setdefault("cases", {})
        return ledger

    def _save_ledger(self, ledger: dict[str, Any]) -> None:
        ledger["updated_at"] = utc_now()
        atomic_json(self.ledger_path, ledger)

    def _recover_stale_attempts(self, ledger: dict[str, Any]) -> None:
        changed = False
        for record in (ledger.get("cases") or {}).values():
            if record.get("status") != "RUNNING":
                continue
            record.update(
                status="INTERRUPTED",
                reason="previous coordinator ended before recording a terminal result",
                completed_at=utc_now(),
                completed_epoch=time.time(),
            )
            log_path = record.get("log")
            if log_path:
                result_path = Path(str(log_path)).parent / "result.json"
                if result_path.parent.exists():
                    atomic_json(result_path, record)
            changed = True
        if changed:
            self._save_ledger(ledger)

    def _commit(self) -> str:
        return git_output(self.repo, "rev-parse", "HEAD")

    def _execution_identity(self, case: dict[str, Any], commit: str) -> dict[str, Any]:
        configured_env = {
            **{str(key): str(value) for key, value in (self.defaults.get("env") or {}).items()},
            **{str(key): str(value) for key, value in (case.get("env") or {}).items()},
        }
        tracked_env = sorted(
            set(self.defaults.get("fingerprint_env") or [])
            | set(case.get("fingerprint_env") or [])
        )
        env_hashes = {
            name: hashlib.sha256(
                configured_env.get(name, os.environ.get(name, "")).encode()
            ).hexdigest()
            for name in tracked_env
        }
        identity_files: dict[str, str] = {}
        for raw_path in [
            *(self.defaults.get("identity_files") or []),
            *(case.get("identity_files") or []),
        ]:
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = self.repo / path
            identity_files[str(raw_path)] = sha256_file(path) if path.is_file() else "MISSING"
        compiler = shutil.which(configured_env.get("CC", os.environ.get("CC", "cc"))) or ""
        compiler_version = ""
        if compiler:
            try:
                compiler_version = subprocess.check_output(
                    [compiler, "--version"], text=True, stderr=subprocess.STDOUT, timeout=10
                ).splitlines()[0]
            except (OSError, subprocess.SubprocessError, IndexError):
                compiler_version = "UNKNOWN"
        return {
            "commit": commit,
            "physical_cpu_ids": self.cpu_ids,
            "thread_count": len(self.cpu_ids),
            "python": {"executable": sys.executable, "version": sys.version.split()[0]},
            "compiler": {"path": compiler, "version": compiler_version},
            "environment_sha256": env_hashes,
            "identity_file_sha256": identity_files,
        }

    def _prepare_checkout(self) -> None:
        if self.args.fetch:
            subprocess.run(
                ["git", "fetch", "origin", "--prune"],
                cwd=self.source_repo,
                check=True,
                timeout=300,
            )
        if not self.args.ref:
            self.repo = self.source_repo
            return
        commit = git_output(self.source_repo, "rev-parse", f"{self.args.ref}^{{commit}}")
        workspace = self.state_dir / "worktrees" / commit
        if workspace.exists():
            observed = git_output(workspace, "rev-parse", "HEAD")
            if observed != commit:
                raise RuntimeError(f"managed worktree has unexpected commit: {workspace}")
        else:
            workspace.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(workspace), commit],
                cwd=self.source_repo,
                check=True,
                timeout=300,
            )
        self.repo = workspace
        if self.config_relative is not None:
            self.config = load_json(self.repo / self.config_relative)
            self._validate_config()
            self.defaults = dict(self.config.get("defaults") or {})
        self._prune_managed_worktrees(commit)

    def _prune_managed_worktrees(self, current_commit: str) -> None:
        root = self.state_dir / "worktrees"
        keep = max(1, int(self.defaults.get("keep_managed_worktrees", 2)))
        candidates = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir()
                and len(path.name) == 40
                and all(char in "0123456789abcdef" for char in path.name)
                and path.name != current_commit
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if root.exists() else []
        source_common = Path(git_output(self.source_repo, "rev-parse", "--git-common-dir"))
        if not source_common.is_absolute():
            source_common = self.source_repo / source_common
        source_common = source_common.resolve()
        for victim in candidates[max(0, keep - 1):]:
            victim_common = Path(git_output(victim, "rev-parse", "--git-common-dir"))
            if not victim_common.is_absolute():
                victim_common = victim / victim_common
            if victim_common.resolve() != source_common:
                raise RuntimeError(f"refusing to remove unowned worktree: {victim}")
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(victim)],
                cwd=self.source_repo,
                check=True,
                timeout=300,
            )

    def _validate_checkout(self) -> None:
        if not self.args.allow_dirty and git_output(self.repo, "status", "--porcelain"):
            raise RuntimeError("refusing a dirty checkout; use a disposable clean worktree")

    def _signal(self, signum: int, _frame: Any) -> None:
        self.stop_reason = f"coordinator received signal {signum}"
        self._terminate_child()

    def _terminate_child(self) -> None:
        process = self.child
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        grace = float(self.defaults.get("terminate_grace_seconds", 15))
        try:
            process.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=grace)

    def _due_cases(self, ledger: dict[str, Any], commit: str) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
        selected = set(self.args.case or [])
        now = time.time()
        due: list[tuple[float, int, dict[str, Any], str, dict[str, Any]]] = []
        seen: set[str] = set()
        for order, case in enumerate(self.config.get("cases") or []):
            case_id = str(case.get("id") or "").strip()
            if not case_id or case_id in seen:
                raise ValueError(f"missing or duplicate case id: {case_id!r}")
            seen.add(case_id)
            if selected and case_id not in selected:
                continue
            execution_identity = self._execution_identity(case, commit)
            fingerprint = case_fingerprint(case, commit, execution_identity)
            previous = (ledger.get("cases") or {}).get(case_id) or {}
            cadence = float(case.get("cadence_hours", 24)) * 3600
            completed = float(previous.get("completed_epoch", 0) or 0)
            same_pass = previous.get("status") == "PASS" and previous.get("fingerprint") == fingerprint
            if same_pass and not getattr(self.args, "rerun", False) and now - completed < cadence:
                continue
            due.append((completed, order, case, fingerprint, execution_identity))
        if selected - seen:
            raise ValueError(f"unknown case ids: {', '.join(sorted(selected - seen))}")
        return [
            (case, fingerprint, execution_identity)
            for _, _, case, fingerprint, execution_identity in sorted(due)
        ]

    @staticmethod
    def _execution_key(row: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
        return (
            str(row.get("kind") or row.get("execution_kind") or ""),
            str(row.get("id") or row.get("execution_id") or ""),
            tuple(str(arg) for arg in (row.get("args") or row.get("execution_args") or [])),
        )

    def _validate_report(
        self,
        case: dict[str, Any],
        values: dict[str, str],
        *,
        attempt_id: str,
        commit: str,
        started_epoch: float,
    ) -> tuple[str, str, str | None]:
        report_template = case.get("report")
        if not report_template:
            return "PASS", "", None
        report_path = Path(str(report_template).format_map(values)).resolve()
        attempt_dir = Path(values["attempt_dir"]).resolve()
        if report_path != attempt_dir and attempt_dir not in report_path.parents:
            return "ERROR", "configured report path escapes attempt directory", str(report_path)
        if not report_path.is_file():
            return "ERROR", f"expected report is missing: {report_path}", str(report_path)
        try:
            report = load_json(report_path)
            report_time = datetime.fromisoformat(str(report["timestamp"]).replace("Z", "+00:00")).timestamp()
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return "ERROR", f"invalid report: {exc}", str(report_path)
        if report_time + 1 < started_epoch:
            return "ERROR", "report timestamp predates this attempt", str(report_path)
        identity = report.get("run_identity") or {}
        if identity.get("attempt_id") != attempt_id:
            return "ERROR", "report attempt identity does not match", str(report_path)
        if identity.get("repository_commit") != commit:
            return "ERROR", "report repository commit does not match", str(report_path)
        results = report.get("results")
        summary = report.get("summary")
        expected = (report.get("selection") or {}).get("expected_executions")
        if not isinstance(results, list) or not isinstance(summary, dict) or not isinstance(expected, list):
            return "ERROR", "report lacks results, summary, or expected execution inventory", str(report_path)
        if not expected:
            return "ERROR", "expected execution inventory is empty", str(report_path)
        actual_counts = {
            status: sum(1 for row in results if row.get("status") == status)
            for status in ("pass", "fail", "skip", "timeout")
        }
        expected_summary = {
            "total": len(results),
            "passed": actual_counts["pass"],
            "failed": actual_counts["fail"],
            "skipped": actual_counts["skip"],
            "timeout": actual_counts["timeout"],
        }
        if any(summary.get(key) != value for key, value in expected_summary.items()):
            return "ERROR", "report summary does not reconcile with result rows", str(report_path)
        expected_keys = [self._execution_key(row) for row in expected]
        actual_keys = [self._execution_key(row) for row in results]
        if len(set(expected_keys)) != len(expected_keys) or len(set(actual_keys)) != len(actual_keys):
            return "ERROR", "report contains duplicate execution identities", str(report_path)
        if set(expected_keys) != set(actual_keys):
            return "INCOMPLETE", "report execution inventory is missing or unexpected", str(report_path)
        if actual_counts["fail"] or actual_counts["timeout"]:
            return "FAIL", "report contains failed or timed-out executions", str(report_path)
        if actual_counts["skip"]:
            return "INCOMPLETE", "report contains unexpected skipped executions", str(report_path)
        if actual_counts["pass"] != len(results):
            return "ERROR", "report contains an unsupported result status", str(report_path)
        capability = report.get("capability_evidence")
        if not isinstance(capability, dict):
            return "ERROR", "report has no capability evidence", str(report_path)
        capability_errors = capability.get("errors")
        if not isinstance(capability_errors, list):
            return "ERROR", "capability evidence has no error inventory", str(report_path)
        if capability_errors:
            return "INCOMPLETE", "capability evidence contains validation errors", str(report_path)
        return "PASS", "", str(report_path)

    @staticmethod
    def _report_findings(report_path: str | None) -> list[dict[str, Any]]:
        if not report_path:
            return []
        try:
            report = load_json(Path(report_path))
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        findings = []
        for row in report.get("results") or []:
            if not isinstance(row, dict) or row.get("status") == "pass":
                continue
            findings.append(
                {
                    "name": str(row.get("name") or ""),
                    "status": str(row.get("status") or ""),
                    "execution_kind": str(row.get("execution_kind") or ""),
                    "execution_id": str(row.get("execution_id") or ""),
                    "execution_args": [str(arg) for arg in (row.get("execution_args") or [])],
                    "error": str(row.get("error_msg") or "")[:4000],
                }
            )
        for error in (report.get("capability_evidence") or {}).get("errors") or []:
            findings.append(
                {
                    "name": "capability evidence",
                    "status": "incomplete",
                    "error": str(error)[:4000],
                }
            )
        return findings

    def _run_case(
        self,
        case: dict[str, Any],
        fingerprint: str,
        ledger: dict[str, Any],
        commit: str,
        host: HostSnapshot,
        execution_identity: dict[str, Any],
    ) -> str:
        case_id = str(case["id"])
        attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-{fingerprint[:12]}"
        attempt_dir = self.state_dir / "attempts" / case_id / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=False)
        values = {
            "attempt_dir": str(attempt_dir),
            "case_id": case_id,
            "commit": commit,
            "repo": str(self.repo),
        }
        command = expand_command(list(case.get("command") or []), values)
        record: dict[str, Any] = {
            "case_id": case_id,
            "fingerprint": fingerprint,
            "commit": commit,
            "attempt_id": attempt_id,
            "status": "RUNNING",
            "started_at": utc_now(),
            "started_epoch": time.time(),
            "command": command,
            "host_before": host.as_dict(),
            "execution_identity": execution_identity,
            "log": str(attempt_dir / "run.log"),
        }
        ledger["cases"][case_id] = record
        self._save_ledger(ledger)
        atomic_json(attempt_dir / "result.json", record)

        if self.args.dry_run:
            record.update(status="INTERRUPTED", reason="dry run", completed_at=utc_now(), completed_epoch=time.time())
            self._save_ledger(ledger)
            atomic_json(attempt_dir / "result.json", record)
            print(f"DRY-RUN {case_id}: {' '.join(command)}")
            return "INTERRUPTED"

        env = os.environ.copy()
        configured_env = {
            **{str(key): str(value) for key, value in (self.defaults.get("env") or {}).items()},
            **{str(key): str(value) for key, value in (case.get("env") or {}).items()},
        }
        env.update(configured_env)
        thread_count = len(self.cpu_ids)
        for name in ("CK_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = configured_env.get(name, str(thread_count))
        env["CK_IDLE_NIGHTLY_ATTEMPT"] = attempt_id

        old_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        deadline = time.monotonic() + float(case.get("timeout_seconds", self.defaults.get("timeout_seconds", 21600)))
        status = "ERROR"
        reason = ""
        log_path = attempt_dir / "run.log"
        try:
            with log_path.open("wb") as log:
                if old_affinity is not None:
                    os.sched_setaffinity(0, self.cpu_ids)
                try:
                    self.child = subprocess.Popen(
                        command,
                        cwd=self.repo,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    record["pid"] = self.child.pid
                    self._save_ledger(ledger)
                    atomic_json(attempt_dir / "result.json", record)
                finally:
                    if old_affinity is not None:
                        os.sched_setaffinity(0, old_affinity)

                monitor = float(self.defaults.get("monitor_interval_seconds", 10))
                swap_limit = int(self.defaults.get("max_swap_activity_pages", 4096))
                swap_start = (host.swap_in_pages, host.swap_out_pages)
                while self.child.poll() is None:
                    if self.stop_reason:
                        status, reason = "INTERRUPTED", self.stop_reason
                        self._terminate_child()
                        break
                    if self.pause_file.exists():
                        status, reason = "INTERRUPTED", f"reservation appeared: {self.pause_file}"
                        self._terminate_child()
                        break
                    current_memory = meminfo().get("MemAvailable", 0)
                    minimum_runtime_memory = float(
                        (case.get("resources") or {}).get("min_runtime_available_memory_gib", 2)
                    ) * GIB
                    if current_memory < minimum_runtime_memory:
                        status, reason = "INTERRUPTED", "available memory fell below runtime reserve"
                        self._terminate_child()
                        break
                    swap_now = swap_counters()
                    if sum(swap_now) - sum(swap_start) > swap_limit:
                        status, reason = "INTERRUPTED", "swap activity exceeded runtime limit"
                        self._terminate_child()
                        break
                    if time.monotonic() >= deadline:
                        status, reason = "TIMEOUT", "case deadline exceeded"
                        self._terminate_child()
                        break
                    time.sleep(monitor)
                else:
                    returncode = self.child.returncode
                    if case.get("report"):
                        report_status, report_reason, report_path = self._validate_report(
                            case,
                            values,
                            attempt_id=attempt_id,
                            commit=commit,
                            started_epoch=record["started_epoch"],
                        )
                        record["report_validation"] = {
                            "status": report_status,
                            "reason": report_reason,
                        }
                        if report_path:
                            record["report"] = report_path
                            record["report_findings"] = self._report_findings(report_path)
                    else:
                        report_status, report_reason = "PASS", ""
                    if returncode == 0:
                        status, reason = report_status, report_reason
                    else:
                        status, reason = "FAIL", f"process exited {returncode}"
                        if report_reason:
                            reason += f"; report validation: {report_reason}"
        except Exception as exc:
            status, reason = "ERROR", f"{type(exc).__name__}: {exc}"
            self._terminate_child()
        finally:
            returncode = self.child.returncode if self.child is not None else None
            self.child = None

        record.update(
            status=status,
            reason=reason,
            returncode=returncode,
            completed_at=utc_now(),
            completed_epoch=time.time(),
            host_after=snapshot(self.repo, self.cpu_ids).as_dict(),
        )
        ledger["cases"][case_id] = record
        self._save_ledger(ledger)
        atomic_json(attempt_dir / "result.json", record)
        print(f"{status} {case_id}: {reason or 'completed'}")
        return status

    def run(self) -> int:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock_stream = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_stream.close()
            print(f"idle nightly coordinator already owns {self.lock_path}")
            return 0
        try:
            self._prepare_checkout()
            self._validate_checkout()
            previous_sigterm = signal.signal(signal.SIGTERM, self._signal)
            previous_sigint = signal.signal(signal.SIGINT, self._signal)
            commit = self._commit()
            ledger = self._ledger()
            self._recover_stale_attempts(ledger)
            due = self._due_cases(ledger, commit)
            if self.args.list:
                due_ids = {case["id"] for case, _, _ in due}
                for case in self.config.get("cases") or []:
                    print(f"{'DUE' if case['id'] in due_ids else 'CURRENT':7} {case['id']}")
                return 0
            if not due:
                print("no idle-nightly cases are due")
                return 0

            max_cases = max(1, int(self.args.max_cases))
            ran = 0
            failures = 0
            for case, fingerprint, execution_identity in due:
                host = snapshot(self.repo, self.cpu_ids)
                reasons = admission_reasons(
                    case,
                    host,
                    pause_file=self.pause_file,
                    max_load_per_core=float(self.defaults.get("max_load_per_physical_core", 0.20)),
                    ignore_load=getattr(self.args, "exclusive", False),
                )
                if reasons:
                    print(f"DEFER {case['id']}: {'; '.join(reasons)}")
                    continue
                status = self._run_case(case, fingerprint, ledger, commit, host, execution_identity)
                ran += 1
                failures += status not in {"PASS", "INTERRUPTED"}
                if ran >= max_cases or self.stop_reason:
                    break
            if ran == 0:
                print("due cases deferred because this host is not currently eligible")
            return 1 if failures else 0
        finally:
            if "previous_sigterm" in locals():
                signal.signal(signal.SIGTERM, previous_sigterm)
                signal.signal(signal.SIGINT, previous_sigint)
            lock_stream.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--ref", help="trusted ref to run in a clean per-commit worktree, for example origin/main")
    parser.add_argument("--fetch", action="store_true", help="fetch origin before resolving --ref")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("CK_IDLE_NIGHTLY_STATE", "~/.local/state/cke/idle-nightly")),
    )
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--case", action="append", help="run only a named registered case")
    parser.add_argument("--max-cases", type=int, default=1, help="maximum cases per invocation")
    parser.add_argument("--rerun", action="store_true", help="ignore completed-result cadence only")
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="reserved-host mode: bypass load admission, but never pause or capacity safety checks",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="development only; cron should use a clean worktree")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return Coordinator(parse_args(argv)).run()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"idle nightly coordinator error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
