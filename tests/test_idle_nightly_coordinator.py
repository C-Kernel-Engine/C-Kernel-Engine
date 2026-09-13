from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "idle_nightly_coordinator.py"


def load_coordinator():
    spec = importlib.util.spec_from_file_location("idle_nightly_coordinator_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_coordinator()


def init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "tracked").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "test fixture"], check=True)


def write_config(path: Path, cases: list[dict], **defaults) -> Path:
    config = path / "cases.json"
    config.write_text(
        json.dumps(
            {
                "schema": "cke.idle-nightly.v1",
                "defaults": {
                    "max_load_per_physical_core": 999,
                    "monitor_interval_seconds": 0.02,
                    "terminate_grace_seconds": 0.05,
                    "max_swap_activity_pages": 10**12,
                    **defaults,
                },
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return config


def args(repo: Path, state: Path, config: Path, **overrides) -> Namespace:
    values = {
        "config": config,
        "repo": repo,
        "ref": None,
        "fetch": False,
        "state_dir": state,
        "pause_file": None,
        "case": [],
        "max_cases": 1,
        "rerun": True,
        "exclusive": True,
        "dry_run": False,
        "list": False,
        "allow_dirty": False,
    }
    values.update(overrides)
    return Namespace(**values)


class IdleNightlyCoordinatorTests(unittest.TestCase):
    def test_checked_capacity_tiers_all_use_the_shared_nightly_runner(self) -> None:
        config = json.loads((ROOT / "scripts" / "idle_nightly_cases.json").read_text(encoding="utf-8"))
        memory_tiers = []
        for case in config["cases"]:
            self.assertEqual(case["command"][:2], ["python3", "scripts/nightly_runner.py"])
            self.assertIn("--prepare", case["command"])
            self.assertIn("--json", case["command"])
            self.assertIn("--markdown", case["command"])
            memory_tiers.append(case["resources"]["min_available_memory_gib"])
        self.assertEqual(memory_tiers, sorted(memory_tiers))

    def test_admission_reports_load_memory_disk_and_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pause = Path(tmp) / "PAUSE"
            pause.touch()
            host = runner.HostSnapshot(8.0, 4, 2 * runner.GIB, 3 * runner.GIB, runner.GIB, 0, 0)
            reasons = runner.admission_reasons(
                {
                    "resources": {
                        "min_available_memory_gib": 4,
                        "min_free_disk_gib": 5,
                        "min_free_swap_gib": 2,
                    }
                },
                host,
                pause_file=pause,
                max_load_per_core=3.0,
            )
        self.assertEqual(len(reasons), 4)
        self.assertTrue(any("reservation" in reason for reason in reasons))
        self.assertTrue(any("available memory" in reason for reason in reasons))
        self.assertTrue(any("free disk" in reason for reason in reasons))
        self.assertTrue(any("free swap" in reason for reason in reasons))

    def test_stale_running_attempt_becomes_interrupted_and_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            attempt = state / "attempts" / "quick" / "old"
            attempt.mkdir(parents=True)
            record = {"case_id": "quick", "status": "RUNNING", "log": str(attempt / "run.log")}
            ledger = {"schema": "cke.idle-nightly-ledger.v1", "cases": {"quick": record}}
            config = write_config(state, [{"id": "quick", "command": ["true"]}])
            coordinator = runner.Coordinator(args(state, state / "state", config, allow_dirty=True))
            coordinator.ledger_path = state / "ledger.json"
            coordinator._recover_stale_attempts(ledger)
            self.assertEqual(record["status"], "INTERRUPTED")
            preserved = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(preserved["status"], "INTERRUPTED")

    def test_fingerprint_changes_with_commit_or_case_contract(self) -> None:
        case = {"id": "quick", "command": ["true"]}
        first = runner.case_fingerprint(case, "a" * 40)
        self.assertNotEqual(first, runner.case_fingerprint(case, "b" * 40))
        self.assertNotEqual(first, runner.case_fingerprint({**case, "cadence_hours": 2}, "a" * 40))

    def test_completed_identical_case_is_reused_until_cadence_expires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_repo(root)
            case = {"id": "quick", "cadence_hours": 24, "command": ["true"]}
            config = write_config(root, [case])
            coordinator = runner.Coordinator(args(root, root / "state", config))
            commit = coordinator._commit()
            execution_identity = coordinator._execution_identity(case, commit)
            fingerprint = runner.case_fingerprint(case, commit, execution_identity)
            ledger = {
                "schema": "cke.idle-nightly-ledger.v1",
                "cases": {"quick": {"status": "PASS", "fingerprint": fingerprint, "completed_epoch": time.time()}},
            }
            coordinator.args.rerun = False
            self.assertEqual(coordinator._due_cases(ledger, commit), [])
            self.assertEqual(len(coordinator._due_cases(ledger, "f" * 40)), 1)

    def test_execution_identity_changes_fingerprint(self) -> None:
        case = {"id": "quick", "command": ["true"]}
        first = runner.case_fingerprint(case, "a" * 40, {"thread_count": 8})
        second = runner.case_fingerprint(case, "a" * 40, {"thread_count": 16})
        self.assertNotEqual(first, second)

    def test_exit_zero_without_required_report_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_repo(root)
            case = {
                "id": "quick",
                "command": [sys.executable, "-c", "raise SystemExit(0)"],
                "report": "{attempt_dir}/nightly.json",
            }
            config = write_config(root, [case])
            state = root / "state"
            coordinator = runner.Coordinator(args(root, state, config, allow_dirty=True))
            self.assertEqual(coordinator.run(), 1)
            record = json.loads((state / "ledger.json").read_text(encoding="utf-8"))["cases"]["quick"]
            self.assertEqual(record["status"], "ERROR")
            self.assertIn("expected report is missing", record["reason"])

    def test_nonzero_exit_retains_structured_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_repo(root)
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            code = (
                "import json,os,pathlib; "
                "p=pathlib.Path(r'{attempt_dir}/nightly.json'); "
                "r=dict(timestamp=__import__('datetime').datetime.now().isoformat(),"
                f"run_identity=dict(attempt_id=os.environ['CK_IDLE_NIGHTLY_ATTEMPT'],repository_commit='{commit}'),"
                "selection=dict(expected_executions=[dict(kind='python',id='broken',args=[])]),"
                "summary=dict(total=1,passed=0,failed=1,skipped=0,timeout=0),"
                "capability_evidence=dict(errors=[]),"
                "results=[dict(name='Broken fixture',status='fail',execution_kind='python',"
                "execution_id='broken',execution_args=[],error_msg='seeded failure')]); "
                "p.write_text(json.dumps(r)); raise SystemExit(4)"
            )
            case = {
                "id": "failure-report",
                "command": [sys.executable, "-c", code],
                "report": "{attempt_dir}/nightly.json",
            }
            config = write_config(root, [case])
            state = root / "state"
            coordinator = runner.Coordinator(args(root, state, config, allow_dirty=True))
            self.assertEqual(coordinator.run(), 1)
            record = json.loads((state / "ledger.json").read_text(encoding="utf-8"))["cases"]["failure-report"]
            self.assertEqual(record["status"], "FAIL")
            self.assertEqual(record["returncode"], 4)
            self.assertEqual(record["report_validation"]["status"], "FAIL")
            self.assertEqual(record["report_findings"][0]["error"], "seeded failure")
            self.assertTrue(Path(record["report"]).is_file())

    def test_report_requires_matching_identity_complete_inventory_and_no_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_repo(root)
            config = write_config(root, [{"id": "quick", "command": ["true"]}])
            coordinator = runner.Coordinator(args(root, root / "state", config))
            attempt = root / "attempt"
            attempt.mkdir()
            report = attempt / "nightly.json"
            now = time.time()
            base = {
                "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "run_identity": {"attempt_id": "attempt-1", "repository_commit": "a" * 40},
                "selection": {
                    "expected_executions": [{"kind": "python", "id": "fixture", "args": []}]
                },
                "summary": {"total": 1, "passed": 1, "failed": 0, "skipped": 0, "timeout": 0},
                "capability_evidence": {"errors": []},
                "results": [
                    {
                        "status": "pass",
                        "execution_kind": "python",
                        "execution_id": "fixture",
                        "execution_args": [],
                    }
                ],
            }
            report.write_text(json.dumps(base), encoding="utf-8")
            values = {"attempt_dir": str(attempt)}
            case = {"report": "{attempt_dir}/nightly.json"}
            status, _, _ = coordinator._validate_report(
                case, values, attempt_id="attempt-1", commit="a" * 40, started_epoch=now - 1
            )
            self.assertEqual(status, "PASS")

            base["selection"]["expected_executions"] = []
            base["results"] = []
            base["summary"].update(total=0, passed=0)
            report.write_text(json.dumps(base), encoding="utf-8")
            status, reason, _ = coordinator._validate_report(
                case, values, attempt_id="attempt-1", commit="a" * 40, started_epoch=now - 1
            )
            self.assertEqual(status, "ERROR")
            self.assertIn("inventory is empty", reason)
            base["selection"]["expected_executions"] = [
                {"kind": "python", "id": "fixture", "args": []}
            ]
            base["results"] = [
                {
                    "status": "pass",
                    "execution_kind": "python",
                    "execution_id": "fixture",
                    "execution_args": [],
                }
            ]
            base["summary"].update(total=1, passed=1)

            base["capability_evidence"]["errors"] = ["stale capability row"]
            report.write_text(json.dumps(base), encoding="utf-8")
            status, reason, _ = coordinator._validate_report(
                case, values, attempt_id="attempt-1", commit="a" * 40, started_epoch=now - 1
            )
            self.assertEqual(status, "INCOMPLETE")
            self.assertIn("capability evidence", reason)
            base["capability_evidence"]["errors"] = []

            base["results"][0]["status"] = "skip"
            base["summary"].update(passed=0, skipped=1)
            report.write_text(json.dumps(base), encoding="utf-8")
            case["allow_skips"] = True
            status, reason, _ = coordinator._validate_report(
                case, values, attempt_id="attempt-1", commit="a" * 40, started_epoch=now - 1
            )
            self.assertEqual(status, "INCOMPLETE")
            self.assertIn("skipped", reason)

            base["run_identity"]["attempt_id"] = "stale-attempt"
            report.write_text(json.dumps(base), encoding="utf-8")
            status, reason, _ = coordinator._validate_report(
                case, values, attempt_id="attempt-1", commit="a" * 40, started_epoch=now - 1
            )
            self.assertEqual(status, "ERROR")
            self.assertIn("attempt identity", reason)

    def test_child_uses_physical_core_affinity_and_bounded_thread_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_repo(root)
            code = (
                "import json,os,pathlib; "
                "pathlib.Path(r'{attempt_dir}/child.json').write_text(json.dumps(dict("
                "threads=os.environ['CK_NUM_THREADS'], affinity=sorted(os.sched_getaffinity(0)))))"
            )
            config = write_config(root, [{"id": "quick", "command": [sys.executable, "-c", code]}])
            state = root / "state"
            coordinator = runner.Coordinator(args(root, state, config, allow_dirty=True))
            self.assertEqual(coordinator.run(), 0)
            record = json.loads((state / "ledger.json").read_text(encoding="utf-8"))["cases"]["quick"]
            child = json.loads((Path(record["log"]).parent / "child.json").read_text(encoding="utf-8"))
            self.assertEqual(child["affinity"], coordinator.cpu_ids)
            self.assertEqual(int(child["threads"]), len(coordinator.cpu_ids))
            self.assertEqual(record["status"], "PASS")

    def test_trusted_ref_runs_in_clean_commit_specific_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            init_repo(root)
            state = Path(tmp) / "state"
            config = write_config(
                Path(tmp),
                [{"id": "quick", "command": [sys.executable, "-c", "raise SystemExit(0)"]}],
            )
            coordinator = runner.Coordinator(
                args(root, state, config, ref="HEAD", allow_dirty=False)
            )
            self.assertEqual(coordinator.run(), 0)
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(coordinator.repo, state / "worktrees" / commit)
            self.assertEqual(runner.git_output(coordinator.repo, "status", "--porcelain"), "")

    def test_failure_does_not_prevent_an_independent_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_repo(root)
            cases = [
                {"id": "fails", "command": [sys.executable, "-c", "raise SystemExit(3)"]},
                {"id": "passes", "command": [sys.executable, "-c", "raise SystemExit(0)"]},
            ]
            config = write_config(root, cases)
            state = root / "state"
            coordinator = runner.Coordinator(args(root, state, config, max_cases=2, allow_dirty=True))
            self.assertEqual(coordinator.run(), 1)
            ledger = json.loads((state / "ledger.json").read_text(encoding="utf-8"))["cases"]
            self.assertEqual(ledger["fails"]["status"], "FAIL")
            self.assertEqual(ledger["passes"]["status"], "PASS")

    def test_reservation_kills_and_reaps_owned_sigterm_ignoring_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_repo(root)
            code = (
                "import os,pathlib,signal,time; "
                "pathlib.Path(r'{attempt_dir}/child.pid').write_text(str(os.getpid())); "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
            )
            config = write_config(root, [{"id": "hang", "command": [sys.executable, "-c", code]}])
            state = root / "state"
            pause = state / "PAUSE"

            def reserve() -> None:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if list((state / "attempts").glob("hang/*/child.pid")):
                        pause.touch()
                        return
                    time.sleep(0.01)

            thread = threading.Thread(target=reserve)
            thread.start()
            coordinator = runner.Coordinator(args(root, state, config, allow_dirty=True))
            self.assertEqual(coordinator.run(), 0)
            thread.join(timeout=3)
            record = json.loads((state / "ledger.json").read_text(encoding="utf-8"))["cases"]["hang"]
            self.assertEqual(record["status"], "INTERRUPTED")
            self.assertIn("reservation appeared", record["reason"])
            with self.assertRaises(ProcessLookupError):
                os.kill(int(record["pid"]), 0)

    def test_lock_makes_overlapping_cron_invocation_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_repo(root)
            config = write_config(root, [{"id": "quick", "command": ["true"]}])
            state = root / "state"
            state.mkdir()
            with (state / "coordinator.lock").open("a+", encoding="utf-8") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                coordinator = runner.Coordinator(args(root, state, config))
                self.assertEqual(coordinator.run(), 0)
            self.assertFalse((state / "ledger.json").exists())


if __name__ == "__main__":
    unittest.main()
