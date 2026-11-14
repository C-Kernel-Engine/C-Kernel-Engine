"""Independent consumer contracts; simulated execution, no remote-auth claim."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.training.jobs import JobError, exchange
from server.training.bridge_consumer import validate, InvalidReply, digest
from server.training.job_consumer import verify_artifacts, InvalidObservation
from server.tests.test_training_jobs import SimulatedJobs
from server.tests import test_training_contract as validation_tests


class QueuedJobs(SimulatedJobs):
    def submit(self, *args, **kwargs):
        return super().submit(*args, **dict(kwargs, start=False))


class TrainingBridge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "jobs"
        self.jobs = QueuedJobs(self.path, principals={"tenant-a", "tenant-b"})
        self.addCleanup(lambda: self.jobs.close())
        helper = validation_tests.TrainingContractTest()
        helper.setUp()
        self.config = helper.request(self.jobs.backend)["config"]
        self.revision = "00" + "f" * 30
        self.caps = self.discover()

    def request(self, operation, **kwargs):
        q = dict(
            schema="cke.training.jobs.v2", request_id="request", operation=operation
        )
        if operation != "discover":
            q.update(
                worker_boot_id=self.caps["worker_boot_id"],
                capabilities_sha256=self.caps["capabilities_sha256"],
                client_revision=self.revision,
            )
        q.update(kwargs)
        return q

    def call(self, request, owner="tenant-a"):
        reply = exchange(self.jobs, owner, request)
        return validate(
            reply,
            request,
            capabilities=getattr(self, "caps", None),
            owner=owner,
            allow_simulation=True,
        )

    def discover(self):
        return self.call(self.request("discover"))

    def prepare(self):
        return self.call(
            self.request(
                "prepare", config=self.config, experiment_id="decision-experiment"
            )
        )

    def submit(self, plan, key="stable-key"):
        return self.call(
            self.request("submit", plan_sha256=plan["plan_sha256"], idempotency_key=key)
        )

    def test_revision_prepare_has_no_execution_and_retry_after_lost_reply(self):
        with patch(
            "subprocess.Popen", side_effect=AssertionError("prepare cannot execute")
        ):
            p = self.prepare()
        self.assertEqual(p["plan"]["client_revision"], self.revision)
        self.assertIsNone(p["plan"]["draft_revision"])
        self.assertEqual(self.jobs.calls, 0)
        first = self.submit(p)  # Simulate losing this reply at the caller.
        retried = self.submit(p)
        self.assertEqual(first, retried | {"observed_at": first["observed_at"]})
        jid = first["job"]["job_id"]
        self.jobs.run_job("tenant-a", jid)
        done = self.submit(p)
        self.assertEqual(done["job"]["status"], "succeeded")
        self.assertEqual(self.jobs.calls, 1)
        self.assertEqual(done["job"]["client_revision"], self.revision)
        self.assertFalse(done["cancel_supported"])
        verified = verify_artifacts(
            done,
            lambda aid, offset: self.call(
                self.request("artifact", job_id=jid, artifact_id=aid, offset=offset)
            ),
        )
        self.assertEqual(len(verified), 2)

    def test_wrong_revision_boot_capabilities_and_caller_identity_reject(self):
        p = self.prepare()
        for field, value, error in (
            ("worker_boot_id", "0" * 32, "STALE_WORKER"),
            ("capabilities_sha256", "0" * 64, "STALE_CAPABILITIES"),
            ("client_revision", "1" * 32, "CLIENT_REVISION_MISMATCH"),
            ("client_revision", 123, "INVALID_CLIENT_REVISION"),
        ):
            q = self.request(
                "submit", plan_sha256=p["plan_sha256"], idempotency_key="key"
            )
            q[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(JobError, error):
                self.call(q)
        for extra in (
            {"principal": "tenant-b"},
            {"path": "/tmp/a"},
            {"draft_revision": 1},
        ):
            q = self.request(
                "prepare", config=self.config, experiment_id="experiment", **extra
            )
            with self.assertRaisesRegex(JobError, "INVALID_REQUEST"):
                self.call(q)
        q = self.request("submit", plan_sha256=p["plan_sha256"], idempotency_key="key")
        with self.assertRaisesRegex(JobError, "UNKNOWN_PLAN"):
            self.call(q, "tenant-b")
        first = self.submit(p)
        jid = first["job"]["job_id"]
        for operation, fields in (
            ("observe", {}),
            ("artifact", {"artifact_id": "0" * 32, "offset": 0}),
        ):
            with self.assertRaisesRegex(JobError, "UNKNOWN_JOB"):
                self.call(self.request(operation, job_id=jid, **fields), "tenant-b")
        self.revision = "1" * 32
        with self.assertRaisesRegex(JobError, "CLIENT_REVISION_MISMATCH"):
            self.submit(p)
        self.assertEqual(self.jobs.calls, 0)

    def test_client_rejects_rehashed_wrong_revision_and_stale_replies(self):
        q = self.request("prepare", config=self.config, experiment_id="experiment")
        reply = exchange(self.jobs, "tenant-a", q)
        for change in ("revision", "config", "boot"):
            bad = copy.deepcopy(reply)
            if change == "revision":
                bad["result"]["plan"]["client_revision"] = "1" * 32
            elif change == "config":
                bad["result"]["plan"]["validation_plan"]["config"]["epochs"] = 8
            else:
                bad["worker_boot_id"] = "0" * 32
            bad["result"]["plan_sha256"] = digest(bad["result"]["plan"])
            with self.subTest(change=change), self.assertRaises(InvalidReply):
                validate(bad, q, capabilities=self.caps, owner="tenant-a")

    def test_restart_requires_discovery_and_never_replays_uncertain_work(self):
        p = self.prepare()
        first = self.submit(p)
        jid = first["job"]["job_id"]
        old = self.request("observe", job_id=jid)
        self.jobs.close()
        self.jobs = QueuedJobs(self.path, principals={"tenant-a", "tenant-b"})
        with self.assertRaisesRegex(JobError, "STALE_WORKER"):
            self.call(old)
        self.caps = self.discover()
        restored = self.call(self.request("observe", job_id=jid))
        self.assertEqual(restored["job"]["status"], "recovery_required")
        self.assertEqual(restored["job"]["client_revision"], self.revision)
        self.assertEqual(self.submit(p)["job"]["job_id"], jid)
        self.assertEqual(self.jobs.calls, 0)
        with self.assertRaisesRegex(JobError, "STALE_WORKER"):
            self.submit(p, "new-key")

    def test_package_or_capability_change_invalidates_request(self):
        p = self.prepare()
        original = self.jobs.backend.capabilities

        def changed():
            caps = original()
            caps["sha256"] = "1" * 64
            return caps

        with patch.object(self.jobs.backend, "capabilities", changed):
            with self.assertRaisesRegex(JobError, "STALE_CAPABILITIES"):
                self.submit(p)

    def test_offsets_corruption_and_truncation_are_not_verified(self):
        p = self.prepare()
        first = self.submit(p)
        jid = first["job"]["job_id"]
        self.jobs.run_job("tenant-a", jid)
        done = self.submit(p)
        for change in ("bytes", "base64", "offset", "eof"):

            def fetch(aid, offset):
                r = self.call(
                    self.request("artifact", job_id=jid, artifact_id=aid, offset=offset)
                )
                if change == "base64":
                    r[change] = ""
                elif change == "eof":
                    r[change] = not r[change]
                else:
                    r[change] += 1
                return r

            with self.subTest(change=change), self.assertRaises(InvalidObservation):
                verify_artifacts(done, fetch)


if __name__ == "__main__":
    unittest.main()
