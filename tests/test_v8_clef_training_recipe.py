"""Public FP32 supplied-feature recipe; independent test data and native reload."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fixture(folder):
    from tests.test_v8_clef_flash_complete_head import synthetic_weights
    from version.v8.decision.training_package import sha

    raw = synthetic_weights()
    for i in range(2):
        stem = f"evidence_layers.{i}.attention."
        for leaf in ("weight", "bias"):
            raw[stem + "in_proj_" + leaf] = np.concatenate(
                [raw.pop(stem + axis + "_proj." + leaf) for axis in ("q", "k", "v")]
            )
    np.savez(folder / "weights.npz", **raw)
    g = dict(
        tokens=6,
        channels=8,
        width=8,
        heads=2,
        feedforward=16,
        vocabulary=16,
        questions=2,
        options=3,
    )
    records = []
    for i, split in enumerate(("train", "train", "validation")):
        arrays = dict(
            hidden=np.random.default_rng(10 + i).normal(0, 0.2, (6, 8)).astype("f4"),
            input_ids=np.arange(1, 7, dtype="i4"),
            question_spans=np.array([[0, 2], [2, 4]], dtype="i4"),
            option_spans=np.array([[1, 3], [2, 5], [4, 6]], dtype="i4"),
            global_span=np.array([[5, 6]], dtype="i4"),
            option_counts=np.array([2, 1], dtype="i4"),
            question_type_ids=np.array([0, 2], dtype="i4"),
            labels=np.array([1 if i != 1 else 0, 0], dtype="i4"),
        )
        file = f"record-{i}.npz"
        np.savez(folder / file, **arrays)
        records.append(
            dict(
                id=f"record-{i}",
                split=split,
                asset=dict(file=file, sha256=sha(folder / file)),
                question_ids=["action", "mode"],
                candidate_ids=[["read", "clarify"], ["normal"]],
            )
        )
    doc = dict(
        schema="cke.clef.training-package.v1",
        model=dict(
            id="synthetic-clef",
            revision="1" * 40,
            backbone_id="supplied-test-backbone",
            backbone_revision="2" * 40,
            tokenizer_revision="3" * 40,
            feature_provenance="supplied",
        ),
        geometry=g,
        weights=dict(file="weights.npz", sha256=sha(folder / "weights.npz")),
        records=records,
    )
    path = folder / "package.json"
    path.write_text(json.dumps(doc))
    return path, sha(path), raw


class PackageContracts(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest("test fixture dependency unavailable: " + str(exc))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.path, self.digest, self.raw = fixture(self.folder)

    def test_round_trip_and_rejections(self):
        from version.v8.decision.training_package import load_package, sha

        doc, raw, records = load_package(self.path, self.digest)
        self.assertEqual(len(records), 3)
        with self.assertRaisesRegex(ValueError, "identity"):
            load_package(self.path, "0" * 64)
        original = (self.folder / "record-0.npz").read_bytes()
        for name, value in [
            ("input_ids", np.full(6, -1, dtype="i4")),
            ("labels", np.array([2, 0], dtype="i4")),
            ("hidden", np.full((6, 8), np.nan, dtype="f4")),
            ("global_span", np.array([[0, 1]], dtype="i4")),
            ("option_counts", np.array([1, 1], dtype="i4")),
        ]:
            with self.subTest(name=name):
                arrays = dict(records[0]["arrays"])
                arrays[name] = value
                np.savez(self.folder / "record-0.npz", **arrays)
                bad = json.loads(json.dumps(doc))
                bad["records"][0]["asset"]["sha256"] = sha(self.folder / "record-0.npz")
                self.path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_package(self.path, sha(self.path))
        (self.folder / "record-0.npz").write_bytes(original)
        self.path.write_text(json.dumps(doc))
        load_package(self.path, sha(self.path))

    def test_registered_plan_rejects_other_precision_and_changed_assets(self):
        from server.training.contract import Backend
        from server.training.clef import RECIPE

        b = Backend(
            clef_packages={"toy": dict(path=str(self.path), sha256=self.digest)}
        )
        caps = b.capabilities()
        self.assertIn(RECIPE, caps["recipes"])
        cfg = dict(
            recipe=RECIPE,
            package_id="toy",
            package_sha256=self.digest,
            dtype="fp32",
            optimizer="adamw",
            method="head_only",
            epochs=2,
            learning_rate=0.001,
        )
        request = dict(
            schema="cke.training.v1",
            request_id="test",
            operation="validate",
            worker_boot_id=b.boot,
            capabilities_sha256=caps["sha256"],
            config=cfg,
        )
        self.assertTrue(b.exchange(request)["ok"])
        self.assertFalse(b.exchange(request)["submitted"])
        from server.training.consumer import validate

        discover = dict(
            schema="cke.training.v1", request_id="discover", operation="discover"
        )
        validate(b.exchange(discover), discover)
        validate(b.exchange(request), request, caps)

        for field, value in [
            ("dtype", "bf16"),
            ("method", "lora"),
            ("package_sha256", "0" * 64),
        ]:
            bad = dict(request, config=dict(cfg, **{field: value}))
            self.assertFalse(b.exchange(bad)["ok"])
        with open(self.folder / "record-0.npz", "ab") as f:
            f.write(b"changed")
        with self.assertRaises(ValueError):
            b.capabilities()


class NativeRecipe(PackageContracts):
    def test_registered_job_and_independent_artifact_client(self):
        self._registered_job()

    def test_opaque_revision_native_job_and_export(self):
        self._registered_job(client_revision="00" + "f" * 30)

    def _registered_job(self, client_revision=None):
        if os.environ.get("CKE_RUN_CLEF_JOB") != "1":
            self.skipTest("real clean-tree job execution is opt-in")
        from server.training.jobs import Jobs
        from server.training.job_consumer import observe, verify_artifacts

        cfg = dict(
            recipe="clef_head_fp32_v1",
            package_id="toy",
            package_sha256=self.digest,
            dtype="fp32",
            optimizer="adamw",
            method="head_only",
            epochs=2,
            learning_rate=0.001,
        )
        jobs = Jobs(
            self.folder / "jobs",
            principals={"tester"},
            python=sys.executable,
            clef_packages={"toy": dict(path=str(self.path), sha256=self.digest)},
        )
        try:
            p = jobs.prepare(
                "tester",
                cfg,
                "synthetic-decision",
                None if client_revision else 1,
                client_revision=client_revision,
            )
            first = jobs.submit(
                "tester",
                p["plan_sha256"],
                "same-request",
                start=False,
                client_revision=client_revision,
            )
            job_id = first["job"]["job_id"]
            self.assertEqual(
                jobs.submit(
                    "tester",
                    p["plan_sha256"],
                    "same-request",
                    start=False,
                    client_revision=client_revision,
                )["job"]["job_id"],
                job_id,
            )
            jobs.run_job("tester", job_id)
            result = jobs.observe("tester", job_id)
            self.assertEqual(result["job"]["status"], "succeeded", result)
            observe(
                result,
                worker_boot_id=jobs.backend.boot,
                owner="tester",
                job_id=job_id,
                plan_sha256=p["plan_sha256"],
                experiment_id="synthetic-decision",
                draft_revision=None if client_revision else 1,
                client_revision=client_revision,
                last=first,
            )
            verified = verify_artifacts(
                result, lambda aid, offset: jobs.artifact("tester", job_id, aid, offset)
            )
            self.assertEqual(len(verified), len(result["job"]["artifacts"]))
            self.assertTrue(
                {"trained_head", "inference_package", "checkpoint_manifest"}
                <= {a["type"] for a in result["job"]["artifacts"]}
            )
            self.assertFalse(result["cancel_supported"])
        finally:
            jobs.close()

    def test_independent_pytorch_updates_on_distinct_records(self):
        source = os.environ.get("CKE_CLEF_FLASH_REFERENCE_SOURCE")
        if not source:
            self.skipTest("pinned independent head source unavailable: NOT_TESTED")
        if not (ROOT / "build/libckernel_engine.so").exists():
            self.skipTest("native engine unavailable")
        import importlib.util
        import torch
        from types import SimpleNamespace as NS
        from version.v8.decision.training_package import load_package
        from version.v8.decision.train_clef_head import run

        identity = json.loads(
            (ROOT / "tests/fixtures/decision/clef_flash/identity.json").read_text()
        )
        self.assertEqual(
            hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            identity["joint_schema_model_sha256"],
        )
        spec = importlib.util.spec_from_file_location("recipe_oracle", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.set_num_threads(1)
        head = module.JointSchemaHead(
            hidden_size=8,
            width=8,
            routing_layers=2,
            layers=4,
            heads=2,
            feedforward=16,
            dropout=0,
        )
        head.load_state_dict(
            {
                k: torch.tensor(self.raw[k]).reshape(v.shape)
                for k, v in head.state_dict().items()
            }
        )
        head.eval()
        opt = torch.optim.AdamW(
            head.parameters(), lr=0.001, eps=1e-4, weight_decay=0.01, foreach=False
        )
        _, _, records = load_package(self.path, self.digest)
        for _ in range(2):
            for record in records:
                if record["split"] != "train":
                    continue
                a = record["arrays"]
                offset = 0
                questions = []
                for i, count in enumerate(a["option_counts"]):
                    questions.append(
                        NS(
                            question_span=tuple(a["question_spans"][i]),
                            question_type=int(a["question_type_ids"][i]),
                            option_spans=[
                                tuple(x)
                                for x in a["option_spans"][offset : offset + count]
                            ],
                        )
                    )
                    offset += count
                opt.zero_grad()
                scores = head(
                    torch.tensor(a["hidden"])[None],
                    torch.tensor(a["input_ids"], dtype=torch.long)[None],
                    torch.ones(1, 6, dtype=torch.long),
                    [NS(questions=questions)],
                    torch.tensor(self.raw["output_embedding.weight"]),
                )[0]
                loss = (
                    sum(
                        torch.nn.functional.cross_entropy(
                            x[None], torch.tensor([int(y)])
                        )
                        for x, y in zip(scores, a["labels"])
                    )
                    / 2
                )
                loss.backward()
                opt.step()
        report = run(self.path, self.digest, self.folder / "oracle", epochs=2)
        with np.load(self.folder / "oracle/trained_head.npz") as trained:
            for name, value in head.state_dict().items():
                np.testing.assert_allclose(
                    trained[name].reshape(value.shape),
                    value.numpy(),
                    atol=3e-5,
                    rtol=3e-4,
                    err_msg=name,
                )
            np.testing.assert_array_equal(
                trained["output_embedding.weight"], self.raw["output_embedding.weight"]
            )
        self.assertEqual(report["completed_steps"], 4)

    def test_native_training_export_and_fresh_resume(self):
        if not (ROOT / "build/libckernel_engine.so").exists():
            self.skipTest("native engine unavailable")
        from version.v8.decision.train_clef_head import run

        baseline = run(self.path, self.digest, self.folder / "baseline", epochs=2)
        self.assertTrue(baseline["passed"])
        self.assertEqual(baseline["numerical_parity"], "NOT_TESTED")
        paused = run(
            self.path, self.digest, self.folder / "resumed", epochs=2, stop_after=2
        )
        self.assertEqual(paused["status"], "CHECKPOINTED")
        args = [
            str(ROOT / "version/v8/decision/train_clef_head.py"),
            "--package",
            str(self.path),
            "--package-sha256",
            self.digest,
            "--run-dir",
            str(self.folder / "resumed"),
            "--epochs",
            "2",
            "--resume",
            paused["checkpoint"],
            "--json-out",
            str(self.folder / "resume-result.json"),
        ]
        # Production training and export must not import a PyTorch executor.
        code = (
            'import sys,runpy;sys.modules["torch"]=None;sys.modules["transformers"]=None;sys.argv='
            + repr(args)
            + ';runpy.run_path(sys.argv[0],run_name="__main__")'
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            timeout=180,
            capture_output=True,
            text=True,
        )
        result = json.loads((self.folder / "resume-result.json").read_text())
        self.assertEqual(
            result["final_optimizer_state_sha256"],
            baseline["final_optimizer_state_sha256"],
        )
        with np.load(self.folder / "baseline/trained_head.npz") as a, np.load(
            self.folder / "resumed/trained_head.npz"
        ) as b:
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(result["records"], baseline["records"])


if __name__ == "__main__":
    unittest.main()
