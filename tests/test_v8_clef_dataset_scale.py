"""Larger bounded data admission, identity-based metrics, native long-run smoke."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tests.test_v8_clef_training_recipe import fixture
from version.v8.decision.training_package import load_package, sha
from version.v8.decision.training_metrics import classification_metrics


class ScaleContracts(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest("fixture construction requires PyTorch: " + str(exc))

    def test_admin_wall_budget(self):
        from server.training.jobs import Jobs, JobError

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(JobError):
                Jobs(Path(tmp) / "bad", principals={"tester"}, wall_seconds=True)
            job = Jobs(Path(tmp) / "good", principals={"tester"}, wall_seconds=3600)
            try:
                self.assertEqual(
                    job.capabilities("tester")["limits"]["wall_seconds"], 3600
                )
            finally:
                job.close()

    def test_sparse_confusion_memory_bound(self):
        names = [f"class{i}" for i in range(300)]
        record = dict(
            id="r",
            split="validation",
            question_ids=["q"],
            candidate_ids=[names],
            arrays=dict(labels=np.array([0]), option_counts=np.array([300])),
        )
        result = classification_metrics([record], {"r": (0.0, np.arange(300.0))})[
            "validation"
        ]["q"]
        self.assertIsNone(result["confusion_matrix"])
        self.assertEqual(len(result["confusion_entries"]), 1)
        self.assertEqual(result["count"], 1)

    def test_stable_grouped_probabilities(self):
        from version.v8.decision.training_metrics import grouped_probabilities

        np.testing.assert_allclose(
            grouped_probabilities([10000, 10000, -10000], [2, 1]), [0.5, 0.5, 1.0]
        )
        for logits, counts in (([1], [2]), ([float("nan")], [1]), ([1], [0])):
            with self.assertRaises(ValueError):
                grouped_probabilities(logits, counts)

    def test_metrics_follow_candidate_identity_not_position(self):
        rows = []
        evaluated = {}
        for i, (names, label, scores) in enumerate(
            (
                (["a", "b"], 0, [2.0, 1.0]),
                (["b", "a"], 0, [1.0, 2.0]),
                (["a", "b"], 1, [1.0, 2.0]),
            )
        ):
            rows.append(
                dict(
                    id=str(i),
                    split="validation",
                    question_ids=["sentiment"],
                    candidate_ids=[names],
                    arrays=dict(labels=np.array([label]), option_counts=np.array([2])),
                )
            )
            evaluated[str(i)] = (0.0, np.array(scores))
        m = classification_metrics(rows, evaluated)["validation"]["sentiment"]
        self.assertEqual(m["confusion_matrix"], [[1, 0], [1, 1]])
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        self.assertAlmostEqual(m["macro_f1"], 2 / 3)
        evaluated["0"] = (0.0, np.array([float("nan"), 0]))
        with self.assertRaises(ValueError):
            classification_metrics(rows, evaluated)

    def test_large_package_admission_and_application_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, digest, _ = fixture(Path(tmp))
            doc = json.loads(path.read_text())
            base = doc["records"][0]
            doc["records"] = [
                dict(copy.deepcopy(base), id=f"r{i}") for i in range(40)
            ] + [doc["records"][-1]]
            path.write_text(json.dumps(doc))
            digest = sha(path)
            self.assertEqual(len(load_package(path, digest)[2]), 41)
            oversized_doc = copy.deepcopy(doc)
            oversized_doc["records"] = [
                dict(copy.deepcopy(base), id=f"big{i}") for i in range(4097)
            ]
            path.write_text(json.dumps(oversized_doc))
            with self.assertRaisesRegex(ValueError, "record count"):
                load_package(path, sha(path))
            path.write_text(json.dumps(doc))

            leaked = copy.deepcopy(doc)
            leaked["records"][-1]["asset"] = copy.deepcopy(base["asset"])
            path.write_text(json.dumps(leaked))
            with self.assertRaisesRegex(ValueError, "crosses"):
                load_package(path, sha(path))
            path.write_text(json.dumps(doc))
            from server.training.contract import Backend
            from server.training.consumer import validate

            b = Backend(clef_packages={"demo": dict(path=str(path), sha256=digest)})
            q = dict(
                schema="cke.training.v1", request_id="discover", operation="discover"
            )
            reply = b.exchange(q)
            validate(reply, q)
            caps = reply["capabilities"]
            self.assertEqual(
                caps["recipes"]["clef_head_fp32_v1"]["limits"]["steps"], 100000
            )
            from server.training.clef import validate as admit

            cfg = dict(
                recipe="clef_head_fp32_v1",
                package_id="demo",
                package_sha256=digest,
                dtype="fp32",
                optimizer="adamw",
                method="head_only",
                epochs=1000,
                learning_rate=0.001,
            )
            self.assertEqual(admit(cfg, caps["clef_packages"]), [])
            oversized = copy.deepcopy(caps["clef_packages"])
            oversized["demo"]["splits"]["train"] = 101
            self.assertIn("STEP_BUDGET_EXCEEDED", str(admit(cfg, oversized)))

    def test_native_longer_tokens_more_records_and_steps(self):
        if not (
            Path(__file__).resolve().parents[1] / "build/libckernel_engine.so"
        ).exists():
            self.skipTest("native engine unavailable")
        from version.v8.decision.train_clef_head import run

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path, _, _ = fixture(folder)
            doc = json.loads(path.read_text())
            doc["geometry"]["tokens"] = 96
            for i in range(3):
                file = folder / f"record-{i}.npz"
                with np.load(file) as f:
                    arrays = {k: f[k] for k in f.files}
                arrays["hidden"] = (
                    np.random.default_rng(i).normal(0, 0.2, (96, 8)).astype("f4")
                )
                arrays["input_ids"] = np.ones(96, dtype="i4")
                arrays["global_span"] = np.array([[95, 96]], dtype="i4")
                np.savez(file, **arrays)
                doc["records"][i]["asset"]["sha256"] = sha(file)
            base = doc["records"][0]
            doc["records"] = [
                dict(copy.deepcopy(base), id=f"r{i}") for i in range(33)
            ] + [doc["records"][-1]]
            path.write_text(json.dumps(doc))
            report = run(path, sha(path), folder / "run", epochs=41)
            self.assertEqual(report["completed_steps"], 1353)
            self.assertEqual(len(report["inference_export_record_ids"]), 32)
            self.assertEqual(
                report["metrics"]["classification"]["validation"]["action"]["count"], 1
            )
            self.assertEqual(
                report["metrics"]["validation_history_segment"][1]["step"], 1000
            )
            self.assertTrue((folder / "run/checkpoint-1000/checkpoint.json").exists())
            self.assertEqual(report["quality"], "NOT_TESTED")
            print(
                json.dumps(
                    dict(
                        experiment="clef_scale_synthetic",
                        records=34,
                        tokens=96,
                        epochs=41,
                        updates=report["completed_steps"],
                        exported_records=len(report["inference_export_record_ids"]),
                        elapsed_seconds=report["metrics"]["elapsed_seconds"],
                        peak_process_rss_bytes=report["metrics"][
                            "peak_process_rss_bytes"
                        ],
                        runtime=report["runtime"],
                        quality="NOT_TESTED",
                    ),
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
