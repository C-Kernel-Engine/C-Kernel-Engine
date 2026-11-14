"""Native held-out loss must be current and must not update training state."""

import ctypes as C
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_v8_clef_training_recipe import fixture

ROOT = Path(__file__).resolve().parents[1]


def grouped_ce(logits, counts, labels):
    """Independent FP64 stable log-sum-exp, averaging questions, not options."""
    result = []
    offset = 0
    for count, target in zip(counts, labels):
        row = [float(x) for x in logits[offset : offset + int(count)]]
        maximum = max(row)
        result.append(
            maximum
            - row[int(target)]
            + math.log(math.fsum(math.exp(x - maximum) for x in row))
        )
        offset += int(count)
    assert offset == len(logits)
    return math.fsum(result) / len(result)


class NativeEvaluation(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest("synthetic fixture dependency unavailable: " + str(exc))
        if not (ROOT / "build/libckernel_engine.so").exists():
            self.skipTest("native engine unavailable")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.path, self.pin, _ = fixture(self.folder)
        from version.v8.decision.training_package import load_package

        self.doc, self.raw, self.records = load_package(self.path, self.pin)

    def runtime(self, name):
        from version.v8.decision.train_clef_head import Runtime

        folder = self.folder / name
        folder.mkdir()
        rt = Runtime(folder, self.doc, self.raw, self.records[0]["arrays"])
        self.addCleanup(rt.close)
        return rt

    def state(self, rt):
        from version.v8.decision.train_clef_head import workflow

        weights = rt.snapshot()
        return dict(
            weights={
                k: weights[v].tobytes()
                for k, v in rt.slices.items()
                if k != "__features.hidden"
            },
            moments=workflow._array_export(rt.lib, "optimizer_state").tobytes(),
            gradients=workflow._array_export(rt.lib, "parameter_gradient").tobytes(),
            step=rt.lib.ck_train_get_opt_step(),
            accum_counter=rt.lib.ck_train_get_accum_counter(),
            accum_tokens=rt.lib.ck_train_get_accum_tokens(),
        )

    def checked_eval(self, rt, arrays):
        before = self.state(rt)
        loss, scores = rt.evaluate(arrays)
        after = self.state(rt)
        for key in before:
            self.assertEqual(after[key], before[key], key)
        self.assertAlmostEqual(
            loss,
            grouped_ce(scores, arrays["option_counts"], arrays["labels"]),
            delta=2e-7,
        )
        return loss, scores

    def test_initial_labels_unequal_groups_and_interleaved_training(self):
        rt = self.runtime("evaluation")
        control = self.runtime("training-only")
        a = self.records[2]["arrays"]
        flipped = dict(a, labels=np.array([0, 0], dtype="i4"))
        regrouped = dict(
            self.records[1]["arrays"],
            option_counts=np.array([1, 2], dtype="i4"),
            labels=np.array([0, 1], dtype="i4"),
        )
        initial, scores = self.checked_eval(rt, a)
        self.assertGreater(initial, 0.01)
        other, other_scores = self.checked_eval(rt, flipped)
        np.testing.assert_array_equal(scores, other_scores)
        self.assertGreater(abs(initial - other), 0.01)
        self.checked_eval(rt, regrouped)
        for record in self.records[:2]:
            train_loss = rt.step(record["arrays"], 0.001)
            self.assertEqual(train_loss, control.step(record["arrays"], 0.001))
            for arrays in (a, flipped, regrouped, a):
                value, _ = self.checked_eval(rt, arrays)
                self.assertGreater(abs(value - train_loss), 1e-5)
            self.assertEqual(self.state(rt), self.state(control))

    def test_rejected_evaluation_does_not_publish_or_update_and_recovers(self):
        from version.v8.decision.train_clef_head import ip

        rt = self.runtime("rejected")
        a = self.records[2]["arrays"]
        rt.step(self.records[0]["arrays"], 0.001)
        self.checked_eval(rt, a)
        before = self.state(rt)
        self.assertLess(rt.lib.ck_train_eval_step(None), 0)
        bad = dict(a, labels=np.array([2, 0], dtype="i4"))
        metadata = rt.batch(bad)
        self.assertGreater(
            rt.lib.ck_train_set_batch_with_i32_ex(
                ip(rt.ids), ip(rt.ids), rt.tokens, ip(metadata), metadata.size
            ),
            0,
        )
        sentinel = C.c_float(71)
        self.assertLess(rt.lib.ck_train_eval_step(C.byref(sentinel)), 0)
        self.assertEqual(sentinel.value, 71)
        self.assertEqual(self.state(rt), before)
        self.checked_eval(rt, a)

    def test_report_metrics_match_logits_after_checkpoint_reload(self):
        from version.v8.decision.train_clef_head import run, workflow
        from version.v8.decision.training_package import sha

        validation = self.records[2]["arrays"]
        other = dict(
            self.records[1]["arrays"],
            option_counts=np.array([1, 2], dtype="i4"),
            labels=np.array([0, 1], dtype="i4"),
        )
        np.savez(self.folder / "record-3.npz", **other)
        self.doc["records"].append(
            dict(
                id="record-3",
                split="validation",
                asset=dict(
                    file="record-3.npz", sha256=sha(self.folder / "record-3.npz")
                ),
                question_ids=["action", "mode"],
                candidate_ids=[["read"], ["normal", "verbose"]],
            )
        )
        self.path.write_text(json.dumps(self.doc))
        self.pin = sha(self.path)
        heldout = {"record-2": validation, "record-3": other}
        rt = self.runtime("initial")
        initial = sum(self.checked_eval(rt, a)[0] for a in heldout.values()) / len(
            heldout
        )
        baseline = run(self.path, self.pin, self.folder / "complete", epochs=2)
        self.assertAlmostEqual(
            baseline["metrics"]["validation_loss_before"], initial, delta=2e-7
        )

        def check_after(report):
            scores = {r["id"]: r["logits"] for r in report["records"]}
            expected = sum(
                grouped_ce(scores[key], a["option_counts"], a["labels"])
                for key, a in heldout.items()
            ) / len(heldout)
            self.assertAlmostEqual(
                report["metrics"]["validation_loss_after"], expected, delta=2e-7
            )
            self.assertNotEqual(
                report["metrics"]["validation_loss_after"],
                report["metrics"]["training_losses"][-1],
            )

        check_after(baseline)
        paused = run(
            self.path, self.pin, self.folder / "resumed", epochs=2, stop_after=2
        )
        # Independently evaluate the actual checkpoint weights, without an update.
        weights = np.fromfile(Path(paused["checkpoint"]) / "weight.f32", dtype="<f4")
        workflow._array_import(rt.lib, "weight", weights)
        resumed_start = sum(
            self.checked_eval(rt, a)[0] for a in heldout.values()
        ) / len(heldout)
        result = run(
            self.path,
            self.pin,
            self.folder / "resumed",
            epochs=2,
            resume=paused["checkpoint"],
        )
        self.assertAlmostEqual(
            result["metrics"]["validation_loss_before"], resumed_start, delta=2e-7
        )
        check_after(result)
        self.assertEqual(
            result["metrics"]["validation_loss_after"],
            baseline["metrics"]["validation_loss_after"],
        )
        self.assertEqual(
            result["final_optimizer_state_sha256"],
            baseline["final_optimizer_state_sha256"],
        )
        self.assertEqual(result["records"], baseline["records"])


if __name__ == "__main__":
    unittest.main()
