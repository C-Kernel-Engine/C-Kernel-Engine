"""Offline native-feature Clef head training and generated inference reload.

The backbone stays frozen. This uses the existing v8 training Runtime, backward
synthesis and native AdamW; it is not an application job or a new training stack.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from version.v8.decision.run_native_clef import (
    file_sha,
    load_float_weights,
    execute_offline_head,
)
from version.v8.decision.capture_clef_features import verify_release
from version.v8.decision.training_package import INPUTS, raw_from_effective, require
from version.v8.decision.train_clef_head import Runtime, workflow
from version.v8.decision.package_clef_head import build_package
from server.decision.native import NativeHead


def label_indices(results, labels):
    if not isinstance(labels, dict) or set(labels) != {
        r["question_id"] for r in results
    }:
        raise ValueError("exactly one label per question is required")
    if any(labels[r["question_id"]] not in r["candidate_ids"] for r in results):
        raise ValueError("label must be a declared candidate")
    return np.array(
        [r["candidate_ids"].index(labels[r["question_id"]]) for r in results],
        dtype="i4",
    )


def train(
    release,
    native_run,
    report_sha,
    labels,
    output,
    *,
    steps=1,
    lr=1e-5,
    workspace_mib=4096,
):
    require(
        type(steps) is int and 1 <= steps <= 40,
        "bounded diagnostic supports 1..40 updates",
    )
    require(
        type(lr) in (int, float) and np.isfinite(lr) and 0 < lr <= 0.01,
        "invalid learning rate",
    )
    require(
        type(workspace_mib) is int and workspace_mib in (2048, 4096),
        "select explicit 2048 or 4096 MiB budget",
    )
    release = Path(release)
    native_run = Path(native_run)
    output = Path(output)
    require(not output.exists(), "output already exists")
    require(
        file_sha(native_run / "result.json") == report_sha,
        "native report identity mismatch",
    )
    report = json.loads((native_run / "result.json").read_text())
    identity = verify_release(release)
    require(
        report.get("schema") == "cke.clef.native-diagnostic.v1"
        and report.get("execution") == "PASS"
        and report.get("backbone_executor") == "cke_generated"
        and report.get("head_executor") == "cke_generated",
        "native execution evidence required",
    )
    require(
        report["model"] == identity["model"]
        and report["revision"] == identity["revision"],
        "native model identity differs",
    )
    require(
        file_sha(native_run / "native-features.npz") == report["feature_sha256"],
        "native feature identity mismatch",
    )
    with np.load(native_run / "native-features.npz", allow_pickle=False) as z:
        inputs = {k: z[k].copy() for k in z.files}
    require(set(inputs) == set(INPUTS), "native input inventory differs")
    ids = np.asarray(report["input_ids"], dtype="i8")
    unique = np.unique(ids)
    require(
        np.array_equal(inputs["input_ids"], np.searchsorted(unique, ids)),
        "head-local token mapping differs",
    )
    raw = load_float_weights(release / "joint_head.safetensors")
    require(
        file_sha(native_run / "embedding.npy") == report.get("embedding_sha256"),
        "native embedding identity mismatch",
    )
    raw["output_embedding.weight"] = np.load(
        native_run / "embedding.npy", allow_pickle=False
    )
    require(
        raw["output_embedding.weight"].shape == (len(unique), inputs["hidden"].shape[1])
        and raw["output_embedding.weight"].dtype == np.float32
        and np.isfinite(raw["output_embedding.weight"]).all(),
        "invalid native embedding rows",
    )
    x = dict(inputs, labels=label_indices(report["results"], labels))
    config = json.loads((release / "joint_head_config.json").read_text())
    geometry = dict(
        tokens=len(ids),
        channels=config["hidden_size"],
        width=config["width"],
        heads=config["heads"],
        feedforward=config["feedforward"],
        vocabulary=len(unique),
        questions=len(x["option_counts"]),
        options=len(x["option_spans"]),
    )
    require(
        inputs["hidden"].shape == (len(ids), config["hidden_size"]),
        "native feature geometry differs",
    )
    output.mkdir(parents=True)
    started = time.perf_counter()
    rt = Runtime(
        output,
        dict(geometry=geometry),
        raw,
        x,
        workspace_limit_bytes=workspace_mib * 1024**2,
    )
    try:
        initial = rt.snapshot()
        before, _ = rt.evaluate(x)
        np.testing.assert_array_equal(initial, rt.snapshot())
        losses = [rt.step(x, lr) for _ in range(steps)]
        after, scores = rt.evaluate(x)
        final = rt.snapshot()
        np.save(
            output / "parameter-gradients.npy",
            workflow._array_export(rt.lib, "parameter_gradient"),
        )
        changed = [
            n for n, s in rt.slices.items() if not np.array_equal(initial[s], final[s])
        ]
        for name in ("__features.hidden", "output_embedding.weight"):
            np.testing.assert_array_equal(
                initial[rt.slices[name]], final[rt.slices[name]]
            )
        effective = {
            n: final[s].reshape(rt.weights[n].shape).copy()
            for n, s in rt.slices.items()
            if n != "__features.hidden"
        }
        updated = raw_from_effective(raw, effective)
        np.savez(output / "trained_head.npz", **updated)
        registrations = [
            dict(
                id=f"question-{i}",
                inputs=inputs,
                question_index=i,
                candidate_ids=r["candidate_ids"],
            )
            for i, r in enumerate(report["results"])
        ]
        digest = build_package(
            output / "inference",
            updated,
            registrations,
            dict(id="clef-native-trained", revision=identity["revision"]),
            heads=config["heads"],
            provenance=dict(
                native_report_sha256=report_sha,
                features="cke_generated",
                backbone_training="frozen",
            ),
        )
        head = NativeHead(output / "inference/package.json", digest)
        try:
            reloaded = execute_offline_head(head, "question-0")
        finally:
            head.close()
        np.testing.assert_allclose(reloaded, scores, atol=2e-6, rtol=0)
        result = dict(
            schema="cke.clef.native-training-diagnostic.v1",
            execution="PASS",
            native_report_sha256=report_sha,
            backbone_training="FROZEN",
            before_loss=before,
            after_loss=after,
            step_losses=losses,
            steps=steps,
            learning_rate=lr,
            updated_parameter_tensors=changed,
            trainable_elements=rt.summary["parameter_gradient_snapshot_floats"],
            workspace_bytes=rt.summary["memory_total_floats"] * 4,
            workspace_budget_bytes=workspace_mib * 1024**2,
            inference_package_sha256=digest,
            trained_head_sha256=file_sha(output / "trained_head.npz"),
            gradient_sha256=file_sha(output / "parameter-gradients.npy"),
            reload_max_abs=float(np.max(np.abs(reloaded - scores))),
            numerical_gradient_parity="NOT_TESTED",
            connected_backbone_parity=report["numerical_parity"],
            held_out_accuracy="NOT_TESTED",
            seconds=time.perf_counter() - started,
        )
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        rt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "native-run", "labels", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--workspace-mib", type=int, choices=[2048, 4096], default=4096)
    a = p.parse_args()
    print(
        json.dumps(
            train(
                a.release,
                a.native_run,
                a.report_sha256,
                json.loads(a.labels.read_text()),
                a.output,
                steps=a.steps,
                lr=a.lr,
                workspace_mib=a.workspace_mib,
            ),
            indent=2,
        )
    )
