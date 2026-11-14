"""Independent publisher-head gradient/AdamW comparison for native features.

Development only. The production trainer does not import this module or PyTorch.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from version.v8.decision.capture_clef_features import file_sha, verify_release
from version.v8.decision.package_clef_head import effective_weights
from version.v8.decision.train_native_clef import label_indices


def compare(release, native, training, labels, report_sha, training_sha):
    import torch
    from safetensors.torch import load_file

    release, native, training = map(Path, (release, native, training))
    identity = verify_release(release)
    if (
        file_sha(native / "result.json") != report_sha
        or file_sha(training / "result.json") != training_sha
    ):
        raise ValueError("report identity mismatch")
    run = json.loads((native / "result.json").read_text())
    trained = json.loads((training / "result.json").read_text())
    if trained["native_report_sha256"] != report_sha or trained["execution"] != "PASS":
        raise ValueError("training belongs to a different native experiment")
    for folder, name, digest in (
        (native, "native-features.npz", run["feature_sha256"]),
        (native, "embedding.npy", run["embedding_sha256"]),
        (training, "trained_head.npz", trained["trained_head_sha256"]),
        (training, "parameter-gradients.npy", trained["gradient_sha256"]),
    ):
        if file_sha(folder / name) != digest:
            raise ValueError("artifact changed: " + name)
    with np.load(native / "native-features.npz", allow_pickle=False) as z:
        x = {k: z[k].copy() for k in z.files}
    rows = np.load(native / "embedding.npy", allow_pickle=False)
    spec = importlib.util.spec_from_file_location(
        "clef_training_oracle", release / "joint_schema_model.py"
    )
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    # Reconstruct identities and spans from the generated request package, not a
    # new tokenization that could accidentally compare a different experiment.
    questions = tuple(
        ref.EncodedQuestion(
            question_id=q["question_id"],
            question_type=int(x["question_type_ids"][i]),
            question_span=tuple(map(int, x["question_spans"][i])),
            option_spans=tuple(
                map(
                    tuple,
                    x["option_spans"][
                        sum(x["option_counts"][:i]) : sum(x["option_counts"][: i + 1])
                    ],
                )
            ),
            option_ids=tuple(q["candidate_ids"]),
        )
        for i, q in enumerate(run["results"])
    )
    record = ref.EncodedRecord(
        input_ids=tuple(map(int, x["input_ids"])),
        questions=questions,
        record_id="native-training",
    )
    torch.set_num_threads(8)
    head = ref.JointSchemaHead(
        **json.loads((release / "joint_head_config.json").read_text())
    )
    head.load_state_dict(load_file(release / "joint_head.safetensors"), strict=True)
    head.float().eval()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=trained["learning_rate"],
        eps=1e-4,
        weight_decay=0.01,
        foreach=False,
    )
    targets = label_indices(run["results"], labels)
    losses = []
    for _ in range(trained["steps"]):
        optimizer.zero_grad(set_to_none=True)
        scores = head(
            torch.from_numpy(x["hidden"])[None],
            torch.from_numpy(x["input_ids"].astype("int64"))[None],
            torch.ones((1, len(x["hidden"])), dtype=torch.int64),
            [record],
            torch.from_numpy(rows),
        )[0]
        loss = torch.stack(
            [
                torch.nn.functional.cross_entropy(s[None], torch.tensor([int(y)]))
                for s, y in zip(scores, targets)
            ]
        ).mean()
        loss.backward()
        losses.append(float(loss.detach()))
        optimizer.step()
    grads = {k: v.grad.numpy().copy() for k, v in head.named_parameters()}
    grads["output_embedding.weight"] = np.zeros_like(rows)
    graph = json.loads((native / "head/circuit.json").read_text())
    effective = effective_weights(graph, grads)
    summary = json.loads((training / "runtime.json").read_text())["summary"]
    actual = np.load(training / "parameter-gradients.npy", allow_pickle=False)
    offset = 0
    comparisons = []
    for name in summary["parameter_gradient_order"]:
        expected = effective[name].ravel()
        got = actual[offset : offset + len(expected)]
        offset += len(expected)
        if not np.isfinite(got).all() or not np.isfinite(expected).all():
            raise ValueError("nonfinite gradient comparison")
        comparisons.append(
            dict(parameter=name, max_abs=float(np.abs(got - expected).max()))
        )
    if offset != len(actual):
        raise ValueError("gradient inventory differs")
    with np.load(training / "trained_head.npz", allow_pickle=False) as updated:
        updates = [
            dict(
                parameter=n,
                max_abs=float(np.abs(p.detach().numpy() - updated[n]).max()),
            )
            for n, p in head.named_parameters()
        ]
    if not all(np.isfinite(u["max_abs"]) for u in updates):
        raise ValueError("nonfinite updated-weight comparison")
    worst = max(comparisons, key=lambda r: r["max_abs"])
    wu = max(updates, key=lambda r: r["max_abs"])
    return dict(
        schema="cke.clef.native-training-oracle.v1",
        native_report_sha256=report_sha,
        training_report_sha256=training_sha,
        torch_version=torch.__version__,
        reference_source_sha256=identity["joint_schema_model_sha256"],
        oracle_script_sha256=file_sha(__file__),
        steps=trained["steps"],
        reference_losses=losses,
        gradient_max=worst,
        updated_weight_max=wu,
        gradient_atol=3e-5,
        update_atol=3e-6,
        gradient_status="PASS" if worst["max_abs"] <= 3e-5 else "FAIL",
        update_status="PASS" if wu["max_abs"] <= 3e-6 else "FAIL",
        parameter_tensors=len(comparisons),
        parameter_elements=offset,
        comparisons=comparisons,
        updates=updates,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "native-run", "training", "labels", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--native-sha256", required=True)
    p.add_argument("--training-sha256", required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("output already exists")
    r = compare(
        a.release,
        a.native_run,
        a.training,
        json.loads(a.labels.read_text()),
        a.native_sha256,
        a.training_sha256,
    )
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in r.items() if k not in ("comparisons", "updates")},
            indent=2,
        )
    )
    if r["gradient_status"] != "PASS" or r["update_status"] != "PASS":
        raise SystemExit(1)
