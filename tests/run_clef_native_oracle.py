"""Development-only same-input and connected Clef comparisons.

The independent publisher head owns expected arithmetic. This tool never feeds
reference tensors into the native text-to-decision runner.
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
from version.v8.decision.replay_clef_features import load_capture
from version.v8.decision.run_native_clef import execute_offline_head
from server.decision.native import NativeHead


def metrics(actual, expected, atol):
    a = np.asarray(actual, dtype="f8")
    b = np.asarray(expected, dtype="f8")
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid comparison arrays")
    d = a - b
    return dict(
        max_abs=float(np.max(np.abs(d))),
        rmse=float(np.sqrt(np.mean(d * d))),
        reference_rms=float(np.sqrt(np.mean(b * b))),
        atol=atol,
        status="PASS" if np.max(np.abs(d)) <= atol else "FAIL",
    )


def compare(release, native_run, native_sha, capture, capture_sha):
    import torch
    from safetensors.torch import load_file

    release = Path(release)
    native_run = Path(native_run)
    identity = verify_release(release)
    if file_sha(native_run / "result.json") != native_sha:
        raise ValueError("native report identity mismatch")
    native = json.loads((native_run / "result.json").read_text())
    oracle, expected_inputs, rows, reference = load_capture(capture, capture_sha)
    for key in ("model", "revision", "request_sha256"):
        if key not in native or native[key] != oracle[key]:
            raise ValueError("different native/oracle experiments")
    if (
        native.get("execution") != "PASS"
        or native.get("backbone_executor") != "cke_generated"
    ):
        raise ValueError("native backbone evidence required")
    if file_sha(native_run / "native-features.npz") != native["feature_sha256"]:
        raise ValueError("native features changed")
    with np.load(native_run / "native-features.npz", allow_pickle=False) as z:
        inputs = {k: z[k].copy() for k in z.files}
    np.testing.assert_array_equal(
        native["input_ids"],
        np.load(Path(capture) / "original_token_ids.npy", allow_pickle=False),
    )
    for name in expected_inputs:
        if name != "hidden":
            np.testing.assert_array_equal(inputs[name], expected_inputs[name])
    if file_sha(native_run / "embedding.npy") != native.get("embedding_sha256"):
        raise ValueError("native embedding identity mismatch")
    np.testing.assert_array_equal(np.load(native_run / "embedding.npy"), rows)
    # Reference records are reconstructed from the independently captured schema;
    # token remapping is already checked against both original token sequences.
    spec = importlib.util.spec_from_file_location(
        "cke_clef_native_oracle_reference", release / "joint_schema_model.py"
    )
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    questions = tuple(
        ref.EncodedQuestion(
            question_id=q["id"],
            question_type=q["type"],
            question_span=tuple(q["span"]),
            option_spans=tuple(map(tuple, q["option_spans"])),
            option_ids=tuple(q["candidate_ids"]),
        )
        for q in oracle["questions"]
    )
    record = ref.EncodedRecord(
        input_ids=tuple(map(int, inputs["input_ids"])),
        questions=questions,
        record_id="native-oracle",
    )
    torch.set_num_threads(8)
    head = ref.JointSchemaHead(
        **json.loads((release / "joint_head_config.json").read_text())
    )
    head.load_state_dict(load_file(release / "joint_head.safetensors"), strict=True)
    head.float().eval()
    with torch.inference_mode():
        out = head(
            torch.from_numpy(inputs["hidden"])[None],
            torch.from_numpy(inputs["input_ids"].astype("int64"))[None],
            torch.ones((1, len(inputs["hidden"])), dtype=torch.int64),
            [record],
            torch.from_numpy(rows),
        )[0]
        same = torch.cat(out).numpy()
    runtime = NativeHead(
        native_run / "head/package.json", native["head_package_sha256"]
    )
    try:
        actual = execute_offline_head(runtime, "question-0")
    finally:
        runtime.close()
    return dict(
        schema="cke.clef.native-oracle.v1",
        native_report_sha256=native_sha,
        capture_sha256=capture_sha,
        reference_source_sha256=identity["joint_schema_model_sha256"],
        torch_version=torch.__version__,
        oracle_script_sha256=file_sha(__file__),
        backbone_reference_transformers=oracle["transformers_version"],
        backbone_reference_schedule=oracle.get("backbone_schedule", "prefill"),
        backbone_reference_attention=oracle.get(
            "attention_implementation", oracle.get("attention")
        ),
        native_logits=actual.tolist(),
        same_input_logits=same.tolist(),
        connected_reference_logits=reference["option_logits"].ravel().tolist(),
        same_input_head=metrics(actual, same, 2e-6),
        connected_logits=metrics(actual, reference["option_logits"].ravel(), 2e-6),
        backbone_exact_storage=metrics(inputs["hidden"], expected_inputs["hidden"], 0),
        note="Connected 2e-6 check retains the strict logit baseline; no broader model tolerance is inferred. Zero-tolerance feature comparison is exact-storage evidence.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "native-run", "capture", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--native-sha256", required=True)
    p.add_argument("--capture-sha256", required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("output already exists")
    r = compare(a.release, a.native_run, a.native_sha256, a.capture, a.capture_sha256)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))
    if (
        r["same_input_head"]["status"] != "PASS"
        or r["connected_logits"]["status"] != "PASS"
    ):
        raise SystemExit(1)
