"""Capture released Karpo's text marker rows and FP32 readout with PyTorch."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

CHECKPOINT_REVISION = "fae75d7bf93fc7e5c37b1ff631b64e9474e4d797"
PUBLISHER_REVISION = "e5826d41e36c58e9e0829e4329c66c955903ddcd"
ARTIFACT_SHA256 = {
    "model-00001-of-00001.safetensors": "656bc586f18e9c26df29033ebd50036b33004de82b2c61e4a670124e88d3143f",
    "readout.safetensors": "e290da9edde55094aff5aa26c7b6d0f5a65b08eac590570af35a2c6fb0d6983e",
    "config.json": "fc3cde8d88ba8f3d23cb7ceb2fbc1e40a693464253f3e0d004772b277ccc85cc",
    "decision_config.json": "f2c545f95e3472c287ed0e15b4f74e1548ca168957ee14e355e16ccd236e8614",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer.json": "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
    "tokenizer_config.json": "44d30b6b485384681a21b3de5073e58439aab66cb650f30f2d3d1073da429af0",
    "processor_config.json": "b4b905eee2d1cf96dcbdac1c68b35113c8151958a4c1f76667242b01168572cd",
}


def digest(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def capture(checkpoint, publisher, output):
    import torch

    command = ["git", "-C", str(publisher)]
    revision = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    if revision != PUBLISHER_REVISION or subprocess.check_output([*command, "status", "--porcelain", "-uno"]):
        raise ValueError("publisher checkout does not match the pinned clean source")
    for name, sha in ARTIFACT_SHA256.items():
        if digest(checkpoint / name) != sha:
            raise ValueError(f"checkpoint hash mismatch: {name}")
    sys.path.insert(0, str(publisher.resolve() / "src"))
    from karpo_decision.checkpoint import load
    from karpo_decision.encoding import Encoder
    from karpo_decision.families.qwen.reference import reference_hidden
    from karpo_decision.templates import template

    torch.set_num_threads(4)
    model = load(checkpoint, device="cpu")
    cfg = model.config
    if cfg.attention != "bidirectional" or set(cfg.temperatures.values()) != {1.0}:
        raise ValueError("unexpected release attention or option temperature")
    encoder = Encoder(model.processor, cfg.codes, template(cfg.template), cfg.marker)
    request = {
        "state": "The store accepts returns within 30 days. This purchase was made 45 days ago.",
        "questions": {
            "eligible": {"type": "noul", "instructions": "Is the purchase eligible for return?"},
            "action": {"type": "choice", "instructions": "How should the agent respond?",
                       "criteria": {"refund": "Refund it", "decline": "Decline the return",
                                    "escalate": "Ask a supervisor"}},
        },
    }
    part, = encoder.plan(request, 4096)
    with torch.inference_mode():
        hidden = reference_hidden(model.backbone.eval(), [part], cfg.attention, encoder.pad_id)
        scores = model.readout.eval()(hidden)
        probabilities = torch.zeros_like(scores)
        for row, count in enumerate(part.counts):
            probabilities[row, :count] = scores[row, :count].softmax(0)
    output.mkdir(parents=True, exist_ok=False)
    data = output / "markers.npz"
    np.savez_compressed(data, ids=part.input_ids.numpy(), markers=part.markers.numpy(),
                        counts=np.asarray(part.counts, dtype=np.int32), hidden=hidden.float().numpy(),
                        scores=scores.numpy(), probabilities=probabilities.numpy())
    report = {"checkpoint_revision": CHECKPOINT_REVISION, "publisher_revision": PUBLISHER_REVISION,
              "torch": torch.__version__, "markers_sha256": digest(data), "question_ids": list(part.keys),
              "counts": list(part.counts), "request": request}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "publisher", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(args.checkpoint, args.publisher, args.output), indent=2))


if __name__ == "__main__":
    main()
