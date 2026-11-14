"""FP32 supplied-feature Clef recipe for the existing v8 native training stack.

Python loads records, invokes generated steps and packages artifacts. Backward and
AdamW are generated C; no PyTorch dependency or Python parameter update exists.
"""

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "version/v8/scripts"))
import run_training_workflow_v8 as workflow
import lower_ir2_backward_v7 as backward
import generate_train_layout_v7 as planner
import codegen_train_runtime_v7 as codegen
from version.v8.decision.build_clef_training_ir import (
    build_training_ir,
    training_registry,
)
from version.v8.decision.package_clef_head import build_package
from version.v8.decision.training_package import (
    load_package,
    sha,
    raw_from_effective,
    INPUTS,
    require,
)

FP = C.POINTER(C.c_float)
IP = C.POINTER(C.c_int32)


def fp(a):
    return a.ctypes.data_as(FP)


def ip(a):
    return a.ctypes.data_as(IP)


def write(path, value):
    workflow._write_json(Path(path), value)


class Runtime:
    def __init__(self, folder, doc, raw, first, *, resume=False):
        self.folder = Path(folder)
        g = doc["geometry"]
        ir, self.weights, self.graph = build_training_ir(
            raw,
            first["hidden"],
            questions=g["questions"],
            options=g["options"],
            heads=g["heads"],
            feedforward=g["feedforward"],
        )
        self.ir = ir
        engine = ROOT / "build/libckernel_engine.so"
        require(engine.is_file(), "native engine unavailable")
        if not resume:
            reg = training_registry()
            lowered = backward.synthesize_ir2_backward(
                ir,
                reg,
                json.loads(
                    (ROOT / "version/v7/kernel_maps/kernel_bindings.json").read_text()
                ),
                json.loads(
                    (ROOT / "version/v8/scripts/grad_rules_v8.json").read_text()
                ),
                strict=True,
                allow_partial=False,
                checkpoint_policy="none",
            )
            layout = planner.build_layout(
                lowered, {"config": lowered["config"]}, 64, strict=True
            )
            require(
                layout["total_bytes"] <= 2 * 1024**3,
                "native workspace exceeds recipe bound",
            )
            source, summary = codegen.generate_c(
                lowered, reg, manifest={"config": lowered["config"]}, layout=layout
            )
            require(not summary["skipped_ops"], "incomplete backward generation")
            (self.folder / "train.c").write_text(source)
            subprocess.run(
                [
                    os.environ.get("CC", "cc"),
                    f"-DCK_NUM_TOKENS={g['tokens']}",
                    "-DCK_ADAMW_EPS=1e-4f",
                    "-O2",
                    "-shared",
                    "-fPIC",
                    "-I" + str(ROOT / "include"),
                    str(self.folder / "train.c"),
                    str(ROOT / "src/kernels/linear_rows_checked.c"),
                    str(engine),
                    "-Wl,-rpath," + str(engine.parent),
                    "-lm",
                    "-o",
                    str(self.folder / "train.so"),
                ],
                check=True,
            )
            write(self.folder / "layout.json", layout)
            write(self.folder / "backward_ir.json", lowered)
            write(
                self.folder / "runtime.json",
                dict(
                    summary=summary,
                    library_sha256=sha(self.folder / "train.so"),
                    engine_sha256=sha(engine),
                    source_sha256=sha(self.folder / "train.c"),
                ),
            )
        runtime = json.loads((self.folder / "runtime.json").read_text())
        self.summary = runtime["summary"]
        require(
            sha(self.folder / "train.so") == runtime["library_sha256"]
            and sha(engine) == runtime["engine_sha256"]
            and sha(self.folder / "train.c") == runtime["source_sha256"],
            "runtime identity mismatch",
        )
        self.lib = C.CDLL(str((self.folder / "train.so").resolve()))
        workflow._configure(self.lib)
        self.lib.ck_train_init.argtypes = [FP, IP, C.c_int]
        self.lib.ck_train_step_with_i32_ex.argtypes = [
            IP,
            IP,
            C.c_int,
            IP,
            C.c_size_t,
            FP,
            C.c_float,
        ]
        self.lib.ck_train_set_batch_with_i32_ex.argtypes = [
            IP,
            IP,
            C.c_int,
            IP,
            C.c_size_t,
        ]
        require(
            hasattr(self.lib, "ck_train_eval_step"),
            "runtime lacks current-loss evaluation; regenerate the training runtime",
        )
        self.lib.ck_train_eval_step.argtypes = [FP]
        self.lib.ck_train_eval_step.restype = C.c_int
        self.sizes = np.asarray(self.summary["init_weight_numel"], dtype="i4")
        initial = np.concatenate(
            [self.weights[n].ravel() for n in self.summary["init_weight_order"]]
        )
        require(
            self.lib.ck_train_init(fp(initial), ip(self.sizes), len(self.sizes))
            == len(self.sizes),
            "native init failed",
        )
        self.lib.ck_set_strict_parity(1)
        self.tokens = g["tokens"]
        self.ids = np.zeros(self.tokens, dtype="i4")
        self.slices = {}
        offset = 0
        for name, count in zip(self.summary["init_weight_order"], self.sizes):
            self.slices[name] = slice(offset, offset + int(count))
            offset += int(count)
        self.frozen_embedding = initial[self.slices["output_embedding.weight"]].copy()

    def close(self):
        self.lib.ck_train_free()

    def snapshot(self):
        return workflow._weight_export(self.lib)

    def batch(self, arrays):
        weights = self.snapshot()
        weights[self.slices["__features.hidden"]] = arrays["hidden"].ravel()
        require(np.isfinite(weights).all(), "nonfinite parameter snapshot")
        workflow._array_import(self.lib, "weight", weights)
        supplied = dict(
            arrays, broadcast_zeros=np.zeros(len(arrays["option_counts"]), dtype="i4")
        )
        metadata = np.concatenate(
            [
                supplied[n.removeprefix("input.")].ravel()
                for n in self.ir["config"]["training_integer_inputs"]
            ]
        ).astype("i4")
        return metadata

    def step(self, arrays, lr):
        metadata = self.batch(arrays)
        loss = C.c_float()
        require(
            self.lib.ck_train_step_with_i32_ex(
                ip(self.ids),
                ip(self.ids),
                self.tokens,
                ip(metadata),
                metadata.size,
                C.byref(loss),
                lr,
            )
            > 0,
            "native training step rejected",
        )
        require(np.isfinite(loss.value), "nonfinite native loss")
        np.testing.assert_array_equal(
            self.snapshot()[self.slices["output_embedding.weight"]],
            self.frozen_embedding,
        )
        return loss.value

    def evaluate(self, arrays):
        metadata = self.batch(arrays)
        require(
            self.lib.ck_train_set_batch_with_i32_ex(
                ip(self.ids), ip(self.ids), self.tokens, ip(metadata), metadata.size
            )
            > 0,
            "native batch rejected",
        )
        loss = C.c_float()
        require(
            self.lib.ck_train_eval_step(C.byref(loss)) > 0, "native evaluation rejected"
        )
        scores = (
            workflow.CERT._activation_snapshot(self.lib, self.summary)[
                "act.option_logits"
            ]
            .ravel()
            .copy()
        )
        require(
            np.isfinite(scores).all() and np.isfinite(loss.value),
            "nonfinite native evaluation",
        )
        return loss.value, scores


def run(
    package, package_sha256, folder, *, epochs=2, lr=0.001, resume=None, stop_after=None
):
    require(type(epochs) is int and 1 <= epochs <= 40, "epochs outside bounds")
    require(
        type(lr) in (int, float) and np.isfinite(lr) and 0 < lr <= 0.01,
        "invalid learning rate",
    )
    require(
        stop_after is None or (type(stop_after) is int and 1 <= stop_after <= 1280),
        "invalid checkpoint step",
    )
    doc, raw, records = load_package(package, package_sha256)
    folder = Path(folder).resolve()
    if resume is None:
        folder.mkdir(parents=True, exist_ok=False)
    else:
        require(folder.is_dir(), "missing resume runtime")
    rt = Runtime(folder, doc, raw, records[0]["arrays"], resume=resume is not None)
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    schedule = train * epochs
    require(len(schedule) <= 1280, "step budget exceeded")
    cfg = dict(
        dtype="fp32",
        optimizer="AdamW",
        lr=float(lr),
        eps=1e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        grad_accum=1,
        epochs=epochs,
        recipe="clef_head_fp32_v1",
    )
    meta = dict(
        runtime_contract_sha256=rt.summary["runtime_contract_sha256"],
        dataset_sha256=package_sha256,
        training_config=cfg,
        training_config_sha256=workflow._json_sha256(cfg),
        total_microsteps=len(schedule),
        grad_accum=1,
        next_microstep=0,
    )
    losses = []
    start = 0
    try:
        if resume:
            restored = workflow._checkpoint_load(Path(resume), rt.lib, meta)
            start = restored["next_microstep"]
            require(start < len(schedule), "checkpoint already completed")
            require(
                stop_after is None or stop_after > start,
                "checkpoint step must follow resumed state",
            )
        before = [rt.evaluate(r["arrays"])[0] for r in validation]
        for step, record in enumerate(schedule[start:], start):
            losses.append(rt.step(record["arrays"], float(lr)))
            write(
                folder / "progress.json",
                dict(
                    stage="training",
                    completed_steps=step + 1,
                    total_steps=len(schedule),
                    record_id=record["id"],
                    loss=losses[-1],
                ),
            )
            if stop_after is not None and step + 1 == stop_after:
                break
        completed = start + len(losses)
        checkpoint = workflow._checkpoint_write(
            folder,
            f"checkpoint-{completed}",
            rt.lib,
            dict(meta, next_microstep=completed),
        )
        if completed < len(schedule):
            return dict(
                status="CHECKPOINTED",
                passed=False,
                checkpoint=str(checkpoint),
                completed_steps=completed,
            )
        final = rt.snapshot()
        moments = workflow._array_export(rt.lib, "optimizer_state")
        # Round-trip uses the existing identity-bound checkpoint implementation.
        workflow._checkpoint_load(checkpoint, rt.lib, meta)
        np.testing.assert_array_equal(rt.snapshot(), final)
        np.testing.assert_array_equal(
            workflow._array_export(rt.lib, "optimizer_state"), moments
        )
        effective = {
            n: final[sl].reshape(rt.weights[n].shape).copy()
            for n, sl in rt.slices.items()
            if n != "__features.hidden"
        }
        updated = raw_from_effective(raw, effective)
        np.savez(folder / "trained_head.npz", **updated)
        expected = {r["id"]: rt.evaluate(r["arrays"]) for r in records}
        artifacts = []
        for r in records:
            for q, candidates in enumerate(r["candidate_ids"]):
                artifacts.append(
                    dict(
                        id=f"{r['id']}-q{q}",
                        inputs={n: r["arrays"][n] for n in INPUTS},
                        question_index=q,
                        candidate_ids=candidates,
                    )
                )
        provenance = dict(
            training_package_sha256=package_sha256,
            base_model=doc["model"],
            training_config=cfg,
            features="supplied",
            embedding="supplied",
            numerical_parity="NOT_TESTED",
        )
        package_identity = build_package(
            folder / "inference",
            updated,
            artifacts,
            dict(id=doc["model"]["id"], revision=sha(folder / "trained_head.npz")),
            heads=doc["geometry"]["heads"],
            provenance=provenance,
        )
        manifest = json.loads((folder / "inference/package.json").read_text())
        files = (
            {"package.json"}
            | {v["file"] for v in manifest["files"].values()}
            | {v["file"] for v in manifest["artifacts"]}
        )
        with zipfile.ZipFile(
            folder / "serving.zip", "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for name in sorted(files):
                archive.write(folder / "inference" / name, name)
        from server.decision.native import NativeHead

        native = NativeHead(folder / "inference/package.json", package_identity)
        try:
            for r in records:
                offset = 0
                for q, candidates in enumerate(r["candidate_ids"]):
                    status, reply = native.evaluate(native.request(f"{r['id']}-q{q}"))
                    require(
                        status == 200 and reply["ok"], "exported head request failed"
                    )
                    scores = [v["logit"] for v in reply["result"]["scores"]]
                    np.testing.assert_allclose(
                        scores,
                        expected[r["id"]][1][offset : offset + len(candidates)],
                        atol=3e-5,
                        rtol=3e-4,
                    )
                    offset += len(candidates)
        finally:
            native.close()
        report = dict(
            schema="cke.clef.training-result.v1",
            status="PASS",
            passed=True,
            implementation_stage="head_only",
            execution_kind="native_generated",
            dtype="fp32",
            optimizer="AdamW",
            completed_steps=completed,
            package_sha256=package_sha256,
            model=doc["model"],
            numerical_parity="NOT_TESTED",
            quality="NOT_TESTED",
            checks={
                name: dict(passed=True)
                for name in (
                    "native_execution",
                    "frozen_embedding",
                    "checkpoint_reload",
                    "inference_export",
                )
            },
            metrics=dict(
                training_losses=losses,
                validation_loss_before=float(np.mean(before)),
                validation_loss_after=float(
                    np.mean([expected[r["id"]][0] for r in validation])
                ),
            ),
            records=[
                dict(
                    id=r["id"],
                    split=r["split"],
                    question_ids=r["question_ids"],
                    candidate_ids=r["candidate_ids"],
                    logits=expected[r["id"]][1].tolist(),
                )
                for r in records
            ],
            artifacts=dict(
                checkpoint=str(checkpoint),
                trained_head=str(folder / "trained_head.npz"),
                inference_manifest=str(folder / "inference/package.json"),
                inference_package_sha256=package_identity,
            ),
            runtime=dict(
                library_sha256=sha(folder / "train.so"),
                engine_sha256=sha(ROOT / "build/libckernel_engine.so"),
                contract_sha256=rt.summary["runtime_contract_sha256"],
            ),
            final_optimizer_state_sha256=hashlib.sha256(moments.tobytes()).hexdigest(),
        )
        write(folder / "result.json", report)
        write(
            folder / "progress.json",
            dict(
                stage="complete", completed_steps=completed, total_steps=len(schedule)
            ),
        )
        return report
    finally:
        rt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--package-sha256", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after", type=int)
    args = p.parse_args()
    report = run(
        args.package,
        args.package_sha256,
        args.run_dir,
        epochs=args.epochs,
        lr=args.lr,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    write(args.json_out, report)
    return 0 if report["status"] in ("PASS", "CHECKPOINTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
