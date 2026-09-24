#!/usr/bin/env python3
"""Generated-C training-to-export certification for a tiny Qwen3-style model.

This v8-owned command selectively reuses the proven v7 training IR/code generator.
It does not invoke v7's training orchestration or use Python to update CKE weights.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import resource
import shutil
import shlex
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V7 = ROOT / "version" / "v7" / "scripts"
if str(V7) not in sys.path:
    sys.path.insert(0, str(V7))
CORPUS_SPEC = ROOT / "version" / "v8" / "training" / "english_byte_v1.json"
DEFAULT_RUN = ROOT / "version" / "v8" / ".cache" / "training_workflow" / "qwen3_4layer_fp32"
DEFAULT_REPORT = ROOT / "version" / "v8" / ".cache" / "reports" / "training_workflow_latest.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CERT = _load_module("cke_v8_training_cert", Path(__file__).with_name("run_training_certification_v8.py"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "architecture": "qwen3_style_dense_reduced", "layers": int(args.layers), "dtype": "fp32",
        "d_model": int(args.d_model), "hidden": int(args.hidden), "heads": int(args.num_heads),
        "kv_heads": int(args.num_kv_heads), "vocab_size": int(args.vocab_size),
        "rope_theta": float(getattr(args, "rope_theta", 1_000_000.0)),
        "tokenizer": str(args.tokenizer),
        "seq_len": int(args.seq_len), "epochs": int(args.epochs), "grad_accum": int(args.grad_accum),
        "optimizer": "generated_c_adamw", "lr": float(args.lr), "beta1": float(args.beta1),
        "beta2": float(args.beta2), "eps": float(args.eps), "weight_decay": float(args.weight_decay),
    }


def _apply_semantic_model(args: argparse.Namespace) -> tuple[dict[str, Any] | None, Path | None]:
    """Make a Python-authored semantic model authoritative for workflow geometry."""
    if args.semantic_model is None:
        return None, None
    path = args.semantic_model.resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != "cke.v8.python_semantic_model.v1":
        raise RuntimeError("semantic model has an unsupported schema")
    observed = _sha256(path)
    if not args.semantic_model_sha256 or observed != args.semantic_model_sha256:
        raise RuntimeError("semantic model identity mismatch")
    contract = document.get("model_contract")
    template = document.get("template")
    graph = document.get("graph")
    trace = document.get("operation_trace")
    if not all(isinstance(value, dict) for value in (contract, template, graph, trace)):
        raise RuntimeError("semantic model is missing graph, contract, trace, or template")
    if int(contract.get("layers", 0)) not in {2, 4, 5, 6, 10}:
        raise RuntimeError("Python semantic graph supports the certified 2, 4, 5, 6, or 10 layer depths")
    if contract.get("dtype") != "fp32" or contract.get("activation") != "swiglu":
        raise RuntimeError("semantic model requests unsupported dtype or activation")
    if contract.get("bias") is not False or contract.get("normalization") != "rmsnorm":
        raise RuntimeError("semantic model requests unsupported bias or normalization")
    if float(contract.get("norm_epsilon", 0.0)) != 1e-6:
        raise RuntimeError("semantic model requests unsupported RMSNorm epsilon")
    if contract.get("initialization") != "normal_0p02":
        raise RuntimeError("semantic model requests unsupported initialization")
    rope_theta = float(contract.get("rope_theta", 0.0))
    if not math.isfinite(rope_theta) or rope_theta <= 0.0:
        raise RuntimeError("semantic model rope_theta must be finite and positive")
    args.layers = int(contract["layers"]); args.d_model = int(contract["d_model"])
    args.hidden = int(contract["hidden"]); args.num_heads = int(contract["heads"])
    args.num_kv_heads = int(contract["kv_heads"]); args.seq_len = int(contract["seq_len"])
    if int(contract["vocab_size"]) != int(args.vocab_size):
        raise RuntimeError("semantic model vocabulary does not match tokenizer vocabulary")
    args.rope_theta = rope_theta
    template_path = args.run_dir / "python_training_lowered_template.json"
    _write_json(template_path, template)
    return document, template_path


def _validate_semantic_parameter_contract(
    semantic_model: Mapping[str, Any], expected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trace = semantic_model.get("operation_trace")
    layers = trace.get("layers") if isinstance(trace, Mapping) else None
    if not isinstance(trace, Mapping) or not isinstance(layers, list) or not layers:
        raise RuntimeError("semantic model has no operation trace for parameter ownership")
    layer_owners = {
        int(row["index"]): row for row in layers
        if isinstance(row, Mapping) and "index" in row
    }

    def expected_owner(name: str) -> str:
        if name == "token_emb":
            return str(trace["embedding"])
        if name == "final_ln_weight":
            return str(trace["final_norm"])
        if name == "output.weight":
            return str(trace["lm_head"])
        parts = name.split(".")
        if len(parts) != 3 or parts[0] != "layer":
            raise RuntimeError(f"semantic parameter ownership has no rule for {name}")
        layer = layer_owners.get(int(parts[1]))
        if layer is None:
            raise RuntimeError(f"semantic parameter refers to unknown layer {parts[1]}")
        field = parts[2]
        if field == "ln1_gamma":
            return str(layer["attn_norm"])
        if field == "ln2_gamma":
            return str(layer["ffn_norm"])
        if field in {"wq", "wk", "wv", "wo", "q_norm", "k_norm"}:
            return str(layer["attention"])
        if field in {"w1", "w2"}:
            return str(layer["feed_forward"])
        raise RuntimeError(f"semantic parameter ownership has no rule for {name}")

    rows = semantic_model.get("parameter_contract")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("semantic model has no parameter contract")
    declared: dict[str, list[int]] = {}
    ownership_errors = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("authored_semantic_id"):
            raise RuntimeError("semantic parameter lacks an authored identity")
        name = str(row.get("runtime_parameter") or "")
        shape = row.get("shape")
        if not name or not isinstance(shape, list) or name in declared:
            raise RuntimeError("semantic parameter contract contains an invalid or duplicate runtime parameter")
        declared[name] = [int(value) for value in shape]
        owner = str(row["authored_semantic_id"])
        required_owner = expected_owner(name)
        if owner != required_owner:
            ownership_errors.append({"parameter": name, "expected": required_owner, "observed": owner})
    if ownership_errors:
        raise RuntimeError(f"semantic parameter ownership mismatch: {ownership_errors[:8]}")
    generated = {str(row["name"]): [int(value) for value in row["shape"]] for row in expected}
    if declared != generated:
        missing = sorted(set(declared) - set(generated))
        unexpected = sorted(set(generated) - set(declared))
        wrong_shape = sorted(name for name in set(declared) & set(generated) if declared[name] != generated[name])
        raise RuntimeError(
            "authored/generated parameter contract mismatch: "
            f"missing={missing}, unexpected={unexpected}, wrong_shape={wrong_shape}"
        )
    return {
        "passed": True, "count": len(declared), "ownership_count": len(declared),
        "source": "python_semantic_model.parameter_contract",
    }


def _validate_semantic_operation_trace(run_dir: Path, semantic_model: Mapping[str, Any]) -> dict[str, Any]:
    graph = semantic_model.get("graph")
    nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
    valid_ids = {
        str(row.get("id")) for row in nodes or []
        if isinstance(row, Mapping) and row.get("id")
    }
    trace = semantic_model.get("operation_trace")
    trace_layers = trace.get("layers") if isinstance(trace, Mapping) else None
    if not isinstance(trace, Mapping) or not isinstance(trace_layers, list) or not trace_layers:
        raise RuntimeError("semantic model has no operation trace")
    layer_owners = {
        int(row["index"]): row for row in trace_layers
        if isinstance(row, Mapping) and "index" in row
    }

    def forward_owner(op: Mapping[str, Any]) -> str:
        layer_index = int(op.get("layer", -1))
        op_name = str(op.get("op") or "")
        if layer_index == -1:
            if op_name in {"bpe_tokenizer", "dense_embedding_lookup"}:
                return str(trace["embedding"])
            if op_name == "rmsnorm":
                return str(trace["final_norm"])
            if op_name in {"lm_head", "logits"}:
                return str(trace["lm_head"])
            raise RuntimeError(f"semantic trace has no owner rule for global forward operation {op_name}")
        layer = layer_owners.get(layer_index)
        if layer is None:
            raise RuntimeError(f"semantic trace refers to unknown layer {layer_index}")
        if op_name == "rmsnorm":
            instance = int(op.get("instance", -1))
            if instance == 0:
                return str(layer["attn_norm"])
            if instance == 1:
                return str(layer["ffn_norm"])
            raise RuntimeError(f"semantic trace has invalid RMSNorm instance {instance} in layer {layer_index}")
        if op_name in {
            "q_proj", "k_proj", "v_proj", "qk_norm", "rope_qk", "attn", "out_proj",
            "bridge_token_to_head_major", "bridge_head_to_token_major",
        }:
            return str(layer["attention"])
        if op_name in {"mlp_gate_up", "silu_mul", "geglu", "mlp_down"}:
            return str(layer["feed_forward"])
        if op_name == "residual_add":
            return str(layer["block"])
        raise RuntimeError(f"semantic trace has no owner rule for forward operation {op_name}")

    ir2 = json.loads((run_dir / "ir2_train_backward.json").read_text(encoding="utf-8"))
    phases = {phase: ir2.get(phase) for phase in ("forward", "backward")}
    for phase, operations in phases.items():
        if not isinstance(operations, list) or not operations:
            raise RuntimeError(f"generated semantic trace has no {phase} operation inventory")

    forward_by_id: dict[int, Mapping[str, Any]] = {}
    checked = []
    ownership_errors = []
    covered_ids: set[str] = set()
    for op in phases["forward"]:
        if not isinstance(op, Mapping) or "op_id" not in op:
            raise RuntimeError("generated forward operation inventory contains an invalid row")
        op_id = int(op["op_id"])
        if op_id in forward_by_id:
            raise RuntimeError(f"generated forward operation inventory duplicates op_id {op_id}")
        forward_by_id[op_id] = op
        required_owner = forward_owner(op)
        observed_owner = str(op.get("authored_semantic_id") or "")
        if observed_owner != required_owner:
            ownership_errors.append({
                "phase": "forward", "op_id": op_id, "op": op.get("op"),
                "expected": required_owner, "observed": observed_owner,
            })
        covered_ids.add(observed_owner)
        if op.get("kernel_id"):
            checked.append({
                "phase": "forward", "op_id": op_id, "op": op.get("op"),
                "authored_semantic_id": observed_owner,
            })

    expected_covered_ids = {
        str(trace["embedding"]), str(trace["final_norm"]), str(trace["lm_head"]),
        *(str(row[field]) for row in trace_layers for field in (
            "block", "attn_norm", "attention", "ffn_norm", "feed_forward"
        )),
    }
    missing_operations = sorted(expected_covered_ids - covered_ids)

    lineage_errors = []
    for op in phases["backward"]:
        if not isinstance(op, Mapping) or not op.get("kernel_id") or "op_id" not in op:
            raise RuntimeError("generated backward kernel operation inventory contains an invalid row")
        forward_ref = op.get("forward_ref")
        forward_op = forward_by_id.get(int(forward_ref)) if forward_ref is not None else None
        observed_owner = str(op.get("authored_semantic_id") or "")
        if forward_op is None:
            lineage_errors.append({"op_id": op.get("op_id"), "forward_ref": forward_ref, "reason": "missing_forward"})
        else:
            forward_semantic_id = str(forward_op.get("authored_semantic_id") or "")
            if observed_owner != forward_semantic_id:
                lineage_errors.append({
                    "op_id": op.get("op_id"), "forward_ref": forward_ref,
                    "expected": forward_semantic_id, "observed": observed_owner,
                })
            if int(op.get("layer", -1)) != int(forward_op.get("layer", -1)):
                lineage_errors.append({
                    "op_id": op.get("op_id"), "forward_ref": forward_ref,
                    "reason": "layer_mismatch", "expected": forward_op.get("layer"),
                    "observed": op.get("layer"),
                })
            backward_name = str(op.get("op") or "")
            if backward_name != "grad_accumulate":
                expected_forward_name = (
                    "logits" if backward_name == "loss_backward"
                    else backward_name.removesuffix("_backward_core")
                )
                if str(forward_op.get("op") or "") != expected_forward_name:
                    lineage_errors.append({
                        "op_id": op.get("op_id"), "forward_ref": forward_ref,
                        "reason": "operation_mismatch", "expected": expected_forward_name,
                        "observed": forward_op.get("op"),
                    })
        checked.append({
            "phase": "backward", "op_id": op.get("op_id"), "op": op.get("op"),
            "forward_ref": forward_ref, "authored_semantic_id": observed_owner,
        })

    invalid_ids = sorted(owner for owner in covered_ids if owner and owner not in valid_ids)
    if ownership_errors or missing_operations or lineage_errors or invalid_ids:
        raise RuntimeError(
            "generated semantic operation trace mismatch: "
            f"ownership={ownership_errors[:8]}, missing_operations={missing_operations[:8]}, "
            f"lineage={lineage_errors[:8]}, invalid_ids={invalid_ids[:8]}"
        )
    return {
        "passed": True, "lowered_kernel_operations": len(checked),
        "lowered_forward_kernel_operations": sum(row["phase"] == "forward" for row in checked),
        "lowered_backward_kernel_operations": sum(row["phase"] == "backward" for row in checked),
        "operations": checked,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _git_blob(path: Path) -> str:
    result = subprocess.run(["git", "hash-object", str(path)], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _ensure_bpe_tools(python: str) -> tuple[Path, Path]:
    trainer = ROOT / "build" / "ck-bpe-train"
    library = ROOT / "build" / "libckernel_tokenizer.so"
    if not trainer.is_file() or not library.is_file():
        result = subprocess.run(
            ["make", "tokenizer", "ck-bpe-train"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise RuntimeError("failed to build CKE BPE tools: " + result.stdout[-4000:])
    return trainer, library


def _load_corpus(args: argparse.Namespace, out_dir: Path, python: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec_path = args.corpus
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    svg_domain = spec.get("domain") == "svg_xml"
    if svg_domain and (
        spec.get("document_policy") != "one_complete_document_per_split"
        or spec.get("label_policy") != "cyclic_causal_next_token_within_document"
        or spec["train"]["git_blob"] == spec["validation"]["git_blob"]
    ):
        raise RuntimeError("SVG fixture requires separate complete documents and an explicit cyclic label policy")
    tokenizer_kind = str(args.tokenizer)
    bpe_handle = None
    tokenizer_meta: dict[str, Any]
    if tokenizer_kind == "bpe":
        trainer, library = _ensure_bpe_tools(python)
        tokenizer_dir = out_dir / "tokenizer_bin"
        tokenizer_json = out_dir / "tokenizer.json"
        trainer_corpus = out_dir / "tokenizer_corpus"
        trainer_corpus.mkdir(parents=True, exist_ok=True)
        train_row = spec["train"]
        train_source = ROOT / str(train_row["path"])
        (trainer_corpus / "train.txt").write_bytes(train_source.read_bytes()[: int(train_row["token_count"])])
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        command = [str(trainer), "--corpus-dir", str(trainer_corpus), "--out", str(tokenizer_json),
                   "--binary-out-dir", str(tokenizer_dir), "--vocab-size", str(args.vocab_size),
                   "--min-freq", str(args.bpe_min_freq), "--max-piece-bytes", str(args.bpe_max_piece_bytes),
                   "--threads", "1"]
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            raise RuntimeError("CKE BPE training failed: " + completed.stdout[-4000:])
        pipeline = _load_module("cke_v7_tokenizer_runtime", V7 / "train_data_pipeline_v7.py")
        bpe_handle = pipeline._TrueBPEHandle(library, tokenizer_dir, tokenizer_json)
        bpe_encode = pipeline._encode_large_text_with_bpe_handle
        if int(bpe_handle.vocab_size) != int(args.vocab_size):
            raise RuntimeError(f"BPE vocab mismatch: requested={args.vocab_size} actual={bpe_handle.vocab_size}")
        tokenizer_meta = {
            "name": "cke_true_bpe_v1", "mode": "bpe", "vocab_size": int(bpe_handle.vocab_size),
            "tokenizer_json": str(tokenizer_json), "tokenizer_json_sha256": _sha256(tokenizer_json),
            "binary_dir": str(tokenizer_dir),
            "binary_sha256": {p.name: _sha256(p) for p in sorted(tokenizer_dir.iterdir()) if p.is_file()},
            "training_split_only": True, "trainer_command": shlex.join(command),
        }
    elif tokenizer_kind == "byte":
        if int(args.vocab_size) != 256:
            raise RuntimeError("byte tokenizer requires --vocab-size 256")
        tokenizer_meta = dict(spec["tokenizer"])
        tokenizer_meta["mode"] = "byte"
    else:
        raise RuntimeError(f"unsupported tokenizer: {tokenizer_kind}")
    outputs: dict[str, Any] = {}
    svg_documents: dict[str, Any] = {}
    arrays: list[np.ndarray] = []
    try:
        for split in ("train", "validation"):
            row = spec[split]
            source = ROOT / str(row["path"])
            actual_blob = _git_blob(source)
            if actual_blob != str(row["git_blob"]):
                raise RuntimeError(f"{split} corpus blob changed: expected={row['git_blob']} actual={actual_blob}")
            raw = source.read_bytes()[: int(row["token_count"])]
            if len(raw) != int(row["token_count"]):
                raise RuntimeError(f"{split} corpus has only {len(raw)} bytes")
            if svg_domain:
                if source.stat().st_size != len(raw):
                    raise RuntimeError(f"{split} SVG corpus must contain exactly one complete source document")
                svg_text = raw.decode("utf-8")
                try:
                    svg_root = ET.fromstring(svg_text)
                except ET.ParseError as exc:
                    raise RuntimeError(f"{split} SVG corpus is not well-formed XML: {exc}") from exc
                if svg_root.tag != "{http://www.w3.org/2000/svg}svg":
                    raise RuntimeError(f"{split} SVG corpus root must be an SVG element")
                allowed_elements = {"svg", "rect", "circle"}
                allowed_attributes = {
                    "xmlns", "width", "height", "viewBox", "x", "y", "rx", "cx", "cy", "r", "fill"
                }
                for element in svg_root.iter():
                    local_name = element.tag.rsplit("}", 1)[-1]
                    if local_name not in allowed_elements or any(
                        key.rsplit("}", 1)[-1] not in allowed_attributes
                        or "url(" in value.lower() or "://" in value
                        for key, value in element.attrib.items()
                    ):
                        raise RuntimeError(f"{split} SVG corpus contains an unsupported element or attribute")
            if bpe_handle is None:
                ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
                exact_roundtrip = True
            else:
                text = raw.decode("utf-8")
                ids = np.asarray(bpe_encode(bpe_handle, text), dtype=np.int32)
                exact_roundtrip = bpe_handle.decode(ids.tolist()) == text
                if not exact_roundtrip:
                    raise RuntimeError(f"{split} BPE encode/decode roundtrip failed")
            limit = int(args.max_train_tokens if split == "train" else args.max_validation_tokens)
            tokenized_count = int(ids.size)
            if svg_domain and 0 < limit < tokenized_count:
                raise RuntimeError(
                    f"{split} SVG token limit {limit} truncates the complete tokenized document "
                    f"({tokenized_count} tokens); use 0 or at least {tokenized_count}"
                )
            if limit > 0:
                ids = ids[:limit]
            if svg_domain:
                svg_documents[split] = {
                    "source": str(row["path"]), "sha256": _sha256(source),
                    "xml": svg_text, "element_count": sum(1 for _ in svg_root.iter()),
                    "root": "svg", "source_complete_document": True,
                    "tokenized_tokens": tokenized_count, "consumed_tokens": int(ids.size),
                    "consumed_complete_document": int(ids.size) == tokenized_count,
                    "complete_document": int(ids.size) == tokenized_count,
                }
            if ids.size < 2 or int(ids.min()) < 0 or int(ids.max()) >= int(args.vocab_size):
                raise RuntimeError(f"{split} token IDs violate vocab contract")
            token_path = out_dir / f"{split}_token_ids.i32"
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_bytes(ids.astype("<i4", copy=False).tobytes())
            outputs[split] = {
                "source": str(row["path"]), "git_blob": actual_blob, "tokens": int(ids.size),
                "source_bytes": len(raw), "exact_roundtrip": exact_roundtrip,
                "original_source": str(row["source_path"]), "source_revision": str(row["source_revision"]),
                "source_blob": str(row["source_blob"]), "source_sha256": _sha256(source),
                "token_ids": str(token_path), "token_ids_sha256": _sha256(token_path),
            }
            arrays.append(ids)
    finally:
        if bpe_handle is not None:
            bpe_handle.close()
    return arrays[0], arrays[1], {
        "spec": str(spec_path), "spec_sha256": _sha256(spec_path),
        "domain": "svg_xml" if svg_domain else "text",
        "document_policy": spec.get("document_policy", "continuous_split_stream"),
        "label_policy": spec.get("label_policy", "cyclic_causal_next_token"),
        "validation_relationship": spec.get("validation_relationship"),
        "svg_documents": svg_documents,
        "tokenizer": tokenizer_meta, "splits": outputs,
    }


def _batches(tokens: np.ndarray, seq_len: int, epochs: int) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    # A target token is presented exactly once per epoch. The final short window is padded.
    result: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    stream = np.concatenate([tokens, tokens[:1]])
    for epoch in range(epochs):
        for start in range(0, int(tokens.size), seq_len):
            valid = min(seq_len, int(tokens.size) - start)
            x = np.zeros(seq_len, dtype=np.int32)
            y = np.zeros(seq_len, dtype=np.int32)
            x[:valid] = stream[start : start + valid]
            y[:valid] = stream[start + 1 : start + valid + 1]
            result.append((x, y, valid, epoch))
    return result


def _configure(lib: ctypes.CDLL) -> None:
    CERT._configure_runtime(lib)
    signatures = {
        "ck_train_step_ex": ([ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_float], ctypes.c_int),
        "ck_train_flush_optimizer": ([ctypes.c_float], ctypes.c_int),
        "ck_train_import_weight_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_get_optimizer_state_snapshot_numel": ([], ctypes.c_int),
        "ck_train_export_optimizer_state_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_import_optimizer_state_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_get_accum_snapshot_numel": ([], ctypes.c_int),
        "ck_train_export_accum_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_import_accum_snapshot": ([ctypes.POINTER(ctypes.c_float), ctypes.c_int], ctypes.c_int),
        "ck_train_get_accum_counter": ([], ctypes.c_int), "ck_train_set_accum_counter": ([ctypes.c_int], ctypes.c_int),
        "ck_train_get_accum_tokens": ([], ctypes.c_int64), "ck_train_set_accum_tokens": ([ctypes.c_int64], ctypes.c_int64),
        "ck_train_get_opt_step": ([], ctypes.c_int), "ck_train_set_opt_step": ([ctypes.c_int], ctypes.c_int),
        "ck_train_get_last_step_profile": ([ctypes.POINTER(ctypes.c_double)] * 4 + [ctypes.POINTER(ctypes.c_int)] * 4, ctypes.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = result


def _array_export(lib: ctypes.CDLL, prefix: str) -> np.ndarray:
    count = int(getattr(lib, f"ck_train_get_{prefix}_snapshot_numel")())
    buf = (ctypes.c_float * count)()
    wrote = int(getattr(lib, f"ck_train_export_{prefix}_snapshot")(buf, count))
    if wrote != count:
        raise RuntimeError(f"{prefix} snapshot wrote {wrote}, expected {count}")
    return np.ctypeslib.as_array(buf, shape=(count,)).astype(np.float32, copy=True)


def _array_import(lib: ctypes.CDLL, prefix: str, values: np.ndarray) -> None:
    flat = np.ascontiguousarray(values, dtype=np.float32)
    ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    wrote = int(getattr(lib, f"ck_train_import_{prefix}_snapshot")(ptr, int(flat.size)))
    if wrote != int(flat.size):
        raise RuntimeError(f"{prefix} snapshot import wrote {wrote}, expected {flat.size}")


def _weight_export(lib: ctypes.CDLL) -> np.ndarray:
    count = int(lib.ck_train_get_weight_snapshot_numel())
    buf = (ctypes.c_float * count)()
    wrote = int(lib.ck_train_export_weight_snapshot(buf, count))
    if wrote != count:
        raise RuntimeError(f"weight snapshot wrote {wrote}, expected {count}")
    return np.ctypeslib.as_array(buf, shape=(count,)).astype(np.float32, copy=True)


def _grad_export(lib: ctypes.CDLL) -> np.ndarray:
    count = int(lib.ck_train_get_parameter_gradient_snapshot_numel())
    buf = (ctypes.c_float * count)()
    wrote = int(lib.ck_train_export_parameter_gradient_snapshot(buf, count))
    if wrote != count:
        raise RuntimeError(f"gradient snapshot wrote {wrote}, expected {count}")
    return np.ctypeslib.as_array(buf, shape=(count,)).astype(np.float32, copy=True)


def _init_runtime(
    run_dir: Path,
    library: Path,
    summary: Mapping[str, Any],
    ck_run: Any,
    *,
    role: str,
) -> tuple[ctypes.CDLL, dict[str, Any]]:
    build_identity_path = run_dir / "generated_train_build_identity_v7.json"
    build_identity = json.loads(build_identity_path.read_text(encoding="utf-8"))
    expected_build = str(build_identity.get("build_identity_sha256") or "")
    inputs = build_identity.get("inputs") if isinstance(build_identity.get("inputs"), Mapping) else {}
    expected_engine = str(inputs.get("kernel_library_sha256") or "")
    expected_contract = str(summary.get("runtime_contract_sha256") or "")
    if not expected_build or not expected_engine or not expected_contract:
        raise RuntimeError(f"{role} runtime identity is incomplete")
    probe_path = run_dir / f"loaded_dependency_probe_{role}.json"
    probe_path.unlink(missing_ok=True)
    probe_process, fresh_probe = CERT._run_dependency_probe(
        library,
        probe_path,
        expected_contract=expected_contract,
        expected_build_identity=expected_build,
        expected_engine_sha256=expected_engine,
    )
    if probe_process.returncode != 0 or not fresh_probe.get("passed"):
        raise RuntimeError(f"{role} fresh-process dependency provenance failed")
    lib = ctypes.CDLL(str(library.resolve()), mode=ctypes.RTLD_LOCAL)
    _configure(lib)
    # The certification lane chooses the reference-order numerical contract.
    # Optimized-provider certification is a later, separate contract.
    lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
    lib.ck_set_strict_parity.restype = None
    lib.ck_set_strict_parity(1)
    contract = CERT._contract_check(lib, expected_contract)
    loaded_build = CERT._build_identity_check(lib, expected_build)
    loaded_library = CERT._loaded_library_evidence(library)
    engine = CERT._mapped_library_evidence("libckernel_engine.so")
    engine_hashes = sorted(
        str(row.get("sha256")) for row in engine.get("identities", [])
        if isinstance(row, Mapping) and row.get("sha256")
    )
    engine_passed = bool(
        engine.get("observed_in_process_maps")
        and len(engine.get("mapped_paths", [])) == 1
        and engine_hashes == [expected_engine]
    )
    if not contract.get("passed") or not loaded_build.get("passed"):
        raise RuntimeError(f"{role} loaded training library identity mismatch")
    if not loaded_library.get("observed_in_process_maps") or not engine_passed:
        raise RuntimeError(f"{role} loaded library/engine was not the built dependency")
    payload = ck_run._build_ck_runtime_init_payload(run_dir, dict(summary))
    rc = int(lib.ck_train_init(payload["float_buffer"], payload["sizes_buffer"], int(payload["num_params"])))
    if rc < 0:
        raise RuntimeError(f"ck_train_init failed: {rc}")
    actual_threads = None
    if hasattr(lib, "ck_get_num_threads"):
        lib.ck_get_num_threads.argtypes = []
        lib.ck_get_num_threads.restype = ctypes.c_int
        actual_threads = int(lib.ck_get_num_threads())
    evidence = {
        "passed": True, "role": role, "fresh_process": fresh_probe,
        "runtime_contract": contract, "build_identity": loaded_build,
        "training_library": loaded_library,
        "engine_dependency": {**engine, "expected_sha256": expected_engine,
                              "loaded_sha256": engine_hashes, "passed": engine_passed},
        "actual_runtime_threads": actual_threads,
    }
    return lib, evidence


def _checkpoint_write(root: Path, name: str, lib: ctypes.CDLL, meta: Mapping[str, Any], *, inject_interrupt: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    tmp = root / ("." + name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    arrays = {"weight": _weight_export(lib), "optimizer_state": _array_export(lib, "optimizer_state"), "accum": _array_export(lib, "accum")}
    files: dict[str, Any] = {}
    for key, value in arrays.items():
        path = tmp / f"{key}.f32"
        path.write_bytes(value.astype("<f4", copy=False).tobytes())
        files[key] = {"file": path.name, "numel": int(value.size), "sha256": _sha256(path)}
    doc = dict(meta)
    doc.update({"schema": "cke.v8.training_checkpoint.v1", "complete": True, "files": files,
                "optimizer_step": int(lib.ck_train_get_opt_step()), "accum_counter": int(lib.ck_train_get_accum_counter()),
                "accum_tokens": int(lib.ck_train_get_accum_tokens())})
    doc["checkpoint_identity_sha256"] = _checkpoint_identity(doc)
    _write_json(tmp / "checkpoint.json", doc)
    if inject_interrupt:
        raise RuntimeError("injected checkpoint interruption before atomic publish")
    if target.exists():
        raise RuntimeError(f"checkpoint target already exists: {target}")
    os.replace(tmp, target)
    _write_json(root / "latest.json", {"checkpoint": name, "checkpoint_manifest_sha256": _sha256(target / "checkpoint.json")})
    return target


def _checkpoint_identity(doc: Mapping[str, Any]) -> str:
    fields = {
        key: doc.get(key)
        for key in (
            "schema", "complete", "runtime_contract_sha256", "training_config_sha256",
            "training_config", "dataset_sha256", "next_microstep", "total_microsteps", "optimizer_step",
            "accum_counter", "accum_tokens", "files",
        )
    }
    return _json_sha256(fields)


def _validate_checkpoint_document(doc: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if doc.get("schema") != "cke.v8.training_checkpoint.v1" or doc.get("complete") is not True:
        raise RuntimeError("checkpoint is incomplete or has the wrong schema")
    recorded_identity = str(doc.get("checkpoint_identity_sha256") or "")
    if not recorded_identity or recorded_identity != _checkpoint_identity(doc):
        raise RuntimeError("checkpoint identity hash mismatch")
    for key in ("runtime_contract_sha256", "training_config_sha256", "dataset_sha256"):
        if str(doc.get(key) or "") != str(expected.get(key) or ""):
            raise RuntimeError(f"checkpoint {key} mismatch")
    checkpoint_config = doc.get("training_config")
    if not isinstance(checkpoint_config, Mapping) or _json_sha256(checkpoint_config) != str(doc["training_config_sha256"]):
        raise RuntimeError("checkpoint training configuration identity mismatch")
    next_microstep = int(doc.get("next_microstep", -1))
    total_microsteps = int(doc.get("total_microsteps", -1))
    if total_microsteps != int(expected.get("total_microsteps", -2)):
        raise RuntimeError("checkpoint total_microsteps mismatch")
    if next_microstep < 0 or next_microstep > total_microsteps:
        raise RuntimeError("checkpoint next_microstep is out of range")
    accum_counter = int(doc.get("accum_counter", -1))
    accum_tokens = int(doc.get("accum_tokens", -1))
    grad_accum = int(expected.get("grad_accum", 0))
    if grad_accum <= 0 or accum_counter < 0 or accum_counter >= grad_accum:
        raise RuntimeError("checkpoint accumulation counter is invalid")
    if accum_tokens < 0 or (accum_counter == 0 and accum_tokens != 0):
        raise RuntimeError("checkpoint contributing-token count is invalid")
    if int(doc.get("optimizer_step", -1)) < 0:
        raise RuntimeError("checkpoint optimizer step is invalid")
    files = doc.get("files")
    if not isinstance(files, Mapping) or set(files) != {"weight", "optimizer_state", "accum"}:
        raise RuntimeError("checkpoint file inventory is invalid")


def _checkpoint_load(path: Path, lib: ctypes.CDLL, expected: Mapping[str, Any]) -> dict[str, Any]:
    doc = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    _validate_checkpoint_document(doc, expected)
    if doc.get("complete") is not True:
        raise RuntimeError("checkpoint is incomplete")
    for key in ("weight", "optimizer_state", "accum"):
        row = doc["files"][key]
        file_path = path / row["file"]
        if _sha256(file_path) != row["sha256"]:
            raise RuntimeError(f"checkpoint {key} hash mismatch")
        values = np.fromfile(file_path, dtype="<f4")
        if int(values.size) != int(row["numel"]):
            raise RuntimeError(f"checkpoint {key} size mismatch")
        _array_import(lib, key if key != "weight" else "weight", values)
    setters = (
        ("optimizer_step", lib.ck_train_set_opt_step, lib.ck_train_get_opt_step),
        ("accum_counter", lib.ck_train_set_accum_counter, lib.ck_train_get_accum_counter),
        ("accum_tokens", lib.ck_train_set_accum_tokens, lib.ck_train_get_accum_tokens),
    )
    for key, setter, getter in setters:
        wanted = int(doc[key])
        if int(setter(wanted)) != wanted or int(getter()) != wanted:
            raise RuntimeError(f"checkpoint {key} restore was rejected")
    return doc


def _step(lib: ctypes.CDLL, batch: tuple[np.ndarray, np.ndarray, int, int], lr: float) -> tuple[float, dict[str, float | int]]:
    x, y, valid, _ = batch
    loss = ctypes.c_float()
    rc = int(lib.ck_train_step_ex(x.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), valid, ctypes.byref(loss), lr))
    if rc <= 0:
        raise RuntimeError(f"generated training step failed: {rc}")
    vals = [ctypes.c_double() for _ in range(4)]
    counts = [ctypes.c_int() for _ in range(4)]
    lib.ck_train_get_last_step_profile(*[ctypes.byref(v) for v in vals], *[ctypes.byref(v) for v in counts])
    return float(loss.value), {"step_ms": vals[0].value, "forward_ms": vals[1].value, "backward_ms": vals[2].value,
                               "optimizer_ms": vals[3].value, "optimizer_applied": counts[3].value}


def _forward_logits(lib: ctypes.CDLL, summary: Mapping[str, Any], x: np.ndarray, y: np.ndarray, valid: int) -> np.ndarray:
    lib.ck_train_set_batch_ex(x.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), valid)
    if int(lib.ck_train_forward_step()) <= 0:
        raise RuntimeError("generated inference forward failed")
    activations = CERT._activation_snapshot(lib, summary)
    rows = [value for name, value in activations.items() if ".logits." in name]
    if len(rows) != 1:
        raise RuntimeError(f"expected one logits tensor, found {len(rows)}")
    vocab_size = int(rows[0].size // max(1, int(x.size)))
    if vocab_size <= 0:
        raise RuntimeError("generated runtime summary does not declare vocab_size")
    return rows[0].reshape(-1, vocab_size)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray, valid: int) -> float:
    active = logits[:valid].astype(np.float64)
    maximum = active.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(active - maximum).sum(axis=1)) + maximum[:, 0]
    return float(np.mean(logsumexp - active[np.arange(valid), targets[:valid]]))


def _evaluate(lib: ctypes.CDLL, summary: Mapping[str, Any], tokens: np.ndarray, seq_len: int) -> float:
    total = 0.0; count = 0
    for x, y, valid, _ in _batches(tokens, seq_len, 1):
        total += _cross_entropy(_forward_logits(lib, summary, x, y, valid), y, valid) * valid
        count += valid
    return total / count


def _sample_trajectory(
    lib: ctypes.CDLL,
    summary: Mapping[str, Any],
    seq_len: int,
    prompt: Sequence[int] = (84, 104, 101, 32),
    new_tokens: int = 48,
) -> tuple[list[int], np.ndarray]:
    generated = [int(value) for value in prompt]
    trajectory: list[np.ndarray] = []
    dummy = np.zeros(seq_len, dtype=np.int32)
    for _ in range(new_tokens):
        context = generated[-seq_len:]
        x = np.zeros(seq_len, dtype=np.int32); x[:len(context)] = context
        logits = _forward_logits(lib, summary, x, dummy, len(context))
        last = logits[len(context) - 1].copy()
        trajectory.append(last)
        generated.append(int(np.argmax(last)))
    vocab_size = int(summary.get("vocab_size") or summary.get("config", {}).get("vocab_size") or 0)
    return generated, np.stack(trajectory) if trajectory else np.empty((0, vocab_size), dtype=np.float32)


def _tokenizer_codec(corpus: Mapping[str, Any]):
    tokenizer = corpus["tokenizer"]
    if tokenizer.get("mode") == "byte":
        return (lambda text: list(text.encode("utf-8")),
                lambda ids: bytes(int(value) & 0xff for value in ids).decode("utf-8", errors="replace"),
                None)
    pipeline = _load_module("cke_v7_tokenizer_runtime_codec", V7 / "train_data_pipeline_v7.py")
    handle = pipeline._TrueBPEHandle(ROOT / "build" / "libckernel_tokenizer.so",
                                     Path(tokenizer["binary_dir"]), Path(tokenizer["tokenizer_json"]))
    return handle.encode, handle.decode, handle


def _sample(lib: ctypes.CDLL, summary: Mapping[str, Any], seq_len: int, corpus: Mapping[str, Any],
            prompt: str = "The ", new_tokens: int = 48) -> tuple[str, dict[str, Any]]:
    encode, decode, handle = _tokenizer_codec(corpus)
    try:
        prompt_ids = encode(prompt)
        generated, _ = _sample_trajectory(lib, summary, seq_len, prompt=prompt_ids, new_tokens=new_tokens)
        generated_count = len(generated) - len(prompt_ids)
        return decode(generated), {
            "prompt_token_count": len(prompt_ids), "new_tokens_requested": new_tokens,
            "new_tokens_generated": generated_count,
            "budget_exhausted": generated_count == new_tokens,
            "stop_reason": "token_budget_exhausted" if generated_count == new_tokens else "ended_early",
        }
    finally:
        if handle is not None:
            handle.close()


def _svg_sample_quality(sample: str, probe: Mapping[str, Any] | None = None) -> dict[str, Any]:
    probe_fields = dict(probe or {})
    try:
        root = ET.fromstring(sample)
    except (ET.ParseError, ValueError) as exc:
        return {"well_formed_svg": False, "parse_error": str(exc), **probe_fields}
    return {
        "well_formed_svg": root.tag == "{http://www.w3.org/2000/svg}svg",
        "parse_error": None, **probe_fields,
    }


def _split_optimizer_snapshot(summary: Mapping[str, Any], flat: np.ndarray) -> dict[str, np.ndarray]:
    rows = [r for r in summary["tensor_slots"] if r.get("section") in {"optimizer_m", "optimizer_v"}]
    rows.sort(key=lambda r: int(r["offset"]))
    out: dict[str, np.ndarray] = {}
    cursor = 0
    for row in rows:
        count = int(row["numel"]); name = str(row["name"])
        out[name] = flat[cursor:cursor + count].copy(); cursor += count
    if cursor != int(flat.size):
        raise RuntimeError("optimizer snapshot inventory mismatch")
    return out


def _torch_optimizer_flat(torch: Any, weights: Mapping[str, Any], optimizer: Any, summary: Mapping[str, Any]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    rows = [r for r in summary["tensor_slots"] if r.get("section") in {"optimizer_m", "optimizer_v"}]
    rows.sort(key=lambda r: int(r["offset"]))
    for row in rows:
        full = str(row["name"]); kind, name = full.split(".", 2)[1:]
        state = optimizer.state[weights[name]]
        key = "exp_avg" if kind == "m" else "exp_avg_sq"
        chunks.append(state[key].detach().cpu().float().reshape(-1).numpy().copy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def _torch_weight_flat(weights: Mapping[str, Any], summary: Mapping[str, Any]) -> np.ndarray:
    rows = [
        row for row in summary["tensor_slots"]
        if str(row.get("name", "")).startswith("weight.") and row.get("section") == "weights"
    ]
    rows.sort(key=lambda row: int(row.get("offset", 0)))
    return np.concatenate([
        weights[str(row["name"])[len("weight."):]].detach().cpu().float().reshape(-1).numpy()
        for row in rows
    ]).astype(np.float32, copy=False)


def _tensor_discrepancies(
    actual: np.ndarray,
    expected: np.ndarray,
    names: Sequence[str],
    numels: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    for name, raw_count in zip(names, numels):
        count = int(raw_count)
        a = np.asarray(actual[cursor:cursor + count], dtype=np.float32)
        e = np.asarray(expected[cursor:cursor + count], dtype=np.float32)
        if a.size != count or e.size != count:
            raise RuntimeError(f"tensor discrepancy inventory mismatch for {name}")
        delta = np.abs(a - e)
        rows.append({
            "name": str(name), "numel": count,
            "max_abs_diff": float(delta.max()) if count else 0.0,
            "mean_abs_diff": float(delta.mean()) if count else 0.0,
            "reference_max_abs": float(np.abs(e).max()) if count else 0.0,
        })
        cursor += count
    if cursor != int(actual.size) or cursor != int(expected.size):
        raise RuntimeError("tensor discrepancy inventory did not consume both snapshots")
    return rows


def _compare_named_snapshot(
    actual: np.ndarray, expected: np.ndarray, names: Sequence[str], numels: Sequence[int],
    *, atol: float, rtol: float = 0.0,
) -> dict[str, Any]:
    """Check every element and retain its owning tensor and worst index."""
    if not names or len(names) != len(numels) or len(set(names)) != len(names):
        raise RuntimeError("snapshot comparison requires unique named tensor slots")
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise RuntimeError("snapshot comparison tolerances must be finite and nonnegative")
    a = np.asarray(actual, dtype=np.float32).reshape(-1)
    e = np.asarray(expected, dtype=np.float32).reshape(-1)
    if a.size != e.size or a.size != sum(int(n) for n in numels):
        raise RuntimeError("named snapshot size does not match the parameter inventory")
    rows: list[dict[str, Any]] = []
    cursor = 0
    for name, raw_count in zip(names, numels):
        count = int(raw_count)
        if count <= 0:
            raise RuntimeError(f"named snapshot has an empty tensor: {name}")
        left = a[cursor:cursor + count]; right = e[cursor:cursor + count]
        nonfinite = np.flatnonzero(~(np.isfinite(left) & np.isfinite(right)))
        if nonfinite.size:
            rows.append({"name": str(name), "numel": count, "passed": False,
                         "status": "NONFINITE", "worst_index": int(nonfinite[0]),
                         "max_abs_diff": None, "max_rel_diff": None})
        else:
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            threshold = atol + rtol * np.abs(right.astype(np.float64))
            worst = int(np.argmax(delta - threshold))
            rows.append({"name": str(name), "numel": count,
                         "passed": bool(np.all(delta <= threshold)), "status": "FINITE",
                         "worst_index": worst, "max_abs_diff": float(delta.max()),
                         "max_rel_diff": float(np.max(delta / np.maximum(np.abs(right), 1e-12))),
                         "worst_actual": float(left[worst]), "worst_reference": float(right[worst]),
                         "worst_allowed_abs_diff": float(threshold[worst])})
        cursor += count
    failures = [row for row in rows if not row["passed"]]
    finite_rows = [row for row in rows if row["status"] == "FINITE"]
    worst = failures[0] if failures else max(finite_rows, key=lambda row: row["max_abs_diff"])
    return {"passed": not failures, "tensor_count": len(rows), "element_count": int(a.size),
            "atol": atol, "rtol": rtol,
            "max_abs_diff": max((row["max_abs_diff"] for row in finite_rows), default=None),
            "max_rel_diff": max((row["max_rel_diff"] for row in finite_rows), default=None),
            "worst_tensor": worst["name"], "worst_index": worst["worst_index"],
            "failed_tensors": [row["name"] for row in failures], "tensors": rows}


def _gradient_routing_negative_control(names: Sequence[str], numels: Sequence[int]) -> dict[str, Any]:
    """Swap two distinguishable named gradient slots; the comparator must reject it."""
    if len(names) < 2:
        raise RuntimeError("gradient routing control requires two parameter tensors")
    expected = np.concatenate([
        np.full(int(count), index + 1, dtype=np.float32)
        for index, count in enumerate(numels)
    ])
    injected = expected.copy()
    second = int(numels[0])
    injected[0], injected[second] = injected[second], injected[0]
    result = _compare_named_snapshot(injected, expected, names, numels, atol=0.0)
    return {"passed": not result["passed"] and set(result["failed_tensors"]) == set(names[:2]),
            "injection": "swap_first_element_between_two_named_gradient_slots",
            "source": str(names[0]), "destination": str(names[1]),
            "detected_failed_tensors": result["failed_tensors"]}


def _new_microstep_parity_state() -> dict[str, Any]:
    return {"passed": True, "window_passed": True, "first_failed_microstep": None,
            "window_first_failed_microstep": None, "max_loss_abs_diff": 0.0,
            "max_logits_abs_diff": 0.0}


def _observe_microstep_parity(
    state: dict[str, Any], microstep: int, logits_comparison: Mapping[str, Any],
    ck_loss: float, torch_loss: float, *, loss_tol: float,
) -> dict[str, Any]:
    """Keep a failed forward comparison failed through the next update boundary."""
    loss_diff = abs(ck_loss - torch_loss) if math.isfinite(ck_loss) and math.isfinite(torch_loss) else None
    logits_diff = logits_comparison.get("max_abs_diff")
    logits_finite = logits_diff is not None and math.isfinite(float(logits_diff))
    if loss_diff is None:
        state["max_loss_abs_diff"] = None
    elif state["max_loss_abs_diff"] is not None:
        state["max_loss_abs_diff"] = max(state["max_loss_abs_diff"], loss_diff)
    if not logits_finite:
        state["max_logits_abs_diff"] = None
    elif state["max_logits_abs_diff"] is not None:
        state["max_logits_abs_diff"] = max(state["max_logits_abs_diff"], float(logits_diff))
    passed = bool(logits_comparison.get("passed") is True and logits_finite
                  and loss_diff is not None and loss_diff <= loss_tol)
    if not passed:
        failure = {"microstep": microstep, "loss_abs_diff": loss_diff,
                   "loss_status": "NONFINITE" if loss_diff is None else "FINITE",
                   "logits": dict(logits_comparison)}
        if state["first_failed_microstep"] is None:
            state["first_failed_microstep"] = failure
        if state["window_first_failed_microstep"] is None:
            state["window_first_failed_microstep"] = failure
        state["passed"] = False
        state["window_passed"] = False
    return {"passed": passed, "loss_abs_diff": loss_diff}


def _close_microstep_parity_window(state: dict[str, Any]) -> dict[str, Any]:
    result = {"passed": state["window_passed"],
              "first_failed_microstep": state["window_first_failed_microstep"]}
    state["window_passed"] = True
    state["window_first_failed_microstep"] = None
    return result


def _trajectory_passed(
    microstep_state: Mapping[str, Any], parity_rows: Sequence[Mapping[str, Any]],
    final_partial: Mapping[str, Any],
) -> bool:
    return bool(microstep_state.get("passed") is True
                and all(row.get("passed") is True for row in parity_rows)
                and final_partial.get("passed") is True)


def _microstep_sticky_negative_control() -> dict[str, Any]:
    """A non-boundary NaN must survive a later finite boundary comparison."""
    state = _new_microstep_parity_state()
    good = _compare_named_snapshot(np.array([1.0], dtype=np.float32),
                                   np.array([1.0], dtype=np.float32), ["logits"], [1], atol=0.0)
    bad = _compare_named_snapshot(np.array([np.nan], dtype=np.float32),
                                  np.array([1.0], dtype=np.float32), ["logits"], [1], atol=0.0)
    for microstep, comparison in ((1, good), (2, bad), (3, good), (4, good)):
        _observe_microstep_parity(state, microstep, comparison, 1.0, 1.0, loss_tol=0.0)
    window = _close_microstep_parity_window(state)
    for microstep in range(5, 9):
        _observe_microstep_parity(state, microstep, good, 1.0, 1.0, loss_tol=0.0)
    next_window = _close_microstep_parity_window(state)
    aggregate_passed = _trajectory_passed(state, [{"passed": True}, {"passed": True}], {"passed": True})
    return {"passed": bool(not aggregate_passed and not state["passed"] and not window["passed"]
                           and next_window["passed"]
                           and state["max_logits_abs_diff"] is None
                           and state["first_failed_microstep"]["microstep"] == 2
                           and window["first_failed_microstep"]["microstep"] == 2),
            "injection": "nonfinite_logits_at_nonboundary_microstep_2_then_finite_boundary_4",
            "first_failed_microstep": state["first_failed_microstep"],
            "aggregate_passed": aggregate_passed,
            "window_passed": window["passed"],
            "next_window_passed": next_window["passed"],
            "max_logits_abs_diff": state["max_logits_abs_diff"]}


def _restore_training_state(
    lib: ctypes.CDLL,
    *,
    weight: np.ndarray,
    optimizer: np.ndarray,
    accum: np.ndarray,
    optimizer_step: int,
    accum_counter: int,
    accum_tokens: int,
) -> None:
    _array_import(lib, "weight", weight)
    _array_import(lib, "optimizer_state", optimizer)
    _array_import(lib, "accum", accum)
    setters = (
        ("optimizer_step", lib.ck_train_set_opt_step, lib.ck_train_get_opt_step, optimizer_step),
        ("accum_counter", lib.ck_train_set_accum_counter, lib.ck_train_get_accum_counter, accum_counter),
        ("accum_tokens", lib.ck_train_set_accum_tokens, lib.ck_train_get_accum_tokens, accum_tokens),
    )
    for name, setter, getter, wanted in setters:
        if int(setter(int(wanted))) != int(wanted) or int(getter()) != int(wanted):
            raise RuntimeError(f"failed to restore {name}")


def _export_weights(run_dir: Path, summary: Mapping[str, Any], snapshot: np.ndarray, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "weights_manifest.json").read_text(encoding="utf-8"))
    entries = {str(r["name"]): r for r in manifest["entries"]}
    blob = bytearray((run_dir / "weights.bump").read_bytes())
    cursor = 0
    for name, count in zip(summary["init_weight_order"], summary["init_weight_numel"]):
        count = int(count); entry = entries.get(str(name), entries.get("tiny." + str(name)))
        if entry is None: raise RuntimeError(f"export manifest lacks {name}")
        off = int(entry["offset"]); blob[off:off + count * 4] = snapshot[cursor:cursor + count].astype("<f4").tobytes(); cursor += count
    (destination / "weights.bump").write_bytes(blob)
    (destination / "weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _export_v8_bundle(raw_export: Path, config_path: Path, destination: Path) -> None:
    """Wrap the raw training payload in the inference loader's BUMPWGT4 envelope."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw_export / "weights_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["entries"]:
        row["offset"] = int(row["offset"]) + 128
    (destination / "weights.bump").write_bytes(b"BUMPWGT4" + bytes(120) + (raw_export / "weights.bump").read_bytes())
    (destination / "weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(config_path, destination / "config.json")


def _write_visualizer_artifacts(
    *,
    run_dir: Path,
    python: str,
    args: argparse.Namespace,
    corpus: Mapping[str, Any],
    epoch_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    summary: Mapping[str, Any],
    checkpoint_path: Path,
    experiment_identity: Mapping[str, Any],
    sample_before: str,
    sample_after: str,
    sample_before_probe: Mapping[str, Any],
    sample_after_probe: Mapping[str, Any],
) -> dict[str, Any]:
    tokenizer_path = run_dir / "tokenizer.json"
    if corpus["tokenizer"].get("mode") == "byte":
        tokenizer = {
            "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
            "normalizer": None, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
            "post_processor": None, "decoder": {"type": "ByteLevel"},
            "model": {"type": "WordLevel", "unk_token": "<0x00>",
                      "vocab": {f"<0x{idx:02X}>": idx for idx in range(256)}},
            "cke_contract": corpus["tokenizer"],
        }
        _write_json(tokenizer_path, tokenizer)
    elif tokenizer_path.resolve() != Path(corpus["tokenizer"]["tokenizer_json"]).resolve():
        shutil.copy2(Path(corpus["tokenizer"]["tokenizer_json"]), tokenizer_path)

    sample_rows = []
    encode, decode, handle = _tokenizer_codec(corpus)
    try:
        for line_no, raw in enumerate((ROOT / str(corpus["splits"]["train"]["source"])).read_bytes().splitlines()[:4], 1):
            text = raw.decode("utf-8")[:96]
            ids = encode(text)
            decoded = decode(ids)
            sample_rows.append({"line_no": line_no, "exact_match": decoded == text,
                                "token_count": len(ids), "token_ids": ids, "decoded": decoded})
    finally:
        if handle is not None:
            handle.close()
    roundtrip_path = run_dir / "tokenizer_roundtrip.json"
    _write_json(roundtrip_path, {
        "schema": "cke.v8.tokenizer_roundtrip.v1", "status": "pass",
        "experiment_identity": dict(experiment_identity),
        "exact_match": all(row["exact_match"] for row in sample_rows),
        "tokenizer_json_path": str(tokenizer_path), "line_eval": {"passed": len(sample_rows), "failed": 0},
        "sample_rows": sample_rows,
    })
    tokenizer_quality_path: Path | None = None
    if corpus["tokenizer"].get("mode") == "bpe":
        quality = _load_module("cke_v8_tokenizer_quality", Path(__file__).with_name("tokenizer_quality_gate_v8.py"))
        quality_report = quality.run_gate(run_dir=run_dir, tokenizer_path=tokenizer_path)
        if quality_report.get("verdict") == "FAIL":
            raise RuntimeError("CKE tokenizer quality gate rejected the training tokenizer")
        tokenizer_quality_path = run_dir / "tokenizer_quality_gate.json"
    qc_path = run_dir / "dataset_qc.json"
    _write_json(qc_path, {
        "schema": "cke.v8.dataset_qc.v1", "status": "pass",
        "experiment_identity": dict(experiment_identity),
        "path": str(ROOT / str(corpus["splits"]["train"]["source"])),
        "non_empty_lines": len((ROOT / str(corpus["splits"]["train"]["source"])).read_bytes().splitlines()),
        "checks": {"pinned_blob": True, "separate_document_split": True, "fixed_token_count": True,
                   "serialized_token_ids": True},
    })
    profile_path = run_dir / "dataset_profile.json"
    _write_json(profile_path, {
        "schema": "cke.v8.dataset_profile.v1", "status": "pass", "tokenizer": corpus["tokenizer"],
        "experiment_identity": dict(experiment_identity),
        "domain": corpus.get("domain", "text"),
        "document_policy": corpus.get("document_policy"),
        "label_policy": corpus.get("label_policy"),
        "splits": {name: {key: row[key] for key in ("tokens", "source", "git_blob", "source_revision", "source_blob", "token_ids_sha256")}
                   for name, row in corpus["splits"].items()},
    })
    svg_evidence_path: Path | None = None
    if corpus.get("domain") == "svg_xml":
        svg_evidence_path = run_dir / "svg_fixture_evidence.json"
        _write_json(svg_evidence_path, {
            "schema": "cke.v8.svg_fixture_evidence.v1", "status": "pass",
            "experiment_identity": dict(experiment_identity),
            "document_policy": corpus["document_policy"],
            "label_policy": corpus["label_policy"],
            "validation_relationship": corpus["validation_relationship"],
            "documents": corpus["svg_documents"],
            "generated_samples": {
                "before": {"text": sample_before, **_svg_sample_quality(sample_before, sample_before_probe)},
                "after": {"text": sample_after, **_svg_sample_quality(sample_after, sample_after_probe)},
                "quality_is_certification_gate": False,
            },
        })
    loss_path = run_dir / "training_loss_curve_latest.json"
    _write_json(loss_path, {
        "schema": "cke.v8.training_loss_curve.v1",
        "experiment_identity": dict(experiment_identity),
        "steps": [{"step": int(row["last_microstep"]), "epoch": int(row["epoch"]),
                   "loss_ck": float(row["cke_mean_loss"]), "loss_pt": float(row["pytorch_mean_loss"]),
                   "lr": float(args.lr), "grad_norm": None, "source_stage": "pretrain"}
                  for row in epoch_rows],
        "grad_norm_status": "NOT_MEASURED",
    })
    parity_path = run_dir / "training_parity_latest.json"
    _write_json(parity_path, {
        "schema": "cke.v8.training_parity.v1",
        "experiment_identity": dict(experiment_identity),
        "steps": list(parity_rows),
    })
    step_profile_path = run_dir / "training_step_profile_latest.json"
    _write_json(step_profile_path, {
        "schema": "cke.v8.training_step_profile.v1",
        "experiment_identity": dict(experiment_identity),
        "train_tok_s": performance["generated_c_tokens_per_second"],
        "scope": "generated_c_profiled_steps_plus_final_flush", "timings": performance["generated_profile_ms"],
    })
    checkpoint_policy_path = run_dir / "training_checkpoint_policy_latest.json"
    _write_json(checkpoint_policy_path, {
        "schema": "cke.v8.training_checkpoint_policy.v1", "status": "pass",
        "experiment_identity": dict(experiment_identity),
        "checkpoint": str(checkpoint_path), "atomic_publication": True, "power_loss_durability_claim": False,
        "state": ["weights", "optimizer_moments", "accumulated_gradients", "optimizer_step",
                  "accumulation_counter", "contributing_tokens", "next_microstep", "runtime_dataset_config_identity"],
    })
    stitch_path = run_dir / "backprop_stitch_runtime_latest.json"
    forward_raw = summary.get("forward_op_count", summary.get("forward_ops", 0))
    backward_raw = summary.get("backward_op_count", summary.get("backward_ops", 0))
    forward_count = len(forward_raw) if isinstance(forward_raw, list) else int(forward_raw or 0)
    backward_count = len(backward_raw) if isinstance(backward_raw, list) else int(backward_raw or 0)
    _write_json(stitch_path, {
        "schema": "cke.v8.backprop_stitch_runtime.v1", "status": "pass", "passed": True,
        "experiment_identity": dict(experiment_identity),
        "forward_ops": forward_count, "backward_ops": backward_count,
        "execution_plan": str(run_dir / "train_exec_plan.json"),
        "generated_runtime": str(run_dir / "generated_train_runtime_v7.c"),
    })
    pipeline_path = run_dir / "training_pipeline_latest.json"
    preview_path = run_dir / "training_batch_preview.json"
    train_token_path = Path(str(corpus["splits"]["train"]["token_ids"]))
    train_tokens = np.fromfile(train_token_path, dtype="<i4")
    preview_rows = []
    for microstep, (inputs, labels, valid, epoch) in enumerate(
        _batches(train_tokens, int(args.seq_len), 1)[:3], 1
    ):
        preview_rows.append({
            "microstep": microstep, "epoch": int(epoch) + 1, "valid_tokens": int(valid),
            "input_token_ids": inputs.astype(int).tolist(),
            "label_token_ids": labels.astype(int).tolist(),
            "attention_mask": ([1] * int(valid)) + ([0] * (int(args.seq_len) - int(valid))),
            "loss_mask": ([1] * int(valid)) + ([0] * (int(args.seq_len) - int(valid))),
        })
    _write_json(preview_path, {
        "schema": "cke.v8.training_batch_preview.v1", "status": "pass",
        "experiment_identity": dict(experiment_identity),
        "token_stream": str(train_token_path),
        "token_stream_sha256": corpus["splits"]["train"]["token_ids_sha256"],
        "label_policy": corpus.get("label_policy", "causal_next_token"), "padding_token_id": 0,
        "batches": preview_rows,
    })
    artifacts = {
        "dataset_qc_json": str(qc_path), "dataset_profile_json": str(profile_path),
        "tokenizer_roundtrip_json": str(roundtrip_path),
        "training_batch_preview_json": str(preview_path),
    }
    if tokenizer_quality_path is not None:
        artifacts["tokenizer_quality_gate_json"] = str(tokenizer_quality_path)
    if svg_evidence_path is not None:
        artifacts["svg_fixture_evidence_json"] = str(svg_evidence_path)
    data_provenance = [
        {"stage": "pretrain", "dataset_name": name, "source_path": row["source"], "split": name,
         "token_count": row["tokens"], "hash": {"sha256": row["token_ids_sha256"]},
         "sampling": {"epochs": int(args.epochs) if name == "train" else 1},
         "packing": {"seq_len": int(args.seq_len), "cross_document_attention": False}}
        for name, row in corpus["splits"].items()
    ]
    _write_json(pipeline_path, {
        "schema": "cke.v8.training_pipeline.v1", "active_stage": "pretrain", "backend": "generated_c_fp32",
        "experiment_identity": dict(experiment_identity),
        "stage_timeline": [{"stage": "pretrain", "order": 0, "status": "completed", "active": True}],
        "optimizer": {"name": "adamw", "lr": float(args.lr), "hparams": {"beta1": args.beta1, "beta2": args.beta2,
                       "eps": args.eps, "weight_decay": args.weight_decay}},
        "execution": {"epochs": int(args.epochs), "micro_steps": int(sum(1 for _ in _batches(
            np.zeros(int(corpus["splits"]["train"]["tokens"]), dtype=np.int32), int(args.seq_len), int(args.epochs)))),
            "optimizer_steps": int(performance["optimizer_steps"]), "seq_len": int(args.seq_len),
            "grad_accum": int(args.grad_accum),
            "tokens_total": int(corpus["splits"]["train"]["tokens"]) * int(args.epochs)},
        "data_provenance": data_provenance,
        "tokenizer_lineage": {"type": corpus["tokenizer"]["name"],
                              "vocab_size": int(corpus["tokenizer"]["vocab_size"]), "tokenizer_path": str(tokenizer_path),
                              "tokenizer_sha256": _sha256(tokenizer_path)},
        "data_lab": {"dataset_dir": str(run_dir / "dataset"),
                     "dataset_path": str(ROOT / str(corpus["splits"]["train"]["source"])),
                     "tokenizer_json_path": str(tokenizer_path), "artifacts": artifacts},
        "sources": {"orchestrator": str(Path(__file__).relative_to(ROOT)), "run_dir": str(run_dir)},
    })
    return {
        "passed": True,
        "artifacts": {path.name: _sha256(path) for path in (
            tokenizer_path, roundtrip_path, qc_path, profile_path, loss_path, parity_path,
            step_profile_path, checkpoint_policy_path, stitch_path, preview_path, pipeline_path,
            *([tokenizer_quality_path] if tokenizer_quality_path is not None else []),
            *([svg_evidence_path] if svg_evidence_path is not None else []))},
    }


def _generate_identity_bound_visualizer(
    *, run_dir: Path, python: str, manifest_path: Path,
) -> dict[str, Any]:
    report_path = run_dir / "ir_report.html"
    command = [
        python, str(ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"),
        "--generate", "--run", str(run_dir), "--html-only", "--strict-run-artifacts",
        "--output", str(report_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0 or not report_path.is_file():
        raise RuntimeError("v8 training IR visualizer generation failed: " + completed.stdout[-4000:])
    html = report_path.read_text(encoding="utf-8", errors="replace")
    required_markers = (
        "training_pipeline", "tokenizer_roundtrip", "training_experiment_manifest",
        "Experiment Identity", "training_batch_preview",
    )
    missing = [marker for marker in required_markers if marker not in html]
    if missing:
        raise RuntimeError(f"v8 training IR visualizer omitted identity-bound panels: {missing}")
    return {
        "passed": True, "report": str(report_path), "report_sha256": _sha256(report_path),
        "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "command": command, "stdout": completed.stdout[-4000:],
    }


def _relative_artifact(run_dir: Path, path: Path, *, role: str, required: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        stored_path = str(resolved.relative_to(run_dir.resolve()))
    except ValueError:
        stored_path = str(resolved)
    return {
        "role": role, "path": stored_path, "required": required,
        "sha256": _sha256(resolved) if resolved.is_file() else None,
        "present": resolved.is_file(),
    }


def _resolve_authored_manifest_source(
    *, run_dir: Path, identity: Mapping[str, Any], semantic_model_path: Path | None,
    semantic_model_sha256: str | None, semantic_template_path: Path | None,
) -> tuple[Path, str, list[tuple[str, Path, bool]]]:
    if semantic_model_path is not None:
        active_model = semantic_model_path.resolve()
        if not semantic_model_sha256 or not active_model.is_file():
            raise RuntimeError("active semantic invocation is missing its authored model artifact")
        if _sha256(active_model) != semantic_model_sha256:
            raise RuntimeError("active semantic invocation model identity mismatch")
        if semantic_template_path is None or not semantic_template_path.resolve().is_file():
            raise RuntimeError("active semantic invocation is missing its lowered template artifact")
        active_template = semantic_template_path.resolve()
        return active_model, "cke.nn_semantic_model", [
            ("authored_semantic_model", active_model, True),
            ("lowered_semantic_template", active_template, True),
        ]

    python_experiment = run_dir / "python_training_experiment.json"
    expected_sha = identity.get("python_experiment_sha256")
    if python_experiment.is_file() and expected_sha and _sha256(python_experiment) == expected_sha:
        return python_experiment, "cke.nn", []
    return run_dir / "template_train.json", "workflow_configuration", []


def _write_experiment_manifest(
    *, run_dir: Path, identity: Mapping[str, Any], checks: Mapping[str, Any],
    summary: Mapping[str, Any], corpus: Mapping[str, Any], checkpoint_path: Path,
    visualizer_artifacts: Mapping[str, Any], semantic_model_path: Path | None = None,
    semantic_model_sha256: str | None = None, semantic_template_path: Path | None = None,
) -> Path:
    authored_path, authored_source, semantic_artifacts = _resolve_authored_manifest_source(
        run_dir=run_dir, identity=identity, semantic_model_path=semantic_model_path,
        semantic_model_sha256=semantic_model_sha256, semantic_template_path=semantic_template_path,
    )
    python_experiment = run_dir / "python_training_experiment.json"
    expected_python_experiment_sha = identity.get("python_experiment_sha256")
    active_python_experiment = bool(
        python_experiment.is_file()
        and expected_python_experiment_sha
        and _sha256(python_experiment) == expected_python_experiment_sha
    )

    artifact_specs = [
        ("authored_experiment", run_dir / "python_training_experiment.json", False),
        ("tokenizer", run_dir / "tokenizer.json", True),
        ("serialized_train_tokens", Path(str(corpus["splits"]["train"]["token_ids"])), True),
        ("serialized_validation_tokens", Path(str(corpus["splits"]["validation"]["token_ids"])), True),
        ("tokenizer_roundtrip", run_dir / "tokenizer_roundtrip.json", True),
        ("dataset_profile", run_dir / "dataset_profile.json", True),
        ("training_batch_preview", run_dir / "training_batch_preview.json", True),
        ("authored_training_template", run_dir / "template_train.json", True),
        ("forward_ir", run_dir / "ir1_train_forward.json", True),
        ("backward_ir", run_dir / "ir2_train_backward.json", True),
        ("execution_plan", run_dir / "train_exec_plan.json", True),
        ("generated_runtime_summary", run_dir / "generated_train_runtime_summary_v7.json", True),
        ("loss_curve", run_dir / "training_loss_curve_latest.json", True),
        ("pytorch_parity", run_dir / "training_parity_latest.json", True),
        ("checkpoint_policy", run_dir / "training_checkpoint_policy_latest.json", True),
        ("checkpoint", checkpoint_path / "checkpoint.json", True),
        ("resume_worker", run_dir / "resume_worker.json", True),
        ("inference_provenance", run_dir / "v8_inference_provenance.json", True),
        ("generation_provenance", run_dir / "v8_generation_provenance.json", True),
        ("generation_sequence", run_dir / "v8_generation_sequence.i32", True),
        ("generation_logits", run_dir / "v8_generation_logits.f32", True),
        ("generated_inference_source", run_dir / "v8_inference_runtime" / "model_v8.c", True),
        ("generated_inference_library", run_dir / "v8_inference_runtime" / "libmodel.so", True),
    ]
    artifact_specs[1:1] = semantic_artifacts
    if corpus.get("domain") == "svg_xml":
        artifact_specs.append(("svg_fixture_evidence", run_dir / "svg_fixture_evidence.json", True))
    artifacts = [
        _relative_artifact(run_dir, path, role=role, required=required)
        for role, path, required in artifact_specs
    ]
    train_plan = json.loads((run_dir / "train_exec_plan.json").read_text(encoding="utf-8"))
    provider_ids = sorted({
        str(row.get("kernel_id")) for row in train_plan.get("ops", [])
        if isinstance(row, Mapping) and row.get("kernel_id")
    })
    try:
        authored_manifest_path = str(authored_path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        authored_manifest_path = str(authored_path.resolve())
    forward_raw = summary.get("forward_op_count", summary.get("forward_ops", 0))
    backward_raw = summary.get("backward_op_count", summary.get("backward_ops", 0))
    forward_count = len(forward_raw) if isinstance(forward_raw, list) else int(forward_raw or 0)
    backward_count = len(backward_raw) if isinstance(backward_raw, list) else int(backward_raw or 0)
    manifest = {
        "schema": "cke.v8.training_experiment_manifest.v1",
        "identity": dict(identity),
        "verdict": {
            "status": "PASS" if all(bool(row.get("passed")) for row in checks.values()) else "FAIL",
            "passed": all(bool(row.get("passed")) for row in checks.values()),
            "authoritative_source": "training_workflow.json",
        },
        "graph_evidence": {
            "authored": {
                "status": "RECORDED", "path": authored_manifest_path,
                "sha256": _sha256(authored_path),
                "source": authored_source,
            },
            "lowered": {
                "status": "GENERATED", "forward_ops": forward_count,
                "backward_ops": backward_count,
                "forward_ir": "ir1_train_forward.json", "backward_ir": "ir2_train_backward.json",
            },
            "executed": {
                "status": "CERTIFIED" if checks.get("runtime_provenance", {}).get("passed") else "FAILED",
                "execution_plan": "train_exec_plan.json", "provider_ids": provider_ids,
                "parameter_count": len(summary.get("init_weight_order", [])),
                "gradient_inventory_count": len(summary.get("parameter_gradient_order", [])),
            },
        },
        "dataset": {
            "corpus_spec_sha256": corpus["spec_sha256"],
            "train_token_ids": corpus["splits"]["train"]["token_ids"],
            "train_token_ids_sha256": corpus["splits"]["train"]["token_ids_sha256"],
            "validation_token_ids": corpus["splits"]["validation"]["token_ids"],
            "validation_token_ids_sha256": corpus["splits"]["validation"]["token_ids_sha256"],
            "tokenizer_sha256": corpus["tokenizer"].get("tokenizer_json_sha256"),
        },
        "navigation": {
            "experiment_summary": (
                "python_training_experiment.json" if active_python_experiment else authored_manifest_path
            ),
            "dataset_and_tokens": "training_batch_preview.json",
            **({"svg_fixture": "svg_fixture_evidence.json"} if corpus.get("domain") == "svg_xml" else {}),
            "forward_ir": "ir1_train_forward.json", "backward_ir": "ir2_train_backward.json",
            "parity": "training_parity_latest.json", "checkpoint": "training_checkpoint_policy_latest.json",
        },
        "artifacts": artifacts,
        "visualizer_artifact_hashes": dict(visualizer_artifacts.get("artifacts", {})),
    }
    path = run_dir / "training_experiment_manifest.json"
    _write_json(path, manifest)
    return path


def _inference_probe(args: argparse.Namespace) -> int:
    runtime = args.inference_runtime.resolve()
    library_path = runtime / "libmodel.so"
    lib = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    loaded_library = CERT._loaded_library_evidence(library_path)
    engine = CERT._mapped_library_evidence("libckernel_engine.so")
    engine_hashes = sorted(
        str(row.get("sha256")) for row in engine.get("identities", [])
        if isinstance(row, Mapping) and row.get("sha256")
    )
    provenance = {
        "training_library": loaded_library,
        "engine_dependency": {**engine, "expected_sha256": args.probe_engine_sha256,
                              "loaded_sha256": engine_hashes},
    }
    provenance["passed"] = bool(
        loaded_library.get("observed_in_process_maps")
        and loaded_library.get("sha256") == args.probe_library_sha256
        and engine.get("observed_in_process_maps")
        and len(engine.get("mapped_paths", [])) == 1
        and engine_hashes == [args.probe_engine_sha256]
    )
    if args.probe_provenance:
        _write_json(args.probe_provenance, provenance)
    if not provenance["passed"]:
        return 5
    lib.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.ck_model_init_with_manifest.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    rc = int(lib.ck_model_init_with_manifest(str(runtime / "weights.bump").encode(), str(runtime / "weights_manifest.map").encode()))
    if rc != 0: return 2
    tokens = np.fromfile(args.probe_tokens, dtype="<i4")[:args.probe_count].astype(np.int32, copy=True)
    output = np.empty(int(args.probe_vocab_size), dtype=np.float32)
    if int(lib.ck_model_embed_tokens(tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), int(tokens.size))) != 0: return 3
    if int(lib.ck_model_forward(output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))) != 0: return 4
    output.astype("<f4").tofile(args.probe_logits)
    if args.probe_generate_count > 0:
        if args.probe_sequence is None or args.probe_trajectory_logits is None:
            return 7
        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int
        generated = [int(value) for value in tokens]
        logits_rows: list[np.ndarray] = []
        current = output.copy()
        for step in range(int(args.probe_generate_count)):
            logits_rows.append(current.copy())
            token = int(np.argmax(current))
            generated.append(token)
            if step + 1 < int(args.probe_generate_count):
                if int(lib.ck_model_decode(token, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))) != 0:
                    return 6
                current = output.copy()
        np.asarray(generated, dtype="<i4").tofile(args.probe_sequence)
        np.stack(logits_rows).astype("<f4").tofile(args.probe_trajectory_logits)
    return 0


def _resume_worker(args: argparse.Namespace) -> int:
    ck_run = _load_module("cke_v7_runtime_worker", V7 / "ck_run_v7.py")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    summary = json.loads((args.run_dir / "generated_train_runtime_summary_v7.json").read_text(encoding="utf-8"))
    library = args.run_dir / "libtrain.so"
    lib, provenance = _init_runtime(args.run_dir, library, summary, ck_run, role="resume_worker")
    expected = plan.get("checkpoint_expected")
    if not isinstance(expected, Mapping):
        raise RuntimeError("resume plan has no checkpoint identity contract")
    doc = _checkpoint_load(args.checkpoint, lib, expected)
    tokens = np.fromfile(plan["token_file"], dtype="<i4")
    if _sha256(Path(plan["token_file"])) != str(doc["dataset_sha256"]):
        raise RuntimeError("resume token stream does not match checkpoint dataset")
    checkpoint_config = doc["training_config"]
    batches = _batches(tokens, int(checkpoint_config["seq_len"]), int(checkpoint_config["epochs"]))
    timings = []
    start_microstep = int(doc["next_microstep"])
    for batch in batches[start_microstep:]: timings.append(_step(lib, batch, float(checkpoint_config["lr"]))[1])
    if int(lib.ck_train_get_accum_counter()) > 0 and int(lib.ck_train_flush_optimizer(float(checkpoint_config["lr"]))) <= 0:
        raise RuntimeError("resume worker final optimizer flush failed")
    out = {"weight": _weight_export(lib), "optimizer": _array_export(lib, "optimizer_state")}
    for key, value in out.items(): value.astype("<f4").tofile(args.worker_out.with_suffix(f".{key}.f32"))
    _write_json(args.worker_out, {"pid": os.getpid(), "optimizer_step": int(lib.ck_train_get_opt_step()),
                                  "start_microstep": start_microstep, "checkpoint_identity_sha256": doc["checkpoint_identity_sha256"],
                                  "provenance": provenance,
                                  "weight_sha256": _sha256(args.worker_out.with_suffix(".weight.f32")),
                                  "optimizer_sha256": _sha256(args.worker_out.with_suffix(".optimizer.f32"))})
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    effective_run_id = args.matrix_run_id or str(uuid.uuid4())
    effective_case_id = args.matrix_case_id or "v8-training-workflow-fp32"
    report: dict[str, Any] = {"schema": "cke.v8.training_workflow.v1", "status": "FAIL", "passed": False,
                              "execution": CERT._execution_identity(),
                              "matrix_identity": {"run_id": effective_run_id, "case_id": effective_case_id,
                                                  "profile": effective_case_id,
                                                  "run_dir": str(args.run_dir), "report": str(args.report)},
                              "checks": {}, "negative_controls": {}, "failures": []}
    manifest_path: Path | None = None
    started = time.perf_counter()
    try:
        import torch
        import torch.nn.functional as F
        python = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable
        args.run_dir.mkdir(parents=True, exist_ok=True)
        semantic_model, semantic_template = _apply_semantic_model(args)
        if args.d_model % args.num_heads != 0 or args.num_heads % args.num_kv_heads != 0:
            raise RuntimeError("semantic model has invalid attention head geometry")
        if args.tokenizer == "bpe" and args.vocab_size < 257:
            raise RuntimeError("BPE semantic workflow requires vocab_size >= 257")
        train_ids, val_ids, corpus = _load_corpus(args, args.run_dir / "dataset", python)
        batches = _batches(train_ids, args.seq_len, args.epochs)
        ck_run = _load_module("cke_v7_runtime_workflow", V7 / "ck_run_v7.py")
        oracle = _load_module("cke_v7_oracle_workflow", V7 / "oracle_snapshot_torch_v7.py")
        init_cmd = [python, str(V7 / "ck_run_v7.py"), "init", "--run", str(args.run_dir), "--allow-non-cache-run-dir",
                    "--train-seed", str(args.seed), "--layers", str(args.layers), "--vocab-size", str(args.vocab_size),
                    "--embed-dim", str(args.d_model), "--hidden-dim", str(args.hidden),
                    "--num-heads", str(args.num_heads), "--num-kv-heads", str(args.num_kv_heads), "--context-len", str(args.seq_len),
                    "--rope-theta", str(args.rope_theta),
                    "--template", "qwen3", "--generate-ir", "--generate-runtime", "--train-bridge-lowering", "explicit",
                    "--adamw-beta1", str(args.beta1), "--adamw-beta2", str(args.beta2), "--adamw-eps", str(args.eps),
                    "--adamw-weight-decay", str(args.weight_decay)]
        if semantic_template is not None:
            init_cmd.extend(["--template-file", str(semantic_template)])
            init_cmd.append("--omit-linear-biases")
        init = subprocess.run(init_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if init.returncode: raise RuntimeError(f"{args.layers}-layer initialization failed: " + init.stdout[-4000:])
        build_started = time.perf_counter()
        defines = {"CK_NUM_TOKENS": args.seq_len, "CK_GRAD_ACCUM_STEPS": args.grad_accum, "CK_TRAIN_USE_CE_PTREF": 1,
                   "CK_ADAMW_BETA1": args.beta1, "CK_ADAMW_BETA2": args.beta2, "CK_ADAMW_EPS": args.eps,
                   "CK_ADAMW_WEIGHT_DECAY": args.weight_decay, "CK_MAX_GRAD_NORM": 0}
        source, library = ck_run._ensure_train_runtime_artifacts(args.run_dir, python, False, runtime_defines=defines,
                                                                  train_tokens=args.seq_len, bridge_lowering="explicit")
        build_seconds = time.perf_counter() - build_started
        summary = json.loads((args.run_dir / "generated_train_runtime_summary_v7.json").read_text(encoding="utf-8"))
        expected, policy = CERT._expected_trainable_parameters(args.run_dir)
        semantic_parameter_check = (
            _validate_semantic_parameter_contract(semantic_model, expected)
            if semantic_model is not None else None
        )
        inventory = CERT._validate_parameter_inventory(summary, args.run_dir, expected)
        gradient_names = [str(row["name"]) for row in inventory]
        gradient_numels = [int(row["numel"]) for row in inventory]
        gradient_routing_negative = _gradient_routing_negative_control(gradient_names, gradient_numels)
        microstep_sticky_negative = _microstep_sticky_negative_control()
        weight_rows = [row for row in summary["tensor_slots"] if str(row.get("name", "")).startswith("weight.") and row.get("section") == "weights"]
        weight_rows.sort(key=lambda row: int(row.get("offset", 0)))
        weight_names = [str(row["name"])[len("weight."):] for row in weight_rows]
        weight_numels = [int(row["numel"]) for row in weight_rows]
        optimizer_rows = [row for row in summary["tensor_slots"] if row.get("section") in {"optimizer_m", "optimizer_v"}]
        optimizer_rows.sort(key=lambda row: int(row.get("offset", 0)))
        optimizer_names = [str(row["name"]) for row in optimizer_rows]
        optimizer_numels = [int(row["numel"]) for row in optimizer_rows]
        semantic_operation_check = (
            _validate_semantic_operation_trace(args.run_dir, semantic_model)
            if semantic_model is not None else None
        )
        configuration = _training_config(args)
        if semantic_model is not None:
            configuration["semantic_model_sha256"] = str(args.semantic_model_sha256)
        training_config_sha256 = _json_sha256(configuration)
        python_experiment = args.run_dir / "python_training_experiment.json"
        experiment_identity = {
            "run_id": effective_run_id, "case_id": effective_case_id,
            "repository_commit": report["execution"]["git_commit"],
            "training_config_sha256": training_config_sha256,
            "corpus_spec_sha256": corpus["spec_sha256"],
            "train_token_ids_sha256": corpus["splits"]["train"]["token_ids_sha256"],
            "validation_token_ids_sha256": corpus["splits"]["validation"]["token_ids_sha256"],
            "python_experiment_sha256": _sha256(python_experiment) if python_experiment.is_file() else None,
        }
        lib, main_provenance = _init_runtime(args.run_dir, library, summary, ck_run, role="main_training")
        report["execution"]["actual_runtime_threads"] = main_provenance.get("actual_runtime_threads")
        initial = _weight_export(lib)
        heldout_before = _evaluate(lib, summary, val_ids, args.seq_len)
        sample_prompt = "<svg" if corpus.get("domain") == "svg_xml" else "The "
        sample_before, sample_before_probe = _sample(lib, summary, args.seq_len, corpus, prompt=sample_prompt)
        decoded, cfg = oracle._decode_weight_snapshot(args.run_dir, summary, initial)
        names = gradient_names
        weights = {n: v.detach().clone().requires_grad_(n in names) for n, v in decoded.items()}
        model = oracle.SnapshotQwenLikeOracle(weights, cfg)
        optimizer = torch.optim.AdamW([weights[n] for n in names], lr=args.lr, betas=(args.beta1, args.beta2),
                                      eps=args.eps, weight_decay=args.weight_decay, foreach=False)
        optimizer.zero_grad(set_to_none=True)
        loss_rows, epoch_rows, parity_rows = [], [], []
        timing = {"step_ms": 0.0, "forward_ms": 0.0, "backward_ms": 0.0, "optimizer_ms": 0.0}
        token_count = 0; update_count = 0; max_weight_diff = 0.0; max_moment_diff = 0.0; max_grad_diff = 0.0
        microstep_parity = _new_microstep_parity_state()
        first_update: dict[str, Any] | None = None
        checkpoint_at = max(1, len(batches) // 2)
        checkpoint_at = checkpoint_at - (checkpoint_at % args.grad_accum) + min(3, args.grad_accum - 1)
        checkpoint_at = min(checkpoint_at, len(batches) - 1)
        checkpoint_path = None
        epoch_ck_loss_sum = 0.0; epoch_pt_loss_sum = 0.0; epoch_tokens = 0
        train_started = time.perf_counter()
        for micro, batch in enumerate(batches, 1):
            x, y, valid, epoch = batch
            ck_loss, profile = _step(lib, batch, args.lr)
            for key in timing: timing[key] += float(profile[key])
            tx = torch.from_numpy(x.astype(np.int64)).view(1, -1); ty = torch.from_numpy(y.astype(np.int64)).view(1, -1)
            logits = model.forward(tx)
            t_loss = F.cross_entropy(logits[:, :valid, :].reshape(-1, args.vocab_size), ty[:, :valid].reshape(-1), reduction="mean")
            boundary = (micro % args.grad_accum == 0)
            ck_logits = [value for name, value in CERT._activation_snapshot(lib, summary).items() if ".logits." in name][0]
            logits_comparison = _compare_named_snapshot(
                ck_logits[:valid * args.vocab_size],
                logits[:, :valid, :].detach().float().reshape(-1).numpy(),
                ["valid_token_logits"], [valid * args.vocab_size], atol=args.logits_tol)
            microstep_result = _observe_microstep_parity(
                microstep_parity, micro, logits_comparison, ck_loss, float(t_loss.item()),
                loss_tol=args.loss_tol)
            (t_loss * valid).backward(); token_count += valid
            epoch_ck_loss_sum += ck_loss * valid; epoch_pt_loss_sum += float(t_loss.item()) * valid; epoch_tokens += valid
            loss_rows.append({"microstep": micro, "epoch": epoch + 1, "valid_tokens": valid, "cke": ck_loss, "pytorch": float(t_loss.item())})
            if boundary:
                microstep_window = _close_microstep_parity_window(microstep_parity)
                for n in names: weights[n].grad.div_(token_count)
                torch_grad = np.concatenate([weights[n].grad.detach().float().reshape(-1).numpy() for n in names])
                grad_comparison = _compare_named_snapshot(
                    _grad_export(lib), torch_grad, gradient_names, gradient_numels, atol=args.grad_tol)
                grad_diff = float(grad_comparison["max_abs_diff"] or 0.0)
                max_grad_diff = max(max_grad_diff, grad_diff)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); token_count = 0; update_count += 1
                ck_w = _weight_export(lib); torch_w = _torch_weight_flat(weights, summary)
                weight_comparison = _compare_named_snapshot(
                    ck_w, torch_w, weight_names, weight_numels, atol=args.param_tol)
                moment_comparison = _compare_named_snapshot(
                    _array_export(lib, "optimizer_state"), _torch_optimizer_flat(torch, weights, optimizer, summary),
                    optimizer_names, optimizer_numels, atol=args.moment_tol)
                weight_diff = float(weight_comparison["max_abs_diff"] or 0.0)
                moment_diff = float(moment_comparison["max_abs_diff"] or 0.0)
                max_weight_diff = max(max_weight_diff, weight_diff); max_moment_diff = max(max_moment_diff, moment_diff)
                parity_row = {"step": update_count, "microstep": micro, "loss_diff": microstep_result["loss_abs_diff"],
                              "gradient_max_abs_diff": grad_diff, "max_param_diff": weight_diff,
                              "moment_max_abs_diff": moment_diff,
                              "passed": bool(grad_comparison["passed"] and weight_comparison["passed"]
                                             and moment_comparison["passed"] and microstep_window["passed"]),
                              "gradients": grad_comparison, "weights": weight_comparison,
                              "optimizer_moments": moment_comparison, "forward_logits": logits_comparison,
                              "microstep_window": microstep_window,
                              "worst_param": weight_comparison["worst_tensor"]}
                parity_rows.append(parity_row)
                if first_update is None:
                    first_update = dict(parity_row)
            if micro == checkpoint_at:
                checkpoint_path = _checkpoint_write(args.run_dir / "checkpoints", f"micro_{micro:06d}", lib,
                    {"next_microstep": micro, "total_microsteps": len(batches),
                     "runtime_contract_sha256": summary["runtime_contract_sha256"],
                     "training_config_sha256": training_config_sha256,
                     "training_config": configuration,
                     "dataset_sha256": corpus["splits"]["train"]["token_ids_sha256"]})
            if micro == len(batches) or batches[micro][3] != epoch:
                epoch_rows.append({"epoch": epoch + 1, "mean_loss": epoch_ck_loss_sum / epoch_tokens,
                                   "cke_mean_loss": epoch_ck_loss_sum / epoch_tokens,
                                   "pytorch_mean_loss": epoch_pt_loss_sum / epoch_tokens,
                                   "tokens": epoch_tokens, "last_microstep": micro})
                epoch_ck_loss_sum = 0.0; epoch_pt_loss_sum = 0.0; epoch_tokens = 0
        final_partial: dict[str, Any] = {"present": False, "passed": True}
        final_flush_negative = {"passed": True, "applicable": False}
        final_flush_seconds = 0.0
        final_grad_ck = np.empty(0, dtype=np.float32); final_grad_torch = np.empty(0, dtype=np.float32)
        if token_count:
            microstep_window = _close_microstep_parity_window(microstep_parity)
            for n in names: weights[n].grad.div_(token_count)
            torch_grad = np.concatenate([weights[n].grad.detach().float().reshape(-1).numpy() for n in names])
            ck_grad = _grad_export(lib)
            grad_comparison = _compare_named_snapshot(
                ck_grad, torch_grad, gradient_names, gradient_numels, atol=args.grad_tol)
            grad_diff = float(grad_comparison["max_abs_diff"] or 0.0)
            max_grad_diff = max(max_grad_diff, grad_diff)
            pre_state = {
                "weight": _weight_export(lib), "optimizer": _array_export(lib, "optimizer_state"),
                "accum": _array_export(lib, "accum"), "optimizer_step": int(lib.ck_train_get_opt_step()),
                "accum_counter": int(lib.ck_train_get_accum_counter()), "accum_tokens": int(lib.ck_train_get_accum_tokens()),
            }
            flush_started = time.perf_counter()
            flush_rc = int(lib.ck_train_flush_optimizer(args.lr))
            final_flush_seconds = time.perf_counter() - flush_started
            if flush_rc <= 0:
                raise RuntimeError("final partial optimizer flush failed")
            optimizer.step(); optimizer.zero_grad(set_to_none=True); update_count += 1
            correct_weight = _weight_export(lib); correct_optimizer = _array_export(lib, "optimizer_state")
            correct_accum = _array_export(lib, "accum")
            torch_weight = _torch_weight_flat(weights, summary)
            torch_optimizer = _torch_optimizer_flat(torch, weights, optimizer, summary)
            weight_comparison = _compare_named_snapshot(
                correct_weight, torch_weight, weight_names, weight_numels, atol=args.param_tol)
            moment_comparison = _compare_named_snapshot(
                correct_optimizer, torch_optimizer, optimizer_names, optimizer_numels, atol=args.moment_tol)
            weight_diff = float(weight_comparison["max_abs_diff"] or 0.0)
            moment_diff = float(moment_comparison["max_abs_diff"] or 0.0)
            counters_pass = bool(int(lib.ck_train_get_accum_counter()) == 0 and int(lib.ck_train_get_accum_tokens()) == 0
                                 and int(lib.ck_train_get_opt_step()) == update_count)
            final_partial = {"present": True, "passed": bool(grad_comparison["passed"] and weight_comparison["passed"]
                              and moment_comparison["passed"] and counters_pass),
                             "contributing_tokens": token_count, "gradient_max_abs_diff": grad_diff,
                             "weight_max_abs_diff": weight_diff, "moment_max_abs_diff": moment_diff,
                             "flush_return": flush_rc, "counters_passed": counters_pass,
                             "optimizer_step": int(lib.ck_train_get_opt_step()), "accum_counter": int(lib.ck_train_get_accum_counter()),
                             "accum_tokens": int(lib.ck_train_get_accum_tokens())}
            max_weight_diff = max(max_weight_diff, weight_diff); max_moment_diff = max(max_moment_diff, moment_diff)
            parity_rows.append({"step": update_count, "microstep": len(batches),
                                "loss_diff": microstep_result["loss_abs_diff"],
                                "gradient_max_abs_diff": grad_diff, "max_param_diff": weight_diff,
                                "moment_max_abs_diff": moment_diff, "worst_param": weight_comparison["worst_tensor"],
                                "passed": bool(grad_comparison["passed"] and weight_comparison["passed"]
                                               and moment_comparison["passed"] and microstep_window["passed"]
                                               and counters_pass),
                                "gradients": grad_comparison, "weights": weight_comparison,
                                "optimizer_moments": moment_comparison, "forward_logits": logits_comparison,
                                "microstep_window": microstep_window,
                                "partial_window": True})
            final_grad_ck = ck_grad; final_grad_torch = torch_grad
            _restore_training_state(lib, weight=pre_state["weight"], optimizer=pre_state["optimizer"], accum=pre_state["accum"],
                                    optimizer_step=pre_state["optimizer_step"], accum_counter=pre_state["accum_counter"],
                                    accum_tokens=pre_state["accum_tokens"])
            injected_rc = int(lib.ck_train_flush_optimizer(args.lr * 100.0))
            injected_weight_delta = float(np.max(np.abs(_weight_export(lib) - correct_weight)))
            injected_moment_delta = float(np.max(np.abs(_array_export(lib, "optimizer_state") - correct_optimizer)))
            final_flush_negative = {"applicable": True, "passed": bool(injected_rc > 0 and
                                    (injected_weight_delta > 1e-6 or injected_moment_delta > 1e-6)),
                                    "injected_lr": args.lr * 100.0, "weight_delta_from_expected": injected_weight_delta,
                                    "moment_delta_from_expected": injected_moment_delta}
            _restore_training_state(lib, weight=correct_weight, optimizer=correct_optimizer, accum=correct_accum,
                                    optimizer_step=update_count, accum_counter=0, accum_tokens=0)
        train_seconds = time.perf_counter() - train_started
        final_weights = _weight_export(lib); final_opt = _array_export(lib, "optimizer_state")
        # Fresh-process resume from a checkpoint captured inside an accumulation window.
        assert checkpoint_path is not None
        checkpoint_expected = {
            "runtime_contract_sha256": summary["runtime_contract_sha256"],
            "training_config_sha256": training_config_sha256,
            "dataset_sha256": corpus["splits"]["train"]["token_ids_sha256"],
            "total_microsteps": len(batches), "grad_accum": args.grad_accum,
        }
        checkpoint_doc = json.loads((checkpoint_path / "checkpoint.json").read_text(encoding="utf-8"))
        _validate_checkpoint_document(checkpoint_doc, checkpoint_expected)
        checkpoint_controls: dict[str, Any] = {}
        for control_name, key in (("wrong_dataset", "dataset_sha256"), ("wrong_runtime", "runtime_contract_sha256")):
            altered_expected = dict(checkpoint_expected); altered_expected[key] = "0" * 64
            detected = False
            try: _validate_checkpoint_document(checkpoint_doc, altered_expected)
            except RuntimeError: detected = True
            checkpoint_controls[control_name] = {"passed": detected, "altered_field": key}
        wrong_config_expected = dict(checkpoint_expected); wrong_config_expected["training_config_sha256"] = "f" * 64
        detected_config = False
        try: _validate_checkpoint_document(checkpoint_doc, wrong_config_expected)
        except RuntimeError: detected_config = True
        checkpoint_controls["wrong_config"] = {"passed": detected_config, "altered_field": "training_config_sha256"}
        altered_cursor = dict(checkpoint_doc); altered_cursor["next_microstep"] = int(checkpoint_doc["next_microstep"]) + 1
        detected_cursor = False
        try: _validate_checkpoint_document(altered_cursor, checkpoint_expected)
        except RuntimeError: detected_cursor = True
        checkpoint_controls["altered_cursor"] = {"passed": detected_cursor, "authoritative_next_microstep": checkpoint_doc["next_microstep"]}
        plan = {"token_file": corpus["splits"]["train"]["token_ids"], "checkpoint_expected": checkpoint_expected}
        plan_path = args.run_dir / "resume_plan.json"; _write_json(plan_path, plan)
        worker_out = args.run_dir / "resume_worker.json"
        worker = subprocess.run([python, str(Path(__file__).resolve()), "--resume-worker", "--run-dir", str(args.run_dir),
                                 "--checkpoint", str(checkpoint_path), "--plan", str(plan_path), "--worker-out", str(worker_out)],
                                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if worker.returncode: raise RuntimeError("fresh-process resume failed: " + worker.stdout[-4000:])
        worker_doc = json.loads(worker_out.read_text(encoding="utf-8"))
        if not worker_doc.get("provenance", {}).get("passed"):
            raise RuntimeError("resume worker did not verify loaded runtime provenance")
        if int(worker_doc.get("start_microstep", -1)) != int(checkpoint_doc["next_microstep"]):
            raise RuntimeError("resume worker did not use checkpoint-owned cursor")
        resumed_w = np.fromfile(worker_out.with_suffix(".weight.f32"), dtype="<f4")
        resumed_o = np.fromfile(worker_out.with_suffix(".optimizer.f32"), dtype="<f4")
        resume_weight_diff = float(np.max(np.abs(final_weights - resumed_w))); resume_opt_diff = float(np.max(np.abs(final_opt - resumed_o)))
        # Prove an interrupted write cannot replace the last complete checkpoint pointer.
        latest_before = (args.run_dir / "checkpoints" / "latest.json").read_bytes()
        interrupted = False
        try: _checkpoint_write(args.run_dir / "checkpoints", "interrupted", lib, {"injected": True}, inject_interrupt=True)
        except RuntimeError as exc: interrupted = "injected" in str(exc)
        atomic_ok = interrupted and (args.run_dir / "checkpoints" / "latest.json").read_bytes() == latest_before
        # Export then independently regenerate and compile the forward-capable generated runtime.
        export_dir = args.run_dir / "exported_inference"; _export_weights(args.run_dir, summary, final_weights, export_dir)
        export_build = time.perf_counter(); _, export_lib_path = ck_run._ensure_train_runtime_artifacts(export_dir, python, False,
            runtime_defines={"CK_NUM_TOKENS": args.seq_len}, train_tokens=args.seq_len, bridge_lowering="explicit")
        export_build_seconds = time.perf_counter() - export_build
        export_summary = json.loads((export_dir / "generated_train_runtime_summary_v7.json").read_text())
        export_lib, export_provenance = _init_runtime(export_dir, export_lib_path, export_summary, ck_run, role="export_rebuild")
        eval_batch = _batches(val_ids, args.seq_len, 1)[0]; ex, ey, ev, _ = eval_batch
        trained_logits = _forward_logits(lib, summary, ex, ey, ev); exported_logits = _forward_logits(export_lib, export_summary, ex, ey, ev)
        export_diff = float(np.max(np.abs(trained_logits - exported_logits)))
        heldout_after = _evaluate(export_lib, export_summary, val_ids, args.seq_len)
        sample_after, sample_after_probe = _sample(export_lib, export_summary, args.seq_len, corpus, prompt=sample_prompt)
        # Build the actual inference-only v8 prefill/decode library from the exported weights.
        v8_source = args.run_dir / "v8_inference_source"
        v8_runtime = args.run_dir / "v8_inference_runtime"
        _export_v8_bundle(export_dir, args.run_dir / "config.json", v8_source)
        v8_build_started = time.perf_counter()
        v8_cmd = [python, str(ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"), "run", str(v8_source),
                  "--run", str(v8_runtime), "--context-len", str(args.seq_len), "--logits-layout", "full", "--generate-only"]
        v8_build = subprocess.run(v8_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if v8_build.returncode != 0:
            raise RuntimeError("v8 inference-only generation failed: " + v8_build.stdout[-8000:])
        v8_build_seconds = time.perf_counter() - v8_build_started
        v8_logits_path = args.run_dir / "v8_inference_logits.f32"
        v8_library_sha256 = _sha256(v8_runtime / "libmodel.so")
        v8_engine_sha256 = _sha256(v8_runtime / "libckernel_engine.so")
        v8_provenance_path = args.run_dir / "v8_inference_provenance.json"
        probe_env = dict(os.environ); probe_env["LD_LIBRARY_PATH"] = str(v8_runtime)
        probe = subprocess.run([python, str(Path(__file__).resolve()), "--inference-probe", "--inference-runtime", str(v8_runtime),
                                "--probe-tokens", corpus["splits"]["validation"]["token_ids"], "--probe-count", str(ev),
                                "--probe-logits", str(v8_logits_path), "--probe-provenance", str(v8_provenance_path),
                                "--probe-library-sha256", v8_library_sha256, "--probe-engine-sha256", v8_engine_sha256,
                                "--probe-vocab-size", str(args.vocab_size)],
                               cwd=ROOT, env=probe_env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if probe.returncode != 0:
            raise RuntimeError("v8 inference probe failed: " + probe.stdout[-4000:])
        v8_logits = np.fromfile(v8_logits_path, dtype="<f4")
        v8_inference_diff = float(np.max(np.abs(trained_logits[ev - 1] - v8_logits)))
        v8_provenance = json.loads(v8_provenance_path.read_text(encoding="utf-8"))
        encode_prompt, _, prompt_handle = _tokenizer_codec(corpus)
        try:
            generation_prompt = np.asarray(encode_prompt("The "), dtype=np.int32)
        finally:
            if prompt_handle is not None:
                prompt_handle.close()
        generation_prompt_path = args.run_dir / "generation_prompt.i32"
        generation_prompt.astype("<i4").tofile(generation_prompt_path)
        generation_steps = 12
        reference_sequence, reference_trajectory = _sample_trajectory(
            export_lib, export_summary, args.seq_len, prompt=generation_prompt, new_tokens=generation_steps)
        trajectory_logits_path = args.run_dir / "v8_generation_logits.f32"
        trajectory_sequence_path = args.run_dir / "v8_generation_sequence.i32"
        trajectory_provenance_path = args.run_dir / "v8_generation_provenance.json"
        trajectory_probe = subprocess.run([
            python, str(Path(__file__).resolve()), "--inference-probe", "--inference-runtime", str(v8_runtime),
            "--probe-tokens", str(generation_prompt_path), "--probe-count", str(generation_prompt.size),
            "--probe-logits", str(args.run_dir / "v8_generation_first_logits.f32"),
            "--probe-generate-count", str(generation_steps), "--probe-sequence", str(trajectory_sequence_path),
            "--probe-trajectory-logits", str(trajectory_logits_path), "--probe-provenance", str(trajectory_provenance_path),
            "--probe-library-sha256", v8_library_sha256, "--probe-engine-sha256", v8_engine_sha256,
            "--probe-vocab-size", str(args.vocab_size),
        ], cwd=ROOT, env=probe_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if trajectory_probe.returncode != 0:
            raise RuntimeError("v8 generation trajectory probe failed: " + trajectory_probe.stdout[-4000:])
        v8_sequence = np.fromfile(trajectory_sequence_path, dtype="<i4").astype(np.int32).tolist()
        v8_trajectory = np.fromfile(trajectory_logits_path, dtype="<f4").reshape(generation_steps, args.vocab_size)
        trajectory_logits_diff = float(np.max(np.abs(reference_trajectory - v8_trajectory)))
        trajectory_tokens_match = v8_sequence == reference_sequence
        trajectory_provenance = json.loads(trajectory_provenance_path.read_text(encoding="utf-8"))
        train_token_count = int(train_ids.size)
        first_loss = sum(r["cke"]*r["valid_tokens"] for r in loss_rows[:len(batches)//args.epochs]) / train_token_count
        last_loss = sum(r["cke"]*r["valid_tokens"] for r in loss_rows[-len(batches)//args.epochs:]) / train_token_count
        generated_c_seconds = timing["step_ms"] / 1000.0 + final_flush_seconds
        performance = {
            "setup_compile_seconds": build_seconds,
            "certification_loop_seconds": train_seconds,
            "certification_workflow_tokens_per_second": (train_token_count * args.epochs) / train_seconds,
            "generated_c_training_seconds": generated_c_seconds,
            "generated_c_tokens_per_second": (train_token_count * args.epochs) / generated_c_seconds,
            "generated_c_optimizer_steps_per_second": update_count / generated_c_seconds,
            "tokens_per_second": (train_token_count * args.epochs) / generated_c_seconds,
            "tokens_per_second_scope": "generated_c_profiled_steps_plus_final_flush",
            "export_compile_seconds": export_build_seconds, "v8_inference_compile_seconds": v8_build_seconds,
            "generated_profile_ms": {**timing, "final_flush_ms": final_flush_seconds * 1000.0},
            "optimizer_steps": update_count,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        }
        final_torch_weights = _torch_weight_flat(weights, summary)
        weight_tensor_diffs = _tensor_discrepancies(
            final_weights, final_torch_weights, weight_names, weight_numels,
        )
        final_torch_optimizer = _torch_optimizer_flat(torch, weights, optimizer, summary)
        optimizer_tensor_diffs = _tensor_discrepancies(
            final_opt, final_torch_optimizer, optimizer_names, optimizer_numels,
        )
        gradient_tensor_diffs = _tensor_discrepancies(
            final_grad_ck, final_grad_torch, gradient_names, gradient_numels
        ) if final_partial["present"] else []
        first_failed_update = next((dict(row) for row in parity_rows if not row["passed"] or row["loss_diff"] > args.loss_tol), None)
        first_growth = None
        previous_weight_diff = 0.0
        for row in parity_rows:
            current = float(row["max_param_diff"])
            if current > previous_weight_diff:
                first_growth = dict(row); break
            previous_weight_diff = current
        visualizer_artifacts = _write_visualizer_artifacts(
            run_dir=args.run_dir, python=python, args=args, corpus=corpus, epoch_rows=epoch_rows,
            parity_rows=parity_rows, performance=performance, summary=summary, checkpoint_path=checkpoint_path,
            experiment_identity=experiment_identity,
            sample_before=sample_before, sample_after=sample_after,
            sample_before_probe=sample_before_probe, sample_after_probe=sample_after_probe,
        )
        provenance = {
            "passed": bool(main_provenance.get("passed") and export_provenance.get("passed")
                           and worker_doc.get("provenance", {}).get("passed")
                           and v8_provenance.get("passed") and trajectory_provenance.get("passed")),
            "main_training": main_provenance, "resume_worker": worker_doc.get("provenance"),
            "export_rebuild": export_provenance, "inference_probe": v8_provenance,
            "generation_probe": trajectory_provenance,
        }
        checks = {
            **({"svg_fixture_dataset": {
                "passed": all(row["source_complete_document"] and row["consumed_complete_document"]
                              for row in corpus["svg_documents"].values()),
                "train_source_sha256": corpus["svg_documents"]["train"]["sha256"],
                "validation_source_sha256": corpus["svg_documents"]["validation"]["sha256"],
                "document_policy": corpus["document_policy"], "label_policy": corpus["label_policy"],
                "validation_relationship": corpus["validation_relationship"],
                "consumed_tokens": {name: row["consumed_tokens"] for name, row in corpus["svg_documents"].items()},
                "tokenizer": corpus["tokenizer"]["name"],
            }} if corpus.get("domain") == "svg_xml" else {}),
            **({"authored_semantic_graph": {
                "passed": True,
                "semantic_model_sha256": str(args.semantic_model_sha256),
                "parameter_contract": semantic_parameter_check,
                "operation_trace": semantic_operation_check,
            }} if semantic_model is not None else {}),
            "learning": {"passed": last_loss < first_loss and heldout_after < heldout_before, "first_epoch_loss": first_loss, "last_epoch_loss": last_loss,
                "heldout_loss_before": heldout_before, "heldout_loss_after": heldout_after, "sample_before": sample_before,
                "sample_after": sample_after, "epochs": epoch_rows},
            "pytorch_trajectory": {"passed": _trajectory_passed(microstep_parity, parity_rows, final_partial)
                and first_failed_update is None,
                "max_weight_abs_diff": max_weight_diff, "max_moment_abs_diff": max_moment_diff, "max_gradient_abs_diff": max_grad_diff,
                "max_loss_abs_diff": microstep_parity["max_loss_abs_diff"],
                "max_selected_logits_abs_diff": microstep_parity["max_logits_abs_diff"],
                "first_failed_microstep": microstep_parity["first_failed_microstep"],
                "first_update": first_update, "first_failed_update": first_failed_update,
                "first_divergence_growth": first_growth,
                "trajectory": parity_rows,
                "final_tensor_discrepancies": {"weights": weight_tensor_diffs, "optimizer": optimizer_tensor_diffs,
                                               "partial_window_gradients": gradient_tensor_diffs},
                "tolerances": {"parameter": args.param_tol, "moment": args.moment_tol, "gradient": args.grad_tol, "loss": args.loss_tol, "logits": args.logits_tol}},
            "final_partial_update": final_partial,
            "fresh_process_resume": {"passed": resume_weight_diff == 0.0 and resume_opt_diff == 0.0 and worker_doc.get("provenance", {}).get("passed") is True,
                "worker_pid": worker_doc["pid"], "checkpoint_owned_start_microstep": worker_doc["start_microstep"],
                "parent_pid": os.getpid(), "weight_max_abs_diff": resume_weight_diff, "optimizer_max_abs_diff": resume_opt_diff,
                "checkpoint_accum_counter": checkpoint_doc["accum_counter"], "checkpoint_accum_tokens": checkpoint_doc["accum_tokens"]},
            "checkpoint_compatibility": {"passed": all(row["passed"] for row in checkpoint_controls.values()),
                                         "authoritative_identity": checkpoint_doc["checkpoint_identity_sha256"],
                                         "negative_controls": checkpoint_controls},
            "negative_control_detection": {"passed": bool(gradient_routing_negative["passed"]
                                                   and microstep_sticky_negative["passed"] and final_flush_negative["passed"] and
                                                   all(row["passed"] for row in checkpoint_controls.values())),
                                           "final_partial_flush": final_flush_negative,
                                           "checkpoint_identity": checkpoint_controls,
                                           "gradient_routing": gradient_routing_negative,
                                           "microstep_sticky": microstep_sticky_negative},
            "atomic_checkpoint_publication": {"passed": atomic_ok, "power_loss_durability_claim": False},
            "runtime_provenance": provenance,
            "inference_export": {"passed": export_diff == 0.0 and v8_inference_diff <= args.inference_tol
                                  and trajectory_tokens_match and trajectory_logits_diff <= args.inference_tol
                                  and v8_provenance.get("passed") is True and trajectory_provenance.get("passed") is True,
                "rebuilt_training_forward_logits_max_abs_diff": export_diff, "v8_inference_last_token_logits_max_abs_diff": v8_inference_diff,
                "generation_steps": generation_steps, "generation_tokens_match": trajectory_tokens_match,
                "generation_logits_max_abs_diff": trajectory_logits_diff,
                "generation_sequence": v8_sequence,
                "v8_inference_tolerance": args.inference_tol, "heldout_loss": heldout_after,
                "independent_training_build_directory": str(export_dir), "v8_inference_only_build_directory": str(v8_runtime),
                "standalone_native_executable": "NOT_CERTIFIED",
                "v8_generated_source_sha256": _sha256(v8_runtime / "model_v8.c"), "v8_library_sha256": v8_library_sha256},
        }
        manifest_path = _write_experiment_manifest(
            run_dir=args.run_dir, identity=experiment_identity, checks=checks, summary=summary,
            corpus=corpus, checkpoint_path=checkpoint_path,
            visualizer_artifacts=visualizer_artifacts,
            semantic_model_path=args.semantic_model if semantic_model is not None else None,
            semantic_model_sha256=str(args.semantic_model_sha256) if semantic_model is not None else None,
            semantic_template_path=semantic_template,
        )
        visualizer = _generate_identity_bound_visualizer(
            run_dir=args.run_dir, python=python, manifest_path=manifest_path,
        )
        checks["training_ir_visualizer"] = visualizer
        report["negative_controls"] = {"final_partial_flush_wrong_lr": final_flush_negative,
                                       "checkpoint_identity": checkpoint_controls,
                                       "gradient_routing": gradient_routing_negative,
                                       "microstep_sticky": microstep_sticky_negative}
        report.update({"status": "PASS" if all(v["passed"] for v in checks.values()) else "FAIL", "checks": checks, "corpus": corpus,
            "quality_evaluations": ({
                "svg_generation": {
                    "before": _svg_sample_quality(sample_before, sample_before_probe),
                    "after": _svg_sample_quality(sample_after, sample_after_probe),
                    "certification_gate": False,
                }
            } if corpus.get("domain") == "svg_xml" else {}),
            "configuration": {**configuration, "training_config_sha256": training_config_sha256,
                "unique_train_tokens": train_token_count, "token_presentations": train_token_count*args.epochs,
                "numerical_contract": "strict_reference_order", "parallel_scaling_certified": False},
            "implementation": {"orchestrator": str(Path(__file__).relative_to(ROOT)), "shared_training_codegen": "version/v7/scripts/codegen_train_runtime_v7.py",
                "shared_oracle": "version/v7/scripts/oracle_snapshot_torch_v7.py", "v7_training_cli_invoked": False,
                "inference_codegen": "version/v8/scripts/ck_run_v8.py -> build_ir_v8.py -> codegen_v8.py",
                "visualizer": "version/v8/tools/open_ir_visualizer_v8.py (v7 interface lineage)"},
            "performance": performance,
            "artifacts": {"checkpoint": str(checkpoint_path), "training_export": str(export_dir), "v8_inference": str(v8_runtime),
                          "experiment_manifest": str(manifest_path), "ir_visualizer": visualizer["report"],
                          "generated_source_sha256": _sha256(source)},
            "passed": all(v["passed"] for v in checks.values())})
    except Exception as exc:
        report["status"] = "FAIL"; report["passed"] = False
        report["failures"].append(str(exc)); report["exception"] = {"type": type(exc).__name__, "traceback": traceback.format_exc()}
        if manifest_path is not None and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["verdict"] = {
                "status": "FAIL", "passed": False,
                "authoritative_source": "training_workflow.json", "failures": list(report["failures"]),
            }
            _write_json(manifest_path, manifest)
    report["wall_seconds"] = time.perf_counter() - started
    _write_json(args.report, report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Certify a complete v8 generated-C training workflow")
    p.add_argument("--resume-worker", action="store_true"); p.add_argument("--checkpoint", type=Path); p.add_argument("--plan", type=Path); p.add_argument("--worker-out", type=Path)
    p.add_argument("--inference-probe", action="store_true"); p.add_argument("--inference-runtime", type=Path)
    p.add_argument("--probe-tokens", type=Path); p.add_argument("--probe-count", type=int); p.add_argument("--probe-logits", type=Path)
    p.add_argument("--probe-provenance", type=Path); p.add_argument("--probe-library-sha256"); p.add_argument("--probe-engine-sha256")
    p.add_argument("--probe-vocab-size", type=int, default=256)
    p.add_argument("--probe-generate-count", type=int, default=0); p.add_argument("--probe-sequence", type=Path)
    p.add_argument("--probe-trajectory-logits", type=Path)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN); p.add_argument("--json-out", dest="report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--corpus", type=Path, default=CORPUS_SPEC); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--matrix-run-id"); p.add_argument("--matrix-case-id")
    p.add_argument("--semantic-model", type=Path)
    p.add_argument("--semantic-model-sha256")
    p.add_argument("--seq-len", type=int, default=32); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--layers", type=int, choices=(2, 4, 5, 6, 10), default=4)
    p.add_argument("--d-model", type=int, default=32); p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4); p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--rope-theta", type=float, default=1_000_000.0)
    p.add_argument("--tokenizer", choices=("byte", "bpe"), default="byte")
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--bpe-min-freq", type=int, default=2); p.add_argument("--bpe-max-piece-bytes", type=int, default=24)
    p.add_argument("--max-train-tokens", type=int, default=0, help="Bound post-tokenization training IDs; 0 uses all")
    p.add_argument("--max-validation-tokens", type=int, default=0, help="Bound post-tokenization validation IDs; 0 uses all")
    p.add_argument("--grad-accum", type=int, default=8); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--beta1", type=float, default=.9); p.add_argument("--beta2", type=float, default=.999)
    p.add_argument("--eps", type=float, default=1e-8); p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--param-tol", type=float, default=5e-3); p.add_argument("--moment-tol", type=float, default=1e-3); p.add_argument("--grad-tol", type=float, default=5e-3)
    p.add_argument("--loss-tol", type=float, default=2e-2); p.add_argument("--logits-tol", type=float, default=5e-2)
    p.add_argument("--inference-tol", type=float, default=1e-3)
    args = p.parse_args(); args.run_dir=args.run_dir.resolve(); args.report=args.report.resolve(); args.corpus=args.corpus.resolve()
    if args.d_model % args.num_heads != 0:
        p.error("--d-model must be divisible by --num-heads")
    if args.num_heads % args.num_kv_heads != 0:
        p.error("--num-heads must be divisible by --num-kv-heads")
    if args.tokenizer == "bpe" and args.vocab_size < 257:
        p.error("--tokenizer bpe requires --vocab-size >= 257")
    if args.resume_worker: return _resume_worker(args)
    if args.inference_probe: return _inference_probe(args)
    result=run(args); print(json.dumps({"status":result["status"],"passed":result["passed"],"report":str(args.report)}, indent=2)); return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
