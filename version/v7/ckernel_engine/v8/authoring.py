from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ..python_authoring.export_v7 import extract_tiny_lm_contract
from ..python_authoring.graph import AuthoringGraph, build_authoring_graph
from ..python_authoring.models import qwen3_tiny
from ..python_authoring.nn import Module


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = REPO_ROOT / "version" / "v8" / "scripts" / "run_training_workflow_v8.py"
KERNEL_MAPS = REPO_ROOT / "version" / "v8" / "kernel_maps"
DEFAULT_CORPUS = REPO_ROOT / "version" / "v8" / "training" / "english_byte_v1.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "version" / "v8" / ".cache" / "python_authoring"
QWEN3_TEMPLATE = REPO_ROOT / "version" / "v7" / "templates" / "qwen3.json"

CommandRunner = Callable[[Sequence[str], Path], None]


@dataclass(frozen=True)
class DatasetConfig:
    corpus: Path = DEFAULT_CORPUS
    max_train_tokens: int = 0
    max_validation_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_train_tokens < 0 or self.max_validation_tokens < 0:
            raise ValueError("dataset token limits must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        path = Path(self.corpus).expanduser().resolve(strict=False)
        return {
            "corpus": str(path),
            "max_train_tokens": int(self.max_train_tokens),
            "max_validation_tokens": int(self.max_validation_tokens),
        }


@dataclass(frozen=True)
class TokenizerConfig:
    kind: str = "bpe"
    vocab_size: int = 384
    min_frequency: int = 2
    max_piece_bytes: int = 24

    def __post_init__(self) -> None:
        if self.kind not in {"byte", "bpe"}:
            raise ValueError("tokenizer kind must be 'byte' or 'bpe'")
        if self.kind == "byte" and self.vocab_size != 256:
            raise ValueError("byte tokenizer requires vocab_size=256")
        if self.vocab_size <= 0 or self.min_frequency <= 0 or self.max_piece_bytes <= 0:
            raise ValueError("tokenizer sizes and frequency must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "vocab_size": int(self.vocab_size),
            "min_frequency": int(self.min_frequency),
            "max_piece_bytes": int(self.max_piece_bytes),
        }


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    grad_accum: int = 8
    seed: int = 42
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    parameter_tolerance: float = 5e-3
    moment_tolerance: float = 1e-3
    gradient_tolerance: float = 5e-3
    loss_tolerance: float = 2e-2
    logits_tolerance: float = 5e-2
    inference_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.grad_accum <= 0:
            raise ValueError("epochs and grad_accum must be > 0")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW beta values must be in [0, 1)")
        optimizer_values = (
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay,
        )
        if any(not math.isfinite(value) for value in optimizer_values):
            raise ValueError("optimizer values must be finite")
        if self.learning_rate <= 0.0 or self.epsilon <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer learning rate/epsilon must be > 0 and weight decay >= 0")
        tolerances = (
            self.parameter_tolerance, self.moment_tolerance, self.gradient_tolerance,
            self.loss_tolerance, self.logits_tolerance, self.inference_tolerance,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("numerical tolerances must be finite and >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": int(self.epochs), "grad_accum": int(self.grad_accum), "seed": int(self.seed),
            "optimizer": "generated_c_adamw", "learning_rate": float(self.learning_rate),
            "beta1": float(self.beta1), "beta2": float(self.beta2), "epsilon": float(self.epsilon),
            "weight_decay": float(self.weight_decay),
            "tolerances": {
                "parameter": float(self.parameter_tolerance), "moment": float(self.moment_tolerance),
                "gradient": float(self.gradient_tolerance), "loss": float(self.loss_tolerance),
                "logits": float(self.logits_tolerance), "inference": float(self.inference_tolerance),
            },
        }


# Semantic requirements of the currently supported dense/GQA lowering. These are
# operation requirements, not provider selections; providers remain owned by the
# v8 kernel maps loaded below.
_DENSE_TRAINING_REQUIREMENTS = (
    ("token embedding", "embedding", "forward"),
    ("token embedding gradient", "embedding_backward", "backward"),
    ("linear projections", "gemm", "forward"),
    ("linear projection gradients", "gemm_backward", "backward"),
    ("RMSNorm", "rmsnorm", "forward"),
    ("RMSNorm gradient", "rmsnorm", "backward"),
    ("causal GQA", "attention", "forward"),
    ("causal GQA gradient", "attention", "backward"),
    ("RoPE", "rope", "forward"),
    ("RoPE gradient", "rope_backward", "backward"),
    ("SwiGLU", "swiglu", "forward"),
    ("SwiGLU gradient", "swiglu", "backward"),
    ("residual add", "residual_add", "forward"),
    ("residual split gradient", "add_backward", "backward"),
    ("cross entropy and logits gradient", "softmax", "backward"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_runner(command: Sequence[str], cwd: Path) -> None:
    subprocess.run([str(part) for part in command], cwd=str(cwd), check=True)


def _is_backward(document: Mapping[str, Any]) -> bool:
    modes = document.get("modes") if isinstance(document.get("modes"), Mapping) else {}
    return bool(modes.get("backward"))


def _is_fp32(document: Mapping[str, Any]) -> bool:
    quant = document.get("quant") if isinstance(document.get("quant"), Mapping) else {}
    values = {str(quant.get("activation", "")), str(quant.get("output", ""))}
    return "fp32" in values or str(document.get("variant", "")).startswith("fp32")


def _has_test_evidence(document: Mapping[str, Any]) -> bool:
    tests = document.get("tests") if isinstance(document.get("tests"), Mapping) else {}
    return bool(tests.get("parity") or tests.get("unit"))


def _load_kernel_maps() -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(KERNEL_MAPS.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and document.get("id") and document.get("op"):
            rows.append((path, document))
    return rows


def _candidate_capability_inventory() -> dict[str, Any]:
    maps = _load_kernel_maps()
    capabilities: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for label, op, direction in _DENSE_TRAINING_REQUIREMENTS:
        want_backward = direction == "backward"
        candidates = []
        for path, document in maps:
            if document.get("op") != op or _is_backward(document) != want_backward:
                continue
            if not _is_fp32(document) or not _has_test_evidence(document):
                continue
            tests = document.get("tests") if isinstance(document.get("tests"), Mapping) else {}
            implementation = document.get("impl") if isinstance(document.get("impl"), Mapping) else {}
            tensor_contract = {
                key: document.get(key, [])
                for key in ("inputs", "weights", "activations", "outputs", "scratch")
            }
            state_tensors = []
            for group, tensors in tensor_contract.items():
                if not isinstance(tensors, list):
                    continue
                for tensor in tensors:
                    if isinstance(tensor, Mapping) and "state" in str(tensor.get("name", "")).lower():
                        state_tensors.append({"group": group, **dict(tensor)})
            candidates.append({
                "provider": document["id"], "map": str(path.relative_to(REPO_ROOT)),
                "variant": document.get("variant"), "modes": document.get("modes", {}),
                "operation_interface": document.get("operation_interface"),
                "numerical_contract": document.get("numerical_contract"),
                "dimensions": document.get("dims", []), "constraints": document.get("constraints", {}),
                "tensor_contract": tensor_contract, "persistent_state_tensors": state_tensors,
                "saved_for_backward": document.get("saved_for_backward", "NOT_DECLARED_IN_KERNEL_MAP"),
                "implementation": {
                    "function": implementation.get("function"),
                    "sources": implementation.get("sources", []),
                },
                "registered_tests": tests,
                "test_evidence_status": "REGISTERED_NOT_EXECUTED_BY_PREFLIGHT",
                "provider_selection_status": "CANDIDATE_NOT_RESOLVED",
            })
        row = {"label": label, "op": op, "direction": direction, "candidates": candidates}
        capabilities.append(row)
        if not candidates:
            missing.append({
                "label": label, "op": op, "direction": direction,
                "action": f"add an FP32 {direction} kernel map with unit/parity evidence for op={op}",
            })
    return {
        "schema": "cke.v8.training_capability_inventory.v1",
        "status": "CANDIDATE_INVENTORY_COMPLETE" if not missing else "CANDIDATE_INVENTORY_INCOMPLETE",
        "candidate_inventory_complete": not missing,
        "can_launch_generated_workflow": not missing,
        "resolved_plan_status": "PENDING_GENERATED_WORKFLOW",
        "executed_numerical_evidence": "PENDING_GENERATED_WORKFLOW",
        "passed": False,
        "kernel_map_root": str(KERNEL_MAPS), "capabilities": capabilities, "missing": missing,
    }


def _module_signature(model: Module) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for path, module in model.named_modules():
        signature[path or "<root>"] = {
            "type": module.__class__.__name__,
            "config": module.spec(),
            "children": [name for name, _child in module.named_children()],
            "parameters": {
                name: parameter.to_dict()
                for name, parameter in module.named_parameters(recurse=False)
            },
        }
    return signature


def _validate_supported_semantics(model: Module, contract: Mapping[str, Any]) -> None:
    rope_theta = float(contract["rope_theta"])
    if not math.isfinite(rope_theta) or rope_theta <= 0.0:
        raise ValueError("v8 generated training requires a finite positive authored rope_theta")
    expected = qwen3_tiny(
        vocab=int(contract["embedding"].vocab), dim=int(contract["dim"]),
        layers=int(contract["layers"]), hidden=int(contract["hidden"]),
        heads=int(contract["heads"]), kv_heads=int(contract["kv_heads"]),
        context_len=int(contract["context_len"]), rope_theta=rope_theta,
        init="normal_0p02", dtype="float32", name=model.name,
    )
    actual_signature = _module_signature(model)
    expected_signature = _module_signature(expected)
    if actual_signature == expected_signature:
        return
    paths = sorted(set(actual_signature) | set(expected_signature))
    for path in paths:
        actual = actual_signature.get(path)
        wanted = expected_signature.get(path)
        if actual != wanted:
            raise ValueError(
                "v8 generated training cannot preserve authored semantics at "
                f"{path}: authored={actual!r}, supported={wanted!r}"
            )
    raise ValueError("v8 generated training cannot preserve the authored model semantics")


def _node_id_by_scope(graph: AuthoringGraph) -> dict[str, str]:
    return {node.scope: node.id for node in graph.nodes}


def _semantic_parameter_contract(graph: AuthoringGraph, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map authored parameters to the generated runtime's stable manifest names."""
    scopes = _node_id_by_scope(graph)
    rows: list[dict[str, Any]] = []

    def add(authored_scope: str, authored_name: str, runtime_name: str, shape: Sequence[int]) -> None:
        rows.append({
            "authored_semantic_id": scopes[authored_scope],
            "authored_parameter": f"{authored_scope}.{authored_name}",
            "runtime_parameter": runtime_name,
            "shape": [int(value) for value in shape],
            "dtype": "fp32",
            "trainable": True,
        })

    dim = int(contract["dim"]); hidden = int(contract["hidden"])
    heads = int(contract["heads"]); kv_heads = int(contract["kv_heads"])
    vocab = int(contract["embedding"].vocab); head_dim = dim // heads
    add("0", "weight", "token_emb", (vocab, dim))
    for layer in range(int(contract["layers"])):
        base = str(layer + 1)
        add(f"{base}.attn_norm", "weight", f"layer.{layer}.ln1_gamma", (dim,))
        # The Python Attention surface exposes a packed logical qkv parameter. The
        # generated runtime materializes its independently addressable projections.
        attention_id = scopes[f"{base}.attention"]
        for authored_name, runtime_name, shape in (
            ("qkv_weight:q", f"layer.{layer}.wq", (dim, dim)),
            ("qkv_weight:k", f"layer.{layer}.wk", (kv_heads * head_dim, dim)),
            ("qkv_weight:v", f"layer.{layer}.wv", (kv_heads * head_dim, dim)),
            ("out_weight", f"layer.{layer}.wo", (dim, dim)),
            ("q_norm_weight", f"layer.{layer}.q_norm", (head_dim,)),
            ("k_norm_weight", f"layer.{layer}.k_norm", (head_dim,)),
        ):
            rows.append({
                "authored_semantic_id": attention_id,
                "authored_parameter": f"{base}.attention.{authored_name}",
                "runtime_parameter": runtime_name,
                "shape": list(shape), "dtype": "fp32", "trainable": True,
            })
        add(f"{base}.ffn_norm", "weight", f"layer.{layer}.ln2_gamma", (dim,))
        add(f"{base}.feed_forward", "gate_up_weight", f"layer.{layer}.w1", (2 * hidden, dim))
        add(f"{base}.feed_forward", "down_weight", f"layer.{layer}.w2", (dim, hidden))
    add(str(int(contract["layers"]) + 1), "weight", "final_ln_weight", (dim,))
    add(str(int(contract["layers"]) + 2), "weight", "output.weight", (vocab, dim))
    return rows


def _semantic_lowering_document(graph: AuthoringGraph, contract: Mapping[str, Any]) -> dict[str, Any]:
    scopes = _node_id_by_scope(graph)
    layers = []
    for layer in range(int(contract["layers"])):
        scope = str(layer + 1)
        layers.append({
            "index": layer,
            "block": scopes[scope],
            "block_label": next(node.label for node in graph.nodes if node.scope == scope),
            "attn_norm": scopes[f"{scope}.attn_norm"],
            "attention": scopes[f"{scope}.attention"],
            "ffn_norm": scopes[f"{scope}.ffn_norm"],
            "feed_forward": scopes[f"{scope}.feed_forward"],
            "rope_theta": float(contract["rope_theta"]),
        })
    model_contract = {
        "family": "qwen3_style_dense_reduced", "dtype": "fp32",
        "layers": int(contract["layers"]), "d_model": int(contract["dim"]),
        "hidden": int(contract["hidden"]), "heads": int(contract["heads"]),
        "kv_heads": int(contract["kv_heads"]), "seq_len": int(contract["context_len"]),
        "vocab_size": int(contract["embedding"].vocab),
        "rope_theta": float(contract["rope_theta"]), "activation": str(contract["activation"]),
        "bias": False, "normalization": "rmsnorm", "norm_epsilon": 1e-6,
        "initialization": "normal_0p02",
    }
    template = copy.deepcopy(json.loads(QWEN3_TEMPLATE.read_text(encoding="utf-8")))
    trace = {
        "schema": "cke.v8.python_semantic_trace.v1",
        "root": graph.root_id, "embedding": scopes["0"],
        "layers": layers,
        "final_norm": scopes[str(int(contract["layers"]) + 1)],
        "lm_head": scopes[str(int(contract["layers"]) + 2)],
    }
    template["python_semantic_lowering"] = trace
    return {
        "schema": "cke.v8.python_semantic_model.v1",
        "graph": graph.to_dict(), "model_contract": model_contract,
        "supported_semantics": {
            "topology": f"Embedding -> ordered TransformerBlock[{int(contract['layers'])}] -> RMSNorm -> Linear",
            "attention": ["dense", "unequal_head_gqa"], "activation": ["swiglu"],
            "bias": [False], "normalization": [{"kind": "rmsnorm", "epsilon": 1e-6}],
            "dtype": ["fp32"], "initialization": ["normal_0p02"],
            "rope_theta": "finite_positive_global_value",
        },
        "operation_trace": trace,
        "parameter_contract": _semantic_parameter_contract(graph, contract),
        "template": template,
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return completed.stdout.strip()


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_artifact_sha256(value: Mapping[str, Any]) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class CompiledTrainingExperiment:
    model: Module
    graph: AuthoringGraph
    contract: Mapping[str, Any]
    run_name: str
    run_dir: Path
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig
    command_runner: CommandRunner = _default_runner
    python: str = sys.executable

    @property
    def report_path(self) -> Path:
        return self.run_dir / "training_workflow.json"

    @property
    def experiment_path(self) -> Path:
        return self.run_dir / "python_training_experiment.json"

    @property
    def preflight_path(self) -> Path:
        return self.run_dir / "training_capability_preflight.json"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "training_experiment_manifest.json"

    @property
    def semantic_model_path(self) -> Path:
        return self.run_dir / "python_training_semantic_model.json"

    def semantic_model_document(self) -> dict[str, Any]:
        return _semantic_lowering_document(self.graph, self.contract)

    def experiment_document(self) -> dict[str, Any]:
        corpus = Path(self.dataset.corpus).expanduser().resolve(strict=False)
        return {
            "schema": "cke.v8.python_training_experiment.v1", "run_name": self.run_name,
            "frontend": "ckernel_engine.v8", "execution_backend": "generated_c",
            "model": {
                "family": "qwen3_style_dense_reduced", "dtype": "fp32",
                "layers": self.contract["layers"], "d_model": self.contract["dim"],
                "hidden": self.contract["hidden"], "heads": self.contract["heads"],
                "kv_heads": self.contract["kv_heads"], "seq_len": self.contract["context_len"],
                "rope_theta": self.contract["rope_theta"],
                "semantic_model": str(self.semantic_model_path),
                "graph": self.graph.to_dict(),
            },
            "dataset": {**self.dataset.to_dict(), "sha256": _sha256(corpus) if corpus.is_file() else None},
            "tokenizer": self.tokenizer.to_dict(), "training": self.training.to_dict(),
            "deployment": {
                "inference_runtime": "independently_generated_v8_c",
                "python_required_by_generated_runtime": False,
                "standalone_native_executable": "NOT_CERTIFIED",
            },
            "unsupported": ["RWKV", "arbitrary torch.nn graph import", "Python arithmetic fallback"],
        }

    def expected_workflow_configuration(self) -> dict[str, Any]:
        c = self.contract; t = self.tokenizer; train = self.training
        return {
            "architecture": "qwen3_style_dense_reduced", "layers": int(c["layers"]),
            "dtype": "fp32", "d_model": int(c["dim"]), "hidden": int(c["hidden"]),
            "heads": int(c["heads"]), "kv_heads": int(c["kv_heads"]),
            "rope_theta": float(c["rope_theta"]),
            "vocab_size": int(t.vocab_size), "tokenizer": t.kind,
            "seq_len": int(c["context_len"]), "epochs": int(train.epochs),
            "grad_accum": int(train.grad_accum), "optimizer": "generated_c_adamw",
            "lr": float(train.learning_rate), "beta1": float(train.beta1),
            "beta2": float(train.beta2), "eps": float(train.epsilon),
            "weight_decay": float(train.weight_decay),
            "semantic_model_sha256": _json_artifact_sha256(self.semantic_model_document()),
        }

    def command(
        self, *, invocation_id: Optional[str] = None, experiment_sha256: Optional[str] = None,
    ) -> list[str]:
        d = self.dataset; t = self.tokenizer; train = self.training
        semantic_sha256 = _json_artifact_sha256(self.semantic_model_document())
        command = [
            self.python, str(WORKFLOW), "--run-dir", str(self.run_dir), "--json-out", str(self.report_path),
            "--corpus", str(Path(d.corpus).expanduser().resolve(strict=False)), "--seed", str(train.seed),
            "--semantic-model", str(self.semantic_model_path),
            "--semantic-model-sha256", semantic_sha256,
            "--epochs", str(train.epochs),
            "--tokenizer", t.kind, "--vocab-size", str(t.vocab_size),
            "--bpe-min-freq", str(t.min_frequency), "--bpe-max-piece-bytes", str(t.max_piece_bytes),
            "--max-train-tokens", str(d.max_train_tokens),
            "--max-validation-tokens", str(d.max_validation_tokens), "--grad-accum", str(train.grad_accum),
            "--lr", str(train.learning_rate), "--beta1", str(train.beta1), "--beta2", str(train.beta2),
            "--eps", str(train.epsilon), "--weight-decay", str(train.weight_decay),
            "--param-tol", str(train.parameter_tolerance), "--moment-tol", str(train.moment_tolerance),
            "--grad-tol", str(train.gradient_tolerance), "--loss-tol", str(train.loss_tolerance),
            "--logits-tol", str(train.logits_tolerance), "--inference-tol", str(train.inference_tolerance),
        ]
        if (invocation_id is None) != (experiment_sha256 is None):
            raise ValueError("invocation_id and experiment_sha256 must be supplied together")
        if invocation_id is not None and experiment_sha256 is not None:
            case_id = f"python-authoring:{experiment_sha256}"
            command.extend(["--matrix-run-id", invocation_id, "--matrix-case-id", case_id])
        return command

    def preflight(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        semantic_model = self.semantic_model_document()
        self.semantic_model_path.write_text(
            json.dumps(semantic_model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        experiment = self.experiment_document()
        self.experiment_path.write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = _candidate_capability_inventory()
        report["experiment"] = str(self.experiment_path)
        report["experiment_sha256"] = _sha256(self.experiment_path)
        report["semantic_model"] = str(self.semantic_model_path)
        report["semantic_model_sha256"] = _sha256(self.semantic_model_path)
        self.preflight_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not report["can_launch_generated_workflow"]:
            details = "; ".join(row["action"] for row in report["missing"])
            raise RuntimeError(f"v8 training capability preflight failed: {details}")
        return report

    def inspect_existing(self) -> dict[str, Any]:
        """Read and validate an existing experiment without modifying its artifacts."""
        paths = {
            "experiment": self.experiment_path, "preflight": self.preflight_path,
            "report": self.report_path, "manifest": self.manifest_path,
        }
        documents: dict[str, Any] = {}
        failures: list[str] = []
        for name, path in paths.items():
            if not path.is_file():
                failures.append(f"missing:{name}")
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                failures.append(f"malformed:{name}")
                continue
            if not isinstance(value, dict):
                failures.append(f"not_object:{name}")
                continue
            documents[name] = value
        report = documents.get("report", {})
        manifest = documents.get("manifest", {})
        identity = manifest.get("identity") if isinstance(manifest.get("identity"), Mapping) else {}
        matrix = report.get("matrix_identity") if isinstance(report.get("matrix_identity"), Mapping) else {}
        configuration = report.get("configuration") if isinstance(report.get("configuration"), Mapping) else {}
        corpus = report.get("corpus") if isinstance(report.get("corpus"), Mapping) else {}
        if manifest and manifest.get("schema") != "cke.v8.training_experiment_manifest.v1":
            failures.append("manifest.schema")
        for key in ("run_id", "case_id"):
            if identity.get(key) != matrix.get(key):
                failures.append(f"identity.{key}")
        if identity.get("training_config_sha256") != configuration.get("training_config_sha256"):
            failures.append("identity.training_config_sha256")
        if identity.get("corpus_spec_sha256") != corpus.get("spec_sha256"):
            failures.append("identity.corpus_spec_sha256")
        train_split = corpus.get("splits", {}).get("train", {}) if isinstance(corpus.get("splits"), Mapping) else {}
        if identity.get("train_token_ids_sha256") != train_split.get("token_ids_sha256"):
            failures.append("identity.train_token_ids_sha256")
        experiment = documents.get("experiment")
        if experiment is not None:
            if identity.get("python_experiment_sha256") != _sha256(self.experiment_path):
                failures.append("identity.python_experiment_sha256")
            if experiment != self.experiment_document():
                failures.append("current_authored_experiment")
        verdict = manifest.get("verdict") if isinstance(manifest.get("verdict"), Mapping) else {}
        if verdict.get("status") != report.get("status") or verdict.get("passed") != report.get("passed"):
            failures.append("verdict")
        artifact_rows = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
        checked_artifacts = []
        for row in artifact_rows:
            if not isinstance(row, Mapping):
                failures.append("artifact.malformed")
                continue
            raw_path = row.get("path")
            artifact_path = Path(str(raw_path)).expanduser() if raw_path else Path()
            if raw_path and not artifact_path.is_absolute():
                artifact_path = self.run_dir / artifact_path
            present = bool(raw_path and artifact_path.is_file())
            observed = _sha256(artifact_path) if present else None
            matched = present and observed == row.get("sha256")
            if matched and artifact_path.suffix == ".json":
                try:
                    child = json.loads(artifact_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    child = None
                child_identity = child.get("experiment_identity") if isinstance(child, dict) else None
                if isinstance(child_identity, Mapping):
                    for key in ("run_id", "case_id", "training_config_sha256", "train_token_ids_sha256"):
                        if child_identity.get(key) != identity.get(key):
                            matched = False
                            failures.append(f"artifact_identity:{row.get('role')}:{key}")
                            break
            checked_artifacts.append({
                "role": row.get("role"), "path": str(artifact_path), "present": present,
                "matched": matched, "expected_sha256": row.get("sha256"), "observed_sha256": observed,
            })
            if row.get("required") is True and not matched:
                failures.append(f"artifact:{row.get('role')}")
            elif present and row.get("sha256") and observed != row.get("sha256"):
                failures.append(f"artifact_hash:{row.get('role')}")
        return {
            "status": "MATCHED" if not failures else "MISMATCH",
            "identity_matched": not failures, "failures": sorted(set(failures)),
            "historical": True, "read_only": True, "documents": documents,
            "checked_artifacts": checked_artifacts,
        }

    def run(self) -> dict[str, Any]:
        preflight = self.preflight()
        experiment_sha256 = str(preflight["experiment_sha256"])
        invocation_id = str(uuid.uuid4())
        case_id = f"python-authoring:{experiment_sha256}"
        try:
            self.report_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"cannot clear prior workflow report {self.report_path}: {exc}") from exc
        started_ns = time.time_ns()
        command = self.command(invocation_id=invocation_id, experiment_sha256=experiment_sha256)
        try:
            self.command_runner(command, REPO_ROOT)
        except subprocess.CalledProcessError as exc:
            if self.report_path.is_file():
                try:
                    report = json.loads(self.report_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as parse_exc:
                    raise RuntimeError(
                        f"generated-C workflow failed and published malformed report={self.report_path}: {parse_exc}"
                    ) from exc
                details = report.get("failures") or report.get("exception") or report.get("status")
                raise RuntimeError(
                    f"generated-C workflow failed; report={self.report_path}; details={details}"
                ) from exc
            raise RuntimeError(
                f"generated-C workflow exited with {exc.returncode} before publishing {self.report_path}"
            ) from exc
        if not self.report_path.is_file():
            raise RuntimeError(f"generated-C workflow did not publish {self.report_path}")
        try:
            report = json.loads(self.report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"generated-C workflow published malformed report={self.report_path}: {exc}") from exc
        if not isinstance(report, dict):
            raise RuntimeError(f"generated-C workflow report root is not an object: {self.report_path}")
        identity_errors: list[str] = []
        if self.report_path.stat().st_mtime_ns < started_ns:
            identity_errors.append("report_stale_mtime")
        if report.get("schema") != "cke.v8.training_workflow.v1":
            identity_errors.append("schema")
        if report.get("status") != "PASS" or report.get("passed") is not True:
            identity_errors.append("workflow_status")
        identity = report.get("matrix_identity") if isinstance(report.get("matrix_identity"), Mapping) else {}
        for key, expected in {
            "run_id": invocation_id, "case_id": case_id, "profile": case_id,
            "run_dir": str(self.run_dir), "report": str(self.report_path),
        }.items():
            if identity.get(key) != expected:
                identity_errors.append(f"matrix_identity.{key}")
        execution = report.get("execution") if isinstance(report.get("execution"), Mapping) else {}
        if execution.get("git_commit") != _git_commit():
            identity_errors.append("execution.git_commit")
        configuration = report.get("configuration") if isinstance(report.get("configuration"), Mapping) else {}
        expected_configuration = self.expected_workflow_configuration()
        for key, expected in expected_configuration.items():
            if configuration.get(key) != expected:
                identity_errors.append(f"configuration.{key}")
        if configuration.get("training_config_sha256") != _json_sha256(expected_configuration):
            identity_errors.append("configuration.training_config_sha256")
        corpus = report.get("corpus") if isinstance(report.get("corpus"), Mapping) else {}
        corpus_path = Path(self.dataset.corpus).expanduser().resolve(strict=False)
        if corpus.get("spec_sha256") != _sha256(corpus_path):
            identity_errors.append("corpus.spec_sha256")
        if identity_errors:
            raise RuntimeError(
                f"generated-C workflow report identity mismatch: {sorted(set(identity_errors))}; "
                f"inspect {self.report_path}"
            )
        preflight.update({
            "status": "RESOLVED_EXECUTION_PASS", "passed": True,
            "resolved_plan_status": "VALIDATED_BY_WORKFLOW_REPORT",
            "executed_numerical_evidence": "PASS", "invocation_id": invocation_id,
            "workflow_report": str(self.report_path), "workflow_report_sha256": _sha256(self.report_path),
        })
        self.preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


def compile(
    model: Module, *, run_name: str, run_dir: Optional[Path] = None,
    dataset: Optional[DatasetConfig] = None, tokenizer: Optional[TokenizerConfig] = None,
    training: Optional[TrainingConfig] = None, command_runner: CommandRunner = _default_runner,
    python: str = sys.executable,
) -> CompiledTrainingExperiment:
    contract = extract_tiny_lm_contract(model, frontend="v8")
    if contract["layers"] not in {4, 5, 6, 10}:
        raise ValueError("v8 generated training currently supports exactly 4, 5, 6, or 10 dense/GQA layers")
    if any(str(parameter.dtype) != "float32" for parameter in model.parameters()):
        raise ValueError("v8 generated training authoring currently supports FP32 parameters only")
    _validate_supported_semantics(model, contract)
    token_cfg = tokenizer or TokenizerConfig()
    if contract["embedding"].vocab != token_cfg.vocab_size:
        raise ValueError("model vocabulary must match tokenizer vocab_size")
    resolved_run_dir = Path(run_dir or (DEFAULT_RUN_ROOT / run_name)).expanduser().resolve(strict=False)
    return CompiledTrainingExperiment(
        model=model, graph=build_authoring_graph(model, name=run_name), contract=contract,
        run_name=str(run_name), run_dir=resolved_run_dir, dataset=dataset or DatasetConfig(),
        tokenizer=token_cfg, training=training or TrainingConfig(), command_runner=command_runner,
        python=str(python),
    )
