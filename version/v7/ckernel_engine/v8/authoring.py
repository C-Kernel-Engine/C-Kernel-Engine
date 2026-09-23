from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ..python_authoring.export_v7 import extract_tiny_lm_contract
from ..python_authoring.graph import AuthoringGraph, build_authoring_graph
from ..python_authoring.nn import Module


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = REPO_ROOT / "version" / "v8" / "scripts" / "run_training_workflow_v8.py"
KERNEL_MAPS = REPO_ROOT / "version" / "v8" / "kernel_maps"
DEFAULT_CORPUS = REPO_ROOT / "version" / "v8" / "training" / "english_byte_v1.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "version" / "v8" / ".cache" / "python_authoring"

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
        if self.learning_rate <= 0.0 or self.epsilon <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer learning rate/epsilon must be > 0 and weight decay >= 0")
        tolerances = (
            self.parameter_tolerance, self.moment_tolerance, self.gradient_tolerance,
            self.loss_tolerance, self.logits_tolerance, self.inference_tolerance,
        )
        if any(value < 0.0 for value in tolerances):
            raise ValueError("numerical tolerances must be >= 0")

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


def _preflight_capabilities() -> dict[str, Any]:
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
                "tests": tests,
            })
        row = {"label": label, "op": op, "direction": direction, "candidates": candidates}
        capabilities.append(row)
        if not candidates:
            missing.append({
                "label": label, "op": op, "direction": direction,
                "action": f"add an FP32 {direction} kernel map with unit/parity evidence for op={op}",
            })
    return {
        "schema": "cke.v8.training_capability_preflight.v1",
        "status": "PASS" if not missing else "FAIL", "passed": not missing,
        "kernel_map_root": str(KERNEL_MAPS), "capabilities": capabilities, "missing": missing,
    }


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

    def command(self) -> list[str]:
        c = self.contract; d = self.dataset; t = self.tokenizer; train = self.training
        return [
            self.python, str(WORKFLOW), "--run-dir", str(self.run_dir), "--json-out", str(self.report_path),
            "--corpus", str(Path(d.corpus).expanduser().resolve(strict=False)), "--seed", str(train.seed),
            "--seq-len", str(c["context_len"]), "--epochs", str(train.epochs),
            "--layers", str(c["layers"]), "--d-model", str(c["dim"]), "--hidden", str(c["hidden"]),
            "--num-heads", str(c["heads"]), "--num-kv-heads", str(c["kv_heads"]),
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

    def preflight(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        experiment = self.experiment_document()
        self.experiment_path.write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = _preflight_capabilities()
        report["experiment"] = str(self.experiment_path)
        report["experiment_sha256"] = _sha256(self.experiment_path)
        self.preflight_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not report["passed"]:
            details = "; ".join(row["action"] for row in report["missing"])
            raise RuntimeError(f"v8 training capability preflight failed: {details}")
        return report

    def run(self) -> dict[str, Any]:
        self.preflight()
        try:
            self.command_runner(self.command(), REPO_ROOT)
        except subprocess.CalledProcessError as exc:
            if self.report_path.is_file():
                report = json.loads(self.report_path.read_text(encoding="utf-8"))
                details = report.get("failures") or report.get("exception") or report.get("status")
                raise RuntimeError(
                    f"generated-C workflow failed; report={self.report_path}; details={details}"
                ) from exc
            raise RuntimeError(
                f"generated-C workflow exited with {exc.returncode} before publishing {self.report_path}"
            ) from exc
        if not self.report_path.is_file():
            raise RuntimeError(f"generated-C workflow did not publish {self.report_path}")
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("passed") is not True:
            raise RuntimeError(f"generated-C workflow failed; inspect {self.report_path}")
        return report


def compile(
    model: Module, *, run_name: str, run_dir: Optional[Path] = None,
    dataset: Optional[DatasetConfig] = None, tokenizer: Optional[TokenizerConfig] = None,
    training: Optional[TrainingConfig] = None, command_runner: CommandRunner = _default_runner,
    python: str = sys.executable,
) -> CompiledTrainingExperiment:
    contract = extract_tiny_lm_contract(model, frontend="v8")
    if contract["layers"] not in {4, 6, 10}:
        raise ValueError("v8 generated training currently supports exactly 4, 6, or 10 dense/GQA layers")
    if any(str(parameter.dtype) != "float32" for parameter in model.parameters()):
        raise ValueError("v8 generated training authoring currently supports FP32 parameters only")
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
