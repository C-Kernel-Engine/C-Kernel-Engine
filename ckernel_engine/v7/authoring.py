from __future__ import annotations

import html
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CK_RUN_V7 = REPO_ROOT / "version" / "v7" / "scripts" / "ck_run_v7.py"
OPEN_IR_VISUALIZER = REPO_ROOT / "version" / "v7" / "tools" / "open_ir_visualizer.py"
OPEN_IR_HUB = REPO_ROOT / "version" / "v7" / "tools" / "open_ir_hub.py"
PREPARE_RUN_VIEWER = REPO_ROOT / "version" / "v7" / "tools" / "prepare_run_viewer.py"


def _default_cache_root() -> Path:
    env = os.environ.get("CK_CACHE_DIR")
    return Path(env).expanduser() if env else (Path.home() / ".cache" / "ck-engine-v7" / "models")


DEFAULT_TRAIN_ROOT = _default_cache_root() / "train"

CommandRunner = Callable[[Sequence[str], Path], None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_python_exec() -> str:
    parity_python = REPO_ROOT / ".venv" / "bin" / "python"
    if parity_python.exists():
        return str(parity_python)
    return sys.executable


def _default_command_runner(command: Sequence[str], cwd: Path) -> None:
    subprocess.run([str(part) for part in command], cwd=str(cwd), check=True)


def _path_is_within(child: Path, parent: Path) -> bool:
    child_resolved = child.expanduser().resolve(strict=False)
    parent_resolved = parent.expanduser().resolve(strict=False)
    try:
        child_resolved.relative_to(parent_resolved)
        return True
    except ValueError:
        return False


def _stringify_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return str(path.expanduser().resolve(strict=False))


def _existing_path(path: Path) -> Optional[Path]:
    candidate = path.expanduser().resolve(strict=False)
    if candidate.exists():
        return candidate
    return None


def _find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        candidate = _existing_path(path)
        if candidate is not None:
            return candidate
    return None


def _path_uri(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return path.expanduser().resolve(strict=False).as_uri()


def _resolve_models_root_for_run(run_dir: Path, models_root: Optional[Path] = None) -> Path:
    if models_root is not None:
        return Path(models_root).expanduser().resolve(strict=False)
    resolved_run_dir = Path(run_dir).expanduser().resolve(strict=False)
    if resolved_run_dir.parent.name in {"train", "inference", "eval"}:
        return resolved_run_dir.parent.parent
    return resolved_run_dir.parent


def _train_report_path(run_dir: Path) -> Optional[Path]:
    return _find_first_existing(
        [
            run_dir / "train_e2e_latest.json",
            run_dir / "train_sanity_latest.json",
            run_dir / "train_parity_latest.json",
        ]
    )


def _libtrain_path(run_dir: Path) -> Optional[Path]:
    return _find_first_existing(
        [
            run_dir / "libtrain.so",
            run_dir / ".ck_build" / "libtrain.so",
        ]
    )


def _artifact_dashboard_rows(run_dir: Path, models_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "label": "Project Plan",
            "path": _existing_path(run_dir / "python_authoring_plan.json"),
            "note": "Python-side project spec and command history.",
        },
        {
            "label": "Python Graph",
            "path": _existing_path(run_dir / "python_authoring_graph.json"),
            "note": "Symbolic module graph exported before the existing v7 materialize/train handoff.",
        },
        {
            "label": "Python Graph Notes",
            "path": _existing_path(run_dir / "python_authoring_graph.md"),
            "note": "Human-readable summary of the same ck.nn module graph.",
        },
        {
            "label": "Compile Config",
            "path": _existing_path(run_dir / "python_authoring_compile_config.json"),
            "note": "Python-side compile target and pass intent for the v7 adapter.",
        },
        {
            "label": "Pass Trace",
            "path": _existing_path(run_dir / "python_authoring_pass_trace.json"),
            "note": "Recorded authoring/lowering pass intent before v7 owns execution.",
        },
        {
            "label": "Weights Manifest",
            "path": _existing_path(run_dir / "weights_manifest.json"),
            "note": "Manifest-first training state and template contract.",
        },
        {
            "label": "Train Report",
            "path": _train_report_path(run_dir),
            "note": "Latest train, sanity, or parity JSON report from the existing v7 runner.",
        },
        {
            "label": "IR Forward",
            "path": _existing_path(run_dir / "ir1_train_forward.json"),
            "note": "Train forward IR lowered by the existing v7 scripts.",
        },
        {
            "label": "IR Backward",
            "path": _existing_path(run_dir / "ir2_train_backward.json"),
            "note": "Backward IR and stitched train graph for generated runtime codegen.",
        },
        {
            "label": "Layout",
            "path": _existing_path(run_dir / "layout_train.json"),
            "note": "Planned memory layout for the training runtime.",
        },
        {
            "label": "Generated Train Runtime",
            "path": _existing_path(run_dir / "generated_train_runtime_v7.c"),
            "note": "Generated C training runtime emitted by codegen.",
        },
        {
            "label": "Compiled Train Runtime",
            "path": _libtrain_path(run_dir),
            "note": "Compiled shared library for the generated training runtime.",
        },
        {
            "label": "IR Report",
            "path": _find_first_existing(
                [
                    run_dir / "ir_report.html",
                    run_dir / ".ck_build" / "ir_report.html",
                ]
            ),
            "note": "Generated IR visualizer for this run.",
        },
        {
            "label": "Embeddings Export",
            "path": _find_first_existing(
                [
                    run_dir / "embeddings.json",
                    run_dir / "dataset" / "embeddings.json",
                ]
            ),
            "note": "Token embedding matrix export used by the dataset viewer and hub.",
        },
        {
            "label": "Attention Export",
            "path": _find_first_existing(
                [
                    run_dir / "attention.json",
                    run_dir / "dataset" / "attention.json",
                ]
            ),
            "note": "Requires tokenizer.json plus a probe path.",
        },
        {
            "label": "Dataset Viewer",
            "path": _find_first_existing(
                [
                    run_dir / "dataset_viewer.html",
                    run_dir / "dataset" / "dataset_viewer.html",
                ]
            ),
            "note": "Requires dataset manifests or a staged dataset workspace.",
        },
        {
            "label": "IR Hub",
            "path": _existing_path(models_root / "ir_hub.html"),
            "note": "Shared run hub under the parent models root.",
        },
        {
            "label": "Run Hub Index",
            "path": _existing_path(models_root / "runs_hub_index.json"),
            "note": "JSON index consumed by ir_hub.html.",
        },
    ]


def notebook_artifact_dashboard_html(
    run_dir: str | Path,
    *,
    models_root: Optional[str | Path] = None,
    title: str = "v7 Run Artifact Dashboard",
) -> str:
    resolved_run_dir = Path(run_dir).expanduser().resolve(strict=False)
    resolved_models_root = _resolve_models_root_for_run(
        resolved_run_dir,
        Path(models_root).expanduser().resolve(strict=False) if models_root is not None else None,
    )
    rows = _artifact_dashboard_rows(resolved_run_dir, resolved_models_root)
    ir_report_path = _find_first_existing(
        [
            resolved_run_dir / "ir_report.html",
            resolved_run_dir / ".ck_build" / "ir_report.html",
        ]
    )
    dataset_viewer_path = _find_first_existing(
        [
            resolved_run_dir / "dataset_viewer.html",
            resolved_run_dir / "dataset" / "dataset_viewer.html",
        ]
    )
    ir_hub_path = _existing_path(resolved_models_root / "ir_hub.html")

    row_html = []
    for row in rows:
        path = row["path"]
        status_text = "ready" if path is not None else "not available"
        status_color = "#47b475" if path is not None else "#ffb400"
        if path is None:
            path_html = "<span style='color:#8a93a6;'>not present for this run</span>"
        else:
            uri = _path_uri(path)
            escaped_name = html.escape(path.name)
            escaped_path = html.escape(str(path))
            path_html = (
                f"<a href='{html.escape(uri or '')}' target='_blank' rel='noopener'>{escaped_name}</a>"
                f"<div style='margin-top:0.18rem;'><code>{escaped_path}</code></div>"
            )
        row_html.append(
            "<tr>"
            f"<td style='padding:0.55rem 0.65rem;vertical-align:top;border-top:1px solid rgba(255,255,255,0.06);'><strong>{html.escape(row['label'])}</strong></td>"
            f"<td style='padding:0.55rem 0.65rem;vertical-align:top;border-top:1px solid rgba(255,255,255,0.06);'><span style='display:inline-block;padding:0.14rem 0.46rem;border-radius:999px;border:1px solid {status_color};color:{status_color};font-size:0.74rem;font-weight:700;text-transform:uppercase;'>{status_text}</span></td>"
            f"<td style='padding:0.55rem 0.65rem;vertical-align:top;border-top:1px solid rgba(255,255,255,0.06);'>{path_html}</td>"
            f"<td style='padding:0.55rem 0.65rem;vertical-align:top;border-top:1px solid rgba(255,255,255,0.06);color:#9aa4b2;'>{html.escape(row['note'])}</td>"
            "</tr>"
        )

    shortcut_specs = [
        ("Open IR Visualizer", ir_report_path, "#8cb4ff"),
        ("Open Dataset Viewer", dataset_viewer_path, "#07adf8"),
        ("Open IR Hub", ir_hub_path, "#47b475"),
    ]
    shortcut_html = []
    for label, path, color in shortcut_specs:
        if path is None:
            shortcut_html.append(
                f"<span style='display:inline-flex;align-items:center;padding:0.28rem 0.62rem;border-radius:999px;border:1px solid rgba(255,180,0,0.35);color:#ffb400;background:rgba(255,180,0,0.08);font-size:0.8rem;'>{html.escape(label)} unavailable</span>"
            )
            continue
        shortcut_html.append(
            f"<a href='{html.escape(_path_uri(path) or '')}' target='_blank' rel='noopener' "
            f"style='display:inline-flex;align-items:center;padding:0.28rem 0.62rem;border-radius:999px;border:1px solid {color};color:{color};background:rgba(17,23,34,0.88);font-size:0.8rem;text-decoration:none;'>{html.escape(label)}</a>"
        )

    return (
        "<div style='margin:1rem 0 1.25rem;border:1px solid #2a2f3a;border-radius:12px;overflow:hidden;background:#151b26;'>"
        "<div style='padding:0.95rem 1rem;border-bottom:1px solid #2a2f3a;background:linear-gradient(145deg, rgba(255,180,0,0.12), rgba(7,173,248,0.08));'>"
        f"<div style='font-size:1rem;font-weight:700;color:#e8e6e3;'>{html.escape(title)}</div>"
        f"<div style='margin-top:0.32rem;color:#9aa4b2;'>Run dir: <code>{html.escape(str(resolved_run_dir))}</code></div>"
        f"<div style='margin-top:0.16rem;color:#9aa4b2;'>Models root: <code>{html.escape(str(resolved_models_root))}</code></div>"
        "<div style='margin-top:0.45rem;color:#c2cad6;'>Notebook front door into the v7 artifact surface. "
        "IR report and IR hub are the first two links to inspect; dataset viewer and attention exports are conditional.</div>"
        "<div style='margin-top:0.7rem;display:flex;gap:0.45rem;flex-wrap:wrap;'>"
        + "".join(shortcut_html)
        + "</div>"
        "</div>"
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>"
        "<thead>"
        "<tr style='background:#111722;'>"
        "<th style='text-align:left;padding:0.6rem 0.65rem;color:#ffb400;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.04em;'>Artifact</th>"
        "<th style='text-align:left;padding:0.6rem 0.65rem;color:#ffb400;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.04em;'>Status</th>"
        "<th style='text-align:left;padding:0.6rem 0.65rem;color:#ffb400;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.04em;'>Path</th>"
        "<th style='text-align:left;padding:0.6rem 0.65rem;color:#ffb400;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.04em;'>Notes</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(row_html)
        + "</tbody></table></div></div>"
    )


@dataclass(frozen=True)
class TemplateSpec:
    """How the Python layer chooses the v7 training template."""

    builtin: str = "qwen3"
    document: Optional[dict[str, Any]] = None
    source_path: Optional[Path] = None

    @classmethod
    def builtin_template(cls, name: str) -> "TemplateSpec":
        text = str(name or "qwen3").strip().lower() or "qwen3"
        return cls(builtin=text)

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        builtin: Optional[str] = None,
    ) -> "TemplateSpec":
        payload = dict(document or {})
        if not payload:
            raise ValueError("template document must not be empty")
        doc_name = str(payload.get("name", "") or "").strip().lower()
        builtin_name = str(builtin or doc_name or "qwen3").strip().lower() or "qwen3"
        return cls(builtin=builtin_name, document=payload)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        builtin: Optional[str] = None,
    ) -> "TemplateSpec":
        template_path = Path(path).expanduser().resolve(strict=False)
        payload = json.loads(template_path.read_text(encoding="utf-8"))
        doc_name = str(payload.get("name", "") or "").strip().lower()
        builtin_name = str(builtin or doc_name or "qwen3").strip().lower() or "qwen3"
        return cls(builtin=builtin_name, document=payload, source_path=template_path)

    def materialize(self, run_dir: Path) -> tuple[str, Optional[Path]]:
        if self.document is None:
            return self.builtin, None
        run_dir.mkdir(parents=True, exist_ok=True)
        template_path = run_dir / "template_python_ui.json"
        template_path.write_text(json.dumps(self.document, indent=2), encoding="utf-8")
        return self.builtin, template_path

    def to_metadata(self) -> dict[str, Any]:
        return {
            "builtin": self.builtin,
            "embedded_document": bool(self.document),
            "source_path": _stringify_path(self.source_path),
        }


@dataclass(frozen=True)
class TinyModelSpec:
    """Shape/init settings for the current tiny v7 run bootstrap path."""

    init: Literal["normal_0p02", "xavier_uniform", "xavier_normal", "kaiming_uniform", "zeros"] = "normal_0p02"
    layers: int = 2
    vocab_size: int = 256
    embed_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 8
    num_kv_heads: int = 4
    context_len: int = 128
    rope_theta: float = 1_000_000.0
    kernel_policy: str = "fp32_reference_first"
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.01
    seed: int = 42

    def validate(self) -> None:
        if self.layers <= 0:
            raise ValueError("layers must be > 0")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.embed_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be > 0")
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("num_heads and num_kv_heads must be > 0")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.context_len <= 0:
            raise ValueError("context_len must be > 0")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be > 0")

    def to_init_args(self) -> list[str]:
        self.validate()
        return [
            "--train-seed", str(self.seed),
            "--init", self.init,
            "--layers", str(self.layers),
            "--vocab-size", str(self.vocab_size),
            "--embed-dim", str(self.embed_dim),
            "--hidden-dim", str(self.hidden_dim),
            "--num-heads", str(self.num_heads),
            "--num-kv-heads", str(self.num_kv_heads),
            "--context-len", str(self.context_len),
            "--rope-theta", str(self.rope_theta),
            "--kernel-policy", self.kernel_policy,
            "--adamw-beta1", str(self.adamw_beta1),
            "--adamw-beta2", str(self.adamw_beta2),
            "--adamw-eps", str(self.adamw_eps),
            "--adamw-weight-decay", str(self.adamw_weight_decay),
        ]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "init": self.init,
            "layers": self.layers,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "context_len": self.context_len,
            "rope_theta": self.rope_theta,
            "kernel_policy": self.kernel_policy,
            "adamw": {
                "beta1": self.adamw_beta1,
                "beta2": self.adamw_beta2,
                "eps": self.adamw_eps,
                "weight_decay": self.adamw_weight_decay,
            },
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TokenizerPlan:
    """Tokenizer planning metadata for the Python-authored workflow."""

    family: str = "runtime_default"
    binary_dir: Optional[Path] = None
    notes: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "binary_dir": _stringify_path(self.binary_dir),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DataSource:
    """Training data selector for the current text/token-file backed v7 flow."""

    kind: Literal["inline_text", "text_file", "token_file"] = "inline_text"
    value: str = "Hello!"
    description: str = ""

    @classmethod
    def inline_text(cls, text: str, *, description: str = "") -> "DataSource":
        return cls(kind="inline_text", value=str(text), description=description)

    @classmethod
    def text_file(cls, path: str | Path, *, description: str = "") -> "DataSource":
        return cls(kind="text_file", value=str(Path(path).expanduser()), description=description)

    @classmethod
    def token_file(cls, path: str | Path, *, description: str = "") -> "DataSource":
        return cls(kind="token_file", value=str(Path(path).expanduser()), description=description)

    def to_cli_args(self) -> list[str]:
        if self.kind == "inline_text":
            return ["--prompt", self.value]
        if self.kind == "text_file":
            return ["--data", str(Path(self.value).expanduser())]
        if self.kind == "token_file":
            return ["--train-token-file", str(Path(self.value).expanduser())]
        raise ValueError(f"unsupported data source kind: {self.kind}")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class MaterializeOptions:
    """Controls the `init -> IR -> runtime` handoff stage."""

    generate_ir: bool = True
    generate_runtime: bool = True
    strict: bool = True
    train_mode: Literal["pretrain", "sft"] = "pretrain"
    bridge_lowering: Literal["legacy", "explicit"] = "legacy"
    checkpoint_policy: Literal["none", "recompute_attn"] = "none"
    dataset_workspace: Optional[Path] = None
    dataset_stage_mode: Literal["copy", "symlink"] = "copy"
    dataset_stage_force: bool = False

    def __post_init__(self) -> None:
        if self.generate_runtime and not self.generate_ir:
            raise ValueError("generate_runtime requires generate_ir=True")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "generate_ir": self.generate_ir,
            "generate_runtime": self.generate_runtime,
            "strict": self.strict,
            "train_mode": self.train_mode,
            "bridge_lowering": self.bridge_lowering,
            "checkpoint_policy": self.checkpoint_policy,
            "dataset_workspace": _stringify_path(self.dataset_workspace),
            "dataset_stage_mode": self.dataset_stage_mode,
            "dataset_stage_force": self.dataset_stage_force,
        }


@dataclass(frozen=True)
class TrainConfig:
    """Training/sanity/parity settings for the generated-runtime surface."""

    mode: Literal["pretrain", "sft"] = "pretrain"
    backend: Literal["ck", "pytorch", "torch", "both"] = "ck"
    strict: bool = True
    epochs: int = 1
    seq_len: int = 8
    total_tokens: int = 64
    grad_accum: int = 2
    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr: float = 5e-4
    seed: int = 42
    parity_regimen: Literal["off", "suggest", "run", "require"] = "suggest"
    kernel_strict_math: bool = False
    bitwise_parity: bool = False
    max_grad_norm: float = 0.0
    bridge_lowering: Literal["legacy", "explicit"] = "legacy"
    checkpoint_policy: Literal["none", "recompute_attn"] = "none"
    memory_check: bool = True
    memory_min_available_gb: Optional[float] = None
    memory_min_available_ratio: Optional[float] = None

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.seq_len <= 0 or self.total_tokens <= 0 or self.grad_accum <= 0:
            raise ValueError("seq_len, total_tokens, and grad_accum must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be >= 0")
        if self.memory_min_available_gb is not None and self.memory_min_available_gb < 0:
            raise ValueError("memory_min_available_gb must be >= 0")
        if self.memory_min_available_ratio is not None and self.memory_min_available_ratio < 0:
            raise ValueError("memory_min_available_ratio must be >= 0")

    def to_cli_args(self) -> list[str]:
        self.validate()
        args = [
            "--train-mode", self.mode,
            "--backend", self.backend,
            "--train-epochs", str(self.epochs),
            "--train-seq-len", str(self.seq_len),
            "--train-total-tokens", str(self.total_tokens),
            "--train-grad-accum", str(self.grad_accum),
            "--train-optimizer", self.optimizer,
            "--train-lr", str(self.lr),
            "--train-seed", str(self.seed),
            "--parity-regimen", self.parity_regimen,
            "--train-bridge-lowering", self.bridge_lowering,
            "--train-checkpoint-policy", self.checkpoint_policy,
            "--train-max-grad-norm", str(self.max_grad_norm),
        ]
        if self.strict:
            args.append("--train-strict")
        if self.kernel_strict_math:
            args.append("--kernel-strict-math")
        if self.bitwise_parity:
            args.append("--bitwise-parity")
        if not self.memory_check:
            args.append("--no-train-memory-check")
        if self.memory_min_available_gb is not None:
            args.extend(["--train-memory-min-available-gb", str(self.memory_min_available_gb)])
        if self.memory_min_available_ratio is not None:
            args.extend(["--train-memory-min-available-ratio", str(self.memory_min_available_ratio)])
        return args

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "strict": self.strict,
            "epochs": self.epochs,
            "seq_len": self.seq_len,
            "total_tokens": self.total_tokens,
            "grad_accum": self.grad_accum,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "seed": self.seed,
            "parity_regimen": self.parity_regimen,
            "kernel_strict_math": self.kernel_strict_math,
            "bitwise_parity": self.bitwise_parity,
            "max_grad_norm": self.max_grad_norm,
            "bridge_lowering": self.bridge_lowering,
            "checkpoint_policy": self.checkpoint_policy,
            "memory_check": self.memory_check,
            "memory_min_available_gb": self.memory_min_available_gb,
            "memory_min_available_ratio": self.memory_min_available_ratio,
        }


@dataclass(frozen=True)
class ExecutionResult:
    action: str
    run_dir: Path
    command: tuple[str, ...]
    report_path: Optional[Path] = None
    project_plan_path: Optional[Path] = None


@dataclass(frozen=True)
class ViewerCommandResult:
    action: str
    command: tuple[str, ...]
    paths: dict[str, Optional[Path]]


@dataclass(frozen=True)
class ViewerArtifacts:
    run_dir: Path
    models_root: Path
    ir_report: Optional[Path] = None
    dataset_viewer: Optional[Path] = None
    embeddings: Optional[Path] = None
    attention: Optional[Path] = None
    ir_hub: Optional[Path] = None
    hub_index: Optional[Path] = None


@dataclass
class TrainingProject:
    """
    Python authoring surface for the existing v7 training pipeline.

    This is intentionally a thin orchestration layer today:
    Python defines model/template/data/training specs, and v7 still owns
    manifest creation, IR lowering, codegen, generated C runtime compilation,
    and runtime execution.
    """

    run_name: str
    model: TinyModelSpec = field(default_factory=TinyModelSpec)
    template: TemplateSpec = field(default_factory=TemplateSpec)
    tokenizer: TokenizerPlan = field(default_factory=TokenizerPlan)
    run_dir: Optional[Path] = None
    repo_root: Path = field(default_factory=lambda: REPO_ROOT)
    python_exec: Optional[str] = None
    command_runner: Optional[CommandRunner] = field(default=_default_command_runner, repr=False)
    _history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).expanduser().resolve(strict=False)
        if self.python_exec is None:
            self.python_exec = _default_python_exec()
        if self.command_runner is None:
            self.command_runner = _default_command_runner
        if self.run_dir is None:
            self.run_dir = (DEFAULT_TRAIN_ROOT / self.run_name).expanduser().resolve(strict=False)
        else:
            self.run_dir = Path(self.run_dir).expanduser().resolve(strict=False)
        self._ck_run_script = self.repo_root / "version" / "v7" / "scripts" / "ck_run_v7.py"
        if not self._ck_run_script.exists():
            raise FileNotFoundError(f"missing ck_run_v7.py: {self._ck_run_script}")

    @property
    def project_plan_path(self) -> Path:
        return Path(self.run_dir) / "python_authoring_plan.json"

    @property
    def models_root(self) -> Path:
        return self._resolve_models_root(None)

    def materialize(self, options: Optional[MaterializeOptions] = None) -> ExecutionResult:
        opts = options or MaterializeOptions()
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        template_name, template_path = self.template.materialize(Path(self.run_dir))

        command = [
            str(self.python_exec),
            str(self._ck_run_script),
            "init",
            "--run", str(self.run_dir),
            "--run-name", self.run_name,
            "--template", template_name,
            "--train-mode", opts.train_mode,
            "--train-bridge-lowering", opts.bridge_lowering,
            "--train-checkpoint-policy", opts.checkpoint_policy,
        ]
        if not _path_is_within(Path(self.run_dir), DEFAULT_TRAIN_ROOT):
            command.append("--allow-non-cache-run-dir")
        if template_path is not None:
            command.extend(["--template-file", str(template_path)])
        if opts.generate_ir:
            command.append("--generate-ir")
        if opts.generate_runtime:
            command.append("--generate-runtime")
        if opts.strict:
            command.append("--strict")
        if opts.dataset_workspace is not None:
            command.extend(["--dataset-workspace", str(opts.dataset_workspace)])
            command.extend(["--dataset-stage-mode", opts.dataset_stage_mode])
            if opts.dataset_stage_force:
                command.append("--dataset-stage-force")
        command.extend(self.model.to_init_args())

        self._run(command)
        self._append_history(
            action="materialize",
            command=command,
            payload={
                "materialize_options": opts.to_metadata(),
                "template_path": _stringify_path(template_path),
            },
        )
        self._write_project_plan()
        return ExecutionResult(
            action="materialize",
            run_dir=Path(self.run_dir),
            command=tuple(command),
            project_plan_path=self.project_plan_path,
        )

    def train(
        self,
        data: DataSource,
        config: Optional[TrainConfig] = None,
        *,
        auto_materialize: bool = True,
    ) -> ExecutionResult:
        return self._run_training_action(
            action="train",
            data=data,
            config=config or TrainConfig(),
            auto_materialize=auto_materialize,
        )

    def sanity(
        self,
        data: DataSource,
        config: Optional[TrainConfig] = None,
        *,
        min_loss_drop: float = 0.0,
        auto_materialize: bool = True,
    ) -> ExecutionResult:
        extra = ["--min-loss-drop", str(min_loss_drop)]
        return self._run_training_action(
            action="sanity",
            data=data,
            config=config or TrainConfig(),
            auto_materialize=auto_materialize,
            extra_args=extra,
            payload={"min_loss_drop": min_loss_drop},
        )

    def parity(
        self,
        data: DataSource,
        config: Optional[TrainConfig] = None,
        *,
        with_fd: bool = True,
        with_replay: bool = True,
        auto_materialize: bool = True,
    ) -> ExecutionResult:
        extra: list[str] = []
        if with_fd:
            extra.append("--with-fd")
        if with_replay:
            extra.append("--with-replay")
        return self._run_training_action(
            action="parity",
            data=data,
            config=config or TrainConfig(),
            auto_materialize=auto_materialize,
            extra_args=extra,
            payload={"with_fd": with_fd, "with_replay": with_replay},
        )

    def generate_ir_report(
        self,
        *,
        output_path: Optional[Path] = None,
        html_only: bool = True,
        strict_run_artifacts: bool = True,
    ) -> ViewerCommandResult:
        report_path = Path(output_path or (Path(self.run_dir) / "ir_report.html")).expanduser().resolve(strict=False)
        command = [
            str(self.python_exec),
            str(OPEN_IR_VISUALIZER),
            "--generate",
            "--run",
            str(self.run_dir),
            "--output",
            str(report_path),
        ]
        if html_only:
            command.append("--html-only")
        if strict_run_artifacts:
            command.append("--strict-run-artifacts")

        self._run(command)
        paths = {"ir_report": _existing_path(report_path)}
        self._append_history(
            action="generate_ir_report",
            command=command,
            payload={
                "html_only": html_only,
                "strict_run_artifacts": strict_run_artifacts,
                "artifacts": {name: _stringify_path(path) for name, path in paths.items()},
            },
        )
        self._write_project_plan()
        return ViewerCommandResult(action="generate_ir_report", command=tuple(command), paths=paths)

    def prepare_run_viewer_artifacts(self, *, force: bool = False) -> ViewerCommandResult:
        command = [
            str(self.python_exec),
            str(PREPARE_RUN_VIEWER),
            str(self.run_dir),
        ]
        if force:
            command.append("--force")

        self._run(command)
        paths = {
            "dataset_viewer": self._dataset_viewer_path(),
            "embeddings": self._embeddings_path(),
            "attention": self._attention_path(),
        }
        self._append_history(
            action="prepare_run_viewer",
            command=command,
            payload={
                "force": force,
                "artifacts": {name: _stringify_path(path) for name, path in paths.items()},
            },
        )
        self._write_project_plan()
        return ViewerCommandResult(action="prepare_run_viewer", command=tuple(command), paths=paths)

    def refresh_ir_hub(
        self,
        *,
        models_root: Optional[Path] = None,
        output_path: Optional[Path] = None,
        index_out: Optional[Path] = None,
    ) -> ViewerCommandResult:
        resolved_models_root = self._resolve_models_root(models_root)
        hub_path = Path(output_path or (resolved_models_root / "ir_hub.html")).expanduser().resolve(strict=False)
        index_path = Path(index_out or (resolved_models_root / "runs_hub_index.json")).expanduser().resolve(strict=False)
        command = [
            str(self.python_exec),
            str(OPEN_IR_HUB),
            "--models-root",
            str(resolved_models_root),
            "--output",
            str(hub_path),
            "--index-out",
            str(index_path),
        ]

        self._run(command)
        paths = {
            "ir_hub": _existing_path(hub_path),
            "hub_index": _existing_path(index_path),
        }
        self._append_history(
            action="refresh_ir_hub",
            command=command,
            payload={
                "models_root": _stringify_path(resolved_models_root),
                "artifacts": {name: _stringify_path(path) for name, path in paths.items()},
            },
        )
        self._write_project_plan()
        return ViewerCommandResult(action="refresh_ir_hub", command=tuple(command), paths=paths)

    def prepare_viewers(
        self,
        *,
        force: bool = False,
        models_root: Optional[Path] = None,
        ir_report_output: Optional[Path] = None,
        hub_output: Optional[Path] = None,
        hub_index_out: Optional[Path] = None,
    ) -> ViewerArtifacts:
        self.generate_ir_report(output_path=ir_report_output)
        self.prepare_run_viewer_artifacts(force=force)
        self.refresh_ir_hub(models_root=models_root, output_path=hub_output, index_out=hub_index_out)
        resolved_models_root = self._resolve_models_root(models_root)
        return ViewerArtifacts(
            run_dir=Path(self.run_dir),
            models_root=resolved_models_root,
            ir_report=self._ir_report_path(),
            dataset_viewer=self._dataset_viewer_path(),
            embeddings=self._embeddings_path(),
            attention=self._attention_path(),
            ir_hub=self._ir_hub_path(resolved_models_root),
            hub_index=self._hub_index_path(resolved_models_root),
        )

    def notebook_artifact_dashboard_html(
        self,
        *,
        title: str = "v7 Run Artifact Dashboard",
        models_root: Optional[Path] = None,
    ) -> str:
        return notebook_artifact_dashboard_html(
            self.run_dir,
            models_root=self._resolve_models_root(models_root),
            title=title,
        )

    def _run_training_action(
        self,
        *,
        action: Literal["train", "sanity", "parity"],
        data: DataSource,
        config: TrainConfig,
        auto_materialize: bool,
        extra_args: Optional[Sequence[str]] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> ExecutionResult:
        if auto_materialize and not (Path(self.run_dir) / "weights_manifest.json").exists():
            self.materialize()

        report_path = Path(self.run_dir) / {
            "train": "train_e2e_latest.json",
            "sanity": "train_sanity_latest.json",
            "parity": "train_parity_latest.json",
        }[action]
        command = [
            str(self.python_exec),
            str(self._ck_run_script),
            action,
            "--run", str(self.run_dir),
            "--train-json-out", str(report_path),
        ]
        if not _path_is_within(Path(self.run_dir), DEFAULT_TRAIN_ROOT):
            command.append("--allow-non-cache-run-dir")
        command.extend(config.to_cli_args())
        command.extend(data.to_cli_args())
        if extra_args:
            command.extend([str(arg) for arg in extra_args])

        self._run(command)
        self._append_history(
            action=action,
            command=command,
            report_path=report_path,
            payload={
                "data": data.to_metadata(),
                "train_config": config.to_metadata(),
                **(payload or {}),
            },
        )
        self._write_project_plan()
        return ExecutionResult(
            action=action,
            run_dir=Path(self.run_dir),
            command=tuple(command),
            report_path=report_path,
            project_plan_path=self.project_plan_path,
        )

    def _run(self, command: Sequence[str]) -> None:
        self.command_runner([str(part) for part in command], self.repo_root)

    def _resolve_models_root(self, models_root: Optional[Path]) -> Path:
        return _resolve_models_root_for_run(Path(self.run_dir), models_root)

    def _ir_report_path(self) -> Optional[Path]:
        return _find_first_existing(
            [
                Path(self.run_dir) / "ir_report.html",
                Path(self.run_dir) / ".ck_build" / "ir_report.html",
            ]
        )

    def _dataset_viewer_path(self) -> Optional[Path]:
        return _find_first_existing(
            [
                Path(self.run_dir) / "dataset_viewer.html",
                Path(self.run_dir) / "dataset" / "dataset_viewer.html",
            ]
        )

    def _embeddings_path(self) -> Optional[Path]:
        return _find_first_existing(
            [
                Path(self.run_dir) / "embeddings.json",
                Path(self.run_dir) / "dataset" / "embeddings.json",
            ]
        )

    def _attention_path(self) -> Optional[Path]:
        return _find_first_existing(
            [
                Path(self.run_dir) / "attention.json",
                Path(self.run_dir) / "dataset" / "attention.json",
            ]
        )

    def _ir_hub_path(self, models_root: Optional[Path] = None) -> Optional[Path]:
        resolved_models_root = self._resolve_models_root(models_root)
        return _existing_path(resolved_models_root / "ir_hub.html")

    def _hub_index_path(self, models_root: Optional[Path] = None) -> Optional[Path]:
        resolved_models_root = self._resolve_models_root(models_root)
        return _existing_path(resolved_models_root / "runs_hub_index.json")

    def _append_history(
        self,
        *,
        action: str,
        command: Sequence[str],
        report_path: Optional[Path] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self._history.append(
            {
                "at": _utc_now_iso(),
                "action": action,
                "command": [str(part) for part in command],
                "report_path": _stringify_path(report_path),
                "payload": payload or {},
            }
        )

    def _write_project_plan(self) -> None:
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        manifest_path = Path(self.run_dir) / "weights_manifest.json"
        doc = {
            "schema": "ck.python_authoring.v1",
            "generated_at": _utc_now_iso(),
            "repo_root": str(self.repo_root),
            "python_exec": str(self.python_exec),
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "template": self.template.to_metadata(),
            "model": self.model.to_metadata(),
            "tokenizer": self.tokenizer.to_metadata(),
            "artifacts": {
                "project_plan": str(self.project_plan_path),
                "python_authoring_graph": _stringify_path(
                    (Path(self.run_dir) / "python_authoring_graph.json")
                    if (Path(self.run_dir) / "python_authoring_graph.json").exists()
                    else None
                ),
                "python_authoring_graph_markdown": _stringify_path(
                    (Path(self.run_dir) / "python_authoring_graph.md")
                    if (Path(self.run_dir) / "python_authoring_graph.md").exists()
                    else None
                ),
                "python_authoring_compile_config": _stringify_path(
                    (Path(self.run_dir) / "python_authoring_compile_config.json")
                    if (Path(self.run_dir) / "python_authoring_compile_config.json").exists()
                    else None
                ),
                "python_authoring_pass_trace": _stringify_path(
                    (Path(self.run_dir) / "python_authoring_pass_trace.json")
                    if (Path(self.run_dir) / "python_authoring_pass_trace.json").exists()
                    else None
                ),
                "weights_manifest": str(manifest_path),
                "weights": str(Path(self.run_dir) / "weights.bump"),
                "config": str(Path(self.run_dir) / "config.json"),
                "template_python_ui": _stringify_path(
                    (Path(self.run_dir) / "template_python_ui.json")
                    if (Path(self.run_dir) / "template_python_ui.json").exists()
                    else None
                ),
                "ir_report": _stringify_path(self._ir_report_path()),
                "dataset_viewer": _stringify_path(self._dataset_viewer_path()),
                "embeddings": _stringify_path(self._embeddings_path()),
                "attention": _stringify_path(self._attention_path()),
                "ir_hub": _stringify_path(self._ir_hub_path()),
                "runs_hub_index": _stringify_path(self._hub_index_path()),
            },
            "history": list(self._history),
            "notes": [
                "Python owns project/spec authoring only.",
                "v7 scripts still own manifest emission, IR lowering, codegen, and generated C runtime execution.",
            ],
        }
        self.project_plan_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
