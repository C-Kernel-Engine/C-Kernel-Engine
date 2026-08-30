from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


TargetName = Literal["cpu", "c", "llvm"]


@dataclass(frozen=True)
class TargetConfig:
    """Target intent for Python-authored compile requests."""

    name: TargetName = "cpu"
    isa: str = "auto"
    threads: Optional[int] = None

    def __post_init__(self) -> None:
        if self.threads is not None and self.threads <= 0:
            raise ValueError("threads must be > 0 when provided")

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "isa": self.isa,
            "threads": self.threads,
        }


@dataclass(frozen=True)
class CompileConfig:
    """
    High-level compile/pass controls for ck.nn authoring.

    These are recorded as intent today. Existing v7 scripts still own the
    concrete lowering, kernel binding, and generated runtime codegen.
    """

    target: TargetConfig = field(default_factory=TargetConfig)
    strict_contracts: bool = True
    vectorize: bool = True
    pack_weights: bool = True
    unroll: int = 1
    debug_dumps: bool = False
    dump_pass_trace: bool = True
    kernel_policy: str = "fp32_reference_first"

    def __post_init__(self) -> None:
        if self.unroll <= 0:
            raise ValueError("unroll must be > 0")

    def to_metadata(self) -> dict[str, object]:
        return {
            "target": self.target.to_metadata(),
            "strict_contracts": self.strict_contracts,
            "vectorize": self.vectorize,
            "pack_weights": self.pack_weights,
            "unroll": self.unroll,
            "debug_dumps": self.debug_dumps,
            "dump_pass_trace": self.dump_pass_trace,
            "kernel_policy": self.kernel_policy,
        }


def default_pass_trace(config: CompileConfig) -> list[dict[str, object]]:
    """Small TileLang-inspired pass trace for notebook/debug inspection."""

    return [
        {
            "name": "python_module_capture",
            "enabled": True,
            "detail": "Capture ck.nn module hierarchy and parameter metadata.",
        },
        {
            "name": "v7_contract_validation",
            "enabled": config.strict_contracts,
            "detail": "Validate authored graph against the current tiny qwen-style v7 LM contract.",
        },
        {
            "name": "template_export",
            "enabled": True,
            "detail": "Embed authoring graph and compile config into template_python_ui.json.",
        },
        {
            "name": "weight_packing_intent",
            "enabled": config.pack_weights,
            "detail": "Record weight packing preference for existing v7 kernel/layout selection.",
        },
        {
            "name": "vectorization_intent",
            "enabled": config.vectorize,
            "detail": f"Record vectorization preference for target {config.target.name}/{config.target.isa}.",
        },
        {
            "name": "unroll_intent",
            "enabled": config.unroll > 1,
            "detail": f"Record requested unroll factor {config.unroll}.",
        },
        {
            "name": "debug_dumps",
            "enabled": config.debug_dumps,
            "detail": "Request extra authoring/debug sidecars when supported by downstream adapters.",
        },
    ]
