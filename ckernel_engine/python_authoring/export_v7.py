from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import CompileConfig, default_pass_trace
from .graph import AuthoringGraph, build_authoring_graph
from .nn import Embedding, Linear, Module, RMSNorm, TransformerBlock
from ..v7.authoring import (
    DataSource,
    ExecutionResult,
    MaterializeOptions,
    TemplateSpec,
    TinyModelSpec,
    TokenizerPlan,
    TrainConfig,
    TrainingProject,
    ViewerArtifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_FAMILIES = {'qwen3', 'qwen35'}


def _ensure_supported_family(family: str) -> str:
    family_name = str(family or 'qwen3').strip().lower() or 'qwen3'
    if family_name not in SUPPORTED_FAMILIES:
        choices = ', '.join(sorted(SUPPORTED_FAMILIES))
        raise ValueError(f'unsupported v7 family {family_name!r}; expected one of: {choices}')
    return family_name


def _load_base_template_document(family: str) -> dict[str, Any]:
    template_path = REPO_ROOT / 'version' / 'v7' / 'templates' / f'{family}.json'
    if not template_path.exists():
        raise FileNotFoundError(f'missing base v7 template: {template_path}')
    return json.loads(template_path.read_text(encoding='utf-8'))


def _top_level_modules(model: Module) -> list[Module]:
    modules = [child for _name, child in model.named_children()]
    if not modules:
        raise ValueError('v7 compile expects a container module with ordered child modules')
    return modules


def _extract_tiny_lm_contract(model: Module) -> dict[str, Any]:
    modules = _top_level_modules(model)
    if not isinstance(modules[0], Embedding):
        raise ValueError('v7 compile currently expects the first top-level module to be ck.nn.Embedding')
    if len(modules) < 4:
        raise ValueError('v7 compile expects at least Embedding -> TransformerBlock -> RMSNorm -> Linear')

    embedding = modules[0]
    lm_head = modules[-1]
    if not isinstance(lm_head, Linear):
        raise ValueError('v7 compile currently expects the final top-level module to be ck.nn.Linear')

    final_norm: Optional[RMSNorm] = None
    body = modules[1:-1]
    if body and isinstance(body[-1], RMSNorm):
        final_norm = body[-1]
        body = body[:-1]
    if final_norm is None:
        raise ValueError('v7 compile currently expects a final ck.nn.RMSNorm before the lm_head')
    if not body:
        raise ValueError('v7 compile requires at least one ck.nn.TransformerBlock')
    if any(not isinstance(module, TransformerBlock) for module in body):
        unsupported = [module.__class__.__name__ for module in body if not isinstance(module, TransformerBlock)]
        raise ValueError(
            'v7 compile currently supports only ck.nn.TransformerBlock in the model body; '
            f'found unsupported modules: {unsupported}'
        )

    blocks = list(body)
    reference = blocks[0]
    if embedding.dim != reference.dim:
        raise ValueError('embedding dim must match transformer block dim')
    if final_norm.dim != reference.dim:
        raise ValueError('final RMSNorm dim must match transformer block dim')
    if lm_head.in_features != reference.dim:
        raise ValueError('lm_head input features must match transformer block dim')
    if lm_head.out_features != embedding.vocab:
        raise ValueError('lm_head output features must match embedding vocab size for the current v7 LM path')

    for index, block in enumerate(blocks, start=1):
        if block.dim != reference.dim:
            raise ValueError(f'transformer block {index} has dim={block.dim}, expected {reference.dim}')
        if block.hidden != reference.hidden:
            raise ValueError(f'transformer block {index} has hidden={block.hidden}, expected {reference.hidden}')
        if block.heads != reference.heads:
            raise ValueError(f'transformer block {index} has heads={block.heads}, expected {reference.heads}')
        if block.kv_heads != reference.kv_heads:
            raise ValueError(f'transformer block {index} has kv_heads={block.kv_heads}, expected {reference.kv_heads}')
        if block.context_len != reference.context_len:
            raise ValueError(f'transformer block {index} has context_len={block.context_len}, expected {reference.context_len}')
        if float(block.rope_theta) != float(reference.rope_theta):
            raise ValueError(f'transformer block {index} has rope_theta={block.rope_theta}, expected {reference.rope_theta}')
        if str(block.activation) != str(reference.activation):
            raise ValueError(f'transformer block {index} has activation={block.activation!r}, expected {reference.activation!r}')

    return {
        'embedding': embedding,
        'blocks': blocks,
        'final_norm': final_norm,
        'lm_head': lm_head,
        'dim': reference.dim,
        'hidden': reference.hidden,
        'heads': reference.heads,
        'kv_heads': reference.kv_heads,
        'context_len': reference.context_len,
        'rope_theta': reference.rope_theta,
        'activation': reference.activation,
        'layers': len(blocks),
    }


def _build_template_spec(
    *,
    family: str,
    graph: AuthoringGraph,
    contract: dict[str, Any],
    compile_config: CompileConfig,
    pass_trace: list[dict[str, object]],
) -> TemplateSpec:
    document = copy.deepcopy(_load_base_template_document(family))
    document['python_authoring'] = {
        'schema': 'ck.python_authoring.graph.v1',
        'entrypoint': 'ck.v7.compile',
        'family': family,
        'graph': graph.to_dict(),
        'compile_config': compile_config.to_metadata(),
        'pass_trace': pass_trace,
        'supported_contract': 'v7_tiny_transformer_lm',
        'model_contract': {
            'layers': contract['layers'],
            'dim': contract['dim'],
            'hidden': contract['hidden'],
            'heads': contract['heads'],
            'kv_heads': contract['kv_heads'],
            'context_len': contract['context_len'],
            'rope_theta': contract['rope_theta'],
            'activation': contract['activation'],
        },
        'notes': [
            'Generated from the thin ck.nn Python authoring surface.',
            'The existing v7 runner still owns manifest emission, IR lowering, codegen, and train execution.',
        ],
    }
    return TemplateSpec.from_document(document, builtin=family)


@dataclass(frozen=True)
class CompiledProject:
    project: TrainingProject
    graph: AuthoringGraph
    family: str
    compile_config: CompileConfig
    pass_trace: tuple[dict[str, object], ...]

    @property
    def run_dir(self) -> Path:
        return Path(self.project.run_dir)

    @property
    def models_root(self) -> Path:
        return Path(self.project.models_root)

    @property
    def graph_path(self) -> Path:
        return self.run_dir / 'python_authoring_graph.json'

    @property
    def graph_markdown_path(self) -> Path:
        return self.run_dir / 'python_authoring_graph.md'

    @property
    def compile_config_path(self) -> Path:
        return self.run_dir / 'python_authoring_compile_config.json'

    @property
    def pass_trace_path(self) -> Path:
        return self.run_dir / 'python_authoring_pass_trace.json'

    @property
    def project_plan_path(self) -> Path:
        return self.project.project_plan_path

    def _write_authoring_artifacts(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path.write_text(self.graph.to_json(indent=2), encoding='utf-8')
        self.graph_markdown_path.write_text(self.graph.to_markdown(), encoding='utf-8')
        self.compile_config_path.write_text(
            json.dumps(
                {
                    'schema': 'ck.python_authoring.compile_config.v1',
                    'family': self.family,
                    'compile_config': self.compile_config.to_metadata(),
                },
                indent=2,
            ),
            encoding='utf-8',
        )
        self.pass_trace_path.write_text(
            json.dumps(
                {
                    'schema': 'ck.python_authoring.pass_trace.v1',
                    'family': self.family,
                    'passes': list(self.pass_trace),
                },
                indent=2,
            ),
            encoding='utf-8',
        )

    def materialize(self, options: Optional[MaterializeOptions] = None) -> ExecutionResult:
        self._write_authoring_artifacts()
        return self.project.materialize(options)

    def train(
        self,
        data: str | DataSource,
        config: Optional[TrainConfig] = None,
        *,
        auto_materialize: bool = True,
    ) -> ExecutionResult:
        self._write_authoring_artifacts()
        source = DataSource.inline_text(data) if isinstance(data, str) else data
        return self.project.train(source, config, auto_materialize=auto_materialize)

    def sanity(
        self,
        data: str | DataSource,
        config: Optional[TrainConfig] = None,
        *,
        min_loss_drop: float = 0.0,
        auto_materialize: bool = True,
    ) -> ExecutionResult:
        self._write_authoring_artifacts()
        source = DataSource.inline_text(data) if isinstance(data, str) else data
        return self.project.sanity(source, config, min_loss_drop=min_loss_drop, auto_materialize=auto_materialize)

    def parity(
        self,
        data: str | DataSource,
        config: Optional[TrainConfig] = None,
        *,
        with_fd: bool = True,
        with_replay: bool = True,
        auto_materialize: bool = True,
    ) -> ExecutionResult:
        self._write_authoring_artifacts()
        source = DataSource.inline_text(data) if isinstance(data, str) else data
        return self.project.parity(source, config, with_fd=with_fd, with_replay=with_replay, auto_materialize=auto_materialize)

    def prepare_viewers(
        self,
        *,
        force: bool = False,
        models_root: Optional[Path] = None,
        ir_report_output: Optional[Path] = None,
        hub_output: Optional[Path] = None,
        hub_index_out: Optional[Path] = None,
    ) -> ViewerArtifacts:
        self._write_authoring_artifacts()
        return self.project.prepare_viewers(
            force=force,
            models_root=models_root,
            ir_report_output=ir_report_output,
            hub_output=hub_output,
            hub_index_out=hub_index_out,
        )

    def generate_ir_report(self, **kwargs: Any):
        self._write_authoring_artifacts()
        return self.project.generate_ir_report(**kwargs)

    def refresh_ir_hub(self, **kwargs: Any):
        self._write_authoring_artifacts()
        return self.project.refresh_ir_hub(**kwargs)

    def show_graph(self, *, format: str = 'markdown') -> str | dict[str, Any]:
        if format == 'markdown':
            return self.graph.to_markdown()
        if format == 'json':
            return self.graph.to_dict()
        raise ValueError("format must be 'markdown' or 'json'")

    def show_ir(self, *, generate: bool = False, **kwargs: Any) -> Optional[Path]:
        current = self.project._ir_report_path()  # noqa: SLF001 - thin adapter over existing v7 surface.
        if current is not None and not generate:
            return current
        result = self.generate_ir_report(**kwargs)
        return result.paths.get('ir_report')

    def notebook_artifact_dashboard_html(self, *, title: str = 'v7 Run Artifact Dashboard') -> str:
        self._write_authoring_artifacts()
        return self.project.notebook_artifact_dashboard_html(title=title)

    def show_compile_config(self) -> dict[str, object]:
        return self.compile_config.to_metadata()

    def show_pass_trace(self) -> list[dict[str, object]]:
        return list(self.pass_trace)


def compile(
    model: Module,
    *,
    run_name: str,
    family: str = 'qwen3',
    run_dir: Optional[str | Path] = None,
    init: str = 'normal_0p02',
    kernel_policy: Optional[str] = None,
    config: Optional[CompileConfig] = None,
    seed: int = 42,
    adamw_beta1: float = 0.9,
    adamw_beta2: float = 0.999,
    adamw_eps: float = 1e-8,
    adamw_weight_decay: float = 0.01,
    tokenizer_family: str = 'runtime_default',
    tokenizer_notes: str = 'Compiled from the ck.nn authoring surface into the existing v7 pipeline.',
    command_runner=None,
    python_exec: Optional[str] = None,
    repo_root: Optional[str | Path] = None,
) -> CompiledProject:
    family_name = _ensure_supported_family(family)
    compile_config = config or CompileConfig(
        kernel_policy=kernel_policy or CompileConfig().kernel_policy,
    )
    if kernel_policy is not None and config is not None and kernel_policy != config.kernel_policy:
        raise ValueError('kernel_policy conflicts with config.kernel_policy')
    graph = build_authoring_graph(model, name=model.name)
    contract = _extract_tiny_lm_contract(model)
    if family_name in {'qwen3', 'qwen35'} and str(contract['activation']) != 'swiglu':
        raise ValueError(f"family {family_name!r} currently expects ck.nn.TransformerBlock(..., activation='swiglu')")
    pass_trace = default_pass_trace(compile_config)
    template = _build_template_spec(
        family=family_name,
        graph=graph,
        contract=contract,
        compile_config=compile_config,
        pass_trace=pass_trace,
    )
    tiny_model = TinyModelSpec(
        init=init,
        layers=contract['layers'],
        vocab_size=contract['embedding'].vocab,
        embed_dim=contract['dim'],
        hidden_dim=contract['hidden'],
        num_heads=contract['heads'],
        num_kv_heads=contract['kv_heads'],
        context_len=contract['context_len'],
        rope_theta=float(contract['rope_theta']),
        kernel_policy=compile_config.kernel_policy,
        adamw_beta1=adamw_beta1,
        adamw_beta2=adamw_beta2,
        adamw_eps=adamw_eps,
        adamw_weight_decay=adamw_weight_decay,
        seed=seed,
    )
    project = TrainingProject(
        run_name=str(run_name),
        run_dir=Path(run_dir).expanduser().resolve(strict=False) if run_dir is not None else None,
        model=tiny_model,
        template=template,
        tokenizer=TokenizerPlan(family=tokenizer_family, notes=tokenizer_notes),
        repo_root=Path(repo_root).expanduser().resolve(strict=False) if repo_root is not None else REPO_ROOT,
        python_exec=python_exec,
        command_runner=command_runner,
    )
    return CompiledProject(
        project=project,
        graph=graph,
        family=family_name,
        compile_config=compile_config,
        pass_trace=tuple(pass_trace),
    )
