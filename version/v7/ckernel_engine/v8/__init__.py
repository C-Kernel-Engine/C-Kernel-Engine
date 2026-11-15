"""Python authoring adapter for the certified v8 generated-C training workflow."""

from .authoring import (
    CompiledTrainingExperiment,
    DatasetConfig,
    TokenizerConfig,
    TrainingConfig,
    compile,
)

__all__ = [
    "CompiledTrainingExperiment",
    "DatasetConfig",
    "TokenizerConfig",
    "TrainingConfig",
    "compile",
]


def __getattr__(name):
    if name in {'LoRAConfig', 'LoRAPlan', 'LoRASession', 'prepare_lora', 'lora_capabilities', 'lora_targets'}:
        from . import lora
        return getattr(lora, name)
    raise AttributeError(name)


__all__ += ['LoRAConfig', 'LoRAPlan', 'LoRASession', 'prepare_lora', 'lora_capabilities', 'lora_targets']
