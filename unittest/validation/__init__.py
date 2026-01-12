"""
Staged Kernel Validation System

This package provides a layered validation approach for kernel development:
  Stage 1: Weight conversion validation (GGUF -> BUMP)
  Stage 2: Dimension and memory planning validation
  Stage 3: Single layer activation validation vs llama.cpp

Usage:
    python validate_kernel.py --stage 1,2,3 --gguf model.gguf

Auto-validation (when gibberish is detected):
    from validation.auto_validate import AutoValidator
    validator = AutoValidator(gguf_path, bump_path, manifest_path)
    validator.check_output(tokens, text)
"""

from .base import StageResult, ValidationReport, ValidationError
from .kernel_registry import KernelSpec, KERNEL_REGISTRY, register_kernel
from .gibberish_detector import detect_gibberish, GibberishResult

__all__ = [
    'StageResult',
    'ValidationReport',
    'ValidationError',
    'KernelSpec',
    'KERNEL_REGISTRY',
    'register_kernel',
    'detect_gibberish',
    'GibberishResult',
]
