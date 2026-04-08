#!/usr/bin/env python3
"""
Kernel Map Validator

Validates the current v7 kernel-map surface without assuming every JSON file
uses the same legacy schema. The validator is intended to be stable enough for
nightly use:

1. Valid JSON syntax
2. Required identity fields present
3. Executable contract present (`impl`, `signature`, or `source`)
4. Dims referenced in shapes are at least declared or recognized
5. Fused kernels reference existing kernel IDs
6. Source files resolve relative to the repo root
7. Optional JSON summary output

Usage:
    python validate_kernel_maps.py
    python validate_kernel_maps.py --verbose
    python validate_kernel_maps.py --strict
    python validate_kernel_maps.py --json-out /tmp/kernel_map_validation.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

SCRIPT_DIR = Path(__file__).parent.resolve()
KERNEL_MAPS_DIR = SCRIPT_DIR
REPO_ROOT = SCRIPT_DIR.parents[2]
METADATA_FILES = {
    "KERNEL_REGISTRY.json",
    "KERNEL_SOURCES.json",
    "kernel_bindings.json",
}

VALID_OPS = {
    "add_backward",
    "add_stream",
    "attention",
    "attention_projection",
    "attention_sliding",
    "attn_gate_sigmoid_mul",
    "attn_gate_sigmoid_mul_backward",
    "bias_add",
    "embedding",
    "embedding_backward",
    "fused_attention_block",
    "fused_linear_bias",
    "fused_mlp_block",
    "fused_qkv_block",
    "gated_deltanet",
    "geglu",
    "gelu",
    "gemm",
    "gemm_backward",
    "gemv",
    "kv_cache_store",
    "logits_store_indexed",
    "optimizer",
    "position_embeddings",
    "qk_norm",
    "qk_norm_backward",
    "qkv_projection",
    "quantize",
    "quantize_batch",
    "recurrent_conv_state_update",
    "recurrent_dt_gate",
    "recurrent_dt_gate_backward",
    "recurrent_norm_gate",
    "recurrent_qk_l2_norm",
    "recurrent_silu",
    "recurrent_split_conv_qkv",
    "recurrent_split_qkv",
    "recurrent_split_qkv_backward",
    "residual_add",
    "residual_save",
    "rmsnorm",
    "rope",
    "rope_backward",
    "rope_init",
    "rowwise_bias_add",
    "softmax",
    "spatial_merge",
    "split_q_gate",
    "split_q_gate_backward",
    "ssm_conv1d",
    "ssm_conv1d_backward",
    "swiglu",
    "tokenizer",
    "vision_patchify",
}

VALID_DTYPES = {
    "bf16",
    "f32",
    "float",
    "float32",
    "fp16",
    "fp32",
    "fp64",
    "i32",
    "int8",
    "int16",
    "int32",
    "int64",
    "mixed",
    "none",
    "quant",
    "q4_0",
    "q4_1",
    "q4_k",
    "q5_0",
    "q5_1",
    "q5_k",
    "q6_k",
    "q8_0",
    "q8_k",
    "string",
    "u8",
    "uint8",
    "uint16",
    "uint32",
}

VALID_DIMS = {
    "1",
    "2I",
    "AAD",
    "AD",
    "AE",
    "AI",
    "B",
    "C",
    "D",
    "E",
    "H",
    "I",
    "K",
    "KV",
    "M",
    "N",
    "O",
    "P",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "aligned_embed",
    "aligned_embed_dim",
    "aligned_head",
    "aligned_head_dim",
    "aligned_intermediate",
    "aligned_intermediate_dim",
    "batch",
    "cache_capacity",
    "channels",
    "context_len",
    "dim",
    "embed_dim",
    "gate_dim",
    "group_dim",
    "head_dim",
    "height",
    "history_len",
    "intermediate",
    "intermediate_dim",
    "k_dim",
    "kv_tokens",
    "max_T",
    "max_context_len",
    "max_seq",
    "max_seq_len",
    "num_channels",
    "num_heads",
    "num_kv_heads",
    "num_layers",
    "num_merges",
    "num_patches",
    "num_seqs",
    "num_tokens",
    "patch",
    "position",
    "q_dim",
    "q_seq_len",
    "rows",
    "seq_len",
    "state_dim",
    "stream_elems",
    "text_len",
    "tokens",
    "total_vocab_bytes",
    "v_dim",
    "vocab",
    "vocab_size",
    "width",
}

VALID_QUANT_KEYS = {"weight", "activation", "output", "note"}
VALID_MODE_KEYS = {"inference", "training", "backward"}
COLLECTION_FIELDS = ("inputs", "outputs", "weights", "activations", "scratch", "params")
EXPR_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
IGNORED_EXPR_IDENTIFIERS = {"min", "max", "ceil", "floor", "align_up", "sizeof", "true", "false"}


def _is_map_file(path: Path) -> bool:
    return path.suffix == ".json" and path.name not in METADATA_FILES


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _extract_expr_symbols(value: Any) -> list[str]:
    if isinstance(value, (int, float)):
        return []
    if isinstance(value, dict):
        if "dim" in value:
            return _extract_expr_symbols(value["dim"])
        return []
    if not isinstance(value, str):
        return []
    return [
        token
        for token in EXPR_IDENTIFIER_RE.findall(value)
        if token not in IGNORED_EXPR_IDENTIFIERS
    ]


class KernelMapValidator:
    def __init__(self, verbose: bool = False, check_paths: bool = True):
        self.verbose = verbose
        self.check_paths = check_paths
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passed = 0
        self.failed = 0
        self.all_kernel_ids: set[str] = set()
        self.kernel_files: dict[str, Path] = {}

    def _emit(self, prefix: str, msg: str, color: str, always: bool = False) -> None:
        if always or self.verbose:
            print(f"  {color}{prefix}{RESET} {msg}")

    def log_pass(self, msg: str) -> None:
        self.passed += 1
        self._emit("✓", msg, GREEN)

    def log_fail(self, msg: str) -> None:
        self.failed += 1
        self.errors.append(msg)
        self._emit("✗", msg, RED, always=self.verbose)

    def log_warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self._emit("!", msg, YELLOW, always=self.verbose)

    def load_all_kernel_maps(self) -> bool:
        json_files = sorted(path for path in KERNEL_MAPS_DIR.glob("*.json") if _is_map_file(path))
        if not json_files:
            self.log_fail("No kernel map JSON files found")
            return False

        for json_file in json_files:
            try:
                data = _load_json(json_file)
            except json.JSONDecodeError as exc:
                self.log_fail(f"Invalid JSON in {json_file.name}: {exc}")
                return False
            except Exception as exc:
                self.log_fail(f"Error loading {json_file.name}: {exc}")
                return False

            if isinstance(data, dict):
                kernel_id = data.get("id")
                if isinstance(kernel_id, str) and kernel_id:
                    self.all_kernel_ids.add(kernel_id)
                    self.kernel_files[kernel_id] = json_file

        if self.verbose:
            print(f"{GREEN}Loaded {len(self.all_kernel_ids)} kernel maps{RESET}")
        return True

    def validate_quant(self, name: str, quant: Any) -> None:
        if not isinstance(quant, dict):
            self.log_fail(f"{name}.quant: Must be an object")
            return

        for key, dtype in quant.items():
            if key not in VALID_QUANT_KEYS:
                self.log_warn(f"{name}.quant: Unknown key '{key}'")
            if key == "note":
                continue
            if dtype is None and key == "weight":
                continue
            if dtype is None:
                self.log_warn(f"{name}.quant.{key}: Null dtype")
                continue
            if not isinstance(dtype, str):
                self.log_warn(f"{name}.quant.{key}: Non-string dtype {type(dtype).__name__}")
                continue
            if dtype in VALID_DTYPES or "|" in dtype:
                continue
            self.log_warn(f"{name}.quant.{key}: Unknown dtype '{dtype}'")

    def validate_io_spec(self, name: str, spec: Any, io_type: str, declared_dims: set[str]) -> None:
        if not isinstance(spec, dict):
            self.log_fail(f"{name}: {io_type} entry must be an object")
            return
        if "name" not in spec:
            self.log_fail(f"{name}: {io_type} missing 'name'")
        if io_type == "param":
            if "dtype" not in spec and "type" not in spec:
                self.log_fail(f"{name}: {io_type} missing 'dtype' or 'type'")
        elif "dtype" not in spec:
            self.log_fail(f"{name}: {io_type} missing 'dtype'")
        dtype = spec.get("dtype")
        if isinstance(dtype, str) and dtype not in VALID_DTYPES and "|" not in dtype:
            self.log_warn(f"{name}: {io_type} dtype uses unknown token '{dtype}'")

        shape = spec.get("shape")
        if isinstance(shape, list):
            for dim in shape:
                for symbol in _extract_expr_symbols(dim):
                    if symbol not in declared_dims and symbol not in VALID_DIMS:
                        self.log_warn(f"{name}: {io_type} shape uses unknown dim '{symbol}'")
        elif shape is not None and not isinstance(shape, (str, dict, int, float)):
            self.log_fail(f"{name}: {io_type} shape must be a list, string, dict, or scalar")

    def validate_dims(self, name: str, dims: Any) -> set[str]:
        if dims is None:
            return set()
        if not isinstance(dims, list):
            self.log_fail(f"{name}.dims: Must be a list")
            return set()
        declared = set()
        for dim in dims:
            if isinstance(dim, str):
                declared.add(dim)
                if dim not in VALID_DIMS:
                    self.log_warn(f"{name}.dims: Unknown dim '{dim}'")
            else:
                self.log_warn(f"{name}.dims: Non-string dim '{dim}'")
        return declared

    def validate_impl(self, name: str, impl: Any) -> None:
        if not isinstance(impl, dict):
            self.log_fail(f"{name}.impl: Must be an object")
            return
        function = impl.get("function")
        if not isinstance(function, str) or not function:
            self.log_fail(f"{name}.impl: Missing 'function'")
        sources = impl.get("sources")
        if not isinstance(sources, list):
            self.log_fail(f"{name}.impl: Missing 'sources'")
            return
        if self.check_paths:
            for src in sources:
                if not isinstance(src, str):
                    self.log_warn(f"{name}.impl.sources: Non-string path '{src}'")
                    continue
                if not _resolve_repo_path(src).exists():
                    self.log_warn(f"{name}: Source file not found: {src}")
        variants = impl.get("variants")
        if variants is not None and not isinstance(variants, list):
            self.log_fail(f"{name}.impl.variants: Must be a list when present")

    def validate_signature_source(self, name: str, data: dict[str, Any]) -> None:
        signature = data.get("signature")
        source = data.get("source")
        if signature is not None and not isinstance(signature, dict):
            self.log_fail(f"{name}.signature: Must be an object")
        if source is not None and not isinstance(source, dict):
            self.log_fail(f"{name}.source: Must be an object")
            return
        if self.check_paths and isinstance(source, dict):
            source_file = source.get("file")
            source_dir = None
            if isinstance(source_file, str):
                source_dir = _resolve_repo_path(source_file).parent
            for key in ("file", "header"):
                value = source.get(key)
                if isinstance(value, str) and not _resolve_repo_path(value).exists():
                    self.log_warn(f"{name}: Source file not found: {value}")
            deps = source.get("dependencies")
            if isinstance(deps, list):
                for dep in deps:
                    if not isinstance(dep, str):
                        continue
                    dep_path = _resolve_repo_path(dep)
                    if dep_path.exists():
                        continue
                    if source_dir is not None and (source_dir / dep).exists():
                        continue
                    self.log_warn(f"{name}: Source file not found: {dep}")

    def validate_fuses(self, name: str, fuses: Any) -> None:
        if not isinstance(fuses, list):
            self.log_fail(f"{name}.fuses: Must be a list")
            return
        for fused_kernel in fuses:
            if not isinstance(fused_kernel, str):
                self.log_warn(f"{name}.fuses: Non-string fused kernel '{fused_kernel}'")
                continue
            if fused_kernel not in self.all_kernel_ids:
                self.log_warn(f"{name}: Fuses unknown kernel '{fused_kernel}'")

    def validate_modes(self, name: str, modes: Any) -> None:
        if modes is None:
            return
        if not isinstance(modes, dict):
            self.log_fail(f"{name}.modes: Must be an object")
            return
        for key, value in modes.items():
            if key not in VALID_MODE_KEYS:
                self.log_warn(f"{name}.modes: Unknown key '{key}'")
            if not isinstance(value, bool):
                self.log_warn(f"{name}.modes.{key}: Expected bool, got {type(value).__name__}")

    def validate_file(self, json_file: Path) -> bool:
        name = json_file.name
        errors_before = len(self.errors)
        try:
            data = _load_json(json_file)
        except json.JSONDecodeError as exc:
            self.log_fail(f"{name}: Invalid JSON - {exc}")
            return False
        except Exception as exc:
            self.log_fail(f"{name}: Could not load - {exc}")
            return False

        if not isinstance(data, dict):
            self.log_fail(f"{name}: Root must be an object")
            return False

        kernel_id = data.get("id")
        op = data.get("op")
        if not isinstance(kernel_id, str) or not kernel_id:
            self.log_fail(f"{name}: Missing required field 'id'")
        if not isinstance(op, str) or not op:
            self.log_fail(f"{name}: Missing required field 'op'")
        elif op not in VALID_OPS:
            self.log_warn(f"{name}: Unknown op '{op}'")

        declared_dims = self.validate_dims(name, data.get("dims"))

        for field in COLLECTION_FIELDS:
            value = data.get(field)
            if value is None:
                continue
            if isinstance(value, list):
                specs = value
            elif isinstance(value, dict):
                specs = []
                for key, spec in value.items():
                    if isinstance(spec, dict) and "name" not in spec:
                        spec = dict(spec)
                        spec["name"] = key
                    specs.append(spec)
            else:
                self.log_fail(f"{name}.{field}: Must be a list or object")
                continue
            for spec in specs:
                self.validate_io_spec(name, spec, field[:-1] if field.endswith("s") else field, declared_dims)

        if "quant" in data:
            self.validate_quant(name, data["quant"])
        self.validate_modes(name, data.get("modes"))

        has_impl = "impl" in data
        has_signature = "signature" in data
        has_source = "source" in data
        if not (has_impl or has_signature or has_source):
            self.log_fail(f"{name}: Missing executable contract ('impl', 'signature', or 'source')")
        if has_impl:
            self.validate_impl(name, data.get("impl"))
        if has_signature or has_source:
            self.validate_signature_source(name, data)
        if "fuses" in data:
            self.validate_fuses(name, data["fuses"])

        passed = len(self.errors) == errors_before
        if passed:
            self.log_pass(name)
        else:
            self.failed += 0
        return passed

    def summary(self, strict: bool = False) -> dict[str, Any]:
        status = "fail" if self.errors or (strict and self.warnings) else "pass"
        return {
            "summary": {
                "status": status,
                "kernel_maps": len(self.kernel_files),
                "passed": self.passed,
                "failed": self.failed,
                "warnings": len(self.warnings),
                "strict": strict,
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def validate_all(self, strict: bool = False) -> bool:
        print(f"\n{CYAN}Validating kernel maps in: {KERNEL_MAPS_DIR}{RESET}\n")
        if not self.load_all_kernel_maps():
            return False
        for json_file in sorted(path for path in KERNEL_MAPS_DIR.glob("*.json") if _is_map_file(path)):
            self.validate_file(json_file)

        summary = self.summary(strict=strict)
        print(f"{'=' * 60}")
        print(f"{CYAN}VALIDATION SUMMARY{RESET}")
        print(f"{'=' * 60}")
        print(f"  {GREEN}Passed: {summary['summary']['passed']}{RESET}")
        print(f"  {RED}Failed: {summary['summary']['failed']}{RESET}")
        print(f"  {YELLOW}Warnings: {summary['summary']['warnings']}{RESET}")

        if self.errors:
            print(f"\n{RED}Errors:{RESET}")
            for error in self.errors[:10]:
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        if self.warnings:
            print(f"\n{YELLOW}Warnings:{RESET}")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        print()
        return summary["summary"]["status"] == "pass"


def validate_kernel_map(path: str | Path, check_paths: bool = True) -> tuple[list[str], list[str]]:
    validator = KernelMapValidator(verbose=False, check_paths=check_paths)
    validator.load_all_kernel_maps()
    validator.validate_file(Path(path))
    return validator.errors, validator.warnings


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate kernel map JSON files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--json-out", type=Path, help="Write JSON summary report")
    args = parser.parse_args()

    validator = KernelMapValidator(verbose=args.verbose)
    success = validator.validate_all(strict=args.strict)
    summary = validator.summary(strict=args.strict)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if success:
        print(f"{GREEN}Kernel map validation passed{RESET}")
        raise SystemExit(0)

    print(f"{RED}Kernel map validation failed{RESET}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
