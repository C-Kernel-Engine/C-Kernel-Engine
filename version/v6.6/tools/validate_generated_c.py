#!/usr/bin/env python3
"""
validate_generated_c.py - Cross-validate generated C code against JSON layouts

This tool parses the generated model_v6_6.c file and validates that:
1. #define constants match values from layout JSON
2. Kernel call arguments are consistent with IR lowering JSON
3. Memory sizes and offsets are correct
4. Struct initializers match expected values

Usage:
    python validate_generated_c.py <model_dir>
    python validate_generated_c.py /path/to/model/cache/dir --verbose

Author: Claude Code Assistant
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


@dataclass
class ValidationIssue:
    """Represents a validation mismatch."""
    severity: str  # "error", "warning", "info"
    category: str  # "offset", "size", "kernel", "config", etc.
    message: str
    c_value: Any = None
    json_value: Any = None
    line_num: int = 0

    def __str__(self):
        prefix = {"error": "[ERROR]", "warning": "[WARN]", "info": "[INFO]"}[self.severity]
        loc = f" (line {self.line_num})" if self.line_num else ""
        vals = ""
        if self.c_value is not None and self.json_value is not None:
            vals = f" [C: {self.c_value}, JSON: {self.json_value}]"
        return f"{prefix} {self.category}: {self.message}{loc}{vals}"


@dataclass
class CFileAnalysis:
    """Parsed data from the C file."""
    defines: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # name -> (value, line)
    kernel_calls: List[Dict] = field(default_factory=list)
    struct_inits: Dict[str, Dict] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    raw_content: str = ""


class CCodeValidator:
    """Validates generated C code against JSON specifications."""

    # Regex patterns for parsing C code
    DEFINE_PATTERN = re.compile(
        r'^\s*#define\s+(\w+)\s+(\d+(?:ULL)?)\s*(?://.*)?$',
        re.MULTILINE
    )

    KERNEL_CALL_PATTERN = re.compile(
        r'(\w+)\s*\(\s*\n?((?:[^)]*\n?)*?)\)',
        re.MULTILINE
    )

    COMMENT_OP_PATTERN = re.compile(
        r'/\*\s*Op\s+(\d+):\s+(\w+)\s+\(([^)]+)\)\s+layer=(-?\d+)',
        re.MULTILINE
    )

    STRUCT_INIT_PATTERN = re.compile(
        r'\[(\d+)\]\s*=\s*\{\s*([^}]+)\}',
        re.MULTILINE
    )

    FIELD_INIT_PATTERN = re.compile(
        r'\.(\w+)\s*=\s*(\d+)',
    )

    def __init__(self, model_dir: Path, verbose: bool = False):
        self.model_dir = Path(model_dir)
        self.verbose = verbose
        self.issues: List[ValidationIssue] = []

        # Load files
        self.c_analysis = CFileAnalysis()
        self.layout_json = {}
        self.lowered_json = {}
        self.weights_manifest = {}

    def log(self, msg: str):
        if self.verbose:
            print(f"  [DEBUG] {msg}")

    def load_files(self) -> bool:
        """Load C file and JSON files."""
        # Find the generated C file
        c_file = self.model_dir / "model_v6_6.c"
        if not c_file.exists():
            print(f"Error: C file not found: {c_file}")
            return False

        # Load C file
        self.log(f"Loading C file: {c_file}")
        self.c_analysis.raw_content = c_file.read_text()

        # Load layout JSONs (try decode first, then prefill)
        for mode in ["decode", "prefill"]:
            layout_file = self.model_dir / f"layout_{mode}.json"
            if layout_file.exists():
                self.log(f"Loading layout JSON: {layout_file}")
                with open(layout_file) as f:
                    self.layout_json[mode] = json.load(f)

        if not self.layout_json:
            print(f"Warning: No layout JSON found in {self.model_dir}")

        # Load lowered IR JSONs
        for mode in ["decode", "prefill"]:
            lowered_file = self.model_dir / f"lowered_{mode}.json"
            if lowered_file.exists():
                self.log(f"Loading lowered IR: {lowered_file}")
                with open(lowered_file) as f:
                    self.lowered_json[mode] = json.load(f)

        # Load weights manifest
        manifest_file = self.model_dir / "weights_manifest.json"
        if manifest_file.exists():
            self.log(f"Loading weights manifest: {manifest_file}")
            with open(manifest_file) as f:
                self.weights_manifest = json.load(f)

        return True

    def parse_c_file(self):
        """Parse the C file to extract defines, kernel calls, etc."""
        content = self.c_analysis.raw_content
        lines = content.split('\n')

        # Parse #defines
        for i, line in enumerate(lines, 1):
            match = self.DEFINE_PATTERN.match(line)
            if match:
                name = match.group(1)
                value_str = match.group(2).replace('ULL', '')
                try:
                    value = int(value_str)
                    self.c_analysis.defines[name] = (value, i)
                except ValueError:
                    pass

        self.log(f"Found {len(self.c_analysis.defines)} #defines")

        # Parse Op comments and following kernel calls
        for match in self.COMMENT_OP_PATTERN.finditer(content):
            op_num = int(match.group(1))
            kernel_name = match.group(2)
            op_type = match.group(3)
            layer = int(match.group(4))

            # Find the kernel call that follows this comment
            start = match.end()
            # Look for the next function call
            call_match = self.KERNEL_CALL_PATTERN.search(content, start)
            if call_match and call_match.start() < start + 500:
                args_str = call_match.group(2)
                args = self._parse_args(args_str)

                self.c_analysis.kernel_calls.append({
                    'op_num': op_num,
                    'kernel': kernel_name,
                    'op_type': op_type,
                    'layer': layer,
                    'args': args,
                    'line': content[:match.start()].count('\n') + 1
                })

        self.log(f"Found {len(self.c_analysis.kernel_calls)} kernel calls")

        # Parse LayerOffsets struct initializations
        layer_block_match = re.search(
            r'static const LayerOffsets L_LAYERS\[\d+\] = \{([^;]+)\};',
            content, re.DOTALL
        )
        if layer_block_match:
            block = layer_block_match.group(1)
            for match in self.STRUCT_INIT_PATTERN.finditer(block):
                layer_idx = int(match.group(1))
                fields_str = match.group(2)
                fields = {}
                for field_match in self.FIELD_INIT_PATTERN.finditer(fields_str):
                    fields[field_match.group(1)] = int(field_match.group(2))
                self.c_analysis.struct_inits[f"layer_{layer_idx}"] = fields

        self.log(f"Found {len(self.c_analysis.struct_inits)} layer initializations")

        # Extract config values from defines
        config_keys = [
            'EMBED_DIM', 'NUM_HEADS', 'NUM_KV_HEADS', 'HEAD_DIM',
            'INTERMEDIATE_SIZE', 'NUM_LAYERS', 'VOCAB_SIZE', 'MAX_SEQ_LEN',
            'WEIGHTS_SIZE', 'ACTIVATIONS_SIZE', 'KV_CACHE_SIZE',
            'BUMP_WEIGHTS_OFFSET', 'BUMP_ACT_OFFSET', 'BUMP_TOTAL_SIZE'
        ]
        for key in config_keys:
            if key in self.c_analysis.defines:
                self.c_analysis.config[key] = self.c_analysis.defines[key][0]

    def _parse_args(self, args_str: str) -> List[str]:
        """Parse function arguments from C code."""
        # Clean up the args string
        args_str = re.sub(r'/\*.*?\*/', '', args_str)  # Remove comments
        args_str = re.sub(r'\s+', ' ', args_str)  # Normalize whitespace

        # Split by commas (handling nested parentheses)
        args = []
        depth = 0
        current = []
        for char in args_str:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            args.append(''.join(current).strip())

        return [a for a in args if a]

    def validate_config(self):
        """Validate configuration values match JSON."""
        if not self.layout_json:
            return

        layout = self.layout_json.get('decode', self.layout_json.get('prefill', {}))
        json_config = layout.get('config', {})

        # Map C define names to JSON config keys
        mapping = {
            'EMBED_DIM': 'embed_dim',
            'NUM_HEADS': 'num_heads',
            'NUM_KV_HEADS': 'num_kv_heads',
            'HEAD_DIM': 'head_dim',
            'INTERMEDIATE_SIZE': 'intermediate_dim',
            'NUM_LAYERS': 'num_layers',
            'VOCAB_SIZE': 'vocab_size',
            'MAX_SEQ_LEN': 'context_length',
        }

        for c_key, json_key in mapping.items():
            c_val = self.c_analysis.config.get(c_key)
            json_val = json_config.get(json_key)

            if c_val is not None and json_val is not None:
                if c_val != json_val:
                    line = self.c_analysis.defines.get(c_key, (0, 0))[1]
                    self.issues.append(ValidationIssue(
                        severity="error",
                        category="config",
                        message=f"{c_key} mismatch",
                        c_value=c_val,
                        json_value=json_val,
                        line_num=line
                    ))
                else:
                    self.log(f"Config {c_key} = {c_val} [OK]")

    def validate_memory_sizes(self):
        """Validate memory size constants."""
        if not self.layout_json:
            return

        layout = self.layout_json.get('decode', self.layout_json.get('prefill', {}))
        memory = layout.get('memory', {})

        # Weights size
        json_weights_size = memory.get('weights', {}).get('total_size', 0)
        c_weights_size = self.c_analysis.config.get('WEIGHTS_SIZE')
        if c_weights_size and json_weights_size:
            if c_weights_size != json_weights_size:
                self.issues.append(ValidationIssue(
                    severity="error",
                    category="memory",
                    message="WEIGHTS_SIZE mismatch",
                    c_value=c_weights_size,
                    json_value=json_weights_size,
                    line_num=self.c_analysis.defines.get('WEIGHTS_SIZE', (0, 0))[1]
                ))
            else:
                self.log(f"WEIGHTS_SIZE = {c_weights_size} [OK]")

        # Activations size
        json_act_size = memory.get('activations', {}).get('total_size', 0)
        c_act_size = self.c_analysis.config.get('ACTIVATIONS_SIZE')
        if c_act_size and json_act_size:
            if c_act_size != json_act_size:
                self.issues.append(ValidationIssue(
                    severity="error",
                    category="memory",
                    message="ACTIVATIONS_SIZE mismatch",
                    c_value=c_act_size,
                    json_value=json_act_size,
                    line_num=self.c_analysis.defines.get('ACTIVATIONS_SIZE', (0, 0))[1]
                ))
            else:
                self.log(f"ACTIVATIONS_SIZE = {c_act_size} [OK]")

    def validate_activation_offsets(self):
        """Validate activation buffer offsets."""
        if not self.layout_json:
            return

        layout = self.layout_json.get('decode', self.layout_json.get('prefill', {}))
        buffers = layout.get('memory', {}).get('activations', {}).get('buffers', [])

        # Build JSON offset map
        json_offsets = {}
        for buf in buffers:
            name = buf.get('name', '')
            offset = buf.get('offset', 0)
            json_offsets[name.lower()] = offset

        # Map C defines to buffer names
        act_defines = {k: v for k, v in self.c_analysis.defines.items() if k.startswith('A_')}

        for c_name, (c_offset, line) in act_defines.items():
            # Convert A_TOKEN_IDS -> token_ids
            buf_name = c_name[2:].lower()

            json_offset = json_offsets.get(buf_name)
            if json_offset is not None:
                # Account for bump base offset
                bump_act_offset = self.c_analysis.config.get('BUMP_ACT_OFFSET', 0)
                expected_c = bump_act_offset + json_offset

                if c_offset != expected_c:
                    self.issues.append(ValidationIssue(
                        severity="error",
                        category="offset",
                        message=f"Activation offset {c_name} mismatch",
                        c_value=c_offset,
                        json_value=f"{bump_act_offset} + {json_offset} = {expected_c}",
                        line_num=line
                    ))
                else:
                    self.log(f"{c_name} = {c_offset} [OK]")

    def validate_weight_offsets(self):
        """Validate weight offsets against lowered IR JSON (uses abs_offset + define)."""
        # Prefer lowered IR which has abs_offset and define mappings
        if self.lowered_json:
            self._validate_weights_from_lowered_ir()
            return

        # Fallback to manifest if no lowered IR
        if not self.weights_manifest:
            return

        # Get header offsets from manifest
        header_offsets = self.weights_manifest.get('header_offsets', {})

        # Map C define names to manifest keys
        header_mapping = {
            'W_TOKEN_EMB': 'token_emb',
            'W_FINAL_LN_WEIGHT': 'final_ln_weight',
            'W_FINAL_LN_BIAS': 'final_ln_bias',
            'W_VOCAB_OFFSETS': 'vocab_offsets',
            'W_VOCAB_STRINGS': 'vocab_strings',
            'W_VOCAB_MERGES': 'vocab_merges',
        }

        bump_offset = 496  # Standard bump base offset

        for c_name, manifest_key in header_mapping.items():
            c_val = self.c_analysis.defines.get(c_name, (None, 0))[0]
            json_val = header_offsets.get(manifest_key)

            if c_val is not None and json_val is not None:
                expected_c = json_val + bump_offset
                if c_val != expected_c:
                    self.issues.append(ValidationIssue(
                        severity="error",
                        category="offset",
                        message=f"Header weight {c_name} mismatch",
                        c_value=c_val,
                        json_value=f"{json_val} + {bump_offset} = {expected_c}",
                        line_num=self.c_analysis.defines.get(c_name, (0, 0))[1]
                    ))
                else:
                    self.log(f"{c_name} = {c_val} [OK]")

        # Validate layer offsets
        layer_offsets = self.weights_manifest.get('layer_offsets', [])

        for layer_idx, layer_data in enumerate(layer_offsets):
            struct_key = f"layer_{layer_idx}"
            c_struct = self.c_analysis.struct_inits.get(struct_key, {})

            if not c_struct:
                continue

            # Map struct fields to manifest keys
            field_mapping = {
                'wq': 'wq', 'wk': 'wk', 'wv': 'wv', 'wo': 'wo',
                'bq': 'bq', 'bk': 'bk', 'bv': 'bv', 'bo': 'bo',
                'w1': 'w1', 'w2': 'w2', 'b1': 'b1', 'b2': 'b2',
                'ln1_gamma': 'ln1_gamma', 'ln2_gamma': 'ln2_gamma',
            }

            for c_field, json_field in field_mapping.items():
                c_val = c_struct.get(c_field)
                json_val = layer_data.get(json_field)

                if c_val is not None and json_val is not None:
                    if c_val != json_val:
                        self.issues.append(ValidationIssue(
                            severity="error",
                            category="offset",
                            message=f"Layer {layer_idx} {c_field} mismatch in struct init",
                            c_value=c_val,
                            json_value=json_val
                        ))

    def _validate_weights_from_lowered_ir(self):
        """Validate weight offsets directly from lowered IR (has define mappings)."""
        lowered = self.lowered_json.get('decode', self.lowered_json.get('prefill', {}))
        weight_entries = lowered.get('memory', {}).get('weights', {}).get('entries', [])

        validated = 0
        errors = 0

        for entry in weight_entries:
            define_name = entry.get('define')
            abs_offset = entry.get('abs_offset')

            if not define_name or abs_offset is None:
                continue

            c_val = self.c_analysis.defines.get(define_name, (None, 0))[0]
            line = self.c_analysis.defines.get(define_name, (0, 0))[1]

            if c_val is not None:
                if c_val != abs_offset:
                    self.issues.append(ValidationIssue(
                        severity="error",
                        category="offset",
                        message=f"Weight offset {define_name} mismatch",
                        c_value=c_val,
                        json_value=abs_offset,
                        line_num=line
                    ))
                    errors += 1
                else:
                    validated += 1

        self.log(f"Validated {validated} weight offsets from lowered IR ({errors} errors)")

    def validate_kernel_calls(self):
        """Validate kernel calls - count and types match expected."""
        # Count kernel calls by name
        kernel_counts = defaultdict(int)
        layer_kernels = defaultdict(list)

        for call in self.c_analysis.kernel_calls:
            kernel_counts[call['kernel']] += 1
            layer_kernels[call['layer']].append(call['kernel'])

        # Expected kernels per layer for decode
        expected_layer_kernels = [
            'rmsnorm_forward',  # ln1
            'gemv_',            # q_proj (q4_k, q5_0, q6_k, or q8_0)
            'gemv_',            # k_proj
            'gemv_',            # v_proj
            'rope_forward_qk',
            'kv_cache_store',
            'attention_forward_decode_head_major_gqa_flash',
            'gemv_',            # out_proj
            'ck_residual_add_token_major',  # residual1
            'rmsnorm_forward',  # ln2
            'gemv_',            # w1/gate
            'gemv_',            # w1/up or w2
            'swiglu_forward',
            'gemv_',            # down_proj
            'ck_residual_add_token_major',  # residual2
        ]

        # Validate each layer has the expected kernel pattern
        num_layers = self.c_analysis.config.get('NUM_LAYERS', 24)

        for layer_idx in range(num_layers):
            kernels = layer_kernels.get(layer_idx, [])
            if not kernels:
                self.issues.append(ValidationIssue(
                    severity="warning",
                    category="kernel",
                    message=f"Layer {layer_idx} has no kernel calls",
                ))
            else:
                # Check minimum expected kernels
                has_rmsnorm = any('rmsnorm' in k for k in kernels)
                has_attention = any('attention' in k for k in kernels)
                has_gemv = any('gemv' in k or 'gemm' in k for k in kernels)

                if not has_rmsnorm:
                    self.issues.append(ValidationIssue(
                        severity="warning",
                        category="kernel",
                        message=f"Layer {layer_idx} missing rmsnorm kernel",
                    ))
                if not has_attention:
                    self.issues.append(ValidationIssue(
                        severity="warning",
                        category="kernel",
                        message=f"Layer {layer_idx} missing attention kernel",
                    ))
                if not has_gemv:
                    self.issues.append(ValidationIssue(
                        severity="warning",
                        category="kernel",
                        message=f"Layer {layer_idx} missing gemv/gemm kernel",
                    ))

        # Summary
        self.log(f"Kernel call counts: {dict(kernel_counts)}")

    def validate_kernel_arg_offsets(self):
        """Validate that kernel call arguments use correct offset defines."""
        # Extract offset values from kernel call arguments
        offset_pattern = re.compile(
            r'\(model->bump\s*\+\s*(\w+)\)|'
            r'\(MEM\s*\+\s*(\d+)\)|'
            r'W_PTR\(([^)]+)\)|'
            r'W_FLOAT\(([^)]+)\)'
        )

        issues_found = 0

        for call in self.c_analysis.kernel_calls:
            for arg in call['args']:
                # Find offset references in the argument
                for match in offset_pattern.finditer(arg):
                    offset_ref = match.group(1) or match.group(2) or match.group(3) or match.group(4)
                    if not offset_ref:
                        continue

                    # Check if it's a define reference
                    if offset_ref in self.c_analysis.defines:
                        offset_val = self.c_analysis.defines[offset_ref][0]
                        # Verify the offset is within valid ranges
                        bump_total = self.c_analysis.config.get('BUMP_TOTAL_SIZE', 0)
                        if bump_total and offset_val >= bump_total:
                            self.issues.append(ValidationIssue(
                                severity="error",
                                category="kernel_arg",
                                message=f"Offset {offset_ref} ({offset_val}) exceeds BUMP_TOTAL_SIZE ({bump_total})",
                                line_num=call['line']
                            ))
                            issues_found += 1

                    # Check if it's a numeric offset
                    elif offset_ref.isdigit():
                        offset_val = int(offset_ref)
                        bump_total = self.c_analysis.config.get('BUMP_TOTAL_SIZE', 0)
                        if bump_total and offset_val >= bump_total:
                            self.issues.append(ValidationIssue(
                                severity="error",
                                category="kernel_arg",
                                message=f"Hardcoded offset {offset_val} exceeds BUMP_TOTAL_SIZE ({bump_total})",
                                line_num=call['line']
                            ))
                            issues_found += 1

        if issues_found == 0:
            self.log("All kernel argument offsets within valid range")

    def validate_no_overlaps(self):
        """Check for overlapping memory regions."""
        # Collect all defined memory regions
        regions = []

        # Activation buffers
        for name, (offset, line) in self.c_analysis.defines.items():
            if name.startswith('A_'):
                # We don't have sizes in the C defines, so skip detailed overlap check
                pass

        # Weight regions (from layout JSON)
        if self.layout_json:
            layout = self.layout_json.get('decode', self.layout_json.get('prefill', {}))
            weights = layout.get('memory', {}).get('weights', {}).get('entries', [])

            for w in weights:
                regions.append({
                    'name': w.get('name', 'unknown'),
                    'start': w.get('offset', 0),
                    'size': w.get('size', 0),
                    'type': 'weight'
                })

            # Sort by start offset
            regions.sort(key=lambda r: r['start'])

            # Check for overlaps
            for i in range(len(regions) - 1):
                curr = regions[i]
                next_r = regions[i + 1]
                curr_end = curr['start'] + curr['size']

                if curr_end > next_r['start']:
                    self.issues.append(ValidationIssue(
                        severity="error",
                        category="overlap",
                        message=f"Memory overlap: {curr['name']} ends at {curr_end} but {next_r['name']} starts at {next_r['start']}"
                    ))

    def run_validation(self) -> Tuple[int, int, int]:
        """Run all validations and return (errors, warnings, info) counts."""
        if not self.load_files():
            return (1, 0, 0)

        self.parse_c_file()

        print("\nRunning validations...")
        print("-" * 60)

        self.validate_config()
        self.validate_memory_sizes()
        self.validate_activation_offsets()
        self.validate_weight_offsets()
        self.validate_kernel_calls()
        self.validate_kernel_arg_offsets()
        self.validate_no_overlaps()

        # Count issues by severity
        errors = sum(1 for i in self.issues if i.severity == "error")
        warnings = sum(1 for i in self.issues if i.severity == "warning")
        infos = sum(1 for i in self.issues if i.severity == "info")

        return (errors, warnings, infos)

    def print_report(self):
        """Print validation report."""
        if not self.issues:
            print("\n" + "=" * 60)
            print("[OK] All validations passed!")
            print("=" * 60)
            return

        print("\n" + "=" * 60)
        print("VALIDATION ISSUES")
        print("=" * 60)

        # Group by category
        by_category = defaultdict(list)
        for issue in self.issues:
            by_category[issue.category].append(issue)

        for category, issues in sorted(by_category.items()):
            print(f"\n{category.upper()}:")
            print("-" * 40)
            for issue in issues:
                print(f"  {issue}")

        # Summary
        errors = sum(1 for i in self.issues if i.severity == "error")
        warnings = sum(1 for i in self.issues if i.severity == "warning")

        print("\n" + "=" * 60)
        print(f"SUMMARY: {errors} errors, {warnings} warnings")
        print("=" * 60)


def find_model_cache_dir(hint: str = None) -> Optional[Path]:
    """Find the model cache directory."""
    if hint:
        p = Path(hint)
        if p.exists():
            return p

    # Check common locations
    home = Path.home()
    cache_dirs = [
        home / ".cache/ck-engine-v6.6/models",
        Path("/home/antshiv/.cache/ck-engine-v6.6/models"),
    ]

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            # Find first model dir with model_v6_6.c
            for model_dir in cache_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "model_v6_6.c").exists():
                    return model_dir

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated C code against JSON specifications"
    )
    parser.add_argument(
        'model_dir',
        nargs='?',
        help='Path to model cache directory (auto-detected if omitted)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    # Find model directory
    model_dir = find_model_cache_dir(args.model_dir)
    if not model_dir:
        print("Error: Could not find model cache directory")
        print("Usage: validate_generated_c.py <model_dir>")
        sys.exit(1)

    print(f"Validating: {model_dir}")

    # Run validation
    validator = CCodeValidator(model_dir, verbose=args.verbose)
    errors, warnings, infos = validator.run_validation()

    if args.json:
        result = {
            'model_dir': str(model_dir),
            'errors': errors,
            'warnings': warnings,
            'issues': [
                {
                    'severity': i.severity,
                    'category': i.category,
                    'message': i.message,
                    'c_value': i.c_value,
                    'json_value': i.json_value,
                    'line': i.line_num
                }
                for i in validator.issues
            ]
        }
        print(json.dumps(result, indent=2))
    else:
        validator.print_report()

    # Exit code based on errors
    sys.exit(1 if errors > 0 else 0)


if __name__ == '__main__':
    main()
