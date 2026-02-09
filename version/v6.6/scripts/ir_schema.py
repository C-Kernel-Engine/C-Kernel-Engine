#!/usr/bin/env python3
"""
IR JSON Schema validation for C-Kernel-Engine v6.6.

Supports validation of:
- IRv1: Intermediate representation format (operations-based)
- Layout: Memory layout specifications
- Kernel Map: Kernel binding specifications
"""

import json
import sys
from typing import Any, Optional, Tuple


# =============================================================================
# IRv1 Schema - Intermediate Representation for model execution
# =============================================================================
IR1_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "IRv1 - C-Kernel Intermediate Representation",
    "type": "object",
    "required": ["version", "operations"],
    "properties": {
        "version": {
            "type": "string",
            "pattern": r"^v?\d+\.\d+$",
            "description": "IR version identifier"
        },
        "metadata": {
            "type": "object",
            "description": "Model metadata"
        },
        "operations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["idx", "op", "kernel"],
                "properties": {
                    "idx": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Operation index"
                    },
                    "op": {
                        "type": "string",
                        "description": "Operation type (e.g., attention, mlp, embedding)"
                    },
                    "kernel": {
                        "type": "string",
                        "description": "Kernel identifier from registry"
                    },
                    "function": {
                        "type": "string",
                        "description": "C function to invoke"
                    },
                    "layer": {
                        "type": "integer",
                        "description": "Layer number"
                    },
                    "section": {
                        "type": "string",
                        "description": "Section name (e.g., encoder, decoder)"
                    },
                    "weights": {
                        "type": "object",
                        "description": "Weight references"
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Input tensor specifications"
                    },
                    "outputs": {
                        "type": "object",
                        "description": "Output tensor specifications"
                    },
                    "dataflow": {
                        "type": "object",
                        "properties": {
                            "inputs": {"type": "object"},
                            "outputs": {"type": "object"}
                        }
                    },
                    "config": {
                        "type": "object",
                        "description": "Kernel-specific configuration"
                    }
                }
            }
        }
    }
}


# =============================================================================
# Layout Schema - Memory layout for activations and weights
# =============================================================================
LAYOUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Layout - Memory Layout Specification",
    "type": "object",
    "required": ["version", "sections"],
    "properties": {
        "version": {
            "type": "string",
            "pattern": r"^v?\d+\.\d+$"
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["header", "layers", "footer"],
                "properties": {
                    "header": {
                        "type": "object",
                        "description": "Section header (config, positions)"
                    },
                    "layers": {
                        "type": "array",
                        "description": "Layer configurations"
                    },
                    "footer": {
                        "type": "object",
                        "description": "Section footer (output, loss)"
                    }
                }
            }
        }
    }
}


# =============================================================================
# Kernel Map Schema - Individual kernel map specifications
# =============================================================================
KERNEL_MAP_ITEM_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Kernel Map Item",
    "type": "object",
    "required": ["id", "op", "params"],
    "properties": {
        "id": {"type": "string"},
        "op": {"type": "string"},
        "variant": {"type": "string"},
        "quant": {
            "type": "object",
            "properties": {
                "weight": {"type": "string"},
                "activation": {"type": "string"},
                "output": {"type": "string"}
            }
        },
        "inputs": {"type": "array"},
        "weights": {"type": "array"},
        "activations": {"type": "array"},
        "outputs": {"type": "array"},
        "scratch": {"type": "array"},
        "dims": {"type": "array"},
        "params": {"type": "array"},
        "parallelization": {"type": "object"},
        "impl": {
            "type": "object",
            "properties": {
                "function": {"type": "string"},
                "c_declaration": {"type": "string"}
            }
        }
    }
}


# =============================================================================
# Kernel Registry Schema
# =============================================================================
KERNEL_REGISTRY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Kernel Registry",
    "type": "object",
    "required": ["_meta", "kernels"],
    "properties": {
        "_meta": {
            "type": "object",
            "required": ["description", "version"],
            "properties": {
                "description": {"type": "string"},
                "version": {"type": "string"},
                "generated_by": {"type": "string"},
                "counts": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "integer"},
                        "by_op": {"type": "object"}
                    }
                }
            }
        },
        "kernels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "op"],
                "properties": {
                    "id": {"type": "string"},
                    "op": {"type": "string"},
                    "variant": {"type": "string"},
                    "mode": {"type": "string", "enum": ["decode", "prefill", "both"]},
                    "quant": {
                        "type": "object",
                        "properties": {
                            "activation": {"type": "string"},
                            "output": {"type": "string"},
                            "weight": {"type": "string"}
                        }
                    },
                    "inputs": {"type": "array"},
                    "outputs": {"type": "array"},
                    "dims": {"type": "array"},
                    "impl": {
                        "type": "object",
                        "properties": {
                            "function": {"type": "string"},
                            "c_declaration": {"type": "string"}
                        }
                    },
                    "depends": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }
}


# Schema name to schema mapping
SCHEMA_MAP = {
    "ir1": IR1_SCHEMA,
    "layout": LAYOUT_SCHEMA,
    "kernel_map": KERNEL_MAP_ITEM_SCHEMA,
    "kernel_map_item": KERNEL_MAP_ITEM_SCHEMA,
    "registry": KERNEL_REGISTRY_SCHEMA
}


def validate_ir_schema(ir_data: dict, schema: dict = IR1_SCHEMA,
                       schema_name: str = "ir1") -> Tuple[bool, list]:
    """Validate IR against schema."""
    errors = []

    # Try to use jsonschema if available
    try:
        import jsonschema
        try:
            jsonschema.validate(ir_data, schema)
            return True, []
        except jsonschema.ValidationError as e:
            return False, [f"Validation error: {e.message} at {' -> '.join(str(p) for p in e.absolute_path)}"]
    except ImportError:
        # Fallback without jsonschema - basic structure check
        pass

    # Basic validation without jsonschema
    if schema_name == "ir1":
        if "version" not in ir_data:
            errors.append("Missing required field: 'version'")
        if "operations" not in ir_data:
            errors.append("Missing required field: 'operations'")
        elif not isinstance(ir_data["operations"], list):
            errors.append("'operations' must be an array")
        elif len(ir_data["operations"]) == 0:
            errors.append("'operations' array cannot be empty")

        # Check operation structure
        for i, op in enumerate(ir_data.get("operations", [])):
            if "idx" not in op:
                errors.append(f"Operation {i}: missing 'idx' field")
            if "op" not in op:
                errors.append(f"Operation {i}: missing 'op' field")
            if "kernel" not in op:
                errors.append(f"Operation {i}: missing 'kernel' field")

    elif schema_name == "layout":
        if "version" not in ir_data:
            errors.append("Missing required field: 'version'")
        if "sections" not in ir_data:
            errors.append("Missing required field: 'sections'")

    elif schema_name in ("kernel_map", "kernel_map_full"):
        # kernel_map files can have version in _meta or at top level
        if "version" not in ir_data and "_meta" not in ir_data:
            errors.append("Missing required field: 'version' or '_meta'")
        if "id" not in ir_data:
            errors.append("Missing required field: 'id'")
        if "op" not in ir_data:
            errors.append("Missing required field: 'op'")

    elif schema_name == "registry":
        if "_meta" not in ir_data:
            errors.append("Missing required field: '_meta'")
        if "kernels" not in ir_data:
            errors.append("Missing required field: 'kernels'")

    return len(errors) == 0, errors


def validate_ir_file(path: str, schema_name: str = "ir1") -> Tuple[bool, list]:
    """Validate an IR file against its schema."""
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]

    if schema_name not in SCHEMA_MAP:
        return False, [f"Unknown schema: {schema_name}"]

    return validate_ir_schema(data, SCHEMA_MAP[schema_name], schema_name)


def auto_detect_schema(data: dict) -> str:
    """Auto-detect schema type from JSON data."""
    if "operations" in data:
        return "ir1"
    elif "sections" in data:
        return "layout"
    elif "_meta" in data and "kernels" in data:
        return "registry"
    elif "bindings" in data:
        return "kernel_map"
    elif "id" in data and "op" in data:
        return "kernel_map"
    return "kernel_map"  # Default


def validate_multiple_files(files: list, schema_name: str = "auto") -> dict:
    """Validate multiple IR files and return summary."""
    results = {
        "valid": [],
        "invalid": [],
        "errors": {}
    }

    for path in files:
        # Auto-detect schema if needed
        if schema_name == "auto":
            with open(path) as f:
                try:
                    data = json.load(f)
                    detected = auto_detect_schema(data)
                    schema_name = detected
                except json.JSONDecodeError:
                    results["invalid"].append(path)
                    results["errors"][path] = ["Invalid JSON"]
                    continue

        valid, errors = validate_ir_file(path, schema_name)
        if valid:
            results["valid"].append(path)
        else:
            results["invalid"].append(path)
            results["errors"][path] = errors

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate IR JSON schema for C-Kernel-Engine v6.6"
    )
    parser.add_argument("file", nargs="?", help="IR JSON file to validate")
    parser.add_argument("--schema", choices=["ir1", "layout", "kernel_map", "registry", "auto"],
                       default="auto", help="Schema type to validate against (auto-detect if not specified)")
    parser.add_argument("--dir", "-d", help="Directory containing IR files")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Recursively find files in directory")
    parser.add_argument("--summary", action="store_true",
                       help="Show summary of all validations")
    args = parser.parse_args()

    if args.file:
        # Auto-detect schema if needed
        schema = args.schema
        if schema == "auto":
            with open(args.file) as f:
                try:
                    data = json.load(f)
                    schema = auto_detect_schema(data)
                except json.JSONDecodeError as e:
                    print(f"[FAIL] {args.file}")
                    print(f"  - Invalid JSON: {e}")
                    return 1

        valid, errors = validate_ir_file(args.file, schema)

        if valid:
            print(f"[PASS] {args.file} (schema: {schema})")
            return 0
        else:
            print(f"[FAIL] {args.file}")
            for e in errors:
                print(f"  - {e}")
            return 1

    elif args.dir:
        import os
        from pathlib import Path

        pattern = "**/*.json" if args.recursive else "*.json"
        files = list(Path(args.dir).glob(pattern))

        if not files:
            print(f"No JSON files found in {args.dir}")
            return 1

        # Auto-detect schema for directory validation
        schema = args.schema if args.schema != "auto" else "auto"

        results = validate_multiple_files([str(f) for f in files], schema)

        print("\n" + "="*60)
        print("IR SCHEMA VALIDATION SUMMARY")
        print("="*60)
        print(f"\nTotal files: {len(files)}")
        print(f"Valid: {len(results['valid'])}")
        print(f"Invalid: {len(results['invalid'])}")

        if results['invalid']:
            print(f"\nInvalid files:")
            for path in results['invalid']:
                print(f"  - {path}")
                for e in results['errors'][path]:
                    print(f"      {e}")

        return 1 if results['invalid'] else 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
