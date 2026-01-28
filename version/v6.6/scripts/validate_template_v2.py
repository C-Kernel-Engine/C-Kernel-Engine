#!/usr/bin/env python3
"""
=============================================================================
EXPERIMENTAL/FUTURE - NOT USED BY CURRENT v6.6 PIPELINE
=============================================================================
This file is a standalone validation utility for template schema v2.
It is NOT called by ck_run_v6_6.py or the current build pipeline.

Part of the experimental op_builders/template system for future IR2 work.
Current pipeline uses: build_ir_v6_6.py with kernel maps directly.
=============================================================================

validate_template_v2.py - Validator for template schema v2

Validates template JSON files against the v2 schema defined in templates/README.md.

USAGE:
    python validate_template_v2.py <template.json>
    python validate_template_v2.py --all  # Validate all templates in templates/

CHECKS:
    - Required fields: version, name, family, flags, sequence, block_types
    - Top-level sequence references valid block_types
    - Each block has sequence field
    - Block sequence references valid phases (header/body/footer)
    - Body has type and ops fields
    - All ops are non-empty strings
    - No duplicate block names
    - No duplicate op names within a phase
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ValidationError(Exception):
    """Raised when template validation fails."""
    pass


class TemplateValidator:
    """Validator for template schema v2."""

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def error(self, msg: str) -> None:
        """Record an error."""
        self.errors.append(msg)

    def warning(self, msg: str) -> None:
        """Record a warning."""
        if self.strict:
            self.errors.append(f"Warning (strict mode): {msg}")
        else:
            self.warnings.append(msg)

    def validate_template(self, template: Dict[str, Any], template_name: str = "template") -> bool:
        """
        Validate a template against v2 schema.

        Args:
            template: Template dictionary
            template_name: Name for error messages

        Returns:
            True if valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()

        # Check required top-level fields
        required_fields = ["version", "name", "family", "flags", "sequence", "block_types"]
        for field in required_fields:
            if field not in template:
                self.error(f"Missing required field: '{field}'")

        if self.errors:
            return False

        # Validate version
        version = template.get("version")
        if not isinstance(version, int):
            self.error(f"'version' must be an integer, got {type(version).__name__}")
        elif version < 2:
            self.warning(f"Template uses legacy version {version}. Consider upgrading to v2.")
        elif version > 2:
            self.warning(f"Template uses future version {version}. This validator supports v2.")

        # Validate name and family
        name = template.get("name")
        if not isinstance(name, str) or not name:
            self.error(f"'name' must be a non-empty string")

        family = template.get("family")
        if not isinstance(family, str) or not family:
            self.error(f"'family' must be a non-empty string")

        # Validate flags
        flags = template.get("flags")
        if not isinstance(flags, dict):
            self.error(f"'flags' must be a dictionary")

        # Validate top-level sequence
        sequence = template.get("sequence", [])
        if not isinstance(sequence, list):
            self.error(f"'sequence' must be an array")
        elif not sequence:
            self.error(f"'sequence' cannot be empty")
        else:
            for i, block_name in enumerate(sequence):
                if not isinstance(block_name, str):
                    self.error(f"sequence[{i}] must be a string, got {type(block_name).__name__}")

        # Validate block_types
        block_types = template.get("block_types", {})
        if not isinstance(block_types, dict):
            self.error(f"'block_types' must be a dictionary")
            return len(self.errors) == 0

        if not block_types:
            self.error(f"'block_types' cannot be empty")
            return len(self.errors) == 0

        # Check that sequence references valid blocks
        for block_name in sequence:
            if isinstance(block_name, str) and block_name not in block_types:
                self.error(f"Top-level sequence references undefined block: '{block_name}'")

        # Validate each block
        for block_name, block_def in block_types.items():
            self._validate_block(block_name, block_def)

        # Check for unreferenced blocks
        referenced_blocks = set(sequence) if isinstance(sequence, list) else set()
        defined_blocks = set(block_types.keys())
        unreferenced = defined_blocks - referenced_blocks
        if unreferenced:
            self.warning(f"Block(s) defined but not referenced in sequence: {', '.join(unreferenced)}")

        return len(self.errors) == 0

    def _validate_block(self, block_name: str, block_def: Any) -> None:
        """Validate a single block definition."""
        if not isinstance(block_def, dict):
            self.error(f"Block '{block_name}' must be a dictionary")
            return

        # Check required block fields
        if "sequence" not in block_def:
            self.error(f"Block '{block_name}' missing required field 'sequence'")

        # Validate block sequence
        block_sequence = block_def.get("sequence", [])
        if not isinstance(block_sequence, list):
            self.error(f"Block '{block_name}' sequence must be an array")
        elif not block_sequence:
            self.error(f"Block '{block_name}' sequence cannot be empty")
        else:
            # Check for valid phase names
            valid_phases = {"header", "body", "footer"}
            seen_phases: Set[str] = set()

            for i, phase in enumerate(block_sequence):
                if not isinstance(phase, str):
                    self.error(f"Block '{block_name}' sequence[{i}] must be a string")
                    continue

                if phase not in valid_phases:
                    self.error(f"Block '{block_name}' sequence[{i}] has invalid phase '{phase}'. "
                             f"Valid: {', '.join(sorted(valid_phases))}")

                if phase in seen_phases:
                    self.warning(f"Block '{block_name}' has duplicate phase '{phase}' in sequence")
                seen_phases.add(phase)

            # Check that referenced phases exist
            for phase in block_sequence:
                if isinstance(phase, str) and phase in valid_phases:
                    if phase not in block_def:
                        self.error(f"Block '{block_name}' sequence references phase '{phase}' but it's not defined")

        # Validate phases
        for phase_name in ["header", "body", "footer"]:
            if phase_name in block_def:
                self._validate_phase(block_name, phase_name, block_def[phase_name])

    def _validate_phase(self, block_name: str, phase_name: str, phase_def: Any) -> None:
        """Validate a phase (header/body/footer)."""
        if phase_name == "body":
            # Body must be a dict with 'type' and 'ops'
            if not isinstance(phase_def, dict):
                self.error(f"Block '{block_name}' {phase_name} must be a dictionary")
                return

            # Check body.type
            if "type" not in phase_def:
                self.error(f"Block '{block_name}' body missing required field 'type'")
            else:
                body_type = phase_def["type"]
                if not isinstance(body_type, str) or not body_type:
                    self.error(f"Block '{block_name}' body.type must be a non-empty string")
                else:
                    valid_types = {"dense", "moe", "sparse"}
                    if body_type not in valid_types:
                        self.warning(f"Block '{block_name}' body.type is '{body_type}'. "
                                   f"Common types: {', '.join(sorted(valid_types))}")

            # Check body.ops
            if "ops" not in phase_def:
                self.error(f"Block '{block_name}' body missing required field 'ops'")
            else:
                ops = phase_def["ops"]
                if not isinstance(ops, list):
                    self.error(f"Block '{block_name}' body.ops must be an array")
                elif not ops:
                    self.error(f"Block '{block_name}' body.ops cannot be empty")
                else:
                    self._validate_ops(f"{block_name}.body", ops)
        else:
            # Header/footer must be arrays of op names
            if not isinstance(phase_def, list):
                self.error(f"Block '{block_name}' {phase_name} must be an array")
            else:
                self._validate_ops(f"{block_name}.{phase_name}", phase_def)

    def _validate_ops(self, context: str, ops: List[Any]) -> None:
        """Validate a list of operations."""
        for i, op in enumerate(ops):
            if not isinstance(op, str):
                self.error(f"{context} ops[{i}] must be a string, got {type(op).__name__}")
                continue

            if not op:
                self.error(f"{context} ops[{i}] is empty")
                continue

            # Check for consecutive duplicates (likely a mistake)
            if i > 0 and op == ops[i - 1]:
                self.warning(f"{context} has consecutive duplicate op '{op}' at index {i}")

    def print_results(self, template_name: str) -> bool:
        """
        Print validation results.

        Returns:
            True if validation passed
        """
        if self.errors:
            print(f"\n❌ {template_name}: FAILED")
            for error in self.errors:
                print(f"  ERROR: {error}")
        else:
            print(f"\n✅ {template_name}: PASSED")

        if self.warnings:
            for warning in self.warnings:
                print(f"  WARNING: {warning}")

        return len(self.errors) == 0


def validate_file(template_path: Path, strict: bool = False) -> bool:
    """
    Validate a single template file.

    Args:
        template_path: Path to template JSON file
        strict: If True, treat warnings as errors

    Returns:
        True if valid
    """
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\n❌ {template_path.name}: INVALID JSON")
        print(f"  ERROR: {e}")
        return False
    except IOError as e:
        print(f"\n❌ {template_path.name}: CANNOT READ")
        print(f"  ERROR: {e}")
        return False

    validator = TemplateValidator(strict=strict)
    is_valid = validator.validate_template(template, template_path.name)
    validator.print_results(template_path.name)

    return is_valid


def find_all_templates(templates_dir: Path) -> List[Path]:
    """Find all .json files in templates directory."""
    if not templates_dir.exists():
        return []

    return sorted([
        p for p in templates_dir.glob("*.json")
        if p.is_file() and not p.name.startswith(".")
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Validate template JSON files against v2 schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_template_v2.py qwen2.json
  python validate_template_v2.py --all
  python validate_template_v2.py --all --strict
        """
    )

    parser.add_argument(
        "template",
        nargs="?",
        help="Path to template JSON file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all templates in templates/ directory"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=Path(__file__).parent.parent / "templates",
        help="Templates directory (default: ../templates)"
    )

    args = parser.parse_args()

    if args.all:
        # Validate all templates
        templates = find_all_templates(args.templates_dir)
        if not templates:
            print(f"No templates found in {args.templates_dir}")
            return 1

        print(f"Validating {len(templates)} template(s) in {args.templates_dir}")

        results = [validate_file(t, strict=args.strict) for t in templates]

        passed = sum(results)
        failed = len(results) - passed

        print(f"\n{'='*60}")
        print(f"Summary: {passed} passed, {failed} failed")

        return 0 if failed == 0 else 1

    elif args.template:
        # Validate single template
        template_path = Path(args.template)

        if not template_path.exists():
            # Try relative to templates dir
            alt_path = args.templates_dir / args.template
            if alt_path.exists():
                template_path = alt_path
            else:
                print(f"Error: Template not found: {args.template}")
                return 1

        is_valid = validate_file(template_path, strict=args.strict)
        return 0 if is_valid else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
