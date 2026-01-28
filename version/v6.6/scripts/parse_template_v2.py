#!/usr/bin/env python3
"""
=============================================================================
EXPERIMENTAL/FUTURE - NOT USED BY CURRENT v6.6 PIPELINE
=============================================================================
This file is part of the experimental op_builders system for future IR2 work.
It is NOT called by ck_run_v6_6.py or the current build pipeline.

Used by: op_builders_auto.py, op_builders_v6_6.py, op_builders_hybrid_v6_6.py
         gen_op_builders_from_registry.py

Current pipeline uses: build_ir_v6_6.py with kernel maps directly.
=============================================================================

parse_template_v2.py - Parser for template schema v2

Parses template JSON files and generates operation sequences for IR1 generation.

USAGE:
    from parse_template_v2 import TemplateParser

    parser = TemplateParser(template_dict, config_dict)
    sequence = parser.build_execution_sequence()

FEATURES:
    - Parses v2 template schema (sequence, header/body/footer)
    - Generates flat operation sequence following template order
    - Expands body operations for all layers
    - Annotates ops with metadata (block, phase, layer index)
    - Handles multi-modal models (vision, audio, text)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TemplateParseError(Exception):
    """Raised when template parsing fails."""
    pass


class OpNode:
    """
    Represents a single operation in the execution graph.

    Attributes:
        op_id: Operation identifier (e.g., "rmsnorm", "qkv_proj")
        block_name: Block this op belongs to (e.g., "decoder", "vision_encoder")
        phase: Phase within block ("header", "body", "footer")
        layer_index: Layer index if in body phase, None otherwise
        metadata: Additional metadata from template
    """

    def __init__(
        self,
        op_id: str,
        block_name: str,
        phase: str,
        layer_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.op_id = op_id
        self.block_name = block_name
        self.phase = phase
        self.layer_index = layer_index
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        if self.layer_index is not None:
            return f"OpNode({self.op_id}, {self.block_name}, {self.phase}, layer={self.layer_index})"
        return f"OpNode({self.op_id}, {self.block_name}, {self.phase})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "op_id": self.op_id,
            "block": self.block_name,
            "phase": self.phase,
        }
        if self.layer_index is not None:
            result["layer"] = self.layer_index
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class BlockExecutionPlan:
    """
    Execution plan for a single block (e.g., decoder, vision_encoder).

    Attributes:
        block_name: Name of the block
        block_type: Type of body layers (dense, moe, sparse)
        num_layers: Number of layers in body (if applicable)
        ops: Flat list of operations in execution order
    """

    def __init__(
        self,
        block_name: str,
        block_type: str,
        num_layers: int,
        ops: List[OpNode]
    ):
        self.block_name = block_name
        self.block_type = block_type
        self.num_layers = num_layers
        self.ops = ops

    def __repr__(self) -> str:
        return f"BlockExecutionPlan({self.block_name}, type={self.block_type}, layers={self.num_layers}, ops={len(self.ops)})"


class ExecutionSequence:
    """
    Complete execution sequence for a model.

    Attributes:
        blocks: List of block execution plans in order
        total_ops: Total number of operations
    """

    def __init__(self, blocks: List[BlockExecutionPlan]):
        self.blocks = blocks
        self.total_ops = sum(len(block.ops) for block in blocks)

    def get_flat_ops(self) -> List[OpNode]:
        """Get flat list of all operations in execution order."""
        result = []
        for block in self.blocks:
            result.extend(block.ops)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "blocks": [
                {
                    "block_name": block.block_name,
                    "block_type": block.block_type,
                    "num_layers": block.num_layers,
                    "ops": [op.to_dict() for op in block.ops]
                }
                for block in self.blocks
            ],
            "total_ops": self.total_ops
        }

    def __repr__(self) -> str:
        return f"ExecutionSequence(blocks={len(self.blocks)}, total_ops={self.total_ops})"


class TemplateParser:
    """
    Parser for template v2 schema.

    Parses template JSON and generates execution sequences for IR1.
    """

    def __init__(self, template: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Initialize parser.

        Args:
            template: Template dictionary (v2 schema)
            config: Model configuration with num_layers, etc. (optional)

        Raises:
            TemplateParseError: If template is invalid
        """
        self.template = template
        self.config = config or {}

        # Validate required fields
        self._validate_required_fields()

        self.version = template["version"]
        self.name = template["name"]
        self.family = template["family"]
        self.flags = template["flags"]
        self.sequence = template["sequence"]
        self.block_types = template["block_types"]

    def _validate_required_fields(self) -> None:
        """Validate that required fields exist."""
        required = ["version", "name", "family", "flags", "sequence", "block_types"]
        missing = [f for f in required if f not in self.template]
        if missing:
            raise TemplateParseError(f"Missing required fields: {', '.join(missing)}")

        if not isinstance(self.template["sequence"], list) or not self.template["sequence"]:
            raise TemplateParseError("'sequence' must be a non-empty array")

        if not isinstance(self.template["block_types"], dict) or not self.template["block_types"]:
            raise TemplateParseError("'block_types' must be a non-empty dictionary")

    def build_execution_sequence(self, num_layers: Optional[int] = None) -> ExecutionSequence:
        """
        Build complete execution sequence from template.

        Args:
            num_layers: Number of layers for body expansion. If None, uses config.

        Returns:
            ExecutionSequence with all blocks and operations

        Raises:
            TemplateParseError: If parsing fails
        """
        # Resolve num_layers
        if num_layers is None:
            num_layers = self.config.get("num_layers", 32)  # Default to 32

        blocks = []

        # Process each block in top-level sequence
        for block_name in self.sequence:
            if block_name not in self.block_types:
                raise TemplateParseError(f"Block '{block_name}' referenced in sequence but not defined in block_types")

            block_def = self.block_types[block_name]
            block_plan = self._parse_block(block_name, block_def, num_layers)
            blocks.append(block_plan)

        return ExecutionSequence(blocks)

    def _parse_block(self, block_name: str, block_def: Dict[str, Any], num_layers: int) -> BlockExecutionPlan:
        """
        Parse a single block definition.

        Args:
            block_name: Name of the block
            block_def: Block definition dictionary
            num_layers: Number of layers for body expansion

        Returns:
            BlockExecutionPlan for this block
        """
        if "sequence" not in block_def:
            raise TemplateParseError(f"Block '{block_name}' missing 'sequence' field")

        block_sequence = block_def["sequence"]
        if not isinstance(block_sequence, list) or not block_sequence:
            raise TemplateParseError(f"Block '{block_name}' sequence must be non-empty array")

        ops = []
        block_type = "dense"  # Default

        # Process phases in order specified by block sequence
        for phase_name in block_sequence:
            if phase_name not in block_def:
                raise TemplateParseError(f"Block '{block_name}' sequence references phase '{phase_name}' but it's not defined")

            phase_def = block_def[phase_name]

            if phase_name == "body":
                # Body phase - expand for all layers
                block_type = phase_def.get("type", "dense")
                body_ops = self._parse_body(block_name, phase_def, num_layers)
                ops.extend(body_ops)
            else:
                # Header/footer phase - run once
                phase_ops = self._parse_phase(block_name, phase_name, phase_def)
                ops.extend(phase_ops)

        return BlockExecutionPlan(
            block_name=block_name,
            block_type=block_type,
            num_layers=num_layers,
            ops=ops
        )

    def _parse_body(self, block_name: str, body_def: Dict[str, Any], num_layers: int) -> List[OpNode]:
        """
        Parse body phase and expand for all layers.

        Args:
            block_name: Name of the block
            body_def: Body definition with 'type' and 'ops'
            num_layers: Number of layers to generate

        Returns:
            List of OpNodes for all layers
        """
        if "ops" not in body_def:
            raise TemplateParseError(f"Block '{block_name}' body missing 'ops' field")

        ops_template = body_def["ops"]
        if not isinstance(ops_template, list) or not ops_template:
            raise TemplateParseError(f"Block '{block_name}' body.ops must be non-empty array")

        body_type = body_def.get("type", "dense")

        result = []
        for layer_idx in range(num_layers):
            for op_id in ops_template:
                if not isinstance(op_id, str) or not op_id:
                    raise TemplateParseError(f"Block '{block_name}' body.ops contains invalid op")

                node = OpNode(
                    op_id=op_id,
                    block_name=block_name,
                    phase="body",
                    layer_index=layer_idx,
                    metadata={"body_type": body_type}
                )
                result.append(node)

        return result

    def _parse_phase(self, block_name: str, phase_name: str, phase_def: List[str]) -> List[OpNode]:
        """
        Parse header/footer phase.

        Args:
            block_name: Name of the block
            phase_name: "header" or "footer"
            phase_def: List of operation IDs

        Returns:
            List of OpNodes
        """
        if not isinstance(phase_def, list):
            raise TemplateParseError(f"Block '{block_name}' {phase_name} must be an array")

        result = []
        for op_id in phase_def:
            if not isinstance(op_id, str) or not op_id:
                raise TemplateParseError(f"Block '{block_name}' {phase_name} contains invalid op")

            node = OpNode(
                op_id=op_id,
                block_name=block_name,
                phase=phase_name,
                layer_index=None
            )
            result.append(node)

        return result

    def get_all_op_ids(self) -> List[str]:
        """
        Get list of unique operation IDs used in template.

        Returns:
            Sorted list of unique op IDs
        """
        op_ids = set()

        for block_def in self.block_types.values():
            for phase_name in ["header", "body", "footer"]:
                if phase_name not in block_def:
                    continue

                phase_def = block_def[phase_name]

                if phase_name == "body" and isinstance(phase_def, dict):
                    ops = phase_def.get("ops", [])
                elif isinstance(phase_def, list):
                    ops = phase_def
                else:
                    continue

                for op_id in ops:
                    if isinstance(op_id, str):
                        op_ids.add(op_id)

        return sorted(op_ids)


def load_template(template_path: Path) -> Dict[str, Any]:
    """Load template JSON from file."""
    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Demo: Parse qwen2.json and print execution sequence."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse template and show execution sequence")
    parser.add_argument("template", help="Path to template JSON file")
    parser.add_argument("--num-layers", type=int, default=32, help="Number of layers (default: 32)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"Error: Template not found: {template_path}")
        return 1

    try:
        template = load_template(template_path)
        config = {"num_layers": args.num_layers}

        parser = TemplateParser(template, config)
        sequence = parser.build_execution_sequence()

        if args.json:
            print(json.dumps(sequence.to_dict(), indent=2))
        else:
            print(f"\n{sequence}")
            print(f"\nBlocks: {len(sequence.blocks)}")
            for block in sequence.blocks:
                print(f"  - {block.block_name}: {len(block.ops)} ops ({block.block_type}, {block.num_layers} layers)")

            print(f"\nUnique operation types:")
            unique_ops = sorted(set(op.op_id for op in sequence.get_flat_ops()))
            for op_id in unique_ops:
                count = sum(1 for op in sequence.get_flat_ops() if op.op_id == op_id)
                print(f"  - {op_id}: {count}x")

    except (TemplateParseError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
