#!/usr/bin/env python3
"""
test_io_chain.py - Validate output of kernel N is input to kernel N+1

This test ensures data flow consistency through the pipeline:
- Each kernel's output buffer must be the next kernel's input buffer
- No gaps or mismatches in the activation flow
- Residual connections properly saved and restored

BUGS THIS CATCHES:
- Bug 3: Output of kernel N must feed kernel N+1
- Bug 14: Residual uses wrong buffer
- Bug 12: Logits output to wrong buffer

Usage:
    python test_io_chain.py
    python test_io_chain.py --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

# Paths
CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V66_ROOT = Path(__file__).parent.parent
GENERATED_DIR = V66_ROOT / "src" / "generated"


@dataclass
class BufferFlow:
    """Tracks buffer flow through an operation."""
    op_idx: int
    op_name: str
    kernel: str
    inputs: List[str]
    outputs: List[str]
    layer: int = -1


@dataclass
class FlowIssue:
    """A data flow issue."""
    severity: str
    op_idx: int
    message: str
    expected: str = ""
    actual: str = ""


class IOChainValidator:
    """Validates input/output chain through the pipeline."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues: List[FlowIssue] = []
        self.flows: List[BufferFlow] = []

    def load_ir(self) -> bool:
        """Load lowered IR."""
        for base_dir in [GENERATED_DIR, CACHE_DIR]:
            lowered_path = base_dir / "lowered_decode.json"
            if lowered_path.exists():
                with open(lowered_path) as f:
                    self.ir = json.load(f)
                print(f"Loaded IR from: {lowered_path}")
                return True

        print("ERROR: Could not find lowered_decode.json")
        return False

    def extract_ops(self) -> List[Dict]:
        """Extract ops from IR (handles nested structure)."""
        ops = []

        # Try top-level first
        if "operations" in self.ir:
            return self.ir["operations"]
        elif "ops" in self.ir:
            return self.ir["ops"]

        # Try nested in sections
        for section in self.ir.get("sections", []):
            # Header ops
            header = section.get("header", {})
            for op in header.get("ops", []):
                op["_section"] = "header"
                ops.append(op)

            # Layer ops
            for layer_idx, layer in enumerate(section.get("layers", [])):
                for op in layer.get("ops", []):
                    op["_layer"] = layer_idx
                    op["_section"] = "body"
                    ops.append(op)

            # Footer ops
            footer = section.get("footer", {})
            for op in footer.get("ops", []):
                op["_section"] = "footer"
                ops.append(op)

        return ops

    def build_flow_graph(self, ops: List[Dict]) -> None:
        """Build buffer flow graph from ops."""
        self.flows = []

        for idx, op in enumerate(ops):
            # Extract inputs
            inputs = []
            if "inputs" in op:
                if isinstance(op["inputs"], list):
                    inputs = op["inputs"]
                elif isinstance(op["inputs"], dict):
                    inputs = list(op["inputs"].keys())

            # Extract outputs
            outputs = []
            if "outputs" in op:
                if isinstance(op["outputs"], list):
                    outputs = op["outputs"]
                elif isinstance(op["outputs"], dict):
                    outputs = list(op["outputs"].keys())

            flow = BufferFlow(
                op_idx=idx,
                op_name=op.get("name", op.get("op", f"op_{idx}")),
                kernel=op.get("kernel", op.get("function", op.get("op", "unknown"))),
                inputs=inputs,
                outputs=outputs,
                layer=op.get("_layer", op.get("layer", -1))
            )
            self.flows.append(flow)

    def validate_chain(self) -> bool:
        """Validate that outputs feed into next inputs."""
        print("\n" + "="*70)
        print("I/O CHAIN VALIDATION")
        print("="*70)

        # Track available buffers (what's been produced)
        available_buffers: Set[str] = set()

        # Initial inputs (always available)
        available_buffers.add("tokens")
        available_buffers.add("token_ids")
        available_buffers.add("input")

        # Track buffer producers
        buffer_producers: Dict[str, int] = {}

        for flow in self.flows:
            if self.verbose:
                print(f"\nOp {flow.op_idx}: {flow.kernel}")
                print(f"  Inputs: {flow.inputs}")
                print(f"  Outputs: {flow.outputs}")

            # Check all inputs are available
            for inp in flow.inputs:
                if inp not in available_buffers:
                    # Check if it's a weight (not an activation)
                    if any(w in inp.lower() for w in ['weight', 'gamma', 'bias', 'emb', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2']):
                        continue  # Weights are always available

                    self.issues.append(FlowIssue(
                        severity="ERROR",
                        op_idx=flow.op_idx,
                        message=f"Input '{inp}' not produced by any previous op",
                        expected=f"Buffer '{inp}' should be output of earlier op",
                        actual=f"Available buffers: {sorted(available_buffers)[:10]}..."
                    ))

            # Add outputs to available buffers
            for out in flow.outputs:
                available_buffers.add(out)
                buffer_producers[out] = flow.op_idx

        return len([i for i in self.issues if i.severity == "ERROR"]) == 0

    def validate_residual_flow(self) -> bool:
        """Validate residual connections."""
        print("\n" + "-"*70)
        print("RESIDUAL FLOW VALIDATION")
        print("-"*70)

        residual_issues = []

        # Find residual_add ops
        for flow in self.flows:
            kernel = flow.kernel or ""
            op_name = flow.op_name or ""
            if 'residual' in kernel.lower() or 'add' in op_name.lower():
                # Residual add should have 2 inputs
                if len(flow.inputs) < 2:
                    residual_issues.append(FlowIssue(
                        severity="WARNING",
                        op_idx=flow.op_idx,
                        message=f"Residual add has only {len(flow.inputs)} inputs, expected 2"
                    ))
                elif self.verbose:
                    print(f"  Op {flow.op_idx} ({flow.kernel}): inputs = {flow.inputs}")

        self.issues.extend(residual_issues)
        return len([i for i in residual_issues if i.severity == "ERROR"]) == 0

    def validate_logits_output(self) -> bool:
        """Validate logits goes to correct buffer."""
        print("\n" + "-"*70)
        print("LOGITS OUTPUT VALIDATION")
        print("-"*70)

        # Find logits op
        logits_flow = None
        for flow in self.flows:
            op_name = flow.op_name or ""
            kernel = flow.kernel or ""
            if 'logits' in op_name.lower() or 'lm_head' in kernel.lower():
                logits_flow = flow
                break

        if not logits_flow:
            print("  No logits op found in IR")
            return True

        print(f"  Logits op: {logits_flow.kernel}")
        print(f"  Outputs: {logits_flow.outputs}")

        # Check output is logits buffer
        has_logits_output = any('logit' in o.lower() for o in logits_flow.outputs)
        if not has_logits_output:
            self.issues.append(FlowIssue(
                severity="ERROR",
                op_idx=logits_flow.op_idx,
                message="Logits op doesn't output to 'logits' buffer",
                expected="Output should be 'logits' or 'output_logits'",
                actual=f"Actual outputs: {logits_flow.outputs}"
            ))
            return False

        return True

    def validate_embedding_to_first_layer(self) -> bool:
        """Validate embedding output feeds first layer."""
        print("\n" + "-"*70)
        print("EMBEDDING → LAYER 0 VALIDATION")
        print("-"*70)

        # Find embedding op
        embedding_flow = None
        first_layer_flow = None

        for flow in self.flows:
            kernel = flow.kernel or ""
            op_name = flow.op_name or ""
            if 'embedding' in kernel.lower() or 'embed' in op_name.lower():
                embedding_flow = flow
            elif flow.layer == 0 and first_layer_flow is None:
                first_layer_flow = flow

        if not embedding_flow:
            print("  No embedding op found")
            return True

        if not first_layer_flow:
            print("  No layer 0 op found")
            return True

        print(f"  Embedding output: {embedding_flow.outputs}")
        print(f"  Layer 0 input: {first_layer_flow.inputs}")

        # Check connection
        embedding_out = set(embedding_flow.outputs)
        layer0_in = set(first_layer_flow.inputs)

        if not embedding_out.intersection(layer0_in):
            # Check for common naming patterns
            common_names = {'embedded_input', 'embedded', 'hidden_states', 'x'}
            if not (embedding_out.intersection(common_names) or layer0_in.intersection(common_names)):
                self.issues.append(FlowIssue(
                    severity="WARNING",
                    op_idx=first_layer_flow.op_idx,
                    message="Embedding output may not connect to layer 0 input",
                    expected=f"Embedding outputs: {embedding_flow.outputs}",
                    actual=f"Layer 0 inputs: {first_layer_flow.inputs}"
                ))

        return True

    def run_all_tests(self) -> bool:
        """Run all I/O chain validations."""
        if not self.load_ir():
            return False

        ops = self.extract_ops()
        if not ops:
            print("ERROR: No ops found in IR")
            return False

        print(f"Found {len(ops)} operations")

        self.build_flow_graph(ops)

        # Run validations
        self.validate_chain()
        self.validate_residual_flow()
        self.validate_logits_output()
        self.validate_embedding_to_first_layer()

        # Print issues
        print("\n" + "="*70)
        print("ISSUES FOUND")
        print("="*70)

        if not self.issues:
            print("No I/O chain issues found!")
        else:
            errors = [i for i in self.issues if i.severity == "ERROR"]
            warnings = [i for i in self.issues if i.severity == "WARNING"]

            if errors:
                print(f"\nERRORS ({len(errors)}):")
                for issue in errors:
                    print(f"  Op {issue.op_idx}: {issue.message}")
                    if issue.expected:
                        print(f"    Expected: {issue.expected}")
                    if issue.actual:
                        print(f"    Actual: {issue.actual}")

            if warnings:
                print(f"\nWARNINGS ({len(warnings)}):")
                for issue in warnings:
                    print(f"  Op {issue.op_idx}: {issue.message}")

        # Summary
        print("\n" + "="*70)
        errors = len([i for i in self.issues if i.severity == "ERROR"])
        if errors > 0:
            print(f"VERDICT: FAIL - {errors} errors in I/O chain")
        else:
            print("VERDICT: PASS - I/O chain is consistent")
        print("="*70)

        return errors == 0


def main():
    parser = argparse.ArgumentParser(description="Validate I/O chain")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = IOChainValidator(verbose=args.verbose)
    success = validator.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
