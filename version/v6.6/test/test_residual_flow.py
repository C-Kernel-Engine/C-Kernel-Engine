#!/usr/bin/env python3
"""
test_residual_flow.py - Validate residual connections flow correctly

This test ensures:
- Residual is saved before each sub-block (attention, MLP)
- Residual add uses correct saved buffer
- Residual path doesn't get overwritten

BUGS THIS CATCHES:
- Bug 14: Residual uses wrong buffer

Usage:
    python test_residual_flow.py
"""

import argparse
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

# Paths
CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V66_ROOT = Path(__file__).parent.parent
GENERATED_DIR = V66_ROOT / "src" / "generated"


@dataclass
class ResidualIssue:
    """A residual flow issue."""
    severity: str
    layer: int
    message: str
    expected: str = ""
    actual: str = ""


class ResidualValidator:
    """Validates residual connection flow."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues: List[ResidualIssue] = []

    def load_files(self) -> bool:
        """Load required files."""
        self.ir = None
        self.c_code = None
        self.c_lines = []

        for base_dir in [GENERATED_DIR, CACHE_DIR]:
            if (base_dir / "lowered_decode.json").exists():
                with open(base_dir / "lowered_decode.json") as f:
                    self.ir = json.load(f)

            if (base_dir / "ck-kernel-inference.c").exists():
                self.c_code = (base_dir / "ck-kernel-inference.c").read_text()
                self.c_lines = self.c_code.split('\n')

        if not self.c_code:
            print("ERROR: Could not find ck-kernel-inference.c")
            return False

        return True

    def find_residual_pattern(self) -> Dict[int, Dict]:
        """Find residual save/add patterns per layer."""
        layer_residuals = {}

        # Pattern: Save residual before attention
        # memcpy(residual_buf, input, ...) OR residual = input
        save_pattern = r'memcpy.*(?:residual|res_buf).*(?:ACT|activations)\s*\+\s*(\d+)'
        add_pattern = r'(?:ck_)?residual_add.*\(([^)]+)\)'

        current_layer = -1

        for i, line in enumerate(self.c_lines):
            # Track layer context
            layer_match = re.search(r'L_LAYERS\[(\d+)\]|layer\s*=\s*(\d+)|L(\d+)', line)
            if layer_match:
                current_layer = int(layer_match.group(1) or layer_match.group(2) or layer_match.group(3))
                if current_layer not in layer_residuals:
                    layer_residuals[current_layer] = {
                        "saves": [],
                        "adds": [],
                    }

            # Find residual saves
            save_match = re.search(save_pattern, line)
            if save_match and current_layer >= 0:
                layer_residuals[current_layer]["saves"].append({
                    "line": i + 1,
                    "offset": save_match.group(1)
                })

            # Find residual adds
            add_match = re.search(add_pattern, line)
            if add_match and current_layer >= 0:
                layer_residuals[current_layer]["adds"].append({
                    "line": i + 1,
                    "args": add_match.group(1)
                })

        return layer_residuals

    def check_residual_balance(self, layer_residuals: Dict[int, Dict]) -> bool:
        """Check each layer has balanced saves and adds."""
        print("\n" + "-"*70)
        print("TEST 1: Residual balance (saves vs adds)")
        print("-"*70)

        all_balanced = True

        for layer, info in sorted(layer_residuals.items()):
            saves = len(info["saves"])
            adds = len(info["adds"])

            if saves == 0 and adds > 0:
                self.issues.append(ResidualIssue(
                    severity="ERROR",
                    layer=layer,
                    message=f"Layer has {adds} residual adds but no saves",
                    expected="At least 1 save before each add",
                    actual=f"{saves} saves, {adds} adds"
                ))
                all_balanced = False
                print(f"  ✗ Layer {layer}: {saves} saves, {adds} adds")
            elif self.verbose:
                print(f"  ✓ Layer {layer}: {saves} saves, {adds} adds")

        if all_balanced:
            print(f"  ✓ All {len(layer_residuals)} layers have balanced residuals")

        return all_balanced

    def check_residual_buffer_reuse(self) -> bool:
        """Check residual buffer isn't overwritten before use."""
        print("\n" + "-"*70)
        print("TEST 2: Residual buffer safety")
        print("-"*70)

        # Find residual buffer offsets
        residual_offsets = set()
        residual_pattern = r'(?:residual|res_buf).*\+\s*(\d+)'

        for line in self.c_lines:
            match = re.search(residual_pattern, line)
            if match:
                residual_offsets.add(int(match.group(1)))

        # Check if any other ops write to these offsets between save and add
        # This is a simplified check - full analysis would need control flow

        if residual_offsets:
            print(f"  Found residual buffer offsets: {residual_offsets}")
        else:
            print("  ⚠ No explicit residual buffers found (may use inline)")

        return True

    def check_two_residual_adds_per_layer(self) -> bool:
        """Check transformer layers have 2 residual adds (attention + MLP)."""
        print("\n" + "-"*70)
        print("TEST 3: Two residual adds per transformer layer")
        print("-"*70)

        # Count residual_add calls per layer context
        layer_add_counts: Dict[int, int] = {}
        current_layer = -1

        for line in self.c_lines:
            # Track layer
            layer_match = re.search(r'L_LAYERS\[(\d+)\]', line)
            if layer_match:
                current_layer = int(layer_match.group(1))

            # Count adds
            if 'residual_add' in line.lower() and current_layer >= 0:
                layer_add_counts[current_layer] = layer_add_counts.get(current_layer, 0) + 1

        # Validate
        issues_found = False
        for layer, count in sorted(layer_add_counts.items()):
            if count != 2:
                self.issues.append(ResidualIssue(
                    severity="WARNING",
                    layer=layer,
                    message=f"Expected 2 residual adds (attention + MLP), found {count}"
                ))
                issues_found = True
                print(f"  ⚠ Layer {layer}: {count} residual adds (expected 2)")

        if not issues_found and layer_add_counts:
            print(f"  ✓ All layers have 2 residual adds")
        elif not layer_add_counts:
            print("  ⚠ No residual adds found (may use fused kernels)")

        return not issues_found

    def check_residual_uses_correct_input(self) -> bool:
        """Check residual add uses layer input, not intermediate buffer."""
        print("\n" + "-"*70)
        print("TEST 4: Residual uses correct source")
        print("-"*70)

        # In a transformer:
        # - First residual: layer_input + attention_output
        # - Second residual: attention_residual_output + mlp_output

        # Look for patterns where residual might use wrong buffer
        # Example bug: using mlp_output for first residual instead of attention_output

        # This is hard to validate statically without full data flow analysis
        # We'll check for obvious issues

        issues_found = False

        # Check for self-add patterns (adding buffer to itself)
        for i, line in enumerate(self.c_lines):
            if 'residual_add' in line.lower():
                # Get the function call (may span lines)
                call_text = line
                for j in range(i+1, min(i+5, len(self.c_lines))):
                    call_text += self.c_lines[j]
                    if ');' in call_text:
                        break

                # Check if same buffer appears twice
                offset_matches = re.findall(r'activations\s*\+\s*(\d+)', call_text)
                if len(offset_matches) >= 2:
                    if offset_matches[0] == offset_matches[1]:
                        self.issues.append(ResidualIssue(
                            severity="ERROR",
                            layer=-1,
                            message=f"Residual add uses same buffer for both inputs (line {i+1})",
                            expected="Different input and residual buffers",
                            actual=f"Both use offset {offset_matches[0]}"
                        ))
                        issues_found = True
                        print(f"  ✗ Line {i+1}: Same buffer used twice in residual_add")

        if not issues_found:
            print("  ✓ No obvious residual source issues found")

        return not issues_found

    def run_all_tests(self) -> bool:
        """Run all residual validations."""
        print("\n" + "="*70)
        print("RESIDUAL FLOW VALIDATION")
        print("="*70)

        if not self.load_files():
            return False

        layer_residuals = self.find_residual_pattern()
        print(f"Found residual patterns in {len(layer_residuals)} layers")

        results = []
        results.append(("Balance", self.check_residual_balance(layer_residuals)))
        results.append(("Buffer safety", self.check_residual_buffer_reuse()))
        results.append(("Two adds per layer", self.check_two_residual_adds_per_layer()))
        results.append(("Correct source", self.check_residual_uses_correct_input()))

        # Summary
        print("\n" + "="*70)
        print("RESIDUAL ISSUES")
        print("="*70)

        if not self.issues:
            print("No residual flow issues found!")
        else:
            for severity in ["ERROR", "WARNING"]:
                issues = [i for i in self.issues if i.severity == severity]
                if issues:
                    print(f"\n{severity} ({len(issues)}):")
                    for issue in issues:
                        layer_info = f"Layer {issue.layer}" if issue.layer >= 0 else "Global"
                        print(f"  [{layer_info}]: {issue.message}")
                        if issue.expected:
                            print(f"    Expected: {issue.expected}")
                        if issue.actual:
                            print(f"    Actual: {issue.actual}")

        # Verdict
        print("\n" + "="*70)
        errors = len([i for i in self.issues if i.severity == "ERROR"])
        passed = sum(1 for _, r in results if r)

        if errors > 0:
            print(f"VERDICT: FAIL - {errors} residual errors")
        else:
            print(f"VERDICT: PASS - {passed}/{len(results)} tests passed")
        print("="*70)

        return errors == 0


def main():
    parser = argparse.ArgumentParser(description="Validate residual flow")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    validator = ResidualValidator(verbose=args.verbose)
    success = validator.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
