#!/usr/bin/env python3
"""
test_weight_offset_consistency.py - Validate weight offsets and structural consistency.

PURPOSE:
    The BUMP file offset for each weight must be correct. If offset is wrong:
    - We read wrong data
    - Dtype check fails
    - All downstream operations produce garbage

This test validates that:
1. Offsets follow expected patterns (layer stride)
2. All weights of same type have same dtype (by reading actual BUMP data)
3. Block structure is consistent
4. Quantization parameters are uniform within weight types

EXIT CODES:
    0 = All weights have consistent offset/structure
    1 = Inconsistencies found (likely offset errors)
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class WeightInfo:
    name: str
    manifest_offset: int
    manifest_dtype: str
    manifest_size: int
    layer: Optional[int]
    weight_type: str  # wq, wk, wv, wo, w1, w2, w3, etc.


@dataclass
class OffsetIssue:
    severity: str  # ERROR, WARNING
    weight_name: str
    issue: str
    details: str


class WeightOffsetValidator:
    """Validates weight offsets and structural consistency in BUMP."""

    # Quantization block sizes
    Q8_0_BLOCK_SIZE = 32  # elements per block
    Q8_0_BYTES_PER_BLOCK = 34  # 2 bytes scale + 32 bytes quants
    Q5_0_BLOCK_SIZE = 32
    Q5_0_BLOCK_MIN_MAX = 2  # 2 bytes per block for min/max
    Q5_0_BYTES_PER_BLOCK = 34  # 2+32+2 (min, quants, max)

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.issues: List[OffsetIssue] = []
        self.weights: List[WeightInfo] = []

    def load_manifest(self) -> bool:
        """Load weights manifest."""
        manifest_path = self.model_dir / "weights_manifest.json"
        if not manifest_path.exists():
            print(f"ERROR: {manifest_path} not found")
            return False

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Build weight list
        for entry in self.manifest.get("entries", []):
            name = entry.get("name", "")
            layer = None
            weight_type = None

            # Parse layer.N.type pattern
            if name.startswith("layer."):
                parts = name.split(".")
                if len(parts) >= 3:
                    try:
                        layer = int(parts[1])
                        weight_type = parts[2]
                    except ValueError:
                        pass

            self.weights.append(WeightInfo(
                name=name,
                manifest_offset=entry.get("file_offset", entry.get("offset", 0)),
                manifest_dtype=entry.get("dtype", "unknown"),
                manifest_size=entry.get("size", 0),
                layer=layer,
                weight_type=weight_type,
            ))

        print(f"Loaded {len(self.weights)} weights from manifest")
        return True

    def validate_dtype_at_offset(self, bump_file, offset: int, expected_dtype: str) -> Tuple[str, int]:
        """
        Read dtype from BUMP at given offset by inspecting block structure.
        Returns (detected_dtype, block_size).
        """
        bump_file.seek(offset)

        # Read enough bytes to detect dtype
        header_bytes = bump_file.read(64)

        if len(header_bytes) < 8:
            return "unknown", 0

        expected = (expected_dtype or "").lower()

        # If manifest expects fp32/f32, allow zero-initialized regions (common for optional biases).
        # Prior logic rejected zeros and produced false "unknown" mismatches in quick sanity.
        if expected in ("fp32", "f32") and len(header_bytes) >= 16:
            try:
                sample = np.frombuffer(header_bytes[:32], dtype=np.float32)
                if sample.size and np.all(np.isfinite(sample)):
                    max_abs = float(np.max(np.abs(sample)))
                    if max_abs <= 1.0e6:
                        return "fp32", 1
            except Exception:
                pass

        # Try to detect by structure

        # Method 1: Check for FP32 data
        # Read first 4 bytes as float32
        try:
            f32_val = struct.unpack('<f', header_bytes[:4])[0]
            # If it's a reasonable float value and not a scale pattern, likely fp32
            if 0 < abs(f32_val) < 1000.0:
                # Check next few values
                all_reasonable = True
                for i in range(0, min(32, len(header_bytes) - 4), 4):
                    v = struct.unpack('<f', header_bytes[i:i+4])[0]
                    if not (0 < abs(v) < 1000.0):
                        all_reasonable = False
                        break
                if all_reasonable:
                    return "fp32", 1
        except:
            pass

        # Method 2: Check for Q8_0 (first 2 bytes = fp16 scale, next 32 = int8 quants)
        try:
            scale = struct.unpack('<e', header_bytes[:2])[0]
            if 0 < abs(scale) < 10.0:  # Scale should be small
                quants = np.frombuffer(header_bytes[2:34], dtype=np.int8)
                if len(quants) == 32:
                    # Verify quants are in int8 range
                    if np.all(quants >= -128) and np.all(quants <= 127):
                        return "q8_0", self.Q8_0_BLOCK_SIZE
        except:
            pass

        # Method 3: Check for Q5_0 (2 bytes min, 32 quants, 2 bytes max)
        try:
            min_val = struct.unpack('<e', header_bytes[:2])[0]
            max_val = struct.unpack('<e', header_bytes[34:36])[0] if len(header_bytes) >= 36 else None
            if max_val is not None and 0 < abs(min_val) < 10.0 and 0 < abs(max_val) < 10.0:
                if min_val <= max_val:
                    return "q5_0", self.Q5_0_BLOCK_SIZE
        except:
            pass

        # Method 4: Q4_K has different structure
        try:
            d_min = struct.unpack('<e', header_bytes[:2])[0]
            if 0 < abs(d_min) < 10.0:
                # Could be q4_k
                return "q4_k", 32
        except:
            pass

        return "unknown", 0

    def validate_all_weight_offsets(self, bump_path: Path) -> bool:
        """
        Main validation: read actual BUMP data at each weight's offset
        and verify structural consistency.
        """
        print("\n" + "=" * 70)
        print("WEIGHT OFFSET & STRUCTURE VALIDATION")
        print("=" * 70)

        all_consistent = True

        with open(bump_path, "rb") as bump_file:
            # Group weights by type
            weights_by_type: Dict[str, List[WeightInfo]] = {}
            for w in self.weights:
                if w.weight_type:
                    if w.weight_type not in weights_by_type:
                        weights_by_type[w.weight_type] = []
                    weights_by_type[w.weight_type].append(w)

            for wtype, weights in weights_by_type.items():
                if len(weights) < 2:
                    continue  # Need at least 2 to check pattern

                print(f"\n  Checking {wtype}: {len(weights)} weights")

                # Check 1: All should have same dtype in manifest
                dtypes = set(w.manifest_dtype for w in weights)
                if len(dtypes) > 1:
                    self.issues.append(OffsetIssue(
                        severity="WARNING",
                        weight_name=wtype,
                        issue="Mixed dtypes in manifest",
                        details=f"Dtypes: {dtypes}"
                    ))

                # Check 2: Read actual dtype from BUMP at first weight offset
                first_weight = weights[0]
                try:
                    detected_dtype, block_size = self.validate_dtype_at_offset(
                        bump_file, first_weight.manifest_offset, first_weight.manifest_dtype
                    )

                    if detected_dtype != first_weight.manifest_dtype:
                        self.issues.append(OffsetIssue(
                            severity="ERROR",
                            weight_name=first_weight.name,
                            issue=f"Dtype mismatch at manifest offset",
                            details=f"Manifest says {first_weight.manifest_dtype}, detected {detected_dtype}"
                        ))
                        all_consistent = False
                except Exception as e:
                    self.issues.append(OffsetIssue(
                        severity="ERROR",
                        weight_name=first_weight.name,
                        issue=f"Failed to read at offset {first_weight.manifest_offset}",
                        details=str(e)
                    ))
                    all_consistent = False

                # Check 3: Verify layer stride pattern
                # Sort by layer and compute offsets
                sorted_weights = sorted([w for w in weights if w.layer is not None],
                                        key=lambda w: w.layer)

                if len(sorted_weights) >= 2:
                    strides = []
                    for i in range(1, len(sorted_weights)):
                        curr = sorted_weights[i]
                        prev = sorted_weights[i-1]
                        stride = curr.manifest_offset - prev.manifest_offset
                        strides.append(stride)

                    # All strides should be equal
                    if len(set(strides)) > 1:
                        self.issues.append(OffsetIssue(
                            severity="ERROR",
                            weight_name=wtype,
                            issue="Inconsistent layer stride",
                            details=f"Strides: {strides[:5]}... (expected consistent)"
                        ))
                        all_consistent = False
                    else:
                        expected_stride = strides[0]
                        print(f"    {wtype} layer stride: {expected_stride:,} bytes")

                # Check 4: Verify offsets are monotonically increasing
                offsets = [w.manifest_offset for w in sorted_weights]
                for i in range(1, len(offsets)):
                    if offsets[i] <= offsets[i-1]:
                        self.issues.append(OffsetIssue(
                            severity="ERROR",
                            weight_name=sorted_weights[i].name,
                            issue="Offset not greater than previous",
                            details=f"Offset {offsets[i]} <= {offsets[i-1]}"
                        ))
                        all_consistent = False

                # Check 5: Sample additional weights to verify consistency
                if len(weights) > 2:
                    sample_indices = [0, len(weights)//2, len(weights)-1]
                    for idx in sample_indices:
                        if idx == 0:
                            continue  # Already checked
                        w = weights[idx]
                        try:
                            # Verify we can read at this offset
                            bump_file.seek(w.manifest_offset)
                            header = bump_file.read(8)
                            if len(header) < 8:
                                self.issues.append(OffsetIssue(
                                    severity="ERROR",
                                    weight_name=w.name,
                                    issue=f"Cannot read {w.manifest_offset}",
                                    details="File may be truncated or offset wrong"
                                ))
                                all_consistent = False
                        except Exception as e:
                            self.issues.append(OffsetIssue(
                                severity="ERROR",
                                weight_name=w.name,
                                issue=f"Seek/read failed at {w.manifest_offset}",
                                details=str(e)
                            ))
                            all_consistent = False

        return all_consistent

    def validate_quantization_blocks(self, bumpPath: Path) -> bool:
        """Verify quantization block structure is valid."""
        print("\n" + "=" * 70)
        print("QUANTIZATION BLOCK VALIDATION")
        print("=" * 70)

        all_valid = True

        with open(bumpPath, "rb") as bump_file:
            for w in self.weights[:10]:  # Sample first 10 weights
                if w.layer is None or w.layer > 2:
                    continue

                dtype = w.manifest_dtype
                offset = w.manifest_offset

                if dtype == "q8_0":
                    bump_file.seek(offset)
                    block_count = 32  # blocks per row for 896 dim

                    valid_blocks = 0
                    for b in range(min(block_count, 10)):  # Check first 10 blocks
                        scale_bytes = bump_file.read(2)
                        if len(scale_bytes) < 2:
                            break

                        scale = struct.unpack('<e', scale_bytes)[0]
                        quants = np.frombuffer(bump_file.read(32), dtype=np.int8)

                        # Valid Q8_0: scale is reasonable fp16, quants in range
                        if 0 < abs(scale) < 1e6 and np.all(quants > -128) and np.all(quants < 128):
                            valid_blocks += 1

                    if valid_blocks > 0:
                        print(f"  {w.name}: {valid_blocks}/10 blocks valid")
                    else:
                        self.issues.append(OffsetIssue(
                            severity="ERROR",
                            weight_name=w.name,
                            issue="No valid Q8_0 blocks found",
                            details=f"Offset {offset} may be wrong"
                        ))
                        all_valid = False
                elif dtype == "q5_0":
                    bump_file.seek(offset)

                    valid_blocks = 0
                    for b in range(min(32, 10)):  # Check first 10 blocks
                        min_bytes = bump_file.read(2)
                        if len(min_bytes) < 2:
                            break

                        min_val = struct.unpack('<e', min_bytes)[0]
                        quants = np.frombuffer(bump_file.read(32), dtype=np.int8)
                        max_bytes = bump_file.read(2)
                        max_val = struct.unpack('<e', max_bytes)[0] if len(max_bytes) >= 2 else 0

                        # Valid Q5_0: min/max reasonable, quants in range
                        if (0 < abs(min_val) < 1e6) and (0 < abs(max_val) < 1e6):
                            if min_val <= max_val:
                                valid_blocks += 1

                    if valid_blocks > 0:
                        print(f"  {w.name}: {valid_blocks}/10 blocks valid")
                    else:
                        self.issues.append(OffsetIssue(
                            severity="ERROR",
                            weight_name=w.name,
                            issue="No valid Q5_0 blocks found",
                            details=f"Offset {offset} may be wrong"
                        ))
                        all_valid = False

        return all_valid

    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70)

        if not self.issues:
            print("No issues found - all weight offsets are consistent!")
            return True

        errors = [i for i in self.issues if i.severity == "ERROR"]
        warnings = [i for i in self.issues if i.severity == "WARNING"]

        print(f"\nErrors: {len(errors)}")
        print(f"Warnings: {len(warnings)}")

        for issue in errors:
            print(f"  [ERROR] {issue.weight_name}")
            print(f"    {issue.issue}")
            print(f"    {issue.details}")

        for issue in warnings:
            print(f"  [WARN] {issue.weight_name}")
            print(f"    {issue.issue}")
            print(f"    {issue.details}")

        print("\n" + "=" * 70)
        if errors:
            print("VERDICT: FAIL - Weight offset inconsistencies detected")
            print("=" * 70)
            return False
        else:
            print("VERDICT: PASS - Weight structure is consistent")
            print("=" * 70)
            return True


def main():
    parser = argparse.ArgumentParser(description="Validate weight offsets in BUMP file")
    parser.add_argument("--model-dir", type=Path,
                       default=Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
    parser.add_argument("--bump", type=Path, help="Path to BUMP file (default: model-dir/weights.bump)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir
    bump_path = args.bump or (model_dir / "weights.bump")

    if not bump_path.exists():
        print(f"ERROR: BUMP file not found: {bump_path}")
        return 1

    validator = WeightOffsetValidator(model_dir)
    if not validator.load_manifest():
        return 1

    offset_ok = validator.validate_all_weight_offsets(bump_path)
    block_ok = validator.validate_quantization_blocks(bump_path)
    report_ok = validator.print_report()

    return 0 if (offset_ok and block_ok and report_ok) else 1


if __name__ == "__main__":
    exit(main())
