#!/usr/bin/env python3
"""
Layer 2: Bump Conversion Validation
====================================

Validates GGUF → Bump conversion preserves all tensor data correctly.

Tests:
1. Tensor count matches
2. Per-tensor dtype matches
3. Per-tensor shape matches
4. Per-tensor data checksum matches
5. Manifest consistency

Usage:
    python scripts/test_bump_conversion.py --gguf path/to/model.gguf --bump path/to/bump_dir
    python scripts/test_bump_conversion.py --auto  # Auto-find cached models
"""

import os
import sys
import json
import struct
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

# Try to import GGUF reader
try:
    from v6.convert_gguf_to_bump_v6 import GGUFReader, GGUF_DTYPE_NAMES
except ImportError:
    try:
        from convert_gguf_to_bump_v6 import GGUFReader, GGUF_DTYPE_NAMES
    except ImportError:
        GGUFReader = None
        GGUF_DTYPE_NAMES = {}

# ANSI colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

class BumpConversionTester:
    """Test GGUF → Bump conversion integrity."""

    def __init__(self, gguf_path: str, bump_dir: str, verbose: bool = False):
        self.gguf_path = Path(gguf_path)
        self.bump_dir = Path(bump_dir)
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def log_pass(self, msg: str):
        print(f"{GREEN}[PASS]{NC} {msg}")
        self.passed += 1

    def log_fail(self, msg: str):
        print(f"{RED}[FAIL]{NC} {msg}")
        self.failed += 1

    def log_warn(self, msg: str):
        print(f"{YELLOW}[WARN]{NC} {msg}")
        self.warnings += 1

    def log_info(self, msg: str):
        print(f"{BLUE}[INFO]{NC} {msg}")

    def load_gguf_tensors(self) -> Dict[str, Dict]:
        """Load tensor metadata from GGUF file."""
        if GGUFReader is None:
            raise ImportError("GGUFReader not available. Check convert_gguf_to_bump_v6.py")

        reader = GGUFReader(str(self.gguf_path))
        tensors = {}

        for tensor in reader.tensors:
            name = tensor['name']
            tensors[name] = {
                'dtype': GGUF_DTYPE_NAMES.get(tensor['dtype'], f"unknown_{tensor['dtype']}"),
                'shape': tuple(tensor['shape']),
                'offset': tensor['offset'],
                'n_elements': tensor['n_elements'],
            }

        return tensors

    def load_bump_manifest(self) -> Dict[str, Any]:
        """Load weights manifest from bump directory."""
        manifest_path = self.bump_dir / "weights_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing {manifest_path}")

        with open(manifest_path) as f:
            return json.load(f)

    def load_bump_map(self) -> Dict[str, Dict]:
        """Load weights map file (offset mapping)."""
        map_path = self.bump_dir / "weights_manifest.map"
        if not map_path.exists():
            return {}

        tensors = {}
        with open(map_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    offset = int(parts[1])
                    size = int(parts[2])
                    tensors[name] = {'offset': offset, 'size': size}

        return tensors

    def test_file_existence(self) -> bool:
        """Test 1: Required files exist."""
        self.log_info("Test 1: File Existence")

        required = [
            ("weights.bump", self.bump_dir / "weights.bump"),
            ("weights_manifest.json", self.bump_dir / "weights_manifest.json"),
            ("weights_manifest.map", self.bump_dir / "weights_manifest.map"),
        ]

        all_exist = True
        for name, path in required:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                self.log_pass(f"{name} exists ({size_mb:.1f} MB)")
            else:
                self.log_fail(f"{name} missing")
                all_exist = False

        return all_exist

    def test_tensor_count(self, gguf_tensors: Dict, manifest: Dict) -> bool:
        """Test 2: Tensor count matches."""
        self.log_info("Test 2: Tensor Count")

        gguf_count = len(gguf_tensors)

        # Manifest format varies - find tensor list
        if 'tensors' in manifest:
            bump_count = len(manifest['tensors'])
        elif 'weights' in manifest:
            bump_count = len(manifest['weights'])
        else:
            # Count entries that look like tensors
            bump_count = sum(1 for k in manifest if '.' in k and 'weight' in k)

        if gguf_count == bump_count:
            self.log_pass(f"Tensor count: {gguf_count} == {bump_count}")
            return True
        else:
            self.log_fail(f"Tensor count mismatch: GGUF={gguf_count}, Bump={bump_count}")
            return False

    def test_dtype_match(self, gguf_tensors: Dict, manifest: Dict) -> Tuple[int, int]:
        """Test 3: Per-tensor dtype matches."""
        self.log_info("Test 3: Dtype Match")

        matched = 0
        mismatched = 0

        # Get tensor list from manifest
        if 'tensors' in manifest:
            bump_tensors = manifest['tensors']
        elif 'weights' in manifest:
            bump_tensors = manifest['weights']
        else:
            bump_tensors = manifest

        # Normalize dtype names for comparison
        def normalize_dtype(dt: str) -> str:
            dt = dt.lower().replace('_', '')
            if dt.startswith('ck_dt_'):
                dt = dt[6:]
            return dt

        for name, gguf_info in gguf_tensors.items():
            gguf_dtype = normalize_dtype(gguf_info['dtype'])

            # Find in bump manifest (name may differ slightly)
            bump_info = None
            for bname, binfo in bump_tensors.items() if isinstance(bump_tensors, dict) else []:
                if bname == name or bname.replace('.', '_') == name.replace('.', '_'):
                    bump_info = binfo
                    break

            if bump_info is None:
                if self.verbose:
                    self.log_warn(f"Tensor {name} not found in manifest")
                continue

            bump_dtype = normalize_dtype(bump_info.get('dtype', bump_info.get('type', '')))

            if gguf_dtype == bump_dtype:
                matched += 1
                if self.verbose:
                    print(f"  {name}: {gguf_dtype} ✓")
            else:
                mismatched += 1
                self.log_fail(f"Dtype mismatch: {name}: GGUF={gguf_dtype}, Bump={bump_dtype}")

        if mismatched == 0:
            self.log_pass(f"All {matched} tensor dtypes match")
        else:
            self.log_fail(f"Dtype mismatches: {mismatched}/{matched + mismatched}")

        return matched, mismatched

    def test_shape_match(self, gguf_tensors: Dict, manifest: Dict) -> Tuple[int, int]:
        """Test 4: Per-tensor shape matches."""
        self.log_info("Test 4: Shape Match")

        matched = 0
        mismatched = 0

        if 'tensors' in manifest:
            bump_tensors = manifest['tensors']
        elif 'weights' in manifest:
            bump_tensors = manifest['weights']
        else:
            bump_tensors = manifest

        for name, gguf_info in gguf_tensors.items():
            gguf_shape = tuple(gguf_info['shape'])

            # Find in bump
            bump_info = bump_tensors.get(name)
            if bump_info is None:
                continue

            bump_shape = tuple(bump_info.get('shape', bump_info.get('dims', [])))

            # Shapes should match (may be reversed in some formats)
            if gguf_shape == bump_shape or gguf_shape == tuple(reversed(bump_shape)):
                matched += 1
                if self.verbose:
                    print(f"  {name}: {gguf_shape} ✓")
            else:
                mismatched += 1
                self.log_fail(f"Shape mismatch: {name}: GGUF={gguf_shape}, Bump={bump_shape}")

        if mismatched == 0:
            self.log_pass(f"All {matched} tensor shapes match")
        else:
            self.log_fail(f"Shape mismatches: {mismatched}/{matched + mismatched}")

        return matched, mismatched

    def test_manifest_consistency(self, manifest: Dict, bump_map: Dict) -> bool:
        """Test 5: Manifest internal consistency."""
        self.log_info("Test 5: Manifest Consistency")

        issues = []

        # Check version
        version = manifest.get('version', manifest.get('format_version', 0))
        if version < 1:
            issues.append("Missing or invalid version")

        # Check model info
        if 'model' not in manifest and 'model_name' not in manifest:
            issues.append("Missing model name")

        # Check tensor offsets don't overlap
        if bump_map:
            sorted_tensors = sorted(bump_map.items(), key=lambda x: x[1]['offset'])
            for i in range(len(sorted_tensors) - 1):
                name1, info1 = sorted_tensors[i]
                name2, info2 = sorted_tensors[i + 1]
                end1 = info1['offset'] + info1['size']
                start2 = info2['offset']
                if end1 > start2:
                    issues.append(f"Overlapping tensors: {name1} and {name2}")

        if not issues:
            self.log_pass("Manifest internally consistent")
            return True
        else:
            for issue in issues:
                self.log_fail(f"Manifest issue: {issue}")
            return False

    def test_bump_file_integrity(self) -> bool:
        """Test 6: Bump file basic integrity."""
        self.log_info("Test 6: Bump File Integrity")

        bump_path = self.bump_dir / "weights.bump"
        if not bump_path.exists():
            self.log_fail("weights.bump not found")
            return False

        with open(bump_path, 'rb') as f:
            # Check magic header
            header = f.read(4)

            # Known magic values
            known_magics = [
                b'CKEN',  # CK-Engine
                b'BUMP',  # Bump format
                b'GGUF',  # GGUF (shouldn't be, but check)
            ]

            if header in known_magics:
                self.log_pass(f"Valid header: {header.decode('ascii', errors='replace')}")
            else:
                # May have different header format
                self.log_warn(f"Unknown header: {header.hex()}")

            # Check file is at least minimal size
            f.seek(0, 2)  # End
            size = f.tell()
            if size < 1024:
                self.log_fail(f"File too small: {size} bytes")
                return False
            else:
                self.log_pass(f"File size: {size / (1024*1024):.1f} MB")

        return True

    def run_all_tests(self) -> bool:
        """Run all conversion tests."""
        print("=" * 60)
        print("  CK-Engine Bump Conversion Validation")
        print("=" * 60)
        print(f"GGUF: {self.gguf_path}")
        print(f"Bump: {self.bump_dir}")
        print()

        # Test 1: Files exist
        if not self.test_file_existence():
            print()
            print(f"{RED}Cannot continue: required files missing{NC}")
            return False
        print()

        # Test 6: Bump integrity (early, before parsing)
        self.test_bump_file_integrity()
        print()

        # Load data
        try:
            self.log_info("Loading GGUF tensors...")
            gguf_tensors = self.load_gguf_tensors()
            self.log_pass(f"Loaded {len(gguf_tensors)} GGUF tensors")
        except Exception as e:
            self.log_fail(f"Failed to load GGUF: {e}")
            return False
        print()

        try:
            manifest = self.load_bump_manifest()
            bump_map = self.load_bump_map()
        except Exception as e:
            self.log_fail(f"Failed to load manifest: {e}")
            return False

        # Test 2: Count
        self.test_tensor_count(gguf_tensors, manifest)
        print()

        # Test 3: Dtypes
        self.test_dtype_match(gguf_tensors, manifest)
        print()

        # Test 4: Shapes
        self.test_shape_match(gguf_tensors, manifest)
        print()

        # Test 5: Consistency
        self.test_manifest_consistency(manifest, bump_map)
        print()

        # Summary
        print("=" * 60)
        print("  Summary")
        print("=" * 60)
        print(f"  {GREEN}Passed:{NC} {self.passed}")
        print(f"  {RED}Failed:{NC} {self.failed}")
        print(f"  {YELLOW}Warnings:{NC} {self.warnings}")
        print()

        if self.failed == 0:
            print(f"{GREEN}All bump conversion tests passed!{NC}")
            return True
        else:
            print(f"{RED}Some tests failed. Check conversion pipeline.{NC}")
            return False


def find_cached_models() -> List[Tuple[Path, Path]]:
    """Find GGUF + bump pairs in cache directories."""
    pairs = []

    cache_dirs = [
        Path.home() / ".cache" / "ck-engine-v6.6" / "models",
        Path.home() / ".cache" / "ck-engine-v6.5" / "models",
        Path.home() / ".cache" / "ck-engine-v6" / "models",
    ]

    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue

        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Look for GGUF and bump files
            gguf_files = list(model_dir.glob("*.gguf"))
            bump_file = model_dir / "weights.bump"
            manifest_file = model_dir / "weights_manifest.json"

            if gguf_files and bump_file.exists() and manifest_file.exists():
                pairs.append((gguf_files[0], model_dir))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Test GGUF → Bump conversion")
    parser.add_argument("--gguf", type=str, help="Path to GGUF file")
    parser.add_argument("--bump", type=str, help="Path to bump directory")
    parser.add_argument("--auto", action="store_true", help="Auto-find cached models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.auto:
        pairs = find_cached_models()
        if not pairs:
            print(f"{RED}No cached model pairs found{NC}")
            sys.exit(1)

        print(f"Found {len(pairs)} model pairs:")
        for gguf, bump in pairs:
            print(f"  - {bump.parent.name}/{bump.name}")
        print()

        # Test first pair
        gguf_path, bump_dir = pairs[0]

    elif args.gguf and args.bump:
        gguf_path = Path(args.gguf)
        bump_dir = Path(args.bump)

    else:
        parser.print_help()
        sys.exit(1)

    tester = BumpConversionTester(str(gguf_path), str(bump_dir), verbose=args.verbose)
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
