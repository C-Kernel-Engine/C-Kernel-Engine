#!/usr/bin/env python3
"""
test_bump_layout_sync.py - Verify bump_layout offsets are consistent across pipeline.

This test ensures that:
1. Converter constants (HEADER_SIZE, EXT_METADATA_SIZE, DATA_START) are correct
2. Manifest.json bump_layout matches converter
3. Layout.json bump_layout matches manifest
4. Generated C code defines match layout.json
5. Actual .bump files have correct structure

Run as part of CI to catch offset mismatches early.

USAGE:
    # Basic validation (just check converter constants are self-consistent)
    python test_bump_layout_sync.py

    # Validate against generated files
    python test_bump_layout_sync.py --manifest=/path/to/weights_manifest.json
    python test_bump_layout_sync.py --layout=/path/to/layout_decode.json
    python test_bump_layout_sync.py --c-file=/path/to/ck-kernel-inference.c

    # Validate actual .bump file structure
    python test_bump_layout_sync.py --bump=/path/to/weights.bump

    # Full pipeline validation
    python test_bump_layout_sync.py --manifest=... --layout=... --c-file=... --bump=...
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

# ============================================================================
# STEP 1: Import converter constants (single source of truth)
# ============================================================================

def get_converter_constants():
    """Import constants from the converter module."""
    # Add scripts directory to path
    scripts_dir = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_dir))

    try:
        # Import the constants from converter
        from convert_gguf_to_bump_v6_6 import HEADER_SIZE, EXT_METADATA_SIZE, DATA_START
        return {
            "header_size": HEADER_SIZE,
            "ext_metadata_size": EXT_METADATA_SIZE,
            "data_start": DATA_START,
            "source": "convert_gguf_to_bump_v6_6.py"
        }
    except ImportError as e:
        print(f"Warning: Could not import converter constants: {e}")
        # Fallback to expected values
        return {
            "header_size": 128,
            "ext_metadata_size": 24,
            "data_start": 152,
            "source": "fallback (import failed)"
        }


# ============================================================================
# STEP 2: Validate self-consistency of converter constants
# ============================================================================

def validate_converter_constants(constants: dict) -> bool:
    """Verify converter constants are self-consistent."""
    print("\n" + "=" * 70)
    print("STEP 1: Validating converter constants")
    print("=" * 70)

    errors = []

    header = constants["header_size"]
    ext_meta = constants["ext_metadata_size"]
    data_start = constants["data_start"]

    print(f"  Source: {constants['source']}")
    print(f"  HEADER_SIZE       = {header}")
    print(f"  EXT_METADATA_SIZE = {ext_meta}")
    print(f"  DATA_START        = {data_start}")

    # Check DATA_START = HEADER_SIZE + EXT_METADATA_SIZE
    expected_data_start = header + ext_meta
    if data_start != expected_data_start:
        errors.append(f"DATA_START mismatch: {data_start} != HEADER_SIZE({header}) + EXT_METADATA_SIZE({ext_meta}) = {expected_data_start}")

    # Check reasonable values
    if header < 64:
        errors.append(f"HEADER_SIZE too small: {header} (expected >= 64)")
    if ext_meta < 0:
        errors.append(f"EXT_METADATA_SIZE negative: {ext_meta}")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    - {e}")
        return False

    print("\n  ✓ Converter constants are self-consistent")
    return True


# ============================================================================
# STEP 3: Validate manifest.json bump_layout
# ============================================================================

def validate_manifest(manifest_path: Path, expected: dict) -> bool:
    """Verify manifest.json bump_layout matches converter."""
    print("\n" + "=" * 70)
    print(f"STEP 2: Validating manifest: {manifest_path}")
    print("=" * 70)

    if not manifest_path.exists():
        print(f"  Skipping: File not found")
        return True  # Not an error if file doesn't exist

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    bump_layout = manifest.get("bump_layout")
    if not bump_layout:
        print("  Warning: No bump_layout in manifest (older format)")
        return True

    errors = []

    print(f"  bump_layout found:")
    print(f"    header_size       = {bump_layout.get('header_size')}")
    print(f"    ext_metadata_size = {bump_layout.get('ext_metadata_size')}")
    print(f"    data_start        = {bump_layout.get('data_start')}")

    # Compare with expected
    for key in ["header_size", "ext_metadata_size", "data_start"]:
        manifest_val = bump_layout.get(key)
        expected_val = expected.get(key)
        if manifest_val != expected_val:
            errors.append(f"{key}: manifest={manifest_val}, expected={expected_val}")

    if errors:
        print("\n  ERRORS (manifest doesn't match converter):")
        for e in errors:
            print(f"    - {e}")
        return False

    print("\n  ✓ Manifest bump_layout matches converter")
    return True


# ============================================================================
# STEP 4: Validate layout.json bump_layout
# ============================================================================

def validate_layout(layout_path: Path, expected: dict) -> bool:
    """Verify layout.json bump_layout matches converter."""
    print("\n" + "=" * 70)
    print(f"STEP 3: Validating layout: {layout_path}")
    print("=" * 70)

    if not layout_path.exists():
        print(f"  Skipping: File not found")
        return True

    with open(layout_path, 'r') as f:
        layout = json.load(f)

    bump_layout = layout.get("bump_layout")
    if not bump_layout:
        print("  Warning: No bump_layout in layout (older format)")
        return True

    errors = []

    print(f"  bump_layout found:")
    print(f"    header_size       = {bump_layout.get('header_size')}")
    print(f"    ext_metadata_size = {bump_layout.get('ext_metadata_size')}")
    print(f"    data_start        = {bump_layout.get('data_start')}")

    # Compare with expected
    for key in ["header_size", "ext_metadata_size", "data_start"]:
        layout_val = bump_layout.get(key)
        expected_val = expected.get(key)
        if layout_val != expected_val:
            errors.append(f"{key}: layout={layout_val}, expected={expected_val}")

    if errors:
        print("\n  ERRORS (layout doesn't match converter):")
        for e in errors:
            print(f"    - {e}")
        return False

    print("\n  ✓ Layout bump_layout matches converter")
    return True


# ============================================================================
# STEP 5: Validate generated C code defines
# ============================================================================

def validate_c_file(c_path: Path, expected: dict) -> bool:
    """Verify generated C code defines match converter."""
    print("\n" + "=" * 70)
    print(f"STEP 4: Validating C code: {c_path}")
    print("=" * 70)

    if not c_path.exists():
        print(f"  Skipping: File not found")
        return True

    with open(c_path, 'r') as f:
        c_code = f.read()

    errors = []
    found = {}

    # Parse #define statements
    patterns = {
        "header_size": r"#define\s+BUMP_HEADER_SIZE\s+(\d+)",
        "ext_metadata_size": r"#define\s+BUMP_EXT_METADATA_SIZE\s+(\d+)",
        "data_start": r"#define\s+BUMP_DATA_START\s+(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, c_code)
        if match:
            found[key] = int(match.group(1))
        else:
            found[key] = None

    print(f"  C defines found:")
    print(f"    BUMP_HEADER_SIZE       = {found.get('header_size')}")
    print(f"    BUMP_EXT_METADATA_SIZE = {found.get('ext_metadata_size')}")
    print(f"    BUMP_DATA_START        = {found.get('data_start')}")

    # Compare with expected
    for key in ["header_size", "ext_metadata_size", "data_start"]:
        c_val = found.get(key)
        expected_val = expected.get(key)
        if c_val is None:
            # Not an error if define not found (might be older codegen)
            print(f"  Warning: BUMP_{key.upper()} not found in C code")
        elif c_val != expected_val:
            errors.append(f"BUMP_{key.upper()}: C={c_val}, expected={expected_val}")

    if errors:
        print("\n  ERRORS (C code doesn't match converter):")
        for e in errors:
            print(f"    - {e}")
        return False

    print("\n  ✓ C code defines match converter")
    return True


# ============================================================================
# STEP 6: Validate actual .bump file structure
# ============================================================================

def validate_bump_file(bump_path: Path, expected: dict) -> bool:
    """Verify actual .bump file has correct structure."""
    print("\n" + "=" * 70)
    print(f"STEP 5: Validating .bump file: {bump_path}")
    print("=" * 70)

    if not bump_path.exists():
        print(f"  Skipping: File not found")
        return True

    errors = []

    with open(bump_path, 'rb') as f:
        # Read magic
        magic = f.read(8)
        print(f"  Magic: {magic}")

        if magic not in (b"BUMPWGT4", b"BUMPWGT5"):
            errors.append(f"Invalid magic: {magic} (expected BUMPWGT4/BUMPWGT5)")
            print("\n  ERRORS:")
            for e in errors:
                print(f"    - {e}")
            return False

        # Seek to DATA_START and try to read dtype_table_len
        data_start = expected["data_start"]
        f.seek(data_start)
        dtype_table_len_bytes = f.read(4)

        if len(dtype_table_len_bytes) < 4:
            errors.append(f"File too small to read dtype_table_len at offset {data_start}")
        else:
            dtype_table_len = struct.unpack("<I", dtype_table_len_bytes)[0]
            print(f"  dtype_table_len at offset {data_start}: {dtype_table_len}")

            # Sanity check - dtype_table_len should be reasonable (< 10000)
            if dtype_table_len > 10000:
                errors.append(f"dtype_table_len too large: {dtype_table_len} (likely reading from wrong offset)")
            elif dtype_table_len == 0:
                errors.append(f"dtype_table_len is 0 (likely reading from wrong offset)")
            else:
                # Try to read dtype_table and verify it looks reasonable
                dtype_table = f.read(dtype_table_len)
                if len(dtype_table) < dtype_table_len:
                    errors.append(f"Could not read full dtype_table")
                else:
                    # dtype values should be 0-10 (CK_DT_* enum values)
                    valid_dtypes = all(b <= 10 for b in dtype_table)
                    if not valid_dtypes:
                        errors.append(f"dtype_table contains invalid values (not CK_DT_* enum)")
                    else:
                        print(f"  dtype_table: {dtype_table_len} entries, all valid")

                        # Calculate weights start
                        weights_start = data_start + 4 + dtype_table_len
                        print(f"  Weights start at: {weights_start}")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    - {e}")
        return False

    print("\n  ✓ .bump file structure is valid")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify bump_layout offsets are consistent across pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--manifest", type=Path, help="Path to weights_manifest.json")
    parser.add_argument("--layout", type=Path, help="Path to layout_decode.json")
    parser.add_argument("--c-file", type=Path, help="Path to generated ck-kernel-inference.c")
    parser.add_argument("--bump", type=Path, help="Path to .bump file to validate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 70)
    print("BUMP LAYOUT SYNC VALIDATION")
    print("=" * 70)

    # Get expected values from converter
    expected = get_converter_constants()

    all_passed = True

    # Step 1: Validate converter constants
    if not validate_converter_constants(expected):
        all_passed = False

    # Step 2: Validate manifest (if provided)
    if args.manifest:
        if not validate_manifest(args.manifest, expected):
            all_passed = False

    # Step 3: Validate layout (if provided)
    if args.layout:
        if not validate_layout(args.layout, expected):
            all_passed = False

    # Step 4: Validate C file (if provided)
    if args.c_file:
        if not validate_c_file(args.c_file, expected):
            all_passed = False

    # Step 5: Validate .bump file (if provided)
    if args.bump:
        if not validate_bump_file(args.bump, expected):
            all_passed = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_passed:
        print("\n  ✓ ALL VALIDATIONS PASSED")
        print("\n  Pipeline offsets are in sync:")
        print(f"    HEADER_SIZE       = {expected['header_size']}")
        print(f"    EXT_METADATA_SIZE = {expected['ext_metadata_size']}")
        print(f"    DATA_START        = {expected['data_start']}")
        return 0
    else:
        print("\n  ✗ SOME VALIDATIONS FAILED")
        print("\n  Check the errors above and ensure all files use the same offsets.")
        print("  The single source of truth is: convert_gguf_to_bump_v6_6.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
