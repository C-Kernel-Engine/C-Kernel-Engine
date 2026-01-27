#!/usr/bin/env python3
"""
Minimal step-by-step validation of the C-Kernel-Engine pipeline.
Tests each component in isolation to identify the failure point.
"""
import json
import struct
from pathlib import Path

CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

def check_bumpwgt5_format():
    """Check BUMPWGT5 file format is valid."""
    print("\n" + "="*60)
    print("CHECK 1: BUMPWGT5 File Format")
    print("="*60)

    bump_path = CACHE_DIR / "weights.bump"
    if not bump_path.exists():
        print(f"ERROR: {bump_path} not found")
        return False

    size = bump_path.stat().st_size
    print(f"File size: {size:,} bytes ({size/1024/1024:.1f} MB)")

    with open(bump_path, 'rb') as f:
        # Read header + extended metadata + dtype_table_len (128 + 24 + 4 = 156 bytes)
        header = f.read(156)

        # Parse header
        magic = header[0:8].decode('utf-8', errors='replace')
        version = struct.unpack('<I', header[8:12])[0]
        bump_size = struct.unpack('<Q', header[16:24])[0]
        weight_count = struct.unpack('<I', header[96:100])[0]
        # dtype_table_len is at offset 152 (128 header + 24 ext_meta)
        dtype_table_len = struct.unpack('<I', header[152:156])[0]

        print(f"Magic: {magic}")
        print(f"Version: {version}")
        print(f"BUMP size: {bump_size:,}")
        print(f"Weight count: {weight_count}")
        print(f"dtype_table_len: {dtype_table_len}")

        # Check if file is big enough
        weights_start = 152 + 4 + dtype_table_len  # header + ext + len + table
        if size < weights_start:
            print(f"ERROR: File too small ({size}) for weights start ({weights_start})")
            return False

        # Read dtype_table
        f.seek(156)
        dtype_table = list(struct.unpack(f'<{dtype_table_len}B', f.read(dtype_table_len)))
        print(f"dtype_table entries: {len(dtype_table)}")
        print(f"First 10 dtype values: {dtype_table[:10]}")

        print("✓ BUMPWGT5 format OK")
        return True

def check_layout_buffers():
    """Check layout has correct buffer offsets."""
    print("\n" + "="*60)
    print("CHECK 2: Memory Layout Buffers")
    print("="*60)

    layout_path = CACHE_DIR / "layout_decode.json"
    if not layout_path.exists():
        print(f"ERROR: {layout_path} not found")
        return False

    with open(layout_path) as f:
        layout = json.load(f)

    memory = layout.get('memory', {})
    activations = memory.get('activations', {})
    buffers = activations.get('buffers', [])

    print("Activation buffers:")
    for buf in buffers:
        print(f"  {buf['name']}: offset={buf['offset']}, size={buf['size']}")

    # Check critical buffers exist
    critical = ['token_ids', 'embedded_input', 'layer_input']
    for name in critical:
        found = any(b['name'] == name for b in buffers)
        if not found:
            print(f"ERROR: Missing critical buffer '{name}'")
            return False
        else:
            buf = next(b for b in buffers if b['name'] == name)
            print(f"  ✓ {name} at offset {buf['offset']}")

    print("✓ Layout OK")
    return True

def check_ir_offsets():
    """Check IR has correct offsets for each op."""
    print("\n" + "="*60)
    print("CHECK 3: IR Offsets")
    print("="*60)

    ir_path = CACHE_DIR / "lowered_decode.json"
    if not ir_path.exists():
        print(f"ERROR: {ir_path} not found")
        return False

    with open(ir_path) as f:
        ir = json.load(f)

    ops = ir.get('operations', [])
    if not ops:
        print("ERROR: No operations in IR")
        return False

    # Check Op 0 (embedding)
    print("\nOp 0 (embedding):")
    op0 = ops[0]
    print(f"  Kernel: {op0.get('kernel', 'N/A')}")

    # Check inputs
    inputs = op0.get('activations', {})
    if 'tokens' in inputs:
        token_offset = inputs['tokens'].get('activation_offset', -1)
        print(f"  Token input offset: {token_offset}")
        if token_offset != 16:
            print(f"  ERROR: Token should be at 16, got {token_offset}")
            return False
        else:
            print(f"  ✓ Token at correct offset 16")

    # Check outputs
    outputs = op0.get('outputs', {})
    for name, out in outputs.items():
        offset = out.get('activation_offset', -1)
        buffer = out.get('buffer', 'N/A')
        print(f"  Output '{name}': buffer={buffer}, offset={offset}")

        if buffer == 'layer_output' and offset > 1000:
            print(f"  ERROR: Embedding outputs to wrong buffer '{buffer}' at {offset}")
            print(f"  Expected: embedded_input at offset 20")
            return False
        elif buffer == 'embedded_input' and offset == 20:
            print(f"  ✓ Embedding outputs to embedded_input at 20")

    print("\n✓ IR offsets OK")
    return True

def check_generated_code():
    """Check generated C code has correct offsets."""
    print("\n" + "="*60)
    print("CHECK 4: Generated C Code")
    print("="*60)

    code_path = CACHE_DIR / "ck-kernel-inference.c"
    if not code_path.exists():
        print(f"ERROR: {code_path} not found")
        return False

    with open(code_path) as f:
        code = f.read()

    # Check token storage offset
    import re
    token_store = re.search(r'/\*\s*Store token.*?\*/\s*\*\(int32_t\*\)\(ACT \+ (\d+)\)', code)
    if token_store:
        token_offset = int(token_store.group(1))
        print(f"Token storage offset: {token_offset}")
        if token_offset != 16:
            print(f"ERROR: Token should be stored at 16, got {token_offset}")
            return False
        else:
            print(f"  ✓ Token stored at correct offset 16")
    else:
        print("WARNING: Could not find token store pattern")

    # Check embedding input offset
    embed_input = re.search(r'embedding_forward.*?ACT \+ (\d+)', code, re.DOTALL)
    if embed_input:
        embed_offset = int(embed_input.group(1))
        print(f"Embedding input offset: {embed_offset}")
        if embed_offset != 16:
            print(f"  NOTE: Embedding reads from offset {embed_offset}")

    print("\n✓ Generated code OK")
    return True

def main():
    print("="*60)
    print("STEP-BY-STEP VALIDATION")
    print("="*60)

    checks = [
        ("BUMPWGT5 Format", check_bumpwgt5_format),
        ("Layout Buffers", check_layout_buffers),
        ("IR Offsets", check_ir_offsets),
        ("Generated Code", check_generated_code),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_passed = False

    if all_passed:
        print("\nAll checks passed! Ready to test inference.")
    else:
        print("\nSome checks failed. Fix issues above before testing.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
