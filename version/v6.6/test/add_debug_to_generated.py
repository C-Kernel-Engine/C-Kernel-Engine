#!/usr/bin/env python3
"""
add_debug_to_generated.py - Add debug fprintf statements to generated_model.c

This modifies the generated C code to print intermediate values at each step,
helping us find exactly where the bug occurs.

USAGE:
    python add_debug_to_generated.py ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/generated_model.c
    # Then recompile:
    gcc -shared -fPIC -O2 generated_model.c -L. -lckernel_engine -o libmodel.so
"""

import sys
import re
from pathlib import Path


def add_debug_hooks(code: str) -> str:
    """Add debug fprintf statements after key operations."""

    # Add debug macro at top
    debug_header = '''
/* ============================================================================
 * DEBUG HOOKS - Added by add_debug_to_generated.py
 * ============================================================================ */
#define DEBUG_V66 1

#ifdef DEBUG_V66
#include <stdio.h>
static void debug_print_array(const char* name, const float* arr, int size) {
    float min_v = arr[0], max_v = arr[0], sum = 0;
    int num_zeros = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] < min_v) min_v = arr[i];
        if (arr[i] > max_v) max_v = arr[i];
        sum += arr[i];
        if (arr[i] == 0.0f) num_zeros++;
    }
    fprintf(stderr, "[DEBUG] %s: min=%.6f, max=%.6f, mean=%.6f, zeros=%d/%d\\n",
            name, min_v, max_v, sum/size, num_zeros, size);
    fprintf(stderr, "        first5: [%.6f, %.6f, %.6f, %.6f, %.6f]\\n",
            arr[0], arr[1], arr[2], arr[3], arr[4]);
}

static void debug_print_int(const char* name, int32_t val) {
    fprintf(stderr, "[DEBUG] %s: %d\\n", name, val);
}
#define DEBUG_ARRAY(name, arr, size) debug_print_array(name, arr, size)
#define DEBUG_INT(name, val) debug_print_int(name, val)
#else
#define DEBUG_ARRAY(name, arr, size)
#define DEBUG_INT(name, val)
#endif

'''

    # Insert after includes
    include_end = code.find('#define QWEN2_DECODE')
    if include_end == -1:
        include_end = code.find('typedef struct')
    code = code[:include_end] + debug_header + code[include_end:]

    # Add debug after token store
    token_store_pattern = r'(\*\(\(int32_t\*\)\(\(uint8_t\*\)model->activations \+ (\d+)\)\) = token;)'
    def add_token_debug(m):
        offset = m.group(2)
        return m.group(1) + f'''
    DEBUG_INT("token_stored_at_offset_{offset}", token);'''
    code = re.sub(token_store_pattern, add_token_debug, code)

    # Add debug after embedding
    emb_pattern = r'(embedding_forward_q8_0\([^;]+\);)'
    def add_emb_debug(m):
        return m.group(1) + '''
    DEBUG_ARRAY("embedding_output", ((float*)((uint8_t*)model->activations + 813751828)), 896);'''
    code = re.sub(emb_pattern, add_emb_debug, code, count=1)

    # Add debug after first rmsnorm
    rms_pattern = r'(rmsnorm_forward\([^;]+\);)'
    rms_count = [0]
    def add_rms_debug(m):
        rms_count[0] += 1
        if rms_count[0] <= 3:  # Only first 3 rmsnorms
            return m.group(1) + f'''
    DEBUG_ARRAY("rmsnorm_{rms_count[0]}_output", ((float*)((uint8_t*)model->activations + 813751828)), 896);'''
        return m.group(1)
    code = re.sub(rms_pattern, add_rms_debug, code)

    # Add debug after first mega_fused_attention
    attn_pattern = r'(mega_fused_attention_prefill\([^;]+\);)'
    attn_count = [0]
    def add_attn_debug(m):
        attn_count[0] += 1
        if attn_count[0] <= 2:  # Only first 2
            return m.group(1) + f'''
    DEBUG_ARRAY("fused_attn_{attn_count[0]}_output", ((float*)((uint8_t*)model->activations + 3604)), 896);'''
        return m.group(1)
    code = re.sub(attn_pattern, add_attn_debug, code)

    # Add debug before return in decode function
    decode_end = code.find('/* Increment position */')
    if decode_end != -1:
        code = code[:decode_end] + '''
    /* Debug: Check layer_input before returning */
    DEBUG_ARRAY("layer_input_final", ((float*)((uint8_t*)model->activations + 3604)), 896);
    DEBUG_ARRAY("layer_output_final", ((float*)((uint8_t*)model->activations + 813751828)), 896);

    ''' + code[decode_end:]

    # Add debug for logits
    logits_pattern = r'(return model->logits;)'
    code = re.sub(logits_pattern, r'''DEBUG_ARRAY("logits_returned", model->logits, 100);
    \1''', code)

    return code


def main():
    if len(sys.argv) < 2:
        print("Usage: python add_debug_to_generated.py <generated_model.c>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        sys.exit(1)

    print(f"Adding debug hooks to {path}...")

    with open(path) as f:
        code = f.read()

    # Backup original
    backup_path = path.with_suffix('.c.orig')
    if not backup_path.exists():
        with open(backup_path, 'w') as f:
            f.write(code)
        print(f"  Backed up to {backup_path}")

    # Add debug hooks
    debug_code = add_debug_hooks(code)

    with open(path, 'w') as f:
        f.write(debug_code)

    print(f"  Added debug hooks")
    print(f"\nNow recompile:")
    print(f"  cd {path.parent}")
    print(f"  gcc -shared -fPIC -O0 -g {path.name} -L. -lckernel_engine -o libmodel.so")


if __name__ == "__main__":
    main()
