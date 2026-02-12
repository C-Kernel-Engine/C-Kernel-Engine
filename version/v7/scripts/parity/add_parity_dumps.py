#!/usr/bin/env python3
"""
add_parity_dumps.py - Add activation dump instrumentation to generated model

Usage:
    python version/v7/scripts/parity/add_parity_dumps.py <model_v7.c> [--output <patched.c>]

This script adds dump_tensor() calls after key ops in the generated model,
enabling activation comparison with llama.cpp.
"""

import re
import sys
import argparse
from pathlib import Path


DUMP_HEADER = '''
/* ============================================================================
 * ACTIVATION PARITY DUMP - Compare with llama.cpp
 * Enable with: CK_DUMP_ACTIVATIONS=1
 * ============================================================================ */
#include <sys/stat.h>

static int g_dump_enabled = 0;
static char g_dump_dir[256] = "ck_parity_dumps";

static void dump_init(void) {
    g_dump_enabled = getenv("CK_DUMP_ACTIVATIONS") != NULL;
    if (g_dump_enabled) {
        const char *dir = getenv("CK_DUMP_DIR");
        if (dir) strncpy(g_dump_dir, dir, 255);
        mkdir(g_dump_dir, 0755);
        fprintf(stderr, "[CK_DUMP] Enabled, output: %s/\\n", g_dump_dir);
    }
}

static void dump_tensor(const char *name, int layer, const float *data, int count) {
    if (!g_dump_enabled || !data) return;

    char path[512];
    if (layer >= 0)
        snprintf(path, sizeof(path), "%s/L%d_%s.bin", g_dump_dir, layer, name);
    else
        snprintf(path, sizeof(path), "%s/%s.bin", g_dump_dir, name);

    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(data, sizeof(float), count, f);
        fclose(f);

        /* Print first 5 values */
        fprintf(stderr, "[CK_DUMP] %s [%d]: %.6f %.6f %.6f %.6f %.6f\\n",
                path, count,
                count > 0 ? data[0] : 0.0f,
                count > 1 ? data[1] : 0.0f,
                count > 2 ? data[2] : 0.0f,
                count > 3 ? data[3] : 0.0f,
                count > 4 ? data[4] : 0.0f);
    }
}
'''


# Patterns for ops we want to dump after
# Format: (op_pattern, dump_name, buffer_pattern, size_pattern)
DUMP_POINTS = [
    # Embedding
    (r'embedding_forward_q8_0\([^)]+\);',
     'embedding', 'A_EMBEDDED_INPUT', '640'),

    # Attention norms (layer 0 only for now)
    (r'/\* Op \d+: rmsnorm_forward \(attn_norm\) layer=0',
     'L0_attn_norm', 'A_EMBEDDED_INPUT', '640'),

    # Q projection (after bias add)
    (r'/\* Op 5: add_inplace_f32 \(bias_add\) layer=0.*?if \(stop_seq == 5\)',
     'L0_q_proj', 'A_Q_SCRATCH', '1024'),

    # K projection (after bias add)
    (r'/\* Op 7: add_inplace_f32 \(bias_add\) layer=0.*?if \(stop_seq == 7\)',
     'L0_k_proj', 'A_K_SCRATCH', '256'),

    # V projection (after bias add)
    (r'/\* Op 9: add_inplace_f32 \(bias_add\) layer=0.*?if \(stop_seq == 9\)',
     'L0_v_proj', 'A_V_SCRATCH', '256'),

    # Post QK-norm
    (r'/\* Op 10: qk_norm_forward.*?if \(stop_seq == 10\)',
     'L0_q_post_norm', 'A_Q_SCRATCH', '1024'),

    # Post RoPE
    (r'/\* Op 11: rope_forward_qk.*?if \(stop_seq == 11\)',
     'L0_q_post_rope', 'A_Q_SCRATCH', '1024'),

    # Attention output
    (r'/\* Op 13: attention_forward.*?if \(stop_seq == 13\)',
     'L0_attn_out', 'A_ATTN_SCRATCH', '1024'),

    # Out projection (after bias)
    (r'/\* Op 15: add_inplace_f32 \(bias_add\) layer=0.*?if \(stop_seq == 15\)',
     'L0_out_proj', 'A_EMBEDDED_INPUT', '640'),

    # FFN norm
    (r'/\* Op 19: rmsnorm_forward \(ffn_norm\) layer=0.*?if \(stop_seq == 19\)',
     'L0_ffn_norm', 'A_EMBEDDED_INPUT', '640'),

    # MLP gate+up (after bias)
    (r'/\* Op 21: add_inplace_f32 \(bias_add\) layer=0.*?if \(stop_seq == 21\)',
     'L0_mlp_gate_up', 'A_MLP_SCRATCH', '4096'),

    # GeGLU output
    (r'/\* Op 22: geglu_forward_fp32.*?if \(stop_seq == 22\)',
     'L0_geglu_out', 'A_MLP_SCRATCH', '2048'),

    # MLP down (after bias)
    (r'/\* Op 25: add_inplace_f32 \(bias_add\) layer=0.*?if \(stop_seq == 25\)',
     'L0_mlp_down', 'A_EMBEDDED_INPUT', '640'),
]


def add_dumps(content):
    """Add dump calls to the model source."""

    # Add dump header after the includes
    marker = '#include "tokenizer/tokenizer.h"'
    if marker in content:
        content = content.replace(marker, marker + '\n' + DUMP_HEADER)
    else:
        # Fallback: add after first #include block
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('#include'):
                insert_idx = i + 1
        lines.insert(insert_idx, DUMP_HEADER)
        content = '\n'.join(lines)

    # Add dump_init() call to ck_model_init / ck_model_init_with_manifest
    if "dump_init();" not in content:
        init_patterns = [
            r'(CK_EXPORT int ck_model_init\([^)]*\) \{)',
            r'(CK_EXPORT int ck_model_init_with_manifest\([^)]*\) \{)',
        ]
        for pat in init_patterns:
            before = content
            content = re.sub(pat, r'\1\n    dump_init();', content, count=1)
            if content != before:
                print(f"  Added dump_init() for pattern: {pat}")
    else:
        print("  dump_init() already present, skipping insert")

    # Add dump calls after specific ops
    for pattern, dump_name, buffer, size in DUMP_POINTS:
        # Find the pattern and add dump after the stop_seq check
        regex = re.compile(pattern, re.DOTALL)
        match = regex.search(content)
        if match:
            end_pos = match.end()
            dump_call = f'\n    dump_tensor("{dump_name}", -1, (float*)(model->bump + {buffer}), {size});'
            content = content[:end_pos] + dump_call + content[end_pos:]
            print(f"  Added dump: {dump_name}")
        else:
            print(f"  Warning: Pattern not found for {dump_name}")

    return content


def main():
    parser = argparse.ArgumentParser(description='Add parity dump instrumentation')
    parser.add_argument('input', type=Path, help='Input model_v7.c file')
    parser.add_argument('--output', '-o', type=Path, help='Output file (default: overwrite input)')
    parser.add_argument('--backup', '-b', action='store_true', help='Create .bak backup')
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1

    print(f"Reading: {args.input}")
    content = args.input.read_text()

    # Check if already instrumented
    if 'g_dump_enabled' in content:
        print("Warning: File already appears to be instrumented")
        print("Remove existing dump code first, or use a fresh model file")
        return 1

    print("Adding dump instrumentation...")
    content = add_dumps(content)

    output_path = args.output or args.input
    if args.backup and args.output is None:
        backup_path = args.input.with_suffix('.c.bak')
        args.input.rename(backup_path)
        print(f"Backup: {backup_path}")

    print(f"Writing: {output_path}")
    output_path.write_text(content)

    print("\nDone! Rebuild with: make clean && make")
    print("Run with: CK_DUMP_ACTIVATIONS=1 ./ck-cli-v7 ...")

    return 0


if __name__ == '__main__':
    sys.exit(main())
