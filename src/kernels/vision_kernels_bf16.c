/**
 * @file vision_kernels_bf16.c
 * @brief Vision kernels for BF16 tensors (im2patch, etc.)
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 */

#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

void im2patch_bf16(const uint16_t *image,
                   uint16_t *patches,
                   int C,
                   int H,
                   int W,
                   int P)
{
    if (!image || !patches || C <= 0 || H <= 0 || W <= 0 || P <= 0) {
        return;
    }

    int num_patches_h = H / P;
    int num_patches_w = W / P;
    int patch_dim = C * P * P;

    for (int ph = 0; ph < num_patches_h; ++ph) {
        for (int pw = 0; pw < num_patches_w; ++pw) {
            int patch_idx = ph * num_patches_w + pw;
            uint16_t *dst_patch = patches + (size_t)patch_idx * (size_t)patch_dim;

            for (int c = 0; c < C; ++c) {
                for (int py = 0; py < P; ++py) {
                    int y = ph * P + py;
                    int x = pw * P;

                    const uint16_t *src_row = image + (size_t)c * (size_t)H * (size_t)W + (size_t)y * (size_t)W + (size_t)x;
                    uint16_t *dst_row = dst_patch + (size_t)c * (size_t)P * (size_t)P + (size_t)py * (size_t)P;

                    memcpy(dst_row, src_row, (size_t)P * sizeof(uint16_t));
                }
            }
        }
    }
}

void patch2im_bf16(const uint16_t *d_patches,
                   uint16_t *d_image,
                   int C,
                   int H,
                   int W,
                   int P)
{
    if (!d_patches || !d_image || C <= 0 || H <= 0 || W <= 0 || P <= 0) {
        return;
    }

    int num_patches_h = H / P;
    int num_patches_w = W / P;
    int patch_dim = C * P * P;

    memset(d_image, 0, (size_t)C * (size_t)H * (size_t)W * sizeof(uint16_t));

    for (int ph = 0; ph < num_patches_h; ++ph) {
        for (int pw = 0; pw < num_patches_w; ++pw) {
            int patch_idx = ph * num_patches_w + pw;
            const uint16_t *src_patch = d_patches + (size_t)patch_idx * (size_t)patch_dim;

            for (int c = 0; c < C; ++c) {
                for (int py = 0; py < P; ++py) {
                    int y = ph * P + py;
                    int x = pw * P;

                    uint16_t *dst_row = d_image + (size_t)c * (size_t)H * (size_t)W + (size_t)y * (size_t)W + (size_t)x;
                    const uint16_t *src_row = src_patch + (size_t)c * (size_t)P * (size_t)P + (size_t)py * (size_t)P;

                    for (int px = 0; px < P; ++px) {
                        float acc = bf16_to_float(dst_row[px]) + bf16_to_float(src_row[px]);
                        dst_row[px] = float_to_bf16(acc);
                    }
                }
            }
        }
    }
}

