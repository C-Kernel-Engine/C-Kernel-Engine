/**
 * @file vision_kernels.c
 * @brief Vision kernels (im2patch, patch embedding, etc.)
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

/**
 * im2patch: Transforms an image into a sequence of flattened patches.
 * 
 * Image Layout: [C, H, W] (Row-major: W is fastest moving)
 * Output Layout: [num_patches, C * P * P]
 * 
 * num_patches = (H/P) * (W/P)
 * P = patch_size
 */
void im2patch(const float *image, 
              float *patches, 
              int C, int H, int W, int P) 
{
    int num_patches_h = H / P;
    int num_patches_w = W / P;
    int patch_dim = C * P * P;

    // ph, pw: patch grid coordinates
    for (int ph = 0; ph < num_patches_h; ++ph) {
        for (int pw = 0; pw < num_patches_w; ++pw) {
            
            int patch_idx = ph * num_patches_w + pw;
            float *dst_patch = patches + (size_t)patch_idx * patch_dim;

            // For each patch, grab pixels from all channels
            for (int c = 0; c < C; ++c) {
                for (int py = 0; py < P; ++py) {
                    int y = ph * P + py;
                    int x = pw * P;
                    
                    // Input row start in the image
                    const float *src_row = image + (size_t)c * H * W + (size_t)y * W + x;
                    
                    // Destination row in the flattened patch sequence
                    float *dst_row = dst_patch + (size_t)c * P * P + (size_t)py * P;
                    
                    // Copy P pixels (one row of the patch)
                    memcpy(dst_row, src_row, P * sizeof(float));
                }
            }
        }
    }
}

/**
 * patch2im: Accumulates gradients from patches back into the image. (Backward pass)
 * 
 * d_patches: [num_patches, C * P * P]
 * d_image: [C, H, W] (Accumulated)
 */
void patch2im(const float *d_patches, 
              float *d_image, 
              int C, int H, int W, int P) 
{
    int num_patches_h = H / P;
    int num_patches_w = W / P;
    int patch_dim = C * P * P;

    // Zero out the image first as we are accumulating gradients
    memset(d_image, 0, (size_t)C * H * W * sizeof(float));

    for (int ph = 0; ph < num_patches_h; ++ph) {
        for (int pw = 0; pw < num_patches_w; ++pw) {
            
            int patch_idx = ph * num_patches_w + pw;
            const float *src_patch = d_patches + (size_t)patch_idx * patch_dim;

            for (int c = 0; c < C; ++c) {
                for (int py = 0; py < P; ++py) {
                    int y = ph * P + py;
                    int x = pw * P;
                    
                    float *dst_row = d_image + (size_t)c * H * W + (size_t)y * W + x;
                    const float *src_row = src_patch + (size_t)c * P * P + (size_t)py * P;
                    
                    // Add the patch gradient to the image gradient
                    for (int px = 0; px < P; ++px) {
                        dst_row[px] += src_row[px];
                    }
                }
            }
        }
    }
}
