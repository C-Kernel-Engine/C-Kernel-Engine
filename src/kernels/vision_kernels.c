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
#include <math.h>

static int tile_order_index_2d(int linear_idx,
                               int grid_h,
                               int grid_w,
                               int merge_size)
{
    if (linear_idx < 0 || grid_h <= 0 || grid_w <= 0 || merge_size <= 0) {
        return 0;
    }

    const int tiles_w = (grid_w + merge_size - 1) / merge_size;
    const int tile_area = merge_size * merge_size;
    const int tile_idx = linear_idx / tile_area;
    const int intra_idx = linear_idx % tile_area;
    const int tile_y = tile_idx / tiles_w;
    const int tile_x = tile_idx % tiles_w;
    const int dy = intra_idx / merge_size;
    const int dx = intra_idx % merge_size;
    const int y = tile_y * merge_size + dy;
    const int x = tile_x * merge_size + dx;

    return y * grid_w + x;
}


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

/**
 * Add learned absolute position embeddings in-place.
 *
 * x layout:        [num_tokens, embed_dim]
 * position_embd:   [num_positions, embed_dim]
 *
 * This first v8 vision path intentionally assumes native resized embeddings are
 * already materialized in the weight tensor, so token i maps directly to
 * position_embd[i]. This is the correct contract for fixed-size bring-up.
 */
void position_embeddings_add(float *x,
                             const float *position_embd,
                             int num_tokens,
                             int embed_dim,
                             int num_positions)
{
    if (x == NULL || position_embd == NULL || num_tokens <= 0 || embed_dim <= 0) {
        return;
    }
    const int limit = num_tokens < num_positions ? num_tokens : num_positions;
    for (int tok = 0; tok < limit; ++tok) {
        float *dst = x + (size_t)tok * embed_dim;
        const float *src = position_embd + (size_t)tok * embed_dim;
        for (int d = 0; d < embed_dim; ++d) {
            dst[d] += src[d];
        }
    }
}

void position_embeddings_add_tiled_2d(float *x,
                                      const float *position_embd,
                                      int grid_h,
                                      int grid_w,
                                      int embed_dim,
                                      int merge_size,
                                      int source_grid_size)
{
    if (x == NULL || position_embd == NULL || grid_h <= 0 || grid_w <= 0 || embed_dim <= 0 || merge_size <= 0) {
        return;
    }

    if (source_grid_size <= 0) {
        source_grid_size = grid_h == grid_w ? grid_h : (grid_h > grid_w ? grid_h : grid_w);
    }

    const int num_tokens = grid_h * grid_w;
    const int source_tokens = source_grid_size * source_grid_size;
    const int needs_resize = source_grid_size != grid_h || source_grid_size != grid_w;

    const float sf_x = needs_resize ? (float) grid_w / (float) source_grid_size : 1.0f;
    const float sf_y = needs_resize ? (float) grid_h / (float) source_grid_size : 1.0f;
    const float pixel_offset = 0.5f;
    const float support_x = needs_resize ? fmaxf(1.0f, 1.0f / sf_x) : 1.0f;
    const float support_y = needs_resize ? fmaxf(1.0f, 1.0f / sf_y) : 1.0f;
    const float invscale_x = needs_resize ? 1.0f / support_x : 1.0f;
    const float invscale_y = needs_resize ? 1.0f / support_y : 1.0f;

    for (int tok = 0; tok < num_tokens; ++tok) {
        const int src_tok = tile_order_index_2d(tok, grid_h, grid_w, merge_size);
        const int dst_y = src_tok / grid_w;
        const int dst_x = src_tok % grid_w;
        float *dst = x + (size_t) tok * (size_t) embed_dim;

        if (!needs_resize) {
            const int flat = dst_y * source_grid_size + dst_x;
            if (flat < 0 || flat >= source_tokens) {
                continue;
            }
            const float *src = position_embd + (size_t) flat * (size_t) embed_dim;
            for (int d = 0; d < embed_dim; ++d) {
                dst[d] += src[d];
            }
            continue;
        }

        const float x_src = ((float) dst_x + pixel_offset) / sf_x;
        const float y_src = ((float) dst_y + pixel_offset) / sf_y;
        int x_min = (int) (x_src - support_x + pixel_offset);
        int x_max = (int) (x_src + support_x + pixel_offset);
        int y_min = (int) (y_src - support_y + pixel_offset);
        int y_max = (int) (y_src + support_y + pixel_offset);
        if (x_min < 0) x_min = 0;
        if (y_min < 0) y_min = 0;
        if (x_max > source_grid_size) x_max = source_grid_size;
        if (y_max > source_grid_size) y_max = source_grid_size;

        for (int d = 0; d < embed_dim; ++d) {
            float val = 0.0f;
            float total_weight = 0.0f;
            for (int sy = y_min; sy < y_max; ++sy) {
                const float wy = fmaxf(1.0f - fabsf(((float) sy - y_src + pixel_offset) * invscale_y), 0.0f);
                if (wy <= 0.0f) {
                    continue;
                }
                for (int sx = x_min; sx < x_max; ++sx) {
                    const float wx = fmaxf(1.0f - fabsf(((float) sx - x_src + pixel_offset) * invscale_x), 0.0f);
                    const float weight = wx * wy;
                    if (weight <= 0.0f) {
                        continue;
                    }
                    const float sample = position_embd[((size_t) sy * (size_t) source_grid_size + (size_t) sx) * (size_t) embed_dim + (size_t) d];
                    val += sample * weight;
                    total_weight += weight;
                }
            }
            if (total_weight > 0.0f) {
                dst[d] += val / total_weight;
            }
        }
    }
}

/**
 * Build merged 2D vision position IDs in the layout expected by vision M-RoPE.
 *
 * Output layout: [4, grid_h * grid_w] flattened as
 *   [y_stream | x_stream | y_stream_dup | x_stream_dup]
 *
 * Tokens are emitted in merged-tile traversal order so the position buffer
 * matches the same 2x2 grouping used by Qwen-style vision encoders.
 */
void vision_position_ids_2d_merge(int32_t *positions,
                                  int grid_h,
                                  int grid_w,
                                  int merge_size)
{
    if (!positions || grid_h <= 0 || grid_w <= 0 || merge_size <= 0) {
        return;
    }

    const int num_tokens = grid_h * grid_w;
    int ptr = 0;

    for (int y = 0; y < grid_h; y += merge_size) {
        for (int x = 0; x < grid_w; x += merge_size) {
            for (int dy = 0; dy < merge_size; ++dy) {
                for (int dx = 0; dx < merge_size; ++dx) {
                    const int yy = y + dy;
                    const int xx = x + dx;
                    if (yy >= grid_h || xx >= grid_w || ptr >= num_tokens) {
                        continue;
                    }
                    positions[ptr] = yy;
                    positions[num_tokens + ptr] = xx;
                    positions[2 * num_tokens + ptr] = yy;
                    positions[3 * num_tokens + ptr] = xx;
                    ++ptr;
                }
            }
        }
    }

    for (; ptr < num_tokens; ++ptr) {
        positions[ptr] = 0;
        positions[num_tokens + ptr] = 0;
        positions[2 * num_tokens + ptr] = 0;
        positions[3 * num_tokens + ptr] = 0;
    }
}

/**
 * Merge 2x2 neighboring tokens into a single wider token.
 *
 * Input layout:   [grid_h * grid_w, embed_dim]
 * Output layout:  [(grid_h/2) * (grid_w/2), embed_dim * 4]
 *
 * Pack order within each merged token:
 *   top-left, top-right, bottom-left, bottom-right
 */
void spatial_merge_2x2(const float *input,
                      float *output,
                      int grid_h,
                      int grid_w,
                      int embed_dim)
{
    if (input == NULL || output == NULL || grid_h <= 0 || grid_w <= 0 || embed_dim <= 0) {
        return;
    }

    const int merged_h = grid_h / 2;
    const int merged_w = grid_w / 2;
    const size_t token_stride = (size_t)embed_dim;
    const size_t merged_stride = (size_t)embed_dim * 4;

    for (int mh = 0; mh < merged_h; ++mh) {
        for (int mw = 0; mw < merged_w; ++mw) {
            const int y0 = mh * 2;
            const int x0 = mw * 2;
            const int in_idx00 = y0 * grid_w + x0;
            const int in_idx01 = in_idx00 + 1;
            const int in_idx10 = in_idx00 + grid_w;
            const int in_idx11 = in_idx10 + 1;
            const int out_idx = mh * merged_w + mw;

            const float *src00 = input + (size_t)in_idx00 * token_stride;
            const float *src01 = input + (size_t)in_idx01 * token_stride;
            const float *src10 = input + (size_t)in_idx10 * token_stride;
            const float *src11 = input + (size_t)in_idx11 * token_stride;
            float *dst = output + (size_t)out_idx * merged_stride;

            memcpy(dst + 0 * embed_dim, src00, (size_t)embed_dim * sizeof(float));
            memcpy(dst + 1 * embed_dim, src01, (size_t)embed_dim * sizeof(float));
            memcpy(dst + 2 * embed_dim, src10, (size_t)embed_dim * sizeof(float));
            memcpy(dst + 3 * embed_dim, src11, (size_t)embed_dim * sizeof(float));
        }
    }
}

void spatial_merge_contiguous_tiled(const float *input,
                                    float *output,
                                    int grid_h,
                                    int grid_w,
                                    int embed_dim,
                                    int merge_size)
{
    if (input == NULL || output == NULL || grid_h <= 0 || grid_w <= 0 || embed_dim <= 0 || merge_size <= 0) {
        return;
    }

    const size_t num_tokens = (size_t) grid_h * (size_t) grid_w;
    const size_t merge_factor = (size_t) merge_size * (size_t) merge_size;
    const size_t merged_tokens = num_tokens / merge_factor;
    memcpy(output, input, merged_tokens * (size_t) embed_dim * merge_factor * sizeof(float));
}

void rowwise_bias_add(float *x,
                      const float *bias,
                      int rows,
                      int dim)
{
    if (!x || !bias || rows <= 0 || dim <= 0) {
        return;
    }

    for (int r = 0; r < rows; ++r) {
        float *row = x + ((size_t) r * (size_t) dim);
        for (int c = 0; c < dim; ++c) {
            row[c] += bias[c];
        }
    }
}

void add_stream_inplace(float *a,
                        const float *b,
                        size_t n)
{
    if (!a || !b || n == 0) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        a[i] += b[i];
    }
}

void add_stream_reorder_2d(float *main_inout,
                           float *aux_scratch,
                           int grid_h,
                           int grid_w,
                           int embed_dim,
                           int merge_size)
{
    if (!main_inout || !aux_scratch || grid_h <= 0 || grid_w <= 0 || embed_dim <= 0 || merge_size <= 0) {
        return;
    }

    const int num_tokens = grid_h * grid_w;
    const size_t total_elems = (size_t) num_tokens * (size_t) embed_dim;

    for (size_t i = 0; i < total_elems; ++i) {
        main_inout[i] += aux_scratch[i];
    }

    for (int tok = 0; tok < num_tokens; ++tok) {
        const int src_tok = tile_order_index_2d(tok, grid_h, grid_w, merge_size);
        const float *src_main = main_inout + (size_t) src_tok * (size_t) embed_dim;
        float *dst = aux_scratch + (size_t) tok * (size_t) embed_dim;
        for (int d = 0; d < embed_dim; ++d) {
            dst[d] = src_main[d];
        }
    }

    memcpy(main_inout, aux_scratch, total_elems * sizeof(float));
}

void feature_concat_2way(const float *main_input,
                         const float *branch_input,
                         float *output,
                         int rows,
                         int main_dim,
                         int branch_slice_dim,
                         int num_branch_slices)
{
    if (!main_input || !branch_input || !output || rows <= 0 || main_dim <= 0 ||
        branch_slice_dim < 0 || num_branch_slices < 0) {
        return;
    }

    const int branch_dim = branch_slice_dim * num_branch_slices;
    const size_t main_bytes = (size_t) main_dim * sizeof(float);
    const size_t branch_bytes = (size_t) branch_dim * sizeof(float);
    const size_t out_stride = (size_t) (main_dim + branch_dim);

    for (int r = 0; r < rows; ++r) {
        const float *src_main = main_input + ((size_t) r * (size_t) main_dim);
        const float *src_branch = branch_input + ((size_t) r * (size_t) branch_dim);
        float *dst = output + ((size_t) r * out_stride);
        memcpy(dst, src_main, main_bytes);
        memcpy(dst + main_dim, src_branch, branch_bytes);
    }
}

void feature_slice_copy(const float *src,
                        float *dst,
                        int rows,
                        int src_dim,
                        int dst_dim,
                        int dst_feature_offset)
{
    if (!src || !dst || rows <= 0 || src_dim <= 0 || dst_dim <= 0 || dst_feature_offset < 0) {
        return;
    }
    if (dst_feature_offset + src_dim > dst_dim) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        const float *src_row = src + (size_t) row * (size_t) src_dim;
        float *dst_row = dst + (size_t) row * (size_t) dst_dim + (size_t) dst_feature_offset;
        memcpy(dst_row, src_row, (size_t) src_dim * sizeof(float));
    }
}

void feature_concat(const float *main_input,
                    const float *branch_input,
                    float *output,
                    int rows,
                    int main_dim,
                    int branch_slice_dim,
                    int num_branch_slices)
{
    if (!main_input || !output || rows <= 0 || main_dim < 0 || branch_slice_dim < 0 || num_branch_slices < 0) {
        return;
    }

    const int branch_total_dim = branch_slice_dim * num_branch_slices;
    const int out_dim = main_dim + branch_total_dim;

    for (int row = 0; row < rows; ++row) {
        const float *src_main = main_input + (size_t) row * (size_t) main_dim;
        float *dst_row = output + (size_t) row * (size_t) out_dim;

        if (main_dim > 0) {
            memcpy(dst_row, src_main, (size_t) main_dim * sizeof(float));
        }

        for (int slice = 0; slice < num_branch_slices; ++slice) {
            const float *src_branch = branch_input
                + (size_t) slice * (size_t) rows * (size_t) branch_slice_dim
                + (size_t) row * (size_t) branch_slice_dim;
            float *dst_branch = dst_row + (size_t) main_dim + (size_t) slice * (size_t) branch_slice_dim;
            if (branch_slice_dim > 0) {
                memcpy(dst_branch, src_branch, (size_t) branch_slice_dim * sizeof(float));
            }
        }
    }
}
