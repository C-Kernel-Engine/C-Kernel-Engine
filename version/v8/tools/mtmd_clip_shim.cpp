#include "clip.h"

extern "C" {

void * ck_mtmd_clip_init(
    const char * mmproj_path,
    int use_gpu,
    int flash_attn_type,
    int image_min_tokens,
    int image_max_tokens,
    int warmup) {
    clip_context_params params{};
    params.use_gpu = use_gpu != 0;
    params.flash_attn_type = static_cast<clip_flash_attn_type>(flash_attn_type);
    params.image_min_tokens = image_min_tokens;
    params.image_max_tokens = image_max_tokens;
    params.warmup = warmup != 0;
    params.cb_eval = nullptr;
    params.cb_eval_user_data = nullptr;

    clip_init_result result = clip_init(mmproj_path, params);
    return result.ctx_v;
}

void ck_mtmd_clip_free(void * ctx) {
    clip_free(static_cast<clip_ctx *>(ctx));
}

int ck_mtmd_clip_n_mmproj_embd(void * ctx) {
    return clip_n_mmproj_embd(static_cast<clip_ctx *>(ctx));
}

size_t ck_mtmd_clip_embd_nbytes_by_img(void * ctx, int img_w, int img_h) {
    return clip_embd_nbytes_by_img(static_cast<clip_ctx *>(ctx), img_w, img_h);
}

int ck_mtmd_clip_encode_float_image(void * ctx, int n_threads, float * img, int h, int w, float * vec) {
    return clip_encode_float_image(static_cast<clip_ctx *>(ctx), n_threads, img, h, w, vec) ? 1 : 0;
}

}
