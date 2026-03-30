#include "clip.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr char CKDUMP_MAGIC[8] = {'C', 'K', 'D', 'M', 'P', '\0', '\0', '\0'};
constexpr uint32_t CKDUMP_VERSION = 1;

struct CKDumpFileHeader {
    char magic[8];
    uint32_t version;
    int32_t layer_id;
    char op_name[32];
    uint32_t dtype;
    uint32_t rank;
    int64_t shape[4];
    uint32_t elem_count;
    int32_t token_id;
    uint8_t reserved[32];
} __attribute__((packed));

static_assert(sizeof(CKDumpFileHeader) == 128, "CKDumpFileHeader must be 128 bytes");

struct DumpState {
    FILE *file = nullptr;
    FILE *meta_file = nullptr;
    bool dump_all = false;
    bool has_exact_layer = false;
    int32_t exact_layer = -1;
    std::unordered_set<std::string> names;
};

struct CKMtmdClipHandle {
    clip_ctx *ctx = nullptr;
    DumpState dump;
};

static const std::array<const char *, 24> kDefaultDumpNames = {
    "patch_bias",
    "inp_pos_emb",
    "ln1",
    "post_ln",
    "Qcur",
    "Kcur",
    "Vcur",
    "v_attn_perm",
    "Qcur_rope",
    "Kcur_rope",
    "kq_scores",
    "kq_softmax",
    "kqv_raw",
    "kqv_out",
    "attn_out",
    "ffn_inp",
    "ffn_inp_normed",
    "ffn_up_b",
    "ffn_gelu",
    "ffn_out",
    "layer_out",
    "projector_in",
    "projector_out",
    "vision_output",
};

static bool parse_layer_suffix(const std::string &raw_name, std::string &base_name, int32_t &layer_id) {
    layer_id = -1;
    base_name = raw_name;
    const size_t dash = raw_name.rfind('-');
    if (dash == std::string::npos || dash + 1 >= raw_name.size()) {
        return false;
    }
    for (size_t i = dash + 1; i < raw_name.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(raw_name[i]))) {
            return false;
        }
    }
    base_name = raw_name.substr(0, dash);
    layer_id = std::atoi(raw_name.c_str() + dash + 1);
    return true;
}

static float get_float_value(
    const uint8_t *data,
    ggml_type type,
    const size_t *nb,
    size_t i0,
    size_t i1,
    size_t i2,
    size_t i3) {
    const size_t idx = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    switch (type) {
        case GGML_TYPE_F32:
            return *(const float *) (data + idx);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (data + idx));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*(const ggml_bf16_t *) (data + idx));
        case GGML_TYPE_I64:
            return (float) *(const int64_t *) (data + idx);
        case GGML_TYPE_I32:
            return (float) *(const int32_t *) (data + idx);
        case GGML_TYPE_I16:
            return (float) *(const int16_t *) (data + idx);
        case GGML_TYPE_I8:
            return (float) *(const int8_t *) (data + idx);
        default:
            return 0.0f;
    }
}

static std::vector<float> flatten_tensor(const ggml_tensor *t, const uint8_t *raw) {
    std::vector<float> out;
    if (!t || !raw) {
        return out;
    }

    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    if (ne0 <= 0 || ne1 < 0 || ne2 < 0 || ne3 < 0) {
        return out;
    }

    out.reserve((size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3);
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    out.push_back(get_float_value(raw, t->type, t->nb, (size_t) i0, (size_t) i1, (size_t) i2, (size_t) i3));
                }
            }
        }
    }
    return out;
}

static bool should_dump_tensor(const DumpState *state, const ggml_tensor *t, std::string &base_name, int32_t &layer_id) {
    if (!state || !state->file || !t) {
        return false;
    }
    const char *raw_name_c = ggml_get_name(t);
    if (!raw_name_c || !raw_name_c[0]) {
        return false;
    }
    parse_layer_suffix(raw_name_c, base_name, layer_id);
    if (state->has_exact_layer && layer_id >= 0 && layer_id != state->exact_layer) {
        return false;
    }
    if (state->dump_all) {
        return true;
    }
    return state->names.find(base_name) != state->names.end();
}

static void write_ckdump_record(FILE *file, const std::string &op_name, int32_t layer_id, const std::vector<float> &flat) {
    if (!file || flat.empty()) {
        return;
    }

    CKDumpFileHeader header{};
    std::memcpy(header.magic, CKDUMP_MAGIC, sizeof(CKDUMP_MAGIC));
    header.version = CKDUMP_VERSION;
    header.layer_id = layer_id;
    std::strncpy(header.op_name, op_name.c_str(), sizeof(header.op_name) - 1);
    header.op_name[sizeof(header.op_name) - 1] = '\0';
    header.dtype = 0;
    header.rank = 1;
    header.shape[0] = (int64_t) flat.size();
    header.elem_count = (uint32_t) flat.size();
    header.token_id = 0;

    std::fwrite(&header, sizeof(header), 1, file);
    std::fwrite(flat.data(), sizeof(float), flat.size(), file);
    std::fflush(file);
}

static bool meta_dump_enabled() {
    const char *v = std::getenv("CK_LLAMA_PARITY_META");
    return v && v[0] && std::strcmp(v, "0") != 0;
}

static void write_meta_record(FILE *file, const std::string &op_name, int32_t layer_id, const ggml_tensor *t) {
    if (!file || !t) {
        return;
    }
    std::fprintf(
        file,
        "{\"name\":\"%s\",\"layer_id\":%d,\"type\":%d,"
        "\"ne\":[%lld,%lld,%lld,%lld],"
        "\"nb\":[%zu,%zu,%zu,%zu]}\n",
        op_name.c_str(),
        layer_id,
        (int) t->type,
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
        (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3]);
    std::fflush(file);
}

static bool ck_mtmd_dump_eval_callback(struct ggml_tensor *t, bool ask, void *user_data) {
    auto *state = static_cast<DumpState *>(user_data);
    std::string base_name;
    int32_t layer_id = -1;
    if (!should_dump_tensor(state, t, base_name, layer_id)) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (ggml_is_quantized(t->type)) {
        return true;
    }

    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) {
        return true;
    }

    std::vector<uint8_t> raw(nbytes);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(raw.data(), t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    }

    std::vector<float> flat = flatten_tensor(t, raw.data());
    write_ckdump_record(state->file, base_name, layer_id, flat);
    write_meta_record(state->meta_file, base_name, layer_id, t);
    return true;
}

static void dump_state_init(DumpState &state) {
    const char *dir_env = std::getenv("CK_LLAMA_PARITY_DIR");
    if (!dir_env || !dir_env[0]) {
        return;
    }

    std::filesystem::create_directories(dir_env);
    std::filesystem::path dump_path = std::filesystem::path(dir_env) / "dump.bin";
    state.file = std::fopen(dump_path.c_str(), "wb");
    if (!state.file) {
        return;
    }
    if (meta_dump_enabled()) {
        std::filesystem::path meta_path = std::filesystem::path(dir_env) / "meta.jsonl";
        state.meta_file = std::fopen(meta_path.c_str(), "wb");
    }

    const char *layer_env = std::getenv("CK_LLAMA_PARITY_LAYER");
    if (layer_env && layer_env[0]) {
        state.has_exact_layer = true;
        state.exact_layer = std::atoi(layer_env);
    }

    const char *all_env = std::getenv("CK_LLAMA_PARITY_ALL");
    state.dump_all = all_env && std::atoi(all_env) != 0;
    const char *names_env = std::getenv("CK_LLAMA_PARITY_NAMES");
    if (names_env && names_env[0]) {
        std::stringstream ss(names_env);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const size_t first = item.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) {
                continue;
            }
            const size_t last = item.find_last_not_of(" \t\r\n");
            const std::string trimmed = item.substr(first, last - first + 1);
            if (trimmed == "*" || trimmed == "all") {
                state.dump_all = true;
                state.names.clear();
                break;
            }
            state.names.insert(trimmed);
        }
    } else if (!state.dump_all) {
        for (const char *name : kDefaultDumpNames) {
            state.names.insert(name);
        }
    }
}

static void dump_state_close(DumpState &state) {
    if (state.file) {
        std::fclose(state.file);
        state.file = nullptr;
    }
    if (state.meta_file) {
        std::fclose(state.meta_file);
        state.meta_file = nullptr;
    }
    state.has_exact_layer = false;
    state.exact_layer = -1;
    state.names.clear();
    state.dump_all = false;
}

static clip_ctx *unwrap_ctx(void *handle_ptr) {
    auto *handle = static_cast<CKMtmdClipHandle *>(handle_ptr);
    return handle ? handle->ctx : nullptr;
}

}  // namespace

extern "C" {

void *ck_mtmd_clip_init(
    const char *mmproj_path,
    int use_gpu,
    int flash_attn_type,
    int image_min_tokens,
    int image_max_tokens,
    int warmup) {
    auto *handle = new CKMtmdClipHandle();
    dump_state_init(handle->dump);

    clip_context_params params{};
    params.use_gpu = use_gpu != 0;
    params.flash_attn_type = static_cast<clip_flash_attn_type>(flash_attn_type);
    params.image_min_tokens = image_min_tokens;
    params.image_max_tokens = image_max_tokens;
    params.warmup = warmup != 0;
    params.cb_eval = handle->dump.file ? ck_mtmd_dump_eval_callback : nullptr;
    params.cb_eval_user_data = handle->dump.file ? &handle->dump : nullptr;

    clip_init_result result = clip_init(mmproj_path, params);
    handle->ctx = result.ctx_v;
    if (!handle->ctx) {
        dump_state_close(handle->dump);
        delete handle;
        return nullptr;
    }
    return handle;
}

void ck_mtmd_clip_free(void *handle_ptr) {
    auto *handle = static_cast<CKMtmdClipHandle *>(handle_ptr);
    if (!handle) {
        return;
    }
    clip_free(handle->ctx);
    dump_state_close(handle->dump);
    delete handle;
}

int ck_mtmd_clip_n_mmproj_embd(void *handle_ptr) {
    return clip_n_mmproj_embd(unwrap_ctx(handle_ptr));
}

size_t ck_mtmd_clip_embd_nbytes_by_img(void *handle_ptr, int img_w, int img_h) {
    return clip_embd_nbytes_by_img(unwrap_ctx(handle_ptr), img_w, img_h);
}

int ck_mtmd_clip_encode_float_image(void *handle_ptr, int n_threads, float *img, int h, int w, float *vec) {
    return clip_encode_float_image(unwrap_ctx(handle_ptr), n_threads, img, h, w, vec) ? 1 : 0;
}

}
