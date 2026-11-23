// Diagnostic only. Use a CPU-only llama.cpp build and stop at the selected layer.
// The other model layers are outside the captured result.
#include "llama.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct capture {
    std::vector<float> rows;
    int expected;
    std::string prefix;
    int layer;
};

static bool capture_layer(ggml_tensor *tensor, bool ask, void *data) {
    const std::string name = tensor->name;
    auto &out = *static_cast<capture *>(data);
    if (out.layer == 6) {
        if (name != "l_out-5" && name != "l_out-6") return !ask;
        if (ask) return true;
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
            ggml_nelements(tensor) != out.expected * 2304) std::abort();
        std::vector<float> rows(static_cast<size_t>(out.expected) * 2304);
        ggml_backend_tensor_get(tensor, rows.data(), 0, ggml_nbytes(tensor));
        if (name == "l_out-6") {
            out.rows.swap(rows);
            return false;
        }
        std::ofstream file(out.prefix + ".l_out-5", std::ios::binary);
        file.write(reinterpret_cast<const char *>(rows.data()), ggml_nbytes(tensor));
        if (!file) std::abort();
        return true;
    }
    const bool final = name == "l_out-0";
    const bool routes = name == "ffn_moe_topk-0";
    const bool logits = name == "ffn_moe_logits-0";
    const bool q = name == "Qcur-0";
    const std::string weight = tensor->op == GGML_OP_MUL_MAT && tensor->src[0] ? tensor->src[0]->name : "";
    const bool q_dot = weight == "blk.0.attn_q.weight";
    const bool ffn_dot = weight == "blk.0.ffn_gate_inp.weight";
    const bool o_dot = weight == "blk.0.attn_output.weight";
    const bool layer_stage = name == "ffn_inp-0" || name == "ffn_moe_out-0" || name == "ffn_out-0";
    if (!final && !routes && !logits && !q && !q_dot && !ffn_dot && !o_dot && !layer_stage) return !ask;
    if (ask) return true;
    if (q_dot || ffn_dot || o_dot) {
        auto *operand = tensor->src[1];
        const int width = o_dot ? 3072 : 2304;
        if (operand->type != GGML_TYPE_F32 || !ggml_is_contiguous(operand) ||
            ggml_nelements(operand) != out.expected * width) std::abort();
        std::vector<float> input(static_cast<size_t>(out.expected) * width);
        ggml_backend_tensor_get(operand, input.data(), 0, ggml_nbytes(operand));
        const char *suffix = q_dot ? ".q-input.f32" : o_dot ? ".o-input.f32" : ".ffn-input.f32";
        std::ofstream file(out.prefix + suffix, std::ios::binary);
        file.write(reinterpret_cast<const char *>(input.data()), ggml_nbytes(operand));
        if (!file) std::abort();
    }
    if (routes || logits || q || q_dot || o_dot || layer_stage) {
        const int64_t width = routes ? 8 : logits ? 128 : (layer_stage || o_dot) ? 2304 : 3072;
        if (tensor->type != (routes ? GGML_TYPE_I32 : GGML_TYPE_F32) ||
            ggml_nelements(tensor) != out.expected * width) std::abort();
        std::ofstream file(out.prefix + "." + (q_dot ? "q-dot.f32" : o_dot ? "o-dot.f32" : name), std::ios::binary);
        if (routes) {
            std::vector<int32_t> row(width);
            for (int i = 0; i < out.expected; ++i) {
                ggml_backend_tensor_get(tensor, row.data(), i * tensor->nb[1], row.size() * sizeof(int32_t));
                file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(int32_t));
            }
        } else {
            if (!ggml_is_contiguous(tensor)) std::abort();
            std::vector<float> values(static_cast<size_t>(out.expected) * width);
            ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
            file.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(ggml_nbytes(tensor)));
        }
        if (!file) std::abort();
        return true;
    }
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
        ggml_nelements(tensor) != out.expected * 2304) {
        std::cerr << "unexpected layer0 output: " << tensor->name << '\n';
        std::abort();
    }
    out.rows.resize(static_cast<size_t>(out.expected) * 2304);
    ggml_backend_tensor_get(tensor, out.rows.data(), 0, ggml_nbytes(tensor));
    return false;
}

static bool piece(llama_context *ctx, capture &state, float *hidden, int pos, int count,
                  const std::string &output) {
    state.expected = count;
    state.prefix = output;
    state.rows.clear();
    llama_seq_id id = 0;
    std::vector<llama_pos> positions(count);
    std::vector<int32_t> num_seq(count, 1);
    std::vector<llama_seq_id *> sequence(count, &id);
    std::vector<int8_t> logits(count, 1);
    for (int i = 0; i < count; ++i) positions[i] = pos + i;
    llama_batch batch{};
    batch.n_tokens = count;
    batch.embd = hidden + pos * 2304;
    batch.pos = positions.data();
    batch.n_seq_id = num_seq.data();
    batch.seq_id = sequence.data();
    batch.logits = logits.data();
    const int status = llama_decode(ctx, batch);
    if (status || state.rows.size() != static_cast<size_t>(count) * 2304) {
        std::cerr << "layer " << state.layer << " capture failed " << status << '\n';
        return false;
    }
    std::ofstream file(output, std::ios::binary);
    file.write(reinterpret_cast<const char *>(state.rows.data()), static_cast<std::streamsize>(state.rows.size() * sizeof(float)));
    return static_cast<bool>(file);
}

int main(int argc, char **argv) {
    if (argc != 4 && (argc != 5 || std::string(argv[4]) != "6")) return 2;
    const int layer = argc == 5 ? 6 : 0;
    const std::string prefix = argv[3];
    std::vector<float> hidden(5 * 2304);
    std::ifstream input(argv[2], std::ios::binary);
    input.read(reinterpret_cast<char *>(hidden.data()), static_cast<std::streamsize>(hidden.size() * sizeof(float)));
    if (!input || input.peek() != EOF) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.use_extra_bufts = false;
    using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
    const model_ptr model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) return 1;
    for (bool split : {false, true}) {
        capture data{};
        data.layer = layer;
        auto cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.n_batch = 16;
        cp.n_ubatch = 16;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.cb_eval = capture_layer;
        cp.cb_eval_user_data = &data;
        using ctx_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
        const ctx_ptr ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) return 1;
        if (split) {
            if (!piece(ctx.get(), data, hidden.data(), 0, 4, prefix + ".prefill.f32") ||
                !piece(ctx.get(), data, hidden.data(), 4, 1, prefix + ".append.f32")) return 1;
        } else if (!piece(ctx.get(), data, hidden.data(), 0, 5, prefix + ".full.f32")) return 1;
    }
    std::ofstream proof(prefix + ".input.f32", std::ios::binary);
    proof.write(reinterpret_cast<const char *>(hidden.data()),
                static_cast<std::streamsize>(hidden.size() * sizeof(float)));
    return proof ? 0 : 1;
}
