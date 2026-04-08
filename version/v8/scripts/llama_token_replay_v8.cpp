#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Args {
    std::string model_path;
    std::vector<int32_t> tokens;
    std::vector<int32_t> tokens_before;
    std::vector<int32_t> tokens_after;
    std::string prompt;
    std::string prefix_f32_path;
    std::string logits_out_path;
    std::string dump_dir;
    std::vector<std::string> dump_names;
    std::string decode_mode = "batched";
    std::string prefix_decode_mode = "batched";
    int ctx_len = 256;
    int top_k = 16;
    int threads = 0;
    int prefix_grid_x = 0;
    int prefix_grid_y = 0;
    int prefix_row_dim = 0;
    int prefix_text_pos = -1;
};

struct DumpState {
    std::filesystem::path dump_dir;
    std::filesystem::path index_path;
    std::unordered_set<std::string> names;
    bool dump_all = false;
    int dumped = 0;
    int32_t current_token_id = 0;
    std::unordered_map<std::string, int> occurrences;
};

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            continue;
        }
        cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static bool parse_tokens(const std::string & csv, std::vector<int32_t> & out_tokens, std::string & err) {
    out_tokens.clear();
    for (const std::string & p : split_csv(csv)) {
        try {
            size_t idx = 0;
            long long v = std::stoll(p, &idx, 10);
            if (idx != p.size()) {
                err = "invalid token id: " + p;
                return false;
            }
            if (v < INT32_MIN || v > INT32_MAX) {
                err = "token id out of int32 range: " + p;
                return false;
            }
            out_tokens.push_back(static_cast<int32_t>(v));
        } catch (...) {
            err = "invalid token id: " + p;
            return false;
        }
    }
    if (out_tokens.empty()) {
        err = "token list is empty";
        return false;
    }
    return true;
}

static bool parse_args(int argc, char ** argv, Args & args, std::string & err) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                err = std::string("missing value for ") + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "--model") {
            const char * v = need_value("--model");
            if (!v) return false;
            args.model_path = v;
            continue;
        }
        if (a == "--tokens") {
            const char * v = need_value("--tokens");
            if (!v) return false;
            if (!parse_tokens(v, args.tokens, err)) return false;
            continue;
        }
        if (a == "--tokens-before") {
            const char * v = need_value("--tokens-before");
            if (!v) return false;
            if (!parse_tokens(v, args.tokens_before, err)) return false;
            continue;
        }
        if (a == "--tokens-after") {
            const char * v = need_value("--tokens-after");
            if (!v) return false;
            if (!parse_tokens(v, args.tokens_after, err)) return false;
            continue;
        }
        if (a == "--prompt") {
            const char * v = need_value("--prompt");
            if (!v) return false;
            args.prompt = v;
            continue;
        }
        if (a == "--prefix-f32") {
            const char * v = need_value("--prefix-f32");
            if (!v) return false;
            args.prefix_f32_path = v;
            continue;
        }
        if (a == "--prefix-grid-x") {
            const char * v = need_value("--prefix-grid-x");
            if (!v) return false;
            args.prefix_grid_x = std::max(0, std::atoi(v));
            continue;
        }
        if (a == "--prefix-grid-y") {
            const char * v = need_value("--prefix-grid-y");
            if (!v) return false;
            args.prefix_grid_y = std::max(0, std::atoi(v));
            continue;
        }
        if (a == "--prefix-row-dim") {
            const char * v = need_value("--prefix-row-dim");
            if (!v) return false;
            args.prefix_row_dim = std::max(0, std::atoi(v));
            continue;
        }
        if (a == "--prefix-text-pos") {
            const char * v = need_value("--prefix-text-pos");
            if (!v) return false;
            args.prefix_text_pos = std::atoi(v);
            continue;
        }
        if (a == "--ctx") {
            const char * v = need_value("--ctx");
            if (!v) return false;
            args.ctx_len = std::max(8, std::atoi(v));
            continue;
        }
        if (a == "--top-k") {
            const char * v = need_value("--top-k");
            if (!v) return false;
            args.top_k = std::max(1, std::atoi(v));
            continue;
        }
        if (a == "--threads") {
            const char * v = need_value("--threads");
            if (!v) return false;
            args.threads = std::max(0, std::atoi(v));
            continue;
        }
        if (a == "--logits-out") {
            const char * v = need_value("--logits-out");
            if (!v) return false;
            args.logits_out_path = v;
            continue;
        }
        if (a == "--dump-dir") {
            const char * v = need_value("--dump-dir");
            if (!v) return false;
            args.dump_dir = v;
            continue;
        }
        if (a == "--dump-names") {
            const char * v = need_value("--dump-names");
            if (!v) return false;
            args.dump_names = split_csv(v);
            continue;
        }
        if (a == "--decode-mode") {
            const char * v = need_value("--decode-mode");
            if (!v) return false;
            args.decode_mode = v;
            if (args.decode_mode != "batched" && args.decode_mode != "sequential") {
                err = "invalid --decode-mode (expected batched or sequential)";
                return false;
            }
            continue;
        }
        if (a == "--prefix-decode-mode") {
            const char * v = need_value("--prefix-decode-mode");
            if (!v) return false;
            args.prefix_decode_mode = v;
            if (args.prefix_decode_mode != "batched" && args.prefix_decode_mode != "sequential") {
                err = "invalid --prefix-decode-mode (expected batched or sequential)";
                return false;
            }
            continue;
        }
        if (a == "-h" || a == "--help") {
            std::cout
                << "Usage: llama_token_replay --model <path.gguf> "
                << "(--tokens <id,id,...> | --prompt <text> | [--tokens-before <id,id,...>] [--tokens-after <id,id,...>]) "
                << "--logits-out <path.bin> [--prefix-f32 <path.f32>] "
                << "[--prefix-grid-x N --prefix-grid-y N] [--prefix-row-dim N] [--prefix-text-pos N] [--ctx N] [--top-k K] [--threads N] "
                << "[--decode-mode batched|sequential] [--prefix-decode-mode batched|sequential] "
                << "[--dump-dir dir --dump-names a,b,c]\n";
            std::exit(0);
        }
        err = "unknown arg: " + a;
        return false;
    }

    if (args.model_path.empty()) {
        err = "missing --model";
        return false;
    }
    const bool segmented = !args.tokens_before.empty() || !args.tokens_after.empty();
    if (segmented) {
        if (!args.tokens.empty() || !args.prompt.empty()) {
            err = "use either --tokens/--prompt or --tokens-before/--tokens-after";
            return false;
        }
    } else if (args.tokens.empty() && args.prompt.empty()) {
        if (args.prefix_f32_path.empty()) {
            err = "pass exactly one of --tokens or --prompt";
            return false;
        }
    } else if (!args.tokens.empty() && !args.prompt.empty()) {
        err = "pass exactly one of --tokens or --prompt";
        return false;
    }
    if ((args.prefix_grid_x > 0) != (args.prefix_grid_y > 0)) {
        err = "pass both --prefix-grid-x and --prefix-grid-y together";
        return false;
    }
    if (args.prefix_grid_x < 0 || args.prefix_grid_y < 0) {
        err = "prefix grid dimensions must be non-negative";
        return false;
    }
    if (args.prefix_text_pos < -1) {
        err = "prefix text position must be >= -1";
        return false;
    }
    return true;
}

static bool load_prefix_embeddings(
    const std::string & path,
    int32_t n_embd,
    int32_t n_embd_inp,
    int32_t forced_row_dim,
    std::vector<float> & out_embd,
    int32_t & out_tokens,
    std::string & err
) {
    out_embd.clear();
    out_tokens = 0;
    if (path.empty()) {
        return true;
    }
    if (n_embd <= 0 || n_embd_inp <= 0) {
        err = "invalid embedding size for prefix embeddings";
        return false;
    }

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "failed opening prefix-f32 file";
        return false;
    }
    const std::streamsize nbytes = f.tellg();
    if (nbytes <= 0 || (nbytes % static_cast<std::streamsize>(sizeof(float))) != 0) {
        err = "prefix-f32 file size must be a positive multiple of 4 bytes";
        return false;
    }
    const size_t n_floats = static_cast<size_t>(nbytes / static_cast<std::streamsize>(sizeof(float)));
    int32_t row_dim = 0;
    if (forced_row_dim > 0) {
        if (forced_row_dim != n_embd && forced_row_dim != n_embd_inp) {
            err = "prefix-row-dim must match model embed_dim or input embed_dim";
            return false;
        }
        if (n_floats % static_cast<size_t>(forced_row_dim) != 0) {
            err = "prefix-f32 row count does not match forced prefix-row-dim";
            return false;
        }
        row_dim = forced_row_dim;
    } else if (n_floats % static_cast<size_t>(n_embd_inp) == 0) {
        row_dim = n_embd_inp;
    } else if (n_floats % static_cast<size_t>(n_embd) == 0) {
        row_dim = n_embd;
    } else {
        err = "prefix-f32 row count does not match model input embedding size";
        return false;
    }

    std::vector<float> raw(n_floats);
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char *>(raw.data()), nbytes);
    if (!f.good()) {
        err = "failed reading prefix-f32 file";
        return false;
    }

    out_tokens = static_cast<int32_t>(n_floats / static_cast<size_t>(row_dim));
    if (out_tokens <= 0) {
        err = "prefix-f32 file contains zero tokens";
        return false;
    }

    if (row_dim == n_embd_inp) {
        out_embd.swap(raw);
        return true;
    }

    out_embd.assign(static_cast<size_t>(out_tokens) * static_cast<size_t>(n_embd_inp), 0.0f);
    for (int32_t tok = 0; tok < out_tokens; ++tok) {
        std::memcpy(
            out_embd.data() + static_cast<size_t>(tok) * static_cast<size_t>(n_embd_inp),
            raw.data() + static_cast<size_t>(tok) * static_cast<size_t>(row_dim),
            static_cast<size_t>(n_embd) * sizeof(float)
        );
    }
    return true;
}

static void print_json_error(const std::string & msg) {
    std::cout << "{\"ok\":false,\"error\":\"";
    for (char c : msg) {
        if (c == '"' || c == '\\') {
            std::cout << '\\' << c;
        } else if (c == '\n') {
            std::cout << "\\n";
        } else {
            std::cout << c;
        }
    }
    std::cout << "\"}\n";
}

static bool should_dump_tensor(const DumpState * state, const ggml_tensor * t) {
    if (!state || state->dump_dir.empty() || !t) {
        return false;
    }
    const char * raw_name = ggml_get_name(t);
    if (!raw_name || !raw_name[0]) {
        return false;
    }
    if (state->dump_all) {
        return true;
    }
    if (state->names.empty()) {
        return false;
    }
    return state->names.find(raw_name) != state->names.end();
}

static std::string json_escape(const std::string & s) {
    std::ostringstream out;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out << '\\' << c;
        } else if (c == '\n') {
            out << "\\n";
        } else {
            out << c;
        }
    }
    return out.str();
}

static void begin_dump_batch(DumpState * state, int32_t token_id) {
    if (!state) {
        return;
    }
    state->current_token_id = std::max<int32_t>(0, token_id);
    state->occurrences.clear();
}

static std::string make_dump_name(const std::string & base_name, int32_t token_id, int occurrence) {
    std::ostringstream name;
    name << base_name
         << "-token-" << std::setw(6) << std::setfill('0') << std::max<int32_t>(0, token_id)
         << "-occ-" << std::setw(3) << std::setfill('0') << std::max(0, occurrence);
    return name.str();
}

static bool append_index_entry(
    const DumpState * state,
    const std::string & dump_name,
    const std::string & base_name,
    const ggml_tensor * t,
    int occurrence
) {
    if (!state || state->index_path.empty()) {
        return false;
    }
    std::ofstream index(state->index_path, std::ios::binary | std::ios::app);
    if (!index) {
        return false;
    }
    index << "{"
          << "\"name\":\"" << json_escape(dump_name) << "\","
          << "\"base_name\":\"" << json_escape(base_name) << "\","
          << "\"token_id\":" << std::max<int32_t>(0, state->current_token_id) << ","
          << "\"occurrence\":" << std::max(0, occurrence) << ","
          << "\"dtype\":" << static_cast<int>(t->type) << ","
          << "\"rank\":" << ggml_n_dims(t) << ","
          << "\"shape\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "],"
          << "\"elem_count\":" << ggml_nelements(t) << ","
          << "\"nbytes\":" << ggml_nbytes(t)
          << "}\n";
    return index.good();
}

static bool dump_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    const DumpState * state = static_cast<const DumpState *>(user_data);
    if (!should_dump_tensor(state, t)) {
        return false;
    }
    if (ask) {
        return true;
    }

    const char * raw_name = ggml_get_name(t);
    if (!raw_name || !raw_name[0]) {
        return true;
    }
    std::string base_name(raw_name);

    const int64_t nbytes = ggml_nbytes(t);
    if (nbytes <= 0) {
        return true;
    }

    std::vector<uint8_t> raw(static_cast<size_t>(nbytes));
    ggml_backend_tensor_get(t, raw.data(), 0, static_cast<size_t>(nbytes));

    DumpState * mut = static_cast<DumpState *>(user_data);
    const int occurrence = mut->occurrences[base_name]++;
    const std::string dump_name = make_dump_name(base_name, mut->current_token_id, occurrence);

    std::filesystem::create_directories(mut->dump_dir);
    const std::filesystem::path bin_path = mut->dump_dir / (dump_name + ".bin");
    std::ofstream f(bin_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    f.write(reinterpret_cast<const char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (!f.good()) {
        return false;
    }

    const std::filesystem::path meta_path = mut->dump_dir / (dump_name + ".json");
    std::ofstream meta(meta_path, std::ios::binary | std::ios::trunc);
    if (meta) {
        meta << "{";
        meta << "\"name\":\"" << json_escape(dump_name) << "\",";
        meta << "\"base_name\":\"" << json_escape(base_name) << "\",";
        meta << "\"token_id\":" << std::max<int32_t>(0, mut->current_token_id) << ",";
        meta << "\"occurrence\":" << std::max(0, occurrence) << ",";
        meta << "\"type\":" << static_cast<int>(t->type) << ",";
        meta << "\"nbytes\":" << nbytes << ",";
        meta << "\"elem_count\":" << ggml_nelements(t) << ",";
        meta << "\"ne\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "],";
        meta << "\"nb\":[" << t->nb[0] << "," << t->nb[1] << "," << t->nb[2] << "," << t->nb[3] << "]";
        meta << "}\n";
    }

    if (!append_index_entry(mut, dump_name, base_name, t, occurrence)) {
        return false;
    }
    mut->dumped += 1;
    return true;
}

static bool tokenize_prompt(
    const llama_vocab * vocab,
    const std::string & prompt,
    std::vector<int32_t> & out_tokens,
    std::string & err
) {
    out_tokens.clear();
    int32_t cap = std::max<int32_t>(32, static_cast<int32_t>(prompt.size()) + 8);
    out_tokens.resize(static_cast<size_t>(cap));
    int32_t n = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        out_tokens.data(),
        static_cast<int32_t>(out_tokens.size()),
        true,
        true
    );
    if (n < 0) {
        cap = -n;
        out_tokens.resize(static_cast<size_t>(cap));
        n = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            out_tokens.data(),
            static_cast<int32_t>(out_tokens.size()),
            true,
            true
        );
    }
    if (n < 0) {
        err = "llama_tokenize failed";
        return false;
    }
    out_tokens.resize(static_cast<size_t>(n));
    if (out_tokens.empty()) {
        err = "tokenized prompt is empty";
        return false;
    }
    return true;
}

static int32_t decode_tokens(
    llama_context * ctx,
    const std::vector<llama_token> & tokens,
    const std::string & decode_mode,
    int32_t pos0,
    DumpState * dump_state
) {
    if (decode_mode == "sequential") {
        for (size_t i = 0; i < tokens.size(); ++i) {
            begin_dump_batch(dump_state, pos0 + static_cast<int32_t>(i));
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.n_tokens = 1;
            batch.token[0] = tokens[i];
            batch.pos[0] = pos0 + static_cast<int32_t>(i);
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            const int32_t rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                return rc;
            }
        }
        return 0;
    }

    begin_dump_batch(dump_state, pos0 + static_cast<int32_t>(tokens.size()) - 1);
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos0 + static_cast<int32_t>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i + 1 == tokens.size()) ? 1 : 0;
    }
    const int32_t rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc;
}

static int32_t decode_prefix_embeddings(
    llama_context * ctx,
    const std::vector<float> & prefix_embd,
    int32_t prefix_tokens,
    int32_t n_embd,
    int prefix_grid_x,
    int prefix_grid_y,
    int32_t prefix_pos0,
    int32_t prefix_text_pos,
    const std::string & prefix_decode_mode,
    DumpState * dump_state,
    int32_t * prefix_text_pos_out
) {
    if (prefix_text_pos_out) {
        *prefix_text_pos_out = prefix_pos0;
    }
    if (prefix_tokens <= 0) {
        return 0;
    }

    const bool use_mrope_2d = prefix_grid_x > 0 && prefix_grid_y > 0;
    const bool use_mrope_1d = !use_mrope_2d && prefix_grid_x == 0 && prefix_grid_y == 0;
    const int pos_width = (use_mrope_2d || use_mrope_1d) ? 4 : 1;
    if (use_mrope_2d && static_cast<int64_t>(prefix_grid_x) * static_cast<int64_t>(prefix_grid_y) != prefix_tokens) {
        return -2;
    }

    std::vector<llama_pos> pos(static_cast<size_t>(prefix_tokens) * static_cast<size_t>(pos_width), 0);
    std::vector<int32_t> n_seq_id(static_cast<size_t>(prefix_tokens), 1);
    std::vector<int8_t> logits(static_cast<size_t>(prefix_tokens), 0);
    std::vector<llama_seq_id *> seq_ids(static_cast<size_t>(prefix_tokens), nullptr);
    llama_seq_id seq0_storage[1] = {0};
    for (int32_t i = 0; i < prefix_tokens; ++i) {
        seq_ids[static_cast<size_t>(i)] = seq0_storage;
    }

    if (use_mrope_2d) {
        for (int32_t y = 0; y < prefix_grid_y; ++y) {
            for (int32_t x = 0; x < prefix_grid_x; ++x) {
                const int32_t i = y * prefix_grid_x + x;
                pos[static_cast<size_t>(i)] = prefix_pos0;
                pos[static_cast<size_t>(i) + static_cast<size_t>(prefix_tokens)] = prefix_pos0 + y;
                pos[static_cast<size_t>(i) + static_cast<size_t>(prefix_tokens) * 2] = prefix_pos0 + x;
                pos[static_cast<size_t>(i) + static_cast<size_t>(prefix_tokens) * 3] = 0;
            }
        }
    } else if (use_mrope_1d) {
        for (int32_t i = 0; i < prefix_tokens; ++i) {
            const llama_pos pos_i = prefix_pos0 + i;
            pos[static_cast<size_t>(i)] = pos_i;
            pos[static_cast<size_t>(i) + static_cast<size_t>(prefix_tokens)] = pos_i;
            pos[static_cast<size_t>(i) + static_cast<size_t>(prefix_tokens) * 2] = pos_i;
            pos[static_cast<size_t>(i) + static_cast<size_t>(prefix_tokens) * 3] = 0;
        }
    } else {
        for (int32_t i = 0; i < prefix_tokens; ++i) {
            pos[static_cast<size_t>(i)] = prefix_pos0 + i;
        }
    }

    int32_t rc = 0;
    if (prefix_decode_mode == "sequential") {
        for (int32_t i = 0; i < prefix_tokens; ++i) {
            llama_seq_id seq0_storage[1] = {0};
            llama_seq_id * seq_ids_1[1] = {seq0_storage};
            int32_t n_seq_id_1[1] = {1};
            int8_t logits_1[1] = {0};
            llama_pos pos_1[4] = {0, 0, 0, 0};
            for (int d = 0; d < pos_width; ++d) {
                pos_1[d] = pos[static_cast<size_t>(i) + static_cast<size_t>(prefix_tokens) * static_cast<size_t>(d)];
            }
            llama_batch batch = {
                /*n_tokens =*/ 1,
                /*token    =*/ nullptr,
                /*embd     =*/ const_cast<float *>(prefix_embd.data() + static_cast<size_t>(i) * static_cast<size_t>(n_embd)),
                /*pos      =*/ pos_1,
                /*n_seq_id =*/ n_seq_id_1,
                /*seq_id   =*/ seq_ids_1,
                /*logits   =*/ logits_1,
            };
            begin_dump_batch(dump_state, prefix_pos0 + i);
            rc = llama_decode(ctx, batch);
            if (rc != 0) {
                return rc;
            }
        }
    } else {
        llama_batch batch = {
            /*n_tokens =*/ prefix_tokens,
            /*token    =*/ nullptr,
            /*embd     =*/ const_cast<float *>(prefix_embd.data()),
            /*pos      =*/ pos.data(),
            /*n_seq_id =*/ n_seq_id.data(),
            /*seq_id   =*/ seq_ids.data(),
            /*logits   =*/ logits.data(),
        };

        begin_dump_batch(dump_state, std::max<int32_t>(prefix_pos0, prefix_pos0 + prefix_tokens - 1));
        rc = llama_decode(ctx, batch);
        if (rc != 0) {
            return rc;
        }
    }

    if (prefix_text_pos_out) {
        if (prefix_text_pos >= 0) {
            *prefix_text_pos_out = prefix_text_pos;
        } else if (use_mrope_2d) {
            *prefix_text_pos_out = prefix_pos0 + std::max(prefix_grid_x, prefix_grid_y);
        } else {
            *prefix_text_pos_out = prefix_pos0 + prefix_tokens;
        }
    }
    return rc;
}

int main(int argc, char ** argv) {
    Args args;
    std::string err;
    if (!parse_args(argc, argv, args, err)) {
        print_json_error(err);
        return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = true;
    mparams.use_mlock = false;

    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (!model) {
        print_json_error("llama_model_load_from_file failed");
        llama_backend_free();
        return 3;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        print_json_error("llama_model_get_vocab returned null");
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    const bool segmented = !args.tokens_before.empty() || !args.tokens_after.empty();
    if (!segmented && args.tokens.empty() && !args.prompt.empty()) {
        if (!tokenize_prompt(vocab, args.prompt, args.tokens, err)) {
            print_json_error(err);
            llama_model_free(model);
            llama_backend_free();
            return 11;
        }
    }

    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_embd_inp = llama_model_n_embd_inp(model);
    std::vector<float> prefix_embd;
    int32_t prefix_tokens = 0;
    if (!load_prefix_embeddings(args.prefix_f32_path, n_embd, n_embd_inp, args.prefix_row_dim, prefix_embd, prefix_tokens, err)) {
        print_json_error(err);
        llama_model_free(model);
        llama_backend_free();
        return 12;
    }

    llama_context_params cparams = llama_context_default_params();
    const int total_tokens = prefix_tokens + static_cast<int>(segmented ? (args.tokens_before.size() + args.tokens_after.size()) : args.tokens.size());
    cparams.n_ctx = static_cast<uint32_t>(std::max(args.ctx_len, total_tokens + 8));
    cparams.n_batch = static_cast<uint32_t>(std::max<int>(32, total_tokens));
    cparams.n_ubatch = cparams.n_batch;
    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    int n_threads = args.threads > 0 ? args.threads : std::max(1, hw_threads);
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    DumpState dump_state;
    if (!args.dump_dir.empty()) {
        dump_state.dump_dir = args.dump_dir;
        dump_state.index_path = dump_state.dump_dir / "index.json";
        dump_state.dump_all = args.dump_names.empty();
        for (const std::string & name : args.dump_names) {
            if (!name.empty()) {
                dump_state.names.insert(name);
            }
        }
        if (dump_state.dump_all || !dump_state.names.empty()) {
            std::filesystem::create_directories(dump_state.dump_dir);
            std::error_code ec;
            std::filesystem::remove(dump_state.index_path, ec);
            cparams.cb_eval = dump_eval_callback;
            cparams.cb_eval_user_data = &dump_state;
        }
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        print_json_error("llama_init_from_model failed");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    std::vector<llama_token> tokens_before(args.tokens_before.begin(), args.tokens_before.end());
    std::vector<llama_token> tokens_after(
        segmented ? args.tokens_after.begin() : args.tokens.begin(),
        segmented ? args.tokens_after.end() : args.tokens.end()
    );
    if (!tokens_before.empty()) {
        int32_t rc = decode_tokens(
            ctx,
            tokens_before,
            args.decode_mode,
            0,
            dump_state.dump_dir.empty() ? nullptr : &dump_state
        );
        if (rc != 0) {
            std::ostringstream oss;
            oss << "llama_decode failed rc=" << rc << " mode=" << args.decode_mode << " while replaying tokens-before";
            print_json_error(oss.str());
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 14;
        }
    }

    int32_t prefix_text_pos = static_cast<int32_t>(tokens_before.size());
    if (prefix_tokens > 0) {
        int32_t rc = decode_prefix_embeddings(
            ctx,
            prefix_embd,
            prefix_tokens,
            n_embd_inp,
            args.prefix_grid_x,
            args.prefix_grid_y,
            static_cast<int32_t>(tokens_before.size()),
            args.prefix_text_pos,
            args.prefix_decode_mode,
            dump_state.dump_dir.empty() ? nullptr : &dump_state,
            &prefix_text_pos
        );
        if (rc != 0) {
            std::ostringstream oss;
            oss << "llama_decode failed rc=" << rc << " while replaying prefix embeddings";
            print_json_error(oss.str());
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 13;
        }
    }
    if (!tokens_after.empty()) {
        int32_t rc = decode_tokens(
            ctx,
            tokens_after,
            args.decode_mode,
            prefix_text_pos,
            dump_state.dump_dir.empty() ? nullptr : &dump_state
        );
        if (rc != 0) {
            std::ostringstream oss;
            oss << "llama_decode failed rc=" << rc << " mode=" << args.decode_mode;
            print_json_error(oss.str());
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 5;
        }
    }

    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        print_json_error("invalid vocab size");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 7;
    }

    const float * logits = nullptr;
    if (args.decode_mode != "sequential" && !tokens_after.empty()) {
        logits = llama_get_logits_ith(ctx, static_cast<int32_t>(tokens_after.size()) - 1);
    }
    if (!logits) {
        logits = llama_get_logits_ith(ctx, -1);
    }
    if (!logits) {
        logits = llama_get_logits(ctx);
    }
    if (!logits) {
        print_json_error("llama logits pointer is null");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 8;
    }

    {
        std::ofstream f(args.logits_out_path, std::ios::binary | std::ios::trunc);
        if (!f) {
            print_json_error("failed opening logits-out file");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 9;
        }
        f.write(reinterpret_cast<const char *>(logits), static_cast<std::streamsize>(n_vocab) * sizeof(float));
        if (!f.good()) {
            print_json_error("failed writing logits-out file");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 10;
        }
    }

    std::vector<int> ids(n_vocab);
    for (int i = 0; i < n_vocab; ++i) ids[i] = i;
    int k = std::min<int>(std::max(1, args.top_k), n_vocab);
    std::partial_sort(
        ids.begin(),
        ids.begin() + k,
        ids.end(),
        [&](int a, int b) { return logits[a] > logits[b]; }
    );

    std::cout << "{";
    std::cout << "\"ok\":true,";
    std::cout << "\"n_vocab\":" << n_vocab << ",";
    std::cout << "\"token_count\":" << (tokens_before.size() + tokens_after.size()) << ",";
    std::cout << "\"token_count_before\":" << tokens_before.size() << ",";
    std::cout << "\"token_count_after\":" << tokens_after.size() << ",";
    std::cout << "\"prefix_token_count\":" << prefix_tokens << ",";
    std::cout << "\"prefix_position_count\":" << std::max<int32_t>(0, prefix_text_pos - static_cast<int32_t>(tokens_before.size())) << ",";
    std::cout << "\"prefix_start_pos\":" << tokens_before.size() << ",";
    std::cout << "\"prefix_text_pos\":" << prefix_text_pos << ",";
    std::cout << "\"tokens\":[";
    for (size_t i = 0; i < tokens_before.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << tokens_before[i];
    }
    for (size_t i = 0; i < tokens_after.size(); ++i) {
        if (!tokens_before.empty() || i) std::cout << ",";
        std::cout << tokens_after[i];
    }
    std::cout << "],";
    std::cout << "\"tokens_before\":[";
    for (size_t i = 0; i < tokens_before.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << tokens_before[i];
    }
    std::cout << "],";
    std::cout << "\"tokens_after\":[";
    for (size_t i = 0; i < tokens_after.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << tokens_after[i];
    }
    std::cout << "],";
    std::cout << "\"decode_mode\":\"" << args.decode_mode << "\",";
    std::cout << "\"dumped\":" << dump_state.dumped << ",";
    std::cout << "\"topk\":[";
    for (int i = 0; i < k; ++i) {
        if (i) std::cout << ",";
        std::cout << "{\"id\":" << ids[i] << ",\"logit\":" << logits[ids[i]] << "}";
    }
    std::cout << "]";
    std::cout << "}\n";

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
