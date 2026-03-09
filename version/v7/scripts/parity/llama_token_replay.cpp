#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct Args {
    std::string model_path;
    std::vector<int32_t> tokens;
    std::string prompt;
    std::string logits_out_path;
    std::string decode_mode = "batched";
    int ctx_len = 256;
    int top_k = 16;
    int threads = 0;
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
        if (a == "--prompt") {
            const char * v = need_value("--prompt");
            if (!v) return false;
            args.prompt = v;
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
        if (a == "-h" || a == "--help") {
            std::cout
                << "Usage: llama_token_replay --model <path.gguf> "
                << "(--tokens <id,id,...> | --prompt <text>) "
                << "--logits-out <path.bin> [--ctx N] [--top-k K] [--threads N] "
                << "[--decode-mode batched|sequential]\n";
            std::exit(0);
        }
        err = "unknown arg: " + a;
        return false;
    }

    if (args.model_path.empty()) {
        err = "missing --model";
        return false;
    }
    if (args.tokens.empty() == args.prompt.empty()) {
        err = "pass exactly one of --tokens or --prompt";
        return false;
    }
    if (args.logits_out_path.empty()) {
        err = "missing --logits-out";
        return false;
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
    const std::string & decode_mode
) {
    if (decode_mode == "sequential") {
        for (size_t i = 0; i < tokens.size(); ++i) {
            llama_token tok = tokens[i];
            llama_batch batch = llama_batch_get_one(&tok, 1);
            const int32_t rc = llama_decode(ctx, batch);
            if (rc != 0) {
                return rc;
            }
        }
        return 0;
    }

    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(tokens.data()),
        static_cast<int32_t>(tokens.size())
    );
    return llama_decode(ctx, batch);
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

    if (args.tokens.empty()) {
        if (!tokenize_prompt(vocab, args.prompt, args.tokens, err)) {
            print_json_error(err);
            llama_model_free(model);
            llama_backend_free();
            return 11;
        }
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(std::max(args.ctx_len, static_cast<int>(args.tokens.size()) + 8));
    cparams.n_batch = static_cast<uint32_t>(std::max<int>(32, static_cast<int>(args.tokens.size())));
    cparams.n_ubatch = cparams.n_batch;
    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    int n_threads = args.threads > 0 ? args.threads : std::max(1, hw_threads);
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        print_json_error("llama_init_from_model failed");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    std::vector<llama_token> tokens(args.tokens.begin(), args.tokens.end());
    int32_t rc = decode_tokens(ctx, tokens, args.decode_mode);
    if (rc != 0) {
        std::ostringstream oss;
        oss << "llama_decode failed rc=" << rc << " mode=" << args.decode_mode;
        print_json_error(oss.str());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 5;
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
    logits = llama_get_logits_ith(ctx, static_cast<int32_t>(tokens.size()) - 1);
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
    std::cout << "\"token_count\":" << tokens.size() << ",";
    std::cout << "\"tokens\":[";
    for (size_t i = 0; i < args.tokens.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << args.tokens[i];
    }
    std::cout << "],";
    std::cout << "\"decode_mode\":\"" << args.decode_mode << "\",";
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
