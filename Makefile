# Auto-detect Intel oneAPI compilers if available (preferred for performance).
# Intel compilers (icx/icpx) typically produce faster code for INT8/AVX-512 workloads.
# (icx-built binaries depend on the Intel runtime, e.g. `libimf.so`.)
#
# To force gcc even with Intel compilers available:
#   make CC=gcc
# To explicitly use Intel oneAPI when auto-detection fails:
#   make CC=icx CXX=icpx

# Auto-detect Intel oneAPI compilers
ICX_CXX := $(shell command -v icpx 2>/dev/null)
ICX_CC := $(shell command -v icx 2>/dev/null)

# Default to gcc for portability, but prefer Intel if available
ifeq ($(origin CC),default)
    ifneq ($(ICX_CC),)
        CC := icx
        CXX := icpx
    else
        CC := gcc
    endif
endif

# Handle CC=cc (some environments export this - reset to gcc)
ifeq ($(CC),cc)
    CC := gcc
    CXX := g++
endif

# If user explicitly set CC=icx, also set CXX
ifneq ($(findstring icx,$(CC)),)
    CXX ?= icpx
endif
# OpenMP flag varies by compiler (icx/icc prefer -qopenmp; gcc/clang use -fopenmp).
OPENMP_FLAG ?= -fopenmp
ifneq (,$(findstring icc,$(CC)))
OPENMP_FLAG := -qopenmp
endif
ifneq (,$(findstring icx,$(CC)))
OPENMP_FLAG := -qopenmp
endif
# You can override AVX/arch flags from the environment if needed, e.g.:
#   make AVX_FLAGS="-mavx2"
#   make AVX_FLAGS=""            # scalar build
# Get ALL flags lines (AVX-512/AMX might be on subsequent lines)
CPU_FLAGS := $(shell grep '^flags' /proc/cpuinfo | tr '\n' ' ' 2>/dev/null)
# Detect FMA support
ifneq (,$(findstring fma,$(CPU_FLAGS)))
FMA_FLAGS := -mfma
else
FMA_FLAGS :=
endif

# Detect AVX level and set appropriate flags for the compiler
# Intel icx uses -x<target> or -xHost, GCC uses -mavx*
DEFAULT_AVX_FLAGS_GCC :=
DEFAULT_AVX_FLAGS_INTEL :=

# Check CPU capabilities
AVX512_SUPPORT := $(findstring avx512f,$(CPU_FLAGS))
AVX2_SUPPORT := $(findstring avx2,$(CPU_FLAGS))
AVX_SUPPORT := $(findstring avx,$(CPU_FLAGS))
AMX_SUPPORT := $(findstring amx_tile,$(CPU_FLAGS))

# GCC flags (used when CC=gcc/clang)
ifneq (,$(AVX512_SUPPORT))
# Full AVX-512 requires F, BW, DQ for all kernels (including _mm512_extractf32x8_ps)
DEFAULT_AVX_FLAGS_GCC := -mavx512f -mavx512bw -mavx512dq $(FMA_FLAGS)
ifneq (,$(findstring avx512vnni,$(CPU_FLAGS)))
DEFAULT_AVX_FLAGS_GCC += -mavx512vnni
endif
ifneq (,$(findstring avx512_vnni,$(CPU_FLAGS)))
DEFAULT_AVX_FLAGS_GCC += -mavx512vnni
endif
else ifneq (,$(AVX2_SUPPORT))
DEFAULT_AVX_FLAGS_GCC := -mavx2 $(FMA_FLAGS)
else ifneq (,$(AVX_SUPPORT))
DEFAULT_AVX_FLAGS_GCC := -mavx $(FMA_FLAGS)
endif

# Intel icx flags - use -xHost to auto-detect CPU, or -xAVX2 for Ivy Bridge
# -xHost auto-detects but generates code for the BUILD machine's CPU
# Since we might build on a different machine, use explicit targeting
ifneq (,$(AVX512_SUPPORT))
# Haswell/Broadwell or newer - use AVX-512
DEFAULT_AVX_FLAGS_INTEL := -xcore-avx512
else ifneq (,$(AVX2_SUPPORT))
# Ivy Bridge/Skylake or newer - use AVX2
DEFAULT_AVX_FLAGS_INTEL := -xAVX2
else
# Older CPUs
DEFAULT_AVX_FLAGS_INTEL := -xAVX
endif

# Use appropriate flags based on compiler
ifneq (,$(findstring icx,$(CC)))
AVX_FLAGS ?= $(DEFAULT_AVX_FLAGS_INTEL)
else
AVX_FLAGS ?= $(DEFAULT_AVX_FLAGS_GCC)
endif

# Detect SSSE3 support (needed for _mm_maddubs_epi16 etc.)
ifneq (,$(findstring ssse3,$(CPU_FLAGS)))
SSSE3_FLAGS := -mssse3
else
SSSE3_FLAGS :=
endif

INCLUDES := -Iinclude
CFLAGS  := -O3 -fPIC $(OPENMP_FLAG) -Wall $(AVX_FLAGS) $(SSSE3_FLAGS) $(INCLUDES)
CXX ?= g++
BENCH_CC ?= gcc
BENCH_CXX ?= $(CXX)

BUILD_DIR := build
BUILD_STAMP := $(BUILD_DIR)/.ck_build_flags

# =============================================================================
# Intel oneAPI Integration (MKL / oneDNN)
# =============================================================================
# Auto-detection: MKL is used automatically if found
# Disable with: make USE_NATIVE=1
# Force MKL:    make USE_MKL=1
# Force oneDNN: make USE_ONEDNN=1

ONEAPI_ROOT ?= /opt/intel/oneapi

# MKL paths
MKL_ROOT ?= $(ONEAPI_ROOT)/mkl/latest
MKL_INC := $(MKL_ROOT)/include
MKL_LIB := $(MKL_ROOT)/lib/intel64

# Auto-detect MKL availability
MKL_AVAILABLE := $(wildcard $(MKL_INC)/mkl.h)

# Auto-enable MKL if available and not explicitly using native/oneDNN
ifndef USE_NATIVE
ifndef USE_ONEDNN
ifndef USE_MKL
ifneq ($(MKL_AVAILABLE),)
USE_MKL := 1
endif
endif
endif
endif

# oneDNN paths
# Prefer oneAPI if installed; otherwise default to /usr/local (typical from-source install).
DNNL_ROOT ?= $(if $(wildcard $(ONEAPI_ROOT)/dnnl/latest/include/dnnl.h),$(ONEAPI_ROOT)/dnnl/latest,/usr/local)
DNNL_INC := $(DNNL_ROOT)/include
DNNL_LIB := $(DNNL_ROOT)/lib

DNNL_HPP := $(wildcard $(DNNL_INC)/dnnl.hpp)

# Add MKL support
ifdef USE_MKL
    # Intel compiler runtime library (libimf.so, etc.) needed by MKL
    INTEL_COMPILER_LIB := $(ONEAPI_ROOT)/compiler/latest/lib
    CFLAGS += -DUSE_MKL -I$(MKL_INC)
    LDFLAGS += -L$(MKL_LIB) -lmkl_rt -Wl,-rpath,$(MKL_LIB)
    LDFLAGS += -L$(INTEL_COMPILER_LIB) -Wl,-rpath,$(INTEL_COMPILER_LIB)
    $(info Building with Intel MKL backend for GEMM)
endif

# Add oneDNN support
ifdef USE_ONEDNN
    # Prefer /usr/local OpenMP-based oneDNN over Intel oneAPI SYCL version
    DNNL_LOCAL := $(wildcard /usr/local/lib/libdnnl.so)
    ifdef DNNL_LOCAL
        DNNL_INC := /usr/local/include
        DNNL_LIB := /usr/local/lib
    else
        # Intel compiler runtime library (libimf.so, etc.) needed by oneAPI oneDNN
        INTEL_COMPILER_LIB := $(ONEAPI_ROOT)/compiler/latest/lib
        LDFLAGS += -L$(INTEL_COMPILER_LIB) -Wl,-rpath,$(INTEL_COMPILER_LIB)
    endif
    CFLAGS += -DUSE_ONEDNN -I$(DNNL_INC)
    LDFLAGS += -L$(DNNL_LIB) -ldnnl -Wl,-rpath,$(DNNL_LIB)
    $(info Building with Intel oneDNN backend for GEMM)
endif

# Default message
ifndef USE_MKL
ifndef USE_ONEDNN
    $(info Building with native AVX kernels for GEMM)
endif
endif

SRCS_v2 := src/v2_legacy/ckernel_ir_v2.c \
            src/v2_legacy/ckernel_ir_v2_builder.c \
            src/v2_legacy/ckernel_ir_v2_lower.c \
            src/v2_legacy/ckernel_codegen_v2.c \
            src/v2_legacy/ckernel_codegen_v2_struct.c \
            src/v2_legacy/ckernel_codegen_v2_dispatch.c \
            src/v2_legacy/ckernel_codegen_v2_schedule.c \
            src/v2_legacy/ckernel_codegen_v2_sections.c \
            src/v2_legacy/ck_tokenizer_v2.c \

SRCS_v4 := src/ckernel_model_load_v4.c \

SRCS    := src/backend_native.c \
           src/ckernel_ir.c \
           src/ckernel_codegen.c \
           src/ckernel_kernel_specs.c \
           src/ckernel_mem_plan.c \
           src/ckernel_alloc.c \
           src/ckernel_strict.c \
           src/ckernel_registry.c \
           src/ckernel_orchestration.c \
           src/ckernel_model_layout.c \
           src/ckernel_model_load.c \
           src/ck_tokenizer.c \
           src/cpu_features.c \
            src/kernels/gemm_kernels.c \
            src/kernels/gemm_fused_kernels.c \
            src/kernels/mlp_fused_decode.c \
            src/kernels/gemm_microkernel.c \
	           src/kernels/layernorm_kernels.c \
	           src/kernels/layernorm_kernels_bf16.c \
	           src/kernels/gelu_kernels.c \
	           src/kernels/gelu_kernels_bf16.c \
	           src/kernels/softmax_kernels.c \
	           src/kernels/softmax_kernels_bf16.c \
	           src/kernels/attention_kernels.c \
	           src/kernels/attention_flash_true.c \
	           src/kernels/attention_decode_fused.c \
	           src/kernels/embedding_kernels.c \
	           src/kernels/embedding_kernels_bf16.c \
	           src/kernels/loss_kernels.c \
	           src/kernels/loss_kernels_bf16.c \
	           src/kernels/mlp_kernels.c \
	           src/kernels/mlp_kernels_bf16.c \
	           src/kernels/rmsnorm_kernels.c \
	            src/kernels/rmsnorm_kernels_bf16.c \
	            src/kernels/rmsnorm_kernels_int8.c \
	            src/kernels/rmsnorm_kernels_int4.c \
	            src/kernels/swiglu_kernels.c \
	           src/kernels/swiglu_kernels_bf16.c \
	           src/kernels/sigmoid_kernels.c \
	           src/kernels/sigmoid_kernels_bf16.c \
	           src/kernels/relu_kernels.c \
	           src/kernels/relu_kernels_bf16.c \
	           src/kernels/vision_kernels.c \
	           src/kernels/vision_kernels_bf16.c \
	           src/kernels/rope_kernels.c \
	           src/kernels/rope_kernels_bf16.c \
	           src/kernels/kv_cache_kernels.c \
	           src/kernels/dequant_kernels.c \
	           src/kernels/gemm_kernels_bf16.c \
	           src/kernels/gemm_kernels_q4_0.c \
	           src/kernels/gemm_kernels_q4_1.c \
	           src/kernels/gemm_kernels_q5_0.c \
	           src/kernels/gemm_kernels_q5_0_sse_v2.c \
	           src/kernels/gemm_kernels_q5_1.c \
	           src/kernels/gemm_kernels_q4k.c \
	           src/kernels/gemm_kernels_q4k_sse.c \
	           src/kernels/gemm_kernels_q4k_avx.c \
	           src/kernels/gemm_kernels_q6k.c \
	           src/kernels/gemm_kernels_q6k_sse.c \
	           src/kernels/gemm_kernels_q4k_q8k.c \
	           src/kernels/gemm_kernels_q4k_q8k_avx2.c \
	           src/kernels/gemm_kernels_q4k_q8k_vnni.c \
	           src/kernels/gemm_kernels_amx.c \
	           src/kernels/gemm_kernels_q6k_q8k.c \
	           src/kernels/gemm_batch_int8.c \
	           src/kernels/fused_rmsnorm_linear.c \
	           src/kernels/gemm_kernels_q8_0.c \
	           src/kernels/quantize_row_q8_k_sse.c \
	           src/kernels/rmsnorm_q8_k_fused.c \
	           src/kernels/gemm_kernels_f16.c \
	           src/kernels/optimizer_kernels.c \
	           src/kernels/optimizer_kernels_bf16.c \
	           src/kernels/add_kernels_bf16.c \
	           src/kernels/topk_kernels.c \
	           src/kernels/axpy_kernels.c
LIB          := $(BUILD_DIR)/libckernel_engine.so
LIB_QUANT    := $(BUILD_DIR)/libckernel_quant.so
LIB_GELU     := $(BUILD_DIR)/libckernel_gelu.so
LIB_RMSNORM  := $(BUILD_DIR)/libckernel_rmsnorm.so
LIB_LN       := $(BUILD_DIR)/libckernel_layernorm.so
LIB_SOFT     := $(BUILD_DIR)/libckernel_softmax.so
LIB_SWIGLU   := $(BUILD_DIR)/libckernel_swiglu.so
LIB_SIGMOID  := $(BUILD_DIR)/libckernel_sigmoid.so
LIB_RELU     := $(BUILD_DIR)/libckernel_relu.so
LIB_VISION   := $(BUILD_DIR)/libckernel_vision.so
LIB_ATTENTION := $(BUILD_DIR)/libckernel_attention.so
LIB_ROPE     := $(BUILD_DIR)/libckernel_rope.so

# Tokenizer library (new - from src/tokenizer/)
SRCS_TOKENIZER := src/tokenizer/murmurhash3.c \
                  src/tokenizer/hash_table.c \
                  src/tokenizer/memory_pool.c \
                  src/tokenizer/utf8.c \
                  src/tokenizer/tokenizer.c \
                  src/tokenizer/true_bpe.c \
                  src/data_structures/tries/trie.c
LIB_TOKENIZER := $(BUILD_DIR)/libckernel_tokenizer.so

IR_DEMO := $(BUILD_DIR)/ck_ir_demo
IR_V2_DEMO := $(BUILD_DIR)/ck_ir_v2_demo
IR_V2_SCRIPT := scripts/build_ir_v2.py
IR_V4_SCRIPT := scripts/v4/build_ir_v4.py
IR_V4_Q4K_SCRIPT := scripts/v4/build_ir_v4_q4k.py
DEFAULT_CONFIG := default.config.json
CONFIG ?= $(DEFAULT_CONFIG)
OUT ?= $(BUILD_DIR)/generated_model.c
IR ?=
IR_V2_OUT ?= $(BUILD_DIR)/ir_v2.json
IR_V2_WEIGHTS ?=
IR_V2_META ?= $(BUILD_DIR)/weights_meta.json
IR_V2_HF ?=
IR_V2_REV ?= main
IR_V2_CTX ?=
IR_V2_KERNEL_DTYPE ?=
IR_V2_ACT_DTYPE ?=
IR_V4_MODEL ?=
IR_V4_CONFIG ?=
IR_V4_PRESET ?=
IR_V4_PREFIX ?=
IR_V4_TOKENS ?=
IR_V4_MODES ?= prefill,decode
IR_V4_EMIT ?= exe
IR_V4_DTYPE ?=
IR_V4_WEIGHTS_HEADER ?=
IR_V4_WEIGHTS_INDEX ?=
IR_V4_KERNEL_SPECS ?=
IR_V4_Q4K_CHECKPOINT ?=
IR_V4_Q4K_GGUF ?=
IR_V4_Q4K_PRESET ?=
IR_V4_Q4K_CONFIG ?=
IR_V4_Q4K_PREFIX ?=
IR_V4_Q4K_MODES ?= prefill,decode
IR_V4_Q4K_TOKENS ?=
IR_V4_Q4K_CONTEXT ?=
IR_V4_Q4K_FUSION ?= off
IR_V4_Q4K_EMIT ?= lib
IR_V4_Q4K_VERBOSE ?=
GGUF ?=
GGUF_OUT ?= $(BUILD_DIR)/gguf_weights.bump
GGUF_CONFIG_OUT ?= $(BUILD_DIR)/gguf_config.json
GGUF_CONTEXT ?=
GGUF_V4_OUT ?= $(BUILD_DIR)/gguf_weights_v4.bump
GGUF_V4_CONFIG_OUT ?= $(BUILD_DIR)/gguf_config_v4.json
GGUF_V4_CONTEXT ?=
HF_V4_CHECKPOINT ?=
HF_V4_OUT ?= $(BUILD_DIR)/weights_v4.bump
HF_V4_CONFIG ?=
HF_V4_DTYPE ?=
HF_V4_CONTEXT ?=
HF_V4_MAP_OUT ?=
TINY_CONFIG ?= tiny.config.json
SMALL_CONFIG ?= small10mb.config.json
TINY_TRAIN_LR ?= 1e-3
TINY_TRAIN_ARGS ?= --dump
TINY_PARITY_ARGS ?=
ALL_TEST_LAYER_ARGS ?= --tokens 256 --embed 64 --heads 4 --kv-heads 2 --intermediate 128 --rope --strict-ref
ALL_TEST_LAYER_TOL ?= 2e-3
SMOLLM_CONFIG ?= smolLM-135.json
SMOLLM_MODEL_DIR ?= $(HOME)/.cache/huggingface/hub/SmolLM-135M
SMOLLM_REPO ?= HuggingFaceTB/SmolLM-135M
SMOLLM_DOWNLOAD ?=
SMOLLM_CONTEXT ?= 2
SMOLLM_DATASET ?= roneneldan/TinyStories
SMOLLM_DATASET_CONFIG ?=
SMOLLM_SPLIT ?= train
SMOLLM_MAX_SAMPLES ?= 4
SMOLLM_TEXT ?= Once upon a time
SMOLLM_TOPK ?= 5
SMOLLM_LAYER ?= 0
SMOLLM_STAGE_TOL ?= 1e-3
SMOLLM_STAGE_DUMP ?=
SMOLLM_BUMP ?= $(BUILD_DIR)/smollm_weights.bin
SMOLLM_OUT_WEIGHTS ?= $(BUILD_DIR)/smollm_weights_after.bin
SMOLLM_MAX_LAYERS ?=

PYTHON  ?= python3
PYTHONFLAGS ?= -B
PY_TESTS := unittest/test_layernorm.py \
            unittest/test_gelu.py \
            unittest/test_softmax.py \
            unittest/test_softmax_backward.py \
            unittest/test_gemm.py \
            unittest/test_q4_k_q8_k_matvec.py \
            unittest/test_gemm_fused.py \
            unittest/test_gemm_microkernel.py \
            unittest/test_mlp.py \
            unittest/test_rmsnorm.py \
            unittest/test_swiglu.py \
            unittest/test_fused_swiglu_decode.py \
            unittest/test_fused_attention_decode.py \
            unittest/test_mega_fused_attention.py \
            unittest/test_sigmoid.py \
            unittest/test_relu.py \
            unittest/test_attention.py \
            unittest/test_attention_backward.py \
            unittest/test_kv_cache_attention.py \
            unittest/test_kv_cache_layer_decode.py \
            unittest/test_rope.py \
            unittest/test_embedding.py \
            unittest/test_cross_entropy.py \
            unittest/test_orchestration_layer.py \
            unittest/test_lm_head_litmus.py \
            unittest/test_optimizer.py \
            unittest/test_gemv_kernels_comprehensive.py

PY_TESTS_BF16 := unittest/bf16/test_sigmoid_bf16.py \
                unittest/bf16/test_rmsnorm_bf16.py \
                unittest/bf16/test_mlp_bf16.py \
                unittest/bf16/test_attention_bf16.py \
                unittest/bf16/test_gelu_bf16.py \
                unittest/bf16/test_layernorm_bf16.py \
                unittest/bf16/test_rope_bf16.py \
                unittest/bf16/test_relu_bf16.py \
                unittest/bf16/test_swiglu_bf16.py \
                unittest/bf16/test_embedding_bf16.py \
                unittest/bf16/test_cross_entropy_bf16.py

LITMUS_DEMO_ARGS ?= --vocab 100 --ctx 100 --embed 64 --intermediate 128 --heads 4 --kv-heads 2
LITMUS_DEMO_SVG ?= $(BUILD_DIR)/litmus_report.svg
LITMUS_DEMO_LOG ?= $(BUILD_DIR)/litmus_demo.log
CK_GELU_TOL ?= 1e-7
TEST_ENV :=
ifneq (,$(findstring icx,$(CC)))
CK_GELU_TOL := 1e-6
TEST_ENV += CK_GELU_TOL=$(CK_GELU_TOL)
endif
ifneq (,$(findstring icc,$(CC)))
CK_GELU_TOL := 1e-6
TEST_ENV += CK_GELU_TOL=$(CK_GELU_TOL)
endif
export CK_GELU_TOL

# MKL compatibility: Force SSE code paths on CPUs without AVX-512
# MKL may incorrectly detect AVX-512 support based on processor family
ifdef USE_MKL
ifeq (,$(findstring avx512f,$(CPU_FLAGS)))
TEST_ENV += MKL_DEBUG_CPU_TYPE=5
endif
endif

FLASH_ATTN_LONG_CONTEXTS ?= 51200
FLASH_ATTN_LONG_HEADS ?= 32
FLASH_ATTN_LONG_KV_HEADS ?= $(FLASH_ATTN_LONG_HEADS)
FLASH_ATTN_TILE_K ?= 32
FLASH_ATTN_FAST_EXP ?= 1
FLASH_ATTN_OMP_THREADS ?= $(shell nproc 2>/dev/null || echo 1)
FLASH_ATTN_LLAMA_PERF ?= 1
FLASH_ATTN_LLAMA_PERF_MAX_TK ?= 51200

all: $(BUILD_DIR) $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_STAMP): | $(BUILD_DIR)
	@printf 'CC=%s\nCFLAGS=%s\n' "$(CC)" "$(CFLAGS)" > $@.tmp
	@if [ ! -f $@ ] || ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm $@.tmp; fi

$(LIB): $(BUILD_STAMP) $(SRCS)
	$(CC) $(CFLAGS) -shared -o $@ $(SRCS) $(LDFLAGS) -lm

$(IR_DEMO): $(BUILD_DIR) src/ckernel_ir.c src/ckernel_ir_demo.c src/ckernel_codegen.c src/ckernel_kernel_specs.c src/ckernel_registry.c include/ckernel_ir.h include/ckernel_codegen.h include/ckernel_registry.h include/ckernel_kernel_specs.h
	$(CC) -O2 -Wall -Iinclude -o $@ src/ckernel_ir.c src/ckernel_codegen.c src/ckernel_kernel_specs.c src/ckernel_registry.c src/ckernel_ir_demo.c

$(IR_V2_DEMO): $(BUILD_DIR) src/ckernel_ir.c src/ckernel_ir_v2.c src/ckernel_ir_v2_builder.c src/ckernel_ir_v2_lower.c src/ckernel_ir_v2_demo.c src/ckernel_codegen_v2.c src/ckernel_codegen_v2_struct.c src/ckernel_codegen_v2_dispatch.c src/ckernel_codegen_v2_schedule.c src/ckernel_codegen_v2_sections.c src/ckernel_mem_plan.c src/ckernel_kernel_specs.c include/ckernel_ir.h include/ckernel_ir_v2.h include/ckernel_ir_v2_lower.h include/ckernel_codegen_v2.h include/ckernel_mem_plan.h include/ckernel_kernel_specs.h
	$(CC) -O2 -Wall -Iinclude -o $@ src/ckernel_ir.c src/ckernel_ir_v2.c src/ckernel_ir_v2_builder.c src/ckernel_ir_v2_lower.c src/ckernel_codegen_v2.c src/ckernel_codegen_v2_struct.c src/ckernel_codegen_v2_dispatch.c src/ckernel_codegen_v2_schedule.c src/ckernel_codegen_v2_sections.c src/ckernel_mem_plan.c src/ckernel_kernel_specs.c src/ckernel_ir_v2_demo.c

ck: $(IR_DEMO)
	@echo "Running $(IR_DEMO) with $(DEFAULT_CONFIG)..."
	./$(IR_DEMO) $(DEFAULT_CONFIG)

ck-v2: $(IR_V2_DEMO)
	@echo "Running $(IR_V2_DEMO) with $(DEFAULT_CONFIG)..."
	./$(IR_V2_DEMO) $(DEFAULT_CONFIG)

ck_V2: ck-v2

# Tokenizer library
$(LIB_TOKENIZER): $(SRCS_TOKENIZER)
	$(CC) $(CFLAGS) -shared -o $@ $(SRCS_TOKENIZER) -lm

tokenizer: $(LIB_TOKENIZER)
	@echo "Tokenizer library built: $(LIB_TOKENIZER)"
	@true

# Tokenizer tests (unified test suite)
test-tokenizer: $(LIB_TOKENIZER)
	@echo ""
	@echo "========================================"
	@echo "  Running Unified Tokenizer Tests"
	@echo "========================================"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_tokenizer_unified.py

test-tokenizer-quick: $(LIB_TOKENIZER)
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_tokenizer_unified.py --quick

test-tokenizer-llama: $(LIB_TOKENIZER)
	@echo ""
	@echo "========================================"
	@echo "  C-Kernel vs llama.cpp Comparison"
	@echo "========================================"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_tokenizer_llamacpp.py

.PHONY: tokenizer test-tokenizer test-tokenizer-quick test-tokenizer-llama

# ═══════════════════════════════════════════════════════════════════════════════
# MEGA-FUSED ATTENTION TESTS (DRAM Pressure & Flamegraph)
# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Test order is ENFORCED - correctness first, then performance!
#
#   1. CORRECTNESS  →  Numerical parity vs PyTorch (MUST PASS FIRST!)
#   2. PERFORMANCE  →  DRAM pressure reduction (perf)
#   3. FLAMEGRAPH   →  Visual confirmation
#
# If correctness fails, performance/flamegraph are skipped!
#
# Key metrics:
#   - cache-misses: LLC misses = DRAM access
#   - LLC-load-misses: Requests that go to DRAM
#   - Expected: 10-100x reduction in DRAM traffic
#
# Run with:
#   make test-mega-fused-correctness    # Step 1: Numerical correctness
#   make test-mega-fused-perf           # Step 2: DRAM pressure (requires correctness)
#   make test-mega-fused-flamegraph     # Step 3: Visual (requires correctness)
#   make test-mega-fused                # All 3 steps in order
# ═══════════════════════════════════════════════════════════════════════════════

TEST_MODEL ?= Qwen2-0.5B-Instruct
TEST_TOKENS ?= 100

test-mega-fused-correctness: $(CK_CLI_V6_5)
	@echo ""
	@echo "========================================"
	@echo "  STEP 1: NUMERICAL CORRECTNESS"
	@echo "  (Must pass before performance tests!)"
	@echo "========================================"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_mega_fused_attention.py --correctness

test-mega-fused-perf: test-mega-fused-correctness $(CK_CLI_V6_5)
	@echo ""
	@echo "========================================"
	@echo "  STEP 2: DRAM PRESSURE TEST"
	@echo "  (THE CRITICAL TEST - fusion's whole point!)"
	@echo "========================================"
	@mkdir -p test_results
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_mega_fused_attention.py --perf --model $(TEST_MODEL) --tokens $(TEST_TOKENS)

test-mega-fused-flamegraph: test-mega-fused-correctness $(CK_CLI_V6_5)
	@echo ""
	@echo "========================================"
	@echo "  STEP 3: FLAMEGRAPH VISUALIZATION"
	@echo "  (Visual confirmation of reduced memory)"
	@echo "========================================"
	@mkdir -p test_results
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_mega_fused_attention.py --flamegraph --model $(TEST_MODEL) --tokens $(TEST_TOKENS)

test-mega-fused: $(CK_CLI_V6_5)
	@echo ""
	@echo "========================================"
	@echo "  MEGA-FUSED ATTENTION: ALL TESTS"
	@echo "  (Correctness → Performance → Flamegraph)"
	@echo "========================================"
	@mkdir -p test_results
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_mega_fused_attention.py --all --model $(TEST_MODEL) --tokens $(TEST_TOKENS)

# Direct perf measurement (without Python)
test-perf-baseline:
	@echo "Measuring baseline (unfused) DRAM pressure..."
	@perf stat -e cycles,instructions,cache-references,cache-misses,LLC-loads \
		-o test_results/perf_baseline.txt -- \
		./$(CK_CLI_V6_5) --model $(TEST_MODEL) --max-tokens $(TEST_TOKENS) --prompt "Test"

test-perf-megafused:
	@echo "Measuring mega-fused DRAM pressure..."
	@perf stat -e cycles,instructions,cache-references,cache-misses,LLC-loads \
		-o test_results/perf_megafused.txt -- \
		./$(CK_CLI_V6_5) --model $(TEST_MODEL) --max-tokens $(TEST_TOKENS) --mega-fused --prompt "Test"

test-perf-compare: test-perf-baseline test-perf-megafused
	@echo ""
	@echo "Comparing results..."
	@python3 scripts/test/compare_perf_results.py test_results/perf_baseline.txt test_results/perf_megafused.txt

.PHONY: test-mega-fused-correctness test-mega-fused-perf test-mega-fused-flamegraph test-mega-fused
.PHONY: test-perf-baseline test-perf-megafused test-perf-compare

ir-v2-meta: $(IR_V2_DEMO)
	@echo "Generating IR v2 from $(CONFIG) + $(IR_V2_META)..."
	./$(IR_V2_DEMO) $(CONFIG) --meta $(IR_V2_META)

ir-v2:
	@echo "Generating IR v2 from $(CONFIG) -> $(IR_V2_OUT)..."
	@if [ -n "$(IR_V2_HF)" ]; then \
	  $(PYTHON) $(PYTHONFLAGS) $(IR_V2_SCRIPT) --hf $(IR_V2_HF) --revision $(IR_V2_REV) --out $(IR_V2_OUT) \
	    $(if $(IR_V2_WEIGHTS),--weights $(IR_V2_WEIGHTS),) \
	    $(if $(IR_V2_META),--meta-out $(IR_V2_META),) \
	    $(if $(IR_V2_CTX),--ctx $(IR_V2_CTX),) \
	    $(if $(IR_V2_KERNEL_DTYPE),--kernel-dtype $(IR_V2_KERNEL_DTYPE),) \
	    $(if $(IR_V2_ACT_DTYPE),--activation-dtype $(IR_V2_ACT_DTYPE),); \
	else \
	  $(PYTHON) $(PYTHONFLAGS) $(IR_V2_SCRIPT) --config $(CONFIG) --out $(IR_V2_OUT) \
	    $(if $(IR_V2_WEIGHTS),--weights $(IR_V2_WEIGHTS),) \
	    $(if $(IR_V2_META),--meta-out $(IR_V2_META),) \
	    $(if $(IR_V2_CTX),--ctx $(IR_V2_CTX),) \
	    $(if $(IR_V2_KERNEL_DTYPE),--kernel-dtype $(IR_V2_KERNEL_DTYPE),) \
	    $(if $(IR_V2_ACT_DTYPE),--activation-dtype $(IR_V2_ACT_DTYPE),); \
	fi

fetch-v2:
	@echo "Fetching config/weights meta -> $(IR_V2_META)..."
	@if [ -n "$(IR_V2_HF)" ]; then \
	  $(PYTHON) $(PYTHONFLAGS) $(IR_V2_SCRIPT) --hf $(IR_V2_HF) --revision $(IR_V2_REV) --meta-out $(IR_V2_META) --meta-only \
	    $(if $(IR_V2_WEIGHTS),--weights $(IR_V2_WEIGHTS),) \
	    --cache-dir $(BUILD_DIR); \
	else \
	  $(PYTHON) $(PYTHONFLAGS) $(IR_V2_SCRIPT) --config $(CONFIG) --meta-out $(IR_V2_META) --meta-only \
	    $(if $(IR_V2_WEIGHTS),--weights $(IR_V2_WEIGHTS),) \
	    --cache-dir $(BUILD_DIR); \
	fi

# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY V4 PIPELINE (Deprecated - use v5 instead)
# ═══════════════════════════════════════════════════════════════════════════════
# Note: v4 targets are kept for backwards compatibility but v5 is recommended.
# v5 provides: explicit per-layer functions, better debugging, parity testing.
# See: make v5 for documentation on the new pipeline.
# ═══════════════════════════════════════════════════════════════════════════════

ir-v4:
	@echo "Generating IR v4 into $(IR_V4_PREFIX)..."
	@if [ -n "$(IR_V4_PRESET)" ]; then \
	  $(PYTHON) $(PYTHONFLAGS) $(IR_V4_SCRIPT) --preset="$(IR_V4_PRESET)" \
	    $(if $(IR_V4_PREFIX),--prefix="$(IR_V4_PREFIX)",) \
	    $(if $(IR_V4_TOKENS),--tokens="$(IR_V4_TOKENS)",) \
	    $(if $(IR_V4_MODES),--modes="$(IR_V4_MODES)",) \
	    $(if $(IR_V4_EMIT),--emit="$(IR_V4_EMIT)",) \
	    $(if $(IR_V4_DTYPE),--dtype="$(IR_V4_DTYPE)",) \
	    $(if $(IR_V4_WEIGHTS_HEADER),--weights-header="$(IR_V4_WEIGHTS_HEADER)",) \
	    $(if $(IR_V4_WEIGHTS_INDEX),--weights-index="$(IR_V4_WEIGHTS_INDEX)",) \
	    $(if $(IR_V4_KERNEL_SPECS),--kernel-specs="$(IR_V4_KERNEL_SPECS)",); \
	elif [ -n "$(IR_V4_CONFIG)" ]; then \
	  $(PYTHON) $(PYTHONFLAGS) $(IR_V4_SCRIPT) --config="$(IR_V4_CONFIG)" \
	    $(if $(IR_V4_PREFIX),--prefix="$(IR_V4_PREFIX)",) \
	    $(if $(IR_V4_TOKENS),--tokens="$(IR_V4_TOKENS)",) \
	    $(if $(IR_V4_MODES),--modes="$(IR_V4_MODES)",) \
	    $(if $(IR_V4_EMIT),--emit="$(IR_V4_EMIT)",) \
	    $(if $(IR_V4_DTYPE),--dtype="$(IR_V4_DTYPE)",) \
	    $(if $(IR_V4_WEIGHTS_HEADER),--weights-header="$(IR_V4_WEIGHTS_HEADER)",) \
	    $(if $(IR_V4_WEIGHTS_INDEX),--weights-index="$(IR_V4_WEIGHTS_INDEX)",) \
	    $(if $(IR_V4_KERNEL_SPECS),--kernel-specs="$(IR_V4_KERNEL_SPECS)",); \
	elif [ -n "$(IR_V4_MODEL)" ]; then \
	  $(PYTHON) $(PYTHONFLAGS) $(IR_V4_SCRIPT) "$(IR_V4_MODEL)" \
	    $(if $(IR_V4_PREFIX),--prefix="$(IR_V4_PREFIX)",) \
	    $(if $(IR_V4_TOKENS),--tokens="$(IR_V4_TOKENS)",) \
	    $(if $(IR_V4_MODES),--modes="$(IR_V4_MODES)",) \
	    $(if $(IR_V4_EMIT),--emit="$(IR_V4_EMIT)",) \
	    $(if $(IR_V4_DTYPE),--dtype="$(IR_V4_DTYPE)",) \
	    $(if $(IR_V4_WEIGHTS_HEADER),--weights-header="$(IR_V4_WEIGHTS_HEADER)",) \
	    $(if $(IR_V4_WEIGHTS_INDEX),--weights-index="$(IR_V4_WEIGHTS_INDEX)",) \
	    $(if $(IR_V4_KERNEL_SPECS),--kernel-specs="$(IR_V4_KERNEL_SPECS)",); \
	else \
	  echo "Usage: make ir-v4 IR_V4_PRESET=name | IR_V4_CONFIG=path | IR_V4_MODEL=hf_id_or_url"; \
	  exit 1; \
	fi

ir-v4-q4k:
	@echo "Generating IR v4 (Q4_K weights) ..."
	@if [ -n "$(IR_V4_Q4K_CHECKPOINT)" ]; then \
	  $(PYTHON) $(PYTHONFLAGS) $(IR_V4_Q4K_SCRIPT) --checkpoint="$(IR_V4_Q4K_CHECKPOINT)" \
	    $(if $(IR_V4_Q4K_PRESET),--preset="$(IR_V4_Q4K_PRESET)",) \
	    $(if $(IR_V4_Q4K_CONFIG),--config="$(IR_V4_Q4K_CONFIG)",) \
	    $(if $(IR_V4_Q4K_PREFIX),--prefix="$(IR_V4_Q4K_PREFIX)",) \
	    $(if $(IR_V4_Q4K_MODES),--modes="$(IR_V4_Q4K_MODES)",) \
	    $(if $(IR_V4_Q4K_TOKENS),--tokens="$(IR_V4_Q4K_TOKENS)",) \
	    $(if $(IR_V4_Q4K_CONTEXT),--context="$(IR_V4_Q4K_CONTEXT)",) \
	    $(if $(IR_V4_Q4K_FUSION),--fusion="$(IR_V4_Q4K_FUSION)",) \
	    $(if $(IR_V4_Q4K_EMIT),--emit="$(IR_V4_Q4K_EMIT)",) \
	    $(if $(IR_V4_Q4K_VERBOSE),--verbose,); \
	elif [ -n "$(IR_V4_Q4K_GGUF)" ]; then \
	  $(PYTHON) $(PYTHONFLAGS) $(IR_V4_Q4K_SCRIPT) --gguf="$(IR_V4_Q4K_GGUF)" \
	    $(if $(IR_V4_Q4K_CONFIG),--config="$(IR_V4_Q4K_CONFIG)",) \
	    $(if $(IR_V4_Q4K_PREFIX),--prefix="$(IR_V4_Q4K_PREFIX)",) \
	    $(if $(IR_V4_Q4K_MODES),--modes="$(IR_V4_Q4K_MODES)",) \
	    $(if $(IR_V4_Q4K_TOKENS),--tokens="$(IR_V4_Q4K_TOKENS)",) \
	    $(if $(IR_V4_Q4K_CONTEXT),--context="$(IR_V4_Q4K_CONTEXT)",) \
	    $(if $(IR_V4_Q4K_FUSION),--fusion="$(IR_V4_Q4K_FUSION)",) \
	    $(if $(IR_V4_Q4K_EMIT),--emit="$(IR_V4_Q4K_EMIT)",) \
	    $(if $(IR_V4_Q4K_VERBOSE),--verbose,); \
	else \
	  echo "Usage: make ir-v4-q4k IR_V4_Q4K_CHECKPOINT=dir | IR_V4_Q4K_GGUF=path"; \
	  exit 1; \
	fi

emit: $(IR_DEMO)
	@echo "Generating runtime from $(CONFIG) -> $(OUT)..."
	./$(IR_DEMO) $(CONFIG) --emit $(OUT)

emit-v2: $(IR_V2_DEMO)
	@if [ -n "$(IR)" ]; then \
	  echo "Generating v2 runtime from $(IR) -> $(OUT)..."; \
	  ./$(IR_V2_DEMO) --ir $(IR) --emit $(OUT); \
	else \
	  echo "Generating v2 runtime from $(CONFIG) -> $(OUT)..."; \
	  ./$(IR_V2_DEMO) $(CONFIG) --emit $(OUT); \
	fi

ck-emit-v2: emit-v2
	@true

ck-emit: emit
	@true

$(LIB_GELU): $(BUILD_STAMP) src/kernels/gelu_kernels.c src/kernels/gelu_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/gelu_kernels.c src/kernels/gelu_kernels_bf16.c -lm

$(LIB_RMSNORM): $(BUILD_STAMP) src/kernels/rmsnorm_kernels.c src/kernels/rmsnorm_kernels_bf16.c src/kernels/rmsnorm_kernels_int8.c src/kernels/rmsnorm_kernels_int4.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/rmsnorm_kernels.c src/kernels/rmsnorm_kernels_bf16.c src/kernels/rmsnorm_kernels_int8.c src/kernels/rmsnorm_kernels_int4.c -lm

$(LIB_LN): $(BUILD_STAMP) src/kernels/layernorm_kernels.c src/kernels/layernorm_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/layernorm_kernels.c src/kernels/layernorm_kernels_bf16.c -lm

$(LIB_SOFT): $(BUILD_STAMP) src/kernels/softmax_kernels.c src/kernels/softmax_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/softmax_kernels.c src/kernels/softmax_kernels_bf16.c -lm

$(LIB_SWIGLU): $(BUILD_STAMP) src/kernels/swiglu_kernels.c src/kernels/swiglu_kernels_bf16.c src/kernels/sigmoid_kernels.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/swiglu_kernels.c src/kernels/swiglu_kernels_bf16.c src/kernels/sigmoid_kernels.c -lm

$(LIB_SIGMOID): $(BUILD_STAMP) src/kernels/sigmoid_kernels.c src/kernels/sigmoid_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/sigmoid_kernels.c src/kernels/sigmoid_kernels_bf16.c -lm

$(LIB_RELU): $(BUILD_STAMP) src/kernels/relu_kernels.c src/kernels/relu_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/relu_kernels.c src/kernels/relu_kernels_bf16.c -lm

$(LIB_VISION): $(BUILD_STAMP) src/kernels/vision_kernels.c src/kernels/vision_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/vision_kernels.c src/kernels/vision_kernels_bf16.c -lm

$(LIB_ATTENTION): $(BUILD_STAMP) src/kernels/attention_kernels.c src/kernels/attention_flash_true.c src/kernels/softmax_kernels.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/attention_kernels.c src/kernels/attention_flash_true.c src/kernels/softmax_kernels.c -lm

$(LIB_ROPE): $(BUILD_STAMP) src/kernels/rope_kernels.c src/kernels/rope_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/rope_kernels.c src/kernels/rope_kernels_bf16.c -lm

$(LIB_QUANT): $(BUILD_STAMP) src/kernels/dequant_kernels.c src/kernels/gemm_kernels_q4_0.c src/kernels/gemm_kernels_q4_1.c src/kernels/gemm_kernels_q5_0.c src/kernels/gemm_kernels_q5_0_sse_v2.c src/kernels/gemm_kernels_q5_1.c src/kernels/gemm_kernels_q4k.c src/kernels/gemm_kernels_q6k.c src/kernels/gemm_kernels_q4k_q8k.c src/kernels/gemm_kernels_q4k_sse.c src/kernels/gemm_kernels_q4k_q8k_avx2.c src/kernels/gemm_kernels_q4k_q8k_vnni.c src/kernels/gemm_kernels_q8_0.c src/kernels/gemm_kernels_f16.c src/kernels/quantize_row_q8_k_sse.c include/ckernel_quant.h include/ckernel_dtype.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/dequant_kernels.c src/kernels/gemm_kernels_q4_0.c src/kernels/gemm_kernels_q4_1.c src/kernels/gemm_kernels_q5_0.c src/kernels/gemm_kernels_q5_0_sse_v2.c src/kernels/gemm_kernels_q5_1.c src/kernels/gemm_kernels_q4k.c src/kernels/gemm_kernels_q6k.c src/kernels/gemm_kernels_q4k_q8k.c src/kernels/gemm_kernels_q4k_sse.c src/kernels/gemm_kernels_q4k_q8k_avx2.c src/kernels/gemm_kernels_q4k_q8k_vnni.c src/kernels/gemm_kernels_q8_0.c src/kernels/gemm_kernels_f16.c src/kernels/quantize_row_q8_k_sse.c -lm

# Convenience alias targets so existing commands still work.
libckernel_gelu.so: $(LIB_GELU)
	@true

libckernel_rmsnorm.so: $(LIB_RMSNORM)
	@true

libckernel_layernorm.so: $(LIB_LN)
	@true

libckernel_softmax.so: $(LIB_SOFT)
	@true

libckernel_swiglu.so: $(LIB_SWIGLU)
	@true

libckernel_sigmoid.so: $(LIB_SIGMOID)
	@true

libckernel_attention.so: $(LIB_ATTENTION)
	@true

libckernel_relu.so: $(LIB_RELU)
	@true

libckernel_vision.so: $(LIB_VISION)
	@true

test-relu: $(LIB_RELU)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_relu.py

test-vision: $(LIB_VISION)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_vision.py

libckernel_rope.so: $(LIB_ROPE)
	@true

libckernel_quant.so: $(LIB_QUANT)
	@true

test-libs: $(LIB_GELU) $(LIB_RMSNORM) $(LIB_LN) $(LIB_SOFT) $(LIB_SWIGLU) $(LIB_SIGMOID) $(LIB_ATTENTION) $(LIB_ROPE) $(LIB_RELU) $(LIB_VISION) $(LIB_QUANT) $(LIB_PARITY)

test-quant: $(LIB_QUANT)
	@set -e; \
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_quant_kernels.py; \
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_q4_k_q8_k_matvec.py; \
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_q6k_kernels.py; \
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_q4_k_quantize.py

unittest:
	@echo ""
	@echo "=========================================="
	@echo "  C-Kernel-Engine Unit Tests"
	@echo "=========================================="
	@echo ""
	@echo "Run with: python3 <script> or make test/nightly"
	@echo ""
	@echo "GEMM Backends:"
	@echo "  - MKL (USE_MKL=1):     Intel MKL cblas_sgemm - best for FP32 training"
	@echo "  - Native (default):   Custom AVX/AVX2/AVX-512 SIMD kernels"
	@echo "  - Quantized:          Custom Q4_K/Q6_K/Q8_K kernels (no MKL needed)"
	@echo ""
	@echo "FP32 Kernel Tests:"
	@echo "  unittest/test_gelu.py                  - GELU forward/backward"
	@echo "  unittest/test_rmsnorm.py               - RMSNorm forward/backward"
	@echo "  unittest/test_sigmoid.py               - Sigmoid forward/backward"
	@echo "  unittest/test_layernorm.py             - LayerNorm forward/backward"
	@echo "  unittest/test_softmax.py               - Causal softmax forward"
	@echo "  unittest/test_softmax_backward.py      - Causal softmax backward"
	@echo "  unittest/test_gemm.py                  - GEMM variants vs PyTorch"
	@echo "  unittest/test_mlp.py                   - MLP block forward/backward"
	@echo "  unittest/test_swiglu.py                - SwiGLU activation"
	@echo "  unittest/test_relu.py                  - ReLU activation"
	@echo "  unittest/test_attention.py             - Attention forward/backward"
	@echo "  unittest/test_rope.py                  - RoPE forward/backward"
	@echo ""
	@echo "Orchestration Tests:"
	@echo "  unittest/test_orchestration_layer.py   - Full layer forward (GQA/MHA)"
	@echo "  unittest/test_fused_swiglu_decode.py   - Fused SwiGLU decode MLP"
	@echo "  unittest/test_fused_attention_decode.py - Fused attention decode"
	@echo ""
	@echo "Quantization Tests:"
	@echo "  unittest/test_q4k_kernels.py           - Q4_K dequant + matvec"
	@echo "  unittest/test_q6k_kernels.py           - Q6_K dequant + matvec"
	@echo "  unittest/test_q4_k_quantize.py         - Q4_K quantization"
	@echo "  unittest/test_q4k_q8k_matvec.py        - Q4_K x Q8_K matmul"
	@echo "  unittest/test_v4_conversion.py         - GGUF -> bump v4 conversion sanity (requires GGUF)"
	@echo ""
	@echo "BF16 Tests:"
	@echo "  unittest/bf16/test_sigmoid_bf16.py     - BF16 sigmoid"
	@echo "  unittest/bf16/test_rmsnorm_bf16.py     - BF16 RMSNorm"
	@echo "  unittest/bf16/test_mlp_bf16.py         - BF16 MLP"
	@echo "  unittest/bf16/test_attention_bf16.py   - BF16 attention"
	@echo ""
	@echo "Tokenizer Tests:"
	@echo "  unittest/test_tokenizer_unified.py     - Unified tokenizer test suite"
	@echo "    - Foundation Tests: Custom vocab, UTF-8, emojis"
	@echo "    - True BPE Parity: 100% HuggingFace parity"
	@echo "    - Performance: Hash vs Trie vs Python"
	@echo ""
	@echo "Batch Commands:"
	@echo "  make test             - Run core kernel tests"
	@echo "  make test-bf16        - Run BF16 kernel tests"
	@echo "  make test-tokenizer   - Run tokenizer tests"
	@echo "  make nightly          - Run full nightly test suite (scripts/nightly_runner.py)"
	@echo ""

# Typo aliases
uittest: unittest
unittest-show: unittest
show_test: showtests

# Show all test-related make targets
showtests:
	@echo ""
	@echo "=========================================="
	@echo "  C-Kernel-Engine Test Targets"
	@echo "=========================================="
	@echo ""
	@echo "Quick Commands:"
	@echo "  make test             Run core kernel tests"
	@echo "  make test-bf16        Run BF16 kernel tests"
	@echo "  make test-quant       Run quantization kernel tests"
	@echo "  make test-flash-attention  Run flash attention tests (50K+ contexts)"
	@echo "  make nightly          Run full test suite (doesn't stop on failure)"
	@echo ""
	@echo "Per-Kernel Libraries:"
	@echo "  make test-libs        Build per-kernel .so files for testing"
	@echo "  make test-relu        Run isolated ReLU C kernel tests (with PyTorch parity)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  QUANTIZATION TESTS (All Formats)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Legacy Quant Formats:"
	@echo "  unittest/test_q4_0_kernels.py     Q4_0 (4-bit, simple block)"
	@echo "  unittest/test_q4_1_kernels.py     Q4_1 (4-bit with min)"
	@echo "  unittest/test_q5_0_kernels.py     Q5_0 (5-bit, simple block)"
	@echo "  unittest/test_q5_1_kernels.py     Q5_1 (5-bit with min)"
	@echo "  unittest/test_q8_0_kernels.py     Q8_0 (8-bit, simple block)"
	@echo ""
	@echo "K-Quant Formats (Recommended):"
	@echo "  unittest/test_q4k_kernels.py      Q4_K dequant/quantize/gemv"
	@echo "  unittest/test_q6k_kernels.py      Q6_K dequant/quantize"
	@echo "  unittest/test_q4_k_quantize.py    Q4_K quantization"
	@echo "  unittest/test_q4_k_q8_k_matvec.py Q4_K x Q8_K matrix-vector"
	@echo "  unittest/test_quant_kernels.py    General quant kernel tests"
	@echo ""
	@echo "Batch Commands:"
	@echo "  make test-quant       Run all quantization tests"
	@echo "  make test-quant-server  Run on VNNI server (AVX-512)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  PARITY TESTS"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "PyTorch Parity:"
	@echo "  make litmus           LM head + cross-entropy backward parity"
	@echo "  make litmus-demo      100x100 litmus demo with SVG output"
	@echo "  make layer-parity     Full layer forward parity (GQA/MHA)"
	@echo "  make layer-parity-scalar  Layer parity with scalar build"
	@echo "  make tiny-parity      Tiny end-to-end training parity (1 step)"
	@echo ""
	@echo "llama.cpp Parity (Q4_K kernel validation):"
	@echo "  make llamacpp-parity      Quick parity vs llama.cpp/ggml"
	@echo "  make llamacpp-parity-full Full parity test (all kernels)"
	@echo "  Note: Requires llama.cpp submodule (git submodule update --init)"
	@echo ""
	@echo "End-to-End Tests:"
	@echo "  make tiny-e2e         Random weights + tiny forward pass"
	@echo "  make tiny-train       Random weights + forward/backward/SGD"
	@echo ""
	@echo "SmolLM Tests (requires HuggingFace weights):"
	@echo "  make smollm-demo      Tiny SmolLM training demo"
	@echo "  make smollm-forward   Forward parity vs PyTorch"
	@echo "  make smollm-layer-diff  Per-stage diffs for one layer"
	@echo "  make smollm-layer-stack  Per-layer outputs across full stack"
	@echo "  make smollm-train-parity  Full forward+backward parity"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  COMPREHENSIVE TEST SUITES"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  make all-tests        Kernel tests + layer parity + tiny parity"
	@echo "  make test-quick       Quick tests (<1 min)"
	@echo "  make test-full        Full tests (5-10 min)"
	@echo "  make test-stress      Stress tests (10+ min)"
	@echo ""
	@echo "Nightly / CI:"
	@echo "  make nightly          Run all tests with summary"
	@echo "  make nightly-quick    Quick subset (~5 min)"
	@echo "  make nightly-json     Run all + JSON report"
	@echo "  make nightly-kernels  Only kernel tests"
	@echo "  make nightly-bf16     Only BF16 tests"
	@echo "  make nightly-quant    Only quantization tests"
	@echo "  make nightly-parity   Only parity tests (PyTorch + llama.cpp)"
	@echo ""
	@echo "For Python unittest scripts: make unittest"
	@echo ""

show-tests: showtests

gguf-inspect:
	@if [ -z "$(GGUF)" ]; then \
	  echo "Usage: make gguf-inspect GGUF=/path/to/model.gguf"; \
	  exit 2; \
	fi
	@$(PYTHON) $(PYTHONFLAGS) scripts/convert_gguf_to_bump.py --gguf "$(GGUF)" --inspect

gguf-list:
	@if [ -z "$(GGUF)" ]; then \
	  echo "Usage: make gguf-list GGUF=/path/to/model.gguf"; \
	  exit 2; \
	fi
	@$(PYTHON) $(PYTHONFLAGS) scripts/convert_gguf_to_bump.py --gguf "$(GGUF)" --list

gguf-to-bump:
	@if [ -z "$(GGUF)" ]; then \
	  echo "Usage: make gguf-to-bump GGUF=/path/to/model.gguf [GGUF_OUT=$(GGUF_OUT)] [GGUF_CONFIG_OUT=$(GGUF_CONFIG_OUT)] [GGUF_CONTEXT=<n>]"; \
	  exit 2; \
	fi
	@$(PYTHON) $(PYTHONFLAGS) scripts/convert_gguf_to_bump.py \
	  --gguf "$(GGUF)" \
	  --output "$(GGUF_OUT)" \
	  $(if $(GGUF_CONFIG_OUT),--config-out "$(GGUF_CONFIG_OUT)") \
	  $(if $(GGUF_CONTEXT),--context "$(GGUF_CONTEXT)")

gguf-to-bump-v4:
	@if [ -z "$(GGUF)" ]; then \
	  echo "Usage: make gguf-to-bump-v4 GGUF=/path/to/model.gguf [GGUF_V4_OUT=$(GGUF_V4_OUT)] [GGUF_V4_CONFIG_OUT=$(GGUF_V4_CONFIG_OUT)] [GGUF_V4_CONTEXT=<n>]"; \
	  exit 2; \
	fi
	@$(PYTHON) $(PYTHONFLAGS) scripts/v4/convert_gguf_to_bump_v4.py \
	  --gguf "$(GGUF)" \
	  --output "$(GGUF_V4_OUT)" \
	  $(if $(GGUF_V4_CONFIG_OUT),--config-out "$(GGUF_V4_CONFIG_OUT)") \
	  $(if $(GGUF_V4_CONTEXT),--context "$(GGUF_V4_CONTEXT)")

hf-to-bump-v4:
	@if [ -z "$(HF_V4_CHECKPOINT)" ]; then \
	  echo "Usage: make hf-to-bump-v4 HF_V4_CHECKPOINT=/path/to/hf_model [HF_V4_OUT=$(HF_V4_OUT)] [HF_V4_DTYPE=q4_k] [HF_V4_CONTEXT=<n>]"; \
	  exit 2; \
	fi
	@$(PYTHON) $(PYTHONFLAGS) scripts/v4/convert_hf_to_bump_v4.py \
	  --checkpoint "$(HF_V4_CHECKPOINT)" \
	  --output "$(HF_V4_OUT)" \
	  $(if $(HF_V4_CONFIG),--config "$(HF_V4_CONFIG)") \
	  $(if $(HF_V4_DTYPE),--dtype "$(HF_V4_DTYPE)") \
	  $(if $(HF_V4_CONTEXT),--context "$(HF_V4_CONTEXT)") \
	  $(if $(HF_V4_MAP_OUT),--map-out "$(HF_V4_MAP_OUT)")

test: $(LIB) test-libs
	@set -e; \
	for t in $(PY_TESTS); do \
	  echo "Running $$t"; \
	  extra_args=""; \
	  case "$$t" in *test_gemm_microkernel.py|*test_gemv_kernels_comprehensive.py) extra_args="--quick";; esac; \
	  LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(TEST_ENV) $(PYTHON) $(PYTHONFLAGS) $$t $$extra_args; \
	done; \
	echo "All Python kernel tests completed."

# Run full benchmarks (including GEMM microkernel performance tests)
test-bench: $(LIB) test-libs
	@echo "Running GEMM microkernel benchmarks (this may take a few minutes)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_gemm_microkernel.py

test-bf16: $(LIB) test-libs
	@failed=0; \
	for t in $(PY_TESTS_BF16); do \
	  echo "Running $$t"; \
	  if ! LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(TEST_ENV) $(PYTHON) $(PYTHONFLAGS) $$t; then \
	    failed=1; \
	  fi; \
	done; \
	if [ $$failed -ne 0 ]; then \
	  echo "BF16 Python kernel tests failed."; \
	  exit 1; \
	fi; \
	echo "BF16 Python kernel tests completed."

test-v4-q4k:
	@if [ -z "$(GGUF_PATH)" ]; then \
	  echo "Usage: make test-v4-q4k GGUF_PATH=/path/to/model.gguf [V4_LAYERS=1,2] [V4_VALIDATE=1] [V4_FULL=1]"; \
	  exit 2; \
	fi
	@$(PYTHON) $(PYTHONFLAGS) scripts/test_v4_q4k_pipeline.py \
	  --gguf "$(GGUF_PATH)" \
	  $(if $(V4_LAYERS),--layers "$(V4_LAYERS)") \
	  $(if $(V4_VALIDATE),--validate) \
	  $(if $(V4_FULL),--full)

test-flash-attention: $(LIB)
	@echo "Running flash attention tests (long context enforced)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		CK_FLASH_ATTN_LONG_CONTEXTS=$(FLASH_ATTN_LONG_CONTEXTS) \
		CK_FLASH_ATTN_LONG_HEADS=$(FLASH_ATTN_LONG_HEADS) \
		CK_FLASH_ATTN_LONG_KV_HEADS=$(FLASH_ATTN_LONG_KV_HEADS) \
		CK_FLASH_ATTN_TILE_K=$(FLASH_ATTN_TILE_K) \
		CK_FLASH_ATTN_FAST_EXP=$(FLASH_ATTN_FAST_EXP) \
		CK_FLASH_ATTN_OMP_THREADS=$(FLASH_ATTN_OMP_THREADS) \
		CK_FLASH_ATTN_LLAMA_PERF=$(FLASH_ATTN_LLAMA_PERF) \
		CK_FLASH_ATTN_LLAMA_PERF_MAX_TK=$(FLASH_ATTN_LLAMA_PERF_MAX_TK) \
		$(PYTHON) $(PYTHONFLAGS) unittest/test_flash_attention.py

test_flash_attention: test-flash-attention

# GEMM benchmark comparing CKernel (Native + MKL if available) vs PyTorch
bench_gemm:
	@echo "Building native kernels..."
	@rm -f $(BUILD_DIR)/libckernel_engine.so $(BUILD_DIR)/libckernel_native.so $(BUILD_DIR)/libckernel_mkl.so
	@# Force true native build even if MKL is auto-detected (or exported via env).
	@$(MAKE) --no-print-directory USE_NATIVE=1 USE_MKL= USE_ONEDNN=
	@cp $(BUILD_DIR)/libckernel_engine.so $(BUILD_DIR)/libckernel_native.so
ifneq ($(MKL_AVAILABLE),)
	@echo "Building MKL kernels..."
	@rm -f $(BUILD_DIR)/libckernel_engine.so
	@$(MAKE) --no-print-directory USE_MKL=1 USE_NATIVE= USE_ONEDNN=
	@cp $(BUILD_DIR)/libckernel_engine.so $(BUILD_DIR)/libckernel_mkl.so
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		CK_NATIVE_LIB=$(BUILD_DIR)/libckernel_native.so \
		CK_MKL_LIB=$(BUILD_DIR)/libckernel_mkl.so \
		$(PYTHON) $(PYTHONFLAGS) benchmarks/bench_gemm_vs_pytorch.py
else
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		CK_NATIVE_LIB=$(BUILD_DIR)/libckernel_native.so \
		CK_MKL_MISSING=1 \
		$(PYTHON) $(PYTHONFLAGS) benchmarks/bench_gemm_vs_pytorch.py
endif

# Benchmark with MKL only
bench_gemm_mkl:
ifneq ($(MKL_AVAILABLE),)
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory USE_MKL=1 USE_NATIVE= USE_ONEDNN=
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH CK_LIB_PATH=$(BUILD_DIR)/libckernel_engine.so \
		$(PYTHON) $(PYTHONFLAGS) benchmarks/bench_gemm_vs_pytorch.py
else
	@echo "Error: MKL not found at $(MKL_INC)"
	@echo "Install Intel oneAPI: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html"
	@exit 1
endif

# Benchmark with native kernels only
bench_gemm_native:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory USE_NATIVE=1 USE_MKL= USE_ONEDNN=
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH CK_LIB_PATH=$(BUILD_DIR)/libckernel_engine.so \
		$(PYTHON) $(PYTHONFLAGS) benchmarks/bench_gemm_vs_pytorch.py

tests-list:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════════╗"
	@echo "║                      C-KERNEL-ENGINE UNIT TESTS                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Run all tests:     make test"
	@echo "Run BF16 tests:    make test-bf16"
	@echo "Run single test:   python3 <script>"
	@echo "Run v4 pipeline:  make test-v4-q4k GGUF_PATH=/path/to/model.gguf"
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  FP32 Unit Tests                                                             │"
	@echo "└──────────────────────────────────────────────────────────────────────────────┘"
	@echo "  unittest/test_gelu.py              - GELU forward/backward vs PyTorch"
	@echo "  unittest/test_rmsnorm.py           - RMSNorm forward/backward vs PyTorch"
	@echo "  unittest/test_sigmoid.py           - Sigmoid forward/backward vs PyTorch"
	@echo "  unittest/test_relu.py              - ReLU forward/backward vs PyTorch"
	@echo "  unittest/test_layernorm.py         - LayerNorm forward/backward vs PyTorch"
	@echo "  unittest/test_softmax.py           - Causal softmax forward vs PyTorch"
	@echo "  unittest/test_softmax_backward.py  - Causal softmax backward vs PyTorch"
	@echo "  unittest/test_gemm.py              - GEMM variants vs PyTorch matmul"
	@echo "  unittest/test_q4_k_q8_k_matvec.py  - Q4_K x Q8_K matvec (inference)"
	@echo "  unittest/test_gemm_fused.py        - Fused GEMM+activation (ReLU/GELU/SiLU/SwiGLU)"
	@echo "  unittest/test_gemm_microkernel.py  - GEMM 8x8 microkernel with register blocking"
	@echo "  unittest/test_mlp.py               - MLP block forward/backward vs PyTorch"
	@echo "  unittest/test_swiglu.py            - SwiGLU activation forward/backward"
	@echo "  unittest/test_attention.py         - Multi-head attention forward vs PyTorch"
	@echo "  unittest/test_attention_backward.py - Attention backward (MHA/GQA)"
	@echo "  unittest/test_kv_cache_attention.py - Flash prefill + KV-cache decode attention"
	@echo "  unittest/test_kv_cache_layer_decode.py - Layer prefill+decode parity (KV cache)"
	@echo "  unittest/test_rope.py              - RoPE forward/backward vs PyTorch"
	@echo "  unittest/test_embedding.py         - Embedding forward/backward vs PyTorch"
	@echo "  unittest/test_cross_entropy.py     - Cross-entropy loss vs PyTorch"
	@echo "  unittest/test_orchestration_layer.py - Full layer stitch (GQA/MHA)"
	@echo "  unittest/test_lm_head_litmus.py    - LM head + CE end-to-end test"
	@echo "  unittest/test_fused_swiglu_decode.py - Fused SwiGLU decode MLP parity"
	@echo "  unittest/test_fused_attention_decode.py - Fused attention decode parity"
	@echo "  unittest/test_v4_conversion.py    - GGUF -> bump v4 conversion sanity (requires GGUF)"
	@echo "  unittest/test_v4_conversion.py    - GGUF -> bump v4 conversion sanity (requires GGUF)"
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  BF16 Unit Tests                                                             │"
	@echo "└──────────────────────────────────────────────────────────────────────────────┘"
	@echo "  unittest/bf16/test_sigmoid_bf16.py   - BF16 sigmoid forward/backward"
	@echo "  unittest/bf16/test_rmsnorm_bf16.py   - BF16 RMSNorm forward/backward"
	@echo "  unittest/bf16/test_gelu_bf16.py      - BF16 GELU forward/backward"
	@echo "  unittest/bf16/test_relu_bf16.py      - BF16 ReLU forward/backward"
	@echo "  unittest/bf16/test_layernorm_bf16.py - BF16 LayerNorm forward/backward"
	@echo "  unittest/bf16/test_mlp_bf16.py       - BF16 MLP forward"
	@echo "  unittest/bf16/test_attention_bf16.py - BF16 attention forward/backward"
	@echo "  unittest/bf16/test_rope_bf16.py      - BF16 RoPE forward/backward"
	@echo "  unittest/bf16/test_swiglu_bf16.py    - BF16 SwiGLU forward/backward"
	@echo "  unittest/bf16/test_embedding_bf16.py - BF16 embedding forward/backward"
	@echo "  unittest/bf16/test_cross_entropy_bf16.py - BF16 cross-entropy loss"
	@echo ""

$(BUILD_DIR)/bench_gemm_gemm_kernels.o: src/kernels/gemm_kernels.c include/ckernel_engine.h
	$(BENCH_CC) -O3 -Wall $(AVX_FLAGS) $(OPENMP_FLAG) -Iinclude -c -o $@ src/kernels/gemm_kernels.c

$(BUILD_DIR)/bench_gemm_strict.o: src/ckernel_strict.c include/ckernel_engine.h
	$(BENCH_CC) -O3 -Wall -Iinclude -c -o $@ src/ckernel_strict.c

rope-test: $(LIB) test-libs
	$(PYTHON) $(PYTHONFLAGS) unittest/test_rope.py

litmus:
	$(PYTHON) $(PYTHONFLAGS) unittest/test_lm_head_litmus.py $(ARGS)

litmus-demo: $(BUILD_DIR)
	@echo "Running litmus demo: $(LITMUS_DEMO_ARGS)"
	@$(PYTHON) $(PYTHONFLAGS) unittest/test_lm_head_litmus.py $(LITMUS_DEMO_ARGS) --svg $(LITMUS_DEMO_SVG) | tee $(LITMUS_DEMO_LOG)

layer-parity: $(LIB)
	$(PYTHON) $(PYTHONFLAGS) unittest/test_orchestration_layer.py $(ARGS) $(if $(TOL),--tol $(TOL),)

layer-parity-scalar:
	$(MAKE) -B $(LIB) AVX_FLAGS=
	$(PYTHON) $(PYTHONFLAGS) unittest/test_orchestration_layer.py $(ARGS) $(if $(TOL),--tol $(TOL),)

gen-specs:
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_kernel_specs.py

tiny-e2e: $(IR_DEMO) $(LIB)
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_bump.py --config $(TINY_CONFIG) --output $(BUILD_DIR)/tiny_weights.bin
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_tokens.py --config $(TINY_CONFIG) --output $(BUILD_DIR)/tiny_tokens.bin
	./$(IR_DEMO) $(TINY_CONFIG) --emit $(BUILD_DIR)/tiny_generated.c
	$(CC) $(CFLAGS) -Iinclude $(BUILD_DIR)/tiny_generated.c $$(cat $(BUILD_DIR)/tiny_generated.c.kernels) -o $(BUILD_DIR)/tiny_model -lm
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(BUILD_DIR)/tiny_model \
	  --model-weights $(BUILD_DIR)/tiny_weights.bin \
	  --tokens $(BUILD_DIR)/tiny_tokens.bin \
	  --out-logits $(BUILD_DIR)/tiny_logits.bin

tiny-train: $(IR_DEMO) $(LIB)
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_bump.py --config $(TINY_CONFIG) --output $(BUILD_DIR)/tiny_weights.bin
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_tokens.py --config $(TINY_CONFIG) --output $(BUILD_DIR)/tiny_tokens.bin --targets $(BUILD_DIR)/tiny_targets.bin
	./$(IR_DEMO) $(TINY_CONFIG) --emit $(BUILD_DIR)/tiny_generated.c
	$(CC) $(CFLAGS) -Iinclude $(BUILD_DIR)/tiny_generated.c $$(cat $(BUILD_DIR)/tiny_generated.c.kernels) -o $(BUILD_DIR)/tiny_model -lm
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(BUILD_DIR)/tiny_model \
	  --model-weights $(BUILD_DIR)/tiny_weights.bin \
	  --tokens $(BUILD_DIR)/tiny_tokens.bin \
	  --targets $(BUILD_DIR)/tiny_targets.bin \
	  --backward --lr $(TINY_TRAIN_LR) $(TINY_TRAIN_ARGS)

tiny-parity: $(IR_DEMO)
	$(PYTHON) $(PYTHONFLAGS) scripts/tiny_train_parity.py --config $(TINY_CONFIG) $(TINY_PARITY_ARGS)

smollm-demo:
	$(PYTHON) $(PYTHONFLAGS) scripts/smollm_train_demo.py \
	  --model-dir $(SMOLLM_MODEL_DIR) \
	  $(if $(SMOLLM_DOWNLOAD),--download-model --repo $(SMOLLM_REPO),) \
	  --context $(SMOLLM_CONTEXT) \
	  --dataset $(SMOLLM_DATASET) \
	  $(if $(SMOLLM_DATASET_CONFIG),--dataset-config $(SMOLLM_DATASET_CONFIG),) \
	  --split $(SMOLLM_SPLIT) \
	  --max-samples $(SMOLLM_MAX_SAMPLES)

smollm-forward:
	$(PYTHON) $(PYTHONFLAGS) scripts/smollm_forward_parity.py \
	  --model-dir $(SMOLLM_MODEL_DIR) \
	  $(if $(SMOLLM_DOWNLOAD),--download-model --repo $(SMOLLM_REPO),) \
	  --context $(SMOLLM_CONTEXT) \
	  --text "$(SMOLLM_TEXT)" \
	  --topk $(SMOLLM_TOPK)

smollm-layer-diff: $(LIB)
	$(PYTHON) $(PYTHONFLAGS) scripts/smollm_layer_stage_diff.py \
	  --model-dir $(SMOLLM_MODEL_DIR) \
	  $(if $(SMOLLM_DOWNLOAD),--download-model --repo $(SMOLLM_REPO),) \
	  --context $(SMOLLM_CONTEXT) \
	  --text "$(SMOLLM_TEXT)" \
	  --layer $(SMOLLM_LAYER) \
	  --tol $(SMOLLM_STAGE_TOL) \
	  $(if $(SMOLLM_STAGE_DUMP),--dump-stages,)

smollm-bump-compare:
	$(PYTHON) $(PYTHONFLAGS) scripts/compare_bump_to_hf.py \
	  --checkpoint $(SMOLLM_MODEL_DIR) \
	  --bump $(SMOLLM_BUMP) \
	  --context $(SMOLLM_CONTEXT) \
	  --layer $(SMOLLM_LAYER)

smollm-weight-check:
	$(PYTHON) $(PYTHONFLAGS) scripts/compare_bump_payload.py \
	  --bump $(SMOLLM_BUMP) \
	  --raw $(SMOLLM_OUT_WEIGHTS)

smollm-layer-stack:
	$(PYTHON) $(PYTHONFLAGS) scripts/smollm_layer_stack_diff.py \
	  --model-dir $(SMOLLM_MODEL_DIR) \
	  --context $(SMOLLM_CONTEXT) \
	  --text "$(SMOLLM_TEXT)" \
	  $(if $(SMOLLM_MAX_LAYERS),--max-layers $(SMOLLM_MAX_LAYERS),) \
	  --tol $(SMOLLM_STAGE_TOL)

smollm-train-parity: $(LIB)
	$(PYTHON) $(PYTHONFLAGS) scripts/tiny_train_parity.py \
	  --checkpoint $(SMOLLM_MODEL_DIR) \
	  --context $(SMOLLM_CONTEXT) \
	  --steps 1 \
	  --lr 1e-4

# llama.cpp parity test (compares CK kernels against llama.cpp/ggml)
# Patches applied: llama.patch, test-kernel-parity.cpp, ck-engine-parity-bench.patch
# AVX-512 is auto-detected and enabled if available
#
# RECOMMENDED: Run 'make llamacpp-parity-rebuild' first time or when patches change

# First time / fix everything: clean rebuild with all patches
llamacpp-parity-rebuild:
	@echo "Force rebuilding llama.cpp with latest patches..."
	@./scripts/run_parity_smoketest.sh --force-rebuild

# Build only - clone, patch, and compile llama.cpp (no tests)
llamacpp-parity-build:
	@echo "Building llama.cpp for parity testing..."
	@./scripts/run_parity_smoketest.sh --skip-tests

# Quick parity test
llamacpp-parity:
	@echo "Running llama.cpp parity smoketest..."
	@./scripts/run_parity_smoketest.sh --quick

# Full parity test (assumes already built)
llamacpp-parity-full:
	@echo "Running full llama.cpp parity test..."
	@./scripts/run_parity_smoketest.sh --skip-build

# Parity tests with performance benchmarks (CK vs llama.cpp)
llamacpp-parity-perf:
	@echo "Running llama.cpp parity + performance benchmarks..."
	@./scripts/run_parity_smoketest.sh --perf

# Parity tests with 7B dimension performance benchmarks
llamacpp-parity-perf-large:
	@echo "Running llama.cpp parity + large (7B) performance benchmarks..."
	@./scripts/run_parity_smoketest.sh --perf-large

# AVX-512 specific parity test
llamacpp-parity-avx512:
	@echo "Running AVX-512 parity tests..."
	@if grep -q avx512f /proc/cpuinfo 2>/dev/null; then \
		python3 scripts/test_avx512_parity.py --cross --full; \
	else \
		echo "SKIP: AVX-512 not available on this machine"; \
	fi

# =============================================================================
# Profiling: Flamegraph and Performance Analysis
# =============================================================================
# Generate flamegraphs comparing CK-Engine vs llama.cpp
# Prerequisites: perf, FlameGraph (auto-cloned)
# Results saved to: profile_results/<timestamp>/

# Profile both CK-Engine and llama.cpp (100 tokens)
profile:
	@./scripts/profile_comparison.sh

# Profile with more tokens for better sampling
profile-extended:
	@./scripts/profile_comparison.sh --tokens 200

# Profile CK-Engine only
profile-ck:
	@./scripts/profile_comparison.sh --ck-only --tokens 200

# Profile llama.cpp only
profile-llama:
	@./scripts/profile_comparison.sh --llama-only --tokens 200

# Quick profile (50 tokens, faster)
profile-quick:
	@./scripts/profile_comparison.sh --tokens 50

# View latest flamegraphs (opens in browser)
profile-view:
	@LATEST=$$(ls -td profile_results/*/ 2>/dev/null | head -1); \
	if [ -n "$$LATEST" ]; then \
		echo "Opening flamegraphs from $$LATEST"; \
		[ -f "$$LATEST/ck_flamegraph.svg" ] && xdg-open "$$LATEST/ck_flamegraph.svg" 2>/dev/null || firefox "$$LATEST/ck_flamegraph.svg" &; \
		[ -f "$$LATEST/llama_flamegraph.svg" ] && xdg-open "$$LATEST/llama_flamegraph.svg" 2>/dev/null || firefox "$$LATEST/llama_flamegraph.svg" &; \
	else \
		echo "No profile results found. Run 'make profile' first."; \
	fi

# Show profiling guide
profile-help:
	@echo ""
	@echo "==================================================================="
	@echo "  Profiling Guide: CK-Engine vs llama.cpp"
	@echo "==================================================================="
	@echo ""
	@echo "QUICK START:"
	@echo "  make profile          # Profile both (100 tokens)"
	@echo "  make profile-view     # Open flamegraphs in browser"
	@echo ""
	@echo "TARGETS:"
	@echo "  make profile          Profile both engines (100 tokens)"
	@echo "  make profile-extended Profile both engines (200 tokens)"
	@echo "  make profile-ck       Profile CK-Engine only"
	@echo "  make profile-llama    Profile llama.cpp only"
	@echo "  make profile-quick    Quick profile (50 tokens)"
	@echo "  make profile-view     Open latest flamegraphs"
	@echo ""
	@echo "READING FLAMEGRAPHS:"
	@echo "  - X-axis: % of total CPU time (wider = slower)"
	@echo "  - Y-axis: Call stack depth (bottom = entry point)"
	@echo "  - Click to zoom into a function"
	@echo "  - Search box to find specific functions"
	@echo ""
	@echo "WHAT TO LOOK FOR:"
	@echo "  [Good] gemv_*, gemm_*, vec_dot_* (GEMM kernels)"
	@echo "  [Bad]  malloc, free, mmap (memory allocation)"
	@echo "  [Bad]  memcpy, memmove (unnecessary copies)"
	@echo "  [Bad]  __intel_*, mkl_* very wide (MKL overhead)"
	@echo ""
	@echo "COMPARING CK vs LLAMA.CPP:"
	@echo "  1. Open both flamegraphs side by side"
	@echo "  2. Compare width of equivalent kernels:"
	@echo "     - CK: gemv_q6_k_q8_k, gemm_nt_q4_k_q8_k"
	@echo "     - llama: ggml_vec_dot_q6_K_q8_K, ggml_vec_dot_q4_K_q8_K"
	@echo "  3. Look for overhead not present in llama.cpp"
	@echo ""
	@echo "See docs/PROFILING_GUIDE.md for detailed analysis instructions."
	@echo ""

.PHONY: profile profile-extended profile-ck profile-llama profile-quick profile-view profile-help

# GEMV kernel performance benchmark
# Links against the full library to get all kernel implementations
benchmark-gemv: $(LIB)
	@echo "Building GEMV performance benchmark..."
	@mkdir -p build
	$(CC) -O3 -mavx -march=native -fopenmp -Iinclude \
		unittest/test_gemv_performance.c \
		-L$(BUILD_DIR) -lckernel_engine \
		-Wl,-rpath,$(BUILD_DIR) \
		-o build/test_gemv_performance -lm
	@echo ""
	@echo "Running benchmark..."
	@./build/test_gemv_performance

benchmark-gemv-quick: $(LIB)
	@$(MAKE) -s benchmark-gemv
	@./build/test_gemv_performance --quick

benchmark-gemv-large: $(LIB)
	@$(MAKE) -s benchmark-gemv
	@./build/test_gemv_performance --large

# Comprehensive GEMV kernel test - accuracy and performance for all quant types
test-gemv-comprehensive: $(LIB_PARITY)
	@echo "Running comprehensive GEMV kernel tests..."
	@$(PYTHON) unittest/test_gemv_kernels_comprehensive.py

test-gemv-comprehensive-quick: $(LIB_PARITY)
	@echo "Running quick GEMV kernel tests..."
	@$(PYTHON) unittest/test_gemv_kernels_comprehensive.py --quick

test-gemv-comprehensive-large: $(LIB_PARITY)
	@echo "Running comprehensive GEMV kernel tests with 7B dimensions..."
	@$(PYTHON) unittest/test_gemv_kernels_comprehensive.py --large

all-tests: $(LIB)
	$(MAKE) test
	$(MAKE) layer-parity-scalar TOL=$(ALL_TEST_LAYER_TOL) ARGS="$(ALL_TEST_LAYER_ARGS)"
	$(MAKE) tiny-parity

# Comprehensive test suite (scripts/run_all_tests.sh)
test-quick: $(LIB)
	@./scripts/run_all_tests.sh quick

test-full: $(LIB)
	@./scripts/run_all_tests.sh full

test-stress: $(LIB)
	@./scripts/run_all_tests.sh stress

# Profiling targets
PROFILE_CFLAGS := -O0 -g
PROFILE_PERF_CFLAGS := -O3 -fno-omit-frame-pointer -g
FLASH_ATTN_TK ?= 8192
FLASH_ATTN_H ?= 4
FLASH_ATTN_HKV ?= 4
FLASH_ATTN_DH ?= 64
FLASH_ATTN_ITERS ?= 200
FLASH_ATTN_WARMUP ?= 20
FLASH_ATTN_THREADS ?= $(shell nproc 2>/dev/null || echo 1)

profile-memory: $(BUILD_DIR)
	@echo "Building with debug symbols..."
	$(MAKE) -B $(LIB) CFLAGS="$(PROFILE_CFLAGS) -fPIC -fopenmp -Wall $(AVX_FLAGS) $(INCLUDES)"
	$(MAKE) tiny-e2e
	@echo "Running Valgrind memcheck..."
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
		--suppressions=valgrind.supp \
		./build/tiny_model --model-weights build/tiny_weights.bin \
		--tokens build/tiny_tokens.bin --out-logits build/tiny_logits.bin

profile-heap: $(BUILD_DIR)
	@echo "Building with debug symbols..."
	$(MAKE) -B $(LIB) CFLAGS="$(PROFILE_CFLAGS) -fPIC -fopenmp -Wall $(AVX_FLAGS) $(INCLUDES)"
	$(MAKE) tiny-e2e
	@echo "Running Valgrind massif..."
	valgrind --tool=massif --pages-as-heap=yes --massif-out-file=$(BUILD_DIR)/massif.out \
		./build/tiny_model --model-weights build/tiny_weights.bin \
		--tokens build/tiny_tokens.bin --out-logits build/tiny_logits.bin
	@echo "Heap profile saved to $(BUILD_DIR)/massif.out"
	@echo "View with: ms_print $(BUILD_DIR)/massif.out"

profile-cpu: $(BUILD_DIR)
	@echo "Building with frame pointers..."
	$(MAKE) -B $(LIB) CFLAGS="$(PROFILE_PERF_CFLAGS) -fPIC -fopenmp -Wall $(AVX_FLAGS) $(INCLUDES)"
	$(MAKE) tiny-e2e
	@echo "Recording with perf..."
	perf record -g -F 99 -o $(BUILD_DIR)/perf.data \
		./build/tiny_model --model-weights build/tiny_weights.bin \
		--tokens build/tiny_tokens.bin --out-logits build/tiny_logits.bin
	@echo "Profile saved to $(BUILD_DIR)/perf.data"
	perf report -i $(BUILD_DIR)/perf.data --stdio --sort=overhead | head -30

profile-flash-attn: $(BUILD_DIR)
	@echo "Building with frame pointers..."
	$(MAKE) -B $(LIB) CFLAGS="$(PROFILE_PERF_CFLAGS) -fPIC -fopenmp -Wall $(AVX_FLAGS) $(INCLUDES)"
	@echo "Building flash attention microbenchmark..."
	$(CC) $(PROFILE_PERF_CFLAGS) -march=native -fopenmp $(AVX_FLAGS) $(SSSE3_FLAGS) -Iinclude \
		benchmarks/perf_flash_attn_micro.c -L$(BUILD_DIR) -lckernel_engine -lm \
		-o $(BUILD_DIR)/perf_flash_attn_micro
	@echo "Recording with perf..."
	OMP_NUM_THREADS=$(FLASH_ATTN_THREADS) LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		perf record -g -F 99 -o $(BUILD_DIR)/perf_flash_attn.perf \
		$(BUILD_DIR)/perf_flash_attn_micro $(FLASH_ATTN_TK) $(FLASH_ATTN_H) $(FLASH_ATTN_HKV) \
		$(FLASH_ATTN_DH) $(FLASH_ATTN_ITERS) $(FLASH_ATTN_WARMUP)
	@echo "Profile saved to $(BUILD_DIR)/perf_flash_attn.perf"
	perf report -i $(BUILD_DIR)/perf_flash_attn.perf --stdio --sort=overhead | head -30

# =============================================================================
# VTune Profiling (Intel Performance Analysis)
# =============================================================================
# Requires: Intel oneAPI VTune (source /opt/intel/oneapi/setvars.sh)

vtune-hotspots:
	@./scripts/vtune_profile.sh hotspots

vtune-memory:
	@./scripts/vtune_profile.sh memory-access

vtune-uarch:
	@./scripts/vtune_profile.sh uarch

vtune-compare:
	@./scripts/vtune_profile.sh compare

vtune-full:
	@./scripts/vtune_profile.sh full

# =============================================================================
# DRAM/Cache Measurement (Fusion Validation)
# =============================================================================
# Measures LLC misses to validate fusion reduces DRAM access

measure-dram:
	@./scripts/measure_dram.sh baseline

measure-dram-fused:
	@./scripts/measure_dram.sh fused

measure-dram-compare:
	@./scripts/measure_dram.sh compare

FLAMEGRAPH_DIR ?= $(HOME)/Programs/FlameGraph

flamegraph: $(BUILD_DIR)/perf.data
	@echo "Generating flamegraph..."
	perf script -i $(BUILD_DIR)/perf.data | $(FLAMEGRAPH_DIR)/stackcollapse-perf.pl | $(FLAMEGRAPH_DIR)/flamegraph.pl > $(BUILD_DIR)/flamegraph.svg
	@echo "Flamegraph saved to $(BUILD_DIR)/flamegraph.svg"

profile-cache: $(BUILD_DIR)
	@echo "Building with debug symbols..."
	$(MAKE) -B $(LIB) CFLAGS="$(PROFILE_CFLAGS) -fPIC -fopenmp -Wall $(AVX_FLAGS) $(INCLUDES)"
	$(MAKE) tiny-e2e
	@echo "Running Valgrind cachegrind..."
	valgrind --tool=cachegrind --cachegrind-out-file=$(BUILD_DIR)/cachegrind.out \
		./build/tiny_model --model-weights build/tiny_weights.bin \
		--tokens build/tiny_tokens.bin --out-logits build/tiny_logits.bin
	@echo "Cache profile saved to $(BUILD_DIR)/cachegrind.out"
	@echo "View with: cg_annotate $(BUILD_DIR)/cachegrind.out"

small-e2e: $(IR_DEMO) $(LIB)
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_bump.py --config $(SMALL_CONFIG) --output $(BUILD_DIR)/small_weights.bin
	$(PYTHON) $(PYTHONFLAGS) scripts/gen_random_tokens.py --config $(SMALL_CONFIG) --output $(BUILD_DIR)/small_tokens.bin
	./$(IR_DEMO) $(SMALL_CONFIG) --emit $(BUILD_DIR)/small_generated.c
	$(CC) $(CFLAGS) -Iinclude $(BUILD_DIR)/small_generated.c $$(cat $(BUILD_DIR)/small_generated.c.kernels) -o $(BUILD_DIR)/small_model -lm
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(BUILD_DIR)/small_model \
	  --model-weights $(BUILD_DIR)/small_weights.bin \
	  --tokens $(BUILD_DIR)/small_tokens.bin \
	  --out-logits $(BUILD_DIR)/small_logits.bin

# ═══════════════════════════════════════════════════════════════════════════════
# VERSION INFO
# ═══════════════════════════════════════════════════════════════════════════════

version:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║  C-Kernel-Engine - Current Version: v5 (Explicit Unrolled Codegen)       ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Quick Start (v5 - Recommended):"
	@echo "    python scripts/ck_run_v5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf"
	@echo ""
	@echo "  Or interactive chat:"
	@echo "    python scripts/ck_chat.py ~/.cache/ck-engine-v5/models/<model>"
	@echo ""
	@echo "  For version history: make version-history"
	@echo "  For all targets:     make help"
	@echo ""

version-history:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║                    C-Kernel-Engine Version History                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  ┌────────┬─────────────────────────────────────────────────────────────┐"
	@echo "  │ v5     │ CURRENT - Explicit Unrolled Codegen                         │"
	@echo "  │ (2024) │ • Per-layer functions (qwen2_layer_0_decode, ...)           │"
	@echo "  │        │ • Mixed quant support (Q4_K, Q6_K per tensor)               │"
	@echo "  │        │ • Debug/parity hooks for llama.cpp comparison               │"
	@echo "  │        │ • make v5, make demo-v5, python scripts/ck_run_v5.py        │"
	@echo "  ├────────┼─────────────────────────────────────────────────────────────┤"
	@echo "  │ v4     │ LEGACY - Graph-based IR with lowering                       │"
	@echo "  │        │ • GGUF support, Q4_K quantization                           │"
	@echo "  │        │ • make ir-v4, make ir-v4-q4k                                │"
	@echo "  │        │ • Deprecated: use v5 for new projects                       │"
	@echo "  ├────────┼─────────────────────────────────────────────────────────────┤"
	@echo "  │ v2     │ LEGACY - JSON IR with Python codegen                        │"
	@echo "  │        │ • HuggingFace config → IR → C runtime                       │"
	@echo "  │        │ • make ir-v2, make emit-v2                                  │"
	@echo "  │        │ • Deprecated: use v5 for new projects                       │"
	@echo "  ├────────┼─────────────────────────────────────────────────────────────┤"
	@echo "  │ v1     │ LEGACY - Original C codegen                                 │"
	@echo "  │        │ • Direct config → C skeleton                                │"
	@echo "  │        │ • make ck, make emit                                        │"
	@echo "  │        │ • Deprecated: use v5 for new projects                       │"
	@echo "  └────────┴─────────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "  Quantization Formats Supported:"
	@echo "    K-Quants (recommended): Q4_K, Q6_K, Q8_K"
	@echo "    Legacy: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0"
	@echo ""
	@echo "  Parity Testing:"
	@echo "    make llamacpp-parity    - Compare vs llama.cpp/ggml"
	@echo "    make layer-parity       - Compare vs PyTorch"
	@echo ""

help:
	@echo "C-Kernel-Engine Make targets:"
	@echo ""
	@echo "  ┌─────────────────────────────────────────────────────────────────────┐"
	@echo "  │  QUICK START (v5 - Recommended)                                     │"
	@echo "  └─────────────────────────────────────────────────────────────────────┘"
	@echo "  make version          Show current version and quick start"
	@echo "  make v5               Show v5 pipeline documentation"
	@echo "  make demo-v5          Run Qwen2-0.5B v5 demo"
	@echo "  python scripts/ck_run_v5.py run <gguf-url-or-path>"
	@echo ""
	@echo "  ┌─────────────────────────────────────────────────────────────────────┐"
	@echo "  │  BUILD                                                              │"
	@echo "  └─────────────────────────────────────────────────────────────────────┘"
	@echo "  make                  Build full engine library ($(LIB))"
	@echo "  make test-libs        Build per-kernel shared libraries"
	@echo "  make ck-cli           Build CLI orchestrator"
	@echo ""
	@echo "  ┌─────────────────────────────────────────────────────────────────────┐"
	@echo "  │  TESTING                                                            │"
	@echo "  └─────────────────────────────────────────────────────────────────────┘"
	@echo "  make test             Run core kernel tests"
	@echo "  make test-quant       Run quantization kernel tests"
	@echo "  make test-bf16        Run BF16 kernel tests"
	@echo "  make showtests        Show ALL available test targets"
	@echo "  make nightly          Run full test suite"
	@echo ""
	@echo "  Parity Tests:"
	@echo "  make layer-parity         PyTorch parity (fp32/bf16)"
	@echo "  make llamacpp-parity      llama.cpp parity (Q4_K)"
	@echo "  make smollm-train-parity  Full training parity"
	@echo ""
	@echo "  ┌─────────────────────────────────────────────────────────────────────┐"
	@echo "  │  QUANTIZATION                                                       │"
	@echo "  └─────────────────────────────────────────────────────────────────────┘"
	@echo "  make gguf-inspect GGUF=path   Inspect GGUF tensor dtypes"
	@echo "  make gguf-list GGUF=path      List all GGUF tensors"
	@echo "  make gguf-to-bump GGUF=path   Convert GGUF → bump weights"
	@echo ""
	@echo "  ┌─────────────────────────────────────────────────────────────────────┐"
	@echo "  │  REPORTS & STATUS                                                   │"
	@echo "  └─────────────────────────────────────────────────────────────────────┘"
	@echo "  make report           Comprehensive status report"
	@echo "  make version          Current version info"
	@echo "  make version-history  All versions and their features"
	@echo "  make show_config      System hardware topology"
	@echo ""
	@echo "  ┌─────────────────────────────────────────────────────────────────────┐"
	@echo "  │  LEGACY VERSIONS (see: make version-history)                        │"
	@echo "  └─────────────────────────────────────────────────────────────────────┘"
	@echo "  make ir-v4 ...        v4 pipeline (deprecated)"
	@echo "  make ir-v2 ...        v2 pipeline (deprecated)"
	@echo "  make emit ...         v1 pipeline (deprecated)"
	@echo ""
	@echo "  ┌─────────────────────────────────────────────────────────────────────┐"
	@echo "  │  PROFILING                                                          │"
	@echo "  └─────────────────────────────────────────────────────────────────────┘"
	@echo "  make profile-memory   Valgrind memcheck"
	@echo "  make profile-cpu      perf CPU profiler"
	@echo "  make profile-flash-attn  perf flash-attn microbench"
	@echo "  make profile-cache    Valgrind cachegrind"
	@echo "  make flamegraph       Generate SVG flamegraph"
	@echo "  make bench_gemm       GEMM benchmarks (Native/MKL/PyTorch)"
	@echo ""
	@echo "Interactive Tools:"
	@echo "  make ck-chat          Build interactive CLI (C-based)"
	@echo "  make ck-server        Build streaming HTTP server (C-based)"
	@echo "  make ck-chat-py       Run interactive chat (Python + C inference)"
	@echo "  make ck-server-py     Run streaming server (Python + C inference)"
	@echo ""
	@echo "  make clean            Remove all built libraries"

clean:
	rm -rf $(BUILD_DIR)
	rm -f ~/.cache/ck-engine-v5/models/*/libmodel.so
	rm -f ~/.cache/ck-engine-v5/models/*/model.c
	rm -f ~/.cache/ck-engine-v5/models/*/*.o

# =============================================================================
# Parity Testing: CK vs llama.cpp/ggml
# =============================================================================
# Tests individual CK kernels against llama.cpp's ggml implementations.

# Source files for CK parity library
PARITY_SRCS := src/ck_parity_api.c \
               src/kernels/dequant_kernels.c \
               src/kernels/gemm_kernels_q4k_q8k.c \
               src/kernels/gemm_kernels_q4k_q8k_avx2.c \
               src/kernels/gemm_kernels_q4k_q8k_vnni.c \
               src/kernels/gemm_kernels_q6k_q8k.c \
               src/kernels/gemm_kernels_q4k_sse.c \
               src/kernels/gemm_kernels_q4k_avx.c \
               src/kernels/gemm_kernels_q5_0.c \
               src/kernels/gemm_kernels_q5_0_sse_v2.c \
               src/kernels/gemm_kernels_q8_0.c \
               src/kernels/gemm_batch_int8.c \
               src/kernels/quantize_row_q8_k_sse.c \
               src/kernels/rmsnorm_kernels.c \
               src/kernels/rope_kernels.c \
               src/kernels/swiglu_kernels.c \
               src/kernels/softmax_kernels.c \
               src/kernels/sigmoid_kernels.c \

LIB_PARITY := $(BUILD_DIR)/libck_parity.so

# Build CK parity testing library
$(LIB_PARITY): $(BUILD_DIR) $(PARITY_SRCS)
	$(CC) $(CFLAGS) -shared -o $@ $(PARITY_SRCS) -lm

libck_parity.so: $(LIB_PARITY)
	@echo "Built CK parity library: $(LIB_PARITY)"

# Build llama.cpp kernel test library
# Requires llama.cpp to be cloned in llama.cpp/ subdirectory
LLAMA_CPP_DIR := llama.cpp
LLAMA_KERNEL_TEST := $(LLAMA_CPP_DIR)/libggml_kernel_test.so

$(LLAMA_KERNEL_TEST):
	@echo "Building llama.cpp kernel test library..."
	@if [ ! -d "$(LLAMA_CPP_DIR)" ]; then \
		echo "ERROR: llama.cpp not found. Clone it first:"; \
		echo "  git clone https://github.com/ggerganov/llama.cpp llama.cpp"; \
		exit 1; \
	fi
	@if [ ! -f "$(LLAMA_CPP_DIR)/build/bin/libggml.so" ] && [ ! -f "$(LLAMA_CPP_DIR)/build/lib/libggml.so" ]; then \
		echo "Building llama.cpp..."; \
		cd $(LLAMA_CPP_DIR) && mkdir -p build && cd build && cmake .. && make -j$$(nproc); \
	fi
	@# Detect library location (build/bin or build/lib)
	cd $(LLAMA_CPP_DIR) && \
	GGML_LIB_DIR=$$(if [ -f build/bin/libggml.so ]; then echo build/bin; else echo build/lib; fi) && \
	$(CXX) -shared -fPIC -o libggml_kernel_test.so \
		tests/test-kernel-parity.cpp \
		-I ggml/include -I ggml/src \
		-L $$GGML_LIB_DIR -lggml -lggml-cpu -lggml-base -lm -lpthread \
		-Wl,-rpath,$(PWD)/$(LLAMA_CPP_DIR)/$$GGML_LIB_DIR

llama_kernel_test: $(LLAMA_KERNEL_TEST)
	@echo "Built llama.cpp kernel test library: $(LLAMA_KERNEL_TEST)"

# Build both parity libraries
parity-libs: $(LIB_PARITY) $(LLAMA_KERNEL_TEST)
	@echo ""
	@echo "Parity testing libraries built:"
	@echo "  CK:        $(LIB_PARITY)"
	@echo "  llama.cpp: $(LLAMA_KERNEL_TEST)"

# Run kernel parity tests
test-kernels: parity-libs
	@echo ""
	@echo "Running kernel parity tests: CK vs llama.cpp/ggml"
	@echo "=================================================="
	$(PYTHON) $(PYTHONFLAGS) scripts/test_kernels_vs_llamacpp.py --all

# Run specific kernel test
test-kernel-%: parity-libs
	$(PYTHON) $(PYTHONFLAGS) scripts/test_kernels_vs_llamacpp.py --kernel $*

.PHONY: libck_parity.so llama_kernel_test parity-libs test-kernels

# Litmus test for full forward pass parity with PyTorch
# ==============================================================================
TEST_HARNESS_SRCS := src/backend_native.c \
	src/ckernel_alloc.c \
	src/ckernel_ir.c \
	src/ckernel_orchestration.c \
	src/ckernel_registry.c \
	src/cpu_features.c \
	src/kernels/attention_kernels.c \
	src/kernels/attention_decode_fused.c \
	src/kernels/gelu_kernels.c \
	src/kernels/gemm_kernels.c \
	src/kernels/gemm_fused_kernels.c \
	src/kernels/mlp_fused_decode.c \
	src/kernels/gemm_microkernel.c \
	src/kernels/layernorm_kernels.c \
	src/kernels/mlp_kernels.c \
	src/kernels/rmsnorm_kernels.c \
	src/kernels/sigmoid_kernels.c \
	src/kernels/relu_kernels.c \
	src/kernels/softmax_kernels.c \
	src/kernels/swiglu_kernels.c \
	src/kernels/rope_kernels.c

.PHONY: litmus-test

litmus-test:
	@echo "--- [Step 1] Generating C Runtime Code ---"
	@make emit OUT=build/generated_model.c
	@echo "\n--- [Step 2] Generating PyTorch Reference Data ---"
	$(PYTHON) $(PYTHONFLAGS) unittest/generate_reference_data.py
	@echo "\n--- [Step 3] Compiling C Test Harness ---"
	$(CC) $(CFLAGS) -Iinclude test_forward_pass.c build/generated_model.c $(TEST_HARNESS_SRCS) -o build/test_forward_pass -lm -lpthread -lrt
	@echo "Compilation complete: build/test_forward_pass"
	@echo "\n--- [Step 4] Running C Test Harness ---"
	./build/test_forward_pass
	@echo "\n--- [Step 5] Comparing C output with PyTorch reference ---"
	$(PYTHON) $(PYTHONFLAGS) unittest/compare_outputs.py

# ============================================================================
# Interactive CLI and Server Tools
# ============================================================================

CK_TOKENIZER := src/ck_tokenizer.c
CK_MAIN := tools/ck_main.c
CK_SERVER := tools/ck_server.c
CK_CLI := tools/ck.c
CK_CLI_V4 := tools/ck_v4.c
CK_CLI_V6 := src/v6/ck_cli_v6.c
CK_CLI_V65 := src/v6.5/ck_cli_v6.5.c

# Main orchestrator (ck run, ck list, etc.)
# Suppress format-truncation warnings - paths are validated at runtime
$(BUILD_DIR)/ck: $(BUILD_DIR) $(CK_CLI)
	$(CC) -O2 -Wall -Wno-format-truncation -Wno-stringop-truncation -o $@ $(CK_CLI) -ldl -lm

# v4 CLI (builder/orchestrator)
$(BUILD_DIR)/ck-v4: $(BUILD_DIR) $(CK_CLI_V4)
	$(CC) -O2 -Wall -Wno-format-truncation -Wno-stringop-truncation -o $@ $(CK_CLI_V4) -lm

# Build CLI with all dependencies (library + IR tool)
ck-cli: $(LIB) $(IR_DEMO) $(BUILD_DIR)/ck
	@echo ""
	@echo "  C-Kernel-Engine CLI built: $(BUILD_DIR)/ck"
	@echo "  Dependencies:"
	@echo "    - $(LIB)"
	@echo "    - $(IR_DEMO)"
	@echo ""
	@echo "  Usage:"
	@echo "    ./$(BUILD_DIR)/ck run HuggingFaceTB/SmolLM-135M"
	@echo "    ./$(BUILD_DIR)/ck run https://huggingface.co/Qwen/Qwen2-0.5B --server"
	@echo "    ./$(BUILD_DIR)/ck list"
	@echo "    ./$(BUILD_DIR)/ck help"
	@echo ""
	@echo "  To install system-wide:"
	@echo "    sudo cp $(BUILD_DIR)/ck /usr/local/bin/"
	@echo ""

$(BUILD_DIR)/ck-cli-v5: src/ck_cli_v5.c src/ck_tokenizer_v2.c $(LIB)
	$(CC) $(CFLAGS) -o $@ src/ck_cli_v5.c src/ck_tokenizer_v2.c -L$(BUILD_DIR) -lckernel_engine -ldl -lpthread -Wl,-rpath,$(BUILD_DIR)

ck-cli-v5-native: $(BUILD_DIR)/ck-cli-v5
	@echo "Native V5 CLI built: $<"

$(BUILD_DIR)/ck-cli-v6: $(CK_CLI_V6) $(LIB_TOKENIZER)
	$(CC) $(CFLAGS) -o $@ $(CK_CLI_V6) -L$(BUILD_DIR) -lckernel_tokenizer -ldl -lpthread -Wl,-rpath,$(BUILD_DIR)

ck-cli-v6: $(BUILD_DIR)/ck-cli-v6
	@echo "Native V6 CLI built: $<"

# v6.5 Native CLI (improved REPL with model discovery, chat templates, sampling)
$(BUILD_DIR)/ck-cli-v6.5: $(CK_CLI_V65) $(LIB_TOKENIZER)
	$(CC) $(CFLAGS) -o $@ $(CK_CLI_V65) -L$(BUILD_DIR) -lckernel_tokenizer -ldl -lpthread -lm -Wl,-rpath,$(BUILD_DIR)

ck-cli-v6.5: $(BUILD_DIR)/ck-cli-v6.5
	@echo ""
	@echo "  $(C_CYAN)C-Kernel-Engine v6.5 CLI$(C_RESET)"
	@echo "  Features: Model discovery, Chat templates, Temperature sampling, REPL"
	@echo ""
	@echo "  Usage:"
	@echo "    ./$(BUILD_DIR)/ck-cli-v6.5 --model qwen           # Auto-discover from cache"
	@echo "    ./$(BUILD_DIR)/ck-cli-v6.5 --list                 # List available models"
	@echo "    ./$(BUILD_DIR)/ck-cli-v6.5 <model.so> <weights.bump>"
	@echo ""

# v4 CLI (Python wrapper for IR v4 pipeline) - LEGACY: use ck-cli-v5 instead
ck-cli-v4: $(LIB)
	@echo ""
	@echo "  $(C_ORANGE)C-Kernel-Engine v4 CLI$(C_RESET)"
	@echo "  Pipeline: download -> convert -> IR v4 -> codegen -> compile -> run"
	@echo ""
	@echo "  Usage:"
	@echo "    python scripts/ck_run_v4.py run HuggingFaceTB/SmolLM-135M"
	@echo "    python scripts/ck_run_v4.py run ./model.gguf --weight-dtype=q4_k"
	@echo "    python scripts/ck_run_v4.py run Qwen/Qwen2-0.5B --generate-only"
	@echo "    python scripts/ck_run_v4.py list"
	@echo "    python scripts/ck_run_v4.py clean"
	@echo ""
	@echo "  Options:"
	@echo "    --weight-dtype=TYPE    Weight type: float32, bf16, q4_k, q4_k_m, q6_k"
	@echo "    --generate-only        Generate C code without running"
	@echo "    --force-convert        Re-convert weights"
	@echo "    --force-compile        Re-generate and recompile"
	@echo ""

# v5 CLI (Python wrapper for IR v5 explicit codegen pipeline)
ck-cli-v5: $(LIB)
	@echo ""
	@echo "  $(C_ORANGE)C-Kernel-Engine v5 CLI$(C_RESET) (Explicit Unrolled Codegen)"
	@echo "  Pipeline: download -> GGUF convert -> IR v5 -> explicit codegen -> compile -> run"
	@echo ""
	@echo "  Usage:"
	@echo "    python scripts/ck_run_v5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf"
	@echo "    python scripts/ck_run_v5.py run ./model.gguf"
	@echo "    python scripts/ck_run_v5.py run MODEL --debug        # Buffer stats after each op"
	@echo "    python scripts/ck_run_v5.py run MODEL --parity       # Save layer outputs for comparison"
	@echo "    python scripts/ck_run_v5.py run MODEL --generate-only"
	@echo "    python scripts/ck_run_v5.py list"
	@echo "    python scripts/ck_run_v5.py clean"
	@echo ""
	@echo "  Options:"
	@echo "    --weight-dtype=TYPE    Weight type: q4_k, q4_k_m, q5_0, q6_k, q8_0"
	@echo "    --debug                Insert debug prints (buffer stats after each op)"
	@echo "    --parity               Save layer outputs for PyTorch/HuggingFace comparison"
	@echo "    --generate-only        Generate C code without running"
	@echo "    --force-convert        Re-convert weights"
	@echo "    --force-compile        Re-generate and recompile"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# V5 PIPELINE (Explicit Unrolled Codegen)
# ═══════════════════════════════════════════════════════════════════════════════

v5:
	@echo ""
	@echo "  $(C_ORANGE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(C_RESET)"
	@echo "  $(C_ORANGE)C-Kernel-Engine v5$(C_RESET) - Explicit Unrolled Codegen"
	@echo "  $(C_ORANGE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(C_RESET)"
	@echo ""
	@echo "  $(C_GREEN)v5 Features:$(C_RESET)"
	@echo "    • Each layer has separate function (qwen2_layer_0_decode, qwen2_layer_1_decode, ...)"
	@echo "    • Explicit kernel names per quant type (gemm_nt_q5_0, gemm_nt_q8_0, gemm_nt_q4_k)"
	@echo "    • Per-layer quant types shown in header comments"
	@echo "    • Easy to insert debug hooks for PyTorch parity testing"
	@echo "    • --debug: Print buffer stats (NaN/Inf/range) after each operation"
	@echo "    • --parity: Save layer outputs to .bin files for comparison"
	@echo ""
	@echo "  $(C_GREEN)Quick Start:$(C_RESET)"
	@echo "    python scripts/ck_run_v5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf"
	@echo ""
	@echo "  $(C_GREEN)Commands:$(C_RESET)"
	@echo "    python scripts/ck_run_v5.py run MODEL              # Full pipeline: download → convert → codegen → compile"
	@echo "    python scripts/ck_run_v5.py run MODEL --debug      # With debug buffer checks"
	@echo "    python scripts/ck_run_v5.py run MODEL --parity     # Save outputs for PyTorch comparison"
	@echo "    python scripts/ck_run_v5.py run MODEL --generate-only  # Generate code, don't run"
	@echo "    python scripts/ck_run_v5.py list                   # List cached v5 models"
	@echo "    python scripts/ck_run_v5.py clean                  # Clean all v5 cache"
	@echo "    python scripts/ck_run_v5.py clean MODEL            # Clean specific model"
	@echo ""
	@echo "  $(C_GREEN)Examples:$(C_RESET)"
	@echo "    # Download Qwen2-0.5B Q4_K_M and generate v5 code:"
	@echo "    python scripts/ck_run_v5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf --generate-only"
	@echo ""
	@echo "    # With debug mode (prints buffer stats after each op):"
	@echo "    python scripts/ck_run_v5.py run MODEL --debug --generate-only"
	@echo ""
	@echo "    # With parity mode (saves .bin files for PyTorch comparison):"
	@echo "    python scripts/ck_run_v5.py run MODEL --parity --generate-only"
	@echo ""
	@echo "  $(C_GREEN)Output Location:$(C_RESET)"
	@echo "    ~/.cache/ck-engine-v5/models/<model-name>/"
	@echo "      ├── generated_*_decode.c   # v5 explicit decode code (24 layers unrolled)"
	@echo "      ├── generated_*_prefill.c  # v5 explicit prefill code"
	@echo "      ├── weights.bump           # Converted weights"
	@echo "      ├── weights_manifest.json  # Per-weight quant types"
	@echo "      └── libmodel.so            # Compiled shared library"
	@echo ""
	@echo "  $(C_GREEN)Makefile Targets:$(C_RESET)"
	@echo "    make v5              # This help"
	@echo "    make demo-v5         # Run full v5 demo with Qwen2-0.5B"
	@echo "    make demo-v5-debug   # Run with --debug enabled"
	@echo ""

# Demo: v5 explicit codegen
demo-v5: $(LIB)
	@echo ""
	@echo "  $(C_ORANGE)Demo: Qwen2-0.5B v5 Explicit Codegen$(C_RESET)"
	@echo "  Generates explicit per-layer kernel calls for mixed Q4_K_M"
	@echo ""
	@python3 scripts/ck_run_v5.py run \
		"hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf" \
		--generate-only
	@echo ""
	@echo "  $(C_GREEN)Generated files:$(C_RESET)"
	@ls -la ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/generated_*.c 2>/dev/null || true
	@echo ""
	@echo "  $(C_GREEN)Per-layer quant types:$(C_RESET)"
	@head -20 ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/generated_qwen2_decode.c 2>/dev/null || true

# Demo: v5 with debug enabled
demo-v5-debug: $(LIB)
	@echo ""
	@echo "  $(C_ORANGE)Demo: Qwen2-0.5B v5 with Debug$(C_RESET)"
	@echo "  Generates code with buffer stats after each operation"
	@echo ""
	@python3 scripts/ck_run_v5.py run \
		"hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf" \
		--debug \
		--generate-only \
		--force-compile
	@echo ""
	@echo "  $(C_GREEN)Debug checks in generated code:$(C_RESET)"
	@grep -n "debug_check_buffer" ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/generated_qwen2_decode.c 2>/dev/null | head -10 || true

# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY V4 DEMOS (Deprecated - use demo-v5 instead)
# ═══════════════════════════════════════════════════════════════════════════════

# Demo: Download Qwen2-0.5B Q4_K_M GGUF and run end-to-end (LEGACY - use demo-v5)
demo-q4: $(LIB)
	@echo ""
	@echo "  $(C_ORANGE)Demo: Qwen2-0.5B Q4_K_M End-to-End$(C_RESET)"
	@echo "  This will download ~400MB GGUF, convert, compile, and run"
	@echo ""
	@python3 scripts/ck_run_v4.py run \
		"hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf" \
		--weight-dtype=q4_k_m \
		--prompt "What is 2+2?" \
		--max-tokens 50

# Demo: Generate code only (no inference)
demo-q4-codegen: $(LIB)
	@echo ""
	@echo "  $(C_ORANGE)Demo: Generate Q4_K code only$(C_RESET)"
	@echo ""
	@python3 scripts/ck_run_v4.py run \
		"hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf" \
		--weight-dtype=q4_k_m \
		--generate-only

# ═══════════════════════════════════════════════════════════════════════════════
# TEST TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

# Run kernel unit tests (fast, catches SIMD bugs)
# Note: test-kernels target at line ~1276 runs Python parity tests vs llama.cpp
# This target (test-kernel-unit) runs standalone C kernel unit tests
test-kernel-unit:
	@echo ""
	@echo "  $(C_ORANGE)Running Kernel Unit Tests$(C_RESET)"
	@echo "  Tests Q4_K x Q8_K GEMV kernels (ref, AVX2, VNNI)"
	@echo ""
	@cd unittest && make clean && make
	@cd unittest && ./test_q4k_kernels
	@echo ""
	@echo "  $(C_GREEN)All kernel tests passed!$(C_RESET)"
	@echo ""

# Test quantization kernels on VNNI-capable server (Xeon)
# This MUST be run on the server to catch VNNI bugs
test-quant-server: $(LIB)
	@echo ""
	@echo "  $(C_ORANGE)Quantization Tests for VNNI Server$(C_RESET)"
	@echo "  Run this on Xeon with AVX-512 VNNI!"
	@echo ""
	@echo "  Checking CPU features..."
	@grep -q avx512vnni /proc/cpuinfo && echo "  $(C_GREEN)✓ AVX-512 VNNI detected$(C_RESET)" || \
		(echo "  $(C_RED)✗ WARNING: No VNNI - tests won't validate VNNI code!$(C_RESET)" && exit 1)
	@echo ""
	@echo "  Step 1: C Kernel Tests (standalone, no library)"
	@cd unittest && make clean && make
	@cd unittest && ./test_q4k_kernels --verbose
	@echo ""
	@echo "  Step 2: Python Quant Tests (uses library)"
	@set -e; \
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_quant_kernels.py; \
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_q4_k_q8_k_matvec.py; \
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_q6k_kernels.py; \
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_q4_k_quantize.py
	@echo ""
	@echo "  $(C_GREEN)All VNNI quantization tests passed!$(C_RESET)"
	@echo ""

# Run unit tests only (fast, no model download)
test-unit: $(LIB) test-kernels
	@echo ""
	@echo "  $(C_ORANGE)Running Unit Tests$(C_RESET)"
	@echo "  (Auto-escalates to DEBUG mode on failure)"
	@echo ""
	@cd unittest && LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH python3 test_kv_cache_layer_decode.py
	@cd unittest && LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH python3 test_multi_layer_parity.py

# Full test: download model, convert, compile, run smoke + parity tests
test-v4: $(LIB)
	@echo ""
	@echo "  $(C_ORANGE)Full Test Suite (Qwen2-0.5B Q4_K)$(C_RESET)"
	@echo "  Downloads model, converts, compiles, runs all tests"
	@echo ""
	@python3 scripts/ck_run_v4.py run \
		https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF \
		--weight-dtype=q4_k \
		--test \
		--test-only

# Full test with force recompile (catches stale build issues)
test-v4-clean: $(LIB)
	@echo ""
	@echo "  $(C_ORANGE)Clean Test (force recompile)$(C_RESET)"
	@echo ""
	@python3 scripts/ck_run_v4.py run \
		https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF \
		--weight-dtype=q4_k \
		--test \
		--test-only \
		--force-compile \
		--force-convert

.PHONY: test-unit test-v4 test-v4-clean test-kernels test-quant-server

# Generate C code for a model without compiling (for inspection)
# Usage: make generate-model MODEL=HuggingFaceTB/SmolLM-135M
generate-model: $(LIB) $(IR_DEMO) $(BUILD_DIR)/ck
ifndef MODEL
	@echo "Usage: make generate-model MODEL=<model-name>"
	@echo "Example: make generate-model MODEL=HuggingFaceTB/SmolLM-135M"
	@exit 1
endif
	@$(BUILD_DIR)/ck run $(MODEL) --generate-only --verbose

$(BUILD_DIR)/ck_main: $(BUILD_DIR) $(CK_MAIN) $(CK_TOKENIZER) include/ck_tokenizer.h
	$(CC) $(CFLAGS) -o $@ $(CK_MAIN) $(CK_TOKENIZER) -lm

$(BUILD_DIR)/ck_server: $(BUILD_DIR) $(CK_SERVER)
	$(CC) $(CFLAGS) -o $@ $(CK_SERVER) -lpthread

ck-chat: $(BUILD_DIR)/ck_main
	@echo "Interactive CLI built: $(BUILD_DIR)/ck_main"
	@echo "Usage: ./$(BUILD_DIR)/ck_main --help"

ck-server: $(BUILD_DIR)/ck_server
	@echo "Server built: $(BUILD_DIR)/ck_server"
	@echo "Usage: ./$(BUILD_DIR)/ck_server --port 8080"

ck-chat-py:
	$(PYTHON) $(PYTHONFLAGS) tools/ck_chat.py --model-dir $(SMOLLM_MODEL_DIR) --context $(SMOLLM_CONTEXT)

ck-server-py:
	$(PYTHON) $(PYTHONFLAGS) tools/ck_server.py --model-dir $(SMOLLM_MODEL_DIR) --context $(SMOLLM_CONTEXT)

# ============================================================================
# System Configuration and Topology
# ============================================================================

SHOW_CONFIG := $(BUILD_DIR)/show_config

$(SHOW_CONFIG): $(BUILD_DIR) src/system_topology.c src/show_config.c include/system_topology.h
	$(CC) -O3 -Wall -Wno-format-truncation -fopenmp -Iinclude -o $@ src/system_topology.c src/show_config.c

show_config: $(SHOW_CONFIG)
	@./$(SHOW_CONFIG)

show-config: show_config

# ============================================================================
# Status and Coverage Reports
# ============================================================================

opt-status:
	@$(PYTHON) scripts/optimization_status.py

opt-pending:
	@$(PYTHON) scripts/optimization_status.py --pending

opt-inference:
	@$(PYTHON) scripts/optimization_status.py --inference

opt-training:
	@$(PYTHON) scripts/optimization_status.py --training

opt-kernels:
	@$(PYTHON) scripts/optimization_status.py --kernels

opt-targets:
	@$(PYTHON) scripts/optimization_status.py --targets

opt-md:
	@$(PYTHON) scripts/optimization_status.py --markdown

kernel-coverage:
	@$(PYTHON) scripts/kernel_coverage.py

kernel-coverage-md:
	@$(PYTHON) scripts/kernel_coverage.py --markdown

test-coverage:
	@$(PYTHON) scripts/test_coverage.py

test-coverage-md:
	@$(PYTHON) scripts/test_coverage.py --markdown

# Nightly test runner (runs all tests, doesn't stop on failure)
nightly:
	@$(PYTHON) scripts/nightly_runner.py

nightly-quick:
	@$(PYTHON) scripts/nightly_runner.py --quick

nightly-json:
	@$(PYTHON) scripts/nightly_runner.py --json $(BUILD_DIR)/nightly_report.json

nightly-baseline:
	@$(PYTHON) scripts/nightly_runner.py --save-baseline

nightly-kernels:
	@$(PYTHON) scripts/nightly_runner.py --category kernels

nightly-bf16:
	@$(PYTHON) scripts/nightly_runner.py --category bf16

nightly-quant:
	@$(PYTHON) scripts/nightly_runner.py --category quant

nightly-parity:
	@$(PYTHON) scripts/nightly_runner.py --category parity

nightly-list:
	@$(PYTHON) scripts/nightly_runner.py --list

# ============================================================================
# Status Reports (reads from meta/kernel_meta.json)
# ============================================================================
# Usage:
#   make report        - Full comprehensive report (kernel status, roadmaps, tests)
#   make opt-status    - Quick kernel implementation table with opt levels
#   make opt-pending   - Show what's not done yet
#   make test-coverage - Show test file coverage
#
# To update the report:
#   1. Edit meta/kernel_meta.json when you add/modify kernels
#   2. Update "opt_level" arrays when adding SIMD/blocking/parallel
#   3. Run "make report" to see updated status
#
# Optional validation (checks JSON matches source code):
#   make meta-check    - Report discrepancies between JSON and code
# ============================================================================

meta-check:
	@$(PYTHON) scripts/sync_kernel_meta.py --check

meta-sync:
	@$(PYTHON) scripts/sync_kernel_meta.py --update

# Comprehensive report - runs all status reports
report:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗"
	@echo "║                              C-KERNEL-ENGINE COMPREHENSIVE REPORT                                ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  1. KERNEL IMPLEMENTATION STATUS (with optimization levels)                                      │"
	@echo "│     Legend: A1=AVX1, A5=AVX512, BF=BF16, AM=AMX, +=blocked/parallel/fused, S=scalar              │"
	@echo "└──────────────────────────────────────────────────────────────────────────────────────────────────┘"
	@$(PYTHON) scripts/optimization_status.py --kernels
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  2. INFERENCE OPTIMIZATION ROADMAP                                                               │"
	@echo "└──────────────────────────────────────────────────────────────────────────────────────────────────┘"
	@$(PYTHON) scripts/optimization_status.py --inference
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  3. TRAINING OPTIMIZATION ROADMAP                                                                │"
	@echo "└──────────────────────────────────────────────────────────────────────────────────────────────────┘"
	@$(PYTHON) scripts/optimization_status.py --training
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  4. SINGLE-CORE PRIORITIES & PERFORMANCE TARGETS                                                 │"
	@echo "└──────────────────────────────────────────────────────────────────────────────────────────────────┘"
	@$(PYTHON) scripts/optimization_status.py --single-core
	@$(PYTHON) scripts/optimization_status.py --targets
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  5. QUANTIZATION FORMAT SUPPORT                                                                  │"
	@echo "│     Formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (legacy) | Q4_K, Q6_K, Q8_K (k-quants, recommended)    │"
	@echo "└──────────────────────────────────────────────────────────────────────────────────────────────────┘"
	@echo "  K-Quants (recommended for inference):"
	@echo "    Q4_K  - 4-bit k-quant with super-blocks (most GGUF models use this)"
	@echo "    Q6_K  - 6-bit k-quant (higher quality)"
	@echo "    Q8_K  - 8-bit k-quant (activation quantization)"
	@echo ""
	@echo "  Legacy Formats (for compatibility):"
	@echo "    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0"
	@echo ""
	@echo "  Parity Tests:"
	@echo "    PyTorch: make layer-parity, make smollm-train-parity"
	@echo "    llama.cpp: make llamacpp-parity (requires submodule)"
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  6. TEST COVERAGE                                                                                │"
	@echo "└──────────────────────────────────────────────────────────────────────────────────────────────────┘"
	@$(PYTHON) scripts/test_coverage.py --summary
	@$(PYTHON) scripts/test_coverage.py --missing
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  7. HIGH-PRIORITY PENDING WORK                                                                   │"
	@echo "└──────────────────────────────────────────────────────────────────────────────────────────────────┘"
	@$(PYTHON) scripts/optimization_status.py --pending
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗"
	@echo "║                                        END OF REPORT                                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════════════════════════════╝"

# Generate markdown report for documentation
report-md:
	@echo "# C-Kernel-Engine Status Report"
	@echo ""
	@echo "Generated: $$(date)"
	@echo ""
	@$(PYTHON) scripts/kernel_coverage.py --markdown
	@echo ""
	@$(PYTHON) scripts/optimization_status.py --markdown

.PHONY: all clean test test-bf16 test-libs test-quant test-flash-attention test_flash_attention unittest unittest-show show_test help litmus litmus-test test-quick test-full test-stress profile-memory profile-heap profile-cpu profile-flash-attn profile-cache flamegraph ck-cli ck-cli-v4 ck-cli-v5 ck-chat ck-server ck-chat-py ck-server-py generate-model gguf-inspect gguf-list gguf-to-bump gguf-to-bump-v4 hf-to-bump-v4 ir-v4 ir-v4-q4k opt-status opt-pending opt-inference opt-training opt-kernels opt-targets opt-md kernel-coverage kernel-coverage-md test-coverage test-coverage-md meta-check meta-sync meta-init report report-md show_config show-config v5 demo-v5 demo-v5-debug llamacpp-parity llamacpp-parity-full showtests version version-history
