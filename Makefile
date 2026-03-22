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
# OpenMP is opt-in for runtime stability (v6.6 uses threadpool parallelism).
# Enable explicitly when needed:
#   make CK_ENABLE_OPENMP=1
CK_ENABLE_OPENMP ?= 0
OPENMP_FLAG :=
ifeq ($(CK_ENABLE_OPENMP),1)
OPENMP_FLAG := -fopenmp
ifneq (,$(findstring icc,$(CC)))
OPENMP_FLAG := -qopenmp
endif
ifneq (,$(findstring icx,$(CC)))
OPENMP_FLAG := -qopenmp
endif
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
ifneq (,$(findstring avx_vnni,$(CPU_FLAGS)))
DEFAULT_AVX_FLAGS_GCC += -mavxvnni
endif
ifneq (,$(findstring avxvnni,$(CPU_FLAGS)))
DEFAULT_AVX_FLAGS_GCC += -mavxvnni
endif
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
# Auto-detection: default to native backend for runtime portability.
# Use MKL/oneDNN explicitly when needed:
#   make USE_MKL=1
#   make USE_ONEDNN=1
# (This avoids pulling Intel OpenMP runtime into default builds.)
USE_NATIVE ?= 1
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
            src/kernels/fused/prefill_fused_gemm.c \
            src/kernels/fused/mega_fused_attention_avx.c \
            src/kernels/fused/mega_fused_attention_prefill.c \
            src/kernels/fused/mega_fused_attention_prefill_q8_0.c \
            src/kernels/fused/mega_fused_outproj_mlp_prefill.c \
            src/kernels/fused/gemv_fused_quant_bias.c \
            src/kernels/gemm_head_major_output.c \
            src/kernels/gemm_microkernel.c \
	           src/kernels/layernorm_kernels.c \
	           src/kernels/layernorm_kernels_bf16.c \
	           src/kernels/gelu_kernels.c \
	           src/kernels/geglu_kernels.c \
	           src/kernels/gelu_kernels_bf16.c \
	           src/kernels/softmax_kernels.c \
	           src/kernels/softmax_kernels_bf16.c \
	           src/kernels/attention_kernels.c \
	           src/kernels/attention_kernels_sliding.c \
	           src/kernels/attention_flash_true.c \
	           src/kernels/ssm_kernels.c \
	           src/kernels/hybrid_attention_kernels.c \
	           src/kernels/recurrent_split_kernels.c \
	           src/kernels/recurrent_gate_kernels.c \
	           src/kernels/recurrent_state_kernels.c \
	           src/kernels/recurrent_qk_norm_kernels.c \
	           src/kernels/recurrent_norm_kernels.c \
	           src/kernels/deltanet_kernels.c \
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
	            src/kernels/qk_norm_kernels.c \
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
	           src/kernels/gemm_kernels_q5_1_q8_1.c \
	           src/kernels/gemm_kernels_q5_k.c \
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
	           src/kernels/fused/fused_rmsnorm_linear.c \
	           src/kernels/gemm_kernels_q8_0.c \
	           src/kernels/gemm_kernels_q8_0_q8_0_contract.c \
	           src/kernels/gemv_omp.c \
	           src/kernels/quantize_row_q8_k_sse.c \
	           src/kernels/quantize_row_q8_k_avx.c \
	           src/kernels/quantize_row_q8_k_avx2.c \
	           src/kernels/fused/rmsnorm_q8_k_fused.c \
	           src/kernels/gemm_kernels_f16.c \
	           src/kernels/optimizer_kernels.c \
	           src/kernels/optimizer_kernels_bf16.c \
	           src/kernels/add_kernels_bf16.c \
	           src/kernels/topk_kernels.c \
	           src/kernels/axpy_kernels.c \
	           src/kernels/fused/rmsnorm_qkv.c \
	           src/kernels/fused/attention_mlp_fused.c \
	           src/ck_threadpool.c \
           src/ck_parallel_train.c
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
LIB_PARITY   := $(BUILD_DIR)/libck_parity.so

# Tokenizer library (new - from src/tokenizer/)
SRCS_TOKENIZER := src/tokenizer/murmurhash3.c \
                  src/tokenizer/hash_table.c \
                  src/tokenizer/memory_pool.c \
                  src/tokenizer/utf8.c \
                  src/tokenizer/tokenizer.c \
                  src/tokenizer/tokenizer_spm.c \
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

PYTHON  ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
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
            unittest/test_qk_norm.py \
            tests/test_deltanet.py \
            unittest/test_swiglu.py \
            unittest/test_fused_swiglu_decode.py \
            unittest/test_fused_attention_decode.py \
            unittest/fusion/test_mega_fused_attention.py \
            unittest/test_sigmoid.py \
            unittest/test_relu.py \
            unittest/test_attention.py \
            unittest/test_attention_sliding_contract.py \
            unittest/test_attention_backward.py \
            unittest/test_kv_cache_attention.py \
            unittest/test_kv_cache_layer_decode.py \
            unittest/test_rope.py \
            unittest/test_embedding.py \
            unittest/test_cross_entropy.py \
            unittest/test_orchestration_layer.py \
            unittest/test_lm_head_litmus.py \
            unittest/test_optimizer.py \
            unittest/test_gemv_kernels_comprehensive.py \
            unittest/test_fused_rmsnorm_qkv.py \
            unittest/test_prefill_fused_rmsnorm_qkv_quant.py \
            unittest/test_prefill_fused_mlp_quant.py \
            unittest/fusion/test_mega_fused_attention_prefill.py \
            unittest/fusion/test_mega_fused_attention_prefill_q8_0.py \
            unittest/fusion/test_mega_fused_outproj_mlp_prefill.py \
            unittest/test_fused_attention_mlp.py

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
	@printf 'CC=%s\nCFLAGS=%s\nLDFLAGS=%s\n' "$(CC)" "$(CFLAGS)" "$(LDFLAGS)" > $@.tmp
	@if [ ! -f $@ ] || ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm $@.tmp; fi

$(LIB): $(BUILD_STAMP) $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -o $@ $(SRCS) $(LDFLAGS) -lm -lpthread

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
	@mkdir -p $(BUILD_DIR)
	# Bind tokenizer-internal symbol references locally to avoid collisions with
	# legacy tokenizer symbols that may also be exported by libckernel_engine.so.
	$(CC) $(CFLAGS) -shared -Wl,-Bsymbolic -o $@ $(SRCS_TOKENIZER) -lm

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

test-tokenizer-special: $(LIB_TOKENIZER)
	@echo ""
	@echo "========================================"
	@echo "  Special Token & Byte Decode Tests"
	@echo "========================================"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_true_bpe_special_tokens.py

test-tokenizer-spm: $(LIB_TOKENIZER)
	@echo ""
	@echo "========================================"
	@echo "  REAL SPM Parity Tests (SentencePiece)"
	@echo "========================================"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_tokenizer_spm_real.py
	@echo ""
	@echo "========================================"
	@echo "  Tokenizer Codegen Sync (init vs C)"
	@echo "========================================"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) version/v6.6/test/test_tokenizer_codegen_sync.py
	@echo ""
	@echo "========================================"
	@echo "  Model Tokenizer Regression (GGUF vs C)"
	@echo "========================================"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) version/v6.6/test/test_tokenizer_model_parity.py

.PHONY: tokenizer test-tokenizer test-tokenizer-quick test-tokenizer-llama test-tokenizer-special test-tokenizer-spm

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
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/fusion/test_mega_fused_attention.py --correctness

test-mega-fused-perf: test-mega-fused-correctness $(CK_CLI_V6_5)
	@echo ""
	@echo "========================================"
	@echo "  STEP 2: DRAM PRESSURE TEST"
	@echo "  (THE CRITICAL TEST - fusion's whole point!)"
	@echo "========================================"
	@mkdir -p test_results
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/fusion/test_mega_fused_attention.py --perf --model $(TEST_MODEL) --tokens $(TEST_TOKENS)

test-mega-fused-flamegraph: test-mega-fused-correctness $(CK_CLI_V6_5)
	@echo ""
	@echo "========================================"
	@echo "  STEP 3: FLAMEGRAPH VISUALIZATION"
	@echo "  (Visual confirmation of reduced memory)"
	@echo "========================================"
	@mkdir -p test_results
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/fusion/test_mega_fused_attention.py --flamegraph --model $(TEST_MODEL) --tokens $(TEST_TOKENS)

test-mega-fused: $(CK_CLI_V6_5)
	@echo ""
	@echo "========================================"
	@echo "  MEGA-FUSED ATTENTION: ALL TESTS"
	@echo "  (Correctness → Performance → Flamegraph)"
	@echo "========================================"
	@mkdir -p test_results
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/fusion/test_mega_fused_attention.py --all --model $(TEST_MODEL) --tokens $(TEST_TOKENS)

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

# ============================================================================
# FUSED KERNEL TESTS (v6.6 Fusion Suite)
# ============================================================================
# Run all fused kernel tests with unified output showing:
#   1. Individual kernel benchmarks (rmsnorm, gemv)
#   2. Non-fused pipeline (baseline)
#   3. Fused kernel (optimized)
#   4. Speedup summary with DRAM analysis

test-fused: $(LIB)
	@echo ""
	@echo "========================================"
	@echo "  FUSED KERNEL TESTS (v6.6 Suite)"
	@echo "========================================"
	@echo ""
	@mkdir -p test_results
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(PYTHON) $(PYTHONFLAGS) unittest/test_fused_rmsnorm_qkv.py

# Quick accuracy-only fused test (no benchmarks)
test-fused-quick: $(LIB)
	@echo "Running fused kernel accuracy tests..."
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(PYTHON) $(PYTHONFLAGS) unittest/test_fused_rmsnorm_qkv.py --quick

# All fused kernel tests including perf analysis
test-fused-all: $(LIB)
	@echo ""
	@echo "========================================"
	@echo "  FUSED KERNEL TESTS (Full Suite)"
	@echo "========================================"
	@mkdir -p test_results
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(PYTHON) $(PYTHONFLAGS) unittest/test_fused_rmsnorm_qkv.py --all

.PHONY: test-fused test-fused-quick test-fused-all

# Fused kernel parity: Test with different model sizes
test-fused-small: $(LIB)
	@echo ""
	@echo "========================================"
	@echo "  FUSED KERNEL TEST: Small Model (Qwen2-0.5B)"
	@echo "========================================"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(PYTHON) $(PYTHONFLAGS) unittest/test_fused_rmsnorm_qkv.py \
		--embed 896 --q-dim 896 --kv-dim 128 --iter 100

test-fused-7b: $(LIB)
	@echo ""
	@echo "========================================"
	@echo "  FUSED KERNEL TEST: Llama 7B Scale"
	@echo "========================================"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(PYTHON) $(PYTHONFLAGS) unittest/test_fused_rmsnorm_qkv.py \
		--embed 4096 --q-dim 4096 --kv-dim 4096 --iter 50

test-fused-13b: $(LIB)
	@echo ""
	@echo "========================================"
	@echo "  FUSED KERNEL TEST: Llama 13B Scale"
	@echo "  (embed_dim > 4096 not yet supported)"
	@echo "========================================"
	@echo "TODO: Increase stack buffer size or use heap allocation"

# Full fused parity suite: all model sizes
test-fused-parity-full: $(LIB)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║         FUSED KERNEL PARITY SUITE (Q4_K Weights)                     ║"
	@echo "╠══════════════════════════════════════════════════════════════════════╣"
	@echo "║  Tests fused vs separate kernels with quantized weights              ║"
	@echo "║  Fusion benefit: normed[] stays in L1 cache                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(MAKE) --no-print-directory test-fused-small
	@echo ""
	@$(MAKE) --no-print-directory test-fused-7b
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║                    FUSED PARITY SUITE COMPLETE                       ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"

.PHONY: test-fused-small test-fused-7b test-fused-13b test-fused-parity-full

# Mega-fused Attention + MLP test
test-mega-fused-block: $(LIB)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  MEGA-FUSED TEST: Attention + MLP Block                              ║"
	@echo "║  Fuses entire block from attention output to next layer input        ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(PYTHON) $(PYTHONFLAGS) unittest/test_fused_attention_mlp.py

test-mega-fused-block-7b: $(LIB)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  MEGA-FUSED TEST: Attention + MLP (Llama 7B Scale)                   ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(PYTHON) $(PYTHONFLAGS) unittest/test_fused_attention_mlp.py \
		--embed 4096 --intermediate 11008 --heads 32 --kv-heads 32 --head-dim 128 --seq-len 512

# Test fused GEMV kernels (quantize + gemv + bias)
test-fusion-gemv: $(LIB)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  FUSED GEMV TEST: quantize_row_q8_0 + gemv + bias_add                ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(PYTHON) $(PYTHONFLAGS) unittest/fusion/test_gemv_fused_quant_bias.py

test-fusion-gemv-quick: $(LIB)
	@LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH \
		$(PYTHON) $(PYTHONFLAGS) unittest/fusion/test_gemv_fused_quant_bias.py --quick

# All fusion tests combined
test-fusion-all: $(LIB)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║         COMPLETE FUSION TEST SUITE (v6.6)                            ║"
	@echo "╠══════════════════════════════════════════════════════════════════════╣"
	@echo "║  1. RMSNorm + QKV fusion                                             ║"
	@echo "║  2. Mega-fused Attention + MLP block                                 ║"
	@echo "║  3. Fused GEMV (quantize + gemv + bias)                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@$(MAKE) --no-print-directory test-fused
	@echo ""
	@$(MAKE) --no-print-directory test-mega-fused-block
	@echo ""
	@$(MAKE) --no-print-directory test-fusion-gemv
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║                    ALL FUSION TESTS COMPLETE                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"

.PHONY: test-mega-fused-block test-mega-fused-block-7b test-fusion-all test-fusion-gemv test-fusion-gemv-quick

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

$(LIB_GELU): $(BUILD_STAMP) src/kernels/gelu_kernels.c src/kernels/geglu_kernels.c src/kernels/gelu_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/gelu_kernels.c src/kernels/geglu_kernels.c src/kernels/gelu_kernels_bf16.c -lm

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

$(LIB_ATTENTION): $(BUILD_STAMP) src/kernels/attention_kernels.c src/kernels/attention_kernels_sliding.c src/kernels/attention_flash_true.c src/kernels/softmax_kernels.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/attention_kernels.c src/kernels/attention_kernels_sliding.c src/kernels/attention_flash_true.c src/kernels/softmax_kernels.c -lm

$(LIB_ROPE): $(BUILD_STAMP) src/kernels/rope_kernels.c src/kernels/rope_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/rope_kernels.c src/kernels/rope_kernels_bf16.c -lm

$(LIB_QUANT): $(BUILD_STAMP) src/kernels/dequant_kernels.c src/kernels/gemm_kernels_q4_0.c src/kernels/gemm_kernels_q4_1.c src/kernels/gemm_kernels_q5_0.c src/kernels/gemm_kernels_q5_0_sse_v2.c src/kernels/gemm_kernels_q5_1.c src/kernels/gemm_kernels_q5_1_q8_1.c src/kernels/gemm_kernels_q4k.c src/kernels/gemm_kernels_q6k.c src/kernels/gemm_kernels_q4k_q8k.c src/kernels/gemm_kernels_q4k_sse.c src/kernels/gemm_kernels_q4k_q8k_avx2.c src/kernels/gemm_kernels_q4k_q8k_vnni.c src/kernels/gemm_kernels_q8_0.c src/kernels/gemm_kernels_q8_0_q8_0_contract.c src/kernels/gemm_kernels_f16.c src/kernels/quantize_row_q8_k_sse.c src/kernels/quantize_row_q8_k_avx.c src/kernels/quantize_row_q8_k_avx2.c include/ckernel_quant.h include/ckernel_dtype.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/dequant_kernels.c src/kernels/gemm_kernels_q4_0.c src/kernels/gemm_kernels_q4_1.c src/kernels/gemm_kernels_q5_0.c src/kernels/gemm_kernels_q5_0_sse_v2.c src/kernels/gemm_kernels_q5_1.c src/kernels/gemm_kernels_q5_1_q8_1.c src/kernels/gemm_kernels_q4k.c src/kernels/gemm_kernels_q6k.c src/kernels/gemm_kernels_q4k_q8k.c src/kernels/gemm_kernels_q4k_sse.c src/kernels/gemm_kernels_q4k_q8k_avx2.c src/kernels/gemm_kernels_q4k_q8k_vnni.c src/kernels/gemm_kernels_q8_0.c src/kernels/gemm_kernels_q8_0_q8_0_contract.c src/kernels/gemm_kernels_f16.c src/kernels/quantize_row_q8_k_sse.c src/kernels/quantize_row_q8_k_avx.c src/kernels/quantize_row_q8_k_avx2.c -lm

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

test-deltanet: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) tests/test_deltanet.py $(ARGS)

test-ssm-conv: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_ssm_conv.py $(ARGS)

test-recurrent-split-qkv: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_recurrent_split_qkv.py $(ARGS)

test-split-q-gate: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_split_q_gate.py $(ARGS)

test-recurrent-dt-gate: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_recurrent_dt_gate.py $(ARGS)

test-recurrent-conv-state-update: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_recurrent_conv_state_update.py $(ARGS)

test-recurrent-silu: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_recurrent_silu.py $(ARGS)

test-recurrent-split-conv-qkv: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_recurrent_split_conv_qkv.py $(ARGS)

test-recurrent-qk-l2-norm: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_recurrent_qk_l2_norm.py $(ARGS)

test-recurrent-norm-gate: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_recurrent_norm_gate.py $(ARGS)

test-attn-gate-sigmoid-mul: $(LIB)
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(PYTHON) $(PYTHONFLAGS) unittest/test_attn_gate_sigmoid_mul.py $(ARGS)

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

# =============================================================================
# E2E Integration Tests
# =============================================================================
# Full end-to-end integration tests: kernel compilation, IR codegen, inference
# Downloads Qwen/SmolLM if not cached, runs full pipeline validation
#
# Usage:
#   make e2e           - Run full integration tests (uses cached model if available)
#   make e2e-quick     - Quick test with smallest available model
#   make e2e-qwen      - Test with Qwen2-0.5B (recommended, ~400MB)
#   make e2e-smollm    - Test with SmolLM2-360M (~200MB)
#
e2e:
	@echo "========================================"
	@echo "  CK-Engine E2E Integration Tests"
	@echo "========================================"
	@bash scripts/full_integration_testing.sh

e2e-quick:
	@echo "Running quick E2E test..."
	@bash scripts/full_integration_testing.sh --quick 2>/dev/null || bash scripts/full_integration_testing.sh

e2e-qwen:
	@echo "Running E2E test with Qwen2-0.5B..."
	@$(PYTHON) scripts/v6.5/ck_run_v6_5.py run "hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf" --prompt "Hello" --max-tokens 10

e2e-smollm:
	@echo "Running E2E test with SmolLM2-360M..."
	@$(PYTHON) scripts/v6.5/ck_run_v6_5.py run "hf://itlwas/SmolLM2-360M-Q4_K_M-GGUF/smollm2-360m-instruct-q4_k_m.gguf" --prompt "Hello" --max-tokens 10

# v6.6 E2E test - runs release gate + local sanity.
# NOTE:
# - `v6.6-gate` is runtime-optional for parity (SKIP is allowed when llama runtime is absent).
# - Use `v6.6-validate-parity-matrix-required` for strict runtime-required parity.
e2e-v66:
	@echo "========================================"
	@echo "  CK-Engine v6.6 E2E Test"
	@echo "========================================"
	@echo "[1/2] Run v6.6 gate suite (contracts + matrix-smoke + parity + long decode)..."
	@$(MAKE) --no-print-directory v6.6-gate || { echo "v6.6 gate failed"; exit 1; }
	@echo "[2/2] Run quick sanity tests..."
	@cd version/v6.6/test && $(MAKE) quick || { echo "Tests failed"; exit 1; }
	@echo "========================================"
	@echo "  v6.6 E2E Test PASSED"
	@echo "========================================"

# v6.6 E2E with full test suite
e2e-v66-full:
	@echo "========================================"
	@echo "  CK-Engine v6.6 Full E2E Test"
	@echo "========================================"
	@$(MAKE) --no-print-directory v6.6-gate || { echo "v6.6 gate failed"; exit 1; }
	@cd version/v6.6/test && $(MAKE) all
	@echo "========================================"
	@echo "  v6.6 Full E2E Test PASSED"
	@echo "========================================"

# =============================================================================
# LOCAL CI - Mirrors .github/workflows/ci-fast.yml
# =============================================================================
# Run this before pushing to catch failures early
# Each step stops on first failure

ci-local:
	@echo "========================================"
	@echo "  CK-Engine Local CI"
	@echo "========================================"
	@echo ""
	@echo "[1/7] Tooling contract validation..."
	@$(PYTHON) version/v6.6/scripts/validate_tooling_contracts.py || { echo "FAIL: Tooling contracts"; exit 1; }
	@echo ""
	@echo "[2/7] Kernel map validation..."
	@$(PYTHON) version/v6.6/kernel_maps/test_validation.py --quick || { echo "FAIL: Kernel map validation"; exit 1; }
	@echo ""
	@echo "[3/7] Registry validation..."
	@$(PYTHON) version/v6.6/scripts/validate_kernel_registry.py || { echo "FAIL: Registry validation"; exit 1; }
	@echo ""
	@echo "[4/7] Unit tests (bindings, IR lowering, templates)..."
	@$(PYTHON) -m pytest version/v6.6/test/test_kernel_bindings.py version/v6.6/test/test_ir_lowering.py version/v6.6/test/test_template_smoke.py -v || { echo "FAIL: Unit tests"; exit 1; }
	@echo ""
	@echo "[5/7] Build CK-Engine..."
	@$(MAKE) clean || true
	@$(MAKE) -j$(nproc) || { echo "FAIL: Build"; exit 1; }
	@echo ""
	@echo "[6/7] E2E Qwen2..."
	@$(PYTHON) version/v6.6/scripts/ck_run_v6_6.py run "hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf" --prompt "Hello" --max-tokens 4 --context-len 512 --force-compile || { echo "FAIL: Qwen2 E2E"; exit 1; }
	@echo ""
	@echo "[7/7] E2E Qwen3..."
	@$(PYTHON) version/v6.6/scripts/ck_run_v6_6.py run "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf" --prompt "Hello" --max-tokens 4 --context-len 512 --force-compile || { echo "FAIL: Qwen3 E2E"; exit 1; }
	@echo ""
	@echo "========================================"
	@echo "  Local CI PASSED - safe to push"
	@echo "========================================"

# Fast version: validation only, no build/inference
ci-local-fast:
	@echo "========================================"
	@echo "  CK-Engine Local CI (Fast)"
	@echo "========================================"
	@echo ""
	@echo "[1/4] Tooling contract validation..."
	@$(PYTHON) version/v6.6/scripts/validate_tooling_contracts.py || { echo "FAIL: Tooling contracts"; exit 1; }
	@echo ""
	@echo "[2/4] Kernel map validation..."
	@$(PYTHON) version/v6.6/kernel_maps/test_validation.py --quick || { echo "FAIL: Kernel map validation"; exit 1; }
	@echo ""
	@echo "[3/4] Registry validation..."
	@$(PYTHON) version/v6.6/scripts/validate_kernel_registry.py || { echo "FAIL: Registry validation"; exit 1; }
	@echo ""
	@echo "[4/4] Unit tests (bindings, IR lowering, templates)..."
	@$(PYTHON) -m pytest version/v6.6/test/test_kernel_bindings.py version/v6.6/test/test_ir_lowering.py version/v6.6/test/test_template_smoke.py -v || { echo "FAIL: Unit tests"; exit 1; }
	@echo ""
	@echo "========================================"
	@echo "  Fast CI PASSED - ready for full CI"
	@echo "========================================"

# =============================================================================
# LAYERED PIPELINE TESTS
# =============================================================================
# These tests validate each stage of the inference pipeline independently.
# Run them in order to pinpoint exactly where failures occur.
#
# Layer 1: Kernel parity with llama.cpp (do kernels match?)
# Layer 2: Bump conversion (are weights correct?)
# Layer 3: IR structure (is computation graph correct?)
# Layer 4: Codegen validation (does code compile?)
# Layer 5: Tensor flow (are dimensions correct?)
# Layer 6: E2E inference (does it produce coherent output?)
# =============================================================================

# Run all 6 layers of testing (stops at first failure)
test-full-pipeline:
	@echo "========================================"
	@echo "  CK-Engine Full Pipeline Validation"
	@echo "========================================"
	@bash scripts/test_full_pipeline.sh

# Quick pipeline test (skips slow kernel parity)
test-pipeline-quick:
	@bash scripts/test_full_pipeline.sh --quick

# Pipeline layers 1..5 only (skip L6 E2E)
test-pipeline-l1-l5:
	@bash scripts/test_full_pipeline.sh --max-layer 5

# Layer 1: Kernel parity with llama.cpp
test-kernel-parity:
	@echo "[Layer 1] Kernel Parity Tests..."
	@if [ -f scripts/test_kernels_vs_llamacpp.py ]; then \
		$(PYTHON) scripts/test_kernels_vs_llamacpp.py --quick; \
	else \
		echo "Kernel parity test not found"; \
	fi

test-kernel-parity-full:
	@echo "[Layer 1] Full Kernel Parity Tests..."
	@$(PYTHON) scripts/test_kernels_vs_llamacpp.py 2>&1 || echo "Run 'make parity-libs' first"

# Layer 2: Bump conversion validation
test-bump-conversion:
	@echo "[Layer 2] Bump Conversion Tests..."
	@$(PYTHON) scripts/test_bump_conversion.py --auto

# Layer 3: IR structure validation
test-ir-validation:
	@echo "[Layer 3] IR Structure Validation..."
	@$(PYTHON) scripts/test_ir_validation.py --auto

# Layer 4: Codegen validation
test-codegen-validation:
	@echo "[Layer 4] Codegen Validation..."
	@$(PYTHON) scripts/test_codegen_validation.py --auto

# Layer 5: Tensor flow validation (the "gibberish" detector)
test-tensor-flow:
	@echo "[Layer 5] Tensor Flow Validation..."
	@$(PYTHON) scripts/test_tensor_flow.py --auto

# Individual layer with specific model
test-layer-%:
	@bash scripts/test_full_pipeline.sh --layer $*

# List all pipeline test layers
test-pipeline-help:
	@echo ""
	@echo "Layered Pipeline Tests:"
	@echo "  make test-full-pipeline      - Run all 6 layers"
	@echo "  make test-pipeline-quick     - Run layers 2-6 (skip kernel parity)"
	@echo "  make test-pipeline-l1-l5     - Run layers 1-5 (skip L6 e2e)"
	@echo ""
	@echo "Individual Layers:"
	@echo "  make test-kernel-parity      - Layer 1: Kernel parity (quick)"
	@echo "  make test-kernel-parity-full - Layer 1: Kernel parity (all)"
	@echo "  make test-bump-conversion    - Layer 2: Bump conversion"
	@echo "  make test-ir-validation      - Layer 3: IR structure"
	@echo "  make test-codegen-validation - Layer 4: Codegen"
	@echo "  make test-tensor-flow        - Layer 5: Tensor flow"
	@echo "  make e2e                     - Layer 6: E2E inference"
	@echo ""
	@echo "Run specific layer:"
	@echo "  make test-layer-1            - Run layer 1 only"
	@echo "  make test-layer-5            - Run layer 5 only"
	@echo ""

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
	@echo "  tests/test_deltanet.py                 - Gated DeltaNet forward/backward"
	@echo "  unittest/test_ssm_conv.py              - SSM causal conv forward/backward"
	@echo "  unittest/test_relu.py                  - ReLU activation"
	@echo "  unittest/test_attention.py             - Attention forward/backward"
	@echo "  unittest/test_attention_sliding_contract.py - Sliding-window attention contract"
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
	@echo "  make visualizer       - Run v7 IR visualizer E2E regression"
	@echo "  make visualizer-full  - Run visualizer E2E + train-runtime ASan artifact checks"
	@echo "  make v7-visualizer-health - Fast HTML/JS health check (tabs, functions, contracts)"
	@echo "  make v7-visualizer-generated-e2e - L3: generate + validate all visualizer artifacts"
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
	@echo "  make llamacpp-parity-full-all-isa-variants Full parity + AVX/AVX2/AVX-512 sweep"
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
	@echo "Mega-Fusion Attention Tests:"
	@echo "  make llamacpp-fusion-test-full     Full mega-fusion test vs llama.cpp"
	@echo "  make llamacpp-fusion-test-quick    Quick mega-fusion test"
	@echo ""
	@echo "Nightly / CI:"
	@echo "  make nightly          Run all tests with summary"
	@echo "  make nightly-quick    Quick subset (~5 min)"
	@echo "  make nightly-json     Run all + JSON report"
	@echo "  make nightly-kernels  Only kernel tests"
	@echo "  make nightly-bf16     Only BF16 tests"
	@echo "  make nightly-quant    Only quantization tests"
	@echo "  make nightly-parity   Only parity tests (PyTorch + llama.cpp + v7 visualizer E2E)"
	@echo "  make visualizer       Run v7 IR visualizer E2E regression"
	@echo "  make visualizer-full  Run visualizer E2E + train-runtime ASan artifact checks"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  LAYERED PIPELINE TESTS (Pinpoint Failures)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Full Pipeline (6 layers):"
	@echo "  make test-full-pipeline   Run all 6 validation layers"
	@echo "  make test-pipeline-quick  Skip kernel parity (faster)"
	@echo "  make test-pipeline-l1-l5  Run L1-L5 only (no e2e)"
	@echo ""
	@echo "Individual Layers:"
	@echo "  make test-kernel-parity      L1: Kernel parity vs llama.cpp"
	@echo "  make test-kernel-parity-full L1: Full kernel parity (all dtypes)"
	@echo "  make test-bump-conversion    L2: GGUF→Bump weight conversion"
	@echo "  make test-ir-validation      L3: IR structure validation"
	@echo "  make test-codegen-validation L4: Generated code validation"
	@echo "  make test-tensor-flow        L5: Tensor dimension flow (*gibberish detector*)"
	@echo "  make e2e                     L6: End-to-end inference"
	@echo ""
	@echo "Run Specific Layer:"
	@echo "  make test-layer-1 through test-layer-6"
	@echo ""
	@echo "E2E Integration:"
	@echo "  make e2e              Full integration test suite"
	@echo "  make e2e-quick        Quick E2E test"
	@echo "  make e2e-qwen         E2E with Qwen2-0.5B"
	@echo "  make e2e-smollm       E2E with SmolLM2-360M"
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
	  extra_env=""; \
	  case "$$t" in \
	    *test_gemm_microkernel.py|*test_gemv_kernels_comprehensive.py) extra_args="--quick";; \
	    *test_qk_norm.py) extra_args="--quick";; \
	    *test_mega_fused_attention.py) extra_args="--correctness";; \
	    *test_orchestration_layer.py) extra_args="--strict-ref";; \
	  esac; \
	  case "$$t" in \
	    *test_gemv_kernels_comprehensive.py) extra_env="CK_SKIP_IF_MISSING=1";; \
	  esac; \
	  LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(TEST_ENV) $$extra_env $(PYTHON) $(PYTHONFLAGS) $$t $$extra_args; \
	done; \
	echo "All Python kernel tests completed."
	@echo ""
	@echo "Running OpenMP GEMV parity tests..."
	@$(MAKE) --no-print-directory test-gemv-omp-quick
	@echo ""
	@echo "Running Thread Pool GEMV parity tests..."
	@$(MAKE) --no-print-directory test-threadpool-parity-quick
	@echo ""
	@echo "Running GEMM AVX vs scalar benchmark (quick)..."
	@$(MAKE) --no-print-directory test-gemm-avx-bench-quick
	@echo ""
	@echo "Kernel-focused suite complete."
	@echo "For v7 visualizer runbook E2E use: make visualizer"

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

# Benchmark parallel GEMV with OpenMP (tests true parallel speedup)
bench_parallel_gemv: $(LIB)
	$(CC) -O3 -qopenmp -Iinclude -o $(BUILD_DIR)/bench_parallel_gemv \
		benchmarks/bench_parallel_gemv.c -L$(BUILD_DIR) -lckernel_engine -lm \
		-Wl,-rpath,$(BUILD_DIR)
	./$(BUILD_DIR)/bench_parallel_gemv

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
	@echo "  tests/test_deltanet.py             - Gated DeltaNet forward/backward vs PyTorch"
	@echo "  unittest/test_ssm_conv.py          - SSM causal conv forward/backward vs PyTorch"
	@echo "  unittest/test_attention.py         - Multi-head attention forward vs PyTorch"
	@echo "  unittest/test_attention_sliding_contract.py - Sliding-window attention contract"
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

test-attention-sliding: $(LIB) test-libs
	$(PYTHON) $(PYTHONFLAGS) unittest/test_attention_sliding_contract.py

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

# Full parity test (assumes already built) — includes OMP kernel parity
llamacpp-parity-full:
	@echo "Running full llama.cpp parity test..."
	@./scripts/run_parity_smoketest.sh --skip-build
	@echo ""
	@echo "Running OpenMP GEMV parity tests (serial vs parallel)..."
	@$(MAKE) --no-print-directory test-gemv-omp
	@echo ""
	@echo "Running Thread Pool GEMV parity tests (serial vs dispatch)..."
	@$(MAKE) --no-print-directory test-threadpool-parity
	@echo ""
	@echo "Running GEMM AVX vs scalar benchmark..."
	@$(MAKE) --no-print-directory test-gemm-avx-bench
	@echo ""
	@echo "Running DeltaNet ISA benchmark..."
	@$(MAKE) --no-print-directory test-deltanet-avx-bench

# Nightly parity profile: keep correctness coverage, but use quick ISA parity
# benches so nightly does not block on long benchmark loops.
llamacpp-parity-nightly:
	@echo "Running nightly llama.cpp parity test..."
	@./scripts/run_parity_smoketest.sh --skip-build
	@echo ""
	@echo "Running OpenMP GEMV parity tests (serial vs parallel)..."
	@$(MAKE) --no-print-directory test-gemv-omp
	@echo ""
	@echo "Running Thread Pool GEMV parity tests (serial vs dispatch)..."
	@$(MAKE) --no-print-directory test-threadpool-parity
	@echo ""
	@echo "Running GEMM AVX vs scalar benchmark (quick)..."
	@$(MAKE) --no-print-directory test-gemm-avx-bench-quick
	@echo ""
	@echo "Running DeltaNet ISA benchmark (quick)..."
	@$(MAKE) --no-print-directory test-deltanet-avx-bench-quick

# Full parity + ISA variant sweep for GEMM AVX benchmarks
llamacpp-parity-full-all-isa-variants:
	@echo "Running ISA variant sweep (AVX/AVX2/AVX-512) + full parity..."
	@./scripts/test_isa_variants.sh
	@$(MAKE) --no-print-directory llamacpp-parity-full

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
# Mega-Fusion Attention Tests
# =============================================================================
# Test CK-Engine's mega-fused attention against llama.cpp implementation

# Mega-Fusion Attention Test (CK-Engine vs llama.cpp)
llamacpp-fusion-test-full:
	@echo "Running mega-fusion attention test with llama.cpp comparison..."
	@./scripts/run_mega_fusion_test.sh --skip-build

# Quick mega-fusion test
llamacpp-fusion-test-quick:
	@echo "Running quick mega-fusion attention test..."
	@./scripts/run_mega_fusion_test.sh --quick --skip-build

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

# =============================================================================
# OpenMP GEMV Parity & Speed Tests
# =============================================================================
# Compare serial GEMV kernels vs OpenMP-parallel variants:
#   - Numerical parity (max abs diff < 1e-3)
#   - Speed comparison (serial vs OMP, thread scaling)
#
# Targets:
#   make test-gemv-omp          Full parity + speed test (all model dimensions)
#   make test-gemv-omp-quick    Quick parity test (small dimensions)
#   make test-gemv-omp-verbose  Full test with detailed diff output

GEMV_OMP_BIN := $(BUILD_DIR)/test_gemv_omp_parity

$(GEMV_OMP_BIN): $(LIB) tests/test_gemv_omp_parity.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -O3 -march=native -fopenmp -Iinclude \
		tests/test_gemv_omp_parity.c \
		-L$(BUILD_DIR) -lckernel_engine -lm \
		-Wl,-rpath,$(BUILD_DIR) \
		-o $(GEMV_OMP_BIN)

test-gemv-omp: $(GEMV_OMP_BIN)
	@echo "Running OpenMP GEMV parity + speed test (full)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(GEMV_OMP_BIN)

test-gemv-omp-quick: $(GEMV_OMP_BIN)
	@echo "Running OpenMP GEMV parity test (quick)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(GEMV_OMP_BIN) --quick

test-gemv-omp-verbose: $(GEMV_OMP_BIN)
	@echo "Running OpenMP GEMV parity + speed test (verbose)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(GEMV_OMP_BIN) --verbose

.PHONY: test-gemv-omp test-gemv-omp-quick test-gemv-omp-verbose

# =============================================================================
# Thread Pool GEMV/GEMM Parity & Speed Tests
# =============================================================================
# Compare serial GEMV/GEMM kernels vs persistent pthread thread pool dispatch
# (ck_parallel_decode.h, ck_parallel_prefill.h). This is the PRODUCTION
# parallelization path — thread pool replaced OpenMP due to:
#   - Zero fork/join overhead (threads spin-wait on atomics, ~0.1us wake)
#   - No core oversubscription (one pool, known thread count)
#   - Consistent (ith, nth) work splitting to kernels
#
# Tests dispatch wrappers against serial baselines:
#   1. gemv_q8_0_q8_0_parallel_dispatch       (decode)
#   2. gemv_q5_0_q8_0_parallel_dispatch       (decode)
#   3. gemv_fused_q5_0_bias_parallel_dispatch  (decode, fused: quantize+gemv+bias)
#   4. Dispatch latency measurement            (overhead on small M)
#   5. gemm_nt_q5_0_q8_0_parallel_dispatch    (prefill, Q8_0 x Q5_0)
#   6. gemm_nt_q8_0_q8_0_parallel_dispatch    (prefill, Q8_0 x Q8_0)
#   7. gemm_nt_q6_k_q8_k_parallel_dispatch    (prefill, Q8_K x Q6_K)
#
# Targets:
#   make test-threadpool-parity          Full parity + speed test
#   make test-threadpool-parity-quick    Quick parity test
#   make test-threadpool-parity-verbose  Full test with detailed diff output

THREADPOOL_BIN := $(BUILD_DIR)/test_threadpool_parity
V66_SRC_DIR    := version/v6.6/src

$(THREADPOOL_BIN): $(LIB) tests/test_threadpool_parity.c $(V66_SRC_DIR)/ck_parallel_decode.c $(V66_SRC_DIR)/ck_parallel_prefill.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -O3 -march=native -Iinclude -I$(V66_SRC_DIR) \
		tests/test_threadpool_parity.c \
		$(V66_SRC_DIR)/ck_parallel_decode.c \
		$(V66_SRC_DIR)/ck_parallel_prefill.c \
		-L$(BUILD_DIR) -lckernel_engine -lm -lpthread \
		-Wl,-rpath,$(BUILD_DIR) \
		-o $(THREADPOOL_BIN)

test-threadpool-parity: $(THREADPOOL_BIN)
	@echo "Running Thread Pool GEMV parity + speed test (full)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(THREADPOOL_BIN)

test-threadpool-parity-quick: $(THREADPOOL_BIN)
	@echo "Running Thread Pool GEMV parity test (quick)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(THREADPOOL_BIN) --quick

test-threadpool-parity-verbose: $(THREADPOOL_BIN)
	@echo "Running Thread Pool GEMV parity + speed test (verbose)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(THREADPOOL_BIN) --verbose

.PHONY: test-threadpool-parity test-threadpool-parity-quick test-threadpool-parity-verbose

# =============================================================================
# GEMM AVX Benchmark: _avx (SSE4.1) vs _ref (scalar)
# =============================================================================
# Measures speedup of the SSE4.1-based gemm_nt_q8_0_q8_0_avx kernel over the
# scalar _ref fallback. Verifies parity and confirms dispatch routing.
#
# Targets:
#   make test-gemm-avx-bench       Full benchmark (5 configs, 5 iters each)
#   make test-gemm-avx-bench-quick Quick parity + benchmark (3 configs, 3 iters)

GEMM_AVX_BENCH_BIN := $(BUILD_DIR)/test_gemm_avx_bench

$(GEMM_AVX_BENCH_BIN): $(LIB) tests/test_gemm_avx_bench.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -O3 -march=native -Iinclude \
		tests/test_gemm_avx_bench.c \
		-L$(BUILD_DIR) -lckernel_engine -lm \
		-Wl,-rpath,$(BUILD_DIR) \
		-o $(GEMM_AVX_BENCH_BIN)

test-gemm-avx-bench: $(GEMM_AVX_BENCH_BIN)
	@echo "Running GEMM Q8_0 AVX vs scalar benchmark (full)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(GEMM_AVX_BENCH_BIN)

test-gemm-avx-bench-quick: $(GEMM_AVX_BENCH_BIN)
	@echo "Running GEMM Q8_0 AVX vs scalar benchmark (quick)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(GEMM_AVX_BENCH_BIN) --quick

.PHONY: test-gemm-avx-bench test-gemm-avx-bench-quick

# =============================================================================
# Gated DeltaNet ISA benchmark
# =============================================================================
# Targets:
#   make test-deltanet-avx-bench       Full benchmark (4 configs, 15 iters each)
#   make test-deltanet-avx-bench-quick Quick parity + benchmark (3 configs, 5 iters)

DELTANET_AVX_BENCH_BIN := $(BUILD_DIR)/test_deltanet_avx_bench

$(DELTANET_AVX_BENCH_BIN): $(LIB) tests/test_deltanet_avx_bench.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -O3 -march=native -Iinclude \
		tests/test_deltanet_avx_bench.c \
		-L$(BUILD_DIR) -lckernel_engine -lm \
		-Wl,-rpath,$(BUILD_DIR) \
		-o $(DELTANET_AVX_BENCH_BIN)

test-deltanet-avx-bench: $(DELTANET_AVX_BENCH_BIN)
	@echo "Running Gated DeltaNet ISA benchmark (full)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(DELTANET_AVX_BENCH_BIN)

test-deltanet-avx-bench-quick: $(DELTANET_AVX_BENCH_BIN)
	@echo "Running Gated DeltaNet ISA benchmark (quick)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(DELTANET_AVX_BENCH_BIN) --quick

.PHONY: test-deltanet-avx-bench test-deltanet-avx-bench-quick

# =============================================================================
# Gated DeltaNet benchmark: CK vs llama.cpp
# =============================================================================
# Targets:
#   make test-deltanet-vs-llamacpp-bench       Full benchmark (4 configs)
#   make test-deltanet-vs-llamacpp-bench-quick Quick benchmark (3 configs)

DELTANET_LLAMA_BENCH_BIN := $(BUILD_DIR)/test_deltanet_vs_llamacpp_bench

$(DELTANET_LLAMA_BENCH_BIN): $(LIB) $(LLAMA_KERNEL_TEST) tests/test_deltanet_vs_llamacpp_bench.c
	@mkdir -p $(BUILD_DIR)
	$(CC) -O3 -march=native -Iinclude \
		tests/test_deltanet_vs_llamacpp_bench.c \
		-L$(BUILD_DIR) -lckernel_engine \
		-L$(LLAMA_CPP_DIR) -lggml_kernel_test \
		-lm -lpthread \
		-Wl,-rpath,$(PWD)/$(BUILD_DIR) \
		-Wl,-rpath,$(PWD)/$(LLAMA_CPP_DIR) \
		-o $(DELTANET_LLAMA_BENCH_BIN)

test-deltanet-vs-llamacpp-bench: $(DELTANET_LLAMA_BENCH_BIN)
	@echo "Running Gated DeltaNet benchmark (CK vs llama.cpp)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$(LLAMA_CPP_DIR):$$LD_LIBRARY_PATH $(DELTANET_LLAMA_BENCH_BIN)

test-deltanet-vs-llamacpp-bench-quick: $(DELTANET_LLAMA_BENCH_BIN)
	@echo "Running Gated DeltaNet benchmark (CK vs llama.cpp, quick)..."
	LD_LIBRARY_PATH=$(BUILD_DIR):$(LLAMA_CPP_DIR):$$LD_LIBRARY_PATH $(DELTANET_LLAMA_BENCH_BIN) --quick

.PHONY: test-deltanet-vs-llamacpp-bench test-deltanet-vs-llamacpp-bench-quick

# =============================================================================
# OpenMP GEMV Profiling (serial vs parallel)
# =============================================================================
# Full profiling suite: perf stat, flamegraph, cachegrind, VTune, massif
#
# Targets:
#   make test-profile-full              All profiling tools
#   make test-profile-perf              perf stat only (HW counters)
#   make test-profile-flamegraph        Flamegraph comparison
#   make test-profile-cachegrind        Cache behavior analysis
#   make test-profile-vtune             Intel VTune analysis
#   make test-profile-massif            Heap memory profiling

test-profile-full: $(GEMV_OMP_BIN)
	@echo "Running full profiling suite (serial vs OpenMP GEMV)..."
	@./scripts/profile_gemv_omp.sh all

test-profile-perf: $(GEMV_OMP_BIN)
	@echo "Running perf stat comparison..."
	@./scripts/profile_gemv_omp.sh perf

test-profile-flamegraph: $(GEMV_OMP_BIN)
	@echo "Generating serial vs OpenMP flamegraphs..."
	@./scripts/profile_gemv_omp.sh flamegraph

test-profile-cachegrind: $(GEMV_OMP_BIN)
	@echo "Running cachegrind comparison..."
	@./scripts/profile_gemv_omp.sh cachegrind

test-profile-vtune: $(GEMV_OMP_BIN)
	@echo "Running VTune analysis..."
	@./scripts/profile_gemv_omp.sh vtune

test-profile-massif: $(GEMV_OMP_BIN)
	@echo "Running heap profiling comparison..."
	@./scripts/profile_gemv_omp.sh massif

.PHONY: test-profile-full test-profile-perf test-profile-flamegraph
.PHONY: test-profile-cachegrind test-profile-vtune test-profile-massif

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
	@echo "    make test-attention-sliding - Sliding-window kernel contract"
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
	@echo "  │  V6.6 PIPELINE                                                      │"
	@echo "  └─────────────────────────────────────────────────────────────────────┘"
	@echo "  make v6.6            Build v6.6 artifacts"
	@echo "  make v6.6-e2e        Manual E2E sweep (Qwen2/Qwen3/Gemma)"
	@echo "  make v6.6-validate-contracts   Static tooling contract checks"
	@echo "  make v6.6-kernel-map-gate      Kernel-map generator/test/registry gate"
	@echo "  make v6.6-validate-matrix      Dynamic 3-model build matrix (Gemma/Qwen2/Qwen3)"
	@echo "  make v6.6-validate-matrix-nightly  Nightly matrix (skip static preflight + retries)"
	@echo "  make v6.6-validate-matrix-smoke  Matrix + smoke checks"
	@echo "  make v6.6-validate-parity-matrix  Parity matrix (runtime-optional)"
	@echo "  make v6.6-validate-parity-matrix-required  Strict parity matrix (runtime-required)"
	@echo "  make v6.6-validate-longdecode  256-token long-decode stability matrix"
	@echo "  make v6.6-gate       Required v6.6 release gate"
	@echo "  make v6.6-memory-signoff  Emit memory_signoff.json (planner + advanced + KV)"
	@echo "  make v6.6-perf-gate  Perf gate (CK_PROFILE + perf stat + flamegraph + budget checks)"
	@echo "     budget env knobs: CK_V66_PERF_MIN_DECODE_TOK_S, CK_V66_PERF_MIN_IPC, CK_V66_PERF_MAX_CACHE_MISS_RATE, CK_V66_PERF_MAX_BRANCH_MISS_RATE"
	@echo "     runtime mode: V66_PERF_RUNTIME=python|cli  (default: cli)"
	@echo "     chat template: V66_CHAT_TEMPLATE=auto|none|qwen|gemma"
	@echo "     vtune capture: V66_WITH_VTUNE=1|0  (default: 1)"
	@echo "     prep/compile: V66_PREP_WITH_PYTHON=1|0, V66_FORCE_COMPILE=1|0  (defaults: 1,1)"
	@echo "     advanced passthrough: V66_RUN_ARGS='--chat-template none', V66_CLI_ARGS='--no-chat-template'"
	@echo "  make v6.6-help       Show v6.6 HF URLs + commands"
	@echo ""
	@echo "  ┌─────────────────────────────────────────────────────────────────────┐"
	@echo "  │  V7 TRAINING FOUNDATION                                             │"
	@echo "  └─────────────────────────────────────────────────────────────────────┘"
	@echo "  make v7-help         Show v7 commands and scope"
	@echo "  make v7-validate-contracts  Static v7 contract checks"
	@echo "  make v7-parity-1tok  Deterministic fp32 1-token parity (C vs PyTorch)"
	@echo "  make v7-gate         Required v7.0 correctness gate"
	@echo "  make v7              Alias for v7-gate"
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
	@echo "  make regression-fast  Run family smoke/coherence regression suite"
	@echo "  make regression-full  Run family regression suite with stitch/parity triage"
	@echo "  make regression-family FAMILY=qwen35  Run regression debug for one family"
	@echo "  make test-attention-sliding  Run sliding-window kernel contract test"
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
               src/ckernel_strict.c \
               src/ck_threadpool.c \
               src/kernels/dequant_kernels.c \
               src/kernels/gemm_kernels_q4k_q8k.c \
               src/kernels/gemm_kernels_q4k_q8k_avx2.c \
               src/kernels/gemm_kernels_q4k_q8k_vnni.c \
               src/kernels/gemm_kernels_q6k_q8k.c \
               src/kernels/gemm_kernels_q4k_sse.c \
               src/kernels/gemm_kernels_q4k_avx.c \
               src/kernels/gemm_kernels_q5_k.c \
               src/kernels/gemm_kernels_q5_0.c \
               src/kernels/gemm_kernels_q5_0_sse_v2.c \
               src/kernels/gemm_kernels_q5_1.c \
               src/kernels/gemm_kernels_q5_1_q8_1.c \
               src/kernels/gemm_kernels_q8_0.c \
               src/kernels/gemm_kernels_q8_0_q8_0_contract.c \
               src/kernels/gemm_batch_int8.c \
               src/kernels/quantize_row_q8_k_sse.c \
               src/kernels/quantize_row_q8_k_avx.c \
               src/kernels/quantize_row_q8_k_avx2.c \
               src/kernels/rmsnorm_kernels.c \
               src/kernels/rope_kernels.c \
               src/kernels/swiglu_kernels.c \
               src/kernels/softmax_kernels.c \
               src/kernels/sigmoid_kernels.c \
               src/kernels/gelu_kernels.c \
               src/kernels/geglu_kernels.c \
	               src/kernels/attention_kernels.c \
	               src/kernels/attention_kernels_sliding.c \
	               src/kernels/attention_flash_true.c \
	               src/kernels/ssm_kernels.c \
	               src/kernels/hybrid_attention_kernels.c \
	               src/kernels/recurrent_split_kernels.c \
	               src/kernels/recurrent_gate_kernels.c \
	               src/kernels/recurrent_state_kernels.c \
	               src/kernels/recurrent_qk_norm_kernels.c \
	               src/kernels/recurrent_norm_kernels.c \
	               src/kernels/deltanet_kernels.c \
               src/kernels/fused/prefill_fused_gemm.c \
               src/kernels/fused/mega_fused_outproj_mlp_prefill.c \
               src/kernels/fused/gemv_fused_quant_bias.c \
               src/kernels/add_kernels_bf16.c

# Build CK parity testing library
$(LIB_PARITY): $(BUILD_DIR) $(PARITY_SRCS)
	$(CC) $(CFLAGS) -shared -o $@ $(PARITY_SRCS) $(LDFLAGS) -lm -lpthread

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
	@mkdir -p "$(LLAMA_CPP_DIR)/tests"
	@cp patches/test-kernel-parity.cpp "$(LLAMA_CPP_DIR)/tests/test-kernel-parity.cpp"
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
	src/kernels/fused/prefill_fused_gemm.c \
	src/kernels/fused/mega_fused_attention_avx.c \
	src/kernels/fused/mega_fused_attention_prefill.c \
	src/kernels/fused/mega_fused_attention_prefill_q8_0.c \
	src/kernels/fused/mega_fused_outproj_mlp_prefill.c \
	src/kernels/fused/gemv_fused_quant_bias.c \
	src/kernels/gemm_head_major_output.c \
	src/kernels/gemm_microkernel.c \
	src/kernels/hybrid_attention_kernels.c \
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
CK_CLI_V66 := version/v6.6/src/ck_cli_v6.6.c
CK_CLI_V7 := version/v7/src/ck_cli_v7.c
CK_BPE_TRAIN_V7 := version/v7/src/ck_bpe_train.c

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
	@mkdir -p $(BUILD_DIR)
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

# v6.6 Native CLI (fusion-enabled kernels, cache-aware optimization)
$(BUILD_DIR)/ck-cli-v6.6: $(CK_CLI_V66) $(LIB_TOKENIZER)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $(CK_CLI_V66) -L$(BUILD_DIR) -lckernel_tokenizer -ldl -lpthread -lm -Wl,-rpath,$(BUILD_DIR)

ck-cli-v6.6: $(BUILD_DIR)/ck-cli-v6.6
	@echo ""
	@echo "  $(C_CYAN)C-Kernel-Engine v6.6 CLI$(C_RESET)"
	@echo "  Features: Fusion kernels, Cache-aware optimization, FP16 KV cache"
	@echo ""
	@echo "  Usage:"
	@echo "    ./$(BUILD_DIR)/ck-cli-v6.6 --model qwen           # Auto-discover from cache"
	@echo "    ./$(BUILD_DIR)/ck-cli-v6.6 --list                 # List available models"
	@echo "    ./$(BUILD_DIR)/ck-cli-v6.6 <model.so> <weights.bump>"
	@echo ""

# v7 Native CLI
$(BUILD_DIR)/ck-cli-v7: $(CK_CLI_V7) $(LIB_TOKENIZER)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $(CK_CLI_V7) -L$(BUILD_DIR) -lckernel_tokenizer -ldl -lpthread -lm -Wl,-rpath,$(BUILD_DIR)

ck-cli-v7: $(BUILD_DIR)/ck-cli-v7
	@echo ""
	@echo "  $(C_CYAN)C-Kernel-Engine v7 CLI$(C_RESET)"
	@echo "  Features: Native runtime profiling path, model discovery, chat templates"
	@echo ""
	@echo "  Usage:"
	@echo "    ./$(BUILD_DIR)/ck-cli-v7 --model qwen"
	@echo "    ./$(BUILD_DIR)/ck-cli-v7 --list"
	@echo "    ./$(BUILD_DIR)/ck-cli-v7 <model.so> <weights.bump>"
	@echo ""

$(BUILD_DIR)/ck-bpe-train: $(CK_BPE_TRAIN_V7)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $(CK_BPE_TRAIN_V7) -lpthread -lm

ck-bpe-train: $(BUILD_DIR)/ck-bpe-train
	@echo ""
	@echo "  $(C_CYAN)ck-bpe-train$(C_RESET)"
	@echo "  Usage: ./$(BUILD_DIR)/ck-bpe-train --corpus-dir <dir> --out tokenizer.json [--binary-out-dir <dir>]"
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

.PHONY: all clean test test-bf16 test-libs test-quant test-flash-attention test_flash_attention unittest unittest-show show_test help litmus litmus-test test-quick test-full test-stress profile-memory profile-heap profile-cpu profile-flash-attn profile-cache flamegraph ck-cli ck-cli-v4 ck-cli-v5 ck-chat ck-server ck-chat-py ck-server-py generate-model gguf-inspect gguf-list gguf-to-bump gguf-to-bump-v4 hf-to-bump-v4 ir-v4 ir-v4-q4k opt-status opt-pending opt-inference opt-training opt-kernels opt-targets opt-md kernel-coverage kernel-coverage-md test-coverage test-coverage-md meta-check meta-sync meta-init report report-md show_config show-config v5 demo-v5 demo-v5-debug llamacpp-parity llamacpp-parity-full llamacpp-parity-full-all-isa-variants showtests version version-history e2e e2e-quick e2e-qwen e2e-smollm e2e-v66 e2e-v66-full v6.6-test-help v6.6-test-quick v6.6-sanity v6.6-test-parity v6.6-test-memory v6.6-test-divergence v6.6-test-nan v6.6-test-all v6.6-test v6.6-download v6.6-kernel-map-regenerate v6.6-kernel-map-gate v6.6-validate-contracts v6.6-validate-matrix v6.6-validate-matrix-nightly v6.6-validate-matrix-smoke v6.6-validate-parity-matrix v6.6-validate-parity-matrix-required v6.6-validate-longdecode v6.6-gate v6.6-build v6.6 v6.6-full v6.6-ir-visualizer v6.6-memory-signoff v6.6-perf-gate v6.6-perf-gate-evaluate v7-help v7-sync-inference v7-infer-run v7-infer-gate v7-validate-contracts v7-parity-1tok v7-train-ir-smoke v7-train-ir-backward v7-train-parity-3 v7-train-parity-5 v7-gate-train v7-gate v7 profile-v6-prepare-runtime profile-v6-decode profile-v6-prefill profile-v6-flamegraph profile-v6-perf-stat profile-v6-vtune profile-v6-cachegrind profile-v6-full profile-v7-prepare-runtime profile-v7-decode profile-v7-prefill profile-v7-flamegraph profile-v7-perf-stat profile-v7-vtune profile-v7-advisor profile-v7-cachegrind profile-v7-full
.PHONY: v7-perf-gate v7-perf-gate-evaluate
.PHONY: v7-inference-smoke
.PHONY: v7-grad-fd v7-replay
.PHONY: v7-backprop-long-epoch v7-backprop-long-epoch-nightly
.PHONY: visualizer visualizer-full v7-ir-visualizer-e2e v7-ir-visualizer-e2e-nightly
.PHONY: v7-visualizer-health v7-visualizer-generated-e2e
.PHONY: v7-dataset-normalize v7-dataset-classify v7-dataset-embeddings v7-dataset-attention v7-dataset-viewer v7-dataset-all
.PHONY: ck-cli-v7 ck-bpe-train

# ============================================================================
# v6.6 Test Suite (delegates to version/v6.6/test/Makefile)
# ============================================================================
# Usage:
#   make v6.6-test-quick        - Quick sanity check
#   make v6.6-test-parity       - Layer-by-layer parity
#   make v6.6-test-memory       - Memory validation
#   make v6.6-test-all          - Run all v6.6 tests
#   make v6.6-download          - Download model
#   make v6.6-build             - Build v6.6 artifacts
#   make v6.6                   - Build v6.6 artifacts (main)
#   make v6.6-e2e               - Manual E2E sweep (Qwen2/Qwen3/Gemma)
# ============================================================================

# Default model used by v6.6 convenience targets.
# Override at invocation time, e.g.:
#   make v6.6-download V66_MODEL='hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf'
V66_MODEL ?= hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf
V66_CHAT_TEMPLATE ?= auto
V66_RUN_ARGS ?= $(if $(filter auto,$(V66_CHAT_TEMPLATE)),,--chat-template $(V66_CHAT_TEMPLATE))
V66_PERF_RUNTIME ?= cli
V66_CLI_ARGS ?=
V66_WITH_VTUNE ?= 1
V66_PREP_WITH_PYTHON ?= 1
V66_FORCE_COMPILE ?= 1
V66_FORCE_COMPILE_ARG := $(if $(filter 1,$(V66_FORCE_COMPILE)),--force-compile,)
V66_CLI_TEMPLATE_ARGS = $(if $(filter none,$(V66_CHAT_TEMPLATE)),--no-chat-template,$(if $(findstring --chat-template none,$(V66_RUN_ARGS)),--no-chat-template,))

v6.6-test-help:
	@cd version/v6.6/test && make help

v6.6-help:
	@echo "=== v6.6 Quick Help ==="
	@echo ""
	@echo "Build v6.6 artifacts:"
	@echo "  make v6.6"
	@echo ""
	@echo "Manual E2E sweep (Qwen2/Qwen3/Gemma):"
	@echo "  make v6.6-e2e"
	@echo ""
	@echo "Required release gate:"
	@echo "  make v6.6-gate"
	@echo "  (kernel-map gate + contracts + matrix-smoke + parity matrix (runtime-optional) + long decode)"
	@echo "  make v6.6-validate-parity-matrix-required"
	@echo "  (strict runtime-required parity matrix for CI/release)"
	@echo ""
	@echo "Manual E2E (HF URLs):"
	@echo "  python3 version/v6.6/scripts/ck_run_v6_6.py run \\"
	@echo "    \"hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf\" \\"
	@echo "    --context-len 1024 --force-compile --prompt \"Hello\" --max-tokens 32"
	@echo "  python3 version/v6.6/scripts/ck_run_v6_6.py run \\"
	@echo "    \"hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf\" \\"
	@echo "    --context-len 1024 --force-compile --prompt \"Hello\" --max-tokens 32"
	@echo "  python3 version/v6.6/scripts/ck_run_v6_6.py run \\"
	@echo "    \"hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf\" \\"
	@echo "    --context-len 1024 --force-compile --prompt \"Hello\" --max-tokens 32"
	@echo ""
	@echo "Docs:"
	@echo "  version/v6.6/run_me.read.md"
	@echo "  version/v6.6/E2E_HELP.md"

v6.6-test-quick v6.6-sanity:
	@cd version/v6.6/test && $(MAKE) quick

v6.6-test-parity:
	@cd version/v6.6/test && $(MAKE) parity

v6.6-test-memory:
	@cd version/v6.6/test && $(MAKE) memory

v6.6-test-divergence:
	@cd version/v6.6/test && $(MAKE) divergence

v6.6-test-nan:
	@cd version/v6.6/test && $(MAKE) nan

v6.6-test-all:
	@cd version/v6.6/test && $(MAKE) all

v6.6-test-trace:
	@cd version/v6.6/test && $(MAKE) trace ARGS=$(ARGS)

v6.6-test: v6.6-test-quick
	@echo "Run 'make v6.6-test-all' for comprehensive testing"

v6.6-download:
	@$(PYTHON) version/v6.6/scripts/ck_run_v6_6.py run "$(V66_MODEL)" --inspect-only

v6.6-kernel-map-regenerate:
	@$(PYTHON) version/v6.6/scripts/gen_kernel_registry_from_maps.py

v6.6-kernel-map-gate:
	@$(PYTHON) version/v6.6/scripts/gen_kernel_registry_from_maps.py --check
	@$(PYTHON) version/v6.6/kernel_maps/check_kernel_map_sync.py
	@$(PYTHON) version/v6.6/scripts/validate_kernel_registry.py

validate-registry:
	@echo "Validating kernel registry..."
	@$(PYTHON) version/v6.6/scripts/validate_kernel_registry.py
	@if [ $$? -ne 0 ]; then \
		echo "Registry validation failed"; \
		exit 1; \
	fi

v6.6-validate-contracts:
	@$(PYTHON) version/v6.6/scripts/validate_tooling_contracts.py --json-out version/v6.6/tools/contract_report_latest.json

v6.6-validate-matrix:
	@$(PYTHON) version/v6.6/scripts/validate_model_matrix_v6_6.py --allow-download --require-all --json-out version/v6.6/tools/model_matrix_report_latest.json

# Nightly matrix variant:
# - skips static preflight that is already covered by v6.6-validate-contracts
# - allows extra retries for transient build/cache fetch issues
# - keeps nightly as compatibility coverage instead of requiring every legacy row
#   to be present in every runner/cache state
v6.6-validate-matrix-nightly:
	@$(PYTHON) version/v6.6/scripts/validate_model_matrix_v6_6.py --allow-download --skip-static-contracts --retries 3 --retry-backoff-sec 5 --json-out version/v6.6/tools/model_matrix_report_latest.json

v6.6-validate-matrix-smoke:
	@$(PYTHON) version/v6.6/scripts/validate_model_matrix_v6_6.py --allow-download --with-smoke --require-all --json-out version/v6.6/tools/model_matrix_report_latest.json

# Runtime-optional parity pass for local developer gates.
# If llama runtime is missing, rows are SKIP and this target still exits 0.
v6.6-validate-parity-matrix:
	@$(PYTHON) version/v6.6/scripts/validate_parity_matrix_v6_6.py --allow-download --json-out version/v6.6/tools/parity_matrix_report_latest.json

v6.6-validate-parity-matrix-required:
	@$(PYTHON) version/v6.6/scripts/validate_parity_matrix_v6_6.py --allow-download --require-runtime --require-all --json-out version/v6.6/tools/parity_matrix_report_latest.json

v6.6-validate-longdecode:
	@$(PYTHON) version/v6.6/scripts/validate_long_decode_stability_v6_6.py --allow-download --require-all --json-out version/v6.6/tools/long_decode_report_latest.json

v6.6-gate:
	@$(MAKE) --no-print-directory v6.6-kernel-map-gate
	@$(PYTHON) version/v6.6/scripts/validate_tooling_contracts.py --strict --json-out version/v6.6/tools/contract_report_latest.json
	@$(PYTHON) version/v6.6/scripts/validate_model_matrix_v6_6.py --allow-download --with-smoke --require-all --skip-static-contracts --json-out version/v6.6/tools/model_matrix_report_latest.json
	@$(PYTHON) version/v6.6/scripts/validate_parity_matrix_v6_6.py --allow-download --json-out version/v6.6/tools/parity_matrix_report_latest.json
	@$(PYTHON) version/v6.6/scripts/validate_long_decode_stability_v6_6.py --allow-download --require-all --json-out version/v6.6/tools/long_decode_report_latest.json

v6.6-build: validate-registry v6.6-gate
	@$(PYTHON) version/v6.6/scripts/ck_run_v6_6.py run "$(V66_MODEL)" --generate-only $(V66_FORCE_COMPILE_ARG) --context-len 128 --max-tokens 1 --prompt "Hello"

v6.6: v6.6-build

v6.6-e2e:
	@./scripts/e2e_manual_v66.sh

v6.6-full: v6.6-download v6.6-build v6.6-test-all

V7_MODEL ?= hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf
V7_SMOKE_MODEL_GEMMA ?= hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf
V7_SMOKE_MODEL_QWEN2 ?= hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf
V7_SMOKE_MODEL_QWEN3 ?= hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
V7_REQUIREMENTS ?= requirements-v7.txt
V7_VENV_PY ?= .venv/bin/python
V7_VENV_STAMP ?= .venv/.v7_requirements.stamp
V7_DOCTOR_SCRIPT ?= version/v7/scripts/v7_doctor.py
V7_DOCTOR_SH ?= version/v7/scripts/v7_doctor.sh
V7_CHAT_TEMPLATE ?= auto
V7_WEIGHT_DTYPE ?=
V7_WEIGHT_DTYPE_ARG := $(if $(strip $(V7_WEIGHT_DTYPE)),--weight-dtype $(V7_WEIGHT_DTYPE),)
V7_RUN_ARGS ?= $(V7_WEIGHT_DTYPE_ARG) $(if $(filter auto,$(V7_CHAT_TEMPLATE)),,--chat-template $(V7_CHAT_TEMPLATE))
V7_PERF_RUNTIME ?= cli
V7_CLI_ARGS ?=
V7_WITH_VTUNE ?= 1
V7_WITH_ADVISOR ?= 1
V7_VTUNE_DEEP ?= 1
V7_PREP_WITH_PYTHON ?= 1
V7_FORCE_COMPILE ?= 1
V7_FORCE_COMPILE_ARG := $(if $(filter 1,$(V7_FORCE_COMPILE)),--force-compile,)
V7_FORCE_CONVERT ?= 1
V7_FORCE_CONVERT_ARG := $(if $(filter 1,$(V7_FORCE_CONVERT)),--force-convert,)
V7_AUTO_OPEN ?= 1
V7_DEMO_CONTEXT ?= 1024
V7_DEMO_PROMPT ?= Hello
V7_DEMO_MAX_TOKENS ?= 16
V7_CAPTURE_CONTEXT ?= 1024
V7_CAPTURE_PROMPT ?= Hello
V7_CAPTURE_MAX_TOKENS ?= 1
V7_REPORT_DIR ?= version/v7/.cache/reports
V7_CLI_TEMPLATE_ARGS = $(if $(filter none,$(V7_CHAT_TEMPLATE)),--no-chat-template,$(if $(findstring --chat-template none,$(V7_RUN_ARGS)),--no-chat-template,))
V7_TRAIN_MANIFEST ?=
V7_TRAIN_MAX_LAYERS ?= 2
V7_TRAIN_ALLOW_PARTIAL ?= 0
V7_TRAIN_STRICT_UNRESOLVED ?= 1
V7_TRAIN_CHECKPOINT_POLICY ?= none
V7_TRAIN_CODEGEN_IR2 ?= $(V7_REPORT_DIR)/ir2_train_backward_latest.json
V7_TRAIN_CODEGEN_OUT ?= $(V7_REPORT_DIR)/generated_train_runtime_v7.c
V7_TRAIN_CODEGEN_SUMMARY ?= $(V7_REPORT_DIR)/generated_train_runtime_v7_summary.json
V7_TRAIN_CODEGEN_OBJ ?= $(V7_REPORT_DIR)/generated_train_runtime_v7.o
V7_TRAIN_LAYOUT_OUT ?= $(V7_REPORT_DIR)/layout_train_latest.json
V7_TRAIN_LAYOUT_AUDIT ?= $(V7_REPORT_DIR)/layout_train_audit_latest.json
V7_TINY_OUT_DIR ?= version/v7/runs/tiny_init
V7_TINY_SEED ?= 42
V7_TINY_LAYERS ?= 2
V7_TINY_VOCAB ?= 256
V7_TINY_EMBED ?= 128
V7_TINY_HIDDEN ?= 256
V7_TINY_HEADS ?= 8
V7_TINY_KV_HEADS ?= 4
V7_TINY_CTX ?= 128
V7_KERNEL_PARITY_OPT_JSON ?= $(V7_REPORT_DIR)/optimizer_parity_latest.json
V7_KERNEL_PARITY_GEMM_JSON ?= $(V7_REPORT_DIR)/gemm_backward_shape_sweep_latest.json
V7_KERNEL_PARITY_RMS_SWIGLU_JSON ?= $(V7_REPORT_DIR)/rms_swiglu_backward_parity_latest.json
V7_GATE_WITH_KERNEL_PARITY ?= 1
V7_KERNEL_PARITY_QK_STRICT_ISA ?= 0
V7_KERNEL_PARITY_QK_JSON ?= $(V7_REPORT_DIR)/qk_norm_backward_parity_isa_latest.json
V7_KERNEL_PARITY_QK_JSON_STRICT ?= $(V7_REPORT_DIR)/qk_norm_backward_parity_isa_strict_latest.json
V7_TRAIN_LONG_HORIZON_EPOCHS ?= 3
V7_TRAIN_LONG_HORIZON_SEQ_LEN ?= 8
V7_TRAIN_LONG_HORIZON_TOTAL_TOKENS ?= 4096
V7_TRAIN_LONG_HORIZON_GRAD_ACCUM ?= 8
V7_TRAIN_LONG_HORIZON_VOCAB ?= 1024
V7_TRAIN_LONG_HORIZON_D_MODEL ?= 256
V7_TRAIN_LONG_HORIZON_HIDDEN ?= 1024
V7_TRAIN_LONG_HORIZON_LR ?= 5e-4
V7_TRAIN_LONG_HORIZON_SEED ?= 42
V7_TRAIN_LONG_HORIZON_TEXT ?= Hello!
V7_TRAIN_LONG_HORIZON_LOSS_TOL ?= 2e-5
V7_TRAIN_LONG_HORIZON_PARAM_TOL ?= 3e-5
V7_TRAIN_LONG_HORIZON_DIAG_EVERY ?= 10
V7_TRAIN_LONG_HORIZON_JSON ?= $(V7_REPORT_DIR)/train_parity_long_horizon_latest.json
V7_TRAIN_DRIFT_SMOKE_STEPS ?= 70
V7_TRAIN_DRIFT_SMOKE_JSON ?= $(V7_REPORT_DIR)/train_parity_drift_smoke_latest.json
V7_TRAIN_PROD_MAX_GRAD_NORM ?= 1.0
V7_TRAIN_ENFORCE_PROD_SAFETY ?= 1
V7_TRAIN_ALLOW_UNSAFE_ADAMW_LR ?= 0
V7_TRAIN_PROD_SAFETY_FLAGS := $(if $(filter 1,$(V7_TRAIN_ENFORCE_PROD_SAFETY)),--enforce-production-safety,) $(if $(filter 1,$(V7_TRAIN_ALLOW_UNSAFE_ADAMW_LR)),--allow-unsafe-adamw-lr,)
V7_TRAIN_DRIFT_LOCALIZE_STEP ?= 65
V7_TRAIN_DRIFT_LOCALIZE_TOL ?= 1e-6
V7_TRAIN_DRIFT_LOCALIZE_SOURCE ?= ck
V7_TRAIN_DRIFT_LOCALIZE_MAX_STEPS ?= 80
V7_TRAIN_DRIFT_LOCALIZE_JSON ?= $(V7_REPORT_DIR)/train_parity_drift_localize_latest.json
V7_TRAIN_REALISTIC_STEPS ?= 320
V7_TRAIN_REALISTIC_TEXT ?= The quick brown fox jumps over the lazy dog. In optimization practice, stable gradients and careful clipping matter. Diverse token windows reduce periodic aliasing and expose long-horizon drift sooner.
V7_TRAIN_REALISTIC_JSON ?= $(V7_REPORT_DIR)/train_parity_realistic_long_horizon_latest.json
V7_TRAIN_RUNTIME_PARITY_RUN_DIR ?= /tmp/v7_runtime_parity
V7_TRAIN_RUNTIME_PARITY_EPOCHS ?= 5
V7_TRAIN_RUNTIME_PARITY_SEQ_LEN ?= $(V7_TRAIN_LONG_HORIZON_SEQ_LEN)
V7_TRAIN_RUNTIME_PARITY_TOTAL_TOKENS ?= $(V7_TRAIN_LONG_HORIZON_TOTAL_TOKENS)
V7_TRAIN_RUNTIME_PARITY_GRAD_ACCUM ?= $(V7_TRAIN_LONG_HORIZON_GRAD_ACCUM)
V7_TRAIN_RUNTIME_PARITY_VOCAB ?= $(V7_TRAIN_LONG_HORIZON_VOCAB)
V7_TRAIN_RUNTIME_PARITY_D_MODEL ?= $(V7_TRAIN_LONG_HORIZON_D_MODEL)
V7_TRAIN_RUNTIME_PARITY_HIDDEN ?= $(V7_TRAIN_LONG_HORIZON_HIDDEN)
V7_TRAIN_RUNTIME_PARITY_SEED ?= $(V7_TRAIN_LONG_HORIZON_SEED)
V7_TRAIN_RUNTIME_PARITY_EVERY ?= $(V7_TRAIN_RUNTIME_PARITY_GRAD_ACCUM)
V7_TRAIN_RUNTIME_PARITY_STRESS_LR ?= 1e-3
V7_TRAIN_RUNTIME_PARITY_STRESS_TEXT ?= $(V7_TRAIN_LONG_HORIZON_TEXT)
V7_TRAIN_RUNTIME_PARITY_STRESS_JSON ?= $(V7_REPORT_DIR)/train_runtime_parity_stress_latest.json
V7_TRAIN_RUNTIME_PARITY_REALISTIC_LR ?= 5e-4
V7_TRAIN_RUNTIME_PARITY_REALISTIC_TEXT ?= $(V7_TRAIN_REALISTIC_TEXT)
V7_TRAIN_RUNTIME_PARITY_REALISTIC_JSON ?= $(V7_REPORT_DIR)/train_runtime_parity_realistic_latest.json
V7_TRAIN_RUNTIME_PARITY_DUMP_ON_DRIFT ?= 0
V7_TRAIN_RUNTIME_PARITY_DUMP_FLAG := $(if $(filter 1,$(V7_TRAIN_RUNTIME_PARITY_DUMP_ON_DRIFT)),--dump-on-drift,)
V7_TRAIN_RUNTIME_PARITY_BITWISE ?= 0
V7_TRAIN_RUNTIME_PARITY_BITWISE_FLAG := $(if $(filter 1,$(V7_TRAIN_RUNTIME_PARITY_BITWISE)),--bitwise-parity,)
V7_BACKPROP_LONG_EPOCH_MODE ?= smoke
V7_BACKPROP_LONG_EPOCH_NIGHTLY_MODE ?= smoke
V7_GATE_WITH_LONG_HORIZON_PARITY ?= 1
V7_GATE_WITH_BPE_TRAIN_PARITY ?= 1
V7_GATE_WITH_REPLAY_ACCUM ?= 1
V7_GATE_WITH_BACKPROP_PLUMBING ?= 1
V7_GATE_WITH_BACKPROP_STITCH_RUNTIME ?= 1
V7_GATE_WITH_SVG_OVERFIT ?= 0
V7_BPE_TRAIN_PARITY_JSON ?= $(V7_REPORT_DIR)/v7_bpe_train_parity_latest.json
V7_SVG_OVERFIT_JSON ?= $(V7_REPORT_DIR)/svg_overfit_regression_latest.json
V7_BACKPROP_PLUMBING_JSON ?= $(V7_REPORT_DIR)/backprop_plumbing_latest.json
V7_BACKPROP_STITCH_JSON ?= $(V7_REPORT_DIR)/backprop_stitch_runtime_latest.json
V7_BACKPROP_STITCH_ACCUM_JSON ?= $(V7_REPORT_DIR)/backprop_stitch_runtime_accum_latest.json
V7_BACKPROP_STITCH_DUMP_CHECK_TOPK ?= 32
V7_BACKPROP_STITCH_RUN_DIR ?= /tmp/v7_backprop_stitch_runtime
V7_BACKPROP_STITCH_ACCUM ?= 4
V7_BACKPROP_STITCH_TOTAL_TOKENS ?= 32
V7_BACKPROP_PLUMBING_RUNTIME_REPORT ?=
V7_BACKPROP_PLUMBING_RUNTIME_SUMMARY ?=
V7_VISUALIZER_E2E_MODEL ?= $(if $(wildcard $(HOME)/.cache/ck-engine-v7/models/Qwen--Qwen3-0.6B-GGUF),$(HOME)/.cache/ck-engine-v7/models/Qwen--Qwen3-0.6B-GGUF,$(if $(wildcard $(HOME)/.cache/ck-engine-v7/models/Qwen--Qwen2-0.5B-Instruct-GGUF),$(HOME)/.cache/ck-engine-v7/models/Qwen--Qwen2-0.5B-Instruct-GGUF,$(V7_SMOKE_MODEL_QWEN2)))
V7_VISUALIZER_E2E_CONTEXT ?= 1024
V7_VISUALIZER_E2E_MAX_TOKENS ?= 1
V7_VISUALIZER_E2E_JSON ?= $(V7_REPORT_DIR)/ir_visualizer_e2e_latest.json
V7_VISUALIZER_E2E_FORCE_COMPILE ?= 1
V7_VISUALIZER_E2E_FORCE_CONVERT ?= 1
V7_VISUALIZER_E2E_WITH_TRAIN ?= 0
V7_VISUALIZER_E2E_SKIP_INFERENCE_PARITY ?= 0
V7_RUNBOOK_E2E_RUN ?= /tmp/v7_runbook_e2e_v7
V7_RUNBOOK_E2E_MODE ?= smoke
V7_RUNBOOK_E2E_DATA ?= version/v7/data/svg_assets_train.txt
V7_RUNBOOK_E2E_JSON ?= $(V7_REPORT_DIR)/runbook_e2e_latest.json
V7_CKTOP_RUN ?= /tmp/v7_runtime_parity
V7_PIPELINE_RUN ?= /tmp/v7_pipeline_run
V7_PIPELINE_TOKENIZER ?= byte
V7_PIPELINE_DATASET_REPEATS ?= 10
V7_PIPELINE_EPOCHS ?= 10
V7_PIPELINE_SEQ_LEN ?= 32
V7_PIPELINE_TOTAL_TOKENS ?= 1024
V7_PIPELINE_GRAD_ACCUM ?= 1
V7_PIPELINE_LR ?= 5e-4
V7_PIPELINE_WITH_TORCH ?= 1
V7_PIPELINE_OPEN_VIS ?= 1
V7_PIPELINE_WORK_DIR ?=
V7_PIPELINE_JSON ?= $(V7_REPORT_DIR)/train_data_pipeline_latest.json
V7_STABILIZATION_RUN_ROOT ?=
V7_STABILIZATION_DATA ?= version/v7/data/svg_assets_train.txt
V7_STABILIZATION_LAYERS ?= 1,2,3,4
V7_STABILIZATION_TOKEN_BUDGETS ?= 2048,4096
V7_STABILIZATION_SEQ_LEN ?= 16
V7_STABILIZATION_GRAD_ACCUM_SWEEP ?= 2,4,8
V7_STABILIZATION_SWEEP_EPOCHS ?= 2
V7_STABILIZATION_FORWARD_EPOCHS ?= 10
V7_STABILIZATION_VOCAB ?= 256
V7_STABILIZATION_D_MODEL ?= 64
V7_STABILIZATION_HIDDEN ?= 128
V7_STABILIZATION_LR ?= 1e-3
V7_STABILIZATION_SEED ?= 42
V7_STABILIZATION_LOSS_TOL ?= 2e-5
V7_STABILIZATION_PARAM_TOL ?= 3e-5
V7_STABILIZATION_CK_LOSS_BACKEND ?= c_ptref
V7_STABILIZATION_RUNTIME_CHECKS ?= 1
V7_STABILIZATION_BACKEND_XRAY ?= 1
V7_STABILIZATION_FORCE ?= 1
V7_STABILIZATION_MAIN_RUN_DIR ?=
V7_STABILIZATION_JSON ?= $(V7_REPORT_DIR)/training_stabilization_scorecard_latest.json
V7_STABILIZATION_MD ?= $(V7_REPORT_DIR)/training_stabilization_scorecard_latest.md
V7_STABILIZATION_HISTORY ?= $(V7_REPORT_DIR)/training_stabilization_history.jsonl

.PHONY: v7-init v7-doctor v7-demo-runtime v7-capture-artifacts v7-profile-dashboard \
	v7-qk-norm-backward-parity v7-qk-norm-backward-parity-isa v7-qk-norm-backward-parity-isa-strict v7-rms-swiglu-backward-parity \
	v7-kernel-parity-train v7-init-tiny v7-train-layout-smoke v7-train-memory-audit v7-train-codegen v7-train-compile-smoke v7-train-c-smoke \
	v7-train-parity-drift-smoke v7-train-parity-drift-localize v7-train-parity-long-horizon v7-train-parity-long-horizon-realistic \
	v7-train-runtime-parity-prepare v7-train-runtime-parity-stress v7-train-runtime-parity-realistic v7-train-runtime-parity-long-horizon \
	v7-backprop-long-epoch v7-backprop-long-epoch-nightly v7-backprop-production-ready test-v7-bpe-train-parity v7-replay-accum \
	v7-backprop-plumbing v7-backprop-stitch-runtime v7-backprop-stitch-runtime-accum test-v7-svg-overfit-regression \
	v7-train-data-pipeline v7-stabilization-nightly v7-ir-visualizer-e2e v7-runbook-e2e \
	v7-ctop v7-ctop-demo

v7-help:
	@echo "=== v7 Training Foundation (fp32 correctness-first) ==="
	@echo ""
	@echo "Targets:"
	@echo "  make v7-init"
	@echo "  make v7-doctor"
	@echo "  make v7-demo-runtime V7_MODEL=$(V7_SMOKE_MODEL_QWEN3)"
	@echo "  make v7-capture-artifacts V7_MODEL=$(V7_SMOKE_MODEL_QWEN3)"
	@echo "  make v7-profile-dashboard V7_MODEL=$(V7_SMOKE_MODEL_QWEN3)"
	@echo "  make v7-sync-inference"
	@echo "  make v7-infer-run"
	@echo "  make v7-infer-gate"
	@echo "  make v7-inference-smoke"
	@echo "  make v7-perf-gate"
	@echo "  make profile-v7-full"
	@echo "  make profile-v7-advisor"
	@echo "  make visualizer"
	@echo "  make visualizer-full"
	@echo "  make v7-runbook-e2e"
	@echo "  make v7-validate-contracts"
	@echo "  make v7-parity-1tok"
	@echo "  make v7-qk-norm-backward-parity"
	@echo "  make v7-qk-norm-backward-parity-isa"
	@echo "  make v7-qk-norm-backward-parity-isa-strict"
	@echo "  make v7-rms-swiglu-backward-parity"
	@echo "  make v7-kernel-parity-train"
	@echo "  make v7-train-ir-smoke"
	@echo "  make v7-train-ir-backward"
	@echo "  make v7-train-layout-smoke"
	@echo "  make v7-train-memory-audit"
	@echo "  make v7-train-codegen"
	@echo "  make v7-train-compile-smoke"
	@echo "  make v7-init-tiny"
	@echo "  make v7-grad-fd"
	@echo "  make v7-replay"
	@echo "  make v7-replay-accum"
	@echo "  make v7-backprop-plumbing"
	@echo "  make v7-backprop-stitch-runtime"
	@echo "  make v7-backprop-stitch-runtime-accum"
	@echo "  make v7-train-parity-3"
	@echo "  make v7-train-parity-5"
	@echo "  make test-v7-bpe-train-parity"
	@echo "  make test-v7-svg-overfit-regression"
	@echo "  make v7-train-data-pipeline"
	@echo "  make v7-stabilization-nightly"
	@echo "     knobs: V7_PIPELINE_TOKENIZER=byte|bpe V7_PIPELINE_WITH_TORCH=0|1 V7_PIPELINE_OPEN_VIS=0|1 (default: 1)"
	@echo "     knobs: V7_STABILIZATION_LAYERS=1,2,3,4 V7_STABILIZATION_TOKEN_BUDGETS=2048,4096 V7_STABILIZATION_GRAD_ACCUM_SWEEP=2,4,8 V7_STABILIZATION_MAIN_RUN_DIR=/path/to/run"
	@echo "  make v7-train-parity-drift-smoke"
	@echo "  make v7-train-parity-drift-localize"
	@echo "  make v7-train-parity-long-horizon"
	@echo "  make v7-train-parity-long-horizon-realistic"
	@echo "  make v7-train-runtime-parity-stress"
	@echo "  make v7-train-runtime-parity-realistic"
	@echo "  make v7-train-runtime-parity-long-horizon"
	@echo "  make v7-ctop"
	@echo "  make v7-ctop-demo"
	@echo "  make v7-backprop-long-epoch"
	@echo "  make v7-backprop-long-epoch-nightly"
	@echo "  make v7-backprop-production-ready"
	@echo "  make v7-gate-train"
	@echo "  make v7-gate"
	@echo "  make v7"
	@echo ""
	@echo "Notes:"
	@echo "  - inference baseline is v7-named under version/v7/"
	@echo "  - run v7-sync-inference only when intentionally re-syncing from version/v6.6"
	@echo "  - v7.0 scope: deterministic fp32 correctness only"
	@echo "  - v7.2 scope: bf16, optimization, threaded fast mode"
	@echo "  - reports written to $(V7_REPORT_DIR)/*.json"
	@echo "  - v7-train-ir-smoke resolves manifest automatically if V7_TRAIN_MANIFEST is unset"
	@echo "  - training IR always reads from weights_manifest.json (generated by converter/init script)"
	@echo "  - v7-train-codegen emits compile-ready C runtime from IR2 backward artifact"
	@echo "  - v7-train-layout-smoke emits layout_train + overlap/bounds audit from IR2 tensor metadata"
	@echo "  - v7-stabilization-nightly runs tokenizer gates (ascii_bpe+bpe roundtrip exact) + parity matrix over depth/token budgets"
	@echo "  - default is strict: V7_TRAIN_STRICT_UNRESOLVED=1 and V7_TRAIN_ALLOW_PARTIAL=0"
	@echo "  - v7-kernel-parity-train uses ISA matrix by default; set V7_KERNEL_PARITY_QK_STRICT_ISA=1 for strict ISA fallback failures"
	@echo "  - set V7_TRAIN_ALLOW_PARTIAL=1 only while iterating unfinished grad-rules"
	@echo "  - long-horizon drift smoke is enforced in v7-gate-train by default (V7_GATE_WITH_LONG_HORIZON_PARITY=1)"
	@echo "  - backprop plumbing audit is enabled in v7-gate-train by default (V7_GATE_WITH_BACKPROP_PLUMBING=1)"
	@echo "  - one-step runtime stitch smoke is enabled in v7-gate-train by default (V7_GATE_WITH_BACKPROP_STITCH_RUNTIME=1)"
	@echo "  - production train safety defaults: max-grad-norm=$(V7_TRAIN_PROD_MAX_GRAD_NORM), enforce=$(V7_TRAIN_ENFORCE_PROD_SAFETY)"
	@echo "  - v7-backprop-long-epoch defaults to smoke mode (set V7_BACKPROP_LONG_EPOCH_MODE=full for full horizon)"
	@echo "  - runtime parity bitwise diagnostics: set V7_TRAIN_RUNTIME_PARITY_BITWISE=1 (forces --bitwise-parity)"
	@echo "  - live terminal monitor: make v7-ctop RUN=/tmp/v7_runtime_parity (or use v7-ctop-demo)"
	@echo "  - profiling toggles: V7_WITH_VTUNE=$(V7_WITH_VTUNE), V7_WITH_ADVISOR=$(V7_WITH_ADVISOR), V7_VTUNE_DEEP=$(V7_VTUNE_DEEP)"

$(V7_VENV_PY):
	@echo "Creating repo-local v7 virtualenv in .venv"
	@python3 -m venv .venv

$(V7_VENV_STAMP): $(V7_REQUIREMENTS) | $(V7_VENV_PY)
	@echo "Installing v7 Python dependencies from $(V7_REQUIREMENTS)"
	@$(V7_VENV_PY) -m pip install --upgrade pip
	@$(V7_VENV_PY) -m pip install -r $(V7_REQUIREMENTS)
	@touch $(V7_VENV_STAMP)

v7-doctor:
	@bash $(V7_DOCTOR_SH)

v7-init: $(V7_VENV_STAMP)
	@echo "v7 env ready: $(V7_VENV_PY)"
	@bash $(V7_DOCTOR_SH) || true
	@echo "Next:"
	@echo "  make v7-doctor"
	@echo "  make v7-demo-runtime V7_MODEL=$(V7_SMOKE_MODEL_QWEN3)"
	@echo "  make v7-capture-artifacts V7_MODEL=$(V7_SMOKE_MODEL_QWEN3)"
	@echo "  make v7-profile-dashboard V7_MODEL=$(V7_SMOKE_MODEL_QWEN3)"

v7-demo-runtime: v7-init
	@echo "Running v7 runtime demo for $(V7_MODEL)"
	@$(PYTHON) version/v7/scripts/ck_run_v7.py run "$(V7_MODEL)" \
		$(V7_FORCE_COMPILE_ARG) $(V7_FORCE_CONVERT_ARG) \
		--context-len $(V7_DEMO_CONTEXT) \
		--prompt "$(V7_DEMO_PROMPT)" --max-tokens $(V7_DEMO_MAX_TOKENS) \
		$(V7_RUN_ARGS)

v7-capture-artifacts: v7-init
	@echo "Capturing v7 operator artifacts for $(V7_MODEL)"
	@$(PYTHON) version/v7/scripts/ck_run_v7.py run "$(V7_MODEL)" \
		$(V7_FORCE_COMPILE_ARG) $(V7_FORCE_CONVERT_ARG) \
		--generate-only --generate-visualizer \
		--context-len $(V7_CAPTURE_CONTEXT) \
		--prompt "$(V7_CAPTURE_PROMPT)" --max-tokens $(V7_CAPTURE_MAX_TOKENS) \
		$(V7_RUN_ARGS)
	@cache_models="$${CK_CACHE_DIR:-$$HOME/.cache/ck-engine-v7/models}"; \
		model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$(V7_MODEL)" )"; \
		report_path="$$model_dir/ir_report.html"; \
		hub_html="$$cache_models/ir_hub.html"; \
		hub_index="$$cache_models/runs_hub_index.json"; \
		$(PYTHON) version/v7/tools/open_ir_visualizer.py --generate --run "$$model_dir" --html-only --strict-run-artifacts --output "$$report_path"; \
		$(PYTHON) version/v7/tools/open_ir_hub.py --models-root "$$cache_models" --output "$$hub_html" --index-out "$$hub_index"; \
		echo "[OK] run dir: $$model_dir"; \
		echo "[OK] report: $$report_path"; \
		echo "[OK] hub: $$hub_html"; \
		if [ "$(V7_AUTO_OPEN)" = "1" ] && command -v xdg-open >/dev/null 2>&1; then \
			xdg-open "$$report_path" >/dev/null 2>&1 || true; \
			xdg-open "$$hub_html" >/dev/null 2>&1 || true; \
		fi

v7-capture-artifacts-run: v7-init
	@if [ -z "$(RUN)" ]; then \
		echo "Set RUN=/path/to/run"; \
		exit 2; \
	fi
	@$(MAKE) --no-print-directory v7-capture-artifacts V7_MODEL="$(RUN)" V7_AUTO_OPEN=$(V7_AUTO_OPEN) V7_FORCE_COMPILE=$(V7_FORCE_COMPILE) V7_FORCE_CONVERT=$(V7_FORCE_CONVERT) V7_CHAT_TEMPLATE="$(V7_CHAT_TEMPLATE)" V7_WEIGHT_DTYPE="$(V7_WEIGHT_DTYPE)"

v7-profile-dashboard: v7-init
	@echo "Capturing v7 profiling dashboard for $(V7_MODEL)"
	@$(MAKE) --no-print-directory v7-capture-artifacts V7_MODEL="$(V7_MODEL)" V7_AUTO_OPEN=0 V7_FORCE_COMPILE=$(V7_FORCE_COMPILE) V7_FORCE_CONVERT=$(V7_FORCE_CONVERT) V7_CHAT_TEMPLATE="$(V7_CHAT_TEMPLATE)" V7_WEIGHT_DTYPE="$(V7_WEIGHT_DTYPE)"
	@$(MAKE) --no-print-directory profile-v7-prepare-runtime V7_MODEL="$(V7_MODEL)" V7_FORCE_COMPILE=$(V7_FORCE_COMPILE) V7_PERF_RUNTIME=$(V7_PERF_RUNTIME) V7_CHAT_TEMPLATE="$(V7_CHAT_TEMPLATE)" V7_WEIGHT_DTYPE="$(V7_WEIGHT_DTYPE)"
	@$(MAKE) --no-print-directory profile-v7-decode V7_MODEL="$(V7_MODEL)" V7_FORCE_COMPILE=0 V7_PERF_RUNTIME=$(V7_PERF_RUNTIME) V7_CHAT_TEMPLATE="$(V7_CHAT_TEMPLATE)" V7_WEIGHT_DTYPE="$(V7_WEIGHT_DTYPE)"
	@$(MAKE) --no-print-directory profile-v7-perf-stat V7_MODEL="$(V7_MODEL)" V7_FORCE_COMPILE=0 V7_PERF_RUNTIME=$(V7_PERF_RUNTIME) V7_CHAT_TEMPLATE="$(V7_CHAT_TEMPLATE)" V7_WEIGHT_DTYPE="$(V7_WEIGHT_DTYPE)"
	@$(MAKE) --no-print-directory profile-v7-flamegraph-decode V7_MODEL="$(V7_MODEL)" V7_FORCE_COMPILE=0 V7_PERF_RUNTIME=$(V7_PERF_RUNTIME) V7_CHAT_TEMPLATE="$(V7_CHAT_TEMPLATE)" V7_WEIGHT_DTYPE="$(V7_WEIGHT_DTYPE)"
	@if command -v valgrind >/dev/null 2>&1 && command -v cg_annotate >/dev/null 2>&1; then \
		$(MAKE) --no-print-directory profile-v7-cachegrind V7_MODEL="$(V7_MODEL)" V7_CHAT_TEMPLATE="$(V7_CHAT_TEMPLATE)" V7_WEIGHT_DTYPE="$(V7_WEIGHT_DTYPE)"; \
	else \
		echo "SKIP: cachegrind capture requires valgrind + cg_annotate"; \
	fi
	@$(MAKE) --no-print-directory v7-perf-gate-evaluate V7_MODEL="$(V7_MODEL)"
	@cache_models="$${CK_CACHE_DIR:-$$HOME/.cache/ck-engine-v7/models}"; \
		model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$(V7_MODEL)" )"; \
		report_path="$$model_dir/ir_report.html"; \
		hub_html="$$cache_models/ir_hub.html"; \
		hub_index="$$cache_models/runs_hub_index.json"; \
		$(PYTHON) version/v7/tools/open_ir_visualizer.py --generate --run "$$model_dir" --html-only --strict-run-artifacts --output "$$report_path"; \
		$(PYTHON) version/v7/tools/open_ir_hub.py --models-root "$$cache_models" --output "$$hub_html" --index-out "$$hub_index"; \
		echo "[OK] profiled run dir: $$model_dir"; \
		echo "[OK] profiled report: $$report_path"; \
		echo "[OK] profiled hub: $$hub_html"; \
		if [ "$(V7_AUTO_OPEN)" = "1" ] && command -v xdg-open >/dev/null 2>&1; then \
			xdg-open "$$report_path" >/dev/null 2>&1 || true; \
			xdg-open "$$hub_html" >/dev/null 2>&1 || true; \
		fi

v7-profile-dashboard-run: v7-init
	@if [ -z "$(RUN)" ]; then \
		echo "Set RUN=/path/to/run"; \
		exit 2; \
	fi
	@$(MAKE) --no-print-directory v7-profile-dashboard V7_MODEL="$(RUN)" V7_AUTO_OPEN=$(V7_AUTO_OPEN) V7_FORCE_COMPILE=$(V7_FORCE_COMPILE) V7_FORCE_CONVERT=$(V7_FORCE_CONVERT) V7_PERF_RUNTIME=$(V7_PERF_RUNTIME) V7_CHAT_TEMPLATE="$(V7_CHAT_TEMPLATE)" V7_WEIGHT_DTYPE="$(V7_WEIGHT_DTYPE)"

v7-ctop:
	@RUN_DIR="$(if $(RUN),$(RUN),$(V7_CKTOP_RUN))"; \
	$(PYTHON) version/v7/tools/cktop_v7.py --run "$$RUN_DIR"

v7-ctop-demo:
	@RUN_DIR="$(if $(RUN),$(RUN),$(V7_CKTOP_RUN))"; \
	$(PYTHON) version/v7/tools/cktop_v7.py --run "$$RUN_DIR" --demo

v7-sync-inference:
	@$(PYTHON) version/v7/scripts/sync_v7_inference_baseline.py

v7-infer-run:
	@$(PYTHON) version/v7/scripts/ck_run_v7.py run "$(V7_MODEL)" --force-compile --context-len 1024 --prompt "Hello" --max-tokens 16

v7-infer-gate:
	@$(PYTHON) version/v7/scripts/gen_kernel_registry_from_maps.py --check
	@$(PYTHON) version/v7/kernel_maps/check_kernel_map_sync.py
	@$(PYTHON) version/v7/scripts/validate_kernel_registry.py
	@$(PYTHON) version/v7/scripts/ck_run_v7.py run "$(V7_MODEL)" --generate-only --profile --force-compile --context-len 128 --max-tokens 1 --prompt "Hello"

v7-inference-smoke:
	@$(PYTHON) version/v7/scripts/ck_run_v7.py run "$(V7_SMOKE_MODEL_GEMMA)" --context-len 1024 --force-compile --force-convert --chat-template none --generate-only --prompt "hi" --max-tokens 1
	@$(PYTHON) version/v7/scripts/ck_run_v7.py run "$(V7_SMOKE_MODEL_QWEN2)" --context-len 1024 --force-compile --force-convert --chat-template none --generate-only --prompt "hi" --max-tokens 1
	@$(PYTHON) version/v7/scripts/ck_run_v7.py run "$(V7_SMOKE_MODEL_QWEN3)" --context-len 1024 --force-compile --force-convert --chat-template none --generate-only --prompt "hi" --max-tokens 1

v7-validate-contracts:
	@$(PYTHON) version/v7/scripts/validate_v7_contracts.py --strict --training-mode --json-out $(V7_REPORT_DIR)/contract_report_latest.json
	@$(PYTHON) version/v7/scripts/validate_tooling_contracts.py --strict --json-out $(V7_REPORT_DIR)/tooling_contract_report_latest.json

regression-fast:
	@echo "Running v7 regression fast suite..."
	@$(PYTHON) version/v7/scripts/run_regression_v7.py --mode fast --force-rebuild $(REGRESSION_ARGS)

regression-full:
	@echo "Running v7 regression full suite..."
	@$(PYTHON) version/v7/scripts/run_regression_v7.py --mode full $(REGRESSION_ARGS)

regression-family:
	@if [ -z "$(FAMILY)" ]; then \
		echo "Usage: make regression-family FAMILY=<family_id>"; \
		exit 2; \
	fi
	@echo "Running v7 regression for family: $(FAMILY)"
	@$(PYTHON) version/v7/scripts/run_regression_v7.py --mode full --family "$(FAMILY)" $(REGRESSION_ARGS)

v7-parity-1tok:
	@$(PYTHON) version/v7/scripts/run_parity_1token_v7.py --json-out $(V7_REPORT_DIR)/parity_1token_latest.json

v7-qk-norm-backward-parity:
	@$(PYTHON) version/v7/scripts/check_qk_norm_backward_parity_v7.py --json-out $(V7_REPORT_DIR)/qk_norm_backward_parity_latest.json

v7-qk-norm-backward-parity-isa:
	@$(PYTHON) version/v7/scripts/check_qk_norm_backward_parity_v7.py --isa-matrix --json-out $(V7_REPORT_DIR)/qk_norm_backward_parity_isa_latest.json

v7-qk-norm-backward-parity-isa-strict:
	@$(PYTHON) version/v7/scripts/check_qk_norm_backward_parity_v7.py --isa-matrix --strict-isa --json-out $(V7_REPORT_DIR)/qk_norm_backward_parity_isa_strict_latest.json

v7-rms-swiglu-backward-parity:
	@$(PYTHON) version/v7/scripts/check_rms_swiglu_backward_parity_v7.py --require-fast --json-out $(V7_KERNEL_PARITY_RMS_SWIGLU_JSON)

v7-kernel-parity-train:
	@$(PYTHON) version/v7/scripts/check_optimizer_parity_v7.py --json-out $(V7_KERNEL_PARITY_OPT_JSON)
	@if [ "$(V7_KERNEL_PARITY_QK_STRICT_ISA)" = "1" ]; then \
		$(PYTHON) version/v7/scripts/check_qk_norm_backward_parity_v7.py --isa-matrix --strict-isa --json-out $(V7_KERNEL_PARITY_QK_JSON_STRICT); \
	else \
		$(PYTHON) version/v7/scripts/check_qk_norm_backward_parity_v7.py --isa-matrix --json-out $(V7_KERNEL_PARITY_QK_JSON); \
	fi
	@$(PYTHON) version/v7/scripts/check_rms_swiglu_backward_parity_v7.py --require-fast --json-out $(V7_KERNEL_PARITY_RMS_SWIGLU_JSON)
	@$(PYTHON) version/v7/scripts/check_gemm_backward_parity_v7.py --json-out $(V7_KERNEL_PARITY_GEMM_JSON)

v7-train-parity-3:
	@$(PYTHON) version/v7/scripts/train_parity_epochs_v7.py --epochs 3 --json-out $(V7_REPORT_DIR)/train_parity_epochs_3_latest.json

v7-train-parity-5:
	@$(PYTHON) version/v7/scripts/train_parity_epochs_v7.py --epochs 5 --json-out $(V7_REPORT_DIR)/train_parity_epochs_5_latest.json


v7-train-parity-drift-smoke:
	@$(PYTHON) version/v7/scripts/train_parity_epochs_v7.py \
		--epochs $(V7_TRAIN_LONG_HORIZON_EPOCHS) \
		--seq-len $(V7_TRAIN_LONG_HORIZON_SEQ_LEN) \
		--total-tokens $(V7_TRAIN_LONG_HORIZON_TOTAL_TOKENS) \
		--grad-accum $(V7_TRAIN_LONG_HORIZON_GRAD_ACCUM) \
		--optimizer adamw \
		--lr $(V7_TRAIN_LONG_HORIZON_LR) \
		--seed $(V7_TRAIN_LONG_HORIZON_SEED) \
		--vocab $(V7_TRAIN_LONG_HORIZON_VOCAB) \
		--d-model $(V7_TRAIN_LONG_HORIZON_D_MODEL) \
		--hidden $(V7_TRAIN_LONG_HORIZON_HIDDEN) \
		--loss-tol $(V7_TRAIN_LONG_HORIZON_LOSS_TOL) \
		--param-tol $(V7_TRAIN_LONG_HORIZON_PARAM_TOL) \
		--diag-every $(V7_TRAIN_LONG_HORIZON_DIAG_EVERY) \
		--max-grad-norm $(V7_TRAIN_PROD_MAX_GRAD_NORM) \
		--unsafe-adamw-lr-threshold 1e-3 \
		$(V7_TRAIN_PROD_SAFETY_FLAGS) \
		--max-steps $(V7_TRAIN_DRIFT_SMOKE_STEPS) \
		--train-text "$(V7_TRAIN_LONG_HORIZON_TEXT)" \
		--json-out "$(V7_TRAIN_DRIFT_SMOKE_JSON)"


v7-train-parity-drift-localize:
	@$(PYTHON) version/v7/scripts/train_parity_epochs_v7.py \
		--epochs $(V7_TRAIN_LONG_HORIZON_EPOCHS) \
		--seq-len $(V7_TRAIN_LONG_HORIZON_SEQ_LEN) \
		--total-tokens $(V7_TRAIN_LONG_HORIZON_TOTAL_TOKENS) \
		--grad-accum $(V7_TRAIN_LONG_HORIZON_GRAD_ACCUM) \
		--optimizer adamw \
		--lr $(V7_TRAIN_LONG_HORIZON_LR) \
		--seed $(V7_TRAIN_LONG_HORIZON_SEED) \
		--vocab $(V7_TRAIN_LONG_HORIZON_VOCAB) \
		--d-model $(V7_TRAIN_LONG_HORIZON_D_MODEL) \
		--hidden $(V7_TRAIN_LONG_HORIZON_HIDDEN) \
		--loss-tol $(V7_TRAIN_LONG_HORIZON_LOSS_TOL) \
		--param-tol $(V7_TRAIN_LONG_HORIZON_PARAM_TOL) \
		--diag-every $(V7_TRAIN_LONG_HORIZON_DIAG_EVERY) \
		--max-grad-norm $(V7_TRAIN_PROD_MAX_GRAD_NORM) \
		--unsafe-adamw-lr-threshold 1e-3 \
		$(V7_TRAIN_PROD_SAFETY_FLAGS) \
		--max-steps $(V7_TRAIN_DRIFT_LOCALIZE_MAX_STEPS) \
		--train-text "$(V7_TRAIN_LONG_HORIZON_TEXT)" \
		--ck-rmsnorm-backend c --ck-swiglu-backend c --ck-loss-backend c \
		--drift-localize-step $(V7_TRAIN_DRIFT_LOCALIZE_STEP) \
		--drift-localize-tol $(V7_TRAIN_DRIFT_LOCALIZE_TOL) \
		--drift-localize-source $(V7_TRAIN_DRIFT_LOCALIZE_SOURCE) \
		--json-out "$(V7_TRAIN_DRIFT_LOCALIZE_JSON)"

v7-train-parity-long-horizon:
	@$(PYTHON) version/v7/scripts/train_parity_epochs_v7.py \
		--epochs $(V7_TRAIN_LONG_HORIZON_EPOCHS) \
		--seq-len $(V7_TRAIN_LONG_HORIZON_SEQ_LEN) \
		--total-tokens $(V7_TRAIN_LONG_HORIZON_TOTAL_TOKENS) \
		--grad-accum $(V7_TRAIN_LONG_HORIZON_GRAD_ACCUM) \
		--optimizer adamw \
		--lr $(V7_TRAIN_LONG_HORIZON_LR) \
		--seed $(V7_TRAIN_LONG_HORIZON_SEED) \
		--vocab $(V7_TRAIN_LONG_HORIZON_VOCAB) \
		--d-model $(V7_TRAIN_LONG_HORIZON_D_MODEL) \
		--hidden $(V7_TRAIN_LONG_HORIZON_HIDDEN) \
		--loss-tol $(V7_TRAIN_LONG_HORIZON_LOSS_TOL) \
		--param-tol $(V7_TRAIN_LONG_HORIZON_PARAM_TOL) \
		--diag-every $(V7_TRAIN_LONG_HORIZON_DIAG_EVERY) \
		--max-grad-norm $(V7_TRAIN_PROD_MAX_GRAD_NORM) \
		--unsafe-adamw-lr-threshold 1e-3 \
		$(V7_TRAIN_PROD_SAFETY_FLAGS) \
		--train-text "$(V7_TRAIN_LONG_HORIZON_TEXT)" \
		--json-out "$(V7_TRAIN_LONG_HORIZON_JSON)"


v7-train-parity-long-horizon-realistic:
	@$(PYTHON) version/v7/scripts/train_parity_epochs_v7.py \
		--epochs $(V7_TRAIN_LONG_HORIZON_EPOCHS) \
		--seq-len $(V7_TRAIN_LONG_HORIZON_SEQ_LEN) \
		--total-tokens $(V7_TRAIN_LONG_HORIZON_TOTAL_TOKENS) \
		--grad-accum $(V7_TRAIN_LONG_HORIZON_GRAD_ACCUM) \
		--optimizer adamw \
		--lr $(V7_TRAIN_LONG_HORIZON_LR) \
		--seed $(V7_TRAIN_LONG_HORIZON_SEED) \
		--vocab $(V7_TRAIN_LONG_HORIZON_VOCAB) \
		--d-model $(V7_TRAIN_LONG_HORIZON_D_MODEL) \
		--hidden $(V7_TRAIN_LONG_HORIZON_HIDDEN) \
		--loss-tol $(V7_TRAIN_LONG_HORIZON_LOSS_TOL) \
		--param-tol $(V7_TRAIN_LONG_HORIZON_PARAM_TOL) \
		--diag-every $(V7_TRAIN_LONG_HORIZON_DIAG_EVERY) \
		--max-grad-norm $(V7_TRAIN_PROD_MAX_GRAD_NORM) \
		--unsafe-adamw-lr-threshold 1e-3 \
		$(V7_TRAIN_PROD_SAFETY_FLAGS) \
		--max-steps $(V7_TRAIN_REALISTIC_STEPS) \
		--train-text "$(V7_TRAIN_REALISTIC_TEXT)" \
		--json-out "$(V7_TRAIN_REALISTIC_JSON)"

v7-train-runtime-parity-prepare:
	@set -e; \
	mkdir -p "$(V7_TRAIN_RUNTIME_PARITY_RUN_DIR)"; \
	if [ -f "$(V7_TRAIN_RUNTIME_PARITY_RUN_DIR)/weights.bump" ] && [ -f "$(V7_TRAIN_RUNTIME_PARITY_RUN_DIR)/weights_manifest.json" ]; then \
		echo "v7-train-runtime-parity-prepare: using existing run dir $(V7_TRAIN_RUNTIME_PARITY_RUN_DIR)"; \
	else \
		echo "v7-train-runtime-parity-prepare: initializing run dir $(V7_TRAIN_RUNTIME_PARITY_RUN_DIR)"; \
		$(PYTHON) version/v7/scripts/ck_run_v7.py init \
			--run "$(V7_TRAIN_RUNTIME_PARITY_RUN_DIR)" \
			--train-seed $(V7_TRAIN_RUNTIME_PARITY_SEED) \
			--layers $(V7_TRAIN_MAX_LAYERS) \
			--vocab-size $(V7_TRAIN_RUNTIME_PARITY_VOCAB) \
			--embed-dim $(V7_TRAIN_RUNTIME_PARITY_D_MODEL) \
			--hidden-dim $(V7_TRAIN_RUNTIME_PARITY_HIDDEN) \
			--context-len $(V7_TINY_CTX); \
	fi

v7-train-runtime-parity-stress: v7-train-runtime-parity-prepare
	@$(PYTHON) version/v7/scripts/ck_run_v7.py train \
		--run "$(V7_TRAIN_RUNTIME_PARITY_RUN_DIR)" \
		--backend ck \
		--train-epochs $(V7_TRAIN_RUNTIME_PARITY_EPOCHS) \
		--train-seq-len $(V7_TRAIN_RUNTIME_PARITY_SEQ_LEN) \
		--train-total-tokens $(V7_TRAIN_RUNTIME_PARITY_TOTAL_TOKENS) \
		--train-grad-accum $(V7_TRAIN_RUNTIME_PARITY_GRAD_ACCUM) \
		--train-optimizer adamw \
		--train-lr $(V7_TRAIN_RUNTIME_PARITY_STRESS_LR) \
		--train-max-grad-norm 0 \
		--enforce-production-safety \
		--allow-unsafe-adamw-lr \
		--train-unsafe-adamw-lr-threshold 1e-3 \
		--train-seed $(V7_TRAIN_RUNTIME_PARITY_SEED) \
		--train-vocab $(V7_TRAIN_RUNTIME_PARITY_VOCAB) \
		--train-d-model $(V7_TRAIN_RUNTIME_PARITY_D_MODEL) \
		--train-hidden $(V7_TRAIN_RUNTIME_PARITY_HIDDEN) \
		--train-loss-tol $(V7_TRAIN_LONG_HORIZON_LOSS_TOL) \
		--train-param-tol $(V7_TRAIN_LONG_HORIZON_PARAM_TOL) \
		--prompt "$(V7_TRAIN_RUNTIME_PARITY_STRESS_TEXT)" \
		--parity-on \
		--parity-every $(V7_TRAIN_RUNTIME_PARITY_EVERY) \
		$(V7_TRAIN_RUNTIME_PARITY_BITWISE_FLAG) \
		$(V7_TRAIN_RUNTIME_PARITY_DUMP_FLAG) --train-json-out "$(V7_TRAIN_RUNTIME_PARITY_STRESS_JSON)"

v7-train-runtime-parity-realistic: v7-train-runtime-parity-prepare
	@$(PYTHON) version/v7/scripts/ck_run_v7.py train \
		--run "$(V7_TRAIN_RUNTIME_PARITY_RUN_DIR)" \
		--backend ck \
		--train-epochs $(V7_TRAIN_RUNTIME_PARITY_EPOCHS) \
		--train-seq-len $(V7_TRAIN_RUNTIME_PARITY_SEQ_LEN) \
		--train-total-tokens $(V7_TRAIN_RUNTIME_PARITY_TOTAL_TOKENS) \
		--train-grad-accum $(V7_TRAIN_RUNTIME_PARITY_GRAD_ACCUM) \
		--train-optimizer adamw \
		--train-lr $(V7_TRAIN_RUNTIME_PARITY_REALISTIC_LR) \
		--train-max-grad-norm $(V7_TRAIN_PROD_MAX_GRAD_NORM) \
		--enforce-production-safety \
		--train-unsafe-adamw-lr-threshold 1e-3 \
		--train-seed $(V7_TRAIN_RUNTIME_PARITY_SEED) \
		--train-vocab $(V7_TRAIN_RUNTIME_PARITY_VOCAB) \
		--train-d-model $(V7_TRAIN_RUNTIME_PARITY_D_MODEL) \
		--train-hidden $(V7_TRAIN_RUNTIME_PARITY_HIDDEN) \
		--train-loss-tol $(V7_TRAIN_LONG_HORIZON_LOSS_TOL) \
		--train-param-tol $(V7_TRAIN_LONG_HORIZON_PARAM_TOL) \
		--prompt "$(V7_TRAIN_RUNTIME_PARITY_REALISTIC_TEXT)" \
		--parity-on \
		--parity-every $(V7_TRAIN_RUNTIME_PARITY_EVERY) \
		$(V7_TRAIN_RUNTIME_PARITY_BITWISE_FLAG) \
		$(V7_TRAIN_RUNTIME_PARITY_DUMP_FLAG) --train-json-out "$(V7_TRAIN_RUNTIME_PARITY_REALISTIC_JSON)"

v7-train-runtime-parity-long-horizon:
	@set -e; \
	echo "v7-train-runtime-parity-long-horizon: realistic"; \
	$(MAKE) --no-print-directory v7-train-runtime-parity-realistic; \
	echo "v7-train-runtime-parity-long-horizon: stress"; \
	$(MAKE) --no-print-directory v7-train-runtime-parity-stress

v7-backprop-long-epoch:
	@if [ "$(V7_BACKPROP_LONG_EPOCH_MODE)" = "full" ]; then \
		echo "v7-backprop-long-epoch: full horizon"; \
		$(MAKE) --no-print-directory v7-train-parity-long-horizon; \
	else \
		echo "v7-backprop-long-epoch: drift smoke"; \
		$(MAKE) --no-print-directory v7-train-parity-drift-smoke; \
	fi

v7-backprop-long-epoch-nightly:
	@set -e; \
	echo "v7-backprop-long-epoch-nightly: realistic long-horizon blocker"; \
	$(MAKE) --no-print-directory v7-train-parity-long-horizon-realistic; \
	echo "v7-backprop-long-epoch-nightly: hello stress monitor (non-blocking)"; \
	if ! $(MAKE) --no-print-directory v7-train-parity-drift-smoke; then \
		echo "WARNING: hello stress monitor failed (non-blocking nightly signal)"; \
	fi

v7-backprop-production-ready:
	@set -e; \
	echo "v7-backprop-production-ready: v7-gate-train + nightly long-horizon with production safety"; \
	$(MAKE) --no-print-directory v7-gate-train V7_TRAIN_ENFORCE_PROD_SAFETY=1; \
	$(MAKE) --no-print-directory v7-backprop-long-epoch-nightly V7_TRAIN_ENFORCE_PROD_SAFETY=1


v7-train-ir-smoke:
	@set -e; \
	manifest="$$V7_TRAIN_MANIFEST"; \
	if [ -z "$$manifest" ]; then \
		manifest="$$( $(PYTHON) version/v7/scripts/resolve_train_manifest_v7.py )" || { \
			echo "ERROR: could not resolve train manifest. Set V7_TRAIN_MANIFEST=/path/to/weights_manifest.json"; exit 1; }; \
	fi; \
	if [ ! -f "$$manifest" ]; then \
		echo "ERROR: manifest not found: $$manifest"; exit 1; \
	fi; \
	model_dir="$$(dirname "$$manifest")"; \
	model_tag="$$(basename "$$model_dir" | tr ' /' '__')"; \
	report_dir="$(V7_REPORT_DIR)"; \
	mkdir -p "$$report_dir"; \
	ir1_out="$$report_dir/ir1_train_forward_$${model_tag}.json"; \
	ir2_out="$$report_dir/ir2_train_backward_$${model_tag}.json"; \
	sum_out="$$report_dir/ir2_train_summary_$${model_tag}.json"; \
	inv_out="$$report_dir/ir_train_invariants_$${model_tag}.json"; \
	layout_out="$$report_dir/layout_train_$${model_tag}.json"; \
	layout_audit_out="$$report_dir/layout_train_audit_$${model_tag}.json"; \
	strict_flag=""; \
	if [ "$(V7_TRAIN_STRICT_UNRESOLVED)" = "1" ]; then strict_flag="--strict"; fi; \
	partial_flag=""; \
	if [ "$(V7_TRAIN_ALLOW_PARTIAL)" = "1" ]; then partial_flag="--allow-partial"; fi; \
	inv_flag=""; \
	if [ "$(V7_TRAIN_STRICT_UNRESOLVED)" = "1" ]; then inv_flag="--strict-unresolved"; fi; \
	inv_partial_flag=""; \
	if [ "$(V7_TRAIN_ALLOW_PARTIAL)" = "1" ]; then inv_partial_flag="--allow-partial"; fi; \
	echo "Using train manifest: $$manifest"; \
	$(PYTHON) version/v7/scripts/build_ir_train_v7.py --manifest "$$manifest" --output "$$ir1_out" --max-layers $(V7_TRAIN_MAX_LAYERS) --report-out "$$report_dir/ir1_train_latest.json"; \
	$(PYTHON) version/v7/scripts/lower_ir2_backward_v7.py --ir1 "$$ir1_out" --output "$$ir2_out" --checkpoint-policy "$(V7_TRAIN_CHECKPOINT_POLICY)" $$strict_flag $$partial_flag --summary-out "$$sum_out"; \
	$(PYTHON) version/v7/scripts/validate_ir_train_invariants_v7.py --ir1 "$$ir1_out" --ir2 "$$ir2_out" --output "$$inv_out" $$inv_flag $$inv_partial_flag; \
	$(PYTHON) version/v7/scripts/generate_train_layout_v7.py --ir2 "$$ir2_out" --manifest "$$manifest" --output "$$layout_out" --align-bytes 64 $$strict_flag; \
	$(PYTHON) version/v7/scripts/validate_train_memory_layout_v7.py --layout "$$layout_out" --ir2 "$$ir2_out" --output "$$layout_audit_out" $$strict_flag; \
	cp "$$ir1_out" "$$report_dir/ir1_train_forward_latest.json"; \
	cp "$$ir2_out" "$$report_dir/ir2_train_backward_latest.json"; \
	cp "$$sum_out" "$$report_dir/ir2_train_summary_latest.json"; \
	cp "$$inv_out" "$$report_dir/ir_train_invariants_latest.json"; \
	cp "$$layout_out" "$$report_dir/layout_train_latest.json"; \
	cp "$$layout_audit_out" "$$report_dir/layout_train_audit_latest.json"; \
	echo "v7 train IR smoke complete: model=$$model_tag"

v7-train-ir-backward: v7-train-ir-smoke

v7-train-layout-smoke: v7-train-ir-smoke
	@echo "layout_train emitted at $(V7_TRAIN_LAYOUT_OUT)"
	@echo "layout_train audit at $(V7_TRAIN_LAYOUT_AUDIT)"

v7-train-memory-audit: v7-train-layout-smoke
	@$(PYTHON) version/v7/scripts/validate_train_memory_layout_v7.py \
		--layout "$(V7_TRAIN_LAYOUT_OUT)" \
		--ir2 "$(V7_TRAIN_CODEGEN_IR2)" \
		--output "$(V7_TRAIN_LAYOUT_AUDIT)" \
		$(if $(filter 1,$(V7_TRAIN_STRICT_UNRESOLVED)),--strict,)

v7-train-codegen: v7-train-memory-audit
	@set -e; \
	manifest="$$V7_TRAIN_MANIFEST"; \
	if [ -z "$$manifest" ]; then \
		manifest="$$( $(PYTHON) version/v7/scripts/resolve_train_manifest_v7.py )" || { \
			echo "ERROR: could not resolve train manifest. Set V7_TRAIN_MANIFEST=/path/to/weights_manifest.json"; exit 1; }; \
	fi; \
	$(PYTHON) version/v7/scripts/codegen_train_runtime_v7.py \
		--ir2 "$(V7_TRAIN_CODEGEN_IR2)" \
		--manifest "$$manifest" \
		--layout "$(V7_TRAIN_LAYOUT_OUT)" \
		--output "$(V7_TRAIN_CODEGEN_OUT)" \
		--summary-out "$(V7_TRAIN_CODEGEN_SUMMARY)"

v7-train-compile-smoke: v7-train-codegen
	@mkdir -p $(V7_REPORT_DIR)
	@$(CC) -std=c11 -O2 -Iinclude -c "$(V7_TRAIN_CODEGEN_OUT)" -o "$(V7_TRAIN_CODEGEN_OBJ)"
	@echo "Compiled training runtime object: $(V7_TRAIN_CODEGEN_OBJ)"

v7-train-c-smoke: v7-train-compile-smoke

v7-init-tiny:
	@$(PYTHON) version/v7/scripts/init_tiny_train_model_v7.py \
		--output-dir "$(V7_TINY_OUT_DIR)" \
		--seed $(V7_TINY_SEED) \
		--layers $(V7_TINY_LAYERS) \
		--vocab-size $(V7_TINY_VOCAB) \
		--embed-dim $(V7_TINY_EMBED) \
		--hidden-dim $(V7_TINY_HIDDEN) \
		--num-heads $(V7_TINY_HEADS) \
		--num-kv-heads $(V7_TINY_KV_HEADS) \
		--context-len $(V7_TINY_CTX)

v7-grad-fd:
	@$(PYTHON) version/v7/scripts/check_fd_gradients_v7.py --json-out $(V7_REPORT_DIR)/fd_gradients_latest.json

v7-replay:
	@$(PYTHON) version/v7/scripts/check_replay_determinism_v7.py --json-out $(V7_REPORT_DIR)/replay_determinism_latest.json

v7-replay-accum:
	@$(PYTHON) version/v7/scripts/check_runtime_replay_accum_v7.py --json-out $(V7_REPORT_DIR)/replay_accum_latest.json

v7-backprop-plumbing:
	@set -e; \
	manifest="$$V7_TRAIN_MANIFEST"; \
	if [ -z "$$manifest" ]; then \
		manifest="$$( $(PYTHON) version/v7/scripts/resolve_train_manifest_v7.py )" || { \
			echo "ERROR: could not resolve train manifest. Set V7_TRAIN_MANIFEST=/path/to/weights_manifest.json"; exit 1; }; \
	fi; \
	if [ ! -f "$$manifest" ]; then \
		echo "ERROR: manifest not found: $$manifest"; exit 1; \
	fi; \
	if [ ! -f "$(V7_TRAIN_CODEGEN_IR2)" ] || [ ! -f "$(V7_TRAIN_LAYOUT_OUT)" ]; then \
		echo "v7-backprop-plumbing: IR2/layout not found; running v7-train-layout-smoke first"; \
		$(MAKE) --no-print-directory v7-train-layout-smoke; \
	fi; \
	runtime_report_args=""; \
	if [ -n "$(V7_BACKPROP_PLUMBING_RUNTIME_REPORT)" ] && [ -f "$(V7_BACKPROP_PLUMBING_RUNTIME_REPORT)" ]; then \
		runtime_report_args="--runtime-report $(V7_BACKPROP_PLUMBING_RUNTIME_REPORT)"; \
	fi; \
	runtime_summary="$(V7_BACKPROP_PLUMBING_RUNTIME_SUMMARY)"; \
	if [ -z "$$runtime_summary" ] && [ -f "$(V7_TRAIN_CODEGEN_SUMMARY)" ]; then \
		runtime_summary="$(V7_TRAIN_CODEGEN_SUMMARY)"; \
	fi; \
	runtime_summary_args=""; \
	if [ -n "$$runtime_summary" ] && [ -f "$$runtime_summary" ]; then \
		runtime_summary_args="--runtime-summary $$runtime_summary"; \
	fi; \
	$(PYTHON) version/v7/scripts/check_backprop_plumbing_v7.py \
		--ir2 "$(V7_TRAIN_CODEGEN_IR2)" \
		--layout "$(V7_TRAIN_LAYOUT_OUT)" \
		--manifest "$$manifest" \
		$$runtime_report_args \
		$$runtime_summary_args \
		$(if $(filter 1,$(V7_TRAIN_STRICT_UNRESOLVED)),--strict,) \
		--json-out "$(V7_BACKPROP_PLUMBING_JSON)"

v7-backprop-stitch-runtime:
	@$(PYTHON) version/v7/scripts/check_backprop_stitch_runtime_v7.py --keep-run-dir "$(V7_BACKPROP_STITCH_RUN_DIR)" --dump-check-topk "$(V7_BACKPROP_STITCH_DUMP_CHECK_TOPK)" --json-out "$(V7_BACKPROP_STITCH_JSON)"

v7-backprop-stitch-runtime-accum:
	@$(PYTHON) version/v7/scripts/check_backprop_stitch_runtime_v7.py --keep-run-dir "$(V7_BACKPROP_STITCH_RUN_DIR)" --grad-accum "$(V7_BACKPROP_STITCH_ACCUM)" --total-tokens "$(V7_BACKPROP_STITCH_TOTAL_TOKENS)" --dump-check-topk "$(V7_BACKPROP_STITCH_DUMP_CHECK_TOPK)" --json-out "$(V7_BACKPROP_STITCH_ACCUM_JSON)"

test-v7-bpe-train-parity: tokenizer ck-bpe-train
	@$(PYTHON) version/v7/scripts/test_bpe_train_parity_v7.py --json-out "$(V7_BPE_TRAIN_PARITY_JSON)"

test-v7-svg-overfit-regression:
	@$(PYTHON) version/v7/scripts/test_svg_overfit_regression_v7.py --json-out "$(V7_SVG_OVERFIT_JSON)"

visualizer: v7-ir-visualizer-e2e

v7-visualizer-health:
	@echo "Running visualizer health checks..."
	@$(PYTHON) version/v7/scripts/test_visualizer_health_v7.py --source --json-out $(V7_REPORT_DIR)/visualizer_health_latest.json
	@echo "Running visualizer JS unit tests..."
	@$(PYTHON) version/v7/scripts/test_visualizer_js_units_v7.py --json-out $(V7_REPORT_DIR)/visualizer_js_units_latest.json

v7-visualizer-generated-e2e:
	@echo "Running Level 3 generated-file E2E..."
	@$(PYTHON) version/v7/scripts/test_visualizer_generated_e2e_v7.py \
		$(if $(RUN),--run $(RUN),) \
		--json-out $(V7_REPORT_DIR)/visualizer_generated_e2e_latest.json

# ── Dataset Pipeline Targets ─────────────────────────────────────────────────
# Usage: make v7-dataset-all RUN=~/.cache/ck-engine-v7/models/train/<run_name>
# Each target maps to a tab in the Dataset Viewer.  See the Pipeline Map
# in the Preflight tab for the full script → artifact → tab mapping.
#
#   Target                  │ Artifact                            │ DV Tab
#   ────────────────────────┼─────────────────────────────────────┼───────────────
#   v7-dataset-normalize    │ normalized_assets_manifest.json     │ Vocabulary, Quality
#   v7-dataset-classify     │ asset_classification_manifest.json  │ Classification, Browse, Gallery
#   v7-dataset-embeddings   │ embeddings.json                     │ Embeddings
#   v7-dataset-attention    │ attention.json                      │ Attention
#   v7-dataset-viewer       │ dataset_viewer.html                 │ All tabs
#   v7-dataset-all          │ normalize+classify+embed+attn+viewer │ All tabs
#
V7_DATASET_RUN ?= $(RUN)

v7-dataset-normalize:
	@test -n "$(V7_DATASET_RUN)" || { echo "Usage: make v7-dataset-normalize RUN=<run_dir>"; exit 1; }
	$(PYTHON) version/v7/scripts/dataset/normalize_svg_assets_v7.py --workspace "$(V7_DATASET_RUN)/dataset" --force

v7-dataset-classify:
	@test -n "$(V7_DATASET_RUN)" || { echo "Usage: make v7-dataset-classify RUN=<run_dir>"; exit 1; }
	$(PYTHON) version/v7/scripts/dataset/classify_svg_assets_v7.py --workspace "$(V7_DATASET_RUN)/dataset" --force

v7-dataset-embeddings:
	@test -n "$(V7_DATASET_RUN)" || { echo "Usage: make v7-dataset-embeddings RUN=<run_dir>"; exit 1; }
	$(PYTHON) version/v7/tools/export_embeddings.py "$(V7_DATASET_RUN)"

v7-dataset-attention:
	@test -n "$(V7_DATASET_RUN)" || { echo "Usage: make v7-dataset-attention RUN=<run_dir>"; exit 1; }
	$(PYTHON) version/v7/tools/export_attention.py "$(V7_DATASET_RUN)" --probe

v7-dataset-viewer:
	@test -n "$(V7_DATASET_RUN)" || { echo "Usage: make v7-dataset-viewer RUN=<run_dir>"; exit 1; }
	$(PYTHON) version/v7/scripts/dataset/build_svg_dataset_visualizer_v7.py \
		--workspace "$(V7_DATASET_RUN)/dataset" \
		--output "$(V7_DATASET_RUN)/dataset_viewer.html"

v7-dataset-all:
	@test -n "$(V7_DATASET_RUN)" || { echo "Usage: make v7-dataset-all RUN=<run_dir>"; exit 1; }
	@echo "── Normalize ──"
	-$(PYTHON) version/v7/scripts/dataset/normalize_svg_assets_v7.py --workspace "$(V7_DATASET_RUN)/dataset" --force 2>&1 || echo "  ⏭ normalize skipped (may not apply to structured-atoms workspaces)"
	@echo "── Classify ──"
	-$(PYTHON) version/v7/scripts/dataset/classify_svg_assets_v7.py --workspace "$(V7_DATASET_RUN)/dataset" --force 2>&1 || echo "  ⏭ classify skipped (may not apply to structured-atoms workspaces)"
	@echo "── Embeddings + Attention + Viewer ──"
	$(PYTHON) version/v7/tools/prepare_run_viewer.py "$(V7_DATASET_RUN)" --force

visualizer-full:
	@$(MAKE) --no-print-directory v7-ir-visualizer-e2e V7_VISUALIZER_E2E_WITH_TRAIN=1

v7-ir-visualizer-e2e-nightly:
	@$(MAKE) --no-print-directory v7-ir-visualizer-e2e V7_VISUALIZER_E2E_WITH_TRAIN=1 V7_VISUALIZER_E2E_SKIP_INFERENCE_PARITY=1

v7-runbook-e2e:
	@$(PYTHON) version/v7/scripts/test_v7_runbook_e2e_v7.py \
		--run-dir "$(V7_RUNBOOK_E2E_RUN)" \
		--mode "$(V7_RUNBOOK_E2E_MODE)" \
		--data "$(V7_RUNBOOK_E2E_DATA)" \
		--json-out "$(V7_RUNBOOK_E2E_JSON)"

v7-ir-visualizer-e2e:
	@set -e; \
	extra_flags=""; \
	if [ "$(V7_VISUALIZER_E2E_FORCE_COMPILE)" = "1" ]; then extra_flags="$$extra_flags --force-compile"; fi; \
	if [ "$(V7_VISUALIZER_E2E_FORCE_CONVERT)" = "1" ]; then extra_flags="$$extra_flags --force-convert"; fi; \
	if [ "$(V7_VISUALIZER_E2E_WITH_TRAIN)" = "1" ]; then extra_flags="$$extra_flags --with-train-runtime"; fi; \
	if [ "$(V7_VISUALIZER_E2E_SKIP_INFERENCE_PARITY)" = "1" ]; then extra_flags="$$extra_flags --skip-inference-parity"; fi; \
	$(PYTHON) version/v7/scripts/test_ir_visualizer_e2e_v7.py \
		--model-input "$(V7_VISUALIZER_E2E_MODEL)" \
		--context-len "$(V7_VISUALIZER_E2E_CONTEXT)" \
		--max-tokens "$(V7_VISUALIZER_E2E_MAX_TOKENS)" \
		--json-out "$(V7_VISUALIZER_E2E_JSON)" \
		$$extra_flags

v7-train-data-pipeline:
	@RUN_DIR="$(if $(RUN),$(RUN),$(V7_PIPELINE_RUN))"; \
	WORK_DIR="$(if $(V7_PIPELINE_WORK_DIR),$(V7_PIPELINE_WORK_DIR),$$RUN_DIR/.ck_pipeline/latest)"; \
	$(PYTHON) version/v7/scripts/train_data_pipeline_v7.py \
		--run "$$RUN_DIR" \
		--init-if-missing \
		--tokenizer "$(V7_PIPELINE_TOKENIZER)" \
		--dataset-repeats "$(V7_PIPELINE_DATASET_REPEATS)" \
		--epochs "$(V7_PIPELINE_EPOCHS)" \
		--seq-len "$(V7_PIPELINE_SEQ_LEN)" \
		--total-tokens "$(V7_PIPELINE_TOTAL_TOKENS)" \
		--grad-accum "$(V7_PIPELINE_GRAD_ACCUM)" \
		--lr "$(V7_PIPELINE_LR)" \
		--work-dir "$$WORK_DIR" \
		--json-out "$(V7_PIPELINE_JSON)" \
		$(if $(filter 1,$(V7_PIPELINE_WITH_TORCH)),--with-torch-ref,) \
		$(if $(filter 0,$(V7_PIPELINE_OPEN_VIS)),--no-open-visualizer,--open-visualizer)

v7-stabilization-nightly:
	@set -e; \
	extra_flags=""; \
	if [ "$(V7_STABILIZATION_RUNTIME_CHECKS)" != "1" ]; then extra_flags="$$extra_flags --no-runtime-checks"; fi; \
	if [ "$(V7_STABILIZATION_BACKEND_XRAY)" != "1" ]; then extra_flags="$$extra_flags --no-backend-xray"; fi; \
	if [ "$(V7_STABILIZATION_FORCE)" = "1" ]; then extra_flags="$$extra_flags --force-regimen"; fi; \
	if [ -n "$(V7_STABILIZATION_RUN_ROOT)" ]; then extra_flags="$$extra_flags --run-root $(V7_STABILIZATION_RUN_ROOT)"; fi; \
	if [ -n "$(V7_STABILIZATION_MAIN_RUN_DIR)" ]; then extra_flags="$$extra_flags --main-run-dir $(V7_STABILIZATION_MAIN_RUN_DIR)"; fi; \
	$(PYTHON) version/v7/scripts/run_v7_stabilization_nightly_v7.py \
		--dataset "$(V7_STABILIZATION_DATA)" \
		--layers "$(V7_STABILIZATION_LAYERS)" \
		--token-budgets "$(V7_STABILIZATION_TOKEN_BUDGETS)" \
		--seq-len "$(V7_STABILIZATION_SEQ_LEN)" \
		--grad-accum-sweep "$(V7_STABILIZATION_GRAD_ACCUM_SWEEP)" \
		--sweep-epochs "$(V7_STABILIZATION_SWEEP_EPOCHS)" \
		--forward-epochs "$(V7_STABILIZATION_FORWARD_EPOCHS)" \
		--vocab "$(V7_STABILIZATION_VOCAB)" \
		--d-model "$(V7_STABILIZATION_D_MODEL)" \
		--hidden "$(V7_STABILIZATION_HIDDEN)" \
		--lr "$(V7_STABILIZATION_LR)" \
		--seed "$(V7_STABILIZATION_SEED)" \
		--loss-tol "$(V7_STABILIZATION_LOSS_TOL)" \
		--param-tol "$(V7_STABILIZATION_PARAM_TOL)" \
		--ck-loss-backend "$(V7_STABILIZATION_CK_LOSS_BACKEND)" \
		--json-out "$(V7_STABILIZATION_JSON)" \
		--md-out "$(V7_STABILIZATION_MD)" \
		--history-jsonl "$(V7_STABILIZATION_HISTORY)" \
		$$extra_flags

v7-gate-train:
	@$(MAKE) --no-print-directory v7-inference-smoke
	@$(MAKE) --no-print-directory v7-validate-contracts
	@$(MAKE) --no-print-directory v7-train-ir-smoke
	@$(MAKE) --no-print-directory v7-train-memory-audit
	@$(MAKE) --no-print-directory v7-train-compile-smoke
	@$(MAKE) --no-print-directory v7-parity-1tok
	@$(MAKE) --no-print-directory v7-grad-fd
	@$(MAKE) --no-print-directory v7-train-parity-3
	@if [ "$(V7_GATE_WITH_BACKPROP_PLUMBING)" = "1" ]; then \
		echo "v7-gate-train: running backprop plumbing audit"; \
		$(MAKE) --no-print-directory v7-backprop-plumbing; \
	else \
		echo "v7-gate-train: skip backprop plumbing audit (set V7_GATE_WITH_BACKPROP_PLUMBING=1 to enable)"; \
	fi
	@if [ "$(V7_GATE_WITH_BACKPROP_STITCH_RUNTIME)" = "1" ]; then \
		echo "v7-gate-train: running one-step runtime stitch smoke"; \
		$(MAKE) --no-print-directory v7-backprop-stitch-runtime; \
	else \
		echo "v7-gate-train: skip one-step runtime stitch smoke (set V7_GATE_WITH_BACKPROP_STITCH_RUNTIME=1 to enable)"; \
	fi
	@if [ "$(V7_GATE_WITH_LONG_HORIZON_PARITY)" = "1" ]; then \
		echo "v7-gate-train: running long-horizon drift smoke"; \
		$(MAKE) --no-print-directory v7-train-parity-drift-smoke; \
	else \
		echo "v7-gate-train: skip long-horizon drift smoke (set V7_GATE_WITH_LONG_HORIZON_PARITY=1 to enable)"; \
	fi
	@if [ "$(V7_GATE_WITH_KERNEL_PARITY)" = "1" ]; then \
		echo "v7-gate-train: running training-kernel parity suite"; \
		$(MAKE) --no-print-directory v7-kernel-parity-train; \
	else \
		echo "v7-gate-train: skip training-kernel parity suite (set V7_GATE_WITH_KERNEL_PARITY=1 to enable)"; \
	fi
	@if [ "$(V7_GATE_WITH_BPE_TRAIN_PARITY)" = "1" ]; then \
		echo "v7-gate-train: running bpe-train parity gate"; \
		$(MAKE) --no-print-directory test-v7-bpe-train-parity; \
	else \
		echo "v7-gate-train: skip bpe-train parity gate (set V7_GATE_WITH_BPE_TRAIN_PARITY=1 to enable)"; \
	fi
	@if [ "$(V7_GATE_WITH_SVG_OVERFIT)" = "1" ]; then \
		echo "v7-gate-train: running svg-overfit regression gate"; \
		$(MAKE) --no-print-directory test-v7-svg-overfit-regression; \
	else \
		echo "v7-gate-train: skip svg-overfit regression gate (set V7_GATE_WITH_SVG_OVERFIT=1 to enable)"; \
	fi
	@if [ "$(V7_GATE_WITH_REPLAY_ACCUM)" = "1" ]; then \
		echo "v7-gate-train: running runtime replay accum gate"; \
		$(MAKE) --no-print-directory v7-replay-accum; \
	else \
		echo "v7-gate-train: skip runtime replay accum gate (set V7_GATE_WITH_REPLAY_ACCUM=1 to enable)"; \
	fi
	@$(MAKE) --no-print-directory v7-replay

v7-gate:
	@$(MAKE) --no-print-directory v7-validate-contracts
	@$(MAKE) --no-print-directory v7-parity-1tok

v7: v7-gate

v6.6-ir-visualizer:
	@if command -v python3 >/dev/null 2>&1; then \
		echo "Opening IR visualizer..."; \
		python3 -m http.server 8080 --directory version/v6.6/tools & \
		sleep 2 && xdg-open http://localhost:8080/ir_visualizer.html 2>/dev/null || \
		open http://localhost:8080/ir_visualizer.html 2>/dev/null || \
		echo "Open http://localhost:8080/ir_visualizer.html in your browser"; \
	fi

# IR Visualizer - New v6.6 targets
.PHONY: vis vis-generate vis-list

vis:
	@echo "Opening IR visualizer..."
	@python version/v6.6/tools/open_ir_visualizer.py

vis-list:
	@echo "Available models in cache:"
	@python version/v6.6/tools/open_ir_visualizer.py --list

vis-%:
	@echo "Opening IR visualizer for $*..."
	@python version/v6.6/tools/open_ir_visualizer.py $*

# =============================================================================
# V6.6 Comprehensive Profiling
# =============================================================================

PROFILE_V6_SCRIPT := version/v6.6/scripts/ck_run_v6_6.py
PERF_ARTIFACTS_V6_SCRIPT := version/v6.6/scripts/perf_artifacts_v6_6.py
VTUNE_ARTIFACTS_V6_SCRIPT := version/v6.6/scripts/vtune_artifacts_v6_6.py
MEMORY_SIGNOFF_V6_SCRIPT := version/v6.6/scripts/memory_signoff_v6_6.py
PERF_GATE_V6_SCRIPT := version/v6.6/scripts/perf_gate_v6_6.py
RESOLVE_MODEL_DIR_V6_SCRIPT := version/v6.6/scripts/resolve_model_dir_v6_6.py
PROFILE_V6_SUMMARY_SCRIPT := version/v6.6/scripts/generate_profile_summary_v6_6.py
PROFILE_V6_PERF_DATA ?= build/ck_v6_perf.data
PROFILE_V6_PERF_FOLDED ?= build/ck_v6_perf.folded
PROFILE_V6_FLAMEGRAPH_SVG ?= build/flamegraph_v6.svg
PROFILE_V6_PERF_STAT_TXT ?= build/ck_v6_perf_stat.txt
PROFILE_V6_DEBUG_CFLAGS ?= -fno-omit-frame-pointer -g
PROFILE_V6_CALLGRAPH ?= dwarf,16384
PROFILE_V6_COMPILER ?= gcc
PROFILE_V6_VTUNE_TEXT ?= build/ck_v6_vtune_hotspots.txt
PROFILE_V6_VTUNE_CSV ?= build/ck_v6_vtune_hotspots.csv

PROFILE_V7_SCRIPT := version/v7/scripts/ck_run_v7.py
PERF_ARTIFACTS_V7_SCRIPT := version/v7/scripts/perf_artifacts_v7.py
VTUNE_ARTIFACTS_V7_SCRIPT := version/v7/scripts/vtune_artifacts_v7.py
ADVISOR_ARTIFACTS_V7_SCRIPT := version/v7/scripts/advisor_artifacts_v7.py
MEMORY_SIGNOFF_V7_SCRIPT := version/v7/scripts/memory_signoff_v7.py
PERF_GATE_V7_SCRIPT := version/v7/scripts/perf_gate_v7.py
RESOLVE_MODEL_DIR_V7_SCRIPT := version/v7/scripts/resolve_model_dir_v7.py
PROFILE_V7_SUMMARY_SCRIPT := version/v7/scripts/generate_profile_summary_v7.py
PROFILE_V7_PERF_DATA ?= build/ck_v7_perf.data
PROFILE_V7_PERF_FOLDED ?= build/ck_v7_perf.folded
PROFILE_V7_FLAMEGRAPH_SVG ?= build/flamegraph_v7.svg
V7_FLAMEGRAPH_MODE ?= decode
PROFILE_V7_PERF_STAT_TXT ?= build/ck_v7_perf_stat.txt
PROFILE_V7_DEBUG_CFLAGS ?= -fno-omit-frame-pointer -g
PROFILE_V7_CALLGRAPH ?= dwarf,16384
PROFILE_V7_COMPILER ?= gcc
PROFILE_V7_VTUNE_TEXT ?= build/ck_v7_vtune_hotspots.txt
PROFILE_V7_VTUNE_CSV ?= build/ck_v7_vtune_hotspots.csv
PROFILE_V7_VTUNE_MEMORY_TEXT ?= build/ck_v7_vtune_memory_summary.txt
PROFILE_V7_VTUNE_MEMORY_CSV ?= build/ck_v7_vtune_memory_summary.csv
PROFILE_V7_VTUNE_UARCH_TEXT ?= build/ck_v7_vtune_uarch_summary.txt
PROFILE_V7_VTUNE_UARCH_CSV ?= build/ck_v7_vtune_uarch_summary.csv
PROFILE_V7_ADVISOR_TEXT ?= build/ck_v7_advisor_roofline.txt
PROFILE_V7_ADVISOR_CSV ?= build/ck_v7_advisor_roofline.csv
PROFILE_V7_ADVISOR_HTML ?= build/ck_v7_advisor_roofline.html

profile-v6-prepare-runtime:
	@if [ "$(V66_PREP_WITH_PYTHON)" != "1" ]; then \
		echo "SKIP: python prep disabled (V66_PREP_WITH_PYTHON=$(V66_PREP_WITH_PYTHON))"; \
	else \
		prep_model="$(V66_MODEL)"; \
		if [ -d "$$prep_model" ]; then \
			gguf_path="$$(find "$$prep_model" -maxdepth 1 -type f -name '*.gguf' | head -n 1)"; \
			if [ -n "$$gguf_path" ]; then prep_model="$$gguf_path"; fi; \
		fi; \
		echo "Preparing runtime via ck_run_v6_6.py (model=$$prep_model, force_compile=$(V66_FORCE_COMPILE))"; \
		CK_PROFILE=1 \
		CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
		CK_V6_EXTRA_CFLAGS="$(PROFILE_V6_DEBUG_CFLAGS)" \
		$(PYTHON) $(PROFILE_V6_SCRIPT) run \
			"$$prep_model" \
			--generate-only \
			--profile \
			$(V66_FORCE_COMPILE_ARG) \
			--context-len 1024 --prompt "Hello" --max-tokens 1 \
			$(V66_RUN_ARGS); \
	fi

profile-v6-decode:
	@if [ "$(V66_PERF_RUNTIME)" = "cli" ]; then \
		model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$(V66_MODEL)" )"; \
		needs_regen=0; regen_reason=""; \
		if [ ! -f "$$model_dir/libmodel.so" ] || [ ! -f "$$model_dir/weights.bump" ]; then \
			needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
		elif ldd "$$model_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
			needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
		fi; \
		if [ "$$needs_regen" -eq 0 ]; then \
			echo "Using existing compiled runtime in $$model_dir"; \
		else \
			echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
			regen_model="$(V66_MODEL)"; \
			if [ -d "$$regen_model" ]; then \
				gguf_path="$$(find "$$regen_model" -maxdepth 1 -type f -name '*.gguf' | head -n 1)"; \
				if [ -n "$$gguf_path" ]; then regen_model="$$gguf_path"; fi; \
			fi; \
			CK_PROFILE=1 \
			CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
			CK_V6_EXTRA_CFLAGS="$(PROFILE_V6_DEBUG_CFLAGS)" \
				$(PYTHON) $(PROFILE_V6_SCRIPT) run \
					"$$regen_model" \
					--generate-only --profile $(V66_FORCE_COMPILE_ARG) \
					--context-len 1024 --prompt "Hello" --max-tokens 1 \
					$(V66_RUN_ARGS); \
				model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$$regen_model" )"; \
			fi; \
		$(MAKE) --no-print-directory ck-cli-v6.6 CFLAGS="$(CFLAGS) $(PROFILE_V6_DEBUG_CFLAGS)"; \
		CK_PROFILE=1 \
		CK_PROFILE_CSV="$$model_dir/profile_decode.csv" \
		CK_PROFILE_JSON="$$model_dir/profile_decode.json" \
		./build/ck-cli-v6.6 "$$model_dir/libmodel.so" "$$model_dir/weights.bump" \
			--prompt "The quick brown fox" --max-tokens 32 --timing \
			$(V66_CLI_TEMPLATE_ARGS) $(V66_CLI_ARGS); \
		$(PYTHON) $(PROFILE_V6_SUMMARY_SCRIPT) --work-dir "$$model_dir"; \
	else \
		CK_PROFILE=1 \
		CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
		$(PYTHON) $(PROFILE_V6_SCRIPT) run \
			"$(V66_MODEL)" \
			--profile $(V66_FORCE_COMPILE_ARG) \
			--prompt "The quick brown fox" --max-tokens 32 \
			$(V66_RUN_ARGS); \
	fi
	@echo "Profile data saved in model cache directory"

profile-v6-prefill:
	@if [ "$(V66_PERF_RUNTIME)" = "cli" ]; then \
		model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$(V66_MODEL)" )"; \
		needs_regen=0; regen_reason=""; \
		if [ ! -f "$$model_dir/libmodel.so" ] || [ ! -f "$$model_dir/weights.bump" ]; then \
			needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
		elif ldd "$$model_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
			needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
		fi; \
		if [ "$$needs_regen" -eq 0 ]; then \
			echo "Using existing compiled runtime in $$model_dir"; \
		else \
			echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
			regen_model="$(V66_MODEL)"; \
			if [ -d "$$regen_model" ]; then \
				gguf_path="$$(find "$$regen_model" -maxdepth 1 -type f -name '*.gguf' | head -n 1)"; \
				if [ -n "$$gguf_path" ]; then regen_model="$$gguf_path"; fi; \
			fi; \
			CK_PROFILE=1 \
			CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
			CK_V6_EXTRA_CFLAGS="$(PROFILE_V6_DEBUG_CFLAGS)" \
				$(PYTHON) $(PROFILE_V6_SCRIPT) run \
					"$$regen_model" \
					--generate-only --profile $(V66_FORCE_COMPILE_ARG) \
					--context-len 1024 --prompt "Hello" --max-tokens 1 \
					$(V66_RUN_ARGS); \
				model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$$regen_model" )"; \
			fi; \
		$(MAKE) --no-print-directory ck-cli-v6.6 CFLAGS="$(CFLAGS) $(PROFILE_V6_DEBUG_CFLAGS)"; \
		CK_PROFILE=1 \
		CK_PROFILE_CSV="$$model_dir/profile_decode.csv" \
		CK_PROFILE_JSON="$$model_dir/profile_decode.json" \
		./build/ck-cli-v6.6 "$$model_dir/libmodel.so" "$$model_dir/weights.bump" \
			--prompt "Explain the theory of relativity in simple terms" --max-tokens 1 --timing \
			$(V66_CLI_TEMPLATE_ARGS) $(V66_CLI_ARGS); \
		$(PYTHON) $(PROFILE_V6_SUMMARY_SCRIPT) --work-dir "$$model_dir"; \
	else \
		CK_PROFILE=1 \
		CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
		$(PYTHON) $(PROFILE_V6_SCRIPT) run \
			"$(V66_MODEL)" \
			--profile $(V66_FORCE_COMPILE_ARG) \
			--prompt "Explain the theory of relativity in simple terms" \
			--max-tokens 1 \
			$(V66_RUN_ARGS); \
	fi
	@echo "Profile data saved in model cache directory"

profile-v6-flamegraph:
	@if ! command -v perf >/dev/null 2>&1; then \
		echo "SKIP: perf not installed"; \
	elif [ ! -x ./FlameGraph/stackcollapse-perf.pl ] || [ ! -x ./FlameGraph/flamegraph.pl ]; then \
		echo "SKIP: FlameGraph tools missing (expected ./FlameGraph/stackcollapse-perf.pl and ./FlameGraph/flamegraph.pl)"; \
	else \
		mkdir -p build; \
		if [ "$(V66_PERF_RUNTIME)" = "cli" ]; then \
			model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$(V66_MODEL)" )"; \
			needs_regen=0; regen_reason=""; \
			if [ ! -f "$$model_dir/libmodel.so" ] || [ ! -f "$$model_dir/weights.bump" ]; then \
				needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
			elif ldd "$$model_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
				needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
			fi; \
			if [ "$$needs_regen" -eq 0 ]; then \
				echo "Using existing compiled runtime in $$model_dir"; \
			else \
				echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
				regen_model="$(V66_MODEL)"; \
				if [ -d "$$regen_model" ]; then \
					gguf_path="$$(find "$$regen_model" -maxdepth 1 -type f -name '*.gguf' | head -n 1)"; \
					if [ -n "$$gguf_path" ]; then regen_model="$$gguf_path"; fi; \
				fi; \
				CK_PROFILE=1 \
				CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
				CK_V6_EXTRA_CFLAGS="$(PROFILE_V6_DEBUG_CFLAGS)" \
				$(PYTHON) $(PROFILE_V6_SCRIPT) run \
					"$$regen_model" \
					--generate-only --profile $(V66_FORCE_COMPILE_ARG) \
					--context-len 1024 --prompt "Hello" --max-tokens 1 \
					$(V66_RUN_ARGS); \
				model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$$regen_model" )"; \
			fi; \
			$(MAKE) --no-print-directory ck-cli-v6.6 CFLAGS="$(CFLAGS) $(PROFILE_V6_DEBUG_CFLAGS)"; \
			perf record --all-user -F 999 --call-graph $(PROFILE_V6_CALLGRAPH) -o $(PROFILE_V6_PERF_DATA) -- \
				./build/ck-cli-v6.6 "$$model_dir/libmodel.so" "$$model_dir/weights.bump" \
				--prompt "The quick brown fox" --max-tokens 32 --timing \
				$(V66_CLI_TEMPLATE_ARGS) $(V66_CLI_ARGS); \
		else \
			CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
			CK_V6_EXTRA_CFLAGS="$(PROFILE_V6_DEBUG_CFLAGS)" \
			perf record --all-user -F 999 --call-graph $(PROFILE_V6_CALLGRAPH) -o $(PROFILE_V6_PERF_DATA) -- \
			$(PYTHON) $(PROFILE_V6_SCRIPT) run \
				"$(V66_MODEL)" \
				$(V66_FORCE_COMPILE_ARG) \
				--prompt "The quick brown fox" --max-tokens 32 \
				$(V66_RUN_ARGS); \
		fi; \
		perf script -i $(PROFILE_V6_PERF_DATA) | \
			./FlameGraph/stackcollapse-perf.pl | \
			tee $(PROFILE_V6_PERF_FOLDED) | \
			./FlameGraph/flamegraph.pl --title="CK v6.6 Decode" > $(PROFILE_V6_FLAMEGRAPH_SVG); \
		$(PYTHON) $(PERF_ARTIFACTS_V6_SCRIPT) \
			--model-input "$(V66_MODEL)" \
			--perf-data $(PROFILE_V6_PERF_DATA) \
			--folded $(PROFILE_V6_PERF_FOLDED) \
			--flamegraph-svg $(PROFILE_V6_FLAMEGRAPH_SVG); \
	fi

profile-v6-perf-stat:
	@if ! command -v perf >/dev/null 2>&1; then \
		echo "SKIP: perf not installed"; \
	else \
		mkdir -p build; \
		if [ "$(V66_PERF_RUNTIME)" = "cli" ]; then \
			model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$(V66_MODEL)" )"; \
			needs_regen=0; regen_reason=""; \
			if [ ! -f "$$model_dir/libmodel.so" ] || [ ! -f "$$model_dir/weights.bump" ]; then \
				needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
			elif ldd "$$model_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
				needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
			fi; \
			if [ "$$needs_regen" -eq 0 ]; then \
				echo "Using existing compiled runtime in $$model_dir"; \
			else \
				echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
				regen_model="$(V66_MODEL)"; \
				if [ -d "$$regen_model" ]; then \
					gguf_path="$$(find "$$regen_model" -maxdepth 1 -type f -name '*.gguf' | head -n 1)"; \
					if [ -n "$$gguf_path" ]; then regen_model="$$gguf_path"; fi; \
				fi; \
				CK_PROFILE=1 \
				CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
				CK_V6_EXTRA_CFLAGS="$(PROFILE_V6_DEBUG_CFLAGS)" \
				$(PYTHON) $(PROFILE_V6_SCRIPT) run \
					"$$regen_model" \
					--generate-only --profile $(V66_FORCE_COMPILE_ARG) \
					--context-len 1024 --prompt "Hello" --max-tokens 1 \
					$(V66_RUN_ARGS); \
				model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$$regen_model" )"; \
			fi; \
			$(MAKE) --no-print-directory ck-cli-v6.6 CFLAGS="$(CFLAGS) $(PROFILE_V6_DEBUG_CFLAGS)"; \
			perf stat --all-user -e cycles,instructions,cache-references,cache-misses,\
LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,\
branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend \
				./build/ck-cli-v6.6 "$$model_dir/libmodel.so" "$$model_dir/weights.bump" \
				--prompt "The quick brown fox" --max-tokens 32 --timing \
				$(V66_CLI_TEMPLATE_ARGS) $(V66_CLI_ARGS) \
				2> $(PROFILE_V6_PERF_STAT_TXT); \
		else \
			CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
			perf stat --all-user -e cycles,instructions,cache-references,cache-misses,\
LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,\
branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend \
			$(PYTHON) $(PROFILE_V6_SCRIPT) run \
				"$(V66_MODEL)" \
				--context-len 1024 --prompt "The quick brown fox" --max-tokens 32 \
				$(V66_RUN_ARGS) \
				2> $(PROFILE_V6_PERF_STAT_TXT); \
		fi; \
		cat $(PROFILE_V6_PERF_STAT_TXT); \
		$(PYTHON) $(PERF_ARTIFACTS_V6_SCRIPT) \
			--model-input "$(V66_MODEL)" \
			--perf-stat $(PROFILE_V6_PERF_STAT_TXT); \
	fi

profile-v6-cachegrind:
	valgrind --tool=cachegrind --cachegrind-out-file=build/cachegrind_v6.out \
	$(PYTHON) $(PROFILE_V6_SCRIPT) run \
		hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf \
		--context-len 1024 --prompt "Hello" --max-tokens 4
	cg_annotate build/cachegrind_v6.out > build/cachegrind_v6_annotated.txt

profile-v6-vtune:
	@if [ "$(V66_WITH_VTUNE)" != "1" ]; then \
		echo "SKIP: VTune probe disabled (V66_WITH_VTUNE=$(V66_WITH_VTUNE))"; \
	elif ! command -v vtune >/dev/null 2>&1; then \
		echo "SKIP: vtune not installed"; \
	elif [ -r /proc/sys/kernel/yama/ptrace_scope ] && [ "$$(cat /proc/sys/kernel/yama/ptrace_scope)" -gt 0 ]; then \
		echo "SKIP: vtune blocked by kernel.yama.ptrace_scope=$$(cat /proc/sys/kernel/yama/ptrace_scope)"; \
		echo "      run: sudo sysctl -w kernel.yama.ptrace_scope=0"; \
	else \
		mkdir -p build; \
		vtune_result="build/ck_v6_vtune_$$(date +%Y%m%d_%H%M%S)"; \
		if [ "$(V66_PERF_RUNTIME)" = "cli" ]; then \
			model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$(V66_MODEL)" )"; \
			needs_regen=0; regen_reason=""; \
			if [ ! -f "$$model_dir/libmodel.so" ] || [ ! -f "$$model_dir/weights.bump" ]; then \
				needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
			elif ldd "$$model_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
				needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
			fi; \
			if [ "$$needs_regen" -eq 0 ]; then \
				echo "Using existing compiled runtime in $$model_dir"; \
			else \
				echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
				regen_model="$(V66_MODEL)"; \
				if [ -d "$$regen_model" ]; then \
					gguf_path="$$(find "$$regen_model" -maxdepth 1 -type f -name '*.gguf' | head -n 1)"; \
					if [ -n "$$gguf_path" ]; then regen_model="$$gguf_path"; fi; \
				fi; \
				CK_PROFILE=1 \
				CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
				CK_V6_EXTRA_CFLAGS="$(PROFILE_V6_DEBUG_CFLAGS)" \
				$(PYTHON) $(PROFILE_V6_SCRIPT) run \
					"$$regen_model" \
					--generate-only --profile $(V66_FORCE_COMPILE_ARG) \
					--context-len 1024 --prompt "Hello" --max-tokens 1 \
					$(V66_RUN_ARGS); \
				model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V6_SCRIPT) --model-input "$$regen_model" )"; \
			fi; \
			$(MAKE) --no-print-directory ck-cli-v6.6 CFLAGS="$(CFLAGS) $(PROFILE_V6_DEBUG_CFLAGS)"; \
			vtune -collect hotspots -result-dir "$$vtune_result" -quiet -- \
				./build/ck-cli-v6.6 "$$model_dir/libmodel.so" "$$model_dir/weights.bump" \
				--prompt "The quick brown fox" --max-tokens 32 --timing \
				$(V66_CLI_TEMPLATE_ARGS) $(V66_CLI_ARGS) || { \
					echo "SKIP: vtune collect failed for CLI runtime"; \
					exit 0; \
				}; \
		else \
			CK_V6_COMPILER="$(PROFILE_V6_COMPILER)" \
			CK_V6_EXTRA_CFLAGS="$(PROFILE_V6_DEBUG_CFLAGS)" \
			vtune -collect hotspots -result-dir "$$vtune_result" -quiet -- \
			$(PYTHON) $(PROFILE_V6_SCRIPT) run \
				"$(V66_MODEL)" \
				$(V66_FORCE_COMPILE_ARG) \
				--prompt "The quick brown fox" --max-tokens 32 \
				$(V66_RUN_ARGS) || { \
					echo "SKIP: vtune collect failed for python runtime"; \
					exit 0; \
				}; \
		fi; \
		vtune -report hotspots -result-dir "$$vtune_result" -format text -report-output $(PROFILE_V6_VTUNE_TEXT) >/dev/null 2>&1 || true; \
		vtune -report hotspots -result-dir "$$vtune_result" -format csv -report-output $(PROFILE_V6_VTUNE_CSV) >/dev/null 2>&1 || true; \
		$(PYTHON) $(VTUNE_ARTIFACTS_V6_SCRIPT) \
			--model-input "$(V66_MODEL)" \
			--result-dir "$$vtune_result" \
			--report-text $(PROFILE_V6_VTUNE_TEXT) \
			--report-csv $(PROFILE_V6_VTUNE_CSV); \
	fi

profile-v6-full:
	$(MAKE) profile-v6-prepare-runtime
	$(MAKE) profile-v6-decode
	$(MAKE) profile-v6-prefill
	$(MAKE) profile-v6-perf-stat
	$(MAKE) profile-v6-flamegraph
	$(MAKE) profile-v6-vtune

# v7 native profiling targets.
profile-v7-prepare-runtime:
	@if [ "$(V7_PREP_WITH_PYTHON)" != "1" ]; then \
		echo "SKIP: python prep disabled (V7_PREP_WITH_PYTHON=$(V7_PREP_WITH_PYTHON))"; \
	else \
		prep_model="$(V7_MODEL)"; \
		# Keep explicit model directory stable; do not remap to *.gguf stem cache dir. \
		echo "Preparing runtime via ck_run_v7.py (model=$$prep_model, force_compile=$(V7_FORCE_COMPILE))"; \
		CK_PROFILE=1 \
		CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
		CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
		$(PYTHON) $(PROFILE_V7_SCRIPT) run \
			"$$prep_model" \
			--generate-only \
			--profile \
			$(V7_FORCE_COMPILE_ARG) \
			--context-len 1024 --prompt "Hello" --max-tokens 1 \
			$(V7_RUN_ARGS); \
	fi

profile-v7-decode:
	@if [ "$(V7_PERF_RUNTIME)" = "cli" ]; then \
		model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$(V7_MODEL)" )"; \
		runtime_dir="$$model_dir"; \
		if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
			runtime_dir="$$model_dir/.ck_build"; \
		fi; \
		needs_regen=0; regen_reason=""; \
		if [ "$(V7_FORCE_COMPILE)" = "1" ]; then \
			needs_regen=1; regen_reason="V7_FORCE_COMPILE=1"; \
		fi; \
		if [ ! -f "$$runtime_dir/libmodel.so" ] || [ ! -f "$$runtime_dir/weights.bump" ]; then \
			needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
		elif ldd "$$runtime_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
			needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
		fi; \
		if [ "$$needs_regen" -eq 0 ]; then \
			echo "Using existing compiled runtime in $$runtime_dir"; \
		else \
			echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
			regen_model="$(V7_MODEL)"; \
			# Keep explicit model directory stable; do not remap to *.gguf stem cache dir. \
			CK_PROFILE=1 \
			CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
			CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
			$(PYTHON) $(PROFILE_V7_SCRIPT) run \
				"$$regen_model" \
				--generate-only --profile $(V7_FORCE_COMPILE_ARG) \
				--context-len 1024 --prompt "Hello" --max-tokens 1 \
				$(V7_RUN_ARGS); \
			model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$$regen_model" )"; \
			runtime_dir="$$model_dir"; \
			if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
				runtime_dir="$$model_dir/.ck_build"; \
			fi; \
		fi; \
		$(MAKE) --no-print-directory ck-cli-v7 CFLAGS="$(CFLAGS) $(PROFILE_V7_DEBUG_CFLAGS)"; \
			decode_log="$$model_dir/profile_decode_run.log"; \
			CK_PROFILE=1 \
			CK_PROFILE_CSV="$$runtime_dir/profile_decode.csv" \
			CK_PROFILE_JSON="$$runtime_dir/profile_decode.json" \
			./build/ck-cli-v7 "$$runtime_dir/libmodel.so" "$$runtime_dir/weights.bump" \
				--prompt "The quick brown fox" --max-tokens 32 --timing --quiet-output \
					$(V7_CLI_TEMPLATE_ARGS) $(V7_CLI_ARGS) > "$$decode_log" 2>&1; \
			cli_rc=$$?; \
			if [ "$$cli_rc" -ne 0 ]; then \
				echo "ERROR: decode profiling run failed (rc=$$cli_rc). Log: $$decode_log"; \
				tail -n 80 "$$decode_log" || true; \
				echo "Hint: if you see libimf.so errors, run: source /opt/intel/oneapi/setvars.sh"; \
				exit "$$cli_rc"; \
			fi; \
			if [ ! -s "$$runtime_dir/profile_decode.csv" ]; then \
				echo "ERROR: missing $$runtime_dir/profile_decode.csv after decode profiling run. Log: $$decode_log"; \
				tail -n 80 "$$decode_log" || true; \
				exit 1; \
			fi; \
			$(PYTHON) $(PROFILE_V7_SUMMARY_SCRIPT) --work-dir "$$model_dir"; \
	else \
		CK_PROFILE=1 \
		CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
		$(PYTHON) $(PROFILE_V7_SCRIPT) run \
			"$(V7_MODEL)" \
			--profile $(V7_FORCE_COMPILE_ARG) \
			--prompt "The quick brown fox" --max-tokens 32 \
			$(V7_RUN_ARGS); \
	fi
	@echo "Profile data saved in model cache directory"

profile-v7-prefill:
	@if [ "$(V7_PERF_RUNTIME)" = "cli" ]; then \
		model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$(V7_MODEL)" )"; \
		runtime_dir="$$model_dir"; \
		if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
			runtime_dir="$$model_dir/.ck_build"; \
		fi; \
		needs_regen=0; regen_reason=""; \
		if [ "$(V7_FORCE_COMPILE)" = "1" ]; then \
			needs_regen=1; regen_reason="V7_FORCE_COMPILE=1"; \
		fi; \
		if [ ! -f "$$runtime_dir/libmodel.so" ] || [ ! -f "$$runtime_dir/weights.bump" ]; then \
			needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
		elif ldd "$$runtime_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
			needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
		fi; \
		if [ "$$needs_regen" -eq 0 ]; then \
			echo "Using existing compiled runtime in $$runtime_dir"; \
		else \
			echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
			regen_model="$(V7_MODEL)"; \
			# Keep explicit model directory stable; do not remap to *.gguf stem cache dir. \
			CK_PROFILE=1 \
			CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
			CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
			$(PYTHON) $(PROFILE_V7_SCRIPT) run \
				"$$regen_model" \
				--generate-only --profile $(V7_FORCE_COMPILE_ARG) \
				--context-len 1024 --prompt "Hello" --max-tokens 1 \
				$(V7_RUN_ARGS); \
			model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$$regen_model" )"; \
			runtime_dir="$$model_dir"; \
			if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
				runtime_dir="$$model_dir/.ck_build"; \
			fi; \
		fi; \
		$(MAKE) --no-print-directory ck-cli-v7 CFLAGS="$(CFLAGS) $(PROFILE_V7_DEBUG_CFLAGS)"; \
		CK_PROFILE=1 \
		CK_PROFILE_CSV="$$runtime_dir/profile_decode.csv" \
		CK_PROFILE_JSON="$$runtime_dir/profile_decode.json" \
		./build/ck-cli-v7 "$$runtime_dir/libmodel.so" "$$runtime_dir/weights.bump" \
			--prompt "Explain the theory of relativity in simple terms" --max-tokens 1 --timing --quiet-output \
				$(V7_CLI_TEMPLATE_ARGS) $(V7_CLI_ARGS) > /dev/null 2>&1; \
		$(PYTHON) $(PROFILE_V7_SUMMARY_SCRIPT) --work-dir "$$model_dir"; \
	else \
		CK_PROFILE=1 \
		CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
		$(PYTHON) $(PROFILE_V7_SCRIPT) run \
			"$(V7_MODEL)" \
			--profile $(V7_FORCE_COMPILE_ARG) \
			--prompt "Explain the theory of relativity in simple terms" \
			--max-tokens 1 \
			$(V7_RUN_ARGS); \
	fi
	@echo "Profile data saved in model cache directory"

profile-v7-flamegraph:
	@if ! command -v perf >/dev/null 2>&1; then \
		echo "SKIP: perf not installed"; \
	elif [ ! -x ./FlameGraph/stackcollapse-perf.pl ] || [ ! -x ./FlameGraph/flamegraph.pl ]; then \
		echo "SKIP: FlameGraph tools missing (expected ./FlameGraph/stackcollapse-perf.pl and ./FlameGraph/flamegraph.pl)"; \
	else \
		mkdir -p build; \
		fg_mode="$(V7_FLAMEGRAPH_MODE)"; \
		if [ "$$fg_mode" != "prefill" ]; then fg_mode="decode"; fi; \
		if [ "$$fg_mode" = "prefill" ]; then \
			fg_prompt="Explain the theory of relativity in simple terms"; \
			fg_max_tokens=1; \
			fg_title="CK v7 Prefill"; \
			fg_perf_data="build/ck_v7_perf_prefill.data"; \
			fg_perf_folded="build/ck_v7_perf_prefill.folded"; \
			fg_svg="build/flamegraph_v7_prefill.svg"; \
		else \
			fg_prompt="The quick brown fox"; \
			fg_max_tokens=32; \
			fg_title="CK v7 Decode"; \
			fg_perf_data="$(PROFILE_V7_PERF_DATA)"; \
			fg_perf_folded="$(PROFILE_V7_PERF_FOLDED)"; \
			fg_svg="$(PROFILE_V7_FLAMEGRAPH_SVG)"; \
		fi; \
		echo "profile-v7-flamegraph mode=$$fg_mode output=$$fg_svg"; \
		if [ "$(V7_PERF_RUNTIME)" = "cli" ]; then \
			model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$(V7_MODEL)" )"; \
		runtime_dir="$$model_dir"; \
		if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
			runtime_dir="$$model_dir/.ck_build"; \
		fi; \
				needs_regen=0; regen_reason=""; \
				if [ "$(V7_FORCE_COMPILE)" = "1" ]; then \
					needs_regen=1; regen_reason="V7_FORCE_COMPILE=1"; \
				fi; \
				if [ ! -f "$$runtime_dir/libmodel.so" ] || [ ! -f "$$runtime_dir/weights.bump" ]; then \
					needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
				elif ldd "$$runtime_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
					needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
				fi; \
			if [ "$$needs_regen" -eq 0 ]; then \
				echo "Using existing compiled runtime in $$runtime_dir"; \
			else \
				echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
				regen_model="$(V7_MODEL)"; \
				# Keep explicit model directory stable; do not remap to *.gguf stem cache dir. \
				CK_PROFILE=1 \
				CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
				CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
				$(PYTHON) $(PROFILE_V7_SCRIPT) run \
					"$$regen_model" \
					--generate-only --profile $(V7_FORCE_COMPILE_ARG) \
					--context-len 1024 --prompt "Hello" --max-tokens 1 \
					$(V7_RUN_ARGS); \
				model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$$regen_model" )"; \
			runtime_dir="$$model_dir"; \
			if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
				runtime_dir="$$model_dir/.ck_build"; \
			fi; \
			fi; \
			$(MAKE) --no-print-directory ck-cli-v7 CFLAGS="$(CFLAGS) $(PROFILE_V7_DEBUG_CFLAGS)"; \
			perf record --all-user -F 999 --call-graph $(PROFILE_V7_CALLGRAPH) -o "$$fg_perf_data" -- \
				./build/ck-cli-v7 "$$runtime_dir/libmodel.so" "$$runtime_dir/weights.bump" \
				--prompt "$$fg_prompt" --max-tokens "$$fg_max_tokens" --timing --quiet-output \
				$(V7_CLI_TEMPLATE_ARGS) $(V7_CLI_ARGS) > /dev/null; \
		else \
			CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
			CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
			perf record --all-user -F 999 --call-graph $(PROFILE_V7_CALLGRAPH) -o "$$fg_perf_data" -- \
			$(PYTHON) $(PROFILE_V7_SCRIPT) run \
				"$(V7_MODEL)" \
				$(V7_FORCE_COMPILE_ARG) \
				--prompt "$$fg_prompt" --max-tokens "$$fg_max_tokens" \
				$(V7_RUN_ARGS); \
		fi; \
		perf script -i "$$fg_perf_data" | \
			./FlameGraph/stackcollapse-perf.pl | \
			tee "$$fg_perf_folded" | \
			./FlameGraph/flamegraph.pl --title="$$fg_title" > "$$fg_svg"; \
		$(PYTHON) $(PERF_ARTIFACTS_V7_SCRIPT) \
			--model-input "$(V7_MODEL)" \
			--perf-data "$$fg_perf_data" \
			--folded "$$fg_perf_folded" \
			--flamegraph-svg "$$fg_svg" \
			--mode "$$fg_mode"; \
	fi

profile-v7-flamegraph-decode:
	@$(MAKE) --no-print-directory profile-v7-flamegraph V7_FLAMEGRAPH_MODE=decode

profile-v7-flamegraph-prefill:
	@$(MAKE) --no-print-directory profile-v7-flamegraph V7_FLAMEGRAPH_MODE=prefill

.PHONY: profile-v7-flamegraph profile-v7-flamegraph-decode profile-v7-flamegraph-prefill

profile-v7-perf-stat:
	@if ! command -v perf >/dev/null 2>&1; then \
		echo "SKIP: perf not installed"; \
	else \
		mkdir -p build; \
		perf_events="cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend,dTLB-loads,dTLB-load-misses,dTLB-stores,dTLB-store-misses,iTLB-load-misses,minor-faults,major-faults"; \
		if perf list 2>/dev/null | grep -q "dtlb_load_misses.walk_completed"; then \
			perf_events="$$perf_events,cpu_core/dtlb_load_misses.walk_completed/,cpu_core/dtlb_store_misses.walk_completed/,cpu_core/itlb_misses.walk_completed/,cpu_core/dtlb_load_misses.stlb_hit/,cpu_core/itlb_misses.stlb_hit/"; \
		fi; \
		if [ "$(V7_PERF_RUNTIME)" = "cli" ]; then \
			model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$(V7_MODEL)" )"; \
		runtime_dir="$$model_dir"; \
		if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
			runtime_dir="$$model_dir/.ck_build"; \
		fi; \
				needs_regen=0; regen_reason=""; \
				if [ "$(V7_FORCE_COMPILE)" = "1" ]; then \
					needs_regen=1; regen_reason="V7_FORCE_COMPILE=1"; \
				fi; \
				if [ ! -f "$$runtime_dir/libmodel.so" ] || [ ! -f "$$runtime_dir/weights.bump" ]; then \
					needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
				elif ldd "$$runtime_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
					needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
				fi; \
			if [ "$$needs_regen" -eq 0 ]; then \
				echo "Using existing compiled runtime in $$runtime_dir"; \
			else \
				echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
				regen_model="$(V7_MODEL)"; \
				# Keep explicit model directory stable; do not remap to *.gguf stem cache dir. \
				CK_PROFILE=1 \
				CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
				CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
				$(PYTHON) $(PROFILE_V7_SCRIPT) run \
					"$$regen_model" \
					--generate-only --profile $(V7_FORCE_COMPILE_ARG) \
					--context-len 1024 --prompt "Hello" --max-tokens 1 \
					$(V7_RUN_ARGS); \
				model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$$regen_model" )"; \
			runtime_dir="$$model_dir"; \
			if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
				runtime_dir="$$model_dir/.ck_build"; \
			fi; \
			fi; \
			$(MAKE) --no-print-directory ck-cli-v7 CFLAGS="$(CFLAGS) $(PROFILE_V7_DEBUG_CFLAGS)"; \
			perf stat --all-user -e "$$perf_events" \
				./build/ck-cli-v7 "$$runtime_dir/libmodel.so" "$$runtime_dir/weights.bump" \
				--prompt "The quick brown fox" --max-tokens 32 --timing --quiet-output \
				$(V7_CLI_TEMPLATE_ARGS) $(V7_CLI_ARGS) > /dev/null \
				2> $(PROFILE_V7_PERF_STAT_TXT); \
		else \
			CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
			perf stat --all-user -e "$$perf_events" \
			$(PYTHON) $(PROFILE_V7_SCRIPT) run \
				"$(V7_MODEL)" \
				--context-len 1024 --prompt "The quick brown fox" --max-tokens 32 \
				$(V7_RUN_ARGS) \
				2> $(PROFILE_V7_PERF_STAT_TXT); \
		fi; \
		cat $(PROFILE_V7_PERF_STAT_TXT); \
		$(PYTHON) $(PERF_ARTIFACTS_V7_SCRIPT) \
			--model-input "$(V7_MODEL)" \
			--perf-stat $(PROFILE_V7_PERF_STAT_TXT); \
	fi

profile-v7-cachegrind:
	valgrind --tool=cachegrind --cachegrind-out-file=build/cachegrind_v7.out \
	$(PYTHON) $(PROFILE_V7_SCRIPT) run \
		"$(V7_MODEL)" \
		--context-len 1024 --prompt "Hello" --max-tokens 4 $(V7_RUN_ARGS)
	cg_annotate build/cachegrind_v7.out > build/cachegrind_v7_annotated.txt


profile-v7-vtune:
	@if [ "$(V7_WITH_VTUNE)" != "1" ]; then \
		echo "SKIP: VTune probe disabled (V7_WITH_VTUNE=$(V7_WITH_VTUNE))"; \
	elif ! command -v vtune >/dev/null 2>&1; then \
		echo "SKIP: vtune not installed"; \
	elif [ -r /proc/sys/kernel/yama/ptrace_scope ] && [ "$$(cat /proc/sys/kernel/yama/ptrace_scope)" -gt 0 ]; then \
		echo "SKIP: vtune blocked by kernel.yama.ptrace_scope=$$(cat /proc/sys/kernel/yama/ptrace_scope)"; \
		echo "      run: sudo sysctl -w kernel.yama.ptrace_scope=0"; \
	else \
		mkdir -p build; \
		vtune_hot_result="build/ck_v7_vtune_hotspots_$$(date +%Y%m%d_%H%M%S)"; \
		vtune_mem_result=""; \
		vtune_uarch_result=""; \
		if [ "$(V7_PERF_RUNTIME)" = "cli" ]; then \
			model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$(V7_MODEL)" )"; \
		runtime_dir="$$model_dir"; \
		if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
			runtime_dir="$$model_dir/.ck_build"; \
		fi; \
				needs_regen=0; regen_reason=""; \
				if [ "$(V7_FORCE_COMPILE)" = "1" ]; then \
					needs_regen=1; regen_reason="V7_FORCE_COMPILE=1"; \
				fi; \
				if [ ! -f "$$runtime_dir/libmodel.so" ] || [ ! -f "$$runtime_dir/weights.bump" ]; then \
					needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
				elif ldd "$$runtime_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
					needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
				fi; \
			if [ "$$needs_regen" -eq 0 ]; then \
				echo "Using existing compiled runtime in $$runtime_dir"; \
			else \
				echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
				regen_model="$(V7_MODEL)"; \
				# Keep explicit model directory stable; do not remap to *.gguf stem cache dir. \
				CK_PROFILE=1 \
				CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
				CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
				$(PYTHON) $(PROFILE_V7_SCRIPT) run \
					"$$regen_model" \
					--generate-only --profile $(V7_FORCE_COMPILE_ARG) \
					--context-len 1024 --prompt "Hello" --max-tokens 1 \
					$(V7_RUN_ARGS); \
				model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$$regen_model" )"; \
			runtime_dir="$$model_dir"; \
			if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
				runtime_dir="$$model_dir/.ck_build"; \
			fi; \
			fi; \
			$(MAKE) --no-print-directory ck-cli-v7 CFLAGS="$(CFLAGS) $(PROFILE_V7_DEBUG_CFLAGS)"; \
			vtune -collect hotspots -result-dir "$$vtune_hot_result" -quiet -- \
				./build/ck-cli-v7 "$$runtime_dir/libmodel.so" "$$runtime_dir/weights.bump" \
				--prompt "The quick brown fox" --max-tokens 32 --timing --quiet-output \
				$(V7_CLI_TEMPLATE_ARGS) $(V7_CLI_ARGS) > /dev/null || { \
					echo "SKIP: vtune collect failed for CLI runtime"; \
					exit 0; \
				}; \
			if [ "$(V7_VTUNE_DEEP)" = "1" ]; then \
				vtune_mem_result="build/ck_v7_vtune_memory_$$(date +%Y%m%d_%H%M%S)"; \
				vtune -collect memory-access -result-dir "$$vtune_mem_result" -quiet -- \
					./build/ck-cli-v7 "$$runtime_dir/libmodel.so" "$$runtime_dir/weights.bump" \
					--prompt "The quick brown fox" --max-tokens 32 --timing --quiet-output \
				$(V7_CLI_TEMPLATE_ARGS) $(V7_CLI_ARGS) > /dev/null || { \
						echo "WARN: vtune memory-access collect failed"; \
						vtune_mem_result=""; \
					}; \
				vtune_uarch_result="build/ck_v7_vtune_uarch_$$(date +%Y%m%d_%H%M%S)"; \
				vtune -collect uarch-exploration -result-dir "$$vtune_uarch_result" -quiet -- \
					./build/ck-cli-v7 "$$runtime_dir/libmodel.so" "$$runtime_dir/weights.bump" \
					--prompt "The quick brown fox" --max-tokens 32 --timing --quiet-output \
				$(V7_CLI_TEMPLATE_ARGS) $(V7_CLI_ARGS) > /dev/null || { \
						echo "WARN: vtune uarch-exploration collect failed"; \
						vtune_uarch_result=""; \
					}; \
			fi; \
		else \
			CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
			CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
			vtune -collect hotspots -result-dir "$$vtune_hot_result" -quiet -- \
			$(PYTHON) $(PROFILE_V7_SCRIPT) run \
				"$(V7_MODEL)" \
				$(V7_FORCE_COMPILE_ARG) \
				--prompt "The quick brown fox" --max-tokens 32 \
				$(V7_RUN_ARGS) || { \
					echo "SKIP: vtune collect failed for python runtime"; \
					exit 0; \
				}; \
			if [ "$(V7_VTUNE_DEEP)" = "1" ]; then \
				vtune_mem_result="build/ck_v7_vtune_memory_$$(date +%Y%m%d_%H%M%S)"; \
				vtune -collect memory-access -result-dir "$$vtune_mem_result" -quiet -- \
					$(PYTHON) $(PROFILE_V7_SCRIPT) run \
					"$(V7_MODEL)" \
					$(V7_FORCE_COMPILE_ARG) \
					--prompt "The quick brown fox" --max-tokens 32 \
					$(V7_RUN_ARGS) || { \
						echo "WARN: vtune memory-access collect failed"; \
						vtune_mem_result=""; \
					}; \
				vtune_uarch_result="build/ck_v7_vtune_uarch_$$(date +%Y%m%d_%H%M%S)"; \
				vtune -collect uarch-exploration -result-dir "$$vtune_uarch_result" -quiet -- \
					$(PYTHON) $(PROFILE_V7_SCRIPT) run \
					"$(V7_MODEL)" \
					$(V7_FORCE_COMPILE_ARG) \
					--prompt "The quick brown fox" --max-tokens 32 \
					$(V7_RUN_ARGS) || { \
						echo "WARN: vtune uarch-exploration collect failed"; \
						vtune_uarch_result=""; \
					}; \
			fi; \
		fi; \
		vtune -report hotspots -result-dir "$$vtune_hot_result" -format text -report-output $(PROFILE_V7_VTUNE_TEXT) >/dev/null 2>&1 || true; \
		vtune -report hotspots -result-dir "$$vtune_hot_result" -format csv -report-output $(PROFILE_V7_VTUNE_CSV) >/dev/null 2>&1 || true; \
		extra_args=""; \
		if [ -n "$$vtune_mem_result" ]; then \
			vtune -report summary -result-dir "$$vtune_mem_result" -format text -report-output $(PROFILE_V7_VTUNE_MEMORY_TEXT) >/dev/null 2>&1 || true; \
			vtune -report summary -result-dir "$$vtune_mem_result" -format csv -report-output $(PROFILE_V7_VTUNE_MEMORY_CSV) >/dev/null 2>&1 || true; \
			extra_args="$$extra_args --analysis-name memory-access --analysis-result-dir $$vtune_mem_result --analysis-report-text $(PROFILE_V7_VTUNE_MEMORY_TEXT) --analysis-report-csv $(PROFILE_V7_VTUNE_MEMORY_CSV)"; \
		fi; \
		if [ -n "$$vtune_uarch_result" ]; then \
			vtune -report summary -result-dir "$$vtune_uarch_result" -format text -report-output $(PROFILE_V7_VTUNE_UARCH_TEXT) >/dev/null 2>&1 || true; \
			vtune -report summary -result-dir "$$vtune_uarch_result" -format csv -report-output $(PROFILE_V7_VTUNE_UARCH_CSV) >/dev/null 2>&1 || true; \
			extra_args="$$extra_args --analysis-name uarch-exploration --analysis-result-dir $$vtune_uarch_result --analysis-report-text $(PROFILE_V7_VTUNE_UARCH_TEXT) --analysis-report-csv $(PROFILE_V7_VTUNE_UARCH_CSV)"; \
		fi; \
		$(PYTHON) $(VTUNE_ARTIFACTS_V7_SCRIPT) \
			--model-input "$(V7_MODEL)" \
			--result-dir "$$vtune_hot_result" \
			--report-text $(PROFILE_V7_VTUNE_TEXT) \
			--report-csv $(PROFILE_V7_VTUNE_CSV) \
			$$extra_args; \
	fi

profile-v7-advisor:
	@if [ "$(V7_WITH_ADVISOR)" != "1" ]; then \
		echo "SKIP: Advisor probe disabled (V7_WITH_ADVISOR=$(V7_WITH_ADVISOR))"; \
	elif ! command -v advisor >/dev/null 2>&1; then \
		echo "SKIP: advisor not installed"; \
	else \
		mkdir -p build; \
		advisor_result="build/ck_v7_advisor_$$(date +%Y%m%d_%H%M%S)"; \
		if [ "$(V7_PERF_RUNTIME)" = "cli" ]; then \
			model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$(V7_MODEL)" )"; \
			runtime_dir="$$model_dir"; \
			if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
				runtime_dir="$$model_dir/.ck_build"; \
			fi; \
				needs_regen=0; regen_reason=""; \
				if [ "$(V7_FORCE_COMPILE)" = "1" ]; then \
					needs_regen=1; regen_reason="V7_FORCE_COMPILE=1"; \
				fi; \
				if [ ! -f "$$runtime_dir/libmodel.so" ] || [ ! -f "$$runtime_dir/weights.bump" ]; then \
					needs_regen=1; regen_reason="missing libmodel.so or weights.bump"; \
				elif ldd "$$runtime_dir/libmodel.so" 2>/dev/null | grep -q "libimf.so => not found"; then \
					needs_regen=1; regen_reason="libimf missing (rebuild with gcc)"; \
				fi; \
			if [ "$$needs_regen" -eq 0 ]; then \
				echo "Using existing compiled runtime in $$runtime_dir"; \
			else \
				echo "Regenerating runtime in $$model_dir ($$regen_reason)"; \
				regen_model="$(V7_MODEL)"; \
				# Keep explicit model directory stable; do not remap to *.gguf stem cache dir. \
				CK_PROFILE=1 \
				CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
				CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
				$(PYTHON) $(PROFILE_V7_SCRIPT) run \
					"$$regen_model" \
					--generate-only --profile $(V7_FORCE_COMPILE_ARG) \
					--context-len 1024 --prompt "Hello" --max-tokens 1 \
					$(V7_RUN_ARGS); \
				model_dir="$$( $(PYTHON) $(RESOLVE_MODEL_DIR_V7_SCRIPT) --model-input "$$regen_model" )"; \
				runtime_dir="$$model_dir"; \
				if [ -f "$$model_dir/.ck_build/libmodel.so" ] && [ -f "$$model_dir/.ck_build/weights.bump" ]; then \
					runtime_dir="$$model_dir/.ck_build"; \
				fi; \
			fi; \
			$(MAKE) --no-print-directory ck-cli-v7 CFLAGS="$(CFLAGS) $(PROFILE_V7_DEBUG_CFLAGS)"; \
			advisor --collect=roofline --project-dir "$$advisor_result" -- \
				./build/ck-cli-v7 "$$runtime_dir/libmodel.so" "$$runtime_dir/weights.bump" \
				--prompt "The quick brown fox" --max-tokens 32 --timing --quiet-output \
				$(V7_CLI_TEMPLATE_ARGS) $(V7_CLI_ARGS) || { \
					echo "SKIP: advisor collect failed for CLI runtime"; \
					exit 0; \
				}; \
		else \
			CK_V7_COMPILER="$(PROFILE_V7_COMPILER)" \
			CK_V7_EXTRA_CFLAGS="$(PROFILE_V7_DEBUG_CFLAGS)" \
			advisor --collect=roofline --project-dir "$$advisor_result" -- \
			$(PYTHON) $(PROFILE_V7_SCRIPT) run \
				"$(V7_MODEL)" \
				$(V7_FORCE_COMPILE_ARG) \
				--prompt "The quick brown fox" --max-tokens 32 \
				$(V7_RUN_ARGS) || { \
					echo "SKIP: advisor collect failed for python runtime"; \
					exit 0; \
				}; \
		fi; \
		advisor --report=roofline --project-dir "$$advisor_result" --format=text --report-output $(PROFILE_V7_ADVISOR_TEXT) >/dev/null 2>&1 || true; \
		advisor --report=roofline --project-dir "$$advisor_result" --format=csv --report-output $(PROFILE_V7_ADVISOR_CSV) >/dev/null 2>&1 || true; \
		advisor --report=roofline --project-dir "$$advisor_result" --format=html --report-output $(PROFILE_V7_ADVISOR_HTML) >/dev/null 2>&1 || true; \
		advisor_args=""; \
		if [ -f "$(PROFILE_V7_ADVISOR_TEXT)" ]; then advisor_args="$$advisor_args --report-text $(PROFILE_V7_ADVISOR_TEXT)"; fi; \
		if [ -f "$(PROFILE_V7_ADVISOR_CSV)" ]; then advisor_args="$$advisor_args --report-csv $(PROFILE_V7_ADVISOR_CSV)"; fi; \
		if [ -f "$(PROFILE_V7_ADVISOR_HTML)" ]; then advisor_args="$$advisor_args --report-html $(PROFILE_V7_ADVISOR_HTML)"; fi; \
		$(PYTHON) $(ADVISOR_ARTIFACTS_V7_SCRIPT) \
			--model-input "$(V7_MODEL)" \
			--project-dir "$$advisor_result" \
			$$advisor_args; \
	fi

profile-v7-full:
	@$(MAKE) --no-print-directory profile-v7-prepare-runtime
	@$(MAKE) --no-print-directory profile-v7-decode
	@$(MAKE) --no-print-directory profile-v7-prefill
	@$(MAKE) --no-print-directory profile-v7-perf-stat
	@$(MAKE) --no-print-directory profile-v7-flamegraph-decode
	@$(MAKE) --no-print-directory profile-v7-flamegraph-prefill
	@$(MAKE) --no-print-directory profile-v7-vtune
	@$(MAKE) --no-print-directory profile-v7-advisor
	@echo "=== Open visualizer: version/v7/tools/ir_visualizer.html ==="
	@echo "=== Load folder from model cache to see Profile tab ==="

v6.6-memory-signoff:
	@$(PYTHON) version/v6.6/scripts/ck_run_v6_6.py run "$(V66_MODEL)" --generate-only $(V66_FORCE_COMPILE_ARG) --context-len 128 --max-tokens 1 --prompt "Hello" $(V66_RUN_ARGS)
	@$(PYTHON) $(MEMORY_SIGNOFF_V6_SCRIPT) --model-input "$(V66_MODEL)"

v6.6-perf-gate:
	@if ! command -v perf >/dev/null 2>&1; then \
		echo "SKIP: v6.6-perf-gate (perf not installed)"; \
	elif [ ! -x ./FlameGraph/stackcollapse-perf.pl ] || [ ! -x ./FlameGraph/flamegraph.pl ]; then \
		echo "SKIP: v6.6-perf-gate (FlameGraph tools missing)"; \
	else \
		$(MAKE) --no-print-directory profile-v6-prepare-runtime; \
		$(MAKE) --no-print-directory profile-v6-decode; \
		$(MAKE) --no-print-directory profile-v6-perf-stat; \
		$(MAKE) --no-print-directory profile-v6-flamegraph; \
		$(MAKE) --no-print-directory profile-v6-vtune || true; \
		$(MAKE) --no-print-directory v6.6-perf-gate-evaluate; \
	fi

v6.6-perf-gate-evaluate:
	@$(PYTHON) $(PERF_GATE_V6_SCRIPT) --model-input "$(V66_MODEL)"

# v7 perf gate.
v7-perf-gate:
	@if ! command -v perf >/dev/null 2>&1; then \
		echo "SKIP: v7-perf-gate (perf not installed)"; \
	elif [ ! -x ./FlameGraph/stackcollapse-perf.pl ] || [ ! -x ./FlameGraph/flamegraph.pl ]; then \
		echo "SKIP: v7-perf-gate (FlameGraph tools missing)"; \
	else \
		$(MAKE) --no-print-directory profile-v7-prepare-runtime; \
		$(MAKE) --no-print-directory profile-v7-decode; \
		$(MAKE) --no-print-directory profile-v7-perf-stat; \
		$(MAKE) --no-print-directory profile-v7-flamegraph-decode; \
		$(MAKE) --no-print-directory profile-v7-flamegraph-prefill; \
		$(MAKE) --no-print-directory profile-v7-vtune || true; \
		$(MAKE) --no-print-directory profile-v7-advisor || true; \
		$(MAKE) --no-print-directory v7-perf-gate-evaluate; \
	fi

v7-perf-gate-evaluate:
	@$(PYTHON) $(PERF_GATE_V7_SCRIPT) --model-input "$(V7_MODEL)"

v7-memory-signoff:
	@$(PYTHON) $(MEMORY_SIGNOFF_V7_SCRIPT) --model-input "$(V7_MODEL)" --gate-profile inference

# =============================================================================
# Thread Pool Tests
# =============================================================================

TEST_THREADPOOL := $(BUILD_DIR)/test_threadpool

THREADPOOL_CFLAGS := -O3 -fPIC -Wall $(AVX_FLAGS) $(SSSE3_FLAGS) $(INCLUDES)

$(TEST_THREADPOOL): tests/test_threadpool.c src/ck_threadpool.c src/ckernel_strict.c include/ck_threadpool.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(THREADPOOL_CFLAGS) -o $@ tests/test_threadpool.c src/ck_threadpool.c src/ckernel_strict.c \
		-Iinclude -lpthread -lm

test-threadpool: $(TEST_THREADPOOL)
	@echo ""
	@echo "========================================"
	@echo "  Thread Pool Unit Tests"
	@echo "========================================"
	./$(TEST_THREADPOOL)

bench-threadpool: $(TEST_THREADPOOL)
	@echo ""
	@echo "========================================"
	@echo "  Thread Pool Benchmark"
	@echo "========================================"
	./$(TEST_THREADPOOL) --bench

.PHONY: test-threadpool bench-threadpool
