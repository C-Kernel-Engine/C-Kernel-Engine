# Default to gcc for portability.
# (icx-built binaries typically depend on the Intel runtime, e.g. `libimf.so`.)
ifeq ($(origin CC),default)
CC := gcc
endif
# Some environments export CC=cc; treat that like "unset".
ifeq ($(CC),cc)
CC := gcc
endif
# Opt-in to icx selection if desired:
#   make CK_USE_ICX=1
ifeq ($(CK_USE_ICX),1)
CC := $(if $(shell command -v icx 2>/dev/null),icx,$(CC))
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
CPU_FLAGS := $(shell grep -m1 '^flags' /proc/cpuinfo 2>/dev/null)
# Detect FMA support
ifneq (,$(findstring fma,$(CPU_FLAGS)))
FMA_FLAGS := -mfma
else
FMA_FLAGS :=
endif
# Detect AVX level
DEFAULT_AVX_FLAGS :=
ifneq (,$(findstring avx512f,$(CPU_FLAGS)))
DEFAULT_AVX_FLAGS := -mavx512f $(FMA_FLAGS)
ifneq (,$(findstring avx512vnni,$(CPU_FLAGS)))
DEFAULT_AVX_FLAGS += -mavx512vnni
endif
ifneq (,$(findstring avx512_vnni,$(CPU_FLAGS)))
DEFAULT_AVX_FLAGS += -mavx512vnni
endif
else ifneq (,$(findstring avx2,$(CPU_FLAGS)))
DEFAULT_AVX_FLAGS := -mavx2 $(FMA_FLAGS)
else ifneq (,$(findstring avx,$(CPU_FLAGS)))
DEFAULT_AVX_FLAGS := -mavx $(FMA_FLAGS)
endif
AVX_FLAGS ?= $(DEFAULT_AVX_FLAGS)
INCLUDES := -Iinclude
CFLAGS  := -O3 -fPIC $(OPENMP_FLAG) -Wall $(AVX_FLAGS) $(INCLUDES)
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

SRCS    := src/backend_native.c \
           src/ckernel_ir.c \
           src/ckernel_ir_v2.c \
           src/ckernel_ir_v2_builder.c \
           src/ckernel_ir_v2_lower.c \
           src/ckernel_codegen.c \
           src/ckernel_codegen_v2.c \
           src/ckernel_codegen_v2_struct.c \
           src/ckernel_codegen_v2_dispatch.c \
           src/ckernel_codegen_v2_schedule.c \
           src/ckernel_codegen_v2_sections.c \
           src/ckernel_kernel_specs.c \
           src/ckernel_mem_plan.c \
           src/ckernel_alloc.c \
           src/ckernel_strict.c \
           src/ckernel_registry.c \
           src/ckernel_orchestration.c \
           src/ckernel_model_layout.c \
           src/ckernel_model_load.c \
           src/ckernel_model_load_v4.c \
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
	           src/kernels/gemm_kernels_q5_1.c \
	           src/kernels/gemm_kernels_q4k.c \
	           src/kernels/gemm_kernels_q6k.c \
	           src/kernels/gemm_kernels_q4k_q8k.c \
	           src/kernels/gemm_kernels_q4k_q8k_avx2.c \
	           src/kernels/gemm_kernels_q4k_q8k_vnni.c \
	           src/kernels/gemm_kernels_q8_0.c \
	           src/kernels/gemm_kernels_f16.c \
	           src/kernels/optimizer_kernels.c \
	           src/kernels/optimizer_kernels_bf16.c \
	           src/kernels/add_kernels_bf16.c
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

IR_DEMO := $(BUILD_DIR)/ck_ir_demo
IR_V2_DEMO := $(BUILD_DIR)/ck_ir_v2_demo
IR_V2_SCRIPT := scripts/build_ir_v2.py
IR_V4_SCRIPT := scripts/build_ir_v4.py
IR_V4_Q4K_SCRIPT := scripts/build_ir_v4_q4k.py
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
            unittest/test_optimizer.py

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
	@true

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

$(LIB_ATTENTION): $(BUILD_STAMP) src/kernels/attention_kernels.c src/kernels/softmax_kernels.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/attention_kernels.c src/kernels/softmax_kernels.c -lm

$(LIB_ROPE): $(BUILD_STAMP) src/kernels/rope_kernels.c src/kernels/rope_kernels_bf16.c include/ckernel_engine.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/rope_kernels.c src/kernels/rope_kernels_bf16.c -lm

$(LIB_QUANT): $(BUILD_STAMP) src/kernels/dequant_kernels.c src/kernels/gemm_kernels_q4_0.c src/kernels/gemm_kernels_q4_1.c src/kernels/gemm_kernels_q5_0.c src/kernels/gemm_kernels_q5_1.c src/kernels/gemm_kernels_q4k.c src/kernels/gemm_kernels_q6k.c src/kernels/gemm_kernels_q4k_q8k.c src/kernels/gemm_kernels_q4k_q8k_avx2.c src/kernels/gemm_kernels_q4k_q8k_vnni.c src/kernels/gemm_kernels_q8_0.c src/kernels/gemm_kernels_f16.c include/ckernel_quant.h include/ckernel_dtype.h
	$(CC) $(CFLAGS) -shared -o $@ src/kernels/dequant_kernels.c src/kernels/gemm_kernels_q4_0.c src/kernels/gemm_kernels_q4_1.c src/kernels/gemm_kernels_q5_0.c src/kernels/gemm_kernels_q5_1.c src/kernels/gemm_kernels_q4k.c src/kernels/gemm_kernels_q6k.c src/kernels/gemm_kernels_q4k_q8k.c src/kernels/gemm_kernels_q4k_q8k_avx2.c src/kernels/gemm_kernels_q4k_q8k_vnni.c src/kernels/gemm_kernels_q8_0.c src/kernels/gemm_kernels_f16.c -lm

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

test-libs: $(LIB_GELU) $(LIB_RMSNORM) $(LIB_LN) $(LIB_SOFT) $(LIB_SWIGLU) $(LIB_SIGMOID) $(LIB_ATTENTION) $(LIB_ROPE) $(LIB_RELU) $(LIB_VISION) $(LIB_QUANT)

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
	@echo "Batch Commands:"
	@echo "  make test             - Run core kernel tests"
	@echo "  make test-bf16        - Run BF16 kernel tests"
	@echo "  make nightly          - Run full nightly test suite (scripts/nightly_runner.py)"
	@echo ""

# Typo aliases
uittest: unittest
unittest-show: unittest

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
	@echo "  make test-v4-q4k      Run v4 Q4_K conversion pipeline (requires GGUF)"
	@echo "  make nightly          Run full test suite (doesn't stop on failure)"
	@echo ""
	@echo "Per-Kernel Libraries:"
	@echo "  make test-libs        Build per-kernel .so files for testing"
	@echo "  make test-relu        Run isolated ReLU C kernel tests (with PyTorch parity)"
	@echo ""
	@echo "Parity Tests (vs PyTorch):"
	@echo "  make litmus           LM head + cross-entropy backward parity"
	@echo "  make litmus-demo      100x100 litmus demo with SVG output"
	@echo "  make layer-parity     Full layer forward parity (GQA/MHA)"
	@echo "  make layer-parity-scalar  Layer parity with scalar build"
	@echo "  make tiny-parity      Tiny end-to-end training parity (1 step)"
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
	@echo "Comprehensive Test Suites:"
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
	@echo "  make nightly-parity   Only parity tests"
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
	@$(PYTHON) $(PYTHONFLAGS) scripts/convert_gguf_to_bump_v4.py \
	  --gguf "$(GGUF)" \
	  --output "$(GGUF_V4_OUT)" \
	  $(if $(GGUF_V4_CONFIG_OUT),--config-out "$(GGUF_V4_CONFIG_OUT)") \
	  $(if $(GGUF_V4_CONTEXT),--context "$(GGUF_V4_CONTEXT)")

hf-to-bump-v4:
	@if [ -z "$(HF_V4_CHECKPOINT)" ]; then \
	  echo "Usage: make hf-to-bump-v4 HF_V4_CHECKPOINT=/path/to/hf_model [HF_V4_OUT=$(HF_V4_OUT)] [HF_V4_DTYPE=q4_k] [HF_V4_CONTEXT=<n>]"; \
	  exit 2; \
	fi
	@$(PYTHON) $(PYTHONFLAGS) scripts/convert_hf_to_bump_v4.py \
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
	  LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(TEST_ENV) $(PYTHON) $(PYTHONFLAGS) $$t; \
	done; \
	echo "All Python kernel tests completed."

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

help:
	@echo "C-Kernel-Engine Make targets:"
	@echo ""
	@echo "  CLI (main entry point):"
	@echo "  make ck-cli          Build CLI tool with all dependencies"
	@echo "  make ck-cli-v4       Build v4 CLI (IR v4 codegen + compile)"
	@echo "  ./ck-v5 run <model>  Run v5 pipeline (manifest-first; inspect-only supported)"
	@echo "  make generate-model MODEL=<name>  Generate C code for model (inspect before compile)"
	@echo ""
	@echo "  Testing:"
	@echo "  make test-unit       Run unit tests only (fast, no model download)"
	@echo "  make test-v4         Full v4 test suite (download, convert, compile, test)"
	@echo "  make test-v4-clean   Full v4 test with force recompile"
	@echo "  ./build/ck run <model> [--generate-only] [--force-convert] [--verbose]"
	@echo ""
	@echo "  Build:"
	@echo "  make                 Build full engine library ($(LIB))"
	@echo "  make test            Build engine + per-kernel libs, run all Python kernel tests"
	@echo "  make $(IR_DEMO)      Build IR + codegen tool (HF config.json -> IR + C skeleton) into $(BUILD_DIR)"
	@echo "  make ck              Build IR tool and run it with $(DEFAULT_CONFIG) (prints forward/backward IR and C skeleton)"
	@echo "  make emit CONFIG=path OUT=path  Emit stitched runtime C file from config/IR (writes OUT.kernels manifest)"
	@echo "  make ck-emit CONFIG=path OUT=path  Alias for emit"
	@echo "  ./$(IR_DEMO) <config> --emit out.c  Emit a stitched runtime C file from config/IR (writes out.c.kernels)"
	@echo "  make $(IR_V2_DEMO)   Build IR v2 + codegen v2 tool into $(BUILD_DIR)"
	@echo "  make ck-v2           Build IR v2 tool and run it with $(DEFAULT_CONFIG) (writes build/ir_v2.json)"
	@echo "  make ir-v2 CONFIG=path [IR_V2_WEIGHTS=path] [IR_V2_CTX=N]  Build IR v2 JSON via Python"
	@echo "  make ir-v2 IR_V2_HF=repo_or_url [IR_V2_REV=main]  Build IR v2 JSON via Python (auto config/weights)"
	@echo "  make fetch-v2 CONFIG=path [IR_V2_WEIGHTS=path]  Download config/weights headers and write weights meta JSON"
	@echo "  make fetch-v2 IR_V2_HF=repo_or_url [IR_V2_REV=main]  Download headers and write weights meta JSON"
	@echo "  make ir-v2-meta CONFIG=path IR_V2_META=path  Build IR v2 JSON from config + weights meta (C-only)"
	@echo "  make emit-v2 CONFIG=path OUT=path  Emit v2 runtime C file from config/IR v2"
	@echo "  make emit-v2 IR=path OUT=path  Emit v2 runtime C file from IR v2 JSON"
	@echo "  make ck-emit-v2 CONFIG=path OUT=path  Alias for emit-v2"
	@echo "  ./$(IR_V2_DEMO) <config> --emit out.c  Emit a v2 runtime C file from config/IR v2"
	@echo "  make ir-v4 IR_V4_PRESET=qwen2-0.5b  Build IR v4 (graph/lowered/layout/codegen)"
	@echo "  make ir-v4 IR_V4_MODEL=Qwen/Qwen2-0.5B  Build IR v4 from HF model ID (config only)"
	@echo "  make ir-v4 IR_V4_CONFIG=path [IR_V4_PREFIX=dir] [IR_V4_TOKENS=N] [IR_V4_MODES=prefill,decode] [IR_V4_EMIT=exe|lib] [IR_V4_DTYPE=fp32|bf16]"
	@echo "  make ir-v4-q4k IR_V4_Q4K_CHECKPOINT=dir [IR_V4_Q4K_PRESET=name]  Build v4 + Q4_K weights"
	@echo "  make ir-v4-q4k IR_V4_Q4K_GGUF=path   Build v4 from GGUF (Q4_K/Q6_K)"
	@echo ""
	@echo "  Quantization:"
	@echo "  make test-quant       Run quantization kernel tests (dequant + q4/q8 gemm)"
	@echo "  make gguf-inspect GGUF=path   Inspect GGUF tensor dtypes (what is quantized?)"
	@echo "  make gguf-list GGUF=path      List all GGUF tensors (name/type/shape)"
	@echo "  make gguf-to-bump GGUF=path   Convert GGUF -> bump weights (GGUF_OUT/ GGUF_CONFIG_OUT / GGUF_CONTEXT)"
	@echo "  make gguf-to-bump-v4 GGUF=path   Convert GGUF -> bump weights (v4 layout)"
	@echo "  make hf-to-bump-v4 HF_V4_CHECKPOINT=path  Convert HF weights dir -> bump v4"
	@echo ""
	@echo "  Tests & CI:"
	@echo "  make test            Run core kernel tests"
	@echo "  make nightly         Run full test suite (all tests, summary report)"
	@echo "  make showtests       Show all available test targets"
	@echo "  make unittest        Show Python unittest scripts"
	@echo ""
	@echo "  make bench_gemm      Compare GEMM: Native vs MKL vs PyTorch"
	@echo "  make bench_gemm_native  Benchmark GEMM: Native vs PyTorch"
	@echo "  make bench_gemm_mkl     Benchmark GEMM: MKL vs PyTorch (requires oneAPI MKL)"
	@echo "  make profile-memory  Run Valgrind memcheck on tiny model"
	@echo "  make profile-heap    Run Valgrind massif heap profiler"
	@echo "  make profile-cpu     Run perf CPU profiler"
	@echo "  make profile-cache   Run Valgrind cachegrind"
	@echo "  make flamegraph      Generate SVG flamegraph (requires FlameGraph tools)"
	@echo "  make show_config     Display system hardware topology and recommendations"
	@echo "  make small-e2e       Same as tiny-e2e but ~10MB weights"
	@echo "  make libckernel_gelu.so      Build GELU-only shared library (outputs to $(LIB_GELU))"
	@echo "  make libckernel_rmsnorm.so   Build RMSNorm-only shared library (outputs to $(LIB_RMSNORM))"
	@echo "  make libckernel_layernorm.so Build LayerNorm-only shared library (AVX-512 if available, outputs to $(LIB_LN))"
	@echo "  make libckernel_softmax.so    Build Softmax-only shared library (outputs to $(LIB_SOFT))"
	@echo "  make libckernel_swiglu.so     Build SwiGLU-only shared library (outputs to $(LIB_SWIGLU))"
	@echo "  make libckernel_sigmoid.so    Build Sigmoid-only shared library (outputs to $(LIB_SIGMOID))"
	@echo "  make libckernel_relu.so       Build ReLU-only shared library (outputs to $(LIB_RELU))"
	@echo "  make libckernel_attention.so  Build attention-only shared library (scalar math + softmax kernel, outputs to $(LIB_ATTENTION))"
	@echo "  make libckernel_rope.so       Build RoPE-only shared library (outputs to $(LIB_ROPE))"
	@echo "  make gen-specs        Regenerate C kernel spec registry from kernel_maps/*.json"
	@echo ""
	@echo "Main CLI (ollama-style orchestrator):"
	@echo "  make ck-cli           Build the 'ck' orchestrator CLI"
	@echo "  make ck-cli-v4        Build the 'ck-v4' IR v4 builder CLI"
	@echo "    ./build/ck run HuggingFaceTB/SmolLM-135M"
	@echo "    ./build/ck run HuggingFaceTB/SmolLM-135M --threads <n> --inference-only --q8k-activations"
	@echo "    ./build/ck run https://huggingface.co/Qwen/Qwen2-0.5B --server"
	@echo "    ./build/ck list     List cached models"
	@echo "    ./build/ck remove <model>  Remove cached model"
	@echo "    ./build/ck-v4 build Qwen/Qwen2-0.5B"
	@echo "    ./ck-v5 run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf --weight-dtype=q4_k_m"
	@echo ""
	@echo "Interactive Tools (llama.cpp style):"
	@echo "  make ck-chat          Build interactive CLI (C-based)"
	@echo "  make ck-server        Build streaming HTTP server (C-based)"
	@echo "  make ck-chat-py       Run interactive chat (Python + C inference)"
	@echo "  make ck-server-py     Run streaming server (Python + C inference)"
	@echo ""
	@echo "  make clean            Remove all built libraries"

clean:
	rm -rf $(BUILD_DIR)

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

# v4 CLI (Python wrapper for IR v4 pipeline)
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

# Demo: Download Qwen2-0.5B Q4_K_M GGUF and run end-to-end
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
test-kernels:
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
	@echo "│  5. TEST COVERAGE                                                                                │"
	@echo "└──────────────────────────────────────────────────────────────────────────────────────────────────┘"
	@$(PYTHON) scripts/test_coverage.py --summary
	@$(PYTHON) scripts/test_coverage.py --missing
	@echo ""
	@echo "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐"
	@echo "│  6. HIGH-PRIORITY PENDING WORK                                                                   │"
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

.PHONY: all clean test test-bf16 test-libs test-quant unittest unittest-show help litmus litmus-test test-quick test-full test-stress profile-memory profile-heap profile-cpu profile-cache flamegraph ck-cli ck-cli-v4 ck-cli-v5 ck-chat ck-server ck-chat-py ck-server-py generate-model gguf-inspect gguf-list gguf-to-bump gguf-to-bump-v4 hf-to-bump-v4 ir-v4 ir-v4-q4k opt-status opt-pending opt-inference opt-training opt-kernels opt-targets opt-md kernel-coverage kernel-coverage-md test-coverage test-coverage-md meta-check meta-sync meta-init report report-md show_config show-config v5 demo-v5 demo-v5-debug
