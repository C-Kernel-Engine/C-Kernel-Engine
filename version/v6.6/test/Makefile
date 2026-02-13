# Makefile for v6.6 Test Suite
# Usage: make <target>
#
# This Makefile provides convenient targets for running tests and
# integrates with the parent Makefile for building.

.PHONY: all quick sanity parity memory debug nan trace clean help build download build-v66 bootstrap full test

# Optional model path (can override with MODEL=...).
# If empty, ck_test_runner auto-resolves from .ck_cache then ~/.cache.
MODEL ?=

# Python interpreter
PYTHON ?= $(if $(wildcard ../../.venv/bin/python),../../.venv/bin/python,python3)

# Directory wiring
TEST_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(TEST_DIR)/../../..)

# NOTE:
# - This test Makefile is invoked by root targets like `make v6.6-test-quick`.
# - Build/download targets below are convenience wrappers only and delegate to root Makefile.
# - The authoritative release pipeline is rooted in top-level `make v6.6-gate`.

# Convenience model for download bootstrap (HF GGUF URL, not local cache dir)
DOWNLOAD_MODEL ?= hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf

# ============================================================================
# QUICK TEST TARGETS
# ============================================================================

# Quick sanity check (embedding, offsets, memory)
quick sanity:
	@echo "Running quick sanity check..."
	@if [ -n "$(MODEL)" ]; then \
		$(PYTHON) ck_test_runner.py --quick --model "$(MODEL)"; \
	else \
		$(PYTHON) ck_test_runner.py --quick; \
	fi

# Full parity check layer-by-layer
parity:
	@echo "Running layer-by-layer parity check..."
	@if [ -n "$(MODEL)" ]; then \
		$(PYTHON) ck_test_runner.py --parity --model "$(MODEL)"; \
	else \
		$(PYTHON) ck_test_runner.py --parity; \
	fi

# Comprehensive memory validation
memory:
	@echo "Running memory validation..."
	@if [ -n "$(MODEL)" ]; then \
		$(PYTHON) ck_test_runner.py --memory --model "$(MODEL)"; \
	else \
		$(PYTHON) ck_test_runner.py --memory; \
	fi

# Find divergence between v6.5 and v6.6
divergence:
	@echo "Finding divergence between v6.5 and v6.6..."
	@if [ -n "$(MODEL)" ]; then \
		$(PYTHON) ck_test_runner.py --divergence --model "$(MODEL)"; \
	else \
		$(PYTHON) ck_test_runner.py --divergence; \
	fi

# Debug NaN issues
nan:
	@echo "Debugging NaN issues..."
	@if [ -n "$(MODEL)" ]; then \
		$(PYTHON) ck_test_runner.py --nan --model "$(MODEL)"; \
	else \
		$(PYTHON) ck_test_runner.py --nan; \
	fi

# Trace specific token
trace: ARGS ?=
trace:
	@echo "Tracing execution..."
	@if [ -n "$(MODEL)" ]; then \
		$(PYTHON) ck_test_runner.py --trace $(ARGS) --model "$(MODEL)"; \
	else \
		$(PYTHON) ck_test_runner.py --trace $(ARGS); \
	fi

# Run all tests
all:
	@echo "Running all tests..."
	@if [ -n "$(MODEL)" ]; then \
		$(PYTHON) ck_test_runner.py --all --model "$(MODEL)"; \
	else \
		$(PYTHON) ck_test_runner.py --all; \
	fi

# ============================================================================
# INDIVIDUAL TEST TARGETS
# ============================================================================

# Weight offset consistency check
test-weight-offsets:
	@echo "Checking weight offset consistency..."
	$(PYTHON) test_weight_offset_consistency.py --model-dir $(MODEL)

# Memory layout validation
test-memory:
	@echo "Validating memory layout..."
	$(PYTHON) test_memory_planner.py --layout=$(MODEL)/layout_decode.json

# Layer-by-layer comparison
test-layer-by-layer: ARGS ?=--token 25
test-layer-by-layer:
	@echo "Running layer-by-layer test..."
	$(PYTHON) test_layer_by_layer.py --model $(MODEL) $(ARGS)

# Numerical parity test
test-numerical-parity:
	@echo "Running numerical parity test..."
	$(PYTHON) test_numerical_parity.py --model $(MODEL)

# Advanced memory validation
test-memory-deep:
	@echo "Running deep memory validation..."
	$(PYTHON) advanced_memory_validator.py --model $(MODEL) --deep

# ============================================================================
# BUILD INTEGRATION
# ============================================================================

# Build v6.6 model (calls parent Makefile)
build:
	@echo "Building v6.6 model via root Makefile..."
	@$(MAKE) --no-print-directory -C $(ROOT_DIR) v6.6

# Download model and build everything
bootstrap: download build-v66

download:
	@echo "Downloading model..."
	$(PYTHON) ../scripts/ck_run_v6_6.py run "$(DOWNLOAD_MODEL)" --inspect-only

build-v66:
	@echo "Building v6.6..."
	@$(MAKE) --no-print-directory -C $(ROOT_DIR) v6.6

# Full pipeline: download, build, test
full: bootstrap test

# Run tests after build
test: all

# ============================================================================
# TRACE TARGETS
# ============================================================================

# Trace divergence
trace-divergence:
	$(PYTHON) trace_divergence.py --token 25

# Trace NaN layer
trace-nan-layer:
	$(PYTHON) trace_nan_layer.py --token 25

# Trace MLP path
trace-mlp:
	$(PYTHON) trace_mlp_path.py --token 25

# Trace residual
trace-residual:
	$(PYTHON) trace_residual.py --token 25

# ============================================================================
# DEBUG TARGETS
# ============================================================================

# Comprehensive debug
debug-comprehensive: ARGS ?=
debug-comprehensive:
	$(PYTHON) v6_6_comprehensive_debug.py $(ARGS)

# Debug embedding
debug-embedding:
	$(PYTHON) debug_embedding_v66.py --model $(MODEL)

# Debug Q bias
debug-q-bias:
	$(PYTHON) debug_q_bias.py --model $(MODEL)

# Debug stop sequence
debug-stop-seq:
	$(PYTHON) debug_stop_seq_v65_v66.py --model $(MODEL)

# ============================================================================
# COMPARISON TARGETS
# ============================================================================

# Compare with llama.cpp
compare-llamacpp:
	$(PYTHON) compare_with_llamacpp.py --model $(MODEL)

# Compare v6.5 vs v6.6
compare-v65-v66:
	$(PYTHON) compare_v65_v66.py --model $(MODEL)

# Compare attention output
compare-attention:
	$(PYTHON) compare_attention_output.py --model $(MODEL)

# ============================================================================
# CLEANUP
# ============================================================================

clean:
	@echo "Cleaning test artifacts..."
	rm -rf __pycache__ .pytest_cache *.pyc
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."

# Clean all generated files
clean-all: clean
	@echo "Removing generated files..."
	rm -f *.so *.o *.bin
	@echo "Cleaned all."

# ============================================================================
# HELP
# ============================================================================

help:
	@echo ""
	@echo "=============================================="
	@echo "C-Kernel-Engine v6.6 Test Suite"
	@echo "=============================================="
	@echo ""
	@echo "Quick Targets:"
	@echo "  make quick/sanity    - Quick sanity check"
	@echo "  make parity          - Layer-by-layer parity"
	@echo "  make memory          - Memory validation"
	@echo "  make divergence      - Find v6.5/v6.6 divergence"
	@echo "  make nan             - Debug NaN issues"
	@echo "  make all             - Run all tests"
	@echo ""
	@echo "Individual Tests:"
	@echo "  make test-weight-offsets"
	@echo "  make test-memory"
	@echo "  make test-layer-by-layer"
	@echo "  make test-numerical-parity"
	@echo "  make test-memory-deep"
	@echo ""
	@echo "Tracing:"
	@echo "  make trace-divergence"
	@echo "  make trace-nan-layer"
	@echo "  make trace-mlp"
	@echo "  make trace-residual"
	@echo ""
	@echo "Debugging:"
	@echo "  make debug-comprehensive ARGS='--stop-at 5'"
	@echo "  make debug-embedding"
	@echo ""
	@echo "Comparison:"
	@echo "  make compare-llamacpp"
	@echo "  make compare-v65-v66"
	@echo "  make compare-attention"
	@echo ""
	@echo "Build Integration:"
	@echo "  make build           - Build v6.6"
	@echo "  make download        - Download model"
	@echo "  make full            - Download, build, test"
	@echo ""
	@echo "Options:"
	@echo "  MODEL=<path>         - Set model path (default: ~/.cache/ck-engine-v6.6/...)"
	@echo "  PYTHON=<interpreter> - Set Python (default: python3)"
	@echo ""
	@echo "Examples:"
	@echo "  make quick MODEL=/path/to/model"
	@echo "  make test-layer-by-layer ARGS='--token 25 --all-layers'"
	@echo "  make debug-comprehensive ARGS='--stop-at 5 --verbose'"
	@echo ""
	@echo "Documentation: see TEST_README.py"
	@echo ""
