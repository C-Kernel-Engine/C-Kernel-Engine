# Makefile for v7 Test Suite
# Usage: make <target>
#
# This Makefile provides convenient targets for running tests and
# integrates with the parent Makefile for building.

.PHONY: all quick sanity parity memory debug nan trace clean help build download build-v7 bootstrap full test

# Optional model path (can override with MODEL=...).
# If empty, ck_test_runner auto-resolves from .ck_cache then ~/.cache.
MODEL ?=

# Python interpreter
PYTHON ?= $(if $(wildcard ../../.venv/bin/python),../../.venv/bin/python,python3)

# Directory wiring
TEST_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(TEST_DIR)/../../..)

# NOTE:
# - This test Makefile is invoked by root targets like `make v7-test-quick`.
# - Build/download targets below are convenience wrappers only and delegate to root Makefile.
# - The authoritative release pipeline is rooted in top-level `make v7-gate`.

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

# Find divergence in v7 decode flow
divergence:
	@echo "Finding divergence in v7 decode flow..."
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
	@echo "Weight-offset checker is folded into advanced memory validator in v7..."
	$(PYTHON) advanced_memory_validator.py --model $(MODEL)

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

# Build v7 model (calls parent Makefile)
build:
	@echo "Building v7 model via root Makefile..."
	@$(MAKE) --no-print-directory -C $(ROOT_DIR) v7

# Download model and build everything
bootstrap: download build-v7

download:
	@echo "Downloading model..."
	$(PYTHON) ../scripts/ck_run_v7.py run "$(DOWNLOAD_MODEL)" --inspect-only

build-v7:
	@echo "Building v7..."
	@$(MAKE) --no-print-directory -C $(ROOT_DIR) v7

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
	@echo "trace-nan-layer retired in v7; use trace-divergence"
	$(PYTHON) trace_divergence.py --token 25

# Trace MLP path
trace-mlp:
	@echo "trace-mlp retired in v7; use trace-divergence"
	$(PYTHON) trace_divergence.py --token 25

# Trace residual
trace-residual:
	@echo "trace-residual retired in v7; use trace-divergence"
	$(PYTHON) trace_divergence.py --token 25

# ============================================================================
# DEBUG TARGETS
# ============================================================================

# Comprehensive debug
debug-comprehensive: ARGS ?=
debug-comprehensive:
	$(PYTHON) v7_comprehensive_debug.py $(ARGS)

# Debug embedding
debug-embedding:
	@echo "debug-embedding retired in v7; use debug-comprehensive"
	$(PYTHON) v7_comprehensive_debug.py --model $(MODEL) --stop-at 1

# Debug Q bias
debug-q-bias:
	@echo "debug-q-bias retired in v7; use debug-comprehensive"
	$(PYTHON) v7_comprehensive_debug.py --model $(MODEL) --stop-at 3

# Debug stop sequence
debug-stop-seq:
	$(PYTHON) v7_comprehensive_debug.py --model $(MODEL)

# ============================================================================
# COMPARISON TARGETS
# ============================================================================

# Compare with llama.cpp
compare-llamacpp:
	@echo "compare-llamacpp retired in v7 test subset; use test-numerical-parity"
	$(PYTHON) test_numerical_parity.py --model $(MODEL)

# Compare v7 variants
compare-v7:
	1000 4 24 27 30 46 100 105 125 1000PYTHON) test_numerical_parity.py --model 1000 4 24 27 30 46 100 105 125 1000MODEL)

# Compare attention output
compare-attention:
	@echo "compare-attention retired in v7 test subset; use test-numerical-parity"
	$(PYTHON) test_numerical_parity.py --model $(MODEL)

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
	@echo "C-Kernel-Engine v7 Test Suite"
	@echo "=============================================="
	@echo ""
	@echo "Quick Targets:"
	@echo "  make quick/sanity    - Quick sanity check"
	@echo "  make parity          - Layer-by-layer parity"
	@echo "  make memory          - Memory validation"
	@echo "  make divergence      - Find v7 divergence"
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
	@echo "  make compare-v7"
	@echo "  make compare-attention"
	@echo ""
	@echo "Build Integration:"
	@echo "  make build           - Build v7"
	@echo "  make download        - Download model"
	@echo "  make full            - Download, build, test"
	@echo ""
	@echo "Options:"
	@echo "  MODEL=<path>         - Set model path (default: ~/.cache/ck-engine-v7/...)"
	@echo "  PYTHON=<interpreter> - Set Python (default: python3)"
	@echo ""
	@echo "Examples:"
	@echo "  make quick MODEL=/path/to/model"
	@echo "  make test-layer-by-layer ARGS='--token 25 --all-layers'"
	@echo "  make debug-comprehensive ARGS='--stop-at 5 --verbose'"
	@echo ""
	@echo "Documentation: see TEST_README.py"
	@echo ""
