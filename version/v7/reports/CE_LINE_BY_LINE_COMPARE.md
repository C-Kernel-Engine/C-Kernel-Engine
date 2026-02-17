# Cross-Entropy Line-by-Line Comparison (CK vs PyTorch vs llama.cpp)

## Snapshot
- CK source: `src/kernels/loss_kernels.c`
- PyTorch source (2.10.0+cpu, commit `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`):
  - `aten/src/ATen/native/LossNLL.cpp`
  - `aten/src/ATen/native/SoftMax.cpp`
- llama.cpp source (master snapshot):
  - `ggml/src/ggml-cpu/ops.cpp`
  - `ggml/src/ggml.c`
  - `ggml/include/ggml.h`

> Note: line references below use upstream repository-relative paths (not local `/tmp` mirrors).

## Core Mapping

1. Target contract
- CK: class-index targets (`int32_t *targets`) per row.  
  Ref: `src/kernels/loss_kernels.c:22`, `src/kernels/loss_kernels.c:42`
- PyTorch: supports index targets and probability targets (same-shape path).  
  Ref: `aten/src/ATen/native/LossNLL.cpp:636`, `aten/src/ATen/native/LossNLL.cpp:645`
- llama.cpp (ggml): requires logits and labels tensors with same shape (distribution labels).  
  Ref: `ggml/src/ggml.c:5992`

2. Numerically stable softmax/log-softmax structure
- CK: max-subtract + exp + sum + log-sum-exp in strict/fast branches.  
  Ref: `src/kernels/loss_kernels.c:51`, `src/kernels/loss_kernels.c:70`, `src/kernels/loss_kernels.c:84`, `src/kernels/loss_kernels.c:99`
- PyTorch: CE index path calls `log_softmax(...)` then NLL loss.  
  Ref: `aten/src/ATen/native/LossNLL.cpp:655`, `aten/src/ATen/native/LossNLL.cpp:656`
- PyTorch softmax kernel uses max-subtract + exp + sum.  
  Ref: `aten/src/ATen/native/SoftMax.cpp:206`, `aten/src/ATen/native/SoftMax.cpp:223`, `aten/src/ATen/native/SoftMax.cpp:234`
- llama.cpp: forward CE computes log-softmax then weighted sum with labels.  
  Ref: `ggml/src/ggml-cpu/ops.cpp:10707`, `ggml/src/ggml-cpu/ops.cpp:10710`

3. Backward formula
- CK: writes gradient as softmax mean-scaled then subtracts target one-hot mean term.  
  Ref: `src/kernels/loss_kernels.c:65`, `src/kernels/loss_kernels.c:73`, `src/kernels/loss_kernels.c:102`, `src/kernels/loss_kernels.c:106`
- PyTorch: from `log_softmax + nll_loss`, autograd equivalent to `(p - one_hot)/N` for index targets (plus ignore/weight semantics).
- llama.cpp: explicit comment and implementation of `(softmax - labels) * grad / nr`.  
  Ref: `ggml/src/ggml-cpu/ops.cpp:10804`, `ggml/src/ggml-cpu/ops.cpp:10805`

4. Reduction semantics
- CK: always divides final loss by `tokens`; no `ignore_index`.
  Ref: `src/kernels/loss_kernels.c:36`, `src/kernels/loss_kernels.c:112`
- PyTorch: reduction handling includes `mean/sum/none`, `ignore_index`, class weights, and label smoothing branches.  
  Ref: `aten/src/ATen/native/LossNLL.cpp:530`, `aten/src/ATen/native/LossNLL.cpp:592`, `aten/src/ATen/native/LossNLL.cpp:646`
- llama.cpp: scalar CE reduced by `-1/nr`; no ignore-index branch in ggml CE op.  
  Ref: `ggml/src/ggml-cpu/ops.cpp:10730`

5. Precision and accumulation
- CK strict path uses `double` for `sum_exp`/loss accumulation and `exp`/`log`; fast path uses `expf` with mixed accumulation.
  Ref: `src/kernels/loss_kernels.c:59`, `src/kernels/loss_kernels.c:70`, `src/kernels/loss_kernels.c:84`, `src/kernels/loss_kernels.c:86`
- PyTorch uses native kernels (`log_softmax`/`nll_loss`) with type-dependent accumulation in backend kernels.
- llama.cpp CE kernels shown here are `f32` implementations for forward/backward.
  Ref: `ggml/src/ggml-cpu/ops.cpp:10658`, `ggml/src/ggml-cpu/ops.cpp:10754`

6. Optimizer context (important for drift analysis)
- CK parity harness (`train_parity_epochs_v7.py`) compares branches under the same PyTorch optimizer.
- CK generated runtime path can run C AdamW.
  Ref: `version/v7/reports/generated_train_runtime_v7.c:946`
- llama.cpp ggml has built-in AdamW op (`ggml_opt_step_adamw`).
  Ref: `ggml/src/ggml.c:6025`

## Practical Implication for v7 Drift
- CK CE math is structurally aligned with standard CE (`p - one_hot` for index targets).
- Largest semantic gaps vs PyTorch are around reduction/feature semantics (`ignore_index`, weight handling, smoothing branches) and low-level numeric kernel details.
- llama.cpp CE in ggml is closer to probability-target CE API (same-shape logits/labels) than PyTorch index-target CE API.
