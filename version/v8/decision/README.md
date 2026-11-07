# Clef-Flash decision-head bring-up

Model topology lives in the circuit builders here. Kernels implement arithmetic;
maps describe ABI, scratch and numerical contracts. Generic operation catalogue
entries do not inspect the model name.

## Current executed boundary

`build_clef_flash_option_summary_circuit.py` extends the canonical ingress and
both evidence-routing layers through grouped option pooling and
`option_summary_norm`. Its single generated entry executes 43 operations.
Question option counts partition consecutive routed option rows. Each question's
projected vector supplies its pooling query. Normalization declares epsilon
`1e-5` locally, independently of the enclosing model configuration.

The grouped provider validates positive counts and exact coverage, finite valid
inputs, physical strides, capacities, alignment and disjoint writable buffers.
Caller-owned scratch stages the result, so rejection does not publish partial
output. Read-only buffers may alias each other. Padding is neither consumed nor
written. Arithmetic follows the registered ordered FP64 reductions with FP32
intermediate storage; no backward capability is advertised.

## Reproduction

```sh
python3 version/v8/decision/build_clef_flash_option_summary_circuit.py --check
python3 -m unittest tests.test_v8_grouped_weighted_rows tests.test_v8_clef_flash_option_summary -v
python3 scripts/nightly_runner.py --tests decision_grouped_weighted_rows decision_clef_flash_option_summary --ci --require-executed
```

Use a Python environment with NumPy; independent oracle tests additionally need
PyTorch. The released-head comparison also needs safetensors and the dependencies
imported by the pinned reference source. Enable that comparison with:

```sh
export CKE_CLEF_FLASH_HEAD_PATH=/path/to/joint_head.safetensors
export CKE_CLEF_FLASH_REFERENCE_SOURCE=/path/to/joint_schema_model.py
```

Both assets are hash-checked against `tests/fixtures/decision/clef_flash/identity.json`.
Missing oracle packages or assets are explicit test skips. The reference test
runs the pinned head's actual forward method and captures its summary
normalization boundary; it does not inject reference evidence outputs into CKE.
The host supplies synthetic hidden states, synthetic output-embedding rows,
IDs and spans. Both implementations use the same inputs and released head weights.
This is **not** a generated-backbone or complete-decision test.

On 2026-11-07, PyTorch 2.8.0 CPU comparisons passed for six tokens with option
counts `[2,1]` and eight tokens with `[1,2]`. Maximum absolute errors were
`1.90735e-6` / `1.43051e-6` for raw summaries and `1.31130e-6` / `1.19209e-6`
for normalized summaries; normalized RMSE was `2.63926e-7` / `3.13428e-7`.
These are fixed compiled geometries, not proof of a dynamic serving interface.
The declared pinned-prefix absolute regression ceiling is `1e-5`.

Synthetic tests also cover reordered options within groups, A→B→A reuse,
invalid counts before parity, undersized arenas, and rejection followed by
successful execution. Model-neutral tests compare explicit FP64/FP32-stage
arithmetic at O0/O3, including singleton groups, unequal groups and padded strides.

## Remaining completion gates

1. Assemble fields from the summary, question/global projections and type embeddings.
2. Connect the four question-interaction decoder layers, final normalization,
   prior and joint scoring, and raw question/option logits.
3. Compare the complete head with the pinned implementation, then connect the
   actual generated backbone with the correct output embeddings.
4. Extend the existing v8 training workflow: audit backward providers, add only
   missing contracts, freeze backbone parameters, compare gradients and AdamW
   updates, then verify checkpoint/resume and inference export.

The current head prefix establishes forward composition. It does not establish
complete Clef decisions, head training, adapters, or backbone fine-tuning.
