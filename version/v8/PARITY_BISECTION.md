# v8 Canonical First-Divergence Attribution

Final-output similarity does not identify where two model executions first
diverge. A high encoder-prefix cosine can still change mixed-prefill rankings
and send greedy decoding down another token path. v8 therefore treats parity as
a sequence of semantic graph checkpoints rather than a collection of ad hoc
backend tensor names.

The implementation uses *bisection* in its ordinary debugging sense: compare a
small ordered set of graph boundaries, find the first failing interval, and
repeat inside only that interval. It does not alter model execution, guess a
kernel, or repair numerical behavior.

## Ownership Boundaries

- Circuits define producer/consumer topology and stable semantic checkpoint
  names.
- Kernel maps define exact numerical, reduction, and threading contracts.
- Parity profiles define backend pairings, comparison thresholds, and the
  sparse-to-granular bisection tree.
- Backend adapters export tensor manifests using the canonical tensor ABI.
- The bisection runner canonicalizes layouts, compares tensors, and reports the
  first divergent semantic edge.

Thresholds do not belong in circuits. Changing a test policy must not change
the model circuit.

## Tensor Manifest ABI

Each backend emits `cke.v8.parity_tensor_manifest` JSON. Every tensor records:

- semantic checkpoint name;
- producer operation and optional numerical contract;
- storage path and format;
- dtype before canonical conversion;
- physical shape and named axes;
- canonical axis order.

For example, CK may export `[head, token, channel]` while PyTorch exports
`[token, head, channel]`. Both manifests set canonical axes to
`[token, head, channel]`; the comparator transposes before computing metrics.
This prevents layout differences from being reported as numerical failures.

Missing artifacts are `unresolved`, never passing. Duplicate checkpoint names,
unknown schema fields, incompatible axes, element-count mismatches, and
canonical shape mismatches are hard faults.

## Bounded Bisection

A parity profile defines ordered groups. A sparse Qwen3-VL profile can start at
layers 0, 8, 16, 24, 26, and the final projector. A failure at layer 16 can
expand to layers 9-16. The first failing layer can then expand into norm, Q/K/V,
pre/post-RoPE, attention, projection, residual, MLP, and layer-output edges.

If granular tensors are absent, the report emits `next_request` containing only
the next group and checkpoint names. Exporters can rerun that bounded request
instead of dumping every tensor.

## RoPE Rule

RoPE and M-RoPE behavior must not be hardcoded in the first-divergence runner or
added as model-name branches in code generation. Circuits must request explicit
position-transform semantics; kernel maps must bind an exact implementation.
The contract must include pairing convention, rotary width, position rank and
axis layout, section interpretation, frequency/scaling precision, input/output
dtype, and threading/reduction guarantees. The runner only verifies that both
backend manifests name the same resolved contract before comparing tensors.

## Metrics

Every comparison reports:

- finite-value status;
- cosine similarity;
- RMSE and relative RMSE;
- mean and maximum absolute error;
- worst logical coordinate and both values.

The first sparse failure is retained as `first_observed_divergence`; after
expansion, `first_divergence` names the deepest available failing edge.

## Validation

```bash
make test-v8-parity-bisection

.venv/bin/python version/v8/scripts/parity_bisect_v8.py \
  --profile build/parity/profile.json \
  --reference-manifest build/parity/pytorch/manifest.json \
  --candidate-manifest build/parity/cke/manifest.json \
  --json-out build/parity/report.json \
  --summary
```

The initial tests cover physical-layout canonicalization, BF16 raw exports,
deepest-edge attribution, bounded follow-up requests, duplicate checkpoints,
and strict schema rejection. Backend adapters and mixed-prefill/teacher-forced
stages build on the same semantic names.
