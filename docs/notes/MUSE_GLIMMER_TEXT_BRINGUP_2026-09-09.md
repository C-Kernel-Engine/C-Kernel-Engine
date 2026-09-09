# Muse-Glimmer text bring-up (2026-09-09)

This note records the first bounded text bring-up for
`meta-models/Muse-Glimmer-30B`. The complete checkpoint converts, both
52-layer graphs generate and compile, and short prefill plus cached decode are
bit exact with the pinned PyTorch eager BF16 reference. The selected attention
provider is an eager parity implementation with quadratic prefill scratch; it
is not a practical long-context implementation.

## Frozen reference contract

- Model configuration and weights: Hugging Face branch `main-30B2`
- Reference implementation: Transformers commit
  `4177486a9f199bd7be520eff14431071d5d41ec5`
- Reference class: `MuseGlimmerForConditionalGeneration`
- Reference attention backend: `eager`
- Storage and activation dtype: BF16
- CPU reference setting: `OMP_NUM_THREADS=16`
- `config.json` SHA-256:
  `5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14`
- `model.safetensors.index.json` SHA-256:
  `7d817b4dccb1b123fc6c1939356c65cee3a0ad462a5b821ac88280990a27d1ba`
- `chat_template.jinja` SHA-256:
  `cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678`

The Ryzen validation host used Transformers 5.15.1, PyTorch 2.13.0+cpu,
oneDNN 3.12, and the SLEEF symbols supplied by that PyTorch build. The full
machine-readable result is in
`docs/notes/artifacts/muse_glimmer_text_parity_128_2026-09-09.json`.

## Implemented contract

- A `muse_glimmer_text` circuit with independent hidden, query, KV, gate,
  and MLP dimensions.
- Three sliding-attention layers followed by one full-attention layer, with
  RoPE enabled only for sliding layers.
- Centered weighted RMSNorm, weighted post-branch RMSNorm, unweighted Q/K
  normalization, query scaling, attention gating, four normalization sites,
  output scaling, and final tanh softcapping.
- Explicit BF16 rounding boundaries for normalization, Q/K scaling, RoPE,
  projections, attention, SwiGLU, residuals, and logits.
- Separate embedding and output-head tensors.
- Strict tensor-family checks and explicit deferral of every vision tensor.
- Caller-owned attention and projection tensor scratch, declared in kernel
  maps and bound through generated call ABIs.
- An explicit aggregate workspace budget enforced during lowering.

The published checkpoint contains 1,436 tensors. Conversion accounts for all
of them: 627 are consumed by the text circuit and 809 vision tensors are
explicitly marked as deferred. There are no unexplained leftovers.

The four Muse attention maps describe their current implementation rather
than broader generic capabilities: AVX-512, oneDNN 3.12, SLEEF, serial CKE
head traversal, and external oneDNN threading. Their selection status remains
`candidate`.

## Numerical X-ray

The complete 59.55 GB checkpoint converted to a 55.5 GiB CKE weight artifact.
Both full prefill and cached decode run on the Ryzen host.

The first multi-token failure was in full-attention cache placement. Global
layers omit RoPE, and decoder lowering had stored K before Q/K normalization.
The corrected graph stores normalized K for these layers.

The remaining X-ray mismatch appeared in layer 1 Q/K normalization. The model
source spells the reciprocal factor as `torch.pow(value, -0.5)`, but PyTorch's
CPU unary dispatch uses its reciprocal-square-root kernel for this exponent.
Selecting the existing reciprocal-square-root path in the Muse weighted and
unweighted RMSNorm wrappers reproduced that dispatch. No new BF16
normalization kernel was required. A 69-token layer X-ray then remained exact
through layers 0 through 50; the end-to-end logit evidence below is the
certification result used for the complete graph.

The attention provider reproduces the reference sequence of BF16 QK matmul,
BF16 scale rounding, FP32 softmax, BF16 probability storage, and BF16 PV
matmul. Synthetic oracles cover multiple KV heads, sliding masks, and decode
where cache capacity exceeds live KV length. Null or undersized scratch and
workspace size overflow fail explicitly.

## End-to-end Ryzen certification

The certification runner separates two questions:

1. Forced-reference-token history compares every float32 logit by its raw
   IEEE-754 bit pattern.
2. Free-running history compares the actual CKE greedy trajectory and retains
   CKE's decoded output independently from the reference output.

It checks finite values, records first divergence, publishes partial failures,
and hashes the runtime libraries, generated C, runtime bundle, and reference
manifest. Three official-chat-template prompts ran for 128 generated tokens
each:

| Prompt | Prompt tokens | Exact logit rows | Greedy token divergence |
| --- | ---: | ---: | --- |
| Complete C function | 69 | 128 / 128 | None |
| Standalone SVG | 69 | 128 / 128 | None |
| CKE v8 architecture analysis | 96 | 128 / 128 | None |

All 384 logit rows were finite and bit exact. Because every free-running token
matched, the forced and free-running histories were identical in this run.
This is PyTorch parity; no llama.cpp Muse implementation was used as an oracle.

## Memory and allocation scope

The eager prefill workspace is
`6*T*C + 4*C*D + 2*T*D` bytes. Its score buffers alone use `6*T*C` bytes:
24 GiB at `T=C=64K` and 96 GiB at `T=C=128K`. Sliding attention currently
materializes this full matrix before masking. The maps therefore impose a
1 GiB call-workspace gate. A bounded-memory provider needs separate numerical
validation before the gate can be expanded.

The selected Muse maps contain no C heap allocation in their attention or
projection tensor workspaces. This is a narrower claim than allocation-free
execution. The projection helper still creates and destroys oneDNN
descriptors, primitives, and memory objects per invocation and executes under
a global mutex; oneDNN may allocate internally. Prepared objects and
user-owned library scratch should be investigated outside this correctness
bring-up, with allocation instrumentation around the full selected call chain.

The repository audit still reports 51 inherited production allocation sites.
Moving Muse projection selection to the workspace ABI reduces mapped
allocating providers without scratch contracts from three to two; the
allocation baseline was tightened accordingly and was not expanded for Muse.

## Certification state

| Gate | State |
| --- | --- |
| Configuration and tensor inventory | Pass |
| Synthetic conversion/lowering/generated C | Pass |
| Component numerical contracts | Pass |
| Short real-weight prefill | Pass, bit exact |
| Short cached decode | Pass, 384 / 384 logit rows bit exact |
| Decode stride with capacity greater than live length | Pass |
| Workspace capacity and overflow rejection | Pass |
| Repeated-call library allocation instrumentation | Not tested |
| 2,047 / 2,048 / 2,049-token boundaries | Not tested |
| 4K through 128K context | Blocked by eager workspace budget |
| Concurrent-session isolation | Not tested |
| Quantization | Not tested |
| Vision and image preprocessing | Deferred |

This establishes a short-context text correctness candidate. Promotion still
requires window-boundary and reset/reuse coverage, library-allocation
measurement, and a bounded-memory attention implementation for practical long
contexts. Vision remains a separate milestone.
