# Unlimited-OCR Generated-Runtime Plan

## Position In The Backlog

Unlimited-OCR is a vision-family candidate, not current CKE support. Begin its
implementation only after the existing Gemma4 and Qwen vision lanes complete
their bounded OCR and multi-token generated-runtime certification. In
particular, finish:

1. Gemma4 40-image OCR, independent encoder/bridge evidence, and multi-token
   decoder parity.
2. Qwen3-VL and Qwen3.6-VL shared-corpus refresh, followed by artifact-specific
   Qwen3.5/Qwen3.8 vision work where such artifacts and contracts exist.
3. Registration of those results in the existing nightly and corpus reports.

Audio optimization remains a separate workstream. Unlimited-OCR changes must
not be combined with Parakeet or Cohere Transcribe performance patches.

## Objective

Determine whether CKE can express and deploy Unlimited-OCR through its normal
standalone boundary:

```text
imported bundle + sidecars
  -> circuits and operation contracts
  -> resolved providers
  -> lowered IR and memory plan
  -> generated C
  -> native single-page and multipage execution
```

Python may perform import, reference capture, and scoring. It must not execute
the production vision graph, decoder, MoE routing, or cache progression.

## Feasibility Inventory

Pin the upstream revision and record source, configuration, tokenizer,
preprocessing, generation policy, and weight hashes. Review custom model code
before execution. Produce a machine-readable inventory with this relationship:

```text
operation -> mathematical contract -> CKE provider -> missing capability
          -> independent oracle case -> generated-runtime status
```

Inventory at least:

- SAM and CLIP visual components;
- resize, normalization, padding, crops, and page ordering;
- the 2048-to-1280 visual projection;
- global/local feature ordering and learned row/view separators;
- image-token assembly, positions, and text/image segmentation;
- ordinary decoder attention and its exact cache behavior;
- 64 routed experts, six selected experts, and two shared experts;
- routing weights, shared-expert combination, and accumulation order;
- repetition suppression, termination, and output-budget behavior;
- raw structured output and any scoring-only normalization.

Existing providers are reusable only when their layout, storage, reduction,
rounding, state, and workspace contracts match. Architecture-name similarity
is not evidence of compatibility.

## Required Cache Contract

Model the preserved prefill prefix and rolling decode region explicitly. Do
not represent the rolling region as ordinary sliding-window attention merely
because both involve a bounded token count.

Tests must cover:

- prefix integrity before and after rolling-region wraparound;
- absolute positions and masks across wraparound;
- segmented versus single-pass prefill where supported;
- session reset and repeated requests;
- capacity and overflow rejection;
- stale or incompatible cache state;
- deterministic continuation from identical state.

## First Executable Milestone

Start with base mode and one pinned image. Compare against the pinned reference
at these boundaries:

1. Preprocessed image tensors.
2. SAM and CLIP checkpoints.
3. Projected visual embeddings.
4. Separator insertion and assembled decoder input.
5. First-token logits.
6. A bounded greedy token trajectory.
7. Native decoded structured output.

Use existing X-Ray semantic checkpoints and adapters. Alignment, coverage, and
numerical verdicts remain separate. Missing or ambiguous alignment yields no
numerical verdict. X-Ray observes generated execution and must not become an
alternate model runtime.

PDF rasterization is an explicit input-adapter concern. First certify native
image or tensor input; a Python PDF helper cannot satisfy standalone model
execution.

## Pull Request Sequence

1. **Import and contracts**: pinned metadata, weight-free lowering fixtures,
   operation/provider gap report, and cache negative controls.
2. **Vision path**: generated visual encoders, projection, separators, and
   independent checkpoint comparisons.
3. **Decoder path**: generated attention, MoE, preserved-prefix/rolling cache,
   and bounded token trajectory.
4. **Single-page E2E**: standalone native execution with no Python, network,
   repository checkout, or undeclared cache.
5. **Crop and multipage**: page ordering, crop/base modes, repeated sessions,
   and bounded memory.
6. **Performance and formats**: profile the certified path before adding
   quantized or specialized providers.

No common lowering branch may select behavior from the family name. Missing
compiler support must be introduced as reusable semantics with explicit
contracts. Inference certification does not imply backward or training
support.

## OCR Certification

Reuse `MODEL_AGNOSTIC_VISION_CORPUS_CERTIFICATION.md` and the existing corpus
runner. Do not create another OCR harness. Cover plain text, columns, tables,
equations, code, small text, blank pages, and multipage reading order.

Report separately:

- conversion and generated execution;
- numerical checkpoint parity;
- raw OCR text and structured output;
- transcription accuracy, reading order, omissions, repetition, and
  truncation;
- latency, peak memory, and output-token count;
- single-page, crop-mode, and multipage scope.

Retain raw output beside normalized scoring output. Postprocessing cannot hide
model omissions or repetitions.

## Regression And Promotion Gates

| Tier | Required evidence |
| --- | --- |
| GitHub PR | New-provider oracle tests, tiny composite fixtures, real-manifest lowering/compile checks, and cache/alignment negative controls |
| GitHub nightly | Bounded generated execution, numerical cases, visualizer checks, and allocation contracts on eligible hosts |
| Idle P3/Ryzen | Pinned real-model single-page and representative shared-corpus cases |
| Extended certification | Multipage, longer outputs, repeated sessions, memory limits, and controlled performance trends |

Shared compiler or kernel changes must select all actual consumers, including
existing Gemma/Qwen vision and text MoE paths. Missing, stale, or blocked
evidence remains visible. One readable document cannot certify the family.

Unlimited-OCR may be called standalone supported only after a clean exported
bundle builds and executes with declared native dependencies and without
Python, the CKE development checkout, network access, or reference code.
