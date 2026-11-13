# Registered FP32 Clef head training

This opt-in recipe trains the complete supported Clef head on **supplied backbone
features**. It does not execute or train a backbone, tokenize text, add adapters,
or execute application actions. A successful job means generated training,
frozen embedding checks, checkpoint reload and generated inference export passed.
It does not certify the submitted dataset's quality, calibrated confidence or
numerical parity against an oracle. Those report fields remain `NOT_TESTED`.

## Ownership and precision

The existing v8 circuit authoring, backward rules and shared training code generator
produce native forward, backward and AdamW execution. Python validates records,
loads input features, invokes complete generated steps and packages artifacts.
It never computes model gradients or updates parameters. No new compiler branch,
model scheduler or training stack is introduced.

The admitted recipe is `clef_head_fp32_v1`: FP32, AdamW, per-question cross entropy,
1–40 epochs, learning rate `(0, 0.01]`, epsilon `1e-4`, betas `(0.9, 0.999)`, weight
decay `0.01`, no dropout, gradient accumulation 1. All head parameters train;
output embedding rows and supplied backbone features are frozen. Mixed BF16,
LoRA and QLoRA are unavailable through this recipe. In particular, the existing
mixed-BF16 trajectory failure has not become a pass.

## Dataset/package contract

`cke.clef.training-package.v1` has exactly five fields:

- `schema`: the version above.
- `model`: `id`, `revision`, `backbone_id`, `backbone_revision`,
  `tokenizer_revision`, and `feature_provenance: "supplied"`. Revisions are pinned
  hexadecimal identities. Supplied provenance records a declaration, not proof
  that CKE executed that backbone.
- `geometry`: `tokens`, `channels`, `width`, `heads`, `feedforward`, `vocabulary`,
  `questions`, `options`. All records share this declared geometry.
- `weights`: local basename `file` and `sha256` of the FP32 NPZ raw head weights,
  including `output_embedding.weight` and packed QKV weights.
- `records`: 2–32 records with `id` (at most 60 characters), `split` (`train` or
  `validation`), `asset` (`file`, `sha256`), `question_ids` and grouped
  `candidate_ids`. Both splits must be present. IDs preserve question/option order.

Each record NPZ contains `hidden` (FP32) and int32 `input_ids`, `question_spans`,
`option_spans`, `global_span`, `option_counts`, `question_type_ids`, `labels`.
Spans are half-open. Global span must select the last valid token. Labels index
that question's allowed options; option counts partition flattened option rows.
Arrays, finite values, IDs, counts, shapes, archive expansion and all asset hashes
are checked before code generation. Archives must not contain Python objects.
See `version/v8/decision/training_package.py` for exact bounds and inventories.
The tests create synthetic, nonzero examples with overlapping spans and two
questions with unequal option counts. Do not use private customer data for the
first client demonstration.

## Run locally

Build `build/libckernel_engine.so` first. Use a trusted package and retain its hash:

```sh
python version/v8/decision/train_clef_head.py \
  --package /absolute/path/package.json --package-sha256 SHA256 \
  --run-dir /private/new-run --json-out /private/result.json --epochs 2
```

The output contains native runtime/source identities, held-out loss before/after,
per-record raw logits and identities, an identity-bound optimizer checkpoint,
`trained_head.npz`, and `serving.zip`. The latter contains the ordinary generated
head package and registered supplied feature artifacts. The recipe loads it with
`NativeHead` and compares its logits with the trained runtime for every record.

For a deliberate CLI checkpoint, add `--stop-after 2`; continue with the same
package, configuration and run directory using `--resume /private/new-run/checkpoint-2`.
This restores weights, AdamW state and step count. The saved runtime and engine
identities must match. Completed checkpoints cannot be resumed as new training.
This is local CLI resume, not a job-service resume or cancellation promise.

## Independent client and job service

The existing `cke.training.v1` discovery/validation API remains non-executing.
An administrator can opt into the head recipe by registering a package mapping:

```json
{"demo":{"path":"/absolute/path/package.json","sha256":"SHA256"}}
```

Launch trusted local stdio, from a clean committed source tree:

```sh
python -m server.training.jobs --store /private/jobs --principal demo \
  --clef-registry /private/registry.json
```

Use discovery before constructing a request. A head configuration is:

```json
{"recipe":"clef_head_fp32_v1","package_id":"demo","package_sha256":"SHA256",
 "dtype":"fp32","optimizer":"adamw","method":"head_only",
 "epochs":2,"learning_rate":0.001}
```

`prepare` binds configuration, worker boot, capability/source/interpreter/package
identities and expiration. `submit` returns a durable acknowledgement with an
idempotency key. Only one job executes, with no backlog; discovery and preparation
do not execute training. `observe` reports durable monotonic progress and metrics.
Head metrics use `step_kind: update` and `unit: nats/question`; they must not be
interpreted as decoder token losses. Artifact retrieval uses opaque IDs, bounded
chunks and hashes. The independent clients in `server/training/consumer.py` and
`job_consumer.py` do not import the worker implementation.

## Readiness and gaps

| Boundary | Status |
| --- | --- |
| Supplied-feature FP32 generated head training and inference export | Implemented; synthetic native regression |
| Independent small-head PyTorch comparison | Optional pinned-source test; explicit skip when unavailable |
| Frozen output embeddings and fresh-process CLI resume | Native regression |
| Held-out loss | Reported; no automatic quality acceptance |
| Backbone execution, text-to-decision | Not provided by this recipe |
| Job cancellation / automatic restart / remote resume | Unsupported; interrupted jobs become `recovery_required` |
| Remote authenticated transport | Unsupported; trusted local stdio only |
| BF16 trajectory certification, LoRA, QLoRA | Not admitted |

Do not expose this interface as an authenticated remote service. A caller-provided
principal is meaningful only inside a trusted local supervisor. A tunnel is a
development bridge, not authorization. Evaluation continues to use the separate
`cke.decision.evaluate.v1` contract; neither evaluation nor control discovery is
a training command. Keep the current decision HTTP worker loopback-only.

The service exports checkpoint metadata and serving artifacts, but remote
checkpoint-state transfer/resume is not yet an application contract. Applications
must surface `recovery_required`, `NOT_TESTED` and unsupported operations explicitly.

## Validation retained for this change

On the local Intel i7-14700T host with PyTorch `2.12.1+cpu` and one requested
OpenMP thread:

- Focused recipe and existing service/client suites: 31 passed, 2 explicit skips
  (opt-in real SVG and Clef jobs).
- Real registered Clef job rerun separately from the clean committed tree: passed,
  including independent observation and chunked artifact hash verification.
- `training_clef_registered_recipe` through the standard nightly runner: passed.
- Existing independent FP32 complete-head 40-update trajectory: passed. Loss
  0.28329045 → 0.06633431; max absolute gradient error `1.374632e-6`, weight error
  `1.668930e-6`, optimizer-moment error `3.147870e-7`. Checkpoint/resume and
  generated inference export passed. These features and head geometry are
  synthetic; this run does not certify released-geometry training or task quality.

Reproduce the new boundary with:

```sh
OMP_NUM_THREADS=1 CKE_CLEF_FLASH_REFERENCE_SOURCE=/path/to/pinned/joint_schema_model.py \
  python -m unittest tests.test_v8_clef_training_recipe \
  server.tests.test_training_contract server.tests.test_training_jobs -v
OMP_NUM_THREADS=1 CKE_RUN_CLEF_JOB=1 python -m unittest \
  tests.test_v8_clef_training_recipe.NativeRecipe.test_registered_job_and_independent_artifact_client -v
python scripts/nightly_runner.py --tests training_clef_registered_recipe --require-executed
```

The oracle source must match the SHA-256 in
`tests/fixtures/decision/clef_flash/identity.json`; no source means explicit skip.
The production resume test blocks imports of `torch` and `transformers`.
