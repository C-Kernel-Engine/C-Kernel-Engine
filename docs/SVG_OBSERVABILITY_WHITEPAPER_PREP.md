# SVG Training Observability White Paper (Prep Draft)

Status: preparation scaffold (not final paper)
Owner: C-Kernel-Engine operator
Scope: v7 SVG training pipeline, parity gates, and run-dir observability

## 1) Working Title

Controlled SVG Domain as a Testbed for End-to-End Training Observability in a C-First LLM Runtime

## 2) Core Thesis

A constrained SVG corpus can be used as a high-signal domain to validate training-system correctness and operator visibility because:
- syntax is deterministic and machine-gated,
- output quality is human-visible,
- dataset/tokenizer/training artifacts can be tracked in one run-dir,
- parity and replay gates can be tied to concrete artifacts.

This is an engineering observability paper, not a frontier-model benchmark paper.

## 3) Non-Goals

- No claim of broad-domain SOTA capability.
- No claim that SVG-only pretraining generalizes to all text/code domains.
- No claim that current model size/context is production-grade for all SVG tasks.

## 4) Candidate Contributions

1. Run-dir as source-of-truth for training observability.
2. Gate stack spanning:
   - dataset quality gates,
   - tokenizer roundtrip gates,
   - parity/replay/runtime gates,
   - output-quality SVG gates.
3. Operator-centric IR visualizer with stage flow and artifact provenance.
4. Reproducible curriculum progression:
   - pretrain (stage_a),
   - midtrain (stage_b),
   - sft bridge (instruction+svg mix).

## 5) Claims -> Evidence Matrix

## Claim A
The training data path is controlled and auditable.
- Evidence artifacts:
  - `dataset_qc.json`
  - `dataset_profile.json`
  - `training_pipeline_latest.json` (`dataset_catalog`, `stage_timeline`, `data_provenance`)
- Required checks:
  - `require_ascii=true` (when using `ascii_bpe`)
  - `ascii_violations=0`
  - `require_svg_rows=true` and svg gate pass

## Claim B
Tokenizer fidelity is measurable and non-lossy on the training dataset.
- Evidence artifacts:
  - `tokenizer_roundtrip.json`
  - `tokenizer.json` / `tokenizer_bin`
- Required checks:
  - `exact_match=true`
  - `line_match_rate=1.0` (or thresholded policy with justification)

## Claim C
Training runtime behavior is reproducible and parity-audited.
- Evidence artifacts:
  - `training_loss_curve_latest.json`
  - `training_parity_latest.json`
  - `training_step_profile_latest.json`
  - optional strict gates: regimen/replay reports
- Required checks:
  - monotonic trend in moving-average loss (allow local oscillation)
  - parity/replay gates documented as pass/fail with thresholds

## Claim D
Output quality is measurable in-domain.
- Evidence artifacts:
  - `post_train_eval.json`
- Required checks:
  - `valid_svg_rate`
  - `closure_success_rate`
  - `repetition_loop_score`

## 6) Minimal Figure List

1. Stage flow diagram (pretrain -> midtrain -> sft).
2. Loss + grad norm + LR chart panel.
3. Tokenizer/Data Lab panel showing corpus coverage and tok-gap warnings.
4. Example prompt/output SVG pairs (good, partial, failure).
5. Decision-strip style gate summary (runbook vs strict parity vs profiler evidence).

## 7) Dataset and Tokenizer Narrative

Required narrative items:
- Corpus sources (repo SVG assets + generated synthetic rows + instruction pairs).
- Why tokenizer must be locked across continuation runs.
- Why tok-gap warnings appear when stage dataset is outside tokenizer corpus.
- Difference between:
  - exploratory continuation (reused tokenizer),
  - production refresh (tokenizer rebuilt on full final corpus).

## 8) Method Section Skeleton

1. Runtime and model config
   - layers, d_model, hidden, heads, vocab, context
   - optimizer and training hyperparameters
2. Curriculum
   - stage_a objective
   - stage_b objective
   - sft bridge objective
3. Gates and thresholds
   - data gates
   - tokenizer roundtrip gates
   - parity/replay/runtime gates
   - output-quality gates
4. Evaluation protocol
   - automated metrics
   - human visual inspection rubric

## 9) Risks and Limitations

- Small model size may memorize local templates without robust composition planning.
- Context-length mismatch can cap output completeness.
- Instruction quality depends on tokenizer coverage and stage-mix schedule.
- Passing parity does not imply high task quality; both are needed.

## 10) Reproducibility Checklist

For each reported run, record:
- run dir path
- git commit hash
- command used
- dataset path + sha256
- tokenizer sha256
- training seed
- key metrics (first/final loss, roundtrip rate, valid_svg_rate, closure rate, loop score)
- strict gate statuses

Template row:

| run | commit | dataset | tokenizer_sha | seed | loss_start | loss_end | roundtrip | valid_svg | closure | loop_score | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `...` | `...` | `...` | `...` | 42 | ... | ... | ... | ... | ... | ... | ... |

## 11) Writing Plan (Incremental)

Phase 1 (now):
- lock paper thesis and non-goals
- gather evidence artifacts from latest stable run
- draft figures and captions

Phase 2:
- draft methods + results sections
- add failure analysis and remediation examples

Phase 3:
- finalize discussion, limitations, and future work
- consistency pass on thresholds and terminology

## 12) Immediate Next Actions

1. Capture one complete run snapshot with all key artifacts present.
2. Export 5-10 representative prompt/output SVG examples.
3. Freeze threshold policy used for the reported run.
4. Start first draft in a separate document from this prep file.

