# Version 8

`version/v8` is the active inference lane for text and multimodal bring-up.

Current scope:
- keep the inference runner, visualizer, run hub, and regression surface versioned as `v8`
- isolate circuit and multimodal bridge evolution inside `version/v8`
- preserve stable operator contracts while new vision ops land
- expose a native `v8` operator surface for text and multimodal bring-up

What is included here right now:
- `scripts/ck_run_v8.py`
- `scripts/cks-v8-run`
- `scripts/build_ir_v8.py`
- `scripts/memory_planner_v8.py`
- `scripts/resolve_model_dir_v8.py`
- `tools/open_ir_visualizer_v8.py`
- `tools/open_ir_hub_v8.py`
- `tools/ir_visualizer.html`
- `circuits/*`
- `contracts/*`
- `schemas/*`
- `kernel_maps/*`

The active compiler input is:

```text
weights + circuits + kernel maps -> deterministic DSL lowering -> generated C
```

Circuits declare graph structure and required semantics. Kernel maps advertise
complete numerical and execution capabilities, including threadpool partition
policies. `build_ir_v8.py` resolves those requirements before GraphIR is built.
GraphIR, LoweredIR, and call-ready IR carry the same provider and contract IDs;
later stages hard-fail if they differ. See `CONTRACT_POLICY.md` before changing a
contract failure: fallbacks, silent defaults, tolerance relaxation, and bypass
flags are forbidden.

Every versioned kernel execution capability is also recorded as
`resolved_execution` in GraphIR and preserved through call-ready IR. This is
operator-generic: the current maps cover attention plus the hot Q4/Q6 GEMM and
GEMV routes, including threadpool runtime, partition, dispatch, and reduction-
order effects.

Q4_K/Q6_K linear maps additionally bind an exact public function, threadpool
entry point, scalar reference function, adapter, fixed comparison tolerance,
and external-oracle status. The compiler validates and carries these bindings;
it does not choose a different function from shape or performance heuristics.

Canonical text bring-up examples:
- `version/v8/scripts/cks-v8-run run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --context-len 1024 --force-compile --force-convert --chat-template=auto --generate-visualizer`
- `version/v8/scripts/cks-v8-run run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf --context-len 1024 --force-compile --force-convert --generate-visualizer`
- `version/v8/scripts/cks-v8-run run hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --context-len 1024 --force-compile --force-convert --generate-visualizer`
- `python3 version/v8/scripts/ck_run_v8.py run hf://unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --force-convert --force-compile --context-len 1034`
- `version/v8/scripts/cks-v8-run run hf://mradermacher/Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf --context-len 1024 --force-compile --force-convert --chat-template auto --generate-visualizer`

Notes:
- `Gemma 3`: use `--chat-template auto` for the instruction/chat path. `--chat-template none` is raw continuation mode now and requires `--allow-raw-prompt` if you intentionally want it.
- `NaanBeige`: if the first reply echoes `<|im_start|>assistant` or starts with `<think>`, keep `--chat-template auto`, do not force `none`, and treat it as a prompt-wrapper/chat-contract symptom rather than a stable expected reply shape.

Canonical vision bring-up example:
- `version/v8/scripts/cks-v8-run run hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf --mmproj hf://Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf --image-path version/v8/test_assets/v8_vision_doc_card_72.ppm --prompt "Explain this image."`

Notes:
- For the validated Qwen3-VL 8B path, omitting `--mmproj` now auto-resolves the matching HF companion projector.
- The `.ppm` regression image is a Portable Pixmap file. CK uses it in CI because the format is simple enough to parse directly: a short header plus raw RGB pixels. That removes PNG/JPEG decoder dependency differences from the vision smoke. It is a test-fixture format, not a requirement for normal use; user-facing runs can still use PNG/JPEG when the local image stack is available.
- Qwen3-VL is an 8B multimodal lane. `make v8-regression-fast` stays text-family focused, while `make test-v8-qwen3vl-e2e-smoke` runs the cached vision E2E path when the decoder and mmproj artifacts are present.

Optional LIKWID profiling:

```text
make profile-v8-likwid \
  V8_MODEL="/path/to/model-or-run-dir" \
  V8_PERF_RUNTIME=cli \
  V8_LIKWID_GROUPS=auto \
  V8_LIKWID_THREADS=1
```

The target detects the counter groups supported by the current processor,
selects up to two portable groups by default, pins the workload to CPUs from the
current affinity set, preserves LIKWID CSV/stdout/stderr, and writes
`likwid_summary.json` for the IR visualizer Profile section. Override
`V8_LIKWID_GROUPS` with a comma-separated list, `V8_LIKWID_MAX_GROUPS` to run
more groups, or `V8_LIKWID_CPUS` with an explicit CPU list. Each group reruns the
workload, so keep the default small when profiling large models.

The wrapper records the CKE workload's exit status separately from LIKWID's
exit status and requires the CLI's model-loading and prefill output before it
reports `pass`. A missing status marker, failed workload, or usage-only CLI run
fails closed instead of accepting counters collected around startup or help.

The v8 IR Hub indexes the same summary and exposes its pass/skip/fail status,
selected groups, pinned CPUs, common normalized metrics, and links to the JSON
and preserved raw artifacts directly on each run card.

LIKWID is optional. If `likwid-perfctr` is unavailable, the Make target reports
`SKIP` and the normal build, runtime, and visualizer paths remain unchanged.
Wrapper measurements include all activity on the pinned CPUs; avoid noisy
neighboring workloads when collecting evidence. Marker API regions can be added
later without changing this artifact contract.

Install the current stable release on Ubuntu (recommended for Xeon 6, Zen 5,
and other recent processors):

```bash
sudo apt-get update
sudo apt-get install -y build-essential git perl gnuplot-nox
git clone --depth 1 --branch v5.5.1 https://github.com/RRZE-HPC/likwid.git
cd likwid
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

The `gnuplot-nox` dependency enables headless SVG/PNG generation. The regular
`gnuplot` desktop package can be used instead when interactive windows are
wanted.

For a quick setup on an older supported processor, Ubuntu also packages LIKWID:

```bash
sudo apt-get update
sudo apt-get install -y likwid
```

Ubuntu LTS repositories may contain an older LIKWID release. If LIKWID does not
recognize the processor or exposes few useful groups, install the current source
release above. On CachyOS, Arch, EndeavourOS, or Manjaro, build the current AUR
package:

```bash
sudo pacman -S --needed base-devel git perl gnuplot
git clone https://aur.archlinux.org/likwid.git
cd likwid
makepkg -si
```

Verify access and discover the groups supported by the actual processor instead
of assuming that an event name exists:

```bash
likwid-perfctr -v
sudo modprobe msr
likwid-perfctr -i
likwid-perfctr -a
likwid-topology -c
```

`likwid-perfscope` is a live plotting frontend and does not automatically leave
a durable image for every run. Gnuplot/feedgnuplot can export SVG or PNG. Once a
plot has been exported, preserve and embed it in the generated IR report with:

```bash
make profile-v8-likwid \
  V8_MODEL="/path/to/model-or-run-dir" \
  V8_LIKWID_PLOT="/path/to/exported-likwid-plot.svg"
```

The profiling script also accepts repeatable `--plot-artifact` arguments when
called directly. SVG, PNG, JPEG, and WebP files are copied under the run's
`likwid/` directory, recorded in `likwid_summary.json`, and embedded in the
Profile page so the standalone report does not depend on the original path.

The Profile page visualizes the evidence flow from the CKE workload through its
pinned CPUs and dynamically selected groups to normalized metric cards. The
audit table and raw CSV/stdout/stderr links remain available beneath it. Metrics
with different units are deliberately not placed on one comparative bar chart.
Automated headless perfscope timeline export can be added later without changing
the normalized summary or plot-artifact contract.

That keeps `v8` small and honest: the version split now includes the inference runner, local kernel registry/maps, multimodal bridge entrypoint, and the `v8`-named operator tooling surface used by the visualizer, hub, and regression entrypoints.

### Pinned Bison BF16 decision replay

Use the checked-in packed two-question fixture and the pinned Bison safetensors.
The host validates the request, prepares the embedding rows and weights, and
loads one generated C entry. That entry executes the packed backbone and returns
raw option scores. The replay uses a temporary file-backed arena. Allow enough
free disk space for the reported bundle size in addition to the checkpoint.

```bash
request=tests/fixtures/decision/block_packed_bison_request.json
python -m version.v8.decision.capture_bison_block_oracle \
  --checkpoint "$checkpoint" --request "$request" --out "$oracle"
CK_BLAS_LIBRARY="$lp64_blas_library" MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  python -m version.v8.decision.run_bison_backbone \
  --checkpoint "$checkpoint" --request "$request" --output "$run" \
  --oracle "$oracle" --max-abs 0.5 --max-rmse 0.25
```

Use an LP64 BLAS library exporting `dgemm_`. `--scalar` selects the slower
ordered FP64 reference provider. `--layers 1` runs only the first layer and saves
its hidden output for diagnosis. `--keep-arena` retains the runtime arena for
replay. Use a fresh output path for each run. The comparison checks raw scores
and winner identities. The harness expects pretokenized requests.

The runner also accepts Groundhog-1 at the revision declared by
`tests/fixtures/decision/block_packed_groundhog_request.json`. Set `request`
to that fixture and `checkpoint` to the pinned Groundhog-1 checkpoint. The
generated graph includes both MLP branches and the routed GeGLU experts. For
diagnosis, the checked MoE fragment retains the router logits, selected IDs,
scaled routing weights and unweighted expert outputs in the replay report.
Equal-logit cutoff ties can select different experts in the publisher's
PyTorch implementation. Compare selected IDs together with weights and inspect
ties before attributing a residual mismatch to expert arithmetic.

For Groundhog's BF16 score comparison, add `--bf16-cblas --partition-topk`
to the replay command. This profile selects compact FP32-accumulating BF16
CBLAS for row projections and score-only partition routing for cutoff ties.
The LP64 library must export `cblas_gemm_bf16bf16f32`. Both dense and
expert projections use the selected compact BF16 CBLAS provider. The pinned acceptance gate uses the
same 0.5 maximum error, 0.25 RMSE and matching winners as Bison. The default
FP64 profile remains available for numerical diagnosis.

### Sigma BF16 denoiser block replay

The generated entry uses explicit positions and key ranges for a nonempty
block within the pinned 4096-token context. The block can read clean prefix
K/V as separate BF16 inputs. It combines them with the current block's K/V in
native code and leaves the prefix unchanged. The fixture compares its output
with a fresh run over the same prefix and block.

Set `CK_SIGMA_BF16_CHECKPOINT` to the pinned Sigma-3B safetensors and
`CK_BLAS_LIBRARY` to an LP64 BLAS library that exports `dgemm_`. Run
`python -m pytest -q tests/test_v8_sigma_bf16_block.py` on an AVX2 host for
the weight-backed comparison. The test verifies tensor hashes, compares
against PyTorch's math attention backend, and reports each checkpoint when
invoked with `-o junit_family=xunit1 --junitxml sigma-block.xml`.

Use the same checkpoint and BLAS settings with
`python -m pytest -q tests/test_v8_sigma_bf16_denoiser_stack.py` to compare
the full 26-layer denoiser and 131,072-token BF16 readout against PyTorch.
Allow about 12 GiB of additional available RAM for generating this replay
bundle. This fixture supplies the hidden input and mask directly. Guidance,
noise transitions, and token commit belong to the surrounding sampling path.

### Drama 3 BF16 shared codebook tower replay

The first layer accepts teacher-forced code rows, backbone hidden rows and the
prepared sum of codebook embeddings unseen by the backbone. The code row
equal to the vocabulary size addresses a zero row for non-semantic tokens.
The generated entry runs the embedding gather, zero-centered norm, mixing,
interleaved partial RoPE, causal gated attention, SwiGLU and final norm.
Checkpoint preparation splits the packed Q/gate/K/V weights by output row.

For the pinned weight replay, fetch the byte ranges from
`fishaudio/drama-3` at revision `78e5ab79e49aa9a25abe8799eff2c4d73a1c3216`. Use
`model-00014-of-00015.safetensors`, bytes `0-1048575` as `shard14.header` and
`382550624-546149983` as `shared0.bin`. The test verifies both SHA-256 hashes
before interpreting weights. Set `CK_DRAMA3_TOWER0_RANGE_DIR` to their directory,
provide an LP64 `CK_BLAS_LIBRARY`, and run
`python -m pytest -q tests/test_v8_drama3_bf16_tower.py`. Without the checkpoint
ranges the synthetic composition still runs and the pinned test reports SKIP.
The replay covers one layer with supplied inputs, before native prediction.

The complete teacher-forced ingress plus greedy codebook feedback replay uses
all four shared layers, nine residual layers, and nine heads. The request
supplies the first semantic code, a hidden row and the initial enrichment
sum. Subsequent code IDs are selected in generated C over the codec's 1,024
legal residual columns and consumed by the next layer. This describes one
isolated frame with no prior tower cache.

Set `CK_DRAMA3_TOWER_RANGES` to a directory containing the first 1 MiB of shards
14 and 15 (`shard14.header`, `shard15.header`), plus the following inclusive
byte ranges from the same pinned safetensors revision.

| Shard | Bytes | Filename |
| --- | --- | --- |
| 14 | 168607328–1036948063 | `tower-shard14.bin` |
| 15 | 15128–1409461015 | `tower-shard15.bin` |

Run `python -m pytest -q tests/test_v8_drama3_bf16_codebook_frame.py` with the
same BLAS settings. The test hashes both payloads, checks per-head logits and
greedy IDs against PyTorch, and records local and propagated layer error.
The pinned real-weight case currently misses the parity gate. The portable
synthetic pass covers feedback, A→B→A, and rejection/recovery.

### Sigma BF16 fixed-noise generation replay

Use the same checkpoint and LP64 BLAS configuration with
`python -m pytest -q tests/test_v8_sigma_native_step.py`. The replay executes
one generated clean-token prefill entry and one reusable generated denoising
entry. The native schedule driver accepts the prefilled clean K/V, a strictly
decreasing FP64 gamma schedule, initial FP64 latent, and one supplied FP64
noise row per update. It resets self-conditioning for each request. The
driver publishes conditional token IDs after the final evaluation and
preserves the caller's output on failure or cancellation.

The pinned replay uses the six prompt IDs for “The capital of France is,”
three generated positions, three DDPM transitions and one terminal
evaluation at guidance 1.75 and score temperature 1. The conditional branch
reads the prefix and feeds back its own mean. The unconditional branch sees
only the current block. Its mean contributes to the guided estimate. The
test records any unconditional argmax differences as diagnostics and checks
the logits and both means against the independent PyTorch calculation.
Conditional IDs must match along the trajectory.

This is a fixed-width, single-block scope with caller-supplied noise and
separate prepared prefill and step arenas. Progression to subsequent
generated blocks, RNG policy, and serving require additional runtime
contracts. The real fixture requires about 20 GiB available memory. To retain
the report, use `-o junit_family=xunit1 --junitxml sigma-native-step.xml`.

### Sigma BF16 bounded cache continuation

Run `python -m pytest -q tests/test_v8_sigma_native_session.py` with the same
Sigma checkpoint and LP64 BLAS settings. The fixture loads one weight arena
and binds three generated entries for clean prefill, clean append, and a
guided DDPM step using the checked execution manifest. Allow about 22 GiB
free memory for the pinned fixture.

The native session reserves a BF16 cache of fixed capacity. The append and
denoise entries use absolute positions and two explicit attention ranges.
They read the valid cached rows and current rows, excluding unused cache
capacity. The append entry stages only the newly selected clean tokens'
keys and values. The native session publishes them with the new cache length
and block IDs after successful evaluation. Rejection or cancellation leaves
the committed cache and token output intact. The arena is single-owner.

The fixture runs the six-token prompt through two three-token blocks with
fixed noise and compares the appended K/V, latent trajectory and committed
IDs against the independent PyTorch calculation. It also exercises recovery
and repeats both blocks with direct native dispatch on the original loaded
library. Prompt and block widths are compiled capacities. An application
serving endpoint and native noise policy are separate milestones.
