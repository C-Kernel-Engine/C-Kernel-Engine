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
