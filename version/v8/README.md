# Version 8

`version/v8` is the next clean inference lane for vision bring-up.

Current scope:
- copy only the critical inference scaffold from `version/v7`
- keep template evolution for vision isolated from `v7`
- prove the copied builder still matches the stable `v7` behavior before new vision ops land
- expose a native `v8` inference runner surface for text and multimodal bring-up

What is duplicated here right now:
- `scripts/ck_run_v8.py`
- `scripts/cks-v8-run`
- `scripts/build_ir_v8.py`
- `scripts/memory_planner_v8.py`
- `tools/open_ir_visualizer_v8.py`
- `tools/ir_visualizer.html`
- `templates/*`
- `kernel_maps/*` (seeded from `v7` plus `v8` overlays)

What is intentionally still reused from `v7`:
- profiling and report generation fallbacks inside the visualizer launcher

Canonical text bring-up examples:
- `version/v8/scripts/cks-v8-run run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --context-len 1024 --force-compile --force-convert --chat-template=none --generate-visualizer`
- `version/v8/scripts/cks-v8-run run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf --context-len 1024 --force-compile --force-convert --generate-visualizer`
- `version/v8/scripts/cks-v8-run run hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf --context-len 1024 --force-compile --force-convert --generate-visualizer`
- `python3 version/v8/scripts/ck_run_v8.py run hf://unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --force-convert --force-compile --context-len 1034`
- `version/v8/scripts/cks-v8-run run hf://mradermacher/Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf --context-len 1024 --force-compile --force-convert --chat-template auto --generate-visualizer`

Canonical vision bring-up example:
- `version/v8/scripts/cks-v8-run run /path/to/decoder.gguf --mmproj /path/to/mmproj.gguf --image-path ./image.png --prompt "Describe the image."`

That keeps `v8` small and honest: the version split now includes the inference runner, local kernel registry/maps, and multimodal bridge entrypoint, while broader profiling/report tooling can still be copied forward later as needed.
