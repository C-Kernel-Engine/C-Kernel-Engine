# Version 8

`version/v8` is the next clean inference lane for vision bring-up.

Current scope:
- copy only the critical inference scaffold from `version/v7`
- keep template evolution for vision isolated from `v7`
- prove the copied builder still matches the stable `v7` behavior before new vision ops land

What is duplicated here right now:
- `scripts/build_ir_v8.py`
- `scripts/memory_planner_v8.py`
- `tools/open_ir_visualizer_v8.py`
- `tools/ir_visualizer.html`
- `templates/*`

What is intentionally still reused from `v7`:
- kernel maps and bindings
- converter/runtime utilities
- profiling and report generation fallbacks inside the visualizer launcher

That keeps `v8` small and honest: the version split exists now, while the runtime substrate can stay shared until vision-specific lowering starts to diverge.
