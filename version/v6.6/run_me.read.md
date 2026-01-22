# v6.6 quick run guide

This is a short, practical checklist for the scripts used in v6.6.

## Kernel maps + registry

Validate kernel map JSON:

```bash
python3 version/v6.6/scripts/validate_kernel_maps.py
```

Generate registry from maps:

```bash
python3 version/v6.6/scripts/gen_kernel_registry_from_maps.py
```

Compare maps vs source functions:

```bash
python3 version/v6.6/scripts/check_kernel_map_sync.py
```

Source scan only (what exists in C):

```bash
python3 version/v6.6/scripts/gen_kernel_registry.py
```

## Graph templates

Validate templates:

```bash
python3 version/v6.6/scripts/validate_templates.py
```

## BUMP conversion (GGUF)

Convert GGUF to BUMPWGT5 + manifest:

```bash
python3 version/v6.6/scripts/convert_gguf_to_bump_v6_6.py \
  --gguf /path/to/model.gguf \
  --output /path/to/weights.bump \
  --manifest-out /path/to/weights_manifest.json
```

Force legacy BUMPWGT4:

```bash
python3 version/v6.6/scripts/convert_gguf_to_bump_v6_6.py \
  --gguf /path/to/model.gguf \
  --output /path/to/weights.bump \
  --manifest-out /path/to/weights_manifest.json \
  --bump-version 4
```

## BUMP inspection + validation

Inspect BUMP header + metadata footer (BUMPWGT5):

```bash
python3 version/v6.6/scripts/bump_inspect.py /path/to/weights.bump \
  --manifest /path/to/weights_manifest.json
```

Dump full metadata JSON:

```bash
python3 version/v6.6/scripts/bump_inspect.py /path/to/weights.bump --dump-meta
```

## IR build (current pipeline)

Build IR using template + manifest + registry:

```bash
python3 version/v6.6/scripts/build_ir_v6_6.py \
  --template qwen2 \
  --weights-manifest /path/to/weights_manifest.json \
  --kernel-specs version/v6.6/kernel_maps/KERNEL_REGISTRY.json
```

## Notes

- Templates live in `version/v6.6/templates/`.
- Kernel maps live in `version/v6.6/kernel_maps/`.
- BUMPWGT5 spec: `version/v6.6/docs/BUMPWGT5_SPEC.md`.

## How the pieces fit

1) Converter (GGUF/HF → BUMP)
   - Produces `weights.bump` + `weights_manifest.json`.
   - `weights_manifest.json` contains file offsets/sizes/dtypes plus template + config (v5).
   - BUMPWGT5 appends metadata (template/config/quant summary) at EOF.

2) IR / layout (build_ir_v6_6.py)
   - Consumes template + config + kernel registry (prefers manifest when present).
   - Uses `weights_manifest.json` for per-layer dtypes and validation.
   - Outputs `weights_manifest.map` (adds *runtime offsets*).

3) Loader (runtime)
   - Reads `weights.bump` and `weights_manifest.map`.
   - Copies each tensor from file offset → runtime offset.
   - Does not use BUMPWGT5 metadata (that’s build-time).
