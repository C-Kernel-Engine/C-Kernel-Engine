# BUMPWGT5 Spec (v6.6)

Goal: keep the existing BUMPWGT4 payload layout intact while adding a
self-describing metadata block. Metadata is appended at EOF and located via
a fixed-size footer.

## File Layout

```
0x0000  [BUMPWGT5 header, 128 bytes]
0x0080  [dtype_table_len (u32) + dtype_table bytes]
...     [weights payload (unchanged from BUMPWGT4)]
...     [metadata JSON blob]
EOF-48  [metadata footer: "BUMPV5MD" + meta_size + meta_sha256]
```

Notes:
- The weights payload and dtype table offsets are unchanged from BUMPWGT4.
- Metadata is appended at EOF to avoid shifting offsets referenced by the
  sidecar manifest.
- The footer is a fixed size (48 bytes) so metadata location is derivable.

## Header (128 bytes)

Same as BUMPWGT4, except `magic="BUMPWGT5"` and `version=5`.
See `include/ckernel_bump_v5.h`.

## Metadata Footer (48 bytes)

```
struct CKBumpMetaFooterV5 {
  char     magic[8];       // "BUMPV5MD"
  uint64_t meta_size;      // bytes of JSON
  uint8_t  meta_sha256[32];// SHA-256 of JSON blob
};
```

To locate metadata:
```
meta_footer = read_last_48_bytes(file)
meta_offset = file_size - 48 - meta_footer.meta_size
```

## Metadata JSON Schema (v1)

Canonical JSON is used for hashing:
`json.dumps(meta, sort_keys=True, separators=(",", ":"))`

```json
{
  "schema_version": 1,
  "template": { /* full template JSON */ },
  "template_hash": "sha256",
  "config": {
    "model": "qwen2",
    "num_layers": 24,
    "embed_dim": 896,
    "num_heads": 14,
    "num_kv_heads": 2,
    "head_dim": 64,
    "intermediate_size": 4864,
    "context_length": 8192,
    "rope_theta": 1000000.0,
    "rms_eps": 1e-5
  },
  "quant_summary": {
    "layer.0": {"wq": "q5_0", "wk": "q5_0", "wv": "q8_0", "wo": "q5_0", "w1": "q5_0", "w2": "q6_k"},
    "layer.1": {"wq": "q5_0", "wk": "q5_0", "wv": "q8_0", "wo": "q5_0", "w1": "q5_0", "w2": "q6_k"}
  },
  "manifest_hash": "sha256",
  "created_by": "convert_gguf_to_bump_v6_6.py",
  "created_at": "2025-01-19T00:00:00Z"
}
```

Required fields:
- `schema_version`
- `template`
- `template_hash`
- `config`
- `quant_summary`
- `manifest_hash`

## Hashing Rules

- `meta_sha256` hashes the canonical JSON metadata blob only.
- The existing header `checksum` continues to hash `dtype_table + weights`.
- `manifest_hash` binds the sidecar `weights_manifest.json` to this bump file.

## Compatibility

- BUMPWGT4 readers will reject BUMPWGT5 due to magic mismatch.
- BUMPWGT5 readers must accept BUMPWGT4 (no metadata footer).

