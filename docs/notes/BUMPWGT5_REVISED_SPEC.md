# BUMPWGT5 Revised Specification (EOF Metadata)

## Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                        BUMPWGT5 FILE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [BUMPWGT5 Header] (64 bytes)                                   │
│  ├─ magic[8] = "BUMPWGT5"                                       │
│  ├─ version = 5                                                 │
│  ├─ header_size = 64                                             │
│  ├─ meta_offset = file_size - metadata_size  ← EOF               │
│  ├─ meta_size                                                     │
│  ├─ meta_sha256[32]                                              │
│  └─ (existing v4 fields: dims, aligned dims, tokenizer, etc.)   │
│                                                                   │
│  [Dtype Table] (offset unchanged from v4!)                        │
│                                                                   │
│  [Weights Payload] (offset unchanged from v4!)                    │
│                                                                   │
│  [Metadata JSON @ EOF] (aligned to 64 bytes)                     │
│  ├─ template                                                     │
│  ├─ config                                                       │
│  ├─ quant_summary                                                │
│  └─ manifest_hash                                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Header Structure

```c
// include/ckernel_bump_v5.h

typedef struct CKBMFV5Header {
    char magic[8];                    // "BUMPWGT5"
    uint32_t version;                // 5
    uint32_t header_size;            // 64
    uint64_t meta_offset;            // Offset to metadata JSON (EOF)
    uint64_t meta_size;             // Size of metadata JSON
    uint8_t  meta_sha256[32];       // SHA256 of metadata JSON
    uint64_t dtype_table_offset;     // Unchanged from v4
    uint32_t dtype_table_len;       // Unchanged from v4
    uint64_t weights_offset;        // Unchanged from v4
    uint64_t weights_size;          // Unchanged from v4
    uint8_t  payload_sha256[32];    // Unchanged from v4
    uint32_t template_id;           // Unchanged from v4
    uint32_t model_dims[8];         // Unchanged from v4
    uint32_t tokenizer_stats[4];    // Unchanged from v4
    uint32_t reserved[4];           // Reserved
} CKBMFV5Header;

// Validate struct is 64 bytes
_Static_assert(sizeof(CKBMFV5Header) == 64, "Header must be 64 bytes");
```

## Metadata JSON Schema

```json
{
  "schema_version": "1.0",
  "format": "BUMPWGT5",

  "template": { ... full template JSON ... },
  "template_hash": "sha256 hex",

  "config": {
    "model_type": "qwen2",
    "embed_dim": 896,
    "num_layers": 32,
    "num_heads": 14,
    "num_kv_heads": 2,
    "head_dim": 64,
    "intermediate": 4864,
    "context_length": 8192,
    "rope_theta": 1000000.0
  },

  "quant_summary": {
    "layer.0.wq": "q5_0",
    "layer.0.wk": "q5_0",
    "layer.0.wv": "q8_0",
    "layer.0.wo": "q5_0",
    "layer.0.w1": "q4_k",
    "layer.0.w2": "q4_k",
    "layer.0.w3": "q4_k",
    "layer.0_norm": "fp32",
    "layer.0_ffn_norm": "fp32",
    "token_embeddings": "q4_k",
    "output_weight": "q5_0"
  },

  "manifest_hash": "sha256 hex of weights_manifest.json",

  "checksums": {
    "header": "sha256 of header fields",
    "dtype_table": "sha256 of dtype table",
    "template": "sha256 of template JSON"
  }
}
```

## Hash Calculation Rules

### 1. Metadata Hash (meta_sha256)

```python
# Canonical JSON for stable hashing
metadata_json = json.dumps(
    metadata,
    sort_keys=True,           # Stable key order
    separators=(',', ':')     # Compact, no spaces
)

# Calculate hash
meta_sha256 = hashlib.sha256(metadata_json.encode()).digest()
```

### 2. Template Hash (template_hash)

```python
# Hash the template portion separately for verification
template_json = json.dumps(
    metadata['template'],
    sort_keys=True,
    separators=(',', ':')
)
template_hash = hashlib.sha256(template_json.encode()).hexdigest()
```

### 3. Manifest Hash (manifest_hash)

```python
# Hash the sidecar manifest (if provided)
if manifest_path:
    with open(manifest_path) as f:
        manifest_json = f.read()
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
```

## Writer Workflow (Phase 2)

```python
def write_bumpv5(gguf, output_path, template, manifest_path):
    # 1. Build metadata
    metadata = build_metadata(gguf, template, manifest_path)

    # 2. Serialize metadata (canonical JSON)
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(',', ':'))

    # 3. Open file
    with open(output_path, 'wb') as fp:
        # 4. Write header (placeholder)
        fp.write(b'\x00' * 64)

        # 5. Write dtype table
        dtype_offset = fp.tell()
        write_dtype_table(fp, gguf.dtype_table)

        # 6. Write weights
        weights_offset = fp.tell()
        write_weights(fp, gguf.weights)

        # 7. Calculate offsets
        file_size = fp.tell()
        meta_offset = file_size
        meta_size = len(metadata_json)

        # 8. Append metadata at EOF
        fp.write(metadata_json.encode())
        fp.write(b'\x00' * (64 - (meta_size % 64)))  # Align

        # 9. Calculate hashes
        meta_sha256 = hashlib.sha256(metadata_json.encode()).digest()

        # 10. Rewrite header
        header = CKBMFV5Header(
            magic=b"BUMPWGT5",
            version=5,
            header_size=64,
            meta_offset=meta_offset,
            meta_size=meta_size,
            meta_sha256=meta_sha256,
            # ... existing fields ...
        )
        fp.seek(0)
        fp.write(header_to_bytes(header))
```

## Reader Workflow (Phase 3)

```c
int load_bumpv5(FILE *fp, CKModel *model, char **metadata, size_t *size) {
    // 1. Read header
    CKBMFV5Header header;
    fread(&header, sizeof(header), 1, fp);

    // 2. Verify magic
    if (strncmp(header.magic, "BUMPWGT5", 8) != 0) {
        return -1;
    }

    // 3. Load dtype table (same offset as v4!)
    fseek(fp, header.dtype_table_offset, SEEK_SET);
    load_dtype_table(fp, header.dtype_table_len, &model->dtype_table);

    // 4. Load weights (same offset as v4!)
    fseek(fp, header.weights_offset, SEEK_SET);
    load_weights(fp, header.weights_size, model);

    // 5. Read metadata from EOF
    if (metadata && size) {
        fseek(fp, header.meta_offset, SEEK_SET);
        *metadata = malloc(header.meta_size);
        fread(*metadata, 1, header.meta_size, fp);
        *size = header.meta_size;

        // 6. Verify hash
        uint8_t hash[32];
        sha256(*metadata, header.meta_size, hash);
        if (memcmp(hash, header.meta_sha256, 32) != 0) {
            free(*metadata);
            return -1;  // Hash mismatch
        }
    }

    return 0;
}
```

## Compatibility Strategy

### BUMPWGT4 → BUMPWGT5 Migration

```python
def migrate_v4_to_v5(bump_v4_path, bump_v5_path, template, manifest):
    """Upgrade BUMPWGT4 to BUMPWGT5."""

    # 1. Read v4 file
    with open(bump_v4_path, 'rb') as fp:
        # Skip v4 header
        fp.seek(64)

        # Read dtype table
        dtype_table = fp.read(...)

        # Read weights
        weights = fp.read(...)

    # 2. Build v5 file
    with open(bump_v5_path, 'wb') as fp:
        # Write header
        write_header_v5(...)

        # Write dtype table
        fp.write(dtype_table)

        # Write weights
        fp.write(weights)

        # Append metadata
        metadata = build_metadata(template, manifest, ...)
        fp.write(json.dumps(metadata, ...))
```

### Version Detection

```c
// Detect version from magic
uint32_t magic;
fread(&magic, 4, 1, fp);

if (magic == CK_BUMP_MAGIC_V5) {
    // BUMPWGT5
    load_bumpv5(fp, model, &metadata, &metadata_size);
} else if (magic == CK_BUMP_MAGIC_V4) {
    // BUMPWGT4
    fseek(fp, 0, SEEK_SET);
    load_bumpv4(fp, model, NULL, NULL);
} else {
    // Unknown
    return -1;
}
```

## Benefits of This Design

✅ **Zero offset changes** - dtype_table_offset, weights_offset unchanged
✅ **Manifest compatibility** - existing manifests work with v5
✅ **Two independent hashes** - payload (existing) + metadata (new)
✅ **Magic-based versioning** - clear version boundary
✅ **Sidecar flexibility** - manifest_hash binds but doesn't force single-file
✅ **Simple migration** - v4 → v5 is just appending metadata

## Next Steps

**Ready to implement Phase 1:**
1. `include/ckernel_bump_v5.h` - Header definition
2. `version/v6.6/docs/BUMPWGT5_SPEC.md` - Documentation

This EOF-based design is bulletproof for backward compatibility.
