# BUMPWGT5 Implementation Plan

## Overview

This plan implements BUMPWGT5: a self-describing weight format with embedded metadata JSON and cryptographic hash verification.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IMPLEMENTATION PHASES                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Specification & Header Definition                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  File: include/ckernel_bump_v5.h                                            │
│  Define: BUMPWGT5 magic, header struct, metadata schema                     │
│                                                                              │
│  Phase 2: Converter Updates (Write BUMPWGT5)                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Files: convert_gguf_to_bump_v6_6.py, convert_hf_to_bump_v6_6.py           │
│  Add: metadata generation, hash calculation, v5 writing                      │
│                                                                              │
│  Phase 3: Loader Updates (Read BUMPWGT5)                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Files: ckernel_model_load_v4.c (rename to v5), test_bump_tokenizer.c       │
│  Add: v5 header parsing, metadata extraction                                 │
│                                                                              │
│  Phase 4: IR Pipeline Integration                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Files: build_ir_v6_6.py                                                    │
│  Add: --bump option, metadata parsing, hash verification                    │
│                                                                              │
│  Phase 5: Tools & Validation                                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Files: bump_inspect.py, validate_bump_v5.py                                │
│  Add: inspection, validation, migration tools                               │
│                                                                              │
│  Phase 6: Tests & Documentation                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Files: test_bump_v5.py, version/v6.6/docs/BUMPWGT5_SPEC.md                │
│  Add: test coverage, spec documentation                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Specification & Header Definition

### File: `include/ckernel_bump_v5.h`

**Purpose:** Define BUMPWGT5 format specification

```c
#ifndef CKERNEL_BUMP_V5_H
#define CKERNEL_BUMP_V5_H

#include <stdint.h>

// ============================================================================
// BUMPWGT5 MAGIC & VERSION
// ============================================================================

#define CK_BUMP_MAGIC_V5 0x35504D42  // "BUMP" in little-endian with version 5
#define CK_BUMP_VERSION_V5 5

// ============================================================================
// BUMPWGT5 HEADER STRUCTURE
// ============================================================================

typedef struct CKBMFV5Header {
    uint32_t magic;              // CK_BUMP_MAGIC_V5
    uint32_t version;            // 5
    uint32_t header_size;        // Size of this header (bytes, 64-byte aligned)
    uint64_t meta_offset;        // Offset to metadata JSON blob
    uint64_t meta_size;         // Size of metadata JSON blob
    uint8_t  meta_sha256[32];   // SHA256 hash of metadata JSON
    uint64_t dtype_table_offset; // Offset to dtype table
    uint32_t dtype_table_len;   // Number of dtype entries
    uint64_t weights_offset;     // Offset to weights payload
    uint64_t weights_size;       // Size of weights payload
    uint8_t  payload_sha256[32]; // SHA256 of weights + metadata
    uint32_t template_id;        // Model template ID (from templates/)
    uint32_t reserved[4];        // Reserved for future use
} CKBMFV5Header;

// Verify struct size is 64 bytes
_Static_assert(sizeof(CKBMFV5Header) == 64, "CKBMFV5Header must be 64 bytes");

// ============================================================================
// METADATA JSON SCHEMA
// ============================================================================

/*
BUMPWGT5 Metadata JSON Structure:

{
  "schema_version": "1.0",
  "format": "BUMPWGT5",
  "created_by": "convert_gguf_to_bump_v6_6.py",
  "created_at": "2026-01-19T10:30:00Z",

  "template": { ... full template JSON ... },
  "template_hash": "sha256 hex string",

  "config": {
    "model_type": "qwen2",
    "embed_dim": 896,
    "num_layers": 32,
    "num_heads": 14,
    "num_kv_heads": 2,
    "head_dim": 64,
    "intermediate": 4864,
    "context_length": 8192,
    "rope_theta": 1000000.0,
    "vocab_size": 151936
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

  "manifest_hash": "sha256 hex string of manifest JSON",
  "checksums": {
    "header": "sha256 of header fields",
    "dtype_table": "sha256 of dtype table",
    "template": "sha256 of template JSON"
  }
}
*/

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Verify BUMPWGT5 header magic
 */
int ck_bumpv5_verify_magic(const CKBMFV5Header *header);

/**
 * Validate header integrity
 */
int ck_bumpv5_validate_header(const CKBMFV5Header *header);

/**
 * Calculate offset to aligned position
 */
uint64_t ck_bumpv5_align_offset(uint64_t offset, uint32_t alignment);

/**
 * Parse metadata JSON from bump file
 */
int ck_bumpv5_parse_metadata(FILE *fp, const CKBMFV5Header *header,
                             char **metadata_json, size_t *metadata_size);

#endif // CKERNEL_BUMP_V5_H
```

## Phase 2: Converter Updates (Write BUMPWGT5)

### File: `version/v6.6/scripts/convert_gguf_to_bump_v6_6.py`

**Changes:**

1. **Add BUMPWGT5 constants and metadata builder**
2. **Update main() with --bump-version flag**
3. **Write BUMPWGT5 with embedded metadata**

```python
#!/usr/bin/env python3
"""
Convert GGUF to BUMP format v6.6

BUMPWGT5 Support:
  --bump-version 5: Write BUMPWGT5 with embedded metadata JSON
  --bump-version 4: Write BUMPWGT4 (legacy)
"""

import argparse
import json
import hashlib
import struct
from pathlib import Path

# BUMPWGT5 constants
BUMP_MAGIC_V5 = 0x35504D42
BUMP_VERSION_V5 = 5
METADATA_ALIGNMENT = 64

def build_bumpv5_metadata(template_data: dict, config: dict, quant_summary: dict,
                         manifest_hash: str, template_hash: str) -> dict:
    """Build metadata JSON for BUMPWGT5."""
    metadata = {
        "schema_version": "1.0",
        "format": "BUMPWGT5",
        "created_by": "convert_gguf_to_bump_v6_6.py",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "template": template_data,
        "template_hash": template_hash,
        "config": config,
        "quant_summary": quant_summary,
        "manifest_hash": manifest_hash,
        "checksums": {
            "template": template_hash,
        }
    }
    return metadata

def write_bumpv5_header(fp, header: CKBMFV5Header):
    """Write BUMPWGT5 header to file."""
    fp.seek(0)
    fp.write(struct.pack('<II', header.magic, header.version))
    fp.write(struct.pack('<I', header.header_size))
    fp.write(struct.pack('<Q', header.meta_offset))
    fp.write(struct.pack('<Q', header.meta_size))
    fp.write(header.meta_sha256)
    fp.write(struct.pack('<Q', header.dtype_table_offset))
    fp.write(struct.pack('<I', header.dtype_table_len))
    fp.write(struct.pack('<Q', header.weights_offset))
    fp.write(struct.pack('<Q', header.weights_size))
    fp.write(header.payload_sha256)
    fp.write(struct.pack('<I', header.template_id))
    fp.write(b'\x00' * 16)  # reserved

def convert_to_bumpv5(gguf_path: Path, output_path: Path, template_id: int,
                     bump_version: int = 5):
    """Convert GGUF to BUMPWGT5."""
    # Load GGUF
    gguf = load_gguf(gguf_path)

    # Load template
    template_data = load_template(gguf.model_type, template_id)
    template_json = json.dumps(template_data, separators=(',', ':'))
    template_hash = hashlib.sha256(template_json.encode()).hexdigest()

    # Build config from GGUF
    config = extract_config_from_gguf(gguf)

    # Build quant summary
    quant_summary = extract_quant_summary_from_gguf(gguf)

    # Create metadata
    metadata = build_bumpv5_metadata(
        template_data, config, quant_summary,
        manifest_hash="",  # TODO: compute after manifest
        template_hash=template_hash
    )
    metadata_json = json.dumps(metadata, separators=(',', ':'))
    meta_sha256 = hashlib.sha256(metadata_json.encode()).digest()

    # Open output file
    with open(output_path, 'wb') as fp:
        # Write temporary header
        header = CKBMFV5Header(
            magic=BUMP_MAGIC_V5,
            version=BUMP_VERSION_V5,
            header_size=64,
            meta_offset=64,  # Right after header
            meta_size=len(metadata_json),
            meta_sha256=meta_sha256,
            dtype_table_offset=0,  # Will update
            dtype_table_len=0,      # Will update
            weights_offset=0,       # Will update
            weights_size=0,         # Will update
            payload_sha256=b'\x00' * 32,  # Will update
            template_id=template_id,
            reserved=[0, 0, 0, 0]
        )
        fp.write(b'\x00' * 64)  # Placeholder header

        # Write metadata (aligned)
        fp.write(metadata_json.encode())
        fp.write(b'\x00' * (METADATA_ALIGNMENT - (len(metadata_json) % METADATA_ALIGNMENT)))

        # Write dtype table
        dtype_offset = fp.tell()
        # ... write dtype table ...

        # Write weights
        weights_offset = fp.tell()
        # ... write weights with hashing ...

        # Update header
        header.dtype_table_offset = dtype_offset
        header.weights_offset = weights_offset
        # ... update sizes ...

        # Rewrite header
        write_bumpv5_header(fp, header)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gguf_path', type=Path)
    parser.add_argument('output_path', type=Path)
    parser.add_argument('--template-id', type=int, required=True)
    parser.add_argument('--bump-version', type=int, default=5,
                       choices=[4, 5], help='BUMP version to write (default: 5)')
    args = parser.parse_args()

    if args.bump_version == 5:
        convert_to_bumpv5(args.gguf_path, args.output_path, args.template_id, 5)
    else:
        # Legacy BUMPWGT4
        convert_to_bumpv4(args.gguf_path, args.output_path, args.template_id)

if __name__ == '__main__':
    main()
```

**Key functions to add:**
- `build_bumpv5_metadata()` - Create metadata JSON
- `write_bumpv5_header()` - Write header struct
- `convert_to_bumpv5()` - Main conversion with v5 format

### File: `version/v6.6/scripts/convert_hf_to_bump_v6_6.py`

**Similar changes for HF (HuggingFace) format:**
- Same BUMPWGT5 header writing
- Extract metadata from HF model config
- Build quant summary from HF model state dict

## Phase 3: Loader Updates (Read BUMPWGT5)

### File: `src/ckernel_model_load_v5.c` (rename from v4)

**Changes:**

1. **Detect BUMPWGT5 vs BUMPWGT4 from magic**
2. **Parse BUMPWGT5 header**
3. **Extract metadata JSON**
4. **Return metadata to caller**

```c
/**
 * Load BUMP model (v4 or v5)
 *
 * Returns: 0 on success, -1 on error
 * Outputs:
 *   - model: loaded model structure
 *   - metadata: JSON metadata (BUMPWGT5 only, NULL for v4)
 *   - metadata_size: size of metadata (0 for v4)
 */
int ck_load_model(const char *bump_path,
                  CKModel *model,
                  char **metadata_json,
                  size_t *metadata_size)
{
    FILE *fp = fopen(bump_path, "rb");
    if (!fp) return -1;

    // Read header
    uint32_t magic, version;
    fread(&magic, sizeof(uint32_t), 1, fp);

    // Detect version from magic
    if (magic == CK_BUMP_MAGIC_V5) {
        // BUMPWGT5
        version = 5;
        return ck_load_model_v5(fp, model, metadata_json, metadata_size);
    } else if (magic == CK_BUMP_MAGIC_V4) {
        // BUMPWGT4 (legacy)
        fseek(fp, 0, SEEK_SET);
        version = 4;
        return ck_load_model_v4(fp, model, NULL, NULL);  // No metadata
    } else {
        fclose(fp);
        return -1;  // Unknown format
    }
}

/**
 * Load BUMPWGT5 model
 */
int ck_load_model_v5(FILE *fp, CKModel *model,
                     char **metadata_json,
                     size_t *metadata_size)
{
    // Read header
    CKBMFV5Header header;
    fread(&header, sizeof(CKBMFV5Header), 1, fp);

    // Validate
    if (!ck_bumpv5_verify_magic(&header)) return -1;
    if (!ck_bumpv5_validate_header(&header)) return -1;

    // Parse metadata
    if (metadata_json && metadata_size) {
        if (ck_bumpv5_parse_metadata(fp, &header,
                                     metadata_json, metadata_size) != 0) {
            return -1;
        }
    }

    // Load dtype table
    fseek(fp, header.dtype_table_offset, SEEK_SET);
    ck_load_dtype_table(fp, header.dtype_table_len, &model->dtype_table);

    // Load weights
    fseek(fp, header.weights_offset, SEEK_SET);
    ck_load_weights_payload(fp, header.weights_size, model);

    // Verify checksums
    // TODO: Implement hash verification

    return 0;
}
```

**Key functions to add:**
- `ck_load_model()` - Universal loader (v4 or v5)
- `ck_load_model_v5()` - BUMPWGT5 specific loader
- `ck_bumpv5_parse_metadata()` - Extract metadata JSON

### File: `version/v6.6/src/test_bump_tokenizer.c`

**Changes to support BUMPWGT5:**

```c
int main(int argc, char **argv) {
    // Load BUMP (v4 or v5)
    char *metadata_json = NULL;
    size_t metadata_size = 0;

    int ret = ck_load_model(argv[1], &model, &metadata_json, &metadata_size);
    if (ret != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // If BUMPWGT5, parse metadata
    if (metadata_json) {
        printf("BUMPWGT5 metadata:\n%s\n", metadata_json);

        // Parse JSON to extract tokenizer info
        cJSON *meta = cJSON_Parse(metadata_json);
        cJSON *config = cJSON_GetObjectItem(meta, "config");
        cJSON *bos_token = cJSON_GetObjectItem(config, "bos_token_id");
        cJSON *eos_token = cJSON_GetObjectItem(config, "eos_token_id");

        printf("BOS token: %d\n", bos_token->valueint);
        printf("EOS token: %d\n", eos_token->valueint);

        cJSON_Delete(meta);
        free(metadata_json);
    }

    // Continue with tokenizer tests...
}
```

## Phase 4: IR Pipeline Integration

### File: `version/v6.6/scripts/build_ir_v6_6.py`

**Changes:**

```python
#!/usr/bin/env python3
"""
Build IR from BUMP model

Supports both BUMPWGT4 and BUMPWGT5
"""

import argparse
import json
import hashlib
from pathlib import Path

def load_model_metadata(bump_path: Path) -> dict:
    """Load model metadata from BUMP file.

    For BUMPWGT5: extracts metadata from embedded JSON
    For BUMPWGT4: requires explicit template + manifest
    """
    with open(bump_path, 'rb') as fp:
        # Read magic
        magic = struct.unpack('<I', fp.read(4))[0]

        if magic == 0x35504D42:  # BUMPWGT5
            # Read header
            header = parse_bumpv5_header(fp)

            # Seek to metadata
            fp.seek(header.meta_offset)
            metadata_json = fp.read(header.meta_size).decode('utf-8')
            metadata = json.loads(metadata_json)

            return {
                'version': 5,
                'template': metadata['template'],
                'config': metadata['config'],
                'quant_summary': metadata['quant_summary'],
                'template_hash': metadata['template_hash'],
                'manifest_hash': metadata['manifest_hash'],
                'metadata_json': metadata_json,  # Keep for IR
            }
        else:
            # BUMPWGT4 - require explicit files
            raise ValueError(
                f"BUMPWGT4 requires explicit --template and --manifest"
            )

def verify_metadata_integrity(bump_path: Path, metadata: dict) -> bool:
    """Verify metadata hash matches actual JSON."""
    # TODO: Implement hash verification
    return True

def build_ir_from_bump(bump_path: Path, output_dir: Path):
    """Build IR from BUMP model."""
    # Load metadata
    metadata = load_model_metadata(bump_path)

    # Verify hash (BUMPWGT5 only)
    if metadata['version'] == 5:
        if not verify_metadata_integrity(bump_path, metadata):
            raise ValueError("Metadata hash mismatch!")

    # Build IR
    ir = IRBuilder.from_metadata(metadata)

    # Save IR
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'ir.json', 'w') as f:
        json.dump(ir.to_dict(), f, indent=2)

    # Save metadata for later stages
    with open(output_dir / 'metadata.json', 'w') as f:
        f.write(metadata['metadata_json'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bump', type=Path, required=True,
                       help='BUMP model file (v4 or v5)')
    parser.add_argument('--template', type=Path,
                       help='Template JSON (required for BUMPWGT4)')
    parser.add_argument('--manifest', type=Path,
                       help='Manifest JSON (required for BUMPWGT4)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-hashes', action='store_true',
                       help='Verify metadata hashes')
    args = parser.parse_args()

    # Load BUMP and build IR
    build_ir_from_bump(args.bump, args.output)

if __name__ == '__main__':
    main()
```

## Phase 5: Tools & Validation

### File: `version/v6.6/scripts/bump_inspect.py`

**Purpose:** Inspect BUMP file metadata

```python
#!/usr/bin/env python3
"""
Inspect BUMP file structure and metadata
"""

import argparse
import json
import struct
from pathlib import Path

def inspect_bump(bump_path: Path):
    """Print BUMP file information."""
    with open(bump_path, 'rb') as fp:
        magic = struct.unpack('<I', fp.read(4))[0]

        if magic == 0x35504D42:
            print(f"Format: BUMPWGT5")
            header = parse_bumpv5_header(fp)
            print(f"  Header size: {header['header_size']} bytes")
            print(f"  Metadata offset: {header['meta_offset']}")
            print(f"  Metadata size: {header['meta_size']} bytes")
            print(f"  Weights offset: {header['weights_offset']}")
            print(f"  Weights size: {header['weights_size']} bytes")
            print(f"  Template ID: {header['template_id']}")

            # Read metadata
            fp.seek(header['meta_offset'])
            metadata_json = fp.read(header['meta_size']).decode('utf-8')
            metadata = json.loads(metadata_json)

            print(f"\nMetadata:")
            print(f"  Created by: {metadata['created_by']}")
            print(f"  Config: {json.dumps(metadata['config'], indent=4)}")
            print(f"  Quant summary: {json.dumps(metadata['quant_summary'], indent=4)}")

        else:
            print(f"Format: BUMPWGT4 (legacy)")
            # TODO: Parse v4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bump_path', type=Path)
    args = parser.parse_args()

    inspect_bump(args.bump_path)
```

### File: `version/v6.6/scripts/validate_bump_v5.py`

**Purpose:** Validate BUMPWGT5 integrity

```python
#!/usr/bin/env python3
"""
Validate BUMPWGT5 file integrity
"""

import argparse
import hashlib
import json
from pathlib import Path

def validate_bumpv5(bump_path: Path, manifest_path: Path = None) -> bool:
    """Validate BUMPWGT5 file."""
    errors = []
    warnings = []

    with open(bump_path, 'rb') as fp:
        # Read and verify header
        header = parse_bumpv5_header(fp)

        # Verify metadata hash
        fp.seek(header.meta_offset)
        metadata_json = fp.read(header.meta_size)
        actual_meta_hash = hashlib.sha256(metadata_json).digest()
        expected_meta_hash = header.meta_sha256

        if actual_meta_hash != expected_meta_hash:
            errors.append(f"Metadata hash mismatch!")

        # Parse metadata
        metadata = json.loads(metadata_json.decode('utf-8'))

        # Verify template hash
        template = json.dumps(metadata['template'], separators=(',', ':'))
        actual_template_hash = hashlib.sha256(template.encode()).hexdigest()
        expected_template_hash = metadata['template_hash']

        if actual_template_hash != expected_template_hash:
            errors.append(f"Template hash mismatch!")

        # Verify manifest hash (if provided)
        if manifest_path and 'manifest_hash' in metadata:
            with open(manifest_path) as f:
                manifest_json = f.read()
            actual_manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
            expected_manifest_hash = metadata['manifest_hash']

            if actual_manifest_hash != expected_manifest_hash:
                errors.append(f"Manifest hash mismatch!")

        # Verify weights hash
        fp.seek(header.weights_offset)
        weights_data = fp.read(header.weights_size)
        # TODO: Include metadata in hash calculation

    # Report results
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("✓ BUMPWGT5 validation passed")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bump_path', type=Path)
    parser.add_argument('--manifest', type=Path, help='Manifest JSON to verify against')
    args = parser.parse_args()

    validate_bumpv5(args.bump_path, args.manifest)
```

## Phase 6: Tests & Documentation

### File: `version/v6.6/tests/test_bump_v5.py`

```python
#!/usr/bin/env python3
"""
Test BUMPWGT5 round-trip (GGUF -> BUMPWGT5 -> Load)
"""

import json
import tempfile
from pathlib import Path

def test_bumpv5_metadata_roundtrip():
    """Test metadata survives conversion."""
    # Convert GGUF to BUMPWGT5
    tmpdir = tempfile.mkdtemp()
    bump_path = Path(tmpdir) / 'model.bump'

    convert_gguf_to_bump_v6_6(
        gguf_path='test_data/model.gguf',
        bump_path=bump_path,
        template_id=1,
        bump_version=5
    )

    # Load and verify
    metadata = load_model_metadata(bump_path)

    assert metadata['version'] == 5
    assert 'template' in metadata
    assert 'config' in metadata
    assert 'quant_summary' in metadata

    print("✓ BUMPWGT5 metadata round-trip test passed")

def test_bumpv4_compatibility():
    """Test BUMPWGT4 files still work."""
    # Load old BUMPWGT4 file
    metadata = load_model_metadata(Path('test_data/model_v4.bump'))

    assert metadata['version'] == 4
    assert metadata['metadata_json'] is None  # No metadata in v4

    print("✓ BUMPWGT4 compatibility test passed")

if __name__ == '__main__':
    test_bumpv5_metadata_roundtrip()
    test_bumpv4_compatibility()
```

### File: `version/v6.6/docs/BUMPWGT5_SPEC.md`

```markdown
# BUMPWGT5 Specification

## Overview

BUMPWGT5 is a self-describing weight format with embedded metadata and cryptographic integrity verification.

## Header Structure

```
Offset  Size  Field
0       4     magic = 0x35504D42
4       4     version = 5
8       4     header_size = 64
12      8     meta_offset
20      8     meta_size
28      32    meta_sha256
60      8     dtype_table_offset
68      4     dtype_table_len
72      8     weights_offset
80      8     weights_size
88      32    payload_sha256
120     4     template_id
124     4     reserved[4]
128
```

## Metadata JSON

Embedded JSON blob containing:
- Template (full model template)
- Config (model parameters)
- Quant summary (quantization types per layer)
- Hashes (integrity verification)

See `include/ckernel_bump_v5.h` for schema.

## Hash Verification

Three levels of integrity:
1. `meta_sha256` - metadata JSON integrity
2. `template_hash` - template integrity
3. `manifest_hash` - sidecar manifest binding

## Backward Compatibility

BUMPWGT4 files remain readable. Converters support both versions via `--bump-version` flag.
```

## Implementation Order & Dependencies

```
Phase 1 (Week 1):
  ├─ include/ckernel_bump_v5.h ← Foundation, no dependencies
  └─ Version bump in Python converters

Phase 2 (Week 1-2):
  ├─ convert_gguf_to_bump_v6_6.py ← Depends on header def
  └─ convert_hf_to_bump_v6_6.py ← Independent

Phase 3 (Week 2):
  ├─ src/ckernel_model_load_v5.c ← Depends on header def
  └─ test_bump_tokenizer.c ← Depends on loader

Phase 4 (Week 2-3):
  └─ build_ir_v6_6.py ← Depends on all above

Phase 5 (Week 3):
  ├─ bump_inspect.py ← Depends on header def
  └─ validate_bump_v5.py ← Depends on all above

Phase 6 (Week 3-4):
  └─ Tests & docs ← Depends on all above
```

## Testing Strategy

1. **Unit tests** - Each phase tested independently
2. **Integration tests** - Full pipeline GGUF → BUMPWGT5 → IR
3. **Compatibility tests** - BUMPWGT4 still works
4. **Hash verification tests** - Tampering detection

## Migration Path

```bash
# Upgrade existing models
python3 scripts/migrate_bump_v4_to_v5.py model_v4.bump --output model_v5.bump

# Verify migration
python3 scripts/validate_bump_v5.py model_v5.bump --manifest manifest.json
```
