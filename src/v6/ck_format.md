# C-Kernel Format (.ck) v0 (Draft)

This file defines the v6 C-Kernel container format used to store weights,
tokenizer data, alignment, and metadata in a single self-contained file.

Goals
- Single file for weights + tokenizer + offsets so codegen can be explicit.
- Cache-aligned tensors and precomputed runtime offsets for bump allocator.
- Optional sections for quant metadata, kernel requirements, training metadata.
- Little-endian, mmap-friendly, deterministic layout.

------------------------------------------------------------------------------
1) File Layout (Little-Endian)
------------------------------------------------------------------------------

File structure:
  [Header]
  [Section Table]
  [Sections...]

All offsets in the header and table are absolute from the start of the file.

Header (fixed 128 bytes):
  magic[4]            = "CKF1"
  version_major       = uint16 (0)
  version_minor       = uint16 (1)
  header_bytes        = uint32 (128)
  section_count       = uint32
  flags               = uint32
  alignment           = uint32 (default 64)
  file_bytes          = uint64 (total file size)
  arena_bytes         = uint64 (total bump arena size)
  section_table_off   = uint64
  section_table_bytes = uint64
  build_id            = uint64 (hash of config+weights+tokenizer or 0)
  reserved[64]

Flags (header):
  CKF_FLAG_HAS_QUANT   = 0x00000001
  CKF_FLAG_HAS_TRAIN   = 0x00000002
  CKF_FLAG_HAS_KERNELS = 0x00000004

Section Table Entry (32 bytes each):
  type      = uint32  (4-char code, e.g., 'CONF', 'TOKN', 'TENS')
  flags     = uint32
  offset    = uint64
  bytes     = uint64
  alignment = uint32
  reserved  = uint32

Section flags are section-specific (see below).

------------------------------------------------------------------------------
2) Section Types
------------------------------------------------------------------------------

Section type codes (uint32, ASCII):
  'CONF'  Config JSON
  'TOKN'  Tokenizer blob
  'TENS'  Tensor table
  'DATA'  Tensor data
  'QMET'  Quant metadata (JSON or binary)
  'KREQ'  Kernel requirements (JSON)
  'TRAI'  Training metadata (JSON)
  'META'  Build metadata (JSON)

------------------------------------------------------------------------------
3) Section Definitions
------------------------------------------------------------------------------

3.1) CONF (Config JSON)
  Payload: UTF-8 JSON (model config + derived fields).
  Suggested fields:
    model_id, revision, vocab_size, num_layers, num_heads, num_kv_heads,
    head_dim, hidden_size, intermediate_size, max_seq_len, rope_theta, dtype.

3.2) TOKN (Tokenizer)
  Payload header:
    tokenizer_format  = uint32 (1=json, 2=sentencepiece, 3=bpe, 4=gguf)
    tokenizer_flags   = uint32
    payload_bytes     = uint64
    runtime_offset    = uint64 (offset in bump arena)
    runtime_bytes     = uint64 (size to reserve in arena)
  Followed by raw tokenizer bytes.

  Notes:
  - Tokenizer bytes are copied into the bump arena at runtime_offset.
  - If runtime_offset = 0xFFFFFFFFFFFFFFFF, tokenizer is not mapped into arena.

3.3) TENS (Tensor Table)
  Header:
    entry_count = uint32
    flags       = uint32
    reserved    = uint64

  Entry (variable size, padded to 8 bytes):
    name_len       = uint16
    dtype          = uint16  (CK_DT_* from ckernel_dtype.h)
    rank           = uint16
    flags          = uint16
    quant_group    = uint32
    reserved0      = uint32
    file_offset    = uint64  (absolute, points into DATA)
    file_bytes     = uint64  (stored bytes)
    runtime_offset = uint64  (offset in bump arena)
    runtime_bytes  = uint64  (reserved bytes in arena)
    alignment      = uint32
    reserved1      = uint32
    name_bytes[name_len]
    dims[rank]     = uint32 each
    padding to 8-byte boundary

  Flags (tensor):
    CK_TENSOR_FLAG_NONE      = 0x0
    CK_TENSOR_FLAG_QUANT     = 0x1
    CK_TENSOR_FLAG_TRAINABLE = 0x2

  Notes:
  - file_offset is absolute from file start (mmap-friendly).
  - runtime_offset + runtime_bytes must fit within arena_bytes.
  - runtime_bytes may exceed file_bytes to satisfy alignment.

3.4) DATA (Tensor Data)
  Raw tensor payloads (aligned per entry).
  All tensor file_offset values reference this section.

3.5) QMET (Quant Metadata)
  Payload: JSON or binary. Suggested JSON schema:
    {
      "entries": [
        {
          "name": "layer.0.wq",
          "quant_type": "q4_k",
          "block_size": 256,
          "scale_dtype": "fp16",
          "zero_dtype": "fp16",
          "group_size": 32
        }
      ]
    }

3.6) KREQ (Kernel Requirements)
  Payload: JSON list of kernel names required by codegen:
    { "kernels": ["gemm_nt_q4_k", "attention_flash_decode", ...] }

3.7) TRAI (Training Metadata)
  Payload: JSON (optimizer, batch plan, memory hints, etc.).
  Example keys: optimizer, lr, weight_decay, grad_accum, batch_size, memory_gb.

3.8) META (Build Metadata)
  Payload: JSON (hashes, ABI version, compiler flags, build timestamp).

------------------------------------------------------------------------------
4) Alignment Rules
------------------------------------------------------------------------------

- Default alignment is 64 bytes.
- Section offsets and tensor data offsets must be aligned to "alignment".
- runtime_offset should match the alignment expected by generated kernels.

------------------------------------------------------------------------------
5) Cache / Invalidation
------------------------------------------------------------------------------

- build_id in the header is a hash of (config + weights + tokenizer + format).
- The cache folder should store a separate build.json for provenance and
  ABI/flags, but the .ck file is the single source of weights+tokenizer.

------------------------------------------------------------------------------
6) Backward Compatibility
------------------------------------------------------------------------------

- Versioned header supports future extension without breaking existing files.
- New sections can be added without changing older readers (unknown sections
  are skipped using the section table).
