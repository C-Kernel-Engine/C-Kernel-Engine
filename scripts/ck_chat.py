#!/usr/bin/env python3
"""
C-Kernel-Engine Chat Interface

Uses the embedded C tokenizer when available, otherwise CK true_bpe/HF/GGUF tokenizer
fallbacks and calls the compiled C model library.

Features:
- Auto-validation: When gibberish is detected, automatically runs staged validation
  to pinpoint kernel issues. Enable with --validate flag.
"""
from __future__ import annotations  # Python 3.9 compatibility

import argparse
import ctypes
import json
import struct
import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np

# Auto-validation support
AUTO_VALIDATE_AVAILABLE = False
try:
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "unittest"))
    from validation.gibberish_detector import detect_gibberish, quick_check
    from validation.auto_validate import AutoValidator
    AUTO_VALIDATE_AVAILABLE = True
except ImportError:
    pass

# Try to import tokenizers (HuggingFace or our GGUF tokenizer)
HF_TOKENIZER_AVAILABLE = False
try:
    from tokenizers import Tokenizer
    HF_TOKENIZER_AVAILABLE = True
except ImportError:
    pass

# Always have GGUF tokenizer available
from gguf_tokenizer import GGUFTokenizer, Tokenizer as GGUFTokenizerWrapper


class _SimpleEncoding:
    """Minimal tokenizers.Encoding-compatible container."""

    def __init__(self, ids: List[int]):
        self.ids = ids


class CKTrueBPETokenizer:
    """Python wrapper around CK true_bpe runtime using binary tokenizer artifacts."""

    def __init__(self, lib_path: Path, bin_dir: Path):
        self.lib_path = Path(lib_path)
        self.bin_dir = Path(bin_dir)
        self._lib = ctypes.CDLL(str(self.lib_path))
        self._bpe = None
        self._vocab_size = 0
        self._num_merges = 0
        self._setup_api()
        self._load_from_binary_artifacts()

    def _setup_api(self) -> None:
        self._lib.ck_true_bpe_create.restype = ctypes.c_void_p
        self._lib.ck_true_bpe_free.argtypes = [ctypes.c_void_p]
        self._lib.ck_true_bpe_load_binary.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self._lib.ck_true_bpe_load_binary.restype = ctypes.c_int
        self._lib.ck_true_bpe_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
        ]
        self._lib.ck_true_bpe_encode.restype = ctypes.c_int
        self._lib.ck_true_bpe_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._lib.ck_true_bpe_decode.restype = ctypes.c_int
        self._lib.ck_true_bpe_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.ck_true_bpe_lookup.restype = ctypes.c_int32
        self._lib.ck_true_bpe_id_to_token.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self._lib.ck_true_bpe_id_to_token.restype = ctypes.c_char_p

    def _load_from_binary_artifacts(self) -> None:
        meta_path = self.bin_dir / "tokenizer_meta.json"
        offsets_path = self.bin_dir / "vocab_offsets.bin"
        strings_path = self.bin_dir / "vocab_strings.bin"
        merges_path = self.bin_dir / "vocab_merges.bin"
        required = [meta_path, offsets_path, strings_path, merges_path]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError("missing tokenizer artifacts: " + ", ".join(missing))

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vocab_size = int(meta.get("vocab_size") or 0)
        num_merges = int(meta.get("num_merges") or 0)
        if vocab_size <= 0 or num_merges < 0:
            raise RuntimeError(f"invalid tokenizer_meta.json values: vocab_size={vocab_size}, num_merges={num_merges}")

        offsets_b = offsets_path.read_bytes()
        merges_b = merges_path.read_bytes()
        strings_b = strings_path.read_bytes()

        expected_offsets = vocab_size * 4
        expected_merges = num_merges * 3 * 4
        if len(offsets_b) != expected_offsets:
            raise RuntimeError(f"bad offsets size: {len(offsets_b)} != {expected_offsets}")
        if len(merges_b) != expected_merges:
            raise RuntimeError(f"bad merges size: {len(merges_b)} != {expected_merges}")

        offsets = list(struct.unpack("<" + ("i" * vocab_size), offsets_b))
        merges = list(struct.unpack("<" + ("i" * (num_merges * 3)), merges_b)) if num_merges > 0 else []

        self._offsets_arr = (ctypes.c_int32 * vocab_size)(*offsets)
        self._merges_arr = (ctypes.c_int32 * (num_merges * 3))(*merges)
        self._strings_buf = ctypes.create_string_buffer(strings_b + b"\x00")

        self._bpe = self._lib.ck_true_bpe_create()
        if not self._bpe:
            raise RuntimeError("ck_true_bpe_create failed")

        rc = self._lib.ck_true_bpe_load_binary(
            self._bpe,
            vocab_size,
            self._offsets_arr,
            ctypes.cast(self._strings_buf, ctypes.c_char_p),
            num_merges,
            self._merges_arr,
        )
        if rc != 0:
            self._lib.ck_true_bpe_free(self._bpe)
            self._bpe = None
            raise RuntimeError(f"ck_true_bpe_load_binary failed rc={rc}")

        self._vocab_size = vocab_size
        self._num_merges = num_merges

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> _SimpleEncoding:
        # true_bpe special-token behavior is configured internally from artifacts.
        _ = add_special_tokens
        if not self._bpe:
            return _SimpleEncoding([])
        text_bytes = text.encode("utf-8")
        max_ids = max(256, len(text_bytes) * 8)
        out = (ctypes.c_int32 * max_ids)()
        n = int(self._lib.ck_true_bpe_encode(self._bpe, text_bytes, -1, out, max_ids))
        if n <= 0:
            return _SimpleEncoding([])
        return _SimpleEncoding([int(out[i]) for i in range(n)])

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        if not self._bpe or not ids:
            return ""
        arr = (ctypes.c_int32 * len(ids))(*[int(x) for x in ids])
        cap = max(256, len(ids) * 32)
        out = ctypes.create_string_buffer(cap)
        n = int(self._lib.ck_true_bpe_decode(self._bpe, arr, len(ids), out, cap))
        if n <= 0:
            return ""
        # Retry with larger buffer if decode appears truncated.
        if n >= cap - 1:
            cap = max(cap * 4, 4096)
            out = ctypes.create_string_buffer(cap)
            n = int(self._lib.ck_true_bpe_decode(self._bpe, arr, len(ids), out, cap))
            if n <= 0:
                return ""
        return out.raw[:n].decode("utf-8", errors="replace")

    def id_to_token(self, token_id: int) -> Optional[str]:
        if not self._bpe:
            return None
        p = self._lib.ck_true_bpe_id_to_token(self._bpe, ctypes.c_int32(int(token_id)))
        if not p:
            return None
        return p.decode("utf-8", errors="replace")

    def lookup_token_id(self, token: str) -> int:
        if not self._bpe:
            return -1
        return int(self._lib.ck_true_bpe_lookup(self._bpe, token.encode("utf-8")))

    def free(self) -> None:
        if self._bpe:
            self._lib.ck_true_bpe_free(self._bpe)
            self._bpe = None


def _is_true_bpe_bin_dir(path: Path) -> bool:
    required = ("tokenizer_meta.json", "vocab_offsets.bin", "vocab_strings.bin", "vocab_merges.bin")
    return path.is_dir() and all((path / name).exists() for name in required)


def _find_true_bpe_bin_dir(model_dir: Path) -> Optional[Path]:
    model_root = model_dir.parent if model_dir.name == ".ck_build" else model_dir

    # Prefer artifacts colocated with the loaded model first.
    preferred = (
        model_dir / "tokenizer_bin",
        model_root / "tokenizer_bin",
        model_dir / "bpe_bin",
        model_root / "bpe_bin",
    )
    for p in preferred:
        if _is_true_bpe_bin_dir(p):
            return p

    # Fallback: latest pipeline artifact if no colocated tokenizer exists.
    pipe_dir = model_root / ".ck_pipeline"
    candidates: List[Path] = []
    if pipe_dir.exists():
        for patt in ("*/tokenizer_bin", "*/bpe_bin"):
            for p in pipe_dir.glob(patt):
                if _is_true_bpe_bin_dir(p):
                    candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_true_bpe_lib(model_dir: Path) -> Optional[Path]:
    model_root = model_dir.parent if model_dir.name == ".ck_build" else model_dir
    project_root = Path(__file__).resolve().parents[1]
    candidates = (
        model_dir / "libckernel_tokenizer.so",
        model_root / "libckernel_tokenizer.so",
        project_root / "build" / "libckernel_tokenizer.so",
    )
    for p in candidates:
        if p.exists():
            return p
    return None


class CKModel:
    """Wrapper for the C model library."""

    # Common EOS token names across different model families
    EOS_TOKEN_NAMES = [
        "<|endoftext|>",      # Qwen, GPT-2
        "<|im_end|>",         # Qwen chat format
        "</s>",               # Llama, Mistral
        "<|eot_id|>",         # Llama 3
        "<eos>",              # Gemma
        "[EOS]",              # Some models
        "<|end|>",            # Phi
    ]

    def __init__(self, model_dir: str, parity_dir: str = None):
        self.model_dir = Path(model_dir)
        self.parity_dir = Path(parity_dir) if parity_dir else None
        self.lib = None
        self.tokenizer = None  # Python tokenizer (fallback)
        self.use_c_tokenizer = False  # Use C tokenizer if available
        self.vocab_size = 0
        self.context_window = 0
        self.has_kv_decode = False
        self.has_parity = False
        self.eos_tokens = set()  # Will be populated during load
        self.logits_stride = None  # Optional: logits stride in floats (0 = last-only)
        self.use_chat_template = True
        self.chat_template_mode = "auto"
        self.default_system_prompt = "You are a helpful assistant."

    def load(self, gguf_path: str = None, force_python_tokenizer: bool = False,
             chat_template: str = "auto") -> bool:
        """Load model library and tokenizer.

        Tokenizer priority (unless force_python_tokenizer=True):
        1. C tokenizer (BPE built into the model library) - fastest, preferred
        2. CK true_bpe via libckernel_tokenizer + tokenizer_bin artifacts
        3. Python HuggingFace tokenizer - compatibility fallback
        4. Python GGUF tokenizer - fallback

        Args:
            gguf_path: Path to GGUF file for tokenizer extraction
            force_python_tokenizer: If True, skip C tokenizer even if available
            chat_template: "auto", "none", or a specific template name (e.g., "qwen")
        """
        # Load C library first (needed to check for C tokenizer)
        lib_path = self.model_dir / "ck-kernel-inference.so"
        if not lib_path.exists():
            lib_path = self.model_dir / "ck-kernel-decode.so"
        if not lib_path.exists():
            lib_path = self.model_dir / "libmodel.so"
        if not lib_path.exists():
            print(f"Error: Model library not found in: {self.model_dir}")
            return False

        self.lib = ctypes.CDLL(str(lib_path))

        # Setup function signatures
        self.lib.ck_model_init.argtypes = [ctypes.c_char_p]
        self.lib.ck_model_init.restype = ctypes.c_int

        self.lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        self.lib.ck_model_embed_tokens.restype = ctypes.c_int

        self.lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self.lib.ck_model_forward.restype = ctypes.c_int

        self.lib.ck_model_get_logits.argtypes = []
        self.lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.ck_model_get_vocab_size.argtypes = []
        self.lib.ck_model_get_vocab_size.restype = ctypes.c_int

        self.lib.ck_model_get_context_window.argtypes = []
        self.lib.ck_model_get_context_window.restype = ctypes.c_int

        self.lib.ck_model_get_active_tokens.argtypes = []
        self.lib.ck_model_get_active_tokens.restype = ctypes.c_int
        # Optional logits stride API (newer v6.6 models)
        try:
            self.lib.ck_model_get_logits_stride.argtypes = []
            self.lib.ck_model_get_logits_stride.restype = ctypes.c_int
            self._has_logits_stride = True
        except AttributeError:
            self._has_logits_stride = False

        self.lib.ck_model_free.argtypes = []
        self.lib.ck_model_free.restype = None

        # Optional KV-cache decode API (newer generated runtimes).
        try:
            self.lib.ck_model_kv_cache_enable.argtypes = [ctypes.c_int]
            self.lib.ck_model_kv_cache_enable.restype = ctypes.c_int
            self.lib.ck_model_kv_cache_reset.argtypes = []
            self.lib.ck_model_kv_cache_reset.restype = None
            self.lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
            self.lib.ck_model_decode.restype = ctypes.c_int
            self.has_kv_decode = True
        except AttributeError:
            self.has_kv_decode = False

        # Optional parity dump API (generated with --parity flag).
        try:
            self.lib.parity_set_output_dir.argtypes = [ctypes.c_char_p]
            self.lib.parity_set_output_dir.restype = None
            self.lib.parity_set_token_index.argtypes = [ctypes.c_int]
            self.lib.parity_set_token_index.restype = None
            self.has_parity = True
        except AttributeError:
            self.has_parity = False

        # Optional C tokenizer API (BPE tokenizer built into the model).
        # If available, we use it instead of Python tokenizers for better performance.
        try:
            self.lib.ck_model_has_tokenizer.argtypes = []
            self.lib.ck_model_has_tokenizer.restype = ctypes.c_int
            self.lib.ck_model_encode_text.argtypes = [ctypes.c_char_p, ctypes.c_int]
            self.lib.ck_model_encode_text.restype = ctypes.c_int
            self.lib.ck_model_decode_tokens.argtypes = [
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.c_char_p, ctypes.c_int
            ]
            self.lib.ck_model_decode_tokens.restype = ctypes.c_int
            self.lib.ck_model_get_token_buffer.argtypes = []
            self.lib.ck_model_get_token_buffer.restype = ctypes.POINTER(ctypes.c_int32)
            self.lib.ck_model_lookup_token.argtypes = [ctypes.c_char_p]
            self.lib.ck_model_lookup_token.restype = ctypes.c_int32
            self._has_c_tokenizer_api = True
        except AttributeError:
            self._has_c_tokenizer_api = False

        # Optional Stop Tokens API (exported from GGUF metadata via codegen).
        # This is the cleanest way to get EOS/BOS tokens - model exports them directly.
        try:
            self.lib.ck_model_get_num_stop_tokens.argtypes = []
            self.lib.ck_model_get_num_stop_tokens.restype = ctypes.c_int
            self.lib.ck_model_get_stop_tokens.argtypes = []
            self.lib.ck_model_get_stop_tokens.restype = ctypes.POINTER(ctypes.c_int32)
            self.lib.ck_model_is_stop_token.argtypes = [ctypes.c_int32]
            self.lib.ck_model_is_stop_token.restype = ctypes.c_int
            self.lib.ck_model_get_eos_token_id.argtypes = []
            self.lib.ck_model_get_eos_token_id.restype = ctypes.c_int32
            self.lib.ck_model_get_bos_token_id.argtypes = []
            self.lib.ck_model_get_bos_token_id.restype = ctypes.c_int32
            self._has_stop_tokens_api = True
        except AttributeError:
            self._has_stop_tokens_api = False

        # Setup parity dir if requested
        if self.parity_dir and self.has_parity:
            self.parity_dir.mkdir(parents=True, exist_ok=True)
            # Keep reference to encoded string to prevent garbage collection
            self._parity_dir_bytes = str(self.parity_dir).encode()
            self.lib.parity_set_output_dir(self._parity_dir_bytes)

        # Initialize model
        weights_path = self.model_dir / "weights.bump"
        if not weights_path.exists():
            print(f"Error: Weights not found: {weights_path}")
            return False

        ret = self.lib.ck_model_init(str(weights_path).encode())
        if ret != 0:
            print(f"Error: Failed to initialize model (code {ret})")
            return False

        # Read logits stride if available (0 = last-only, >0 = full history)
        if self._has_logits_stride:
            try:
                self.logits_stride = int(self.lib.ck_model_get_logits_stride())
            except Exception:
                self.logits_stride = None

        self.vocab_size = self.lib.ck_model_get_vocab_size()
        self.context_window = self.lib.ck_model_get_context_window()

        # Check if C tokenizer is available (preferred - faster)
        if self._has_c_tokenizer_api and self.lib.ck_model_has_tokenizer() and not force_python_tokenizer:
            self.use_c_tokenizer = True
            print(f"Using C BPE tokenizer (built into model)")
        else:
            if force_python_tokenizer:
                print(f"Forcing Python tokenizer (--python-tokenizer flag)")
            # Fall back to Python tokenizer
            self.use_c_tokenizer = False
            # Tokenizer files may live either in model root or .ck_build output dir.
            model_root = self.model_dir.parent if self.model_dir.name == ".ck_build" else self.model_dir
            tokenizer_candidates = [self.model_dir / "tokenizer.json", model_root / "tokenizer.json"]
            vocab_candidates = [self.model_dir / "vocab.json", model_root / "vocab.json"]

            tokenizer_json = next((p for p in tokenizer_candidates if p.exists()), tokenizer_candidates[0])
            vocab_json = next((p for p in vocab_candidates if p.exists()), vocab_candidates[0])

            true_bpe_bin = _find_true_bpe_bin_dir(self.model_dir)
            true_bpe_lib = _find_true_bpe_lib(self.model_dir)
            if true_bpe_bin and true_bpe_lib:
                try:
                    self.tokenizer = CKTrueBPETokenizer(true_bpe_lib, true_bpe_bin)
                    print(f"Loaded CK true_bpe tokenizer from {true_bpe_bin}")
                except Exception as e:
                    print(f"Warning: failed to load CK true_bpe tokenizer ({e})")
                    self.tokenizer = None

            if self.tokenizer is not None:
                pass
            elif tokenizer_json.exists() and HF_TOKENIZER_AVAILABLE:
                # Use HuggingFace tokenizer if available
                self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
                print(f"Loaded HuggingFace tokenizer from {tokenizer_json}")
            elif vocab_json.exists():
                # Use GGUF-compatible wrapper (now supports plain vocab maps too)
                self.tokenizer = GGUFTokenizerWrapper.from_file(str(vocab_json))
                print(f"Loaded GGUF tokenizer from {vocab_json}")
            elif tokenizer_json.exists():
                # Fallback: parse tokenizer.json via GGUF wrapper if tokenizers package is missing.
                self.tokenizer = GGUFTokenizerWrapper.from_file(str(tokenizer_json))
                print(f"Loaded tokenizer via GGUF wrapper from {tokenizer_json}")
            elif gguf_path and Path(gguf_path).exists():
                # Extract directly from GGUF
                print(f"Extracting tokenizer from GGUF: {gguf_path}")
                self.tokenizer = GGUFTokenizerWrapper(GGUFTokenizer.from_gguf(gguf_path))
                # Save for next time
                self.tokenizer._tokenizer.save(str(vocab_json))
                print(f"Saved vocab to {vocab_json}")
            else:
                print(f"Error: No tokenizer found. Tried:")
                print(f"  - C tokenizer (not available)")
                print(f"  - {tokenizer_json}")
                print(f"  - {vocab_json}")
                if gguf_path:
                    print(f"  - {gguf_path}")
                return False

        # Detect EOS tokens from tokenizer
        self._detect_eos_tokens()

        # Configure chat template usage
        self._configure_chat_template(chat_template)

        return True

    def _load_model_meta(self) -> dict:
        """Load chat-related metadata from config/manifest if present."""
        meta = {}
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                meta.update({k: cfg.get(k) for k in ("chat_template", "finetune", "model_name", "name", "model_type", "default_system_prompt")})
            except Exception:
                pass
        manifest_path = self.model_dir / "weights_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                cfg = manifest.get("config", {})
                meta.update({k: cfg.get(k) for k in ("chat_template", "finetune", "model_name", "name", "model_type", "default_system_prompt")})
            except Exception:
                pass
        return meta

    def _configure_chat_template(self, mode: str) -> None:
        """Select chat template based on metadata or explicit override."""
        mode = (mode or "auto").lower()
        self.chat_template_mode = mode

        if mode == "none":
            self.use_chat_template = False
            return
        if mode == "qwen":
            self.use_chat_template = True
            return
        if mode == "gemma":
            self.use_chat_template = True
            return

        # Auto mode: require chat_template metadata + instruct finetune
        meta = self._load_model_meta()
        chat_template = meta.get("chat_template") or ""
        finetune = str(meta.get("finetune") or "").lower()
        model_name = str(meta.get("model_name") or meta.get("name") or "").lower()
        model_type = str(meta.get("model_type") or "").lower()
        default_system = meta.get("default_system_prompt")

        if chat_template and (
            "instruct" in finetune or "chat" in finetune or "instruct" in model_name or "it" in finetune
            or "gemma" in model_type
        ):
            # ChatML-style templates (Qwen)
            if "<|im_start|>" in chat_template and "<|im_end|>" in chat_template:
                self.use_chat_template = True
                self.chat_template_mode = "qwen"
                # Default system prompt behavior:
                # - Qwen2 templates inject a default system prompt if none is provided.
                # - Qwen3 templates do NOT inject a default system prompt.
                # Use model_type (if available) to avoid forcing a system prompt on Qwen3.
                if isinstance(default_system, str) and default_system.strip():
                    self.default_system_prompt = default_system
                elif model_type == "qwen3":
                    self.default_system_prompt = ""
                else:
                    self.default_system_prompt = "You are a helpful assistant."
                return

            # Gemma-style templates
            if "<start_of_turn>" in chat_template and "<end_of_turn>" in chat_template:
                self.use_chat_template = True
                self.chat_template_mode = "gemma"
                if isinstance(default_system, str) and default_system.strip():
                    self.default_system_prompt = default_system
                else:
                    self.default_system_prompt = ""
                return

        # Default: no chat template (base models or unknown templates)
        self.use_chat_template = False
        self.chat_template_mode = "none"

    def _detect_eos_tokens(self):
        """Detect EOS (End-Of-Sequence) token IDs from tokenizer vocabulary.

        WHY THIS MATTERS:
        =================
        When generating text, we need to know when to STOP. EOS tokens signal
        "generation is complete." Without proper EOS detection:

        - Model generates forever (until max_tokens)
        - Or worse: stops immediately on common tokens

        BUG WE HAD (2025-01):
        =====================
        We assumed tokens 0,1,2 are always special (PAD, BOS, EOS) like in GPT-2.
        But in Qwen tokenizer:
            Token 0 = "!"
            Token 1 = '"'
            Token 2 = "#"

        So if model generated "Hello!" -> token "!" (ID=0) -> treated as EOS -> STOP
        Result: Only generated one token before stopping.

        FIX (2026-01): Model now exports stop tokens directly from GGUF metadata!
        Flow: GGUF -> manifest -> IR -> generated code -> ck_model_get_stop_tokens()
        This is the cleanest approach - no guessing, model tells us its stop tokens.

        Fallback: Only use low token IDs as fallback if NO model-specific EOS tokens found.
        """
        self.eos_tokens = set()

        # PREFERRED: Use model's exported stop tokens API (from GGUF metadata)
        # This is the cleanest approach - model tells us its EOS tokens directly.
        if hasattr(self, '_has_stop_tokens_api') and self._has_stop_tokens_api:
            try:
                num_stop = self.lib.ck_model_get_num_stop_tokens()
                if num_stop > 0:
                    stop_tokens_ptr = self.lib.ck_model_get_stop_tokens()
                    for i in range(num_stop):
                        self.eos_tokens.add(stop_tokens_ptr[i])

                # Also get explicit EOS/BOS IDs
                eos_id = self.lib.ck_model_get_eos_token_id()
                if eos_id >= 0:
                    self.eos_tokens.add(eos_id)

                # If we got stop tokens from the model, we're done
                if self.eos_tokens:
                    return
            except Exception:
                pass  # Fall through to tokenizer lookup

        # FALLBACK: Use tokenizer lookup (original method)
        if self.use_c_tokenizer:
            # For C tokenizer, use ck_model_lookup_token to find EOS IDs
            for name in self.EOS_TOKEN_NAMES:
                try:
                    text_bytes = name.encode('utf-8')
                    token_id = self.lib.ck_model_lookup_token(text_bytes)
                    if token_id >= 0:  # Valid token found
                        self.eos_tokens.add(token_id)
                except Exception:
                    pass
        else:
            # Try to get vocab from Python tokenizer
            vocab = None
            try:
                # HuggingFace tokenizer
                if hasattr(self.tokenizer, 'get_vocab'):
                    vocab = self.tokenizer.get_vocab()
                # Our GGUF tokenizer wrapper
                elif hasattr(self.tokenizer, '_tokenizer') and hasattr(self.tokenizer._tokenizer, 'vocab'):
                    vocab = self.tokenizer._tokenizer.vocab
            except Exception:
                pass

            if vocab:
                # Look for known EOS token names (model-specific)
                for name in self.EOS_TOKEN_NAMES:
                    if name in vocab:
                        self.eos_tokens.add(vocab[name])
            else:
                lookup = getattr(self.tokenizer, "lookup_token_id", None)
                if callable(lookup):
                    for name in self.EOS_TOKEN_NAMES:
                        try:
                            token_id = int(lookup(name))
                        except Exception:
                            token_id = -1
                        if token_id >= 0:
                            self.eos_tokens.add(token_id)

        # Model-family specific EOS tokens (when C tokenizer lookup fails)
        # These are hardcoded because special tokens like <|im_end|> may not
        # be findable via direct vocab lookup (encoding/string differences)
        QWEN_EOS_TOKENS = {151643, 151645}  # <|endoftext|>, <|im_end|>
        LLAMA_EOS_TOKENS = {128001, 128009}  # <|end_of_text|>, <|eot_id|>

        # Add model-specific EOS tokens based on vocab size
        if self.vocab_size > 150000:  # Qwen family (151936 vocab)
            self.eos_tokens.update(QWEN_EOS_TOKENS)
        elif self.vocab_size > 127000:  # Llama 3 family (128256 vocab)
            self.eos_tokens.update(LLAMA_EOS_TOKENS)

        # IMPORTANT: Only use low token IDs as fallback if we found NOTHING
        # Different tokenizers assign different meanings to low IDs!
        if not self.eos_tokens:
            self.eos_tokens.update([0, 1, 2])

    def is_eos_token(self, token_id: int) -> bool:
        """Check if token is an EOS token."""
        return token_id in self.eos_tokens

    def kv_cache_enable(self, capacity: Optional[int] = None) -> bool:
        if not self.has_kv_decode:
            return False
        if capacity is None:
            capacity = self.context_window
        ret = self.lib.ck_model_kv_cache_enable(int(capacity))
        return ret == 0

    def kv_cache_reset(self):
        if self.has_kv_decode:
            self.lib.ck_model_kv_cache_reset()

    def set_parity_token_index(self, idx: int):
        """Set current token index for parity dumps."""
        if self.has_parity and self.parity_dir:
            self.lib.parity_set_token_index(idx)

    def encode(self, text: str) -> list:
        """Tokenize text.

        Uses C tokenizer if available (faster), otherwise Python tokenizer.
        """
        if self.use_c_tokenizer:
            # Use C tokenizer - encode directly into model's token buffer
            text_bytes = text.encode('utf-8')
            num_tokens = self.lib.ck_model_encode_text(text_bytes, len(text_bytes))
            # Read tokens back from model's internal buffer
            token_buf_ptr = self.lib.ck_model_get_token_buffer()
            if token_buf_ptr and num_tokens > 0:
                return [token_buf_ptr[i] for i in range(num_tokens)]
            return []
        else:
            return self.tokenizer.encode(text).ids

    def encode_to_buffer(self, text: str) -> int:
        """Encode text directly into model's token buffer (C tokenizer only).

        Returns: number of tokens encoded
        """
        if self.use_c_tokenizer:
            text_bytes = text.encode('utf-8')
            return self.lib.ck_model_encode_text(text_bytes, len(text_bytes))
        else:
            # Fallback: encode with Python, then we'd need to copy to buffer
            # This path shouldn't be hit in normal usage
            token_ids = self.tokenizer.encode(text).ids
            return len(token_ids)

    def decode(self, token_ids: list) -> str:
        """Decode token IDs to text."""
        if self.use_c_tokenizer:
            # Use C tokenizer for decoding
            num_ids = len(token_ids)
            ids_array = (ctypes.c_int32 * num_ids)(*token_ids)
            # Allocate output buffer (generous size)
            out_buf = ctypes.create_string_buffer(num_ids * 16)
            out_len = self.lib.ck_model_decode_tokens(ids_array, num_ids, out_buf, len(out_buf))
            return out_buf.value[:out_len].decode('utf-8', errors='replace')
        else:
            return self.tokenizer.decode(token_ids)

    def token_piece(self, token_id: int) -> Optional[str]:
        """Return raw vocabulary piece for a token ID when Python tokenizer is active."""
        if self.use_c_tokenizer or self.tokenizer is None:
            return None
        tid = int(token_id)

        # HuggingFace tokenizers.Tokenizer API
        id_to_token = getattr(self.tokenizer, "id_to_token", None)
        if callable(id_to_token):
            try:
                piece = id_to_token(tid)
                if isinstance(piece, str):
                    return piece
            except Exception:
                pass

        # GGUF wrapper fallback (scripts/gguf_tokenizer.py)
        inner = getattr(self.tokenizer, "_tokenizer", None)
        toks = getattr(inner, "tokens", None)
        if isinstance(toks, list) and 0 <= tid < len(toks):
            piece = toks[tid]
            if isinstance(piece, str):
                return piece
        return None

    def format_chat_prompt(self, user_message: str, system_prompt: str = None) -> str:
        """Format user message with chat template for instruction models.

        WHY THIS IS NEEDED:
        ===================
        Language models predict "what comes next" - they don't inherently know
        they should answer questions. Without a chat template:

            Input:  "hello"
            Output: Random continuation (could be anything from training data)

        Instruction-tuned models (Qwen-Instruct, Llama-Instruct, etc.) are trained
        on conversations with specific markers. The model learns:
        - <|im_start|>user means "human is talking"
        - <|im_start|>assistant means "I should respond helpfully"

        With chat template:
            Input:  "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
            Output: "Hello! How can I help you today?"

        The weights don't "know" to be helpful - they learned patterns from
        training data that used this exact format.

        FORMAT (ChatML - used by Qwen, and many others):
        ================================================
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {model generates response here...}
        """
        if not self.use_chat_template:
            return user_message

        if system_prompt is None:
            # Qwen ChatML default system prompt can differ by model family.
            # Gemma chat templates are commonly used without one.
            if self.chat_template_mode == "gemma":
                system_prompt = ""
            else:
                system_prompt = self.default_system_prompt

        if self.chat_template_mode == "qwen":
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt

        if self.chat_template_mode == "gemma":
            user_block = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
            prompt = "<bos>"
            prompt += f"<start_of_turn>user\n{user_block}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"
            return prompt

        return user_message

    def forward(self, token_ids: list) -> np.ndarray:
        """Run forward pass and return logits for last position."""
        n = len(token_ids)
        tokens = (ctypes.c_int32 * n)(*token_ids)

        self.lib.ck_model_embed_tokens(tokens, n)
        self.lib.ck_model_forward(None)

        # Get logits pointer
        logits_ptr = self.lib.ck_model_get_logits()
        active_tokens = self.lib.ck_model_get_active_tokens()
        stride = self.logits_stride if self.logits_stride is not None else self.vocab_size
        if stride <= 0:
            logits_array = np.ctypeslib.as_array(logits_ptr, shape=(self.vocab_size,))
            return logits_array.copy()

        # Get last position logits (stride-aware)
        active = max(int(active_tokens), 1)
        last_pos_offset = (active - 1) * stride
        logits_array = np.ctypeslib.as_array(logits_ptr, shape=(active * stride,))
        return logits_array[last_pos_offset:last_pos_offset + self.vocab_size].copy()

    def prefill(self, token_ids: list) -> np.ndarray:
        """Prefill KV cache by running a full forward once (returns last logits)."""
        return self.forward(token_ids)

    def decode_step(self, token_id: int) -> np.ndarray:
        """Decode one token using KV cache (returns logits for that token)."""
        ret = self.lib.ck_model_decode(ctypes.c_int32(int(token_id)), None)
        if ret != 0:
            raise RuntimeError(f"ck_model_decode failed (code {ret})")

        logits_ptr = self.lib.ck_model_get_logits()
        active_tokens = self.lib.ck_model_get_active_tokens()
        stride = self.logits_stride if self.logits_stride is not None else self.vocab_size
        if stride <= 0:
            logits_array = np.ctypeslib.as_array(logits_ptr, shape=(self.vocab_size,))
            return logits_array.copy()

        active = max(int(active_tokens), 1)
        last_pos_offset = (active - 1) * stride
        logits_array = np.ctypeslib.as_array(logits_ptr, shape=(active * stride,))
        return logits_array[last_pos_offset:last_pos_offset + self.vocab_size].copy()

    def free(self):
        """Free model resources."""
        tok_free = getattr(self.tokenizer, "free", None)
        if callable(tok_free):
            tok_free()
        if self.lib:
            self.lib.ck_model_free()


def sample_top_k(logits: np.ndarray, k: int = 40, temperature: float = 0.7) -> int:
    """Sample from top-k logits with temperature."""
    if temperature <= 0:
        return int(np.argmax(logits))

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_indices]

    # Softmax
    max_logit = np.max(top_k_logits)
    exp_logits = np.exp(top_k_logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)

    # Sample
    idx = np.random.choice(len(top_k_indices), p=probs)
    return int(top_k_indices[idx])


def _escape_text_for_display(text: str, ascii_only: bool = False, escape_newlines: bool = False) -> str:
    """Render text safely for terminal display (no control-char side effects)."""
    if not text:
        return ""
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if ch == "\n":
            if escape_newlines:
                out.append("\\n")
            else:
                out.append("\n")
            continue
        if ch == "\r":
            out.append("\\r")
            continue
        if ch == "\t":
            out.append("\\t")
            continue
        if code < 0x20 or code == 0x7F:
            out.append(f"\\x{code:02x}")
            continue
        if ch == "\ufffd":
            out.append("\\uFFFD")
            continue
        if ascii_only and code > 0x7E:
            if code <= 0xFFFF:
                out.append(f"\\u{code:04X}")
            else:
                out.append(f"\\U{code:08X}")
            continue
        out.append(ch)
    return "".join(out)


def _piece_for_debug(piece: str) -> str:
    """Render raw vocab pieces in a byte/escape form for easier debugging."""
    if not piece:
        return ""
    out: List[str] = []
    for ch in piece:
        code = ord(ch)
        if ch == "\n":
            out.append("\\n")
            continue
        if ch == "\r":
            out.append("\\r")
            continue
        if ch == "\t":
            out.append("\\t")
            continue
        if 0x20 <= code <= 0x7E:
            out.append(ch)
            continue
        if code <= 0xFF:
            out.append(f"\\x{code:02X}")
            continue
        if code <= 0xFFFF:
            out.append(f"\\u{code:04X}")
        else:
            out.append(f"\\U{code:08X}")
    return "".join(out)


def generate(model: CKModel, prompt: str, max_tokens: int = 50,
             temperature: float = 0.7, verbose: bool = False,
             show_stats: bool = True,
             validator: Optional['AutoValidator'] = None,
             check_every_n: int = 20,
             no_prefill: bool = False,
             safe_display: bool = True,
             ascii_display: bool = False,
             escape_newlines: bool = False,
             show_token_ids: bool = False,
             show_token_pieces: bool = False) -> str:
    """Generate text from prompt.

    Args:
        validator: Optional AutoValidator for gibberish detection
        check_every_n: Check for gibberish every N tokens
    """
    # Tokenize prompt
    token_ids = model.encode(prompt)
    prompt_tokens = len(token_ids)

    # Track generated tokens for validation
    generated_tokens: List[int] = []
    generated_text: str = ""
    gibberish_detected: bool = False

    if verbose:
        print(f"[Prompt tokens: {prompt_tokens}]")

    generated = []
    sample_times = []
    decode_times = []
    prefill_time = 0.0
    start_time = time.time()

    if model.has_kv_decode and model.kv_cache_enable():
        # Each generate() call is a fresh prompt pass.
        # Reset KV state so prior turns do not leak into the new prefill/decode.
        model.kv_cache_reset()
        # KV-cache path: prefill once, then decode token-by-token.
        t0 = time.time()
        if no_prefill:
            # Slow path: feed prompt tokens via decode to avoid prefill crashes
            logits = None
            for idx, tok in enumerate(token_ids):
                model.set_parity_token_index(idx)
                logits = model.decode_step(tok)
        else:
            model.set_parity_token_index(0)  # Token 0 for prefill output
            logits = model.prefill(token_ids)
        prefill_time = time.time() - t0

        for i in range(max_tokens):
            # Sample
            t_sample = time.time()
            next_token = sample_top_k(logits, k=40, temperature=temperature)
            sample_times.append(time.time() - t_sample)

            if model.is_eos_token(next_token):
                break
            generated.append(next_token)
            generated_tokens.append(next_token)
            token_ids.append(next_token)
            token_text = model.decode([next_token])
            generated_text += token_text
            display_text = _escape_text_for_display(
                token_text, ascii_only=ascii_display, escape_newlines=escape_newlines
            ) if safe_display else token_text
            if show_token_ids:
                if show_token_pieces:
                    piece = model.token_piece(next_token)
                    piece_txt = _piece_for_debug(piece if piece is not None else "?")
                    print(f"<{next_token}|{piece_txt}:{display_text}>", end='', flush=True)
                else:
                    print(f"<{next_token}:{display_text}>", end='', flush=True)
            else:
                print(display_text, end='', flush=True)

            # Periodic gibberish check
            if validator and (i + 1) % check_every_n == 0 and len(generated_tokens) >= 10:
                if AUTO_VALIDATE_AVAILABLE and quick_check(generated_text):
                    result = detect_gibberish(tokens=generated_tokens, text=generated_text,
                                             vocab_size=model.vocab_size)
                    if result.is_gibberish and result.confidence > 0.5:
                        gibberish_detected = True
                        print(f"\n\033[91m[GIBBERISH DETECTED at token {i+1}]\033[0m")
                        break

            if len(token_ids) >= model.context_window - 1:
                break

            # Set parity token index before decode
            model.set_parity_token_index(prompt_tokens + i)

            # Decode step
            t_decode = time.time()
            logits = model.decode_step(next_token)
            decode_times.append(time.time() - t_decode)
    else:
        for i in range(max_tokens):
            # Forward pass (first is prefill, rest are decode)
            t0 = time.time()
            logits = model.forward(token_ids)
            fwd_time = time.time() - t0

            if i == 0:
                prefill_time = fwd_time
            else:
                decode_times.append(fwd_time)

            # Sample next token
            t_sample = time.time()
            next_token = sample_top_k(logits, k=40, temperature=temperature)
            sample_times.append(time.time() - t_sample)

            # Check for EOS token
            if model.is_eos_token(next_token):
                break

            generated.append(next_token)
            generated_tokens.append(next_token)
            token_ids.append(next_token)

            # Decode and print incrementally
            token_text = model.decode([next_token])
            generated_text += token_text
            display_text = _escape_text_for_display(
                token_text, ascii_only=ascii_display, escape_newlines=escape_newlines
            ) if safe_display else token_text
            if show_token_ids:
                if show_token_pieces:
                    piece = model.token_piece(next_token)
                    piece_txt = _piece_for_debug(piece if piece is not None else "?")
                    print(f"<{next_token}|{piece_txt}:{display_text}>", end='', flush=True)
                else:
                    print(f"<{next_token}:{display_text}>", end='', flush=True)
            else:
                print(display_text, end='', flush=True)

            # Periodic gibberish check
            if validator and (i + 1) % check_every_n == 0 and len(generated_tokens) >= 10:
                if AUTO_VALIDATE_AVAILABLE and quick_check(generated_text):
                    result = detect_gibberish(tokens=generated_tokens, text=generated_text,
                                             vocab_size=model.vocab_size)
                    if result.is_gibberish and result.confidence > 0.5:
                        gibberish_detected = True
                        print(f"\n\033[91m[GIBBERISH DETECTED at token {i+1}]\033[0m")
                        break

            # Check context limit
            if len(token_ids) >= model.context_window - 1:
                break

    total_time = time.time() - start_time
    gen_count = len(generated)

    # Final gibberish check and validation
    if validator and not gibberish_detected and len(generated_tokens) >= 10:
        if AUTO_VALIDATE_AVAILABLE:
            result = detect_gibberish(tokens=generated_tokens, text=generated_text,
                                     vocab_size=model.vocab_size)
            if result.is_gibberish and result.confidence > 0.5:
                gibberish_detected = True
                print(f"\n\033[91m[GIBBERISH DETECTED in final output]\033[0m")

    # Run validation if gibberish was detected
    if gibberish_detected and validator and AUTO_VALIDATE_AVAILABLE:
        print()  # newline
        if not validator.check_output(tokens=generated_tokens, text=generated_text,
                                     vocab_size=model.vocab_size):
            validator.run_validation()
            validator.print_debug_instructions()

    # Print statistics (llama.cpp style)
    if show_stats and gen_count > 0:
        print()  # newline after generated text

        # Prefill stats
        prefill_ms = prefill_time * 1000
        prefill_ms_per_token = prefill_ms / prompt_tokens if prompt_tokens > 0 else 0
        prefill_tps = prompt_tokens / prefill_time if prefill_time > 0 else 0

        # Decode stats
        total_decode_time = sum(decode_times) if decode_times else 0
        decode_ms = total_decode_time * 1000
        decode_count = len(decode_times)
        decode_ms_per_token = decode_ms / decode_count if decode_count > 0 else 0
        decode_tps = decode_count / total_decode_time if total_decode_time > 0 else 0

        # Sample stats
        total_sample_time = sum(sample_times) if sample_times else 0
        sample_ms = total_sample_time * 1000
        sample_count = len(sample_times)
        sample_ms_per_token = sample_ms / sample_count if sample_count > 0 else 0

        # Total stats
        total_ms = total_time * 1000
        total_tokens = prompt_tokens + gen_count

        print(f"\n\033[90m" +
              f"prompt eval: {prefill_ms:8.2f} ms / {prompt_tokens:4d} tokens ({prefill_ms_per_token:7.2f} ms/tok, {prefill_tps:7.2f} tok/s)\n" +
              f"      decode: {decode_ms:8.2f} ms / {decode_count:4d} runs   ({decode_ms_per_token:7.2f} ms/tok, {decode_tps:7.2f} tok/s)\n" +
              f"      sample: {sample_ms:8.2f} ms / {sample_count:4d} runs   ({sample_ms_per_token:7.2f} ms/tok)\n" +
              f"       total: {total_ms:8.2f} ms / {total_tokens:4d} tokens\033[0m")

    elif verbose:
        tokens_per_sec = gen_count / total_time if total_time > 0 else 0
        print(f"\n[Generated {gen_count} tokens in {total_time:.2f}s ({tokens_per_sec:.1f} tok/s)]")

    return model.decode(generated)


def chat_loop(model: CKModel, temperature: float = 0.7, max_tokens: int = 100,
              show_stats: bool = True, validator: Optional['AutoValidator'] = None,
              no_prefill: bool = False, safe_display: bool = True,
              ascii_display: bool = False, escape_newlines: bool = False,
              show_token_ids: bool = False,
              show_token_pieces: bool = False):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("  C-Kernel-Engine Chat")
    commands = "/exit, /help, /stats"
    if validator:
        commands += ", /validate"
    print(f"  Type your message and press Enter. Commands: {commands}")
    print("=" * 60 + "\n")

    conversation = []

    while True:
        try:
            user_input = input("\033[92mYou: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
            print("Goodbye!")
            break

        if user_input.lower() == '/help':
            print("  Commands:")
            print("    /exit, /quit  - Exit the chat")
            print("    /stats        - Toggle performance stats display")
            if validator:
                print("    /validate     - Run manual kernel validation")
            print("    /help         - Show this help")
            continue

        if user_input.lower() == '/stats':
            show_stats = not show_stats
            print(f"  Performance stats: {'ON' if show_stats else 'OFF'}")
            continue

        if user_input.lower() == '/validate' and validator:
            print("\n  \033[96mRunning manual validation...\033[0m")
            validator.run_validation()
            validator.print_debug_instructions()
            continue

        # Build prompt with chat template
        prompt = model.format_chat_prompt(user_input)

        # Generate response
        print("\033[94mAssistant: \033[0m", end='', flush=True)
        response = generate(model, prompt, max_tokens=max_tokens,
                          temperature=temperature, verbose=False,
                          show_stats=show_stats, validator=validator,
                          no_prefill=no_prefill,
                          safe_display=safe_display,
                          ascii_display=ascii_display,
                          escape_newlines=escape_newlines,
                          show_token_ids=show_token_ids,
                          show_token_pieces=show_token_pieces)
        print()


def main():
    parser = argparse.ArgumentParser(description="C-Kernel-Engine Chat Interface")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--gguf", help="Path to GGUF file (for tokenizer extraction)")
    parser.add_argument("--prompt", help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--stats", action="store_true", default=True,
                       help="Show performance stats (default: on)")
    parser.add_argument("--no-stats", action="store_false", dest="stats",
                       help="Disable performance stats")
    parser.add_argument("--parity", action="store_true",
                       help="Save intermediate buffers for comparison (requires --parity in codegen)")
    parser.add_argument("--parity-dir",
                       help="Directory for parity dumps (default: model-dir/parity)")
    parser.add_argument("--validate", action="store_true",
                       help="Enable auto-validation: detect gibberish and run staged validation")
    parser.add_argument("--check-every", type=int, default=20,
                       help="Check for gibberish every N tokens (default: 20)")
    parser.add_argument("--no-prefill", action="store_true",
                       help="Disable prefill; feed prompt tokens via decode (slow)")
    parser.add_argument("--python-tokenizer", action="store_true",
                       help="Force Python tokenizer instead of C tokenizer")
    parser.add_argument("--unsafe-display", action="store_true",
                       help="Print raw token text (may include control/invalid characters)")
    parser.add_argument("--ascii-display", action="store_true",
                       help="Escape all non-ASCII output as \\uXXXX sequences")
    parser.add_argument("--escape-newlines", action="store_true",
                       help="Render newline tokens as literal \\n instead of line breaks")
    parser.add_argument("--show-token-ids", action="store_true",
                       help="Print token IDs inline as <id:text>")
    parser.add_argument("--show-token-pieces", action="store_true",
                       help="With --show-token-ids, also show raw vocab piece as <id|piece:text>")
    parser.add_argument("--chat-template", choices=["auto", "none", "qwen", "gemma"], default="auto",
                       help="Chat template mode: auto (from GGUF), none, qwen, or gemma")
    parser.add_argument("--no-chat-template", action="store_true",
                       help="Disable chat template formatting (same as --chat-template=none)")
    args = parser.parse_args()

    # Determine parity directory
    parity_dir = None
    if args.parity:
        parity_dir = args.parity_dir or str(Path(args.model_dir) / "parity")
        print(f"Parity dumps will be saved to: {parity_dir}")

    # Setup validator for auto-validation
    validator = None
    if args.validate:
        if AUTO_VALIDATE_AVAILABLE:
            model_dir = Path(args.model_dir)
            gguf_path = args.gguf or ""
            bump_path = str(model_dir / "weights.bump")
            manifest_path = str(model_dir / "weights_manifest.json")
            validator = AutoValidator(
                gguf_path=gguf_path,
                bump_path=bump_path,
                manifest_path=manifest_path,
                work_dir=str(model_dir),
                verbose=args.verbose
            )
            print(f"\033[96mAuto-validation enabled (checking every {args.check_every} tokens)\033[0m")
        else:
            print("\033[93mWarning: Auto-validation requested but validation module not available\033[0m")

    # Load model
    print(f"Loading model from {args.model_dir}...")
    model = CKModel(args.model_dir, parity_dir=parity_dir)

    chat_template = "none" if args.no_chat_template else args.chat_template
    if not model.load(gguf_path=args.gguf, force_python_tokenizer=args.python_tokenizer,
                      chat_template=chat_template):
        sys.exit(1)

    print(f"Model loaded! Vocab: {model.vocab_size}, Context: {model.context_window}")

    try:
        if args.prompt:
            # Single prompt mode
            print(f"\nPrompt: {args.prompt}")
            print("Response: ", end='', flush=True)
            generate(model, args.prompt, max_tokens=args.max_tokens,
                    temperature=args.temperature, verbose=args.verbose,
                    show_stats=args.stats, validator=validator,
                    check_every_n=args.check_every,
                    no_prefill=args.no_prefill,
                    safe_display=not args.unsafe_display,
                    ascii_display=args.ascii_display,
                    escape_newlines=args.escape_newlines,
                    show_token_ids=args.show_token_ids,
                    show_token_pieces=args.show_token_pieces)
            print()
        else:
            # Interactive chat mode
            chat_loop(model, temperature=args.temperature, max_tokens=args.max_tokens,
                     show_stats=args.stats, validator=validator,
                     no_prefill=args.no_prefill,
                     safe_display=not args.unsafe_display,
                     ascii_display=args.ascii_display,
                     escape_newlines=args.escape_newlines,
                     show_token_ids=args.show_token_ids,
                     show_token_pieces=args.show_token_pieces)
    finally:
        model.free()


if __name__ == "__main__":
    main()
