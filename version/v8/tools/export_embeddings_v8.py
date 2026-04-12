#!/usr/bin/env python3
"""
Export the token embedding matrix from a C-Kernel-Engine v8 run directory.

Reads weights.bump (using weights_manifest.json for offsets) and tokenizer.json
to produce an embeddings.json file loadable by the dataset_viewer.html Embeddings tab.

Usage:
    python3 version/v8/tools/export_embeddings_v8.py <run_dir>
    python3 version/v8/tools/export_embeddings_v8.py <run_dir> -o /tmp/my_emb.json
    python3 version/v8/tools/export_embeddings_v8.py <run_dir> --checkpoint 300
    python3 version/v8/tools/export_embeddings_v8.py <run_dir> --tensor output.weight

Examples:
    python3 version/v8/tools/export_embeddings_v8.py \\
        ~/.cache/ck-engine-v8/models/train/toy_svg_atoms_ctx512_d64_h128

    python3 version/v8/tools/export_embeddings_v8.py \\
        ~/.cache/ck-engine-v8/models/train/toy_svg_structured_atoms_ctx512_d64_h128 \\
        --vocab version/v8/data/generated/toy_svg_structured_atoms_vocab.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


# ── Token group classification ────────────────────────────────────────────────

def classify_group(token: str) -> str:
    """Assign a semantic group label to a token string."""
    t = token
    if (t.startswith("<|") and t.endswith("|>")) or t in ("<unk>", "<s>", "</s>", "<pad>"):
        return "system"
    if t in ("[OUT]", "[task:svg]", "[task:card]", "[task:chart]", "[task:plot]"):
        return "prompt"
    if t.startswith("[shape:") or t.startswith("[color:") or t.startswith("[size:") \
            or t.startswith("[theme:") or t.startswith("[chart:") or t.startswith("[curve:"):
        return "prompt"
    if t in ("[svg]", "[/svg]") or t.startswith("[svg:") or t.startswith("[/svg"):
        return "svg_structure"
    if t in ("[circle]", "[rect]", "[polygon]", "[ellipse]", "[line]", "[path]", "[g]", "[/g]") \
            or t.startswith("[circle") or t.startswith("[rect") or t.startswith("[polygon") \
            or t.startswith("[card:") or t.startswith("[/card") or t.startswith("[plot:"):
        return "svg_structure"
    if t.startswith("[fill:") or t.startswith("[stroke:") or t.startswith("[sw:") \
            or t.startswith("[opacity:") or t.startswith("[color:"):
        return "svg_style"
    if t.startswith("[cx:") or t.startswith("[cy:") or t.startswith("[r:") \
            or t.startswith("[x:") or t.startswith("[y:") \
            or t.startswith("[w:") or t.startswith("[h:") \
            or t.startswith("[width:") or t.startswith("[height:") \
            or t.startswith("[rx:") or t.startswith("[ry:") \
            or t.startswith("[points:") or t.startswith("[d:"):
        return "svg_attr"
    if t.startswith("[layout:") or t.startswith("[size:"):
        return "svg_attr"
    if t.startswith("[") and t.endswith("]"):
        return "dsl_other"
    if t.startswith("<") and t.endswith(">"):
        return "system"
    return "ascii"


# ── .bump reader ──────────────────────────────────────────────────────────────

def read_f32_tensor(bump_path: Path, entry: dict) -> np.ndarray:
    dtype = entry.get("dtype", "fp32")
    if dtype != "fp32":
        raise ValueError(
            f"Tensor '{entry['name']}' has dtype '{dtype}' — only fp32 is supported. "
            "Quantized weights cannot be visualised as embeddings without dequantization."
        )
    with open(bump_path, "rb") as f:
        f.seek(entry["offset"])
        raw = f.read(entry["size"])
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    return arr.reshape(entry["shape"])


def resolve_paths(run_dir: Path, checkpoint: str | None):
    """Return (bump_path, manifest_path) for the given run/checkpoint."""
    if checkpoint is not None:
        step = int(checkpoint)
        bump_path = run_dir / "checkpoints" / f"weights_step_{step:08d}.bump"
        # Checkpoints don't always carry their own manifest; fall back to run manifest.
        manifest_path = run_dir / "weights_manifest.json"
    else:
        bump_path = run_dir / "weights.bump"
        manifest_path = run_dir / "weights_manifest.json"
    return bump_path, manifest_path


# ── Vocab helpers ─────────────────────────────────────────────────────────────

def load_vocab_from_tokenizer(tokenizer_path: Path) -> dict[int, str]:
    """Extract {token_id → token_string} from a HF-style tokenizer.json."""
    data = json.loads(tokenizer_path.read_text())
    raw = data.get("model", {}).get("vocab", {})
    if not raw:
        raw = data.get("vocab", {})
    if isinstance(raw, dict):
        return {v: k for k, v in raw.items()}
    if isinstance(raw, list):
        return {i: tok for i, tok in enumerate(raw)}
    return {}


def load_vocab_from_spec(spec_path: Path, id_to_token: dict[int, str]) -> dict[int, str]:
    """Merge group info from a spec vocab JSON into the id→token map.
    Returns a dict[int, str] enriched with any extra tokens from the spec."""
    spec = json.loads(spec_path.read_text())
    for key, val in spec.items():
        if not isinstance(val, list):
            continue
        for tok in val:
            if isinstance(tok, str) and tok not in id_to_token.values():
                # Unknown id: append at end
                next_id = max(id_to_token.keys(), default=-1) + 1
                id_to_token[next_id] = tok
    return id_to_token


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Export token embeddings from a v8 run dir to JSON for the dataset viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("run_dir", help="Path to run directory (contains weights.bump + tokenizer.json)")
    p.add_argument("-o", "--output", help="Output path (default: <run_dir>/embeddings.json)")
    p.add_argument("--tensor", default="token_emb",
                   help="Tensor name to export (default: token_emb). Use 'output.weight' for the LM head.")
    p.add_argument("--checkpoint", metavar="STEP",
                   help="Load from a checkpoint step (e.g. --checkpoint 300)")
    p.add_argument("--vocab", metavar="VOCAB_JSON",
                   help="Optional spec vocab JSON to enrich token group labels "
                        "(e.g. version/v8/data/generated/toy_svg_atoms_vocab.json... "
                        "actually that file doesn't exist, use toy_svg_structured_atoms_vocab.json)")
    p.add_argument("--list-tensors", action="store_true",
                   help="List all tensors in the manifest and exit")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"ERROR: Not a directory: {run_dir}", file=sys.stderr)
        return 1

    bump_path, manifest_path = resolve_paths(run_dir, args.checkpoint)

    if not manifest_path.exists():
        print(f"ERROR: Missing manifest: {manifest_path}", file=sys.stderr)
        return 1
    if not bump_path.exists():
        print(f"ERROR: Missing weights file: {bump_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries", [])

    if args.list_tensors:
        print(f"Tensors in {manifest_path.name}:")
        for e in entries:
            shape_str = "×".join(str(d) for d in e.get("shape", []))
            print(f"  {e['name']:<40} {e.get('dtype','fp32'):<6} {shape_str}")
        return 0

    entry = next((e for e in entries if e["name"] == args.tensor), None)
    if entry is None:
        names = [e["name"] for e in entries]
        print(f"ERROR: Tensor '{args.tensor}' not found in manifest.", file=sys.stderr)
        print(f"Available tensors: {names}", file=sys.stderr)
        return 1

    print(f"  tensor : {entry['name']}")
    print(f"  shape  : {entry['shape']}")
    print(f"  dtype  : {entry.get('dtype','fp32')}")
    print(f"  offset : {entry['offset']} bytes")

    emb = read_f32_tensor(bump_path, entry)
    V, D = emb.shape
    print(f"  loaded : {V} tokens × {D} dims  ({emb.nbytes/1024:.1f} KB in memory)")

    # Build vocab
    tokenizer_path = run_dir / "tokenizer.json"
    id_to_token: dict[int, str] = {}
    if tokenizer_path.exists():
        id_to_token = load_vocab_from_tokenizer(tokenizer_path)
        print(f"  vocab  : {len(id_to_token)} tokens from tokenizer.json")
    else:
        print("  vocab  : tokenizer.json not found — using numeric IDs", file=sys.stderr)

    if args.vocab:
        vpath = Path(args.vocab).expanduser()
        if vpath.exists():
            id_to_token = load_vocab_from_spec(vpath, id_to_token)
            print(f"  vocab+ : merged spec vocab from {vpath.name}")

    # Build vocab list (only tokens within embedding table range)
    vocab_list = []
    for tok_id in range(V):
        tok_str = id_to_token.get(tok_id, f"<{tok_id}>")
        vocab_list.append({
            "id": tok_id,
            "token": tok_str,
            "group": classify_group(tok_str),
        })

    # Trim to tokens that actually have string representations (non-padding)
    # Keep all V rows — the viewer handles empty rows gracefully
    matrix_rows = emb.tolist()

    # Stats
    flat = emb.flatten()
    stats = {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "vocab_size": V,
        "embed_dim": D,
        "nonzero_rows": int(np.any(emb != 0, axis=1).sum()),
    }

    step = manifest.get("step")
    out = {
        "format": "ck-embeddings.v1",
        "run_id": run_dir.name,
        "step": step,
        "tensor": args.tensor,
        "shape": [V, D],
        "vocab": vocab_list,
        "matrix": matrix_rows,
        "stats": stats,
    }

    output_path = Path(args.output).expanduser() if args.output else run_dir / "embeddings.json"
    output_path.write_text(json.dumps(out, separators=(",", ":")))
    size_kb = output_path.stat().st_size / 1024
    print(f"\nWritten → {output_path}  ({size_kb:.1f} KB)")
    if size_kb > 5000:
        print("  ⚠  File is large. Consider --checkpoint to use an earlier step,")
        print("     or use --tensor with a specific layer to reduce size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
