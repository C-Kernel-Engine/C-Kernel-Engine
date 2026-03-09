#!/usr/bin/env python3
"""
Generate a tiny structured SVG DSL dataset plus a fixed no-merge tokenizer.

The output representation is intentionally NOT raw XML. The point of this toy
is to force compositional learning through stable atoms instead of giant BPE
template pieces.
"""

from __future__ import annotations

import argparse
import itertools
import json
import struct
from dataclasses import dataclass
from pathlib import Path


SYSTEM_TOKENS = ["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"]

PROMPT_TOKENS = [
    "[task:svg]",
    "[OUT]",
    "[shape:circle]",
    "[shape:rect]",
    "[shape:triangle]",
    "[color:red]",
    "[color:blue]",
    "[color:green]",
    "[size:small]",
    "[size:big]",
]

SVG_ATOM_TOKENS = [
    "[svg]",
    "[/svg]",
    "[w:128]",
    "[h:128]",
    "[circle]",
    "[rect]",
    "[polygon]",
    "[cx:64]",
    "[cy:64]",
    "[r:18]",
    "[r:36]",
    "[x:42]",
    "[y:48]",
    "[width:44]",
    "[height:32]",
    "[x:26]",
    "[y:35]",
    "[width:76]",
    "[height:58]",
    "[rx:6]",
    "[points:64,34|36,86|92,86]",
    "[points:64,20|22,98|106,98]",
    "[fill:red]",
    "[fill:blue]",
    "[fill:green]",
    "[stroke:black]",
    "[sw:2]",
]

ASCII_BASE = ["\t", "\n", "\r"] + [chr(i) for i in range(32, 127)]

SHAPES = ("circle", "rect", "triangle")
COLORS = ("red", "blue", "green")
SIZES = ("small", "big")

HOLDOUT_COMBOS = {
    ("circle", "red", "big"),
    ("circle", "blue", "small"),
    ("rect", "green", "big"),
    ("rect", "red", "small"),
    ("triangle", "blue", "big"),
    ("triangle", "green", "small"),
}


@dataclass(frozen=True)
class Combo:
    shape: str
    color: str
    size: str

    @property
    def prompt_tokens(self) -> list[str]:
        return [
            "[task:svg]",
            f"[shape:{self.shape}]",
            f"[color:{self.color}]",
            f"[size:{self.size}]",
            "[OUT]",
        ]

    @property
    def prompt_text(self) -> str:
        return " ".join(self.prompt_tokens)


def _output_tokens(combo: Combo) -> list[str]:
    color = f"[fill:{combo.color}]"
    base = ["[svg]", "[w:128]", "[h:128]"]
    tail = [color, "[stroke:black]", "[sw:2]", "[/svg]"]
    if combo.shape == "circle":
        r = "[r:18]" if combo.size == "small" else "[r:36]"
        return base + ["[circle]", "[cx:64]", "[cy:64]", r] + tail
    if combo.shape == "rect":
        if combo.size == "small":
            geom = ["[rect]", "[x:42]", "[y:48]", "[width:44]", "[height:32]", "[rx:6]"]
        else:
            geom = ["[rect]", "[x:26]", "[y:35]", "[width:76]", "[height:58]", "[rx:6]"]
        return base + geom + tail
    if combo.shape == "triangle":
        points = (
            "[points:64,34|36,86|92,86]"
            if combo.size == "small"
            else "[points:64,20|22,98|106,98]"
        )
        return base + ["[polygon]", points] + tail
    raise RuntimeError(f"unsupported shape: {combo.shape}")


def _row(combo: Combo) -> str:
    return " ".join(combo.prompt_tokens + _output_tokens(combo))


def _render_svg(combo: Combo) -> str:
    width = 128
    height = 128
    stroke = "black"
    stroke_width = 2
    color = combo.color
    if combo.shape == "circle":
        r = 18 if combo.size == "small" else 36
        body = (
            f'<circle cx="64" cy="64" r="{r}" '
            f'fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    elif combo.shape == "rect":
        w = 44 if combo.size == "small" else 76
        h = 32 if combo.size == "small" else 58
        x = (width - w) // 2
        y = (height - h) // 2
        body = (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
            f'fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    elif combo.shape == "triangle":
        points = "64,34 36,86 92,86" if combo.size == "small" else "64,20 22,98 106,98"
        body = (
            f'<polygon points="{points}" '
            f'fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    else:
        raise RuntimeError(f"unsupported shape: {combo.shape}")
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _build_tokenizer_payload(domain_tokens: list[str]) -> tuple[dict, dict]:
    vocab_list: list[str] = []
    seen: set[str] = set()

    def _push(token: str) -> None:
        if token in seen:
            return
        seen.add(token)
        vocab_list.append(token)

    for token in SYSTEM_TOKENS:
        _push(token)
    for token in domain_tokens:
        _push(token)
    for token in ASCII_BASE:
        _push(token)

    vocab = {token: idx for idx, token in enumerate(vocab_list)}
    added_tokens = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in (SYSTEM_TOKENS + domain_tokens)
    ]
    tokenizer = {
        "version": "1.0",
        "ck_mode": "ascii_bpe",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<|unk|>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "vocab": vocab,
            "merges": [],
        },
    }
    meta = {
        "schema": "ck.bpe.binary.v1",
        "mode": "ascii_bpe",
        "vocab_size": len(vocab_list),
        "num_merges": 0,
        "num_reserved_special_tokens": len(domain_tokens),
        "max_piece_bytes": 0,
    }
    return tokenizer, meta


def _write_tokenizer_artifacts(tokenizer: dict, meta: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = out_dir / "tokenizer.json"
    tokenizer_json.write_text(json.dumps(tokenizer, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    vocab = tokenizer["model"]["vocab"]
    sorted_vocab = sorted(vocab.items(), key=lambda item: int(item[1]))
    offsets: list[int] = []
    strings_blob = b""
    current_offset = 0
    for token, _ in sorted_vocab:
        token_bytes = token.encode("utf-8")
        offsets.append(current_offset)
        strings_blob += token_bytes + b"\0"
        current_offset += len(token_bytes) + 1

    bin_dir = out_dir / "tokenizer_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "vocab_offsets.bin").write_bytes(struct.pack("<" + ("i" * len(offsets)), *offsets))
    (bin_dir / "vocab_strings.bin").write_bytes(strings_blob)
    (bin_dir / "vocab_merges.bin").write_bytes(b"")

    meta_doc = dict(meta)
    meta_doc["offsets"] = str((bin_dir / "vocab_offsets.bin").resolve())
    meta_doc["strings"] = str((bin_dir / "vocab_strings.bin").resolve())
    meta_doc["merges"] = str((bin_dir / "vocab_merges.bin").resolve())
    (bin_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta_doc, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return tokenizer_json, bin_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate structured SVG toy dataset with fixed vocab tokenizer")
    ap.add_argument("--out-dir", default="version/v7/data/generated", help="Output directory")
    ap.add_argument("--prefix", default="toy_svg_structured_atoms", help="Output file prefix")
    ap.add_argument("--train-repeats", type=int, default=12, help="Repeats for seen combos")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="Repeats for holdout combos")
    args = ap.parse_args()

    if args.train_repeats < 1:
        raise SystemExit("--train-repeats must be >= 1")
    if args.holdout_repeats < 1:
        raise SystemExit("--holdout-repeats must be >= 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [Combo(shape=s, color=c, size=z) for s, c, z in itertools.product(SHAPES, COLORS, SIZES)]
    holdout = [combo for combo in combos if (combo.shape, combo.color, combo.size) in HOLDOUT_COMBOS]
    train = [combo for combo in combos if (combo.shape, combo.color, combo.size) not in HOLDOUT_COMBOS]

    train_rows = [_row(combo) for combo in train for _ in range(int(args.train_repeats))]
    holdout_rows = [_row(combo) for combo in holdout for _ in range(int(args.holdout_repeats))]
    tokenizer_corpus = [_row(combo) for combo in combos] + [combo.prompt_text for combo in combos]
    seen_prompts = [combo.prompt_text for combo in train]
    holdout_prompts = [combo.prompt_text for combo in holdout]
    render_rows = [
        {
            "prompt": combo.prompt_text,
            "output_tokens": " ".join(_output_tokens(combo)),
            "svg_xml": _render_svg(combo),
            "split": "holdout" if combo in holdout else "train",
        }
        for combo in combos
    ]

    domain_tokens = PROMPT_TOKENS + SVG_ATOM_TOKENS
    tokenizer, tokenizer_meta = _build_tokenizer_payload(domain_tokens)
    tokenizer_json, tokenizer_bin = _write_tokenizer_artifacts(
        tokenizer,
        tokenizer_meta,
        out_dir / f"{args.prefix}_tokenizer",
    )

    _write_lines(out_dir / f"{args.prefix}_train.txt", train_rows)
    _write_lines(out_dir / f"{args.prefix}_holdout.txt", holdout_rows)
    _write_lines(out_dir / f"{args.prefix}_tokenizer_corpus.txt", tokenizer_corpus)
    _write_lines(out_dir / f"{args.prefix}_seen_prompts.txt", seen_prompts)
    _write_lines(out_dir / f"{args.prefix}_holdout_prompts.txt", holdout_prompts)
    _write_lines(out_dir / f"{args.prefix}_reserved_control_tokens.txt", domain_tokens)

    vocab_spec = {
        "format": "svg-structured-fixed-vocab.v1",
        "prefix": args.prefix,
        "system_tokens": SYSTEM_TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "svg_atom_tokens": SVG_ATOM_TOKENS,
        "ascii_base_count": len(ASCII_BASE),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_bin": str(tokenizer_bin),
        "vocab_size": int(tokenizer_meta["vocab_size"]),
        "num_merges": int(tokenizer_meta["num_merges"]),
    }
    (out_dir / f"{args.prefix}_vocab.json").write_text(
        json.dumps(vocab_spec, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / f"{args.prefix}_render_catalog.json").write_text(
        json.dumps(render_rows, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "format": "toy-svg-structured-atoms-manifest.v1",
        "prefix": args.prefix,
        "out_dir": str(out_dir),
        "train_repeats": int(args.train_repeats),
        "holdout_repeats": int(args.holdout_repeats),
        "counts": {
            "unique_total": len(combos),
            "unique_train": len(train),
            "unique_holdout": len(holdout),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "tokenizer_rows": len(tokenizer_corpus),
            "vocab_size": int(tokenizer_meta["vocab_size"]),
            "domain_token_count": len(domain_tokens),
        },
        "artifacts": {
            "train": str(out_dir / f"{args.prefix}_train.txt"),
            "holdout": str(out_dir / f"{args.prefix}_holdout.txt"),
            "tokenizer_corpus": str(out_dir / f"{args.prefix}_tokenizer_corpus.txt"),
            "seen_prompts": str(out_dir / f"{args.prefix}_seen_prompts.txt"),
            "holdout_prompts": str(out_dir / f"{args.prefix}_holdout_prompts.txt"),
            "reserved_control_tokens": str(out_dir / f"{args.prefix}_reserved_control_tokens.txt"),
            "vocab": str(out_dir / f"{args.prefix}_vocab.json"),
            "render_catalog": str(out_dir / f"{args.prefix}_render_catalog.json"),
            "tokenizer_json": str(tokenizer_json),
            "tokenizer_bin": str(tokenizer_bin),
        },
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "prefix": args.prefix,
                "unique_train": len(train),
                "unique_holdout": len(holdout),
                "train_rows": len(train_rows),
                "holdout_rows": len(holdout_rows),
                "vocab_size": int(tokenizer_meta["vocab_size"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
