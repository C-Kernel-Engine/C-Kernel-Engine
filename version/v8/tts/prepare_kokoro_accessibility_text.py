#!/usr/bin/env python3
"""Pinned text conversion and bounded chunk planning for generated Kokoro.

This frontend performs no neural inference. Its output is a list of phoneme-ID
requests for native-kokoro-dispatch; every model stage still runs in generated C.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys


SCHEMA = "cke.kokoro.accessibility-text-plan.v1"
MAX_TEXT_BYTES = 8192
MAX_CHUNKS = 512  # Bounded incremental playback; aggregate WAV stays at 64.
MAX_REQUEST_IDS = 40
PINNED_SHA256 = {
    "config": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    "misaki_en": "a9c36c90a3fa6c6084e9042fd8e19f6049add4f1435d8fe5ca1b77f3529a98f4",
    "kokoro_pipeline": "499982bbc23a1019681cdb4c327c1d911e7034e240721b95686ff1a8f15c2a9a",
    "us_gold": "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f",
    "espeak_library": "b15ce0803b26254e8a0d088c6b9281a2637b2b02613147cc320cfdefaf2c3ccf",
    "espeak_data": "e1a4e7c181fdfb2042e3d7c8da98a4f9f8390d471d833eff9c15340eb9f033bc",
}


class TextPlanError(ValueError):
    """Input cannot be represented safely by this pinned Kokoro frontend."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_file(label: str, path: Path) -> str:
    actual = sha256(path)
    if actual != PINNED_SHA256[label]:
        raise TextPlanError(f"{label} identity mismatch: {actual}")
    return actual


def _checked_espeak_data(root: Path) -> str:
    if not root.is_dir():
        raise TextPlanError("pinned eSpeak data directory unavailable")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    actual = digest.hexdigest()
    if actual != PINNED_SHA256["espeak_data"]:
        raise TextPlanError(f"espeak_data identity mismatch: {actual}")
    return actual


def _source_spans(text: str, tokens: list) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for token in tokens:
        token_text = token.text + token.whitespace
        end = cursor + len(token_text)
        if text[cursor:end] != token_text:
            raise TextPlanError(f"phonemizer changed source text near character {cursor}")
        spans.append((cursor, end))
        cursor = end
    if cursor != len(text):
        raise TextPlanError(f"phonemizer did not cover source text at character {cursor}")
    return spans


def _phonemes(tokens: list) -> str:
    return "".join(token.phonemes + (" " if token.whitespace else "")
                   for token in tokens).strip()


def _boundary_rank(token: object) -> int:
    sound = token.phonemes
    if sound in (".", "!", "?", "…", ";", ":"):
        return 2
    if sound in (",", "—"):
        return 1
    return 0


def _legal_boundary(tokens: list, first: int, end: int) -> bool:
    if end == len(tokens):
        return True
    last, following = tokens[end - 1], tokens[end]
    if following.phonemes in (".", ",", "!", "?", ";", ":") and not last.whitespace:
        return False
    if last.text.strip().casefold() in {"and", "or", "but", "to", "of", "the", "a", "an"}:
        return False
    # A list number and its dot belong to the following list item.
    if (last.text == "." and end - first >= 2 and
            tokens[end - 2].text.strip().isdigit() and
            (tokens[end - 2].text.startswith("\n") or end - first == 2)):
        return False
    return True


def plan_tokens(text: str, tokens: list, vocab: dict[str, int],
                supported_counts: set[int]) -> list[dict]:
    """Choose checked request boundaries from one full-context G2P pass."""
    if not text.strip():
        raise TextPlanError("empty text")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TextPlanError(f"text exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
    if not tokens:
        raise TextPlanError("phonemizer produced no tokens")
    spans = _source_spans(text, tokens)
    for index, token in enumerate(tokens):
        if not isinstance(token.phonemes, str) or not token.phonemes:
            raise TextPlanError(f"unspoken source token {token.text!r} at token {index}")
        missing = sorted(set(token.phonemes) - vocab.keys())
        if missing:
            raise TextPlanError(f"unsupported phonemes {missing!r} for {token.text!r}")
    if not supported_counts or min(supported_counts) < 3:
        raise TextPlanError("bundle has no valid supported phoneme counts")
    if max(supported_counts) > MAX_REQUEST_IDS:
        raise TextPlanError("bundle exceeds the native 40-ID request contract")

    chunks = []
    first = 0
    while first < len(tokens):
        candidates = []
        for end in range(first + 1, len(tokens) + 1):
            ps = _phonemes(tokens[first:end])
            count = len(ps) + 2
            if count > MAX_REQUEST_IDS:
                break
            if count in supported_counts and _legal_boundary(tokens, first, end):
                candidates.append((end, ps, _boundary_rank(tokens[end - 1]), count))
        if not candidates:
            raise TextPlanError(
                f"no supported chunk ending after {tokens[first].text!r}; "
                f"bundle counts={sorted(supported_counts)}"
            )
        # Use the last complete sentence within capacity, then a clause. For
        # an overlong sentence without a legal clause break, balance the two
        # resulting chunks instead of stranding its last word.
        strong = [item for item in candidates if item[2] == 2]
        weak = [item for item in candidates if item[2] == 1 and item[3] >= 20]
        if strong or weak:
            end, ps, _, _ = max(strong or weak, key=lambda item: item[0])
        else:
            natural_end = first + 1
            while (natural_end < len(tokens) and
                   _boundary_rank(tokens[natural_end - 1]) != 2):
                natural_end += 1
            remaining_count = len(_phonemes(tokens[first:natural_end])) + 2
            target = (remaining_count / 2 if remaining_count <= 2 * MAX_REQUEST_IDS
                      else MAX_REQUEST_IDS)
            end, ps, _, _ = min(
                candidates, key=lambda item: (abs(item[3] - target), -item[3]))
        start_char, end_char = spans[first][0], spans[end - 1][1]
        chunk_text = text[start_char:end_char].strip()
        ids = [0] + [vocab[char] for char in ps] + [0]
        chunks.append({
            "index": len(chunks),
            "source_start": start_char,
            "source_end": end_char,
            "source_text": chunk_text,
            "phonemes": ps,
            "phoneme_ids": ids,
            "valid_token_count": len(ids),
            "voice_row": len(ps) - 1,
        })
        if len(chunks) > MAX_CHUNKS:
            raise TextPlanError(f"request requires more than {MAX_CHUNKS} chunks")
        first = end
    if "".join(text[item["source_start"]:item["source_end"]]
               for item in chunks) != text:
        raise TextPlanError("chunk spans did not preserve the complete input")
    return chunks


def prepare(text: str, config_path: Path, bundle_path: Path) -> dict:
    spoken_text = text.strip()
    if not spoken_text:
        raise TextPlanError("empty text")
    if len(spoken_text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise TextPlanError(f"text exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
    _checked_file("config", config_path)
    config = json.loads(config_path.read_text())
    vocab = config.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise TextPlanError("pinned config has no phoneme vocabulary")
    manifest_path = bundle_path / "bundle_manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    counts = manifest.get("supported_phoneme_counts")
    if not isinstance(counts, list) or not all(type(n) is int for n in counts):
        raise TextPlanError("bundle has invalid supported phoneme counts")

    try:
        import espeakng_loader
        from kokoro.pipeline import KPipeline
        from misaki import en
        import spacy
    except ImportError as error:
        raise TextPlanError(f"pinned text frontend dependency unavailable: {error}") from error
    if not spacy.util.is_package("en_core_web_sm"):
        raise TextPlanError("pinned en_core_web_sm package unavailable")
    paths = {
        "misaki_en": Path(inspect.getfile(en)),
        "kokoro_pipeline": Path(inspect.getfile(KPipeline)),
        "us_gold": Path(inspect.getfile(en)).parent / "data/us_gold.json",
        "espeak_library": Path(espeakng_loader.get_library_path()),
    }
    for label, path in paths.items():
        _checked_file(label, path)
    _checked_espeak_data(Path(espeakng_loader.get_data_path()))
    for package, version in (("kokoro", "0.9.4"), ("misaki", "0.9.4"),
                             ("espeakng-loader", "0.2.4")):
        if importlib.metadata.version(package) != version:
            raise TextPlanError(f"pinned {package} version mismatch")
    if importlib.metadata.version("en_core_web_sm") != "3.8.0":
        raise TextPlanError("pinned en_core_web_sm version mismatch")

    pipeline = KPipeline(lang_code="a", model=False,
                         repo_id="hexgrad/Kokoro-82M")
    if pipeline.g2p.fallback is None:
        raise TextPlanError("pinned English G2P fallback unavailable")
    reference_phonemes, tokens = pipeline.g2p(spoken_text)
    if reference_phonemes != _phonemes(tokens):
        raise TextPlanError("pinned phonemizer text and token phonemes disagree")
    chunks = plan_tokens(spoken_text, tokens, vocab, set(counts))
    return {
        "schema": SCHEMA,
        "language": "en-us",
        "voice": "af_heart",
        "input_text": text,
        "source_text": spoken_text,
        "full_reference_phonemes": reference_phonemes,
        "normalizations": (["trimmed boundary whitespace"] if text != spoken_text else []),
        "bundle_manifest_sha256": manifest_hash,
        "bundle_identity_sha256": manifest.get("bundle_identity_sha256"),
        "frontend_sha256": PINNED_SHA256,
        "spacy_model_version": "3.8.0",
        "max_request_ids": MAX_REQUEST_IDS,
        "chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        text = args.text if args.text is not None else args.text_file.read_text()
        plan = prepare(text, args.config, args.bundle_dir)
        serialized = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
    except (OSError, ValueError, KeyError, TextPlanError) as error:
        print(f"Kokoro text preparation failed: {error}", file=sys.stderr)
        return 2
    print(f"prepared {len(plan['chunks'])} checked Kokoro chunks: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
