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
import re
import sys
import threading
from types import SimpleNamespace


SCHEMA = "cke.kokoro.accessibility-text-plan.v1"
MAX_TEXT_BYTES = 8192
MAX_CHUNKS = 512  # Shared bound for incremental playback and aggregate WAV.
MAX_REQUEST_IDS = 48
PREFERRED_REQUEST_IDS = 40  # Reserve 41..48 for an indivisible G2P token.
PINNED_SHA256 = {
    "config": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    "misaki_en": "a9c36c90a3fa6c6084e9042fd8e19f6049add4f1435d8fe5ca1b77f3529a98f4",
    "misaki_espeak": "ebc39e807eeae7b99322109d8477aa16b605427f9cc852a4b1f6f72c45806a09",
    "kokoro_pipeline": "499982bbc23a1019681cdb4c327c1d911e7034e240721b95686ff1a8f15c2a9a",
    "us_gold": "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f",
    "espeak_library": "b15ce0803b26254e8a0d088c6b9281a2637b2b02613147cc320cfdefaf2c3ccf",
    "espeak_data": "e1a4e7c181fdfb2042e3d7c8da98a4f9f8390d471d833eff9c15340eb9f033bc",
}
PATH_SEPARATORS = {"/": "slash", "-": "dash", "_": "underscore",
                   ".": "dot"}
EMBEDDED_VERSION = re.compile(r"([A-Za-z]+)([0-9]+\.[0-9]+)\Z")
SIGNED_DECIMAL = re.compile(r"([+-])([0-9]+\.[0-9]+)\Z")
_PINNED_PIPELINE_LOCK = threading.RLock()
_PINNED_PIPELINE = None
_PINNED_PIPELINE_PID = None


class TextPlanError(ValueError):
    """Input cannot be represented safely by this pinned Kokoro frontend."""


def _build_pinned_pipeline(pipeline_type, fallback_type):
    candidate = pipeline_type(lang_code="a", model=False,
                              repo_id="hexgrad/Kokoro-82M")
    fallback = getattr(getattr(candidate, "g2p", None), "fallback", None)
    if not isinstance(fallback, fallback_type):
        raise TextPlanError("pinned English EspeakFallback unavailable")
    return candidate


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


def expand_technical_decimals(tokens: list, pronounce) -> tuple[list, list[dict]]:
    """Say embedded versions and signed decimals with an audible point.

    The pinned G2P currently elides the point in ``Qwen3.8`` and ``+3.5``.
    This declared frontend adaptation retains the original token and span.
    """
    expanded, changes = [], []
    cursor = 0
    for token in tokens:
        raw = token.text
        version = EMBEDDED_VERSION.fullmatch(raw)
        signed = SIGNED_DECIMAL.fullmatch(raw)
        if version:
            major, minor = version.group(2).split(".", 1)
            parts = (version.group(1), major, "point", minor)
        elif signed:
            major, minor = signed.group(2).split(".", 1)
            parts = ("minus" if signed.group(1) == "-" else "plus",
                     major, "point", minor)
        else:
            expanded.append(token)
            cursor += len(raw + token.whitespace)
            continue
        sounds = [pronounce(part) for part in parts]
        if any(not sound for sound in sounds):
            raise TextPlanError(f"decimal pronunciation is incomplete for {raw!r}")
        expanded.append(SimpleNamespace(text=raw, whitespace=token.whitespace,
                                        phonemes=" ".join(sounds)))
        changes.append({"source_start": cursor, "source_end": cursor + len(raw),
                        "source_text": raw, "spoken_text": " ".join(parts),
                        "kind": "embedded_version" if version else "signed_decimal"})
        cursor += len(raw + token.whitespace)
    return expanded, changes


def expand_technical_paths(tokens: list, pronounce,
                           mode: str = "natural") -> tuple[list, list[dict]]:
    """Give path characters explicit speech while retaining exact source spans.

    `pronounce` is the pinned G2P, supplied by prepare(). The same adapter can
    be exercised without model assets using a deterministic test G2P.
    """
    if mode not in ("natural", "exact"):
        raise TextPlanError("technical path mode must be natural or exact")
    expanded, changes = [], []
    cursor = 0
    for index, token in enumerate(tokens):
        raw = token.text
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        if "/" not in raw:
            expanded.append(token)
            cursor += len(raw + token.whitespace)
            continue
        if mode == "exact":
            parts = list(raw)
        else:
            parts = re.findall(r"[A-Za-z0-9]+|[/._-]", raw)
        if "".join(parts) != raw:
            raise TextPlanError(f"unsupported character in technical path {raw!r}")
        root = raw.startswith("/") and (
            len(raw) > 1 or
            (next_token is not None and not token.whitespace and
             "/" in next_token.text))
        spoken = []
        for part in parts:
            if part in PATH_SEPARATORS:
                term = "root slash" if part == "/" and root else PATH_SEPARATORS[part]
                root = False
                pieces = [(part, term)]
            elif mode == "exact":
                pieces = [(part, part.upper() + "." if part.isalpha() else part)]
            else:
                sound = pronounce(part)
                if not sound:
                    raise TextPlanError(f"unspoken technical path component {part!r}")
                pieces = ([(letter, letter.upper() + "." if letter.isalpha()
                            else letter) for letter in part]
                          if len(sound) + 2 > MAX_REQUEST_IDS else [(part, part)])
            for source, word in pieces:
                sound = pronounce(word)
                if not sound or len(sound) + 2 > MAX_REQUEST_IDS:
                    raise TextPlanError(f"unspoken or oversized technical path part {source!r}")
                spoken.append(word)
                expanded.append(SimpleNamespace(text=source, whitespace="",
                                                phonemes=sound + " "))
        expanded[-1].whitespace = token.whitespace
        if token.whitespace:
            expanded[-1].phonemes = expanded[-1].phonemes.rstrip()
        changes.append({"source_start": cursor, "source_end": cursor + len(raw),
                        "source_text": raw, "mode": mode,
                        "spoken_text": " ".join(spoken)})
        cursor += len(raw + token.whitespace)
    return expanded, changes


def _boundary_rank(token: object) -> int:
    sound = token.phonemes
    if sound in (".", "!", "?", "…", ";", ":"):
        return 2
    if sound in (",", "—"):
        return 1
    return 0


_CONNECTORS = {"and", "or", "but", "to", "of", "the", "a", "an"}
_COMPOUND_TENS = {"twenty", "thirty", "forty", "fifty", "sixty",
                  "seventy", "eighty", "ninety"}
_COMPOUND_UNITS = {"one", "two", "three", "four", "five",
                   "six", "seven", "eight", "nine"}


def _legal_boundary(tokens: list, first: int, end: int,
                    *, allow_connector_tail: bool = False) -> bool:
    if end == len(tokens):
        return True
    last, following = tokens[end - 1], tokens[end]
    if following.phonemes in (".", ",", "!", "?", ";", ":") and not last.whitespace:
        return False
    # Splitting an English compound number introduces an audible pause and
    # can make ASR render "forty two" as two separate quantities.
    if ("\n" not in last.whitespace and "\r" not in last.whitespace and
            last.text.strip().casefold() in _COMPOUND_TENS and
            following.text.strip().casefold() in _COMPOUND_UNITS):
        return False
    if not allow_connector_tail and last.text.strip().casefold() in _CONNECTORS:
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
        raise TextPlanError(f"bundle exceeds the native {MAX_REQUEST_IDS}-ID request contract")

    # A locally attractive boundary can strand a connector before a long
    # indivisible number/name. Compute suffix feasibility before selecting
    # each chunk. Relax connector-tail boundaries only when a complete plan
    # requires them; punctuation and list-item guards remain mandatory.
    candidate_cache = []
    for first in range(len(tokens)):
        overlong_token = len(_phonemes(tokens[first:first + 1])) + 2 > PREFERRED_REQUEST_IDS
        request_limit = MAX_REQUEST_IDS if overlong_token else PREFERRED_REQUEST_IDS
        candidates = []
        for end in range(first + 1, len(tokens) + 1):
            ps = _phonemes(tokens[first:end])
            count = len(ps) + 2
            if count > request_limit:
                break
            if count in supported_counts and _legal_boundary(
                    tokens, first, end, allow_connector_tail=True):
                forced_connector = (end < len(tokens) and
                    tokens[end - 1].text.strip().casefold() in _CONNECTORS)
                starts_connector = (end < len(tokens) and
                    tokens[end].text.strip().casefold() in _CONNECTORS)
                candidates.append((end, ps, _boundary_rank(tokens[end - 1]),
                                   count, int(forced_connector), int(starts_connector)))
        candidate_cache.append(candidates)
    best_cost = [None] * (len(tokens) + 1)
    best_cost[-1] = (0, 0)
    for first in range(len(tokens) - 1, -1, -1):
        costs = [(candidate[4] + best_cost[candidate[0]][0],
                  candidate[5] + best_cost[candidate[0]][1])
                 for candidate in candidate_cache[first]
                 if best_cost[candidate[0]] is not None]
        if costs:
            best_cost[first] = min(costs)

    chunks = []
    first = 0
    while first < len(tokens):
        candidates = [candidate for candidate in candidate_cache[first]
                      if best_cost[candidate[0]] is not None and
                      (candidate[4] + best_cost[candidate[0]][0],
                       candidate[5] + best_cost[candidate[0]][1]) == best_cost[first]]
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
        overlong_token = len(_phonemes(tokens[first:first + 1])) + 2 > PREFERRED_REQUEST_IDS
        if overlong_token:
            # Keep a long number/name by itself, attaching punctuation only
            # when the source-span boundary would otherwise be invalid.
            end, ps, _, _ = min(candidates, key=lambda item: item[0])[:4]
        elif strong or weak:
            end, ps, _, _ = max(strong or weak, key=lambda item: item[0])[:4]
        else:
            natural_end = first + 1
            while (natural_end < len(tokens) and
                   _boundary_rank(tokens[natural_end - 1]) != 2):
                natural_end += 1
            remaining_count = len(_phonemes(tokens[first:natural_end])) + 2
            target = (remaining_count / 2 if remaining_count <= 2 * MAX_REQUEST_IDS
                      else MAX_REQUEST_IDS)
            end, ps, _, _ = min(
                candidates, key=lambda item: (abs(item[3] - target), -item[3]))[:4]
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


def prepare(text: str, config_path: Path, bundle_path: Path,
            *, technical_mode: str = "natural") -> dict:
    global _PINNED_PIPELINE, _PINNED_PIPELINE_PID
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
        from misaki import en, espeak
        import spacy
    except ImportError as error:
        raise TextPlanError(f"pinned text frontend dependency unavailable: {error}") from error
    if not spacy.util.is_package("en_core_web_sm"):
        raise TextPlanError("pinned en_core_web_sm package unavailable")
    paths = {
        "misaki_en": Path(inspect.getfile(en)),
        "misaki_espeak": Path(inspect.getfile(espeak)),
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

    # KPipeline silently switches to a neural fallback if constructing a new
    # EspeakFallback fails (for example, when its temporary library copy hits
    # disk quota). Reuse the checked instance within this process and reject
    # any different fallback instead of changing pronunciation mid-session.
    with _PINNED_PIPELINE_LOCK:
        if _PINNED_PIPELINE is None or _PINNED_PIPELINE_PID != os.getpid():
            _PINNED_PIPELINE = _build_pinned_pipeline(
                KPipeline, espeak.EspeakFallback)
            _PINNED_PIPELINE_PID = os.getpid()
        pipeline = _PINNED_PIPELINE
        if not isinstance(pipeline.g2p.fallback, espeak.EspeakFallback):
            raise TextPlanError("pinned English EspeakFallback changed")
        reference_phonemes, tokens = pipeline.g2p(spoken_text)
        if reference_phonemes != _phonemes(tokens):
            raise TextPlanError("pinned phonemizer text and token phonemes disagree")
        tokens, decimal_expansions = expand_technical_decimals(
            tokens, lambda word: pipeline.g2p(word)[0])
        tokens, path_expansions = expand_technical_paths(
            tokens, lambda word: pipeline.g2p(word)[0], technical_mode)
    expansions = decimal_expansions + path_expansions
    chunks = plan_tokens(spoken_text, tokens, vocab, set(counts))
    return {
        "schema": SCHEMA,
        "language": "en-us",
        "voice": "af_heart",
        "input_text": text,
        "source_text": spoken_text,
        "full_reference_phonemes": reference_phonemes,
        "planned_phonemes": _phonemes(tokens),
        "technical_mode": technical_mode,
        "pronunciation_expansions": expansions,
        "normalizations": (["trimmed boundary whitespace"] if text != spoken_text else []) +
                          (["explicit technical decimal pronunciation"]
                           if decimal_expansions else []) +
                          (["explicit technical path pronunciation"]
                           if path_expansions else []),
        "bundle_manifest_sha256": manifest_hash,
        "bundle_identity_sha256": manifest.get("bundle_identity_sha256"),
        "frontend_sha256": PINNED_SHA256,
        "g2p_fallback": "misaki.espeak.EspeakFallback",
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
    parser.add_argument("--technical-mode", choices=("natural", "exact"),
                        default="natural")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        text = args.text if args.text is not None else args.text_file.read_text()
        plan = prepare(text, args.config, args.bundle_dir,
                       technical_mode=args.technical_mode)
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
