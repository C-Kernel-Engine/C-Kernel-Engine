#!/usr/bin/env python3
"""Plan a long document through CKE's pinned bounded Kokoro text frontend.

Document windows are only a text-capacity envelope. The existing frontend
remains the sole pronunciation and phoneme-ID authority for every window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from prepare_kokoro_accessibility_text import (
    MAX_TEXT_BYTES, TextPlanError, prepare)

SCHEMA = "cke.kokoro.document-text-plan.v1"
MAX_DOCUMENT_BYTES = 1_000_000
MAX_DOCUMENT_CHUNKS = 8192


def text_windows(text: str, *, max_bytes: int = MAX_TEXT_BYTES) -> list[tuple[int, int]]:
    """Choose bounded UTF-8 windows at sentence or whitespace boundaries."""
    if max_bytes < 4:
        raise ValueError("window capacity is too small")
    if not text.strip():
        raise TextPlanError("empty document")
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise TextPlanError(f"document exceeds {MAX_DOCUMENT_BYTES} UTF-8 bytes")
    windows = []
    start = 0
    while start < len(text):
        end = start
        consumed = 0
        while end < len(text):
            width = len(text[end].encode("utf-8"))
            if consumed + width > max_bytes:
                break
            consumed += width
            end += 1
        if end == len(text):
            windows.append((start, end))
            break
        if end == start:
            raise TextPlanError(f"character at source offset {start} exceeds window capacity")
        sentence = [index for index in range(start + 1, end + 1)
                    if text[index - 1] in ".!?\n" and
                    (index == len(text) or text[index].isspace()) and
                    text[start:index].strip()]
        whitespace = [index for index in range(start + 1, end + 1)
                      if text[index - 1].isspace() and text[start:index].strip()]
        cut = (sentence[-1] if sentence else
               whitespace[-1] if whitespace else None)
        if cut is None:
            raise TextPlanError(f"no safe document boundary near source offset {start}:{end}")
        windows.append((start, cut))
        start = cut
    return windows


def prepare_document(text: str, config_path: Path, bundle_path: Path,
                     *, technical_mode: str = "natural",
                     prepare_window=prepare) -> dict:
    windows = text_windows(text)
    chunks = []
    window_rows = []
    expansions = []
    identity = None
    for index, (start, end) in enumerate(windows):
        fragment = text[start:end]
        if not fragment.strip():
            if not chunks:
                raise TextPlanError("document contains no speakable text")
            chunks[-1]["source_end"] = end
            chunks[-1]["source_text"] = text[
                chunks[-1]["source_start"]:end].strip()
            continue
        try:
            local = prepare_window(fragment, config_path, bundle_path,
                                   technical_mode=technical_mode)
        except TextPlanError as error:
            raise TextPlanError(f"document source span {start}:{end}: {error}") from error
        if local.get("source_text") != fragment.strip() or not local.get("chunks"):
            raise TextPlanError(f"window source mismatch at {start}:{end}")
        current_identity = tuple(local.get(key) for key in (
            "language", "voice", "bundle_manifest_sha256",
            "bundle_identity_sha256", "frontend_sha256"))
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise TextPlanError("document frontend or bundle identity changed between windows")
        leading = len(fragment) - len(fragment.lstrip())
        first_chunk = len(chunks)
        for item in local["chunks"]:
            absolute = dict(item)
            absolute["index"] = len(chunks)
            absolute["window_index"] = index
            absolute["source_start"] = start + leading + item["source_start"]
            absolute["source_end"] = start + leading + item["source_end"]
            absolute["source_text"] = text[
                absolute["source_start"]:absolute["source_end"]].strip()
            chunks.append(absolute)
            if len(chunks) > MAX_DOCUMENT_CHUNKS:
                raise TextPlanError(f"document exceeds {MAX_DOCUMENT_CHUNKS} speech chunks")
        chunks[first_chunk]["source_start"] = start
        chunks[first_chunk]["source_text"] = text[
            start:chunks[first_chunk]["source_end"]].strip()
        chunks[-1]["source_end"] = end
        chunks[-1]["source_text"] = text[
            chunks[-1]["source_start"]:end].strip()
        for item in local.get("pronunciation_expansions", []):
            expanded = dict(item)
            expanded["source_start"] += start + leading
            expanded["source_end"] += start + leading
            expansions.append(expanded)
        window_rows.append({"index": index, "source_start": start,
                            "source_end": end, "first_chunk": first_chunk,
                            "chunk_count": len(chunks) - first_chunk})
    if not chunks:
        raise TextPlanError("document contains no speakable text")
    chunks[0]["source_start"] = 0
    chunks[0]["source_text"] = text[:chunks[0]["source_end"]].strip()
    chunks[-1]["source_end"] = len(text)
    chunks[-1]["source_text"] = text[chunks[-1]["source_start"]:].strip()
    if (any(left["source_end"] != right["source_start"]
            for left, right in zip(chunks, chunks[1:])) or
            "".join(text[row["source_start"]:row["source_end"]]
                    for row in chunks) != text):
        raise TextPlanError("document chunks did not preserve the complete source")
    return {"schema": SCHEMA, "source_text": text,
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "language": identity[0], "voice": identity[1],
            "bundle_manifest_sha256": identity[2],
            "bundle_identity_sha256": identity[3],
            "frontend_sha256": identity[4],
            "technical_mode": technical_mode,
            "pronunciation_expansions": expansions,
            "windows": window_rows, "chunks": chunks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--technical-mode", choices=("natural", "exact"),
                        default="natural")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = prepare_document(args.text_file.read_text(encoding="utf-8"),
                                args.config, args.bundle_dir,
                                technical_mode=args.technical_mode)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, TextPlanError, ValueError) as error:
        print(f"Kokoro document preparation failed: {error}", file=sys.stderr)
        return 2
    print(f"prepared {len(plan['chunks'])} checked document chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
