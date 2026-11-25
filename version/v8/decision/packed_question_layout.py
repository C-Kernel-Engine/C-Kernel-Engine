"""Pack a shared prefix and isolated question blocks for bounded decision circuits."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    name: str
    tokens: tuple[int, ...]
    options: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PackedQuestionLayout:
    token_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    full_ranges: tuple[tuple[int, int, int, int], ...]
    sliding_ranges: tuple[tuple[int, int, int, int], ...]
    answers: tuple[tuple[str, int, tuple[tuple[str, int], ...]], ...]


def pack_questions(
    prefix: tuple[int, ...],
    questions: tuple[Question, ...],
    *,
    vocab_size: int,
    sliding_window: int,
    capacity: int,
) -> PackedQuestionLayout:
    """Return half-open physical key ranges and branch-local rotary positions.

    Full layers see the shared state and the complete current block. Sliding
    layers see the same logical branch through a causal window. The readout
    row is the last token of that question's suffix.
    """
    if not prefix or not questions:
        raise ValueError("a shared prefix and at least one question are required")
    if any(type(n) is not int or not 0 < n <= 2**31 - 1 for n in (vocab_size, sliding_window, capacity)):
        raise ValueError("vocabulary, window and capacity must be positive int32")
    total = len(prefix) + sum(len(question.tokens) for question in questions)
    if total > capacity:
        raise ValueError("packed request exceeds token capacity")
    if len({q.name for q in questions}) != len(questions):
        raise ValueError("question names must be unique")
    ids = prefix + tuple(token for q in questions for token in q.tokens)
    if any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError("invalid request token ID")

    prefix_end = len(prefix)
    positions = list(range(prefix_end))
    full = [(0, prefix_end, prefix_end, prefix_end)] * prefix_end
    sliding = [(max(0, i + 1 - sliding_window), i + 1, i + 1, i + 1) for i in positions]
    answers = []
    start = prefix_end
    for q in questions:
        if not q.name or not q.tokens or not q.options:
            raise ValueError("question name, suffix and options are required")
        labels, slots = zip(*q.options, strict=True)
        if not all(labels) or len(set(labels)) != len(labels) or len(set(slots)) != len(slots):
            raise ValueError("option names and slot IDs must be distinct per question")
        if any(type(i) is not int or not 0 <= i < vocab_size for i in slots):
            raise ValueError("invalid answer slot token ID")

        end = start + len(q.tokens)
        for physical in range(start, end):
            position = prefix_end + physical - start
            low = max(0, position + 1 - sliding_window)
            positions.append(position)
            full.append((0, prefix_end, start, end))
            sliding.append((min(low, prefix_end), prefix_end, start + max(0, low - prefix_end), physical + 1))
        answers.append((q.name, end - 1, q.options))
        start = end
    return PackedQuestionLayout(ids, tuple(positions), tuple(full), tuple(sliding), tuple(answers))
