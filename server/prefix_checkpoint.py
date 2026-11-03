"""Byte-budgeted checkpoint cache for serialized HTTP execution.

Only exact rendered token prefixes are reusable. Snapshots contain native
sequence state, not logits or request sampling state. A prepared entry becomes
visible only after its request succeeds; cancellation discards it.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from math import lcm
from time import monotonic
from typing import Any, Callable

from .session_v8 import (
    CK_SESSION_REQUEST_CONTINUE_STATE,
    CK_SESSION_REQUEST_RAW_PROMPT,
    CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION,
    SessionError,
    SequenceCheckpointV8,
)


@dataclass(frozen=True)
class PreparedPrefix:
    prompt: str
    flags: int
    input_tokens: int
    reused_tokens: int
    suffix_tokens: int
    save_seconds: float
    restore_seconds: float
    candidate: tuple[tuple[int, ...], SequenceCheckpointV8] | None = None
    prompt_ids: tuple[int, ...] | None = None
    owner_key: str = "anonymous"
    request_id: str = ""


class ExactPrefixCheckpointCache:
    """One checkpoint per owner; caller serializes it with native work."""

    def __init__(self, session: Any, *, max_bytes: int, identity: str) -> None:
        if type(max_bytes) is not int or max_bytes <= 0 or not identity:
            raise ValueError("prefix cache requires a positive byte budget and identity")
        minimum, alignment, group = session.prefill_contract()
        size, snapshot_alignment = session.checkpoint_requirements()
        allocation_bytes = size + snapshot_alignment
        if allocation_bytes > max_bytes:
            raise ValueError(f"prefix checkpoint needs {allocation_bytes} bytes; budget is {max_bytes}")
        self.session = session
        self.identity = identity
        self.minimum = minimum
        self.step = lcm(alignment, group)
        self.size = size
        self.allocation_bytes = allocation_bytes
        self.snapshot_alignment = snapshot_alignment
        self.max_bytes = max_bytes
        self._entries: OrderedDict[
            str, tuple[tuple[int, ...], SequenceCheckpointV8]
        ] = OrderedDict()
        self._entry_request_ids: dict[str, str] = {}
        self._reserved_bytes = 0
        self.hits = 0
        self.misses = 0

    @property
    def bytes_used(self) -> int:
        return self.allocation_bytes * len(self._entries) + self._reserved_bytes

    def clear(self) -> None:
        self._entries.clear()
        self._entry_request_ids.clear()

    def discard(self, owner_key: str, *, request_id: str | None = None) -> None:
        """Drop an entry when delivery later observes cancellation."""
        if request_id is not None and self._entry_request_ids.get(owner_key) != request_id:
            return
        self._entries.pop(owner_key, None)
        self._entry_request_ids.pop(owner_key, None)

    def prepare(self, prompt: str, *, output_tokens: int, flags: int,
                owner_key: str = "anonymous",
                request_id: str = "",
                cancelled: Callable[[], bool] | None = None) -> PreparedPrefix:
        if not owner_key:
            raise ValueError("prefix cache owner key is empty")
        ids = self.session.encode_ids(prompt)
        if cancelled is not None and cancelled():
            raise SessionError(-9, "request cancelled before prefix preparation")
        ordinary = PreparedPrefix(prompt, flags, len(ids), 0, len(ids), 0.0, 0.0,
                                  owner_key=owner_key, request_id=request_id)
        if (not flags & CK_SESSION_REQUEST_RAW_PROMPT or output_tokens < 1 or
                len(ids) + output_tokens > self.session.context_length):
            self.misses += 1
            return ordinary
        snapshot = None
        entry = self._entries.get(owner_key)
        if entry is not None:
            prefix, snapshot = entry
            if len(ids) > len(prefix) and tuple(ids[:len(prefix)]) == prefix:
                suffix_ids = ids[len(prefix):]
                if cancelled is not None and cancelled():
                    raise SessionError(-9, "request cancelled before checkpoint restore")
                started = monotonic()
                try:
                    self.session.checkpoint_restore(snapshot)
                except SessionError as exc:
                    self.discard(owner_key)
                    if exc.status not in (-1, -5):
                        raise
                    self.misses += 1
                    return ordinary
                if cancelled is not None and cancelled():
                    raise SessionError(-9, "request cancelled after checkpoint restore")
                self.hits += 1
                self._entries.move_to_end(owner_key)
                return PreparedPrefix(
                    "", flags | CK_SESSION_REQUEST_CONTINUE_STATE |
                    CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION,
                    len(ids), len(prefix), len(suffix_ids), 0.0,
                    monotonic() - started, prompt_ids=tuple(suffix_ids),
                    owner_key=owner_key, request_id=request_id)
        # Do not keep an old owner snapshot alive in this stack frame when
        # eviction makes room for its replacement.
        entry = None
        snapshot = None
        # Leave a bounded suffix outside the first snapshot. Harnesses may
        # revise trailing prompt items (for example, removing a first-turn
        # reminder) before appending a tool result. Exact ID comparison still
        # decides whether the checkpoint is reusable. Keep at least two IDs
        # for generation because snapshots contain no logits.
        latest = ((len(ids) - 2) // self.step) * self.step
        earliest = ((self.minimum + self.step - 1) // self.step) * self.step
        if latest < earliest:
            self.misses += 1
            return ordinary
        tail = min(1024, max(2, len(ids) // 4))
        target = ((len(ids) - tail) // self.step) * self.step
        cut = max(earliest, min(latest, target))
        suffix_ids = ids[cut:]
        # Evict before constructing the candidate. Counting only committed
        # entries permits two full snapshots to coexist at a one-entry budget.
        while self._entries and self.bytes_used + self.allocation_bytes > self.max_bytes:
            self.discard(next(iter(self._entries)))
        self._reserved_bytes += self.allocation_bytes
        try:
            if cancelled is not None and cancelled():
                raise SessionError(-9, "request cancelled before prefix prefill")
            candidate = SequenceCheckpointV8(self.size, self.snapshot_alignment)
            self.session.prefill_ids(ids[:cut], reserved_tokens=len(suffix_ids) + output_tokens)
            if cancelled is not None and cancelled():
                raise SessionError(-9, "request cancelled after prefix prefill")
            started = monotonic()
            self.session.checkpoint_save(candidate)
            if cancelled is not None and cancelled():
                raise SessionError(-9, "request cancelled after checkpoint save")
        except SessionError as exc:
            self._reserved_bytes -= self.allocation_bytes
            if exc.status == -9:
                raise
            if exc.status not in (-1, -5):
                raise
            self.misses += 1
            return ordinary
        except BaseException:
            self._reserved_bytes -= self.allocation_bytes
            raise
        save_seconds = monotonic() - started
        self.misses += 1
        return PreparedPrefix(
            "", flags | CK_SESSION_REQUEST_CONTINUE_STATE |
            CK_SESSION_REQUEST_REJECT_PROMPT_TRUNCATION,
            len(ids), 0, len(ids), save_seconds, 0.0,
            (tuple(ids[:cut]), candidate), tuple(suffix_ids), owner_key,
            request_id)

    def settle(self, prepared: PreparedPrefix | None, *, success: bool) -> None:
        if prepared is not None and prepared.candidate is not None:
            self._reserved_bytes -= self.allocation_bytes
        if not success:
            if prepared is not None:
                self.discard(prepared.owner_key)
        elif prepared is not None and prepared.candidate is not None:
            self.discard(prepared.owner_key)
            self._entries[prepared.owner_key] = prepared.candidate
            self._entry_request_ids[prepared.owner_key] = prepared.request_id
