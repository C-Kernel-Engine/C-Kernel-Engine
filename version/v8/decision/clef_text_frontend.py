"""Text-only Clef schema encoder, without a PyTorch model dependency.

Adapted from Cloudflare/clef-flash joint_schema_model.py, revision
fb4417d31c1d0f9e7e77f1141b40aaede7491e00 (Apache-2.0).
The reference serialization and span rules are preserved. The public checked
entry rejects truncation and media; tokenization is a declared external frontend.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field as dataclass_field
from typing import Any


def _encode_media(processor, record):
    if record.get("images") or record.get("videos"):
        raise ValueError("text-only frontend; media unsupported")
    return [], None


SYSTEM_PROMPT = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options."

QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted(
            ((str(key), value) for key, value in question["criteria"].items())
        )
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str
    media: dict[str, Any] | None = dataclass_field(
        default=None, compare=False, repr=False
    )


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 16384,
    max_state_tokens: int | None = None,
    processor: Any | None = None,
) -> EncodedRecord:
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(
        record["questions"].items()
    ):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))
        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        for option_index, (option_id, description) in enumerate(
            question_options(question)
        ):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[str(question["type"])],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )
    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    media_ids, media = _encode_media(processor, record)
    if media is not None:
        media["token_offset"] = len(prefix_ids)
        prefix_ids = prefix_ids + media_ids
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(
            f"schema requires {fixed_length} tokens before state; maximum is {max_length}"
        )
    state_ids = state_ids[: max_length - fixed_length]
    schema_offset = len(prefix_ids) + len(state_ids)
    shifted_questions = tuple(
        (
            EncodedQuestion(
                question_id=question.question_id,
                question_type=question.question_type,
                question_span=(
                    question.question_span[0] + schema_offset,
                    question.question_span[1] + schema_offset,
                ),
                option_spans=tuple(
                    (
                        (start + schema_offset, end + schema_offset)
                        for start, end in question.option_spans
                    )
                ),
                option_ids=question.option_ids,
            )
            for question in questions
        )
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not input_ids or not shifted_questions:
        raise ValueError("record produced no model input or questions")
    return EncodedRecord(
        input_ids=input_ids,
        questions=shifted_questions,
        record_id=str(record.get("id", "unknown")),
        media=media,
    )


def prepare(tokenizer, record, *, max_tokens=512):
    from .capture_clef_features import strict_encode
    import sys

    return strict_encode(sys.modules[__name__], tokenizer, record, max_tokens)


class Tokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer as Impl

        self.impl = Impl.from_file(str(path))

    def __call__(self, text, *, add_special_tokens=False):
        from types import SimpleNamespace

        return SimpleNamespace(
            input_ids=self.impl.encode(text, add_special_tokens=add_special_tokens).ids
        )
