"""Detached classification reporting; never model or optimizer arithmetic."""

from collections import Counter
import numpy as np


def classification_metrics(records, evaluated):
    """Per-split, per-question metrics keyed by declared candidate identities.

    A question ID must represent the same task across records. Candidate order
    may change; matrix columns/rows use sorted identities, not local indices.
    Macro-F1 includes every declared class; undefined class F1 is zero.
    """
    groups = {}
    for record in records:
        loss, logits = evaluated[record["id"]]
        logits = np.asarray(logits, dtype=np.float64)
        counts = record["arrays"]["option_counts"]
        if (
            logits.shape != (int(counts.sum()),)
            or not np.isfinite(logits).all()
            or not np.isfinite(loss)
        ):
            raise ValueError("invalid evaluated logits")
        offset = 0
        for q, candidates in enumerate(record["candidate_ids"]):
            scores = logits[offset : offset + len(candidates)]
            expected = candidates[int(record["arrays"]["labels"][q])]
            predicted = candidates[int(np.argmax(scores))]
            group = groups.setdefault(
                (record["split"], record["question_ids"][q]),
                {"classes": set(), "pairs": []},
            )
            group["classes"].update(candidates)
            group["pairs"].append((expected, predicted))
            offset += len(candidates)
    result = {}
    dense_budget = 65536
    for (split, question), group in sorted(groups.items()):
        classes = sorted(group["classes"])
        index = {name: i for i, name in enumerate(classes)}
        pairs = Counter((index[a], index[b]) for a, b in group["pairs"])
        support = np.zeros(len(classes), dtype=np.int64)
        predicted_counts = np.zeros(len(classes), dtype=np.int64)
        correct = np.zeros(len(classes), dtype=np.int64)
        for (a, b), count in pairs.items():
            support[a] += count
            predicted_counts[b] += count
            if a == b:
                correct[a] += count
        denominator = support + predicted_counts
        f1 = np.divide(
            2 * correct, denominator, out=np.zeros(len(classes)), where=denominator != 0
        )
        matrix = None
        if len(classes) ** 2 <= dense_budget:
            dense_budget -= len(classes) ** 2
            matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
            for (a, b), count in pairs.items():
                matrix[a, b] = count
        result.setdefault(split, {})[question] = dict(
            count=len(group["pairs"]),
            accuracy=float(correct.sum() / len(group["pairs"])),
            macro_f1=float(f1.mean()),
            classes=classes,
            support=support.tolist(),
            confusion_matrix=None if matrix is None else matrix.tolist(),
            confusion_entries=[
                [a, b, count] for (a, b), count in sorted(pairs.items())
            ],
            axes="rows=true, columns=predicted",
            zero_division=0,
        )
    return result


def grouped_probabilities(logits, counts):
    """Stable softmax per question; these are scores, not calibrated confidence."""
    logits = np.asarray(logits, dtype=np.float64)
    counts = np.asarray(counts)
    if (
        counts.ndim != 1
        or counts.dtype.kind not in "iu"
        or np.any(counts <= 0)
        or logits.shape != (int(counts.sum()),)
        or not np.isfinite(logits).all()
    ):
        raise ValueError("invalid grouped logits")
    result = []
    offset = 0
    for count in counts:
        scores = logits[offset : offset + int(count)]
        values = np.exp(scores - scores.max())
        result.extend((values / values.sum()).tolist())
        offset += int(count)
    return result
