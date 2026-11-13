"""Checked local Clef head-training data; supplied features, no backbone executor."""

import hashlib
import json
import re
import math
import zipfile
from pathlib import Path

import numpy as np

MAX_BYTES = 512 * 1024**2
IDENT = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
INPUTS = (
    "hidden",
    "input_ids",
    "question_spans",
    "option_spans",
    "global_span",
    "option_counts",
    "question_type_ids",
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def weight_shapes(g):
    c, w, f, v = (g[k] for k in ("channels", "width", "feedforward", "vocabulary"))
    shapes = {"output_embedding.weight": (v, c), "type_embedding.weight": (3, w)}

    def norm(stem, n):
        shapes[stem + ".weight"] = (n,)
        shapes[stem + ".bias"] = (n,)

    def linear(stem, a, b, bias=True):
        shapes[stem + ".weight"] = (b, a)
        if bias:
            shapes[stem + ".bias"] = (b,)

    def attention(stem):
        shapes[stem + ".in_proj_weight"] = (3 * w, w)
        shapes[stem + ".in_proj_bias"] = (3 * w,)
        linear(stem + ".out_proj", w, w)

    norm("hidden_norm", c)
    for stem in (
        "memory",
        "question",
        "option_question",
        "global",
        "option_context",
        "option_lexical",
    ):
        linear(stem + "_projection", c, w, False)
    for i in range(2):
        stem = f"evidence_layers.{i}"
        for name in ("query_norm", "memory_norm", "feedforward_norm"):
            norm(stem + "." + name, w)
        attention(stem + ".attention")
        linear(stem + ".feedforward.0", w, f)
        linear(stem + ".feedforward.3", f, w)
    for i in range(4):
        stem = f"layers.{i}"
        for name in ("norm1", "norm2", "norm3"):
            norm(stem + "." + name, w)
        for name in ("self_attn", "multihead_attn"):
            attention(stem + "." + name)
        linear(stem + ".linear1", w, f)
        linear(stem + ".linear2", f, w)
    for name in ("option_summary_norm", "field_norm", "option_norm"):
        norm(name, w)
    linear("residual_scorer.0", 4 * w, w)
    linear("residual_scorer.3", w, 1)
    for name in ("prior_logit_scale", "joint_logit_scale", "residual_gate"):
        shapes[name] = ()
    return shapes


def archive(folder, descriptor):
    require(
        type(descriptor) is dict and set(descriptor) == {"file", "sha256"},
        "invalid asset descriptor",
    )
    name = descriptor["file"]
    require(
        type(name) is str and Path(name).name == name and name not in (".", ".."),
        "invalid asset path",
    )
    path = folder / name
    require(
        not path.is_symlink() and path.is_file() and path.stat().st_size <= MAX_BYTES,
        "invalid asset file",
    )
    require(
        type(descriptor["sha256"]) is str
        and HASH.fullmatch(descriptor["sha256"])
        and sha(path) == descriptor["sha256"],
        "asset hash mismatch",
    )
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        require(
            len(entries) <= 256 and len({i.filename for i in entries}) == len(entries),
            "invalid archive inventory",
        )
        require(
            sum(i.file_size for i in entries) <= MAX_BYTES, "expanded archive too large"
        )
        for entry in entries:
            require(
                entry.filename.endswith(".npy") and "/" not in entry.filename,
                "invalid array member",
            )
            with z.open(entry) as f:
                version = np.lib.format.read_magic(f)
                require(version in ((1, 0), (2, 0)), "unsupported array version")
                shape, _, dtype = (
                    np.lib.format.read_array_header_1_0(f)
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0(f)
                )
                require(
                    not dtype.hasobject
                    and dtype.itemsize == 4
                    and len(shape) <= 2
                    and all(type(n) is int and n >= 0 for n in shape),
                    "unsupported array storage",
                )
                require(
                    math.prod(shape) * dtype.itemsize <= entry.file_size - f.tell(),
                    "invalid array extent",
                )
    with np.load(path, allow_pickle=False) as z:
        return {name: np.array(z[name], copy=True) for name in z.files}


def load_package(path, expected_sha256):
    path = Path(path).resolve()
    require(
        path.stat().st_size <= 1024**2 and sha(path) == expected_sha256,
        "package identity mismatch",
    )
    doc = json.loads(path.read_text())
    require(
        set(doc) == {"schema", "model", "geometry", "weights", "records"},
        "invalid package fields",
    )
    require(
        doc["schema"] == "cke.clef.training-package.v1", "unsupported package schema"
    )
    model = doc["model"]
    require(
        type(model) is dict
        and set(model)
        == {
            "id",
            "revision",
            "backbone_id",
            "backbone_revision",
            "tokenizer_revision",
            "feature_provenance",
        },
        "invalid model identity",
    )
    require(
        all(type(model[k]) is str and 0 < len(model[k]) <= 128 for k in model),
        "invalid model identity",
    )
    require(
        IDENT.fullmatch(model["id"]) and model["feature_provenance"] == "supplied",
        "unsupported feature provenance",
    )
    require(
        all(
            re.fullmatch("[0-9a-f]{40,64}", model[k])
            for k in ("revision", "backbone_revision", "tokenizer_revision")
        ),
        "unpinned model identity",
    )
    g = doc["geometry"]
    limits = dict(
        tokens=64,
        channels=4096,
        width=1024,
        heads=16,
        feedforward=4096,
        vocabulary=65536,
        questions=8,
        options=32,
    )
    require(type(g) is dict and set(g) == set(limits), "invalid geometry")
    require(
        all(type(g[k]) is int and 0 < g[k] <= n for k, n in limits.items())
        and g["width"] % g["heads"] == 0,
        "unsupported geometry",
    )
    raw = archive(path.parent, doc["weights"])
    expected = weight_shapes(g)
    require(set(raw) == set(expected), "weight inventory mismatch")
    for name, shape in expected.items():
        a = raw[name]
        require(
            a.dtype == np.float32
            and (a.shape == shape or shape == () and a.shape == (1,))
            and np.isfinite(a).all(),
            "invalid weight: " + name,
        )
    require(
        type(doc["records"]) is list and 2 <= len(doc["records"]) <= 32,
        "record count outside bounds",
    )
    records = []
    seen = set()
    splits = set()
    total = 0
    t, c, q, o = (g[k] for k in ("tokens", "channels", "questions", "options"))
    shapes = dict(
        hidden=(t, c),
        input_ids=(t,),
        question_spans=(q, 2),
        option_spans=(o, 2),
        global_span=(1, 2),
        option_counts=(q,),
        question_type_ids=(q,),
        labels=(q,),
    )
    for record in doc["records"]:
        require(
            set(record) == {"id", "split", "asset", "question_ids", "candidate_ids"},
            "invalid record fields",
        )
        require(
            type(record["id"]) is str
            and IDENT.fullmatch(record["id"])
            and len(record["id"]) <= 60
            and record["id"] not in seen,
            "invalid record identity",
        )
        seen.add(record["id"])
        require(record["split"] in ("train", "validation"), "invalid split")
        splits.add(record["split"])
        arrays = archive(path.parent, record["asset"])
        require(set(arrays) == set(shapes), "record inventory mismatch")
        total += sum(a.nbytes for a in arrays.values())
        require(total <= MAX_BYTES, "dataset too large")
        for name, shape in shapes.items():
            a = arrays[name]
            require(
                a.shape == shape
                and a.dtype == ("float32" if name == "hidden" else "int32")
                and np.isfinite(a).all(),
                "invalid record tensor: " + name,
            )
        require(
            np.all(
                (arrays["input_ids"] >= 0) & (arrays["input_ids"] < g["vocabulary"])
            ),
            "invalid token ID",
        )
        for name in ("question_spans", "option_spans"):
            a = arrays[name]
            require(
                np.all((a[:, 0] >= 0) & (a[:, 0] < a[:, 1]) & (a[:, 1] <= t)),
                "invalid token span",
            )
        require(
            np.array_equal(arrays["global_span"], [[t - 1, t]]),
            "global span must select final token",
        )
        counts = arrays["option_counts"]
        require(np.all(counts > 0) and int(counts.sum()) == o, "invalid option groups")
        require(
            np.all((arrays["labels"] >= 0) & (arrays["labels"] < counts)),
            "invalid label",
        )
        require(
            np.all(
                (arrays["question_type_ids"] >= 0) & (arrays["question_type_ids"] < 3)
            ),
            "invalid question type",
        )
        ids = record["question_ids"]
        candidates = record["candidate_ids"]
        require(
            type(ids) is list
            and len(ids) == q
            and all(type(x) is str and IDENT.fullmatch(x) for x in ids)
            and len(set(ids)) == q,
            "invalid question identities",
        )
        require(
            type(candidates) is list and len(candidates) == q,
            "invalid candidate identities",
        )
        for names, count in zip(candidates, counts):
            require(
                type(names) is list
                and len(names) == count
                and all(type(x) is str and IDENT.fullmatch(x) for x in names)
                and len(set(names)) == len(names),
                "invalid candidate identities",
            )
        records.append(dict(record, arrays=arrays))
    require(splits == {"train", "validation"}, "train and validation splits required")
    return doc, raw, records


def raw_from_effective(original, effective):
    """Reverse packaging only; no neural or optimizer arithmetic."""
    result = {}
    for name, value in original.items():
        if name.endswith(("in_proj_weight", "in_proj_bias")):
            suffix = "in_proj_weight" if name.endswith("weight") else "in_proj_bias"
            leaf = "weight" if suffix.endswith("weight") else "bias"
            result[name] = np.concatenate(
                [
                    effective[name[: -len(suffix)] + axis + "_proj." + leaf]
                    for axis in ("q", "k", "v")
                ]
            ).reshape(value.shape)
        else:
            result[name] = effective[name].reshape(value.shape).copy()
    return result
