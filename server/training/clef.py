"""Optional administrator-registered FP32 head recipe; no client filesystem paths."""

from pathlib import Path
import math

from version.v8.decision.training_limits import MAX_EPOCHS, MAX_STEPS

RECIPE = "clef_head_fp32_v1"


def packages(registry):
    from version.v8.decision.training_package import load_package

    result = {}
    if len(registry) > 8:
        raise ValueError("too many registered head packages")
    for name, entry in registry.items():
        from .contract import ident

        if not ident(name) or set(entry) != {"path", "sha256"}:
            raise ValueError("invalid training registration")
        doc, _, records = load_package(entry["path"], entry["sha256"])
        result[name] = dict(
            id=name,
            sha256=entry["sha256"],
            model=doc["model"],
            geometry=doc["geometry"],
            splits={
                split: sum(r["split"] == split for r in records)
                for split in ("train", "validation")
            },
            feature_provenance="supplied",
        )
    return result


def validate(config, available):
    errors = []

    def error(field, code):
        errors.append(dict(field="/config/" + field, code=code))

    fields = {
        "recipe",
        "package_id",
        "package_sha256",
        "dtype",
        "optimizer",
        "method",
        "epochs",
        "learning_rate",
    }
    for field in sorted(fields - set(config)):
        error(field, "REQUIRED")
    for field in sorted(set(config) - fields):
        error(field, "UNKNOWN_FIELD")
    for field, value in (
        ("recipe", RECIPE),
        ("dtype", "fp32"),
        ("optimizer", "adamw"),
        ("method", "head_only"),
    ):
        if config.get(field) != value:
            error(field, "UNSUPPORTED_VALUE")
    name = config.get("package_id")
    package = available.get(name) if type(name) is str else None
    if package is None:
        error("package_id", "UNKNOWN_PACKAGE")
    elif config.get("package_sha256") != package["sha256"]:
        error("package_sha256", "IDENTITY_MISMATCH")
    n = config.get("epochs")
    if type(n) is not int or not 1 <= n <= MAX_EPOCHS:
        error("epochs", "OUT_OF_RANGE")
    elif package is not None and n * package["splits"]["train"] > MAX_STEPS:
        error("epochs", "STEP_BUDGET_EXCEEDED")
    lr = config.get("learning_rate")
    if type(lr) not in (float, int) or not math.isfinite(lr) or not 0 < lr <= 0.01:
        error("learning_rate", "OUT_OF_RANGE")
    return errors
