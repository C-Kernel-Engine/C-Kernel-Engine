"""Bounded FP32 supplied-feature recipe admission, shared with local jobs."""

MAX_RECORDS = 4096
MAX_TOKENS = 512
MAX_EPOCHS = 1000
MAX_STEPS = 100000
MAX_DATA_BYTES = 512 * 1024**2
MAX_MANIFEST_BYTES = 8 * 1024**2
MAX_SERVING_ARTIFACTS = 64


def limits():
    return dict(
        epochs=[1, MAX_EPOCHS],
        grad_accum=1,
        records=MAX_RECORDS,
        tokens=MAX_TOKENS,
        steps=MAX_STEPS,
        dataset_bytes=MAX_DATA_BYTES,
        lengths="exact_geometry; padding and mixed lengths unsupported",
    )
