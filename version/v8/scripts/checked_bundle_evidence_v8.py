"""Checked-bundle warning admission and loaded-artifact identity verification."""

import ctypes
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
WARNING_POLICY = ROOT / "version/v8/contracts/checked_provider_warnings.json"
IDENTITY_SYMBOL = "ck_checked_bundle_build_identity"


def digest_json(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def check_provider_warnings(stderr, *, policy=None):
    policy = json.loads(WARNING_POLICY.read_text()) if policy is None else policy
    if policy.get("schema") != "cke.checked-provider-warnings.v1":
        raise ValueError("unsupported provider warning policy")
    allowed = {}
    for row in policy.get("exceptions", []):
        key = (row["source"], row["option"], row["message"])
        if (
            key in allowed
            or type(row.get("max_count")) is not int
            or row["max_count"] < 1
            or not row.get("reason")
        ):
            raise ValueError("invalid reviewed warning exception")
        allowed[key] = row["max_count"]
    observed = Counter()
    for line in stderr.splitlines():
        if "warning:" not in line:
            continue
        match = re.fullmatch(
            r"(.+?):[0-9]+:[0-9]+: warning: (.*?) \[(-W[^\]]+)\]", line
        )
        if not match:
            raise ValueError("unreviewed compiler warning: " + line)
        source, message, option = match.groups()
        try:
            source = str(Path(source).resolve().relative_to(ROOT))
        except ValueError:
            raise ValueError("compiler warning outside declared source tree") from None
        message = message.replace("‘", "'").replace("’", "'")
        key = source, option, message
        observed[key] += 1
        if observed[key] > allowed.get(key, 0):
            raise ValueError(
                "unreviewed provider warning or increased occurrence count: " + line
            )
    return [
        dict(source=s, option=o, message=m, count=n)
        for (s, o, m), n in sorted(observed.items())
    ]


def verify_loaded_identity(library, expected):
    try:
        getter = getattr(library, IDENTITY_SYMBOL)
    except AttributeError:
        raise ValueError("loaded bundle lacks embedded build identity") from None
    getter.argtypes = []
    getter.restype = ctypes.c_char_p
    value = getter()
    if value != expected.encode("ascii"):
        raise ValueError("loaded checked-bundle identity mismatch")


def load_verified_checked_bundle(library_path, identity_path, expected_identity_sha256):
    raw = Path(identity_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_identity_sha256:
        raise ValueError("checked-bundle metadata hash mismatch")
    identity = json.loads(raw)
    expected = identity.get("build_identity_sha256")
    if (
        not isinstance(expected, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
        or digest_json(identity.get("identity_inputs")) != expected
    ):
        raise ValueError("invalid checked-bundle identity inputs")
    library_path = Path(library_path).resolve()
    if hashlib.sha256(library_path.read_bytes()).hexdigest() != identity.get(
        "library_sha256"
    ):
        raise ValueError("checked-bundle library hash mismatch")
    loaded = ctypes.CDLL(str(library_path))
    verify_loaded_identity(loaded, expected)
    return loaded
