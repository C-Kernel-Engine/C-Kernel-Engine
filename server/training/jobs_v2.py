"""Versioned local training lifecycle; authentication belongs to a trusted adapter.

This contract does not open a socket. Never obtain principal from request JSON.
"""

import re

from .contract import digest, encode, ident, MAX_BYTES
from .jobs import require

SCHEMA = "cke.training.jobs.v2"


def hexid(value, size):
    return type(value) is str and re.fullmatch("[a-f0-9]{%d}" % size, value) is not None


def capabilities(jobs, principal):
    body = jobs.capabilities(principal)
    body.update(
        schema=SCHEMA,
        client_revision="opaque_lowercase_hex_32; preserved verbatim",
        freshness="pin worker_boot_id and capabilities_sha256 after discovery",
        transport="trusted_local_only; remote authentication NOT_IMPLEMENTED",
    )
    body["capabilities_sha256"] = digest(body)
    return body


def exchange(jobs, principal, request):
    jobs._auth(principal)
    require(
        type(request) is dict and request.get("schema") == SCHEMA, "INVALID_REQUEST"
    )
    require(ident(request.get("request_id")), "INVALID_REQUEST")
    operation = request.get("operation")
    fields = {
        "discover": set(),
        "prepare": {"config", "experiment_id", "client_revision"},
        "submit": {"plan_sha256", "idempotency_key", "client_revision"},
        "observe": {"job_id", "client_revision"},
        "artifact": {"job_id", "artifact_id", "offset", "client_revision"},
    }
    require(type(operation) is str and operation in fields, "UNSUPPORTED_OPERATION")
    expected = {"schema", "request_id", "operation"} | fields[operation]
    if operation != "discover":
        expected |= {"worker_boot_id", "capabilities_sha256"}
    require(
        set(request) == expected and len(encode(request)) <= MAX_BYTES,
        "INVALID_REQUEST",
    )
    caps = capabilities(jobs, principal)
    if operation != "discover":
        require(
            hexid(request["worker_boot_id"], 32)
            and request["worker_boot_id"] == jobs.backend.boot,
            "STALE_WORKER",
        )
        require(
            hexid(request["capabilities_sha256"], 64)
            and request["capabilities_sha256"] == caps["capabilities_sha256"],
            "STALE_CAPABILITIES",
        )
        require(hexid(request["client_revision"], 32), "INVALID_CLIENT_REVISION")
    if operation == "discover":
        result = caps
    elif operation == "prepare":
        result = jobs.prepare(
            principal,
            request["config"],
            request["experiment_id"],
            client_revision=request["client_revision"],
        )
    elif operation == "submit":
        require(hexid(request["plan_sha256"], 64), "INVALID_PLAN_IDENTITY")
        result = jobs.submit(
            principal,
            request["plan_sha256"],
            request["idempotency_key"],
            client_revision=request["client_revision"],
        )
    else:
        require(hexid(request["job_id"], 32), "UNKNOWN_JOB")
        with jobs.guard:
            job = jobs._read(principal, request["job_id"])
            require(
                job.get("client_revision") == request["client_revision"],
                "CLIENT_REVISION_MISMATCH",
            )
        if operation == "observe":
            result = jobs.observe(principal, request["job_id"])
        else:
            require(hexid(request["artifact_id"], 32), "UNKNOWN_ARTIFACT")
            result = jobs.artifact(
                principal, request["job_id"], request["artifact_id"], request["offset"]
            )
    return dict(
        schema=SCHEMA,
        request_id=request["request_id"],
        request_sha256=digest(request),
        worker_boot_id=jobs.backend.boot,
        capabilities_sha256=caps["capabilities_sha256"],
        ok=True,
        result=result,
        error=None,
    )
