"""Independent local v2 client validation; no training worker imports."""

import hashlib
import json
import math
import re
import time


class InvalidReply(ValueError):
    pass


def check(value):
    if not value:
        raise InvalidReply("INVALID_OR_STALE_REPLY")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def hexid(value, length):
    return (
        type(value) is str and re.fullmatch("[a-f0-9]{%d}" % length, value) is not None
    )


def validate(reply, request, *, capabilities=None, owner=None, allow_simulation=False):
    check(
        type(reply) is dict
        and set(reply)
        == {
            "schema",
            "request_id",
            "request_sha256",
            "worker_boot_id",
            "capabilities_sha256",
            "ok",
            "result",
            "error",
        }
    )
    check(reply["schema"] == request["schema"] == "cke.training.jobs.v2")
    check(
        reply["request_id"] == request["request_id"]
        and reply["request_sha256"] == digest(request)
    )
    check(reply["ok"] is True and reply["error"] is None)
    check(
        hexid(reply["worker_boot_id"], 32) and hexid(reply["capabilities_sha256"], 64)
    )
    result = reply["result"]
    if request["operation"] == "discover":
        check(type(result) is dict and result.get("schema") == reply["schema"])
        check(
            result.get("worker_boot_id") == reply["worker_boot_id"]
            and result.get("capabilities_sha256") == reply["capabilities_sha256"]
        )
        check(
            digest({k: v for k, v in result.items() if k != "capabilities_sha256"})
            == reply["capabilities_sha256"]
        )
        check(
            result.get("cancel") == "unsupported"
            and result.get("operations")
            == ["discover", "prepare", "submit", "observe", "artifact"]
        )
        check(
            result.get("transport")
            == "trusted_local_only; remote authentication NOT_IMPLEMENTED"
        )
        check(
            result.get("client_revision")
            == "opaque_lowercase_hex_32; preserved verbatim"
        )
        return result
    check(
        type(capabilities) is dict
        and capabilities.get("schema") == "cke.training.jobs.v2"
    )
    check(
        digest({k: v for k, v in capabilities.items() if k != "capabilities_sha256"})
        == capabilities.get("capabilities_sha256")
    )
    check(
        reply["worker_boot_id"]
        == request["worker_boot_id"]
        == capabilities["worker_boot_id"]
    )
    check(
        reply["capabilities_sha256"]
        == request["capabilities_sha256"]
        == capabilities["capabilities_sha256"]
    )
    check(hexid(request["client_revision"], 32))
    if request["operation"] == "prepare":
        check(
            type(result) is dict
            and set(result) == {"plan", "plan_sha256", "submitted"}
            and result["submitted"] is False
        )
        p = result["plan"]
        check(
            type(p) is dict
            and p.get("schema") == "cke.training.execution-plan.v2"
            and digest(p) == result["plan_sha256"]
        )
        check(p.get("owner") == owner and type(owner) is str and bool(owner))
        check(
            p.get("draft_revision") is None
            and p.get("client_revision") == request["client_revision"]
            and p.get("experiment_id") == request["experiment_id"]
        )
        check(
            p.get("worker_boot_id") == reply["worker_boot_id"]
            and p.get("submittable") is True
        )
        check(
            type(p.get("expires_at")) in (float, int)
            and math.isfinite(p["expires_at"])
            and time.time() < p["expires_at"]
        )
        validation = p.get("validation_plan")
        check(
            type(validation) is dict
            and validation.get("config") == request["config"]
            and digest(validation) == p.get("validation_plan_sha256")
        )
        check(
            validation.get("worker_boot_id") == reply["worker_boot_id"]
            and validation.get("capabilities_sha256")
            == capabilities["validation"]["sha256"]
        )
    elif request["operation"] in ("submit", "observe"):
        from .job_consumer import observe

        check(type(result) is dict and type(result.get("job")) is dict)
        j = result["job"]
        if request["operation"] == "submit":
            check(j.get("plan_sha256") == request["plan_sha256"])
        else:
            check(j.get("job_id") == request["job_id"])
        observe(
            result,
            worker_boot_id=reply["worker_boot_id"],
            owner=owner,
            job_id=j["job_id"],
            plan_sha256=j["plan_sha256"],
            experiment_id=j["experiment_id"],
            client_revision=request["client_revision"],
            allow_simulation=allow_simulation,
        )
    elif request["operation"] == "artifact":
        # Full stream verification additionally checks cumulative bytes and SHA-256.
        check(
            type(result) is dict
            and result.get("artifact_id") == request["artifact_id"]
            and result.get("offset") == request["offset"]
        )
    else:
        raise InvalidReply("UNSUPPORTED_OPERATION")
    return result
