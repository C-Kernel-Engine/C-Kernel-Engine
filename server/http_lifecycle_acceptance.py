#!/usr/bin/env python3
"""Black-box cancellation and recovery acceptance for a running CKE endpoint.

This script uses only HTTP and a declared loaded-artifact identity. It does not
import the Python server or inspect its native session, so the same cases can
certify a future server implementation against the same generated runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx


TERMINALS = {"response.completed", "response.incomplete", "response.cancelled", "response.failed"}
BASE = ""
MODEL = ""
HOST = ""
PORT = 0
PREFILL_LINES = 1500


def request(method: str, path: str, *, payload: dict | None = None, timeout: float = 40.0) -> httpx.Response:
    with httpx.Client(timeout=timeout) as client:
        return client.request(method, BASE + path, json=payload)


def stream_request(prompt: str, *, max_tokens: int, disconnect_at: str | None = None,
                   disconnect_delay: float = 0.0):
    messages: queue.Queue = queue.Queue()
    done = threading.Event()
    state: dict = {"events": [], "error": None, "status_code": None}

    def worker() -> None:
        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                with client.stream(
                    "POST", BASE + "/responses",
                    json={"model": MODEL, "input": prompt, "max_output_tokens": max_tokens,
                          "stream": True, "store": True},
                ) as response:
                    state["status_code"] = response.status_code
                    response.raise_for_status()
                    event = None
                    for line in response.iter_lines():
                        if line.startswith("event: "):
                            event = line[7:]
                        elif line.startswith("data: ") and event:
                            payload = json.loads(line[6:])
                            state["events"].append(event)
                            messages.put((event, payload))
                            if event == disconnect_at:
                                time.sleep(disconnect_delay)
                                break
                            if event in TERMINALS:
                                break
                            event = None
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            messages.put(("error", state["error"]))
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return messages, done, state, thread


def wait_event(messages: queue.Queue, name: str, *, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event, payload = messages.get(timeout=max(0.01, deadline - time.monotonic()))
        if event == "error":
            raise RuntimeError(payload)
        if event == name:
            return payload
        if event in TERMINALS:
            raise RuntimeError(f"terminal {event} arrived before {name}")
    raise TimeoutError(f"missing {name}")


def response_id_from_created(messages: queue.Queue) -> str:
    payload = wait_event(messages, "response.created")
    return payload["response"]["id"]


def followup(label: str) -> dict:
    start = time.monotonic()
    response = request("POST", "/responses", payload={
        "model": MODEL, "input": "Reply with one word: ready", "max_output_tokens": 2,
    })
    result = {"label": label, "http_status": response.status_code,
              "latency_seconds": round(time.monotonic() - start, 3)}
    if response.status_code == 200:
        body = response.json()
        result.update(status=body.get("status"), output_tokens=body.get("usage", {}).get("output_tokens"))
    else:
        result["error"] = response.text[:300]
    if (response.status_code != 200
            or result.get("status") not in {"completed", "incomplete"}
            or not (result.get("output_tokens") or 0) > 0):
        raise RuntimeError(f"{label}: follow-up failed: {result}")
    return result


def run_cancel(label: str, prompt: str, *, wait_for: str, delay: float = 0.0) -> dict:
    messages, done, state, thread = stream_request(prompt, max_tokens=512)
    response_id = response_id_from_created(messages)
    if wait_for != "response.created":
        wait_event(messages, wait_for, timeout=30.0)
    if delay:
        time.sleep(delay)
    start = time.monotonic()
    cancelled = request("POST", f"/responses/{response_id}/cancel", timeout=20.0)
    ack_latency = round(time.monotonic() - start, 3)
    if not done.wait(25.0):
        raise TimeoutError(f"{label}: stream worker did not finish")
    thread.join(timeout=1.0)
    stored = request("GET", f"/responses/{response_id}")
    row = {"label": label, "response_id": response_id,
           "trigger_event": wait_for,
           "phase_evidence": (
               "output_delta_observed" if wait_for == "response.output_text.delta"
               else "in_progress_event_only_native_phase_unverified"
           ),
           "cancel_http_status": cancelled.status_code,
           "cancel_ack_seconds": ack_latency,
           "stream_events": state["events"][-6:], "stream_error": state["error"],
           "stored_http_status": stored.status_code,
           "stored_status": stored.json().get("status") if stored.status_code == 200 else None}
    if (cancelled.status_code != 200 or stored.status_code != 200
            or row["stored_status"] != "cancelled" or state["error"] is not None
            or "response.cancelled" not in state["events"]):
        raise RuntimeError(f"{label}: cancellation not acknowledged: {row}")
    row["followup"] = followup(label + "-followup")
    stale = request("POST", f"/responses/{response_id}/cancel")
    row["stale_cancel_http_status"] = stale.status_code
    if stale.status_code != 409:
        raise RuntimeError(f"{label}: stale cancellation not rejected: {row}")
    return row


def run_disconnect(label: str, prompt: str) -> dict:
    messages, done, state, thread = stream_request(prompt, max_tokens=512,
                                                   disconnect_at="response.in_progress",
                                                   disconnect_delay=3.0)
    response_id = response_id_from_created(messages)
    if not done.wait(10.0):
        raise TimeoutError(f"{label}: client did not disconnect")
    thread.join(timeout=1.0)
    start = time.monotonic()
    errors = []
    while time.monotonic() - start < 30.0:
        try:
            next_result = followup(label + "-followup")
            stored = request("GET", f"/responses/{response_id}")
            stored_status = stored.json().get("status")
            if stored_status != "cancelled":
                raise RuntimeError(f"{label}: stored status is {stored_status!r}")
            return {"label": label, "response_id": response_id,
                    "stream_events": state["events"], "stream_error": state["error"],
                    "time_to_followup_completion_seconds": round(time.monotonic() - start, 3),
                    "followup": next_result, "stored_status": stored_status,
                    "busy_attempts": len(errors)}
        except RuntimeError as exc:
            errors.append(str(exc))
            time.sleep(0.5)
    raise RuntimeError(f"{label}: no follow-up recovery within 30s: {errors[-2:]}")


def run_client_timeout(label: str, prompt: str) -> dict:
    payload = json.dumps({"model": MODEL, "input": prompt, "max_output_tokens": 512,
                          "stream": True, "store": True}).encode()
    connection = socket.create_connection((HOST, PORT), timeout=30.0)
    try:
        connection.sendall(
            b"POST /v1/responses HTTP/1.1\r\nHost: "
            + f"{HOST}:{PORT}".encode() + b"\r\n"
            b"Content-Type: application/json\r\nContent-Length: "
            + str(len(payload)).encode() + b"\r\n\r\n" + payload
        )
        received = b""
        while b"response.in_progress" not in received:
            chunk = connection.recv(4096)
            if not chunk:
                raise RuntimeError(f"{label}: stream closed before in-progress")
            received += chunk
        match = re.search(rb"resp_[0-9a-f]{24}", received)
        if match is None:
            raise RuntimeError(f"{label}: response id missing")
        response_id = match.group().decode()
        deadline = time.monotonic() + 0.3
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            connection.settimeout(remaining)
            try:
                chunk = connection.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                raise RuntimeError(f"{label}: server closed before the client deadline")
    finally:
        connection.close()
    start = time.monotonic()
    errors = []
    while time.monotonic() - start < 30.0:
        try:
            result = followup(label + "-followup")
            stored = request("GET", f"/responses/{response_id}")
            row = {"label": label, "response_id": response_id, "client_timeout_seconds": 0.3,
                   "time_to_followup_completion_seconds": round(time.monotonic() - start, 3),
                   "followup": result, "stored_status": stored.json().get("status"),
                   "busy_attempts": len(errors)}
            if row["stored_status"] != "cancelled":
                raise RuntimeError(f"{label}: timed-out response did not become cancelled: {row}")
            return row
        except RuntimeError as exc:
            errors.append(str(exc))
            time.sleep(0.5)
    raise RuntimeError(f"{label}: no follow-up recovery within 30s: {errors[-2:]}")


def _check_identity(identity: dict, *, expected_serving: str,
                    expected_session: str, model: str) -> None:
    for key, expected in (
        ("serving_identity", expected_serving),
        ("session_library_sha256", expected_session),
        ("model", model),
    ):
        if identity.get(key) != expected:
            raise RuntimeError(f"loaded identity {key} differs from the required artifact")
    if not identity.get("server_instance_id"):
        raise RuntimeError("loaded identity has no server instance ID")


def _sse_events(body: str) -> list[tuple[str, dict]]:
    """Parse complete SSE events while ignoring comments and keep-alives."""
    events = []
    event = None
    data = []
    for line in body.splitlines() + [""]:
        if not line:
            if event is not None:
                if not data:
                    raise RuntimeError(f"SSE event {event} has no data")
                payload = json.loads("\n".join(data))
                if payload.get("type") != event:
                    raise RuntimeError(f"SSE event {event} has a mismatched type")
                events.append((event, payload))
            event, data = None, []
        elif line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data.append(line[6:])
    return events


def _plain_chat_request(reply: str, label: str, cases: list[dict]) -> None:
    response = request("POST", "/chat/completions", payload={
        "model": MODEL, "messages": [{"role": "user", "content": f"Reply with exactly: {reply}"}],
        "max_tokens": 16, "temperature": 0,
    })
    row = {"label": label, "http_status": response.status_code}
    cases.append(row)
    if response.status_code != 200:
        row["error"] = response.text[:300]
        raise RuntimeError(f"{label}: chat request failed: HTTP {response.status_code}")
    body = response.json()
    choices = body.get("choices") or []
    choice = choices[0] if len(choices) == 1 else {}
    row.update(output=choice.get("message", {}).get("content"),
               finish_reason=choice.get("finish_reason"),
               output_tokens=body.get("usage", {}).get("completion_tokens"))
    row["protocol_status"] = (choice.get("finish_reason") == "stop"
                              and isinstance(row["output"], str)
                              and bool(row["output"])
                              and (row["output_tokens"] or 0) > 0)
    row["answer_status"] = row["output"] == reply
    if not row["protocol_status"] or not row["answer_status"]:
        raise RuntimeError(f"{label}: incomplete or unexpected answer: {row}")


def run_plain_chat(expected_reply: str, identity: dict, cases: list[dict]) -> None:
    """Exercise both HTTP routes and recovery without assuming a tool protocol."""
    _plain_chat_request(expected_reply, "chat-completions", cases)
    streamed = request("POST", "/responses", payload={
        "model": MODEL, "input": f"Reply with exactly: {expected_reply}",
        "max_output_tokens": 16, "temperature": 0, "stream": True,
    })
    row = {"label": "responses-stream", "http_status": streamed.status_code}
    cases.append(row)
    if streamed.status_code != 200:
        row["error"] = streamed.text[:300]
        raise RuntimeError(f"responses-stream: HTTP {streamed.status_code}")
    events = _sse_events(streamed.text)
    names = [name for name, _ in events]
    terminal = [(name, payload) for name, payload in events if name in TERMINALS]
    deltas = [payload.get("delta") for name, payload in events
              if name == "response.output_text.delta"]
    completed = terminal[0][1].get("response", {}) if len(terminal) == 1 else {}
    row.update(events=names, output=completed.get("output_text"),
               terminal=terminal[0][0] if len(terminal) == 1 else None,
               output_tokens=completed.get("usage", {}).get("output_tokens"))
    row["protocol_status"] = (len(terminal) == 1 and names[-1] == "response.completed"
                              and names[0] == "response.created"
                              and completed.get("status") == "completed"
                              and bool(deltas) and "".join(deltas) == row["output"]
                              and (row["output_tokens"] or 0) > 0)
    row["answer_status"] = row["output"] == expected_reply
    if not row["protocol_status"] or not row["answer_status"]:
        raise RuntimeError(f"responses-stream: incomplete or unexpected answer: {row}")

    multi = request("POST", "/chat/completions", payload={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "The codeword is cobalt. Acknowledge with okay."},
            {"role": "assistant", "content": "okay"},
            {"role": "user", "content": "What was the codeword? Reply with only the word."},
        ],
        "max_tokens": 16, "temperature": 0,
    })
    row = {"label": "multi-turn", "http_status": multi.status_code}
    cases.append(row)
    if multi.status_code != 200:
        row["error"] = multi.text[:300]
        raise RuntimeError(f"multi-turn: HTTP {multi.status_code}")
    multi_body = multi.json()
    multi_choices = multi_body.get("choices") or []
    multi_choice = multi_choices[0] if len(multi_choices) == 1 else {}
    row.update(output=multi_choice.get("message", {}).get("content"),
               finish_reason=multi_choice.get("finish_reason"),
               output_tokens=multi_body.get("usage", {}).get("completion_tokens"))
    row["protocol_status"] = (row["finish_reason"] == "stop"
                              and isinstance(row["output"], str)
                              and (row["output_tokens"] or 0) > 0)
    row["answer_status"] = row["output"] == "cobalt"
    if not row["protocol_status"] or not row["answer_status"]:
        raise RuntimeError(f"multi-turn: incomplete or unexpected answer: {row}")

    limited = request("POST", "/responses", payload={
        "model": MODEL,
        "input": "Write the integers from 1 to 100, separated by commas. Do not stop early.",
        "max_output_tokens": 1, "temperature": 0,
    })
    row = {"label": "output-limit", "http_status": limited.status_code}
    cases.append(row)
    if limited.status_code != 200:
        row["error"] = limited.text[:300]
        raise RuntimeError(f"output-limit: HTTP {limited.status_code}")
    limited_body = limited.json()
    row.update(status=limited_body.get("status"),
               incomplete_reason=(limited_body.get("incomplete_details") or {}).get("reason"),
               output_tokens=limited_body.get("usage", {}).get("output_tokens"))
    if (row["status"] != "incomplete" or row["incomplete_reason"] != "max_output_tokens"
            or row["output_tokens"] != 1):
        raise RuntimeError(f"output-limit: exhaustion was not reported truthfully: {row}")

    if identity.get("output_protocol") == "none":
        rejected = request("POST", "/chat/completions", payload={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Read a file"}],
            "tools": [{"type": "function", "function": {
                "name": "read_file", "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }}],
            "max_tokens": 8,
        })
        code = rejected.json().get("error", {}).get("code")
        row = {"label": "undeclared-tools", "http_status": rejected.status_code,
               "error_code": code}
        cases.append(row)
        if rejected.status_code != 501 or code != "tool_protocol_undeclared":
            raise RuntimeError(f"undeclared-tools: rejection contract failed: {row}")
        _plain_chat_request("recovered", "chat-after-rejection", cases)
    else:
        cases.append({"label": "undeclared-tools", "status": "not_applicable",
                      "reason": "bundle declares an output protocol"})


def run_budget_boundary(identity: dict, cases: list[dict]) -> None:
    """Reject a request beyond the declared capacity, then serve another.

    A configured output cap changes an oversized reservation into a smaller
    effective allowance. Test a configured rendered-byte limit when present;
    otherwise construct a token-count overflow without relying on the output
    reservation to exceed context.
    """
    effective = identity.get("effective_serving") or {}
    capacity = effective.get("active_context_limit")
    if type(capacity) is not int or capacity <= 1:
        raise RuntimeError("loaded runtime has no valid context capacity")
    if effective.get("request_output_cap") is None:
        mode = "output_reservation"
        prompt = "hello"
        allowance = capacity
    else:
        byte_cap = effective.get("request_prompt_byte_cap")
        if byte_cap is not None and (type(byte_cap) is not int or not 1 <= byte_cap <= 262144):
            raise RuntimeError("loaded prompt byte cap is outside the supported probe range")
        if byte_cap is None:
            if capacity > 262144:
                raise RuntimeError("loaded context exceeds the bounded token-overflow probe range")
            mode = "counted_token_limit"
            prompt = "🙂 " * (capacity + 1)
        else:
            mode = "rendered_byte_limit"
            prompt = "x" * (byte_cap + 1)
        allowance = 1
    response = request("POST", "/responses", payload={
        "model": MODEL, "input": prompt, "max_output_tokens": allowance,
    })
    try:
        error_code = response.json().get("error", {}).get("code")
    except ValueError:
        error_code = None
    row = {"label": "context-boundary", "mode": mode, "http_status": response.status_code,
           "error_code": error_code, "input_bytes": len(prompt.encode()),
           "requested_output_tokens": allowance, "compiled_capacity": capacity}
    cases.append(row)
    expected_status, expected_error = (
        (413, "prompt_bytes_limit_exceeded") if mode == "rendered_byte_limit" else
        (400, "context_length_exceeded"))
    if response.status_code != expected_status or error_code != expected_error:
        raise RuntimeError(f"context-boundary: overflow was not rejected: {row}")
    row["counted_token_exceedance_proven"] = error_code == "context_length_exceeded"
    row["byte_limit_proven"] = error_code == "prompt_bytes_limit_exceeded"
    row["followup"] = followup("context-boundary-followup")


def run_recovery_replay(oversized_prompt: str, cases: list[dict]) -> None:
    """Compare repeated deterministic output before and after admission rejection.

    This proves HTTP admission isolation, not recovery from a native failure
    after generated model state has been mutated.
    """
    if not oversized_prompt or len(oversized_prompt.encode("utf-8")) > 8_000_000:
        raise ValueError("oversized prompt must contain 1 to 8,000,000 UTF-8 bytes")
    settings = {"temperature": 0, "top_p": 1}
    prompt = "Hello"
    for route in ("responses", "chat-completions"):
        def send(input_text: str) -> httpx.Response:
            if route == "responses":
                return request("POST", "/responses", payload={
                    "model": MODEL, "input": input_text, "max_output_tokens": 16,
                    **settings,
                }, timeout=120.0)
            return request("POST", "/chat/completions", payload={
                "model": MODEL,
                "messages": [{"role": "user", "content": input_text}],
                "max_tokens": 16, **settings,
            }, timeout=120.0)

        baseline = None
        for phase in ("before", "reject", "after"):
            for index in range(3 if phase != "reject" else 1):
                response = send(oversized_prompt if phase == "reject" else prompt)
                row = {"label": f"{route}-{phase}-{index + 1}",
                       "route": route, "phase": phase, "http_status": response.status_code}
                cases.append(row)
                try:
                    body = response.json()
                except ValueError as exc:
                    raise RuntimeError(f"{row['label']}: invalid JSON response") from exc
                if phase == "reject":
                    row["error_code"] = (body.get("error") or {}).get("code")
                    if response.status_code != 400 or row["error_code"] != "context_length_exceeded":
                        raise RuntimeError(f"{row['label']}: oversized prompt not rejected: {row}")
                    continue
                if response.status_code != 200:
                    row["error_code"] = (body.get("error") or {}).get("code")
                    raise RuntimeError(f"{row['label']}: normal request failed: {row}")
                if route == "responses":
                    result = (body.get("status"), body.get("output_text"),
                              (body.get("usage") or {}).get("output_tokens"))
                else:
                    choices = body.get("choices") or []
                    choice = choices[0] if len(choices) == 1 else {}
                    result = (choice.get("finish_reason"),
                              (choice.get("message") or {}).get("content"),
                              (body.get("usage") or {}).get("completion_tokens"))
                if (result[0] not in {"completed", "incomplete", "stop", "length"}
                        or not isinstance(result[1], str) or not result[1]
                        or type(result[2]) is not int or result[2] <= 0):
                    raise RuntimeError(f"{row['label']}: invalid result: {result!r}")
                row.update(terminal=result[0], output_sha256=hashlib.sha256(
                    result[1].encode("utf-8")).hexdigest(), output_tokens=result[2])
                if baseline is None:
                    baseline = result
                elif result != baseline:
                    raise RuntimeError(f"{row['label']}: deterministic replay diverged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Local HTTP base URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-serving-identity", required=True)
    parser.add_argument("--expected-session-library-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    parser.add_argument("--prefill-lines", type=int, default=1500)
    parser.add_argument("--suite", choices=("lifecycle", "plain-chat", "budget-boundary", "recovery-replay"),
                        default="lifecycle")
    parser.add_argument("--oversized-prompt-file", type=Path,
                        help="UTF-8 prompt known to exceed token context for recovery-replay")
    parser.add_argument("--expected-reply", default="hello",
                        help="Exact one-word reply for the plain-chat suite")
    args = parser.parse_args()
    parsed = urlparse(args.endpoint.rstrip("/"))
    if parsed.scheme != "http" or not parsed.hostname or parsed.path != "/v1":
        parser.error("endpoint must be an HTTP URL with /v1 as its path")
    if args.prefill_lines < 1:
        parser.error("prefill-lines must be positive")
    if not re.fullmatch(r"[A-Za-z]{1,32}", args.expected_reply):
        parser.error("expected-reply must be one ASCII word of at most 32 letters")
    if args.suite == "recovery-replay" and args.oversized_prompt_file is None:
        parser.error("recovery-replay requires --oversized-prompt-file")
    oversized_prompt = None
    if args.oversized_prompt_file is not None:
        if args.oversized_prompt_file.stat().st_size > 8_000_000:
            parser.error("oversized prompt exceeds 8,000,000 bytes")
        oversized_prompt = args.oversized_prompt_file.read_text(encoding="utf-8")
        if not oversized_prompt:
            parser.error("oversized prompt file is empty")
    for label, digest in (
        ("expected-serving-identity", args.expected_serving_identity),
        ("expected-session-library-sha256", args.expected_session_library_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            parser.error(f"{label} must be a lowercase SHA-256 digest")
    if args.output.exists():
        parser.error("output report already exists; acceptance evidence is immutable")

    global BASE, MODEL, HOST, PORT, PREFILL_LINES
    BASE = args.endpoint.rstrip("/")
    MODEL = args.model
    HOST = parsed.hostname
    PORT = parsed.port or 80
    PREFILL_LINES = args.prefill_lines

    report = {
        "schema": "cke.http_lifecycle_acceptance.v1",
        "endpoint": BASE, "model": MODEL,
        "expected_serving_identity": args.expected_serving_identity,
        "expected_session_library_sha256": args.expected_session_library_sha256,
        "started_at_unix": time.time(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "suite": args.suite,
        "oversized_prompt_sha256": (hashlib.sha256(oversized_prompt.encode("utf-8")).hexdigest()
                                     if oversized_prompt is not None else None),
        "replay_settings": ({"ordinary_prompt_sha256": hashlib.sha256(b"Hello").hexdigest(),
                             "temperature": 0, "top_p": 1, "max_output_tokens": 16,
                             "requests_per_side_per_route": 3}
                            if args.suite == "recovery-replay" else None),
        "prefill_lines": PREFILL_LINES,
        "identity_before": None,
        "identity_after": None, "cases": [], "status": "fail",
    }
    try:
        before = request("GET", "/cke/loaded-identity")
        before.raise_for_status()
        report["identity_before"] = before.json()
        _check_identity(report["identity_before"],
                        expected_serving=args.expected_serving_identity,
                        expected_session=args.expected_session_library_sha256,
                        model=MODEL)
        if args.suite == "plain-chat":
            run_plain_chat(args.expected_reply, report["identity_before"], report["cases"])
        elif args.suite == "budget-boundary":
            run_budget_boundary(report["identity_before"], report["cases"])
        elif args.suite == "recovery-replay":
            run_recovery_replay(oversized_prompt, report["cases"])
        else:
            long_prompt = "Test data follows.\n" + "alpha beta gamma delta\n" * PREFILL_LINES
            report["cases"].append(run_cancel(
                "long-input-timed-cancel", long_prompt,
                wait_for="response.in_progress", delay=3.0))
            report["cases"].append(run_cancel(
                "decode", "Count upward forever, one number per line, beginning at one.",
                wait_for="response.output_text.delta"))
            report["cases"].append(run_disconnect("stream-disconnect", long_prompt))
            report["cases"].append(run_client_timeout("client-read-timeout", long_prompt))
        after = request("GET", "/cke/loaded-identity")
        after.raise_for_status()
        report["identity_after"] = after.json()
        _check_identity(report["identity_after"],
                        expected_serving=args.expected_serving_identity,
                        expected_session=args.expected_session_library_sha256,
                        model=MODEL)
        if report["identity_after"] != report["identity_before"]:
            raise RuntimeError("loaded identity changed during acceptance run")
        report["status"] = "pass"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = round(time.time() - report["started_at_unix"], 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "status": report["status"],
                      "error": report.get("error")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
