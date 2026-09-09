#!/usr/bin/env python3
"""Bounded nightly localhost E2E for the v8 live server.

Starts ``ck_serve_v8.py`` with Qwen3-0.6B (context 1024) on an ephemeral
localhost port, waits for readiness with a hard timeout, sends one streamed
``POST /v1/responses`` request, and validates:

  * schema (Responses + SSE event shapes),
  * non-empty emitted text,
  * terminal ``response.completed`` event,
  * ``GET /v1/responses/{id}`` retrievability,
  * server process exit / port cleanup.

All HTTP traffic targets ``127.0.0.1`` only (no external network access in
the request phase; model artifact download happens once inside the server
child via the standard ``ck_run`` pipeline when the cache is cold).

On failure the server log tail and the SSE transcript are printed and saved
under ``build/v8_serve_e2e_last_failure/``.

Exit status: 0 on pass, 1 on any failure (model prep, startup, routing,
streaming, tokenization, native inference, or termination).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]
SERVE_SCRIPT = SCRIPTS_DIR / "ck_serve_v8.py"
FAILURE_DIR = PROJECT_ROOT / "build" / "v8_serve_e2e_last_failure"

DEFAULT_MODEL = "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
DEFAULT_CONTEXT_LEN = 1024
# Ready timeout must cover cold artifact build (download+convert+compile)
# plus session open; keeps total still inside the 10 min nightly budget.
DEFAULT_READY_TIMEOUT = 480.0
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_TOTAL_TIMEOUT = 600.0
DEFAULT_CLEANUP_RESERVE = 20.0
POLL_INTERVAL = 0.5
TERMINATE_GRACE_SEC = 10.0
KILL_GRACE_SEC = 5.0
LOG_TAIL_CHARS = 40000


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # SO_REUSEADDR lets a fresh listen succeed even while the previous
        # listener's connections are in TIME_WAIT; without it the bind check
        # falsely reports "still held" immediately after close.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError:
            pass
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _wait_port_free(port: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_is_free(port):
            return True
        time.sleep(0.2)
    return _port_is_free(port)


def _http(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 10.0,
    accept: str | None = None,
) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if accept:
        req.add_header("Accept", accept)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return int(exc.code), body


def _wait_for_health(
    base: str, deadline: float, proc: subprocess.Popen | None = None
) -> None:
    last_error = "no attempt yet"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"server exited during readiness (code={proc.returncode}); last error: {last_error}"
            )
        try:
            remaining = deadline - time.monotonic()
            status, body = _http(
                "GET", f"{base}/v1/health", timeout=max(0.001, min(5.0, remaining))
            )
            if status == 200:
                try:
                    doc = json.loads(body)
                except json.JSONDecodeError as exc:
                    last_error = f"health returned invalid JSON: {exc}"
                    time.sleep(POLL_INTERVAL)
                    continue
                if doc.get("status") == "ok":
                    return
                last_error = f"health not ok: {body[:300]}"
            else:
                last_error = f"health status={status} body={body[:300]}"
        except Exception as exc:  # connection refused while booting
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(max(0.0, min(POLL_INTERVAL, deadline - time.monotonic())))
    raise RuntimeError(f"server readiness timeout; last error: {last_error}")


def _set_response_timeout(response, timeout: float) -> None:
    """Bound the next blocking response read where urllib exposes its socket."""
    candidates = (
        getattr(response, "fp", None),
        getattr(getattr(response, "fp", None), "raw", None),
    )
    for candidate in candidates:
        sock = getattr(candidate, "_sock", None)
        if sock is not None:
            sock.settimeout(timeout)
            return


def _read_response_incrementally(
    response,
    deadline: float,
    *,
    now=time.monotonic,
    on_chunk=None,
) -> str:
    """Read an HTTP response without allowing a trickle to reset the deadline."""
    chunks: list[bytes] = []
    while True:
        remaining = deadline - now()
        if remaining <= 0:
            raise TimeoutError("streaming response exceeded its overall deadline")
        _set_response_timeout(response, max(0.001, remaining))
        chunk = response.readline()
        if not chunk:
            break
        chunks.append(chunk)
        if on_chunk is not None:
            on_chunk(chunk.decode("utf-8", "replace"))
    return b"".join(chunks).decode("utf-8", "replace")


def _stream_http(
    url: str,
    payload: dict,
    deadline: float,
    *,
    on_chunk=None,
) -> tuple[int, str]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("request deadline expired before POST")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=remaining) as response:
            return int(response.status), _read_response_incrementally(
                response, deadline, on_chunk=on_chunk
            )
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse SSE ``event: X\\ndata: {...}\\n\\n`` blocks."""
    events: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name: str | None = None
        payload: dict | None = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                try:
                    payload = json.loads(line.split(":", 1)[1].strip())
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"non-JSON SSE data line: {line[:300]} ({exc})")
        if name is None or payload is None:
            raise RuntimeError(f"malformed SSE block: {block[:300]}")
        events.append((name, payload))
    return events


def _validate_stream(events: list[tuple[str, dict]]) -> tuple[str, str, str, dict]:
    """Structural + schema validation of the streamed response.

    Returns (response_id, emitted_text, terminal_event_name).
    """
    if len(events) < 3:
        raise RuntimeError(f"expected >=3 SSE events, got {len(events)}")
    names = [name for name, _ in events]
    if names[0] != "response.created" or names[1] != "response.in_progress":
        raise RuntimeError(f"bad SSE prefix: {names[:3]}")

    # Importing and applying the actual union is mandatory. A missing schema
    # dependency means this E2E cannot establish its advertised contract.
    sys.path.insert(0, str(PROJECT_ROOT))
    from pydantic import TypeAdapter
    from server.schemas.streaming import ResponseStreamEvent

    adapter = TypeAdapter(ResponseStreamEvent)

    # sequence_number must be contiguous 0..N-1.
    seqs: list[int] = []
    for name, payload in events:
        if payload.get("type") != name:
            raise RuntimeError(
                f"SSE event/payload type mismatch: event={name!r} "
                f"payload={payload.get('type')!r}"
            )
        try:
            adapter.validate_python(payload)
        except Exception as exc:
            raise RuntimeError(f"event {name!r} failed schema validation: {exc}") from exc
        seq = payload.get("sequence_number")
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise RuntimeError(f"event {name!r} missing int sequence_number")
        seqs.append(seq)
    if seqs != list(range(len(events))):
        raise RuntimeError(f"non-contiguous sequence_numbers: {seqs[:12]}")

    response_ids: set[str] = set()
    for name, payload in events:
        resp = payload.get("response")
        if isinstance(resp, dict) and isinstance(resp.get("id"), str):
            response_ids.add(resp["id"])
    if not response_ids:
        raise RuntimeError("no response id found in SSE prefix events")
    if len(response_ids) != 1:
        raise RuntimeError(f"inconsistent response ids in SSE events: {sorted(response_ids)}")
    response_id = next(iter(response_ids))

    deltas: list[str] = []
    for name, payload in events:
        if name != "response.output_text.delta":
            continue
        delta = payload.get("delta")
        if not isinstance(delta, str):
            raise RuntimeError("output_text.delta delta must be a string")
        deltas.append(delta)
    if not deltas:
        raise RuntimeError("no response.output_text.delta events emitted")
    for n, p in events:
        if n == "response.output_text.delta" and p.get("logprobs") is not None:
            raise RuntimeError("output_text.delta logprobs must be None")
    emitted = "".join(deltas)
    if not emitted.strip():
        raise RuntimeError("emitted text is empty")

    terminal = names[-1]
    if terminal != "response.completed":
        raise RuntimeError(
            f"expected terminal response.completed, got {terminal!r}; "
            f"events={names[-4:]}"
        )
    terminal_response = events[-1][1].get("response")
    if not isinstance(terminal_response, dict):
        raise RuntimeError("response.completed is missing its response payload")
    if terminal_response.get("output_text") != emitted:
        raise RuntimeError("streamed deltas do not match response.completed output_text")
    return response_id, emitted, terminal, terminal_response


def _validate_final_response(
    doc: dict, response_id: str, emitted: str, terminal_response: dict
) -> None:
    if doc.get("id") != response_id:
        raise RuntimeError(
            f"GET id mismatch: {doc.get('id')!r} != {response_id!r}"
        )
    if doc.get("status") != "completed":
        raise RuntimeError(f"final status is {doc.get('status')!r}, want completed")
    if not isinstance(doc.get("output_text"), str) or not doc["output_text"].strip():
        raise RuntimeError("final output_text is empty")
    if doc["output_text"] != emitted:
        raise RuntimeError("stored response output_text does not match streamed deltas")
    if terminal_response.get("id") != doc.get("id"):
        raise RuntimeError("stored response id does not match terminal response")
    if terminal_response.get("output_text") != doc.get("output_text"):
        raise RuntimeError("stored response text does not match terminal response")
    sys.path.insert(0, str(PROJECT_ROOT))
    from server.schemas.response import Response

    Response.model_validate(doc)


def _stop_server(proc: subprocess.Popen, port: int) -> None:
    """Terminate the child on all paths; raise if it refuses to exit."""
    if proc.poll() is not None:
        if not _wait_port_free(port, timeout=5.0):
            raise RuntimeError(f"server exited but port {port} still held")
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=TERMINATE_GRACE_SEC)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=KILL_GRACE_SEC)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("server process refused to terminate") from exc
    if not _wait_port_free(port, timeout=5.0):
        raise RuntimeError(f"server exited but port {port} still held")


def _save_failure(log_text: str, transcript: str, summary: dict) -> None:
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    (FAILURE_DIR / "server.log").write_text(log_text[-LOG_TAIL_CHARS:], encoding="utf-8")
    (FAILURE_DIR / "transcript.sse").write_text(transcript[-LOG_TAIL_CHARS:], encoding="utf-8")
    (FAILURE_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def _combine_failure(
    failure: Exception | None, cleanup_error: Exception
) -> Exception:
    cleanup_failure = RuntimeError(f"cleanup failed: {cleanup_error}")
    if failure is None:
        return cleanup_failure
    return RuntimeError(f"{failure}; {cleanup_failure}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded localhost E2E: serve Qwen3-0.6B and stream one Responses request.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--total-timeout", type=float, default=DEFAULT_TOTAL_TIMEOUT)
    parser.add_argument("--cleanup-reserve", type=float, default=DEFAULT_CLEANUP_RESERVE)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    args = parser.parse_args(argv)

    if args.total_timeout <= args.cleanup_reserve:
        parser.error("--total-timeout must exceed --cleanup-reserve")

    if not SERVE_SCRIPT.is_file():
        print(f"FAIL: serve script missing: {SERVE_SCRIPT}", flush=True)
        return 1

    port = _find_free_port()
    base = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile(
        prefix="ck-serve-e2e-", suffix=".log", delete=False, mode="w", encoding="utf-8"
    )
    log_path = log_file.name
    log_file.close()

    cmd = [
        sys.executable,
        str(SERVE_SCRIPT),
        args.model,
        "--context-len",
        str(args.context_len),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-viz",
    ]
    if args.run_dir:
        cmd += ["--run", args.run_dir]

    proc: subprocess.Popen | None = None
    transcript = ""
    response_id = ""
    summary: dict = {"model": args.model, "port": port}
    failure: Exception | None = None
    pass_details: tuple[str, int] | None = None
    overall_deadline = time.monotonic() + args.total_timeout
    work_deadline = overall_deadline - args.cleanup_reserve

    def _handle_signal(signum, _frame):
        raise RuntimeError(f"interrupted by signal {int(signum)}")

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    try:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            kw: dict = {"cwd": str(PROJECT_ROOT), "stdout": log_fh, "stderr": subprocess.STDOUT}
            if os.name == "posix":
                kw["start_new_session"] = True
            proc = subprocess.Popen(cmd, **kw)  # noqa: SUBW001 - localhost E2E child

        # Hard readiness bound; also fails fast if child exits during model prep.
        _wait_for_health(
            base,
            min(work_deadline, time.monotonic() + args.ready_timeout),
            proc,
        )
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup (code={proc.returncode})")

        transcript_chunks: list[str] = []
        status, transcript = _stream_http(
            f"{base}/v1/responses",
            {
                "model": "ck-v8",
                "input": "Say hello in five words or less.",
                "max_output_tokens": args.max_output_tokens,
                "stream": True,
                "store": True,
            },
            min(work_deadline, time.monotonic() + args.request_timeout),
            on_chunk=transcript_chunks.append,
        )
        transcript = "".join(transcript_chunks) or transcript
        if status != 200:
            raise RuntimeError(f"POST /v1/responses status={status}: {transcript[:500]}")
        events = _parse_sse(transcript)
        response_id, emitted, terminal, terminal_response = _validate_stream(events)

        remaining = work_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("overall deadline expired before stored-response retrieval")
        get_status, get_body = _http(
            "GET",
            f"{base}/v1/responses/{response_id}",
            timeout=min(15.0, remaining),
        )
        if get_status != 200:
            raise RuntimeError(f"GET /v1/responses/{{id}} status={get_status}")
        _validate_final_response(
            json.loads(get_body), response_id, emitted, terminal_response
        )

        summary.update(
            {"status": "pass", "response_id": response_id, "terminal": terminal}
        )
        pass_details = (response_id, len(emitted))
    except Exception as exc:
        failure = exc
    finally:
        # Restore previous signal handlers before cleanup.
        try:
            signal.signal(signal.SIGTERM, prev_sigterm)
            signal.signal(signal.SIGINT, prev_sigint)
        except Exception:
            pass
        if proc is not None:
            try:
                _stop_server(proc, port)
            except Exception as exc:
                failure = _combine_failure(failure, exc)

    if failure is None and pass_details is not None:
        response_id, emitted_chars = pass_details
        print(
            f"PASS serve-localhost-e2e model=Qwen3-0.6B-Q8_0.gguf context-len={args.context_len} "
            f"port={port} id={response_id} chars={emitted_chars}",
            flush=True,
        )
        return 0

    log_text = ""
    try:
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    failure = failure or RuntimeError("test ended without a result")
    summary.update({"status": "fail", "error": str(failure), "response_id": response_id})
    _save_failure(log_text, transcript, summary)
    print(f"FAIL serve-localhost-e2e: {failure}", flush=True)
    if log_text:
        print(f"--- server log tail ({log_path}) ---", flush=True)
        print(log_text[-LOG_TAIL_CHARS:], flush=True)
    if transcript:
        print("--- response transcript tail ---", flush=True)
        print(transcript[-LOG_TAIL_CHARS:], flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
