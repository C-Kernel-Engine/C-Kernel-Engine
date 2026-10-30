#!/usr/bin/env python3
"""Confirmed Whisper WAV -> CKE Responses -> interruptible local speech.

This is a bounded, single-target workstation client. It opens a microphone
only after an explicit Enter on --record, using the selected ALSA device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave


class VoiceError(RuntimeError):
    pass


def _append(journal: Path, record: dict) -> None:
    journal.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time_ns": time.time_ns(), **record}
    descriptor = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND |
                         os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise VoiceError("journal must be a regular file")
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def transcribe(wav: Path, encoder: Path, decoder: Path,
               runner: Path) -> str:
    if not wav.is_file():
        raise VoiceError(f"input WAV missing: {wav}")
    with tempfile.TemporaryDirectory(prefix="cke-voice-asr-") as folder:
        report = Path(folder) / "whisper.json"
        command = [sys.executable, str(runner), "run", "--encoder-run-dir",
                   str(encoder), "--decoder-run-dir", str(decoder),
                   "--wav", str(wav), "--output", str(report)]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=600, check=False)
        if result.returncode != 0:
            raise VoiceError(f"CKE Whisper failed ({result.returncode}): "
                             f"{result.stderr[-400:]}")
        try:
            text = json.loads(report.read_text(encoding="utf-8"))["decoder"]["transcript_text"]
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise VoiceError("CKE Whisper did not publish a transcript") from error
    if not isinstance(text, str) or not text.strip():
        raise VoiceError("CKE Whisper returned an empty transcript")
    return text.strip()


def record_to_wav(output: Path, device: str, recorder: Path, read=input) -> bool:
    """Explicit press/release recording; caller owns the temporary WAV path."""
    if not device or "\x00" in device:
        raise VoiceError("an explicit capture device is required")
    if not recorder.is_file() or not os.access(recorder, os.X_OK):
        raise VoiceError(f"audio recorder unavailable: {recorder}")
    start = read("Press Enter to begin recording, or type cancel: ").strip().lower()
    if start == "cancel":
        return False
    if start:
        raise VoiceError("recording requires an explicit Enter keypress")
    command = [str(recorder), "-q", "-D", device, "-f", "S16_LE",
               "-r", "16000", "-c", "1", "-t", "wav", str(output)]
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             start_new_session=True)
    stop = ""
    try:
        stop = read("Recording. Press Enter to stop or type cancel: ").strip().lower()
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
        try:
            child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
        raise
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
        try:
            _, errors = child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            _, errors = child.communicate()
            raise VoiceError("audio recorder did not stop promptly")
    if child.returncode not in (0, -signal.SIGINT, 128 + signal.SIGINT):
        raise VoiceError(f"audio capture failed: {errors.decode('utf-8', 'replace')[-300:]}")
    if stop == "cancel":
        return False
    if stop:
        raise VoiceError("unrecognized recording control")
    try:
        with wave.open(str(output), "rb") as audio:
            valid = (audio.getnchannels(), audio.getsampwidth(),
                     audio.getframerate()) == (1, 2, 16000) and audio.getnframes() > 0
    except (OSError, EOFError, wave.Error):
        valid = False
    if not valid:
        raise VoiceError("audio capture did not produce 16 kHz mono PCM16")
    return True


def confirm(draft: str, agent: str, project: str, read=input) -> str | None:
    """A correction invalidates the old draft; only a fresh yes confirms it."""
    text = draft
    while True:
        print(f"Target: {agent} / {project}\nTranscript: {text}", flush=True)
        choice = read("Type yes to submit, edit to correct, or cancel: ").strip().lower()
        if choice == "cancel":
            return None
        if choice == "yes":
            return text
        if choice == "edit":
            replacement = read("Corrected request: ").strip()
            if not replacement:
                print("Empty correction rejected.", flush=True)
                continue
            text = replacement
            continue
        print("Request not submitted.", flush=True)


def _events(stream):
    name = None
    data = []
    for raw in stream:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if name and data:
                try:
                    yield name, json.loads("\n".join(data))
                except ValueError as error:
                    raise VoiceError("invalid server event JSON") from error
            name, data = None, []
        elif line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if name and data:
        raise VoiceError("server event stream ended mid-event")


def _cancel_response(server_url: str, response_id: str) -> bool:
    route = (server_url.rstrip("/") + "/v1/responses/" +
             urllib.parse.quote(response_id, safe="") + "/cancel")
    try:
        with urllib.request.urlopen(urllib.request.Request(route, method="POST"),
                                    timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _verify_model(server_url: str, model: str) -> None:
    route = (server_url.rstrip("/") + "/v1/models/" +
             urllib.parse.quote(model, safe=""))
    try:
        with urllib.request.urlopen(route, timeout=5) as response:
            selected = json.load(response)
    except (OSError, ValueError) as error:
        raise VoiceError("selected CKE model could not be verified") from error
    if not isinstance(selected, dict) or selected.get("id") != model:
        raise VoiceError("CKE server returned a different model identity")


def submit(text: str, *, server_url: str, model: str, journal: Path,
           request_id: str, agent: str, project: str,
           timeout: float = 600) -> str:
    parsed = urllib.parse.urlsplit(server_url)
    try:
        local_port = parsed.port
    except ValueError as error:
        raise VoiceError("invalid server port") from error
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
            or parsed.username or parsed.password or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment or local_port is None):
        raise VoiceError("first-loop server must use an explicit local address")
    _verify_model(server_url, model)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = {"request_id": request_id, "agent_id": agent,
                "project_id": project, "text_sha256": digest}
    # The intent must survive a crash before the network request starts.
    _append(journal, {**identity, "state": "submitting", "text": text})
    body = json.dumps({"model": model, "input": text,
                       "instructions": (
                           "You are a helpful assistant. This voice client does not "
                           "provide file or project tools. If asked for live project "
                           "status, state clearly that you cannot inspect it."),
                       "max_output_tokens": 512, "stream": True}).encode("utf-8")
    request = urllib.request.Request(server_url.rstrip("/") + "/v1/responses",
                                     body, {"Content-Type": "application/json",
                                            "Accept": "text/event-stream"})
    response_id = None
    sequence = -1
    deltas = []
    terminal = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise VoiceError(f"server returned HTTP {response.status}")
            for name, payload in _events(response):
                if payload.get("type") != name:
                    raise VoiceError("server event type mismatch")
                number = payload.get("sequence_number")
                if not isinstance(number, int) or isinstance(number, bool) or number != sequence + 1:
                    raise VoiceError("server event sequence mismatch")
                sequence = number
                current = payload.get("response") or {}
                current_id = current.get("id") or payload.get("response_id")
                if name == "response.created":
                    if response_id or not isinstance(current_id, str) or not current_id:
                        raise VoiceError("missing or duplicate server acceptance ID")
                    response_id = current_id
                    _append(journal, {**identity, "state": "acknowledged",
                                      "response_id": response_id})
                elif response_id and current_id and current_id != response_id:
                    raise VoiceError("server response ID changed")
                if name == "response.output_text.delta":
                    if not response_id or not isinstance(payload.get("delta"), str):
                        raise VoiceError("uncorrelated output text")
                    deltas.append(payload["delta"])
                if name in ("response.completed", "response.incomplete",
                            "response.failed", "response.cancelled"):
                    if not response_id or not isinstance(current_id, str):
                        raise VoiceError("uncorrelated terminal event")
                    if name != "response.completed":
                        _append(journal, {**identity, "state": name.split(".")[1],
                                          "response_id": response_id})
                        terminal = True
                        raise VoiceError(f"server ended with {name}")
                    answer = current.get("output_text")
                    if not isinstance(answer, str) or not answer.strip() or (
                            deltas and answer != "".join(deltas)):
                        raise VoiceError("server completed with inconsistent or empty text")
                    _append(journal, {**identity, "state": "completed",
                                      "response_id": response_id, "answer": answer})
                    terminal = True
                    return answer
        if not terminal:
            raise VoiceError("server stream ended without a terminal event")
    except urllib.error.HTTPError as error:
        error.close()
        _append(journal, {**identity, "state": "rejected",
                          "http_status": error.code})
        raise VoiceError(f"server rejected request: HTTP {error.code}") from error
    except (OSError, VoiceError, UnicodeError) as error:
        if not terminal:
            _append(journal, {**identity, "state": "delivery_uncertain",
                              "response_id": response_id, "reason": str(error)})
        raise
    except KeyboardInterrupt:
        if response_id:
            _append(journal, {**identity, "state": "cancel_requested",
                              "response_id": response_id})
            stopped = _cancel_response(server_url, response_id)
            _append(journal, {**identity,
                              "state": "cancelled" if stopped else "delivery_uncertain",
                              "response_id": response_id})
        else:
            _append(journal, {**identity, "state": "delivery_uncertain",
                              "reason": "interrupted before acceptance receipt"})
        raise
    raise VoiceError("server returned no answer")


def speak_and_control(text: str, speech_exe: Path, read=input) -> None:
    if not speech_exe.is_file() or not os.access(speech_exe, os.X_OK):
        raise VoiceError(f"system speech executable unavailable: {speech_exe}")
    print("System speech backend selected.", flush=True)
    child = None
    try:
        while True:
            if child is None:
                child = subprocess.Popen([str(speech_exe), text],
                                         start_new_session=True)
            choice = read("Speech: enter=finish, repeat=restart, stop=silence: ").strip().lower()
            if choice == "":
                if child.wait() != 0:
                    raise VoiceError("system speech failed")
                return
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            child = None
            if choice == "repeat":
                continue
            return
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record", action="store_true",
                        help="record only after explicit Enter; delete temporary audio")
    source.add_argument("--wav", type=Path, help="recorded PCM WAV to transcribe")
    source.add_argument("--text", help="typed request for testing without a microphone")
    parser.add_argument("--encoder-run-dir", type=Path)
    parser.add_argument("--decoder-run-dir", type=Path)
    parser.add_argument("--whisper-runner", type=Path,
                        default=Path(__file__).with_name("run_whisper_v8.py"))
    parser.add_argument("--capture-device", help="explicit ALSA capture device for --record")
    parser.add_argument("--recorder-exe", type=Path, default=Path("/usr/bin/arecord"))
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="ck-v8")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--speech-exe", type=Path, help="blocking local speech command accepting text as one argument")
    args = parser.parse_args(argv)
    try:
        if (args.wav or args.record) and (not args.encoder_run_dir or not args.decoder_run_dir):
            parser.error("audio input requires both Whisper run directories")
        if args.record and not args.capture_device:
            parser.error("--record requires --capture-device")
        if args.record:
            with tempfile.TemporaryDirectory(prefix="cke-voice-capture-") as folder:
                wav = Path(folder) / "microphone.wav"
                if not record_to_wav(wav, args.capture_device, args.recorder_exe):
                    print("Recording cancelled before capture.")
                    return 0
                draft = transcribe(wav, args.encoder_run_dir, args.decoder_run_dir,
                                   args.whisper_runner)
        elif args.wav:
            draft = transcribe(args.wav, args.encoder_run_dir, args.decoder_run_dir,
                               args.whisper_runner)
        else:
            draft = args.text
        if not draft.strip():
            raise VoiceError("empty request")
        confirmed = confirm(draft.strip(), args.agent_id, args.project_id)
        if confirmed is None:
            print("Cancelled before delivery.")
            return 0
        request_id = str(uuid.uuid4())
        answer = submit(confirmed, server_url=args.server_url, model=args.model,
                        journal=args.journal, request_id=request_id,
                        agent=args.agent_id, project=args.project_id)
        print(f"Answer from {args.agent_id} [{request_id}]:\n{answer}", flush=True)
        if args.speech_exe:
            speak_and_control(answer, args.speech_exe)
        else:
            print("No speech executable selected; answer was displayed only.", flush=True)
        return 0
    except (VoiceError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Voice request failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; consult the journal before resubmitting.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
