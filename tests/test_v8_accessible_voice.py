"""Contract tests for the single-target accessible voice client."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "version/v8/scripts/ck_accessible_voice.py"
SPEC = importlib.util.spec_from_file_location("ck_accessible_voice", PATH)
assert SPEC and SPEC.loader
voice = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(voice)


class Handler(BaseHTTPRequestHandler):
    mode = "complete"
    received = []
    cancelled = []

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/v1/models/ck-v8":
            self.send_error(404)
            return
        payload = {"id": "other" if self.mode == "wrong_model" else "ck-v8"}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.endswith("/cancel"):
            self.__class__.cancelled.append(self.path)
            self.send_response(200)
            self.end_headers()
            return
        self.__class__.received.append(json.loads(self.rfile.read(
            int(self.headers["Content-Length"]))))
        if self.path != "/v1/responses":
            self.send_error(404)
            return
        if self.mode == "reject":
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            ("response.created", {"response": {"id": "resp-1"}}),
            ("response.in_progress", {"response": {"id": "resp-1"}}),
            ("response.output_text.delta", {"response_id": "resp-1", "delta": "Status is good."}),
            ("response.completed", {"response": {"id": "resp-1", "output_text": "Status is good."}}),
        ]
        if self.mode == "disconnect":
            events = events[:2]
        if self.mode == "wrong_id":
            events[-1][1]["response"]["id"] = "resp-other"
        if self.mode == "bad_sequence":
            events[-1][1]["sequence_number"] = 9
        for index, (name, payload) in enumerate(events):
            payload = {"type": name, "sequence_number": index, **payload}
            self.wfile.write((f"event: {name}\ndata: {json.dumps(payload)}\n\n").encode())
        self.wfile.flush()


class AccessibleVoiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.journal = Path(self.temp.name) / "events.jsonl"
        Handler.received = []
        Handler.cancelled = []
        Handler.mode = "complete"
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_port}"

    def states(self):
        return [json.loads(line)["state"] for line in self.journal.read_text().splitlines()]

    def send(self):
        return voice.submit("Read project status", server_url=self.url,
                            model="ck-v8", journal=self.journal,
                            request_id="req-1", agent="cke", project="engine")

    def test_confirmation_requires_fresh_yes_after_edit(self):
        answers = iter(("edit", "Read only status", "maybe", "yes"))
        self.assertEqual(voice.confirm("wrong", "cke", "engine", lambda _: next(answers)),
                         "Read only status")
        self.assertIsNone(voice.confirm("draft", "cke", "engine", lambda _: "cancel"))

    def test_stream_acceptance_and_correlated_completion(self):
        self.assertEqual(self.send(), "Status is good.")
        self.assertEqual(self.states(), ["submitting", "acknowledged", "completed"])
        rows = [json.loads(line) for line in self.journal.read_text().splitlines()]
        self.assertEqual({row["request_id"] for row in rows}, {"req-1"})
        self.assertEqual(rows[-1]["response_id"], "resp-1")
        self.assertEqual(Handler.received[0]["input"], "Read project status")
        self.assertTrue(rows[0]["text_sha256"])

    def test_typed_cli_requires_confirmation_before_delivery(self):
        original_stdin = sys.stdin
        sys.stdin = io.StringIO("cancel\n")
        arguments = ["--text", "Read project status", "--agent-id", "cke-chat",
                     "--project-id", "engine", "--server-url", self.url,
                     "--journal", str(self.journal)]
        try:
            self.assertEqual(voice.main(arguments), 0)
            self.assertEqual(Handler.received, [])
            self.assertFalse(self.journal.exists())
            sys.stdin = io.StringIO("yes\n")
            self.assertEqual(voice.main(arguments), 0)
        finally:
            sys.stdin = original_stdin
        self.assertEqual(len(Handler.received), 1)
        self.assertEqual(self.states(), ["submitting", "acknowledged", "completed"])

    def test_missing_terminal_remains_delivery_uncertain(self):
        Handler.mode = "disconnect"
        with self.assertRaisesRegex(voice.VoiceError, "without a terminal"):
            self.send()
        self.assertEqual(self.states(), ["submitting", "acknowledged", "delivery_uncertain"])
        self.assertEqual(len(Handler.received), 1)

    def test_explicit_rejection_is_not_delivery_uncertain(self):
        Handler.mode = "reject"
        with self.assertRaisesRegex(voice.VoiceError, "HTTP 400"):
            self.send()
        self.assertEqual(self.states(), ["submitting", "rejected"])

    def test_model_mismatch_prevents_submission(self):
        Handler.mode = "wrong_model"
        with self.assertRaisesRegex(voice.VoiceError, "different model identity"):
            self.send()
        self.assertEqual(Handler.received, [])
        self.assertFalse(self.journal.exists())

    def test_private_journal_rejects_symlink(self):
        target = Path(self.temp.name) / "outside.txt"
        target.write_text("keep")
        self.journal.symlink_to(target)
        with self.assertRaises(OSError):
            self.send()
        self.assertEqual(target.read_text(), "keep")
        self.assertEqual(Handler.received, [])

    def test_wrong_identity_and_sequence_fail_closed(self):
        for mode in ("wrong_id", "bad_sequence"):
            with self.subTest(mode=mode):
                self.journal.unlink(missing_ok=True)
                Handler.mode = mode
                with self.assertRaises(voice.VoiceError):
                    self.send()
                self.assertEqual(self.states()[-1], "delivery_uncertain")

    def test_interrupt_after_receipt_requests_native_cancel(self):
        original = voice._events

        def interrupted(stream):
            yield next(original(stream))
            raise KeyboardInterrupt

        voice._events = interrupted
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.send()
        finally:
            voice._events = original
        self.assertEqual(self.states(), ["submitting", "acknowledged",
                                         "cancel_requested", "cancelled"])
        self.assertEqual(Handler.cancelled, ["/v1/responses/resp-1/cancel"])

    def test_whisper_adapter_uses_checked_transcript(self):
        wav = Path(self.temp.name) / "input.wav"
        wav.write_bytes(b"test fixture")
        runner = Path(self.temp.name) / "runner.py"
        runner.write_text(
            "import json,sys\n"
            "from pathlib import Path\n"
            "Path(sys.argv[sys.argv.index('--output')+1]).write_text("
            "json.dumps({'decoder':{'transcript_text':'Project status'}}))\n")
        self.assertEqual(voice.transcribe(wav, Path("encoder"), Path("decoder"), runner),
                         "Project status")
        wav.unlink()
        with self.assertRaisesRegex(voice.VoiceError, "WAV missing"):
            voice.transcribe(wav, Path("encoder"), Path("decoder"), runner)

    def test_recording_requires_action_and_can_be_cancelled(self):
        marker = Path(self.temp.name) / "recorder.started"
        output = Path(self.temp.name) / "microphone.wav"
        recorder = Path(self.temp.name) / "recorder"
        recorder.write_text(
            "#!/usr/bin/env python3\n"
            "import time,wave\nfrom pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('started')\n"
            "with wave.open(__import__('sys').argv[-1], 'wb') as out:\n"
            " out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)\n"
            " out.writeframes(b'\\x00\\x00' * 160)\n"
            " try: time.sleep(30)\n"
            " except KeyboardInterrupt: pass\n")
        recorder.chmod(0o700)
        self.assertFalse(voice.record_to_wav(output, "sony-source", recorder,
                                             lambda _: "cancel"))
        self.assertFalse(marker.exists())
        with self.assertRaisesRegex(voice.VoiceError, "explicit Enter"):
            voice.record_to_wav(output, "sony-source", recorder, lambda _: "anything")
        self.assertFalse(marker.exists())

        calls = iter(("", ""))

        def press(_prompt):
            value = next(calls)
            if marker.exists():
                return value
            if _prompt.startswith("Recording"):
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
            return value

        self.assertTrue(voice.record_to_wav(output, "sony-source", recorder, press))
        self.assertTrue(output.exists())

    def test_stop_terminates_owned_system_voice(self):
        marker = Path(self.temp.name) / "speech.pid"
        executable = Path(self.temp.name) / "system-speech"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import os,time\nfrom pathlib import Path\n"
            f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(30)\n")
        executable.chmod(0o700)

        def stop_when_running(_prompt):
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists())
            return "stop"

        voice.speak_and_control("Status is good.", executable, stop_when_running)
        pid = int(marker.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
