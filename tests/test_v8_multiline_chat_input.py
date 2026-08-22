import importlib.util
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_chat_module():
    path = ROOT / "scripts" / "ck_chat.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ck_chat_multiline_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MultilineChatInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chat = _load_chat_module()

    def test_multiline_turn_preserves_blank_lines_and_submits_once(self) -> None:
        lines = iter(
            [
                "Summarize this message:",
                "",
                "Hi Ownr Support,",
                "The first charge was on May 30.",
                "/send",
            ]
        )
        prompts = []

        def read(prompt: str) -> str:
            prompts.append(prompt)
            return next(lines)

        turn = self.chat._read_interactive_turn(
            multiline_input=True,
            input_fn=read,
        )

        self.assertEqual(
            turn,
            "Summarize this message:\n\nHi Ownr Support,\n"
            "The first charge was on May 30.",
        )
        self.assertEqual(len(prompts), 5)

    def test_multiline_mode_accepts_slash_terminator(self) -> None:
        lines = iter(["line one", "line two", "/"])
        turn = self.chat._read_interactive_turn(
            multiline_input=True,
            input_fn=lambda _prompt: next(lines),
        )
        self.assertEqual(turn, "line one\nline two")

    def test_repl_command_does_not_require_multiline_terminator(self) -> None:
        calls = []

        def read(_prompt: str) -> str:
            calls.append(1)
            return "/exit"

        turn = self.chat._read_interactive_turn(
            multiline_input=True,
            input_fn=read,
        )
        self.assertEqual(turn, "/exit")
        self.assertEqual(len(calls), 1)

    def test_memory_flag_replays_complete_conversation(self) -> None:
        class FakeModel:
            def __init__(self):
                self.conversations = []
                self.use_chat_template = False

            def format_chat_conversation(self, conversation):
                self.conversations.append(list(conversation))
                return f"formatted-{len(self.conversations)}"

        model = FakeModel()
        lines = iter(["first turn", "second turn", "/exit"])
        responses = iter(["first response", "second response"])
        original_generate = self.chat.generate
        self.chat.generate = lambda *_args, **_kwargs: next(responses)
        try:
            with redirect_stdout(io.StringIO()):
                self.chat.chat_loop(
                    model,
                    memory_enabled=True,
                    show_stats=False,
                    input_fn=lambda _prompt: next(lines),
                )
        finally:
            self.chat.generate = original_generate

        self.assertEqual(
            model.conversations,
            [
                [("user", "first turn")],
                [
                    ("user", "first turn"),
                    ("assistant", "first response"),
                    ("user", "second turn"),
                ],
            ],
        )

    def test_runner_and_chat_help_advertise_same_flag(self) -> None:
        commands = [
            [sys.executable, str(ROOT / "scripts" / "ck_chat.py"), "--help"],
            [
                sys.executable,
                str(ROOT / "version" / "v8" / "scripts" / "ck_run_v8.py"),
                "run",
                "--help",
            ],
        ]
        for command in commands:
            with self.subTest(command=command):
                proc = subprocess.run(
                    command,
                    cwd=ROOT,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("--multiline-input", proc.stdout)


if __name__ == "__main__":
    unittest.main()
