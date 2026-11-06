import copy
import tempfile
import unittest
from pathlib import Path

from server.deepagents_edit_acceptance import VALIDATOR_OUTPUT, assess, validate_fixture


def good_report():
    identity = {"server_instance_id": "instance-1", "serving_identity": "artifact-1"}
    return {
        "identity_before": identity, "identity_after": copy.deepcopy(identity),
        "calls": [
            {"name": "read_fixture", "arguments": {"path": "note.txt"}, "outcome": "digest"},
            {"name": "edit_fixture", "arguments": {
                "path": "module.py", "old": "a - b", "new": "a + b"}, "outcome": "digest"},
            {"name": "run_validator", "arguments": {}, "outcome": {
                "exit_code": 0, "assertion_completed": True, "output": VALIDATOR_OUTPUT}},
        ],
        "messages": [
            {"type": "AIMessage", "tool_calls": [{"id": "r1", "name": "read_fixture",
                                                   "args": {"path": "note.txt"}}]},
            {"type": "ToolMessage", "tool_call_id": "r1"},
            {"type": "AIMessage", "tool_calls": [{"id": "e1", "name": "edit_fixture",
                                                   "args": {"path": "module.py", "old": "a - b", "new": "a + b"}}]},
            {"type": "ToolMessage", "tool_call_id": "e1"},
            {"type": "AIMessage", "tool_calls": [{"id": "v1", "name": "run_validator", "args": {}}]},
            {"type": "ToolMessage", "tool_call_id": "v1"},
            {"type": "AIMessage", "content": "cke-value PASS", "tool_calls": [],
             "finish_reason": "stop"},
        ],
        "validation": {"exit_code": 0, "assertion_completed": True,
                       "stdout": VALIDATOR_OUTPUT},
        "validator_sha256_before": "validator-digest",
        "validator_sha256_after": "validator-digest",
    }


class DeepAgentsEditAcceptanceTests(unittest.TestCase):
    def test_valid_trace(self):
        self.assertEqual(assess(good_report(), "cke-value"), [])

    def test_injected_read_error_must_be_retried_before_edit(self):
        report = good_report()
        report["calls"].insert(0, {"name": "read_fixture",
                                   "arguments": {"path": "note.txt"},
                                   "outcome": "injected_failure"})
        report["messages"][:0] = [
            {"type": "AIMessage", "tool_calls": [{"id": "failed-read",
                "name": "read_fixture", "args": {"path": "note.txt"}}]},
            {"type": "ToolMessage", "tool_call_id": "failed-read",
             "content": "ERROR: temporary read failure; retry the same path."},
        ]
        self.assertEqual(assess(report, "cke-value", expect_read_recovery=True), [])
        report["messages"][1]["content"] = "ordinary file content"
        self.assertIn("read_failure_result_missing",
                      assess(report, "cke-value", expect_read_recovery=True))
        report["messages"][1]["content"] = "ERROR: temporary read failure; retry the same path."
        report["calls"][1]["arguments"]["path"] = "module.py"
        self.assertIn("read_failure_not_recovered",
                      assess(report, "cke-value", expect_read_recovery=True))
        report["calls"].pop(0)
        report["messages"] = report["messages"][2:]
        self.assertIn("read_failure_not_recovered",
                      assess(report, "cke-value", expect_read_recovery=True))

    def test_missing_tool_result(self):
        report = good_report()
        report["messages"] = [m for m in report["messages"]
                              if m.get("tool_call_id") != "e1"]
        self.assertIn("unresolved_tool_call", assess(report, "cke-value"))

    def test_result_before_call(self):
        report = good_report()
        result = report["messages"].pop(3)
        report["messages"].insert(2, result)
        self.assertIn("orphan_or_duplicate_tool_result", assess(report, "cke-value"))

    def test_validator_failure_and_modified_fixture(self):
        report = good_report()
        report["calls"][-1]["outcome"]["exit_code"] = 1
        report["validator_sha256_after"] = "changed"
        errors = assess(report, "cke-value")
        self.assertIn("validator_not_passed_by_agent", errors)
        self.assertIn("validator_modified", errors)

    def test_identity_and_answer_must_match(self):
        report = good_report()
        report["identity_after"]["server_instance_id"] = "instance-2"
        report["messages"][-1]["content"] = "Done"
        errors = assess(report, "cke-value")
        self.assertIn("loaded_identity_changed_or_missing", errors)
        self.assertIn("final_answer_missing_value", errors)

    def test_emitted_tool_must_execute(self):
        report = good_report()
        report["calls"].pop(1)
        self.assertIn("tool_execution_mismatch", assess(report, "cke-value"))

    def test_reused_call_id_cannot_share_one_result(self):
        report = good_report()
        for message in report["messages"]:
            for call in message.get("tool_calls") or []:
                call["id"] = "duplicate"
        report["messages"] = [message for message in report["messages"]
                              if message.get("type") != "ToolMessage"]
        report["messages"].insert(-1, {"type": "ToolMessage", "tool_call_id": "duplicate"})
        self.assertIn("reused_tool_call_id", assess(report, "cke-value"))

    def test_orphan_result_is_rejected(self):
        report = good_report()
        report["messages"].insert(-1, {"type": "ToolMessage", "tool_call_id": "orphan"})
        self.assertIn("orphan_or_duplicate_tool_result", assess(report, "cke-value"))

    def test_duplicate_result_and_argument_mismatch_are_rejected(self):
        report = good_report()
        report["messages"].insert(2, {"type": "ToolMessage", "tool_call_id": "r1"})
        report["calls"][0]["arguments"] = {"path": "wrong.txt"}
        errors = assess(report, "cke-value")
        self.assertIn("orphan_or_duplicate_tool_result", errors)
        self.assertIn("tool_arguments_mismatch", errors)

    def test_trusted_validator_reports_completed_assertion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "module.py").write_text("def add(a, b):\n    return a + b\n")
            (root / "test_module.py").write_text(
                "from module import add\nassert add(2, 3) == 5\n"
                "print('PASS: add(2, 3) == 5')\n")
            result = validate_fixture(root)
            self.assertEqual(result["exit_code"], 0)
            self.assertTrue(result["assertion_completed"])
            self.assertEqual(result["output"], VALIDATOR_OUTPUT)

    def test_early_exit_cannot_fake_assertion_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "module.py").write_text(
                "def add(a, b):\n    return __import__('sys').exit(0)\n")
            (root / "test_module.py").write_text(
                "from module import add\nassert add(2, 3) == 5\n"
                "print('PASS: add(2, 3) == 5')\n")
            result = validate_fixture(root)
            self.assertNotEqual(result["exit_code"], 0)
            self.assertFalse(result["assertion_completed"])
            self.assertNotEqual(result["output"], VALIDATOR_OUTPUT)

    def test_zero_exit_without_completion_cannot_pass_assessor(self):
        report = good_report()
        report["calls"][-1]["outcome"]["assertion_completed"] = False
        report["calls"][-1]["outcome"]["output"] = ""
        report["validation"]["assertion_completed"] = False
        report["validation"]["stdout"] = ""
        self.assertIn("validator_not_passed_by_agent", assess(report, "cke-value"))


if __name__ == "__main__":
    unittest.main()
