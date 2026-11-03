import copy
import unittest

from server.deepagents_edit_acceptance import assess


def good_report():
    identity = {"server_instance_id": "instance-1", "serving_identity": "artifact-1"}
    return {
        "identity_before": identity, "identity_after": copy.deepcopy(identity),
        "calls": [
            {"name": "read_fixture", "outcome": "digest"},
            {"name": "edit_fixture", "outcome": "digest"},
            {"name": "run_validator", "outcome": {"exit_code": 0}},
        ],
        "messages": [
            {"type": "AIMessage", "tool_calls": [{"id": "r1", "name": "read_fixture"}]},
            {"type": "ToolMessage", "tool_call_id": "r1"},
            {"type": "AIMessage", "tool_calls": [{"id": "e1", "name": "edit_fixture"}]},
            {"type": "ToolMessage", "tool_call_id": "e1"},
            {"type": "AIMessage", "tool_calls": [{"id": "v1", "name": "run_validator"}]},
            {"type": "ToolMessage", "tool_call_id": "v1"},
            {"type": "AIMessage", "content": "cke-value PASS", "tool_calls": [],
             "finish_reason": "stop"},
        ],
        "validation": {"exit_code": 0},
        "validator_sha256_before": "validator-digest",
        "validator_sha256_after": "validator-digest",
    }


class DeepAgentsEditAcceptanceTests(unittest.TestCase):
    def test_valid_trace(self):
        self.assertEqual(assess(good_report(), "cke-value"), [])

    def test_missing_tool_result(self):
        report = good_report()
        report["messages"] = [m for m in report["messages"]
                              if m.get("tool_call_id") != "e1"]
        self.assertIn("missing_or_duplicate_tool_result", assess(report, "cke-value"))

    def test_result_before_call(self):
        report = good_report()
        result = report["messages"].pop(3)
        report["messages"].insert(2, result)
        self.assertIn("missing_or_duplicate_tool_result", assess(report, "cke-value"))

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


if __name__ == "__main__":
    unittest.main()
