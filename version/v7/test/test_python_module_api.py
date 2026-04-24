#!/usr/bin/env python3
"""Focused tests for the ck.nn -> v7 adapter surface."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ckernel_engine as ck  # noqa: E402


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, command, cwd) -> None:
        self.calls.append(([str(part) for part in command], Path(cwd)))


def _build_model() -> ck.nn.Sequential:
    return ck.nn.Sequential(
        ck.nn.Embedding(vocab=256, dim=128, init='xavier_uniform', name='tokens'),
        ck.nn.TransformerBlock(dim=128, hidden=256, heads=8, kv_heads=4, context_len=128, init='xavier_uniform', name='block0'),
        ck.nn.TransformerBlock(dim=128, hidden=256, heads=8, kv_heads=4, context_len=128, init='xavier_uniform', name='block1'),
        ck.nn.RMSNorm(128, name='final_norm'),
        ck.nn.Linear(128, 256, bias=False, init='xavier_uniform', name='lm_head'),
        name='tiny_qwen3_module_api',
    )


class PythonModuleApiTest(unittest.TestCase):
    def test_compile_builds_graph_and_v7_project(self) -> None:
        fake_runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            compiled = ck.v7.compile(
                _build_model(),
                run_name='py-module-api',
                run_dir=Path(tmp) / 'run',
                family='qwen3',
                init='xavier_uniform',
                command_runner=fake_runner,
            )

            self.assertEqual(compiled.family, 'qwen3')
            self.assertEqual(compiled.project.model.layers, 2)
            self.assertEqual(compiled.project.model.embed_dim, 128)
            self.assertEqual(compiled.project.model.hidden_dim, 256)
            self.assertIn('# tiny_qwen3_module_api', compiled.show_graph())
            graph_payload = compiled.show_graph(format='json')
            self.assertEqual(graph_payload['schema'], 'ck.python_authoring.graph.v1')
            self.assertGreater(graph_payload['node_count'], 0)
            self.assertEqual(fake_runner.calls, [])

    def test_materialize_writes_graph_artifacts_and_template_metadata(self) -> None:
        fake_runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            compiled = ck.v7.compile(
                _build_model(),
                run_name='py-module-api-materialize',
                run_dir=run_dir,
                family='qwen3',
                init='xavier_uniform',
                command_runner=fake_runner,
            )

            result = compiled.materialize()

            self.assertEqual(result.action, 'materialize')
            self.assertEqual(len(fake_runner.calls), 1)
            command, cwd = fake_runner.calls[0]
            self.assertEqual(cwd, REPO_ROOT)
            self.assertIn('init', command)
            self.assertIn('--template-file', command)

            graph_path = run_dir / 'python_authoring_graph.json'
            graph_markdown_path = run_dir / 'python_authoring_graph.md'
            self.assertTrue(graph_path.exists())
            self.assertTrue(graph_markdown_path.exists())
            graph_payload = json.loads(graph_path.read_text(encoding='utf-8'))
            self.assertEqual(graph_payload['schema'], 'ck.python_authoring.graph.v1')
            self.assertEqual(graph_payload['name'], 'tiny_qwen3_module_api')

            template_payload = json.loads((run_dir / 'template_python_ui.json').read_text(encoding='utf-8'))
            self.assertIn('python_authoring', template_payload)
            self.assertEqual(template_payload['python_authoring']['family'], 'qwen3')
            self.assertEqual(template_payload['python_authoring']['graph']['schema'], 'ck.python_authoring.graph.v1')

            plan_payload = json.loads((run_dir / 'python_authoring_plan.json').read_text(encoding='utf-8'))
            self.assertEqual(plan_payload['artifacts']['python_authoring_graph'], str(graph_path))
            self.assertEqual(plan_payload['artifacts']['python_authoring_graph_markdown'], str(graph_markdown_path))
            self.assertEqual(plan_payload['history'][0]['action'], 'materialize')

    def test_compile_rejects_unsupported_family_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            ck.v7.compile(_build_model(), run_name='unsupported-family', family='gemma3')

    def test_compile_rejects_unsupported_topology(self) -> None:
        unsupported = ck.nn.Sequential(
            ck.nn.Embedding(vocab=256, dim=128),
            ck.nn.Linear(128, 256, bias=False),
            name='unsupported',
        )

        with self.assertRaises(ValueError):
            ck.v7.compile(unsupported, run_name='unsupported-graph')


if __name__ == '__main__':
    unittest.main()
