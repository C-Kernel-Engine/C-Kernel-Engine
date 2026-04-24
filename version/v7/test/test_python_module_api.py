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
    return ck.models.qwen3_tiny(
        vocab=256,
        dim=128,
        layers=2,
        hidden=256,
        heads=8,
        kv_heads=4,
        context_len=128,
        init='xavier_uniform',
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
                config=ck.CompileConfig(
                    target=ck.TargetConfig(name='cpu', isa='auto'),
                    vectorize=True,
                    pack_weights=True,
                    unroll=2,
                    kernel_policy='fp32_reference_first',
                ),
                command_runner=fake_runner,
            )

            self.assertEqual(compiled.family, 'qwen3')
            self.assertEqual(compiled.project.model.layers, 2)
            self.assertEqual(compiled.project.model.embed_dim, 128)
            self.assertEqual(compiled.project.model.hidden_dim, 256)
            self.assertIn('# tiny_qwen3_module_api', compiled.show_graph())
            self.assertEqual(compiled.show_compile_config()['unroll'], 2)
            self.assertTrue(compiled.show_pass_trace())
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
            compile_config_path = run_dir / 'python_authoring_compile_config.json'
            pass_trace_path = run_dir / 'python_authoring_pass_trace.json'
            self.assertTrue(graph_path.exists())
            self.assertTrue(graph_markdown_path.exists())
            self.assertTrue(compile_config_path.exists())
            self.assertTrue(pass_trace_path.exists())
            graph_payload = json.loads(graph_path.read_text(encoding='utf-8'))
            self.assertEqual(graph_payload['schema'], 'ck.python_authoring.graph.v1')
            self.assertEqual(graph_payload['name'], 'tiny_qwen3_module_api')

            template_payload = json.loads((run_dir / 'template_python_ui.json').read_text(encoding='utf-8'))
            self.assertIn('python_authoring', template_payload)
            self.assertEqual(template_payload['python_authoring']['family'], 'qwen3')
            self.assertEqual(template_payload['python_authoring']['graph']['schema'], 'ck.python_authoring.graph.v1')
            self.assertEqual(template_payload['python_authoring']['compile_config']['target']['name'], 'cpu')
            self.assertTrue(template_payload['python_authoring']['pass_trace'])

            compile_payload = json.loads(compile_config_path.read_text(encoding='utf-8'))
            self.assertEqual(compile_payload['schema'], 'ck.python_authoring.compile_config.v1')
            self.assertEqual(compile_payload['compile_config']['kernel_policy'], 'fp32_reference_first')
            pass_payload = json.loads(pass_trace_path.read_text(encoding='utf-8'))
            self.assertEqual(pass_payload['schema'], 'ck.python_authoring.pass_trace.v1')
            self.assertTrue(pass_payload['passes'])

            plan_payload = json.loads((run_dir / 'python_authoring_plan.json').read_text(encoding='utf-8'))
            self.assertEqual(plan_payload['artifacts']['python_authoring_graph'], str(graph_path))
            self.assertEqual(plan_payload['artifacts']['python_authoring_graph_markdown'], str(graph_markdown_path))
            self.assertEqual(plan_payload['artifacts']['python_authoring_compile_config'], str(compile_config_path))
            self.assertEqual(plan_payload['artifacts']['python_authoring_pass_trace'], str(pass_trace_path))
            self.assertEqual(plan_payload['history'][0]['action'], 'materialize')

    def test_compile_rejects_conflicting_kernel_policy(self) -> None:
        with self.assertRaises(ValueError):
            ck.v7.compile(
                _build_model(),
                run_name='conflicting-kernel-policy',
                kernel_policy='fp32_reference_first',
                config=ck.CompileConfig(kernel_policy='other_policy'),
            )

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
