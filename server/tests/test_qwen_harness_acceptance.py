"""The local Qwen task must agree across tool events, artifact and recording."""

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from server import qwen_harness_acceptance as acceptance
from server.qwen_harness_acceptance import EXPECTED_HTML, evaluate


def test_outer_termination_reaps_qwen_client_process_group(tmp_path):
    """A coordinator timeout must not leave a detached Qwen process running."""
    pid_file = tmp_path / 'child.pid'
    ready_file = tmp_path / 'ready.txt'
    done_file = tmp_path / 'done.txt'
    source = (
        'import subprocess,sys; from pathlib import Path; '
        'from server.qwen_harness_acceptance import _wait_for_client; '
        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"], '
        'start_new_session=True); Path(sys.argv[1]).write_text(str(p.pid)); '
        'Path(sys.argv[3]).write_text(str(_wait_for_client(p,60, '
        'lambda: Path(sys.argv[2]).write_text("ready"))))'
    )
    runner = subprocess.Popen([sys.executable, '-c', source, str(pid_file),
                               str(ready_file), str(done_file)])
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_file.exists()
        child_pid = int(pid_file.read_text())
        os.kill(runner.pid, signal.SIGTERM)
        assert runner.wait(timeout=5) == 0
        assert done_file.read_text() == str(128 + signal.SIGTERM)
        assert not Path(f'/proc/{child_pid}').exists()
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()
        if child_pid is not None and Path(f'/proc/{child_pid}').exists():
            os.killpg(child_pid, signal.SIGKILL)


SESSION = '7e218e94-5a31-4ccb-a284-130c935bdd79'
MODEL = 'qwen-test'
SUMMARY = 'Changed the heading and ran python3 test_html.py; PASS: expected heading present.'


def _evidence():
    events = [
        {'type': 'system', 'subtype': 'init', 'session_id': SESSION, 'model': MODEL},
    ]
    recording = [{'type': 'user', 'sessionId': SESSION}]
    for number, name in enumerate(('read_file', 'edit', 'run_shell_command'), 1):
        call_id = f'call_{number}'
        arguments = (
            {'file_path': 'index.html'} if name == 'read_file' else
            {'file_path': 'index.html', 'old_string': '<h1>Before</h1>',
             'new_string': '<h1>CKE tool round trip</h1>'} if name == 'edit' else
            {'command': 'python3 test_html.py'}
        )
        events.append({'type': 'assistant', 'session_id': SESSION, 'message': {
            'content': [{'type': 'tool_use', 'id': call_id, 'name': name,
                         'input': arguments}],
        }})
        events.append({'type': 'user', 'session_id': SESSION, 'message': {
            'content': [{'type': 'tool_result', 'tool_use_id': call_id,
                         'is_error': False,
                         'content': 'PASS: expected heading present' if number == 3 else 'ok'}],
        }})
        recording.append({'type': 'assistant', 'provenance': 'assistant_output',
                          'sessionId': SESSION, 'message': {'parts': [
                              {'functionCall': {'id': call_id, 'name': name,
                                                'args': arguments.copy()}}]}})
        recording.append({'type': 'tool_result', 'sessionId': SESSION,
                          'toolCallResult': {'callId': call_id, 'status': 'success'}})
    recording.append({'type': 'assistant', 'provenance': 'assistant_output',
                      'sessionId': SESSION, 'message': {'parts': [{'text': SUMMARY}]}})
    events.append({'type': 'result', 'session_id': SESSION, 'subtype': 'success',
                   'is_error': False, 'result': SUMMARY})
    return events, recording


def _check(events, recording, *, html=EXPECTED_HTML, exit_code=0):
    return evaluate(events, recording, session_id=SESSION, html=html,
                    exit_code=exit_code, expected_model=MODEL)


def test_complete_tool_task_passes():
    events, recording = _evidence()
    result = _check(events, recording)
    assert result['status'] == 'pass'
    assert result['tool_sequence'] == ['read_file', 'edit', 'run_shell_command']


@pytest.mark.parametrize('task_status,certification_status,expected_exit', [
    ('pass', 'pass', 0),
    ('pass', 'incomplete', 1),
    ('pass', 'fail', 1),
    ('fail', 'pass', 1),
])
def test_cli_requires_task_and_artifact_certification_pass(
        monkeypatch, tmp_path, capsys, task_status, certification_status, expected_exit):
    monkeypatch.setattr(acceptance, 'run', lambda _args: {
        'result': {'status': task_status, 'errors': []},
        'certification': {'status': certification_status, 'reason': 'test'},
    })
    monkeypatch.setattr(sys, 'argv', [
        'qwen_harness_acceptance', '--endpoint', 'http://example.test/v1',
        '--model', MODEL, '--run-dir', str(tmp_path),
    ])
    assert acceptance.main() == expected_exit
    summary = json.loads(capsys.readouterr().out)
    assert summary['status'] == task_status
    assert summary['certification_status'] == certification_status


def test_missing_stream_final_fails_even_if_recording_and_artifact_pass():
    events, recording = _evidence()
    result = _check(events[:-1], recording)
    assert result['status'] == 'fail'
    assert 'stream_final_missing_or_failed' in result['errors']
    assert result['recorded_summary_present']


def test_wrong_call_id_fails_before_final_summary_can_certify():
    events, recording = _evidence()
    events[2]['message']['content'][0]['tool_use_id'] = 'call_from_other_response'
    result = _check(events, recording)
    assert result['status'] == 'fail'
    assert 'missing_or_unmatched_tool_result' in result['errors']


def test_failed_test_or_unrelated_edit_fails():
    events, recording = _evidence()
    events[-2]['message']['content'][0]['content'] = 'FAIL: missing heading'
    result = _check(events, recording, html=EXPECTED_HTML + '<!-- unrelated edit -->')
    assert result['status'] == 'fail'
    assert 'validator_result_missing' in result['errors']
    assert 'artifact_mismatch' in result['errors']


def test_recording_mismatch_fails_even_when_stream_claims_success():
    events, recording = _evidence()
    recording[-1]['message']['parts'][0]['text'] = 'A different result.'
    result = _check(events, recording)
    assert result['status'] == 'fail'
    assert 'stream_recording_summary_mismatch' in result['errors']


def test_optional_read_of_validator_is_allowed_but_other_paths_are_rejected():
    events, recording = _evidence()
    events.insert(3, {'type': 'assistant', 'session_id': SESSION, 'message': {
        'content': [{'type': 'tool_use', 'id': 'call_extra', 'name': 'read_file',
                     'input': {'file_path': 'test_html.py'}}],
    }})
    events.insert(4, {'type': 'user', 'session_id': SESSION, 'message': {
        'content': [{'type': 'tool_result', 'tool_use_id': 'call_extra',
                     'is_error': False, 'content': 'validator source'}],
    }})
    recording.insert(3, {'type': 'assistant', 'provenance': 'assistant_output',
                         'sessionId': SESSION, 'message': {'parts': [
                             {'functionCall': {'id': 'call_extra', 'name': 'read_file',
                                               'args': {'file_path': 'test_html.py'}}}]}})
    recording.insert(4, {'type': 'tool_result', 'sessionId': SESSION,
                         'toolCallResult': {'callId': 'call_extra', 'status': 'success'}})
    assert _check(events, recording)['status'] == 'pass'
    assert evaluate(events, recording, session_id=SESSION, html=EXPECTED_HTML,
                    exit_code=0, expected_model=MODEL,
                    fixture_dir=Path('/tmp/fixture'))['status'] == 'pass'
    events[3]['message']['content'][0]['input']['file_path'] = '../private.txt'
    result = evaluate(events, recording, session_id=SESSION, html=EXPECTED_HTML,
                      exit_code=0, expected_model=MODEL,
                      fixture_dir=Path('/tmp/fixture'))
    assert 'unexpected_tool_path' in result['errors']


def test_shell_command_must_be_exact_validator():
    events, recording = _evidence()
    events[1]['message']['content'][0]['input'] = {'file_path': 'index.html'}
    events[3]['message']['content'][0]['input'] = {'file_path': 'index.html'}
    events[5]['message']['content'][0]['input'] = {
        'command': 'python3 test_html.py; echo misleading PASS'}
    result = evaluate(events, recording, session_id=SESSION, html=EXPECTED_HTML,
                      exit_code=0, expected_model=MODEL,
                      fixture_dir=Path('/tmp/fixture'))
    assert 'unexpected_validation_command' in result['errors']


def test_stale_session_or_missing_recorded_tool_result_fails():
    events, recording = _evidence()
    events[1]['session_id'] = 'older-session'
    recording[2]['toolCallResult']['callId'] = 'another-call'
    result = _check(events, recording)
    assert result['status'] == 'fail'
    assert 'mixed_stream_sessions' in result['errors']
    assert 'stream_recording_tool_results_mismatch' in result['errors']


def test_nonzero_harness_exit_cannot_pass_with_good_artifact():
    events, recording = _evidence()
    result = _check(events, recording, exit_code=124)
    assert result['status'] == 'fail'
    assert 'qwen_nonzero_exit' in result['errors']


def test_tool_result_must_follow_its_own_call_before_next_step():
    events, recording = _evidence()
    events[2], events[4] = events[4], events[2]
    result = _check(events, recording)
    assert result['status'] == 'fail'
    assert 'tool_result_order_mismatch' in result['errors']


def test_parallel_reads_finish_before_edit():
    events, recording = _evidence()
    events[1]['message']['content'].append(
        {'type': 'tool_use', 'id': 'call_extra', 'name': 'read_file',
         'input': {'file_path': 'index.html'}})
    events.insert(3, {'type': 'user', 'session_id': SESSION, 'message': {
        'content': [{'type': 'tool_result', 'tool_use_id': 'call_extra',
                     'is_error': False, 'content': 'validator source'}]}})
    recording[1]['message']['parts'].append(
        {'functionCall': {'id': 'call_extra', 'name': 'read_file',
                          'args': {'file_path': 'index.html'}}})
    recording.insert(3, {'type': 'tool_result', 'sessionId': SESSION,
                         'toolCallResult': {'callId': 'call_extra', 'status': 'success'}})
    assert _check(events, recording)['status'] == 'pass'

    early_edit = list(events)
    early_edit[3], early_edit[4] = early_edit[4], early_edit[3]
    result = _check(early_edit, recording)
    assert 'tool_result_order_mismatch' in result['errors']


def test_recorded_tool_argument_mismatches_fail_for_path_edit_and_command():
    changes = (
        (0, 'file_path', '../elsewhere.html'),
        (1, 'new_string', '<h1>Unrelated edit</h1>'),
        (2, 'command', 'python3 unrelated.py'),
    )
    for call_index, key, value in changes:
        events, recording = _evidence()
        recording[1 + 2 * call_index]['message']['parts'][0]['functionCall']['args'][key] = value
        result = _check(events, recording)
        assert result['status'] == 'fail'
        assert 'stream_recording_tool_arguments_mismatch' in result['errors']


def test_terminal_must_follow_all_tool_results_and_end_tool_activity():
    events, recording = _evidence()
    early = [events[0], events[-1], *events[1:-1]]
    result = _check(early, recording)
    assert result['status'] == 'fail'
    assert 'stream_terminal_order_mismatch' in result['errors']
    assert 'tool_activity_after_terminal' in result['errors']

    events, recording = _evidence()
    events.append(events[-1].copy())
    assert 'stream_terminal_order_mismatch' in _check(events, recording)['errors']


def _loaded_identity(*, digest='a' * 64, instance='b' * 32):
    return {
        'schema': acceptance.LOADED_IDENTITY_SCHEMA,
        'model': MODEL,
        'serving_identity': digest,
        'session_library_sha256': 'c' * 64,
        'server_instance_id': instance,
        'effective_serving': {
            'schema': 'cke.effective_serving.v1',
            'configured_mode': 'templated', 'output_protocol': 'qwen_xml',
            'active_context_limit': 16384, 'default_max_output_tokens': 512,
            'stop_on_text': [], 'stop_at_eos': True,
        },
        'assets_sha256': {name: 'd' * 64 for name in (
            'libmodel.so', 'libckernel_engine.so', 'libckernel_tokenizer.so')},
    }


def test_loaded_identity_verdict_requires_expected_bundle_and_stable_server():
    observed = _loaded_identity()
    identity, certification = acceptance._identity_verdict(
        observed, observed.copy(), expected='a' * 64, expected_session='c' * 64,
        model=MODEL, task_passed=True)
    assert identity['level'] == 'expected_loaded_bundle'
    assert certification['status'] == 'pass'
    for after, reason in ((None, 'loaded_artifact_identity_changed'),
                          (_loaded_identity(instance='e' * 32), 'loaded_artifact_identity_changed')):
        _, certification = acceptance._identity_verdict(
            observed, after, expected='a' * 64, expected_session='c' * 64,
            model=MODEL, task_passed=True)
        assert certification == {'status': 'fail', 'reason': reason}
    _, certification = acceptance._identity_verdict(
        observed, observed, expected=None, expected_session=None,
        model=MODEL, task_passed=True)
    assert certification['status'] == 'incomplete'
    _, certification = acceptance._identity_verdict(
        observed, observed, expected='a' * 64, expected_session='c' * 64,
        model=MODEL, task_passed=False)
    assert certification['status'] == 'fail'
    _, certification = acceptance._identity_verdict(
        observed, observed, expected='a' * 64, expected_session=None,
        model=MODEL, task_passed=True)
    assert certification['reason'] == 'expected_session_library_identity_not_supplied'


@pytest.mark.parametrize('expected,expected_session,status,reason', [
    (None, None, 'incomplete', 'expected_serving_identity_not_supplied'),
    ('a' * 64, None, 'incomplete', 'expected_session_library_identity_not_supplied'),
    (None, 'c' * 64, 'incomplete', 'expected_serving_identity_not_supplied'),
    ('a' * 64, 'c' * 64, 'pass', None),
    ('e' * 64, None, 'fail', 'loaded_artifact_identity_mismatch'),
    (None, 'e' * 64, 'fail', 'loaded_session_library_identity_mismatch'),
    ('e' * 64, 'c' * 64, 'fail', 'loaded_artifact_identity_mismatch'),
    ('a' * 64, 'e' * 64, 'fail', 'loaded_session_library_identity_mismatch'),
])
def test_identity_expectations_are_independent(expected, expected_session, status, reason):
    observed = _loaded_identity()
    assert acceptance._identity_expectation_failure(
        observed, expected=expected, expected_session=expected_session, model=MODEL
    ) == (reason if status == 'fail' else None)
    _, certification = acceptance._identity_verdict(
        observed, observed, expected=expected, expected_session=expected_session,
        model=MODEL, task_passed=True)
    assert certification == {'status': status, 'reason': reason}


@pytest.mark.parametrize('expected,expected_session', [
    ('a' * 64, None), (None, 'c' * 64), ('a' * 64, 'c' * 64),
])
def test_expected_identity_requires_endpoint(expected, expected_session):
    _, certification = acceptance._identity_verdict(
        None, None, expected=expected, expected_session=expected_session,
        model=MODEL, task_passed=True)
    assert certification == {'status': 'fail', 'reason': 'loaded_artifact_identity_unavailable'}


def test_loaded_identity_probe_rejects_malformed_or_incomplete_assets(monkeypatch):
    document = _loaded_identity()
    document['assets_sha256'] = {}
    monkeypatch.setattr(acceptance, 'urlopen',
                        lambda *_args, **_kwargs: io.BytesIO(json.dumps(document).encode()))
    with pytest.raises(ValueError, match='malformed loaded serving identity'):
        acceptance._read_loaded_identity('http://example.test/v1')


def test_loaded_identity_requires_effective_serving_and_detects_setting_change(monkeypatch):
    document = _loaded_identity()
    document.pop('effective_serving')
    monkeypatch.setattr(acceptance, 'urlopen',
                        lambda *_args, **_kwargs: io.BytesIO(json.dumps(document).encode()))
    with pytest.raises(ValueError, match='malformed loaded serving identity'):
        acceptance._read_loaded_identity('http://example.test/v1')
    before = _loaded_identity()
    after = _loaded_identity()
    after['effective_serving']['active_context_limit'] = 8192
    _, certification = acceptance._identity_verdict(
        before, after, expected='a' * 64, expected_session='c' * 64,
        model=MODEL, task_passed=True)
    assert certification == {'status': 'fail', 'reason': 'loaded_artifact_identity_changed'}


@pytest.mark.parametrize('malformed', [False, True])
def test_bad_identity_does_not_start_harness(monkeypatch, tmp_path, malformed):
    observed = _loaded_identity(digest='e' * 64)
    if malformed:
        observed['assets_sha256'] = {}

    def fake_urlopen(url, **_kwargs):
        payload = ({'data': [{'id': MODEL}]} if url.endswith('/models') else observed)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(acceptance, 'urlopen', fake_urlopen)
    monkeypatch.setattr(acceptance.subprocess, 'run',
                        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, '0.24.6\n'))
    monkeypatch.setattr(acceptance.subprocess, 'Popen',
                        lambda *_args, **_kwargs: pytest.fail('harness started on stale bundle'))
    args = argparse.Namespace(timeout=10, max_output_tokens=16, qwen_bin='qwen',
                              expected_qwen_version='0.24.6', run_dir=tmp_path / 'task',
                              endpoint='http://example.test/v1', model=MODEL,
                              expected_serving_identity='a' * 64,
                              expected_session_library_sha256='c' * 64)
    report = acceptance.run(args)
    assert report['result']['status'] == 'not_run'
    assert report['certification']['reason'] == (
        'loaded_identity_preflight_probe_failed' if malformed
        else 'loaded_artifact_identity_mismatch')
    assert report['qwen_code_version'] == '0.24.6'
    assert report['qwen_profile'] == 'bare'
    assert report['time_budget_seconds'] == 10
    assert json.loads((tmp_path / 'task/report.json').read_text()) == report


@pytest.mark.parametrize('expected,expected_session,reason', [
    ('e' * 64, None, 'loaded_artifact_identity_mismatch'),
    (None, 'e' * 64, 'loaded_session_library_identity_mismatch'),
    ('e' * 64, 'c' * 64, 'loaded_artifact_identity_mismatch'),
    ('a' * 64, 'e' * 64, 'loaded_session_library_identity_mismatch'),
])
def test_individual_identity_mismatch_prevents_client_launch(
        monkeypatch, tmp_path, expected, expected_session, reason):
    def fake_urlopen(url, **_kwargs):
        payload = {'data': [{'id': MODEL}]} if url.endswith('/models') else _loaded_identity()
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(acceptance, 'urlopen', fake_urlopen)
    monkeypatch.setattr(acceptance.subprocess, 'run',
                        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, '0.24.6\n'))
    monkeypatch.setattr(acceptance.subprocess, 'Popen',
                        lambda *_args, **_kwargs: pytest.fail('harness started on mismatched identity'))
    args = argparse.Namespace(timeout=10, max_output_tokens=16, qwen_bin='qwen',
                              expected_qwen_version='0.24.6', run_dir=tmp_path / 'task',
                              endpoint='http://example.test/v1', model=MODEL,
                              expected_serving_identity=expected,
                              expected_session_library_sha256=expected_session)
    report = acceptance.run(args)
    assert report['result']['status'] == 'not_run'
    assert report['certification'] == {'status': 'fail', 'reason': reason, 'detail': None}


@pytest.mark.parametrize('version,profile,expected_bare,budget_mode', [
    ('0.24.6', 'bare', True, None),
    ('0.25.1', 'normal', False, 'runner'),
    ('0.25.1', 'normal', False, 'client'),
])
def test_versioned_qwen_profile_retains_settings_and_time_budget(
        monkeypatch, tmp_path, version, profile, expected_bare, budget_mode):
    def fake_urlopen(url, **_kwargs):
        payload = {'data': [{'id': MODEL}]} if url.endswith('/models') else _loaded_identity()
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(acceptance, 'urlopen', fake_urlopen)
    monkeypatch.setattr(acceptance.subprocess, 'run',
                        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, version + '\n'))
    launched = []
    def fake_popen(command, **kwargs):
        launched.append((command, kwargs))
        if budget_mode == 'client':
            kwargs['stderr'].write(
                b'Run aborted: wall-clock budget of 14400s exceeded (--max-wall-time).\n')
        return object()

    monkeypatch.setattr(acceptance.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(acceptance, '_session_file', lambda *_args: tmp_path / 'empty.jsonl')
    (tmp_path / 'empty.jsonl').write_text('')
    settings = tmp_path / 'settings.json'
    settings.write_text('{"model":{"maxWallTimeSeconds":14400,"generationConfig":{"contextWindowSize":32768}}}')

    def wait(_process, _timeout, on_ready=None, on_timeout=None):
        if budget_mode == 'runner':
            on_timeout()
            return 124
        if budget_mode == 'client':
            return 55
        return 0

    monkeypatch.setattr(acceptance, '_wait_for_client', wait)
    args = argparse.Namespace(timeout=14400, max_output_tokens=2048, qwen_bin='qwen',
                              qwen_profile=profile, expected_qwen_version=version,
                              run_dir=tmp_path / profile, settings=str(settings),
                              endpoint='http://example.test/v1', model=MODEL,
                              expected_serving_identity='a' * 64,
                              expected_session_library_sha256='c' * 64,
                              api_key='cke-local-only')
    report = acceptance.run(args)
    command, kwargs = launched[0]
    assert ('--bare' in command) is expected_bare
    assert kwargs['env']['QWEN_CODE_SYSTEM_SETTINGS_PATH'] == str(settings.resolve())
    assert report['qwen_code_version'] == version
    assert report['qwen_profile'] == profile
    assert report['qwen_settings_sha256'] == acceptance._sha256(settings)
    assert report['declared_client_wall_time_seconds'] == 14400
    assert report['time_budget_seconds'] == 14400
    assert report['time_budget_exceeded'] is (budget_mode == 'runner')
    assert report['client_time_budget_exceeded'] is (budget_mode == 'client')
    if budget_mode:
        assert report['result']['status'] == 'incomplete'
        assert report['result']['incomplete_reason'] == 'time_budget_exceeded'


def test_client_budget_abort_is_incomplete_only_with_explicit_stderr(tmp_path):
    stderr = tmp_path / 'stderr.log'
    stderr.write_text('Run aborted: wall-clock budget of 1200s exceeded (--max-wall-time).\n')
    assert acceptance._client_time_budget_exceeded(55, stderr)
    assert not acceptance._client_time_budget_exceeded(0, stderr)
    stderr.write_text('Runtime error during generation.\n')
    assert not acceptance._client_time_budget_exceeded(55, stderr)
