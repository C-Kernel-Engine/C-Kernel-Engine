"""Bounded, local Qwen Code serving acceptance with session-bound evidence.

This runner exercises the existing Qwen Code client against an already-running
OpenAI-compatible endpoint. It never starts a model or executes server code.
The disposable task is intentionally small: read, edit, validate, summarize.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid
from urllib.error import HTTPError
from urllib.request import urlopen


SOURCE_HTML = ('<!doctype html>\n<html lang="en">\n'
               '<head><meta charset="utf-8"><title>Serving probe</title></head>\n'
               '<body><main><h1>Before</h1><p>Local harness fixture.</p></main></body>\n'
               '</html>\n')
EXPECTED_HTML = SOURCE_HTML.replace('<h1>Before</h1>', '<h1>CKE tool round trip</h1>')
TEST_SCRIPT = ('from pathlib import Path\n'
               'text = Path("index.html").read_text(encoding="utf-8")\n'
               'assert "<h1>CKE tool round trip</h1>" in text\n'
               'print("PASS: expected heading present")\n')
TASK_PROMPT = ('Read index.html. Change only its h1 text from Before to CKE tool round trip '
               'using edit. Run python3 test_html.py with run_shell_command. Summarize the '
               'test result. Do not edit test_html.py or files outside this directory.')
EXPECTED_TOOLS = ('read_file', 'edit', 'run_shell_command')
LOADED_IDENTITY_SCHEMA = 'cke.loaded_serving_identity.v1'


def _wait_for_client(process: subprocess.Popen, timeout: int, on_ready=None,
                     on_timeout=None) -> int:
    """Wait for Qwen Code and reap its separate process group on interruption."""
    class Interrupted(Exception):
        def __init__(self, signum: int):
            self.signum = signum

    def on_signal(signum, _frame):
        raise Interrupted(signum)

    previous_term = signal.signal(signal.SIGTERM, on_signal)
    previous_int = signal.signal(signal.SIGINT, on_signal)
    try:
        if on_ready is not None:
            on_ready()
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, Interrupted) as exc:
            if isinstance(exc, subprocess.TimeoutExpired) and on_timeout is not None:
                on_timeout()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            return 124 if isinstance(exc, subprocess.TimeoutExpired) else 128 + exc.signum
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _client_time_budget_exceeded(exit_code: int, stderr_path: Path) -> bool:
    if exit_code == 0:
        return False
    try:
        tail = stderr_path.read_bytes()[-8192:].decode('utf-8', 'replace')
    except OSError:
        return False
    return 'Run aborted: wall-clock budget of ' in tail and ' exceeded' in tail


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in '0123456789abcdef' for char in value)


def _valid_effective_serving(value: object) -> bool:
    if not isinstance(value, dict) or value.get('schema') != 'cke.effective_serving.v1':
        return False
    limit = value.get('active_context_limit')
    allowance = value.get('default_max_output_tokens')
    protocol = value.get('output_protocol')
    stops = value.get('stop_on_text')
    return (value.get('configured_mode') in ('raw', 'templated')
            and isinstance(limit, int) and not isinstance(limit, bool) and limit > 0
            and isinstance(allowance, int) and not isinstance(allowance, bool) and allowance > 0
            and (protocol is None or isinstance(protocol, str) and bool(protocol))
            and isinstance(stops, list) and all(isinstance(s, str) for s in stops)
            and isinstance(value.get('stop_at_eos'), bool))


def _read_loaded_identity(endpoint: str) -> dict | None:
    try:
        with urlopen(endpoint + '/cke/loaded-identity', timeout=10) as response:
            document = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:  # Older or deliberately unassessed serving path.
            return None
        raise
    if (not isinstance(document, dict)
            or document.get('schema') != LOADED_IDENTITY_SCHEMA
            or not _valid_sha256(document.get('serving_identity'))
            or not _valid_sha256(document.get('session_library_sha256'))
            or not isinstance(document.get('model'), str)
            or not document['model']
            or not isinstance(document.get('server_instance_id'), str)
            or len(document['server_instance_id']) != 32
            or any(char not in '0123456789abcdef'
                   for char in document['server_instance_id'])
            or not isinstance(document.get('assets_sha256'), dict)
            or not _valid_effective_serving(document.get('effective_serving'))
            or not {'libmodel.so', 'libckernel_engine.so', 'libckernel_tokenizer.so'}
                   <= document['assets_sha256'].keys()
            or not all(_valid_sha256(value) for value in document['assets_sha256'].values())):
        raise ValueError('malformed loaded serving identity response')
    return document


def _identity_verdict(before: dict | None, after: dict | None, *,
                      expected: str | None, expected_session: str | None,
                      model: str, task_passed: bool) -> tuple[dict, dict]:
    if before is None and after is None:
        identity = {'level': 'endpoint_model_id_only', 'before': None, 'after': None}
        mismatch = _identity_expectation_failure(before, expected=expected,
                                                 expected_session=expected_session, model=model)
        if mismatch is not None:
            return identity, {'status': 'fail', 'reason': mismatch}
        reason = 'loaded_artifact_identity_unavailable'
    elif before is None or after is None or before != after:
        identity = {'level': 'changed_or_missing', 'before': before, 'after': after}
        return identity, {'status': 'fail', 'reason': 'loaded_artifact_identity_changed'}
    else:
        identity = {'level': 'observed_loaded_bundle', 'before': before, 'after': after}
        mismatch = _identity_expectation_failure(before, expected=expected,
                                                 expected_session=expected_session, model=model)
        if mismatch is not None:
            return identity, {'status': 'fail', 'reason': mismatch}
        if expected is None:
            reason = 'expected_serving_identity_not_supplied'
        elif expected_session is None:
            reason = 'expected_session_library_identity_not_supplied'
        else:
            identity['level'] = 'expected_loaded_bundle'
            return identity, {'status': 'pass' if task_passed else 'fail',
                              'reason': None if task_passed else 'harness_task_failed'}
    return identity, {'status': 'incomplete' if task_passed else 'fail',
                      'reason': reason if task_passed else 'harness_task_failed'}


def _identity_expectation_failure(identity: dict | None, *, expected: str | None,
                                  expected_session: str | None, model: str) -> str | None:
    """Reject supplied mismatches before interpreting absent expectations."""
    if identity is None:
        return ('loaded_artifact_identity_unavailable'
                if expected is not None or expected_session is not None else None)
    if identity['model'] != model:
        return 'loaded_artifact_model_mismatch'
    if expected is not None and identity['serving_identity'] != expected:
        return 'loaded_artifact_identity_mismatch'
    if (expected_session is not None
            and identity['session_library_sha256'] != expected_session):
        return 'loaded_session_library_identity_mismatch'
    return None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path.name}:{number}: invalid JSON: {exc}') from exc
        if not isinstance(row, dict):
            raise ValueError(f'{path.name}:{number}: expected object')
        rows.append(row)
    return rows


def _session_file(qwen_bin: str, cwd: Path, session_id: str) -> Path:
    result = subprocess.run(
        [qwen_bin, 'sessions', 'list', '--json', '--limit', '100'],
        cwd=cwd, capture_output=True, text=True, timeout=30, check=True,
    )
    for line in result.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('sessionId') == session_id:
            path = Path(row.get('filePath', ''))
            if path.is_file():
                return path
    raise ValueError(f'Qwen Code recording for session {session_id} is missing')


def _valid_tool_phases(order: list[tuple[str, str]], calls: list[dict]) -> bool:
    """Allow parallel reads, but require each phase to finish before the next."""
    by_id = {call['id']: call['name'] for call in calls}
    outstanding: set[str] = set()
    phase = 0
    phases = ('read_file', 'edit', 'run_shell_command')
    for event, call_id in order:
        if event == 'call':
            name = by_id.get(call_id)
            if name not in phases:
                return False
            next_phase = phases.index(name)
            if next_phase < phase or next_phase > phase + 1:
                return False
            if next_phase > phase:
                if outstanding:
                    return False
                phase = next_phase
            elif phase != 0 and outstanding:
                return False
            if call_id in outstanding:
                return False
            outstanding.add(call_id)
        elif event == 'result':
            if call_id not in outstanding:
                return False
            outstanding.remove(call_id)
        else:
            return False
    return phase == 2 and not outstanding


def _tool_arguments(value: object) -> dict | None:
    """Return the Qwen tool argument object, regardless of JSON key order."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def evaluate(events: list[dict], recording: list[dict], *, session_id: str,
             html: str, exit_code: int, expected_model: str,
             fixture_dir: Path | None = None) -> dict:
    """Compare Qwen's machine stream and retained conversation, fail closed."""
    errors: list[str] = []
    init = next((row for row in events if row.get('type') == 'system'
                 and row.get('subtype') == 'init'), None)
    if not init or init.get('session_id') != session_id:
        errors.append('stream_session_identity_missing_or_mismatched')
    if not init or init.get('model') != expected_model:
        errors.append('stream_model_identity_missing_or_mismatched')
    if any(row.get('session_id') not in (None, session_id) for row in events):
        errors.append('mixed_stream_sessions')
    if any(row.get('sessionId') not in (None, session_id) for row in recording):
        errors.append('mixed_recorded_sessions')

    calls: list[dict] = []
    results: dict[str, dict] = {}
    stream_order: list[tuple[str, str]] = []
    last_tool_result_index = -1
    tool_activity_after_terminal = False
    terminal_indices = [i for i, row in enumerate(events) if row.get('type') == 'result']
    first_terminal_index = terminal_indices[0] if terminal_indices else len(events)
    for event_index, row in enumerate(events):
        message = row.get('message') or {}
        if row.get('type') == 'assistant':
            for part in message.get('content', []):
                if part.get('type') == 'tool_use':
                    arguments = _tool_arguments(part.get('input'))
                    if arguments is None:
                        errors.append('invalid_stream_tool_arguments')
                    calls.append({'id': str(part.get('id', '')),
                                  'name': str(part.get('name', '')),
                                  'input': arguments or {}})
                    stream_order.append(('call', calls[-1]['id']))
                    tool_activity_after_terminal |= event_index > first_terminal_index
        elif row.get('type') == 'user':
            for part in message.get('content', []):
                if part.get('type') == 'tool_result':
                    call_id = str(part.get('tool_use_id', ''))
                    if call_id in results:
                        errors.append('duplicate_tool_result')
                    results[call_id] = part
                    stream_order.append(('result', call_id))
                    last_tool_result_index = event_index
                    tool_activity_after_terminal |= event_index > first_terminal_index
    names = tuple(call['name'] for call in calls)
    valid_order = (len(names) in (3, 4) and names[-2:] == EXPECTED_TOOLS[-2:]
                   and all(name == 'read_file' for name in names[:-2]))
    if not valid_order:
        errors.append('wrong_tool_sequence')
    if len({call['id'] for call in calls}) != len(calls) or any(not call['id'] for call in calls):
        errors.append('invalid_call_ids')
    if set(results) != {call['id'] for call in calls}:
        errors.append('missing_or_unmatched_tool_result')
    if not _valid_tool_phases(stream_order, calls):
        errors.append('tool_result_order_mismatch')
    if any(result.get('is_error') for result in results.values()):
        errors.append('tool_error')
    recorded_calls: list[tuple[str, str, dict | None]] = []
    recorded_results: dict[str, str] = {}
    recorded_order: list[tuple[str, str]] = []
    for row in recording:
        if row.get('type') == 'assistant':
            for part in (row.get('message') or {}).get('parts') or []:
                function = part.get('functionCall') if isinstance(part, dict) else None
                if isinstance(function, dict):
                    arguments = _tool_arguments(function.get('args'))
                    if arguments is None:
                        errors.append('invalid_recorded_tool_arguments')
                    recorded_calls.append((str(function.get('id', '')),
                                           str(function.get('name', '')), arguments))
                    recorded_order.append(('call', recorded_calls[-1][0]))
        elif row.get('type') == 'tool_result':
            result = row.get('toolCallResult') or {}
            if result.get('callId'):
                recorded_results[str(result['callId'])] = str(result.get('status', ''))
                recorded_order.append(('result', str(result['callId'])))
    if [(call_id, name) for call_id, name, _ in recorded_calls] != [
            (call['id'], call['name']) for call in calls]:
        errors.append('stream_recording_tool_calls_mismatch')
    if [args for _, _, args in recorded_calls] != [call['input'] for call in calls]:
        errors.append('stream_recording_tool_arguments_mismatch')
    if set(recorded_results) != set(results):
        errors.append('stream_recording_tool_results_mismatch')
    if not _valid_tool_phases(recorded_order, calls):
        errors.append('recorded_tool_result_order_mismatch')
    if any(status != 'success' for status in recorded_results.values()):
        errors.append('recorded_tool_error')
    if valid_order and 'PASS: expected heading present' not in str(
            results.get(calls[-1]['id'], {}).get('content', '')):
        errors.append('validator_result_missing')
    if fixture_dir is not None:
        fixture_dir = fixture_dir.resolve()
        read_index = False
        for call in calls:
            supplied = call['input']
            if call['name'] in ('read_file', 'edit'):
                path = Path(str(supplied.get('file_path', '')))
                if not path.is_absolute():
                    path = fixture_dir / path
                allowed = ({fixture_dir / 'index.html', fixture_dir / 'test_html.py'}
                           if call['name'] == 'read_file' else {fixture_dir / 'index.html'})
                if path.resolve() not in allowed:
                    errors.append('unexpected_tool_path')
                if call['name'] == 'read_file' and path.resolve() == fixture_dir / 'index.html':
                    read_index = True
            elif call['name'] == 'run_shell_command':
                if supplied.get('command') != 'python3 test_html.py':
                    errors.append('unexpected_validation_command')
                directory = supplied.get('directory')
                if directory and Path(str(directory)).resolve() != fixture_dir:
                    errors.append('unexpected_validation_directory')
        if not read_index:
            errors.append('index_read_missing')
    if html != EXPECTED_HTML:
        errors.append('artifact_mismatch')
    if exit_code != 0:
        errors.append('qwen_nonzero_exit')

    terminal = [row for row in events if row.get('type') == 'result']
    stream_final = terminal[-1] if terminal else None
    if not stream_final or stream_final.get('subtype') != 'success' or stream_final.get('is_error'):
        errors.append('stream_final_missing_or_failed')
    if len(terminal_indices) != 1 or (terminal_indices and terminal_indices[0] <= last_tool_result_index):
        errors.append('stream_terminal_order_mismatch')
    if tool_activity_after_terminal:
        errors.append('tool_activity_after_terminal')
    stream_text = str((stream_final or {}).get('result') or '').strip()
    if not stream_text:
        errors.append('stream_summary_missing')
    elif 'test_html.py' not in stream_text or 'PASS' not in stream_text:
        errors.append('summary_missing_test_outcome')

    last_tool_index = max((i for i, row in enumerate(recording)
                           if row.get('type') == 'tool_result'), default=-1)
    recorded_texts = []
    for row in recording[last_tool_index + 1:]:
        if row.get('type') != 'assistant' or row.get('provenance') != 'assistant_output':
            continue
        parts = (row.get('message') or {}).get('parts') or []
        recorded_texts.extend(str(part['text']).strip() for part in parts
                              if isinstance(part, dict) and part.get('text')
                              and not part.get('thought'))
    recorded_final = '\n'.join(recorded_texts).strip()
    if not recorded_final:
        errors.append('recorded_summary_missing')
    if stream_text and recorded_final and stream_text != recorded_final:
        errors.append('stream_recording_summary_mismatch')

    return {
        'status': 'pass' if not errors else 'fail',
        'errors': errors,
        'tool_sequence': [call['name'] for call in calls],
        'tool_call_ids': [call['id'] for call in calls],
        'stream_summary_present': bool(stream_text),
        'recorded_summary_present': bool(recorded_final),
        'turn_usage': [row['message'].get('usage') for row in events
                       if row.get('type') == 'assistant'
                       and isinstance(row.get('message'), dict)
                       and isinstance(row['message'].get('usage'), dict)],
        'client_usage': (stream_final or {}).get('usage'),
        'client_turns': (stream_final or {}).get('num_turns'),
        'artifact_sha256': hashlib.sha256(html.encode('utf-8')).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict:
    if args.timeout <= 0 or args.max_output_tokens <= 0:
        raise ValueError('timeout and max-output-tokens must be positive')
    version = subprocess.run([args.qwen_bin, '--version'], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
    if version != args.expected_qwen_version:
        raise ValueError(f'Qwen Code version {version!r} differs from pinned '
                         f'{args.expected_qwen_version!r}')
    profile = getattr(args, 'qwen_profile', 'bare')
    if profile not in ('bare', 'normal'):
        raise ValueError('qwen-profile must be bare or normal')
    settings_path = Path(args.settings).resolve() if getattr(args, 'settings', None) else None
    settings_sha256 = _sha256(settings_path) if settings_path else None
    declared_client_wall_time = None
    if settings_path:
        settings_document = json.loads(settings_path.read_text(encoding='utf-8'))
        if not isinstance(settings_document, dict):
            raise ValueError('Qwen Code settings must be a JSON object')
        model_settings = settings_document.get('model')
        if isinstance(model_settings, dict):
            declared_client_wall_time = model_settings.get('maxWallTimeSeconds')
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise ValueError('run directory must be empty; existing artifacts are immutable')
    (run_dir / 'index.html').write_text(SOURCE_HTML, encoding='utf-8')
    (run_dir / 'test_html.py').write_text(TEST_SCRIPT, encoding='utf-8')
    session_id = str(uuid.uuid4())
    started = time.time()
    endpoint = args.endpoint.rstrip('/')
    with urlopen(endpoint + '/models', timeout=10) as response:
        model_list = json.load(response)
    advertised = next((row for row in model_list.get('data', [])
                       if row.get('id') == args.model), None)
    if advertised is None:
        raise ValueError(f'model {args.model!r} is not advertised by endpoint')
    expected_identity = args.expected_serving_identity
    expected_session = args.expected_session_library_sha256
    if expected_identity is not None and not _valid_sha256(expected_identity):
        raise ValueError('expected-serving-identity must be a lowercase SHA-256 hex digest')
    if expected_session is not None and not _valid_sha256(expected_session):
        raise ValueError('expected-session-library-sha256 must be a lowercase SHA-256 hex digest')
    try:
        identity_before = _read_loaded_identity(endpoint)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        identity_before = None
        identity_probe_error = str(exc)
    else:
        identity_probe_error = None
    mismatch = _identity_expectation_failure(
        identity_before, expected=expected_identity,
        expected_session=expected_session, model=args.model)
    if identity_probe_error is not None or mismatch is not None:
        reason = 'loaded_identity_preflight_probe_failed' if identity_probe_error is not None else mismatch
        report = {
            'schema': 'cke.serving_harness_acceptance.v1',
            'run_id': run_dir.name, 'session_id': session_id,
            'model': args.model, 'endpoint': endpoint,
            'expected_serving_identity': expected_identity,
            'expected_session_library_sha256': expected_session,
            'started_at_unix': started,
            'qwen_code_version': version,
            'qwen_profile': profile,
            'qwen_settings_sha256': settings_sha256,
            'declared_client_wall_time_seconds': declared_client_wall_time,
            'max_output_tokens': args.max_output_tokens,
            'time_budget_seconds': args.timeout,
            'result': {'status': 'not_run', 'errors': ['loaded_identity_preflight_failed']},
            'identity': {'level': 'preflight_rejected', 'before': identity_before, 'after': None},
            'certification': {'status': 'fail', 'reason': reason,
                              'detail': identity_probe_error},
        }
        (run_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return report
    cmd = [args.qwen_bin]
    if profile == 'bare':
        cmd.append('--bare')
    cmd += ['--auth-type', 'openai-responses',
           '--model', args.model, '--session-id', session_id,
           '--chat-recording', '--allowed-tools', 'read_file', 'edit',
           'run_shell_command', '--max-tool-calls', '6',
           '--output-format', 'stream-json', '--prompt', TASK_PROMPT]
    env = os.environ.copy()
    env.update(OPENAI_BASE_URL=endpoint, OPENAI_MODEL=args.model,
               OPENAI_API_KEY=args.api_key,
               QWEN_CODE_MAX_OUTPUT_TOKENS=str(args.max_output_tokens))
    if settings_path:
        env['QWEN_CODE_SYSTEM_SETTINGS_PATH'] = str(settings_path)
    timed_out = False
    def mark_timeout():
        nonlocal timed_out
        timed_out = True
    with (run_dir / 'events.jsonl').open('wb') as output, (run_dir / 'stderr.log').open('wb') as errors:
        process = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=output,
                                   stderr=errors, start_new_session=True)
        exit_code = _wait_for_client(process, args.timeout, on_timeout=mark_timeout)
    client_budget_exceeded = _client_time_budget_exceeded(exit_code, run_dir / 'stderr.log')
    try:
        session_path = _session_file(args.qwen_bin, run_dir, session_id)
        shutil.copyfile(session_path, run_dir / 'session.jsonl')
        recording = _read_jsonl(run_dir / 'session.jsonl')
    except (ValueError, subprocess.SubprocessError, OSError) as exc:
        recording = []
        recording_error = str(exc)
    else:
        recording_error = None
    try:
        events = _read_jsonl(run_dir / 'events.jsonl')
        event_error = None
    except (ValueError, OSError) as exc:
        events, event_error = [], str(exc)
    verdict = evaluate(events, recording, session_id=session_id,
                       html=(run_dir / 'index.html').read_text(encoding='utf-8'),
                       exit_code=exit_code, expected_model=args.model,
                       fixture_dir=run_dir)
    if recording_error:
        verdict['errors'].append('recording_unavailable: ' + recording_error)
    if event_error:
        verdict['errors'].append('event_stream_unavailable: ' + event_error)
    if verdict['errors']:
        verdict['status'] = 'fail'
    if timed_out or client_budget_exceeded:
        verdict['status'] = 'incomplete'
        verdict['incomplete_reason'] = 'time_budget_exceeded'
    try:
        identity_after = _read_loaded_identity(endpoint)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        identity_after = None
        identity_probe_error = str(exc)
    else:
        identity_probe_error = None
    identity, certification = _identity_verdict(
        identity_before, identity_after, expected=expected_identity,
        expected_session=expected_session,
        model=args.model, task_passed=verdict['status'] == 'pass')
    if identity_probe_error:
        certification = {'status': 'fail', 'reason': 'loaded_identity_followup_probe_failed',
                         'detail': identity_probe_error}
    report = {
        'schema': 'cke.serving_harness_acceptance.v1',
        'run_id': run_dir.name,
        'session_id': session_id,
        'model': args.model,
        'endpoint': endpoint,
        'expected_serving_identity': expected_identity,
        'expected_session_library_sha256': expected_session,
        'started_at_unix': started,
        'elapsed_seconds': time.time() - started,
        'qwen_code_version': version,
        'qwen_profile': profile,
        'qwen_settings_sha256': settings_sha256,
        'declared_client_wall_time_seconds': declared_client_wall_time,
        'max_output_tokens': args.max_output_tokens,
        'time_budget_seconds': args.timeout,
        'time_budget_exceeded': timed_out,
        'client_time_budget_exceeded': client_budget_exceeded,
        'advertised_model': advertised,
        'exit_code': exit_code,
        'runner_sha256': _sha256(Path(__file__)),
        'fixture_sha256': _sha256(run_dir / 'test_html.py'),
        'events_sha256': _sha256(run_dir / 'events.jsonl'),
        'session_sha256': _sha256(run_dir / 'session.jsonl')
            if (run_dir / 'session.jsonl').is_file() else None,
        'result': verdict,
        'identity': identity,
        'certification': certification,
    }
    (run_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True, help='Base URL ending in /v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--qwen-bin', default='qwen')
    parser.add_argument('--expected-qwen-version', default='0.24.6')
    parser.add_argument('--qwen-profile', choices=('bare', 'normal'), default='bare')
    parser.add_argument('--settings')
    parser.add_argument('--api-key', default='cke-local-only')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--max-output-tokens', type=int, default=768)
    parser.add_argument('--expected-serving-identity',
                        help='SHA-256 identity from the intended resolved serving.json bundle')
    parser.add_argument('--expected-session-library-sha256',
                        help='SHA-256 of the intended libck_session_v8.so runtime ABI')
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(f'acceptance setup failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'report': str(Path(args.run_dir).resolve() / 'report.json'),
                      'status': report['result']['status'],
                      'certification_status': report['certification']['status'],
                      'certification_reason': report['certification']['reason'],
                      'errors': report['result']['errors']}))
    return 0 if (report['result']['status'] == 'pass'
                 and report['certification']['status'] == 'pass') else 1


if __name__ == '__main__':
    raise SystemExit(main())
