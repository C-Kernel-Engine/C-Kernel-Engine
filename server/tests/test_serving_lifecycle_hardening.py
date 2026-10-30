"""Execution ownership and unsupported-input regressions for every text model."""
import asyncio
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from server import live
from server.schemas.response import CreateResponseRequest
from version.v8.scripts.ck_serve_v8 import create_app

TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}"


class Session:
    def __init__(self):
        self.calls = 0
        self.cancel_calls = 0
        self.release = threading.Event()
        self.finished = threading.Event()

    def generate(self, *args, on_token, **kwargs):
        self.calls += 1
        on_token(0, 'answer')
        if self.calls == 1:
            assert self.release.wait(5)
        self.finished.set()
        return {'prompt_tokens': 1, 'generated_tokens': 1, 'stop_reason': 1}

    def cancel(self):
        self.cancel_calls += 1


@pytest.mark.parametrize('stream', [False, True])
def test_native_failure_releases_slot_and_followup_succeeds(stream):
    class FailOnceSession:
        def __init__(self):
            self.calls = 0

        def generate(self, *args, on_token, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('injected native failure')
            on_token(0, 'recovered')
            return {'prompt_tokens': 1, 'generated_tokens': 1, 'stop_reason': 1}

        def cancel(self):
            pass

    session = FailOnceSession()
    app = create_app(session, model='test', chat_template=TEMPLATE)
    client = TestClient(app)
    first = client.post('/v1/responses', json={'model': 'test', 'input': 'first', 'stream': stream})
    assert first.status_code == 200
    if stream:
        assert 'response.failed' in first.text
    else:
        assert first.json()['status'] == 'failed'
    assert not app.state.flight_lock.locked()
    second = client.post('/v1/responses', json={'model': 'test', 'input': 'second'})
    assert second.status_code == 200
    assert second.json()['output_text'] == 'recovered'


def test_cancel_timeout_retains_native_owner_until_worker_finishes(monkeypatch):
    class SlowCancelSession:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.cancel_calls = 0
            self.calls = 0

        def generate(self, *args, on_token, **kwargs):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                assert self.release.wait(5)
            on_token(0, 'finished')
            return {'prompt_tokens': 1, 'generated_tokens': 1, 'stop_reason': 1}

        def cancel(self):
            self.cancel_calls += 1

    monkeypatch.setattr(live, '_FLIGHT_WAIT_SECONDS', .01)
    session = SlowCancelSession()
    app = create_app(session, model='test', chat_template=TEMPLATE, cancel_wait_seconds=.01)
    client = TestClient(app)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, '/v1/responses', json={
            'model': 'test', 'input': 'first', 'stream': True,
        })
        assert session.started.wait(5)
        with app.state.active_streams_lock:
            response_id = next(iter(app.state.active_streams))
        try:
            cancelled = client.post(f'/v1/responses/{response_id}/cancel')
            assert cancelled.status_code == 504
            assert session.cancel_calls == 1
            assert app.state.flight_lock.locked()
            busy = client.post('/v1/responses', json={'model': 'test', 'input': 'second'})
            assert busy.status_code == 429
            assert busy.headers['retry-after'] == '1'
            assert session.calls == 1
        finally:
            session.release.set()
        assert first.result(timeout=5).status_code == 200
    assert not app.state.flight_lock.locked()
    followup = client.post('/v1/responses', json={'model': 'test', 'input': 'third'})
    assert followup.status_code == 200
    assert followup.json()['output_text'] == 'finished'


def test_native_completion_releases_slot_without_consuming_terminal_event():
    """A paused/disconnected SSE consumer cannot retain the native slot."""
    session = Session()
    app = create_app(session, model='test', chat_template=TEMPLATE)
    def routes(items):
        for route in items:
            if hasattr(route, 'original_router'):
                yield from routes(route.original_router.routes)
            else:
                yield route
    endpoint = next(r.endpoint for r in routes(app.routes) if getattr(r, 'path', '').endswith('/responses') and 'POST' in r.methods)
    request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/responses', 'headers': []})

    async def run():
        response = endpoint(CreateResponseRequest(model='test', input='first', stream=True), request)
        iterator = response.body_iterator
        while 'response.output_text.delta' not in await anext(iterator):
            pass
        assert app.state.flight_lock.locked()
        session.release.set()
        for _ in range(100):
            if not app.state.flight_lock.locked():
                break
            await asyncio.sleep(.01)
        assert not app.state.flight_lock.locked()
        assert not app.state.active_streams
        # The first HTTP iterator is still paused before its terminal event.
        second = endpoint(CreateResponseRequest(model='test', input='second'), request)
        assert second['status'] == 'completed'
        response.lease.disconnect()
        assert session.cancel_calls == 0
        await iterator.aclose()

    asyncio.run(run())


def test_stale_owner_never_cancels_new_request_and_active_owner_retains_slot():
    lock = threading.Lock()
    session = Session()
    lock.acquire()
    old = live._FlightLease(lock, session)
    old.start_worker(threading.Event())
    old.release()
    lock.acquire()
    current = live._FlightLease(lock, session)
    cancelled = threading.Event()
    current.start_worker(cancelled)
    old.disconnect()
    old.cancel()
    assert session.cancel_calls == 0
    current.disconnect()
    assert session.cancel_calls == 1 and cancelled.is_set()
    assert lock.locked()  # Cancellation request is not native completion.
    current.release()
    current.release()
    assert not lock.locked()


def test_bounded_admission_is_fifo_and_rejects_overflow():
    lock = threading.Lock()
    admission = live._FlightAdmission(lock, max_waiters=2)
    assert lock.acquire(blocking=False)
    release_first = threading.Event()
    order = []

    def wait_for_queue_size(size):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with admission.guard:
                if len(admission.waiters) == size:
                    return
            threading.Event().wait(.001)
        pytest.fail(f"expected {size} queued requests")

    def contender(name):
        result = admission.acquire(5)
        order.append((name, result))
        if result == "acquired":
            if name == "first":
                assert release_first.wait(5)
            lock.release()
            admission.notify_release()

    first = threading.Thread(target=contender, args=("first",))
    second = threading.Thread(target=contender, args=("second",))
    first.start()
    wait_for_queue_size(1)
    second.start()
    wait_for_queue_size(2)
    assert admission.acquire(0) == "full"
    lock.release()
    admission.notify_release()
    try:
        deadline = time.monotonic() + 5
        while not order and time.monotonic() < deadline:
            threading.Event().wait(.001)
        assert order == [("first", "acquired")]
    finally:
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert order == [("first", "acquired"), ("second", "acquired")]
    assert not lock.locked()


def test_timed_out_admission_removes_its_ticket():
    lock = threading.Lock()
    admission = live._FlightAdmission(lock, max_waiters=1)
    lock.acquire()
    assert admission.acquire(.01) == "timeout"
    assert not admission.waiters
    lock.release()
    assert admission.acquire(0) == "acquired"
    lock.release()


def test_disconnected_admission_removes_ticket_before_native_ownership():
    lock = threading.Lock()
    admission = live._FlightAdmission(lock, max_waiters=1)
    disconnected = threading.Event()
    lock.acquire()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        queued = pool.submit(admission.acquire, 5, cancelled=disconnected.is_set)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with admission.guard:
                if admission.waiters:
                    break
            threading.Event().wait(.001)
        else:
            pytest.fail('request did not join admission queue')
        disconnected.set()
        assert queued.result(timeout=1) == 'cancelled'
    assert not admission.waiters
    assert lock.locked()
    lock.release()
    assert admission.acquire(0) == 'acquired'
    lock.release()


def test_admission_disconnect_probe_does_not_hold_cleanup_lock():
    lock = threading.Lock()
    admission = live._FlightAdmission(lock, max_waiters=1)
    probe_entered = threading.Event()
    cleanup_done = threading.Event()
    release_probe = threading.Event()
    probes = 0

    def disconnect_probe():
        nonlocal probes
        probes += 1
        if probes == 1:
            return False
        probe_entered.set()
        assert release_probe.wait(2)
        return True

    def cleanup():
        admission.notify_release()
        cleanup_done.set()
        release_probe.set()

    lock.acquire()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        queued = pool.submit(admission.acquire, 5, cancelled=disconnect_probe)
        try:
            assert probe_entered.wait(1)
            notifier = pool.submit(cleanup)
            # Before the fix, cleanup needed guard while the callback held it.
            assert cleanup_done.wait(.3)
            notifier.result(timeout=1)
            assert queued.result(timeout=1) == 'cancelled'
        finally:
            release_probe.set()
            lock.release()
    assert not admission.waiters


def test_disconnect_during_admission_handoff_releases_native_owner():
    lock = threading.Lock()
    admission = live._FlightAdmission(lock, max_waiters=1)
    observations = iter((False, True))
    assert admission.acquire(0, cancelled=lambda: next(observations)) == 'cancelled'
    assert not lock.locked()
    assert admission.acquire(0) == 'acquired'
    lock.release()


def test_http_disconnected_waiter_never_starts_native_work(monkeypatch):
    disconnected = threading.Event()
    original = Request.is_disconnected

    async def simulated_disconnect(request):
        if request.headers.get('x-test-disconnect') == 'yes':
            return disconnected.is_set()
        return await original(request)

    monkeypatch.setattr(Request, 'is_disconnected', simulated_disconnect)
    monkeypatch.setattr(live, '_FLIGHT_WAIT_SECONDS', 5.0)
    session = Session()
    session.release.set()
    app = create_app(session, model='test', chat_template=TEMPLATE)
    client = TestClient(app)
    assert app.state.flight_lock.acquire(blocking=False)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            queued = pool.submit(
                client.post, '/v1/responses',
                json={'model': 'test', 'input': 'queued'},
                headers={'x-test-disconnect': 'yes'},
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with app.state.flight_admission.guard:
                    if app.state.flight_admission.waiters:
                        break
                threading.Event().wait(.001)
            else:
                pytest.fail('request did not join admission queue')
            disconnected.set()
            response = queued.result(timeout=1)
            assert response.status_code == 499
            assert response.json()['error']['code'] == 'client_disconnected'
            assert not app.state.flight_admission.waiters
            assert session.calls == 0
    finally:
        app.state.flight_lock.release()
        app.state.flight_admission.notify_release()
    followup = client.post('/v1/responses', json={'model': 'test', 'input': 'followup'})
    assert followup.status_code == 200
    assert session.calls == 1


@pytest.mark.parametrize("route", ["responses", "chat/completions"])
def test_http_admission_overflow_does_not_start_native_work(monkeypatch, route):
    monkeypatch.setattr(live, "_FLIGHT_MAX_WAITERS", 1)
    monkeypatch.setattr(live, "_FLIGHT_WAIT_SECONDS", 5.0)
    session = Session()
    session.release.set()
    app = create_app(session, model="test", chat_template=TEMPLATE)
    client = TestClient(app)
    payload = ({"model": "test", "input": "queued"} if route == "responses"
               else {"model": "test", "messages": [{"role": "user", "content": "queued"}]})
    assert app.state.flight_lock.acquire(blocking=False)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            queued = pool.submit(client.post, f"/v1/{route}", json=payload)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with app.state.flight_admission.guard:
                    if len(app.state.flight_admission.waiters) == 1:
                        break
                threading.Event().wait(.001)
            else:
                pytest.fail("request did not join admission queue")
            full = client.post(f"/v1/{route}", json=payload)
            assert full.status_code == 429
            assert "queue is full" in full.json()["detail"]
            assert session.calls == 0
            app.state.flight_lock.release()
            app.state.flight_admission.notify_release()
            assert queued.result(timeout=5).status_code == 200
    finally:
        if app.state.flight_lock.locked():
            app.state.flight_lock.release()
    assert session.calls == 1


def test_disconnect_before_stream_iteration_releases_reserved_slot():
    lock = threading.Lock()
    lock.acquire()
    lease = live._FlightLease(lock, Session())
    lease.disconnect()
    assert not lock.locked()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('route', ['responses', 'chat/completions'])
@pytest.mark.parametrize('media', ['image', 'file', 'tool_image', 'tool_file'])
def test_media_rejected_before_renderer_tokenizer_and_native_execution(monkeypatch, stream, route, media):
    def forbidden(*args, **kwargs):
        pytest.fail('unsupported media reached computation')
    session = Session()
    session.generate = session.count_tokens = forbidden
    monkeypatch.setattr(live, '_render_with_chat_templates', forbidden)
    app = create_app(session, model='test', context_length=4096, chat_template=TEMPLATE)
    client = TestClient(app)
    part = {'type': 'input_image', 'image_url': 'https://example.org/a.png'} if media.endswith('image') else {'type': 'input_file', 'file_id': 'example'}
    if route == 'responses':
        content = [{'type': 'function_call_output', 'call_id': 'call_1', 'output': [part]}] if media.startswith('tool') else [{'type': 'message', 'role': 'user', 'content': [part]}]
        payload = {'model': 'test', 'input': content, 'stream': stream}
    else:
        chat_part = {'type': 'image_url', 'image_url': {'url': 'https://example.org/a.png'}} if media.endswith('image') else {'type': 'file', 'file': {'file_id': 'example'}}
        payload = {'model': 'test', 'messages': [{'role': 'tool' if media.startswith('tool') else 'user', 'tool_call_id': 'call_1', 'content': [chat_part]}], 'stream': stream}
    result = client.post('/v1/' + route, json=payload)
    assert result.status_code in (422, 400)
    assert session.calls == 0


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('user_text', ['literal <think>', 'literal </think>', '/tmp/image.png'])
def test_literal_user_tags_cannot_open_output_reasoning(stream, user_text):
    session = Session()
    session.release.set()
    template = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}assistant\n{% endif %}"
    client = TestClient(create_app(session, model='test', chat_template=template))
    result = client.post('/v1/responses', json={'model': 'test', 'input': user_text, 'stream': stream, 'reasoning': {'effort': 'medium'}})
    assert result.status_code == 200
    if stream:
        assert 'response.reasoning_text.delta' not in result.text
        assert 'response.output_text.delta' in result.text
    else:
        assert result.json()['output_text'] == 'answer'


def test_render_only_order_and_pinned_publisher_fixture():
    root = Path(__file__).parent / 'fixtures'
    provenance = json.loads((root / 'qwen_template_provenance.json').read_text())
    for filename, item in provenance.items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == item['sha256']
    parts = [{'type': 'input_text', 'text': 'A'}, {'type': 'input_image', 'image_url': 'https://example.org/a.png'}, {'type': 'input_text', 'text': 'B'}, {'type': 'input_image', 'image_url': 'https://example.org/b.png'}]
    normalized = live._ordered_content_parts(parts)
    assert [p['type'] for p in normalized] == ['text', 'image', 'text', 'image']
    body = CreateResponseRequest(model='test', input='text')
    template = (root / 'qwen35_08b_publisher.jinja').read_text()
    rendered = live._render_with_chat_templates(template, None, [{'role': 'user', 'content': normalized}], body)
    assert rendered.index('A') < rendered.index('<|image_pad|>') < rendered.index('B') < rendered.rindex('<|image_pad|>')
