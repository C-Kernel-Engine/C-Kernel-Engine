"""Aggregate telemetry must not change lifecycle behavior or retain speech content."""
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import json
import threading
import time

from fastapi.testclient import TestClient
import pytest

from server.speech_usage import SpeechUsage, SpeechUsageMiddleware
from server.whisper_http import create_app, TranscriptionCancelled
from server.tests.test_whisper_http import _FakeBackend, _wav


def test_http_success_validation_failure_and_hint_are_counted_without_content():
    backend = _FakeBackend(1)
    with TestClient(create_app(lambda: backend)) as client:
        assert client.post('/audio/transcriptions', content=_wav(1),
                           headers={'content-type':'audio/wav','x-cke-request-id':'private-id'}).status_code==200
        assert client.post('/audio/transcriptions',content=b'private audio').status_code==415
        assert client.post('/audio/transcriptions',content=_wav(9),
                           headers={'content-type':'audio/wav'}).status_code==503
        data=client.get('/telemetry').json()
        c=data['counters']
        assert c['http_received']==c['http_finished']==3 and c['http_inflight']==0
        assert (c['http_2xx'],c['http_4xx'],c['http_5xx'])==(1,1,1)
        assert c['admitted']==2 and c['jobs_succeeded']==c['jobs_failed']==1
        assert data['admitted_audio_seconds_total']==pytest.approx(.02)
        assert data['successful_audio_seconds_total']==pytest.approx(.01)
        assert data['successful_inference_seconds_total']==.01
        assert sum(data['histograms']['http_wall_seconds']['noncumulative_counts'])==3
        assert sum(data['histograms']['inference_success_seconds']['noncumulative_counts'])==1
        text=json.dumps(data)
        assert all(x not in text for x in ('private-id','private audio','heard-1','loaded_artifact'))
        assert client.get('/telemetry').json()['counters']==c  # polling is not usage


def test_queued_cancellation_and_overload_do_not_double_count():
    backend=_FakeBackend(1,block=True)
    with TestClient(create_app(lambda:backend,max_waiters=1)) as client:
        def post(ident):
            return client.post('/audio/transcriptions',content=_wav(1),
                               headers={'content-type':'audio/wav','x-cke-request-id':ident})
        with ThreadPoolExecutor(max_workers=2) as pool:
            active=pool.submit(post,'active')
            assert backend.started.wait(2)
            queued=pool.submit(post,'waiting')
            for _ in range(200):
                if client.get('/health').json()['queued']==1:break
                time.sleep(.01)
            assert client.get('/health').json()['queued']==1
            assert post('overflow').status_code==429
            assert client.post('/audio/transcriptions/waiting/cancel').status_code==200
            assert queued.result(timeout=3).status_code==409
            assert client.post('/audio/transcriptions/waiting/cancel').status_code==404
            backend.released.set()
            assert active.result(timeout=3).status_code==200
        c=client.get('/telemetry').json()['counters']
        assert c['admitted']==2 and c['jobs_cancelled']==1 and c['jobs_succeeded']==1
        assert c['http_received']==3 and c['http_4xx']==2 and c['http_inflight']==0


def test_hint_usage_boot_identity_and_invalid_timings():
    class Hinted(_FakeBackend):
        def run(self,data,**kwargs):return super().run(data)
    with TestClient(create_app(lambda:Hinted(1),vocabulary_validator=lambda _:None)) as client:
        response=client.post('/audio/transcriptions?vocabulary_hint=Antsand',content=_wav(1),
                             headers={'content-type':'audio/wav'})
        assert response.status_code==200
        usage=client.get('/telemetry').json()
        assert usage['counters']['admitted_with_hint']==1
        assert 'Antsand' not in json.dumps(usage)
    with TestClient(create_app(lambda:Hinted(1))) as client:
        fresh=client.get('/telemetry').json()
        assert usage['boot_id']!=fresh['boot_id']
        assert fresh['counters']['http_received']==0
    usage=SpeechUsage()
    result=Future();result.set_result({'queue_seconds':float('nan'),'inference_seconds':-1})
    usage.settled(result,TranscriptionCancelled,160)
    snap=usage.snapshot()
    assert snap['counters']['invalid_timing_observations']==2
    assert snap['successful_inference_seconds_total']==0
    json.dumps(snap,allow_nan=False)


def test_asgi_disconnect_and_exception_are_observed_without_payload_capture():
    usage=SpeechUsage()
    async def app(scope,receive,send):
        assert (await receive())['type']=='http.request'
        assert (await receive())['type']=='http.disconnect'
        raise RuntimeError('private detail')
    async def exercise():
        events=iter([{'type':'http.request','body':b'private recording'},
                     {'type':'http.disconnect'}])
        async def receive():return next(events)
        async def send(_):pass
        with pytest.raises(RuntimeError):
            await SpeechUsageMiddleware(app,usage)(
                {'type':'http','method':'POST','path':'/audio/transcriptions'},receive,send)
    asyncio.run(exercise())
    data=usage.snapshot();c=data['counters']
    assert c['http_disconnects']==c['http_no_response']==c['http_finished']==1
    assert c['http_inflight']==0 and c['uploaded_bytes']==len(b'private recording')
    assert 'private' not in json.dumps(data)


def test_concurrent_counter_updates_are_bounded_and_exact():
    usage=SpeechUsage()
    def update(_):
        usage.begin();usage.admit(160,False)
        future=Future();future.set_result({'inference_seconds':.25,'queue_seconds':.125})
        usage.settled(future,TranscriptionCancelled,160)
        usage.finish(200,False,100,.5)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(update,range(500)))
    snap=usage.snapshot()
    assert snap['counters']['http_received']==snap['counters']['jobs_succeeded']==500
    assert snap['counters']['http_inflight']==0
    assert snap['successful_inference_seconds_total']==125
    assert sum(snap['histograms']['http_wall_seconds']['noncumulative_counts'])==500
    assert len(json.dumps(snap))<2500
