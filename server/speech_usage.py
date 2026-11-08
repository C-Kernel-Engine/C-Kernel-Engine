"""Bounded, process-local speech usage counters. Never retain request content."""
from datetime import datetime, timezone
import math
import threading
import time
import uuid


class SpeechUsage:
    BOUNDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0)

    def __init__(self):
        self.lock = threading.Lock()
        self.boot_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = time.monotonic()
        self.counters = dict.fromkeys((
            'http_received', 'http_finished', 'http_inflight', 'http_disconnects',
            'http_2xx', 'http_4xx', 'http_5xx', 'http_other', 'http_no_response',
            'uploaded_bytes', 'admitted', 'admitted_with_hint', 'jobs_succeeded',
            'jobs_failed', 'jobs_cancelled', 'invalid_timing_observations'), 0)
        self.audio_seconds = 0.0
        self.success_audio_seconds = 0.0
        self.inference_seconds = 0.0
        self.queue_seconds = 0.0
        self.http_seconds = 0.0
        self.histograms = {k: [0] * (len(self.BOUNDS) + 1)
                           for k in ('http_wall_seconds', 'inference_success_seconds')}

    def begin(self):
        with self.lock:
            self.counters['http_received'] += 1
            self.counters['http_inflight'] += 1

    def _histogram(self, name, seconds):
        for index, bound in enumerate((*self.BOUNDS, math.inf)):
            if seconds <= bound:
                self.histograms[name][index] += 1
                return

    def finish(self, status, disconnected, uploaded, elapsed):
        with self.lock:
            self.counters['http_finished'] += 1
            self.counters['http_inflight'] -= 1
            self.counters['http_disconnects'] += int(disconnected)
            self.counters['uploaded_bytes'] += uploaded
            category = ('http_no_response' if status is None else
                        'http_2xx' if 200 <= status < 300 else
                        'http_4xx' if 400 <= status < 500 else
                        'http_5xx' if 500 <= status < 600 else 'http_other')
            self.counters[category] += 1
            self.http_seconds += elapsed
            self._histogram('http_wall_seconds', elapsed)

    def admit(self, audio_frames, has_hint):
        with self.lock:
            self.counters['admitted'] += 1
            self.counters['admitted_with_hint'] += int(has_hint)
            self.audio_seconds += audio_frames / 16000.0

    def settled(self, result, cancelled_type, audio_frames):
        with self.lock:
            try:
                response = result.result()
            except cancelled_type:
                self.counters['jobs_cancelled'] += 1
                return
            except BaseException:
                self.counters['jobs_failed'] += 1
                return
            self.counters['jobs_succeeded'] += 1
            self.success_audio_seconds += audio_frames / 16000.0
            for field, target in (('inference_seconds', 'inference_seconds'),
                                  ('queue_seconds', 'queue_seconds')):
                value = response.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    self.counters['invalid_timing_observations'] += 1
                    continue
                setattr(self, target, getattr(self, target) + value)
                if field == 'inference_seconds':
                    self._histogram('inference_success_seconds', value)

    def snapshot(self):
        with self.lock:
            return {'schema': 'cke.whisper.usage.v1', 'boot_id': self.boot_id,
                    'started_at': self.started_at, 'uptime_seconds': time.monotonic() - self.started,
                    'retention': 'process-local; resets on restart',
                    'counters': dict(self.counters),
                    'admitted_audio_seconds_total': self.audio_seconds,
                    'successful_audio_seconds_total': self.success_audio_seconds,
                    'successful_inference_seconds_total': self.inference_seconds,
                    'successful_queue_seconds_total': self.queue_seconds,
                    'http_wall_seconds_total': self.http_seconds,
                    'histograms': {key: {'upper_bounds_seconds': [*self.BOUNDS, None],
                                        'noncumulative_counts': list(values)}
                                   for key, values in self.histograms.items()}}


class SpeechUsageMiddleware:
    """Pure ASGI instrumentation: preserve disconnect and streaming behavior."""
    def __init__(self, app, usage):
        self.app, self.usage = app, usage

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope.get('path') != '/audio/transcriptions':
            return await self.app(scope, receive, send)
        started = time.monotonic()
        status, disconnected, uploaded = None, False, 0
        self.usage.begin()
        async def observed_receive():
            nonlocal disconnected, uploaded
            message = await receive()
            if message['type'] == 'http.disconnect':
                disconnected = True
            elif message['type'] == 'http.request':
                uploaded += len(message.get('body', b''))
            return message
        async def observed_send(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
            await send(message)
        try:
            await self.app(scope, observed_receive, observed_send)
        finally:
            self.usage.finish(status, disconnected, uploaded, time.monotonic() - started)
