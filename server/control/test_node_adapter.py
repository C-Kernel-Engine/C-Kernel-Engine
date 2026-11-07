"""Real-socket contract tests. Set CKE_NODE_ADAPTER to a built C binary."""
import json
import os
import socket
import subprocess
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import zmq


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Health(BaseHTTPRequestHandler):
    mode = 'ok'

    def do_GET(self):
        if self.mode == 'slow':
            time.sleep(1)
        self.send_response(200)
        self.end_headers()
        value = (b'broken' if self.mode == 'malformed' else json.dumps({
            'ready': True, 'active': 'private-request-id', 'queued': 2,
            'worker_generation': 3, 'startup_error': '/private/secret',
            'token': 'secret', 'model_path': '/private/model',
        }).encode())
        try:
            self.wfile.write(value)
        except BrokenPipeError:
            pass

    def log_message(self, *args):
        pass


class Adapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = os.environ.get('CKE_NODE_ADAPTER')
        if not cls.binary:
            raise unittest.SkipTest('build server/control/node-adapter and set CKE_NODE_ADAPTER')
        cls.ctx = zmq.Context()
        cls.http = ThreadingHTTPServer(('127.0.0.1', 0), Health)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.ctx.term()

    def setUp(self):
        Health.mode = 'ok'
        self.port = port()
        self.start()

    def start(self):
        self.proc = subprocess.Popen([self.binary, 'test-node', str(self.port),
                                     f'speech={self.http.server_port}', f'offline={port()}'])
        for _ in range(30):
            try:
                self.call('node.describe', timeout=100)
                return
            except zmq.Again:
                time.sleep(.02)
        self.fail('adapter did not start')

    def tearDown(self):
        self.proc.terminate()
        self.proc.wait(timeout=3)

    def call(self, op='workers.list', params=None, raw=None, timeout=1500):
        s = self.ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, timeout)
        s.setsockopt(zmq.SNDTIMEO, timeout)
        s.connect(f'tcp://127.0.0.1:{self.port}')
        try:
            s.send(raw if raw is not None else json.dumps({
                'schema': 'cke.control.v1', 'request_id': 'r-1',
                'node_id': 'test-node', 'operation': op, 'params': params or {},
            }).encode())
            return s.recv_json()
        finally:
            s.close()

    def test_discovery_and_status_privacy(self):
        r = self.call()
        self.assertTrue(r['ok'])
        self.assertEqual(r['request_id'], 'r-1')
        self.assertEqual(r['data']['workers'], [{'worker_id': 'speech'}, {'worker_id': 'offline'}])
        r = self.call('worker.status', {'worker_id': 'speech'})
        self.assertTrue(r['data']['ready'])
        self.assertTrue(r['data']['busy'])
        self.assertEqual(r['data']['queued'], 2)
        self.assertIsNone(r['data']['boot_id'])
        self.assertNotIn('private', json.dumps(r))
        self.assertNotIn('secret', json.dumps(r))
        r = self.call('worker.capabilities', {'worker_id': 'speech'})
        self.assertEqual(r['data']['capabilities_status'], 'unsupported')
        self.assertIsNone(r['data']['capabilities'])

    def test_invalid_and_unsupported(self):
        for raw in (b'no', b'[]', b'{}', b'{} trailing', b'{"schema":"x"}', b'null'):
            self.assertFalse(self.call(raw=raw)['ok'])
        self.assertEqual(self.call('model.load')['error']['code'], 'UNSUPPORTED_OPERATION')
        self.assertEqual(self.call('workers.list', {'x': 1})['error']['code'], 'INVALID_PARAMS')
        self.assertEqual(self.call('worker.status', {'worker_id': 'missing'})['error']['code'], 'WORKER_NOT_FOUND')
        self.assertTrue(self.call()['ok'])

    def test_unavailable_timeout_and_recovery(self):
        r = self.call('worker.status', {'worker_id': 'offline'})
        self.assertTrue(r['ok'])  # observation succeeded, worker unavailable
        self.assertEqual(r['data']['observation_status'], 'unavailable')
        self.assertIsNone(r['data']['ready'])
        for mode in ('slow', 'malformed'):
            Health.mode = mode
            start = time.monotonic()
            r = self.call('worker.status', {'worker_id': 'speech'})
            self.assertLess(time.monotonic() - start, 1.2)
            self.assertEqual(r['data']['observation_status'], 'unavailable')
        Health.mode = 'ok'
        self.assertTrue(self.call('worker.status', {'worker_id': 'speech'})['data']['ready'])

    def test_oversized_transport_rejection_and_recovery(self):
        with self.assertRaises(zmq.Again):
            self.call(raw=b'x' * 8193, timeout=300)
        self.assertTrue(self.call()['ok'])

    def test_envelope_identity_and_configuration(self):
        request = {'schema': 'cke.control.v1', 'request_id': 'test-id',
                   'node_id': 'test-node', 'operation': 'workers.list', 'params': {}}
        for key, value in (('schema', 'cke.control.v2'), ('node_id', 'other'),
                           ('request_id', 'a' * 65), ('params', [])):
            changed = dict(request, **{key: value})
            self.assertEqual(self.call(raw=json.dumps(changed).encode())['error']['code'],
                             'INVALID_REQUEST')
        for args in (['bad/id', '8120'], ['node', '0'], ['node', '8120', 'a=80'],
                     ['node', '8120', 'a=8110', 'a=8111']):
            result = subprocess.run([self.binary, *args], capture_output=True, timeout=2)
            self.assertEqual(result.returncode, 2)

    def test_restart_identity(self):
        old = self.call()['adapter_boot_id']
        self.proc.terminate()
        self.proc.wait(timeout=3)
        with self.assertRaises(zmq.Again):
            self.call(timeout=100)
        self.start()
        self.assertNotEqual(old, self.call()['adapter_boot_id'])

    def test_multipart_rejection_and_recovery(self):
        for _ in range(12):
            s = self.ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(f'tcp://127.0.0.1:{self.port}')
            s.send_multipart([b'{}', b'{}'])
            time.sleep(.05)
            s.close()
            self.assertTrue(self.call()['ok'])



if __name__ == '__main__':
    unittest.main()
