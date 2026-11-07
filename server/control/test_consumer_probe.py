import copy
import json
from pathlib import Path
import tempfile
import threading
import subprocess
import sys
import unittest

import zmq

from consumer_probe import Consumer, ContractError, ZmqExchange, atomic_write, validate
import test_node_adapter as native

FIXTURE = Path(__file__).parent / 'fixtures' / 'consumer-contract.json'


def scripted(request):
    fixture = json.loads(FIXTURE.read_text())
    response = copy.deepcopy(fixture['responses'][request['operation']])
    response.update({k: request[k] for k in ('schema', 'request_id', 'node_id')})
    return response


class ConsumerContract(unittest.TestCase):
    def test_fixture_positive_and_negative_outcomes(self):
        fixture = json.loads(FIXTURE.read_text())
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                count = 0
                def exchange(request):
                    nonlocal count
                    count += 1
                    r = scripted(request)
                    if count == case.get('mutate_exchange'):
                        target = r
                        path = case['path']
                        for key in path[:-1]:
                            target = target[key]
                        target[path[-1]] = case['value']
                    return r
                state = Consumer('fixture-node', exchange).run()
                self.assertEqual(state['last_attempt']['status'], case['expected_status'])
                self.assertEqual(state['last_attempt']['error'], case.get('expected_error'))
                if case['expected_status'] == 'FAIL':
                    self.assertIsNone(state['last_success'])

    def test_nulls_are_preserved_and_fields_are_sanitized(self):
        def exchange(request):
            r = scripted(request)
            r['secret'] = '/private/token'
            r['data']['secret'] = '/private/model'
            return r
        result = Consumer('fixture-node', exchange).run()
        self.assertEqual(result['last_attempt']['status'], 'PASS')
        self.assertNotIn('private', json.dumps(result))
        for e in result['last_success']['exchanges'][2:4]:
            self.assertIsNone(e['response']['data']['boot_id'])
            self.assertIsNone(e['response']['data']['capabilities'])

    def test_history_and_incomplete_publication(self):
        published = []
        c = Consumer('fixture-node', scripted, publish=lambda s: published.append(copy.deepcopy(s)))
        success = c.run()['last_success']
        def timeout(_):
            raise ContractError('ADAPTER_TIMEOUT')
        c.exchange = timeout
        failed = c.run()
        self.assertEqual(failed['last_success'], success)
        self.assertEqual(failed['last_attempt']['status'], 'FAIL')
        self.assertEqual(published[-2]['last_attempt']['status'], 'INCOMPLETE')
        self.assertEqual(published[-2]['last_success'], success)
        error = failed['last_error']
        c.exchange = scripted
        recovered = c.run()
        self.assertEqual(recovered['last_error'], error)
        self.assertNotEqual(recovered['last_success']['attempt_id'], success['attempt_id'])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'state.json'
            atomic_write(p, recovered)
            restored = Consumer('fixture-node', scripted, json.loads(p.read_text())).run()
            self.assertEqual(restored['last_attempt']['status'], 'PASS')

    def test_missing_fields_and_wrong_types(self):
        request = dict(schema='cke.control.v1', request_id='r', node_id='fixture-node',
                       operation='worker.status', params={'worker_id': 'speech'})
        original = scripted(request)
        for key in original['data']:
            broken = copy.deepcopy(original)
            del broken['data'][key]
            with self.subTest(missing=key), self.assertRaises(ContractError):
                validate(broken, request)
        for key, value in [('queued', True), ('worker_generation', -1), ('busy', 1),
                           ('ready', 'true'), ('boot_id', 'unknown'),
                           ('observed_at', 'yesterday'), ('capabilities', {})]:
            broken = copy.deepcopy(original)
            broken['data'][key] = value
            with self.subTest(type=key), self.assertRaises(ContractError):
                validate(broken, request)

    def test_worker_failure_is_not_adapter_failure(self):
        def exchange(request):
            r = scripted(request)
            if request['operation'].startswith('worker.'):
                d = r['data']
                d.update(observation_status='unavailable', observation_error='WORKER_UNAVAILABLE',
                         ready=None, busy=None, queued=None, worker_generation=None)
            return r
        state = Consumer('fixture-node', exchange).run()
        self.assertEqual(state['last_attempt']['status'], 'PASS')
        self.assertEqual(state['last_success']['exchanges'][2]['response']['data']['observation_status'], 'unavailable')

    def test_cli_configuration_failure_replaces_current_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'state.json'
            old = Consumer('fixture-node', scripted).run()
            atomic_write(state, old)
            cmd = [sys.executable, str(Path(__file__).with_name('consumer_probe.py')),
                   '--node-id', 'fixture-node', '--port', '0', '--state', str(state)]
            result = subprocess.run(cmd, capture_output=True, timeout=3)
            self.assertEqual(result.returncode, 1)
            current = json.loads(state.read_text())
            self.assertEqual(current['last_attempt']['error'], 'INVALID_CONFIG')
            self.assertEqual(current['last_success'], old['last_success'])
            state.write_text('invalid json')
            cmd[cmd.index('--port') + 1] = '8120'
            result = subprocess.run(cmd, capture_output=True, timeout=3)
            self.assertEqual(result.returncode, 1)
            current = json.loads(state.read_text())
            self.assertEqual(current['last_attempt']['error'], 'INVALID_STATE')
            self.assertIsNone(current['last_success'])

    def test_partial_results_not_published_as_success(self):
        calls = 0
        def exchange(request):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise ContractError('MALFORMED_REPLY')
            return scripted(request)
        state = Consumer('fixture-node', exchange).run()
        self.assertEqual(len(state['last_attempt']['exchanges']), 3)
        self.assertIsNone(state['last_success'])


class ConsumerSocket(native.Adapter):
    def test_live_sequence_shutdown_restart_and_worker_failure(self):
        c = Consumer('test-node', ZmqExchange(self.port, 300))
        first = c.run()
        self.assertEqual(first['last_attempt']['status'], 'PASS')
        self.assertEqual([e['request']['operation'] for e in first['last_success']['exchanges']],
                         ['node.describe', 'workers.list', 'worker.status', 'worker.capabilities',
                          'worker.status', 'worker.capabilities', 'node.describe'])
        observations = [e['response']['data'] for e in first['last_success']['exchanges']
                        if e['request']['operation'] == 'worker.status']
        self.assertEqual([o['observation_status'] for o in observations], ['available', 'unavailable'])
        self.proc.terminate()
        self.proc.wait(timeout=3)
        failed = c.run()
        self.assertEqual(failed['last_attempt']['error'], 'ADAPTER_TIMEOUT')
        self.assertEqual(failed['last_success'], first['last_success'])
        self.start()
        final = c.run()
        self.assertEqual(final['last_attempt']['status'], 'PASS')
        self.assertNotEqual(first['last_success']['exchanges'][0]['response']['adapter_boot_id'],
                            final['last_success']['exchanges'][0]['response']['adapter_boot_id'])

    def test_real_restart_mid_discovery(self):
        exchange = ZmqExchange(self.port)
        count = 0
        def restarting(request):
            nonlocal count
            count += 1
            if count == 3:
                self.proc.terminate()
                self.proc.wait(timeout=3)
                self.start()
            return exchange(request)
        result = Consumer('test-node', restarting).run()
        self.assertEqual(result['last_attempt']['error'], 'ADAPTER_RESTARTED')
        self.assertIsNone(result['last_success'])


class MalformedPeer(unittest.TestCase):
    def test_real_socket_malformed_replies(self):
        for payload in [b'not json', b'{"ok":true,"ok":false}', b'[]', b'\xff',
                        b'{"schema":"cke.control.v1","request_id":"wrong"}',
                        [b'{}', b'{}']]:
            with self.subTest(payload=payload), zmq.Context() as ctx:
                ready = threading.Event()
                endpoint = []
                def serve():
                    with ctx.socket(zmq.REP) as s:
                        s.setsockopt(zmq.LINGER, 0)
                        s.setsockopt(zmq.RCVTIMEO, 2000)
                        endpoint.append(s.bind_to_random_port('tcp://127.0.0.1'))
                        ready.set()
                        s.recv()
                        if isinstance(payload, list):
                            s.send_multipart(payload)
                        else:
                            s.send(payload)
                t = threading.Thread(target=serve)
                t.start()
                self.assertTrue(ready.wait(2))
                result = Consumer('fixture-node', ZmqExchange(endpoint[0])).run()
                t.join(3)
                self.assertFalse(t.is_alive())
                self.assertEqual(result['last_attempt']['status'], 'FAIL')
                self.assertIsNone(result['last_success'])
                self.assertNotIn('not json', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
