"""Independent, read-only simulated consumer of cke.control.v1.

Not an application registry. Successful discovery does not certify worker readiness.
Only validated allowlisted fields are retained; malformed peer payloads are not logged.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

SCHEMA = 'cke.control.v1'
STATE_SCHEMA = 'cke.control.consumer.v1'
IDENT = re.compile(r'[A-Za-z0-9_.-]{1,64}\Z')
BOOT = re.compile(r'[a-f0-9]{32}\Z')
ERRORS = {'MALFORMED_MESSAGE', 'INVALID_REQUEST', 'INVALID_PARAMS',
          'UNSUPPORTED_OPERATION', 'WORKER_NOT_FOUND'}


class ContractError(Exception):
    """Safe fixed-code error; never includes peer text or local filesystem paths."""


def require(condition, code='INVALID_RESPONSE'):
    if not condition:
        raise ContractError(code)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def timestamp(value):
    require(isinstance(value, str) and len(value) <= 32 and value.endswith('Z'))
    try:
        datetime.fromisoformat(value[:-1] + '+00:00')
    except ValueError:
        raise ContractError('INVALID_RESPONSE') from None
    return value


def identifier(value):
    return isinstance(value, str) and IDENT.fullmatch(value) is not None


def integer(value, maximum=None):
    return type(value) is int and value >= 0 and (maximum is None or value <= maximum)


def revision(value):
    return value is None or value == 'unknown' or (
        isinstance(value, str) and re.fullmatch(r'[a-f0-9]{40,64}', value) is not None)


def validate(reply, request):
    require(type(reply) is dict)
    require(all(k in reply for k in ('schema', 'request_id', 'node_id', 'adapter_boot_id',
                                    'observed_at', 'ok', 'data', 'error')))
    require(all(reply[k] == request[k] for k in ('schema', 'request_id', 'node_id')),
            'CORRELATION_MISMATCH')
    require(isinstance(reply['adapter_boot_id'], str) and BOOT.fullmatch(reply['adapter_boot_id']))
    timestamp(reply['observed_at'])
    require(type(reply['ok']) is bool)
    if not reply['ok']:
        require(reply['data'] is None and type(reply['error']) is dict)
        require(reply['error'].get('code') in ERRORS)
        raise ContractError('REMOTE_' + reply['error']['code'])
    require(reply['error'] is None and type(reply['data']) is dict)
    d = reply['data']
    op = request['operation']
    if op == 'node.describe':
        keys = ('adapter_health', 'cke_revision', 'transport', 'worker_discovery', 'worker_count')
        require(all(k in d for k in keys))
        require(d['adapter_health'] == 'ready' and d['transport'] == 'loopback-only'
                and d['worker_discovery'] == 'explicit-registration')
        require(revision(d['cke_revision']) and integer(d['worker_count'], 32))
        clean = {k: d[k] for k in keys}
    elif op == 'workers.list':
        require(type(d.get('workers')) is list and len(d['workers']) <= 32)
        ids = []
        for w in d['workers']:
            require(type(w) is dict and identifier(w.get('worker_id')))
            require(w['worker_id'] not in ids, 'DUPLICATE_WORKER')
            ids.append(w['worker_id'])
        clean = {'workers': [{'worker_id': wid} for wid in ids]}
    else:
        keys = ('worker_id', 'observed_at', 'observation_status', 'observation_error',
                'boot_id', 'cke_revision', 'capabilities_status', 'capabilities',
                'ready', 'busy', 'queued', 'worker_generation')
        require(all(k in d for k in keys))
        require(d['worker_id'] == request['params']['worker_id'], 'WORKER_MISMATCH')
        timestamp(d['observed_at'])
        require(d['boot_id'] is None or (isinstance(d['boot_id'], str) and BOOT.fullmatch(d['boot_id'])))
        require(revision(d['cke_revision']))
        # v1 currently declares no capability schema. Do not infer support from registration.
        require(d['capabilities_status'] == 'unsupported' and d['capabilities'] is None)
        require(all(d[k] is None or type(d[k]) is bool for k in ('ready', 'busy')))
        require(all(d[k] is None or integer(d[k]) for k in ('queued', 'worker_generation')))
        require(d['observation_status'] in ('available', 'unavailable'))
        if d['observation_status'] == 'available':
            require(d['observation_error'] is None)
        else:
            require(d['observation_error'] in ('WORKER_TIMEOUT', 'WORKER_UNAVAILABLE'))
            require(all(d[k] is None for k in ('ready', 'busy', 'queued', 'worker_generation')))
        clean = {k: d[k] for k in keys}
    return {k: reply[k] for k in ('schema', 'request_id', 'node_id', 'adapter_boot_id',
                                 'observed_at', 'ok', 'error')} | {'data': clean}


class ZmqExchange:
    def __init__(self, port, timeout_ms=1000):
        require(type(port) is int and 1024 <= port <= 65535, 'INVALID_CONFIG')
        require(type(timeout_ms) is int and 1 <= timeout_ms <= 5000, 'INVALID_CONFIG')
        self.port, self.timeout_ms = port, timeout_ms

    def __call__(self, request):
        import zmq
        # Fresh REQ socket after every attempt, including timeout/EFSM cases.
        with zmq.Context() as ctx:
            with ctx.socket(zmq.REQ) as sock:
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.MAXMSGSIZE, 8192)
                sock.setsockopt(zmq.RCVHWM, 1)
                sock.setsockopt(zmq.SNDHWM, 1)
                sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                sock.connect(f'tcp://127.0.0.1:{self.port}')
                try:
                    sock.send_json(request)
                    raw = sock.recv()
                    require(not sock.getsockopt(zmq.RCVMORE), 'MALFORMED_REPLY')
                except zmq.Again:
                    raise ContractError('ADAPTER_TIMEOUT') from None
                except zmq.ZMQError:
                    raise ContractError('TRANSPORT_ERROR') from None
        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result, 'MALFORMED_REPLY')
                result[key] = value
            return result
        try:
            return json.loads(raw, object_pairs_hook=pairs,
                              parse_constant=lambda _: require(False, 'MALFORMED_REPLY'))
        except (ValueError, UnicodeError, RecursionError):
            raise ContractError('MALFORMED_REPLY') from None


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.cke-observation-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Consumer:
    def __init__(self, node_id, exchange, previous=None, publish=lambda state: None):
        require(identifier(node_id), 'INVALID_CONFIG')
        self.node_id, self.exchange, self.publish = node_id, exchange, publish
        self.state = {'schema': STATE_SCHEMA, 'node_id': node_id,
                      'last_success': None, 'last_attempt': None, 'last_error': None}
        if previous is not None:
            require(type(previous) is dict and previous.get('schema') == STATE_SCHEMA
                    and previous.get('node_id') == node_id, 'INVALID_STATE')
            # State belongs to this local consumer. Never import it from a peer response.
            self.state = copy.deepcopy(previous)

    def run(self):
        attempt = {'attempt_id': str(uuid.uuid4()), 'started_at': now(),
                   'finished_at': None, 'status': 'INCOMPLETE', 'error': None, 'exchanges': []}
        self.state['last_attempt'] = attempt
        self.publish(self.state)
        boot = None
        def call(op, params=None):
            nonlocal boot
            request = {'schema': SCHEMA, 'request_id': str(uuid.uuid4()),
                       'operation': op, 'node_id': self.node_id, 'params': params or {}}
            reply = validate(self.exchange(request), request)
            if boot is None:
                boot = reply['adapter_boot_id']
            require(reply['adapter_boot_id'] == boot, 'ADAPTER_RESTARTED')
            attempt['exchanges'].append({'request': request, 'response': reply})
            return reply['data']
        try:
            description = call('node.describe')
            listing = call('workers.list')['workers']
            require(description['worker_count'] == len(listing), 'WORKER_SET_CHANGED')
            for worker in listing:
                params = {'worker_id': worker['worker_id']}
                status = call('worker.status', params)
                capabilities = call('worker.capabilities', params)
                require(status['boot_id'] == capabilities['boot_id'], 'WORKER_RESTARTED')
            # Fence the pass: even an empty worker list must not pass across an adapter restart.
            final = call('node.describe')
            require(final == description, 'NODE_DESCRIPTION_CHANGED')
            attempt['status'] = 'PASS'
        except ContractError as exc:
            attempt['status'], attempt['error'] = 'FAIL', str(exc)
        except Exception:
            attempt['status'], attempt['error'] = 'FAIL', 'CLIENT_ERROR'
        attempt['finished_at'] = now()
        if attempt['status'] == 'PASS':
            self.state['last_success'] = copy.deepcopy(attempt)
        else:
            self.state['last_error'] = {'attempt_id': attempt['attempt_id'],
                                        'observed_at': attempt['finished_at'], 'code': attempt['error']}
        self.publish(self.state)
        return copy.deepcopy(self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-id', required=True)
    parser.add_argument('--port', type=int, default=8120)
    parser.add_argument('--timeout-ms', type=int, default=1000)
    parser.add_argument('--state', type=Path, required=True,
                        help='single-writer local state file; keeps historical success separately')
    args = parser.parse_args()
    previous = None
    try:
        if args.state.exists():
            require(args.state.stat().st_size <= 1024 * 1024, 'INVALID_STATE')
            try:
                previous = json.loads(args.state.read_text())
            except (ValueError, UnicodeError):
                raise ContractError('INVALID_STATE') from None
        consumer = Consumer(args.node_id, ZmqExchange(args.port, args.timeout_ms), previous,
                            lambda state: atomic_write(args.state, state))
        result = consumer.run()
    except ContractError as exc:
        # Do not leave yesterday's PASS looking like the current CLI verdict.
        code = str(exc)
        def failed(_):
            raise ContractError(code)
        safe_previous = previous if (type(previous) is dict
            and previous.get('schema') == STATE_SCHEMA
            and previous.get('node_id') == args.node_id) else None
        consumer = Consumer(args.node_id, failed, safe_previous,
                            lambda state: atomic_write(args.state, state))
        result = consumer.run()
    print(json.dumps({'status': result['last_attempt']['status'],
                      'error': result['last_attempt']['error']}))
    return 0 if result['last_attempt']['status'] == 'PASS' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ContractError, OSError) as exc:
        print(json.dumps({'status': 'FAIL', 'error': str(exc) if isinstance(exc, ContractError) else 'STATE_IO_ERROR'}))
        raise SystemExit(1)
