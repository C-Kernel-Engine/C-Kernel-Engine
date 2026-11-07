"""Standalone discovery client; requires pyzmq, no application dependencies."""
import argparse
import json
import uuid
import zmq

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, default=8120)
p.add_argument('--node-id', required=True)
p.add_argument('--operation', default='workers.list')
p.add_argument('--worker-id')
a = p.parse_args()
request_id = str(uuid.uuid4())
with zmq.Context() as ctx:
    with ctx.socket(zmq.REQ) as s:
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 2000)
        s.setsockopt(zmq.SNDTIMEO, 2000)
        s.connect(f'tcp://127.0.0.1:{a.port}')
        s.send_json({'schema': 'cke.control.v1', 'request_id': request_id,
                     'operation': a.operation, 'node_id': a.node_id,
                     'params': {'worker_id': a.worker_id} if a.worker_id else {}})
        try:
            response = s.recv_json()
        except zmq.Again:
            raise SystemExit('Adapter unavailable: request timed out; discard this REQ socket before retry.')
        if any(response.get(k) != v for k, v in {
            'schema': 'cke.control.v1', 'request_id': request_id, 'node_id': a.node_id
        }.items()):
            raise SystemExit('Invalid response correlation')
        print(json.dumps(response, indent=2))
        raise SystemExit(0 if response['ok'] else 1)
