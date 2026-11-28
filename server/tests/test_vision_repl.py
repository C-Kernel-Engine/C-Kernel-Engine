from email.message import Message
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.certify_pi_vision_replacement import call as call_pi
from scripts.vision_repl import VisionConversation, VisionHTTP, read_stream


RID = 'vision-' + 'a' * 32
PNG = b'\x89PNG\r\n\x1a\n'


def event(text='', stop=None, request_id=RID):
    value = {'id': request_id, 'choices': [{'delta': {'content': text}, 'finish_reason': stop}]}
    return ('data: ' + json.dumps(value) + '\n\n').encode()


class VisionReplTests(unittest.TestCase):
    def test_followup_replacement_reset_and_failure_are_transactional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.png').write_bytes(PNG + b'one')
            (root / 'b.png').write_bytes(PNG + b'two')
            transport = Mock()
            transport.reply.return_value = {'text': 'answer', 'stop_reason': 'stop'}
            session = VisionConversation(root, transport)
            stage = Mock()
            with self.assertRaises(ValueError):
                session.ask('no image', stage)
            session.attach('a.png')
            session.ask('first', stage)
            first_image = session.image
            session.ask('follow-up', stage)
            messages = transport.reply.call_args.args[0]
            self.assertEqual([msg['role'] for msg in messages], ['user', 'assistant', 'user'])
            self.assertEqual(messages[0]['content'][0], first_image)
            self.assertEqual(messages[2]['content'], [{'type': 'text', 'text': 'follow-up'}])

            old_history = session.history.copy()
            with self.assertRaises(OSError):
                session.attach('does-not-exist.png')
            self.assertEqual(session.history, old_history)
            transport.reply.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                session.ask('cancel this', stage)
            transport.reply.side_effect = None
            transport.reply.return_value = {'text': 'partial', 'stop_reason': 'length'}
            session.ask('do not keep partial output', stage)
            self.assertEqual(session.history, old_history)

            session.attach('b.png')
            session.ask('replacement', stage)
            messages = transport.reply.call_args.args[0]
            self.assertEqual(len(messages), 1)
            self.assertNotEqual(messages[0]['content'][0], first_image)
            session.reset()
            self.assertIsNone(session.image)
            self.assertEqual(session.history, [])

    def test_requires_terminal_event_and_matching_identity(self):
        stage = Mock()
        data = b': native stage\n\n' + event('Red') + event(stop='stop') + b'data: [DONE]\n\n'
        self.assertEqual(read_stream(io.BytesIO(data), RID, stage)['text'], 'Red')
        stage.assert_called_once_with(RID)
        for invalid in (event('Red'), event(stop='stop'), b'data: [DONE]\n',
                        event(request_id='different'), b'data: {"error":{}}\n',
                        event(stop='stop') + event('extra') + b'data: [DONE]\n'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                read_stream(io.BytesIO(invalid), RID, stage)

    def test_interrupt_cancels_matching_request_and_closes_response(self):
        client = VisionHTTP('http://127.0.0.1:8120', 'test', 10, 16)
        client.opener = Mock()
        response = io.BytesIO(b': native stage\n\n')
        response.headers = Message()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['X-Request-ID'] = RID
        client.opener.open.side_effect = [response, io.BytesIO()]
        with self.assertRaises(KeyboardInterrupt):
            client.reply([], Mock(side_effect=KeyboardInterrupt))
        self.assertTrue(response.closed)
        req = client.opener.open.call_args_list[1].args[0]
        self.assertEqual(req.get_method(), 'DELETE')
        self.assertEqual(req.full_url, 'http://127.0.0.1:8120/v1/requests/' + RID)

    def test_reject_non_loopback_endpoints(self):
        for endpoint in ('http://example.org:8120', 'http://127.0.0.1:8120/login', 'http://127.0.0.1:65536'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                VisionHTTP(endpoint, 'test', 10, 16)

    def test_pi_replacement_fails_on_stale_images_and_unacknowledged_turns(self):
        part = {'type': 'image', 'mimeType': 'image/png', 'data': 'Ag=='}
        command = {'id': 'new', 'type': 'prompt', 'images': [part]}
        ack = {'type': 'response', 'id': 'new', 'command': 'prompt', 'success': True}
        user = {'role': 'user', 'content': [part]}
        answer = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'blue'}],
                  'stopReason': 'stop', 'responseId': RID}

        def pipe(events):
            return Mock(stdin=io.StringIO(), stdout=io.StringIO('\n'.join(map(json.dumps, events))))

        def end(messages):
            return {'type': 'agent_end', 'messages': messages}

        result = call_pi(pipe([ack, end([user, answer])]), command)
        self.assertEqual(result['text'], 'blue')
        self.assertEqual(result['request_id'], RID)
        for events in ([end([user, answer])], [ack, end([user, user, answer])],
                       [ack, end([{'role': 'user', 'content': [dict(part, data='AA==')]}, answer])]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                call_pi(pipe(events), command)
        reset = {'type': 'response', 'id': 'reset', 'command': 'new_session', 'success': True,
                 'data': {'cancelled': True}}
        with self.assertRaisesRegex(ValueError, 'did not replace'):
            call_pi(pipe([reset]), {'type': 'new_session', 'id': 'reset'})


if __name__ == '__main__':
    unittest.main()
