import unittest
from version.v8.tts.ming3_text_segments import sentence_spans,validate_spans
class TextSegmentTests(unittest.TestCase):
    def test_lossless_explicit_sentence_policy(self):
        text='First, count 3 items. Then read 42 words! Is this complete? Yes.'
        spans=sentence_spans(text);validate_spans(text,spans)
        self.assertEqual(''.join(text[a:b] for a,b in spans),text);self.assertEqual(len(spans),4)
        for bad in (spans[:-1],[(0,3),(4,len(text))],[(0,len(text)),(1,2)]):
            with self.assertRaises(ValueError):validate_spans(text,bad)
if __name__=='__main__':unittest.main()
