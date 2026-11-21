"""Coverage and bounded WAV assembly tests for Ming long text."""
import tempfile
from pathlib import Path
import unittest
import wave

import numpy as np

from version.v8.tts.capture_ming3_synthesis_reference import digest
from version.v8.tts.ming3_text_segments import sentence_spans, validate_spans
from version.v8.tts.run_ming3_paragraph_v8 import join_chunk_wavs, write_wav


class TextSegmentTests(unittest.TestCase):
    def test_lossless_explicit_sentence_policy(self):
        text='First, count 3 items. Then read 42 words! Is this complete? Yes.'
        spans=sentence_spans(text);validate_spans(text,spans)
        self.assertEqual(''.join(text[a:b] for a,b in spans),text);self.assertEqual(len(spans),4)
        for bad in (spans[:-1],[(0,3),(4,len(text))],[(0,len(text)),(1,2)]):
            with self.assertRaises(ValueError):validate_spans(text,bad)

    def test_wav_assembly_preserves_pcm_order_and_rejects_corrupt_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            chunks, pcm = [], []
            for index, samples in enumerate((35_001, 11)):
                path = output / f"chunk-{index}.wav"
                write_wav(path, np.linspace(-0.5, 0.5, samples, dtype=np.float32), 44100)
                with wave.open(str(path), "rb") as reader:
                    pcm.append(reader.readframes(samples))
                chunks.append(dict(index=index, measurements={"samples": samples},
                                   wav_sha256=digest(path)))
            path = output / "paragraph.wav"
            self.assertEqual(join_chunk_wavs(output, chunks, 44100), 35_012)
            with wave.open(str(path), "rb") as reader:
                self.assertEqual(reader.getnframes(), 35_012)
                self.assertEqual(reader.readframes(35_012), b"".join(pcm))
            complete = path.read_bytes()
            with (output / "chunk-1.wav").open("ab") as file:
                file.write(b"changed")
            with self.assertRaisesRegex(ValueError, "chunk integrity"):
                join_chunk_wavs(output, chunks, 44100)
            self.assertEqual(path.read_bytes(), complete)
            self.assertEqual(list(output.glob(".paragraph-*")), [])

if __name__=='__main__':unittest.main()
