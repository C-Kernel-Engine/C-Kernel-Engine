import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pack_nemotron_vision import pack
from server.vision_bundle import DECODER_FILES, RADIO_ENCODER_FILES, RADIO_SCHEMA, load, pin
from server.vision_nemotron import VisionWeights, prompt_segments


class FakeTokenizer:
    def __init__(self, ids):
        self.ids = ids

    def encode(self, text, *, add_bos):
        self.text = text
        self.add_bos = add_bos
        return self.ids


class NemotronVisionContractTests(unittest.TestCase):
    def test_pack_copies_only_vision_bits(self):
        prefix = 'vision_model.radio_model.'
        types = {prefix + 'input_conditioner.norm_' + key: 'F32' for key in ('mean', 'std')}
        types.update({prefix + 'model.patch_generator.' + key: 'BF16'
                      for key in ('cls_token.token', 'embedder.weight', 'pos_embed')})
        types.update({key: 'BF16' for key in ('unrelated', 'mlp1.0.weight', 'mlp1.1.weight', 'mlp1.3.weight')})
        index, bits = {}, bytearray()
        for name, dtype in types.items():
            value = bytes([128, 63]) if dtype == 'BF16' else bytes([0, 0, 128, 63])
            index[name] = {'dtype': dtype, 'shape': [1], 'data_offsets': [len(bits), len(bits) + len(value)]}
            bits.extend(value)
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / 'source', Path(tmp) / 'dest'
            head = json.dumps(index).encode()
            source.write_bytes(len(head).to_bytes(8, 'little') + head + bits)
            with patch('version.v8.scripts.nemotron_radio_checked_v8.LAYERS', 0):
                pack(source, dest)
            packed = VisionWeights(dest)
            self.assertEqual(set(packed.entries), set(index) - {'unrelated'})
            self.assertEqual(packed.read('mlp1.0.weight').tolist(), [0x3f80])
            self.assertEqual(packed.read(prefix + 'input_conditioner.norm_mean', dtype='F32').tolist(), [1.0])

    def test_full_tokenized_span_preserves_delimiters(self):
        tok = FakeTokenizer([1, 77, 131073, 131072, 131072, 131074, 99])
        self.assertEqual(prompt_segments(tok, 'before<image>after', 2),
                         ([1, 77, 131073], [131074, 99]))
        self.assertEqual(tok.text, 'before<img><image><image></img>after')
        self.assertFalse(tok.add_bos)

    def test_reject_truncated_and_injected_spans(self):
        for ids in ([1, 131073, 131072, 131074, 99],
                    [1, 131072, 131073, 131072, 131072, 131074, 99],
                    [1, 131073, 131072, 131072, 131074, 131073, 99]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                prompt_segments(FakeTokenizer(ids), '<image>', 2)

    def test_profile_pins_weights_blas_and_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder, decoder = root / 'encoder', root / 'decoder'
            for folder, names in ((encoder, RADIO_ENCODER_FILES), (decoder, (*DECODER_FILES, 'tokenizer.json'))):
                folder.mkdir()
                for name in names:
                    (folder / name).write_bytes(b'original')
            gguf = root / 'tokenizer.gguf'
            gguf.touch()
            profile = root / 'profile.json'
            data = pin(encoder, decoder, gguf, 'test', '<image>', schema=RADIO_SCHEMA)
            profile.write_text(json.dumps(data))
            self.assertEqual(load(profile)['schema'], RADIO_SCHEMA)
            for relative in ('encoder/libblas.so', 'encoder/vision.safetensors', 'decoder/libmodel.so'):
                with self.subTest(relative=relative):
                    artifact = root / relative
                    artifact.write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'bundle changed'):
                        load(profile)
                    artifact.write_bytes(b'original')
            with self.assertRaises(ValueError):
                pin(encoder, decoder, gguf, 'test', '<wrong>', schema=RADIO_SCHEMA)

    def test_exact_bf16_descriptor_and_truncated_extent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.safetensors'
            head = json.dumps({'a': {'dtype': 'BF16', 'data_offsets': [0, 4], 'shape': [2]}}).encode()
            body = len(head).to_bytes(8, 'little') + head + bytes([128, 63, 0, 192])
            path.write_bytes(body)
            tensor = VisionWeights(path).read('a')
            self.assertEqual(tensor.tolist(), [0x3f80, 0xc000])
            del tensor
            path.write_bytes(body[:-1])
            with self.assertRaises(ValueError):
                VisionWeights(path).read('a')


if __name__ == '__main__':
    unittest.main()
