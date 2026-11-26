"""Pinned generated Kokoro text embedding joined to generated duration alignment."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena, prepare_duration_fixture)
from tests import test_v8_kokoro_generated_albert_layer as first_layer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'version/v8/tts'))
import build_kokoro_text_embedding_circuit as author


class KokoroGeneratedTextEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence=os.environ.get('CKE_KOKORO_TEXT_EMBEDDING_EVIDENCE_DIR')
        if evidence:
            root=Path(evidence).resolve()/'text_embedding'
            root.mkdir(parents=True,exist_ok=True)
            cls.temp=SimpleNamespace(name=str(root),cleanup=lambda:None)
        else:
            cls.temp=tempfile.TemporaryDirectory()
            root=Path(cls.temp.name)
        cls.root=root
        fixture_path=ROOT/'tests/fixtures/tts/kokoro_text_embedding_pinned.npz'
        meta=json.loads(fixture_path.with_suffix('.json').read_text())
        if hashlib.sha256(fixture_path.read_bytes()).hexdigest()!=meta['fixture_sha256']:
            raise RuntimeError('text embedding fixture hash mismatch')
        with np.load(fixture_path) as archive:
            cls.text={name:archive[name].copy() for name in archive.files}
        for name,digest in meta['arrays_sha256'].items():
            if hashlib.sha256(cls.text[name].tobytes()).hexdigest()!=digest:
                raise RuntimeError('text embedding array hash mismatch: '+name)
        circuit = materialize_stage_circuit(root, author)
        fixture=prepare_duration_fixture(root,circuit,
            {'acoustic_text_encoder.embedding.weight':cls.text['table']})
        for key,value in vars(fixture).items(): setattr(cls,key,value)
        cls.layout,cls.calls,cls.library,cls.loaded,cls.fn=compile_native_graph(
            root,cls.source,circuit)
        cls.fn.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_int32)]
        cls.buffers={item['name']:item for item in
            cls.layout['memory']['activations']['buffers']}

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def view(self,arena,name,dtype=np.float32):
        item=self.buffers[name]
        return np.ndarray((item['size']//np.dtype(dtype).itemsize,),dtype,
            buffer=arena,offset=item['abs_offset'])

    def arena(self):
        arena=populated_duration_arena(self.layout,self.entries,self.bump,
                                       self.encoder,self.duration)
        self.view(arena,'text_features')[:]=-91.
        self.view(arena,'text_expanded')[:]=-98.
        return arena

    def test_connected_embedding_and_expansion(self):
        self.assertEqual(self.calls['errors'],[])
        self.assertEqual(len(self.calls['operations']),163)
        lookup=next(op for op in self.calls['operations'] if op['function']==
                      'audio_text_embedding_channel_major_f32')
        args={arg['name']:arg for arg in lookup['args']}
        self.assertEqual(args['ids']['buffer_ref'],'word_ids')
        self.assertEqual(args['table']['weight_ref'],
                         'acoustic_text_encoder.embedding.weight')
        arena=self.arena()
        frames=ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        self.assertEqual(frames.value,103)
        embedding=self.view(arena,'text_features').reshape(512,40)
        np.testing.assert_array_equal(embedding[:,:36].view(np.uint32),
                                      self.text['expected'].view(np.uint32))
        self.assertTrue(np.all(embedding[:,36:]==-91.))
        expanded=self.view(arena,'text_expanded').reshape(512,128)
        durations=self.view(arena,'runtime_values',np.int32)[:36]
        np.testing.assert_array_equal(durations,self.duration['durations'])
        expected=np.repeat(self.text['expected'],durations,axis=1)
        np.testing.assert_array_equal(expanded[:,:103].view(np.uint32),
                                      expected.view(np.uint32))
        self.assertTrue(np.all(expanded[:,103:]==-98.))
        print('CKE_NUMERICAL_CASE '+json.dumps({
            'case_id':'kokoro.text_embedding.generated-alignment-v1',
            'name':'generated text embedding and duration expansion',
            'provider':'audio_text_embedding_channel_major_f32',
            'dtype':'fp32','direction':'inference','oracle':'pinned-pytorch',
            'backend_version':'2.8.0+cpu','max_diff':0.0,'tolerance':0.0,
            'status':'pass','configuration':'tokens=36; frames=103; channels=512',
            'reproduction_command':'python3 -m unittest '+self.id()}))

    def test_failed_embedding_stops_text_expansion_and_output_publication(self):
        arena=self.arena()
        ids=self.view(arena,'word_ids',np.int32)
        ids[-1]=178
        frames=ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        self.assertTrue(np.all(self.view(arena,'text_features')==-91.))
        self.assertTrue(np.all(self.view(arena,'text_expanded')==-98.))
        self.assertEqual(frames.value,-999)

    def test_xray_captures_valid_regions_and_provenance(self):
        arena=self.arena()
        frames=ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        builder,xray=first_layer.xray_builder,first_layer.xray
        tensor_report={'torch':{'tensors':{}},'comparisons':{}}
        expected={
            'text_features':(self.text['expected'],40,36),
            'text_expanded':(np.repeat(self.text['expected'],
                self.duration['durations'],axis=1),128,frames.value)}
        points={}
        for tensor,(oracle,stride,valid) in expected.items():
            point=next(point for op in self.calls['operations']
                for point in op.get('semantic_checkpoints',[])
                if point['tensor']==tensor)
            points[tensor]=point
            if tensor=='text_features':
                self.assertEqual(point['resolved_contract_id'],
                                 'audio_text_embedding_exact_channel_major_fp32')
            else:
                self.assertEqual(point['resolved_contract_id'],
                    'audio_duration_expand_checked_strided_copy_fp32')
            selector=tensor if point['layer']<0 else f'{tensor}@{point["layer"]}'
            native_path=self.root/f'{tensor}-native.f32'
            oracle_path=self.root/f'{tensor}-pytorch.f32'
            self.view(arena,tensor).tofile(native_path)
            oracle.tofile(oracle_path)
            tensor_report['torch']['tensors'][selector]={
                'path':str(oracle_path),'shape':[512,valid]}
            tensor_report['comparisons'][selector]={
                'ck_path':str(native_path),'shape':[512,valid],
                'physical_shape':[512,stride], 'capacity_shape':[512,stride],
                'valid_shape':[512,valid], 'physical_strides':[stride,1]}
        runtime=builder.capture_runtime_library_identity(
            self.loaded,'ck_kokoro_text_embedding_bounded')
        subject=builder.build_manifest(backend='ck',call_ir=self.calls,
            tensor_report=tensor_report,model='kokoro_text_embedding_bounded',
            source='generated_native',phase='prefill',
            loaded_library=self.library,runtime_library=runtime)
        oracle=builder.build_manifest(backend='pytorch',call_ir=self.calls,
            tensor_report=tensor_report,model='kokoro_text_embedding_bounded',
            source='pinned_capture',phase='prefill')
        for tensor,point in points.items():
            profile={'schema':'cke.parity_profile','schema_version':1,
                'name':'kokoro_text_embedding','backend':'pytorch',
                'contract_schema_version':1,
                'required_match_fields':['checkpoint_id','producer',
                    'logical_layout','axis_names','resolved_contract_id',
                    'kernel_id','function'],
                'observed_storage':{'default':'fp32','checkpoints':{}},
                'dtype_thresholds':{'fp32':{'cosine_min':0.99999,
                    'rmse_max':0.0,'relative_rmse_max':0.0,
                    'max_abs_max':0.0,'finite_required':True}},
                'checkpoint_order':[point['id']],
                'interval_expansions':{},'backend_mappings':{}}
            one=lambda manifest:{**manifest,'checkpoints':[item for item in
                manifest['checkpoints'] if item['checkpoint_id']==point['id']]}
            result=xray.compare_manifests(one(subject),one(oracle),profile)
            self.assertEqual(result['status'],'pass',(tensor,result))
            (self.root/f'{tensor}-xray.json').write_text(json.dumps(result,indent=2))
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        (self.root/'native-checkpoints.json').write_text(json.dumps(subject,indent=2))
        (self.root/'pytorch-checkpoints.json').write_text(json.dumps(oracle,indent=2))


if __name__=='__main__': unittest.main()
