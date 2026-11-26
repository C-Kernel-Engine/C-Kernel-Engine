"""Pinned generated Kokoro acoustic text encoder joined to generated durations."""
import ctypes
import copy
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph,build_ir_v8
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit
from tests.v8_kokoro_duration_fixture_support import (
    populated_duration_arena,prepare_duration_fixture)
from tests import test_v8_kokoro_generated_albert_layer as xray_support

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'version/v8/tts'))
import build_kokoro_text_encoder_circuit as author

CHECKPOINT_ATOL={
    'text_conv0':1.3e-5,'text_norm0':8e-6,'text_act0':8e-6,
    'text_conv1':5e-6,'text_norm1':7e-6,'text_act1':7e-6,
    'text_conv2':9e-6,'text_norm2':6e-6,'text_act2':6e-6,
    'text_encoder_features':3e-6,'text_expanded':3e-6,
    'duration_expanded':3e-5,
}


class KokoroGeneratedTextEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence=os.environ.get('CKE_KOKORO_TEXT_ENCODER_EVIDENCE_DIR')
        if evidence:
            root=Path(evidence).resolve()/'text_encoder';root.mkdir(parents=True,exist_ok=True)
            cls.temp=SimpleNamespace(name=str(root),cleanup=lambda:None)
        else:cls.temp=tempfile.TemporaryDirectory()
        root=Path(cls.temp.name);cls.root=root
        path=ROOT/'tests/fixtures/tts/kokoro_text_encoder_pinned.npz'
        meta=json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['fixture_sha256']:
            raise RuntimeError('text encoder fixture hash mismatch')
        with np.load(path) as archive:
            cls.text={name:archive[name].copy() for name in archive.files}
        path=ROOT/'tests/fixtures/tts/kokoro_text_embedding_pinned.npz'
        with np.load(path) as archive:
            table=archive['table'].copy()
        weights={'acoustic_text_encoder.embedding.weight':table}
        for index in range(3):
            for kind in ('weight','bias'):
                weights[f'acoustic_text_encoder.conv{index}.{kind}']=cls.text[f'conv{index}_{kind}']
            weights[f'acoustic_text_encoder.norm{index}.weight']=cls.text[f'norm{index}_gamma']
            weights[f'acoustic_text_encoder.norm{index}.bias']=cls.text[f'norm{index}_beta']
        for kind in ('weight_ih','weight_hh','bias_ih','bias_hh'):
            weights[f'acoustic_text_encoder.lstm.{kind}']=cls.text[f'lstm_{kind}']
        circuit = materialize_stage_circuit(root, author)
        fixture=prepare_duration_fixture(root,circuit,weights)
        for key,value in vars(fixture).items():setattr(cls,key,value)
        cls.layout,cls.calls,cls.library,cls.loaded,cls.fn=compile_native_graph(
            root,cls.source,circuit)
        cls.fn.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_int32)]
        cls.buffers={item['name']:item for item in
                     cls.layout['memory']['activations']['buffers']}

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def view(self,arena,name,dtype=np.float32):
        item=self.buffers[name]
        return np.ndarray((item['size']//np.dtype(dtype).itemsize,),dtype,
            buffer=arena,offset=item['abs_offset'])

    def arena(self):
        arena=populated_duration_arena(self.layout,self.entries,self.bump,
                                       self.encoder,self.duration)
        for name in ('text_features','text_expanded','text_encoder_features',
                     'duration_expanded'):
            self.view(arena,name)[:]=-91.
        return arena

    def test_generated_text_encoder_and_expansion(self):
        self.assertEqual(self.calls['errors'],[])
        arena=self.arena();frames=ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        self.assertEqual(frames.value,103)
        self.assertEqual(len([op for op in self.calls['operations'] if
            op['function']=='audio_conv1d_checked_channel_major_f32']),3)
        for index in range(3):
            for name,key,shape,stride,limit in (
                (f'text_conv{index}',f'conv{index}_output',(512,40),40,36),
                (f'text_norm{index}',f'norm{index}_output',(36,512),512,512),
                (f'text_act{index}',f'act{index}_output',(36,512),512,512)):
                actual=self.view(arena,name).reshape(shape)
                expected=self.text[key] if name.startswith('text_conv') else self.text[key].T
                diff=np.abs(actual[:,:limit]-expected)
                self.assertTrue(np.isfinite(actual[:,:limit]).all(),name)
                self.assertLessEqual(float(diff.max()),CHECKPOINT_ATOL[name],
                                     (name,float(diff.max())))
        features=self.view(arena,'text_encoder_features').reshape(36,512)
        np.testing.assert_allclose(features,self.text['lstm_output_token_major'],
            rtol=0,atol=3e-6)
        expanded=self.view(arena,'text_expanded').reshape(512,128)
        np.testing.assert_allclose(expanded[:,:103],self.text['text_expanded'],
            rtol=0,atol=3e-6)
        self.assertTrue(np.all(expanded[:,103:]==-91.))
        durations=self.view(arena,'runtime_values',np.int32)[:36]
        np.testing.assert_array_equal(durations,self.duration['durations'])
        self.assertEqual(int(durations.sum()),frames.value)
        duration_expanded=self.view(arena,'duration_expanded').reshape(640,128)
        expected_duration=np.repeat(self.duration['predictor_output'].T,
                                    durations,axis=1)
        np.testing.assert_allclose(duration_expanded[:,:frames.value],
                                   expected_duration,rtol=0,
                                   atol=CHECKPOINT_ATOL['duration_expanded'])
        self.assertTrue(np.all(duration_expanded[:,frames.value:]==-91.))
        print('CKE_NUMERICAL_CASE '+json.dumps({'case_id':'kokoro.text-encoder.generated-v1',
            'name':'generated complete acoustic text encoder and expansion',
            'provider':'generated_text_encoder','dtype':'fp32','direction':'inference',
            'oracle':'pinned-pytorch','backend_version':'2.8.0+cpu','status':'pass',
            'max_diff':float(np.max(np.abs(expanded[:,:103]-self.text['text_expanded']))),
            'tolerance':3e-6,'configuration':'36 phonemes; 103 frames; 512 channels; af_heart',
            'reproduction_command':'python3 -m unittest '+self.id()}))

    def test_repeated_request_resets_lstm_and_keeps_shared_extent(self):
        arena=self.arena()
        frames=ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        first=self.view(arena,'text_encoder_features').copy()
        first_expanded=self.view(arena,'text_expanded').copy()
        self.view(arena,'text_encoder_features')[:]=-17.
        self.view(arena,'text_expanded')[:]=-19.
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        self.assertEqual(frames.value,103)
        np.testing.assert_array_equal(self.view(arena,'text_encoder_features'),first)
        np.testing.assert_array_equal(self.view(arena,'text_expanded').reshape(512,128)[:,:103],
            first_expanded.reshape(512,128)[:,:103])
        self.assertTrue(np.all(self.view(arena,'text_expanded').reshape(512,128)[:,103:]==-19.))

    def test_failed_conv_stops_downstream_and_does_not_publish_frames(self):
        arena=self.arena()
        weight=next(item for item in self.layout['memory']['weights']['entries']
            if item['name']=='acoustic_text_encoder.conv0.weight')
        np.ndarray((1,),np.float32,buffer=arena,offset=weight['abs_offset'])[0]=np.nan
        frames=ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        self.assertEqual(frames.value,-999)
        self.assertTrue(np.all(self.view(arena,'text_encoder_features')==-91.))
        self.assertTrue(np.all(self.view(arena,'text_expanded')==-91.))

    def test_excess_duration_rejects_before_either_expansion(self):
        arena=self.arena()
        for name,value in (('duration_prosody.duration_head.weight',0.),
                           ('duration_prosody.duration_head.bias',1.)):
            weight=next(item for item in self.layout['memory']['weights']['entries']
                        if item['name']==name)
            np.ndarray((weight['size']//4,),np.float32,buffer=arena,
                       offset=weight['abs_offset'])[:]=value
        self.view(arena,'duration_expanded')[:]=-97.
        frames=ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        self.assertEqual(frames.value,-999)
        self.assertEqual(int(self.view(arena,'runtime_valid_extent',np.int32)[0]),-999)
        self.assertTrue(np.all(self.view(arena,'duration_expanded')==-97.))
        self.assertTrue(np.all(self.view(arena,'text_expanded')==-91.))

    def test_selected_contracts_and_bound_dimensions(self):
        providers={op['function'] for op in self.calls['operations']}
        self.assertIn('audio_conv1d_checked_channel_major_f32',providers)
        self.assertIn('transpose_strided_f32_checked',providers)
        self.assertIn('leaky_relu_strided_f32_checked',providers)
        self.assertIn('audio_lstm_bidirectional_scan_f32',providers)
        for op in self.calls['operations']:
            if op['function'] in ('audio_conv1d_checked_channel_major_f32',
                    'transpose_strided_f32_checked','leaky_relu_strided_f32_checked',
                    'audio_duration_expand_token_major_f32'):
                for point in op.get('semantic_checkpoints',[]):
                    self.assertNotEqual(point['resolved_contract_id'],'unresolved',point)
        conv=next(op for op in self.calls['operations'] if
            op['function']=='audio_conv1d_checked_channel_major_f32')
        args={arg['name']:arg for arg in conv['args']}
        self.assertEqual(args['weight']['weight_ref'],
                         'acoustic_text_encoder.conv0.weight')

    def test_wrong_channel_and_weight_capacity_claims_fail_lowering(self):
        registry=build_ir_v8.load_kernel_registry()
        for field,value in (('conv_output_channels',513),
                            ('conv_weight_elements',512*512*5+1)):
            source=copy.deepcopy(self.source)
            circuit=source['template']
            op=circuit['block_types']['text_encoder']['body']['ops'][0]
            op['params']['call_constants'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(
                    RuntimeError,'HARD CALL CONSTANT FAULT'):
                with contextlib.redirect_stdout(io.StringIO()):
                    ir=build_ir_v8.build_ir1_direct(source,None,mode='prefill')
                    lower1=build_ir_v8.generate_ir_lower_1(
                        ir,registry,source,'prefill')
                    layout=build_ir_v8.generate_memory_layout(
                        lower1,source,registry,mode='prefill',context_len=36)
                    build_ir_v8.generate_ir_lower_2(
                        lower1,layout,source,registry,mode='prefill')

    def test_xray_selected_provider_and_valid_region(self):
        arena=self.arena();frames=ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        self.assertEqual(frames.value,103)
        expected={
            'text_conv0':(self.text['conv0_output'],(512,36),(512,40),
                          CHECKPOINT_ATOL['text_conv0']),
            'text_encoder_features':(self.text['lstm_output_token_major'],
                                     (36,512),(36,512),3e-6),
            'text_expanded':(self.text['text_expanded'],
                             (512,103),(512,128),3e-6)}
        tensor_report={'torch':{'tensors':{}},'comparisons':{}}
        points={}
        for name,(oracle,valid,physical,_limit) in expected.items():
            point=next(item for op in self.calls['operations']
                       for item in op.get('semantic_checkpoints',[])
                       if item['tensor']==name)
            self.assertNotEqual(point['resolved_contract_id'],'unresolved')
            points[name]=point
            selector=name if point['layer']<0 else f'{name}@{point["layer"]}'
            native_path=self.root/f'{name}-native.f32'
            oracle_path=self.root/f'{name}-pytorch.f32'
            self.view(arena,name).tofile(native_path)
            oracle.tofile(oracle_path)
            tensor_report['torch']['tensors'][selector]={
                'path':str(oracle_path),'shape':list(valid)}
            tensor_report['comparisons'][selector]={
                'ck_path':str(native_path),'shape':list(valid),
                'physical_shape':list(physical),
                'capacity_shape':list(physical),'valid_shape':list(valid),
                'physical_strides':[physical[1],1]}
        builder=xray_support.xray_builder
        runtime=builder.capture_runtime_library_identity(
            self.loaded,'ck_kokoro_text_encoder_bounded')
        subject=builder.build_manifest(backend='ck',call_ir=self.calls,
            tensor_report=tensor_report,model='kokoro_text_encoder_bounded',
            source='generated_native',phase='prefill',
            loaded_library=self.library,runtime_library=runtime)
        oracle_manifest=builder.build_manifest(backend='pytorch',
            call_ir=self.calls,tensor_report=tensor_report,
            model='kokoro_text_encoder_bounded',source='pinned_capture',
            phase='prefill')
        for name,(reference,valid,physical,limit) in expected.items():
            point=points[name]
            profile={'schema':'cke.parity_profile','schema_version':1,
                'name':'kokoro_text_encoder','backend':'pytorch',
                'contract_schema_version':1,
                'required_match_fields':['checkpoint_id','producer',
                    'logical_layout','axis_names','resolved_contract_id',
                    'kernel_id','function'],
                'observed_storage':{'default':'fp32','checkpoints':{}},
                'dtype_thresholds':{'fp32':{'cosine_min':0.99999,
                    'rmse_max':limit,'relative_rmse_max':limit,
                    'max_abs_max':limit,'finite_required':True}},
                'checkpoint_order':[point['id']],
                'interval_expansions':{},'backend_mappings':{}}
            one=lambda manifest:{**manifest,'checkpoints':[item for item in
                manifest['checkpoints'] if item['checkpoint_id']==point['id']]}
            report=xray_support.xray.compare_manifests(
                one(subject),one(oracle_manifest),profile)
            self.assertEqual(report['status'],'pass',(name,report))
            (self.root/f'{name}-xray.json').write_text(json.dumps(report,indent=2))
        self.assertEqual(subject['run']['artifact_library']['sha256'],
                         subject['run']['runtime_library']['sha256'])
        (self.root/'native-checkpoints.json').write_text(json.dumps(subject,indent=2))

    def test_wrong_text_edge_is_detected_by_independent_capture(self):
        graph=copy.deepcopy(self.source['template'])
        ops=graph['block_types']['text_encoder']['body']['ops']
        wrong=next(op for op in ops if op['id']=='text_conv1')
        wrong['graph_slots']['inputs']['input']='text_features'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'wrong-edge.json'
            path.write_text(json.dumps(graph))
            source=copy.deepcopy(self.source);source['template']=graph
            layout,calls,library,loaded,fn=compile_native_graph(root,source,path)
            fn.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_int32)]
            arena=populated_duration_arena(layout,self.entries,self.bump,
                                           self.encoder,self.duration)
            frames=ctypes.c_int32(-999)
            self.assertEqual(fn(arena,len(arena),ctypes.byref(frames)),0)
            self.assertEqual(frames.value,103)
            buffers={item['name']:item for item in
                layout['memory']['activations']['buffers']}
            item=buffers['text_expanded']
            changed=np.ndarray((512,128),np.float32,buffer=arena,
                offset=item['abs_offset'])[:,:103]
            self.assertGreater(float(np.max(np.abs(
                changed-self.text['text_expanded']))),1e-3)


if __name__=='__main__':unittest.main()
