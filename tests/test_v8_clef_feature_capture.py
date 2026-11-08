"""Capture contracts; synthetic tests never count as actual-backbone evidence."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from version.v8.decision.capture_clef_features import compact_inputs,strict_encode,verify_lock,file_sha,IDENTITY

class FeatureCaptureTest(unittest.TestCase):
    def encoded(self):
        return SimpleNamespace(input_ids=(7,2,7,11,2,3),questions=[SimpleNamespace(question_id='mode',question_type=1,question_span=(0,2),option_spans=((1,3),(2,5)),option_ids=('a','b')),SimpleNamespace(question_id='confirm',question_type=0,question_span=(3,6),option_spans=((5,6),),option_ids=('yes',))])
    def test_compact_rows_preserve_full_head_logits(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch unavailable: compaction oracle NOT_TESTED')
        from tests.test_v8_clef_flash_complete_head import synthetic_weights,reference
        raw=synthetic_weights();encoded=self.encoded();hidden=np.random.default_rng(5).normal(0,.2,(6,8)).astype('f4')
        ids=np.array(encoded.input_ids,dtype='i4');unique=np.unique(ids);rows=raw['output_embedding.weight'][unique]
        inputs=compact_inputs(encoded,hidden,rows,unique)
        original=reference(hidden,ids,inputs['question_spans'],inputs['option_spans'],inputs['option_counts'],inputs['question_type_ids'],raw)
        compact=dict(raw,**{'output_embedding.weight':rows})
        remapped=reference(hidden,inputs['input_ids'],inputs['question_spans'],inputs['option_spans'],inputs['option_counts'],inputs['question_type_ids'],compact)
        for name in original:np.testing.assert_array_equal(original[name],remapped[name])
        np.testing.assert_array_equal(unique[inputs['input_ids']],ids)
    def test_reject_invalid_row_and_span_contracts(self):
        e=self.encoded();hidden=np.ones((6,8),dtype='f4');ids=np.unique(e.input_ids);rows=np.ones((len(ids),8),dtype='f4')
        with self.assertRaisesRegex(ValueError,'identities'):compact_inputs(e,hidden,rows,ids[::-1])
        bad=hidden.copy();bad[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):compact_inputs(e,bad,rows,ids)
        bad=copy.deepcopy(e);bad.questions[0].option_spans=((1,7),)
        with self.assertRaisesRegex(ValueError,'span'):compact_inputs(bad,hidden,rows,ids)
        with self.assertRaisesRegex(ValueError,'shape'):compact_inputs(e,hidden[:-1],rows,ids)
    def test_pinned_encoder_never_silently_truncates(self):
        source=os.environ.get('CKE_CLEF_FLASH_REFERENCE_SOURCE')
        if not source:self.skipTest('pinned reference source unavailable: encoder test NOT_TESTED')
        self.assertEqual(file_sha(source),json.loads(IDENTITY.read_text())['joint_schema_model_sha256'])
        spec=importlib.util.spec_from_file_location('clef_capture_test_reference',source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        class CharacterTokenizer:
            def __call__(self,text,**kwargs):return SimpleNamespace(input_ids=[ord(x) for x in text])
        tokenizer=CharacterTokenizer();record=dict(id='test',state='A'*400,questions={'mode':dict(type='choice',instructions='Select',criteria={'a':'read','b':'wait'})})
        full=strict_encode(module,tokenizer,record,4096)
        self.assertEqual(strict_encode(module,tokenizer,record,len(full.input_ids)),full)
        # Demonstrate that the upstream encoder accepts and truncates this case.
        shorter=module.encode_record(tokenizer,record,max_length=len(full.input_ids)-1)
        self.assertNotEqual(shorter.input_ids,full.input_ids)
        with self.assertRaisesRegex(ValueError,'truncate'):strict_encode(module,tokenizer,record,len(full.input_ids)-1)
        with self.assertRaisesRegex(ValueError,'text-only'):strict_encode(module,tokenizer,dict(record,images=['image']),4096)
        bad=copy.deepcopy(record);bad['questions']['mode']['criteria']={'read all':'description'}
        with self.assertRaisesRegex(ValueError,'candidate identity'):strict_encode(module,tokenizer,bad,4096)
    def test_capture_loader_rejects_wrong_mapping_and_reference_shape(self):
        from version.v8.decision.replay_clef_features import load_capture
        # Synthetic serialization fixture, not evidence of backbone execution.
        pin=json.loads(IDENTITY.read_text());e=self.encoded();hidden=np.ones((6,4096),dtype='f4');tokens=np.array(e.input_ids);vocab=np.unique(tokens);rows=np.ones((len(vocab),4096),dtype='f4')
        inputs=compact_inputs(e,hidden,rows,vocab)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);np.savez(p/'features.npz',**inputs);np.save(p/'embedding.npy',rows);np.save(p/'original_token_ids.npy',tokens);np.save(p/'embedding_token_ids.npy',vocab)
            refs={f'fields{i}_output':np.zeros((2,1024),dtype='f4') for i in range(4)};refs.update(final_fields=np.zeros((2,1024),dtype='f4'),option_logits=np.zeros((3,1),dtype='f4'));np.savez(p/'reference.npz',**refs)
            m=dict(schema='cke.clef.feature-capture.v1',model=pin['model'],revision=pin['revision'],reference_source_sha256=pin['joint_schema_model_sha256'],head_sha256=pin['joint_head_weights_sha256'],backbone_executor='pytorch_reference',backbone_dtype='bf16',head_dtype='fp32',tokens=6,questions=[dict(id=q.question_id,type=q.question_type,span=list(q.question_span),option_spans=[list(s) for s in q.option_spans],candidate_ids=list(q.option_ids)) for q in e.questions])
            names=['features.npz','embedding.npy','original_token_ids.npy','embedding_token_ids.npy','reference.npz']
            def save():
                m['files']={name:file_sha(p/name) for name in names};(p/'capture.json').write_text(json.dumps(m));return file_sha(p/'capture.json')
            identity=save();load_capture(p,identity)
            with self.assertRaisesRegex(ValueError,'identity'):load_capture(p,'0'*64)
            ids=inputs['input_ids'].copy();inputs['input_ids'][0]=999;np.savez(p/'features.npz',**inputs)
            with self.assertRaisesRegex(ValueError,'file mismatch'):load_capture(p,identity)
            with self.assertRaisesRegex(ValueError,'remapping'):load_capture(p,save())
            inputs['input_ids']=ids;np.savez(p/'features.npz',**inputs)
            refs['option_logits']=np.zeros((300,1),dtype='f4');np.savez(p/'reference.npz',**refs)
            with self.assertRaisesRegex(ValueError,'reference shape'):load_capture(p,save())

    def test_lock_requires_every_model_shard_before_execution(self):
        identity=json.loads(IDENTITY.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);files={}
            for name in ('config.json','joint_head_config.json','joint_schema_model.py','joint_head.safetensors','tokenizer.json','tokenizer_config.json'):
                (folder/name).write_bytes(b'fixture');files[name]=file_sha(folder/name)
            (folder/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'layer':'model-1.safetensors'}}));files['model.safetensors.index.json']=file_sha(folder/'model.safetensors.index.json')
            lock=dict(model=identity['model'],revision=identity['revision'],files=files);p=folder/'lock.json'
            def save():p.write_text(json.dumps(lock));return file_sha(p)
            with self.assertRaisesRegex(ValueError,'unlocked'):verify_lock(folder,p,save())
            (folder/'model-1.safetensors').write_bytes(b'weight');files['model-1.safetensors']=file_sha(folder/'model-1.safetensors')
            self.assertEqual(verify_lock(folder,p,save()),lock)
            (folder/'model-1.safetensors').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'file mismatch'):verify_lock(folder,p,file_sha(p))
            with self.assertRaisesRegex(ValueError,'lock identity'):verify_lock(folder,p,'0'*64)

class RealBackboneCaptureTest(unittest.TestCase):
    def test_local_released_backbone_then_generated_head(self):
        from version.v8.decision.capture_clef_features import capture
        from version.v8.decision.replay_clef_features import replay
        folder=os.environ.get('CKE_CLEF_RELEASE_PATH');lock=os.environ.get('CKE_CLEF_RELEASE_LOCK');sha=os.environ.get('CKE_CLEF_RELEASE_LOCK_SHA256')
        if not all((folder,lock,sha)):self.skipTest('released backbone/lock unavailable: real-feature parity NOT_TESTED')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);request=p/'request.json'
            request.write_text(json.dumps(dict(id='accessibility-review',state='Two checks failed. The user asks to hear the failures first.',questions={'reading':dict(type='choice',instructions='What should be read first?',criteria={'all':'Complete summary','failures':'Failed checks','nothing':'Remain silent'}),'approval':dict(type='noul',instructions='Has the user asked to hear the failures?')})))
            capture(folder,lock,sha,request,p/'capture',max_tokens=512,threads=4)
            report=replay(p/'capture',file_sha(p/'capture/capture.json'),Path(folder)/'joint_head.safetensors',p/'replay')
            self.assertEqual(report['execution'],'PASS');self.assertEqual(report['numerical'],'PASS')
            self.assertEqual(report['scope'],'head_only')

if __name__=='__main__':unittest.main()
