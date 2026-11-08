"""Complete generated decision head; supplied backbone features, not full model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from tests.test_v8_clef_flash_head_ingress import compile_case
from tests.test_v8_clef_flash_evidence0 import prepare, torch_layer
from tests.test_v8_clef_flash_evidence_complete import both_synthetic_weights

# Independent FP32 trajectory bounds; logits retain a separate tighter gate.
INTERMEDIATE_ATOL = 1e-5
LOGIT_ATOL = 2e-6

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'version/v8/decision'))
from build_clef_flash_complete_head_circuit import OUTPUT,build_circuit


from version.v8.decision.package_clef_head import effective_weights


def synthetic_weights():
    import torch
    raw=both_synthetic_weights(8,8,16,16)
    rng=np.random.default_rng(707)
    for stem in ('option_summary_norm','field_norm','option_norm'):
        raw[stem+'.weight']=rng.uniform(.7,1.3,8).astype('f4')
        raw[stem+'.bias']=rng.normal(0,.05,8).astype('f4')
    raw['type_embedding.weight']=rng.normal(0,.2,(3,8)).astype('f4')
    for stem,shape in [('residual_scorer.0',(8,32)),('residual_scorer.3',(1,8))]:
        raw[stem+'.weight']=rng.normal(0,.1,shape).astype('f4')
        raw[stem+'.bias']=rng.normal(0,.05,shape[0]).astype('f4')
    for k,v in [('prior_logit_scale',.4),('joint_logit_scale',.7),('residual_gate',-.8)]:
        raw[k]=np.array([v],dtype='f4')
    torch.manual_seed(611)
    for i in range(4):
        layer=torch.nn.TransformerDecoderLayer(8,2,16,dropout=0,activation='gelu',
                                               batch_first=True,norm_first=True)
        raw.update({f'layers.{i}.{k}':v.detach().numpy().copy() for k,v in layer.state_dict().items()})
    return raw


def reference(hidden, ids, questions, spans, counts, types, raw):
    import torch
    import torch.nn.functional as F
    t=lambda k:torch.from_numpy(raw[k])
    norm=lambda x,k:F.layer_norm(x,(x.shape[-1],),t(k+'.weight'),t(k+'.bias'),eps=1e-5)
    proj=lambda x,k:F.linear(x,t(k+'.weight'))
    x=norm(torch.from_numpy(hidden),'hidden_norm')
    q=torch.stack([x[a:b].mean(0) for a,b in questions]); glob=x[-1]
    lex=torch.stack([t('output_embedding.weight')[ids[a:b]].mean(0) for a,b in spans])
    ctx=torch.stack([x[a:b].mean(0) for a,b in spans])
    owners=torch.repeat_interleave(torch.arange(len(counts)),torch.tensor(counts))
    memory=proj(x,'memory_projection')
    routed=proj(ctx,'option_context_projection')+proj(lex,'option_lexical_projection')+proj(q,'option_question_projection')[owners]
    captures={}
    for i in range(2):
        routed=torch_layer(routed.detach().numpy(),memory.detach().numpy(),raw,2,
                           prefix=f'evidence_layers.{i}')['evidence0_output']
    base=proj(q,'question_projection'); summaries=[]; start=0
    for i,n in enumerate(counts):
        rows=routed[start:start+n]
        summaries.append((torch.softmax(rows@base[i]/(8**.5),0)[:,None]*rows).sum(0)); start+=n
    fields=base+norm(torch.stack(summaries),'option_summary_norm')+proj(glob,'global_projection')+t('type_embedding.weight')[types]
    captures['fields_input']=fields
    for i in range(4):
        layer=torch.nn.TransformerDecoderLayer(8,2,16,dropout=0,activation='gelu',batch_first=True,norm_first=True)
        layer.load_state_dict({k[len(f'layers.{i}.'):]:torch.from_numpy(v) for k,v in raw.items() if k.startswith(f'layers.{i}.')})
        layer.eval();fields=layer(fields[None],memory[None])[0]
        captures[f'fields{i}_output']=fields
    fields=norm(fields,'field_norm'); opts=norm(routed,'option_norm'); paired=fields[owners]
    anchor=F.normalize(q+glob,dim=-1); prior_cos=(F.normalize(lex,dim=-1)*anchor[owners]).sum(-1)
    cosine=F.cosine_similarity(paired,opts,dim=-1)
    features=torch.cat((paired,opts,paired*opts,(paired-opts).abs()),-1)
    residual=F.linear(F.gelu(F.linear(features,t('residual_scorer.0.weight'),t('residual_scorer.0.bias'))),t('residual_scorer.3.weight'),t('residual_scorer.3.bias'))[:,0]
    scores=t('prior_logit_scale').clamp(max=np.log(100.)).exp()*prior_cos+torch.sigmoid(t('residual_gate'))*(t('joint_logit_scale').clamp(max=np.log(100.)).exp()*cosine+residual)
    captures.update(final_fields=fields,final_options=opts,scorer_features=features,
                    prior_cosine=prior_cos,joint_cosine=cosine,option_logits=scores[:,None])
    return {k:v.detach().numpy() for k,v in captures.items()}


class CompleteHeadTest(unittest.TestCase):
    def test_regeneration(self):
        self.assertEqual(OUTPUT.read_text(),json.dumps(build_circuit(),indent=2)+'\n')
        for kwargs in ({'heads':0},{'feedforward':0},{'tokens':0}):
            with self.assertRaises(ValueError):build_circuit(**kwargs)

    def test_synthetic_full_head_and_recovery(self):
        try: import torch
        except ImportError:self.skipTest('PyTorch oracle unavailable')
        raw=synthetic_weights()
        for tokens,counts in ((6,[2,1]),(8,[1,2])):
            with self.subTest(tokens=tokens),tempfile.TemporaryDirectory() as temp:
                graph=build_circuit(tokens=tokens,channels=8,width=8,heads=2,feedforward=16)
                arena,execute,view,calls,_=compile_case(Path(temp),graph,effective_weights(graph,raw))
                self.assertEqual(calls['errors'],[])
                hidden=np.random.default_rng(tokens+40).normal(0,.4,(tokens,8)).astype('f4')
                prepare(view,hidden);view('option_counts')[:]=counts;view('question_type_ids')[:]=[0,2]
                self.assertEqual(execute(arena,len(arena)),0)
                expected=reference(hidden,view('input_ids').copy().astype('int64'),view('question_spans').copy(),
                    view('option_spans').copy(),counts,[0,2],raw)
                for name,value in expected.items():
                    np.testing.assert_allclose(view(name),value,atol=1e-5,rtol=0,err_msg=name)
                baseline=view('option_logits').copy()
                # Question permutation carries its option block and type identity.
                original_spans=view('option_spans').copy()
                original_questions=view('question_spans').copy()
                order=list(range(counts[0],3))+list(range(counts[0]))
                view('question_spans')[:]=original_questions[::-1]
                view('option_spans')[:]=original_spans[order]
                view('option_counts')[:]=counts[::-1]
                view('question_type_ids')[:]=[2,0]
                self.assertEqual(execute(arena,len(arena)),0)
                np.testing.assert_allclose(view('option_logits'),baseline[order],atol=1e-5,rtol=0)
                prepare(view,hidden);view('option_counts')[:]=counts;view('question_type_ids')[:]=[0,2]
                for bad in ('input_ids','question_spans','option_counts','hidden'):
                    saved=view(bad).copy();view('option_logits')[:]=-901
                    view(bad).flat[0]=np.nan if bad=='hidden' else -1
                    self.assertNotEqual(execute(arena,len(arena)),0)
                    np.testing.assert_array_equal(view('option_logits'),-901)
                    view(bad)[:]=saved
                self.assertEqual(execute(arena,len(arena)),0)
                np.testing.assert_array_equal(view('option_logits'),baseline)
                # Corrupt a finite scoring weight: the independent output gate must detect it.
                layout=json.loads((Path(temp)/'layout.json').read_text())
                entry=next(x for x in layout['memory']['weights']['entries'] if x['name']=='residual_scorer.3.bias')
                weight=np.ndarray((1,),dtype='f4',buffer=arena,offset=entry['abs_offset'])
                saved=weight.copy();weight[:]+=1
                self.assertEqual(execute(arena,len(arena)),0)
                self.assertGreater(float(np.max(np.abs(view('option_logits')-baseline))),.1)
                weight[:]=saved
                gate_entry=next(x for x in layout['memory']['weights']['entries'] if x['name']=='residual_gate')
                gate=np.ndarray((1,),dtype='f4',buffer=arena,offset=gate_entry['abs_offset'])
                saved_gate=gate.copy();gate[:]=np.nan;view('option_logits')[:]=-901
                self.assertNotEqual(execute(arena,len(arena)),0)
                np.testing.assert_array_equal(view('option_logits'),-901)
                gate[:]=saved_gate
                self.assertEqual(execute(arena,len(arena)),0)
                np.testing.assert_array_equal(view('option_logits'),baseline)
                view('question_type_ids')[:]=[3,0];view('option_logits')[:]=-901
                self.assertNotEqual(execute(arena,len(arena)),0)
                np.testing.assert_array_equal(view('option_logits'),-901)
                view('question_type_ids')[:]=[0,2]
                self.assertEqual(execute(arena,len(arena)),0)
                np.testing.assert_array_equal(view('option_logits'),baseline)
                prepare(view,hidden+.2);view('option_counts')[:]=counts
                self.assertEqual(execute(arena,len(arena)),0)
                self.assertFalse(np.array_equal(view('option_logits'),baseline))
                prepare(view,hidden);view('option_counts')[:]=counts
                self.assertEqual(execute(arena,len(arena)),0)
                np.testing.assert_array_equal(view('option_logits'),baseline)
                self.assertNotEqual(execute(arena,len(arena)-1),0)
                np.testing.assert_array_equal(view('option_logits'),baseline)

    def test_direct_pinned_complete_head(self):
        source=os.environ.get('CKE_CLEF_FLASH_REFERENCE_SOURCE');asset=os.environ.get('CKE_CLEF_FLASH_HEAD_PATH')
        if not source or not asset:self.skipTest('pinned head/reference assets unavailable')
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError:self.skipTest('PyTorch/safetensors unavailable')
        identity=json.loads((ROOT/'tests/fixtures/decision/clef_flash/identity.json').read_text())
        for file,key in ((source,'joint_schema_model_sha256'),(asset,'joint_head_weights_sha256')):
            self.assertEqual(hashlib.sha256(Path(file).read_bytes()).hexdigest(),identity[key])
        spec=importlib.util.spec_from_file_location('clef_complete_reference',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        state=load_file(asset)
        head=module.JointSchemaHead(hidden_size=4096,width=1024,routing_layers=2,layers=4,heads=16,feedforward=4096,dropout=0)
        head.load_state_dict(state,strict=True);head.eval()
        raw={k:v.float().numpy() for k,v in state.items()}
        raw['output_embedding.weight']=np.random.default_rng(113).normal(0,.02,(16,4096)).astype('f4')
        captures={};handles=[]
        def hook(name):
            def f(_module,_args,output):captures[name]=output.detach().numpy().squeeze(0) if name.startswith('fields') else output.detach().numpy()
            return f
        for i,layer in enumerate(head.layers):handles.append(layer.register_forward_hook(hook(f'fields{i}_output')))
        handles.append(head.field_norm.register_forward_hook(hook('final_fields')))
        try:
            for tokens,counts in ((6,[2,1]),(8,[1,2])):
                with self.subTest(tokens=tokens),tempfile.TemporaryDirectory() as temp:
                    graph=build_circuit(tokens=tokens)
                    arena,execute,view,calls,_=compile_case(Path(temp),graph,effective_weights(graph,raw))
                    hidden=np.random.default_rng(42+tokens).normal(0,.2,(tokens,4096)).astype('f4')
                    prepare(view,hidden);view('option_counts')[:]=counts;view('question_type_ids')[:]=[0,2]
                    self.assertEqual(execute(arena,len(arena)),0)
                    spans=[tuple(map(int,x)) for x in view('option_spans')];questions=[];start=0
                    for q,n in enumerate(counts):
                        questions.append(SimpleNamespace(question_span=tuple(map(int,view('question_spans')[q])),question_type=[0,2][q],option_spans=spans[start:start+n]));start+=n
                    with torch.no_grad():
                        logits=head(torch.from_numpy(hidden)[None],torch.from_numpy(view('input_ids').copy().astype('int64'))[None],
                            torch.ones((1,tokens),dtype=torch.int64),[SimpleNamespace(questions=questions)],torch.from_numpy(raw['output_embedding.weight']))[0]
                    captures['option_logits']=torch.cat(logits).numpy()[:,None]
                    for name,expected in captures.items():
                        actual=view(name);err=np.abs(actual.astype('f8')-expected)
                        print(f'complete head tokens={tokens} {name}: max_abs={err.max():.9g} rmse={np.sqrt(np.mean(err**2)):.9g}')
                        np.testing.assert_allclose(actual,expected,atol=LOGIT_ATOL if name=='option_logits' else INTERMEDIATE_ATOL,rtol=0,err_msg=name)
        finally:
            for handle in handles:handle.remove()

if __name__=='__main__':unittest.main()
