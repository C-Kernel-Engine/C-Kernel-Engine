"""Connected pinned phonemes/style to duration logits through one generated entry."""
import ctypes
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tests.v8_checked_graph_test_support import compile_native_graph
from tests.v8_checked_graph_test_support import build_ir_v8
from tests.v8_kokoro_stage_circuit_support import materialize_stage_circuit
from tests.v8_kokoro_duration_fixture_support import (
    prepare_duration_fixture, populated_duration_arena)
from tests import test_v8_kokoro_generated_albert_layer as first_layer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/tts'))
import build_kokoro_duration_circuit as author


class KokoroGeneratedDurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence=os.environ.get('CKE_KOKORO_DURATION_EVIDENCE_DIR')
        if evidence:
            root=Path(evidence).resolve()/'duration'
            root.mkdir(parents=True,exist_ok=True)
            cls.temp=SimpleNamespace(name=str(root),cleanup=lambda:None)
        else:
            cls.temp=tempfile.TemporaryDirectory()
            root=Path(cls.temp.name)
        circuit = materialize_stage_circuit(root, author)
        fixture = prepare_duration_fixture(root, circuit)
        for key, value in vars(fixture).items():
            setattr(cls, key, value)
        cls.layout,cls.calls,cls.library,cls.loaded,cls.fn = compile_native_graph(root,cls.source,circuit)
        cls.root = root
        cls.buffers = {item['name']:item for item in cls.layout['memory']['activations']['buffers']}
        cls.planned_weights = {item['name']:item for item in cls.layout['memory']['weights']['entries']}
        cls.fn.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def arena(self):
        return populated_duration_arena(self.layout, self.entries, self.bump,
                                        self.encoder, self.duration)

    def view(self, arena, name, dtype, shape):
        return np.ndarray(shape,dtype,buffer=arena,offset=self.buffers[name]['abs_offset'])

    def primitive_abis(self):
        """Load the same production providers outside generated model execution."""
        exports=self.root/'primitive-exports.map'
        exports.write_text('{ global: audio_lstm_bidirectional_scan_f32; audio_adaptive_layer_norm_f32; local: *; };\n')
        library=self.root/'primitive-predictor.so'
        if not library.exists():
            subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror',
                '-shared','-fPIC','-ffp-contract=off','-ffunction-sections','-fdata-sections',
                '-Wl,--gc-sections','-Wl,--no-undefined','-Wl,--version-script='+str(exports),
                str(ROOT/'src/kernels/audio_lstm_scan.c'),
                str(ROOT/'src/kernels/audio_kernels.c'),
                str(ROOT/'src/kernels/audio_adaptive_layer_norm.c'),
                '-I',str(ROOT/'include'),'-lm','-o',str(library)],
                check=True,capture_output=True,text=True)
        loaded=ctypes.CDLL(str(library)); pointer=ctypes.POINTER(ctypes.c_float)
        scan=loaded.audio_lstm_bidirectional_scan_f32
        scan.argtypes=[pointer,ctypes.c_size_t]*9+[ctypes.c_int]*3+[ctypes.c_size_t]*2
        scan.restype=ctypes.c_int
        norm=loaded.audio_adaptive_layer_norm_f32
        norm.argtypes=[pointer,ctypes.c_size_t]*6+[ctypes.c_int]*3+[ctypes.c_size_t]*2+[ctypes.c_float]
        norm.restype=ctypes.c_int
        return scan,norm,lambda x:x.ctypes.data_as(pointer)

    def check_sensitive_stage_propagation(self, arena):
        """Separate local provider error from upstream-input amplification."""
        scan,norm,ptr=self.primitive_abis()
        results={}
        for kind in ('scan0','norm0'):
            input_name=kind+'_input' if kind=='scan0' else 'scan0_output'
            expected_input=self.duration[input_name]
            connected_input=self.view(arena,input_name,np.float32,expected_input.shape)
            expected_output=self.duration[kind+'_output']
            generated_output=self.view(arena,kind+'_output',np.float32,expected_output.shape)
            prefix=('duration_prosody.text_encoder.scan0' if kind=='scan0' else
                    'duration_prosody.text_encoder.norm0')
            def replay(source):
                output=np.full(expected_output.shape,-777.,np.float32)
                if kind=='scan0':
                    weights=[self.duration['weight__'+prefix+'__'+part] for part in
                             ('weight_ih','weight_hh','bias_ih','bias_hh')]
                    hidden=np.full(512,13.,np.float32); cell=np.full(512,17.,np.float32)
                    gates=np.full(1024,19.,np.float32)
                    args=[ptr(source),source.size]
                    for weight in weights: args.extend((ptr(weight),weight.size))
                    args.extend((ptr(output),output.size,ptr(hidden),hidden.size,
                                 ptr(cell),cell.size,ptr(gates),gates.nbytes,
                                 36,640,256,640,512))
                    self.assertEqual(scan(*args),0,kind)
                else:
                    style=self.duration['predictor_style']
                    weight=self.duration['weight__'+prefix+'__projection_weight']
                    bias=self.duration['weight__'+prefix+'__projection_bias']
                    scratch=np.zeros(1024,np.float32)
                    self.assertEqual(norm(ptr(source),source.size,ptr(style),style.size,
                        ptr(weight),weight.size,ptr(bias),bias.size,ptr(output),output.size,
                        ptr(scratch),scratch.nbytes,36,512,128,512,512,1e-5),0)
                self.assertTrue(np.isfinite(output).all(),kind)
                return output
            from_oracle=replay(expected_input)
            from_connected=replay(connected_input)
            np.testing.assert_array_equal(from_connected,generated_output)
            local=np.abs(from_oracle.astype(np.float64)-expected_output.astype(np.float64))
            propagated=np.abs(from_connected.astype(np.float64)-from_oracle.astype(np.float64))
            input_error=np.abs(connected_input.astype(np.float64)-expected_input.astype(np.float64))
            self.assertLessEqual(float(local.max()),3e-6,kind)
            # The first scan may amplify a bounded encoder perturbation before
            # it becomes the normalization input; keep those limits distinct.
            self.assertLessEqual(float(input_error.max()),
                                 3e-5 if kind=='scan0' else 4e-5,kind)
            self.assertLessEqual(float(propagated.max()),7e-5,kind)
            results[kind]={'local_provider_max_abs':float(local.max()),
                           'upstream_input_max_abs':float(input_error.max()),
                           'propagated_max_abs':float(propagated.max()),
                           'generated_matches_direct_provider':'exact'}
        (self.root/'duration-propagation.json').write_text(json.dumps(results,indent=2))
        print('KOKORO_DURATION_PROPAGATION '+json.dumps(results,sort_keys=True))
        return results

    def test_connected_predictor_stages_and_exact_durations(self):
        self.assertEqual(self.calls['errors'],[])
        self.assertEqual(len(self.calls['operations']),160)
        arena=self.arena(); frames=ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        self.check_sensitive_stage_propagation(arena)
        report={}
        failures=[]
        pairs={'phoneme_features':'encoder_features',
               'head_scan_output':'head_scan_output',
               'duration_logits':'duration_logits'}
        for i in range(3):
            pairs[f'scan{i}_input']=f'scan{i}_input'
            pairs[f'scan{i}_output']=f'scan{i}_output'
            pairs[f'norm{i}_output']=f'norm{i}_output'
        pairs['predictor_features']='predictor_output'
        for name,oracle in pairs.items():
            expected=self.duration[oracle]
            actual=self.view(arena,name,np.float32,expected.shape)
            self.assertTrue(np.isfinite(actual).all(),name)
            error=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
            report[name]={'max_abs':float(error.max()),'rmse':float(np.sqrt(np.mean(error*error))),
                          'worst':list(map(int,np.unravel_index(error.argmax(),error.shape)))}
            connected_limit = (7.3e-5 if name in ('scan0_output','norm0_output','scan1_input')
                else 5e-5 if name=='duration_logits' else 3e-5)
            point=next(point for op in self.calls['operations']
                for point in op.get('semantic_checkpoints',[]) if point['tensor']==name)
            print('CKE_NUMERICAL_CASE '+json.dumps({
                'case_id':f'kokoro.duration.{name}.connected-vs-pytorch28',
                'name':name,'provider':point['kernel_id'],'dtype':'fp32','direction':'inference',
                'oracle':'pytorch','backend_version':self.duration_meta['dependencies']['torch'],
                'configuration':'tokens=36; af_heart row=33; connected generated graph',
                'evidence_kind':'numerical','max_diff':report[name]['max_abs'],
                'tolerance':connected_limit,'worst_sample':report[name]['worst'],
                'rmse':report[name]['rmse'],
                'status':'pass' if np.all(error <= connected_limit) else 'fail',
                'reproduction_command':'python3 -m unittest '+self.id()},sort_keys=True))
            if not np.all(error <= connected_limit) or report[name]['rmse'] > 1e-5:
                failures.append((name,report[name],connected_limit))
        (self.root/'duration-errors.json').write_text(json.dumps(report,indent=2))
        print('KOKORO_DURATION_STAGE_ERRORS '+json.dumps(report,sort_keys=True))
        self.assertEqual(failures, [], failures)
        self.verify_xray(arena, pairs)
        np.testing.assert_array_equal(self.view(arena,'runtime_values',np.int32,(36,)),
                                      self.duration['durations'])
        self.assertEqual(frames.value,int(self.duration['expanded_frames'][0]))
        # The checked entry publishes this scalar via out_frames. The declared
        # arena slot is not a second output until lowering binds a consumer.
        self.assertEqual(int(self.duration['expanded_frames'][0]),103)

    def test_second_compiled_phoneme_geometry_executes_and_recovers(self):
        # This proves circuit specialization and generated execution at a
        # different input geometry. It is not a runtime-variable WAV entry or
        # an independent model oracle for this eight-token sequence.
        tokens = 8
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            circuit = root / 'duration-eight-tokens.json'
            circuit.write_text(json.dumps(author.build_circuit(tokens)))
            fixture = prepare_duration_fixture(root, circuit)
            layout, calls, _, _, fn = compile_native_graph(
                root, fixture.source, circuit)
            self.assertEqual(calls['errors'], [])
            fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_int32)]
            buffers = {item['name']: item for item in
                       layout['memory']['activations']['buffers']}
            ids = self.encoder['word_ids'][:tokens].copy()

            def run(word_ids):
                arena = populated_duration_arena(
                    layout, fixture.entries, fixture.bump, fixture.encoder,
                    fixture.duration, word_ids=word_ids)
                frames = ctypes.c_int32(-999)
                self.assertEqual(fn(arena, len(arena), ctypes.byref(frames)), 0)
                durations = np.ndarray((tokens,), np.int32, buffer=arena,
                    offset=buffers['runtime_values']['abs_offset']).copy()
                logits = np.ndarray((tokens, 50), np.float32, buffer=arena,
                    offset=buffers['duration_logits']['abs_offset']).copy()
                self.assertTrue(np.isfinite(logits).all())
                self.assertGreater(frames.value, 0)
                self.assertLessEqual(frames.value, 128)
                self.assertEqual(int(durations.sum(dtype=np.int64)), frames.value)
                return logits, durations, frames.value

            first = run(ids)
            alternate = ids.copy()
            alternate[1] = (int(alternate[1]) + 1) % 178
            second = run(alternate)
            self.assertFalse(np.array_equal(first[0], second[0]))
            third = run(ids)
            np.testing.assert_array_equal(first[0], third[0])
            np.testing.assert_array_equal(first[1], third[1])
            self.assertEqual(first[2], third[2])

            arena = populated_duration_arena(
                layout, fixture.entries, fixture.bump, fixture.encoder,
                fixture.duration, word_ids=ids)
            frames = ctypes.c_int32(-999)
            untouched = np.ndarray((tokens,), np.int32, buffer=arena,
                offset=buffers['runtime_values']['abs_offset'])
            self.assertNotEqual(fn(arena, len(arena)-1, ctypes.byref(frames)), 0)
            self.assertEqual(frames.value, -999)
            np.testing.assert_array_equal(untouched, np.full(tokens, -999, np.int32))

    def verify_xray(self, arena, pairs):
        builder, xray = first_layer.xray_builder, first_layer.xray
        identity=builder.capture_runtime_library_identity(self.loaded,'ck_kokoro_duration_predictor')
        tensor_report={'torch':{'tensors':{}},'comparisons':{}}
        for name,oracle in pairs.items():
            expected=self.duration[oracle]
            native=self.view(arena,name,np.float32,expected.shape)
            native_path=self.root/f'{name}-native.f32'
            oracle_path=self.root/f'{name}-pytorch.f32'
            native.tofile(native_path); expected.tofile(oracle_path)
            point=next(point for op in self.calls['operations']
                for point in op.get('semantic_checkpoints',[]) if point['tensor']==name)
            selector=name if point['layer']<0 else f'{name}@{point["layer"]}'
            tensor_report['torch']['tensors'][selector]={'path':str(oracle_path),'shape':list(expected.shape)}
            tensor_report['comparisons'][selector]={
                'ck_path':str(native_path),'shape':list(expected.shape),
                'capacity_shape':list(expected.shape),'valid_shape':list(expected.shape),
                'physical_strides':[expected.shape[1],1]}
        subject=builder.build_manifest(backend='ck',call_ir=self.calls,tensor_report=tensor_report,
            model='kokoro_duration_predictor_bounded',source='generated_native',phase='prefill',
            loaded_library=self.library,runtime_library=identity)
        expected=builder.build_manifest(backend='pytorch',call_ir=self.calls,tensor_report=tensor_report,
            model='kokoro_duration_predictor_bounded',source='pinned_capture',phase='prefill')
        reports=[]
        for op in self.calls['operations']:
            for point in op.get('semantic_checkpoints',[]):
                if point['tensor'] not in pairs:
                    continue
                self.assertNotEqual(point['resolved_contract_id'],'unresolved')
                limit=(7.3e-5 if point['tensor'] in ('scan0_output','norm0_output','scan1_input')
                    else 5e-5 if point['tensor']=='duration_logits' else 3e-5)
                profile={'schema':'cke.parity_profile','schema_version':1,
                    'name':'kokoro_connected_duration','backend':'pytorch','contract_schema_version':1,
                    'required_match_fields':['checkpoint_id','producer','logical_layout','axis_names',
                        'resolved_contract_id','kernel_id','function'],
                    'observed_storage':{'default':'fp32','checkpoints':{}},
                    'dtype_thresholds':{'fp32':{'cosine_min':0.99999,'rmse_max':1e-5,
                        'relative_rmse_max':1e-5,'max_abs_max':limit,'finite_required':True}},
                    'checkpoint_order':[point['id']],'interval_expansions':{},'backend_mappings':{}}
                one=lambda manifest:{**manifest,'checkpoints':[entry for entry in manifest['checkpoints']
                    if entry['checkpoint_id']==point['id']]}
                result=xray.compare_manifests(one(subject),one(expected),profile)
                self.assertEqual(result['status'],'pass',point['id'])
                reports.append(result)
        self.assertEqual(len(reports),len(pairs))
        for name,value in (('native-checkpoints.json',subject),('pytorch-checkpoints.json',expected),
                           ('xray-reports.json',reports)):
            (self.root/name).write_text(json.dumps(value,indent=2))

    def test_failed_producer_preserves_downstream_and_extent(self):
        arena=self.arena()
        style=self.view(arena,'predictor_style',np.float32,(128,))
        style[0]=np.nan
        expected=self.view(arena,'duration_logits',np.float32,(36,50)).copy()
        frames=ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        np.testing.assert_array_equal(self.view(arena,'duration_logits',np.float32,(36,50)),expected)
        self.assertEqual(frames.value,-999)
        self.assertEqual(self.view(arena,'runtime_valid_extent',np.int32,(1,))[0],-999)

    def test_capacity_rejection_preserves_durations_and_extent(self):
        arena=self.arena()
        bias=self.planned_weights['duration_prosody.duration_head.bias']
        np.ndarray((50,),np.float32,buffer=arena,offset=bias['abs_offset'])[:]=1000.
        durations=self.view(arena,'runtime_values',np.int32,(36,))
        frames=ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        np.testing.assert_array_equal(durations,np.full(36,-999,np.int32))
        self.assertEqual(frames.value,-999)

    def test_repeated_native_calls_and_small_arena(self):
        arena=self.arena(); frames=ctypes.c_int32(-999)
        self.assertNotEqual(self.fn(arena,len(arena)-1,ctypes.byref(frames)),0)
        self.assertEqual(frames.value,-999)
        self.assertEqual(self.view(arena,'runtime_values',np.int32,(36,))[0],-999)
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        first=self.view(arena,'duration_logits',np.float32,(36,50)).copy()
        self.assertEqual(frames.value,103)
        self.view(arena,'duration_logits',np.float32,(36,50))[:]=-777.
        self.view(arena,'runtime_values',np.int32,(36,))[:]=-999
        frames.value=-999
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        np.testing.assert_array_equal(self.view(arena,'duration_logits',np.float32,(36,50)),first)
        np.testing.assert_array_equal(self.view(arena,'runtime_values',np.int32,(36,)),self.duration['durations'])
        self.assertEqual(frames.value,103)

    def test_compiler_rejects_false_provider_capacity_claims(self):
        registry=build_ir_v8.load_kernel_registry()
        for operation, argument, bad_value in (
            ('duration_scan0','scan_input_elements',36*640+1),
            ('duration_norm0','adaln_output_elements',36*512+1)):
            with self.subTest(operation=operation):
                source=copy.deepcopy(self.source)
                ops=source['template']['block_types']['duration_predictor']['body']['ops']
                target=next(op for op in ops if op['id']==operation)
                target['params']['call_constants'][argument]=bad_value
                circuit=self.root/f'bad-{operation}.json'
                circuit.write_text(json.dumps(source['template']))
                ir1=build_ir_v8.build_ir1_direct(source,circuit,mode='prefill')
                lower1=build_ir_v8.generate_ir_lower_1(ir1,registry,source,'prefill')
                layout=build_ir_v8.generate_memory_layout(lower1,source,registry,
                    mode='prefill',context_len=36)
                with self.assertRaises((RuntimeError,ValueError)):
                    lower2=build_ir_v8.generate_ir_lower_2(lower1,layout,source,registry,mode='prefill')
                    build_ir_v8.generate_ir_lower_3(lower2,mode='prefill')

    def test_oracle_fed_production_scan_and_adaptive_norm(self):
        scan,norm,ptr=self.primitive_abis()
        results={}
        for index in range(4):
            name='head_scan' if index==3 else f'scan{index}'
            prefix=('duration_prosody.head_scan' if index==3 else
                    f'duration_prosody.text_encoder.scan{index}')
            source=self.duration[name+'_input']
            expected=self.duration[name+'_output']
            weights=[self.duration['weight__'+prefix+'__'+kind] for kind in
                     ('weight_ih','weight_hh','bias_ih','bias_hh')]
            output=np.full(expected.shape,-777.,np.float32)
            hidden=np.full(512,13.,np.float32); cell=np.full(512,17.,np.float32)
            gates=np.full(1024,19.,np.float32)
            args=[ptr(source),source.size]
            for weight in weights: args.extend((ptr(weight),weight.size))
            args.extend((ptr(output),output.size,ptr(hidden),hidden.size,
                         ptr(cell),cell.size,ptr(gates),gates.nbytes,
                         36,640,256,640,512))
            self.assertEqual(scan(*args),0,name)
            diff=np.abs(output.astype(np.float64)-expected.astype(np.float64))
            self.assertTrue(np.isfinite(output).all(),name)
            results[name]=float(diff.max())
        style=self.duration['predictor_style']
        for index in range(3):
            source=self.duration[f'norm{index}_input']
            expected=self.duration[f'norm{index}_output']
            prefix=f'duration_prosody.text_encoder.norm{index}'
            weight=self.duration['weight__'+prefix+'__projection_weight']
            bias=self.duration['weight__'+prefix+'__projection_bias']
            output=np.full(expected.shape,-777.,np.float32)
            scratch=np.zeros(1024,np.float32)
            self.assertEqual(norm(ptr(source),source.size,ptr(style),style.size,
                ptr(weight),weight.size,ptr(bias),bias.size,ptr(output),output.size,
                ptr(scratch),scratch.nbytes,36,512,128,512,512,1e-5),0)
            diff=np.abs(output.astype(np.float64)-expected.astype(np.float64))
            self.assertTrue(np.isfinite(output).all(),prefix)
            results[f'norm{index}']=float(diff.max())
        print('KOKORO_DURATION_ORACLE_FED '+json.dumps(results,sort_keys=True))
        (self.root/'duration-oracle-fed.json').write_text(json.dumps(results,indent=2))
        for name, maximum in results.items():
            self.assertLessEqual(maximum, 3e-6, (name, maximum))

    def test_connected_duration_projection_matches_independent_fp64(self):
        arena=self.arena(); frames=ctypes.c_int32(-999)
        self.assertEqual(self.fn(arena,len(arena),ctypes.byref(frames)),0)
        source=self.view(arena,'head_scan_output',np.float32,(36,512)).astype(np.float64)
        weight=self.duration['weight__duration_prosody.duration_head__weight'].astype(np.float64)
        bias=self.duration['weight__duration_prosody.duration_head__bias'].astype(np.float64)
        expected=np.broadcast_to(bias,(36,50)).copy()
        for channel in range(512):
            expected += source[:,channel,None]*weight[None,:,channel]
        actual=self.view(arena,'duration_logits',np.float32,(36,50))
        np.testing.assert_array_equal(actual,expected.astype(np.float32))

    def test_standalone_duration_replay_without_python_or_checkout(self):
        arena=self.arena()
        output=self.buffers['runtime_values']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'arena.bin').write_bytes(bytes(arena))
            shutil.copyfile(self.library,root/'generated.so')
            source=root/'host.c'
            source.write_text(f'''#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
extern int ck_kokoro_duration_predictor(uint8_t *, size_t, int32_t *);
int main(int argc, char **argv) {{
    if (argc != 3) return 10;
    const size_t bytes = {len(arena)}u;
    uint8_t *arena = aligned_alloc(64, (bytes+63u)&~(size_t)63u);
    if (!arena) return 11;
    FILE *in = fopen(argv[1], "rb");
    if (!in) {{ free(arena); return 12; }}
    size_t received = fread(arena, 1, bytes, in);
    int extra = fgetc(in); fclose(in);
    if (received != bytes || extra != EOF) {{ free(arena); return 13; }}
    int32_t frames = -999;
    int status = ck_kokoro_duration_predictor(arena, bytes, &frames);
    if (status) {{ free(arena); return 14; }}
    FILE *out = fopen(argv[2], "wb");
    if (!out) {{ free(arena); return 15; }}
    size_t written = fwrite(arena+{output['abs_offset']}u, 1, {36*4}u, out);
    written += fwrite(&frames, 1, sizeof(frames), out);
    int closed = fclose(out); free(arena);
    return written == {37*4}u && closed == 0 ? 0 : 16;
}}
''')
            binary=root/'native-duration'
            subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror',str(source),
                '-L',str(root),'-l:generated.so','-Wl,-rpath,$ORIGIN','-o',str(binary)],
                check=True,capture_output=True,text=True)
            subprocess.run([str(binary),'arena.bin','duration.bin'],cwd=root,check=True)
            result=np.fromfile(root/'duration.bin',np.int32)
            np.testing.assert_array_equal(result[:36],self.duration['durations'])
            self.assertEqual(result[36],103)
            replay=self.root/'standalone-duration'; replay.mkdir(exist_ok=True)
            for name in ('host.c','native-duration','generated.so','arena.bin','duration.bin'):
                shutil.copy2(root/name,replay/name)
            (root/'arena.bin').write_bytes(b'invalid')
            rejected=subprocess.run([str(binary),'arena.bin','rejected.bin'],cwd=root)
            self.assertEqual(rejected.returncode,13)
            self.assertFalse((root/'rejected.bin').exists())


if __name__=='__main__': unittest.main()
