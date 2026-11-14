#!/usr/bin/env python3
"""Opt-in generated visual-session equivalence on a retained bridge fixture.

Uses the SAME generated arithmetic for direct and session execution. This is not
an independent model oracle or HTTP/harness certification. Missing inputs fail
with a retained report; nothing is inferred from an older PASS.
"""
from array import array
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from server.session_v8 import SessionV8, SESSION_LIB_PATH
from certify_sequence_checkpoint_v8 import declared_state_regions, signature, sha_file
from run_multimodal_bridge_v8 import _resolved_symbol_library


def checked_loaded_identity(library, symbol, expected):
    path = _resolved_symbol_library(library, symbol)
    actual_hash = sha_file(path)
    if actual_hash != sha_file(expected):
        raise RuntimeError(f"loaded {symbol} differs from requested artifact")
    return {"path": str(path), "sha256": actual_hash}


def certify(report_path, prefix_path, reference_logits_path):
    report = json.loads(report_path.read_text())
    prefix = array('f'); prefix.frombytes(prefix_path.read_bytes())
    if report.get('prefix_dump_sha256') != sha_file(prefix_path):
        raise ValueError('visual prefix does not match retained bridge report')
    expected_ids = report['generated_token_ids']
    if not expected_ids or report['generation_stop_reason'] != 'max_tokens':
        raise ValueError('probe requires a nonempty, token-budget-limited trajectory')
    runtime = Path(report['decoder_runtime']['so_path']).resolve()
    directory = runtime.parent
    layout = json.loads((directory/'layout_decode.json').read_text())
    required = declared_state_regions(layout)
    before = report['prompt_tokens_before_image']; after = report['prompt_tokens_after_image']
    args = dict(prefix_tokens=report['prefix_tokens'], prefix_width=report['prefix_embed_dim'],
        grid_x=report['prefix_grid_x'], grid_y=report['prefix_grid_y'],
        text_position=report['prefix_text_pos'], max_tokens=len(expected_ids))
    with tempfile.TemporaryDirectory(prefix='cke-visual-session-') as folder:
        bundle = Path(folder)
        for name, source in [('libmodel.so',runtime),('weights.bump',directory/'weights.bump'),
                             ('weights_manifest.map',directory/'weights_manifest.map'),
                             ('libckernel_engine.so',directory/'libckernel_engine.so'),
                             ('libckernel_tokenizer.so',directory/'libckernel_tokenizer.so')]:
            (bundle/name).symlink_to(source)
        session = SessionV8.open(bundle,context_length=report['decoder_context_len'],num_threads=4)
        try:
            model = C.CDLL(str(runtime))
            model.ck_model_get_logits.restype=C.POINTER(C.c_float)
            model.ck_model_get_named_activation_ptr.argtypes=[C.c_char_p]
            model.ck_model_get_named_activation_ptr.restype=C.c_size_t
            model.ck_model_get_named_activation_nbytes.argtypes=[C.c_char_p]
            model.ck_model_get_named_activation_nbytes.restype=C.c_ssize_t
            model.ck_model_decode.argtypes=[C.c_int32,C.POINTER(C.c_float)]
            model.ck_model_forward_segments_grid_ex.argtypes=[C.POINTER(C.c_int32),C.c_int,
                C.POINTER(C.c_float),C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,
                C.POINTER(C.c_int32),C.c_int,C.POINTER(C.c_float)]
            loaded = {}
            for library, symbol, expected in [
                (model, 'ck_model_forward_segments_grid_ex', runtime),
                (model, 'ck_set_num_threads', directory/'libckernel_engine.so'),
                (model, 'ck_tokenizer_create', directory/'libckernel_tokenizer.so'),
                (session.lib, 'ck_session_v8_generate_visual', SESSION_LIB_PATH)]:
                loaded[symbol] = checked_loaded_identity(library, symbol, expected)
            vocab=model.ck_model_get_vocab_size()
            if vocab<=0:raise ValueError('invalid loaded vocabulary')
            def capture():
                pointer=model.ck_model_get_logits()
                stride=model.ck_model_get_logits_stride();pos=model.ck_model_get_active_tokens()
                if not pointer or pos<=0 or stride<0 or (stride and stride<vocab):
                    raise ValueError('invalid logits layout')
                address=C.cast(pointer,C.c_void_p).value + (pos-1)*stride*4
                logits=(C.c_float*vocab).from_buffer_copy(C.string_at(address,vocab*4))
                return logits,signature(model,logits,required)
            b=(C.c_int32*len(before))(*before);a=(C.c_int32*len(after))(*after)
            p=(C.c_float*len(prefix)).from_buffer(prefix)
            model.ck_model_kv_cache_reset()
            rc=model.ck_model_forward_segments_grid_ex(b,len(before),p,args['prefix_tokens'],
                args['prefix_width'],args['grid_x'],args['grid_y'],args['text_position'],a,len(after),None)
            if rc:raise RuntimeError(f'direct prefill failed: {rc}')
            rows=[];ids=[]
            for index in range(len(expected_ids)):
                logits,state=capture()
                if index==0 and bytes(logits)!=reference_logits_path.read_bytes():
                    raise ValueError('direct logits differ from retained standalone run')
                ids.append(max(range(vocab),key=lambda n:logits[n]));rows.append(state)
                if index+1<len(expected_ids) and model.ck_model_decode(ids[-1],None):
                    raise RuntimeError('direct decode failed')
            if ids!=expected_ids:raise ValueError('direct tokens differ from standalone run')
            trials=[]
            for trial in range(3):
                observed=[];seen=[];errors=[]
                def callback(token,text):
                    try:
                        _,state=capture();observed.append(state);seen.append(token)
                    except Exception as error:
                        errors.append(str(error));return 1
                    return 0
                result=session.generate_visual(before,prefix,after,on_token=callback,**args)
                if errors or seen!=ids or observed!=rows or result['stop_reason']!=2:
                    raise ValueError(f'session comparison failed in trial {trial}: {errors}')
                trials.append({'result':result,'steps':observed,'tokens_match':True})
            return {'status':'PASS','comparison':'direct generated versus native session',
                'http_tested':False,'independent_oracle':False,
                'state_scope':'complete declared state arenas including padding and active position; internal bridge scratch not captured',
                'loaded_libraries':loaded,'bridge_report_sha256':sha_file(report_path),
                'prefix_sha256':sha_file(prefix_path),'weights_sha256':sha_file(directory/'weights.bump'),
                'generated_source_sha256':sha_file(Path(report['decoder_runtime']['c_path'])),
                'vocab_size':vocab,'visual_rows':args['prefix_tokens'],
                'visual_width':args['prefix_width'],'direct_steps':rows,'session_trials':trials}
        finally:
            session.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge-report',type=Path,required=True)
    parser.add_argument('--prefix-f32',type=Path,required=True)
    parser.add_argument('--reference-logits-f32',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'status':'INCOMPLETE'})+'\n')
    try:result=certify(args.bridge_report,args.prefix_f32,args.reference_logits_f32)
    except Exception as error:
        result={'status':'FAIL','error_type':type(error).__name__,'error':str(error)}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status']);return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
