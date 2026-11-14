"""One-request native vision worker. Parent owns admission and process lifetime."""
import json
import os
from pathlib import Path
import sys
import time


def run(profile_path, payload, emit):
    from server.vision_bundle import load, sha
    profile = load(profile_path)
    scripts = Path(__file__).resolve().parents[1] / 'version/v8/scripts'
    sys.path.insert(0,str(scripts))
    import ctypes as C
    import hashlib
    from server.live import _render_with_chat_templates
    from server.session_v8 import SessionV8, SESSION_LIB_PATH
    from run_multimodal_bridge_v8 import (
        _load_prebuilt_encoder_runtime, _planned_encoder_prefix_tokens,
        _run_encoder, _load_encoder_lib, _resolved_symbol_library,
        _encode_prompt_segment, _segment_text_has_bos_prefix,
        _local_prefix_text_pos_for_policy, _vision_prefix_position_policy,
        _vision_prefix_decode_policy, GGUFTokenizer,
    )
    started = time.monotonic()
    encdir, decdir = Path(profile['encoder']), Path(profile['decoder'])
    encoder = _load_prebuilt_encoder_runtime(encdir)
    enc_cfg = json.loads((encdir/'layout.json').read_text())['config']
    dec_cfg = json.loads((decdir/'layout_decode.json').read_text())['config']
    bridge = dec_cfg.get('multimodal_bridge_contract', {})
    if (bridge.get('position_policy') != 'mrope_2d' or
            bridge.get('cache_policy') != 'persistent_decoder_kv' or
            _vision_prefix_position_policy(enc_cfg) != 'mrope_2d'):
        raise ValueError('unsupported visual position/state contract')
    _vision_prefix_decode_policy(enc_cfg)
    capacity = dec_cfg['context_length']
    rows = _planned_encoder_prefix_tokens(encoder)
    if type(capacity) is not int or capacity <= 0 or rows <= 0:
        raise ValueError('invalid generated capacity')
    template = (decdir/'chat_template.jinja').read_text()
    placeholder = profile['image_placeholder']
    messages = payload.get('messages', [{'role':'user','content':[{'type':'image'}, {'type':'text','text':'Describe.'}]}])
    rendered = _render_with_chat_templates(template,None,messages,None)
    if not rendered or rendered.count(placeholder) != 1:
        raise ValueError('template must produce exactly one declared image placeholder')
    left, right = rendered.split(placeholder)
    tokenizer = GGUFTokenizer.from_gguf(profile['tokenizer_gguf'])
    before = _encode_prompt_segment(tokenizer,left,add_bos=not _segment_text_has_bos_prefix(left))
    after = _encode_prompt_segment(tokenizer,right,add_bos=False)
    budget = payload.get('max_tokens',1)
    if len(before)+rows+len(after)+budget > capacity:
        emit({'type':'error','code':'context_length_exceeded','message':'Rendered text, visual rows and output exceed capacity'})
        return
    session = SessionV8.open(decdir,context_length=capacity,num_threads=payload['threads'])
    try:
        model = C.CDLL(str(decdir/'libmodel.so'))
        width = model.ck_model_get_visual_prefix_width()
        if width <= 0 or not hasattr(model,'ck_model_forward_segments_grid_ex'):
            raise ValueError('missing visual session capability')
        identities = {'session_library':sha(SESSION_LIB_PATH)}
        for lib,symbol,expected in [
            (model,'ck_model_forward_segments_grid_ex',decdir/'libmodel.so'),
            (model,'ck_set_num_threads',decdir/'libckernel_engine.so'),
            (model,'ck_true_bpe_decode',decdir/'libckernel_tokenizer.so')]:
            loaded = _resolved_symbol_library(lib,symbol)
            if sha(loaded) != sha(expected): raise ValueError('loaded artifact mismatch')
            identities[symbol] = sha(loaded)
        if payload.get('probe'):
            lib = _load_encoder_lib(encoder['so_path'],engine_so=encoder['engine_so'])
            if lib.ck_model_init_with_manifest(str(encoder['weights_bump']).encode(),str(encoder['manifest_map']).encode()) != 0:
                raise ValueError('encoder initialization failed')
            lib.ck_model_free()
            identities['encoder_model']=sha(_resolved_symbol_library(lib,'ck_model_init_with_manifest'))
            identities['encoder_engine']=sha(_resolved_symbol_library(lib,'ck_set_num_threads'))
            if identities['encoder_engine'] != profile['files']['encoder/libckernel_engine.so']:
                raise ValueError('loaded encoder engine mismatch')
            if identities['encoder_model'] != profile['files']['encoder/libencoder_v8.so']:
                raise ValueError('loaded encoder mismatch')
            emit({'type':'complete','capabilities':{'experimental':True,'vision':True,
                'api':['chat_completions'],'tools':False,'max_images':1,'context_length':capacity,
                'visual_rows':rows,'visual_width':width,'resident':False,
                'image_geometry':[enc_cfg.get('image_width'),enc_cfg.get('image_height')]},
                'bundle_identity':profile['identity'],'loaded_libraries':identities})
            return
        emit({'type':'stage','stage':'encoder'})
        enc_start = time.monotonic()
        encoded = _run_encoder(encoder,'checker',image_path=Path(payload['image_path']))
        enc_seconds = time.monotonic()-enc_start
        if encoded['prefix_tokens'] != rows or encoded['embed_dim'] != width:
            raise ValueError('encoder/decoder visual extent mismatch')
        gx,gy = encoded['prefix_grid_x'],encoded['prefix_grid_y']
        if gx*gy != rows: raise ValueError('visual grid mismatch')
        text_pos = len(before)+_local_prefix_text_pos_for_policy('mrope_2d',rows,gx,gy)
        model.ck_model_get_logits.restype=C.POINTER(C.c_float)
        vocab = model.ck_model_get_vocab_size()
        def logits_bytes():
            stride=model.ck_model_get_logits_stride();pos=model.ck_model_get_active_tokens()
            pointer=model.ck_model_get_logits()
            if not pointer or pos<=0 or stride<0 or (stride and stride<vocab):raise ValueError('invalid logits view')
            return C.string_at(C.cast(pointer,C.c_void_p).value+(pos-1)*stride*4,vocab*4)
        reference = None
        if payload.get('verify_reference'):
            model.ck_model_forward_segments_grid_ex.argtypes=[C.POINTER(C.c_int32),C.c_int,C.POINTER(C.c_float),
                C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int32),C.c_int,C.POINTER(C.c_float)]
            b=(C.c_int32*len(before))(*before);a=(C.c_int32*len(after))(*after)
            p=(C.c_float*len(encoded['embeddings'])).from_buffer(encoded['embeddings'])
            model.ck_model_kv_cache_reset()
            if model.ck_model_forward_segments_grid_ex(b,len(before),p,rows,width,gx,gy,text_pos,a,len(after),None):
                raise ValueError('direct reference failed')
            reference=hashlib.sha256(logits_bytes()).hexdigest()
        emit({'type':'stage','stage':'decoder'})
        initial=[];tokens=[];callback_errors=[]
        def callback(token,text):
            try:
                if not initial: initial.append(hashlib.sha256(logits_bytes()).hexdigest())
                tokens.append(token);emit({'type':'delta','text':text})
                return 0
            except Exception as error:
                callback_errors.append(error)
                return 1
        result=session.generate_visual(before,encoded['embeddings'],after,prefix_tokens=rows,
            prefix_width=width,grid_x=gx,grid_y=gy,text_position=text_pos,max_tokens=budget,on_token=callback)
        if callback_errors: raise callback_errors[0]
        if not initial: initial.append(hashlib.sha256(logits_bytes()).hexdigest())
        if result['stop_reason'] not in (1,2): raise ValueError('generation did not complete normally')
        if reference is not None and (not initial or initial[0] != reference):
            raise ValueError('HTTP session/direct reference logits differ')
        emit({'type':'complete','usage':{'prompt_tokens':result['prompt_tokens'],
            'completion_tokens':result['generated_tokens'],'total_tokens':result['prompt_tokens']+result['generated_tokens']},
            'finish_reason':'stop' if result['stop_reason']==1 else 'length',
            'evidence':{'bundle_identity':profile['identity'],'loaded_libraries':identities,
                'encoder_model_sha256':encoded['model_library_sha256'],'image_sha256':encoded['image_sha256'],
                'prefix_sha256':hashlib.sha256(encoded['embeddings'].tobytes()).hexdigest(),
                'rendered_prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
                'token_ids_sha256':hashlib.sha256(json.dumps([before,after]).encode()).hexdigest(),
                'initial_logits_sha256':initial[0] if initial else None,
                'direct_reference_match':True if reference is not None else None,
                'encoder_seconds':enc_seconds,'prefill_ms':result['prefill_time_ms'],
                'decode_ms':result['decode_time_ms'],'total_seconds':time.monotonic()-started}})
    finally:
        session.close()


def main():
    # Keep C printf and library diagnostics away from the JSON event pipe.
    events = os.fdopen(os.dup(1),'w',buffering=1)
    os.dup2(2,1)
    def emit(event): events.write(json.dumps(event)+'\n');events.flush()
    try:
        import resource
        payload=json.loads(sys.stdin.buffer.read(256*1024+1))
        memory=payload.pop('address_space_bytes')
        resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
        run(sys.argv[1],payload,emit)
    except Exception as error:
        print(f'vision worker failed: {type(error).__name__}',file=sys.stderr)
        emit({'type':'error','code':'native_vision_failed','message':'Vision worker failed; request state discarded'})
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
