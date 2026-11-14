"""Compiled native lifecycle checks; synthetic arithmetic is not model parity."""
from array import array
import ctypes as C
from pathlib import Path
import subprocess
import threading
import time

import pytest
from server.session_v8 import SessionV8, SessionError, CK_SESSION_REQUEST_RAW_PROMPT

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def built(tmp_path_factory):
    folder=tmp_path_factory.mktemp('visual-session')
    run=subprocess.run(['make','--no-print-directory','ck-session-v8'],cwd=ROOT,capture_output=True,text=True,timeout=180)
    assert run.returncode==0,run.stdout+run.stderr
    for name,flag in [('current',[]),('old',['-DOMIT_WIDTH'])]:
        target=folder/name;target.mkdir()
        run=subprocess.run(['cc','-std=c11','-shared','-fPIC',*flag,'-I',str(ROOT/'include'),
            str(Path(__file__).parent/'fixtures/visual_session_model.c'),'-lm','-o',str(target/'libmodel.so')],capture_output=True,text=True,timeout=30)
        assert run.returncode==0,run.stderr
        (target/'weights.bump').write_bytes(b'fixture');(target/'weights_manifest.map').write_text('')
    return folder

@pytest.fixture
def model(built):
    session=SessionV8.open(built/'current',num_threads=1)
    native=C.CDLL(str(built/'current/libmodel.so'))
    native.test_state.restype=C.c_float
    yield session,native
    session.close()


def generate(session,**changes):
    ids=[]
    args=dict(tokens_before=[1,2],prefix=array('f',[0.25,1.5,-0.5,2.25]*2),
        tokens_after=[3,4],prefix_tokens=2,prefix_width=4,grid_x=2,grid_y=1,
        text_position=4,max_tokens=3,on_token=lambda token,text:ids.append(token) or 0)
    args.update(changes)
    return session.generate_visual(**args),ids


def test_direct_segment_reference_repeat_and_text_recovery(model):
    session,native=model
    native.ck_model_forward_segments_grid_ex.argtypes=[C.POINTER(C.c_int32),C.c_int,C.POINTER(C.c_float),
        C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.POINTER(C.c_int32),C.c_int,C.POINTER(C.c_float)]
    native.ck_model_get_logits.restype=C.POINTER(C.c_float)
    native.ck_model_decode.argtypes=[C.c_int32,C.POINTER(C.c_float)]
    native.ck_model_kv_cache_reset()
    before=(C.c_int32*2)(1,2);after=(C.c_int32*2)(3,4);prefix=(C.c_float*8)(*([.25,1.5,-.5,2.25]*2))
    assert native.ck_model_forward_segments_grid_ex(before,2,prefix,2,4,2,1,4,after,2,None)==0
    expected=[]
    for i in range(3):
        logits=native.ck_model_get_logits();expected.append(max(range(8),key=lambda n:logits[n]))
        if i<2: assert native.ck_model_decode(expected[-1],None)==0
    expected_state=native.test_state()
    a,ids=generate(session);assert ids==expected and native.test_state()==expected_state
    assert a['prompt_tokens']==6 and a['generated_tokens']==3
    assert a['stop_reason']==2
    generate(session,prefix=array('f',[1.0]*8))
    _,ids=generate(session);assert ids==expected and native.test_state()==expected_state
    # Visual state is deliberately not checkpoint/continuation eligible.
    with pytest.raises(SessionError):
        session.generate(None,'x',max_tokens=1,temperature=0,top_p=1,on_token=lambda *a:0,
            flags=CK_SESSION_REQUEST_RAW_PROMPT | (1<<3))
    session.generate(None,'x',max_tokens=1,temperature=0,top_p=1,on_token=lambda *a:0,
        flags=CK_SESSION_REQUEST_RAW_PROMPT)
    assert native.ck_model_get_active_tokens()==1


@pytest.mark.parametrize('change',[
    {'tokens_before':[99]}, {'prefix_width':2,'prefix':array('f',[1]*4)},
    {'prefix':array('f',[float('nan')]*8)}, {'max_tokens':30}, {'text_position':31},
])
def test_rejection_leaves_state_untouched(model,change):
    session,native=model;generate(session)
    state=native.test_state();calls=native.test_calls();resets=native.test_resets()
    with pytest.raises(SessionError): generate(session,**change)
    assert (native.test_state(),native.test_calls(),native.test_resets())==(state,calls,resets)
    assert generate(session)[0]['stop_reason']==2


def test_native_failure_resets_then_recovers(model):
    session,native=model;native.test_fail()
    with pytest.raises(SessionError):generate(session)
    assert native.ck_model_get_active_tokens()==0 and native.test_state()==0
    assert generate(session)[0]['stop_reason']==2


def test_cancel_inflight_prefill_releases_state_and_recovers(model):
    session,native=model;native.test_hold(1);results=[]
    worker=threading.Thread(target=lambda:results.append(generate(session)))
    worker.start()
    try:
        deadline=time.monotonic()+5
        while not native.test_entered() and time.monotonic()<deadline:time.sleep(.005)
        assert native.test_entered()
        assert session.progress()['phase']=='prefill'
        session.cancel();worker.join(5);assert not worker.is_alive()
    finally:
        native.test_hold(0);worker.join(5)
    assert results[0][0]['stop_reason']==3 and results[0][1]==[]
    assert native.ck_model_get_active_tokens()==0
    assert generate(session)[0]['stop_reason']==2


def test_cancel_during_decode_and_followup(model):
    session,native=model
    def cancel(*args):session.cancel();return 0
    result,_=generate(session,on_token=cancel)
    assert result['stop_reason']==3 and native.ck_model_get_active_tokens()==0
    assert generate(session)[0]['stop_reason']==2


def test_old_model_without_width_contract_fails_closed(built):
    session=SessionV8.open(built/'old',num_threads=1)
    try:
        with pytest.raises(SessionError) as error:generate(session)
        assert error.value.status==-5
    finally:session.close()


def test_native_direct_geometry_and_extent_validation(model):
    from server.session_v8 import _VisualInput, _GenerateRequest, _GenerateResult, _configure_abi
    session,native=model;generate(session)
    before=(C.c_int32*2)(1,2);after=(C.c_int32*2)(3,4);prefix=(C.c_float*8)(*([1]*8))
    v=_VisualInput(C.sizeof(_VisualInput),1,before,2,prefix,8,2,4,2,1,4,after,2)
    request=_GenerateRequest(struct_size=C.sizeof(_GenerateRequest),abi_version=1,
        user_text=b'',max_tokens=3,temperature=0,top_p=1,flags=1)
    result=_GenerateResult(struct_size=C.sizeof(_GenerateResult),abi_version=1)
    _configure_abi(session.lib,'ck_session_v8_generate_visual')
    for name,value in [('grid_x',2147483647),('prefix_float_count',7),
                       ('tokens_before_count',-1),('text_position',0),('prefix_tokens',2147483647)]:
        old=getattr(v,name);setattr(v,name,value)
        saved=(native.test_state(),native.test_calls(),native.test_resets())
        status=session.lib.ck_session_v8_generate_visual(session.session,C.byref(request),C.byref(v),None,None,C.byref(result))
        assert status==-1
        assert (native.test_state(),native.test_calls(),native.test_resets())==saved
        setattr(v,name,old)


def test_busy_native_reset_and_close_wait_for_visual_owner(model):
    session,native=model;native.test_hold(1);results=[]
    worker=threading.Thread(target=lambda:results.append(generate(session)))
    worker.start()
    try:
        deadline=time.monotonic()+5
        while not native.test_entered() and time.monotonic()<deadline:time.sleep(.005)
        assert native.test_entered()
        reset=session.lib.ck_session_v8_reset;reset.argtypes=[C.c_void_p];reset.restype=C.c_int
        assert reset(session.session)==-6
        closer=threading.Thread(target=session.close);closer.start()
        time.sleep(.02);assert closer.is_alive()
        session.cancel();worker.join(5);closer.join(5)
        assert not worker.is_alive() and not closer.is_alive() and session.session is None
    finally:
        native.test_hold(0);worker.join(5)
    with pytest.raises(SessionError):generate(session)


def test_nonfinite_native_output_resets_and_recovers(model):
    session,native=model;native.test_bad_logits()
    with pytest.raises(SessionError):generate(session)
    assert native.ck_model_get_active_tokens()==0
    assert generate(session)[0]['stop_reason']==2
