#!/usr/bin/env python3
"""Explicit sentence-chunked Ming trial using one resident native session."""
import argparse,json,os,sys,tempfile,time,wave
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from version.v8.tts.run_ming3_native_v8 import NativeMingSession,prepare_tokens,IncompleteSynthesis,CancelledSynthesis
from version.v8.tts.ming3_text_segments import sentence_spans,validate_spans
from version.v8.tts.capture_ming3_synthesis_reference import digest

def write_wav(path,audio,sample_rate):
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sample_rate)
        f.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())

def join_chunk_wavs(output, chunks, sample_rate):
    """Assemble the published PCM from validated chunks using bounded memory."""
    final = output / "paragraph.wav"
    samples = sum(chunk["measurements"]["samples"] for chunk in chunks)
    if not 0 < samples <= (2**32 - 36 - 1) // 2:
        raise ValueError("assembled speech exceeds the PCM16 WAV size limit")
    fd, temp_path = tempfile.mkstemp(prefix=".paragraph-", suffix=".wav", dir=output)
    pending = Path(temp_path)
    try:
        with os.fdopen(fd, "wb") as file, wave.open(file, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.setnframes(samples)
            for chunk in chunks:
                path = output / f"chunk-{chunk['index']}.wav"
                if digest(path) != chunk["wav_sha256"]:
                    raise ValueError(f"chunk integrity check failed: {path.name}")
                with wave.open(str(path), "rb") as reader:
                    remaining = chunk["measurements"]["samples"]
                    if (reader.getnchannels(), reader.getsampwidth(),
                        reader.getframerate(), reader.getnframes()) != (1, 2, sample_rate, remaining):
                        raise ValueError(f"chunk WAV format or sample count differs: {path.name}")
                    while remaining:
                        count = min(remaining, 32_768)
                        pcm = reader.readframes(count)
                        if len(pcm) != count * 2:
                            raise ValueError(f"truncated PCM: {path.name}")
                        writer.writeframesraw(pcm)
                        remaining -= count
        pending.replace(final)
    finally:
        pending.unlink(missing_ok=True)
    return samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','publisher-root','text-file','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--capacity',type=int,default=512)
    p.add_argument('--max-frames',type=int,default=200);args=p.parse_args()
    if not 1<=args.threads<=64:p.error('threads must be 1..64')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'INCOMPLETE','segmentation':'explicit English punctuation sentence spans; not whole-utterance equivalence',
        'chunks':[],'listening_verdict':'NOT_TESTED','numerical_verdict':'NOT_CERTIFIED','threads':args.threads}
    def save():
        target=args.output/'report.json';temporary=args.output/'report.json.tmp'
        temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(target)
    save()
    try:
        if args.text_file.stat().st_size>400000:raise ValueError('text file exceeds limit')
        text=args.text_file.read_text();spans=sentence_spans(text);validate_spans(text,spans)
        tokens=[prepare_tokens(text[a:b],args.checkpoint,args.publisher_root) for a,b in spans]
        report.update(text_sha256=digest(args.text_file),word_count=len(text.split()),source_characters=len(text),
            checkpoint_sha256=digest(args.checkpoint/'model.safetensors'),tokenizer_sha256=digest(args.checkpoint/'tokenizer.json'));save()
        os.environ['OMP_NUM_THREADS']=str(args.threads)
        import torch
        torch.set_num_threads(args.threads)
        start=time.monotonic();session=NativeMingSession(args.checkpoint,args.output/'native',capacity=args.capacity,max_frames=args.max_frames,cached_backbone=True)
        report.update(load_seconds=time.monotonic()-start,artifacts=session.identities,execution_descriptor_sha256=session.plan['descriptor_sha256']);save()
        if any(len(ids)+session.plan['minimum_frames']-1>session.plan['capacity'] for ids in tokens):raise ValueError('a sentence exceeds prompt capacity')
        last_progress=[0.]
        for index,((a,b),ids) in enumerate(zip(spans,tokens)):
            report['active_chunk']=index;report['chunks'].append({'index':index,'source_start':a,'source_end':b,'token_ids':ids,'status':'RUNNING'});save()
            def progress():
                now=time.monotonic()
                if now-last_progress[0]>30:
                    last_progress[0]=now;print(json.dumps({'chunk':index,'elapsed_seconds':now-start}),flush=True)
                return False
            audio,measurements,latents=session.synthesize(ids,seed=731+index,should_cancel=progress)
            write_wav(args.output/f'chunk-{index}.wav',audio,session.plan['sample_rate']);np.save(args.output/f'latents-{index}.npy',latents)
            report['chunks'][-1].update(status='GENERATED_NOT_CERTIFIED',measurements=measurements,wav_sha256=digest(args.output/f'chunk-{index}.wav'))
            save()
        samples = join_chunk_wavs(args.output, report["chunks"], session.plan["sample_rate"])
        report.update(status='GENERATED_NOT_CERTIFIED',active_chunk=None,samples=samples,audio_seconds=samples/session.plan['sample_rate'],
            join_policy='ordered direct sample concatenation; no invented overlap or omitted samples',wav_sha256=digest(args.output/'paragraph.wav'));save()
    except (CancelledSynthesis,KeyboardInterrupt) as exc:report.update(status='CANCELLED',error=str(exc));save();raise
    except IncompleteSynthesis as exc:report.update(status='INCOMPLETE',error=str(exc));save();raise
    except BaseException as exc:report.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));save();raise
    print(json.dumps(report))
if __name__=='__main__':main()
