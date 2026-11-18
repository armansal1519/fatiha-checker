"""Paced transport, emitted transcript revisions and targeted word feedback on known repair fixtures."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_recitation_pack import load_pcm
from evaluation.compare_recitation_runs import control
from verify_recitation_cases import normalized_words
from verify_audio_challenges import challenge_cases
from verify_stream_runtime import RATE, check_health, stream_v2


def fixtures(pack):
    manifest=pack/'manifest.json'
    by_id={row['id']:row for row in json.loads(manifest.read_text())['clips']}
    chapter=pack/'passages/audio/transition-hafs-112-113.wav'
    yield 'chapter_join','auto',load_pcm(chapter),{},sha256(chapter.read_bytes()).hexdigest()
    ids=['control-110-2-repetition','control-107-3-repetition',
         'learner-e5e672b2-chunk0','learner-e5e672b2-chunk3']
    for mode in ('auto','manual'):
        for key in ids:
            if mode=='manual' and key==ids[0]: continue
            row=by_id[key]
            path=pack/row['audio']
            digest=sha256(path.read_bytes()).hexdigest()
            if digest!=row['sha256']: raise ValueError('Source hash mismatch: '+key)
            yield key,mode,load_pcm(path),row,digest
    for name,pcm,rate,want,kind in challenge_cases(manifest,ROOT/'.cache/hallucination-repro',
                                                   ROOT/'.cache/challenge-audio/breathing.wav'):
        if kind=='ordinary' or name=='quiet_then_20s_noise':
            assert rate==RATE
            yield name,'auto',pcm,by_id.get(name,dict(spoken_text=want)),None


def assessment(name,clip,tokens,gaps,events):
    raw=''.join(t['text'] for t in tokens)
    words=normalized_words(raw)
    all_words=[normalized_words(v['raw']) for v in events]
    if name=='chapter_join':
        phrase=normalized_words('اشتركوا في القناة')
        def has(text):
            return any(text[i:i+len(phrase)]==phrase for i in range(len(text)-len(phrase)+1))
        return dict(final_suffix_correct=words[-4:]==normalized_words('من شر ما خلق'),
                    final_has_boilerplate=has(words),ever_had_boilerplate=any(has(w) for w in all_words))
    if name.startswith('control-'):
        scored=control(clip,dict(tokens=tokens,gaps=gaps))
        gold=normalized_words(clip['spoken_text'])
        pos=clip['events'][0]['heard_indices'][0]
        word=gold[pos]
        def pair(sequence):
            return any(a==b==word for a,b in zip(sequence,sequence[1:]))
        return dict(spoken_text=clip['spoken_text'],final_exact=scored['exact_sequence'],
                    final_exact_with_marks=scored['exact_sequence_and_marks'],
                    target=word,final_pair=pair(words),ever_pair=any(pair(v) for v in all_words),
                    typed_repetition=scored['by_type']['repetition'])
    if name.startswith('learner-'):
        spelling=next(iter(clip['source_annotations']['canonical_word_errors']))
        gold=normalized_words(clip['canonical_text'])
        pos=gold.index(normalized_words(spelling)[0])
        def at_site(t):
            ref=t.get('reference') or {}
            return ref.get('surah')==clip['surah'] and ref.get('ayah')==clip['ayah'] and ref.get('word_index')==pos
        hits=[t for t in tokens if at_site(t) and t['state']=='difference' and any(
            e.get('type')=='letter' for e in t.get('errors',[]))]
        return dict(label=spelling,target_index=pos,final_target_letter_warning=bool(hits),
                    final_warned_raw=[t['text'] for t in hits],registered_annotation='Letters at canonical site; verbatim realization not independently known')
    if name=='quiet_then_20s_noise':
        expected=normalized_words(clip['spoken_text'])
        return dict(final_exact=words==expected,tail_has_extra=any(e['end_audio_s']>2.14 and
                    any(w not in expected for w in normalized_words(e['raw'])) for e in events))
    return dict(nonempty=bool(words),any_quran_verdict=any(e['reviewed'] for e in events),
                final_exact=words==normalized_words(clip['spoken_text']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--engine-root',type=Path,default=ROOT)
    parser.add_argument('--server-pid',type=int,required=True)
    parser.add_argument('--pack',type=Path,default=ROOT/'.cache/evaluation-pack')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=('transcription.py','live_socket.py','transcript_tracking.py','word_alignment.py','repeat_recovery.py')
    key=dict(engine={n:sha256((args.engine_root/n).read_bytes()).hexdigest() for n in paths if (args.engine_root/n).exists()},
             url=args.url,pid=args.server_pid,script=sha256(Path(__file__).read_bytes()).hexdigest(),
             pack=sha256((args.pack/'manifest.json').read_bytes()).hexdigest())
    report=dict(key=key,health=check_health(args.url),complete=False,results=[],
                method='Real PCM over v2 WebSocket in paced 128 ms chunks. Original public and controlled files. Model throughput changes window boundaries. Not a latency benchmark.')
    if args.output.exists():
        report=json.loads(args.output.read_text())
        if report['key']!=key: raise ValueError('Report provenance differs. Use a new path.')
    seen={(r['id'],r['mode']) for r in report['results']}
    def publish():
        p=args.output.with_suffix('.partial');p.write_text(json.dumps(report,ensure_ascii=False,indent=1));p.replace(args.output)
    for name,mode,pcm,clip,source_hash in fixtures(args.pack):
        if (name,mode) in seen: continue
        history=[]
        def observe(event,acc,elapsed):
            if event['type'] in ('transcript','feedback'):
                history.append(dict(type=event['type'],revision=event['revision'],at_wall_s=round(elapsed,3),
                                    start_audio_s=event['start_sample']/RATE,end_audio_s=event['end_sample']/RATE,
                                    raw=''.join(t['text'] for t in acc.tokens),reviewed=sum(t.get('state') in {'match','difference'} for t in acc.tokens),
                                    warnings=[dict(id=t['id'],raw=t['text'],details=t.get('errors',[]),reference=t.get('reference')) for t in acc.tokens if t.get('state')=='difference'],
                                    committed_count=acc.committed_count))
        run=stream_v2(args.url,lambda lo,hi:pcm[lo:hi],len(pcm),sample_rate=RATE,on_event=observe,
                      server_pid=args.server_pid,start_surah=clip.get('surah') if mode=='manual' else None,
                      start_ayah=clip.get('ayah') if mode=='manual' else None)
        tokens,gaps=run['final_tokens'],run['final_gaps']
        m=run['done']['metrics']
        row=dict(id=name,mode=mode,source_sha256=source_hash,pcm_sha256=sha256(pcm.tobytes()).hexdigest(),
                 seconds=len(pcm)/RATE,session_id=run['session_id'],samples_sent=run['samples_sent'],
                 acked=run['acknowledged_samples'],coverage_complete=m['coverage_complete'],
                 analyzed=m['analyzed_samples'],committed=m['committed_samples'],stage_calls=run['model_stage_calls'],
                 final_raw=''.join(t['text'] for t in tokens),final_reference=run['reference'],tokens=tokens,gaps=gaps,
                 assessment=assessment(name,clip,tokens,gaps,history),history=history)
        report['results'].append(row);publish()
        print(name,mode,row['final_raw'],row['assessment'],flush=True)
    report['complete']=True;publish()


if __name__=='__main__':
    main()
