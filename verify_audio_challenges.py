"""Paced audio checks for real breaths, quiet recitation, pauses, noise, ordinary speech and resampling."""
import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path

import numpy as np

from benchmark_recitation_pack import load_pcm
from evaluate_transcription import word_error_rate
from verify_recitation_cases import normalized_words
from verify_stream_runtime import RATE, check_health, stream_v2

ROOT=Path(__file__).resolve().parent


def challenge_cases(manifest, words_dir, breathing):
    rows=json.loads(manifest.read_text())['clips']
    for clip in rows:
        if clip['source_kind']=='noise':
            yield clip['id'],load_pcm(manifest.parent/clip['audio']),RATE,'','nonspeech'
        if clip['source_kind']=='ordinary-speech':
            yield clip['id'],load_pcm(manifest.parent/clip['audio']),RATE,clip['spoken_text'],'ordinary'
    yield 'recorded_breath',load_pcm(breathing),RATE,'','nonspeech'
    a,b=[load_pcm(words_dir/f'wbw-112-003-{i:03d}.mp3') for i in (3,4)]
    phrase=np.concatenate([a,b])
    rng=np.random.default_rng(356)
    for rate in (16000,44100,48000):
        pcm=np.interp(np.arange(round(len(phrase)*rate/RATE))*RATE/rate,np.arange(len(phrase)),phrase)
        yield f'walam_yulad_{rate}',pcm,rate,'ولم يولد','recitation'
    yield 'quiet_with_inner_pause',np.concatenate([a,np.zeros(RATE//2),b])*.1,RATE,'ولم يولد','recitation'
    yield 'quiet_then_20s_noise',np.concatenate([phrase*.1,rng.normal(0,.006,20*RATE)]),RATE,'ولم يولد','recitation'
    # Two simulated reflections of a human voice at 83 and 171 ms.
    reflected = np.zeros(len(phrase)+round(.171*RATE),dtype=np.float32)
    for delay,gain in [(0,.65),(.083,.22),(.171,.1)]:
        offset = round(delay*RATE)
        reflected[offset:offset+len(phrase)] += gain*phrase
    yield 'human_phrase_simulated_echo',reflected,RATE,'ولم يولد','recitation'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url')
    parser.add_argument('--prepare',action='store_true',help='Fetch and verify the four pinned public stream fixtures.')
    parser.add_argument('--manifest',type=Path,default=ROOT/'.cache/evaluation-pack/manifest.json')
    parser.add_argument('--words',type=Path,default=ROOT/'.cache/hallucination-repro')
    parser.add_argument('--breathing',type=Path,default=ROOT/'.cache/challenge-audio/breathing.wav')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.prepare:
        from prepare_evaluation_pack import AssetCache, digest
        for asset in json.loads((ROOT/'evaluation/stream-assets.json').read_text()):
            path=ROOT/'.cache'/asset['file']
            if path.is_file():
                with path.open('rb') as f:
                    if file_digest(f,'sha256').hexdigest()==asset['sha256']:
                        continue
            cache=AssetCache(ROOT/'.cache/stream-downloads',{digest(asset['url'].encode()):asset['sha256']},False)
            downloaded=cache.obtain(asset['url'])
            path.parent.mkdir(exist_ok=True,parents=True)
            temp=path.with_suffix('.verified')
            temp.unlink(missing_ok=True)
            os.link(downloaded,temp)
            temp.replace(path)
        print('Stream fixtures verified.')
        return 0
    if not args.url or not args.output:
        parser.error('--url and --output are required for the live checks')
    report=dict(health=check_health(args.url),cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for name,pcm,rate,want,kind in challenge_cases(args.manifest,args.words,args.breathing):
        history=[]
        def observe(event,acc,elapsed):
            if event['type'] in {'transcript','feedback'}:
                history.append(dict(at=round(elapsed,3),raw=''.join(t['text'] for t in acc.tokens),
                                    analyzed_to_s=event['end_sample']/RATE,
                                    reviewed=sum(t['state'] in {'match','difference'} for t in acc.tokens),
                                    activity=(event.get('audio_activity') or {}).get('status')))
        run=stream_v2(args.url,lambda start,end:pcm[start:end],len(pcm),sample_rate=rate,on_event=observe)
        raw=''.join(t['text'] for t in run['final_tokens'])
        exact=normalized_words(raw)==normalized_words(want)
        if kind=='ordinary':
            passed=bool(normalized_words(raw)) and not any(e['reviewed'] for e in history)
        elif kind=='nonspeech':
            passed=exact and not any(e['raw'].strip() for e in history)
        else:
            passed=exact and all(e['raw']==history[i-1]['raw'] for i,e in enumerate(history)
                                 if i and e['activity'] in {'no_speech','uncertain'})
        lexical=word_error_rate(normalized_words(want),normalized_words(raw))
        tail_errors=[]
        if name=='quiet_then_20s_noise':
            last_voice_end=len(pcm)/rate-20
            for obs in history:
                if obs['analyzed_to_s']<=last_voice_end:
                    continue
                errors=word_error_rate(normalized_words(want),normalized_words(obs['raw']))
                if errors['insertions'] or errors['substitutions']:
                    tail_errors.append(dict(at=obs['at'],raw=obs['raw']))
            passed=passed and not tail_errors
        result=dict(id=name,kind=kind,expected=want,raw=raw,exact=exact,passed=passed,
                    lexical=lexical,unexpected_tail_text=tail_errors,
                    sample_rate=rate,stream_coverage=run['done']['metrics']['coverage_complete'],
                    model_stage_calls=run['model_stage_calls'],history=history)
        report['cases'].append(result)
        report['status']='running'
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(name,passed,repr(raw),flush=True)
    report['status']='passed' if all(c['passed'] for c in report['cases']) else 'failed'
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
