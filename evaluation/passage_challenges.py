"""Prepare and replay a continuous 2:282 excerpt and a declared chapter join."""
import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_evaluation_pack import AssetCache, RATE, PACK, chunk_bytes, digest, publish, write_atomic

SELECTION=Path(__file__).with_name('passage-challenges-v1.json')
LOCK=Path(__file__).with_name('passage-challenge-lock.json')


def extract(path, start, end):
    """Sample numbers on the decoded track clock, bounded PCM even for long surahs."""
    pieces=[]
    cursor=0
    def accept(converted):
        nonlocal cursor
        for frame in converted:
            lo,hi=cursor,cursor+frame.samples
            if hi>start and (end is None or lo<end):
                kept=frame.to_ndarray().reshape(-1)[max(0,start-lo):None if end is None else max(0,end-lo)]
                if len(kept): pieces.append(kept.copy())
            cursor=hi
    with av.open(str(path)) as container:
        resampler=av.AudioResampler(format='s16',layout='mono',rate=RATE)
        for frame in container.decode(audio=0):
            accept(resampler.resample(frame))
            if end is not None and cursor>=end: break
        else:
            accept(resampler.resample(None))
    if not pieces: raise ValueError('Requested excerpt exceeds available audio')
    return np.concatenate(pieces).astype('<i2')


def prepare(root, offline):
    spec_bytes=SELECTION.read_bytes()
    spec=json.loads(spec_bytes)
    lock=json.loads(LOCK.read_bytes()) if LOCK.exists() else {}
    if lock and lock['selection_sha256']!=digest(spec_bytes):
        raise ValueError('Selection hash changed')
    main=json.loads((ROOT/'evaluation/assets-v1.json').read_bytes())
    cache=AssetCache(root,{**main['assets'],**lock.get('assets',{})},offline)
    outdir=root/'passages'
    clips=[]
    for case in spec['cases']:
        pieces, segments, details=[],[],[]
        cursor=0
        for part in case['parts']:
            s=part['surah']
            src=spec['reader']['folder_url']+f'{s:03}.mp3'
            timing=f'https://www.mp3quran.net/api/v3/ayat_timing?surah={s}&read={spec["reader"]["id"]}'
            marks=json.loads(cache.obtain(timing).read_bytes())
            selected=[m for m in marks if part['start_ayah']<=m['ayah']<=part['end_ayah']]
            if [m['ayah'] for m in selected]!=list(range(part['start_ayah'],part['end_ayah']+1)):
                raise ValueError('Missing or unordered publisher timing')
            start_ms=0 if part.get('from_file_start') else selected[0]['start_time']
            marked_end=selected[-1]['end_time']
            end_ms=None if part.get('through_file_end') else marked_end
            if 'max_seconds' in part: end_ms=min(end_ms,start_ms+1000*part['max_seconds'])
            wave=extract(cache.obtain(src),start_ms*16,None if end_ms is None else end_ms*16)
            actual_end=start_ms+len(wave)/16
            details.append(dict(surah=s,audio_url=src,timing_url=timing,source_start_ms=start_ms,
                source_end_ms=actual_end,selected_ayah_start_ms=selected[0]['start_time'],
                selected_ayah_end_ms=marked_end,has_full_selected_ayahs=actual_end>=marked_end-50,
                source_sha256=cache.hashes[digest(src.encode())],timing_sha256=cache.hashes[digest(timing.encode())],
                begins_at_clip_ms=cursor/16,ends_at_clip_ms=(cursor+len(wave))/16))
            for m in marks:
                lo=max(start_ms,m['start_time'])
                hi=min(actual_end,m['end_time'])
                if hi>lo:
                    segments.append(dict(start_ms=cursor/16+lo-start_ms,end_ms=cursor/16+hi-start_ms,
                        surah=s,ayah=m['ayah'],role='publisher_opening' if m['ayah']==0 else 'recited_ayah'))
            cursor+=len(wave)
            pieces.append(wave)
        payload=chunk_bytes(np.concatenate(pieces))
        first=case['parts'][0]
        entry=dict(id=case['id'],source_kind='supplemental-challenge',challenge=case['kind'],
                   surah=first['surah'],ayah=first['start_ayah'],group='mp3quran-1',split='supplement',
                   spoken_text=None,spoken_text_verified=False,error_labels_complete=False,
                   audio='audio/'+case['id']+'.wav',sha256=digest(payload),duration_s=cursor/RATE,
                   parts=details,expected_ref_timeline=segments)
        if lock and lock['clips'][entry['id']]!=entry['sha256']:
            raise ValueError('Rendered challenge hash differs from frozen output')
        write_atomic(outdir/entry['audio'],payload)
        clips.append(entry)
    manifest=dict(complete=True,reader=spec['reader'],selection_sha256=digest(spec_bytes),clips=clips,
        constraints=['A long single-verse excerpt ends at 90 seconds. Its raw words have not been independently transcribed.',
        'The chapter transition joins two official recordings with no crossfade or inserted silence. The second track retains its untrimmed opening including publisher ayah 0. A zero index is an opening interval, not an API reference.',
        'Publisher intervals describe verse spans including pauses. Long-source decoding counts samples sequentially from the start, with bounded retained PCM.'])
    publish(outdir/'manifest.json',manifest)
    receipt=dict(selection_sha256=digest(spec_bytes),manifest_sha256=digest((outdir/'manifest.json').read_bytes()),
                 assets=dict(sorted(cache.hashes.items())), clips={c['id']:c['sha256'] for c in clips})
    if lock and lock!=receipt: raise ValueError('Receipt mismatch')
    publish(outdir/'asset-lock.json',receipt)
    return manifest


def score(clip,result):
    intervals=clip['expected_ref_timeline']
    transitions=[]
    for event in result['timeline']:
        point=min(event['end']*1000,clip['duration_s']*1000)-.01
        active=next(((m['surah'],m['ayah']) for m in intervals if m['start_ms']<=point<m['end_ms'] and m['ayah']>0),None)
        submitted=tuple(event['location']) if event['location']!=[None,None] else None
        transitions.append(dict(start_audio_s=event['start'],end_audio_s=event['end'],reference=submitted,
                                publisher_at_window_end=active,raw=event['raw'],raw_window=event['raw_window']))
    expected=(clip['surah'],clip['ayah'])
    exact=next((r['end_audio_s'] for r in transitions if r['reference']==expected),None)
    from verify_recitation_cases import normalized_words
    attributed=[t['reference']['word_index'] for t in result['tokens'] if t.get('reference')
                and (t['reference'].get('surah'),t['reference'].get('ayah'))==expected
                and isinstance(t['reference'].get('word_index'),int)]
    out=dict(id=clip['id'],duration_s=clip['duration_s'],media_sha256=clip['sha256'],first_target_ref_audio_s=exact,
             final_ref=transitions[-1]['reference'],final_raw=result['raw'],analysis_windows=len(transitions),
             final_lexical_words=len(normalized_words(result['raw'])),final_tokens=result['tokens'],gaps=result['gaps'],
             initial_verse_distinct_attached_positions=len(set(attributed)),
             last_attached_word_index_of_initial_verse=max(attributed,default=None),
             window_start_offsets=len({r['start_audio_s'] for r in transitions}),trace=transitions)
    if len(clip['parts'])>1:
        joined=clip['parts'][1]['begins_at_clip_ms']/1000
        first_verse=next(m['start_ms']/1000 for m in intervals if m['surah']==113 and m['ayah']>0)
        post=[r for r in transitions if r['end_audio_s']>=first_verse and r['reference'] and r['reference'][0]==113]
        out.update(join_at_s=joined,first_numbered_113_verse_s=first_verse,
                   first_113_ref_after_spoken_verse_s=post[0]['end_audio_s'] if post else None,
                   first_113_delay_source_s=post[0]['end_audio_s']-first_verse if post else None)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--root',type=Path,default=PACK)
    parser.add_argument('--score',type=Path)
    parser.add_argument('--engine-root',type=Path,default=ROOT)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    args=parser.parse_args()
    manifest=prepare(args.root,args.offline)
    print([(c['id'],c['duration_s']) for c in manifest['clips']],flush=True)
    if not args.score:return
    from benchmark_recitation_pack import ModelEvidence,run_clip
    sys.path.insert(0,str(args.engine_root.resolve()))
    from quran_corpus import QuranCorpus
    data=args.root/'passages'
    corpus=QuranCorpus()
    evidence=ModelEvidence(ROOT/'.cache/evaluation-evidence',args.device)
    key=dict(engine={p:digest((args.engine_root/p).read_bytes()) for p in ['transcription.py','recitation.py','speech_activity.py','transcript_tracking.py','quran_corpus.py','word_alignment.py']},
             scorer_sha256=digest(Path(__file__).read_bytes()),benchmarker_sha256=digest((ROOT/'benchmark_recitation_pack.py').read_bytes()),
             device=args.device, manifest_sha256=digest((data/'manifest.json').read_bytes()))
    result=dict(key=key,method='Offline fixed acoustic windows with real ASR and phonetic evidence. No wall-clock transport or independent verbatim scoring.',complete=False,results=[])
    if args.score.exists():
        result=json.loads(args.score.read_bytes())
        if result['key']!=key: parser.error('Existing report uses different frozen inputs')
    done={r['id'] for r in result['results']}
    for c in manifest['clips']:
        if c['id'] in done: continue
        row=run_clip(c,data/c['audio'],evidence,corpus)
        result['results'].append(score(c,row))
        result['complete']=len(result['results'])==len(manifest['clips'])
        publish(args.score,result)
        print(c['id'],flush=True)


if __name__=='__main__':
    main()
