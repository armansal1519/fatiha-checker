"""Rebuild the new listening controls. Reuse the earlier verified public audio cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_evaluation_pack import AssetCache, PACK, RATE, chunk_bytes, digest, prepare_clip, publish, read_pcm, write_atomic
from quran_transcript import Aya

SPEC = ROOT / 'evaluation/repair-voices-v1.json'
SOURCES = ROOT / 'evaluation/repair-sources-v1.json'
LOCK = ROOT / 'evaluation/repair-voices-lock.json'
DATA = PACK / 'repairs'


def words(surah, ayah):
    return Aya(surah, ayah).get().imlaey.split()


def save(case, parts, *, destination=DATA):
    wave = np.concatenate(parts)
    payload = chunk_bytes(wave)
    case.update(audio='audio/'+case['id']+'.wav',sha256=digest(payload),duration_s=len(wave)/RATE,split='fresh')
    write_atomic(destination/case['audio'], payload)
    return case


def reader_cases(reader, cache):
    reader_id = reader['id']
    tracks, times, sources = {}, {}, {}
    for surah in (93,112,113):
        url = reader['folder_url']+f'{surah:03d}.mp3'
        times_url = f'https://www.mp3quran.net/api/v3/ayat_timing?surah={surah}&read={reader_id}'
        times[surah] = {r['ayah']:r for r in json.loads(cache.obtain(times_url).read_bytes())}
        tracks[surah] = read_pcm(cache.obtain(url))
        sources[surah] = dict(audio_url=url,timing_url=times_url,license_url='https://www.mp3quran.net/ar/contact-us')
    def section(s,a,b,*,to_end=False,from_start=False):
        lo = 0 if from_start else times[s][a]['start_time']*16
        hi = len(tracks[s]) if to_end else min(times[s][b]['end_time']*16,len(tracks[s]))
        if not 0 <= lo < hi:
            raise ValueError(f'Invalid bounds for {reader_id}, {s}:{a}–{b}')
        return tracks[s][lo:hi], dict(**sources[s],surah=s,start_ayah=a,end_ayah=b,
                                       source_start_sample=lo,source_end_sample=hi)
    base = dict(group=f'mp3quran-{reader_id}',reader=reader['name'],spoken_text_verified=False,
                spoken_text=None,source_kind='professional',error_labels_complete=False)
    tail = section(112,3,4,to_end=True)
    head = section(113,1,2,from_start=True)
    def clip(name, refs, **kwargs):
        return save(dict(base,id=f'new-{reader_id}-{name}',sources=[s for _,s in refs],**kwargs),[w for w,_ in refs])
    ayahs = section(93,1,3)
    first = section(93,1,1)
    return [clip('transition',[tail,head],surah=112,ayah=3,case='chapter_join',
                 join_s=len(tail[0])/RATE,last_ayah_start_s=len(tail[0])/RATE+times[113][2]['start_time']/1000),
            clip('once',[ayahs],surah=93,ayah=1,case='clean_short_ayah',repeat_count=1,target_word=words(93,1)[0]),
            clip('twice',[first,ayahs],surah=93,ayah=1,case='short_ayah_reread',repeat_count=2,target_word=words(93,1)[0])]


def word_asset(s,a,i):
    return f'https://audio.qurancdn.com/wbw/{s:03}_{a:03}_{i:03}.mp3'


def edited_case(s,a,position,donor,case,cache,*,destination=DATA):
    original=words(s,a)
    indices=[(s,a,i) for i in range(1,len(original)+1)]
    if case=='midword_repeat':
        indices.insert(position,indices[position-1])
    elif case=='single_letter':
        indices[position-1]=tuple(donor)
    heard=[words(cs,ca)[pos-1] for cs,ca,pos in indices]
    parts,segments=[],[]
    clock=0
    for ref, word in zip(indices,heard):
        url=word_asset(*ref)
        wave=read_pcm(cache.obtain(url))
        parts.append(wave)
        segments.append(dict(source=url,word=word,start_ms=clock/16,end_ms=(clock+len(wave))/16))
        clock+=len(wave)
    return save(dict(id=f'words-{s}-{a}-{case}',surah=s,ayah=a,group='Quran.com word recordings',
                     source_kind='constructed-sequence',case=case,position=position,
                     target_word=original[position-1],donor=donor,segments=segments,
                     reading=heard,spoken_text=' '.join(heard),spoken_text_verified=True,
                     canonical_text=' '.join(original),error_labels_complete=False,
                     terms='https://quran.com/terms-and-conditions'),parts,destination=destination)


def prepare(offline):
    spec=json.loads(SPEC.read_text())
    prior=json.loads((ROOT/'evaluation/assets-v1.json').read_text())
    lock=json.loads(LOCK.read_text()) if LOCK.exists() else {}
    cache=AssetCache(PACK,prior['assets'] | lock.get('assets',{}),offline)
    sources=json.loads(SOURCES.read_bytes())
    readers={r['id']:r for r in sources['readers']}
    learners=sources['learners']
    def learner_clip(selection):
        group, filename=selection
        src=next(r for r in learners if r['recording_id']==group and Path(r['file_name']).name==filename)
        assert src['riwayah']=='Hafs' and not src['errors']
        rev=sources['learner_revision']
        url=f'https://huggingface.co/datasets/sobolev210/quran-recitation-errors/resolve/{rev}/'+src['file_name']
        key='new-learner-'+group[:8]
        return prepare_clip(dict(id=key,group=group,source_kind='natural-learner',split='fresh',
            audio='audio/'+key+'.wav',surah=int(src['surah']),ayah=int(src['ayah']),spoken_text=None,
            canonical_text=src['text'],spoken_text_verified=False,error_labels_complete=False,case='learner_clean',
            source_annotations={'canonical_word_errors':{}},source=dict(audio_url=url,revision=rev,license='MIT')),
            cache,DATA)
    with ThreadPoolExecutor(max_workers=5) as pool:
        pros=[pool.submit(reader_cases,readers[k],cache) for k in spec['reader_ids']]
        students=[pool.submit(learner_clip,v) for v in spec['learner_recordings']]
        controls=[]
        for s,a in spec['known_repeats']:
            for kind in ('midword_repeat','identity'):
                controls.append(pool.submit(edited_case,s,a,2,None,kind,cache))
        for edit in spec['single_letter_edits']:
            for kind in ('single_letter','identity'):
                controls.append(pool.submit(edited_case,edit['surah'],edit['ayah'],edit['position'],
                                            edit['donor'] if kind=='single_letter' else None,kind,cache))
        cases=[c for f in pros for c in f.result()]+[f.result() for f in students+controls]
    manifest=dict(selection_sha256=digest(SPEC.read_bytes()),clips=cases)
    receipt=dict(selection_sha256=manifest['selection_sha256'],assets=dict(sorted(cache.hashes.items())),
                 clips={c['id']:c['sha256'] for c in cases})
    if lock and receipt!=lock: raise ValueError('Frozen test audio or source differs')
    if not lock: publish(LOCK,receipt)
    publish(DATA/'manifest.json',manifest)
    print(len(cases),'verified clips',DATA/'manifest.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    prepare(parser.parse_args().offline)
