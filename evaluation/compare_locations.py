"""Compare automatic verse references emitted by frozen baseline/candidate replays."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
from statistics import median
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verify_recitation_cases import normalized_words

ABSTAIN='abstained'
EXACT='exact_target'
EQUIVALENT='identical_vocalized_text_other_verse'
HOMOGRAPH='same_normalized_words_other_verse'
WRONG='different_verse_text'
SPURIOUS='reference_on_non_quran_input'


def corpus_index():
    from quran_transcript import Aya
    aya=Aya(1,1)
    by_position={}
    same_text=defaultdict(list)
    for s in range(1,115):
        for a in range(1,aya.set_new(s,1).get().num_ayat_in_sura+1):
            raw=aya.set_new(s,a).get().imlaey
            vocalized=' '.join(unicodedata.normalize('NFC',raw).split())
            by_position[s,a]=(vocalized,tuple(normalized_words(raw)))
            same_text[vocalized].append((s,a))
    if len(by_position)!=6236:
        raise ValueError('Unexpected Hafs corpus size')
    return by_position,same_text


def classify(target,observed,corpus):
    if observed==[None,None] or observed is None:
        return ABSTAIN
    observed=tuple(observed)
    if observed not in corpus:
        raise ValueError(f'Invalid emitted Quran reference {observed!r}')
    if target is None:
        return SPURIOUS
    if observed==target:
        return EXACT
    actual,canonical=corpus[observed],corpus[target]
    if actual[0]==canonical[0]: return EQUIVALENT
    if actual[1]==canonical[1]: return HOMOGRAPH
    return WRONG


def location_case(clip,row,corpus,identicals):
    target=None if clip['source_kind'] in {'noise','ordinary-speech'} else (clip['surah'],clip['ayah'])
    events=[(float(e['end']),classify(target,e['location'],corpus),e['location']) for e in row['timeline']]
    if not events or any(a[0]>=b[0] for a,b in zip(events,events[1:])):
        raise ValueError(f'Invalid source clock for {clip["id"]}')
    def first(kinds):
        return next((time for time,kind,_ in events if kind in kinds),None)
    def later(then,kinds):
        return next((time for time,kind,_ in events if then is not None and time>then and kind in kinds),None)
    different=first({WRONG})
    off_target=first({WRONG,HOMOGRAPH,EQUIVALENT})
    same_positions=identicals[corpus[target][0]] if target else []
    return dict(id=clip['id'],source_kind=clip['source_kind'],target=target,
                other_identically_vocalized_positions=[p for p in same_positions if p!=target],
                identical_vocalized_target=len(same_positions)>1,
                final=events[-1][1],final_reference=events[-1][2],
                any_lock=any(kind!=ABSTAIN for _,kind,_ in events),
                first_exact_source_s=first({EXACT}),first_equivalent_or_exact_source_s=first({EQUIVALENT,EXACT}),
                first_different_text_source_s=different,first_other_numeric_target_source_s=off_target,
                exact_recovery_after_different_text_source_s=later(different,{EXACT}),
                exact_recovery_after_other_numeric_reference_source_s=later(off_target,{EXACT}),
                source_windows_by_reference=Counter(kind for _,kind,_ in events))


def ratio(n,total):
    return {'n':n,'total':total,'rate':n/total if total else None}


def distribution(values):
    v=sorted(x for x in values if x is not None)
    return dict(n=len(v),median=median(v) if v else None,p95=v[math.ceil(.95*len(v))-1] if v else None)


def summarize(rows):
    final=Counter(r['final'] for r in rows)
    n=len(rows)
    positives=sum(r['target'] is not None for r in rows)
    pos_final=Counter(r['final'] for r in rows if r['target'] is not None)
    pos_issued=positives-pos_final[ABSTAIN]
    neg=n-positives
    with_wrong=[r for r in rows if r['first_different_text_source_s'] is not None]
    with_other=[r for r in rows if r['first_other_numeric_target_source_s'] is not None]
    recovered_exact=[r for r in with_wrong if r['exact_recovery_after_different_text_source_s'] is not None]
    windows=Counter()
    for r in rows: windows.update(r['source_windows_by_reference'])
    return dict(clips=n,expected_quran=positives,negative_inputs=neg,
        final_counts={key:final[key] for key in (EXACT,EQUIVALENT,HOMOGRAPH,WRONG,SPURIOUS,ABSTAIN)},
        source_window_counts=dict(windows),
        final_reference_coverage_among_quran_inputs=ratio(pos_issued,positives), ever_reference_coverage_among_quran_inputs=ratio(sum(r['any_lock'] for r in rows if r['target'] is not None),positives),
        final_numeric_precision_among_published_targets=ratio(pos_final[EXACT],pos_issued),
        final_full_text_precision_among_published_targets=ratio(pos_final[EXACT]+pos_final[EQUIVALENT],pos_issued),
        exact_final_rate_among_all_targets=ratio(pos_final[EXACT],positives),
        any_reference_on_negative=ratio(sum(r['any_lock'] for r in rows if r['target'] is None),neg),
        different_text_then_exact=ratio(len(recovered_exact),len(with_wrong)),
        different_text_then_ended_exact=ratio(sum(r['final']==EXACT for r in with_wrong),len(with_wrong)),
        other_numeric_target_then_exact=ratio(sum(r['exact_recovery_after_other_numeric_reference_source_s'] is not None for r in with_other),len(with_other)),
        first_exact_reference_source_seconds=distribution(r['first_exact_source_s'] for r in rows),
        first_exact_or_identical_verse_source_seconds=distribution(r['first_equivalent_or_exact_source_s'] for r in rows),
        elapsed_source_seconds_to_recover_different_text=distribution(r['exact_recovery_after_different_text_source_s']-r['first_different_text_source_s'] for r in recovered_exact))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,default=ROOT/'reports')
    p.add_argument('--output',type=Path,default=ROOT/'reports/upgrade-location.json')
    p.add_argument('--split',choices=['dev','holdout','all'],default='all')
    args=p.parse_args()
    mb=(ROOT/'.cache/evaluation-pack/manifest.json').read_bytes()
    manifest_sha=sha256(mb).hexdigest()
    if manifest_sha!=json.loads((ROOT/'evaluation/assets-v1.json').read_text())['manifest_sha256']:
        raise ValueError('Frozen manifest mismatch')
    clips={c['id']:c for c in json.loads(mb)['clips']}
    corpus,equal=corpus_index()
    out=dict(manifest_sha256=manifest_sha,eval_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        corpus_version=version('quran-transcript'),vocalized_text_sha256=sha256('\n'.join(v[0] for v in corpus.values()).encode()).hexdigest(),
        notes=['A lock means an emitted verse reference in a saved analysis timeline. The final reference, any interim wrong reference, and abstention are reported separately. The model source clock measures received fixture audio and cannot measure browser latency.',
            'Identical verses have identical complete vocalized Hafs imlaey wording after NFC/space normalization. A separate class counts other typed verse IDs equal only after ASR lexical normalization. An isolated identical verse cannot determine an absolute verse number from audio alone.',
            'Source-provided learner references may describe only part of a recording. Single-verse attribution does not measure within-verse word-offset errors. Counts are per clip; window counts are also provided. Timings exclude clips without an exact reference and give the observed count.'],
        runs={})
    requested=['dev','holdout'] if args.split=='all' else [args.split]
    for build in ('baseline','candidate'):
        out['runs'][build]={}
        for split in requested:
            source=(args.reports/f'upgrade-{build}-{split}.json').read_bytes()
            run=json.loads(source)
            expected={key for key,c in clips.items() if c['split']==split}
            if run.get('status')!='completed' or run['key']['manifest_sha256']!=manifest_sha or run['key']['split']!=split or {r['id'] for r in run['results']}!=expected or len(run['results'])!=len(expected):
                raise ValueError(f'Incomplete or inconsistent {build} {split} run')
            scored=[location_case(clips[r['id']],r,corpus,equal) for r in run['results']]
            groups=sorted({x['source_kind'] for x in scored})
            out['runs'][build][split]=dict(source_sha256=sha256(source).hexdigest(),overall=summarize(scored),
                by_source={g:summarize([r for r in scored if r['source_kind']==g]) for g in groups},
                targets_with_identical_full_verses=summarize([r for r in scored if r['other_identically_vocalized_positions']]),
                targets_with_unique_full_verse=summarize([r for r in scored if r['target'] and not r['other_identically_vocalized_positions']]),clips=scored)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=1)+'\n')
    print(f'Location comparison completed for {requested}')


if __name__=='__main__':
    main()
