"""Rebuild and score independently timed human recitation from the pinned annex."""

import argparse
import json
from pathlib import Path
from statistics import median
import sys

from prepare_evaluation_pack import AssetCache, PACK, ROOT, RATE, chunk_bytes, digest, publish, read_pcm, write_atomic
from verify_recitation_cases import normalized_words

SPEC = ROOT / 'evaluation/timing-v1.json'


def prepare(annex, offline):
    spec = json.loads(SPEC.read_bytes())
    expected = {digest(v['audio_url'].encode()):v['sha256'] for v in spec['source_audio'].values()}
    cache = AssetCache(annex, expected, offline)
    for name, record in spec['source_audio'].items():
        path = annex / (name + '.mp3')
        if not path.exists() or digest(path.read_bytes()) != record['sha256']:
            path = cache.obtain(record['audio_url'])
        pcm = read_pcm(path)
        for clip in spec['clips']:
            if clip['group'] != name:
                continue
            audio = chunk_bytes(pcm[clip['publisher_start_sample']:clip['publisher_end_sample']])
            if digest(audio) != clip['sha256']:
                raise ValueError(f'Rendered audio differs for {clip["id"]}')
            write_atomic(annex/clip['audio'], audio)
    write_atomic(annex/'manifest.json', SPEC.read_bytes())
    return spec


def observations(tokens):
    result = []
    for token in tokens:
        words = normalized_words(token['text'])
        interval = None
        if len(words) == 1 and token.get('start_ms') is not None and token.get('end_ms') is not None:
            interval = (token['start_ms'], token['end_ms'])
        result.extend((word, interval) for word in words)
    return result


def timed_alignment(gold, times, observed):
    """Minimize edit distance first, then prefer the closest equal-word occurrence."""
    width = len(observed) + 1
    costs = [[(j, 0) for j in range(width)]]
    parent = {}
    for i, word in enumerate(gold, 1):
        costs.append([(i, 0)])
        for j, (other, interval) in enumerate(observed, 1):
            c, t = costs[i-1][j-1]
            displacement = abs(sum(times[i-1])-sum(interval))/2 if interval else 0
            options = [((c + (word != other), t + (displacement if word == other else 0)), 'pair'),
                       ((costs[i-1][j][0]+1,costs[i-1][j][1]),'skip_gold'),
                       ((costs[i][j-1][0]+1,costs[i][j-1][1]),'skip_heard')]
            cost, direction = min(options, key=lambda p:p[0])
            costs[i].append(cost)
            parent[i,j] = direction
    i, j = len(gold), len(observed)
    equal = {}
    while i and j:
        op = parent[i,j]
        if op == 'pair':
            if gold[i-1] == observed[j-1][0]:
                equal[i-1] = j-1
            i -= 1
            j -= 1
        elif op == 'skip_gold':
            i -= 1
        else:
            j -= 1
    return costs[-1][-1][0], equal


def measure(clip, tokens):
    gold = normalized_words(clip['spoken_text'])
    times, ids = clip['spoken_word_times_ms'], clip['reviewed_word_ids']
    if not len(gold) == len(times) == len(ids):
        raise ValueError(f'Unmatched annotation lengths for {clip["id"]}')
    heard = observations(tokens)
    orthographic_errors, _ = timed_alignment(gold, times, heard)
    # This annotated utterance repeats alif-lam-mim. Both the letter names and
    # the written abbreviation describe the same recitation; retain strict WER.
    if set(ids) == {'31:1:1'}:
        merged = []
        index = 0
        while index < len(heard):
            window = heard[index:index+3]
            if [w for w,_ in window] == ['الف','لام','ميم']:
                interval = (window[0][1][0],window[-1][1][1]) if all(t for _,t in window) else None
                merged.append((gold[0],interval))
                index += 3
            else:
                merged.append(heard[index])
                index += 1
        heard = merged
    errors, equal = timed_alignment(gold, times, heard)
    seen, rows = set(), []
    for idx, word_id in enumerate(ids):
        match = equal.get(idx)
        interval = heard[match][1] if match is not None else None
        rows.append(dict(reference_word_id=word_id,word=gold[idx],repeat=word_id in seen,
                         match=match,reviewed_ms=times[idx],observed_ms=interval))
        seen.add(word_id)
    return dict(id=clip['id'], spoken=clip['spoken_text'], raw=''.join(t['text'] for t in tokens),
                expected_words=len(gold),predicted_words=len(heard),orthographic_edit_distance=orthographic_errors,
                lexical_edit_distance=errors,order_exact=errors == 0,words=rows)


def aggregate(results):
    words = [w for r in results for w in r['words']]
    timed = [w for w in words if w['observed_ms'] is not None]
    def distribution(values):
        if not values: return None
        import numpy as np
        return dict(p50=float(median(values)),p95=float(np.percentile(values, 95)),max=max(values))
    repeats = [w for w in words if w['repeat']]
    return dict(clips=len(results),reviewed_occurrences=len(words),
                exact_order_clips=sum(r['order_exact'] for r in results),
                lexical_edit_distance=sum(r['lexical_edit_distance'] for r in results),
                orthographic_edit_distance=sum(r['orthographic_edit_distance'] for r in results),
                repeat_occurrences=len(repeats),lexically_matched_repeat_occurrences=sum(w['match'] is not None for w in repeats),
                repeat_occurrences_overlapping_reviewed_audio=sum(bool(w['observed_ms']) and min(w['observed_ms'][1], w['reviewed_ms'][1]) >
                    max(w['observed_ms'][0],w['reviewed_ms'][0]) for w in repeats),
                timed_lexical_matches=len(timed),
                absolute_start_error_ms=distribution([abs(w['observed_ms'][0]-w['reviewed_ms'][0]) for w in timed]),
                absolute_end_error_ms=distribution([abs(w['observed_ms'][1]-w['reviewed_ms'][1]) for w in timed]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PACK/'qud')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--score', type=Path, help='Run actual local inference and write scores to this JSON')
    parser.add_argument('--device', choices=['mps','cpu'], default='mps')
    parser.add_argument('--engine-root', type=Path, default=ROOT)
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache/evaluation-evidence')
    args = parser.parse_args()
    spec = prepare(args.root, args.offline)
    if args.score is None:
        print(f"Verified {len(spec['clips'])} reviewed timing clips")
        return
    from benchmark_recitation_pack import ModelEvidence, run_clip
    sys.path.insert(0, str(args.engine_root.resolve()))
    from quran_corpus import QuranCorpus
    key = dict(device=args.device,manifest_sha256=digest(SPEC.read_bytes()),
               engine={f:digest((args.engine_root/f).read_bytes()) for f in ('transcript_tracking.py','quran_corpus.py',
                    'word_alignment.py','transcription.py','speech_activity.py','recitation.py')},
               runners={f:digest((ROOT/f).read_bytes()) for f in ('evaluate_timed_recitations.py','benchmark_recitation_pack.py')})
    repeat_source=args.engine_root/'repeat_recovery.py'
    if repeat_source.is_file():
        key['engine'][repeat_source.name]=digest(repeat_source.read_bytes())
    report = dict(key=key, results=[],complete=False)
    if args.score.exists():
        report = json.loads(args.score.read_text())
        if report['key'] != key:
            parser.error('Source or model settings differ from the existing result')
    done = {r['id'] for r in report['results']}
    corpus = QuranCorpus()
    evidence = ModelEvidence(args.cache, args.device)
    for clip in spec['clips']:
        if clip['id'] in done: continue
        result = run_clip(clip, args.root/clip['audio'], evidence, corpus)
        row = measure(clip,result['tokens'])
        row['timing_summary'] = aggregate([row])
        row['timeline'] = result['timeline']
        report['results'].append(row)
        report['summary'] = aggregate(report['results'])
        report['complete'] = len(report['results'])==len(spec['clips'])
        publish(args.score, report)
        print(f"{len(report['results'])}/{len(spec['clips'])}: {clip['id']}",flush=True)


if __name__ == '__main__':
    main()
