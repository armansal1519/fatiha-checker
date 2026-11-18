"""Evaluate whole-Quran text retrieval on saved real ASR transcripts.

When verified acoustic evidence is available for every window of a clip, also
replay the actual tracker, including transcript revisions and word grading.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_recitation_pack import ModelEvidence, RATE, apply_asr_window, digest, input_windows, load_pcm
from evaluation.compare_locations import ABSTAIN, EXACT, classify, corpus_index, distribution
from quran_corpus import QuranCorpus
from transcript_tracking import TranscriptTracker


def model_windows(wave, row, cache, cache_prefix):
    """Require final evidence or a verified, unchanged primary model result."""
    result = []
    intervals = list(input_windows(len(wave)))
    if len(intervals) != len(row['timeline']):
        raise ValueError(f"Window count changed for {row['id']}")
    for (start, end, settled), recorded in zip(intervals, row['timeline']):
        if (start / RATE, end / RATE) != (recorded['start'], recorded['end']):
            raise ValueError(f"Audio intervals changed for {row['id']}")
        info = recorded['evidence']
        payload = wave[start:end].tobytes()
        if info['pcm_sha256'] != sha256(payload).hexdigest():
            raise ValueError(f"Decoded audio changed for {row['id']}")
        final = cache / (sha256(cache_prefix + payload).hexdigest() + '.json')
        if final.is_file():
            evidence = json.loads(final.read_bytes())
        elif (info['source'] == 'verified_original' and not info['confirmation_changed']
              and info['subclip_asr_calls'] == 0):
            source = cache / (info['upstream_cache_key'] + '.json')
            if not source.is_file():
                return None
            if digest(source) != info['cache_file_sha256']:
                raise ValueError(f"Cached primary evidence changed for {row['id']}")
            old = json.loads(source.read_bytes())
            evidence = dict(heard=old[info['primary_field']], phones=old['phones'])
        else:
            return None
        if evidence['heard']['text'] != recorded['raw_window']:
            raise ValueError(f"Model output changed for {row['id']}")
        result.append((start, end, settled, evidence))
    return result


def compact(location):
    return dict(status=location['status'], total=location['total_candidates'],
                positions=[[r['surah'], r['ayah']] for r in location['candidates']])


def baseline(position):
    if position == [None, None]:
        return dict(status='none', total=0, positions=[])
    return dict(status='confirmed', total=1, positions=[position])


def result_for(hint, target, index):
    if hint['status'] == 'ambiguous':
        return 'ambiguous'
    positions = hint['positions']
    return classify(target, positions[0] if positions else None, index)


def report_summary(rows, arm, index):
    counts, status_counts = Counter(), Counter()
    first, wrong, visible, wrong_confirmed = [], 0, 0, 0
    for row in rows:
        timeline = row['timeline']
        target = tuple(row['target']) if row['target'] else None
        kinds = [result_for(event[arm], target, index) for event in timeline]
        counts[kinds[-1]] += 1
        status_counts[(timeline[-1][arm]['status'], kinds[-1])] += 1
        first.append(next((event['end_s'] for event, kind in zip(timeline, kinds) if kind == EXACT), None))
        wrong += any(kind not in (EXACT, ABSTAIN, 'ambiguous') for kind in kinds)
        wrong_confirmed += any(event[arm]['status'] == 'confirmed' and kind not in (EXACT, ABSTAIN)
                               for event, kind in zip(timeline, kinds))
        visible += any(event[arm]['positions'] for event in timeline)
    return dict(clips=len(rows), final=dict(sorted(counts.items())),
                final_by_status={f'{s}/{k}': n for (s, k), n in sorted(status_counts.items())},
                ever_single_non_target=wrong, ever_confirmed_non_target=wrong_confirmed,
                ever_candidate_list=visible, first_single_exact_source_seconds=distribution(first))


def compare_clip(clip, row, pack_path, corpus, index, duplicates, cache, cache_prefix):
    media = pack_path.parent / clip['audio']
    if digest(media) != clip['sha256']:
        raise ValueError(f"Fixture changed for {row['id']}")
    windows = model_windows(load_pcm(media), row, cache, cache_prefix)
    tracker = TranscriptTracker(corpus) if windows else None
    target = None
    cohort = 'negative'
    if clip['source_kind'] not in ('ordinary-speech', 'noise'):
        target = (clip['surah'], clip['ayah'])
        cohort = 'duplicate' if len(duplicates[index[target][0]]) > 1 else 'unique'
    case = dict(id=row['id'], split=clip['split'], source_kind=clip['source_kind'],
                target=target, cohort=cohort, acoustic_replay=bool(tracker), timeline=[])
    for i, old in enumerate(row['timeline']):
        event = dict(end_s=old['end'], old=baseline(old['location']),
                     text=compact(corpus.text_location(old['raw'])))
        if tracker is not None:
            start, end, settled, observed = windows[i]
            apply_asr_window(tracker, observed, start, end, settled)
            feedback = tracker.update_phonemes(observed['phones'], start, end, settled)
            if ''.join(t['text'] for t in tracker.tokens) != old['raw']:
                raise ValueError(f"Spoken transcript changed for {row['id']} at {end / RATE}")
            event['live'] = compact(tracker.location_fields()['display_location'])
        case['timeline'].append(event)
    if tracker:
        case['final_feedback_changed'] = tracker.tokens != row['tokens'] or feedback['gaps'] != row['gaps']
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / '.cache/evaluation-pack/manifest.json')
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports')
    parser.add_argument('--cache', type=Path, default=ROOT / '.cache/evaluation-evidence')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--typed-text', help='Optional user-reported example to replay incrementally as text')
    args = parser.parse_args()
    manifest_hash = digest(args.manifest)
    if manifest_hash != json.loads((ROOT / 'evaluation/assets-v1.json').read_bytes())['manifest_sha256']:
        raise ValueError('Frozen pack manifest changed')
    pack = json.loads(args.manifest.read_bytes())
    by_id = {clip['id']: clip for clip in pack['clips']}
    corpus = QuranCorpus()
    index, duplicates = corpus_index()
    cache_prefix = '|'.join(ModelEvidence(args.cache, 'mps').parts).encode()
    engine_files = ('quran_corpus.py', 'transcript_tracking.py', 'live_socket.py')
    engine = {name: digest(ROOT / name) for name in engine_files}
    rows, splits = [], {}
    for split in ('dev', 'holdout'):
        path = args.reports / f'repair-final-{split}.json'
        source = json.loads(path.read_bytes())
        splits[split] = digest(path)
        if source['status'] != 'completed' or source['key']['manifest_sha256'] != manifest_hash:
            raise ValueError(f'Missing matched primary report for {split}')
        for row in source['results']:
            clip = by_id[row['id']]
            if clip['split'] != split:
                raise ValueError(f"Clip in unexpected split: {row['id']}")
            rows.append(compare_clip(clip, row, args.manifest, corpus, index, duplicates,
                                     args.cache, cache_prefix))
    if set(by_id) != {r['id'] for r in rows} or len(rows) != len(by_id):
        raise ValueError('Incomplete paired set')
    stats = {}
    integrated = [r for r in rows if r['acoustic_replay']]
    for arm, eligible in [('text', rows), ('live', integrated)]:
        groups = dict(all=eligible)
        for cohort in ('unique', 'duplicate', 'negative'):
            groups[cohort] = [r for r in eligible if r['cohort'] == cohort]
        stats[arm] = {cohort: {a: report_summary(group, a, index) for a in ('old', arm)}
                      for cohort, group in groups.items()}
    paired_seconds = {'old': [], 'live': []}
    for row in integrated:
        if row['cohort'] != 'unique':
            continue
        target = tuple(row['target'])
        first = {
            arm: next((event['end_s'] for event in row['timeline']
                       if result_for(event[arm], target, index) == EXACT), None)
            for arm in paired_seconds
        }
        if all(value is not None for value in first.values()):
            for arm, value in first.items():
                paired_seconds[arm].append(value)
    data = dict(
        paired_first_correct_source_seconds={arm: distribution(values) for arm, values in paired_seconds.items()},
        sha256=dict(manifest=manifest_hash, **splits, evaluator=digest(Path(__file__)), engine=engine),
        method=('Text retrieval uses merged transcripts from previous real MPS/CPU audio runs. Actual paired '
                'tracker replay requires complete verified acoustic windows. A source clip label may cover '
                'only its first verse. An exact numeric hit on identical complete verses can be coincidental. '
                'Source-clock timing excludes inference and UI. This is diagnostic reuse of previous clips.'),
        summary=stats,
        replay=dict(clips=len(integrated), windows=sum(len(r['timeline']) for r in integrated),
                    all_text_windows=sum(len(r['timeline']) for r in rows),
                    changed_final_grades=sum(r['final_feedback_changed'] for r in integrated),
                    cohorts=dict(Counter(r['source_kind'] for r in integrated))),
        rows=rows,
    )
    if args.typed_text:
        parts = args.typed_text.split()
        data['typed_example'] = dict(
            source='User-reported text only, without a microphone recording or word timestamps.',
            events=[dict(heard_prefix=' '.join(parts[:i]),
                         location=compact(corpus.text_location(' '.join(parts[:i]))))
                    for i in range(1, len(parts) + 1)],
        )
    if any(digest(ROOT / name) != h for name, h in engine.items()):
        raise ValueError('Engine changed during evaluation')
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(data['replay']), flush=True)


if __name__ == '__main__':
    main()
