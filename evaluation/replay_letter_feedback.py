"""Replay frozen model windows through the current tracker and compare final letter warnings."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_recitation_pack import run_clip
from evaluation.compare_recitation_runs import learner
from model_assets import ACTIVITY_MODEL_REVISION, MODEL_REVISION, TRANSCRIPTION_MODEL_REVISION
from quran_corpus import QuranCorpus


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def cached_evidence(key, cache):
    revisions = [MODEL_REVISION, TRANSCRIPTION_MODEL_REVISION, ACTIVITY_MODEL_REVISION]
    engines = key['engine_sha256']
    parts = [*revisions, *(engines[f] for f in ('transcription.py', 'speech_activity.py', 'recitation.py'))]
    if 'repeat_recovery.py' in engines:
        parts.append(engines['repeat_recovery.py'])
    prefix = '|'.join([*parts, key['device']]).encode()

    def read(samples):
        filename = sha256(prefix + samples.tobytes()).hexdigest() + '.json'
        # Refuse to fill any missing frozen observation with a different live engine.
        return json.loads((cache / filename).read_bytes())

    return read


def summarize(clip, row):
    kind = clip['source_kind']
    sites = learner(clip, row)['sites'] if kind == 'natural-learner' else []
    changes = []
    for token in row['tokens']:
        if token['state'] != 'difference':
            continue
        details = [dict(expected=e['expected'], heard=e['heard']) for e in token['errors'] if e['type'] == 'letter']
        if details:
            changes.append(dict(raw=token['text'], start_ms=token['start_ms'], end_ms=token['end_ms'],
                                reference=token['reference'], contrasts=details))
    counts = Counter(clips=1, letter_warnings=len(changes), clips_with_letter_warnings=int(bool(changes)),
                     clips_with_any_warning=int(bool(row['gaps']) or any(t['state'] == 'difference' for t in row['tokens'])))
    for site in sites:
        if site['label'] == 'Letters' and site['eligible']:
            counts['eligible_published_letter_sites'] += 1
            counts['matched_letter_sites'] += int(site['typed_hit'])
    return counts, dict(sites=sites, letter_warnings=changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'.cache/evaluation-pack/manifest.json')
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache/evaluation-evidence')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/letter-repair-evidence.json')
    args = parser.parse_args()
    manifest_hash = digest(args.manifest)
    clips = {c['id']: c for c in json.loads(args.manifest.read_bytes())['clips']}
    dependencies = ('word_alignment.py', 'transcript_tracking.py', 'quran_corpus.py',
                    'benchmark_recitation_pack.py', 'evaluation/compare_recitation_runs.py')
    report = dict(
        method='Same frozen per-window ASR and CTC observations. Only downstream tracking and grading are replayed.',
        limits=['Learner labels identify reference sites and categories. They do not verify a verbatim transcript '
                'or the actual substituted consonant, and the labelled word may be outside this audio chunk.',
                'Professional recordings and learner files without published errors are negative controls. '
                'Their warning counts are proxies for false alarms, not independently adjudicated false positives.',
                'The historical holdout was inspected for diagnosis; this replay measures regression on known material.'],
        manifest_sha256=manifest_hash, evaluator_sha256=digest(Path(__file__)),
        tracker_sha256={f: digest(ROOT / f) for f in dependencies},
        model_revisions=[MODEL_REVISION, TRANSCRIPTION_MODEL_REVISION, ACTIVITY_MODEL_REVISION], splits={})
    corpus = QuranCorpus()
    for split in ('dev', 'holdout'):
        source = ROOT / 'reports' / f'upgrade-candidate-{split}.json'
        previous = json.loads(source.read_bytes())
        key = previous['key']
        if key['manifest_sha256'] != manifest_hash or key['split'] != split or previous['status'] != 'completed':
            raise ValueError(f'Incompatible previous run: {source}')
        read = cached_evidence(key, args.cache)
        groups = defaultdict(lambda: dict(before=Counter(), after=Counter()))
        cases = []
        for baseline in previous['results']:
            clip = clips[baseline['id']]
            group = clip['source_kind']
            if group not in {'professional', 'natural-learner', 'controlled-edit-human', 'ordinary-speech'}:
                continue
            path = args.manifest.parent / clip['audio']
            if digest(path) != clip['sha256']:
                raise ValueError(f'Changed audio: {path}')
            new = run_clip(clip, path, read, corpus)
            if group == 'natural-learner':
                group += '-annotated' if clip['source_annotations']['canonical_word_errors'] else '-reviewed-clean'
            case = dict(id=clip['id'], source_group=clip['group'], audio_sha256=clip['sha256'], cohort=group)
            for name, row in (('before', baseline), ('after', new)):
                counts, case[name] = summarize(clip, row)
                groups[group][name].update(counts)
            if group.startswith('natural-learner') or case['before'] != case['after']:
                cases.append(case)
        report['splits'][split] = dict(previous_report_sha256=digest(source), acoustic_key=key,
                                      cohorts=groups, cases=cases)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
