"""Produce a side-by-side review of two complete, source-matched new-voice runs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def overview(row):
    fields = ('final_raw', 'expected_text', 'expected_target', 'expected_count', 'count', 'adjacent_pairs',
              'count_matches', 'any_target_warning', 'any_target_repeat_warning', 'expected_tail',
              'final_ayah_found', 'exact_ending', 'words_after_last_ayah', 'transient_or_final_credit_phrases',
              'exact_sequence', 'adjacent_at_expected', 'second_only_repeat', 'second_only_with_timed_ownership',
              'all_repeat_mark_ids', 'expected_canonical', 'expected_spoken', 'expected_contrast',
              'correct_raw_target', 'correct_contrast_at_location', 'raw_and_typed_contrast',
              'clean_target_has_warning', 'canonical_context', 'source_labels', 'states', 'gaps')
    out = {k:row[k] for k in fields if k in row}
    out['warnings'] = [dict(id=t['id'], text=t['text'], clock_ms=[t['start_ms'],t['end_ms']],
                            reference=t['reference'], errors=t['errors']) for t in row['warnings']]
    for which in ('target','second'):
        if which in row:
            obs=row[which]
            out[which] = {k:obs[k] for k in ('spoken_index','expected_spoken','source_start_ms','source_end_ms',
                        'edit_operation','token_time_overlaps_source','token')}
    return out


def compare(baseline, candidate):
    if not baseline['complete'] or not candidate['complete']:
        raise ValueError('Both runs must complete')
    for k in ('evaluation','manifest','selection','lock','device'):
        if baseline['key'][k] != candidate['key'][k]:
            raise ValueError(f'Unpaired evaluator or inputs: {k}')
    before = {r['id']:r for r in baseline['results']}
    if set(before) != {r['id'] for r in candidate['results']}:
        raise ValueError('Clip IDs differ')
    rows = []
    for updated in candidate['results']:
        prior = before[updated['id']]
        if updated['audio_sha256'] != prior['audio_sha256']:
            raise ValueError('Audio changed between versions')
        modes = {m:dict(baseline=overview(prior['modes'][m]),candidate=overview(updated['modes'][m]))
                 for m in updated['modes']}
        rows.append(dict(id=updated['id'],case=updated['case'],reader=updated['reader'],group=updated['group'],
                         audio_sha256=updated['audio_sha256'],requested_start=updated['requested_start'],
                         verified_spoken_text=updated['spoken_verbatim_verified'],modes=modes))
    return dict(method=candidate['method'],clip_count=len(rows),baseline_key=baseline['key'],
        candidate_key=candidate['key'],groups={k:dict(baseline=baseline['summary'][k],candidate=v)
                                              for k,v in candidate['summary'].items()},clips=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    baseline,candidate=args.baseline.read_bytes(),args.candidate.read_bytes()
    report=compare(json.loads(baseline),json.loads(candidate))
    report['report_sha256']={args.baseline.name:sha256(baseline).hexdigest(),
                             args.candidate.name:sha256(candidate).hexdigest()}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=1)+'\n')


if __name__=='__main__':
    main()
