"""Compare two fixed real-browser runs, explicitly retaining missing milestones."""
import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import median


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    observed = sorted(value for value in values if value is not None)
    rank = ceil(.95 * len(values))
    return {
        'total': len(values), 'observed': len(observed), 'missing': len(values)-len(observed),
        'observed_median_ms': median(observed) if observed else None,
        'observed_p95_ms': observed[ceil(.95*len(observed))-1] if observed else None,
        'all_clips_p95_ms': observed[rank-1] if rank and rank <= len(observed) else None,
    }


def timing_pairs(keys, before, after, metric):
    baseline = {key: before[key].get(metric) for key in keys}
    candidate = {key: after[key].get(metric) for key in keys}
    shared = [key for key in keys if baseline[key] is not None and candidate[key] is not None]
    old = [baseline[key] for key in shared]
    new = [candidate[key] for key in shared]
    paired_median_change = 100*(1-median(new)/median(old)) if shared else None
    return {
        'baseline': summarize(list(baseline.values())), 'candidate': summarize(list(candidate.values())),
        'seen_in_both': shared, 'newly_seen': [k for k in keys if baseline[k] is None and candidate[k] is not None],
        'lost': [k for k in keys if baseline[k] is not None and candidate[k] is None],
        'never_seen': [k for k in keys if baseline[k] is None and candidate[k] is None],
        'paired_baseline_median_ms': median(old) if shared else None,
        'paired_final_median_ms': median(new) if shared else None,
        'reduction_of_paired_medians_pct': paired_median_change,
        'median_individual_paired_reduction_pct': median(100*(1-b/a) for a,b in zip(old,new)) if shared else None,
        'paired_20pct_reduction': paired_median_change >= 20 if shared else None,
    }


def comparison(old_path, new_path):
    old, new = [json.loads(path.read_text()) for path in (old_path,new_path)]
    if old['browser'] != new['browser'] or old['version'] != new['version']:
        raise ValueError('Browser engines or versions differ')
    if old.get('frozen', {}).get('sha256') != new.get('frozen', {}).get('sha256') or old.get('checkedAudio') != new.get('checkedAudio'):
        raise ValueError('Browser recordings or panel selection differ')
    def indexed(doc):
        runs={row.get('id', row['file'])+'|'+row['mode']: row for row in doc['runs']}
        if len(runs) != len(doc['runs']):
            raise ValueError('Duplicate run keys')
        return runs
    baseline, candidate = indexed(old), indexed(new)
    if not baseline or set(baseline) != set(candidate):
        raise ValueError('Run keys differ or there are no completed runs')
    text = timing_pairs(baseline,baseline,candidate,'textInViewport')
    verdict = timing_pairs(baseline,baseline,candidate,'feedbackInViewport')
    value = text['candidate']['all_clips_p95_ms']
    common_ready = verdict['baseline']['all_clips_p95_ms'] is not None and verdict['candidate']['all_clips_p95_ms'] is not None
    return {
        'baseline_report': old_path.name, 'baseline_sha256':digest(old_path),
        'candidate_report':new_path.name, 'candidate_sha256':digest(new_path),
        'baseline_status':old['status'], 'candidate_status':new['status'],
        'browser':new['browser'], 'version':new['version'],
        'first_text_in_viewport':text, 'first_verdict_in_viewport':verdict,
        'first_text_all_clips_p95_at_most_3500ms': value is not None and value<=3500,
        'verdict_all_clips_p95_20pct_reduction': (
            verdict['candidate']['all_clips_p95_ms'] <= .8*verdict['baseline']['all_clips_p95_ms']
            if common_ready else None),
    }


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-controls',required=True,type=Path)
    parser.add_argument('--final-controls',required=True,type=Path)
    parser.add_argument('--baseline-panel',required=True,type=Path)
    parser.add_argument('--final-panel',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report={
        'measure':'First binary PCM to visible token or definitive verdict intersecting the viewport, with a browser animation-frame presentation proxy.',
        'quantiles':'Nearest rank p95; missing observations count as unreachable. Medians here use the arithmetic mean of middle observations for even counts.',
        'paired_comparison':'Both implementations must produce the milestone on the same audio and mode. Missing milestones and their change in coverage are separately reported.',
        'controls':comparison(args.baseline_controls,args.final_controls),
        'readers':comparison(args.baseline_panel,args.final_panel),
    }
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
