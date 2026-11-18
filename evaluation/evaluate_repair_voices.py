"""Run the pinned new voices through an archived or current engine on identical windows."""
import argparse
from collections import Counter
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_recitation_pack import ModelEvidence, RATE, apply_asr_window, digest, input_windows, load_pcm
from evaluate_transcription import align_words
from verify_recitation_cases import normalized_words

SPEC = ROOT / 'evaluation/repair-voices-v1.json'
LOCK = ROOT / 'evaluation/repair-voices-lock.json'
MODES = ('automatic', 'manual')
TAIL = 'مِنْ شَرِّ مَا خَلَقَ'
CREDIT = ('اشتركوا في القناة', 'ترجمة نانسي')


def spans(sequence, phrase):
    length = len(phrase)
    return [i for i in range(len(sequence) - length + 1) if sequence[i:i+length] == phrase]


def compact(token):
    return {key: token[key] for key in ('id', 'text', 'start_ms', 'end_ms', 'state', 'reference', 'errors')}


def overlaps(a, b):
    return min(a['end_ms'], b['end_ms']) > max(a['start_ms'], b['start_ms'])


def target_observation(tokens, pieces, aligned, index, clip, windows):
    """Alignment is to the spoken construction. Surface substitutions stay visible."""
    op = aligned[index]
    hi = op['hypothesis_index']
    segment = clip['segments'][index]
    window = max(windows, key=lambda w: max(0, min(w['end_ms'], segment['end_ms'])
        - max(w['start_ms'], segment['start_ms'])))
    owner = pieces[hi][1] if hi is not None else None
    token = tokens[owner] if owner is not None else None
    return dict(spoken_index=index, expected_spoken=segment['word'],
                source_start_ms=segment['start_ms'], source_end_ms=segment['end_ms'],
                edit_operation=op['operation'], observed_piece_index=hi,
                token=compact(token) if token else None,
                token_time_overlaps_source=bool(token and overlaps(token, segment)),
                ctc_window_ms=[window['start_ms'], window['end_ms']],
                ctc_groups_overlapping_source=[g for g in window['groups'] if overlaps(g, segment)])


def is_marked(token, kind):
    return bool(token and token['state'] == 'difference' and
                any(e['type'] == kind for e in token['errors']))


def at_reference(token, clip):
    ref = (token or {}).get('reference') or {}
    return (ref.get('surah'), ref.get('ayah'), ref.get('word_index')) == (
        clip['surah'], clip['ayah'], clip['position']-1)


def editing_score(clip, tokens, pieces, words, edits, windows):
    gold = normalized_words(clip['spoken_text'])
    aligned = {op['reference_index']: op for op in align_words(gold, words) if op['reference_index'] is not None}
    position = clip['position']-1
    first = target_observation(tokens, pieces, aligned, position, clip, windows)
    out = dict(expected_text=clip['spoken_text'], canonical_text=clip['canonical_text'],
               exact_sequence=words == gold, target=first)
    if clip['case'] == 'midword_repeat':
        second = target_observation(tokens, pieces, aligned, position+1, clip, windows)
        pair = [first, second]
        both = (all(o['edit_operation'] == 'equal' for o in pair) and
                second['observed_piece_index'] == first['observed_piece_index']+1 and
                second['token']['id'] != first['token']['id'])
        marked = [t['id'] for t in tokens if is_marked(t, 'repetition')]
        only_second = bool(both and second['token']['id'] in marked
                           and marked == [second['token']['id']])
        out.update(adjacent_pairs=len(spans(words, [gold[position]]*2)),
                   adjacent_at_expected=both, second=second, second_only_repeat=only_second,
                   second_only_with_timed_ownership=only_second and all(o['token_time_overlaps_source'] for o in pair),
                   all_repeat_mark_ids=marked)
    elif clip['case'] == 'single_letter':
        edit = next(e for e in edits if (e['surah'], e['ayah']) == (clip['surah'], clip['ayah']))
        token = first['token']
        found = [e for e in (token or {}).get('errors', []) if is_marked(token, 'letter') and e['type']=='letter']
        from recitation import basic_form
        exact_contrast = any([basic_form(e['expected'])[:1], basic_form(e['heard'])[:1]] == edit['changes']
                             for e in found)
        out.update(expected_contrast=edit['changes'], expected_canonical=edit['intended'],
                   expected_spoken=edit['heard'], actual_letter_errors=found,
                   correct_raw_target=first['edit_operation']=='equal',
                   correct_contrast_at_location=bool(exact_contrast and at_reference(token, clip)
                                                    and first['token_time_overlaps_source']),
                   raw_and_typed_contrast=bool(first['edit_operation']=='equal' and exact_contrast
                                               and at_reference(token, clip) and first['token_time_overlaps_source']))
    else:
        out['clean_target_has_warning'] = bool(first['token'] and first['token']['state']=='difference')
    return out


def score(clip, tokens, gaps, edits, windows):
    pieces = [(piece, i) for i, token in enumerate(tokens) for piece in normalized_words(token['text'])]
    words = [word for word, i in pieces]
    kind = clip['case']
    warnings = [compact(t) for t in tokens if t.get('state') == 'difference']
    out = dict(case=kind, final_raw=''.join(t['text'] for t in tokens), warnings=warnings,
               gaps=gaps, states=dict(Counter(t['state'] for t in tokens)),
               tokens=[compact(t) for t in tokens])
    if clip['spoken_text_verified']:
        out.update(editing_score(clip, tokens, pieces, words, edits, windows))
    elif kind in ('clean_short_ayah', 'short_ayah_reread'):
        target = normalized_words(clip['target_word'])[0]
        matches = spans(words, [target])
        observed = [compact(tokens[pieces[m][1]]) for m in matches]
        expected = clip['repeat_count']
        out.update(expected_target=clip['target_word'], expected_count=expected,
                   count=len(matches), count_matches=len(matches)==expected,
                   adjacent_pairs=len(spans(words, [target, target])), target_occurrences=observed,
                   any_target_warning=any(t['state']=='difference' for t in observed),
                   any_target_repeat_warning=any(is_marked(t, 'repetition') for t in observed),
                   correct_count_without_target_warning=len(matches)==expected and
                       (expected==1 or bool(spans(words,[target,target]))) and
                       all(t['state']!='difference' for t in observed))
    elif kind == 'chapter_join':
        expected = normalized_words(TAIL)
        matches = spans(words, expected)
        start = matches[-1] if matches else None
        out.update(expected_tail=TAIL, last_ayah_source_start_ms=1000*clip['last_ayah_start_s'],
                   final_ayah_found=start is not None,
                   exact_ending=bool(matches and start+len(expected)==len(words)),
                   words_after_last_ayah=words[start+len(expected):] if matches else None,
                   matching_tail_tokens=[compact(tokens[pieces[i][1]]) for i in
                       range(start, start+len(expected))] if matches else [],
                   final_credit_phrases=[p for p in CREDIT if spans(words, normalized_words(p))])
    else:
        out.update(canonical_context=clip['canonical_text'],
                   source_labels=clip['source_annotations']['canonical_word_errors'])
    return out


def run(clip, audio_path, evidence, corpus, edits):
    from transcript_tracking import TranscriptTracker
    trackers = {'automatic': TranscriptTracker(corpus), 'manual': TranscriptTracker(corpus, clip['surah'], clip['ayah'])}
    samples = load_pcm(audio_path)
    windows, history, gaps = [], {m: [] for m in MODES}, {}
    for lo, hi, settled in input_windows(len(samples)):
        observed = evidence(samples[lo:hi])
        offset, end_ms = lo*1000/RATE, hi*1000/RATE
        phones, heard = observed['phones'], observed['heard']
        windows.append(dict(start_ms=offset, end_ms=end_ms, primary_raw=observed.get('preview', heard)['text'],
            verified_raw=heard['text'], activity=heard.get('audio_activity', {}).get('status'),
            groups=[dict(text=g['text'], start_ms=g['start_ms']+offset, end_ms=g['end_ms']+offset)
                    for g in phones['groups']]))
        for mode, tracker in trackers.items():
            apply_asr_window(tracker, observed, lo, hi, settled)
            result = tracker.update_phonemes(phones, lo, hi, settled)
            gaps[mode] = result['gaps']
            history[mode].append(dict(end_ms=end_ms, raw=''.join(t['text'] for t in tracker.tokens),
                reference=result['reference'], reviewed=result['summary']['reviewed']))
    modes = {m: score(clip, trackers[m].tokens, gaps[m], edits, windows) for m in MODES}
    for m in MODES:
        modes[m]['timeline'] = history[m]
        if clip['case']=='chapter_join':
            modes[m]['transient_or_final_credit_phrases'] = [p for p in CREDIT if any(
                spans(normalized_words(row['raw']), normalized_words(p)) for row in history[m])]
    return dict(id=clip['id'], case=clip['case'], reader=clip.get('reader'), group=clip['group'],
                audio_sha256=clip['sha256'], spoken_verbatim_verified=clip['spoken_text_verified'],
                requested_start=[clip['surah'], clip['ayah']], duration_s=clip['duration_s'],
                modes=modes, windows=windows)


def summary(rows):
    keys = ('exact_sequence', 'adjacent_at_expected', 'second_only_repeat', 'second_only_with_timed_ownership',
            'correct_raw_target', 'correct_contrast_at_location', 'raw_and_typed_contrast', 'clean_target_has_warning',
            'count_matches', 'any_target_warning', 'any_target_repeat_warning',
            'correct_count_without_target_warning', 'final_ayah_found', 'exact_ending')
    by_case = {}
    for kind in sorted({r['case'] for r in rows}):
        by_case[kind] = {}
        for m in MODES:
            selected = [r['modes'][m] for r in rows if r['case']==kind]
            stats = {k: sum(s[k] for s in selected) for k in keys if k in selected[0]}
            states, typed = Counter(), Counter()
            for s in selected:
                states.update(s['states'])
                for warning in s['warnings']:
                    typed.update({e['type'] for e in warning['errors']})
                typed['omission'] += len(s['gaps'])
            stats.update(clips=len(selected), clips_with_any_warning=sum(bool(s['warnings'] or s['gaps']) for s in selected),
                         warning_types_by_token=dict(typed), states=dict(states))
            if kind=='chapter_join':
                stats['clips_with_credit_at_any_window'] = sum(bool(s['transient_or_final_credit_phrases']) for s in selected)
            by_case[kind][m]=stats
    return by_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path, default=ROOT/'.cache/evaluation-pack/repairs/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('mps','cpu'), default='mps')
    args=parser.parse_args()
    engine = args.engine_root.resolve()
    sys.path.insert(0, str(engine))
    from quran_corpus import QuranCorpus
    files = ['model_assets.py', 'transcript_tracking.py', 'word_alignment.py', 'quran_corpus.py',
             'transcription.py', 'speech_activity.py', 'recitation.py']
    if (engine/'repeat_recovery.py').is_file():
        files.append('repeat_recovery.py')
    manifest = json.loads(args.manifest.read_bytes())
    clips = manifest['clips']
    lock = json.loads(LOCK.read_bytes())
    if (manifest['selection_sha256'] != digest(SPEC) or lock['selection_sha256'] != digest(SPEC)
            or lock['clips'] != {c['id']: c['sha256'] for c in clips} or len(clips) != len(lock['clips'])):
        raise ValueError('Locked media and selection must match')
    key = dict(manifest=digest(args.manifest),selection=digest(SPEC),lock=digest(LOCK),device=args.device,
        engine={f:digest(engine/f) for f in files},
        evaluation={f:digest(ROOT/f) for f in ('evaluation/evaluate_repair_voices.py','benchmark_recitation_pack.py',
            'evaluate_transcription.py','verify_stream_runtime.py','verify_recitation_cases.py')})
    report = dict(key=key,complete=False,results=[],method='Identical growing audio windows; ASR and CTC offline, automatic and seeded tracking. Only constructed word sequences have exact spoken gold. The chapter ending and full-verse reread are source-timed professional proxies. Unlabelled learner recordings yield observational warning rates. Joined source segment boundaries include recording padding. Window CTC groups overlap source clocks and are not hand-aligned phones.')
    if args.output.exists():
        report = json.loads(args.output.read_bytes())
        if report['key']!=key:
            raise ValueError('Output provenance differs; choose a fresh output path')
    seen = {r['id'] for r in report['results']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def publish():
        partial = args.output.with_suffix('.partial')
        partial.write_text(json.dumps(report,ensure_ascii=False,indent=1)+'\n')
        partial.replace(args.output)
    corpus = QuranCorpus()
    evidence = ModelEvidence(ROOT/'.cache/evaluation-evidence', args.device)
    report['acoustic_key'] = evidence.parts
    edits = json.loads(SPEC.read_bytes())['single_letter_edits']
    for c in clips:
        audio = args.manifest.parent/c['audio']
        if digest(audio)!=c['sha256']:
            raise ValueError(f"Wrong audio hash for {c['id']}")
        if c['id'] not in seen:
            row = run(c,audio,evidence,corpus,edits)
            report['results'].append(row)
            publish()
            print(len(report['results']), '/', len(clips), c['id'], flush=True)
    report.update(complete=True,summary=summary(report['results']))
    publish()


if __name__=='__main__':
    main()
