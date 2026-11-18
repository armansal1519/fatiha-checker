"""Independently score saved baseline/candidate tokens and gaps against the frozen pack."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_timed_recitations import timed_alignment
from evaluate_transcription import align_words, word_error_rate
from verify_recitation_cases import normalized_words

TYPES = ('repetition','omission','transposition','substitution','insertion')
VERDICTS = {'match','difference'}


def fraction(numerator, denominator):
    return {'n':numerator, 'total':denominator, 'rate':numerator/denominator if denominator else None}


def lexical_alignment(clip, tokens):
    """Return exact word owners and minimum edits with time as a tie-breaker."""
    actual, ids = [], []
    for token in tokens:
        parts = normalized_words(token['text'])
        interval = (token['start_ms'], token['end_ms']) if len(parts)==1 else None
        for word in parts:
            actual.append((word,interval))
            ids.append(token['id'] if len(parts)==1 else None)
    gold = normalized_words(clip['spoken_text'])
    times = [(m['start_ms'],m['end_ms']) for m in clip['source']['asset_segments']]
    if len(gold) != len(times):
        raise ValueError('Incorrect gold tokenization')
    distance, exact = timed_alignment(gold,times,actual)
    aligned, skipped = {}, []
    edges = [(-1,-1),*sorted(exact.items()),(len(gold),len(actual))]
    for (prev_g,prev_h),(next_g,next_h) in zip(edges,edges[1:]):
        g,h = prev_g+1, prev_h+1
        ops=align_words(gold[g:next_g],[w for w,_ in actual[h:next_h]])
        for op in ops:
            if op['reference_index'] is None:
                continue
            k = g+op['reference_index']
            if op['hypothesis_index'] is None:
                skipped.append(k)
            else:
                aligned[k] = h+op['hypothesis_index']
        if next_g != len(gold): aligned[next_g] = next_h
    owners = {ids[h]:g for g,h in exact.items() if ids[h] is not None}
    return gold,actual,distance,aligned,owners


def control(clip, row):
    tokens = row['tokens']
    gold,heard,distance,aligned,owners = lexical_alignment(clip,tokens)
    graded = {owners[t['id']] for t in tokens if t['id'] in owners and t['state'] in VERDICTS}
    expected = Counter()
    fidelity = Counter()
    opportunity = Counter()
    wrong_positions = set()
    canonical = normalized_words(clip['canonical_text'])
    editions = clip.get('published_canonical_words') or canonical
    def expected_visible(expected_text, idx):
        if not isinstance(idx,int) or not 0<=idx<len(canonical):
            return False
        text=normalized_words(expected_text)
        return text==[canonical[idx]] or text==normalized_words(editions[idx])
    for e in clip['events']:
        kind = e['kind']
        if kind == 'omission':
            left = e['after_index']
            for ref in e['reference_indices']:
                expected[kind,(left,left+1),ref] += 1
                opportunity[kind] += left in graded and left+1 in graded
                fidelity['target_reference_omissions'] += 1
                lo, hi = aligned.get(left), aligned.get(left+1)
                if lo is not None and hi is not None and (heard[lo][0],heard[hi][0])==(gold[left],gold[left+1]):
                    fidelity['omissions_with_two_exact_anchors'] += 1
                    fidelity['filled_with_omitted_word'] += any(
                        heard[pos][0]==canonical[ref] for pos in range(lo+1,hi))
            continue
        for j,g in enumerate(e['heard_indices']):
            ref=e['reference_indices'][j] if kind in {'substitution','transposition'} else None
            expected[kind,g,ref] += 1
            opportunity[kind] += g in graded
            wrong_positions.add(g)
            value = heard[aligned[g]][0] if g in aligned else None
            fidelity['targeted_wrong_occurrences'] += 1
            if value==gold[g]: fidelity['retained'] += 1
            elif value is None: fidelity['deleted'] += 1
            else: fidelity['changed'] += 1
            if kind=='substitution':
                fidelity['substitution_opportunities'] += 1
                fidelity['substitutions_canonicalized'] += value==canonical[ref]
    warnings = []
    marked = []
    for t in tokens:
        if t['state']!='difference': continue
        anchor=owners.get(t['id'])
        marked.append(anchor)
        for kind in {e['type'] for e in t.get('errors',[])} & set(TYPES):
            details=[e for e in t['errors'] if e['type']==kind]
            full_details = [e for e in details if normalized_words(e.get('heard',''))==normalized_words(t['text'])]
            entire = bool(full_details)
            if kind=='insertion' and not entire:
                continue  # A phoneme insertion can share the wire type.
            ref = t.get('reference') or {}
            ref_index = None
            payload_ok=entire
            if kind in {'transposition','substitution'}:
                if (ref.get('surah'),ref.get('ayah'))==(clip['surah'],clip['ayah']):
                    ref_index=ref.get('word_index')
                payload_ok = entire and expected_visible(ref.get('word',''),ref_index) and any(
                    expected_visible(e.get('expected',''),ref_index) for e in full_details)
            warnings.append((t['id'],(kind,anchor,ref_index),payload_ok))
    for gap in row['gaps']:
        ref=gap.get('reference') or {}
        idx=ref.get('word_index') if (ref.get('surah'),ref.get('ayah'))==(clip['surah'],clip['ayah']) else None
        pair=(owners.get(gap.get('before_token_id')),owners.get(gap.get('after_token_id')))
        payload_ok=expected_visible(ref.get('word',''),idx) and expected_visible(gap.get('expected',''),idx)
        warnings.append((gap['id'],('omission',pair,idx),payload_ok))
    unmatched=expected.copy()
    counts={k:Counter(tp=0,fp=0,fn=0,eligible=sum(v for e,v in expected.items() if e[0]==k),
                      reviewed_opportunities=opportunity[k]) for k in TYPES}
    stray=[]
    for identifier,key,valid in warnings:
        kind=key[0]
        if valid and unmatched[key]:
            counts[kind]['tp'] += 1
            unmatched[key] -= 1
        else:
            counts[kind]['fp'] += 1
            stray.append({'id':identifier,'kind':kind,'heard_owner':key[1],'canonical_owner':key[2],'valid_detail':valid})
    for key,n in unmatched.items():
        counts[key[0]]['fn'] += n
    # Each changed word is a separate visual warning, including spurious words
    # whose recognized spelling has no exact match in the source.
    matching_marks=(Counter(marked) & Counter(wrong_positions)).total()
    exact_marks=Counter(marked)==Counter(wrong_positions)
    exact_struct=not stray and not any(unmatched.values()) and exact_marks
    legacy={k:row.get('structural',{}).get('by_type',{}).get(k,{}) for k in TYPES}
    return dict(id=clip['id'], operation=clip.get('operation'), by_type=counts,
                exact_sequence=distance==0, exact_sequence_and_marks=distance==0 and exact_struct,
                lexical_distance=distance, reference_occurrences=len(gold), reviewed_gold_occurrences=len(graded),
                flagged_words=len(marked), tagged_expected_heard=len(wrong_positions), matched_marks=matching_marks,
                gaps=len(row['gaps']), spurious_warnings=stray,
                expected_unmatched=[[kind,anchor,ref,n] for (kind,anchor,ref),n in unmatched.items() if n],
                fidelity=fidelity, legacy_by_type=legacy)


def learner(clip,row):
    target=normalized_words(clip['canonical_text'])
    sparse=clip['source_annotations']['canonical_word_errors']
    categories={'Letters':{'letter'},'Tashkeel':{'vowel','gemination'}}
    actual=[]
    def add(record, kind, identifier):
        ref=record.get('reference') or {}
        index=ref.get('word_index') if (ref.get('surah'),ref.get('ayah'))==(clip['surah'],clip['ayah']) else None
        actual.append(dict(id=identifier,kind=kind,index=index))
    for token in row['tokens']:
        if token['state']=='difference':
            for kind in sorted({e['type'] for e in token['errors']}):
                add(token,kind,token['id'])
    for gap in row['gaps']: add(gap,'omission',gap['id'])
    sites=[]
    for word,labels in sparse.items():
        indices=[i for i,w in enumerate(target) if [w]==normalized_words(word)]
        for label in labels:
            i=indices[0] if len(indices)==1 else None
            measurable=i is not None and label in categories
            hit=measurable and any(x['index']==i and x['kind'] in categories[label] for x in actual)
            any_at=i is not None and any(x['index']==i for x in actual)
            sites.append(dict(word=word, label=label, index=i, eligible=measurable,typed_hit=bool(hit),any_flag_at_site=any_at))
    for warning in actual:
        matches=[s for s in sites if s['index'] is not None and s['index']==warning['index']]
        warning['status']=('outside_reference_ayah_or_unmapped' if warning['index'] is None else
            'matched_published_type' if any(warning['kind'] in categories.get(s['label'],set()) for s in matches) else
            'different_or_ungraded_type_at_published_site' if matches else 'no_published_label_at_site')
    return dict(id=row['id'], source_recording_id=clip['group'], source_has_labels=bool(sparse), sites=sites, warnings=actual)


def count_states(rows):
    counts=Counter(t['state'] for r in rows for t in r['tokens'])
    return dict(states=dict(counts),reviewed=fraction(sum(counts[k] for k in VERDICTS),counts.total()))


def cohort(clip_rows):
    controls=[control(c,r) for c,r in clip_rows if c['source_kind']=='controlled-edit-human']
    learners=[learner(c,r) for c,r in clip_rows if c['source_kind']=='natural-learner']
    classes={k:count_states([r for c,r in clip_rows if c['source_kind']==k])
             for k in sorted({c['source_kind'] for c,r in clip_rows})}
    structure={}
    for k in TYPES:
        total=Counter()
        legacy=Counter()
        for x in controls:
            total.update(x['by_type'][k])
            legacy.update(x['legacy_by_type'][k])
        tp,fp,fn=total['tp'],total['fp'],total['fn']
        structure[k]={**total,'precision':fraction(tp,tp+fp),'recall':fraction(tp,tp+fn),
                      'coverage':fraction(total['reviewed_opportunities'],total['eligible']),
                      'previous_saved_scorer':dict(legacy)}
    totals=Counter()
    by_operation=defaultdict(Counter)
    for x in controls:
        totals.update(x['fidelity'])
        by_operation[x['operation']].update(x['fidelity'])
    sites=[s for x in learners for s in x['sites']]
    labels={}
    for kind in sorted({s['label'] for s in sites}):
        source=[s for s in sites if s['label']==kind]
        eligible=[s for s in source if s['eligible']]
        hits=sum(s['typed_hit'] for s in eligible)
        labels[kind]=dict(published=len(source),positional_and_supported=len(eligible),typed_hits=hits,
                          missed=len(eligible)-hits,any_warning_at_same_site=sum(s['any_flag_at_site'] for s in source))
    readers=defaultdict(list)
    negatives=[]
    for c,r in clip_rows:
        if c['source_kind']=='professional': readers[c['source']['reciter']].append((c,r))
        if c['source_kind'] in {'noise','ordinary-speech'}:
            early=[s for s in r['timeline'] if normalized_words(s['raw'])]
            target=normalized_words(c['spoken_text'])
            lexical=word_error_rate(target,normalized_words(r['raw']))
            negatives.append(dict(id=r['id'],kind=c['source_kind'],reference=c['spoken_text'],final=r['raw'],
                any_visible_text=bool(early),final_words=lexical['hypothesis_words'],
                any_quran_location=any(s['location'] != [None,None] for s in r['timeline']),
                any_quran_verdict=any(s['reviewed']>0 for s in r['timeline']),lexical=lexical))
    pro={}
    for reader,items in readers.items():
        metrics=[word_error_rate(normalized_words(c['canonical_text']),normalized_words(r['raw'])) for c,r in items]
        errors=sum(x['errors'] for x in metrics)
        denominator=sum(x['reference_words'] for x in metrics)
        pro[reader]=dict(clips=len(items), canonical_reference_wer_proxy=fraction(errors,denominator),
                         **count_states([r for _,r in items]))
    clean=[x for x in learners if not x['source_has_labels']]
    identities=[x for x in controls if x['operation']=='identity']
    return dict(clips=len(clip_rows),coverage_by_source=classes,structural=structure,
        control_summary=dict(clips=len(controls),typed_exact_and_verbatim=sum(x['exact_sequence_and_marks'] for x in controls),
            exact_verbatim=sum(x['exact_sequence'] for x in controls),
            clean_siblings=len(identities),clean_siblings_with_warnings=sum(x['flagged_words']>0 or x['gaps']>0 for x in identities),
            final_word_warnings_at_expected_spoken_positions=sum(x['matched_marks'] for x in controls),
            final_word_warnings_outside_constructed_targets=sum(x['flagged_words']-x['matched_marks'] for x in controls),
            expected_spoken_error_positions_without_final_word_warning=sum(x['tagged_expected_heard']-x['matched_marks'] for x in controls),
            lexical_wer=fraction(sum(x['lexical_distance'] for x in controls),sum(x['reference_occurrences'] for x in controls)),
            target_word_retention=fraction(totals['retained'],totals['targeted_wrong_occurrences']),
            deletion_of_target=fraction(totals['deleted'],totals['targeted_wrong_occurrences']),
            changed_target=fraction(totals['changed'],totals['targeted_wrong_occurrences']),
            canonicalized_substitutions=fraction(totals['substitutions_canonicalized'],totals['substitution_opportunities']),
            omission_anchor_observability=fraction(totals['omissions_with_two_exact_anchors'],totals['target_reference_omissions']),
            omitted_word_generated_between_anchors=fraction(totals['filled_with_omitted_word'],totals['omissions_with_two_exact_anchors']),
            fidelity_by_edit={k:dict(v) for k,v in by_operation.items()}),
        learner_summary=dict(clips=len(learners),published_labels=labels,
            warning_instances=dict(Counter(w['status'] for x in learners for w in x['warnings'])),
            no_published_error_clips=len(clean),no_published_error_clips_with_warnings=sum(bool(x['warnings']) for x in clean)),
        professional_by_reader=pro,negative_challenges=negatives,control_cases=controls,learner_cases=learners)


NOTES = [
    'Structural precision and recall score individual final word and gap warnings. Multiple occurrences without a matching ASR anchor remain separate false warnings. Transposition counts each affected word. Correct details require heard text and canonical reference ownership.',
    'Minimum-edit word alignment uses reviewed construction intervals only to break equal-word ties such as adjacent duplicates. The former scorer used sets and a text-only tie break, and omitted display-payload checks. Historical counts are shown per type for traceability.',
    'Structural opportunity coverage is the fraction of expected event occurrences for which the affected heard word has an exact lexical owner and a final match/difference verdict. A missing-word gap requires both its observed neighboring anchors to be reviewed. Insertion has no positive controls independent of repetition.',
    'False warning rates on source-reviewed learner recordings cannot be computed from sparse error dictionaries. An unmatched alert can reflect a real unlabelled error, an alignment error, or a false alert. Published-site recall omits ambiguous canonical word ownership and unsupported categories.',
    'Professional WER compares against the published target verse. Home capture, additional phrases, and phonetic errors do not have independent full transcripts in that cohort. Professional reader identities determine the split, while learners are split by recording ID.',
    'Arabic everyday speech may correctly yield text; any Quran verdict is scored separately. For generated nonspeech any published provisional or final word is unwanted.',
    'Constructed wrong-word denominators come from explicit edited indices. Repetition, transposed words, and substitution all contribute. An omitted reference word has no spoken occurrence. Canonicalization here counts only deliberate lexical substitutions.',
]


def paired_summary(baseline, candidate):
    counts = ('tp','fp','fn','reviewed_opportunities','eligible')
    before = {r['id']:r for r in baseline['control_cases']}
    improvements, regressions = [], []
    for row in candidate['control_cases']:
        original = before[row['id']]['exact_sequence_and_marks']
        final = row['exact_sequence_and_marks']
        if final and not original:
            improvements.append(row['id'])
        elif original and not final:
            regressions.append(row['id'])
    return dict(delta_candidate_minus_baseline={
        kind:{metric:candidate['structural'][kind][metric]-baseline['structural'][kind][metric] for metric in counts}
        for kind in TYPES}, new_exact_controls=improvements, lost_exact_controls=regressions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=('dev','holdout'),required=True)
    p.add_argument('--manifest', type=Path,default=ROOT/'.cache/evaluation-pack/manifest.json')
    p.add_argument('--baseline',type=Path)
    p.add_argument('--candidate',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    manifest_bytes=args.manifest.read_bytes()
    manifest_hash=sha256(manifest_bytes).hexdigest()
    expected=json.loads((ROOT/'evaluation/assets-v1.json').read_bytes())['manifest_sha256']
    if manifest_hash!=expected: p.error('Frozen manifest digest mismatch')
    manifest=json.loads(manifest_bytes)
    clips={c['id']:c for c in manifest['clips'] if c['split']==args.split}
    # Clean sibling assets give an external Uthmani word for omitted positions.
    canonical_editions={c['group']:[w['published_word'] for w in c['source']['asset_segments']]
                        for c in clips.values() if c.get('operation')=='identity'}
    for c in clips.values():
        if c['source_kind']=='controlled-edit-human':
            c['published_canonical_words']=canonical_editions[c['group']]
    dependencies=['evaluate_timed_recitations.py','evaluate_transcription.py','verify_recitation_cases.py']
    out=dict(split=args.split,manifest_sha256=manifest_hash,evaluator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
             scoring_dependencies_sha256={f:sha256((ROOT/f).read_bytes()).hexdigest() for f in dependencies},notes=NOTES,runs={})
    groups={}
    for kind in sorted(manifest['summary']):
        selected=[c for c in clips.values() if c['source_kind']==kind]
        groups[kind]=dict(clips=len(selected),source_groups=len({c['group'] for c in selected}),
                          duration_s=sum(c['duration_s'] for c in selected),
                          annotation_scopes=sorted({c['label_scope'] for c in selected}))
    out['provenance']=dict(original_selection_sha256=manifest['original_selection_sha256'],
        identity_extension_sha256=manifest['identity_extension_sha256'],
        source_lock='evaluation/assets-v1.json',cohorts=groups,
        annotation_limits=['No generic nonrepeated insertion positive in the frozen controls.',
            'Learner recording IDs do not establish independent speaker identity; no full verbatim learner transcript supplied.',
            'The benchmark does not verify advanced tajweed, isolated vowel or shadda recall, or calibration by learner demographic.',
            'Typed word warnings on ordinary speech and noise are negative controls. Only reviewed pronunciation judgments can measure false phoneme alarms.'])
    for name in ('baseline','candidate'):
        path=getattr(args,name) or ROOT/f'reports/upgrade-{name}-{args.split}.json'
        data=path.read_bytes()
        report=json.loads(data)
        if report['key']['manifest_sha256']!=manifest_hash or report['key']['split']!=args.split or report.get('status')!='completed':
            p.error(f'Incomplete/incompatible saved input: {path}')
        rows=report['results']
        if len(rows)!=len(clips) or set(r['id'] for r in rows)!=set(clips):
            p.error(f'Split ID coverage mismatch: {path}')
        for r in rows:
            if r['source_kind']!=clips[r['id']]['source_kind'] or r['raw']!=''.join(t['text'] for t in r['tokens']):
                raise ValueError(f'Invalid saved token/source consistency: {r["id"]}')
        out['runs'][name]=dict(source_sha256=sha256(data).hexdigest(),engine=report['key'],**cohort([(clips[r['id']],r) for r in rows]))
    out['paired']=paired_summary(out['runs']['baseline'],out['runs']['candidate'])
    output=args.output or ROOT/f'reports/recitation-release-qa-{args.split}.json'
    output.write_text(json.dumps(out,ensure_ascii=False,indent=1)+'\n')
    print(f'Scored {len(clips)} pairs, {output}')


if __name__=='__main__':
    main()
