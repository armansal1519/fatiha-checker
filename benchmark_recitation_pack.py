"""Compare offline recognizers/trackers on frozen public fixtures.

The test clock below is deterministic. Timing/throughput acceptance uses a
separate paced browser/WS run. Only explicitly supplied evaluation fixtures
are cached. This code is never called by the live app.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from evaluate_transcription import has_spoken_gold, transcription_metrics, word_error_rate
from verify_recitation_cases import normalized_words
from verify_stream_runtime import RATE

ROOT = Path(__file__).resolve().parent
KINDS = {'repeat': 'repetition', 'swap': 'transposition', 'omit': 'omission',
         'delete': 'omission', 'extra': 'insertion', 'substitute': 'substitution'}
STRUCTURAL = {'repetition', 'transposition', 'insertion', 'substitution', 'omission'}


def load_pcm(path):
    """Evaluation can include silence and media longer than a 30-second HTTP upload."""
    from verify_stream_runtime import RecordingSource
    frames = RecordingSource(path, RATE).frames
    arrays = []
    count = 0
    try:
        for block in frames:
            count += block.size
            if count > 1800 * RATE:
                raise ValueError('Fixture exceeds maximum session duration')
            arrays.append(block)
    finally:
        frames.close()
    if not count:
        raise ValueError('No audio frames')
    pcm = np.clip(np.concatenate(arrays), -1, 1)
    if not np.isfinite(pcm).all():
        raise ValueError('Invalid audio')
    return (pcm * 32767).astype('<i2').astype(np.float32) / 32768


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def input_windows(length):
    """Same acoustic intervals for the two trackers, including complete leading context."""
    end = min(length, 2 * RATE)
    start = 0
    while True:
        yield start, end, end == length
        if end == length:
            return
        if end-start >= 8*RATE:
            start = end-2*RATE
        end = min(length, start+8*RATE, end+3*RATE)


class ModelEvidence:
    def __init__(self, cache, device):
        from model_assets import MODEL_REVISION, TRANSCRIPTION_MODEL_REVISION, ACTIVITY_MODEL_REVISION
        from recitation import RecitationService
        from transcription import TranscriptionService
        self.cache = cache
        self.speech = TranscriptionService(device)
        self.phones = RecitationService()
        self.parts = [MODEL_REVISION, TRANSCRIPTION_MODEL_REVISION, ACTIVITY_MODEL_REVISION,
                      *(digest(Path(sys.modules[m].__file__)) for m in ('transcription', 'speech_activity', 'recitation'))]
        if hasattr(self.speech,'recover_repetitions'):
            import repeat_recovery
            self.parts.append(digest(Path(repeat_recovery.__file__)))
        self.cache.mkdir(parents=True, exist_ok=True)
        self.parts.append(device)

    def __call__(self, samples):
        key = sha256('|'.join(self.parts).encode()+samples.tobytes()).hexdigest()
        path = self.cache/(key+'.json')
        if path.exists():
            return json.loads(path.read_text())
        self.speech.load()
        self.phones.load()
        if not self.speech.ready or not self.phones.ready:
            raise RuntimeError('Both offline models must be available')
        heard = self.speech.transcribe(samples)
        activity = heard.get('audio_activity', {})
        phones = (self.phones.predict_live_timed(samples) if activity.get('status')=='speech'
                  else {'audio_activity': activity, 'text': '', 'groups': []})
        item = {'heard': heard, 'phones': phones}
        verify = getattr(self.speech,'recover_repetitions',None)
        if verify and heard.get('words'):
            confirmed = verify(heard,phones,samples)
            if confirmed is not heard:
                item.update(heard=confirmed,preview=heard)
        with tempfile.NamedTemporaryFile(mode='w',dir=self.cache,delete=False) as temp:
            json.dump(item,temp,ensure_ascii=False)
        Path(temp.name).replace(path)
        return item


def typed_control_score(clip, tokens, gaps):
    from evaluate_transcription import align_words
    spoken = normalized_words(clip['spoken_text'])
    pieces = [w for t in tokens for w in normalized_words(t['text'])]
    aligned = {op['hypothesis_index']: op['reference_index'] for op in align_words(spoken, pieces)
               if op['operation'] == 'equal'}
    offsets = {}
    actual = Counter()
    marked_indices = set()
    offset = 0
    def reference_index(record):
        ref = record.get('reference') or {}
        if (ref.get('surah'), ref.get('ayah')) != (clip.get('surah'), clip.get('ayah')):
            return None
        return ref.get('word_index')
    for t in tokens:
        count = len(normalized_words(t['text']))
        anchor = aligned.get(offset) if count == 1 else None
        offsets[t['id']] = anchor
        if t.get('state') == 'difference':
            marked_indices.add(anchor)
            notices=set()
            for error in t.get('errors', []):
                kind = error['type']
                if kind in STRUCTURAL:
                    notices.add((kind, anchor, reference_index(t) if kind in {'transposition','substitution'} else None))
            actual.update(notices)
        offset += count
    for g in gaps:
        actual['omission', (offsets.get(g.get('before_token_id')), offsets.get(g.get('after_token_id'))), reference_index(g)] += 1
    gold = Counter()
    for event in clip.get('events', []):
        kind = KINDS.get(event['kind'], event['kind'])
        refs = event.get('reference_indices', [])
        if kind == 'omission':
            for idx in refs:
                left = event['after_index']
                gold[kind, (left, left+1), idx] += 1
        else:
            for i, index in enumerate(event['heard_indices']):
                ref = refs[i] if kind in {'transposition','substitution'} else None
                gold[kind,index,ref] += 1
    def count_kind(counter, kind):
        return sum(n for entry,n in counter.items() if entry[0]==kind)
    by_type = {k:dict(tp=count_kind(actual & gold,k),fp=count_kind(actual-gold,k),
                      fn=count_kind(gold-actual,k)) for k in sorted(STRUCTURAL)}
    expected_marks={index for event in clip.get('events',[]) for index in event.get('heard_indices',[])}
    return dict(exact_sequence=pieces==spoken, exact_annotations=actual==gold and marked_indices==expected_marks, by_type=by_type,
                actual=sorted(actual.elements(),key=repr), expected=sorted(gold.elements(),key=repr))


def score(clip, tokens, gaps):
    raw = ''.join(t['text'] for t in tokens)
    states = Counter(t.get('state') for t in tokens)
    result = dict(id=clip['id'],split=clip.get('split'),source_kind=clip['source_kind'],
                  group=clip.get('speaker_id') or clip.get('group') or 'unknown',
                  raw=raw,states=dict(states),tokens=tokens,gaps=gaps)
    if has_spoken_gold(clip):
        observed, target = normalized_words(raw), normalized_words(clip['spoken_text'])
        result['lexical_wer']=word_error_rate(target, observed)
        if clip.get('canonical_text') and clip['source_kind'] != 'ordinary-speech':
            result['fidelity']=transcription_metrics(clip['spoken_text'],clip['canonical_text'],raw)
        if clip['source_kind']=='controlled-edit-human':
            result['structural']=typed_control_score(clip,tokens,gaps)
    if clip['source_kind']=='professional' and clip.get('canonical_text'):
        result['canonical_agreement_proxy']=word_error_rate(normalized_words(clip['canonical_text']),normalized_words(raw))
    if clip['source_kind']=='natural-learner':
        marked = clip.get('source_annotations',{}).get('canonical_word_errors',{})
        expected = normalized_words(clip['canonical_text'])
        # Published errors are keyed by spelling, not token ID. Only a unique
        # canonical occurrence can be compared to a positional model warning.
        sites = []
        for spelling, categories in marked.items():
            indices = [i for i,w in enumerate(expected) if [w] == normalized_words(spelling)]
            for category in categories or []:
                mapped = {'Letters':'letter', 'Tashkeel':'vowel'}.get(category)
                if len(indices)!=1 or not mapped:
                    sites.append(dict(label=category, measurable=False))
                    continue
                index=indices[0]
                at_site=[t for t in tokens if (t.get('reference') or {}).get('word_index')==index
                         and (t.get('reference') or {}).get('surah')==clip['surah']
                         and (t.get('reference') or {}).get('ayah')==clip['ayah']]
                sites.append(dict(label=category, measurable=True,index=index,
                                  detected=any(t.get('state')=='difference' and
                                               any(e.get('type')==mapped for e in t.get('errors',[])) for t in at_site)))
        result['published_sites']=sites
    return result


def apply_asr_window(tracker, evidence, start, end, settled):
    """Replay the live preview and confirmation before the CTC feedback stage."""
    heard=evidence['heard']
    primary=evidence.get('preview',heard)
    if (hasattr(tracker,'preview_transcript') and primary.get('words')
            and primary.get('audio_activity',{}).get('status') not in {'no_speech','uncertain'}):
        visible=tracker.preview_transcript(primary,start,end,settled)
        if visible is not None and 'preview' not in evidence:
            tracker.commit_preview(end,settled)
            return
    tracker.update_transcript(heard,start,end,settled)


def run_clip(clip, path, evidence, corpus):
    from transcript_tracking import TranscriptTracker
    samples = load_pcm(path)
    tracker=TranscriptTracker(corpus)
    timeline=[]
    prior_prefix=[]
    for start,end,settled in input_windows(len(samples)):
        evidence_window = evidence(samples[start:end])
        heard, phones = evidence_window['heard'], evidence_window['phones']
        apply_asr_window(tracker,evidence_window,start,end,settled)
        prefix = [(t['id'], t['text']) for t in tracker.tokens[:tracker.committed_count]]
        if prefix[:len(prior_prefix)] != prior_prefix:
            raise AssertionError('Previously committed raw words changed')
        prior_prefix = prefix
        feedback=tracker.update_phonemes(phones,start,end,settled)
        ref = feedback.get('reference') or {}
        timeline.append(dict(start=start/RATE,end=end/RATE,activity=heard.get('audio_activity',{}).get('status'),
                             raw_window=heard['text'],raw=''.join(t['text'] for t in tracker.tokens),
                             reviewed=feedback['summary']['reviewed'],location=[ref.get('surah'),ref.get('ayah')]))
    result = score(clip,tracker.tokens,feedback['gaps'])
    result['timeline']=timeline
    place=[clip.get('surah'),clip.get('ayah')]
    locations=[event for event in timeline if event['location']!=[None,None]]
    correct=[event['end'] for event in locations if event['location']==place]
    wrong=[event['end'] for event in locations if event['location']!=place]
    result['location']=dict(first_correct_audio_s=next(iter(correct),None),
                            off_target=bool(wrong),
                            recovered_audio_s=next((t for t in correct if wrong and t>wrong[0]),None))
    if clip['source_kind'] in {'noise','ordinary-speech'}:
        result['out_of_quran']=dict(any_quran_verdict=any(e['reviewed']>0 for e in timeline),
                                    any_location=bool(locations),
                                    max_words=max(len(normalized_words(e['raw'])) for e in timeline))
    return result


def summary(rows):
    def weighted(items, field):
        selected=[r[field] for r in items if field in r]
        total=sum(m['reference_words'] for m in selected)
        errors=sum(m['errors'] for m in selected)
        return dict(clips=len(selected),words=total,errors=errors,rate=errors/total if total else None)
    def group_stats(items):
        states=Counter()
        for r in items:
            states.update(r.get('states',{}))
        reviewed=states['match']+states['difference']
        verified=[r['fidelity'] for r in items if 'fidelity' in r]
        def fraction(n,d):
            a,b=sum(m[n] for m in verified),sum(m[d] for m in verified)
            return dict(numerator=a,denominator=b,rate=a/b if b else None)
        structural=[r['structural'] for r in items if 'structural' in r]
        by_type={k:{m:sum(s['by_type'][k][m] for s in structural) for m in ('tp','fp','fn')}
                 for k in sorted(STRUCTURAL)}
        for counts in by_type.values():
            tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
            counts.update(precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
        sites=[s for row in items for s in row.get('published_sites',[])]
        return dict(clips=len(items), states=dict(states), reviewed=reviewed,
                    coverage=reviewed/states.total() if states.total() else None,
                    spoken_wer=weighted(items,'lexical_wer'),canonical_proxy=weighted(items,'canonical_agreement_proxy'),
                    retained_wrong=fraction('retained_wrong_spoken_words','wrong_spoken_words'),
                    silently_corrected=fraction('canonicalized_substitutions','eligible_spoken_substitutions'),
                    controlled=len(structural),exact_sequence=sum(s['exact_sequence'] for s in structural),
                    exact_sequence_and_marks=sum(s['exact_sequence'] and s['exact_annotations'] for s in structural),
                    by_error_type=by_type,
                    published_learner_sites={kind:dict(total=sum(s['label']==kind for s in sites),
                        eligible=sum(s['label']==kind and s['measurable'] for s in sites),
                        detected=sum(s['label']==kind and s.get('detected',False) for s in sites))
                        for kind in sorted({s['label'] for s in sites})})
    keys=('source_kind','group','split')
    return dict(overall=group_stats(rows), **{k:{g:group_stats([r for r in rows if r.get(k)==g])
                for g in sorted({r[k] for r in rows},key=str)} for k in keys})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--engine-root',type=Path,default=ROOT)
    parser.add_argument('--split',choices=['dev','holdout','all'],required=True)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--cache',type=Path,default=ROOT/'.cache/evaluation-evidence')
    args=parser.parse_args()
    sys.path.insert(0,str(args.engine_root.resolve()))
    from quran_corpus import QuranCorpus
    engine={str(n):digest(args.engine_root/n) for n in ['transcript_tracking.py','quran_corpus.py','word_alignment.py','transcription.py','speech_activity.py','recitation.py']}
    repeat_source=args.engine_root/'repeat_recovery.py'
    if repeat_source.is_file():
        engine[repeat_source.name]=digest(repeat_source)
    manifest=json.loads(args.manifest.read_text())
    if manifest.get('complete') is not True:
        parser.error('Benchmark requires a complete frozen manifest')
    lock_path=ROOT/'evaluation/assets-v1.json'
    if digest(args.manifest)!=json.loads(lock_path.read_text())['manifest_sha256']:
        parser.error('Manifest does not match the pinned evaluation pack')
    clips=manifest['clips']
    selected=[c for c in clips if args.split=='all' or c['split']==args.split]
    if not selected:
        parser.error('No clips in requested split')
    key={'manifest_sha256':digest(args.manifest),'engine_sha256':engine,'device':args.device,'split':args.split,
         'evaluator_sha256':digest(Path(__file__))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(key=key,method='real local models, fixed growing acoustic windows; no browser latency claim',results=[])
    if args.output.exists():
        report=json.loads(args.output.read_text())
        if report['key']!=key:
            parser.error('Report inputs differ. Use a new output file.')
    done={r['id'] for r in report['results']}
    corpus=QuranCorpus()
    evidence=ModelEvidence(args.cache,args.device)
    for clip in selected:
        if clip['id'] in done: continue
        media=args.manifest.parent/clip['audio']
        if digest(media)!=clip['sha256']:
            raise ValueError(f'Audio digest mismatch for {clip["id"]}')
        row=run_clip(clip,media,evidence,corpus)
        report['results'].append(row)
        report['status']='running'
        temp=args.output.with_suffix('.partial')
        temp.write_text(json.dumps(report,ensure_ascii=False))
        temp.replace(args.output)
        print(f'{len(report["results"])}/{len(selected)}',clip['id'],flush=True)
    report['summary']=summary(report['results'])
    report['status']='completed'
    temp=args.output.with_suffix('.partial')
    temp.write_text(json.dumps(report,ensure_ascii=False))
    temp.replace(args.output)


if __name__=='__main__':
    main()
