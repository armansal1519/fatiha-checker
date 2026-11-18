"""Evaluate real v2 ASR against a user-supplied, human-labelled local manifest.

Example manifest (audio paths are relative to the manifest):
{"schema_version": 1, "clips": [{"id": "learner-01", "audio": "clip.wav",
 "spoken_text": "لفظ غلط", "canonical_text": "لفظ صحيح",
 "surah": 1, "ayah": 5, "error_spoken_word_indices": [1],
 "spoken_word_times_ms": [[0, 300], [300, 700]]}]}

Only spoken_text and canonical_text are mandatory labels. Error indices and
word times are optional and must be human annotations, not model predictions.
The tool never substitutes the canonical text for raw ASR and never creates a
synthetic dataset when real recordings are absent. Audio stays on localhost.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import unicodedata

from verify_stream_runtime import (RATE, RecordingSource, StreamRunError, check_health, local_base_url,
                                   percentile_summary, stream_v2)


def words(text, *, remove_diacritics=False):
    """Unicode word tokenization; preserve all Arabic letters and raw display text.

    Main WER keeps diacritics. A separately named secondary WER removes marks
    and tatweel only; it never folds hamza/alif or changes a spoken letter.
    """
    tokens=[]
    current=[]
    for character in unicodedata.normalize('NFC',text):
        category=unicodedata.category(character)[0]
        if remove_diacritics and (category=='M' or character=='ـ'):
            continue
        if category in {'L','M','N'}:
            current.append(character)
        elif current:
            tokens.append(''.join(current))
            current=[]
    if current:
        tokens.append(''.join(current))
    return tokens


def align_words(reference,hypothesis):
    """Deterministic minimal word edit alignment, with positional repeats kept."""
    n,m=len(reference),len(hypothesis)
    costs=[[0]*(m+1) for _ in range(n+1)]
    moves=[[None]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        costs[i][0]=i
        moves[i][0]='delete'
    for j in range(1,m+1):
        costs[0][j]=j
        moves[0][j]='insert'
    for i in range(1,n+1):
        for j in range(1,m+1):
            equal=reference[i-1]==hypothesis[j-1]
            candidates=[(costs[i-1][j-1]+(not equal),'equal' if equal else 'replace'),
                        (costs[i-1][j]+1,'delete'),(costs[i][j-1]+1,'insert')]
            costs[i][j],moves[i][j]=min(candidates,key=lambda pair:pair[0])
    i,j=n,m
    operations=[]
    while i or j:
        operation=moves[i][j]
        ri=i-1 if operation!='insert' else None
        hi=j-1 if operation!='delete' else None
        operations.append(dict(operation=operation,reference_index=ri,hypothesis_index=hi))
        if operation!='insert': i-=1
        if operation!='delete': j-=1
    operations.reverse()
    return operations


def word_error_rate(reference,hypothesis):
    alignment=align_words(reference,hypothesis)
    counts={name:sum(item['operation']==name for item in alignment) for name in ('replace','delete','insert')}
    errors=sum(counts.values())
    return dict(reference_words=len(reference),hypothesis_words=len(hypothesis),
                substitutions=counts['replace'],deletions=counts['delete'],insertions=counts['insert'],
                errors=errors,wer=errors/len(reference) if reference else (0.0 if not hypothesis else None))


def transcription_metrics(spoken_text,canonical_text,asr_text,*,error_spoken_word_indices=None,
                          predicted_error_word_indices=None):
    """Compare against human spoken words, independently of expected Quran text."""
    # Word-retention compares lexical letters, not Quran display diacritics.
    # Raw WER below remains strict; vowel/shadda errors need the optional human
    # phonetic/error labels rather than an invented orthographic ASR verdict.
    retention=lambda text:[w.replace('ٱ','ا') for w in words(text,remove_diacritics=True)]
    spoken,canonical,hypothesis=retention(spoken_text),retention(canonical_text or ''),retention(asr_text)
    canonical_spoken=align_words(canonical,spoken) if canonical_text is not None else []
    spoken_asr=align_words(spoken,hypothesis)
    asr_by_spoken={op['reference_index']:op for op in spoken_asr if op['reference_index'] is not None}
    wrong={}
    omitted=set()
    substitutions=0
    for operation in canonical_spoken:
        kind=operation['operation']
        if kind=='replace':
            wrong[operation['hypothesis_index']]=canonical[operation['reference_index']]
            substitutions+=1
        elif kind=='insert':
            wrong[operation['hypothesis_index']]=None
        elif kind=='delete':
            omitted.add(operation['reference_index'])
    retained=canonicalized=extra_retained=extra_deleted=0
    for index,expected in wrong.items():
        observed=asr_by_spoken[index]
        if observed['operation']=='equal':
            retained+=1
            if expected is None: extra_retained+=1
        elif expected is not None and observed['hypothesis_index'] is not None and hypothesis[observed['hypothesis_index']]==expected:
            canonicalized+=1
        elif expected is None and observed['operation']=='delete':
            extra_deleted+=1
    generated_omissions=sum(op['operation']=='equal' and op['reference_index'] in omitted
                            for op in align_words(canonical,hypothesis))
    result=dict(
        canonical_reference_available=canonical_text is not None,
        strict_spoken_wer=word_error_rate(words(spoken_text),words(asr_text)),
        diacritic_insensitive_spoken_wer=word_error_rate(words(spoken_text,remove_diacritics=True),
                                                        words(asr_text,remove_diacritics=True)),
        wrong_spoken_words=len(wrong),retained_wrong_spoken_words=retained,
        wrong_word_retention=retained/len(wrong) if wrong else None,
        eligible_spoken_substitutions=substitutions,canonicalized_substitutions=canonicalized,
        silent_canonical_correction_rate=canonicalized/substitutions if substitutions else None,
        extra_spoken_words=sum(value is None for value in wrong.values()),
        retained_extra_spoken_words=extra_retained,deleted_extra_spoken_words=extra_deleted,
        omitted_canonical_words=len(omitted),unspoken_canonical_words_generated=generated_omissions,
        unspoken_canonical_generation_rate=generated_omissions/len(omitted) if omitted else None,
        alignment_note='Lexical retention removes diacritics/tatweel and maps Quran alif-wasla to alif for comparison only. Strict WER keeps the human spelling. Minimal edit alignments can be ambiguous for repeated phrases.',
    )
    if error_spoken_word_indices is not None:
        gold=set(error_spoken_word_indices)
        predicted=set(predicted_error_word_indices or [])
        hyp_to_spoken={op['hypothesis_index']:op['reference_index'] for op in spoken_asr
                       if op['hypothesis_index'] is not None}
        detected={hyp_to_spoken[h] for h in predicted if h in hyp_to_spoken and hyp_to_spoken[h] is not None}
        tp=len(gold&detected)
        fn=len(gold-detected)
        fp=sum(hyp_to_spoken.get(h) not in gold for h in predicted)
        result['spoken_word_error_detection']=dict(
            true_positives=tp,false_positives=fp,false_negatives=fn,
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
            scope='Difference marks on human-labelled spoken words; omission gaps are not included in this metric.')
    else:
        result['spoken_word_error_detection']=None
    return result


def token_observations(tokens):
    pieces=[]
    flagged=set()
    for token in tokens:
        token_words=words(token.get('text',''))
        for word in token_words:
            index=len(pieces)
            timed=(len(token_words)==1 and isinstance(token.get('start_ms'),(int,float))
                   and isinstance(token.get('end_ms'),(int,float)))
            pieces.append(dict(text=word,start_ms=token.get('start_ms') if timed else None,
                               end_ms=token.get('end_ms') if timed else None))
            if token.get('state')=='difference':
                flagged.add(index)
    return pieces,flagged


def timestamp_metrics(spoken_text,observations,gold_times):
    valid=sum(isinstance(w.get('start_ms'),(int,float)) and isinstance(w.get('end_ms'),(int,float))
              and 0<=w['start_ms']<=w['end_ms'] for w in observations)
    result=dict(predicted_words=len(observations),words_with_valid_intervals=valid,
                human_timed_matches=0,start_absolute_error_ms=None,end_absolute_error_ms=None)
    if gold_times is None:
        return result
    starts,ends=[],[]
    for op in align_words(words(spoken_text),[word['text'] for word in observations]):
        if op['operation']!='equal': continue
        observed=observations[op['hypothesis_index']]
        if observed['start_ms'] is None or observed['end_ms'] is None: continue
        expected=gold_times[op['reference_index']]
        starts.append(abs(observed['start_ms']-expected[0]))
        ends.append(abs(observed['end_ms']-expected[1]))
    result.update(human_timed_matches=len(starts),start_absolute_error_ms=percentile_summary(starts),
                  end_absolute_error_ms=percentile_summary(ends))
    return result


def has_spoken_gold(clip):
    """A published target ayah does not describe everything a learner actually said."""
    if clip.get('source_kind') == 'natural-learner' and clip.get('label_scope') != 'human_verbatim':
        return False
    if clip.get('label_scope') in {'canonical_target_only', 'reviewed_reference_sites_only'}:
        return False
    return isinstance(clip.get('spoken_text'), str) and clip.get('spoken_text_verified', True) is True


def load_manifest(path):
    path=Path(path).resolve()
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or data.get('schema_version',1)!=1 or not isinstance(data.get('clips'),list) or not data['clips']:
        raise ValueError('Manifest must contain schema_version=1 and a nonempty clips list')
    clips=[]
    identifiers=set()
    for index,item in enumerate(data['clips']):
        if not isinstance(item,dict):
            raise ValueError(f'Clip {index} must be an object')
        clip=dict(item)
        identifier=clip.get('id',f'clip-{index+1}')
        if not isinstance(identifier,str) or not identifier or identifier in identifiers:
            raise ValueError('Clip IDs must be nonempty and unique')
        identifiers.add(identifier)
        non_recitation=clip.get('source_kind') in {'noise','ordinary-speech'}
        for field in (('audio',) if non_recitation else ('audio','canonical_text')):
            if not isinstance(clip.get(field),str):
                raise ValueError(f'{identifier}: {field} must be a string')
        if non_recitation:
            clip.setdefault('canonical_text', None)
        if clip.get('spoken_text') is None and not clip.get('source_kind') and clip.get('spoken_text_verified') is not False:
            raise ValueError(f'{identifier}: supply a verbatim transcript or explicitly mark it unverified')
        if clip.get('spoken_text') is not None and not isinstance(clip['spoken_text'], str):
            raise ValueError(f'{identifier}: spoken_text must be a string or null')
        audio=Path(clip['audio']).expanduser()
        clip['audio_path']=(path.parent/audio).resolve() if not audio.is_absolute() else audio.resolve()
        if not clip['audio_path'].is_file():
            raise ValueError(f'{identifier}: local audio file not found: {clip["audio_path"]}')
        count=len(words(clip.get('spoken_text') or ''))
        labels=clip.get('error_spoken_word_indices')
        if labels is not None and (not isinstance(labels,list) or any(type(i) is not int or not 0<=i<count for i in labels)
                                   or len(labels)!=len(set(labels))):
            raise ValueError(f'{identifier}: error_spoken_word_indices must be unique zero-based spoken word indices')
        intervals=clip.get('spoken_word_times_ms')
        if intervals is not None:
            if not isinstance(intervals,list) or len(intervals)!=count:
                raise ValueError(f'{identifier}: word times must match spoken word count')
            for pair in intervals:
                if (not isinstance(pair,list) or len(pair)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pair)
                            or not 0<=pair[0]<=pair[1]<=1800000):
                    raise ValueError(f'{identifier}: invalid human word timing')
        surah,ayah=clip.get('surah'),clip.get('ayah')
        if surah is not None or ayah is not None:
            if type(surah) is not int or not 1<=surah<=114 or type(ayah) is not int or ayah<1:
                raise ValueError(f'{identifier}: surah and ayah must be supplied together')
        clip['id']=identifier
        clips.append(clip)
    return clips


def evaluate_clip(clip,base_url,manual_reference=False):
    audio_path=clip['audio_path']
    if audio_path.stat().st_size>100*1024**2:
        raise ValueError('Live evaluation files must be at most 100 MiB')
    # Discover the exact decoded duration without retaining the recording. The
    # second sequential decode is paced on the original audio clock.
    probe=RecordingSource(audio_path,RATE)
    count=0
    try:
        for frame in probe.frames:
            count+=len(frame)
            if count>1800*RATE:
                raise ValueError('Live evaluation files must be at most 30 minutes')
    finally:
        probe.frames.close()
    if clip.get('spoken_word_times_ms') and clip['spoken_word_times_ms'][-1][1]>count/RATE*1000+100:
        raise ValueError('Human word timing exceeds the decoded clip duration')
    if manual_reference and clip.get('surah') is None:
        raise ValueError('--manual-reference requires surah and ayah for every clip')
    audio=RecordingSource(audio_path,RATE)
    try:
        stream=stream_v2(base_url,audio,count,start_surah=clip.get('surah') if manual_reference else None,
                         start_ayah=clip.get('ayah') if manual_reference else None)
    finally:
        audio.frames.close()
    raw=''.join(token['text'] for token in stream['final_tokens'])
    observations,flagged=token_observations(stream['final_tokens'])
    verified = has_spoken_gold(clip)
    labels = clip.get('error_spoken_word_indices') if clip.get('error_labels_complete', True) else None
    return dict(id=clip['id'],audio=str(audio_path),duration_seconds=count/RATE,
                human_spoken_text=clip['spoken_text'] if verified else None,canonical_text=clip['canonical_text'],raw_asr_text=raw,
                reference_mode='manual' if manual_reference else 'automatic',reference=stream['reference'],
                metrics=transcription_metrics(clip['spoken_text'],clip['canonical_text'],raw,
                    error_spoken_word_indices=labels,predicted_error_word_indices=flagged) if verified else None,
                timing=timestamp_metrics(clip['spoken_text'],observations,clip.get('spoken_word_times_ms')) if verified else None,
                latency=dict(first_transcript_seconds=stream['first_transcript_seconds'],
                             first_feedback_seconds=stream['first_feedback_seconds'],wall_seconds=stream['wall_seconds'],
                             first_reviewed_seconds=stream.get('first_reviewed_seconds'),
                             ack_latency_ms=stream['ack_latency_ms']),
                final_tokens=stream['final_tokens'],gaps=stream['final_gaps'],stream_metrics=stream['done']['metrics'],
                status='evaluated' if verified else 'observed')


def aggregate_metrics(results):
    evaluated=[row for row in results if row.get('status')=='evaluated']
    denominator=sum(row['metrics']['strict_spoken_wer']['reference_words'] for row in evaluated)
    errors=sum(row['metrics']['strict_spoken_wer']['errors'] for row in evaluated)
    wrong=sum(row['metrics']['wrong_spoken_words'] for row in evaluated)
    retained=sum(row['metrics']['retained_wrong_spoken_words'] for row in evaluated)
    substituted=sum(row['metrics']['eligible_spoken_substitutions'] for row in evaluated)
    corrected=sum(row['metrics']['canonicalized_substitutions'] for row in evaluated)
    labelled=[row['metrics']['spoken_word_error_detection'] for row in evaluated
              if row['metrics']['spoken_word_error_detection'] is not None]
    tp=sum(row['true_positives'] for row in labelled)
    fp=sum(row['false_positives'] for row in labelled)
    fn=sum(row['false_negatives'] for row in labelled)
    return dict(evaluated_clips=len(evaluated),failed_clips=sum(row.get('status')=='failed' for row in results),
                observation_only_clips=sum(row.get('status')=='observed' for row in results),
                strict_spoken_words=denominator,strict_word_errors=errors,
                strict_spoken_wer=errors/denominator if denominator else None,
                wrong_spoken_words=wrong,retained_wrong_spoken_words=retained,
                wrong_word_retention=retained/wrong if wrong else None,
                eligible_spoken_substitutions=substituted,canonicalized_substitutions=corrected,
                silent_canonical_correction_rate=corrected/substituted if substituted else None,
                labelled_clips=len(labelled),
                spoken_word_error_precision=tp/(tp+fp) if tp+fp else None,
                spoken_word_error_recall=tp/(tp+fn) if tp+fn else None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest',required=True,type=Path,help='Existing local human-labelled JSON manifest.')
    parser.add_argument('--base-url',default='http://127.0.0.1:5001')
    parser.add_argument('--manual-reference',action='store_true',help='Give only the checker (not ASR) each clip\'s surah/ayah.')
    parser.add_argument('--output',type=Path,default=Path('reports/transcription-evaluation.json'))
    args=parser.parse_args(argv)
    try:
        clips=load_manifest(args.manifest)
        base=local_base_url(args.base_url)
    except (OSError,ValueError,json.JSONDecodeError) as exc:
        parser.error(str(exc))
    report=dict(kind='Human-labelled local recitation/transcription evaluation',
                created_utc=datetime.now(UTC).isoformat(),manifest=str(args.manifest.resolve()),
                limitations=['Human spoken_text is the ASR ground truth; canonical_text is never substituted for it.',
                             'Wrong-word retention is computed by minimal edit alignment and can be ambiguous for repeated phrases.',
                             'Error precision/recall is only scored when complete human spoken-word error labels are provided.',
                             'Replay timings do not measure physical microphone or browser-paint latency.'],results=[])
    try:
        report['health']=check_health(base)
        for clip in clips:
            try:
                row=evaluate_clip(clip,base,args.manual_reference)
            except (OSError,ValueError,RuntimeError,StreamRunError) as exc:
                row=dict(id=clip['id'],status='failed',error=f'{type(exc).__name__}: {exc}')
            report['results'].append(row)
            print(clip['id'],row['status'],flush=True)
        report['summary']=aggregate_metrics(report['results'])
        report['status']='evaluated' if report['summary']['failed_clips']==0 else 'incomplete'
    except Exception as exc:
        report.update(status='failed',failure=f'{type(exc).__name__}: {exc}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Evaluation:',report['status'],'report:',args.output.resolve(),flush=True)
    return 0 if report['status']=='evaluated' else 1


if __name__=='__main__':
    raise SystemExit(main())
