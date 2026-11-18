"""Isolated, one-pass ASR comparison on known-order development audio only.

Run with --model turbo first. --download-large explicitly fetches the optional
snapshot. Then run --model large offline. Nothing here is used by the app.
"""
import argparse
from collections import Counter
from hashlib import file_digest, sha1, sha256
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np

from benchmark_recitation_pack import ModelEvidence, STRUCTURAL, digest, load_pcm, run_clip, summary
from evaluate_transcription import transcription_metrics, word_error_rate
from model_assets import (ACTIVITY_MODEL_REVISION, BASE, TRANSCRIPTION_MODEL_ID,
                          TRANSCRIPTION_MODEL_REVISION, hub_cache_directory)
from verify_recitation_cases import normalized_words

LARGE_ID = 'openai/whisper-large-v3'
LARGE_REV = '06f233fe06e710322aca913c1bc4249a0d71fce1'
# Pinned upstream git blob identifiers for metadata. The weight is LFS SHA-256.
LARGE_FILES = {
    'model.safetensors': (3087130976, 'a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95'),
    'config.json': (1272, '14c6c8cf48b64ebb1cb8b637e2b0fab3a9774972'),
    'added_tokens.json': (34648, '1b33526d33aaa60d79f78ae8651dae50b730185a'),
    'generation_config.json': (3903, 'f3294dfe3654ac4e362570867369ec48104af59f'),
    'merges.txt': (493869, '6038932a2a1f09a66991b1c2adae0d14066fa29e'),
    'normalizer.json': (52666, 'dd6ae819ad738ac1a546e9f9282ef325c33b9ea0'),
    'preprocessor_config.json': (340, '931c77a740890c46365c7ae0c9d350ba3cca908f'),
    'special_tokens_map.json': (2072, '3c18e97162d572f2b1d62e7e1c7b805f1be11dbe'),
    'tokenizer.json': (2480617, '3a5e2ba63acdcac9a19ba56cf9bd27f185bfff61'),
    'tokenizer_config.json': (282843, '06ffdc8308eae6bb7bd1fdd81e94b0a881a539ab'),
    'vocab.json': (1036558, '0f3456460629e21d559c6daa23ab6ce3644e8271'),
}


def large_snapshot(download):
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(LARGE_ID, revision=LARGE_REV, cache_dir=hub_cache_directory(),
                                  allow_patterns=list(LARGE_FILES), local_files_only=not download))
    for name,(size,oid) in LARGE_FILES.items():
        item = path/name
        if item.stat().st_size != size:
            raise ValueError('Invalid snapshot size: ' + name)
        if len(oid)==64:
            with item.open('rb') as f:
                actual = file_digest(f,'sha256').hexdigest()
        else:
            actual = sha1(f'blob {size}\0'.encode()+item.read_bytes()).hexdigest()
        if actual != oid:
            raise ValueError('Invalid snapshot digest: ' + name)
    return path


def development_inputs(manifest):
    doc = json.loads(manifest.read_text())
    if not doc['complete']:
        raise ValueError('Wait for the frozen fixture pack to complete')
    for clip in doc['clips']:
        if clip['split']!='dev' or clip['source_kind'] not in {'controlled-edit-human','noise'}:
            continue
        audio = manifest.parent/clip['audio']
        if sha256(audio.read_bytes()).hexdigest()!=clip['sha256']:
            raise ValueError('Audio digest mismatch: ' + clip['id'])
        wave = load_pcm(audio)
        operation = 'noise' if clip['source_kind']=='noise' else clip['operation']
        if operation=='identity':
            operation = 'clean'
        yield clip['id'], operation, wave, clip.get('spoken_text') or '', clip.get('canonical_text') or ''


def load_service(args):
    from transcription import TranscriptionService
    service = TranscriptionService(device=args.device)
    if args.model=='large':
        # Use the very same service loading/warmup/decoding/VAD logic, with
        # only the local verified snapshot substituted in this isolated process.
        with patch('transcription.ensure_transcription_model', return_value=str(large_snapshot(False))):
            service.load()
    else:
        service.load()
    if not service.ready:
        raise RuntimeError(service.error)
    service.warm(np.zeros(32000,dtype=np.float32))
    return service


def run(args):
    from transcription import GENERATION_OPTIONS
    service = load_service(args)
    rows = []
    for label,operation,wave,spoken,canonical in development_inputs(args.manifest):
        if args.initial_windows:
            if operation != 'clean':
                continue
            wave=wave[:32768]
        if len(wave)>30*16000:
            raise ValueError('One-pass diagnostic needs clips under 30 seconds')
        start = time.perf_counter()
        prediction = service.transcribe(wave)
        elapsed = time.perf_counter()-start
        raw = prediction['text']
        metrics = transcription_metrics(spoken,canonical,raw)
        lexical = word_error_rate(normalized_words(spoken),normalized_words(raw))
        rows.append(dict(id=label, operation=operation, pcm_sha256=sha256(wave.tobytes()).hexdigest(),
                         seconds=len(wave)/16000, runtime_seconds=round(elapsed,4),
                         spoken=spoken, canonical=canonical, raw_asr=raw, word_chunks=prediction['words'],
                         activity=prediction['audio_activity']['status'], lexical=lexical,
                         wrong_word_metrics=metrics))
    if args.initial_windows:
        ordered=sorted(r['runtime_seconds'] for r in rows)
        if not ordered: raise ValueError('No development identity recordings available')
        report=dict(model=LARGE_ID if args.model=='large' else TRANSCRIPTION_MODEL_ID,device=service.device,
                    revision=LARGE_REV if args.model=='large' else TRANSCRIPTION_MODEL_REVISION,
                    first_input_seconds=32768/16000,clip_count=len(rows),
                    p50_asr_and_vad_s=statistics.median(ordered),
                    p95_asr_and_vad_s=float(np.percentile(ordered,95)),
                    limitation='Warm offline VAD and ASR only; excludes capture, queue, WebSocket and paint.',
                    results=[{k:r[k] for k in ['id','seconds','runtime_seconds','pcm_sha256','raw_asr','activity']}
                             for r in rows])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(report['p50_asr_and_vad_s'],report['p95_asr_and_vad_s'])
        return
    edited = [r for r in rows if r['operation'] not in {'noise','clean'}]
    clean = [r for r in rows if r['operation']=='clean']
    speech = edited+clean
    wrong = sum(r['wrong_word_metrics']['wrong_spoken_words'] for r in edited)
    retained = sum(r['wrong_word_metrics']['retained_wrong_spoken_words'] for r in edited)
    report = dict(model=LARGE_ID if args.model=='large' else TRANSCRIPTION_MODEL_ID,
                  revision=LARGE_REV if args.model=='large' else TRANSCRIPTION_MODEL_REVISION,
                  device=service.device, dtype=str(service.model.dtype), generation=GENERATION_OPTIONS,
                  vad_revision=ACTIVITY_MODEL_REVISION, manifest_sha256=sha256(args.manifest.read_bytes()).hexdigest(),
                  evaluation='Single independent decode on entire dev fixture, no window merge or Quran prompts.',
                  limitation='Edited human words test the exact construction only. Full-context errors here precede merging. Live windows may introduce additional errors.',
                  summary=dict(control_exact=sum(r['lexical']['errors']==0 for r in edited),controls=len(edited),
                               clean_exact=sum(r['lexical']['errors']==0 for r in clean),clean=len(clean),
                               wrong_retained=retained,wrong_words=wrong,
                               lexical_word_errors=sum(r['lexical']['errors'] for r in speech),
                               lexical_reference_words=sum(r['lexical']['reference_words'] for r in speech),
                               noise_words=sum(len(normalized_words(r['raw_asr'])) for r in rows if r['operation']=='noise'),
                               speech_median_runtime_seconds=statistics.median(r['runtime_seconds'] for r in speech)),
                  results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['summary']),flush=True)


def run_growing(args):
    from quran_corpus import QuranCorpus
    from evaluate_transcription import token_observations
    from model_assets import MODEL_REVISION
    fixed=ModelEvidence(BASE/'.cache/evaluation-evidence',args.device)
    service=load_service(args) if args.model=='large' else None
    revision=LARGE_REV if service else TRANSCRIPTION_MODEL_REVISION
    manifest_digest=digest(args.manifest)
    if manifest_digest!=json.loads((BASE/'evaluation/assets-v1.json').read_text())['manifest_sha256']:
        raise ValueError('Development manifest differs from frozen assets')
    corpus=QuranCorpus()
    corpus.prepare_live()
    cache=BASE/'.cache/development-large-outputs'
    cache.mkdir(exist_ok=True)
    unique={}
    uses=[]
    def evidence(samples):
        pcm=samples.tobytes()
        turbo_key=sha256('|'.join(fixed.parts).encode()+pcm).hexdigest()
        # Missing evidence is an error. Never silently run a new Muaalem pass
        # for one side; every frame and model revision must be identical.
        original=json.loads((fixed.cache/(turbo_key+'.json')).read_text())
        large_parts=list(fixed.parts)
        large_parts[1]=LARGE_REV
        large_key=sha256('|'.join(large_parts).encode()+pcm).hexdigest()
        uses.append(turbo_key)
        if turbo_key not in unique:
            unique[turbo_key]=dict(samples=len(samples),large_key=large_key,
                phone_sha256=sha256(json.dumps(original['phones'],ensure_ascii=False,sort_keys=True).encode()).hexdigest())
        if service is None:
            return original
        path=cache/(large_key+'.json')
        if path.exists():
            decoded=json.loads(path.read_text())
        else:
            began=time.perf_counter()
            heard=service.transcribe(samples)
            checked=service.recover_repetitions(heard,original['phones'],samples)
            elapsed=time.perf_counter()-began
            decoded=dict(turbo_key=turbo_key,large_key=large_key,seconds=elapsed,heard=checked)
            if checked is not heard:
                decoded['preview']=heard
            temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(decoded,ensure_ascii=False))
            temp.replace(path)
        if (decoded['turbo_key']!=turbo_key or decoded['large_key']!=large_key
                or decoded['heard']['audio_activity']!=original['heard']['audio_activity']):
            raise RuntimeError('Model cache key or VAD observations differ across paired windows')
        unique[turbo_key]['seconds']=decoded['seconds']
        reading=dict(heard=decoded['heard'],phones=original['phones'])
        if 'preview' in decoded:
            reading['preview']=decoded['preview']
        return reading
    rows=[]
    for clip in json.loads(args.manifest.read_text())['clips']:
        if clip['split']!='dev' or clip['source_kind'] not in {'controlled-edit-human','noise'}:
            continue
        media=args.manifest.parent/clip['audio']
        if digest(media)!=clip['sha256']:
            raise ValueError('Audio digest mismatch')
        row=run_clip(clip,media,evidence,corpus)
        if clip['source_kind']=='controlled-edit-human':
            structural_tokens=[dict(t,state='difference' if t['state']=='difference' and
                                    any(e.get('type') in STRUCTURAL for e in t['errors']) else 'unflagged')
                               for t in row['tokens']]
            _,flagged=token_observations(structural_tokens)
            row['word_detection']=transcription_metrics(clip['spoken_text'],clip['canonical_text'],row['raw'],
                error_spoken_word_indices=clip['error_spoken_word_indices'],predicted_error_word_indices=flagged)['spoken_word_error_detection']
        rows.append(row)
    metrics=summary(rows)
    counts={k:sum(r['word_detection'][k] for r in rows if 'word_detection' in r)
            for k in ('true_positives','false_positives','false_negatives')}
    elapsed=sum(e.get('seconds',0) for e in unique.values())
    audio=sum(e['samples'] for e in unique.values())/16000
    report=dict(model=revision,muaalem=MODEL_REVISION,device=args.device,
                tracker_sha256=digest(BASE/'transcript_tracking.py'),corpus_sha256=digest(BASE/'quran_corpus.py'),
                scoring_sha256=digest(BASE/'benchmark_recitation_pack.py'),
                manifest_sha256=manifest_digest,window_calls=len(uses),distinct_windows=len(unique),
                paired_keys=unique,summary=metrics,scored_structural_words=counts,
                asr_and_vad_unique_compute_s=elapsed if service else None,unique_window_audio_s=audio,
                asr_and_vad_window_rtf=elapsed/audio if service else None,
                zero_reference_word_noises=sum(r['out_of_quran']['max_words']==0 for r in rows if 'out_of_quran' in r),
                note='Frozen dev inputs, automatic location, identical CTC frames and tracker for both sides. Spoken-word detection covers structural errors. Gaps scored separately in typed metrics; no human pronunciation gold. Cached Turbo timings unavailable for these specific windows.',
                results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False))
    print(counts,'window RTF',report['asr_and_vad_window_rtf'],flush=True)


def compare(paths, live_report, output):
    first,second = (json.loads(path.read_text()) for path in paths)
    if 'paired_keys' in first:
        for key in ('device','muaalem','tracker_sha256','corpus_sha256','scoring_sha256','manifest_sha256','window_calls','distinct_windows'):
            if first[key]!=second[key]:
                raise ValueError(f'Incremental comparisons differ on {key}')
        if first['paired_keys'].keys()!=second['paired_keys'].keys():
            raise ValueError('Incremental window sets differ')
        for window,base in first['paired_keys'].items():
            if any(base[k]!=second['paired_keys'][window][k] for k in ('phone_sha256','samples','large_key')):
                raise ValueError('Incremental model evidence differs')
        def describe_windows(payload):
            return {key:payload[key] for key in ('model','device','scored_structural_words','asr_and_vad_window_rtf',
                        'zero_reference_word_noises')} | {'accuracy':payload['summary']['overall']}
        report=dict(first=describe_windows(first),second=describe_windows(second),
                    manifest_sha256=first['manifest_sha256'],distinct_windows=first['distinct_windows'],
                    total_windows=first['window_calls'],comparison='Identical frozen dev fixtures, time windows, CTC frames, VAD and tracking source hashes verified.',
                    limitation=first['note'])
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        return
    if (first['manifest_sha256'] != second['manifest_sha256'] or first['generation']!=second['generation']
            or first['vad_revision']!=second['vad_revision'] or first['device']!=second['device']):
        raise ValueError('The paired reports must use identical audio, decoding, VAD and device')
    rows = {r['id']:r for r in second['results']}
    if rows.keys()!={r['id'] for r in first['results']}:
        raise ValueError('The model reports contain different fixtures')
    live={}
    if live_report:
        live={r['id']:r for r in json.loads(live_report.read_text())['results'] if r['split']=='dev'}
    pairs,counts,baseline=[],Counter(),Counter()
    for a in first['results']:
        b=rows[a['id']]
        if a['pcm_sha256']!=b['pcm_sha256'] or a['spoken']!=b['spoken'] or a['operation']!=b['operation']:
            raise ValueError('Unpaired audio or label: '+a['id'])
        if a['operation'] in {'clean','noise'}:
            continue
        exact_a,exact_b=a['lexical']['errors']==0,b['lexical']['errors']==0
        pair=('both_exact' if exact_a and exact_b else 'both_wrong' if not exact_a and not exact_b
              else 'first_only_exact' if exact_a else 'second_only_exact')
        counts[pair]+=1
        row=dict(id=a['id'],spoken=a['spoken'],first_raw=a['raw_asr'],second_raw=b['raw_asr'],result=pair)
        if a['id'] in live:
            raw=live[a['id']]['raw']
            exact_live=normalized_words(raw)==normalized_words(a['spoken'])
            baseline[f'full_{"exact" if exact_a else "wrong"}_live_{"exact" if exact_live else "wrong"}']+=1
            row.update(streaming_baseline_raw=raw,streaming_baseline_exact=exact_live)
        pairs.append(row)
    def describe(payload):
        speech=[r for r in payload['results'] if r['operation']!='noise']
        wall=sum(r['runtime_seconds'] for r in speech)
        duration=sum(r['seconds'] for r in speech)
        return dict(model=payload['model'],revision=payload['revision'],summary=payload['summary'],
                    one_pass_speech_wall_s=round(wall,4),one_pass_speech_audio_s=round(duration,4),
                    real_time_factor=round(wall/duration,4))
    report=dict(first=describe(first),second=describe(second),
                manifest=first['manifest_sha256'],common_device=first['device'],paired_edit_counts=dict(counts),
                first_full_clip_vs_streamed_edits=dict(baseline),pairs=pairs,
                limitations=['Known-order development edits, no independent human population score.',
                             'Runtimes describe the two sequential processes on this Mac under prevailing load.',
                             'Incorrect complete-clip output demonstrates an ASR-level error for this input. Different incremental outputs may also reflect acoustic context or merge decisions.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(counts)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-large',action='store_true')
    parser.add_argument('--compare',nargs=2,type=Path,metavar=('FIRST_REPORT','SECOND_REPORT'))
    parser.add_argument('--stream-report',type=Path)
    parser.add_argument('--model',choices=['turbo','large'],default='turbo')
    parser.add_argument('--manifest',type=Path,default=BASE/'.cache/evaluation-pack/manifest.json')
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--initial-windows',action='store_true',help='Time 2.048 seconds of each clean dev control')
    parser.add_argument('--growing',action='store_true',help='Replay paired dev growing windows, sharing exact cached Muaalem evidence')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.download_large:
        print(large_snapshot(True))
    elif args.output is None:
        parser.error('--output is required when evaluating')
    elif args.compare:
        compare(args.compare,args.stream_report,args.output)
    elif args.growing:
        run_growing(args)
    else:
        run(args)
