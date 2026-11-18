"""Replay the same local human-audio controls through the real v2 WebSocket.

The four Quran.com word-asset controls have exact, constructed spoken sequences.
Their checks distinguish transcript preservation from correct error diagnosis:
a red vowel mark does not prove that a swapped verb was detected. The duplicated
whole-ayah clip has only a secondary preservation check. Natural learner clips
remain observational because a target ayah/reviewer label is not necessarily a
complete verbatim transcript of the recording.

Examples:
  python verify_recitation_cases.py --modes automatic,manual
  python verify_recitation_cases.py --case control-repeat-nabudu --modes manual
  python verify_recitation_cases.py --input-report reports/real-recitation-live-v2.json

--input-report applies the same criteria to saved results without inference or
network requests. Otherwise WAVs are decoded in memory and replayed at capture
speed using verify_stream_runtime.stream_v2. No microphone is opened. No text
from this manifest is supplied to the transcription model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import file_digest
import json
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parent
NORMALIZATION = ('Comparison only: NFC, remove Unicode combining marks and tatweel, '
                 'fold alif/hamza forms أ إ آ ٱ to ا, and split punctuation/whitespace. '
                 'No stemming, fuzzy matching, letter deletion, canonical replacement, '
                 'or repeated-word removal. Raw model text remains unchanged.')
CANONICAL = 'إياك نعبد وإياك نستعين'
CONTROL_SPECS = {
    'control-word-order-correct': dict(kind='correct', spoken_text=CANONICAL,
                                     source_word_sequence=[1, 2, 3, 4]),
    'control-repeat-nabudu': dict(kind='repeat', spoken_text='إياك نعبد نعبد وإياك نستعين',
                                source_word_sequence=[1, 2, 2, 3, 4]),
    'control-swap-verbs': dict(kind='swap', spoken_text='إياك نستعين وإياك نعبد',
                             source_word_sequence=[1, 4, 3, 2]),
    'control-omit-nabudu': dict(kind='omit', spoken_text='إياك وإياك نستعين',
                              source_word_sequence=[1, 3, 4]),
    'control-full-ayah-reread': dict(kind='reread', spoken_text=f'{CANONICAL} {CANONICAL}'),
}
SOURCE_FILES = ('app.py', 'live_socket.py', 'transcript_tracking.py', 'transcription.py',
                'recitation.py', 'word_alignment.py', 'quran_corpus.py', 'static/live.js',
                'verify_stream_runtime.py', 'verify_recitation_cases.py')


def normalized_words(text):
    """Limited orthographic comparison, never a transformation of displayed ASR."""
    if not isinstance(text, str):
        return []
    tokens, current = [], []
    for character in unicodedata.normalize('NFC', text):
        category = unicodedata.category(character)[0]
        if category == 'M' or character == 'ـ':
            continue
        if character in 'أإآٱ':
            character = 'ا'
        if category in {'L', 'N'}:
            current.append(character)
        elif current:
            tokens.append(''.join(current))
            current = []
    if current:
        tokens.append(''.join(current))
    return tokens


def _check(name, passed, *, expected=None, observed=None, note=None):
    result = dict(name=name, status='passed' if passed else 'failed')
    if expected is not None:
        result['expected'] = expected
    if observed is not None:
        result['observed'] = observed
    if note:
        result['note'] = note
    return result


def _lexical_error(token, heard, expected, kinds):
    """Require the complete word pair, rather than incidental phoneme/vowel red."""
    if token is None or token.get('state') != 'difference':
        return False
    for error in token.get('errors', []):
        if not isinstance(error, dict) or error.get('type') not in kinds:
            continue
        # Full-word aliases permit phonetic detail to remain in heard/expected.
        actual_heard = error.get('heard_word', error.get('heard'))
        actual_expected = error.get('expected_word', error.get('expected'))
        if (isinstance(actual_heard, str) and isinstance(actual_expected, str)
                and normalized_words(actual_heard) == normalized_words(heard)
                and normalized_words(actual_expected) == normalized_words(expected)):
            return True
    return False


def _reference_matches(reference, clip, word_index, expected):
    return (isinstance(reference, dict)
            and reference.get('surah') == clip.get('surah')
            and reference.get('ayah') == clip.get('ayah')
            and reference.get('word_index') == word_index
            and normalized_words(reference.get('word')) == normalized_words(expected))


def assess_case(clip, result):
    """Pure, positional acceptance checks; no audio/model/network operations.

    A dropped word cannot acquire credit from a red mark on a surviving word.
    Unknown natural spoken variants are never filled in from canonical_text.
    """
    spec = CONTROL_SPECS.get(clip.get('id'))
    if spec is None or clip.get('source_kind') != 'controlled-edit-human':
        return dict(status='skipped', scope='observation_only', checks=[dict(
            name='complete_verbatim_ground_truth', status='skipped',
            note='A published target ayah or sparse reviewer category is not complete '
                 'verbatim ground truth. No preservation or diagnostic accuracy is scored.')])

    tokens = result.get('final_tokens', [])
    gaps = result.get('gaps', result.get('final_gaps', []))
    raw = ''.join(token.get('text', '') for token in tokens)
    expected_words = normalized_words(spec['spoken_text'])
    observed_words = normalized_words(raw)
    flat = [token for token in tokens for _ in normalized_words(token.get('text'))]
    exact = observed_words == expected_words
    ids = [token.get('id') for token in tokens]
    independent = (all(isinstance(token.get('text'), str)
                       and len(normalized_words(token['text'])) == 1 for token in tokens)
                   and all(isinstance(identifier, str) and identifier for identifier in ids)
                   and len(ids) == len(set(ids)))
    checks = [_check('spoken_sequence_preserved', exact, expected=expected_words,
                     observed=observed_words, note=NORMALIZATION),
              _check('independent_word_tokens', independent,
                     note='Each lexical word, including each repeat, needs its own token ID.')]
    if 'raw_asr_text' in result:
        checks.append(_check('raw_text_matches_tokens', result['raw_asr_text'] == raw,
                             expected=raw, observed=result['raw_asr_text']))

    def token_at(index):
        # Partial alignments of repeated phrases are ambiguous. A diagnosis
        # cannot pass a positional test until the complete sequence survives.
        return flat[index] if exact and independent else None

    def diagnostic(index, heard, expected, kinds, reference_index=None):
        token = token_at(index)
        passed = _lexical_error(token, heard, expected, kinds)
        if reference_index is not None:
            passed = passed and _reference_matches(
                token.get('reference'), clip, reference_index, expected)
        return _check(f'spoken_word_{index}_diagnosed', passed,
                      expected=dict(spoken_word_index=index, heard_word=heard,
                                    expected_word=expected, types=sorted(kinds),
                                    reference_word_index=reference_index),
                      observed=deepcopy(token),
                      note='Requires a difference mark with full heard/expected words. '
                           'An unrelated vowel flag does not satisfy this check.')

    def no_false_word_flags(indices):
        if not exact and any(token.get('state') == 'difference' for token in tokens):
            return dict(name='unaffected_spoken_words_not_flagged', status='skipped',
                        note='The changed/missing sequence prevents reliable assignment '
                             'of red marks to unaffected spoken positions. No false-positive claim is made.')
        flagged = [dict(id=token.get('id'), text=token.get('text'),
                        errors=deepcopy(token.get('errors', [])))
                   for token in tokens if token.get('state') == 'difference'
                   and (not exact or token in [token_at(index) for index in indices])]
        return _check('unaffected_spoken_words_not_flagged', not flagged, observed=flagged,
                      note='Uncertain/pending words are abstentions, not correct verdicts.')

    kind = spec['kind']
    if kind == 'correct':
        checks.append(no_false_word_flags(range(len(expected_words))))
    elif kind == 'repeat':
        checks.append(diagnostic(2, 'نعبد', '', {'insertion', 'repetition'}))
        checks.append(no_false_word_flags([0, 1, 3, 4]))
    elif kind == 'swap':
        checks.append(diagnostic(1, 'نستعين', 'نعبد', {'substitution', 'transposition'}, 1))
        checks.append(diagnostic(3, 'نعبد', 'نستعين', {'substitution', 'transposition'}, 3))
        checks.append(no_false_word_flags([0, 2]))
    elif kind == 'omit':
        before, after = token_at(0), token_at(1)
        valid_gaps = [gap for gap in gaps if before is not None and after is not None
                      and gap.get('before_token_id') == before.get('id')
                      and gap.get('after_token_id') == after.get('id')
                      and before.get('state') in {'match', 'difference'}
                      and after.get('state') in {'match', 'difference'}
                      and normalized_words(gap.get('expected')) == normalized_words('نعبد')
                      and _reference_matches(gap.get('reference'), clip, 1, 'نعبد')]
        checks.append(_check('internal_omission_with_two_anchors', len(valid_gaps) == 1,
                             expected=dict(word='نعبد', before='إياك', after='وإياك',
                                           reference_word_index=1), observed=deepcopy(gaps),
                             note='The missing word is a gap, never a spoken token. '
                                  'Both adjacent spoken anchors must be reviewed.'))
        checks.append(no_false_word_flags(range(len(expected_words))))
        checks.append(_check('no_unexpected_omission_gaps', len(gaps) == len(valid_gaps) and len(valid_gaps) <= 1,
                             observed=deepcopy(gaps)))
    if kind in {'correct', 'repeat', 'swap'}:
        checks.append(_check('no_unexpected_omission_gaps', not gaps, observed=deepcopy(gaps)))
    scope = 'secondary_source_utterance_preservation' if kind == 'reread' else 'primary_exact_word_asset_control'
    return dict(status='passed' if all(check['status'] == 'passed' for check in checks) else 'failed',
                scope=scope, checks=checks,
                source_label_caveat=(
                    'Whole utterance copied twice; its source-provided transcript was not '
                    'independently re-certified by listening. No error labels are asserted for rereading.'
                    if kind == 'reread' else
                    'Exact ordering of human-recited word assets is known by construction. '
                    'Edits are controlled tests, not naturally occurring human mistakes or a population accuracy estimate.'),
                final_word_states=dict(Counter(token.get('state', 'missing') for token in tokens)))


def sha256(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def load_manifest(path):
    """Validate local inputs and original audio hashes without trusting sparse gold."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(manifest, dict) or manifest.get('schema_version', 1) != 1
            or not isinstance(manifest.get('clips'), list) or not manifest['clips']):
        raise ValueError('Manifest must have schema_version=1 and a nonempty clips list')
    clips, identifiers = [], set()
    declared_hashes = manifest.get('source_audio_sha256', {})
    if not isinstance(declared_hashes, dict):
        raise ValueError('source_audio_sha256 must be an object')
    for item in manifest['clips']:
        if not isinstance(item, dict):
            raise ValueError('Each clip must be an object')
        clip = deepcopy(item)
        identifier = clip.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('Clip IDs must be nonempty and unique')
        identifiers.add(identifier)
        if not isinstance(clip.get('audio'), str):
            raise ValueError(f'{identifier}: audio must be a local path string')
        audio = Path(clip['audio']).expanduser()
        audio = (path.parent / audio).resolve() if not audio.is_absolute() else audio.resolve()
        if not audio.is_file():
            raise ValueError(f'{identifier}: audio file not found: {audio}')
        if audio.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f'{identifier}: evaluation clips must be at most 10 MiB and 30 seconds')
        if clip.get('spoken_text') is not None and not isinstance(clip['spoken_text'], str):
            raise ValueError(f'{identifier}: spoken_text must be a string or null')
        surah, ayah = clip.get('surah'), clip.get('ayah')
        if (surah is not None or ayah is not None) and (
                type(surah) is not int or not 1 <= surah <= 114
                or type(ayah) is not int or ayah < 1):
            raise ValueError(f'{identifier}: valid surah and ayah must be supplied together')
        spec = CONTROL_SPECS.get(identifier)
        if spec is not None:
            if (clip.get('source_kind') != 'controlled-edit-human'
                    or clip.get('spoken_text_verified') is not True
                    or normalized_words(clip.get('spoken_text')) != normalized_words(spec['spoken_text'])
                    or normalized_words(clip.get('canonical_text')) != normalized_words(CANONICAL)
                    or (surah, ayah) != (1, 5)):
                raise ValueError(f'{identifier}: declared labels do not match the fixed control specification')
            if ('source_word_sequence' in spec
                    and clip.get('source', {}).get('source_word_sequence') != spec['source_word_sequence']):
                raise ValueError(f'{identifier}: word-asset provenance/order does not match this control')
        actual_hash = sha256(audio)
        declared_hash = clip.get('audio_sha256', declared_hashes.get(audio.name))
        if declared_hash is not None and declared_hash != actual_hash:
            raise ValueError(f'{identifier}: SHA-256 mismatch; this is not the original declared audio')
        clip.update(audio_path=audio, audio_sha256=actual_hash,
                    audio_hash_matches_manifest=declared_hash is not None)
        clips.append(clip)
    return clips


def capture_case(clip, base_url, mode):
    """Real paced PCM/ACK path; recording wraps only the local client accumulator."""
    from time import perf_counter
    from unittest.mock import patch

    from audio_processing import decode_audio
    import verify_stream_runtime as runtime

    history = []
    original_accumulator = runtime.StreamAccumulator

    class RecordingAccumulator(original_accumulator):
        def accept(self, event, elapsed):
            super().accept(event, elapsed)
            if event.get('type') != 'ack':
                history.append(dict(elapsed_seconds=elapsed, event=deepcopy(event),
                                    raw_text_after_event=''.join(token['text'] for token in self.tokens),
                                    committed_count=self.committed_count))

    began = perf_counter()
    try:
        audio = decode_audio(clip['audio_path'].read_bytes(), clip['audio_path'].name)
        with patch.object(runtime, 'StreamAccumulator', RecordingAccumulator):
            stream = runtime.stream_v2(
                base_url, lambda start, end: audio[start:end], len(audio),
                start_surah=clip.get('surah') if mode == 'manual' else None,
                start_ayah=clip.get('ayah') if mode == 'manual' else None)
        result = dict(status='completed', duration_seconds=len(audio) / runtime.RATE,
                      raw_asr_text=''.join(token['text'] for token in stream['final_tokens']),
                      final_tokens=stream['final_tokens'], gaps=stream['final_gaps'],
                      reference=stream['reference'], stream=stream)
    except Exception as exc:
        result = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
        if isinstance(exc, runtime.StreamRunError):
            result['partial_stream'] = exc.report
    result.update(id=clip['id'], mode=mode, stage_history=history,
                  client_wall_seconds=round(perf_counter() - began, 4),
                  transcript_hypotheses=[item['raw_text_after_event'] for item in history
                                         if item['event'].get('type') == 'transcript'])
    return result


def summarize(rows):
    counts = Counter(row['verification']['status'] for row in rows)
    return dict(completed_runs=len(rows), runtime_failed=sum(row.get('status') == 'failed' for row in rows),
                passed_cases=counts['passed'], failed_cases=counts['failed'],
                observational_cases=counts['skipped'],
                failed_checks=[dict(id=row['id'], mode=row['mode'], check=check['name'])
                               for row in rows for check in row['verification']['checks']
                               if check['status'] == 'failed'])


def write_report(path, report):
    """Publish a complete JSON snapshot after each run, including before inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=ROOT / '.cache/real-recitation-eval/manifest.json')
    parser.add_argument('--base-url', default='http://127.0.0.1:5001')
    parser.add_argument('--modes', default='automatic,manual', help='Comma-separated automatic,manual; default both.')
    parser.add_argument('--case', dest='cases', action='append', help='Only this clip ID; repeat for several cases.')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/real-recitation-regression-v2.json')
    parser.add_argument('--input-report', type=Path,
                        help='Reassess a saved actual-WS report instead of replaying audio. Makes no network requests.')
    args = parser.parse_args(argv)
    modes = list(dict.fromkeys(part.strip() for part in args.modes.split(',') if part.strip()))
    if not modes or any(mode not in {'automatic', 'manual'} for mode in modes):
        parser.error('--modes must contain automatic and/or manual')
    try:
        clips = load_manifest(args.manifest)
        if args.cases:
            missing = set(args.cases) - {clip['id'] for clip in clips}
            if missing:
                raise ValueError('Unknown clip IDs: ' + ', '.join(sorted(missing)))
            clips = [clip for clip in clips if clip['id'] in args.cases]
        if 'manual' in modes and any(clip.get('surah') is None for clip in clips):
            raise ValueError('Manual mode requires surah and ayah for every selected clip')
        saved = None
        if args.input_report is not None:
            if args.input_report.resolve() == args.output.resolve():
                raise ValueError('--output must differ from --input-report; original results are preserved')
            original = json.loads(args.input_report.read_text(encoding='utf-8'))
            if not isinstance(original.get('results'), list):
                raise ValueError('The input report must contain a results list')
            saved = {(row['id'], row['mode']): row for row in original['results']}
            if len(saved) != len(original['results']):
                raise ValueError('The input report has duplicate case/mode rows')
            if any((clip['id'], mode) not in saved for clip in clips for mode in modes):
                raise ValueError('The input report is missing a selected case/mode')
        else:
            import verify_stream_runtime as runtime
            args.base_url = runtime.local_base_url(args.base_url)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    report = dict(schema_version=3, kind='same-audio real-WebSocket recitation regression',
                  created_utc=datetime.now(UTC).isoformat(), status='running',
                  execution_path='saved actual-WS report; no replay' if saved is not None else
                                 'in-memory local audio -> paced PCM/ACK -> real /live/ws v2 -> raw text/feedback',
                  manifest=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
                  base_url=args.base_url, modes=modes, normalization=NORMALIZATION,
                  client_observed_source_sha256={name: sha256(ROOT / name) for name in SOURCE_FILES
                                                 if (ROOT / name).is_file()},
                  source_hash_note='On-disk files at client start; not server-process attestation.',
                  limitations=[
                      'Exact word-asset controls are deliberate edits of real human audio, not natural mistakes or population accuracy.',
                      'Whole-ayah duplication uses the source transcript as a secondary preservation check; it is not independently re-certified.',
                      'Natural target ayahs/reviewer categories are observation-only; no canonical text is promoted to complete spoken gold.',
                      'Lexical normalization permits spelling/diacritic variants only; these checks do not establish phonetic/tajweed accuracy.',
                      'Transcript preservation and full-word error diagnosis are separate checks; unrelated red vowel flags earn no diagnostic credit.',
                      'Unknown/pending is abstention; absence of a false red mark is not evidence that every word was verified.',
                      'Audio is not stored by this client; the existing local source recordings remain unchanged.'],
                  audio_sources=[], results=[])
    if saved is not None:
        report.update(input_report=str(args.input_report.resolve()), input_report_sha256=sha256(args.input_report),
                      original_client_observed_source_sha256=deepcopy(original.get('client_observed_source_sha256')))
    for clip in clips:
        record = {key: deepcopy(value) for key, value in clip.items() if key != 'audio_path'}
        record['audio'] = str(clip['audio_path'])
        record['audio_bytes'] = clip['audio_path'].stat().st_size
        report['audio_sources'].append(record)
    report['summary'] = summarize([])
    write_report(args.output, report)
    exit_code = 1
    try:
        if saved is None:
            report['health'] = runtime.check_health(args.base_url)
            write_report(args.output, report)
        for clip in clips:
            for mode in modes:
                if saved is None:
                    print(f"Replaying {clip['id']} mode={mode}", flush=True)
                    row = capture_case(clip, args.base_url, mode)
                else:
                    row = deepcopy(saved[(clip['id'], mode)])
                    if row.get('audio_sha256') not in {None, clip['audio_sha256']}:
                        raise ValueError(f"{clip['id']}: saved report audio hash differs from the manifest audio")
                row.update(audio_sha256=clip['audio_sha256'], source_kind=clip.get('source_kind'),
                           label_origin=clip.get('label_origin'))
                row['verification'] = assess_case(clip, row)
                if row['verification']['scope'] == 'observation_only':
                    # A legacy report may have scored a target ayah as though it
                    # covered the complete clip. Preserve source annotations and
                    # history, but never propagate that unsupported gold claim.
                    row.pop('human_spoken_text', None)
                    row.update(metrics=None, metrics_evaluated=False,
                               lexical_reference_unknown=True,
                               source_target_text=clip.get('canonical_text'),
                               metric_scope='observation_only_incomplete_verbatim_labels')
                if row.get('status') == 'failed':
                    row['verification']['status'] = 'failed'
                    row['verification']['checks'].insert(0, _check(
                        'complete_real_websocket_run', False, observed=row.get('error', 'Stream failed')))
                report['results'].append(row)
                report['summary'] = summarize(report['results'])
                write_report(args.output, report)
                print(json.dumps(dict(id=clip['id'], mode=mode, verification=row['verification']['status'],
                                      failed_checks=[check['name'] for check in row['verification']['checks']
                                                     if check['status'] == 'failed']), ensure_ascii=False), flush=True)
        summary = report['summary']
        report['status'] = ('failed' if summary['failed_cases'] else
                            'passed' if summary['passed_cases'] else 'observed')
        exit_code = 1 if report['status'] == 'failed' else 0
    except KeyboardInterrupt:
        report.update(status='interrupted', failure='Interrupted; completed rows/history are preserved.')
        exit_code = 130
    except Exception as exc:
        report.update(status='failed', failure=f'{type(exc).__name__}: {exc}')
    finally:
        write_report(args.output, report)
    print('Recitation regression:', report['status'], 'report:', args.output.resolve(), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
