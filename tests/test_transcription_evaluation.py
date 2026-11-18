"""Controlled strings and events validate metrics/protocol, never ASR accuracy."""
import json

import pytest

from evaluate_transcription import (aggregate_metrics, align_words, load_manifest,
    timestamp_metrics, token_observations, transcription_metrics, word_error_rate, words)
from verify_stream_runtime import (StreamAccumulator, local_base_url, model_stage_calls,
                                   percentile_summary, tone_bursts)


def test_wrong_spoken_word_is_the_asr_ground_truth_not_the_canonical_word():
    metrics=transcription_metrics('قل هوب الله','قل هو الله','قل هوب الله')
    assert metrics['strict_spoken_wer']['wer']==0
    assert metrics['wrong_spoken_words']==1
    assert metrics['retained_wrong_spoken_words']==1
    assert metrics['wrong_word_retention']==1
    assert metrics['silent_canonical_correction_rate']==0
    assert metrics['spoken_word_error_detection'] is None


def test_silent_canonical_substitution_is_reported_as_lost_wrong_speech():
    metrics=transcription_metrics('قل هوب الله','قل هو الله','قل هو الله')
    assert metrics['strict_spoken_wer']['wer']==pytest.approx(1/3)
    assert metrics['wrong_word_retention']==0
    assert metrics['canonicalized_substitutions']==1
    assert metrics['silent_canonical_correction_rate']==1


def test_extra_spoken_word_retention_and_deletion_are_measured_separately():
    retained=transcription_metrics('قل كلمة هو الله','قل هو الله','قل كلمة هو الله')
    dropped=transcription_metrics('قل كلمة هو الله','قل هو الله','قل هو الله')
    assert retained['extra_spoken_words']==1 and retained['retained_extra_spoken_words']==1
    assert dropped['deleted_extra_spoken_words']==1 and dropped['wrong_word_retention']==0
    assert dropped['silent_canonical_correction_rate'] is None


def test_unspoken_canonical_word_generation_is_detected_from_human_omission():
    preserved=transcription_metrics('قل الله احد','قل هو الله احد','قل الله احد')
    invented=transcription_metrics('قل الله احد','قل هو الله احد','قل هو الله احد')
    assert preserved['omitted_canonical_words']==1 and preserved['unspoken_canonical_words_generated']==0
    assert invented['unspoken_canonical_words_generated']==1
    assert invented['unspoken_canonical_generation_rate']==1


def test_absent_error_examples_do_not_create_perfect_retention_or_detection_scores():
    metrics=transcription_metrics('قل هو الله','قل هو الله','قل هو الله')
    assert metrics['wrong_word_retention'] is None
    assert metrics['silent_canonical_correction_rate'] is None
    assert metrics['unspoken_canonical_generation_rate'] is None
    assert metrics['spoken_word_error_detection'] is None


def test_main_wer_keeps_letters_and_diacritics_while_secondary_is_named_explicitly():
    metrics=transcription_metrics('قُلْ','قُلْ','قل')
    assert metrics['strict_spoken_wer']['wer']==1
    assert metrics['diacritic_insensitive_spoken_wer']['wer']==0
    assert words('أحد، احد كلم قلم')==['أحد','احد','كلم','قلم']


def test_error_precision_recall_require_human_labels_and_count_missing_asr_words():
    found=transcription_metrics('قل هوب الله','قل هو الله','قل هوب الله',
        error_spoken_word_indices=[1],predicted_error_word_indices={1,2})['spoken_word_error_detection']
    assert found['true_positives']==1 and found['false_positives']==1 and found['false_negatives']==0
    assert found['precision']==.5 and found['recall']==1
    missed=transcription_metrics('قل هوب الله','قل هو الله','قل الله',
        error_spoken_word_indices=[1],predicted_error_word_indices=set())['spoken_word_error_detection']
    assert missed['false_negatives']==1 and missed['recall']==0 and missed['precision'] is None


def test_edit_alignment_preserves_duplicate_occurrences_instead_of_deduplicating():
    alignment=align_words(['قل','هو','هو','الله'],['قل','هو','هو','الله'])
    assert len(alignment)==4 and all(op['operation']=='equal' for op in alignment)
    assert word_error_rate(['قل'],['قل','قل'])['insertions']==1
    assert word_error_rate([],[])['wer']==0
    assert word_error_rate([],['قل'])['wer'] is None


def test_token_observations_do_not_invent_times_for_multiword_fallback_text():
    observations,flagged=token_observations([
        dict(id='w1',text='قل',start_ms=0,end_ms=200,state='pending'),
        dict(id='w2',text=' هو الله',start_ms=200,end_ms=900,state='difference')])
    assert [x['text'] for x in observations]==['قل','هو','الله']
    assert observations[1]['start_ms'] is None and observations[2]['end_ms'] is None
    assert flagged=={1,2}
    timed=timestamp_metrics('قل هو الله',observations,[[0,250],[250,500],[500,900]])
    assert timed['human_timed_matches']==1
    assert timed['end_absolute_error_ms']['median']==50
    assert timestamp_metrics('قل',observations,None)['start_absolute_error_ms'] is None


def test_aggregate_wer_uses_word_weighting_and_reports_failures():
    rows=[]
    for spoken,hypothesis in [('قل','قل'),('قل هو الله','قل الله')]:
        rows.append(dict(status='evaluated',metrics=transcription_metrics(spoken,spoken,hypothesis)))
    rows.append(dict(status='failed',error='unavailable clip'))
    summary=aggregate_metrics(rows)
    assert summary['strict_spoken_wer']==.25
    assert summary['failed_clips']==1 and summary['evaluated_clips']==2
    assert summary['labelled_clips']==0 and summary['wrong_word_retention'] is None


def manifest_file(tmp_path,item):
    audio=tmp_path/'clip.wav'
    audio.write_bytes(b'not decoded in manifest validation')
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1,clips=[dict(audio='clip.wav',spoken_text='قل هو',canonical_text='قل هو',**item)])))
    return path


def test_manifest_resolves_existing_local_audio_and_accepts_optional_gold_labels(tmp_path):
    path=manifest_file(tmp_path,dict(id='sample',error_spoken_word_indices=[1],spoken_word_times_ms=[[0,100],[100,500]]))
    clip=load_manifest(path)[0]
    assert clip['audio_path']==tmp_path/'clip.wav'
    assert clip['id']=='sample'


@pytest.mark.parametrize('extra',[
    dict(error_spoken_word_indices=[2]),dict(error_spoken_word_indices=[True]),
    dict(error_spoken_word_indices=[1,1]),dict(spoken_word_times_ms=[[0,100]]),
    dict(spoken_word_times_ms=[[0,100],[200,100]]),dict(surah=1),
])
def test_manifest_rejects_invalid_human_labels_instead_of_silently_scoring_them(tmp_path,extra):
    with pytest.raises(ValueError):
        load_manifest(manifest_file(tmp_path,extra))


def accumulator():
    return StreamAccumulator(dict(type='ready',version=2,session_id='current',revision=1))


def token(token_id,text,state='pending'):
    return dict(id=token_id,text=text,start_ms=0,end_ms=300,state=state,errors=[])


def test_client_applies_raw_tail_revisions_and_metadata_to_committed_words():
    state=accumulator()
    state.accept(dict(type='transcript',session_id='current',revision=2,replace_from=0,
                      tokens=[token('w1','قل'),token('w2',' هوب')],committed_count=1),1)
    state.accept(dict(type='feedback',session_id='current',revision=3,
                      updates=[dict(id='w2',state='difference',errors=[dict(type='letter')])],gaps=[]),1.1)
    state.accept(dict(type='transcript',session_id='current',revision=4,replace_from=1,
                      tokens=[token('w3',' هب')],committed_count=1),2)
    assert [t['id'] for t in state.tokens]==['w1','w3']
    assert state.tokens[1]['errors']==[]
    state.accept(dict(type='feedback',session_id='current',revision=5,
                      updates=[dict(id='w1',state='difference',errors=[])],gaps=[]),2.1)
    assert state.tokens[0]['text']=='قل' and state.tokens[0]['state']=='difference'


def test_client_keeps_real_repeated_words_as_distinct_ids():
    state=accumulator()
    state.accept(dict(type='transcript',session_id='current',revision=2,replace_from=0,
                      tokens=[token('w1','هو'),token('w2',' هو')],committed_count=2),1)
    assert ''.join(t['text'] for t in state.tokens)=='هو هو'


@pytest.mark.parametrize('event',[
    dict(type='transcript',session_id='foreign',revision=2,replace_from=0,tokens=[],committed_count=0),
    dict(type='feedback',session_id='current',revision=1,updates=[],gaps=[]),
    dict(type='transcript',session_id='current',revision=2,replace_from=2,tokens=[],committed_count=0),
])
def test_client_verifier_rejects_stale_sessions_revisions_and_invalid_deltas(event):
    with pytest.raises(AssertionError):
        accumulator().accept(event,1)


def test_client_verifier_rejects_canonical_rewrite_and_unknown_feedback_ids():
    state=accumulator()
    state.accept(dict(type='transcript',session_id='current',revision=2,replace_from=0,
                      tokens=[token('w1','هوب')],committed_count=1),1)
    with pytest.raises(AssertionError,match='replace'):
        state.accept(dict(type='feedback',session_id='current',revision=3,
                          updates=[dict(id='w1',text='هو',state='match')],gaps=[]),2)
    state=accumulator()
    with pytest.raises(AssertionError,match='unknown'):
        state.accept(dict(type='feedback',session_id='current',revision=2,updates=[dict(id='removed',state='match')]),1)


@pytest.mark.parametrize('url',['https://example.com','http://example.com','http://localhost:5001/path','http://user@localhost:5001','http://localhost:5001?secret=x'])
def test_audio_evaluation_never_transmits_to_non_loopback_or_credential_urls(url):
    with pytest.raises(ValueError): local_base_url(url)


def test_tone_generator_is_streaming_and_summary_handles_missing_observations():
    assert local_base_url('http://127.0.0.1:5001/')=='http://127.0.0.1:5001'
    assert len(tone_bursts(0,2048))==2048
    assert not tone_bursts(3*16000,3*16000+2048).any()
    assert percentile_summary([])['p95'] is None


def test_stage_counts_require_execution_evidence_and_distinguish_vad_frames():
    timings = [
        dict(transcription_skipped=True, phonetic_skipped=True, vad_frames=63,
             transcription_ms=12, phonetic_ms=0),
        dict(transcription_skipped=False, phonetic_skipped=False, vad_frames=250),
        dict(transcription_skipped=True, phonetic_skipped=True, vad_frames=0),
        dict(transcription_ms=1000, phonetic_ms=2000),  # Legacy timing alone proves no execution.
    ]
    assert model_stage_calls(timings) == dict(asr=1, phonetic=1, vad=2,
        vad_frame_evaluations=313, unknown=dict(asr=1, phonetic=1, vad=1))


def test_synthetic_soak_passes_transport_when_vad_rejects_heavy_inference(monkeypatch, tmp_path):
    import verify_stream_runtime
    timings = [dict(transcription_skipped=True, phonetic_skipped=True,
                    vad_frames=63, transcription_ms=12, phonetic_ms=0)]
    monkeypatch.setattr(verify_stream_runtime, 'check_health', lambda base: dict(ready=True))
    monkeypatch.setattr(verify_stream_runtime, 'stream_v2', lambda *args, **kwargs: dict(
        status='passed', final_tokens=[], final_gaps=[], memory_samples=[],
        done=dict(metrics=dict(stage_timings=timings))))
    output = tmp_path / 'synthetic.json'
    assert verify_stream_runtime.main(['--duration', '.5', '--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['status'] == 'passed'
    assert report['model_stage_calls']['asr'] == report['model_stage_calls']['phonetic'] == 0
    assert report['model_stage_calls']['vad'] == 1
    assert 'does not establish' in report['throughput_note']
    assert 'verify_recitation_cases.py' in report['throughput_note']


def test_quran_display_diacritics_do_not_create_fake_wrong_word_examples():
    metrics=transcription_metrics('قل هو الله احد','قُلْ هُوَ ٱللَّهُ أَحَدٌ','قل هو الله احد')
    # The hamza in أَحَد remains a real orthographic difference; the other
    # fully-vowelled reference words do not manufacture three error examples.
    assert metrics['wrong_spoken_words']==1
    assert metrics['strict_spoken_wer']['wer']==0
    same=transcription_metrics('الحمد لله','ٱلْحَمْدُ لِلَّهِ','الحمد لله')
    assert same['wrong_spoken_words']==0 and same['wrong_word_retention'] is None
