"""Acceptance criteria use invented event fixtures, never claims of ASR accuracy."""
from copy import deepcopy
import json

import pytest

from verify_recitation_cases import (CANONICAL, CONTROL_SPECS, assess_case, load_manifest,
                                     main, normalized_words, sha256)


def fixture(identifier):
    clip = dict(id=identifier, source_kind='controlled-edit-human', surah=1, ayah=5,
                spoken_text=CONTROL_SPECS[identifier]['spoken_text'], spoken_text_verified=True,
                canonical_text=CANONICAL,
                source=dict(source_word_sequence=CONTROL_SPECS[identifier].get('source_word_sequence')))
    tokens = [dict(id=f'w{index}', text=' ' + word, state='match', errors=[],
                   start_ms=index * 1000, end_ms=(index + 1) * 1000)
              for index, word in enumerate(clip['spoken_text'].split())]
    return clip, dict(id=identifier, mode='manual', status='completed', final_tokens=tokens,
                      raw_asr_text=''.join(token['text'] for token in tokens), gaps=[])


def verdict(result, name):
    return next(check['status'] for check in result['checks'] if check['name'] == name)


def word_error(token, heard, expected, kind, index=None):
    token.update(state='difference', errors=[dict(type=kind, heard=heard, expected=expected)])
    if index is not None:
        token['reference'] = dict(surah=1, ayah=5, word_index=index, word=expected)


def test_comparison_normalization_preserves_letters_and_repeated_words():
    assert normalized_words(' إِيَّاكَ، إياك نَعْبُدُ!') == ['اياك', 'اياك', 'نعبد']
    assert normalized_words('ٱلْحَمْدُ') == normalized_words('الحمد')
    assert normalized_words('نعبدو') != normalized_words('نعبد')
    assert normalized_words('ستعين') != normalized_words('نستعين')
    assert normalized_words('قلم') != normalized_words('كلم')


def test_correct_control_requires_all_spoken_words_but_uncertain_is_not_false_red():
    clip, row = fixture('control-word-order-correct')
    row['final_tokens'][0]['state'] = 'uncertain'
    before = deepcopy(row)
    result = assess_case(clip, row)
    assert result['status'] == 'passed'
    assert result['final_word_states']['uncertain'] == 1
    assert row == before
    row['final_tokens'].pop(1)
    row['raw_asr_text'] = ''.join(token['text'] for token in row['final_tokens'])
    result = assess_case(clip, row)
    assert verdict(result, 'spoken_sequence_preserved') == 'failed'
    assert verdict(result, 'unaffected_spoken_words_not_flagged') == 'passed'


@pytest.mark.parametrize('false_mark', ['word', 'gap'])
def test_correct_control_rejects_false_word_marks_or_omission_gaps(false_mark):
    clip, row = fixture('control-word-order-correct')
    if false_mark == 'word':
        word_error(row['final_tokens'][1], 'َ', '', 'insertion')
    else:
        row['gaps'] = [dict(expected='نعبد')]
    assert assess_case(clip, row)['status'] == 'failed'


def test_repeat_must_preserve_and_diagnose_the_second_occurrence():
    clip, row = fixture('control-repeat-nabudu')
    assert verdict(assess_case(clip, row), 'spoken_word_2_diagnosed') == 'failed'
    word_error(row['final_tokens'][2], 'نعبد', '', 'repetition')
    assert assess_case(clip, row)['status'] == 'passed'
    row['final_tokens'][2]['errors'][0]['heard'] = 'َ'
    assert verdict(assess_case(clip, row), 'spoken_word_2_diagnosed') == 'failed'
    word_error(row['final_tokens'][1], 'نعبد', '', 'insertion')
    row['final_tokens'][2].update(state='uncertain', errors=[])
    result = assess_case(clip, row)
    assert verdict(result, 'spoken_word_2_diagnosed') == 'failed'
    assert verdict(result, 'unaffected_spoken_words_not_flagged') == 'failed'


def test_a_repeat_cannot_be_hidden_inside_one_multiword_token():
    clip, row = fixture('control-repeat-nabudu')
    row['final_tokens'][1]['text'] += row['final_tokens'][2]['text']
    row['final_tokens'].pop(2)
    result = assess_case(clip, row)
    assert verdict(result, 'spoken_sequence_preserved') == 'passed'
    assert verdict(result, 'independent_word_tokens') == 'failed'
    assert verdict(result, 'spoken_word_2_diagnosed') == 'failed'


def test_swap_needs_both_full_word_pairs_and_correct_reference_slots():
    clip, row = fixture('control-swap-verbs')
    word_error(row['final_tokens'][1], 'نستعين', 'نعبد', 'substitution', 1)
    word_error(row['final_tokens'][3], 'نعبد', 'نستعين', 'transposition', 3)
    assert assess_case(clip, row)['status'] == 'passed'
    row['final_tokens'][3]['reference']['word_index'] = 1
    assert verdict(assess_case(clip, row), 'spoken_word_3_diagnosed') == 'failed'


def test_incidental_red_vowel_does_not_get_credit_for_swapped_verb():
    clip, row = fixture('control-swap-verbs')
    word_error(row['final_tokens'][1], 'نستعين', 'نعبد', 'substitution', 1)
    word_error(row['final_tokens'][3], 'َ', '', 'insertion', 1)
    result = assess_case(clip, row)
    assert verdict(result, 'spoken_word_1_diagnosed') == 'passed'
    assert verdict(result, 'spoken_word_3_diagnosed') == 'failed'
    assert result['status'] == 'failed'


def test_partial_swap_does_not_invent_a_false_positive_assignment():
    clip, row = fixture('control-swap-verbs')
    row['final_tokens'].pop(1)
    row['raw_asr_text'] = ''.join(token['text'] for token in row['final_tokens'])
    word_error(row['final_tokens'][-1], 'َ', '', 'insertion', 1)
    result = assess_case(clip, row)
    assert verdict(result, 'spoken_sequence_preserved') == 'failed'
    assert verdict(result, 'spoken_word_3_diagnosed') == 'failed'
    assert verdict(result, 'unaffected_spoken_words_not_flagged') == 'skipped'


def test_full_word_aliases_can_coexist_with_phonetic_error_detail():
    clip, row = fixture('control-swap-verbs')
    for index, heard, expected in [(1, 'نستعين', 'نعبد'), (3, 'نعبد', 'نستعين')]:
        word_error(row['final_tokens'][index], 'َ', 'ِ', 'substitution', index)
        row['final_tokens'][index]['errors'][0].update(heard_word=heard, expected_word=expected)
        row['final_tokens'][index]['reference']['word'] = expected
    assert assess_case(clip, row)['status'] == 'passed'


def omission_gap(row):
    return dict(id='g1', expected='نَعْبُدُ', before_token_id=row['final_tokens'][0]['id'],
                after_token_id=row['final_tokens'][1]['id'],
                reference=dict(surah=1, ayah=5, word_index=1, word='نَعْبُدُ'))


def test_omission_is_a_gap_between_two_reviewed_adjacent_spoken_anchors():
    clip, row = fixture('control-omit-nabudu')
    assert verdict(assess_case(clip, row), 'internal_omission_with_two_anchors') == 'failed'
    row['gaps'] = [omission_gap(row)]
    assert assess_case(clip, row)['status'] == 'passed'
    row['final_tokens'][0]['state'] = 'uncertain'
    assert verdict(assess_case(clip, row), 'internal_omission_with_two_anchors') == 'failed'
    row['final_tokens'][0]['state'] = 'match'
    row['gaps'][0]['after_token_id'] = row['final_tokens'][2]['id']
    assert verdict(assess_case(clip, row), 'internal_omission_with_two_anchors') == 'failed'


def test_generating_the_omitted_word_or_extra_gaps_fails_the_omission_control():
    clip, row = fixture('control-omit-nabudu')
    row['gaps'] = [omission_gap(row), dict(expected='كلمة', before_token_id='w0', after_token_id='w1')]
    assert verdict(assess_case(clip, row), 'no_unexpected_omission_gaps') == 'failed'
    clip, row = fixture('control-omit-nabudu')
    row['final_tokens'].insert(1, dict(id='invented', text=' نعبد', state='match', errors=[]))
    row['raw_asr_text'] = ''.join(token['text'] for token in row['final_tokens'])
    assert verdict(assess_case(clip, row), 'spoken_sequence_preserved') == 'failed'


def test_natural_target_text_is_not_promoted_to_complete_spoken_gold():
    clip = dict(id='natural-114-4-correct', source_kind='natural-learner', spoken_text_verified=True,
                spoken_text='من شر الوسواس الخناس', canonical_text='من شر الوسواس الخناس')
    result = assess_case(clip, dict(raw_asr_text='ملك الناس إله الناس من شر الوسواس الخناس'))
    assert result['status'] == 'skipped'
    assert result['scope'] == 'observation_only'
    assert not any(check['status'] == 'failed' for check in result['checks'])


def test_whole_ayah_reread_preservation_is_secondary_without_claiming_reread_is_error():
    clip, row = fixture('control-full-ayah-reread')
    result = assess_case(clip, row)
    assert result['status'] == 'passed'
    assert result['scope'] == 'secondary_source_utterance_preservation'
    assert not any('diagnosed' in check['name'] for check in result['checks'])
    row['final_tokens'] = row['final_tokens'][:4]
    row['raw_asr_text'] = ''.join(token['text'] for token in row['final_tokens'])
    assert assess_case(clip, row)['status'] == 'failed'


def write_manifest(tmp_path, identifier='control-word-order-correct'):
    clip, _ = fixture(identifier)
    audio = tmp_path / 'source.wav'
    audio.write_bytes(b'local-test-placeholder-no-decoding')
    clip['audio'] = audio.name
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, clips=[clip],
                                       source_audio_sha256={audio.name: sha256(audio)})))
    return manifest, audio


def test_loader_verifies_original_audio_hash_and_fixed_control_provenance(tmp_path):
    manifest, audio = write_manifest(tmp_path)
    assert load_manifest(manifest)[0]['audio_hash_matches_manifest'] is True
    audio.write_bytes(b'changed-audio')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        load_manifest(manifest)
    manifest, _ = write_manifest(tmp_path)
    data = json.loads(manifest.read_text())
    data['clips'][0]['source']['source_word_sequence'] = [1, 4, 3, 2]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='provenance/order'):
        load_manifest(manifest)


def test_offline_cli_fails_a_saved_bad_result_and_preserves_history_without_network(tmp_path):
    manifest, _ = write_manifest(tmp_path)
    clip, row = fixture('control-word-order-correct')
    row['final_tokens'].pop(1)
    row['raw_asr_text'] = ''.join(token['text'] for token in row['final_tokens'])
    row['stage_history'] = [dict(event=dict(type='transcript'), raw_text_after_event=CANONICAL)]
    saved = tmp_path / 'saved.json'
    saved.write_text(json.dumps(dict(results=[row])))
    output = tmp_path / 'verified.json'
    assert main(['--manifest', str(manifest), '--input-report', str(saved), '--output', str(output),
                 '--modes', 'manual']) == 1
    report = json.loads(output.read_text())
    assert report['status'] == 'failed'
    assert report['results'][0]['stage_history'] == row['stage_history']
    assert report['summary']['failed_cases'] == 1
    assert report['audio_sources'][0]['audio_sha256'] == load_manifest(manifest)[0]['audio_sha256']
    assert 'health' not in report
    assert json.loads(saved.read_text())['results'][0] == row


def test_offline_cli_passes_good_result_and_refuses_to_overwrite_input_report(tmp_path):
    manifest, _ = write_manifest(tmp_path)
    _, row = fixture('control-word-order-correct')
    saved = tmp_path / 'saved.json'
    saved.write_text(json.dumps(dict(results=[row])))
    output = tmp_path / 'verified.json'
    assert main(['--manifest', str(manifest), '--input-report', str(saved), '--output', str(output),
                 '--modes', 'manual', '--case', row['id']]) == 0
    assert json.loads(output.read_text())['status'] == 'passed'
    with pytest.raises(SystemExit):
        main(['--manifest', str(manifest), '--input-report', str(saved), '--output', str(saved),
              '--modes', 'manual'])


def test_offline_legacy_natural_metrics_are_not_propagated_as_gold(tmp_path):
    manifest, _ = write_manifest(tmp_path)
    data = json.loads(manifest.read_text())
    data['clips'][0].update(id='natural-source-target', source_kind='natural-learner')
    manifest.write_text(json.dumps(data))
    _, row = fixture('control-word-order-correct')
    row.update(id='natural-source-target', human_spoken_text=CANONICAL,
               metrics=dict(strict_spoken_wer=dict(wer=0)), metrics_evaluated=True)
    saved = tmp_path / 'saved.json'
    saved.write_text(json.dumps(dict(results=[row])))
    output = tmp_path / 'observations.json'
    assert main(['--manifest', str(manifest), '--input-report', str(saved), '--output', str(output),
                 '--modes', 'manual']) == 0
    report = json.loads(output.read_text())
    observed = report['results'][0]
    assert report['status'] == 'observed'
    assert observed['verification']['status'] == 'skipped'
    assert observed['metrics'] is None and observed['metrics_evaluated'] is False
    assert 'human_spoken_text' not in observed
    assert observed['source_target_text'] == CANONICAL
