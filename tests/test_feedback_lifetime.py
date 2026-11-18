"""Mocked evidence isolates verdict lifetime; it never tests model accuracy."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from transcript_tracking import RATE, TranscriptTracker


def reviewed_tokens(start_ms, first_state='match'):
    words = ['نعبد', 'نستعين']
    tokens = []
    for index, word in enumerate(words):
        state = first_state if index == 0 else 'match'
        tokens.append(dict(id=f'w{index}', text=' ' + word,
                           start_ms=start_ms + index * 1000,
                           end_ms=start_ms + index * 1000 + 800,
                           state=state,
                           errors=[dict(type='vowel', expected='ُ', heard='َ',
                                        message='Existing reviewed difference')]
                                  if state == 'difference' else [],
                           reference=dict(surah=1, ayah=5, word_index=index, word=word)))
    return tokens


def mock_alignment(tokens):
    references = [deepcopy(token['reference']) for token in tokens]
    entries = [dict(id=token['id'], text=token['text'], owner=index, source_owner=index,
                    confidence='anchored', op='match')
               for index, token in enumerate(tokens)]
    expected = [dict(owner=index, text=token['text'].strip())
                for index, token in enumerate(tokens)]
    return references, expected, dict(tokens=entries, gaps=[])


def supported_evidence(tokens, groups, own_groups, **kwargs):
    # Evidence identity is deliberately predetermined. A lifetime regression
    # must not depend on the recognizer's acoustic performance.
    return {token['id']: dict(status='supported', expected=own_groups[token['id']],
                             actual=own_groups[token['id']], alignment=[])
            for token in tokens}


def saved_gap(tokens):
    return dict(id='existing-gap', before_token_id=tokens[0]['id'],
                after_token_id=tokens[1]['id'], expected='مثال',
                reference=dict(surah=1, ayah=5, word_index=2, word='مثال'),
                message='Previously established gap fixture')


def test_old_window_cannot_revoke_future_word_matches_or_their_gap():
    tracker = TranscriptTracker(object())
    tracker.tokens = reviewed_tokens(10000)
    tracker.committed_count = len(tracker.tokens)
    tracker._history = [dict(start=0, end=12 * RATE, prediction=dict(groups=[]))]
    tracker._gaps = {'existing-gap': saved_gap(tracker.tokens)}
    before_tokens, before_gaps = deepcopy(tracker.tokens), deepcopy(tracker._gaps)
    tracker._feedback('tracking')  # Establish exactly what was already published.
    references, expected, alignment = mock_alignment(tracker.tokens)

    with patch('transcript_tracking.align_words', return_value=alignment), patch(
            'word_alignment.align_word_phone_evidence', side_effect=supported_evidence):
        # Before the fix, the old 5s endpoint made both later words appear to
        # be moving edges and erased their existing omission annotation.
        tracker._check_words(references, expected, 0, 5 * RATE, settled=False)

    assert tracker.tokens == before_tokens
    assert tracker._gaps == before_gaps
    assert tracker._pending == {}
    feedback = tracker._feedback('tracking')
    assert feedback['updates'] == []
    assert feedback['gaps'] == list(before_gaps.values())
    assert feedback['summary']['reviewed'] == 2
    assert feedback['summary']['missing'] == 1


@pytest.mark.parametrize('first_state', ['match', 'difference'])
def test_expired_cached_evidence_cannot_downgrade_older_reviewed_words(first_state):
    tracker = TranscriptTracker(object())
    tracker.tokens = reviewed_tokens(0, first_state)
    tracker.committed_count = len(tracker.tokens)
    # All retained evidence is newer than these already reviewed words. Its
    # absence for the older interval is not contradictory acoustic evidence.
    tracker._history = [dict(start=10 * RATE, end=12 * RATE,
                             prediction=dict(groups=[])) for _ in range(16)]
    tracker._gaps = {'existing-gap': saved_gap(tracker.tokens)}
    before_tokens, before_gaps = deepcopy(tracker.tokens), deepcopy(tracker._gaps)
    tracker._feedback('tracking')
    references, expected, alignment = mock_alignment(tracker.tokens)

    with patch('transcript_tracking.align_words', return_value=alignment), patch(
            'word_alignment.align_word_phone_evidence') as acoustic_assignment:
        tracker._check_words(references, expected, 10 * RATE, 12 * RATE, settled=True)

    acoustic_assignment.assert_not_called()  # No retained context overlaps them.
    assert tracker.tokens == before_tokens
    assert tracker._gaps == before_gaps
    assert tracker._pending == {}
    feedback = tracker._feedback('tracking')
    assert feedback['updates'] == []
    assert feedback['gaps'] == list(before_gaps.values())
    assert feedback['summary']['reviewed'] == 2
    assert feedback['summary']['differences'] == (first_state == 'difference')
