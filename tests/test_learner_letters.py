"""Regressions using actual model observations of public recitation recordings."""
import json
from pathlib import Path

import pytest

from quran_corpus import QuranCorpus
from transcript_tracking import TranscriptTracker
from word_alignment import lexical

FIXTURES = json.loads(
    (Path(__file__).parent / 'fixtures/learner-letter-ctc.json').read_text()
)['cases']


@pytest.fixture(scope='module')
def corpus():
    return QuranCorpus()


@pytest.mark.parametrize('case', FIXTURES, ids=[c['id'] for c in FIXTURES])
def test_letter_judgment_stays_on_a_timed_spoken_occurrence(corpus, case):
    tracker = TranscriptTracker(corpus)
    for window in case['windows']:
        region = window['start'], window['end'], window['settled']
        tracker.update_transcript(window['asr'], *region)
        raw = [(w['id'], w['text']) for w in tracker.tokens]
        tracker.update_phonemes({'groups': window['ctc']}, *region)
        assert raw == [(w['id'], w['text']) for w in tracker.tokens]
    occurrences = [t for t in tracker.tokens if lexical(t['text']) == lexical(case['raw_word'])]
    assert len(occurrences) == 1
    token = occurrences[0]
    contrast = case['expected_letter_change']
    if contrast is None:
        # Disagreement, an ambiguous reference neighbor, or a clipped CTC onset.
        assert token['state'] == 'uncertain'
        return
    assert token['state'] == 'difference'
    reference = token['reference']
    assert (reference['surah'], reference['ayah']) == (case['surah'], case['ayah'])
    assert lexical(reference['word']) == lexical(case['reference_word'])
    letters = [(e['expected'][0], e['heard'][0]) for e in token['errors'] if e['type'] == 'letter']
    assert letters == [tuple(contrast)]


def test_dictionary_backed_uthmani_and_modern_word_anchors():
    assert lexical('كَـٰذِبَةٍ') == lexical('كاذبة')
    assert lexical('ٱلرَّحْمَـٰنِ') == lexical('الرحمن')
    assert lexical('كذبة') != lexical('كاذبة')
