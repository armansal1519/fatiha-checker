"""Location is a property of a transcript revision, independent of word grades."""
import pytest

from live_socket import LiveSession
from quran_corpus import QuranCorpus
from transcript_tracking import TranscriptTracker, display_location


@pytest.fixture(scope="module")
def corpus():
    return QuranCorpus()


def reference_ids(location):
    return [(candidate["surah"], candidate["ayah"]) for candidate in location["candidates"]]


def test_typed_ikhlas_without_timings_keeps_actual_spelling_and_no_grade(corpus):
    raw = "قل هو الله أحد الله و سمد لم يلد ولم يولاد ولم يكن"
    session = LiveSession(dict(type="start", version=2, sample_rate=16000, channels=1), corpus)
    job = dict(start=0, end=128000, settled=True)
    event = session.apply_stage(dict(stage="preview", result=dict(text=raw, words=[]), job=job, timings={}))
    assert event["type"] == "transcript"
    assert reference_ids(event["display_location"]) == [(112, 4)]
    assert event["display_location"]["status"] == "probable"
    assert [word["text"] for word in event["tokens"]] == [raw]
    assert event["tokens"][0]["reference"] is None and not event["tokens"][0]["errors"]
    session.tracker.update_phonemes({"audio_activity": {"status": "no_speech"}}, 0, 128000)
    done = session.done_fields()
    assert done["display_location"] == event["display_location"] and done["reference"] is None
    assert done["metrics"]["first_text_location_ms"] is not None
    assert done["metrics"]["first_position_ms"] is None and done["summary"]["reviewed"] == 0


def test_revising_a_phrase_withdraws_a_display_only_match(corpus):
    tracker = TranscriptTracker(corpus)
    result = tracker.preview_transcript(dict(text="قد أفلح من تزكى", words=[]), 0, 32000)
    assert reference_ids(result["display_location"]) == [(87, 14)]
    result = tracker.preview_transcript(dict(text="موعد الحافلة الساعة عشرة", words=[]), 0, 40000)
    assert result["display_location"] == dict(status="none", candidates=[], total_candidates=0)
    assert tracker.current_reference is None


def test_manual_choice_wins_and_does_not_replace_shared_words(corpus):
    tracker = TranscriptTracker(corpus)
    raw = "فبأي آلاء ربكما تكذبان"
    shared = tracker.update_transcript(dict(text=raw, words=[]), 0, 40000)
    assert shared["display_location"]["status"] == "ambiguous"
    assert shared["display_location"]["total_candidates"] == 31
    event = tracker.select_reference(55, 16)
    assert event["display_location"]["status"] == "manual"
    assert reference_ids(event["display_location"]) == [(55, 16)]
    assert [word["text"] for word in tracker.tokens] == [raw]
    assert tracker.summary()["reviewed"] == 0


def test_pause_starts_a_new_text_query_and_keeps_earlier_transcript(corpus):
    tracker = TranscriptTracker(corpus)
    first = dict(text="قل هو الله أحد", start_ms=50, end_ms=1400)
    assert reference_ids(tracker.update_transcript(dict(words=[first]), 0, 32000, settled=True)["display_location"]) == [(112, 1)]
    other = dict(text="صباح الخير", start_ms=200, end_ms=750)
    event = tracker.update_transcript(dict(words=[other]), 80000, 96000)
    assert event["display_location"]["status"] == "none"
    assert [word["text"] for word in tracker.tokens] == [first["text"], other["text"]]


def test_disagreeing_models_do_not_upgrade_text_guess(corpus):
    textual = corpus.text_location("قد أفلح من تزكى")
    another = corpus.text_location("قل هو الله أحد")["candidates"][0]
    assert display_location(textual, another)["status"] == "probable"
    assert reference_ids(display_location(textual, another)) == [(87, 14)]
    assert display_location(textual, textual["candidates"][0])["status"] == "confirmed"
    shared = corpus.text_location("فبأي آلاء ربكما تكذبان")
    assert display_location(shared, shared["candidates"][0])["status"] == "ambiguous"
