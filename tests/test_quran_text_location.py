"""Regression cases for display-only lookup of raw Arabic transcripts."""
import pytest
from quran_transcript import Aya

from quran_corpus import QuranCorpus


@pytest.fixture(scope="module")
def corpus():
    return QuranCorpus()


def assert_only(corpus, text, surah, ayah):
    result = corpus.text_location(text)
    assert result["status"] == "probable", (text, result)
    assert result["total_candidates"] == 1
    candidate, = result["candidates"]
    assert (candidate["surah"], candidate["ayah"]) == (surah, ayah)
    assert candidate["surah_name"] and candidate["text"]


@pytest.mark.parametrize("text, surah, ayah", [
    ("والعصر", 103, 1),  # Distinctive one-word verse.
    ("الله الصمد", 112, 2),
    ("- یومئذ، تحدث أخبارها!", 99, 4),
    ("كلا إذا بلغت التراقی", 75, 26),
    ("إذا تداينتم بدين إلى أجل مسمى", 2, 282),  # Partial long verse.
    ("الذي مالا وعدده", 104, 2),  # Unspoken interior word.
    ("وأرسل طيرا عليهم أبابيل", 105, 3),  # Transposed words.
    ("وحس لما في الصدور", 100, 10),  # Learner ASR shifts the ل across a space.
    ("وشفاء لما في الصدور", 10, 57),  # Similar phrase with a different beginning.
    ("الف لام ميم ذلك الكتاب لا ريب فيه", 2, 2),
    ("في جيدها حبل من مسد قل هو الله أحد", 112, 1),  # Cross-surah.
    ("أعوذ بالله من الشيطان الرجيم قل هو الله أحد", 112, 1),
])
def test_verse_search(corpus, text, surah, ayah):
    assert_only(corpus, text, surah, ayah)


@pytest.mark.parametrize("text, count", [
    ("فبأي آلاء ربكما تكذبان", 31),
    ("بسم الله الرحمن الرحيم", 2),
    ("ألف لام ميم", 6),
    ("ولم يكن", 6),
])
def test_ambiguous_evidence_retains_the_number_of_possible_ayahs(corpus, text, count):
    result = corpus.text_location(text)
    assert result["status"] == "ambiguous"
    assert result["total_candidates"] == count
    assert len(result["candidates"]) == min(3, count)


@pytest.mark.parametrize("text", [
    "", " 123! ", "الله", "أعوذ بالله من الشيطان الرجيم",
    "أريد أن أذهب إلى السوق", "اشتركوا في القناة", "ترجمة نانسي قنقر",
    "قل هو الله أحد وأنا أبحث عن التسجيل المفقود",
])
def test_empty_ordinary_or_unrelated_speech_has_no_current_location(corpus, text):
    assert corpus.text_location(text) == {
        "status": "none", "total_candidates": 0, "candidates": [],
    }


def test_noisy_ikhlas_location_needs_independent_words_in_next_verse(corpus):
    heard = "قل هو الله أحد الله و سمد لم يلد ولم يولاد ولم يكن".split()
    # The final conjunction of a trailing window does not advance a whole ayah.
    expected = [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4]
    for length, last_supported_ayah in enumerate(expected, start=3):
        assert_only(corpus, " ".join(heard[:length]), 112, last_supported_ayah)


@pytest.mark.parametrize("text, ayah", [
    ("يلد ولم يولد ولم", 3), ("111. لم يلد ولم يولد ولم يدر", 3),
    ("الله الصمد لم يتحق", 2),
])
def test_one_noisy_boundary_word_does_not_advance_real_recitations(corpus, text, ayah):
    assert_only(corpus, text, 112, ayah)


@pytest.mark.parametrize("surah, ayah", [(7, 144), (15, 16), (20, 19), (21, 14)])
def test_uthmani_spellings_with_different_word_boundaries(corpus, surah, ayah):
    verse = Aya(surah, ayah).get()
    assert_only(corpus, verse.uthmani, surah, ayah)
    assert_only(corpus, verse.imlaey, surah, ayah)


def test_cached_search_is_independent_of_phonetic_and_caller_state(corpus):
    result = corpus.text_location("الله الصمد")
    result["candidates"][0]["ayah"] = 100
    assert_only(corpus, "الله الصمد", 112, 2)
    assert corpus._reference.cache_info().currsize == 0
    assert corpus._searcher is None
