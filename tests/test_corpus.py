from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quran_corpus import QuranCorpus
from recitation import RecitationService


@pytest.fixture(scope="module")
def corpus():
    return QuranCorpus(cache_size=2)


def test_complete_hafs_corpus_metadata_and_text(corpus):
    assert len(corpus.surahs) == 114
    assert sum(item["ayah_count"] for item in corpus.surahs) == 6236
    assert corpus.surahs[1]["ayah_count"] == 286
    assert corpus.surahs[-1] == dict(surah=114, name="الناس", ayah_count=6)
    assert corpus.validate() == 6236
    assert corpus.verses(114, start=6, limit=10)[0]["text"] == "مِنَ ٱلْجِنَّةِ وَٱلنَّاسِ"


@pytest.mark.parametrize("surah,ayah", [(0,1),(115,1),(2,287),(114,7),(1,0),(True,1),(1,False)])
def test_reference_rejects_invalid_positions(corpus, surah, ayah):
    with pytest.raises(ValueError):
        corpus.reference(surah, ayah)


def test_browsing_is_lazy_and_reference_cache_is_bounded():
    corpus = QuranCorpus(cache_size=2)
    assert len(corpus.verses(2, start=280, limit=20)) == 7
    assert corpus._reference.cache_info().currsize == 0
    first = corpus.reference(112, 1)
    assert first.surah == 112 and first.ayah == 1 and first.phonetic.phonemes
    assert corpus.reference(112, 1) is first
    corpus.reference(113, 1)
    corpus.reference(114, 1)
    assert corpus._reference.cache_info().currsize == 2
    assert corpus.reference(112, 1) is not first


def test_precomputed_phonetic_search_can_find_non_fatiha_reference(corpus):
    phonemes = corpus.reference(112, 1).phonetic.phonemes
    results = corpus.search(phonemes, error_ratio=0)
    assert any(result.start.sura_idx == 112 and result.start.aya_idx == 1 for result in results)
    searcher = corpus._searcher
    assert corpus.search(phonemes, error_ratio=0)
    assert corpus._searcher is searcher
    assert corpus.search("") == []


def test_precise_analysis_accepts_other_surahs_without_changing_legacy_references(monkeypatch):
    service = RecitationService()
    assert set(service.references) == set(range(1, 8))
    ref = service.reference(114, 6)
    monkeypatch.setattr(service, "predict", lambda wave, actual: actual.phonetic.phonemes)
    result = service.analyze(np.ones(16000, dtype=np.float32), 6, surah=114)
    assert result["surah"] == 114 and result["ayah"] == 6 and result["status"] == "no_difference"
    assert ref.surah == 114


def timed_service(frame_ids, vocab):
    service = RecitationService()
    calls = []
    logits = torch.full((1, len(frame_ids), len(vocab)), -10.0)
    for index, token in enumerate(frame_ids):
        logits[0, index, token] = 10

    def processor(waves, **kwargs):
        calls.append(1)
        return {"input_values": torch.ones((1, 1))}

    service.model = SimpleNamespace(
        processor=processor, model=lambda **kwargs: ({"phonemes": logits},),
        device="cpu", dtype=torch.float32,
        multi_level_tokenizer=SimpleNamespace(id_to_vocab={"phonemes": vocab}),
    )
    service.ready = True
    return service, calls


def test_timed_ctc_groups_preserve_official_text_and_frame_intervals():
    service, calls = timed_service([0,1,1,2,0,3,3,0], {0:"",1:"ب",2:"َ",3:"ت"})
    wave = np.ones(16000, dtype=np.float32)
    result = service.predict_live_timed(wave)
    assert result == {"text":"بَت", "groups":[
        {"text":"بَ", "start_ms":125, "end_ms":500},
        {"text":"ت", "start_ms":625, "end_ms":875},
    ]}
    assert service.predict_live(wave) == "بَت"
    result["groups"][0]["start_ms"] = 999
    assert service.predict_live_timed(wave)["groups"][0]["start_ms"] == 125
    assert len(calls) == 1


def test_timing_does_not_fix_the_official_decoder_edge_behavior():
    # In quran-muaalem0.2.2 a final token following a blank is omitted.
    service, _ = timed_service([0,0,1], {0:"",1:"ب"})
    assert service.predict_live_timed(np.ones(16000, dtype=np.float32)) == {"text":"", "groups":[]}


def test_spoken_candidate_ignores_refrains_and_unknown_passages():
    from quran_corpus import QuranCorpus
    corpus=QuranCorpus()
    def find(text):
        return corpus.spoken_passage([{'text':w} for w in text.split()])
    assert find('الذي مالا وعدده')==(104,2,0)
    assert find('وأرسل طيرا عليهم أبابيل')==(105,3,0)
    assert find('فبأي آلاء ربكما تكذبان') is None
    assert find('أنا هنا مع الصديق') is None
    assert find('ولم يولد') is None


def test_raw_lexical_candidate_cannot_grade_words_without_timed_phones():
    from quran_corpus import QuranCorpus
    from transcript_tracking import TranscriptTracker
    tracker=TranscriptTracker(QuranCorpus())
    raw='الذي مالا وعدده'
    tracker.update_transcript(dict(speech=True,text=raw,words=[
        dict(text=w,start_ms=100+i*700,end_ms=700+i*700) for i,w in enumerate(raw.split())]),0,48000,True)
    before=[w['text'] for w in tracker.tokens]
    wrong=[dict(text='ح',start_ms=100+i*200,end_ms=250+i*200) for i in range(10)]
    assert tracker._spoken_location(wrong) is None
    assert [w['text'] for w in tracker.tokens]==before
    assert tracker.current_reference is None
