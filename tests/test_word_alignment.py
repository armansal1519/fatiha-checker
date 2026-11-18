"""Structural and independent acoustic identity checks; no models or audio."""
from copy import deepcopy

import pytest

from word_alignment import align_words, align_word_phone_evidence, word_phone_evidence


def reference(words, surah=1, ayah=5):
    return [dict(surah=surah, ayah=ayah, word_index=i, word=w)
            for i,w in enumerate(words.split())]


def tokens(words):
    return [dict(id=f"w{i}",text=w,start_ms=i*1000,end_ms=(i+1)*1000)
            for i,w in enumerate(words.split())]


FATIHA = reference("إياك نعبد وإياك نستعين")


def test_baseline_preserves_raw_tokens_and_expected_positions():
    heard=tokens("إيَّاكَ نعبد وَإِيَّاكَ نستعين")
    heard[1]["text"]=" نعبد"
    before=deepcopy(heard)
    result=align_words(FATIHA,heard,preferred_owner=0)
    assert heard==before
    assert [(t["id"],t["text"]) for t in result["tokens"]]==[(t["id"],t["text"]) for t in heard]
    assert [t["owner"] for t in result["tokens"]]==[0,1,2,3]
    assert all(t["op"]=="match" for t in result["tokens"])
    assert result["gaps"]==[] and not result["ambiguous"]


def test_second_repeated_occurrence_is_extra_not_the_first():
    heard=tokens("إياك نعبد نعبد وإياك نستعين")
    result=align_words(FATIHA,heard,preferred_owner=0)
    assert [t["op"] for t in result["tokens"]]==["match","match","repeat","match","match"]
    extra=result["tokens"][2]
    assert extra["id"]=="w2" and extra["owner"] is None and extra["source_owner"]==1
    assert extra["confidence"]=="anchored" and result["gaps"]==[]


def test_swapped_verbs_keep_positional_expected_slots():
    result=align_words(FATIHA,tokens("إياك نستعين وإياك نعبد"),preferred_owner=0)
    assert [t["owner"] for t in result["tokens"]]==[0,1,2,3]
    assert [t["op"] for t in result["tokens"]]==["match","transpose","match","transpose"]
    assert result["tokens"][1]["source_owner"]==3
    assert result["tokens"][3]["source_owner"]==1
    assert result["tokens"][1]["reference"]["word"]=="نعبد"
    assert result["tokens"][3]["reference"]["word"]=="نستعين"
    assert result["gaps"]==[]


def test_omission_is_only_a_gap_between_present_exact_anchors():
    result=align_words(FATIHA,tokens("إياك وإياك نستعين"),preferred_owner=0)
    assert [t["owner"] for t in result["tokens"]]==[0,2,3]
    assert result["gaps"]==[dict(owner=1,before_id="w0",after_id="w1",confidence="anchored")]
    assert len(result["tokens"])==3


@pytest.mark.parametrize("heard",["إياك نعبد","وإياك نستعين","نعبد وإياك"])
def test_unread_reference_edges_are_not_omissions(heard):
    assert align_words(FATIHA,tokens(heard))["gaps"]==[]


def test_full_ayah_reread_preserves_both_cycles_without_repeat_error():
    result=align_words(FATIHA,tokens("إياك نعبد وإياك نستعين إياك نعبد وإياك نستعين"),preferred_owner=0)
    assert len(result["tokens"])==8
    assert [t["owner"] for t in result["tokens"]]==[0,1,2,3,0,1,2,3]
    assert all(t["op"]=="match" for t in result["tokens"])
    assert all(t.get("reread") for t in result["tokens"][4:])
    assert result["gaps"]==[]


def test_complete_one_word_ayah_reread_before_next_ayah():
    from quran_corpus import QuranCorpus

    corpus=QuranCorpus()
    first,following=corpus.reference(20,1),corpus.reference(20,2)
    assert first.words==["طه"]
    refs=reference(" ".join(first.words),surah=20,ayah=1)
    refs+=reference(" ".join(following.words),surah=20,ayah=2)
    heard=tokens(" ".join(first.words+first.words+following.words))
    before=deepcopy(heard)
    result=align_words(refs,heard,preferred_owner=0)
    assert heard==before
    assert [entry["owner"] for entry in result["tokens"]]==[0,0,*range(1,len(refs))]
    assert all(entry["op"]=="match" and entry["confidence"]=="anchored"
               for entry in result["tokens"])
    assert result["tokens"][1]["reread"]
    assert not result["ambiguous"] and result["gaps"]==[]


def test_generic_adjacent_transposition_and_unrelated_insertion():
    refs=reference("كتب زيد درسا نافعا",surah=2,ayah=1)
    swapped=align_words(refs,tokens("كتب درسا زيد نافعا"))
    assert [t["op"] for t in swapped["tokens"]]==["match","transpose","transpose","match"]
    inserted=align_words(refs,tokens("كتب زيد اليوم درسا نافعا"))
    assert inserted["tokens"][2]["op"]=="insert"
    assert inserted["tokens"][2]["confidence"]=="anchored"
    assert inserted["gaps"]==[]
    pair=align_words(reference("كتب زيد"),tokens("زيد كتب"))
    assert [t["op"] for t in pair["tokens"]]==["transpose","transpose"]


def test_duplicate_reference_phrase_remains_ambiguous_without_position():
    refs=reference("قال ربنا")+reference("قال ربنا",surah=2,ayah=2)
    result=align_words(refs,tokens("قال ربنا"))
    assert result["ambiguous"]
    assert all(t["confidence"]=="ambiguous" for t in result["tokens"])
    selected=align_words(refs,tokens("قال ربنا"),preferred_owner=0)
    assert not selected["ambiguous"]
    assert [t["owner"] for t in selected["tokens"]]==[0,1]


def test_one_heard_word_inside_identical_reference_run_cannot_locate_gap():
    refs=reference("قال هيهات هيهات هذا")
    heard=tokens("قال هيهات هذا")
    before=deepcopy(heard)
    result=align_words(refs,heard,preferred_owner=0)
    assert heard==before and len(result["tokens"])==3
    assert [entry["confidence"] for entry in result["tokens"]]==["anchored","ambiguous","anchored"]
    assert [entry["text"] for entry in result["tokens"]]==[token["text"] for token in heard]
    assert result["gaps"]==[]


def phone_groups(texts, bounds):
    return [dict(text=text,start_ms=a,end_ms=b) for text,(a,b) in zip(texts,bounds)]


def test_sequence_phone_partition_does_not_take_previous_words_last_phone():
    heard=tokens("نعبد نستعين")
    heard[0].update(start_ms=0,end_ms=990)
    heard[1].update(start_ms=990,end_ms=2000)
    groups=phone_groups(["نَ","ع","بُ","دُ","نَ","س","تَ","عِ","ۦۦ","ن"],
                        [(0,250),(250,500),(500,750),(750,1080),(1080,1200),
                         (1200,1300),(1300,1450),(1450,1600),(1600,1800),(1800,2050)])
    own={"w0":["نَ","ع","بُ","دُ"],"w1":["نَ","س","تَ","عِ","ۦۦ","ن"]}
    result=align_word_phone_evidence(heard,groups,own)
    assert result["w0"]["status"]==result["w1"]["status"]=="supported"
    assert result["w0"]["actual"]==own["w0"]
    assert result["w1"]["actual"]==own["w1"]
    assert not set(result["w0"]["indices"])&set(result["w1"]["indices"])


def test_identity_support_keeps_a_real_vowel_difference_for_grading():
    word=dict(id="w",text="نعبد",start_ms=0,end_ms=1000)
    groups=phone_groups(["نَ","ع","بَ","دُ"],[(0,250),(250,500),(500,750),(750,1000)])
    result=align_word_phone_evidence([word],groups,{"w":["نَ","ع","بُ","دُ"]})["w"]
    assert result["status"]=="supported" and not result["exact"]
    assert result["actual"][2]=="بَ" and result["expected"][2]=="بُ"


def test_misaligned_asr_start_cannot_hide_an_earlier_vowel_error():
    word=dict(id="w",text="إياك",start_ms=500,end_ms=1000)
    groups=phone_groups(["ءَ","ييَ","اا","كَ"],[(0,250),(250,500),(500,750),(750,1000)])
    result=align_word_phone_evidence([word],groups,{"w":["ءِ","ييَ","اا","كَ"]})["w"]
    assert result["status"]=="uncertain"
    assert result["reason"]=="incomplete_word_phone_coverage"
    assert result["actual"][0]=="ءَ"


def test_extra_phone_is_not_trimmed_to_obtain_exact_match():
    word=dict(id="w",text="نعبد",start_ms=0,end_ms=1250)
    groups=phone_groups(["نَ","ع","بُ","دُ","ه"],[(0,250),(250,500),(500,750),(750,1000),(1000,1250)])
    result=align_word_phone_evidence([word],groups,{"w":["نَ","ع","بُ","دُ"]})["w"]
    assert result["status"]=="supported" and not result["exact"]
    assert result["actual"][-1]=="ه" and len(result["indices"])==5


def test_phone_disagreement_and_missing_timing_remain_ungradable():
    word=dict(id="w",text="نعبد",start_ms=0,end_ms=1000)
    groups=phone_groups(["زَ","سَ","قَ","هَ"],[(0,250),(250,500),(500,750),(750,1000)])
    result=align_word_phone_evidence([word],groups,{"w":["نَ","ع","بُ","دُ"]})["w"]
    assert result["status"] in {"disagreement","uncertain"}
    word["start_ms"]=None
    assert align_word_phone_evidence([word],groups,{"w":["نَ","ع","بُ","دُ"]})["w"]["status"]=="uncertain"


def test_standalone_partial_suffix_never_confirms_complete_word():
    word=dict(id="w",text="إياك",start_ms=500,end_ms=1000)
    groups=phone_groups(["اا","كَ"],[(500,750),(750,1000)])
    assert word_phone_evidence(word,groups,["ءِ","ييَ","اا","كَ"])["status"]=="uncertain"
