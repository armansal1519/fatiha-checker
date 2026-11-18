"""Controlled strings test mapping/state, never human recognition accuracy."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from quran_transcript import chunck_phonemes
from quran_corpus import QuranCorpus
from transcript_tracking import TranscriptTracker


@pytest.fixture(scope='module')
def corpus():
    return QuranCorpus()


def evidence(corpus, surah=1, ayah=5, omit=(), change=None):
    ref=corpus.reference(surah,ayah)
    phone_map={}
    for ci,mapping in enumerate(ref.phonetic.mappings):
        if mapping is not None:
            for p in range(*mapping.pos): phone_map.setdefault(p,ci)
    groups=[]; pos=0
    for group in chunck_phonemes(ref.phonetic.phonemes):
        char=phone_map.get(pos,phone_map.get(pos-1,0))
        wi=next(i for i,(a,b) in enumerate(ref.word_spans) if a<=char<b)
        pos+=len(group)
        if wi not in omit: groups.append((wi,group))
    surviving=[i for i in range(len(ref.words)) if i not in omit]
    words=[dict(text=ref.words[i],start_ms=j*1000,end_ms=(j+1)*1000) for j,i in enumerate(surviving)]
    timed=[]
    for wi in surviving:
        own=[text for owner,text in groups if owner==wi]
        for k,text in enumerate(own):
            if change: text=change(text)
            j=surviving.index(wi)
            timed.append(dict(text=text,start_ms=j*1000+k*1000/len(own),end_ms=j*1000+(k+1)*1000/len(own)))
    return dict(text=' '.join(w['text'] for w in words),words=words,speech=True),dict(text=''.join(g['text'] for g in timed),groups=timed),len(words)*16000


def test_raw_misspelling_and_real_repetitions_are_not_canonicalized(corpus):
    tracker=TranscriptTracker(corpus)
    raw=dict(text='الهمد الهمد',speech=True,words=[dict(text='الهمد',start_ms=0,end_ms=900),dict(text=' الهمد',start_ms=1000,end_ms=1900)])
    event=tracker.update_transcript(raw,0,32000)
    assert [t['text'] for t in event['tokens']]==['الهمد',' الهمد']
    assert len({t['id'] for t in event['tokens']})==2


def test_tail_revision_revokes_changed_token_and_its_feedback(corpus):
    tracker=TranscriptTracker(corpus)
    tracker.update_transcript(dict(text='خطا',words=[dict(text='خطا',start_ms=1000,end_ms=1700)]),0,32000)
    old=tracker.tokens[0]['id']
    tracker.tokens[0].update(state='difference',errors=[dict(type='letter')])
    result=tracker.update_transcript(dict(text='صحيح',words=[dict(text='صحيح',start_ms=1000,end_ms=1700)]),0,40000)
    assert result['replace_from']==0
    assert result['tokens'][0]['id']!=old
    assert result['tokens'][0]['state']=='pending' and result['tokens'][0]['errors']==[]


def test_missing_timing_retains_text_without_fabricating_boundaries(corpus):
    tracker=TranscriptTracker(corpus)
    result=tracker.update_transcript(dict(text='كلام غير واضح',words=[]),0,32000,settled=True)
    assert result['tokens'][0]['text']=='كلام غير واضح'
    assert result['tokens'][0]['start_ms'] is None
    assert result['tokens'][0]['state']=='uncertain'


def test_untimed_raw_phrase_survives_after_audio_leaves_overlap(corpus):
    tracker=TranscriptTracker(corpus)
    tracker.update_transcript(dict(text='كلام سابق',words=[]),0,128000)
    tracker.update_transcript(dict(text='التالي',words=[dict(text='التالي',start_ms=0,end_ms=1000)]),96000,224000)
    assert [t['text'] for t in tracker.tokens]==['كلام سابق','التالي']
    assert tracker.tokens[0]['state']=='uncertain' and tracker.tokens[0]['start_ms'] is None


def test_window_start_inside_word_cannot_erase_complete_occurrence(corpus):
    tracker=TranscriptTracker(corpus)
    tracker.update_transcript(dict(words=[dict(text='إياك',start_ms=0,end_ms=1300),
        dict(text=' نعبد',start_ms=1456,end_ms=3016)]),0,73728)
    assert tracker.committed_count==1
    old_id=tracker.tokens[1]['id']
    event=tracker.update_transcript(dict(words=[dict(text=' وإياك',start_ms=0,end_ms=2720)]),41728,92160)
    assert [t['text'].strip() for t in tracker.tokens]==['إياك','نعبد','وإياك']
    assert tracker.tokens[1]['id']==old_id
    assert event['replace_from']==2


def test_eight_millisecond_overlap_preserves_word(corpus):
    tracker=TranscriptTracker(corpus)
    tracker.update_transcript(dict(words=[dict(text='إياك',start_ms=48,end_ms=1208),
        dict(text=' نعبد',start_ms=1400,end_ms=2900)]),0,51200)
    tracker.update_transcript(dict(words=[dict(text=' نعبد',start_ms=200,end_ms=1700)]),19200,70400)
    assert [t['text'].strip() for t in tracker.tokens]==['إياك','نعبد']


def test_clipped_echo_is_suppressed_by_audio_overlap_but_real_repeat_survives(corpus):
    tracker=TranscriptTracker(corpus)
    tracker.update_transcript(dict(words=[dict(text='نعبد',start_ms=0,end_ms=1000)]),0,32000)
    tracker.update_transcript(dict(words=[dict(text='نعبد',start_ms=0,end_ms=900),
        dict(text=' نعبد',start_ms=1000,end_ms=1800)]),8000,40000)
    assert [t['text'].strip() for t in tracker.tokens]==['نعبد','نعبد']
    assert len({t['id'] for t in tracker.tokens})==2


def test_next_analysis_keeps_whole_mutable_word_in_context(corpus):
    tracker=TranscriptTracker(corpus)
    tracker.update_transcript(dict(words=[dict(text='نستعين',start_ms=5400,end_ms=7900)]),0,128000)
    assert tracker.analysis_start(96000)==86400
    assert tracker.analysis_start(32000)==32000


@pytest.mark.parametrize('acoustically_supported',[False,True])
def test_confirmed_repeat_survives_later_asr_collapse(corpus,acoustically_supported):
    tracker=TranscriptTracker(corpus)
    raw=dict(words=[dict(text='إياك',start_ms=0,end_ms=900),
        dict(text=' نعبد',start_ms=1000,end_ms=1800),
        dict(text=' نعبد',start_ms=2000,end_ms=2800)])
    tracker.update_transcript(raw,0,56000)
    tracker.update_transcript(raw,0,60800)
    repeated_id=tracker.tokens[2]['id']
    if acoustically_supported:
        tracker._acoustic_supported.add(repeated_id)
    tracker.update_transcript(dict(words=[dict(text='إياك',start_ms=0,end_ms=900),
        dict(text=' نعبد',start_ms=1000,end_ms=1800),
        dict(text=' وإياك',start_ms=3000,end_ms=4000)]),0,80000)
    heard=[t['text'].strip() for t in tracker.tokens]
    assert heard==(['إياك','نعبد','نعبد','وإياك'] if acoustically_supported else ['إياك','نعبد','وإياك'])


@pytest.mark.parametrize('status',['no_speech','uncertain'])
def test_trailing_noise_cannot_erase_heard_words_or_feedback(corpus,status):
    tracker=TranscriptTracker(corpus,112,3)
    asr,phones,end=evidence(corpus,112,3)
    tracker.update_transcript(asr,0,end,True)
    tracker.update_phonemes(phones,0,end,True)
    previous=deepcopy(tracker.tokens)
    activity=dict(status=status,speech_samples=0,rms=.01,peak=.03)
    delta=tracker.update_transcript(dict(text='',words=[],speech=False,audio_activity=activity),end,end+128000)
    feedback=tracker.update_phonemes(dict(text='',groups=[],audio_activity=activity),end,end+128000)
    assert tracker.tokens==previous
    assert delta['replace_from']==len(previous) and delta['tokens']==[]
    assert feedback['updates']==[]


def test_manual_exact_phones_grade_heard_words(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus)
    tracker.update_transcript(asr,0,end,settled=True)
    result=tracker.update_phonemes(phones,0,end,settled=True)
    assert result['status']=='tracking'
    assert tracker.summary()['reviewed']==4
    assert tracker.summary()['differences']==0
    assert [t['text'] for t in tracker.tokens]==[w['text'] for w in asr['words']]


def test_vowel_error_is_on_heard_token_not_reference_token(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus,change=lambda p:'بَ' if p=='بُ' else p)
    tracker.update_transcript(asr,0,end,True)
    result=tracker.update_phonemes(phones,0,end,True)
    errors=[u for u in result['updates'] if u['state']=='difference']
    assert errors and any(e['type']=='vowel' for t in errors for e in t['errors'])
    assert any(t['id']==errors[0]['id'] and t['text']==asr['words'][1]['text'] for t in tracker.tokens)


def test_whole_word_omission_is_gap_only_between_spoken_anchors(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus,omit=(1,))
    tracker.update_transcript(asr,0,end,True)
    result=tracker.update_phonemes(phones,0,end,True)
    assert len(tracker.tokens)==3
    assert len(result['gaps'])==1
    assert result['gaps'][0]['expected']==corpus.reference(1,5).words[1]
    assert result['gaps'][0]['expected'] not in [t['text'] for t in tracker.tokens]


def test_unread_tail_is_never_reported_as_omission(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus,omit=(2,3))
    tracker.update_transcript(asr,0,end,True)
    result=tracker.update_phonemes(phones,0,end,True)
    assert result['gaps']==[]


def test_contradicted_alignment_withdraws_omission_gap(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus,omit=(1,))
    tracker.update_transcript(asr,0,end,True)
    assert tracker.update_phonemes(phones,0,end,True)['gaps']
    bad=dict(text='زَزَزَزَ',groups=[dict(text='زَ',start_ms=i*600,end_ms=(i+1)*600) for i in range(4)])
    feedback=tracker.update_phonemes(bad,0,end+1600,True)
    assert feedback['status']=='listening' and feedback['gaps']==[]


def test_repeated_phrase_does_not_move_reference_back_to_basmala(corpus):
    tracker=TranscriptTracker(corpus,1,2)
    asr,phones,end=evidence(corpus,1,2)
    tracker.update_transcript(asr,0,end,True)
    tracker.update_phonemes(phones,0,end,True)
    asr,phones,next_end=evidence(corpus,1,3)
    phones['groups']=phones['groups'][1:]  # valid linked hamzat-al-wasl omission
    phones['text']=''.join(g['text'] for g in phones['groups'])
    tracker.update_transcript(asr,end,end+next_end,True)
    feedback=tracker.update_phonemes(phones,end,end+next_end,True)
    assert feedback['reference']['ayah']==3
    assert all(t['reference']['ayah']==3 for t in tracker.tokens if t['start_ms']>=end/16 and t['reference'])


def test_asr_and_phonetic_disagreement_cannot_turn_green(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus)
    asr['words'][0]['text']='شيء'
    tracker.update_transcript(asr,0,end,True)
    tracker.update_phonemes(phones,0,end,True)
    assert tracker.tokens[0]['text']=='شيء'
    assert tracker.tokens[0]['state']=='uncertain'


def test_incomplete_asr_timing_cannot_hide_phonetic_error_and_turn_green(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus)
    phones['groups'][0]['text']='ءَ'
    phones['text']=''.join(g['text'] for g in phones['groups'])
    asr['words'][0]['start_ms']=500
    tracker.update_transcript(asr,0,end,True)
    tracker.update_phonemes(phones,0,end,True)
    assert tracker.tokens[0]['state']=='uncertain'


def test_selecting_next_reference_preserves_reviewed_previous_reading(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus)
    tracker.update_transcript(asr,0,end,True)
    tracker.update_phonemes(phones,0,end,True)
    previous=deepcopy(tracker.tokens)
    tracker.select_reference(112,1)
    assert tracker.tokens==previous and tracker.summary()['reviewed']==4
    asr,phones,next_end=evidence(corpus,112,1)
    tracker.update_transcript(asr,end,end+next_end,True)
    tracker.update_phonemes(phones,end,end+next_end,True)
    assert tracker.tokens[:4]==previous


def test_ambiguous_bismillah_waits_for_reference(corpus):
    tracker=TranscriptTracker(corpus)
    asr,phones,end=evidence(corpus,1,1)
    tracker.update_transcript(asr,0,end,True)
    result=tracker.update_phonemes(phones,0,end,True)
    assert result['status']=='listening'
    assert tracker.summary()['reviewed']==0
    before=deepcopy([(t['id'],t['text']) for t in tracker.tokens])
    feedback=tracker.select_reference(1,1)
    assert [(t['id'],t['text']) for t in tracker.tokens]==before
    assert feedback['status']=='tracking'
    assert tracker.summary()['reviewed']>0


def test_auto_localizes_outside_fatiha(corpus):
    tracker=TranscriptTracker(corpus)
    asr,phones,end=evidence(corpus,112,1)
    tracker.update_transcript(asr,0,end,True)
    result=tracker.update_phonemes(phones,0,end,True)
    assert result['reference']['surah']==112
    assert result['reference']['ayah']==1
    assert tracker.summary()['reviewed']>0


def test_long_acoustic_anchor_is_not_diluted_by_short_common_phrase(corpus):
    from types import SimpleNamespace
    _,phones,_=evidence(corpus,omit=(1,))
    calls=[]
    def search(query,error_ratio):
        size=len(chunck_phonemes(query))
        calls.append((size,error_ratio))
        if error_ratio==0 and size==6:
            raise AssertionError('A longer successful anchor must take priority')
        return corpus.search(query,error_ratio=error_ratio)
    view=SimpleNamespace(reference=corpus.reference,surahs=corpus.surahs,search=search)
    tracker=TranscriptTracker(view)
    assert tracker._locate(phones['groups']) is not None
    assert (8,0) in calls


@pytest.mark.parametrize('start', [55, 75, 90])
def test_start_deep_inside_long_ayah_tracks_exact_word_coordinates(corpus, start):
    raw, phones, _ = evidence(corpus, 2, 282)
    lo, hi = 1000 * start, 1000 * (start + 6)
    def rebase(items):
        return [dict(x, start_ms=x['start_ms']-lo, end_ms=x['end_ms']-lo)
                for x in items if lo <= x['start_ms'] < hi]

    tracker = TranscriptTracker(corpus)
    tracker.update_transcript(dict(words=rebase(raw['words'])), 0, 96000, settled=True)
    event = tracker.update_phonemes(dict(groups=rebase(phones['groups'])), 0, 96000, settled=True)
    assert event['status'] == 'tracking'
    assert [t['reference']['word_index'] for t in tracker.tokens] == list(range(start, start+6))
    assert all(t['state']=='match' and t['reference']['ayah']==282 for t in tracker.tokens)


def test_repeated_verse_requires_distinct_recent_audio_to_disambiguate(corpus):
    tracker = TranscriptTracker(corpus)
    _, previous, end = evidence(corpus, 55, 12)
    _, repeated, _ = evidence(corpus, 55, 13)
    current = tracker._absolute_groups(repeated,end)
    assert tracker._locate(current) is None
    assert tracker._locate(current, current) is None
    assert tracker._locate(tracker._absolute_groups(repeated,end+32000), previous['groups']) is None
    words, groups, path = tracker._locate(current, previous['groups'])
    positions = {words[groups[i]['owner']]['ayah'] for i,h,op in path if h is not None and op!='insert'}
    assert positions == {13}


def test_acoustic_coincidence_in_ordinary_arabic_does_not_grade_it(corpus):
    tracker = TranscriptTracker(corpus)
    raw, phones, end = evidence(corpus, 112, 1)
    ordinary = ['قل', 'كيف', 'تعمل', 'اليوم']
    for token, word in zip(raw['words'], ordinary, strict=True):
        token['text'] = word
    tracker.update_transcript(raw, 0, end, True)
    event = tracker.update_phonemes(phones, 0, end, True)
    assert event['status'] == 'listening' and event['reference'] is None
    assert tracker.summary()['reviewed'] == 0
    assert [t['text'] for t in tracker.tokens] == ordinary


@pytest.mark.parametrize('actual_word,expected_state', [('شجر','difference'),('نعبد','uncertain')])
def test_wrong_word_outside_local_reference_requires_its_own_phones(corpus,actual_word,expected_state):
    raw, phones, end = evidence(corpus)
    tracker = TranscriptTracker(corpus, 1, 5)
    raw['words'][1]['text'] = ' شجر'
    if actual_word != 'نعبد':
        # Hand-specified independent CTC-like input for شجر.
        middle = [dict(text=p, start_ms=1000+i*300, end_ms=1300+i*300)
                  for i,p in enumerate(['شَ','جَ','ر'])]
        phones['groups'] = ([p for p in phones['groups'] if p['end_ms']<=1000] + middle
                            + [p for p in phones['groups'] if p['start_ms']>=2000])
    tracker.update_transcript(raw, 0, end, True)
    tracker.update_phonemes(phones, 0, end, True)
    token = tracker.tokens[1]
    assert token['text'] == ' شجر' and token['state'] == expected_state
    if expected_state == 'difference':
        assert [(e['type'], e['heard'], e['expected']) for e in token['errors']] == [
            ('substitution','شجر',corpus.reference(1,5).words[1])]


def test_early_green_needs_stable_raw_word_plus_exact_ctc_and_interior(corpus):
    raw, phones, end = evidence(corpus)
    tracker = TranscriptTracker(corpus, 1, 5)
    tracker.update_transcript(dict(words=raw['words'][:2]), 0, 32000)
    tracker.update_transcript(raw, 0, end+1600)
    tracker.update_phonemes(phones, 0, end+1600)
    assert tracker.tokens[0]['state']=='match'
    assert tracker.tokens[-1]['state'] not in {'match','difference'}


def test_positionally_scrambled_speech_can_use_unique_partial_acoustic_location(corpus):
    raw, phones, end = evidence(corpus)
    order = [0,3,2,1]
    heard_words,heard_phones=[],[]
    for j,i in enumerate(order):
        heard_words.append(dict(raw['words'][i],start_ms=1000*j,end_ms=1000*(j+1)))
        for g in phones['groups']:
            if 1000*i <= g['start_ms'] < 1000*(i+1):
                heard_phones.append(dict(g,start_ms=g['start_ms']+1000*(j-i),
                                         end_ms=g['end_ms']+1000*(j-i)))
    tracker = TranscriptTracker(corpus)
    tracker.update_transcript(dict(words=heard_words),0,end,True)
    tracker.update_phonemes(dict(groups=heard_phones),0,end,True)
    assert [t['errors'][0]['type'] for t in tracker.tokens if t['state']=='difference'] == [
        'transposition','transposition']


@pytest.mark.parametrize('ordinary_tail',[False,True])
def test_late_swap_partner_updates_the_earlier_mark_only_with_fresh_phones(corpus, ordinary_tail):
    windows=json.loads((Path(__file__).parent/'fixtures/rolling-swap.json').read_text())['windows']
    tracker=TranscriptTracker(corpus)
    earlier_token=None
    for window in windows:
        start,end=window['samples']
        final=window is windows[-1]
        words=[dict(text=text,start_ms=a,end_ms=b) for text,a,b in window['words']]
        phones=[dict(text=text,start_ms=a,end_ms=b) for text,a,b in window['phones']]
        if ordinary_tail and final:
            for phone in phones:
                phone['text']='زَ'
            for word in words:
                word['text']=' زجاج'
        tracker.update_transcript(dict(words=words),start,end,settled=final)
        event=tracker.update_phonemes(dict(groups=phones),start,end,settled=final)
        if window is windows[-2]:
            earlier_token=tracker.tokens[1]['id']
            assert tracker.tokens[1]['errors'][0]['type']=='substitution'
    if ordinary_tail:
        assert event['status']=='listening'
        assert all(t['state']=='uncertain' for t in tracker.tokens if t['text']==' زجاج')
    else:
        assert tracker.tokens[1]['id']==earlier_token
        assert [w['text'].strip() for w in tracker.tokens]==['إياك','نستعين','وإياك','نعبد']
        assert [w['state'] for w in tracker.tokens]==['match','difference','match','difference']
        assert [w['errors'][0]['type'] for w in tracker.tokens if w['errors']]==['transposition']*2


def test_metadata_updates_are_deltas(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus)
    tracker.update_transcript(asr,0,end,True)
    first=tracker.update_phonemes(phones,0,end,True)
    again=tracker.update_phonemes(phones,0,end,True)
    assert first['updates']
    assert again['updates']==[]


def test_partial_observations_need_fresh_consistent_evidence(corpus):
    tracker=TranscriptTracker(corpus,1,5)
    asr,phones,end=evidence(corpus)
    tracker.update_transcript(asr,0,end)
    tracker.update_phonemes(phones,0,end)
    assert not any(t['state']=='match' for t in tracker.tokens)
    tracker.update_phonemes(phones,0,end)
    assert not any(t['state']=='match' for t in tracker.tokens)
    tracker.update_phonemes(phones,0,end+1600)
    assert any(t['state']=='match' for t in tracker.tokens)


def test_late_location_recheck_cannot_revoke_newer_word_verdicts(corpus):
    tracker=TranscriptTracker(corpus)
    asr,phones,end=evidence(corpus)
    first=dict(text=asr['words'][0]['text'],words=asr['words'][:1],speech=True)
    early=dict(groups=[g for g in phones['groups'] if g['end_ms']<=1000])
    early['text']=''.join(g['text'] for g in early['groups'])
    tracker.update_transcript(first,0,16000)
    tracker.update_phonemes(early,0,16000)
    tracker.update_transcript(asr,0,end,True)
    tracker.update_phonemes(phones,0,end,True)
    assert [t['state'] for t in tracker.tokens]==['match']*4


def test_repetition_is_marked_with_fresh_live_evidence_before_stopping(corpus):
    asr,phones,_=evidence(corpus)
    order=[0,1,1,2,3]
    words=[];groups=[]
    for j,i in enumerate(order):
        words.append(dict(text=asr['words'][i]['text'],start_ms=j*1000,end_ms=(j+1)*1000))
        for group in phones['groups']:
            if i*1000 <= group['start_ms'] < (i+1)*1000:
                groups.append(dict(text=group['text'],start_ms=group['start_ms']+(j-i)*1000,
                                   end_ms=group['end_ms']+(j-i)*1000))
    transcript=dict(words=words,speech=True)
    prediction=dict(text=''.join(g['text'] for g in groups),groups=groups)
    tracker=TranscriptTracker(corpus,1,5)
    tracker.update_transcript(transcript,0,80000)
    tracker.update_phonemes(prediction,0,80000)
    tracker.update_phonemes(prediction,0,81600)
    repeated=tracker.tokens[2]
    assert repeated['state']=='difference'
    assert repeated['errors'][0]['type']=='repetition'
    assert repeated['errors'][0]['heard']==repeated['text'].strip()
