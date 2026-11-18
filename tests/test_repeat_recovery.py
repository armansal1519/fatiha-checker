import numpy as np
import pytest

from repeat_recovery import recover_repetitions
from transcript_tracking import TranscriptTracker


def timed(text,start,end):
    return dict(text=text,start_ms=start,end_ms=end)


def phones():
    return [timed(text,start,start+100) for text,start in
            [('كَ',200),('تَ',400),('بَ',600),('كَ',950),('تَ',1150),('بَ',1350)]]


def primary():
    words=[timed(' كتب',0,750),timed(' اليوم',750,2100)]
    return dict(text=''.join(w['text'] for w in words),words=words)


def verified(_):
    return dict(words=[timed(' كَتَبَ',0,500),timed(' اليوم',510,1100)])


def test_recovered_occurrence_uses_independently_spoken_spelling_and_ctc_times():
    raw=primary()
    recovered=recover_repetitions(raw,dict(groups=phones()),np.zeros(40000),verified)
    assert raw['text']==' كتب اليوم'
    assert recovered['text']==' كتب كَتَبَ اليوم'
    assert [(w['start_ms'],w['end_ms']) for w in recovered['words']]==[(200,700),(950,1450),(1450,2100)]


@pytest.mark.parametrize('second',[
    [timed(' كتاب',0,500),timed(' اليوم',510,1100)],
    [timed(' كتب',0,500),timed(' أمس',510,1100)],
    [timed(' كتب',900,1000),timed(' اليوم',1000,1100)],
    [],
])
def test_unconfirmed_or_misplaced_subclip_cannot_add_text(second):
    raw=primary()
    assert recover_repetitions(raw,dict(groups=phones()),np.zeros(40000),lambda _:dict(words=second)) is raw


def test_second_ctc_cycle_required_and_no_double_insertion_for_verified_pair():
    raw=primary()
    def unused(_):
        raise AssertionError('unexpected extra decoder call')
    assert recover_repetitions(raw,dict(groups=phones()[:-1]),np.zeros(40000),unused) is raw
    raw['words'].insert(1,timed(' كتب',730,1400))
    raw['text']=' كتب كتب اليوم'
    assert recover_repetitions(raw,dict(groups=phones()),np.zeros(40000),unused)['text']==raw['text']


def test_later_real_identical_token_cannot_be_moved_to_an_earlier_ctc_cycle():
    raw=dict(words=[timed(' كتب',200,700),timed(' كتب',3100,3500),timed(' اليوم',3500,4200)])
    groups=phones()+[timed(t,a,a+100) for t,a in [('كَ',3100),('تَ',3250),('بَ',3400)]]
    assert recover_repetitions(raw,dict(groups=groups),np.zeros(70000),verified) is raw


def test_same_window_verification_can_split_previously_supported_mutable_neighbor():
    from quran_corpus import QuranCorpus
    from benchmark_recitation_pack import apply_asr_window
    tracker=TranscriptTracker(QuranCorpus(),112,3)
    raw=dict(words=[timed(' لم',200,700),timed(' يلد',700,1800)])
    # Two earlier windows agree about both words and provide independent CTC
    # support for both. Only the first leaves the normal mutable interval.
    prior=dict(groups=[timed(t,a,b) for t,a,b in [('لَ',200,450),('مۡ',450,700),
                  ('يَ',700,1050),('لِ',1050,1450),('دۡ',1450,1800)]])
    for end in (32000,48000):
        apply_asr_window(tracker,dict(heard=raw),0,end,False)
        tracker.update_phonemes(prior,0,end,False)
    first,neighbor=[t['id'] for t in tracker.tokens]
    assert tracker.committed_count==1 and neighbor in tracker._acoustic_supported
    preview=tracker.preview_transcript(raw,0,64000,True)
    assert preview['committed_count']==1
    cycles=[timed(t,a,b) for t,a,b in [('لَ',200,450),('مۡ',450,700),('لَ',950,1200),('مۡ',1200,1450)]]
    confirmed=recover_repetitions(raw,dict(groups=cycles),np.zeros(64000),
        lambda _:dict(words=[timed('لم',0,500),timed(' يلد',510,1050)]))
    tracker.update_transcript(confirmed,0,64000,True)
    assert [t['text'].strip() for t in tracker.tokens]==['لم','لم','يلد']
    assert tracker.committed_count==3 and tracker.tokens[0]['id']==first
    assert tracker.tokens[2]['id']==neighbor


def test_preview_cannot_erase_visible_repeat_or_lock_following_token_before_verification():
    tracker=TranscriptTracker(None)
    first=dict(words=[timed(' كتب',200,700),timed(' كتب',950,1190)])
    tracker.update_transcript(first,0,20000)
    repeated_id=tracker.tokens[1]['id']
    assert tracker.preview_transcript(primary(),0,40000,settled=True) is None
    assert tracker.tokens[1]['id']==repeated_id
    confirmed=recover_repetitions(primary(),dict(groups=phones()),np.zeros(40000),verified)
    tracker.update_transcript(confirmed,0,40000,settled=True)
    assert len(tracker.tokens)==tracker.committed_count==3

    fresh=TranscriptTracker(None)
    preview=fresh.preview_transcript(primary(),0,40000,settled=True)
    assert preview['committed_count']==0 and len(preview['tokens'])==2
    fresh.update_transcript(confirmed,0,40000,settled=True)
    assert [w['text'] for w in fresh.tokens]==[' كتب',' كَتَبَ',' اليوم']


def test_live_worker_publishes_tentative_text_before_acoustic_recovery_and_feedback():
    from queue import Queue
    from threading import Event, Lock
    from types import SimpleNamespace
    from live_socket import LiveSession, LiveSocketManager

    raw=primary()
    job=dict(start=0,end=40000,settled=True,wave=np.zeros(40000))
    published=Queue(maxsize=4)
    def ctc(wave):
        assert published.qsize()==1
        return dict(groups=phones())
    service=SimpleNamespace(predict_live_timed=ctc)
    asr=SimpleNamespace(transcribe=lambda wave:raw,
        recover_repetitions=lambda heard,evidence,wave:recover_repetitions(heard,evidence,wave,verified))
    lock=Lock()
    manager=LiveSocketManager(service,asr,None,lock)
    try:
        lock.acquire()
        manager.infer(job,published,Event())
        session=LiveSession(dict(type='start',version=2,sample_rate=16000,channels=1),None)
        preview=session.apply_stage(published.get_nowait())
        revision=session.apply_stage(published.get_nowait())
        assert preview['type']==revision['type']=='transcript'
        assert preview['committed_count']==0 and revision['committed_count']==3
        assert [w['text'] for w in revision['tokens']]==[' كتب',' كَتَبَ',' اليوم']
        assert published.get_nowait()['stage']=='feedback' and published.empty()
    finally:
        manager.shutdown()
