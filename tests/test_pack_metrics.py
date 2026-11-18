import json
from pathlib import Path
import wave

import numpy as np
import pytest

from benchmark_recitation_pack import input_windows, score, typed_control_score, summary
from evaluate_transcription import has_spoken_gold
from verify_stream_runtime import RecordingSource, RATE, StreamAccumulator


def test_published_learner_target_is_not_spoken_gold_even_with_old_verified_flag():
    c=dict(id='learner',source_kind='natural-learner',surah=1,ayah=2,
           canonical_text='الحمد لله',spoken_text='الحمد لله',spoken_text_verified=True,
           source_annotations={'canonical_word_errors':{'الحمد':['Letters']}})
    assert not has_spoken_gold(c)
    result=score(c,[],[])
    assert 'lexical_wer' not in result
    assert result['published_sites']==[dict(label='Letters',measurable=True,index=0,detected=False)]


def test_gap_requires_the_actual_pair_and_correct_reference_word():
    clip=dict(id='omission',source_kind='controlled-edit-human',surah=1,ayah=5,
              spoken_text='ألف جيم',canonical_text='ألف باء جيم',
              events=[dict(kind='omission',heard_indices=[],reference_indices=[1],after_index=0)])
    tokens=[dict(id='a',text=' ألف',state='match'),dict(id='c',text=' جيم',state='match')]
    g=dict(before_token_id='a',after_token_id='c',reference=dict(surah=1,ayah=5,word_index=1))
    assert typed_control_score(clip,tokens,[g])['exact_annotations']
    different=dict(g,after_token_id='a')
    scored=typed_control_score(clip,tokens,[different])
    assert not scored['exact_annotations'] and scored['by_type']['omission']==dict(tp=0,fp=1,fn=1)
    tokens[1].update(state='difference',errors=[{'type':'letter'}])
    assert not typed_control_score(clip,tokens,[g])['exact_annotations']


def test_streaming_fixture_decodes_once_and_reports_short_input(tmp_path):
    pcm=(np.arange(8000,dtype=np.int16)-4000)
    path=tmp_path/'test.wav'
    with wave.open(str(path),'wb') as output:
        output.setparams((1,2,16000,0,'NONE','not compressed'))
        output.writeframes(pcm.astype('<i2').tobytes())
    source=RecordingSource(path,RATE)
    first=source(0,527)
    with pytest.raises(ValueError,match='sequential'):
        source(0,1)
    assert np.allclose(np.concatenate([first,source(527,len(pcm))]),pcm/32768)
    with pytest.raises(ValueError,match='ended'):
        source(len(pcm),len(pcm)+1)


def test_fixed_model_windows_cover_last_sample_and_overlap():
    for duration in [1,2,5,8,11,18,30]:
        windows=list(input_windows(duration*RATE))
        assert windows[0][0]==0 and windows[-1][1:]==(duration*RATE,True)
        assert all(end-start<=8*RATE for start,end,_ in windows)
        assert all(b[0]<=a[1]-2*RATE and b[1]>a[1] for a,b in zip(windows,windows[1:]))


def test_first_definitive_verdict_excludes_uncertain_and_duplicate_locations():
    acc=StreamAccumulator(dict(type='ready',version=2,session_id='s'))
    def event(name,r,time,**kw):
        acc.accept(dict(type=name,revision=r,session_id='s',**kw),time)
    event('transcript',1,1,replace_from=0,committed_count=0,tokens=[dict(id='1',text=' قل')])
    ref=dict(surah=112,ayah=1)
    for r,state in [(2,'uncertain'),(3,'match'),(4,'difference')]:
        event('feedback',r,r,updates=[dict(id='1',state=state)],reference=ref)
    assert acc.first_feedback_seconds==2 and acc.first_reviewed_seconds==3
    assert len(acc.reference_changes)==1


def test_empty_denominators_and_observational_proxy_stay_separate():
    row=score(dict(id='p',source_kind='professional',group='r',split='dev',
                   spoken_text_verified=False,canonical_text='قل'),[],[])
    stats=summary([row])['overall']
    assert stats['spoken_wer']['rate'] is None and stats['canonical_proxy']['rate']==1
