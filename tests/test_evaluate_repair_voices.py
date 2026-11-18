from evaluation.evaluate_repair_voices import TAIL, score


def token(i, text, kind=None):
    return dict(id=f'w{i}', text=' '+text, start_ms=i*500, end_ms=(i+1)*500,
                state='difference' if kind else 'match', reference=None,
                errors=[dict(type=kind, expected='ب', heard='خ')] if kind else [])


def repeat(marked):
    reading=['فإن','مع','مع','العسر']
    clip=dict(case='midword_repeat', spoken_text_verified=True, spoken_text=' '.join(reading),
              canonical_text='فإن مع العسر', position=2,
              segments=[dict(word=w,start_ms=i*500,end_ms=(i+1)*500) for i,w in enumerate(reading)])
    tokens=[token(i,w,'repetition' if marked==i else None) for i,w in enumerate(reading)]
    return score(clip,tokens,[],[],[dict(start_ms=0,end_ms=2000,groups=[])])


def test_repeat_mark_requires_second_spoken_occurrence():
    assert repeat(1)['adjacent_at_expected']
    assert not repeat(1)['second_only_repeat']
    assert repeat(2)['second_only_with_timed_ownership']


def test_unheard_tail_has_unknown_extras_and_observed_brand_text_visible():
    clip=dict(case='chapter_join', spoken_text_verified=False, last_ayah_start_s=0)
    short = score(clip,[token(0,'اشتركوا في القناة')],[],[],[])
    assert not short['exact_ending'] and short['words_after_last_ayah'] is None
    assert short['final_credit_phrases']
    extra = score(clip,[token(0,TAIL),token(1,'كلمة')],[],[],[])
    assert extra['final_ayah_found'] and not extra['exact_ending']
    assert extra['words_after_last_ayah']==['كلمة']


def test_substitution_warning_or_wrong_consonant_does_not_count_as_letter_hit():
    clip=dict(case='single_letter',spoken_text_verified=True,spoken_text='يد أخي',canonical_text='يد أبي',
              position=2,surah=111,ayah=1,segments=[dict(word=w,start_ms=i*500,end_ms=(i+1)*500)
              for i,w in enumerate(['يد','أخي'])])
    edit=dict(surah=111,ayah=1,changes=['ب','خ'],intended='أبي',heard='أخي')
    def graded(kind,heard):
        tokens=[token(0,'يد'),token(1,'أخي',kind)]
        tokens[1]['reference']=dict(surah=111,ayah=1,word_index=1)
        tokens[1]['errors'][0]['heard']=heard
        return score(clip,tokens,[],[edit],[dict(start_ms=0,end_ms=1000,groups=[])])
    assert not graded('substitution','خ')['correct_contrast_at_location']
    assert not graded('letter','ح')['correct_contrast_at_location']
    assert graded('letter','خ')['raw_and_typed_contrast']
