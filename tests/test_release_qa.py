"""Independent scoring from final UI word instances and gaps, with no model I/O."""

from evaluation.compare_recitation_runs import control, learner


def clip(words, expected=None, events=None):
    return dict(id='test', surah=1, ayah=2, operation='identity', spoken_text=' '.join(words),
                canonical_text=' '.join(expected or words), events=events or [],
                source={'asset_segments':[dict(start_ms=i*500,end_ms=(i+1)*500) for i in range(len(words))]})


def token(k, text, lo, state='match', errors=(), ref=None):
    return dict(id=k,text=text,start_ms=lo,end_ms=lo+480,state=state,errors=list(errors),reference=ref)


def test_multiple_unrecognized_insertions_count_individually_without_duplicate_count_per_word():
    c=clip(['هو','الله'])
    alerts=[dict(type='insertion',heard='خطأ',expected='')]*2
    words=[token('w0','هو',0),token('w1','خطأ',300,'difference',alerts),
           token('w2','خطأ',600,'difference',alerts),token('w3','الله',900)]
    scored=control(c,dict(tokens=words,gaps=[]))
    assert scored['by_type']['insertion']['fp']==2
    assert [s['id'] for s in scored['spurious_warnings']]==['w1','w2']
    assert not scored['exact_sequence_and_marks']


def test_repeat_time_tiebreaker_preserves_a_late_occurrence_when_first_occurrence_is_absent():
    c=clip(['هو','هو','الله'], ['هو','الله'], [dict(kind='repetition',heard_indices=[1],reference_indices=[0])])
    words=[token('w0','هو',510,'difference',[dict(type='repetition',heard='هو',expected='')]),
           token('w1','الله',1000)]
    scored=control(c,dict(tokens=words,gaps=[]))
    assert scored['by_type']['repetition']['tp']==1
    assert scored['fidelity']['retained']==1
    assert not scored['exact_sequence']


def test_gap_requires_both_adjacent_owners_and_expected_word_on_the_same_reference():
    c=clip(['هو','أحد'],['هو','الله','أحد'],[dict(kind='omission',heard_indices=[],reference_indices=[1],after_index=0)])
    words=[token('left','هو',0),token('right','أحد',500)]
    ref=dict(surah=1,ayah=2,word_index=1,word='الله')
    correct=dict(id='g1',reference=ref,expected='الله',before_token_id='left',after_token_id='right')
    spurious={**correct,'id':'g2','after_token_id':'unknown'}
    scored=control(c,dict(tokens=words,gaps=[correct,spurious]))
    assert scored['by_type']['omission']['tp']==scored['by_type']['omission']['fp']==1
    assert scored['by_type']['omission']['eligible']==scored['by_type']['omission']['reviewed_opportunities']==1
    wrong=control(c,dict(tokens=words,gaps=[{**correct,'expected':'غير'}]))
    assert wrong['by_type']['omission']['fn']==wrong['by_type']['omission']['fp']==1


def test_position_and_word_payload_both_matter_for_a_substitution():
    event=dict(kind='substitution',heard_indices=[0],reference_indices=[0])
    c=clip(['رب','أحد'],['هو','أحد'],[event])
    ref=dict(surah=1,ayah=2,word_index=0,word='هو')
    token0=token('w0','رب',0,'difference',[dict(type='substitution',heard='رب',expected='باطل')],ref)
    scored=control(c,dict(tokens=[token0,token('w1','أحد',500)],gaps=[]))
    assert scored['by_type']['substitution']['fp']==scored['by_type']['substitution']['fn']==1
    assert scored['fidelity']['retained']==1
    corrected=control(c,dict(tokens=[token('w0','هو',0),token('w1','أحد',500)],gaps=[]))
    assert corrected['fidelity']['substitutions_canonicalized']==1
    assert corrected['fidelity']['substitution_opportunities']==1


def test_sparse_source_labels_distinguish_wrong_type_unjudged_and_unlocatable():
    c=dict(surah=1,ayah=2,canonical_text='هو الله أحد أحد',group='recording-1',
           source_annotations={'canonical_word_errors':{'الله':['Letters'],'أحد':['Letters'],'هو':['Tajweed']}})
    src=dict(surah=1,ayah=2,word_index=1,word='الله')
    words=[token('w1','الله',100,'difference',[{'type':'vowel'}],src),
           token('w2','هو',500,'difference',[{'type':'letter'}],None)]
    out=learner(c,dict(id='natural',tokens=words,gaps=[]))
    assert [(s['eligible'],s['typed_hit']) for s in out['sites']]==[(True,False),(False,False),(False,False)]
    assert [w['status'] for w in out['warnings']]==['different_or_ungraded_type_at_published_site','outside_reference_ayah_or_unmapped']


def test_self_consistent_but_wrong_reference_and_display_word_do_not_match_source():
    event=dict(kind='substitution',heard_indices=[0],reference_indices=[0])
    c=clip(['رب','أحد'],['هو','أحد'],[event])
    bad=dict(surah=1,ayah=2,word_index=0,word='آخر')
    tok=token('w0','رب',0,'difference',[dict(type='substitution',heard='رب',expected='آخر')],bad)
    score=control(c,dict(tokens=[tok,token('w1','أحد',500)],gaps=[]))
    assert score['by_type']['substitution']['tp']==0
    assert score['by_type']['substitution']['fp']==score['by_type']['substitution']['fn']==1


def test_silently_filled_omission_requires_bounded_matching_anchors():
    c=clip(['هو','أحد'],['هو','الله','أحد'],[dict(kind='omission',heard_indices=[],reference_indices=[1],after_index=0)])
    words=[token('left','هو',0),token('spurious','الله',300),token('right','أحد',600)]
    filled=control(c,dict(tokens=words,gaps=[]))['fidelity']
    assert filled['omissions_with_two_exact_anchors']==filled['filled_with_omitted_word']==1
    partial=control(c,dict(tokens=words[1:],gaps=[]))['fidelity']
    assert partial['target_reference_omissions']==1 and partial['omissions_with_two_exact_anchors']==0
