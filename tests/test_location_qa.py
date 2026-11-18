import pytest
from evaluation.compare_locations import (ABSTAIN, EQUIVALENT, EXACT, HOMOGRAPH, SPURIOUS,
                                         WRONG, classify, location_case, summarize)

CORPUS={(1,1):('قُلْ',('قل',)),(2,1):('قُلْ',('قل',)),(3,1):('قَلَّ',('قل',)),(4,1):('هُوَ',('هو',))}
EQUAL={'قُلْ':[(1,1),(2,1)],'قَلَّ':[(3,1)],'هُوَ':[(4,1)]}


def case(kind, positions):
    clip=dict(id='abc',source_kind=kind,surah=1,ayah=1)
    timeline=[dict(end=i*2,location=item) for i,item in enumerate(positions,1)]
    return location_case(clip,{'timeline':timeline},CORPUS,EQUAL)


def test_numeric_other_identical_and_homographic_verses_are_distinct():
    predictions=[[None,None],[2,1],[3,1],[4,1],[1,1]]
    assert [classify((1,1),p,CORPUS) for p in predictions]==[ABSTAIN,EQUIVALENT,HOMOGRAPH,WRONG,EXACT]
    scored=case('professional',predictions)
    assert scored['first_exact_source_s']==10 and scored['first_equivalent_or_exact_source_s']==4
    assert scored['first_different_text_source_s']==8 and scored['exact_recovery_after_different_text_source_s']==10
    result=summarize([scored])
    assert result['different_text_then_exact']['rate']==1
    assert result['elapsed_source_seconds_to_recover_different_text']['median']==2


def test_abstaining_or_transient_negative_reference_does_not_imply_exact_target():
    quiet=case('noise',[[None,None]])
    speech=case('ordinary-speech',[[2,1],[None,None]])
    waiting=case('natural-learner',[[None,None]])
    result=summarize([quiet,speech,waiting])
    assert result['any_reference_on_negative']=={'n':1,'total':2,'rate':.5}
    assert result['source_window_counts'][SPURIOUS]==1
    assert result['final_counts'][SPURIOUS]==0
    assert result['final_reference_coverage_among_quran_inputs']=={'n':0,'total':1,'rate':0}
    assert result['final_numeric_precision_among_published_targets']['rate'] is None
    assert result['first_exact_reference_source_seconds']['n']==0


def test_unrecognized_numeric_reference_is_invalid_data():
    with pytest.raises(ValueError): classify((1,1),[1,88],CORPUS)
