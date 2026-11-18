import numpy as np

from evaluation.passage_challenges import extract, score
from prepare_evaluation_pack import chunk_bytes


def test_bounded_extractor_preserves_the_pcm_clock_and_file_tail(tmp_path):
    pcm = np.arange(32000).astype('<i2')
    path=tmp_path/'test.wav'
    path.write_bytes(chunk_bytes(pcm))
    assert np.array_equal(extract(path,1111,20000),pcm[1111:20000])
    assert np.array_equal(extract(path,22000,None),pcm[22000:])


def test_join_scoring_excludes_the_opening_from_numbered_verse_latency():
    clip=dict(id='join',sha256='abc',duration_s=5,parts=[{},dict(begins_at_clip_ms=2000)],
              surah=112,ayah=4,expected_ref_timeline=[
        dict(start_ms=0,end_ms=2000,surah=112,ayah=4),
        dict(start_ms=2000,end_ms=3000,surah=113,ayah=0),
        dict(start_ms=3000,end_ms=5000,surah=113,ayah=1)])
    trace=[dict(start=0,end=end,location=ref,raw='ت',raw_window=' ت') for end,ref in [(2,[112,4]),(2.5,[113,1]),(5,[113,1])]]
    result=score(clip,dict(raw=' ت',timeline=trace,tokens=[],gaps=[]))
    assert result['first_113_ref_after_spoken_verse_s']==5 and result['first_113_delay_source_s']==2
    assert result['trace'][1]['publisher_at_window_end'] is None
