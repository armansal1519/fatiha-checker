import json
from collections import Counter

import numpy as np
import pytest

from prepare_evaluation_pack import AssetCache, SELECTION, chunk_bytes, digest, edit_order, nonspeech, read_pcm


def test_repair_catalog_can_locate_locked_media_without_a_local_catalog_dump():
    from evaluation.prepare_repair_voices import LOCK, SOURCES, SPEC
    from evaluation import repair_confirmation

    spec = json.loads(SPEC.read_bytes())
    confirmation = json.loads(repair_confirmation.SPEC.read_bytes())
    catalog = json.loads(SOURCES.read_bytes())
    readers = {row['id']: row for row in catalog['readers']}
    assert set(readers) == set(spec['reader_ids'] + confirmation['reader_ids'])
    for selection, chapters, lock in ((spec, (93, 112, 113), LOCK),
                                      (confirmation, confirmation['chapters'], repair_confirmation.LOCK)):
        assets = json.loads(lock.read_bytes())['assets']
        for reader in selection['reader_ids']:
            for surah in chapters:
                url = readers[reader]['folder_url'] + f'{surah:03d}.mp3'
                assert digest(url.encode()) in assets
    selected = set(map(tuple, spec['learner_recordings']))
    assert {(row['recording_id'], row['file_name'].rsplit('/', 1)[1]) for row in catalog['learners']} == selected
    assets = json.loads(LOCK.read_bytes())['assets']
    for row in catalog['learners']:
        url = ('https://huggingface.co/datasets/sobolev210/quran-recitation-errors/resolve/'
               f"{catalog['learner_revision']}/{row['file_name']}")
        assert digest(url.encode()) in assets


def test_selection_size_is_frozen_and_sources_are_disjoint():
    selection = json.loads(SELECTION.read_bytes())
    rows = selection['clips']
    assert len({c['id'] for c in rows}) == len(rows)
    assert Counter(c['source_kind'] for c in rows) == {
        'professional': 168, 'natural-learner': 84, 'controlled-edit-human': 48,
        'ordinary-speech': 4, 'noise': 4,
    }
    groups = {split: {c['group'] for c in rows if c['split'] == split} for split in ('dev', 'holdout')}
    assert groups['holdout'].isdisjoint(groups['dev'])
    learners = [c for c in rows if c['source_kind'] == 'natural-learner']
    assert len([c for c in learners if c['source_annotations']['canonical_word_errors']]) == 42
    assert set(selection['excluded_source_recordings']).isdisjoint(c['group'] for c in learners)
    assert all(c['spoken_text'] is None and not c['error_labels_complete'] for c in learners)
    professionals = [c for c in rows if c['source_kind'] == 'professional']
    assert all(c['source']['riwayah'] == 'حفص عن عاصم' for c in professionals)
    assert Counter(c['group'] for c in professionals).most_common()[-1][1] == 12
    assert len({c['group'] for c in professionals}) == 14


@pytest.mark.parametrize(('kind','sequence','heard','refs','after'), [
    ('repetition',[0,1,1,2,3],[2],[1],None),
    ('omission',[0,2,3],[],[1],0),
    ('transposition',[0,2,1,3],[1,2],[1,2],None),
    ('substitution',[0,3,2,3],[1],[1],None),
])
def test_edit_events_keep_exact_heard_and_reference_ownership(kind,sequence,heard,refs,after):
    actual, event = edit_order(4, kind)
    assert actual == sequence
    assert event == dict(kind=kind, heard_indices=heard, reference_indices=refs, after_index=after)


def test_offline_asset_checks_reject_wrong_bytes(tmp_path):
    url = 'https://example.test/sample.wav'
    assets = tmp_path / 'assets'
    assets.mkdir()
    path = assets / digest(url.encode())
    path.write_bytes(b'wrong')
    cache = AssetCache(tmp_path, {path.name:digest(b'right')}, offline=True)
    with pytest.raises(OSError, match='damaged'):
        cache.obtain(url)
    path.write_bytes(b'right')
    assert cache.obtain(url) == path


def test_generated_challenges_round_trip_without_silence_fabrication(tmp_path):
    for kind in ['silence','noise','fan','impulses']:
        pcm = nonspeech(kind)
        file = tmp_path / (kind + '.wav')
        file.write_bytes(chunk_bytes(pcm))
        assert np.array_equal(pcm, read_pcm(file))
        assert bool(np.any(pcm)) == (kind != 'silence')


def test_identity_extension_preserves_the_original_splits_and_has_no_marked_words():
    from prepare_evaluation_pack import IDENTITIES
    originals = json.loads(SELECTION.read_bytes())['clips']
    controls = {r['group']:r for r in originals if r.get('operation') == 'repetition'}
    identities = json.loads(IDENTITIES.read_bytes())['clips']
    assert len(identities) == len(controls) == 12
    for row in identities:
        source = controls[row['group']]
        assert (row['source'],row['split']) == (source['source'],source['split'])
        n = len(row['canonical_text'].split())
        assert edit_order(n, row['operation']) == (list(range(n)),None)


def test_timed_duplicate_alignment_uses_time_to_identify_the_real_occurrence():
    from evaluate_timed_recitations import timed_alignment
    errors, equal = timed_alignment(['هو','هو'], [[0,430],[890,1380]], [('هو',(900,1320))])
    assert errors == 1 and equal == {1:0}


def test_timed_annex_locks_local_source_provenance_and_repeats():
    from evaluate_timed_recitations import SPEC, aggregate, measure
    spec = json.loads(SPEC.read_bytes())
    assert len(spec['decisions']) == 16 and len(spec['source_audio']) == 3
    results = []
    for clip in spec['clips']:
        assert len(clip['anchors']) == 3
        for match in clip['anchors']:
            assert abs(match['corr']) > .65
            assert abs(match['offset_s'] - clip['anchor_offset_s']) < .1
        tokens = [dict(text=' '+w,start_ms=s,end_ms=e) for w,(s,e) in zip(
            clip['spoken_text'].split(), clip['spoken_word_times_ms'], strict=True)]
        results.append(measure(clip,tokens))
    total = aggregate(results)
    assert total['timed_lexical_matches'] == total['reviewed_occurrences'] == 107
    assert total['repeat_occurrences_overlapping_reviewed_audio'] == total['repeat_occurrences'] == 23
    assert total['exact_order_clips'] == 7


def test_download_reuses_partial_bytes_after_read_timeout(tmp_path, monkeypatch):
    from pathlib import Path
    from subprocess import CompletedProcess
    calls = []
    def fetch(args, capture_output):
        part = Path(args[args.index('--output') + 1])
        assert '--continue-at' in args
        calls.append(part.read_bytes() if part.exists() else b'')
        if len(calls) == 1:
            part.write_bytes(b'partial')
            return CompletedProcess(args, 28, b'', b'timeout')
        part.write_bytes(part.read_bytes()+b'-complete')
        return CompletedProcess(args, 0, b'', b'')
    monkeypatch.setattr('prepare_evaluation_pack.subprocess.run', fetch)
    url = 'https://example.test/recitation.mp3'
    expected = digest(b'partial-complete')
    cache = AssetCache(tmp_path, {digest(url.encode()):expected}, offline=False)
    assert cache.obtain(url).read_bytes() == b'partial-complete'
    assert calls == [b'', b'partial']
