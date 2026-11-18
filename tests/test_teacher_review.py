from hashlib import sha256
from io import BytesIO
import json
import wave
from zipfile import ZipFile

import pytest

from evaluation import teacher_review as review


def test_locked_teacher_tasks_use_pinned_public_audio_and_unique_sites():
    spec = json.loads(review.SPEC.read_bytes())
    locks = json.loads((review.ROOT / "evaluation/assets-v1.json").read_bytes())["assets"]
    assert spec["selection"]["eligible_sites"] == 37
    assert spec["selection"]["previously_missed"] == 32
    assert len(spec["tasks"]) == len({(t["clip"], t["word_index"]) for t in spec["tasks"]}) == 43
    assert len(spec["clips"]) == 40
    for t in spec["tasks"]:
        row = spec["clips"][t["clip"]]
        assert row["words"][t["word_index"]]
        assert locks[sha256(review.source_url(spec, row).encode()).hexdigest()] == row["sha256"]


def test_selection_matches_published_reports_when_available():
    spec = json.loads(review.SPEC.read_bytes())
    expected, clean = set(), set()
    missed = 0
    for split, sha in spec["selection"]["qa_sha256"].items():
        report = review.ROOT / f"reports/repair-final-qa-{split}.json"
        if not report.exists():
            pytest.skip("Local results from the full model evaluation are optional.")
        assert review.digest(report.read_bytes()) == sha
        rows = json.loads(report.read_bytes())["runs"]["candidate"]["learner_cases"]
        for r in rows:
            if not r["source_has_labels"]:
                clean.add(r["id"])
            for site in r["sites"]:
                if site["eligible"] and site["label"] == "Letters":
                    expected.add((r["id"], site["index"]))
                    missed += not site["typed_hit"]
    selected = {(t["clip"], t["word_index"]) for t in spec["tasks"]}
    assert len(expected) == 37 and missed == 32
    assert expected <= selected
    assert len(selected - expected) == 6 and all(key in clean for key, pos in selected - expected)


def test_packet_is_standalone_blind_and_reproducible(tmp_path, monkeypatch):
    source = BytesIO()
    with wave.open(source, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(32000))
    data = source.getvalue()
    spec = dict(dataset="sobolev210/quran-recitation-errors", revision="abc", license="MIT",
                selection={"model_verdict_should_be_hidden": True},
                clips={"secret-test-id": dict(path="audio/input.wav", sha256=review.digest(data), seconds=1.,
                                              surah=112, ayah=1, words=["أحد"])},
                tasks=[{"id": "R01", "word_index": 0, "clip": "secret-test-id"}])
    spec_path = tmp_path / "manifest.json"
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr(review, "SPEC", spec_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    key = review.digest(review.source_url(spec, spec["clips"]["secret-test-id"]).encode())
    (assets / key).write_bytes(data)
    output = tmp_path / "review.html"
    review.build(output, assets, offline=True)
    html = output.read_text()
    assert "model_verdict_should_be_hidden" not in html and "secret-test-id" not in html
    assert "data:font/woff2;base64," in html and '<script src=' not in html
    assert review.encoded(data) in html and "Permission is hereby granted" in html
    archive = output.with_suffix(".zip")
    original = archive.read_bytes()
    with ZipFile(archive) as zipped:
        assert zipped.namelist() == ["review.html"]
        assert zipped.read(output.name) == output.read_bytes()
    review.build(output, assets, offline=True)
    assert original == archive.read_bytes()
    (assets / key).write_bytes(b"broken")
    with pytest.raises(ValueError, match="Missing or changed"):
        review.build(output, assets, offline=True)
    assert archive.read_bytes() == original


def response():
    spec = review.SPEC.read_bytes()
    row = dict(task_id="R01", verdict="difference", expected="ح", heard="ه", start_s=0., end_s=.2, note="")
    return dict(schema="hamkhan-teacher-answers/v1", packet_id=review.packet_identity(spec), answers=[row])


def test_returned_answers_join_to_review_targets_and_preserve_uncertainty(tmp_path):
    payload = response()
    payload["answers"].append(dict(payload["answers"][0], task_id="R02", verdict="uncertain", start_s=None, end_s=None,
                                   note="=1+2"))
    stats, table = review.check_answers(payload, review.SPEC.read_bytes())
    assert stats == dict(tasks=43, completed=2, pending=41, match=0, difference=1, uncertain=1,
                         typed_contrasts=1, timed_typed_contrasts=1)
    assert table[0]["clip"] and table[0]["word"] and table[0]["timed_typed_contrast"]
    assert table[2]["verdict"] == "pending"
    path = tmp_path / "joined.csv"
    review.export_table(path, table)
    assert "'=1+2" in path.read_text(encoding="utf-8-sig")


@pytest.mark.parametrize("field,value", [("task_id", "unknown"), ("start_s", -1), ("end_s", 0),
                                         ("end_s", None), ("end_s", 1e6), ("start_s", True),
                                         ("heard", 1), ("verdict", "unknown")])
def test_reject_invalid_response_field(field, value):
    payload = response()
    payload["answers"][0][field] = value
    with pytest.raises(ValueError):
        review.check_answers(payload, review.SPEC.read_bytes())


def test_wrong_packet_and_duplicate_rows_are_rejected():
    spec = review.SPEC.read_bytes()
    payload = response()
    with pytest.raises(ValueError, match="different worksheet"):
        review.check_answers(dict(payload, packet_id="stale"), spec)
    with pytest.raises(ValueError, match="duplicate"):
        review.check_answers(dict(payload, answers=payload["answers"]*2), spec)
