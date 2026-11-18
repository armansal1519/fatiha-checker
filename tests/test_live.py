import io

import numpy as np
import pytest

from app import create_app
from live_tracking import LiveFollower
from recitation import RecitationService, make_reference
from test_app import wav_bytes


@pytest.fixture(scope="module")
def tracker():
    return LiveFollower()


def test_all_seven_ayat_follow_in_order_without_future_errors(tracker):
    cursor, seen = 0, set()
    for last in range(1, len(tracker.words)):
        a = tracker.word_ranges[max(0, last - 5)][0]
        b = tracker.word_ranges[last][1]
        prediction = "".join(tracker.groups[a:b])
        result = tracker.align(prediction, cursor, settled=last == len(tracker.words) - 1)
        assert result["status"] == "tracking"
        assert result["cursor"] >= cursor
        assert all(u["index"] <= last and u["state"] == "match" for u in result["updates"])
        seen.update(u["index"] for u in result["updates"])
        cursor = result["cursor"]
    assert seen == set(range(29))
    assert result["complete"] and cursor == 29


def test_unread_tail_and_clipped_word_never_red(tracker):
    result = tracker.align("".join(tracker.groups[:5]))
    assert result["cursor"] <= 2
    assert all(u["index"] == 0 and u["state"] == "match" for u in result["updates"])
    a, b = tracker.word_ranges[14]
    end = tracker.word_ranges[16][1]
    result = tracker.align("".join(tracker.groups[a + 1:end - 1]), 15, start_ayah=5, settled=True)
    assert 14 not in [u["index"] for u in result["updates"]]
    assert 16 not in [u["index"] for u in result["updates"]]


@pytest.mark.parametrize("old,new,kind", [("نَعبُدُ", "نَهبُدُ", "letter"), ("بُ", "بَ", "vowel"), ("ييَ", "يَ", "gemination"), ("عبُ", "بُ", "deletion")])
def test_live_basic_errors(tracker, old, new, kind):
    prediction = tracker.references[5].phonetic.phonemes.replace(old, new, 1)
    result = tracker.align(prediction, 13, 5, settled=True)
    errors = [e for word in result["updates"] for e in word["errors"]]
    assert any(e["type"] == kind for e in errors)


def test_repeated_rahman_rahim_does_not_jump_ahead(tracker):
    p = tracker.references[3].phonetic.phonemes
    result = tracker.align(p, 8, settled=True)
    assert {u["ayah"] for u in result["updates"]} == {3}
    assert result["cursor"] == 10
    first = tracker.align(tracker.references[1].phonetic.phonemes, 0, settled=True)
    assert first["cursor"] == 4


def test_stopping_or_joining_ayat_accepted(tracker):
    text = tracker.references[1].text + " " + tracker.references[2].text
    connected = make_reference(1, text).phonetic.phonemes
    result = tracker.align(connected, settled=True)
    assert result["status"] == "tracking"
    assert all(u["state"] == "match" for u in result["updates"])


def test_length_only_ignored_and_missing_long_vowel_reported(tracker):
    prediction = tracker.references[4].phonetic.phonemes
    changed_length = tracker.align(prediction.replace("اا", "اااا"), 10, 4, True)
    assert all(u["state"] == "match" for u in changed_length["updates"])
    missing = tracker.align(prediction.replace("اا", ""), 10, 4, True)
    assert any(u["state"] == "difference" for u in missing["updates"])


@pytest.mark.parametrize("prediction", ["", "زَ", "زَزَزَزَزَ"])
def test_empty_and_unrelated_prediction_cannot_color_words(tracker, prediction):
    result = tracker.align(prediction, 13, 5)
    assert result["status"] == "listening" and result["updates"] == [] and result["cursor"] == 13


@pytest.fixture
def live_client(monkeypatch):
    service = RecitationService()
    service.ready = True
    monkeypatch.setattr(service, "predict_live", lambda *args: service.references[5].phonetic.phonemes)
    app = create_app(service)
    return app.test_client()


def send(client, **fields):
    return client.post("/live/analyze", data={"cursor":"13", "start_ayah":"5", "sequence":"7", "settled":"true", "audio":(io.BytesIO(wav_bytes(2)), "live.wav"), **fields})


def test_live_page_and_api(live_client):
    page = live_client.get("/live")
    assert page.status_code == 200 and page.text.count('class="live-word"') == 0
    assert 'id="heard-transcript"' in page.text
    for filename in ["live.js", "live.css", "live-audio.js", "live-cadence.js", "pcm-worklet.js"]:
        assert live_client.get("/static/" + filename).status_code == 200
    result = send(live_client)
    assert result.status_code == 200 and result.json["sequence"] == 7
    assert len(result.json["updates"]) == 4


@pytest.mark.parametrize("fields", [{"cursor":"-1"},{"cursor":"30"},{"start_ayah":"8"},{"sequence":"bad"},{"sequence":"-1"}])
def test_invalid_live_metadata(live_client, fields):
    assert send(live_client, **fields).status_code == 400


def test_live_limit_model_and_shared_lock(live_client):
    assert send(live_client, audio=(io.BytesIO(wav_bytes(12.2)), "long.wav")).status_code == 400
    assert send(live_client, audio=(io.BytesIO(wav_bytes(silent=True)), "silence.wav")).status_code == 400
    lock = live_client.application.extensions["inference_lock"]
    lock.acquire()
    assert send(live_client).status_code == 409
    lock.release()
    assert live_client.post("/live/analyze", headers={"Origin":"https://example.org"}).status_code == 403
    live_client.application.extensions["recitation_service"].ready = False
    assert send(live_client).status_code == 503
    assert not lock.locked()


def test_live_inference_failure_unlocks_and_does_not_return_old_success(live_client, monkeypatch):
    def broken(*args):
        raise RuntimeError("Intentional live model failure")
    monkeypatch.setattr(live_client.application.extensions["recitation_service"], "predict_live", broken)
    response = send(live_client)
    assert response.status_code == 500 and response.json["status"] == "error"
    assert not live_client.application.extensions["inference_lock"].locked()


def test_live_cache_requires_ready_model_and_audio_match(monkeypatch):
    import torch
    from types import SimpleNamespace
    service = RecitationService()
    wave = np.ones(16000, dtype=np.float32)
    with pytest.raises(RuntimeError):
        service.predict_live(wave)
    calls = []
    def processor(waves, **kwargs):
        calls.append(waves[0].copy())
        return {"input_values":torch.tensor([[1.0]])}
    def network(**kwargs):
        assert torch.is_inference_mode_enabled()
        # CTC blank, repeated ب, blank, ب, trailing blank => بب.
        return ({"phonemes":torch.tensor([[[10.,0.],[0.,10.],[0.,10.],[10.,0.],[0.,10.],[10.,0.]]])},)
    service.model = SimpleNamespace(processor=processor,model=network,device="cpu",dtype=torch.float32,
                                   multi_level_tokenizer=SimpleNamespace(id_to_vocab={"phonemes":{0:"",1:"ب"}}))
    service.ready = True
    assert service.predict_live(wave) == "بب"
    assert service.predict_live(wave.copy()) == "بب"
    assert len(calls) == 1
    wave[0] = .25
    assert service.predict_live(wave) == "بب" and len(calls) == 2
