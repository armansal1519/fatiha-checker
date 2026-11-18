import io
import wave

import numpy as np
import pytest

from app import create_app
from audio_processing import AudioError, MAX_BYTES, decode_audio
from recitation import RecitationService, compare_prediction, make_reference


def wav_bytes(seconds=1, rate=16000, silent=False, channels=1):
    time = np.arange(int(seconds * rate)) / rate
    samples = np.zeros(len(time), dtype=np.int16) if silent else (np.sin(2 * np.pi * 220 * time) * 5000).astype(np.int16)
    if channels == 2:
        samples = np.column_stack([samples, samples]).reshape(-1)
    data = io.BytesIO()
    with wave.open(data, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.tobytes())
    return data.getvalue()


@pytest.fixture(scope="module")
def service():
    return RecitationService()


@pytest.mark.parametrize("ayah", range(1, 8))
def test_reference_and_stop(service, ayah):
    ref = service.references[ayah]
    assert ref.words and ref.text == " ".join(ref.words)
    assert ref.phonetic.phonemes[-1] in "من"
    assert compare_prediction(ref, ref.phonetic.phonemes) == []


def test_letter_vowel_gemination_and_word_mapping(service):
    ref = service.references[2]
    prediction = ref.phonetic.phonemes.replace("حَ", "هَ", 1)
    errors = compare_prediction(ref, prediction)
    assert len(errors) == 1 and errors[0]["type"] == "letter" and errors[0]["word_index"] == 0
    prediction = ref.phonetic.phonemes.replace("دُ", "دَ", 1)
    errors = compare_prediction(ref, prediction)
    assert len(errors) == 1 and errors[0]["type"] == "vowel"
    prediction = ref.phonetic.phonemes.replace("ببِ", "بِ", 1)
    errors = compare_prediction(ref, prediction)
    assert len(errors) == 1 and errors[0]["type"] == "gemination" and errors[0]["word_index"] == 2


@pytest.mark.parametrize("ayah,old,new,word_index", [
    (1, "حِ", "هِ", 3), (2, "عَاا", "غَاا", 3),
    (3, "حْم", "خْم", 0), (4, "ددِ", "ددُ", 2),
    (5, "تَ", "طَ", 3), (6, "قِ", "كِ", 2), (7, "ضُ", "ظُ", 5),
])
def test_error_maps_to_known_word_in_each_ayah(service, ayah, old, new, word_index):
    # QPS represents sukun through the absence of a vowel mark.
    old, new = old.replace("ْ", ""), new.replace("ْ", "")
    ref = service.references[ayah]
    assert old in ref.phonetic.phonemes
    errors = compare_prediction(ref, ref.phonetic.phonemes.replace(old, new, 1))
    assert len(errors) == 1
    assert errors[0]["word_index"] == word_index
    assert errors[0]["word"] == ref.words[word_index]


def test_length_only_ignored_but_missing_long_vowel_retained(service):
    ref = service.references[4]
    assert compare_prediction(ref, ref.phonetic.phonemes.replace("اا", "اااا", 1)) == []
    errors = compare_prediction(ref, ref.phonetic.phonemes.replace("اا", "", 1))
    assert errors and errors[0]["type"] == "deletion" and errors[0]["word_index"] == 0


def test_insertions_and_multi_phone_deletions(service):
    ref = service.references[5]
    errors = compare_prediction(ref, ref.phonetic.phonemes.replace("نَعبُدُ", "", 1))
    assert len(errors) >= 4
    assert {e["word_index"] for e in errors} == {1}
    errors = compare_prediction(ref, "بَ" + ref.phonetic.phonemes)
    assert errors[0]["type"] == "insertion" and errors[0]["word_index"] == 0


def test_unequal_replacement_does_not_drop_remaining_deletions(service):
    ref = service.references[5]
    errors = compare_prediction(ref, ref.phonetic.phonemes.replace("نَعبُدُ", "زَ", 1))
    assert len(errors) >= 4


def test_audio_conversion():
    samples = decode_audio(wav_bytes(rate=48000, channels=2), "stereo.wav")
    assert samples.dtype == np.float32 and samples.ndim == 1 and len(samples) == 16000


@pytest.mark.parametrize("data,name", [(b"", "a.wav"), (b"broken", "a.wav"), (b"broken", "a.txt"), (wav_bytes(silent=True), "silent.wav"), (wav_bytes(.1), "short.wav"), (wav_bytes(30.2), "long.wav"), (b"x" * (MAX_BYTES + 1), "huge.wav")], ids=["empty", "broken", "extension", "silence", "short", "long", "oversize"])
def test_audio_rejects_invalid_input(data, name):
    with pytest.raises(AudioError):
        decode_audio(data, name)


@pytest.fixture
def client(service):
    # HTTP tests isolate transport; real model tests live in evaluate_dataset.py.
    class TransportStub:
        ready = True
        verses = service.verses
        def analyze(self, audio, ayah, surah=1):
            return dict(status="no_difference", surah=surah, ayah=ayah, errors=[], message="آزمون انتقال")
    app = create_app(TransportStub())
    app.config["TESTING"] = True
    return app.test_client()


def upload(client, **fields):
    return client.post("/analyze", data={"ayah": "1", "audio": (io.BytesIO(wav_bytes()), "sample.wav"), **fields})


def test_page_assets_and_api(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'dir="rtl"' in response.text
    assert 'id="precise-surah"' in response.text
    assert 'id="verse-nav"' in response.text
    assert len(client.get('/api/quran/ayahs?surah=1').json['ayahs']) == 7
    for path in ["/static/style.css", "/static/app.js"]:
        assert client.get(path).status_code == 200
    response = upload(client, ayah="5")
    assert response.status_code == 200 and response.json["ayah"] == 5
    assert "elapsed_seconds" in response.json


@pytest.mark.parametrize("fields", [{"ayah": "0"}, {"ayah": "8"}, {"ayah": "abc"}, {"surah": "115"}, {"surah":"2","ayah":"287"}])
def test_invalid_reference(client, fields):
    assert upload(client, **fields).status_code == 400


def test_all_quran_browsing_and_non_fatiha_analysis(client):
    catalog=client.get('/api/quran/surahs').json['surahs']
    assert len(catalog)==114 and sum(s['ayah_count'] for s in catalog)==6236
    page=client.get('/api/quran/ayahs?surah=2&start=255&limit=2')
    assert page.status_code==200 and page.json['total']==286
    assert [a['ayah'] for a in page.json['ayahs']]==[255,256]
    assert all(a['surah']==2 and a['words'] for a in page.json['ayahs'])
    response=upload(client,surah='114',ayah='6')
    assert response.status_code==200 and response.json['surah']==114
    for query in ['surah=115','surah=2&start=287','surah=2&limit=0','limit=51','surah=no']:
        assert client.get('/api/quran/ayahs?'+query).status_code==400


def test_combined_health_requires_both_models_and_speech_detector(client):
    assert client.get('/health').status_code==503
    transcriber=client.application.extensions['transcription_service']
    transcriber.ready=True
    transcriber.device='mps'
    assert client.get('/health').status_code==503
    transcriber.activity_detector.ready=True
    health=client.get('/health')
    assert health.status_code==200 and health.json['ready']
    assert health.json['models']['transcription']['device']=='mps'
    assert health.json['models']['speech_activity']['device']=='cpu'
    assert health.json['models']['speech_activity']['ready']
    assert health.json['protocol_version']==2


def test_errors_busy_model_and_lock_release(client):
    lock = client.application.extensions["inference_lock"]
    lock.acquire()
    assert upload(client).status_code == 409
    lock.release()
    assert client.post("/analyze", data={"ayah": "1"}).status_code == 400
    assert not lock.locked()
    assert upload(client, audio=(io.BytesIO(b"bad"), "bad.wav")).status_code == 400
    assert not lock.locked()
    service = client.application.extensions["recitation_service"]
    service.ready = False
    assert upload(client).status_code == 503
    assert client.get("/health").status_code == 503


def test_oversize_and_cross_origin(client):
    response = client.post("/analyze", data=b"x" * (MAX_BYTES + 200000), content_type="multipart/form-data")
    assert response.status_code == 413
    response = client.post("/analyze", headers={"Origin": "https://example.org"})
    assert response.status_code == 403


def test_no_prediction_and_wrong_content_are_not_success(service, monkeypatch):
    monkeypatch.setattr(service, "predict", lambda *args: "")
    assert service.analyze(np.ones(16000), 1)["status"] == "needs_retry"
    monkeypatch.setattr(service, "predict", lambda *args: "زَزَزَزَزَ")
    assert service.analyze(np.ones(16000), 1)["status"] == "needs_retry"


def test_failed_model_load_is_not_ready(monkeypatch):
    import huggingface_hub
    def unavailable(*args, **kwargs):
        raise OSError("Intentional model download failure for test")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", unavailable)
    failed = RecitationService()
    failed.load()
    assert not failed.ready and failed.model is None
    client = create_app(failed).test_client()
    assert client.get("/health").status_code == 503
    assert "مدل آماده نیست" in client.get("/").text
    assert upload(client).status_code == 503


def test_model_runtime_failure_returns_json_and_unlocks(client, monkeypatch):
    service = client.application.extensions["recitation_service"]
    def broken(*args):
        raise RuntimeError("Intentional inference failure for test")
    monkeypatch.setattr(service, "analyze", broken)
    response = upload(client)
    assert response.status_code == 500 and response.json["status"] == "error"
    assert not client.application.extensions["inference_lock"].locked()
