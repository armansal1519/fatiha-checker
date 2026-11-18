"""Deterministic VAD state/transport tests; these do not measure speech accuracy."""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event, Lock
from time import sleep
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from live_socket import LiveSession, LiveSocketManager, RATE
from speech_activity import (CONTINUE_THRESHOLD, FRAME_SAMPLES, MIN_SPEECH_SAMPLES,
                             POST_PAD_SAMPLES, PRE_PAD_SAMPLES, SpeechActivityDetector,
                             _speech_spans)


class FrameModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.index = 0
        self.resets = 0
        self.frames = []

    def eval(self):
        return self

    def reset_states(self):
        self.resets += 1
        self.index = 0

    def __call__(self, frame, sample_rate):
        assert sample_rate == RATE
        assert frame.device.type == "cpu"
        assert frame.dtype == torch.float32
        assert frame.shape == (FRAME_SAMPLES,)
        assert torch.is_inference_mode_enabled()
        self.frames.append(frame.numpy().copy())
        probability = self.probabilities[self.index]
        self.index += 1
        if isinstance(probability, Exception):
            raise probability
        return torch.tensor(probability)


def detector(probabilities):
    activity = SpeechActivityDetector()
    activity.ready = True
    activity.model = FrameModel(probabilities)
    return activity


def audio(frames, amplitude=.1, extra=0):
    return np.full(frames*FRAME_SAMPLES+extra, amplitude, dtype=np.float32)


def test_padded_short_word_has_clipped_sample_coordinates():
    activity = detector([0]*10 + [.9, .8, .4] + [0]*20)
    wave = audio(33)
    result = activity.detect(wave)
    start, end = 10*FRAME_SAMPLES, 13*FRAME_SAMPLES
    assert result["status"] == "speech"
    assert result["spans"] == [dict(start_sample=start-PRE_PAD_SAMPLES,
                                    end_sample=end+POST_PAD_SAMPLES)]
    assert result["speech_samples"] == end+POST_PAD_SAMPLES-start+PRE_PAD_SAMPLES
    assert result["evaluated_frames"] == 33
    assert result["rms"] == pytest.approx(.1)
    assert result["peak"] == pytest.approx(.1)


def test_onset_continuation_minimum_and_nearby_padding_merge():
    # Speech can fall below the onset threshold after beginning, while two
    # distinct short onsets separated by quiet must not fabricate a word.
    assert _speech_spans([.9, .4, .4], MIN_SPEECH_SAMPLES) == [dict(
        start_sample=0, end_sample=MIN_SPEECH_SAMPLES)]
    assert _speech_spans([.9, .9, 0, .9, .9], 5*FRAME_SAMPLES) == []
    probabilities = [.9]*3 + [0]*8 + [.9]*3
    assert _speech_spans(probabilities, len(probabilities)*FRAME_SAMPLES) == [dict(
        start_sample=0, end_sample=len(probabilities)*FRAME_SAMPLES)]


def test_distinct_speech_spans_preserve_the_intervening_silence():
    probabilities = [.9]*3 + [0]*20 + [.9]*3
    assert _speech_spans(probabilities, len(probabilities)*FRAME_SAMPLES) == [
        dict(start_sample=0, end_sample=3*FRAME_SAMPLES+POST_PAD_SAMPLES),
        dict(start_sample=23*FRAME_SAMPLES-PRE_PAD_SAMPLES, end_sample=26*FRAME_SAMPLES),
    ]


@pytest.mark.parametrize("probabilities,extra,reason", [
    ([.9, .9], 0, "speech_too_short"),
    ([.9, .9, .9], 1, "speech_too_short"),
    ([.4]*8, 0, "borderline_activity"),
])
def test_unconfirmed_activity_is_uncertain(probabilities, extra, reason):
    # A partial final frame cannot turn 64ms+1 sample into a confirmed 96ms word.
    count = len(probabilities)*FRAME_SAMPLES if not extra else 2*FRAME_SAMPLES+extra
    result = detector(probabilities).detect(np.full(count, .1, np.float32))
    assert result["status"] == "uncertain"
    assert result["reason"] == reason
    assert result["spans"] == [] and result["speech_samples"] == 0


def test_high_energy_noise_is_not_speech_and_quiet_input_can_be_speech():
    assert detector([.02]*20).detect(audio(20, .3))["status"] == "no_speech"
    assert detector([.9]*3).detect(audio(3, .0005))["status"] == "speech"


def test_exact_probability_threshold_is_included():
    result = detector([.5, CONTINUE_THRESHOLD, CONTINUE_THRESHOLD]).detect(audio(3))
    assert result["status"] == "speech"


def test_zero_and_empty_input_skip_frame_inference():
    activity = detector([])
    for wave in [np.zeros(FRAME_SAMPLES, np.float32), np.array([], np.float32)]:
        result = activity.detect(wave)
        assert result["status"] == "no_speech" and result["reason"] == "below_audio_floor"
    assert activity.model.frames == []


def test_not_ready_and_invalid_input_fail_closed():
    activity = SpeechActivityDetector()
    assert activity.detect(audio(4))["reason"] == "detector_not_ready"
    for wave in [np.array([np.nan]), np.array([np.inf]), np.zeros((2, 2))]:
        result = activity.detect(wave)
        assert result["status"] == "uncertain"
        assert result["reason"] == "invalid_audio" and result["spans"] == []


def test_frame_padding_and_per_window_state_are_isolated():
    activity = detector([.9, .9, .9, 0])
    wave = audio(3, extra=7)
    first = activity.detect(wave)
    second = activity.detect(wave)
    assert first == second
    assert activity.model.resets == 4  # start and finally for both windows
    assert len(activity.model.frames) == 8
    assert np.all(activity.model.frames[3][:7] == np.float32(.1))
    assert np.all(activity.model.frames[3][7:] == 0)
    assert not hasattr(activity, "wave") and not hasattr(activity, "probabilities")


@pytest.mark.parametrize("failure", [RuntimeError("bad model"), float("nan"), 1.1])
def test_inference_failure_resets_then_disables_detector(failure):
    activity = detector([failure])
    model = activity.model
    result = activity.detect(audio(1))
    assert result["status"] == "uncertain" and result["reason"] == "detector_failed"
    assert not activity.ready and activity.model is None and activity.error
    assert model.resets == 2
    assert activity.detect(audio(1))["reason"] == "detector_not_ready"


def test_concurrent_windows_are_serialized():
    activity = detector([.9]*3)
    base_model = activity.model
    active = 0
    maximum = 0
    state_lock = Lock()

    class SlowModel:
        def reset_states(self):
            base_model.reset_states()

        def __call__(self, frame, rate):
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(active, maximum)
            sleep(.005)
            output = base_model(frame, rate)
            with state_lock:
                active -= 1
            return output

    activity.model = SlowModel()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(activity.detect, [audio(3), audio(3)]))
    assert all(result["status"] == "speech" for result in results)
    assert maximum == 1 and base_model.resets == 4


def test_load_uses_only_offline_verified_cpu_asset(monkeypatch, tmp_path):
    import model_assets
    calls = []
    model = FrameModel([.9]*3)
    path = tmp_path / "silero_vad.jit"

    def asset(*, offline):
        calls.append(("asset", offline))
        return path

    def load(path_string, *, map_location):
        calls.append(("load", path_string, map_location))
        return model

    monkeypatch.setattr(model_assets, "ensure_activity_model", asset)
    monkeypatch.setattr(torch.jit, "load", load)
    activity = SpeechActivityDetector()
    activity.load()
    activity.load()
    assert calls == [("asset", True), ("load", str(path), "cpu")]
    assert activity.ready and activity.error is None and model.resets == 1


def test_failed_local_load_stays_uncertain(monkeypatch):
    import model_assets

    def unavailable(**kwargs):
        raise FileNotFoundError("offline checkpoint absent")

    monkeypatch.setattr(model_assets, "ensure_activity_model", unavailable)
    activity = SpeechActivityDetector()
    activity.load()
    assert not activity.ready and "offline checkpoint absent" in activity.error
    assert activity.detect(audio(3))["status"] == "uncertain"


@pytest.mark.parametrize("status", ["no_speech", "uncertain"])
def test_worker_publishes_activity_but_skips_phonetic_model(status):
    activity = dict(status=status, reason="vad_test", spans=[], rms=.1, peak=.2,
                    speech_samples=0, evaluated_frames=3)
    transcript = dict(text="", words=[], speech=False, audio_activity=activity)

    def unexpected_model(wave):
        raise AssertionError("Rejected activity must not run the phonetic model")

    lock = Lock()
    manager = LiveSocketManager(SimpleNamespace(predict_live_timed=unexpected_model),
                                SimpleNamespace(transcribe=lambda wave: transcript), None, lock)
    queue = Queue(maxsize=4)
    lock.acquire()
    try:
        timings = manager.infer(dict(wave=audio(3)), queue, Event())
        first, second = queue.get_nowait(), queue.get_nowait()
        assert first["stage"] == "transcript" and first["result"]["audio_activity"] == activity
        assert second["stage"] == "feedback" and second["result"] == dict(audio_activity=activity)
        assert timings["phonetic_skipped"] and timings["phonetic_ms"] == 0
        assert timings["transcription_skipped"] and timings["vad_frames"] == 3
        assert first["timings"]["transcription_skipped"]
        assert queue.empty() and not lock.locked()
    finally:
        manager.shutdown()


def test_confirmed_speech_with_blank_asr_still_runs_independent_phonetics():
    calls = []
    lock = Lock()
    transcript = dict(text="", words=[], speech=False, audio_activity=dict(status="speech"))
    manager = LiveSocketManager(SimpleNamespace(predict_live_timed=lambda wave: calls.append(True) or dict(groups=[])),
                                SimpleNamespace(transcribe=lambda wave: transcript), None, lock)
    queue = Queue(maxsize=4)
    lock.acquire()
    try:
        timings = manager.infer(dict(wave=audio(3)), queue, Event())
        assert calls == [True] and queue.qsize() == 2
        assert timings["phonetic_skipped"] is False
        assert timings["transcription_skipped"] is False
    finally:
        manager.shutdown()


def test_activity_metadata_reaches_transcript_event_separately_from_words():
    tracker = SimpleNamespace(update_transcript=lambda *args, **kwargs: dict(
        replace_from=0, tokens=[], committed_count=0, summary={}))
    session = LiveSession(dict(type="start", version=2, sample_rate=RATE, channels=1), None,
                          tracker_factory=lambda *args: tracker)
    activity = dict(status="uncertain", reason="borderline_activity", spans=[])
    event = session.apply_stage(dict(stage="transcript", result=dict(audio_activity=activity),
                                    job=dict(start=0, end=100, settled=False), timings={}))
    assert event["audio_activity"] == activity
    assert event["tokens"] == [] and event["replace_from"] == 0
