"""Protocol tests use a labelled fake predictor; real-model evaluation is separate."""
import json
import struct
from threading import Event, Thread
from time import monotonic, sleep

import numpy as np
import pytest
from simple_websocket import Client, ConnectionClosed, ConnectionError
from werkzeug.serving import make_server

from app import create_app
from live_socket import AudioRing, CAPACITY, LiveSession, MAX_MESSAGE, PcmInput, ProtocolError, StableWords
from recitation import RecitationService


class PortAwareClient(Client):
    def handshake(self):
        # simple-websocket1.1 Client drops the nonstandard URL port in Host.
        # Browsers include it; retain it here to exercise a genuine origin check.
        self.host = f"{self.host}:{self.port}"
        super().handshake()


def packet(sequence, offset, samples):
    return struct.pack("<II", sequence, offset) + np.asarray(samples, dtype="<i2").tobytes()


def voice(count=14400):
    return (np.sin(np.arange(count) * .09) * 5000).astype("<i2")


def receive_until(ws, event_type, timeout=5):
    deadline, seen = monotonic() + timeout, []
    while monotonic() < deadline:
        raw = ws.receive(timeout=max(.01, deadline-monotonic()))
        if raw is not None:
            event = json.loads(raw)
            seen.append(event)
            if event["type"] == event_type:
                return event, seen
    raise AssertionError(f"No {event_type} event: {seen}")


@pytest.fixture
def socket_server(monkeypatch):
    service = RecitationService()
    service.ready = True
    monkeypatch.setattr(service, "predict_live", lambda wave: service.references[5].phonetic.phonemes)
    app = create_app(service)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    clients = []

    def connect(start=True, rate=16000):
        ws = PortAwareClient(url.replace("http:", "ws:") + "/live/ws", headers={"Origin": url})
        clients.append(ws)
        if start:
            ws.send(json.dumps(dict(type="start", version=1, start_ayah=5, sample_rate=rate, channels=1)))
            event, _ = receive_until(ws, "ready")
            assert event["version"] == 1 and event["session_id"]
        return ws

    yield app, service, connect, url
    for ws in clients:
        try:
            ws.close()
        except ConnectionClosed:
            pass
    app.extensions["live_socket_manager"].shutdown()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def stop(ws, sequence, samples):
    ws.send(json.dumps(dict(type="stop", last_sequence=sequence, total_samples=samples)))
    return receive_until(ws, "done")


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_continuous_resampler_keeps_sample_count_and_waveform(rate):
    count = rate * 2
    samples = (np.sin(np.arange(count) * 2*np.pi*440/rate) * 12000).astype("<i2")
    source = PcmInput(rate)
    chunks = []
    offset = 0
    for sequence, start in enumerate(range(0, count, 2048)):
        chunk = samples[start:start+2048]
        chunks.append(source.convert(packet(sequence, offset, chunk)))
        offset += len(chunk)
    chunks.append(source.flush())
    wave = np.concatenate(chunks)
    assert len(wave) == 32000
    assert source.input_samples == count
    expected = np.sin(np.arange(32000) * 2*np.pi*440/16000) * (12000/32768)
    assert np.max(np.abs(wave[30:-30] - expected[30:-30])) < .002
    assert source.flush().size == 0


def test_ring_keeps_latest_audio_and_bounded_storage():
    ring = AudioRing()
    wave = np.arange(13*16000, dtype=np.float32)
    for chunk in np.array_split(wave, 37):
        ring.append(chunk)
    assert ring.data.size == CAPACITY and ring.total == len(wave)
    np.testing.assert_array_equal(ring.snapshot(), wave[-CAPACITY:])
    np.testing.assert_array_equal(ring.snapshot(end=len(wave)-1000, seconds=6), wave[-97000:-1000])
    assert ring.snapshot(end=16000).size == 0


@pytest.mark.parametrize("data,code", [
    (packet(1, 0, voice(20)), "sequence"),
    (packet(0, 4, voice(20)), "sequence"),
    (b"bad", "audio_format"),
    (b"x" * (MAX_MESSAGE+2), "message_size"),
], ids=["sequence", "offset", "short", "oversize"])
def test_invalid_pcm_rejected_before_use(data, code):
    source = PcmInput(16000)
    with pytest.raises(ProtocolError) as error:
        source.convert(data)
    assert error.value.code == code
    assert source.input_samples == 0


def test_source_duration_exact_cap_then_reject():
    source = PcmInput(16000)
    for seq in range(180):
        source.convert(packet(seq, seq*16000, np.zeros(16000, dtype="<i2")))
    assert source.input_samples == 180*16000
    with pytest.raises(ProtocolError, match="سه دقیقه"):
        source.convert(packet(180, 180*16000, [0]))


def test_stabilization_requires_new_audio_and_same_error_evidence():
    stable = StableWords(29, 13)
    word = dict(index=13, state="match", errors=[])
    stable.observe([word], 1000, False)
    stable.observe([word], 1000, False)
    assert not stable.words
    stable.observe([word], 2000, False)
    assert stable.words[13]["state"] == "match"
    changed = dict(index=13, state="difference", errors=[dict(type="letter", message="حرف")])
    stable.observe([changed], 3000, False)
    assert stable.words[13]["state"] == "match"
    stable.observe([changed], 3100, True)
    assert stable.words[13]["state"] == "difference"
    assert stable.summary() == dict(reviewed=1, differences=1, unreviewed=15)


def test_cadence_pause_tail_and_silence_are_bounded(socket_server):
    app, _, _, _ = socket_server
    session = LiveSession(dict(type="start", version=1, start_ayah=5, sample_rate=16000, channels=1),
                          app.extensions["live_follower"])
    session.append(np.ones(12000, dtype=np.float32)*.1)
    assert session.candidate(monotonic()) is None  # first800ms not yet available
    session.append(np.ones(800, dtype=np.float32)*.1)
    first = session.candidate(monotonic())
    assert first and not first["settled"] and len(first["wave"]) == 12800
    session.last_analyzed = session.last_scheduled_voice = session.voice_end
    session.append(np.ones(4000, dtype=np.float32)*.1)
    assert session.candidate(monotonic()) is None  # less than500ms fresh audio
    session.append(np.zeros(8000, dtype=np.float32))
    paused = session.candidate(monotonic())
    assert paused["settled"] and paused["end"] == session.voice_end + 4000
    session.last_settled_voice = session.last_scheduled_voice = session.voice_end
    session.last_analyzed = paused["end"]
    session.append(np.zeros(16000, dtype=np.float32))
    assert session.candidate(monotonic()) is None  # long silence is not re-inferred


def test_socket_real_handshake_ack_result_and_stop(socket_server):
    _, _, connect, _ = socket_server
    ws = connect()
    samples = voice()
    ws.send(packet(0, 0, samples))
    ack, _ = receive_until(ws, "ack")
    assert ack["sequence"] == 0 and ack["input_samples"] == len(samples)
    first, _ = receive_until(ws, "result")
    assert first["updates"] == []  # one provisional observation cannot paint
    done, events = stop(ws, 0, len(samples))
    assert done["revision"] > first["revision"]
    assert len(done["updates"]) == 4
    assert done["summary"]["reviewed"] == 4
    assert done["metrics"]["total_received_samples"] == len(samples)
    assert done["metrics"]["inference_max_inflight"] == 1
    assert all(e["session_id"] == ack["session_id"] for e in events)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_resampled_socket_results(socket_server, rate):
    _, _, connect, _ = socket_server
    ws = connect(rate=rate)
    # Keep each packet below the64KB message limit.
    chunk_size = int(rate * .128)
    total = int(rate * 1.1)
    seq = -1
    for seq, offset in enumerate(range(0, total, chunk_size)):
        ws.send(packet(seq, offset, voice(min(chunk_size, total-offset))))
    done, _ = stop(ws, seq, total)
    assert abs(done["metrics"]["output_samples"] - 17600) <= 1
    assert done["updates"]


def test_second_connection_and_invalid_origin_are_rejected(socket_server):
    app, _, connect, url = socket_server
    first = connect()
    second = connect(start=False)
    error, _ = receive_until(second, "error")
    assert error["code"] == "session_busy" and error["fatal"]
    with pytest.raises(ConnectionError):
        PortAwareClient(url.replace("http:", "ws:") + "/live/ws", headers={"Origin":"https://example.com"})
    assert app.test_client().get("/live/ws", base_url="http://evil.example", headers={"Origin":"http://evil.example"}).status_code == 403
    stop(first, -1, 0)


def test_receive_continues_during_slow_inference(socket_server, monkeypatch):
    app, service, connect, _ = socket_server
    entered, release = Event(), Event()

    def slow(_wave):
        entered.set()
        assert release.wait(3)
        return service.references[5].phonetic.phonemes

    monkeypatch.setattr(service, "predict_live", slow)
    ws = connect()
    ws.send(packet(0, 0, voice()))
    receive_until(ws, "ack")
    assert entered.wait(1)
    started = monotonic()
    ws.send(packet(1, 14400, voice(2048)))
    ack, _ = receive_until(ws, "ack")
    assert ack["sequence"] == 1 and monotonic()-started < .5
    assert app.extensions["inference_lock"].locked()
    release.set()
    done, _ = stop(ws, 1, 16448)
    assert done["metrics"]["queued_inference_jobs"] == 0


def test_disconnect_releases_session_but_running_model_finishes_safely(socket_server, monkeypatch):
    app, service, connect, _ = socket_server
    entered, release = Event(), Event()

    def slow(_wave):
        entered.set()
        assert release.wait(3)
        return service.references[5].phonetic.phonemes

    monkeypatch.setattr(service, "predict_live", slow)
    ws = connect()
    ws.send(packet(0, 0, voice()))
    receive_until(ws, "ack")
    assert entered.wait(1)
    ws.close()
    manager = app.extensions["live_socket_manager"]
    deadline = monotonic()+1
    while manager.session_lock.locked() and monotonic() < deadline:
        sleep(.01)
    fresh = connect()
    release.set()
    stop(fresh, -1, 0)
    deadline = monotonic()+1
    while app.extensions["inference_lock"].locked() and monotonic() < deadline:
        sleep(.01)
    assert not app.extensions["inference_lock"].locked()


def test_model_busy_does_not_queue_and_failure_unlocks(socket_server, monkeypatch):
    app, service, connect, _ = socket_server
    lock = app.extensions["inference_lock"]
    lock.acquire()
    ws = connect()
    ws.send(packet(0, 0, voice()))
    error, _ = receive_until(ws, "error")
    assert error["code"] == "model_busy" and not error["fatal"]
    lock.release()

    def fail(_wave):
        raise RuntimeError("intentional transport-test predictor failure")

    monkeypatch.setattr(service, "predict_live", fail)
    error, events = receive_until(ws, "error")
    assert error["code"] == "processing" and error["fatal"]
    assert not any(e["type"] == "result" for e in events)
    assert not lock.locked()


@pytest.mark.parametrize("message,code", [
    (packet(3, 0, voice(2048)), "sequence"),
    (json.dumps(dict(type="stop", last_sequence=0, total_samples=1)), "sequence"),
    ("not json", "json"),
])
def test_socket_protocol_errors_close_cleanly(socket_server, message, code):
    _, _, connect, _ = socket_server
    ws = connect()
    ws.send(message)
    error, _ = receive_until(ws, "error")
    assert error["code"] == code and error["fatal"]


def test_oversized_message_is_closed_by_websocket_transport(socket_server):
    app, _, connect, _ = socket_server
    ws = connect()
    ws.send(b"x"*(MAX_MESSAGE+2))
    with pytest.raises(ConnectionClosed):
        ws.receive(timeout=2)
    assert not app.extensions["inference_lock"].locked()


def test_silence_and_not_ready_do_not_produce_colored_words(socket_server):
    _, service, connect, _ = socket_server
    ws = connect()
    ws.send(packet(0, 0, np.zeros(16000, dtype="<i2")))
    done, _ = stop(ws, 0, 16000)
    assert done["updates"] == [] and done["metrics"]["analyses"] == 0
    service.ready = False
    second = connect(start=False)
    error, _ = receive_until(second, "error")
    assert error["code"] == "not_ready"
