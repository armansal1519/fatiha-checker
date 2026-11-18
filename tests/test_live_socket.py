"""Version 2 transport tests use labelled fake ASR and phonetic services.

These checks establish ordering, coverage and isolation, not model accuracy.
"""
import json
import struct
from threading import Event, Lock, Thread
from time import monotonic, sleep
from types import SimpleNamespace

from flask import Flask
import numpy as np
import pytest
from simple_websocket import Client, ConnectionClosed, ConnectionError
from werkzeug.serving import make_server

from live_socket import (AudioRing, CAPACITY, FIRST_AUDIO, FRESH_AUDIO, LiveSession,
                         MAX_MESSAGE, MAX_SECONDS, OVERLAP, PcmInput, ProtocolError,
                         RATE, WINDOW, parse_json, register_live_socket)


class PortAwareClient(Client):
    def handshake(self):
        self.host = f"{self.host}:{self.port}"
        super().handshake()


class FakeCorpus:
    def reference(self, surah, ayah):
        limits = {1: 7, 2: 286, 36: 83, 114: 6}
        if surah not in limits or not 1 <= ayah <= limits[surah]:
            raise ValueError("Invalid test reference")
        return SimpleNamespace(surah=surah, ayah=ayah)


class FakeTracker:
    def __init__(self, corpus, surah=None, ayah=None):
        self.tokens = []
        self.committed_count = 0
        self.current_reference = {"surah": surah, "ayah": ayah} if surah else None

    def update_transcript(self, result, start_sample, end_sample, settled=False):
        self.tokens = [{"id": f"w{i}", "text": w["text"],
                        "start_ms": start_sample / 16 + w["start_ms"],
                        "end_ms": start_sample / 16 + w["end_ms"], "state": "pending"}
                       for i, w in enumerate(result["words"])]
        self.committed_count = len(self.tokens) if settled else 0
        return dict(replace_from=0, tokens=self.tokens, committed_count=self.committed_count, summary=self.summary())

    def update_phonemes(self, result, start_sample, end_sample, settled=False):
        return dict(updates=[dict(id=w["id"], state="difference", errors=[dict(type="letter", message="خطای آزمایشی")])
                             for w in self.tokens], gaps=[], reference=self.current_reference,
                    status="tracking" if self.current_reference else "ambiguous", message="آزمون انتقال", summary=self.summary())

    def select_reference(self, surah, ayah):
        self.current_reference = dict(surah=surah, ayah=ayah)
        return dict(updates=[], gaps=[], reference=self.current_reference, status="tracking", summary=self.summary())

    def summary(self):
        return dict(reviewed=0, differences=0, unreviewed=len(self.tokens))


def packet(sequence, offset, samples):
    return struct.pack("<II", sequence, offset) + np.asarray(samples, dtype="<i2").tobytes()


def voice(count=FIRST_AUDIO):
    return (np.sin(np.arange(count) * .09) * 5000).astype("<i2")


def receive_until(ws, event_type, timeout=8):
    deadline, seen = monotonic()+timeout, []
    while monotonic() < deadline:
        raw = ws.receive(timeout=max(.01, deadline-monotonic()))
        if raw is not None:
            event = json.loads(raw)
            seen.append(event)
            if event["type"] == event_type:
                return event, seen
    raise AssertionError(f"No {event_type} event: {seen}")


def start_message(**overrides):
    return dict(type="start", version=2, start_surah=None, start_ayah=None,
                sample_rate=RATE, channels=1, **overrides)


@pytest.fixture
def socket_server():
    service = SimpleNamespace(ready=True, predict_live_timed=lambda wave: dict(text="اختبار", groups=[]))
    transcriber = SimpleNamespace(ready=True, transcribe=lambda wave: dict(
        text="لفظ غلط", words=[dict(text="لفظ", start_ms=0, end_ms=400),
                                dict(text=" غلط", start_ms=400, end_ms=800)], speech=True))
    app = Flask(__name__)
    lock = Lock()
    app.extensions["inference_lock"] = lock
    manager = register_live_socket(app, service, transcriber, FakeCorpus(), lock, tracker_factory=FakeTracker)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    clients = []

    def connect(start=True, rate=RATE, **kwargs):
        ws = PortAwareClient(url.replace("http:", "ws:") + "/live/ws", headers={"Origin": url})
        clients.append(ws)
        if start:
            message = start_message()
            message.update(sample_rate=rate, **kwargs)
            ws.send(json.dumps(message))
            event, _ = receive_until(ws, "ready")
            assert event["version"] == 2 and event["session_id"]
        return ws

    yield app, service, transcriber, connect, url
    for ws in clients:
        try:
            ws.close()
        except ConnectionClosed:
            pass
    manager.shutdown()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def stop(ws, sequence, samples):
    ws.send(json.dumps(dict(type="stop", last_sequence=sequence, total_samples=samples)))
    return receive_until(ws, "done")


def send_audio(ws, samples, rate=RATE, *, offset=0, sequence=0):
    block = round(.128 * rate)
    for begin in range(0, len(samples), block):
        part = samples[begin:begin+block]
        ws.send(packet(sequence, offset+begin, part))
        sequence += 1
    return sequence-1, offset+len(samples)


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_continuous_resampler_keeps_sample_count_and_waveform(rate):
    count = rate*2
    source = PcmInput(rate)
    samples = (np.sin(np.arange(count)*2*np.pi*440/rate)*12000).astype("<i2")
    chunks = []
    for sequence, start in enumerate(range(0, count, 2048)):
        chunks.append(source.convert(packet(sequence, start, samples[start:start+2048])))
    chunks.append(source.flush())
    wave = np.concatenate(chunks)
    assert len(wave) == 32000
    expected = np.sin(np.arange(32000)*2*np.pi*440/16000)*(12000/32768)
    assert np.max(np.abs(wave[30:-30]-expected[30:-30])) < .002
    assert source.flush().size == 0


def test_ring_keeps_thirty_seconds_and_strict_slice_rejects_lost_audio():
    ring = AudioRing()
    wave = np.arange(33*RATE, dtype=np.float32)
    for chunk in np.array_split(wave, 37):
        ring.append(chunk)
    assert ring.data.size == CAPACITY
    np.testing.assert_array_equal(ring.snapshot(), wave[-CAPACITY:])
    np.testing.assert_array_equal(ring.slice(4*RATE, 5*RATE), wave[4*RATE:5*RATE])
    with pytest.raises(ProtocolError, match="عقب"):
        ring.slice(0, RATE)


@pytest.mark.parametrize("data,code", [
    (packet(1, 0, voice(20)), "sequence"), (packet(0, 4, voice(20)), "sequence"),
    (b"bad", "audio_format"), (b"x"*(MAX_MESSAGE+2), "message_size"),
], ids=['sequence','offset','short','oversize'])
def test_invalid_pcm_rejected_before_use(data, code):
    source = PcmInput(RATE)
    with pytest.raises(ProtocolError) as error:
        source.convert(data)
    assert error.value.code == code
    assert source.input_samples == 0


def test_source_duration_exact_thirty_minute_cap_then_reject():
    source = PcmInput(RATE)
    for seq in range(MAX_SECONDS):
        source.convert(packet(seq, seq*RATE, np.zeros(RATE, dtype="<i2")))
    assert source.input_samples == MAX_SECONDS*RATE
    with pytest.raises(ProtocolError, match="۳۰ دقیقه"):
        source.convert(packet(MAX_SECONDS, MAX_SECONDS*RATE, [0]))


@pytest.mark.parametrize("overrides,code", [
    ({"version": 1}, "version"), ({"sample_rate": 8000}, "sample_rate"),
    ({"sample_rate": True}, "sample_rate"), ({"channels": 2}, "sample_rate"),
    ({"start_surah": 2, "start_ayah": None}, "ayah"),
    ({"start_surah": 114, "start_ayah": 7}, "ayah"),
])
def test_session_start_validation_and_explicit_v1_refresh(overrides, code):
    message = start_message()
    message.update(overrides)
    with pytest.raises(ProtocolError) as error:
        LiveSession(message, FakeCorpus(), FakeTracker)
    assert error.value.code == code


def complete_job(session, job):
    session.job = job
    session.future = object()
    session.finish_job(dict(worker_ms=1))
    session.last_completed = 0


def test_cadence_contiguous_backlog_pause_and_stop_never_skip_old_audio():
    session = LiveSession(start_message(), FakeCorpus(), FakeTracker)
    session.append(np.ones(FIRST_AUDIO-1, dtype=np.float32)*.1)
    assert session.candidate(monotonic()) is None
    session.append(np.ones(1, dtype=np.float32)*.1)
    first = session.candidate(monotonic())
    assert (first["start"], first["end"]) == (0, FIRST_AUDIO)
    complete_job(session, first)
    session.append(np.ones(FRESH_AUDIO-1, dtype=np.float32)*.1)
    assert session.candidate(monotonic()) is None
    session.append(np.ones(1, dtype=np.float32)*.1)
    second = session.candidate(monotonic())
    assert second["start"] == 0 and second["end"] == FIRST_AUDIO+FRESH_AUDIO
    complete_job(session, second)
    # A slow job leaves more than one window of unprocessed speech. The next
    # job must extend the oldest coverage, not pick the newest eight seconds.
    session.append(np.ones(24*RATE, dtype=np.float32)*.1)
    session.stopping = True
    while True:
        old = session.analyzed_samples
        job = session.candidate(monotonic())
        assert job["start"] <= old and len(job["wave"]) <= WINDOW
        assert job["end"] > old
        complete_job(session, job)
        if job["final"]:
            break
    assert session.analyzed_samples == session.ring.total
    assert session.committed_samples == session.ring.total
    assert all(a[1] == b[0] for a,b in zip(session.coverage_intervals, session.coverage_intervals[1:]))
    assert sum(b-a for a,b,_ in session.coverage_intervals) == session.ring.total


def test_backlog_stops_before_overwriting_unprocessed_samples():
    session = LiveSession(start_message(), FakeCorpus(), FakeTracker)
    session.append(np.ones(CAPACITY, dtype=np.float32)*.1)
    with pytest.raises(ProtocolError) as error:
        session.append(np.ones(1, dtype=np.float32)*.1)
    assert error.value.code == "backlog"
    assert session.ring.total == CAPACITY


def test_thirty_minute_silence_has_bounded_buffer_and_explicit_complete_coverage():
    session = LiveSession(start_message(), FakeCorpus(), FakeTracker)
    for _ in range(MAX_SECONDS):
        session.append(np.zeros(RATE, dtype=np.float32))
        assert session.candidate(monotonic()) is None
    assert session.analyzed_samples == MAX_SECONDS*RATE
    assert session.skipped_silence_samples == MAX_SECONDS*RATE
    assert session.max_retained_samples == CAPACITY
    assert session.model_samples == 0


def test_socket_handshake_raw_transcript_feedback_and_final_coverage(socket_server):
    _, _, _, connect, _ = socket_server
    ws = connect(start_surah=2, start_ayah=255)
    seq, total = send_audio(ws, voice(FIRST_AUDIO+6000))
    transcript, seen = receive_until(ws, "transcript")
    assert "".join(w["text"] for w in transcript["tokens"]) == "لفظ غلط"
    feedback, _ = receive_until(ws, "feedback")
    assert feedback["reference"] == dict(surah=2, ayah=255)
    assert transcript["revision"] < feedback["revision"]
    done, tail = stop(ws, seq, total)
    assert done["metrics"]["coverage_complete"]
    assert done["analyzed_samples"] == total
    assert done["metrics"]["model_samples"] == total
    assert done["metrics"]["queued_inference_jobs"] == 0
    assert done["metrics"]["inference_max_inflight"] == 1
    assert all(e["session_id"] == transcript["session_id"] for e in seen+tail)
    revisions = [e["revision"] for e in tail if "revision" in e]
    assert revisions == sorted(set(revisions))


@pytest.mark.parametrize("rate", [44100, 48000])
def test_resampled_socket_results_cover_exact_received_audio(socket_server, rate):
    _, _, _, connect, _ = socket_server
    ws = connect(rate=rate)
    seq, total = send_audio(ws, voice(int(rate*2.2)), rate)
    done, _ = stop(ws, seq, total)
    assert abs(done["metrics"]["output_samples"]-35200) <= 1
    assert done["metrics"]["coverage_complete"]


def test_asr_stage_arrives_while_phonetic_inference_is_still_blocked(socket_server, monkeypatch):
    app, service, _, connect, _ = socket_server
    entered, release = Event(), Event()
    def slow(_wave):
        entered.set()
        assert release.wait(5)
        return dict(text="test", groups=[])
    monkeypatch.setattr(service, "predict_live_timed", slow)
    ws = connect()
    seq,total = send_audio(ws, voice())
    transcript,_ = receive_until(ws, "transcript")
    assert entered.is_set() and transcript["tokens"]
    started = monotonic()
    ws.send(packet(seq+1,total,voice(2048)))
    ack,_ = receive_until(ws,"ack")
    assert ack["sequence"] == seq+1 and monotonic()-started < .5
    assert app.extensions["inference_lock"].locked()
    release.set()
    done,_ = stop(ws,seq+1,total+2048)
    assert done["metrics"]["coverage_complete"]


def test_stop_drains_multiple_backlogged_windows_even_during_cooldown(socket_server, monkeypatch):
    _, _, transcriber, connect, _ = socket_server
    entered, release = Event(), Event()
    original = transcriber.transcribe
    def slow_first(wave):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(wave)
    monkeypatch.setattr(transcriber,"transcribe",slow_first)
    ws = connect()
    seq,total = send_audio(ws,voice(FIRST_AUDIO))
    assert entered.wait(2)
    seq,total = send_audio(ws,voice(20*RATE),offset=total,sequence=seq+1)
    ws.send(json.dumps(dict(type="stop",last_sequence=seq,total_samples=total)))
    release.set()
    done,_ = receive_until(ws,"done")
    assert done["metrics"]["analyses"] >= 4
    assert done["metrics"]["model_samples"] == total
    intervals = done["metrics"]["coverage_intervals"]
    assert intervals[0][0] == 0 and intervals[-1][1] == total
    assert all(a[1] == b[0] for a,b in zip(intervals,intervals[1:]))


def test_manual_reference_changes_metadata_without_starting_a_new_session(socket_server):
    _, _, _, connect, _ = socket_server
    ws=connect()
    ws.send(json.dumps(dict(type="reference",surah=114,ayah=1)))
    feedback,_=receive_until(ws,"feedback")
    assert feedback["reference"] == dict(surah=114,ayah=1)
    done,_=stop(ws,-1,0)
    assert done["session_id"] == feedback["session_id"]


def test_second_connection_and_invalid_origin_are_rejected(socket_server):
    app, _, _, connect,url = socket_server
    first=connect()
    second=connect(start=False)
    error,_=receive_until(second,"error")
    assert error["code"] == "session_busy" and error["fatal"]
    with pytest.raises(ConnectionError):
        PortAwareClient(url.replace("http:","ws:")+"/live/ws",headers={"Origin":"https://example.com"})
    assert app.test_client().get('/live/ws',base_url='http://evil.example',headers={'Origin':'http://evil.example'}).status_code == 403
    stop(first,-1,0)


def test_disconnect_releases_session_without_cross_session_results(socket_server, monkeypatch):
    app,service,_,connect,_=socket_server
    entered,release=Event(),Event()
    def slow(_wave):
        entered.set()
        assert release.wait(5)
        return dict(text="test",groups=[])
    monkeypatch.setattr(service,"predict_live_timed",slow)
    old=connect()
    send_audio(old,voice())
    assert entered.wait(2)
    old.close()
    manager=app.extensions['live_socket_manager']
    deadline=monotonic()+2
    while manager.session_lock.locked() and monotonic()<deadline:
        sleep(.01)
    fresh=connect()
    release.set()
    done,events=stop(fresh,-1,0)
    assert not any(e['type'] in {'transcript','feedback'} for e in events)
    assert all(e['session_id']==done['session_id'] for e in events)
    deadline=monotonic()+2
    while app.extensions['inference_lock'].locked() and monotonic()<deadline:
        sleep(.01)
    assert not app.extensions['inference_lock'].locked()


def test_busy_model_does_not_queue_and_inference_failure_unlocks(socket_server, monkeypatch):
    app,service,_,connect,_=socket_server
    lock=app.extensions['inference_lock']
    lock.acquire()
    ws=connect()
    send_audio(ws,voice())
    error,_=receive_until(ws,'error')
    assert error['code']=='model_busy' and not error['fatal']
    lock.release()
    def fail(_wave):
        raise RuntimeError('intentional fake phonetic failure')
    monkeypatch.setattr(service,'predict_live_timed',fail)
    error,events=receive_until(ws,'error')
    assert error['code']=='processing' and error['fatal']
    assert not any(e['type']=='feedback' for e in events)
    assert not lock.locked()


@pytest.mark.parametrize('message,code',[
    (packet(3,0,voice(2048)),'sequence'),
    (json.dumps(dict(type='stop',last_sequence=0,total_samples=1)),'sequence'),
    ('not json','json'),
    (json.dumps(dict(type='reference',surah=114,ayah=7)),'ayah'),
], ids=['sequence','stop-mismatch','json','reference'])
def test_socket_protocol_errors_close_cleanly(socket_server,message,code):
    _,_,_,connect,_=socket_server
    ws=connect()
    ws.send(message)
    error,_=receive_until(ws,'error')
    assert error['code']==code and error['fatal']


def test_old_protocol_is_rejected_with_refresh_message(socket_server):
    _,_,_,connect,_=socket_server
    ws=connect(start=False)
    ws.send(json.dumps(dict(type='start',version=1,start_ayah=1,sample_rate=RATE,channels=1)))
    error,_=receive_until(ws,'error')
    assert error['code']=='version' and 'تازه' in error['message']


def test_oversized_message_is_closed_by_websocket_transport(socket_server):
    app,_,_,connect,_=socket_server
    ws=connect()
    ws.send(b'x'*(MAX_MESSAGE+2))
    with pytest.raises(ConnectionClosed):
        ws.receive(timeout=2)
    assert not app.extensions['inference_lock'].locked()


def test_silence_and_missing_transcriber_do_not_create_text(socket_server):
    _,_,transcriber,connect,_=socket_server
    ws=connect()
    seq,total=send_audio(ws,np.zeros(3*RATE,dtype='<i2'))
    done,events=stop(ws,seq,total)
    assert not any(e['type'] in {'transcript','feedback'} for e in events)
    assert done['metrics']['analyses']==0
    assert done['metrics']['skipped_silence_samples']==total
    transcriber.ready=False
    second=connect(start=False)
    error,_=receive_until(second,'error')
    assert error['code']=='not_ready'


def test_running_snapshot_protects_its_next_overlap_but_not_an_unbounded_backlog():
    session=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    session.append(np.ones(WINDOW,dtype=np.float32)*.1)
    job=session.candidate(monotonic())
    session.job=job
    session.future=object()
    saved=job['wave'].copy()
    # Anchored/growing ASR contexts protect the full eight-second history,
    # not only the next nominal two-second overlap.
    session.append(np.ones(22*RATE,dtype=np.float32)*.2)
    assert session.ring.oldest == 0
    np.testing.assert_array_equal(job['wave'],saved)
    with pytest.raises(ProtocolError,match='عقب'):
        session.append(np.ones(1,dtype=np.float32)*.1)
    session.finish_job(dict(worker_ms=1))
    session.last_completed=0
    follow=session.candidate(monotonic())
    assert follow['start']==WINDOW-OVERLAP
    assert follow['end']==2*WINDOW-OVERLAP


def test_silence_coverage_metadata_is_coalesced_not_per_pcm_packet():
    session=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    for _ in range(10000):
        session.append(np.zeros(1,dtype=np.float32))
    assert session.coverage_intervals == [[0,10000,'silence']]
    assert session.metrics()['coverage_complete']


def test_quiet_audio_accepted_by_asr_is_not_silently_skipped_by_transport():
    from transcription import TranscriptionService
    wave=(.0005*np.sin(2*np.pi*220*np.arange(FIRST_AUDIO)/RATE)).astype(np.float32)
    service=TranscriptionService()
    service.ready=True
    # This tests energy transport, with speech classification independently
    # supplied; the synthetic tone is not evidence of actual speech accuracy.
    service.activity_detector=SimpleNamespace(detect=lambda samples: dict(
        status='speech',spans=[dict(start_sample=0,end_sample=len(samples))]))
    calls=[]
    service.pipeline=lambda *args,**kwargs: calls.append(1) or dict(
        text=' كلام',chunks=[dict(text=' كلام',timestamp=(0.0,1.5))])
    assert service.transcribe(wave)['speech'] and calls==[1]
    session=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    for block in np.array_split(wave,16): session.append(block)
    job=session.candidate(monotonic())
    assert session.voice_end==FIRST_AUDIO and session.skipped_silence_samples==0
    assert job is not None and job['start']==0 and job['end']==FIRST_AUDIO


def test_transport_audio_floor_is_inclusive_and_zero_stays_silence(monkeypatch):
    # Use an exactly representable binary value to test >= independently of
    # float32 rounding at the production 0.0001 boundary.
    monkeypatch.setattr('live_socket.MIN_AUDIO_RMS',2**-10)
    at_floor=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    at_floor.append(np.full(FIRST_AUDIO,2**-10,dtype=np.float32))
    assert at_floor.voice_end==FIRST_AUDIO and at_floor.candidate(monotonic()) is not None
    below=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    below.append(np.full(FIRST_AUDIO,2**-11,dtype=np.float32))
    assert below.voice_end==0 and below.skipped_silence_samples==FIRST_AUDIO
    zero=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    zero.append(np.zeros(FIRST_AUDIO,dtype=np.float32))
    assert zero.voice_end==0 and zero.skipped_silence_samples==FIRST_AUDIO


def test_quiet_pcm_reaches_both_stream_stages_instead_of_claiming_silence(socket_server,monkeypatch):
    _,_,transcriber,connect,_=socket_server
    calls=[]
    original=transcriber.transcribe
    def record(wave):
        calls.append(float(np.sqrt(np.mean(wave.astype(np.float64)**2))))
        return original(wave)
    monkeypatch.setattr(transcriber,'transcribe',record)
    ws=connect()
    quiet=(.0005*np.sin(2*np.pi*220*np.arange(FIRST_AUDIO)/RATE)*32767).astype('<i2')
    seq,total=send_audio(ws,quiet)
    done,events=stop(ws,seq,total)
    assert calls and all(.0001<rms<.001 for rms in calls)
    assert any(event['type']=='transcript' for event in events)
    assert any(event['type']=='feedback' for event in events)
    assert done['metrics']['analyses']>=1 and done['metrics']['model_samples']==total
    assert done['metrics']['skipped_silence_samples']==0


def test_fast_inference_grows_anchored_context_instead_of_cutting_every_word():
    session=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    windows=[]
    for second in range(1,16):
        session.append(np.ones(RATE,dtype=np.float32)*.1)
        job=session.candidate(monotonic())
        if job is not None:
            windows.append((job['start']//RATE,job['end']//RATE))
            complete_job(session,job)
    assert windows==[(0,second) for second in range(2,9)]+[(6,second) for second in range(9,15)]+[(12,15)]
    assert session.analyzed_samples==15*RATE


def test_rollover_can_move_left_to_keep_a_complete_mutable_word():
    class BoundaryTracker(FakeTracker):
        def analysis_start(self,proposed):
            return proposed-RATE//2 if proposed==6*RATE else proposed
    session=LiveSession(start_message(),FakeCorpus(),BoundaryTracker)
    session.append(np.ones(WINDOW,dtype=np.float32)*.1)
    complete_job(session,session.candidate(monotonic()))
    session.append(np.ones(RATE,dtype=np.float32)*.1)
    job=session.candidate(monotonic())
    assert job['start']==int(5.5*RATE) and job['end']==9*RATE
    assert len(job['wave'])<=WINDOW
    complete_job(session,job)
    assert session.analyzed_samples==9*RATE


def test_an_overlong_word_context_request_cannot_stall_new_audio_coverage():
    class OverlongTracker(FakeTracker):
        def analysis_start(self,proposed): return 0
    session=LiveSession(start_message(),FakeCorpus(),OverlongTracker)
    session.append(np.ones(WINDOW,dtype=np.float32)*.1)
    complete_job(session,session.candidate(monotonic()))
    session.append(np.ones(RATE,dtype=np.float32)*.1)
    job=session.candidate(monotonic())
    assert job['start']==6*RATE and job['end']==9*RATE
    assert len(job['wave'])<=WINDOW


def test_final_recheck_keeps_complete_anchored_context_without_new_audio():
    session=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    session.append(np.ones(6*RATE,dtype=np.float32)*.1)
    complete_job(session,session.candidate(monotonic()))
    session.stopping=True
    final=session.candidate(monotonic())
    assert final['start']==0 and final['end']==6*RATE and final['settled'] and final['final']


@pytest.mark.parametrize('settled', [False, True])
def test_session_merge_timing_counts_preview_and_confirmation(monkeypatch, settled):
    session = LiveSession(start_message(), FakeCorpus())
    session.append(np.ones(FIRST_AUDIO, dtype=np.float32))
    session.stopping = settled
    job = session.candidate(monotonic())
    heard = dict(words=[dict(text=' لفظ', start_ms=0, end_ms=500)])
    times = iter([10, 10.003, 10.003, 10.01])
    monkeypatch.setattr('live_socket.perf_counter', lambda: next(times))
    preview = session.apply_stage(dict(stage='preview', job=job, result=heard, timings={}))
    confirm = session.apply_stage(dict(stage='confirm', job=job, result=heard, timings={}, revised=False))
    assert preview['timings']['merge_ms'] == 3
    # An unchanged provisional prefix can consume merge time without emitting an event.
    if settled:
        assert confirm['timings']['merge_ms'] == 7
    else:
        assert confirm is None
    session.job = job
    session.finish_job({})
    assert session.metrics()['stage_timings'][0]['merge_ms'] == 10


def test_initial_silence_does_not_bypass_two_second_first_analysis_requirement():
    session=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    session.append(np.zeros(RATE//2,dtype=np.float32))
    session.append(np.ones(RATE,dtype=np.float32)*.1)
    assert session.analyzed_samples==RATE//2
    assert session.candidate(monotonic()) is None
    session.append(np.ones(RATE//2,dtype=np.float32)*.1)
    first=session.candidate(monotonic())
    assert first['start']==0 and first['end']==FIRST_AUDIO


def test_thirty_minute_continuous_fast_windows_remain_bounded_and_cover_all_audio():
    session=LiveSession(start_message(),FakeCorpus(),FakeTracker)
    for _ in range(MAX_SECONDS):
        session.append(np.ones(RATE,dtype=np.float32)*.1)
        job=session.candidate(monotonic())
        if job is not None:
            assert len(job['wave'])<=WINDOW and job['start']<=session.analyzed_samples
            complete_job(session,job)
    session.stopping=True
    final=session.candidate(monotonic())
    assert final['final']
    complete_job(session,final)
    assert session.analyzed_samples==MAX_SECONDS*RATE
    assert session.model_samples==MAX_SECONDS*RATE and session.skipped_silence_samples==0
    assert session.committed_samples==MAX_SECONDS*RATE
    assert session.max_retained_samples==CAPACITY
    assert session.coverage_intervals==[[0,MAX_SECONDS*RATE,'model']]
