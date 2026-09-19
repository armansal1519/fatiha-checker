"""Bounded, local WebSocket audio transport and server-owned live results.

One receiving thread keeps draining PCM while one shared executor runs the model.
No audio jobs are queued: a completed job is followed by the newest ring snapshot.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import logging
import struct
from threading import Lock
from time import monotonic, perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import av
import numpy as np
from flask import jsonify, request
from flask_sock import Sock
from simple_websocket import ConnectionClosed

RATE = 16000
CAPACITY = 10 * RATE
MAX_MESSAGE = 64 * 1024
MAX_SECONDS = 180
RATES = {16000, 44100, 48000}
log = logging.getLogger(__name__)


class ProtocolError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class AudioRing:
    def __init__(self, capacity=CAPACITY):
        self.data = np.empty(capacity, dtype=np.float32)
        self.capacity = capacity
        self.total = 0

    def append(self, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        original_size = len(samples)
        if original_size >= self.capacity:
            samples = samples[-self.capacity:]
        start = (self.total + original_size - len(samples)) % self.capacity
        first = min(len(samples), self.capacity - start)
        self.data[start:start + first] = samples[:first]
        self.data[:len(samples) - first] = samples[first:]
        self.total += original_size

    def snapshot(self, end=None, seconds=10):
        end = min(self.total, self.total if end is None else end)
        start = max(0, self.total - self.capacity, end - int(seconds * RATE))
        if end <= start:
            return np.empty(0, dtype=np.float32)
        indices = np.arange(start, end) % self.capacity
        return self.data[indices].copy()


class PcmInput:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.sequence = -1
        self.input_samples = 0
        self.resampler = av.AudioResampler(format="fltp", layout="mono", rate=RATE)
        self.flushed = False

    def convert(self, packet):
        if len(packet) > MAX_MESSAGE:
            raise ProtocolError("message_size", "قطعهٔ صوتی بیش از حد بزرگ است. دوباره شروع کنید.")
        if len(packet) < 10 or len(packet) % 2:
            raise ProtocolError("audio_format", "قالب قطعهٔ صوتی معتبر نیست. دوباره شروع کنید.")
        sequence, offset = struct.unpack_from("<II", packet)
        if sequence != self.sequence + 1 or offset != self.input_samples:
            raise ProtocolError("sequence", "پیوستگی صدا از بین رفت. لطفاً دوباره شروع کنید.")
        pcm = np.frombuffer(packet, dtype="<i2", offset=8).copy()
        if self.input_samples + len(pcm) > self.sample_rate * MAX_SECONDS:
            raise ProtocolError("duration", "زمان هر نوبت حداکثر سه دقیقه است.")
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self.input_samples
        frame.time_base = Fraction(1, self.sample_rate)
        self.sequence = sequence
        self.input_samples += len(pcm)
        return self._samples(self.resampler.resample(frame))

    @staticmethod
    def _samples(frames):
        chunks = [frame.to_ndarray().reshape(-1) for frame in frames]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.empty(0, dtype=np.float32)

    def flush(self):
        if self.flushed:
            return np.empty(0, dtype=np.float32)
        self.flushed = True
        return self._samples(self.resampler.resample(None))


class StableWords:
    def __init__(self, total_words, start_index):
        self.words = {}
        self.pending = {}
        self.total_words = total_words
        self.start_index = start_index

    def observe(self, updates, analyzed_samples, settled):
        for update in updates:
            index = update["index"]
            if not self.start_index <= index < self.total_words:
                continue
            signature = (update["state"], tuple((e["type"], e["message"]) for e in update["errors"]))
            old_signature, count, endpoint = self.pending.get(index, (None, 0, -1))
            count = count + 1 if signature == old_signature and analyzed_samples > endpoint else 1
            self.pending[index] = (signature, count, analyzed_samples)
            if settled or count >= 2:
                self.words[index] = update

    def summary(self):
        return {"reviewed": len(self.words),
                "differences": sum(w["state"] == "difference" for w in self.words.values()),
                "unreviewed": self.total_words - self.start_index - len(self.words)}


class LiveSession:
    def __init__(self, start, follower):
        if not isinstance(start, dict) or start.get("type") != "start" or start.get("version") != 1:
            raise ProtocolError("start", "اطلاعات شروع قرائت معتبر نیست. صفحه را تازه کنید.")
        rate, ayah = start.get("sample_rate"), start.get("start_ayah")
        if type(rate) is not int or rate not in RATES or start.get("channels") != 1:
            raise ProtocolError("sample_rate", "قالب صدای مرورگر پشتیبانی نمی‌شود.")
        if type(ayah) is not int or ayah not in follower.ayah_starts:
            raise ProtocolError("ayah", "آیهٔ شروع معتبر نیست.")
        self.id = uuid4().hex
        self.input = PcmInput(rate)
        self.ring = AudioRing()
        self.start_ayah = ayah
        self.cursor = follower.ayah_starts[ayah]
        self.current_word = None
        self.words = StableWords(len(follower.words), self.cursor)
        self.started = monotonic()
        self.received_at = self.started
        self.last_activity = self.started
        self.voice_end = 0
        self.last_scheduled_voice = 0
        self.last_analyzed = 0
        self.last_settled_voice = 0
        self.last_completed = 0
        self.future = None
        self.job = None
        self.revision = 0
        self.stopping = False
        self.final_scheduled = False
        self.last_status = "listening"
        self.last_message = "چند کلمهٔ پیوسته بخوانید."
        self.timings = {}
        self.first_progress_ms = None
        self.first_color_ms = None
        self.inference_times = []
        self.last_busy = 0

    def append(self, samples):
        self.ring.append(samples)
        if len(samples) and float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) > .001:
            self.voice_end = self.ring.total

    def accept(self, message):
        self.last_activity = self.received_at = monotonic()
        if isinstance(message, bytes):
            if self.stopping:
                raise ProtocolError("stopped", "صدا پس از پایان نوبت دریافت شد. دوباره شروع کنید.")
            self.append(self.input.convert(message))
            return {"type": "ack", "sequence": self.input.sequence, "input_samples": self.input.input_samples}
        command = parse_json(message)
        if command.get("type") != "stop" or self.stopping:
            raise ProtocolError("command", "پیام قرائت شناخته نشد. دوباره شروع کنید.")
        if command.get("last_sequence") != self.input.sequence or command.get("total_samples") != self.input.input_samples:
            raise ProtocolError("sequence", "همهٔ بخش‌های صدا دریافت نشد. دوباره شروع کنید.")
        self.append(self.input.flush())
        self.stopping = True
        return None

    def candidate(self, now):
        if self.future is not None or now - self.last_completed < .1:
            return None
        if self.stopping and self.final_scheduled:
            return None
        settled = self.stopping or self.ring.total - self.voice_end >= int(.5 * RATE)
        if not self.voice_end or self.ring.total < int(.8 * RATE):
            return None
        end = min(self.ring.total, self.voice_end + int(.25 * RATE)) if settled else self.ring.total
        new_audio = end - self.last_analyzed >= int(.5 * RATE)
        new_voice = self.voice_end > self.last_scheduled_voice
        new_pause = settled and self.voice_end > self.last_settled_voice
        if not self.stopping and not ((new_audio and new_voice) or new_pause):
            return None
        seconds = 10 if settled or self.last_status == "listening" else 6
        wave = self.ring.snapshot(end=end, seconds=seconds)
        if len(wave) < int(.5 * RATE):
            return None
        return dict(wave=wave, end=end, voice=self.voice_end, settled=settled, final=self.stopping,
                    receive_to_dispatch_ms=round((now-self.received_at)*1000, 1))

    def finish_result(self, result):
        job = self.job
        self.future = self.job = None
        self.last_completed = monotonic()
        self.last_analyzed = job["end"]
        if job["settled"]:
            self.last_settled_voice = job["voice"]
        self.revision += 1
        self.cursor = max(self.cursor, result["cursor"])
        self.current_word = result.get("current_word")
        self.last_status = result.get("status", "listening")
        self.last_message = result.get("message", self.last_message)
        self.words.observe(result.get("updates", []), job["end"], job["settled"])
        elapsed = round((self.last_completed-self.started)*1000, 1)
        if self.last_status == "tracking" and self.first_progress_ms is None:
            self.first_progress_ms = elapsed
        if self.words.words and self.first_color_ms is None:
            self.first_color_ms = elapsed
        self.timings = dict(result.get("timings", {}), server_receive_to_dispatch_ms=job["receive_to_dispatch_ms"])
        self.inference_times.append(self.timings.get("inference_ms", 0))

    def result_fields(self):
        return dict(revision=self.revision, cursor=self.cursor, current_word=self.current_word,
                    analyzed_samples=self.last_analyzed,
                    updates=[self.words.words[k] for k in sorted(self.words.words)],
                    summary=self.words.summary(), timings=self.timings)

    def metrics(self):
        return dict(first_progress_ms=self.first_progress_ms, first_color_ms=self.first_color_ms,
                    first_position_ms=self.first_progress_ms,
                    analyses=len(self.inference_times), inference_ms=self.inference_times,
                    input_samples=self.input.input_samples, output_samples=self.ring.total,
                    total_received_samples=self.input.input_samples,
                    ring_max_samples=min(self.ring.total, self.ring.capacity),
                    inference_max_inflight=1 if self.inference_times else 0, queued_inference_jobs=0,
                    buffer_capacity_samples=self.ring.capacity,
                    elapsed_ms=round((monotonic()-self.started)*1000, 1))


def parse_json(message):
    if not isinstance(message, str) or len(message.encode("utf-8")) > MAX_MESSAGE:
        raise ProtocolError("message_size", "پیام قرائت معتبر نیست.")
    try:
        value = json.loads(message)
    except (ValueError, TypeError) as exc:
        raise ProtocolError("json", "پیام قرائت قابل‌خواندن نیست.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("json", "پیام قرائت معتبر نیست.")
    return value


class LiveSocketManager:
    def __init__(self, service, follower, inference_lock):
        self.service, self.follower, self.inference_lock = service, follower, inference_lock
        self.session_lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-inference")

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def infer(self, wave, cursor, start_ayah, settled):
        started = perf_counter()
        try:
            result = self.follower.analyze(self.service, wave, cursor, start_ayah, settled)
            result.setdefault("timings", {})["worker_ms"] = round((perf_counter()-started)*1000, 1)
            return result
        finally:
            self.inference_lock.release()

    def serve(self, ws):
        session = None
        owns_session = False

        def send(event):
            if session is not None:
                event["session_id"] = session.id
            ws.send(json.dumps(event, ensure_ascii=False, separators=(",", ":")))

        try:
            if not self.service.ready:
                raise ProtocolError("not_ready", "مدل هنوز آماده نیست. برنامه را دوباره راه‌اندازی کنید.")
            owns_session = self.session_lock.acquire(blocking=False)
            if not owns_session:
                raise ProtocolError("session_busy", "یک نوبت قرائت زنده در حال اجراست. ابتدا آن را پایان دهید.")
            first = ws.receive(timeout=10)
            if first is None:
                raise ProtocolError("timeout", "شروع قرائت دریافت نشد. دوباره تلاش کنید.")
            session = LiveSession(parse_json(first), self.follower)
            send(dict(type="ready", version=1))
            while True:
                message = ws.receive(timeout=.02)
                if message is not None:
                    ack = session.accept(message)
                    if ack:
                        send(ack)
                now = monotonic()
                if now - session.started > MAX_SECONDS + 30 or (not session.stopping and now-session.last_activity > 15):
                    raise ProtocolError("timeout", "ارتباط دریافت صدا متوقف شد. دوباره شروع کنید.")
                if session.future is not None and session.future.done():
                    result = session.future.result()
                    final = session.job["final"]
                    session.finish_result(result)
                    send(dict(type="progress", revision=session.revision, cursor=session.cursor,
                              current_word=session.current_word, analyzed_samples=session.last_analyzed,
                              phase=session.last_status, message=session.last_message))
                    send(dict(type="result", **session.result_fields()))
                    if final:
                        send(dict(type="done", **session.result_fields(), metrics=session.metrics()))
                        return
                job = session.candidate(now)
                if job is not None:
                    if self.inference_lock.acquire(blocking=False):
                        try:
                            session.future = self.executor.submit(self.infer, job["wave"], session.cursor,
                                                                  session.start_ayah, job["settled"])
                        except BaseException:
                            self.inference_lock.release()
                            raise
                        session.job = job
                        session.last_scheduled_voice = job["voice"]
                        session.final_scheduled = job["final"]
                    elif now - session.last_busy > 2:
                        send(dict(type="error", code="model_busy", message="مدل در حال بررسی صدای قبلی است؛ لحظه‌ای صبر کنید.", fatal=False))
                        session.last_busy = now
                elif session.stopping and session.future is None and now - session.last_completed >= .1:
                    # Silence or too little usable audio: finish without invented results.
                    send(dict(type="done", **session.result_fields(), metrics=session.metrics()))
                    return
        except ProtocolError as exc:
            try:
                send(dict(type="error", code=exc.code, message=str(exc), fatal=True))
            except ConnectionClosed:
                pass
        except ConnectionClosed:
            pass
        except Exception:
            log.exception("Live WebSocket analysis failed")
            try:
                send(dict(type="error", code="processing", message="بررسی این نوبت انجام نشد. لطفاً دوباره شروع کنید.", fatal=True))
            except ConnectionClosed:
                pass
        finally:
            # Running native inference cannot safely be interrupted. It owns its
            # snapshot and releases the shared lock, but can no longer publish.
            if session is not None:
                session.ring.data = np.empty(0, dtype=np.float32)
            if owns_session:
                self.session_lock.release()
            try:
                ws.close()
            except ConnectionClosed:
                pass


def register_live_socket(app, service, follower, inference_lock):
    manager = LiveSocketManager(service, follower, inference_lock)
    app.extensions["live_socket_manager"] = manager
    app.config["SOCK_SERVER_OPTIONS"] = {"max_message_size": MAX_MESSAGE, "ping_interval": 10}

    @app.before_request
    def protect_socket_origin():
        if request.path != "/live/ws":
            return None
        try:
            host = urlsplit("http://" + request.host)
            origin = urlsplit(request.headers.get("Origin", ""))
            valid = (host.hostname in {"127.0.0.1", "localhost", "::1"}
                     and origin.scheme == request.scheme
                     and origin.netloc.lower() == request.host.lower()
                     and origin.path in {"", "/"} and not origin.query and not origin.fragment)
        except ValueError:
            valid = False
        if not valid:
            return jsonify(status="error", message="قرائت زنده باید از صفحهٔ همین برنامه اجرا شود."), 403

    sock = Sock(app)

    @sock.route("/live/ws")
    def live_socket(ws):
        manager.serve(ws)

    return manager
