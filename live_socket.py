"""Version 2 local PCM transport and bounded two-stage live inference.

The receiver owns session state. A single worker publishes raw transcription
before checking the same audio with Muaalem. Every new audio interval is either
processed in order or explicitly classified as silence; recent audio never
silently replaces an older, unprocessed interval.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import logging
from queue import Empty, Full, Queue
import struct
from threading import Event, Lock
from time import monotonic, perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import av
import numpy as np
from flask import jsonify, request
from flask_sock import Sock
from simple_websocket import ConnectionClosed

from audio_processing import MIN_AUDIO_RMS

RATE = 16000
CAPACITY = 30 * RATE
WINDOW = 8 * RATE
OVERLAP = 2 * RATE
FIRST_AUDIO = 2 * RATE
FRESH_AUDIO = RATE
MAX_MESSAGE = 64 * 1024
MAX_SECONDS = 30 * 60
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

    @property
    def oldest(self):
        return max(0, self.total - self.capacity)

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

    def slice(self, start, end):
        if start < self.oldest or end > self.total or end < start:
            raise ProtocolError("backlog", "بررسی از صدا عقب افتاد؛ این نوبت کامل نیست. دوباره شروع کنید.")
        return self.data[np.arange(start, end) % self.capacity].copy()

    def snapshot(self, end=None, seconds=30):
        end = min(self.total, self.total if end is None else end)
        start = max(self.oldest, end - int(seconds * RATE))
        if end <= start:
            return np.empty(0, dtype=np.float32)
        return self.slice(start, end)


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
            raise ProtocolError("duration", "زمان هر نوبت حداکثر ۳۰ دقیقه است.")
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


def _make_tracker(corpus, start_surah, start_ayah):
    from transcript_tracking import TranscriptTracker
    return TranscriptTracker(corpus, start_surah=start_surah, start_ayah=start_ayah)


def validate_reference(corpus, surah, ayah):
    if type(surah) is not int or type(ayah) is not int:
        raise ProtocolError("ayah", "سوره و آیهٔ شروع معتبر نیست.")
    try:
        corpus.reference(surah, ayah)
    except (ValueError, KeyError, IndexError, AssertionError):
        raise ProtocolError("ayah", "سوره و آیهٔ شروع معتبر نیست.") from None


class LiveSession:
    def __init__(self, start, corpus, tracker_factory=None):
        if not isinstance(start, dict) or start.get("type") != "start":
            raise ProtocolError("start", "اطلاعات شروع قرائت معتبر نیست. صفحه را تازه کنید.")
        if start.get("version") != 2:
            raise ProtocolError("version", "نسخهٔ ارتباط زنده تغییر کرده است؛ صفحه را تازه کنید.")
        rate = start.get("sample_rate")
        if type(rate) is not int or rate not in RATES or start.get("channels") != 1:
            raise ProtocolError("sample_rate", "قالب صدای مرورگر پشتیبانی نمی‌شود.")
        surah, ayah = start.get("start_surah"), start.get("start_ayah")
        if surah is not None or ayah is not None:
            validate_reference(corpus, surah, ayah)
        self.id = uuid4().hex
        self.corpus = corpus
        self.tracker = (tracker_factory or _make_tracker)(corpus, surah, ayah)
        self.input = PcmInput(rate)
        self.ring = AudioRing()
        self.stages = Queue(maxsize=4)
        self.closed = Event()
        self.started = self.last_activity = self.received_at = monotonic()
        self.voice_end = 0
        self.last_settled_voice = 0
        self.analyzed_samples = 0
        # Reuse a growing, anchored ASR context instead of cutting the left
        # edge one second forward after every fast inference.
        self.context_start = 0
        self.committed_samples = 0
        self.skipped_silence_samples = 0
        self.model_samples = 0
        self.coverage_intervals = []
        self.last_completed = 0
        self.future = self.job = None
        self.revision = 0
        self.stopping = self.final_scheduled = False
        self.last_status = "listening"
        self.last_message = "چند کلمهٔ پیوسته بخوانید."
        self.last_busy = 0
        self.first_transcript_ms = self.first_feedback_ms = self.first_position_ms = None
        self.first_text_location_ms = None
        self.inference_times = []
        self.max_retained_samples = 0
        self.peak_unanalyzed_samples = 0
        self.peak_waiting_stages = 0

    def _cover(self, end, *, silence=False):
        """Advance only across the immediately next interval, never a hole."""
        if end < self.analyzed_samples or end > self.ring.total:
            raise RuntimeError("Invalid analyzed audio cursor")
        count = end - self.analyzed_samples
        if count:
            kind = "silence" if silence else "model"
            if (self.coverage_intervals and self.coverage_intervals[-1][1] == self.analyzed_samples
                    and self.coverage_intervals[-1][2] == kind):
                self.coverage_intervals[-1][1] = end
            else:
                self.coverage_intervals.append([self.analyzed_samples, end, kind])
            self.analyzed_samples = end
            if silence:
                self.skipped_silence_samples += count
            else:
                self.model_samples += count

    def append(self, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(samples):
            return
        has_energy = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) >= MIN_AUDIO_RMS
        # A growing context may still need its complete eight-second history,
        # including a word that straddles the next nominal overlap boundary.
        # An in-flight worker owns its snapshot, but its next context must also
        # remain available. Stop before either context or new audio is erased.
        owned_end = self.job["end"] if self.future is not None else self.analyzed_samples
        protected_start = max(0, owned_end - WINDOW)
        after_oldest = max(0, self.ring.total + len(samples) - self.ring.capacity)
        if after_oldest > protected_start:
            raise ProtocolError("backlog", "بررسی از صدا عقب افتاد؛ این نوبت کامل نیست. دوباره شروع کنید.")
        can_skip = (not has_energy and self.future is None
                    and self.analyzed_samples == self.ring.total
                    and self.last_settled_voice >= self.voice_end)
        self.ring.append(samples)
        self.max_retained_samples = max(self.max_retained_samples, min(self.ring.total, self.ring.capacity))
        if has_energy:
            self.voice_end = self.ring.total
        elif can_skip:
            self._cover(self.ring.total, silence=True)
            self.committed_samples = self.analyzed_samples
        self.peak_unanalyzed_samples = max(self.peak_unanalyzed_samples,
                                          self.ring.total-self.analyzed_samples)

    def accept(self, message):
        self.last_activity = self.received_at = monotonic()
        if isinstance(message, bytes):
            if self.stopping:
                raise ProtocolError("stopped", "صدا پس از پایان نوبت دریافت شد. دوباره شروع کنید.")
            self.append(self.input.convert(message))
            return {"type": "ack", "sequence": self.input.sequence, "input_samples": self.input.input_samples}
        command = parse_json(message)
        if self.stopping:
            raise ProtocolError("command", "این نوبت پایان یافته است. دوباره شروع کنید.")
        if command.get("type") == "reference":
            surah, ayah = command.get("surah"), command.get("ayah")
            validate_reference(self.corpus, surah, ayah)
            return dict(type="feedback", **self.tracker.select_reference(surah, ayah))
        if command.get("type") != "stop":
            raise ProtocolError("command", "پیام قرائت شناخته نشد. دوباره شروع کنید.")
        if command.get("last_sequence") != self.input.sequence or command.get("total_samples") != self.input.input_samples:
            raise ProtocolError("sequence", "همهٔ بخش‌های صدا دریافت نشد. دوباره شروع کنید.")
        self.append(self.input.flush())
        self.stopping = True
        return None

    def _analysis_start(self, *, require_progress):
        """Grow an anchor to eight seconds, then roll by at most six seconds.

        The tracker may move the next left boundary back to a complete mutable
        word. A word longer than the whole analysis window cannot hold up the
        stream forever: retain transport continuity and let the tracker keep
        that incomplete occurrence ungraded instead.
        """
        covered = self.analyzed_samples
        proposed = self.context_start
        if covered - proposed >= WINDOW and require_progress:
            proposed = max(0, covered - OVERLAP)
        helper = getattr(self.tracker, "analysis_start", None)
        adjusted = helper(proposed) if callable(helper) else proposed
        if type(adjusted) is not int or not 0 <= adjusted <= proposed:
            raise RuntimeError("Tracker returned an invalid analysis context boundary")
        if require_progress and adjusted + WINDOW <= covered:
            adjusted = proposed
        if adjusted < self.ring.oldest:
            raise ProtocolError("backlog", "زمینهٔ لازم برای بررسی از بافر خارج شد؛ این نوبت کامل نیست. دوباره شروع کنید.")
        return adjusted

    def candidate(self, now):
        if self.future is not None or self.final_scheduled or now - self.last_completed < .05:
            return None
        total = self.ring.total
        if not total or not self.voice_end:
            return None
        paused = total - self.voice_end >= RATE // 2
        pause_review = paused and self.voice_end > self.last_settled_voice
        new = total - self.analyzed_samples
        if self.stopping and not new and self.last_settled_voice >= self.voice_end:
            return None
        if not self.stopping:
            if not self.inference_times and total < FIRST_AUDIO:
                return None
            if new < FRESH_AUDIO and not pause_review:
                return None
            if self.voice_end <= self.analyzed_samples and not pause_review:
                # Only received silence remains and the spoken edge was checked.
                self._cover(total, silence=True)
                self.committed_samples = self.analyzed_samples
                return None
        start = self._analysis_start(require_progress=new > 0)
        end = min(total, start + WINDOW)
        if end < self.analyzed_samples:
            raise RuntimeError("Analysis window cannot move backwards")
        settled = end >= self.voice_end and (self.stopping or paused)
        return dict(start=start, end=end, wave=self.ring.slice(start, end),
                    settled=settled, final=self.stopping and end == total,
                    voice=self.voice_end,
                    receive_to_dispatch_ms=round((now-self.received_at)*1000, 1))

    def apply_stage(self, stage):
        job, kind = stage["job"], stage["stage"]
        stage_start = perf_counter()
        kwargs = dict(start_sample=job["start"], end_sample=job["end"], settled=job["settled"])
        elapsed = round((monotonic()-self.started)*1000, 1)
        if kind != "feedback":
            if kind == "preview":
                result = self.tracker.preview_transcript(stage["result"], **kwargs)
                job["preview_deferred"] = result is None
            elif kind == "confirm" and not stage["revised"] and not job.get("preview_deferred"):
                result = self.tracker.commit_preview(job["end"],job["settled"])
            else:
                result = self.tracker.update_transcript(stage["result"], **kwargs)
            kind = "transcript"
            if result and result.get("tokens") and self.first_transcript_ms is None:
                self.first_transcript_ms = elapsed
        else:
            result = self.tracker.update_phonemes(stage["result"], **kwargs)
            self.last_status = result.get("status", "listening")
            self.last_message = result.get("message", self.last_message)
            if self.last_status == "tracking" and self.first_position_ms is None:
                self.first_position_ms = elapsed
            if any(u.get("state") in {"match", "difference"} for u in result.get("updates", [])) and self.first_feedback_ms is None:
                self.first_feedback_ms = elapsed
        timing_key = "merge_ms" if kind == "transcript" else "alignment_ms"
        stage_ms = round((perf_counter()-stage_start)*1000, 2)
        job[timing_key] = round(job.get(timing_key, 0) + stage_ms, 2)
        if result is None:
            return None
        if self.first_text_location_ms is None and result.get('text_location', {}).get('status') == 'probable':
            self.first_text_location_ms = round((monotonic()-self.started)*1000, 1)
        event = dict(type=kind if kind == "transcript" else "feedback", **result,
                     start_sample=job["start"], end_sample=job["end"],
                     timings={**stage["timings"], timing_key: stage_ms})
        if kind == "transcript" and isinstance(stage["result"].get("audio_activity"), dict):
            event["audio_activity"] = dict(stage["result"]["audio_activity"])
        return event

    def finish_job(self, timings):
        job = self.job
        if job["start"] > self.analyzed_samples:
            raise RuntimeError("An audio interval was skipped")
        self._cover(job["end"])
        self.context_start = job["start"]
        if job["settled"]:
            self.last_settled_voice = max(self.last_settled_voice, min(job["voice"], job["end"]))
        self.committed_samples = max(self.committed_samples, job["end"] if job["settled"] else max(0, job["end"]-OVERLAP))
        self.inference_times.append({**timings, **{k:job[k] for k in ("merge_ms","alignment_ms") if k in job}})
        self.last_completed = monotonic()
        self.future = self.job = None

    def metrics(self):
        return dict(first_transcript_ms=self.first_transcript_ms, first_feedback_ms=self.first_feedback_ms,
                    first_position_ms=self.first_position_ms, first_text_location_ms=self.first_text_location_ms,
                    analyses=len(self.inference_times),
                    inference_ms=[x.get("worker_ms", 0) for x in self.inference_times],
                    stage_timings=self.inference_times,
                    input_samples=self.input.input_samples, output_samples=self.ring.total,
                    total_received_samples=self.input.input_samples,
                    received_samples=self.ring.total, analyzed_samples=self.analyzed_samples,
                    committed_samples=self.committed_samples,
                    skipped_silence_samples=self.skipped_silence_samples, model_samples=self.model_samples,
                    coverage_complete=self.analyzed_samples == self.ring.total,
                    coverage_intervals=self.coverage_intervals,
                    ring_max_samples=self.max_retained_samples, buffer_capacity_samples=self.ring.capacity,
                    peak_unanalyzed_samples=self.peak_unanalyzed_samples, peak_waiting_stages=self.peak_waiting_stages,
                    inference_max_inflight=1 if self.inference_times else 0, queued_inference_jobs=0,
                    elapsed_ms=round((monotonic()-self.started)*1000, 1))

    def done_fields(self):
        return dict(summary=self.tracker.summary(), reference=getattr(self.tracker, "current_reference", None),
                    status=self.last_status, received_samples=self.ring.total,
                    analyzed_samples=self.analyzed_samples, committed_samples=self.committed_samples,
                    committed_count=getattr(self.tracker, "committed_count", 0), metrics=self.metrics(),
                    **(self.tracker.location_fields() if hasattr(self.tracker, "location_fields") else {}))


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
    def __init__(self, service, transcriber, corpus, inference_lock, tracker_factory=None):
        self.service, self.transcriber, self.corpus = service, transcriber, corpus
        self.inference_lock = inference_lock
        self.tracker_factory = tracker_factory
        self.session_lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-inference")

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def infer(self, job, stages, closed):
        began = perf_counter()
        timings = {}
        try:
            transcript = self.transcriber.transcribe(job["wave"])
            timings["transcription_ms"] = round((perf_counter()-began)*1000, 1)
            activity = transcript.get("audio_activity")
            status = activity.get("status") if isinstance(activity, dict) else None
            timings["transcription_skipped"] = status in {"no_speech", "uncertain"}
            frames = activity.get("evaluated_frames", 0) if isinstance(activity, dict) else 0
            timings["vad_frames"] = frames if type(frames) is int and frames >= 0 else None
            if status is not None:
                timings["activity_status"] = status
            if closed.is_set():
                return timings
            verify = getattr(self.transcriber,"recover_repetitions",None)
            preview = bool(verify and transcript.get("words") and not timings["transcription_skipped"])
            stages.put_nowait(dict(stage="preview" if preview else "transcript", result=transcript,
                                   job=job, timings=dict(timings)))
            if timings["transcription_skipped"]:
                # Activity detection runs inside this worker, never on the ACK
                # receiver. Ambient noise must not trigger phonetic verdicts.
                timings["phonetic_ms"] = 0.0
                timings["phonetic_skipped"] = True
                timings["worker_ms"] = round((perf_counter()-began)*1000, 1)
                if not closed.is_set():
                    stages.put_nowait(dict(stage="feedback", result=dict(audio_activity=dict(activity)),
                                           job=job, timings=dict(timings)))
                return timings
            phonemes_at = perf_counter()
            phonemes = self.service.predict_live_timed(job["wave"])
            timings["phonetic_ms"] = round((perf_counter()-phonemes_at)*1000, 1)
            timings["phonetic_skipped"] = False
            if preview and not closed.is_set():
                checked_at = perf_counter()
                confirmed = verify(transcript,phonemes,job["wave"])
                timings["repeat_check_ms"] = round((perf_counter()-checked_at)*1000,1)
                if not closed.is_set():
                    stages.put_nowait(dict(stage="confirm",result=confirmed,revised=confirmed is not transcript,
                                           job=job,timings=dict(timings)))
            timings["worker_ms"] = round((perf_counter()-began)*1000, 1)
            if not closed.is_set():
                stages.put_nowait(dict(stage="feedback", result=phonemes, job=job, timings=dict(timings)))
            return timings
        except Full:
            raise RuntimeError("Live result channel exceeded its fixed capacity") from None
        finally:
            self.inference_lock.release()

    def serve(self, ws):
        session = None
        owns_session = False

        def send(event):
            if session is not None:
                event["session_id"] = session.id
                if event.get("type") != "ack":
                    session.revision += 1
                    event["revision"] = session.revision
            ws.send(json.dumps(event, ensure_ascii=False, separators=(",", ":")))

        def drain_stages():
            session.peak_waiting_stages = max(session.peak_waiting_stages, session.stages.qsize())
            while True:
                try:
                    stage = session.stages.get_nowait()
                except Empty:
                    return
                event = session.apply_stage(stage)
                if event is not None:
                    send(event)

        try:
            if not self.service.ready or not self.transcriber.ready:
                raise ProtocolError("not_ready", "مدل‌های رونویسی و بررسی آماده نیستند. پیام راه‌اندازی برنامه را بررسی کنید.")
            owns_session = self.session_lock.acquire(blocking=False)
            if not owns_session:
                raise ProtocolError("session_busy", "یک نوبت قرائت زنده در حال اجراست. ابتدا آن را پایان دهید.")
            first = ws.receive(timeout=10)
            if first is None:
                raise ProtocolError("timeout", "شروع قرائت دریافت نشد. دوباره تلاش کنید.")
            session = LiveSession(parse_json(first), self.corpus, self.tracker_factory)
            send(dict(type="ready", version=2, sample_rate=RATE, max_seconds=MAX_SECONDS))
            while True:
                message = ws.receive(timeout=.02)
                if message is not None:
                    event = session.accept(message)
                    if event:
                        send(event)
                now = monotonic()
                if now-session.started > MAX_SECONDS+120 or (not session.stopping and now-session.last_activity > 15):
                    raise ProtocolError("timeout", "ارتباط دریافت صدا متوقف شد. دوباره شروع کنید.")
                drain_stages()
                if session.future is not None and session.future.done():
                    # A worker may have published its last stage just after the
                    # previous drain; consume it before marking coverage done.
                    timings = session.future.result()
                    drain_stages()
                    final = session.job["final"]
                    session.finish_job(timings)
                    if final:
                        send(dict(type="done", **session.done_fields()))
                        return
                job = session.candidate(now)
                if job is not None:
                    if self.inference_lock.acquire(blocking=False):
                        session.job = job
                        try:
                            session.future = self.executor.submit(self.infer, job, session.stages, session.closed)
                        except BaseException:
                            session.job = None
                            self.inference_lock.release()
                            raise
                        session.final_scheduled = job["final"]
                    elif now-session.last_busy > 2:
                        send(dict(type="error", code="model_busy", message="مدل در حال بررسی صدای قبلی است؛ لحظه‌ای صبر کنید.", fatal=False))
                        session.last_busy = now
                elif (session.stopping and session.future is None
                      and session.voice_end <= session.analyzed_samples
                      and session.last_settled_voice >= session.voice_end):
                    # Only an empty stream or a fully reviewed spoken edge can
                    # finish without another job. A scheduling cooldown must
                    # never classify an unprocessed backlog as silence.
                    if session.analyzed_samples < session.ring.total:
                        session._cover(session.ring.total, silence=True)
                        session.committed_samples = session.analyzed_samples
                    send(dict(type="done", **session.done_fields()))
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
            if session is not None:
                session.closed.set()
                session.ring.data = np.empty(0, dtype=np.float32)
            if owns_session:
                self.session_lock.release()
            try:
                ws.close()
            except ConnectionClosed:
                pass


def register_live_socket(app, service, transcriber, corpus, inference_lock, *, tracker_factory=None):
    manager = LiveSocketManager(service, transcriber, corpus, inference_lock, tracker_factory)
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
