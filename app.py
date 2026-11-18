"""Local-only Flask entrypoint. Run with: python app.py [--port 5001]"""
import argparse
import logging
from io import BytesIO
from threading import Lock
from time import perf_counter

from flask import Flask, Request, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from audio_processing import AudioError, MAX_BYTES, decode_audio


class MemoryOnlyRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        # Werkzeug otherwise spools uploads larger than 500 KB to disk.
        return BytesIO()


def create_app(service=None, transcriber=None, corpus=None):
    app = Flask(__name__)
    app.request_class = MemoryOnlyRequest
    app.config.update(MAX_CONTENT_LENGTH=MAX_BYTES + 128 * 1024, MAX_FORM_PARTS=5)
    app.json.ensure_ascii = False
    if service is None:
        from recitation import RecitationService
        service = RecitationService()
    if transcriber is None:
        from transcription import TranscriptionService
        transcriber = TranscriptionService()
    if corpus is None:
        from quran_corpus import QuranCorpus
        corpus = QuranCorpus()
    app.extensions["recitation_service"] = service
    app.extensions["transcription_service"] = transcriber
    app.extensions["quran_corpus"] = corpus
    inference_lock = Lock()
    app.extensions["inference_lock"] = inference_lock
    from live_tracking import LiveFollower
    follower = LiveFollower(getattr(service, "references", None))
    app.extensions["live_follower"] = follower
    from live_socket import register_live_socket
    register_live_socket(app, service, transcriber, corpus, inference_lock)

    def failure(message, code=400):
        return jsonify(status="error", message=message), code

    @app.after_request
    def local_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/")
    def index():
        return render_template("index.html", verses=service.verses, surahs=corpus.surahs, ready=service.ready)

    @app.get("/live")
    def live():
        return render_template("live.html", surahs=corpus.surahs, ready=service.ready and transcriber.ready)

    @app.get("/api/quran/surahs")
    def surahs():
        return jsonify(surahs=corpus.surahs)

    @app.get("/api/quran/ayahs")
    def ayahs():
        try:
            surah = int(request.args.get("surah", "1"))
            start = int(request.args.get("start", "1"))
            limit = int(request.args.get("limit", "10"))
            if not 1 <= limit <= 50:
                raise ValueError("هر بار بین ۱ تا ۵۰ آیه درخواست کنید.")
            verses = corpus.verses(surah, start, limit)
            return jsonify(surah=surah, ayahs=verses, total=corpus.surahs[surah-1]["ayah_count"])
        except (ValueError, TypeError):
            return failure("شمارهٔ سوره، آیه یا تعداد آیات معتبر نیست.")

    @app.post("/live/analyze")
    def live_analyze():
        from live_tracking import MAX_LIVE_SECONDS
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            return failure("درخواست باید از صفحهٔ همین برنامه ارسال شود.", 403)
        if not service.ready:
            return failure("مدل آماده نیست. برنامه را دوباره راه‌اندازی کنید.", 503)
        if not inference_lock.acquire(blocking=False):
            return failure("مدل در حال بررسی صدای قبلی است؛ لحظه‌ای صبر کنید.", 409)
        try:
            try:
                cursor = int(request.form.get("cursor", "0"))
                start_ayah = int(request.form.get("start_ayah", "1"))
                sequence = int(request.form.get("sequence", "0"))
            except (ValueError, TypeError):
                return failure("اطلاعات دنبال‌کردن قرائت معتبر نیست.")
            if not 0 <= cursor <= len(follower.words) or start_ayah not in follower.ayah_starts or not 0 <= sequence <= 100000:
                return failure("موقعیت خواندن معتبر نیست.")
            audio = request.files.get("audio")
            if audio is None or not audio.filename:
                return failure("صدایی دریافت نشد.")
            wave = decode_audio(audio.read(MAX_BYTES + 1), audio.filename)
            if len(wave) > MAX_LIVE_SECONDS * 16000:
                return failure("هر قطعهٔ بررسی زنده باید حداکثر ۱۲ ثانیه باشد.")
            started = perf_counter()
            result = follower.analyze(service, wave, cursor, start_ayah, request.form.get("settled") == "true")
            result.update(sequence=sequence, elapsed_seconds=round(perf_counter() - started, 2))
            return jsonify(result)
        except AudioError as exc:
            return failure(str(exc))
        except RequestEntityTooLarge:
            return too_large(None)
        except Exception:
            app.logger.exception("Live tracking failed")
            return failure("این بخش از صدا بررسی نشد. دوباره تلاش کنید.", 500)
        finally:
            inference_lock.release()

    @app.get("/health")
    def health():
        from model_assets import MODEL_ID, MODEL_REVISION, TRANSCRIPTION_MODEL_ID, TRANSCRIPTION_MODEL_REVISION
        activity=getattr(transcriber,'activity_detector',None)
        ready = service.ready and transcriber.ready and (activity is None or activity.ready)
        models=dict(recitation=dict(ready=service.ready,device='cpu',model=MODEL_ID,revision=MODEL_REVISION),
                    transcription=dict(ready=transcriber.ready,device=getattr(transcriber,'device',None),
                                       model=TRANSCRIPTION_MODEL_ID,revision=TRANSCRIPTION_MODEL_REVISION))
        if activity is not None:
            from model_assets import ACTIVITY_MODEL_ID, ACTIVITY_MODEL_REVISION
            models['speech_activity']=dict(ready=activity.ready,device='cpu',model=ACTIVITY_MODEL_ID,
                                           revision=ACTIVITY_MODEL_REVISION)
        return jsonify(ready=ready, model="obadx/muaalem-model-v3_2", device="cpu", protocol_version=2,
                       quran=dict(surahs=114, ayahs=6236, rewaya="hafs"),
                       models=models), (200 if ready else 503)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return failure("حجم فایل باید حداکثر ۱۰ مگابایت باشد.", 413)

    @app.post("/analyze")
    def analyze():
        # Do not permit another website to use this local model via a form POST.
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            return failure("درخواست باید از صفحهٔ همین برنامه ارسال شود.", 403)
        if not service.ready:
            return failure("مدل آماده نیست. پیام راه‌اندازی برنامه را بررسی و برنامه را دوباره اجرا کنید.", 503)
        if not inference_lock.acquire(blocking=False):
            return failure("در حال بررسی صدای قبلی هستیم. کمی صبر کنید.", 409)
        try:
            try:
                surah = int(request.form.get("surah", "1"))
                ayah = int(request.form.get("ayah", ""))
            except (ValueError, TypeError):
                return failure("شمارهٔ آیه معتبر نیست.")
            try:
                corpus.reference(surah, ayah)
            except ValueError:
                return failure("شمارهٔ سوره یا آیه معتبر نیست.")
            audio = request.files.get("audio")
            if audio is None or not audio.filename:
                return failure("ابتدا صدایتان را ضبط یا یک فایل صوتی انتخاب کنید.")
            wave = decode_audio(audio.read(MAX_BYTES + 1), audio.filename)
            start = perf_counter()
            result = service.analyze(wave, ayah) if surah == 1 else service.analyze(wave, ayah, surah=surah)
            result["elapsed_seconds"] = round(perf_counter() - start, 2)
            return jsonify(result)
        except AudioError as exc:
            return failure(str(exc))
        except RequestEntityTooLarge:
            return too_large(None)
        except Exception:
            app.logger.exception("Recitation analysis failed")
            return failure("بررسی صدا انجام نشد. دوباره تلاش کنید؛ جزئیات خطا در پنجرهٔ اجرای برنامه ثبت شده است.", 500)
        finally:
            inference_lock.release()

    return app


def warm_runtime(application):
    """Compile/cache the local inference paths before accepting user audio."""
    service=application.extensions['recitation_service']
    transcriber=application.extensions['transcription_service']
    if not service.ready or not transcriber.ready:
        return
    import numpy as np
    wave=(.01*np.sin(2*np.pi*220*np.arange(32000)/16000)).astype(np.float32)
    logging.info('Warming local speech models; no microphone or user recording is used.')
    try:
        getattr(transcriber,'warm',transcriber.transcribe)(wave)
        service.predict_live_timed(wave)
        corpus=application.extensions.get('quran_corpus')
        if hasattr(corpus, 'prepare_live'):
            corpus.prepare_live()
    except Exception as exc:
        transcriber.ready=False
        transcriber.error=str(exc)
        logging.exception('Speech warm-up failed; live transcription remains unavailable.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local Hamkhan app.")
    parser.add_argument("--port", type=int, default=5000, help="Local server port (default: 5000).")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    application = create_app()
    service = application.extensions["recitation_service"]
    service.load()
    application.extensions["transcription_service"].load()
    warm_runtime(application)
    if service.ready and application.extensions["transcription_service"].ready:
        logging.info("Open http://127.0.0.1:%s/live", args.port)
    application.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)
