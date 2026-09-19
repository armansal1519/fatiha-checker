"""Bounded, in-memory decoding of browser recordings and uploaded audio."""
from io import BytesIO

import av
import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 30
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "mp4", "webm", "ogg"}


class AudioError(ValueError):
    pass


def decode_audio(data: bytes, filename: str) -> np.ndarray:
    if not data:
        raise AudioError("فایل صوتی خالی است. دوباره ضبط کنید.")
    if len(data) > MAX_BYTES:
        raise AudioError("حجم فایل باید حداکثر ۱۰ مگابایت باشد.")
    if filename.rsplit(".", 1)[-1].lower() not in ALLOWED_EXTENSIONS:
        raise AudioError("فایل WAV، MP3، M4A، WebM یا OGG انتخاب کنید.")
    chunks, sample_count = [], 0
    try:
        # Never resolve external media references or open a user-supplied path.
        with av.open(BytesIO(data), options={"protocol_whitelist": ""}) as container:
            if not container.streams.audio:
                raise AudioError("این فایل صدای قابل‌خواندن ندارد.")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().reshape(-1)
                    sample_count += samples.size
                    if sample_count > MAX_SECONDS * SAMPLE_RATE:
                        raise AudioError("هر ضبط باید حداکثر ۳۰ ثانیه و مربوط به یک آیه باشد.")
                    chunks.append(samples)
            for converted in resampler.resample(None):
                samples = converted.to_ndarray().reshape(-1)
                sample_count += samples.size
                if sample_count > MAX_SECONDS * SAMPLE_RATE:
                    raise AudioError("هر ضبط باید حداکثر ۳۰ ثانیه باشد.")
                chunks.append(samples)
    except AudioError:
        raise
    except (av.FFmpegError, ValueError, OSError) as exc:
        raise AudioError("فایل صوتی قابل‌خواندن نیست. فایل دیگری انتخاب کنید یا دوباره ضبط کنید.") from exc
    if not chunks or sample_count < SAMPLE_RATE // 2:
        raise AudioError("صدا خیلی کوتاه است. یک آیه را کامل بخوانید.")
    wave = np.concatenate(chunks).astype(np.float32)
    if not np.isfinite(wave).all():
        raise AudioError("دادهٔ صوتی معتبر نیست. دوباره ضبط کنید.")
    # Reject near-silence, not quiet speech. This is not a speech/noise classifier.
    if float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))) < 0.0001:
        raise AudioError("صدای قابل‌شنیدن دریافت نشد. میکروفن را بررسی و دوباره ضبط کنید.")
    return wave
