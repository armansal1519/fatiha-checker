"""Local speech-activity evidence, independent of recitation and ASR text.

The pinned Silero JIT model runs on CPU inside the shared inference worker.
Each call has its own reset state and retains no recording, frame scores or
cross-window audio. Padding changes the analysis context, never the PCM itself.
"""
from __future__ import annotations

import logging
import math
from threading import Lock

import numpy as np

from audio_processing import MIN_AUDIO_RMS

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
SPEECH_THRESHOLD = .5
# Silero emits float32 scores; compare at the same representable threshold.
CONTINUE_THRESHOLD = float(np.float32(.35))
MIN_SPEECH_SAMPLES = round(.096*SAMPLE_RATE)
MIN_SILENCE_SAMPLES = round(.240*SAMPLE_RATE)
PRE_PAD_SAMPLES = round(.160*SAMPLE_RATE)
POST_PAD_SAMPLES = round(.240*SAMPLE_RATE)
log = logging.getLogger(__name__)


def _result(status, reason, rms, peak, *, spans=None, frames=0):
    spans = spans or []
    return dict(status=status, reason=reason, spans=spans, rms=rms, peak=peak,
                speech_samples=sum(span['end_sample']-span['start_sample'] for span in spans),
                evaluated_frames=frames)


def _speech_spans(probabilities, sample_count):
    """Consecutive onset evidence, hysteresis, then clipped context padding."""
    pending_start = active_start = last_speech_end = silence_start = None
    cores = []
    for index, probability in enumerate(probabilities):
        start, end = index*FRAME_SAMPLES, min((index+1)*FRAME_SAMPLES, sample_count)
        if end <= start:
            break
        if active_start is None:
            if probability >= SPEECH_THRESHOLD or (pending_start is not None and probability >= CONTINUE_THRESHOLD):
                if pending_start is None:
                    pending_start = start
                if end-pending_start >= MIN_SPEECH_SAMPLES:
                    active_start, last_speech_end = pending_start, end
                    silence_start = None
            else:
                pending_start = None
        elif probability >= CONTINUE_THRESHOLD:
            last_speech_end = end
            silence_start = None
        else:
            if silence_start is None:
                silence_start = start
            if end-silence_start >= MIN_SILENCE_SAMPLES:
                cores.append((active_start,last_speech_end))
                pending_start = active_start = last_speech_end = silence_start = None
    if active_start is not None:
        cores.append((active_start,last_speech_end))
    padded = []
    for start,end in cores:
        start, end = max(0,start-PRE_PAD_SAMPLES), min(sample_count,end+POST_PAD_SAMPLES)
        if padded and start <= padded[-1]['end_sample']:
            padded[-1]['end_sample'] = max(padded[-1]['end_sample'],end)
        else:
            padded.append(dict(start_sample=start,end_sample=end))
    return padded


class SpeechActivityDetector:
    def __init__(self):
        self.ready = False
        self.error = None
        self.model = None
        self._lock = Lock()

    def load(self):
        with self._lock:
            if self.ready:
                return
            try:
                import torch
                from model_assets import ensure_activity_model
                path = ensure_activity_model(offline=True)
                model = torch.jit.load(str(path),map_location='cpu').eval()
                model.reset_states()
                self.model = model
                self.ready = True
                self.error = None
                log.info('Local speech-activity detector ready on CPU.')
            except Exception as exc:
                self.model = None
                self.ready = False
                self.error = str(exc)
                log.exception('Speech-activity detector could not be loaded from the local pinned checkpoint.')

    def detect(self, wave):
        wave = np.asarray(wave,dtype=np.float32)
        if wave.ndim != 1 or not np.isfinite(wave).all():
            return _result('uncertain','invalid_audio',None,None)
        count = len(wave)
        rms = float(np.sqrt(np.mean(wave.astype(np.float64)**2))) if count else 0.0
        peak = float(np.max(np.abs(wave))) if count else 0.0
        with self._lock:
            if not self.ready or self.model is None:
                return _result('uncertain','detector_not_ready',rms,peak)
            if not count or rms < MIN_AUDIO_RMS:
                return _result('no_speech','below_audio_floor',rms,peak)
            model = self.model
            probabilities = []
            failure = None
            try:
                import torch
                model.reset_states()
                with torch.inference_mode():
                    for start in range(0,count,FRAME_SAMPLES):
                        frame = torch.zeros(FRAME_SAMPLES,dtype=torch.float32,device='cpu')
                        available = min(FRAME_SAMPLES,count-start)
                        frame[:available].copy_(torch.from_numpy(wave[start:start+available]))
                        output = model(frame,SAMPLE_RATE)
                        probability = float(output.item())
                        if not math.isfinite(probability) or not 0 <= probability <= 1:
                            raise RuntimeError('Speech detector returned an invalid frame probability')
                        probabilities.append(probability)
            except Exception as exc:
                failure = str(exc)
            finally:
                try:
                    model.reset_states()
                except Exception as exc:
                    failure = str(exc)
            if failure is not None:
                # Failed VAD never licenses ASR hallucination or a green verdict.
                self.ready = False
                self.model = None
                self.error = failure
                log.error('Speech-activity detector failed: %s',failure)
                return _result('uncertain','detector_failed',rms,peak,frames=len(probabilities))
            spans = _speech_spans(probabilities,count)
            if spans:
                return _result('speech','speech_detected',rms,peak,spans=spans,frames=len(probabilities))
            maximum = max(probabilities,default=0.0)
            if maximum >= CONTINUE_THRESHOLD:
                reason = 'speech_too_short' if maximum >= SPEECH_THRESHOLD else 'borderline_activity'
                return _result('uncertain',reason,rms,peak,frames=len(probabilities))
            return _result('no_speech','vad_no_speech',rms,peak,frames=len(probabilities))
