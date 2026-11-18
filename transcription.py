"""Unprompted Arabic ASR. Raw heard text never depends on Quran matching."""
import logging
import math

import numpy as np

from model_assets import ensure_transcription_model
from audio_processing import MIN_AUDIO_RMS
from speech_activity import SpeechActivityDetector

SAMPLE_RATE = 16000
GENERATION_OPTIONS = {
    "language": "ar", "task": "transcribe", "condition_on_prev_tokens": False,
    "prompt_ids": None, "num_beams": 1, "do_sample": False, "temperature": 0.0,
    "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
}

def _crossed_pause(words, spans):
    """Find a substantial pause that Whisper put inside a single long word."""
    best_gap = 0
    cut = None
    for left, right in zip(spans, spans[1:]):
        before, after = left["end_sample"], right["start_sample"]
        gap = after - before
        if gap < round(.24*SAMPLE_RATE) or gap <= best_gap:
            continue
        for word in words:
            if word["start_ms"] is None or word["end_ms"] is None:
                continue
            start = round(word["start_ms"] * SAMPLE_RATE/1000)
            end = round(word["end_ms"] * SAMPLE_RATE/1000)
            if (end-start >= 1.6*SAMPLE_RATE
                    and before-max(start,left["start_sample"]) >= .16*SAMPLE_RATE
                    and min(end,right["end_sample"])-after >= .16*SAMPLE_RATE):
                cut = (before+after)//2
                best_gap = gap
                break
    return cut


class TranscriptionService:
    def __init__(self, device=None, activity_detector=None):
        self.ready = False
        self.device = device
        self.error = None
        self.model = None
        self.processor = None
        self.pipeline = None
        self.activity_detector = activity_detector or SpeechActivityDetector()

    def load(self):
        if self.ready:
            return
        try:
            import torch
            self.activity_detector.load()
            if not self.activity_detector.ready:
                raise RuntimeError(self.activity_detector.error or 'Local speech detector is not ready')
            snapshot = ensure_transcription_model(offline=True)
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            from transformers.pipelines import pipeline

            self.device = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
            if self.device not in {"mps", "cpu"}:
                raise ValueError("Only local MPS and CPU transcription are supported")
            dtype = torch.float16 if self.device == "mps" else torch.float32
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                snapshot, local_files_only=True, use_safetensors=True,
                dtype=dtype, attn_implementation="eager",
            ).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
            self.pipeline = pipeline(
                "automatic-speech-recognition", model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device, dtype=dtype,
            )
            self.error = None
            self.ready = True
            logging.info("Raw Arabic transcription model ready on %s.", self.device)
        except Exception as exc:
            self.model = self.processor = self.pipeline = None
            self.ready = False
            self.error = str(exc)
            logging.exception("Transcription model could not be loaded. Run setup.py and check local MPS/CPU availability.")

    def warm(self, wave):
        """Compile ASR on an internal test signal; publish no user transcript."""
        if not self.ready: return
        activity=self.activity_detector.detect(np.asarray(wave,dtype=np.float32))
        if activity.get('reason') in {'detector_failed','detector_not_ready'}:
            raise RuntimeError('Local speech detector failed during warm-up')
        self.pipeline({'raw':np.asarray(wave,dtype=np.float32),'sampling_rate':SAMPLE_RATE},
                      return_timestamps='word',generate_kwargs=dict(GENERATION_OPTIONS))

    def transcribe(self, wave):
        if not self.ready:
            raise RuntimeError("Transcription model is not ready")
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1 or not np.isfinite(wave).all():
            raise ValueError("Transcription audio must be finite mono PCM at 16 kHz")
        rms=float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))) if len(wave) else 0.0
        peak=float(np.max(np.abs(wave))) if len(wave) else 0.0
        empty = dict(text="", words=[], speech=False,
                     audio_activity=dict(status='no_speech',reason='below_audio_floor',
                                         rms=rms,peak=peak,speech_samples=0))
        if not len(wave) or rms < MIN_AUDIO_RMS:
            return empty
        activity=self.activity_detector.detect(wave)
        if activity.get('reason') in {'detector_failed','detector_not_ready'}:
            self.ready=False
            self.error='Local speech detector failed; restart the app after checking its model.'
            raise RuntimeError(self.error)
        if activity.get('status')!='speech':
            return dict(empty,audio_activity=activity)
        spans=activity.get('spans',[])
        if not spans or any(type(s.get('start_sample')) is not int or type(s.get('end_sample')) is not int
                            or not 0<=s['start_sample']<s['end_sample']<=len(wave) for s in spans):
            raise RuntimeError('Speech detector returned invalid audio spans')
        # Quran recitation can begin with a soft vowel that VAD recognizes
        # late. Keep the complete left context; VAD is not a word cutter.
        detected_end=max(s['end_sample'] for s in spans)
        # Trim only a clearly long non-speech tail, keeping extra end context
        # beyond the detector padding for quiet final consonants and vowels.
        last=(min(len(wave),detected_end+SAMPLE_RATE//4)
              if len(wave)-detected_end>=SAMPLE_RATE else len(wave))
        speech_wave=wave[:last]
        output = self._decode(speech_wave)
        cut = _crossed_pause(output["words"], spans)
        if cut is not None:
            output = self._verify_pause(speech_wave, output, cut)
        return dict(**output, speech=bool(output["text"].strip()), audio_activity=activity)

    def _verify_pause(self, wave, original, cut):
        """Require agreement between whole audio and independently decoded speech on each side."""
        padding = round(.16*SAMPLE_RATE)
        whole = self._decode(wave, leading_padding=padding)
        candidate = whole["text"].split()
        if (not candidate or candidate == original["text"].split() or not whole["words"]
                or any(w["start_ms"] is None or w["end_ms"] <= w["start_ms"] for w in whole["words"])):
            return original
        left = self._decode(wave[:cut], leading_padding=padding)
        left_words = left["text"].split()
        if not left_words or candidate[:len(left_words)] != left_words:
            return original
        right = self._decode(wave[cut:], leading_padding=padding)
        if not right["text"].strip() or left_words + right["text"].split() != candidate:
            return original
        # Keep a single decoder's complete raw text, including repetitions and
        # lexical mistakes. The split decodes provide evidence, never prompts.
        return whole

    def recover_repetitions(self, transcript, phones, wave):
        """Verify possible repeated occurrences against their independent audio intervals."""
        from repeat_recovery import recover_repetitions
        return recover_repetitions(transcript, phones, wave, self._decode)

    def _decode(self, wave, *, leading_padding=0):
        """Unprompted decode, reporting word times on the unpadded audio clock."""
        speech_wave = np.pad(wave, (leading_padding,0)) if leading_padding else wave
        output = self.pipeline(
            {"raw": speech_wave, "sampling_rate": SAMPLE_RATE}, return_timestamps="word",
            generate_kwargs=dict(GENERATION_OPTIONS),
        )
        text = output.get("text", "")
        if not isinstance(text, str):
            raise RuntimeError("ASR returned invalid raw text")
        words = []
        duration = len(speech_wave) / SAMPLE_RATE
        pad_seconds=leading_padding/SAMPLE_RATE
        for chunk in output.get("chunks", []):
            word = chunk.get("text")
            if not isinstance(word, str) or not word:
                continue
            times = chunk.get("timestamp")
            start, end = times if isinstance(times, (tuple, list)) and len(times) == 2 else (None, None)
            if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                    or not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end < start or end > duration + 0.1
                    or start > duration + 0.1):
                # Keep exactly what ASR heard, but do not invent alignment or a
                # correctness verdict for an incomplete window-edge boundary.
                words.append(dict(text=word, start_ms=None, end_ms=None))
            else:
                words.append(dict(text=word, start_ms=round(max(0,min(start,duration)-pad_seconds)*1000),
                                  end_ms=round(max(0,min(end,duration)-pad_seconds)*1000)))
        return dict(text=text, words=words)
