from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transformers
import transformers.pipelines

import transcription
from transcription import TranscriptionService


class FakeActivityDetector:
    ready=True
    error=None
    def load(self): pass
    def detect(self,wave):
        return dict(status='speech',spans=[dict(start_sample=0,end_sample=len(wave))],
                    speech_samples=len(wave),rms=.1,peak=.1)


def without_activity(result):
    return {k:v for k,v in result.items() if k!='audio_activity'}


def ready_service(output, detector=None):
    service = TranscriptionService(activity_detector=detector or FakeActivityDetector())
    calls = []

    def infer(audio, **kwargs):
        calls.append((audio, kwargs))
        return output

    service.pipeline = infer
    service.ready = True
    return service, calls


def test_spoken_error_and_repetition_remain_raw_without_quran_prompt():
    output = {"text":" هَمْدُ هَمْدُ لله", "chunks":[
        {"text":" هَمْدُ", "timestamp":(.1,.3)},
        {"text":" هَمْدُ", "timestamp":(.3,.6)},
        {"text":" لله", "timestamp":(.6,.9)},
    ]}
    service, calls = ready_service(output)
    result = service.transcribe(np.ones(16000, dtype=np.float32) * .1)
    assert without_activity(result) == {"text":output["text"], "speech":True, "words":[
        {"text":" هَمْدُ", "start_ms":100, "end_ms":300},
        {"text":" هَمْدُ", "start_ms":300, "end_ms":600},
        {"text":" لله", "start_ms":600, "end_ms":900},
    ]}
    audio, options = calls[0]
    assert audio["sampling_rate"] == 16000 and audio["raw"].dtype == np.float32
    assert options["return_timestamps"] == "word"
    generate = options["generate_kwargs"]
    assert generate["language"] == "ar" and generate["task"] == "transcribe"
    assert generate["prompt_ids"] is None and not generate["condition_on_prev_tokens"]
    assert generate["repetition_penalty"] == 1 and generate["no_repeat_ngram_size"] == 0
    assert "hotwords" not in generate and "prefix_allowed_tokens_fn" not in generate


def test_silence_does_not_invoke_asr_or_fabricate_text():
    service, calls = ready_service({"text":"not allowed"})
    assert without_activity(service.transcribe(np.zeros(16000, dtype=np.float32))) == {"text":"", "words":[], "speech":False}
    assert not calls


@pytest.mark.parametrize("times", [None, (None,.2), (.2,None), (-.1,.2), (.5,.2), (0,float("nan")), (0,5)])
def test_invalid_word_timing_is_not_replaced_with_invented_evidence(times):
    service, _ = ready_service({"text":"خطأ", "chunks":[{"text":"خطأ", "timestamp":times}]})
    assert without_activity(service.transcribe(np.ones(16000, dtype=np.float32))) == {
        "text":"خطأ", "speech":True,
        "words":[{"text":"خطأ", "start_ms":None, "end_ms":None}]}


def test_phrase_without_word_timing_remains_raw_and_ungradable():
    service, _ = ready_service({"text":"هذا ما قرأه المستخدم"})
    assert without_activity(service.transcribe(np.ones(16000, dtype=np.float32))) == {
        "text":"هذا ما قرأه المستخدم", "speech":True, "words":[]}


def test_invalid_audio_and_not_ready_do_not_run_pipeline():
    with pytest.raises(RuntimeError, match="not ready"):
        TranscriptionService().transcribe(np.ones(16000))
    service, calls = ready_service({})
    for wave in [np.ones((2,100)), np.array([np.nan])]:
        with pytest.raises(ValueError):
            service.transcribe(wave)
    assert not calls


@pytest.mark.parametrize("mps,dtype,device", [(True,torch.float16,"mps"),(False,torch.float32,"cpu")])
def test_load_uses_pinned_local_assets_eager_attention_and_device_precision(monkeypatch, mps, dtype, device):
    captured = {}

    class Model:
        def eval(self):
            return self
        def to(self, actual):
            captured["device"] = actual
            return self

    def load_model(path, **kwargs):
        captured["model"] = (path, kwargs)
        return Model()

    def assets(**kwargs):
        assert kwargs == {"offline":True}
        return "pinned-snapshot"

    def make_pipeline(task, **kwargs):
        captured["pipeline"] = kwargs
        return lambda *args, **kwargs: {}

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(transcription, "ensure_transcription_model", assets)
    monkeypatch.setattr(transformers.AutoModelForSpeechSeq2Seq, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs:
                        SimpleNamespace(tokenizer="tokenizer",feature_extractor="features"))
    monkeypatch.setattr(transformers.pipelines, "pipeline", make_pipeline)
    service = TranscriptionService(activity_detector=FakeActivityDetector())
    service.load()
    assert service.ready and service.device == device and service.error is None
    assert captured["model"] == ("pinned-snapshot", {"local_files_only":True,
        "use_safetensors":True,"dtype":dtype,"attn_implementation":"eager"})
    assert captured["device"] == device
    assert captured["pipeline"]["dtype"] == dtype


def test_load_failure_is_explicit_and_keeps_transcription_unready(monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("Pinned ASR snapshot missing")

    monkeypatch.setattr(transcription, "ensure_transcription_model", unavailable)
    service = TranscriptionService(device="cpu",activity_detector=FakeActivityDetector())
    service.load()
    assert not service.ready and service.pipeline is None
    assert service.error == "Pinned ASR snapshot missing"


@pytest.mark.parametrize('status',['no_speech','uncertain'])
def test_nonzero_noise_and_uncertain_activity_cannot_invoke_whisper(status):
    detector=SimpleNamespace(detect=lambda wave:dict(status=status,spans=[],speech_samples=0,rms=.01,peak=.02))
    service,calls=ready_service({'text':' ترجمة نانسي قنقر'},detector)
    wave=np.random.default_rng(7).normal(0,.01,128000).astype(np.float32)
    result=service.transcribe(wave)
    assert not calls and not result['speech'] and result['text']=='' and result['words']==[]
    assert result['audio_activity']['status']==status


def test_speech_gate_does_not_blacklist_specific_words():
    service,calls=ready_service({'text':' ترجمة نانسي قنقر','chunks':[]})
    result=service.transcribe(np.ones(16000,dtype=np.float32)*.1)
    assert calls and result['text']==' ترجمة نانسي قنقر'
    assert result['speech']


def test_speech_crop_preserves_raw_wrong_word_and_complete_left_context():
    detector=SimpleNamespace(detect=lambda wave:dict(status='speech',
        spans=[dict(start_sample=16000,end_sample=32000)],speech_samples=16000,rms=.1,peak=.2))
    service,calls=ready_service({'text':' ولم يلمد','chunks':[{'text':' ولم','timestamp':(.1,.3)},
        {'text':' يلمد','timestamp':(.3,.8)}]},detector)
    wave=np.arange(48000,dtype=np.float32)/48000
    result=service.transcribe(wave)
    np.testing.assert_array_equal(calls[0][0]['raw'],wave[:36000])
    assert result['text']==' ولم يلمد'
    assert result['words']==[dict(text=' ولم',start_ms=100,end_ms=300),
                             dict(text=' يلمد',start_ms=300,end_ms=800)]
    assert not calls[0][1]['generate_kwargs']['condition_on_prev_tokens']


def test_late_vad_onset_and_short_silence_cannot_cut_quiet_recitation_edges():
    detector=SimpleNamespace(detect=lambda wave:dict(status='speech',
        spans=[dict(start_sample=16000,end_sample=40000)],speech_samples=24000,rms=.02,peak=.1))
    service,calls=ready_service({'text':' إياك نعبد','chunks':[]},detector)
    wave=np.ones(48000,dtype=np.float32)*.02
    service.transcribe(wave)
    np.testing.assert_array_equal(calls[0][0]['raw'],wave)


def test_invalid_activity_spans_fail_instead_of_fabricating_audio():
    detector=SimpleNamespace(detect=lambda wave:dict(status='speech',spans=[dict(start_sample=5,end_sample=999999)]))
    service,calls=ready_service({'text':'fake'},detector)
    with pytest.raises(RuntimeError,match='invalid audio spans'):
        service.transcribe(np.ones(16000,dtype=np.float32)*.1)
    assert not calls


def test_activity_detector_failure_is_not_reported_as_successful_silence():
    detector=SimpleNamespace(detect=lambda wave:dict(status='uncertain',reason='detector_failed'))
    service,calls=ready_service({'text':'fake'},detector)
    with pytest.raises(RuntimeError,match='speech detector failed'):
        service.transcribe(np.ones(16000,dtype=np.float32)*.1)
    assert not service.ready and not calls


def test_pause_redecode_requires_agreement_and_unpads_absolute_word_times():
    spans=[dict(start_sample=0,end_sample=24000),dict(start_sample=32000,end_sample=64000)]
    detector=SimpleNamespace(detect=lambda wave:dict(status='speech',spans=spans))
    original={'text':' أول شيء','chunks':[{'text':' أول','timestamp':(0,2.6)},{'text':' شيء','timestamp':(2.6,3.5)}]}
    whole={'text':' دار دار شجر','chunks':[{'text':' دار','timestamp':(0,.66)},
           {'text':' دار','timestamp':(.86,1.46)},{'text':' شجر','timestamp':(2.4,3.5)}]}
    left={'text':' دار دار','chunks':[]}
    right={'text':' شجر','chunks':[]}
    def replay(right_confirmation):
        service,_=ready_service(original,detector)
        outputs=iter([original,whole,left,right_confirmation])
        samples=[]
        def decode(buf,**kwargs):
            samples.append(buf['raw'].copy())
            assert kwargs['generate_kwargs']==transcription.GENERATION_OPTIONS
            return next(outputs)
        service.pipeline=decode
        return service.transcribe(np.ones(64000,dtype=np.float32)*.1),samples
    accepted,calls=replay(right)
    assert [len(x) for x in calls]==[64000,66560,30560,38560]
    assert np.all(calls[1][:2560]==0) and np.all(calls[1][2560:]==calls[0])
    assert accepted['text']==whole['text']
    assert [(w['start_ms'],w['end_ms']) for w in accepted['words']]==[(0,500),(700,1300),(2240,3340)]
    disagreed,_=replay({'text':' ورد','chunks':[]})
    assert disagreed['text']==original['text']


def test_real_long_word_in_one_speech_span_and_legitimate_phrase_are_kept():
    output={'text':' اشتركوا في القناة','chunks':[{'text':' اشتركوا','timestamp':(.1,2.3)},
            {'text':' في','timestamp':(2.4,2.8)}, {'text':' القناة','timestamp':(2.8,3.8)}]}
    service,calls=ready_service(output)
    assert service.transcribe(np.ones(64000,dtype=np.float32)*.1)['text']==output['text']
    assert len(calls)==1
