"""Offline, real-model CPU profiling; does not change the running server."""
import json
import logging
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from audio_processing import decode_audio
from live_tracking import LiveFollower
from recitation import RecitationService


def main():
    import torch
    from quran_muaalem.decode import phonemes_level_greedy_decode

    base = Path(__file__).resolve().parent
    wave = decode_audio((base / "tests/audio_samples/verses_2_and_3.wav").read_bytes(), "sample.wav")
    # Repetition supplies longer real speech for throughput measurement only.
    wave = np.tile(wave, 3)
    service = RecitationService()
    service.load()
    if not service.ready:
        raise RuntimeError("Model failed to load")
    logging.getLogger().setLevel(logging.WARNING)
    follower = LiveFollower(service.references)
    model = service.model
    rows = []
    with torch.inference_mode():
        for threads in (2, 4, 6):
            torch.set_num_threads(threads)
            for seconds in (2, 6, 10):
                sample = wave[:seconds * 16000]
                measured = []
                for repeat in range(3):
                    started = perf_counter()
                    features = model.processor([sample], sampling_rate=16000, return_tensors="pt")
                    features = {k: v.to(model.device, dtype=model.dtype) for k, v in features.items()}
                    prepared = perf_counter()
                    logits = model.model(**features, return_dict=False)[0]["phonemes"]
                    forwarded = perf_counter()
                    probs = logits.softmax(-1).cpu().float()
                    predicted = phonemes_level_greedy_decode(probs, model.multi_level_tokenizer.id_to_vocab["phonemes"])[0].text
                    decoded = perf_counter()
                    follower.align(predicted, cursor=4, start_ayah=2)
                    aligned = perf_counter()
                    measured.append(dict(features=prepared-started, forward=forwarded-prepared,
                                         decode=decoded-forwarded, align=aligned-decoded, total=aligned-started))
                row = dict(threads=threads, seconds=seconds,
                           **{k: round(median(r[k] for r in measured[1:]), 4) for k in measured[0]})
                rows.append(row)
                print(row, flush=True)
        # Compare with the complete official API, including unused sifat.
        torch.set_num_threads(6)
        complete = []
        for _ in range(3):
            start = perf_counter()
            text = service.predict(wave[:96000], follower.model_reference)
            complete.append(perf_counter()-start)
    report = dict(torch=torch.__version__, cpu_threads_before=6, samples=rows,
                  official_api_6_seconds_median=round(median(complete),4),
                  note="Long clips repeat a real recording for load measurement, not accuracy validation.")
    (base / "reports/live-profile-before.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
