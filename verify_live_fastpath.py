"""Compare the optimized path to Muaalem's official API on identical audio."""
import json
import logging
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

from audio_processing import decode_audio
from benchmark_live_latency import CASES
from live_tracking import LiveFollower
from recitation import RecitationService


def main():
    base = Path(__file__).resolve().parent
    service = RecitationService()
    service.load()
    if not service.ready:
        raise RuntimeError("Model failed to load")
    logging.getLogger().setLevel(logging.WARNING)
    follower = LiveFollower(service.references)
    results = []
    with ZipFile(base.parent / "drive-download-20260913T170716Z-1-001.zip") as archive:
        for path, _, _ in CASES:
            audio = decode_audio(archive.read(path), "audio.wav")
            for end in sorted(set((12800, 25600, min(len(audio),96000),min(len(audio),160000)))):
                wave = audio[:end]
                start = perf_counter()
                expected = service.predict(wave, follower.model_reference)
                official_seconds = perf_counter()-start
                start = perf_counter()
                actual = service.predict_live(wave)
                live_seconds = perf_counter()-start
                assert actual == expected, f"Different phonemes: {path} at {end/16000}s"
                start = perf_counter()
                cached = service.predict_live(wave.copy())
                cache_seconds = perf_counter()-start
                assert cached == actual
                results.append(dict(source=path, seconds=end/16000, identical=True,
                                    official_seconds=round(official_seconds,4), live_seconds=round(live_seconds,4),
                                    cache_seconds=round(cache_seconds,5)))
            print(Path(path).name, "phonemes identical",flush=True)
    (base / "reports/live-fastpath-verification.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("Verified",len(results),"real audio windows",flush=True)


if __name__ == "__main__":
    main()
