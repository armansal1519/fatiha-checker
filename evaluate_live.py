"""Send rolling windows of user-provided recordings to the real running server."""
import csv
import io
import json
import wave
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import requests

from audio_processing import decode_audio

BASE = Path(__file__).resolve().parent


def wav_bytes(samples):
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(16000)
        writer.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return out.getvalue()


def main():
    health = requests.get("http://127.0.0.1:5000/health", timeout=10)
    health.raise_for_status()
    assert health.json()["ready"]
    records = []
    with ZipFile(BASE.parent / "drive-download-20260913T170716Z-1-001.zip") as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("test_set.csv").decode("utf-8-sig"))))
        span = next(r for r in rows if "/sentence1/" in r["path"] and "/native_speakers/" in r["path"] and r["canon"] == r["ans"])
        chosen = [(span["path"].split("/dataset/", 1)[-1], 2, 4),
                  ("native_speakers/males/speaker3/sentence2/wav/Sp3_NM_S2_5.wav", 5, 13),
                  ("non-native_speakers/males/speaker1/sentence2/wav/Sp1_M_S2_2.wav", 5, 13),
                  ("non-native_speakers/males/speaker15/sentence2/wav/Sp15_M_S2_3.wav", 5, 13)]
        for path, start_ayah, cursor in chosen:
            raw = archive.read(path)
            audio = decode_audio(raw, "input.wav")
            ends = list(range(32000, len(audio), 32000)) + [len(audio)]
            snapshots = []
            for sequence, end in enumerate(ends, 1):
                chunk = audio[max(0, end - 160000):end]
                start = perf_counter()
                response = requests.post("http://127.0.0.1:5000/live/analyze",
                    files={"audio":("live.wav", wav_bytes(chunk), "audio/wav")},
                    data=dict(cursor=cursor, start_ayah=start_ayah, sequence=sequence, settled=str(end == len(audio)).lower()), timeout=30)
                response.raise_for_status()
                result = response.json()
                snapshots.append(dict(audio_until_seconds=round(end / 16000, 3),
                                      wall_seconds=round(perf_counter() - start, 3), response=result))
                cursor = result["cursor"]
                print(Path(path).name, round(end / 16000, 2), result["status"], cursor,
                      [(u["index"], u["state"]) for u in result["updates"]], flush=True)
            records.append(dict(source=path, start_ayah=start_ayah, snapshots=snapshots))
            samples = BASE / "tests/audio_samples"
            samples.mkdir(exist_ok=True)
            if start_ayah == 2:
                (samples / "verses_2_and_3.wav").write_bytes(raw)
            elif path.startswith("non-native"):
                (samples / "known_letter_errors.wav").write_bytes(raw)
    report = dict(model=health.json()["model"], cadence_seconds=2, window_seconds=10, records=records,
                  limitations=["Recorded audio windows, not a live microphone benchmark.",
                               "Multi-verse audio covers ayat 2 and 3; a complete human recording of all seven ayat was not used.",
                               "Colors and cursor can change between snapshots; this is not a religious correctness score."])
    (BASE / "reports/live-evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
