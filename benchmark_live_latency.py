"""Real HTTP inference, simulated capture clock; no microphone latency claim.

Run before/after against the corresponding server. Audio never leaves localhost.
Colors use the same two-observation rule and a settled final snapshot.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter
from zipfile import ZipFile

import requests

from audio_processing import decode_audio
from evaluate_live import wav_bytes

BASE = Path(__file__).resolve().parent
CASES = [
    ("native_speakers/females/speaker8/sentence1/wav/Sp8_NF_S1_3.wav", 2, 4),
    ("native_speakers/males/speaker3/sentence2/wav/Sp3_NM_S2_5.wav", 5, 13),
    ("non-native_speakers/males/speaker1/sentence2/wav/Sp1_M_S2_2.wav", 5, 13),
    ("non-native_speakers/males/speaker15/sentence2/wav/Sp15_M_S2_3.wav", 5, 13),
    ("native_speakers/males/speaker11/sentence1/wav/Sp11_NM_S1_5.wav", 2, 4),
    ("non-native_speakers/females/speaker11/sentence1/wav/Sp11_F_S1_1.wav", 2, 4),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cadence", type=float, default=0.8)
    parser.add_argument("--window", type=float, default=6)
    args = parser.parse_args()
    report = dict(label=args.label, cadence=args.cadence, window=args.window, records=[],
                  note="Measured HTTP inference on real recordings; capture clock simulated without pauses. Not a human pronunciation accuracy benchmark.")
    with ZipFile(BASE.parent / "drive-download-20260913T170716Z-1-001.zip") as archive:
        for path, ayah, cursor in CASES:
            audio = decode_audio(archive.read(path), "input.wav")
            duration = len(audio) / 16000
            now, next_send = 0.0, args.cadence
            observations, states, snapshots = {}, {}, []
            first_position = first_color = None
            needs_context = False
            sequence = 0
            while True:
                now = max(now, next_send)
                end = min(duration, now)
                settled = end == duration
                # Final review always retains ten seconds of context.
                window = 10 if settled or needs_context else args.window
                chunk = audio[max(0, int((end-window)*16000)):int(end*16000)]
                started = perf_counter()
                sequence += 1
                response = requests.post("http://127.0.0.1:5000/live/analyze",
                    files={"audio": ("live.wav", wav_bytes(chunk), "audio/wav")},
                    data=dict(cursor=cursor, start_ayah=ayah, sequence=sequence, settled=str(settled).lower()), timeout=30)
                response.raise_for_status()
                result = response.json()
                needs_context = result["status"] != "tracking"
                elapsed = perf_counter()-started
                now = max(end, now if settled else end) + elapsed
                if result["status"] == "tracking":
                    if first_position is None:
                        first_position = round(now,3)
                    for update in result["updates"]:
                        index = update["index"]
                        old, count = observations.get(index, (None, 0))
                        count = count+1 if old == update["state"] else 1
                        observations[index] = update["state"], count
                        if count >= 2 or settled:
                            states[index] = update
                    if states and first_color is None:
                        first_color = round(now,3)
                cursor = result["cursor"]
                snapshots.append(dict(audio_until=round(end,3), duration=round(len(chunk)/16000,3),
                                      wall_seconds=round(elapsed,3), result=result))
                if settled:
                    break
                next_send = math.ceil(max(end+args.cadence, now+0.08)*10)/10
                next_send = min(duration, next_send)
            row = dict(source=path, start_ayah=ayah, duration=round(duration,3), first_position=first_position,
                       first_color=first_color, final_states=list(states.values()), snapshots=snapshots)
            report["records"].append(row)
            print(Path(path).name, "first color",first_color,"requests",len(snapshots),"states",[(k,v["state"]) for k,v in states.items()],flush=True)
    times = [s["wall_seconds"] for r in report["records"] for s in r["snapshots"]]
    report["median_request_seconds"] = round(median(times),3)
    report["requests"] = len(times)
    (BASE / "reports" / f"live-latency-{args.label}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
