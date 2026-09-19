"""Paced, real-wall-clock HTTP/WebSocket comparison against the real local model.

No model is loaded here and no synthetic prediction is used. The user-provided
recordings stay on localhost. A warm-up request is excluded from the report.
Run only while the application is otherwise idle.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import threading
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter, sleep
from zipfile import ZipFile

import numpy as np
import requests
from simple_websocket import Client, ConnectionClosed

from audio_processing import decode_audio
from benchmark_live_latency import BASE, CASES
from evaluate_live import wav_bytes

RATE = 16000
BLOCK_SAMPLES = 2048


class LocalClient(Client):
    def handshake(self):
        # simple-websocket 1.1.0 omits a non-default port from the Host header.
        # The socket is connected already; make its upgrade match browsers and
        # the server's strict same-origin check without relaxing that check.
        self.host = f"[{self.host}]:{self.port}" if ":" in self.host else f"{self.host}:{self.port}"
        return super().handshake()


def number_summary(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None}
    return dict(count=len(values), median=round(median(values), 4),
                p95=round(values[max(0, math.ceil(.95 * len(values)) - 1)], 4),
                min=round(values[0], 4), max=round(values[-1], 4))


def state_map(states):
    return {str(s["index"]): s["state"] for s in states}


def state_difference(left, right):
    a, b = state_map(left), state_map(right)
    return [{"index": int(k), "left": a.get(k), "right": b.get(k)}
            for k in sorted(set(a) | set(b), key=int) if a.get(k) != b.get(k)]


def pcm_frame(sequence, offset, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    return struct.pack("<II", sequence, offset) + pcm


def connect(base_url):
    url = base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    return LocalClient.connect(url.rstrip("/") + "/live/ws",
                          headers={"Origin": base_url.rstrip("/")},
                          max_message_size=1024 * 1024)


def receive_json(client, timeout=30):
    message = client.receive(timeout=timeout)
    if message is None:
        raise TimeoutError("Timed out waiting for the local WebSocket server")
    if not isinstance(message, str):
        raise AssertionError("Server messages must be JSON text")
    return json.loads(message)


def start_connection(base_url, ayah, rate=RATE):
    client = connect(base_url)
    try:
        client.send(json.dumps(dict(type="start", version=1, start_ayah=ayah,
                                    sample_rate=rate, channels=1)))
        ready = receive_json(client)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Session was not accepted: {ready}")
        return client, ready
    except BaseException:
        close_connection(client)
        raise


def close_connection(client):
    try:
        client.close()
    except (ConnectionClosed, OSError):
        pass


def run_websocket(audio, ayah, cursor, base_url, *, rate=RATE, block_samples=None,
                  on_event=None, allow_automatic_done=False):
    """Send each frame when its capture time has really elapsed, while receiving.

    This deliberately measures a recording replay, not physical microphone or
    browser rendering latency. Receiver timestamps are independent of sending.
    """
    block_samples = block_samples or round(.128 * rate)
    client, ready = start_connection(base_url, ayah, rate)
    events, ack_seconds, errors = [], [], []
    sent = {}
    finished = threading.Event()
    started = perf_counter()
    outcome = dict(first_position=None, first_color=None, final_states=[], done=None)

    def reader():
        try:
            while not finished.is_set():
                result = receive_json(client, timeout=35)
                now = perf_counter()
                elapsed = round(now - started, 4)
                kind = result.get("type")
                if kind == "ack":
                    sequence = result.get("sequence")
                    if sequence in sent:
                        ack_seconds.append(now - sent[sequence])
                else:
                    events.append(dict(received_seconds=elapsed, payload=result))
                if kind == "progress" and result.get("phase") == "tracking":
                    if outcome["first_position"] is None:
                        outcome["first_position"] = elapsed
                if kind in ("result", "done"):
                    updates = result.get("updates", [])
                    if updates:
                        if outcome["first_color"] is None:
                            outcome["first_color"] = elapsed
                        outcome["final_states"] = updates
                if on_event:
                    on_event(result, elapsed)
                if kind == "error":
                    if result.get("fatal", True):
                        errors.append(result)
                        finished.set()
                if kind == "done":
                    outcome["done"] = result
                    finished.set()
        except BaseException as exc:
            if not finished.is_set():
                errors.append(dict(type=type(exc).__name__, message=str(exc)))
                finished.set()

    receiver = threading.Thread(target=reader, daemon=True, name="benchmark-ws-receiver")
    receiver.start()
    sequence, offset = 0, 0
    try:
        while offset < len(audio) and not finished.is_set():
            end = min(len(audio), offset + block_samples)
            due = started + end / rate
            if finished.wait(max(0, due - perf_counter())):
                break
            # Timestamp before send because localhost acks can arrive immediately.
            sent[sequence] = perf_counter()
            client.send(pcm_frame(sequence, offset, audio[offset:end]))
            sequence += 1
            offset = end
        if not finished.is_set():
            try:
                client.send(json.dumps(dict(type="stop", last_sequence=sequence - 1,
                                            total_samples=offset)))
            except ConnectionClosed:
                if not allow_automatic_done:
                    raise
        if not finished.wait(35):
            raise TimeoutError("No done event after the final audio frame")
        receiver.join(timeout=2)
        if errors:
            raise RuntimeError(f"WebSocket session failed: {errors}")
        if outcome["done"] is None:
            raise AssertionError("The WebSocket closed without a done event")
        # A revision is emitted in both progress and result; count it just once.
        inference_by_revision = {}
        for event in events:
            p = event["payload"]
            if p.get("timings", {}).get("inference_ms") is not None:
                inference_by_revision[p.get("revision", len(inference_by_revision))] = p["timings"]["inference_ms"]
        return dict(transport="websocket", duration=round(len(audio)/rate, 4),
                    wall_seconds=round(perf_counter()-started, 4),
                    first_position=outcome["first_position"], first_color=outcome["first_color"],
                    final_states=outcome["final_states"], session_id=ready.get("session_id"),
                    frames_sent=sequence, samples_sent=offset,
                    ack_seconds=number_summary(ack_seconds),
                    ack_latencies_ms=[round(value * 1000, 3) for value in ack_seconds],
                    inference_ms=list(inference_by_revision.values()),
                    done=outcome["done"], events=events)
    finally:
        finished.set()
        close_connection(client)
        receiver.join(timeout=2)


def run_http(audio, ayah, cursor, base_url):
    """Previous client policy, paced by the real monotonic capture clock.

    One request at a time, 800ms cadence, 6s normal/10s recovery/final context.
    Silence decisions use actual 128ms chunks, matching the old energy gate.
    """
    started = perf_counter()
    duration = len(audio) / RATE
    next_poll = 0.1
    last_sent = 0.0
    last_finished = float("-inf")
    last_voice = 0.0
    voice_sample = 0
    sent_voice = -1
    pause_sent = False
    scanned = 0
    sequence = 0
    observations, states, snapshots = {}, {}, []
    first_position = first_color = None
    needs_context = False
    final_sent = False
    with requests.Session() as http:
        while not final_sent:
            sleep(max(0, started + min(next_poll, duration) - perf_counter()))
            now = perf_counter() - started
            end = min(len(audio), int(now * RATE))
            while scanned < end:
                block_end = min(end, scanned + BLOCK_SAMPLES)
                block = audio[scanned:block_end]
                if float(np.sqrt(np.mean(block ** 2))) > .001:
                    voice_sample = block_end
                    last_voice = block_end / RATE
                    pause_sent = False
                scanned = block_end
            final = end == len(audio)
            quiet = now - last_voice >= .5
            pause_due = quiet and not pause_sent and now - last_sent >= .35
            moving_due = not quiet and voice_sample > sent_voice and now - last_sent >= .8
            due = now - last_finished >= .08 and (pause_due or moving_due)
            next_poll = now + .1
            if not voice_sample:
                if final:
                    break
                continue
            if not final and not due:
                continue
            settled = final or pause_due
            context = 10 if settled or needs_context else 6
            trimmed_end = min(end, voice_sample + int(.25 * RATE))
            wave = audio[max(0, trimmed_end - int(context * RATE)):trimmed_end]
            if len(wave) < .75 * RATE:
                if final:
                    break
                continue
            sequence += 1
            last_sent = now
            sent_voice = voice_sample
            pause_sent = settled
            requested = perf_counter()
            response = http.post(base_url.rstrip("/") + "/live/analyze",
                                 files={"audio": ("live.wav", wav_bytes(wave), "audio/wav")},
                                 data=dict(cursor=cursor, start_ayah=ayah, sequence=sequence,
                                           settled=str(settled).lower()), timeout=35)
            response.raise_for_status()
            result = response.json()
            received = perf_counter() - started
            request_seconds = perf_counter() - requested
            last_finished = received
            needs_context = result["status"] != "tracking"
            if not needs_context:
                if first_position is None:
                    first_position = round(received, 4)
                for update in result["updates"]:
                    old, count = observations.get(update["index"], (None, 0))
                    count = count + 1 if old == update["state"] else 1
                    observations[update["index"]] = update["state"], count
                    if count >= 2 or settled:
                        states[update["index"]] = update
                if states and first_color is None:
                    first_color = round(received, 4)
            cursor = result["cursor"]
            snapshots.append(dict(received_seconds=round(received, 4),
                                  audio_until_seconds=round(end / RATE, 4),
                                  context_seconds=round(len(wave) / RATE, 4),
                                  request_seconds=round(request_seconds, 4), result=result))
            final_sent = final
            next_poll = min(duration, received + .08)
    return dict(transport="http", duration=round(duration, 4),
                wall_seconds=round(perf_counter()-started, 4), first_position=first_position,
                first_color=first_color, final_states=list(states.values()), snapshots=snapshots,
                inference_ms=[s["result"]["timings"]["inference_ms"] for s in snapshots
                              if "inference_ms" in s["result"].get("timings", {})])


def load_cases():
    with ZipFile(BASE.parent / "drive-download-20260913T170716Z-1-001.zip") as archive:
        return [(path, ayah, cursor, decode_audio(archive.read(path), "input.wav"))
                for path, ayah, cursor in CASES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--case-limit", type=int, default=6)
    parser.add_argument("--output", type=Path, default=BASE / "reports/websocket-benchmark.json")
    args = parser.parse_args()
    if args.runs < 1 or not 1 <= args.case_limit <= len(CASES):
        parser.error("runs must be positive; case-limit must be 1..6")
    health = requests.get(args.base_url + "/health", timeout=10)
    health.raise_for_status()
    assert health.json()["ready"]
    cases = load_cases()
    _, warm_ayah, warm_cursor, warm_audio = cases[1]
    warm = requests.post(args.base_url + "/live/analyze",
                         files={"audio": ("warm.wav", wav_bytes(warm_audio), "audio/wav")},
                         data=dict(cursor=warm_cursor, start_ayah=warm_ayah, settled="true"), timeout=35)
    warm.raise_for_status()
    baseline = json.loads((BASE / "reports/live-latency-after.json").read_text(encoding="utf-8"))
    baseline_by_path = {row["source"]: row for row in baseline["records"]}
    benchmark_started = perf_counter()
    report = dict(started_at=datetime.now(UTC).isoformat(), model=health.json(),
                  clock="perf_counter; real-time capture replay; no simulated clock",
                  rate=RATE, chunk_samples=BLOCK_SAMPLES, warmup_excluded=True, runs=args.runs,
                  records=[], comparisons=[], limitations=[
                      "Local recording replay, not microphone acquisition or browser paint latency.",
                      "Existing baseline used a simulated capture clock; real HTTP runs here are the direct comparison.",
                      "Different inference schedules can change tentative/final model alignment; no pronunciation accuracy score.",
                      "Dataset character annotations are not complete vowel/tashdid ground truth."])
    for path, ayah, cursor, audio in cases[:args.case_limit]:
        for run in range(1, args.runs + 1):
            paired = {}
            # Alternate order to reduce sustained-load/thermal ordering bias.
            order = ("http", "websocket") if run % 2 else ("websocket", "http")
            for transport in order:
                row = (run_http if transport == "http" else run_websocket)(audio, ayah, cursor, args.base_url)
                row.update(source=path, start_ayah=ayah, run=run)
                report["records"].append(row)
                paired[transport] = row
                print(Path(path).name, run, transport, "position", row["first_position"],
                      "color", row["first_color"], "states", state_map(row["final_states"]), flush=True)
            comparison = dict(source=path, run=run,
                              websocket_vs_http=state_difference(paired["websocket"]["final_states"], paired["http"]["final_states"]),
                              websocket_vs_old_baseline=state_difference(paired["websocket"]["final_states"], baseline_by_path[path]["final_states"]))
            report["comparisons"].append(comparison)
            # Keep partial real measurements if later interrupted; never fabricate completion.
            report["complete"] = False
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["summary"] = {}
    for transport in ("http", "websocket"):
        rows = [r for r in report["records"] if r["transport"] == transport]
        report["summary"][transport] = dict(records=len(rows),
            first_position_seconds=number_summary([r["first_position"] for r in rows]),
            first_color_seconds=number_summary([r["first_color"] for r in rows]),
            inference_ms=number_summary([t for r in rows for t in r["inference_ms"]]))
        if transport == "websocket":
            report["summary"][transport]["ack_ms"] = number_summary(
                [t for r in rows for t in r["ack_latencies_ms"]])
    report["complete"] = True
    report["wall_seconds"] = round(perf_counter() - benchmark_started, 3)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
