"""Real model, cached restart with blocked networking, and audio format checks.

Run after evaluate_dataset.py --export-samples. Does not change OS networking.
"""
import io
import json
import logging
import socket
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import av
import numpy as np

from app import create_app
from audio_processing import decode_audio
from recitation import MODEL_ID, MODEL_REVISION, RecitationService

BASE = Path(__file__).resolve().parent


def encode(samples, container_format, codec, rate=16000):
    output = io.BytesIO()
    with av.open(output, mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=rate)
        stream.layout = "mono"
        stream.bit_rate = 64000
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="fltp", layout="mono")
        frame.sample_rate = 16000
        resampler = av.AudioResampler(format=stream.codec_context.format, layout="mono", rate=rate)
        for converted in resampler.resample(frame) + resampler.resample(None):
            for packet in stream.encode(converted):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


def main():
    logging.basicConfig(level=logging.INFO)
    source = (BASE / "tests/audio_samples/no_annotated_character_error.wav").read_bytes()
    attempts = []

    def deny_network(*args, **kwargs):
        attempts.append(repr(args))
        raise OSError("Network blocked for this offline test")

    started = perf_counter()
    with patch.object(socket.socket, "connect", deny_network), patch.object(socket, "getaddrinfo", deny_network):
        service = RecitationService()
        service.load()
        assert service.ready, "Cached model must load with networking blocked"
        load_seconds = round(perf_counter() - started, 2)
        client = create_app(service).test_client()
        assert client.get("/health").json["ready"]
        formats = [("wav", source)]
        samples = decode_audio(source, "sample.wav")
        for extension, format_name, codec, rate in [
            ("mp3", "mp3", "libmp3lame", 16000),
            ("m4a", "ipod", "aac", 16000),
            ("webm", "webm", "libopus", 48000),
            ("ogg", "ogg", "libopus", 48000),
        ]:
            formats.append((extension, encode(samples, format_name, codec, rate)))
        results = []
        for extension, data in formats:
            response = client.post("/analyze", data={"ayah": "5", "audio": (io.BytesIO(data), f"sample.{extension}")})
            assert response.status_code == 200, response.json
            assert response.json["status"] in ("differences", "no_difference")
            results.append(dict(format=extension, bytes=len(data), response=response.json))
            print(extension, response.json["status"], response.json["elapsed_seconds"], flush=True)
    assert not attempts, f"Unexpected network calls: {attempts}"
    report = dict(model=MODEL_ID, revision=MODEL_REVISION, network="socket connect and DNS blocked in test process",
                  network_attempts=attempts, model_load_seconds=load_seconds, results=results)
    (BASE / "reports/runtime-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Offline startup and all five real-model format requests passed.")


if __name__ == "__main__":
    main()
