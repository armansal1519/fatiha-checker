"""Opt-in real-model WebSocket soak and cached offline restart checks.

The soak sends 180 seconds of real, paced audio to the running local server.
The offline mode loads a separate real model and binds an ephemeral loopback
server, with all non-loopback socket/DNS access blocked inside this process.
Run the modes sequentially while the application is otherwise idle.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import threading
from pathlib import Path
from time import perf_counter, sleep
from unittest.mock import patch

import numpy as np
import requests
import av

from benchmark_websocket import (BASE, RATE, close_connection, load_cases,
                                 pcm_frame, run_websocket, start_connection)


def process_memory(pid):
    """Read Windows working set/private bytes without extra dependencies."""
    if not pid or os.name != "nt":
        return None

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
    handle = kernel.OpenProcess(0x1000, 0, pid)
    if not handle:
        return {"error": f"OpenProcess failed: {ctypes.get_last_error()}"}
    try:
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return {"error": f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}"}
        return dict(working_set_bytes=counters.WorkingSetSize, private_bytes=counters.PrivateUsage)
    finally:
        kernel.CloseHandle(handle)


def resample_fixture(samples, rate):
    converter = av.AudioResampler(format="fltp", layout="mono", rate=rate)
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="fltp", layout="mono")
    frame.sample_rate = RATE
    frames = converter.resample(frame) + converter.resample(None)
    return np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames]).astype(np.float32)


def soak(base_url, server_pid):
    _, ayah, cursor, recording = load_cases()[1]
    # Repetition is a stress fixture, not a claimed complete human recitation.
    unit = np.concatenate([recording, np.zeros(RATE // 2, dtype=np.float32)])
    samples = np.resize(unit, 180 * RATE).astype(np.float32)
    memory = []
    last_sample = -10.0

    def observe(event, elapsed):
        nonlocal last_sample
        if elapsed - last_sample >= 5:
            last_sample = elapsed
            memory.append(dict(elapsed_seconds=round(elapsed, 3), memory=process_memory(server_pid)))

    result = run_websocket(samples, ayah, cursor, base_url, on_event=observe,
                           allow_automatic_done=True)
    metrics = result["done"]["metrics"]
    assert result["samples_sent"] == 180 * RATE, result["samples_sent"]
    assert result["wall_seconds"] >= 180, result["wall_seconds"]
    assert metrics["total_received_samples"] == 180 * RATE, metrics
    assert metrics["output_samples"] == 180 * RATE, metrics
    assert metrics["ring_max_samples"] <= metrics["buffer_capacity_samples"] == 10 * RATE, metrics
    assert metrics["inference_max_inflight"] <= 1, metrics
    assert metrics["queued_inference_jobs"] == 0, metrics
    return dict(passed=True, duration_seconds=180,
                fixture="Repeated native ayah-5 recording separated by 500ms silence; not a pronunciation-quality test.",
                source="native_speakers/males/speaker3/sentence2/wav/Sp3_NM_S2_5.wav",
                bounded_ring_verified=True, no_inference_queue_verified=True,
                process_memory_samples=memory, session=result)


def reconnect(base_url):
    """Disconnect during actual inference, then establish a fresh real session."""
    _, ayah, cursor, recording = load_cases()[1]
    client, old_ready = start_connection(base_url, ayah)
    started = perf_counter()
    try:
        for seq, offset in enumerate(range(0, min(len(recording), int(RATE * 1.28)), 2048)):
            end = min(len(recording), offset + 2048)
            sleep(max(0, started + end / RATE - perf_counter()))
            client.send(pcm_frame(seq, offset, recording[offset:end]))
    finally:
        close_connection(client)
    # An already-started inference is allowed to finish before the next session.
    sleep(2)
    fresh = run_websocket(recording, ayah, cursor, base_url)
    assert fresh["session_id"] != old_ready["session_id"]
    assert all(e["payload"].get("session_id", fresh["session_id"]) == fresh["session_id"]
               for e in fresh["events"])
    return dict(passed=True, discarded_session_id=old_ready["session_id"], fresh_session=fresh)


def offline():
    """Actual fresh model load + actual socket transport, remote access denied."""
    from app import create_app
    from recitation import MODEL_ID, MODEL_REVISION, RecitationService
    from werkzeug.serving import make_server

    attempts = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def local_host(host):
        return host in ("127.0.0.1", "localhost", "::1", b"127.0.0.1", b"localhost", b"::1")

    def checked_address(address):
        if not isinstance(address, tuple) or not local_host(address[0]):
            attempts.append(repr(address))
            raise OSError("Offline validation blocks all non-loopback connections")

    def connect_local(sock, address):
        checked_address(address)
        return original_connect(sock, address)

    def connect_ex_local(sock, address):
        checked_address(address)
        return original_connect_ex(sock, address)

    def lookup_local(host, *args, **kwargs):
        if not local_host(host):
            attempts.append(f"DNS {host!r}")
            raise OSError("Offline validation blocks external DNS")
        return original_getaddrinfo(host, *args, **kwargs)

    started = perf_counter()
    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}), \
         patch.object(socket.socket, "connect", connect_local), \
         patch.object(socket.socket, "connect_ex", connect_ex_local), \
         patch.object(socket, "getaddrinfo", lookup_local):
        service = RecitationService()
        service.load()
        assert service.ready, "A fresh model instance must load from the local cache"
        load_seconds = perf_counter() - started
        application = create_app(service)
        server = make_server("127.0.0.1", 0, application, threaded=True)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True, name="offline-validation-server")
        server_thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            health = requests.get(base_url + "/health", timeout=10)
            health.raise_for_status()
            page = requests.get(base_url + "/live", timeout=10)
            page.raise_for_status()
            _, ayah, cursor, audio = load_cases()[1]
            session = run_websocket(audio, ayah, cursor, base_url)
            assert session["final_states"], "Real cached model must produce a checked word"
            rate_sessions = []
            for rate in (44100, 48000):
                converted = resample_fixture(audio, rate)
                alternate = run_websocket(converted, ayah, cursor, base_url, rate=rate)
                assert alternate["final_states"], f"No checked word for {rate}Hz real-model input"
                assert alternate["done"]["metrics"]["total_received_samples"] == len(converted)
                assert abs(alternate["done"]["metrics"]["output_samples"] - len(audio)) <= 1
                rate_sessions.append(dict(input_rate=rate, session=alternate))
            fonts = []
            for path in sorted(p for p in (BASE / "static/fonts").rglob("*")
                               if p.suffix in {".woff2", ".ttf"}):
                relative = path.relative_to(BASE / "static").as_posix()
                response = requests.get(base_url + "/static/" + relative, timeout=10)
                response.raise_for_status()
                assert response.content == path.read_bytes()
                fonts.append(relative)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            application.extensions["live_socket_manager"].shutdown()
    assert not attempts, attempts
    return dict(passed=True, model=MODEL_ID, revision=MODEL_REVISION,
                network="All non-loopback socket connect/connect_ex and DNS blocked in this fresh process",
                external_attempts=attempts, model_load_seconds=round(load_seconds, 3),
                served_local_fonts=fonts, session=session, resampled_sessions=rate_sessions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("soak", "reconnect", "offline"))
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = offline() if args.mode == "offline" else soak(args.base_url, args.server_pid) if args.mode == "soak" else reconnect(args.base_url)
    output = args.output or BASE / "reports" / f"websocket-{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.mode}: passed; report {output}", flush=True)


if __name__ == "__main__":
    main()
