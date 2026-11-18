"""Re-evaluate the frozen pack from pre-confirmation ASR and timed CTC.

Completed clips of the original repair pass supply auditable primary evidence.
Missing clips and cache entries are evaluated using all real local models.
The benchmark's live preview, confirmation, and feedback stages run anew.
"""
import argparse
from collections import Counter
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_recitation_pack as benchmark
import model_assets
from quran_corpus import QuranCorpus
from transcription import GENERATION_OPTIONS


SOURCE_ENGINE = {
    "transcript_tracking.py": "7740e69df29c17f89d0ca63cdde68c444ff910ebc6a47b9808af15c945d32e35",
    "quran_corpus.py": "2010b8bebd8da1a63087c19a05f4950a70696d6b84efa61664e6b64e7969a56d",
    "word_alignment.py": "29f530b1dc0c118abdde8dba5236a6ff3f2bd5f0dbf855ed29c4a2e301ebfa61",
    "transcription.py": "53d733ef9b9bf6d7b57312494f14a32d4614e6bac63a7437f695215d5a3cabdc",
    "speech_activity.py": "6db1f9966d845a57e7ff0b3b254610c248fd74ea96ab820fdf0ba483e03955fc",
    "recitation.py": "bea4903e4917d4e875278c49bf873980a4952f3fbe92df7e472cc5aba22b26d2",
    "repeat_recovery.py": "ebcb1a5081c55eba10589166f2bd391b1f6ba6cd7995ca6254e8859aa8b7a5e6",
}
OLD_BENCHMARK_HASH = "d1ce8eb01943af0ed543583ff1fe225fcd0f13a25dea6c714b058e50271b715d"
PINNED_MODELS = {
    "muaalem": ["obadx/muaalem-model-v3_2", "01a1ef9fbe40d144ef845101e89ff924aed3fef5"],
    "whisper": ["openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"],
    "silero": ["snakers4/silero-vad", "5cd7945676eb32225748052e2e6a0580e4686a08"],
}
ACOUSTIC_FILES = ("transcription.py", "speech_activity.py", "recitation.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_report(path, report):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(report, ensure_ascii=False))
    tmp.replace(path)


def read_source(manifest_path, report_path):
    manifest = json.loads(manifest_path.read_bytes())
    manifest_sha = benchmark.digest(manifest_path)
    lock = json.loads((ROOT / "evaluation/assets-v1.json").read_bytes())
    require(manifest["complete"] and manifest_sha == lock["manifest_sha256"], "Frozen pack mismatch")
    raw = report_path.read_bytes()
    report = json.loads(raw)
    expected = dict(manifest_sha256=manifest_sha, engine_sha256=SOURCE_ENGINE, device="mps",
                    split="all", evaluator_sha256=OLD_BENCHMARK_HASH)
    require(report["key"] == expected, "Upstream report engine, evaluator, device or pack changed")
    require(report.get("status") in {"running", "completed"}, "Unexpected upstream report status")
    rows = report["results"]
    by_id = {r["id"]: r for r in rows}
    require(len(rows) == len(by_id) and list(by_id) == [c["id"] for c in manifest["clips"][:len(rows)]],
            "Upstream results are not a valid checkpoint of this frozen pack")
    if report["status"] == "completed":
        require(len(rows) == len(manifest["clips"]), "Completed upstream report is missing clips")
    return manifest, by_id, expected, sha256(raw).hexdigest()


class PrimaryReplay:
    def __init__(self, cache, fallback_dir, engine_hashes):
        models = {
            "muaalem": [model_assets.MODEL_ID, model_assets.MODEL_REVISION],
            "whisper": [model_assets.TRANSCRIPTION_MODEL_ID, model_assets.TRANSCRIPTION_MODEL_REVISION],
            "silero": [model_assets.ACTIVITY_MODEL_ID, model_assets.ACTIVITY_MODEL_REVISION],
        }
        require(models == PINNED_MODELS, "Models differ from upstream pins")
        for name in ACOUSTIC_FILES:
            require(engine_hashes[name] == SOURCE_ENGINE[name], f"Incompatible acoustic implementation: {name}")
        self.cache = cache
        self.engine = benchmark.ModelEvidence(fallback_dir, "mps")
        parts = ([models[k][1] for k in ("muaalem", "whisper", "silero")] +
                 [engine_hashes[f] for f in ACOUSTIC_FILES])
        require(self.engine.parts == parts + [engine_hashes["repeat_recovery.py"], "mps"],
                "ModelEvidence cache key format changed")
        self.old_prefix = "|".join(parts + [SOURCE_ENGINE["repeat_recovery.py"], "mps"]).encode()
        self.engine.speech.load()
        speech = self.engine.speech
        require(speech.ready and speech.model.device.type == "mps", "Offline MPS ASR unavailable")
        # Count only genuine decoder calls made from the final confirmation hook.
        # Both wrappers forward the real result and preserve its object identity.
        decode, recover = speech._decode, speech.recover_repetitions
        self.in_confirmation = False
        self.subclip_decodes = self.confirmations = 0

        @wraps(decode)
        def counted_decode(*args, **kwargs):
            if self.in_confirmation:
                self.subclip_decodes += 1
            return decode(*args, **kwargs)

        @wraps(recover)
        def counted_recovery(*args, **kwargs):
            self.in_confirmation = True
            self.confirmations += 1
            try:
                return recover(*args, **kwargs)
            finally:
                self.in_confirmation = False

        speech._decode = counted_decode
        speech.recover_repetitions = counted_recovery

    def window(self, samples, expected, audit):
        if expected is not None:
            require(len(samples) == round((expected["end"] - expected["start"]) * benchmark.RATE),
                    "Upstream window size mismatch")
        pcm = samples.tobytes()
        key = sha256(self.old_prefix + pcm).hexdigest()
        path = self.cache / (key + ".json")
        entry = dict(upstream_cache_key=key, pcm_sha256=sha256(pcm).hexdigest())
        count = self.subclip_decodes
        n = self.confirmations
        if expected is not None and path.is_file():
            raw = path.read_bytes()
            old = json.loads(raw)
            primary = old.get("preview", old["heard"])
            phones = old["phones"]
            require(isinstance(phones["text"], str) and isinstance(phones["groups"], list)
                    and isinstance(primary["words"], list) and isinstance(primary["text"], str)
                    and isinstance(primary["audio_activity"], dict), "Malformed model evidence")
            require(old["heard"]["text"] == expected["raw_window"] and
                    primary["audio_activity"]["status"] == expected["activity"],
                    "Cached evidence does not match the upstream checkpoint")
            verified = self.engine.speech.recover_repetitions(primary, phones, samples)
            result = {"heard": verified, "phones": phones}
            if verified is not primary:
                result["preview"] = primary
            entry.update(source="verified_original", primary_field="preview" if "preview" in old else "heard",
                         cache_file_sha256=sha256(raw).hexdigest())
        else:
            # Empty per-window cache forces actual Whisper, CTC and final confirmation.
            # A repeated fixture never skips its independent subclip re-decodes.
            with TemporaryDirectory(dir=self.engine.cache) as temp:
                fallback_root = self.engine.cache
                self.engine.cache = Path(temp)
                try:
                    result = self.engine(samples)
                finally:
                    self.engine.cache = fallback_root
            if self.confirmations == n:
                require(not result["heard"]["words"], "Full inference skipped verification")
                self.engine.speech.recover_repetitions(result["heard"], result["phones"], samples)
            entry.update(source="fresh_full_inference",
                         reason="unreported_source_clip" if expected is None else "original_cache_missing")
        require(self.confirmations == n + 1, "Every waveform must run final confirmation exactly once")
        entry.update(subclip_asr_calls=self.subclip_decodes - count,
                     confirmation_changed="preview" in result)
        audit.append(entry)
        return result

    def clip(self, clip, media, original, corpus):
        if original is not None:
            require(original["split"] == clip["split"] and original["source_kind"] == clip["source_kind"],
                    "Mismatched upstream fixture")
        audit = []
        def evidence(wave):
            expected = original["timeline"][len(audit)] if original is not None else None
            return self.window(wave, expected, audit)

        result = benchmark.run_clip(clip, media, evidence, corpus)
        timeline = result["timeline"]
        require(len(audit) == len(timeline), "Missing or extra acoustic window")
        if original is not None:
            expected = original["timeline"]
            require(len(timeline) == len(expected) and
                    all((e["start"], e["end"]) == (s["start"], s["end"]) for e, s in zip(timeline, expected)),
                    "Upstream acoustic intervals changed")
        for event, observation in zip(timeline, audit):
            event["evidence"] = observation
        return result


def run_split(split, manifest_path, manifest, source_rows, key, replay, corpus, out_dir):
    selected = [clip for clip in manifest["clips"] if clip["split"] == split]
    out = out_dir / f"repair-final-{split}.json"
    key = dict(key, split=split)
    report = dict(key=key, method="Fixed 16 kHz windows. Verified primary ASR and timed CTC from the original "
                  "pre-review run or live inference on cache miss. Fresh final repeated-word ASR subdecodes, "
                  "live preview/confirmation semantics, final tracker and scores. No wall-clock latency claim.",
                  results=[], status="running")
    if out.exists():
        prior = json.loads(out.read_bytes())
        require(prior.get("key") == key, f"Existing {out} has different provenance")
        report = prior
    completed = {row["id"] for row in report["results"]}
    require(len(completed) == len(report["results"]) and
            completed.issubset({c["id"] for c in selected}), "Unexpected checkpoint IDs")
    for clip in selected:
        if clip["id"] in completed:
            continue
        media = manifest_path.parent / clip["audio"]
        require(benchmark.digest(media) == clip["sha256"], f"File digest mismatch: {media}")
        row = replay.clip(clip, media, source_rows.get(clip["id"]), corpus)
        report["results"].append(row)
        if len(report["results"]) % 10 == 0:
            atomic_report(out, report)
            print(f"{split}: {len(report['results'])}/{len(selected)}", flush=True)
    require({row["id"] for row in report["results"]} == {c["id"] for c in selected},
            "Final split missing fixtures")
    require(all(benchmark.digest(ROOT / f) == h for f, h in key["engine_sha256"].items()) and
            benchmark.digest(Path(benchmark.__file__)) == key["evaluator_sha256"], "Engine changed mid-run")
    windows = [e["evidence"] for row in report["results"] for e in row["timeline"]]
    report.update(status="completed", summary=benchmark.summary(report["results"]),
                  replay=dict(windows=len(windows), sources=dict(Counter(e["source"] for e in windows)),
                              primary_fields=dict(Counter(e.get("primary_field") for e in windows)),
                              subclip_asr_calls=sum(e["subclip_asr_calls"] for e in windows),
                              confirmed_revisions=sum(e["confirmation_changed"] for e in windows)))
    atomic_report(out, report)
    print(f"{out.name}: {report['replay']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / ".cache/evaluation-pack/manifest.json")
    parser.add_argument("--original", type=Path, default=ROOT / "reports/repair-all-originals.json")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/evaluation-evidence")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--split", choices=["both", "dev", "holdout"], default="both")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    manifest, source_rows, upstream, upstream_digest = read_source(args.manifest, args.original)
    engine = {f: benchmark.digest(ROOT / f) for f in SOURCE_ENGINE}
    require(engine["repeat_recovery.py"] != SOURCE_ENGINE["repeat_recovery.py"], "Helper still at pre-review revision")
    key = dict(manifest_sha256=upstream["manifest_sha256"], engine_sha256=engine, device="mps",
               evaluator_sha256=benchmark.digest(Path(benchmark.__file__)), replay_sha256=benchmark.digest(Path(__file__)),
               upstream_sha256=upstream_digest, upstream_key=upstream, upstream_completed_clips=len(source_rows),
               model_revisions=PINNED_MODELS,
               generation_options=GENERATION_OPTIONS, runtime_devices=dict(whisper="mps", muaalem="cpu", silero="cpu"),
               live_socket_sha256=benchmark.digest(ROOT / "live_socket.py"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus = QuranCorpus()
    splits = ("dev", "holdout") if args.split == "both" else (args.split,)
    with TemporaryDirectory() as temp:
        replay = PrimaryReplay(args.cache, Path(temp), engine)
        for split in splits:
            run_split(split, args.manifest, manifest, source_rows, key, replay, corpus, args.out_dir)


if __name__ == "__main__":
    main()
