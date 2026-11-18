"""Prepare or score the additional independent controls for the two residual errors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_evaluation_pack import AssetCache, PACK, digest, publish, read_pcm
from evaluation.prepare_repair_voices import SOURCES, edited_case, save, words

SPEC = ROOT / "evaluation/repair-confirmation-v1.json"
LOCK = ROOT / "evaluation/repair-confirmation-lock.json"
DATA = PACK / "confirmation"


def read(path):
    return json.loads(path.read_bytes())


def verse_cases(reader, surah, cache):
    source = reader["folder_url"] + f"{surah:03d}.mp3"
    timing = f"https://www.mp3quran.net/api/v3/ayat_timing?surah={surah}&read={reader['id']}"
    spans = {row["ayah"]: row for row in read(cache.obtain(timing))}
    track = read_pcm(cache.obtain(source))
    start = 16 * spans[1]["start_time"]
    end_first = 16 * spans[1]["end_time"]
    end_third = 16 * spans[3]["end_time"]
    # Publisher endpoints are integer milliseconds. The decoded track can
    # end a few samples before its rounded final timestamp.
    if not 0 <= start < end_first < end_third <= len(track) + 16:
        raise ValueError(f"Publisher interval unavailable for {source}")
    end_third = min(end_third, len(track))
    target, = words(surah, 1)
    clips = []
    for repeats in (1, 2):
        intervals = [(start, end_first)] * (repeats - 1) + [(start, end_third)]
        row = dict(id=f"reader-{reader['id']}-{surah}-{repeats}", reader=reader["name"],
                   group=f"mp3quran-{reader['id']}", source_kind="professional",
                   case="clean_short_ayah" if repeats == 1 else "short_ayah_reread",
                   surah=surah, ayah=1, target_word=target, repeat_count=repeats,
                   spoken_text_verified=False, error_labels_complete=False, spoken_text=None,
                   source_audio=source, source_timing=timing, source_intervals=intervals,
                   terms="https://www.mp3quran.net/ar/contact-us")
        clips.append(save(row, [track[lo:hi] for lo, hi in intervals], destination=DATA))
    return clips


def prepare(spec, offline):
    existing = read(LOCK) if LOCK.exists() else None
    known = read(ROOT / "evaluation/assets-v1.json")["assets"].copy()
    known.update(read(ROOT / "evaluation/repair-voices-lock.json")["assets"])
    if existing:
        known.update(existing["assets"])
    cache = AssetCache(PACK, known, offline)
    readers = {r["id"]: r for r in read(SOURCES)["readers"]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        passages = [pool.submit(verse_cases, readers[reader_id], surah, cache)
                    for reader_id in spec["reader_ids"] for surah in spec["chapters"]]
        changes = [pool.submit(edited_case, e["surah"], e["ayah"], e["position"],
                               e["donor"] if kind == "single_letter" else None, kind,
                               cache, destination=DATA)
                   for e in spec["single_letter_edits"] for kind in ("single_letter", "identity")]
        clips = [c for job in passages for c in job.result()] + [job.result() for job in changes]
    key = digest(SPEC.read_bytes())
    lock = dict(selection_sha256=key, clips={r["id"]: r["sha256"] for r in clips},
                assets=dict(sorted(cache.hashes.items())))
    if existing and existing != lock:
        raise ValueError("Existing control lock differs from rebuilt sources or selection")
    publish(DATA / "manifest.json", dict(selection_sha256=key, clips=clips))
    if not existing:
        publish(LOCK, lock)
    print(len(clips), "confirmed audio files", flush=True)


def score(spec, engine, output):
    from benchmark_recitation_pack import ModelEvidence
    from evaluation.evaluate_repair_voices import run, summary
    sys.path.insert(0, str(engine.resolve()))
    from quran_corpus import QuranCorpus
    manifest, lock = read(DATA / "manifest.json"), read(LOCK)
    media = {r["id"]: r["sha256"] for r in manifest["clips"]}
    selection = digest(SPEC.read_bytes())
    if lock["clips"] != media or selection != lock["selection_sha256"] or selection != manifest["selection_sha256"]:
        raise ValueError("Confirmation audio differs from the locked selection")
    modules = ("transcription", "recitation", "speech_activity", "repeat_recovery",
               "transcript_tracking", "quran_corpus", "word_alignment")
    hashes = {}
    for module in modules:
        source = Path(find_spec(module).origin).resolve()
        requested = engine.resolve() / f"{module}.py"
        if source != (requested if requested.exists() else ROOT / f"{module}.py"):
            raise ValueError(f"Unexpected module loaded for {module}")
        hashes[module] = digest(source.read_bytes())
    key = dict(engine=hashes, selection=selection, lock=digest(LOCK.read_bytes()), device="mps",
               runner=digest(Path(__file__).read_bytes()),
               scoring=digest((ROOT / "evaluation/evaluate_repair_voices.py").read_bytes()))
    report = read(output) if output.exists() else dict(key=key, complete=False, results=[])
    if report["key"] != key:
        raise ValueError("Existing output belongs to another build")
    observed = {r["id"] for r in report["results"]}
    evidence = ModelEvidence(ROOT / ".cache/evaluation-evidence", "mps")
    corpus = QuranCorpus()
    for clip in manifest["clips"]:
        path = DATA / clip["audio"]
        if digest(path.read_bytes()) != clip["sha256"]:
            raise ValueError(f"Incorrect waveform for {clip['id']}")
        if clip["id"] not in observed:
            report["results"].append(run(clip, path, evidence, corpus, spec["single_letter_edits"]))
            publish(output, report)
            print(len(report["results"]), len(media), clip["id"], flush=True)
    report.update(complete=True, summary=summary(report["results"]))
    publish(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Verify cached source audio during preparation")
    parser.add_argument("--score", type=Path, metavar="OUTPUT", help="Evaluate the previously prepared files")
    parser.add_argument("--engine-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.score:
        score(read(SPEC), args.engine_root, args.score)
    else:
        prepare(read(SPEC), args.offline)
