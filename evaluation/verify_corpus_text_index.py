"""Check every canonical ayah against the public local search (no ASR involved)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from quran_transcript import Aya

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quran_corpus import QuranCorpus


def check():
    corpus, source = QuranCorpus(), Aya(1, 1)
    counts = {script: Counter() for script in ("imlaey", "uthmani")}
    failures = []
    for chapter in corpus.surahs:
        surah = chapter["surah"]
        for ayah in range(1, chapter["ayah_count"] + 1):
            original = source.set_new(surah, ayah).get()
            for script, result in counts.items():
                located = corpus.text_location(getattr(original, script))
                visible = [(c["surah"], c["ayah"]) for c in located["candidates"]]
                if located["status"] == "probable" and visible == [(surah, ayah)]:
                    result["correct_single"] += 1
                elif located["status"] == "ambiguous" and (surah, ayah) in visible:
                    result["among_visible_choices"] += 1
                elif located["status"] == "ambiguous" and len(visible) < located["total_candidates"]:
                    result["ambiguous_beyond_shortlist_unverified"] += 1
                else:
                    result["missing_or_wrong"] += 1
                    failures.append(dict(script=script, surah=surah, ayah=ayah,
                                         status=located["status"], visible=visible))
    return dict(unit="canonical ayah text", counts=counts, failures=failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = check()
    if options.output:
        options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"]))
    raise SystemExit(int(bool(report["failures"])))
