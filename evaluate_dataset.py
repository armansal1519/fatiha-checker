"""Run the real application against labeled, local L2-KSU recordings.

No training occurs. The dataset's character labels are not a complete gold
standard for harakat or tajweed; this is an integration/qualitative evaluation.
"""
import argparse
import csv
import io
import json
import logging
from collections import Counter
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

from app import create_app
from recitation import RecitationService


def select_samples(archive, per_group=4):
    rows = list(csv.DictReader(io.StringIO(archive.read("test_set.csv").decode("utf-8-sig"))))
    candidates = [row for row in rows if "/sentence2/" in row["path"]]
    selected, used = [], set()
    for has_error in (False, True):
        for row in candidates:
            if (row["canon"].split() != row["ans"].split()) != has_error:
                continue
            path = row["path"].split("/dataset/", 1)[-1]
            speaker = path.split("/sentence", 1)[0]
            if speaker in used or path not in archive.namelist():
                continue
            selected.append({**row, "archive_path": path, "annotated_character_error": has_error})
            used.add(speaker)
            if sum(x["annotated_character_error"] == has_error for x in selected) >= per_group:
                break
    return rows, candidates, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(__file__).parent.parent / "drive-download-20260913T170716Z-1-001.zip")
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--export-samples", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    service = RecitationService()
    start = perf_counter()
    service.load()
    if not service.ready:
        raise SystemExit("The real model could not be loaded; no evaluation was produced.")
    client = create_app(service).test_client()
    results = []
    with ZipFile(args.archive) as archive:
        rows, candidates, selected = select_samples(archive, args.per_group)
        inventory = dict(wav_files=sum(n.lower().endswith(".wav") for n in archive.namelist()),
                         test_rows=len(rows), ayah5_test_rows=len(candidates), selected=len(selected))
        for idx, row in enumerate(selected):
            audio = archive.read(row["archive_path"])
            response = client.post("/analyze", data={"ayah": "5", "audio": (io.BytesIO(audio), "sample.wav")})
            result = {"source": row["archive_path"], "canonical_characters": row["canon"],
                      "annotated_characters": row["ans"], "annotated_character_error": row["annotated_character_error"],
                      "http_status": response.status_code, "response": response.json}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if args.export_samples and idx in (0, args.per_group):
                output = Path(__file__).parent / "tests" / "audio_samples"
                output.mkdir(parents=True, exist_ok=True)
                filename = "annotated_error.wav" if row["annotated_character_error"] else "no_annotated_character_error.wav"
                (output / filename).write_bytes(audio)
        # Controlled omission from a real recording: use annotated word intervals
        # in an accompanying TextGrid in future research, never label synthetic
        # omissions as naturally occurring learner errors.
    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report = dict(dataset="L2-KSU (user-provided archive)", model="obadx/muaalem-model-v3_2",
                  inventory=inventory, total_seconds=round(perf_counter() - start, 2), results=results,
                  limitations=["Only verse 1:5 is evaluated here; some other corpus sentences span or split ayat.",
                               "canon/ans label character errors, not a complete harakat/gemination assessment.",
                               "This small selection is not a Persian-speaker accuracy benchmark."])
    (report_dir / "dataset-evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    statuses = Counter(r["response"].get("status") for r in results)
    print("Evaluation complete:", dict(statuses), "Report: reports/dataset-evaluation.json", flush=True)


if __name__ == "__main__":
    main()
