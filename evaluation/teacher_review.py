"""Build a self-contained listening worksheet from the pinned public WAV selection."""
import argparse
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from html import escape
from io import BytesIO, StringIO
import json
import math
import os
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from urllib.request import Request, urlopen
import wave
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "evaluation/teacher-review-v1.json"
TEMPLATE = ROOT / "evaluation/teacher-review.html.in"
SCRIPT = ROOT / "evaluation/teacher-review.js"
FONTS = ROOT / "static/fonts"
DEFAULT_OUTPUT = ROOT / "reports/hamkhan-teacher-review.html"


def digest(data):
    return sha256(data).hexdigest()


def encoded(data):
    return b64encode(data).decode("ascii")


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        name = temp.name
        try:
            temp.write(data)
            temp.close()
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)


def source_url(spec, row):
    return f"https://huggingface.co/datasets/{spec['dataset']}/resolve/{spec['revision']}/{quote(row['path'])}"


def packet_identity(spec_bytes):
    return f"hamkhan-letters-v1-{digest(spec_bytes)[:16]}"


def load_audio(spec, row, assets, offline):
    url = source_url(spec, row)
    path = assets / digest(url.encode())
    data = path.read_bytes() if path.is_file() else b""
    if digest(data) != row["sha256"]:
        if offline:
            raise ValueError(f"Missing or changed cached audio for {row['path']}. Rerun without --offline.")
        request = Request(url, headers={"User-Agent": "Hamkhan-review-packet/1"})
        with urlopen(request, timeout=60) as response:
            data = response.read(10 * 1024 * 1024)
        if digest(data) != row["sha256"]:
            raise ValueError(f"Publisher audio differs from the pinned SHA-256 for {row['path']}")
        write_atomic(path, data)
    with wave.open(BytesIO(data), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
    if duration != row["seconds"]:
        raise ValueError(f"Wrong audio duration for {row['path']}")
    return {"duration": duration, "wav": encoded(data)}


def build(output, assets, offline):
    spec_bytes = SPEC.read_bytes()
    spec = json.loads(spec_bytes)
    packet_id = packet_identity(spec_bytes)
    clips = spec["clips"]
    # Render neutral task identifiers only. Keep evaluation outcomes and model
    # expectations outside the worksheet and the specialist's returned file.
    index = {key: str(i + 1) for i, key in enumerate(clips)}
    tasks = []
    for task in spec["tasks"]:
        row = clips[task["clip"]]
        tasks.append(dict(id=task["id"], clip=index[task["clip"]], surah=row["surah"], ayah=row["ayah"],
                          words=row["words"], word_index=task["word_index"]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        recordings = list(pool.map(lambda row: load_audio(spec, row, assets, offline), clips.values()))
    payload = dict(id=packet_id, tasks=tasks, clips={index[key]: rec for key, rec in zip(clips, recordings)})
    sources = "".join(f'<li><a href="{escape(source_url(spec, row), quote=True)}" '
                      f'rel="noreferrer">فایل {i}</a> <code>{row["sha256"]}</code></li>'
                      for i, row in enumerate(clips.values(), 1))
    template = Template(TEMPLATE.read_text(encoding="utf-8"))
    html = template.substitute(
        vazir=encoded((FONTS / "Vazirmatn-Variable.woff2").read_bytes()),
        amiri=encoded((FONTS / "Amiri-Regular.ttf").read_bytes()),
        notices=escape("\n\n".join((FONTS / name).read_text(encoding="utf-8")
                                  for name in ("Amiri-OFL.txt", "Vazirmatn-OFL.txt"))),
        dataset=f"https://huggingface.co/datasets/{spec['dataset']}/blob/{spec['revision']}/README.md",
        sources=sources, payload=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"),
        script=SCRIPT.read_text(encoding="utf-8"))
    raw = html.encode("utf-8")
    write_atomic(output, raw)
    buffer = BytesIO()
    # Fixed metadata makes identical source inputs produce identical archives.
    with ZipFile(buffer, "w") as archive:
        info = ZipInfo(output.name)
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, raw)
    zipped = output.with_suffix(".zip")
    write_atomic(zipped, buffer.getvalue())
    print(f"{len(tasks)} listening tasks, {len(recordings)} WAVs. {packet_id}\n{output}\n{zipped}")


def check_answers(payload, spec_bytes):
    """Join the returned independent notes to pinned source positions."""
    spec = json.loads(spec_bytes)
    if (not isinstance(payload, dict) or payload.get("schema") != "hamkhan-teacher-answers/v1"
            or payload.get("packet_id") != packet_identity(spec_bytes)):
        raise ValueError("The response is from a different worksheet or schema version.")
    tasks = {t["id"]: t for t in spec["tasks"]}
    if not isinstance(payload.get("answers"), list) or len(payload["answers"]) > len(tasks):
        raise ValueError("Invalid response count.")
    answers = {}
    for row in payload["answers"]:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise ValueError("Invalid response.")
        key = row["task_id"]
        if key not in tasks or key in answers:
            raise ValueError(f"Unknown or duplicate task ID: {key}")
        if "verdict" not in row or row["verdict"] not in (None, "match", "difference", "uncertain"):
            raise ValueError(f"Invalid verdict in {key}")
        for field, limit in (("expected", 200), ("heard", 200), ("note", 1000)):
            if not isinstance(row.get(field), str) or len(row[field]) > limit:
                raise ValueError(f"Invalid {field} in {key}")
        if "start_s" not in row or "end_s" not in row:
            raise ValueError(f"Missing time field in {key}")
        start, end = row["start_s"], row["end_s"]
        if start is not None or end is not None:
            duration = spec["clips"][tasks[key]["clip"]]["seconds"]
            if (not all(type(n) in (int, float) and math.isfinite(n) for n in (start, end))
                    or not 0 <= start < end <= duration):
                raise ValueError(f"Invalid source interval in {key}")
        answers[key] = row
    table = []
    for key, task in tasks.items():
        clip = spec["clips"][task["clip"]]
        pos = task["word_index"]
        a = answers.get(key, {})
        status = a.get("verdict") or "pending"
        typed = status == "difference" and bool(a["heard"].strip() and a["expected"].strip()
                                                and a["heard"].strip() != a["expected"].strip())
        table.append(dict(task_id=key, clip=task["clip"], surah=clip["surah"], ayah=clip["ayah"],
                          word_index=pos, word=clip["words"][pos], verdict=status,
                          expected=a.get("expected", ""), heard=a.get("heard", ""),
                          start_s=a.get("start_s"), end_s=a.get("end_s"), note=a.get("note", ""),
                          typed_contrast=typed, timed_typed_contrast=typed and a["start_s"] is not None))
    counts = {s: sum(r["verdict"] == s for r in table) for s in ("pending", "match", "difference", "uncertain")}
    stats = dict(tasks=len(tasks), completed=len(tasks)-counts["pending"], **counts,
                 typed_contrasts=sum(r["typed_contrast"] for r in table),
                 timed_typed_contrasts=sum(r["timed_typed_contrast"] for r in table))
    return stats, table


def export_table(path, table):
    def cell(value):
        # Do not let a reviewer-entered note become a spreadsheet formula.
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=table[0].keys())
    writer.writeheader()
    writer.writerows({k: cell(v) for k, v in r.items()} for r in table)
    write_atomic(path, buffer.getvalue().encode("utf-8-sig"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets", type=Path, default=ROOT / ".cache/evaluation-pack/assets")
    parser.add_argument("--offline", action="store_true", help="Use previously verified source WAVs only")
    parser.add_argument("--check-answers", type=Path, metavar="JSON", help="Validate a returned worksheet")
    parser.add_argument("--csv", type=Path, metavar="PATH", help="Save the joined responses with --check-answers")
    args = parser.parse_args()
    if args.check_answers:
        if args.check_answers.stat().st_size > 500000:
            parser.error("Answer file is too large.")
        try:
            stats, table = check_answers(json.loads(args.check_answers.read_bytes()), SPEC.read_bytes())
        except ValueError as exc:
            parser.error(str(exc))
        if args.csv:
            if args.csv.resolve() == args.check_answers.resolve():
                parser.error("--csv must differ from the input answer file")
            export_table(args.csv, table)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        raise SystemExit(0)
    if args.csv:
        parser.error("--csv requires --check-answers")
    if args.output.suffix.lower() != ".html":
        parser.error("--output must name an HTML file")
    build(args.output, args.assets, args.offline)
