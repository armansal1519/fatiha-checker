"""Send fixed public recitations and ordinary speech over the live v2 microphone socket."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_recitation_pack import digest, load_pcm
from evaluation.passage_challenges import extract
from prepare_evaluation_pack import AssetCache
from verify_stream_runtime import check_health, stream_v2


def ikhlas_audio(pack):
    """Crop the four verses using timings from the published, checksum-pinned chapter."""
    selection = json.loads((ROOT / 'evaluation/passage-challenges-v1.json').read_bytes())
    receipt = json.loads((ROOT / 'evaluation/passage-challenge-lock.json').read_bytes())
    timing = 'https://www.mp3quran.net/api/v3/ayat_timing?surah=112&read=1'
    audio = selection['reader']['folder_url'] + '112.mp3'
    assets = AssetCache(pack, receipt['assets'], offline=True)
    for url in (timing, audio):
        if sha256(url.encode()).hexdigest() not in receipt['assets']:
            raise ValueError('Unpinned supplemental asset')
    marks = [m for m in json.loads(assets.obtain(timing).read_bytes()) if m['ayah'] > 0]
    if [m['ayah'] for m in marks] != [1, 2, 3, 4]:
        raise ValueError('Expected publisher timings for all four verses')
    start = marks[0]['start_time']
    path = assets.obtain(audio)
    pcm = extract(path, start * 16, marks[-1]['end_time'] * 16).astype('float32') / 32768
    return pcm, dict(target=[112, 4], source=digest(path), audio_url=audio,
                     verse_intervals=[dict(ayah=m['ayah'], start_ms=m['start_time'] - start,
                                          end_ms=m['end_time'] - start) for m in marks])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:5001')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pack', type=Path, default=ROOT / '.cache/evaluation-pack')
    parser.add_argument('--clips', nargs='+', default=[
        'pro-17-87-14', 'pro-4-55-13', 'fleurs-ar-0-1993', 'public-ikhlas-1-4',
    ])
    args = parser.parse_args()
    manifest_path = args.pack / 'manifest.json'
    expected = json.loads((ROOT / 'evaluation/assets-v1.json').read_bytes())['manifest_sha256']
    if digest(manifest_path) != expected:
        raise ValueError('Unexpected media pack')
    clips = {c['id']: c for c in json.loads(manifest_path.read_bytes())['clips']}
    report = dict(url=args.url, health=check_health(args.url), complete=False,
                  manifest_sha256=expected, results=[])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    for clip_id in args.clips:
        if clip_id == 'public-ikhlas-1-4':
            pcm, metadata = ikhlas_audio(args.pack)
        else:
            clip = clips[clip_id]
            path = args.pack / clip['audio']
            if digest(path) != clip['sha256']:
                raise ValueError(f'Changed clip {clip_id}')
            pcm = load_pcm(path)
            metadata = dict(target=[clip.get('surah'), clip.get('ayah')], source=clip['sha256'])
        events = []

        def observe(event, state, elapsed):
            if event['type'] in ('transcript', 'feedback', 'done'):
                events.append(dict(
                    type=event['type'], wall_s=round(elapsed, 3), revision=event['revision'],
                    end_sample=event.get('end_sample'), text=''.join(t['text'] for t in state.tokens),
                    grade_reference=state.reference,
                    review_count=sum(t['state'] in ('match', 'difference') for t in state.tokens),
                    display=event.get('display_location'),
                ))

        run = stream_v2(args.url, lambda a, b: pcm[a:b], len(pcm), on_event=observe)
        report['results'].append(dict(id=clip_id, **metadata, samples=run['samples_sent'],
                                      acked=run['acknowledged_samples'],
                                      coverage=run['done']['metrics']['coverage_complete'], events=events))
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
        print(clip_id, events[-1]['text'], flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
