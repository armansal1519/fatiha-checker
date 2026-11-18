"""Repeat the waveform identity checks for the three MP3Quran source recordings.

Compare three short, in-memory QUD previews per selected excerpt against the
publisher's pinned MP3. This program does not store preview bytes.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

import av
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_evaluation_pack import AssetCache, PACK, digest, publish

FPS = 4000
ROWS = ('https://datasets-server.huggingface.co/rows?dataset=QUD-Technologies%2F'
        'quran-alignment-benchmark&config=v1&split=test&offset=0&length=16')


def waveform(source):
    parts = []
    with av.open(source, format='mp3') as container:
        converter = av.AudioResampler(format='fltp', layout='mono', rate=FPS)
        for packet in container.demux(audio=0):
            try:
                frames = packet.decode()
            except av.FFmpegError:
                # HTTP byte ranges can begin between two MP3 frames.
                continue
            for frame in frames:
                parts.extend(t.to_ndarray().reshape(-1) for t in converter.resample(frame))
        parts.extend(t.to_ndarray().reshape(-1) for t in converter.resample(None))
    return np.concatenate(parts)


def best_match(bank, sample):
    length = len(sample)
    if len(bank) < length or not length:
        raise ValueError('Cannot compare short audio')
    size = 1 << (len(bank)+length-2).bit_length()
    cross = np.fft.irfft(np.fft.rfft(bank,size)*np.fft.rfft(sample[::-1],size), size)[length-1:len(bank)]
    total = np.r_[0,np.cumsum(bank.astype(float)**2)]
    energy = (total[length:]-total[:-length]) * (sample.astype(float)**2).sum()
    correlation = cross/np.sqrt(np.maximum(energy,1e-9))
    index = int(np.argmax(np.abs(correlation)))
    return index/FPS, float(correlation[index])


def verify(spec, source_root):
    r = requests.get(ROWS, timeout=45)
    r.raise_for_status()
    rows = {item['row']['id']:item['row'] for item in r.json()['rows']}
    results = []
    for name, source in spec['source_audio'].items():
        path = source_root/(name+'.mp3')
        if not path.exists():
            # A reconstruction from scratch uses AssetCache's URL-derived key.
            cache = AssetCache(source_root, {digest(source['audio_url'].encode()):source['sha256']}, True)
            path = cache.obtain(source['audio_url'])
        if digest(path.read_bytes()) != source['sha256']:
            raise ValueError(f'Publisher MP3 differs: {name}')
        bank = waveform(path)
        row = rows[name]
        address = row['audio'][0]['src']
        if ('/'+spec['source_revision']+'/') not in address:
            raise ValueError('Preview revision changed; audit the new QUD metadata first')
        head = requests.head(address, timeout=30)
        head.raise_for_status()
        length = int(head.headers['content-length'])
        for clip in spec['clips']:
            if clip['group'] != name: continue
            positions = []
            st,en = clip['qud_start_s'],clip['qud_end_s']
            for when in sorted({max(1, t) for t in (st+.4,(st+en)/2,en-4.5)}):
                start = round(length*(when-1)/row['duration_s'])
                with requests.get(address, headers={'Range':f'bytes={start}-{start+111999}'}, stream=True, timeout=40) as response:
                    if response.status_code != 206:
                        raise ValueError('Origin did not return a short preview range')
                    preview = waveform(BytesIO(response.content))
                sample = preview[FPS:4*FPS]
                approx = start/length * row['duration_s']
                lo,hi = int(max(approx-5,0)*FPS), int(min(approx+25,len(bank)/FPS)*FPS)
                found, corr = best_match(bank[lo:hi], sample)
                offset = found + lo/FPS - approx - 1
                if abs(corr) < .65 or abs(offset-clip['anchor_offset_s']) > .10:
                    raise ValueError(f'Origin mismatch: {clip["id"]} at {when:.3f}s')
                positions.append(dict(offset_s=offset, correlation=corr))
            results.append(dict(id=clip['id'],matched=positions))
    return results


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=PACK/'qud')
    p.add_argument('--output', type=Path, default=PACK/'qud/source-verification.json')
    args=p.parse_args()
    spec=json.loads((ROOT/'evaluation/timing-v1.json').read_bytes())
    results=verify(spec,args.root)
    publish(args.output, results)
    print(f'Verified {sum(len(r["matched"]) for r in results)} audio anchors in {len(results)} clips')
