"""Acquire the pinned evaluation selection. Audio stays in the local cache."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Lock
from urllib.parse import urlparse
import wave

import av
import numpy as np
from filelock import FileLock

ROOT = Path(__file__).resolve().parent
SELECTION = ROOT / 'evaluation/selection-v1.json'
IDENTITIES = ROOT / 'evaluation/identity-v1.json'
ASSET_LOCK = ROOT / 'evaluation/assets-v1.json'
PACK = ROOT / '.cache/evaluation-pack'
RATE = 16000
FLEURS_INDEX = 'https://datasets-server.huggingface.co/rows?dataset=google%2Ffleurs&config=ar_eg&split=test&offset=0&length=4'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    Path(tmp.name).replace(path)


def publish(path, obj):
    write_atomic(path, (json.dumps(obj, ensure_ascii=False, indent=1) + '\n').encode())


class AssetCache:
    def __init__(self, root, expected, offline):
        self.root = root
        self.expected = expected
        self.offline = offline
        self.hashes = {}
        self.locks = defaultdict(Lock)
        self.guard = Lock()

    def obtain(self, address, signed_url=None):
        # Stable provider URLs form the on-disk keys; expiring URLs stay in memory.
        key = digest(address.encode())
        path = self.root / 'assets' / key
        with self.guard:
            mutex = self.locks[key]
        with mutex:
            if key in self.hashes:
                return path
            expected = self.expected.get(key)
            if path.exists() and (not expected or digest(path.read_bytes()) == expected):
                self.hashes[key] = expected or digest(path.read_bytes())
                return path
            if self.offline:
                raise OSError(f'Offline asset missing or damaged: {address}')
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix('.part')
            if urlparse(address).scheme != 'https':
                raise ValueError('Only HTTPS sources are supported')
            command = ['curl', '--fail', '--silent', '--show-error', '--location',
                       '--proto', '=https', '--proto-redir', '=https',
                       '--speed-time', '45', '--speed-limit', '256', '--connect-timeout', '15',
                       '--user-agent', 'FatihaChecker-local-evaluation/1']
            url = signed_url or address
            for _ in range(3):
                result = subprocess.run(command + ['--continue-at', '-', '--output', str(part), url], capture_output=True)
                if expected and part.exists() and digest(part.read_bytes()) == expected:
                    # A previous transfer may have completed just before the disconnect.
                    break
                if result.returncode in (33, 36):
                    # The origin explicitly rejected byte-range resumption.
                    result = subprocess.run(command + ['--output', str(part), url], capture_output=True)
                if not result.returncode or result.returncode not in (18, 28, 52, 55, 56, 92):
                    break
            if result.returncode and not (expected and part.exists() and digest(part.read_bytes()) == expected):
                raise RuntimeError(f'curl {result.returncode}: {address}: ' + result.stderr.decode()[-160:])
            actual = digest(part.read_bytes())
            if expected and actual != expected:
                part.unlink()
                raise ValueError(f'Publisher bytes changed at {address}; expected {expected}, got {actual}')
            part.replace(path)
            self.hashes[key] = actual
            return path


def read_pcm(path):
    with av.open(BytesIO(path.read_bytes()), options={'protocol_whitelist': ''}) as container:
        resampler = av.AudioResampler(format='s16', layout='mono', rate=RATE)
        parts = []
        for frame in container.decode(audio=0):
            parts.extend(x.to_ndarray().ravel() for x in resampler.resample(frame))
        parts.extend(x.to_ndarray().ravel() for x in resampler.resample(None))
    return np.concatenate(parts).astype('<i2')


def chunk_bytes(pcm):
    out = BytesIO()
    with wave.open(out, 'wb') as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(RATE)
        dst.writeframes(pcm.astype('<i2').tobytes())
    return out.getvalue()


def edit_order(n, operation):
    if n < 4:
        raise ValueError('Four source words required for bounded edits')
    order = list(range(n))
    if operation == 'identity':
        return order, None
    middle = n // 2 - 1 if n > 4 else 1
    if operation == 'repetition':
        order.insert(middle + 1, middle)
        heard, refs, after = [middle + 1], [middle], None
    elif operation == 'omission':
        del order[middle]
        heard, refs, after = [], [middle], middle - 1
    elif operation == 'transposition':
        order[middle:middle+2] = [middle + 1, middle]
        heard, refs, after = [middle, middle + 1], [middle, middle + 1], None
    elif operation == 'substitution':
        order[middle] = n - 1
        heard, refs, after = [middle], [middle], None
    else:
        raise ValueError(operation)
    return order, dict(kind=operation, heard_indices=heard, reference_indices=refs, after_index=after)


def nonspeech(op):
    rng = np.random.default_rng(6718)
    t = np.arange(8 * RATE) / RATE
    if op == 'silence':
        x = np.zeros_like(t)
    elif op == 'fan':
        x = 0.009 * np.sin(2*np.pi*100*t) + 0.003 * np.sin(2*np.pi*200*t)
    elif op == 'noise':
        x = 0.012 * rng.standard_normal(len(t))
    elif op == 'impulses':
        x = np.zeros_like(t)
        for i in range(0, len(t) - 160, RATE // 3):
            x[i:i+160] = 0.2 * rng.standard_normal(160) * np.exp(-np.arange(160) / 18)
    else:
        raise ValueError(op)
    return np.rint(np.clip(x,-1,1)*32767).astype('<i2')


def prepare_clip(spec, cache, root):
    row = dict(spec)
    source = dict(row['source'])
    row['source'] = source
    kind = row['source_kind']
    if kind == 'professional':
        marks = json.loads(cache.obtain(source['timing_url']).read_bytes())
        mark = next(m for m in marks if m['ayah'] == row['ayah'])
        full = read_pcm(cache.obtain(source['audio_url']))
        start, end = mark['start_time'], mark['end_time']
        if start < 0 or end <= start or start*16 >= len(full):
            raise ValueError(f'Invalid publisher timestamps {start}–{end} vs {len(full)/16} ms')
        pcm = full[start*16:end*16]
        source.update(start_ms=start, end_ms=min(end, len(full)/16), published_end_ms=end)
    elif kind == 'natural-learner':
        pcm = read_pcm(cache.obtain(source['audio_url']))
    elif kind == 'controlled-edit-human':
        metadata = json.loads(cache.obtain(source['metadata_url']).read_bytes())['verse']
        if metadata['verse_key'] != f"{row['surah']}:{row['ayah']}":
            raise ValueError('Wrong source verse')
        words = [w for w in metadata['words'] if w['char_type_name']=='word']
        canonical = row['canonical_text'].split()
        if len(words) != len(canonical):
            raise ValueError(f'Canonical word count differs: {row["id"]}')
        order, event = edit_order(len(words), row['operation'])
        parts, segments, cursor = [], [], 0
        for index in order:
            w = words[index]
            url = f"https://audio.qurancdn.com/{w['audio_url']}"
            if w['position'] != index+1:
                raise ValueError('Unexpected word position')
            piece = read_pcm(cache.obtain(url))
            parts.append(piece)
            segments.append(dict(word=canonical[index], published_word=w['text_uthmani'],
                                 ref_index=index, url=url, start_ms=round(cursor/16),
                                 end_ms=round((cursor+len(piece))/16)))
            cursor += len(piece)
        pcm = np.concatenate(parts)
        row.update(spoken_text=' '.join(canonical[i] for i in order), spoken_text_verified=True,
                   events=[event] if event else [], error_labels_complete=True,
                   error_spoken_word_indices=event['heard_indices'] if event else [])
        row['source_annotations'] = dict(expected_omission=canonical[event['reference_indices'][0]]
                                         if event and event['kind']=='omission' else None)
        source.update(asset_segments=segments, source_word_sequence=order)
    elif kind == 'ordinary-speech':
        # The published asset path pins the FLEURS snapshot. Its temporary CDN
        # query string can expire and is deliberately excluded from this pack.
        address = source['asset']
        if cache.offline or (cache.root/'assets'/digest(address.encode())).exists():
            path = cache.obtain(address)
        else:
            with tempfile.TemporaryDirectory() as directory:
                fresh = AssetCache(Path(directory), {}, False)
                row_url = json.loads(fresh.obtain(FLEURS_INDEX).read_bytes())['rows'][source['index']]['row']
            live_url = next(a['src'] for a in row_url['audio'] if a['src'].split('?')[0] == address)
            if row_url['id'] != source['id']:
                raise ValueError('FLEURS row changed')
            path = cache.obtain(address, live_url)
        pcm = read_pcm(path)
    elif kind == 'noise':
        pcm = nonspeech(row['operation'])
        row.update(spoken_text='', spoken_text_verified=True)
    else:
        raise ValueError(kind)
    if not len(pcm): raise ValueError(f'Empty output: {row["id"]}')
    audio = chunk_bytes(pcm)
    write_atomic(root/row['audio'], audio)
    return {**row, 'sha256':digest(audio), 'duration_s':len(pcm)/RATE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PACK)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Verify generated clips against the frozen lock, without network')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with FileLock(args.root/'builder.lock'):
        return build(args)


def build(args):
    selection_bytes = SELECTION.read_bytes()
    identity_bytes = IDENTITIES.read_bytes()
    selection = json.loads(selection_bytes)['clips'] + json.loads(identity_bytes)['clips']
    selection_hash = digest(selection_bytes + identity_bytes)
    lock = json.loads(ASSET_LOCK.read_text()) if ASSET_LOCK.exists() else {}
    if lock and lock['selection_sha256'] != selection_hash:
        raise ValueError('Selection differs from the frozen asset lock')
    if args.verify:
        if not lock: raise ValueError('Missing frozen lock')
        manifest_bytes = (args.root/'manifest.json').read_bytes()
        if digest(manifest_bytes) != lock['manifest_sha256']:
            raise ValueError('Manifest metadata differs from the frozen lock')
        manifest = json.loads(manifest_bytes)
        if manifest['selection_sha256'] != lock['selection_sha256'] or manifest.get('complete') is not True:
            raise ValueError('Incomplete manifest or wrong source selection')
        if {r['id'] for r in manifest['clips']} != set(lock['clips']):
            raise ValueError('Manifest clip IDs differ from lock')
        for row in manifest['clips']:
            if digest((args.root/row['audio']).read_bytes()) != lock['clips'][row['id']] or row['sha256'] != lock['clips'][row['id']]:
                raise ValueError(f'Clip SHA-256 differs: {row["id"]}')
        print(f"Verified {len(manifest['clips'])} clips")
        return 0
    cache = AssetCache(args.root, lock.get('assets', {}), args.offline)
    results, errors = {}, {}
    def flush():
        rows = [results[x['id']] for x in selection if x['id'] in results]
        doc = dict(schema_version=1, selection_sha256=selection_hash,
                   original_selection_sha256=digest(selection_bytes), identity_extension_sha256=digest(identity_bytes),
                   complete=len(rows)==len(selection), expected=len(selection),
                   clips=rows, failed=errors, summary=dict(Counter(x['source_kind'] for x in rows)))
        publish(args.root/('manifest.json' if doc['complete'] else 'partial-manifest.json'), doc)

    # Interleave groups so early partial reports include more than one voice.
    ordered = sorted(selection, key=lambda x:digest(x['id'].encode()))
    with ThreadPoolExecutor(max_workers=max(1,min(args.jobs,12))) as pool:
        tasks = {pool.submit(prepare_clip, x, cache, args.root):x['id'] for x in ordered}
        for index, task in enumerate(as_completed(tasks), 1):
            clip_id = tasks[task]
            try:
                row = task.result()
                if lock and lock['clips'][clip_id] != row['sha256']:
                    raise ValueError('Rendered audio differs from frozen lock')
                results[clip_id] = row
            except Exception as exc:
                errors[clip_id] = f'{type(exc).__name__}: {exc}'
                print(f'{clip_id}: {errors[clip_id]}', file=sys.stderr, flush=True)
            if index % 8 == 0:
                flush()
                print(f'{index}/{len(tasks)}, {len(errors)} failed', flush=True)
    flush()
    if errors: return 1
    report = dict(selection_sha256=selection_hash, manifest_sha256=digest((args.root/'manifest.json').read_bytes()),
                  assets=dict(sorted(cache.hashes.items())),
                  clips={k:v['sha256'] for k,v in sorted(results.items())})
    publish(args.root/'asset-lock.json',report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
