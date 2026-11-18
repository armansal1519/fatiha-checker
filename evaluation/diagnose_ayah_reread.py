"""Inspect the preparation boundaries and independent decodes of the three 93:1 copies."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_evaluation_pack import AssetCache, PACK, RATE, digest, publish, read_pcm
from recitation import RecitationService
from transcription import TranscriptionService
from verify_recitation_cases import normalized_words


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    args=parser.parse_args()
    pack=PACK/'repairs'
    lock=json.loads((ROOT/'evaluation/repair-voices-lock.json').read_bytes())
    assets=AssetCache(PACK,lock['assets'],True)
    manifest=json.loads((pack/'manifest.json').read_bytes())
    asr=TranscriptionService(args.device)
    phones=RecitationService()
    asr.load()
    phones.load()
    if not asr.ready or not phones.ready:
        raise ValueError('Both offline models are required')
    report=dict(method='Post-evaluation bounded diagnostic. Three independent unprompted crops per voice. The exact duplicated waveform and publisher timings are verifiable. ASR/CTC agreement is model evidence only. Source context includes material from adjacent verses.',
        manifest_sha256=digest((pack/'manifest.json').read_bytes()),device=args.device,
        dependencies={f:digest((ROOT/f).read_bytes()) for f in ('transcription.py','recitation.py','model_assets.py',
                'evaluation/diagnose_ayah_reread.py')}, results=[])
    for clip in (c for c in manifest['clips'] if c['case']=='short_ayah_reread'):
        original=clip['sources'][0]
        original_pcm=read_pcm(assets.obtain(original['audio_url']))
        left,right=(original[k] for k in ('source_start_sample','source_end_sample'))
        width=right-left
        media=pack/clip['audio']
        if digest(media.read_bytes())!=lock['clips'][clip['id']]:
            raise ValueError('Changed fixture')
        pcm=read_pcm(media)
        exact=np.array_equal(original_pcm[left:right],pcm[:width]) and np.array_equal(pcm[:width],pcm[width:2*width])
        if not exact:
            raise ValueError('First and second copies diverged from source')
        start=max(0,left-round(.4*RATE))
        source_slice=original_pcm[start:right+round(.55*RATE)]
        views=[('exact_source_cut',pcm[:width],0),('source_with_context',source_slice,left-start),
               ('second_copy_with_next_context',pcm[width:2*width+round(.7*RATE)],0)]
        row=dict(id=clip['id'],reader=clip['reader'],publisher_source=original['audio_url'],
                 source_interval_ms=[left/16,right/16],exact_double_copy=exact,
                 expected_word=clip['target_word'],samples=[])
        for kind,samples,offset in views:
            wave=samples.astype(np.float32)/32768
            reading=asr._decode(wave)
            ctc=phones.predict_live_timed(wave)
            row['samples'].append(dict(kind=kind,duration_ms=len(wave)/16,exact_cut_ms=[offset/16,(offset+width)/16],
                rms=float(np.sqrt(np.mean(wave*wave))),whisper=reading,ctc=ctc,
                exact_spelling_present=normalized_words(clip['target_word'])[0] in normalized_words(reading['text'])))
        report['results'].append(row)
        print(row['id'],[s['whisper']['text'] for s in row['samples']],flush=True)
    publish(args.output, report)


if __name__=='__main__':
    main()
