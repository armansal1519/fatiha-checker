"""Compare the raw Arabic ASR with and without acoustic pause verification."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_recitation_pack import input_windows, load_pcm
from transcription import SAMPLE_RATE, TranscriptionService
from verify_recitation_cases import normalized_words


def window(service, audio):
    decodes = []
    decode = service._decode

    def record(samples, *, leading_padding=0):
        result = decode(samples, leading_padding=leading_padding)
        decodes.append(dict(pcm_samples=len(samples), leading_padding=leading_padding,
                            raw=result['text'], words=result['words']))
        return result

    with patch.object(service, '_decode', side_effect=record):
        result = service.transcribe(audio)
    return dict(raw=result['text'], single_pass=decodes[0]['raw'] if decodes else '',
                speech_spans=result['audio_activity'].get('spans', []), decodes=decodes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',type=Path,default=ROOT/'.cache/evaluation-pack')
    parser.add_argument('--spoken-control',type=Path,required=True,
                        help='An independently spoken or locally synthesized اشتركوا في القناة recording.')
    parser.add_argument('--control-source',required=True)
    parser.add_argument('--device',choices=('cpu','mps'),default='mps')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    service=TranscriptionService(args.device)
    service.load()
    if not service.ready: raise RuntimeError(service.error)
    inputs=[('chapter_join',args.pack/'passages/audio/transition-hafs-112-113.wav')]
    inputs.append(('literal_phrase',args.spoken_control))
    for clip in json.loads((args.pack/'manifest.json').read_text())['clips']:
        if clip['source_kind']=='ordinary-speech':
            inputs.append((clip['id'],args.pack/clip['audio']))
    records=[]
    for name,path in inputs:
        wave=load_pcm(path)
        times=[(0,len(wave),True)] if name=='literal_phrase' else input_windows(len(wave))
        results=[]
        for start,end,_ in times:
            record=window(service,wave[start:end])
            results.append(dict(start_audio_s=start/SAMPLE_RATE,end_audio_s=end/SAMPLE_RATE,**record))
        records.append(dict(id=name,sha256=sha256(path.read_bytes()).hexdigest(),windows=results))
    # Jitter the reported analysis edges by a few real capture batches.
    wave=load_pcm(inputs[0][1])
    neighborhood=[]
    for start_seconds in (17.84,17.92,18.,18.08,18.16):
        for omitted_tail in (.32,.16,0.):
            end=len(wave)-round(omitted_tail*SAMPLE_RATE)
            observed=window(service,wave[round(start_seconds*SAMPLE_RATE):end])
            neighborhood.append(dict(start_audio_s=start_seconds,end_audio_s=end/SAMPLE_RATE,**observed))
    phrase=normalized_words('اشتركوا في القناة')
    final=records[0]['windows'][-1]
    control=records[1]['windows'][-1]
    summary=dict(original_tail_reproduced=normalized_words(final['single_pass'])==phrase,
                 repaired_tail=final['raw'],spoken_phrase=control['raw'],
                 literal_phrase_preserved=normalized_words(control['raw'])==phrase,
                 ordinary_windows=sum(len(r['windows']) for r in records[2:]),
                 ordinary_windows_changed=sum(w['raw']!=w['single_pass'] for r in records[2:] for w in r['windows']))
    def contains_phrase(text):
        words=normalized_words(text)
        return any(words[i:i+len(phrase)]==phrase for i in range(len(words)-len(phrase)+1))
    summary['neighborhood_windows']=len(neighborhood)
    summary['single_pass_boilerplate_windows']=sum(contains_phrase(w['single_pass']) for w in neighborhood)
    summary['verified_boilerplate_windows']=sum(contains_phrase(w['raw']) for w in neighborhood)
    report=dict(summary=summary,device=service.device,control_source=args.control_source,
                transcription_sha256=sha256((ROOT/'transcription.py').read_bytes()).hexdigest(),
                method='Independent unprompted ASR on fixed source windows. This test does not measure browser latency or human diagnostic accuracy.',
                results=records,chapter_join_neighborhood=neighborhood)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    passed=(summary['literal_phrase_preserved'] and not summary['verified_boilerplate_windows']
            and normalized_words(final['raw'])[-4:]==normalized_words('من شر ما خلق'))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
