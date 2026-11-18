"""Reproduce the recorded 18.000-second Whisper window through a paced local WebSocket."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_recitation_pack import load_pcm
from verify_recitation_cases import normalized_words
from verify_stream_runtime import RATE, check_health, stream_v2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    path=ROOT/'.cache/evaluation-pack/passages/audio/transition-hafs-112-113.wav'
    source_digest=sha256(path.read_bytes()).hexdigest()
    lock=json.loads((ROOT/'evaluation/passage-challenge-lock.json').read_text())
    assert source_digest==lock['clips']['transition-hafs-112-113']
    wave=load_pcm(path)[18*RATE:]
    history=[]
    def observe(event,acc,elapsed):
        if event['type']=='transcript':
            history.append(dict(revision=event['revision'],start_sample=event['start_sample'],
                                end_sample=event['end_sample'],raw=''.join(t['text'] for t in acc.tokens)))
    health=check_health(args.url)
    run=stream_v2(args.url,lambda start,end:wave[start:end],len(wave),on_event=observe)
    raw=''.join(t['text'] for t in run['final_tokens'])
    target=normalized_words('اشتركوا في القناة')
    def contains(raw):
        words=normalized_words(raw)
        return any(words[i:i+len(target)]==target for i in range(len(words)-len(target)+1))
    report=dict(health=health,url=args.url,source_file=path.name,source_sha256=source_digest,
                pcm_sha256=sha256(wave.tobytes()).hexdigest(),source_start_s=18,
                input_samples=len(wave),session=run['session_id'],samples_acknowledged=run['acknowledged_samples'],
                coverage_complete=run['done']['metrics']['coverage_complete'],stages=run['model_stage_calls'],
                final_raw=raw,ever_boilerplate=any(contains(e['raw']) for e in history),
                final_boilerplate=contains(raw),final_four_match=normalized_words(raw)[-4:]==normalized_words('من شر ما خلق'),
                revisions=history,final_tokens=run['final_tokens'])
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=1)+'\n')
    print(raw,report['ever_boilerplate'],report['final_four_match'],flush=True)


if __name__=='__main__':
    main()
