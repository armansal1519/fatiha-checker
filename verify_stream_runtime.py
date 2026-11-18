"""Paced v2 transport and real-model verification on the local WebSocket.

The default 30-minute run creates audio in small blocks, retains no recording,
checks both real-model readiness and exact stream coverage, and samples optional
server RSS through the operating system. Pass --recording to decode a real
recording incrementally and require successful ASR and phonetic stage calls.
Only the recording mode provides evidence about speech throughput.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import file_digest
import json
import math
from pathlib import Path
from statistics import median
import struct
import subprocess
from threading import Event, Lock, Thread
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np
import requests
from simple_websocket import Client, ConnectionClosed

RATE = 16000
BLOCK = 2048
MAX_SECONDS = 1800


class LocalClient(Client):
    def handshake(self):
        self.host = f'[{self.host}]:{self.port}' if ':' in self.host else f'{self.host}:{self.port}'
        return super().handshake()


def local_base_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1','localhost','::1'}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {'','/'}):
        raise ValueError('Use an HTTP loopback base URL, for example http://127.0.0.1:5001')
    return value.rstrip('/')


def percentile_summary(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return dict(count=0, median=None, p95=None, minimum=None, maximum=None)
    return dict(count=len(values), median=round(median(values),3),
                p95=round(values[max(0,math.ceil(.95*len(values))-1)],3),
                minimum=round(values[0],3),maximum=round(values[-1],3))


def model_stage_calls(timings):
    """Count executed stages, never infer a model call from elapsed time alone."""
    result = {}
    unknown = {}
    for stage,flag in [('asr','transcription_skipped'),('phonetic','phonetic_skipped')]:
        result[stage] = sum(item.get(flag) is False for item in timings)
        unknown[stage] = sum(type(item.get(flag)) is not bool for item in timings)
    frames = [item.get('vad_frames') for item in timings]
    known = [count for count in frames if type(count) is int and count >= 0]
    result['vad'] = sum(count > 0 for count in known)
    result['vad_frame_evaluations'] = sum(known)
    unknown['vad'] = len(frames)-len(known)
    result['unknown'] = unknown
    return result


def check_health(base_url):
    response = requests.get(local_base_url(base_url)+'/health',timeout=15,allow_redirects=False)
    if response.status_code != 200:
        raise RuntimeError(f'The two-model local server is not ready (HTTP {response.status_code})')
    health = response.json()
    if health.get('protocol_version') != 2 or health.get('ready') is not True:
        raise RuntimeError('The local server must report ready=true and protocol_version=2')
    models = health.get('models',{})
    expected = {'recitation':'obadx/muaalem-model-v3_2','transcription':'openai/whisper-large-v3-turbo'}
    for key,model_id in expected.items():
        if models.get(key,{}).get('ready') is not True or models[key].get('model') != model_id:
            raise RuntimeError(f'Real {key} model readiness was not confirmed by /health')
    return health


def pcm_frame(sequence, offset, audio):
    pcm = (np.clip(np.asarray(audio,dtype=np.float32),-1,1)*32767).astype('<i2').tobytes()
    return struct.pack('<II',sequence,offset)+pcm


class StreamAccumulator:
    """Apply exactly the same raw-text deltas and token-ID feedback as the UI."""
    def __init__(self, ready):
        if ready.get('type') != 'ready' or ready.get('version') != 2 or not ready.get('session_id'):
            raise ValueError(f'Invalid v2 ready event: {ready.get("type")}')
        self.session_id = ready['session_id']
        self.revision = ready.get('revision',0)
        self.tokens = []
        self.committed_count = 0
        self.gaps = []
        self.reference = None
        self.done = None
        self.first_transcript_seconds = self.first_feedback_seconds = None
        self.first_reviewed_seconds = None
        self.reference_changes = []
        self.transcript_events = self.feedback_events = 0
        self.timing_events = []

    def accept(self,event,elapsed):
        if event.get('session_id') != self.session_id:
            raise AssertionError('A foreign session event reached the live stream')
        if event.get('type') == 'ack':
            return
        revision = event.get('revision')
        if type(revision) is not int or revision <= self.revision:
            raise AssertionError('Non-ACK revisions must increase strictly')
        self.revision = revision
        kind = event.get('type')
        if kind == 'transcript':
            start = event.get('replace_from')
            incoming = event.get('tokens')
            if type(start) is not int or not self.committed_count <= start <= len(self.tokens) or not isinstance(incoming,list):
                raise AssertionError('Invalid mutable-tail replacement')
            self.tokens[start:] = [dict(token) for token in incoming]
            ids = [token.get('id') for token in self.tokens]
            if any(not isinstance(token_id,str) for token_id in ids) or len(ids) != len(set(ids)):
                raise AssertionError('Transcript token IDs must be unique strings')
            count = event.get('committed_count')
            if type(count) is not int or not self.committed_count <= count <= len(self.tokens):
                raise AssertionError('Invalid committed token count')
            self.committed_count = count
            self.transcript_events += 1
            if incoming and self.first_transcript_seconds is None:
                self.first_transcript_seconds = elapsed
        elif kind == 'feedback':
            by_id = {token['id']:token for token in self.tokens}
            for update in event.get('updates',[]):
                token_id = update.get('id')
                if token_id not in by_id:
                    raise AssertionError('Feedback refers to a removed or unknown raw token')
                if 'text' in update and update['text'] != by_id[token_id]['text']:
                    raise AssertionError('Feedback must never replace a raw spoken token')
                by_id[token_id].update({k:v for k,v in update.items() if k != 'text'})
            self.gaps = event.get('gaps',[])
            self.reference = event.get('reference')
            self.feedback_events += 1
            if event.get('updates') and self.first_feedback_seconds is None:
                self.first_feedback_seconds = elapsed
            if self.first_reviewed_seconds is None and any(
                    update.get('state') in {'match', 'difference'} for update in event.get('updates', [])):
                self.first_reviewed_seconds = elapsed
            place = (self.reference['surah'], self.reference['ayah']) if self.reference else None
            previous = self.reference_changes[-1]['place'] if self.reference_changes else None
            if place != previous:
                self.reference_changes.append(dict(place=place, seconds=elapsed))
        elif kind == 'done':
            self.done = event
            self.reference = event.get('reference',self.reference)
        if event.get('timings'):
            self.timing_events.append(dict(event['timings']))


def process_rss_kib(pid):
    if pid is None:
        return None
    try:
        result = subprocess.run(['ps','-o','rss=','-p',str(pid)],capture_output=True,text=True,timeout=2,check=True)
        return int(result.stdout.strip())
    except (OSError,ValueError,subprocess.SubprocessError):
        return None


class StreamRunError(RuntimeError):
    def __init__(self,message,report):
        super().__init__(message)
        self.report=report


def stream_v2(base_url, source, total_samples, *, sample_rate=RATE, start_surah=None,
              start_ayah=None, server_pid=None, on_progress=None, on_event=None):
    """Send source(start,end) at real capture speed and collect final raw text.

    This is recording replay timing. It does not measure a microphone, physical
    capture latency or browser painting. No model is loaded by this client.
    """
    base_url=local_base_url(base_url)
    if type(total_samples) is not int or not 0 < total_samples <= MAX_SECONDS*sample_rate:
        raise ValueError('Audio must have a positive duration of at most 30 minutes')
    if sample_rate not in {16000,44100,48000}:
        raise ValueError('Unsupported PCM input rate')
    url=base_url.replace('http://','ws://',1)+'/live/ws'
    client=LocalClient.connect(url,headers={'Origin':base_url},max_message_size=8*1024*1024)
    finished=Event()
    mutex=Lock()
    pending={}
    ack_latencies=[]
    send_lateness=[]
    ack_sequence=-1
    ack_samples=0
    errors=[]
    notices=[]
    memory=[]
    accumulator=None
    receiver=None
    sequence=offset=0
    began=perf_counter()
    try:
        client.send(json.dumps(dict(type='start',version=2,start_surah=start_surah,start_ayah=start_ayah,
                                    sample_rate=sample_rate,channels=1)))
        raw=client.receive(timeout=15)
        if raw is None:
            raise TimeoutError('No ready event from the local server')
        ready=json.loads(raw)
        if ready.get('type') == 'error':
            raise RuntimeError(ready.get('message','Session rejected'))
        accumulator=StreamAccumulator(ready)
        began=perf_counter()

        def reader():
            nonlocal ack_sequence,ack_samples
            try:
                while not finished.is_set():
                    raw=client.receive(timeout=1)
                    if raw is None:
                        continue
                    if not isinstance(raw,str):
                        raise AssertionError('Server events must be JSON text')
                    event=json.loads(raw)
                    now=perf_counter()
                    accumulator.accept(event,round(now-began,4))
                    if on_event is not None and event.get('type') != 'ack':
                        on_event(event, accumulator, now-began)
                    if event.get('type') == 'ack':
                        with mutex:
                            sent=pending.pop(event.get('sequence'),None)
                            if (event.get('sequence') != ack_sequence+1 or sent is None
                                    or event.get('input_samples') != sent[1]):
                                raise AssertionError('ACK order or sample endpoint does not match sent PCM')
                            ack_sequence=event['sequence']
                            ack_samples=event['input_samples']
                            ack_latencies.append((now-sent[0])*1000)
                    elif event.get('type') == 'error':
                        if event.get('fatal',True):
                            errors.append(dict(code=event.get('code'),message=event.get('message')))
                            finished.set()
                        else:
                            notices.append(dict(code=event.get('code'),message=event.get('message')))
                    elif event.get('type') == 'done':
                        finished.set()
            except Exception as exc:
                if not finished.is_set():
                    errors.append(dict(type=type(exc).__name__,message=str(exc)))
                    finished.set()

        receiver=Thread(target=reader,daemon=True,name='v2-verifier-receiver')
        receiver.start()
        next_rss=next_progress=0.0
        block=round(.128*sample_rate)
        while offset<total_samples and not finished.is_set():
            end=min(total_samples,offset+block)
            deadline=began+end/sample_rate
            if finished.wait(max(0,deadline-perf_counter())):
                break
            now=perf_counter()
            samples=np.asarray(source(offset,end),dtype=np.float32).reshape(-1)
            if len(samples) != end-offset or not np.isfinite(samples).all():
                raise ValueError('Audio source returned an invalid block')
            with mutex:
                if end-ack_samples > sample_rate:
                    raise RuntimeError('Unacknowledged audio exceeded one second; stream would drop continuity')
                pending[sequence]=(now,end)
            client.send(pcm_frame(sequence,offset,samples))
            send_lateness.append(max(0,(now-deadline)*1000))
            sequence+=1
            offset=end
            elapsed=now-began
            if elapsed>=next_rss:
                rss=process_rss_kib(server_pid)
                if server_pid is not None:
                    memory.append(dict(elapsed_seconds=round(elapsed,3),rss_kib=rss))
                next_rss=elapsed+5
            if on_progress is not None and elapsed>=next_progress:
                on_progress(dict(elapsed_seconds=round(elapsed,1),samples_sent=offset,
                                 acknowledged_samples=ack_samples,transcript_events=accumulator.transcript_events,
                                 feedback_events=accumulator.feedback_events,
                                 rss_kib=memory[-1]['rss_kib'] if memory else None))
                next_progress=elapsed+30
        if not finished.is_set():
            client.send(json.dumps(dict(type='stop',last_sequence=sequence-1,total_samples=offset)))
        if not finished.wait(120):
            raise TimeoutError('Final transcript/feedback did not drain within 120 seconds')
        receiver.join(timeout=2)
        done=accumulator.done
        report=dict(status='passed' if not errors else 'failed', session_id=accumulator.session_id,
                    input_duration_seconds=round(total_samples/sample_rate,4),sample_rate=sample_rate,
                    frames_sent=sequence,samples_sent=offset,acknowledged_samples=ack_samples,
                    wall_seconds=round(perf_counter()-began,4),
                    first_transcript_seconds=accumulator.first_transcript_seconds,
                    first_feedback_seconds=accumulator.first_feedback_seconds,
                    first_reviewed_seconds=accumulator.first_reviewed_seconds,
                    reference_changes=accumulator.reference_changes,
                    transcript_events=accumulator.transcript_events,feedback_events=accumulator.feedback_events,
                    final_tokens=accumulator.tokens,final_gaps=accumulator.gaps,reference=accumulator.reference,
                    ack_latency_ms=percentile_summary(ack_latencies),sender_lateness_ms=percentile_summary(send_lateness),
                    memory_samples=memory,notices=notices,errors=errors,done=done)
        if errors:
            raise StreamRunError('The local v2 stream reported an error',report)
        if done is None or offset != total_samples or ack_samples != total_samples:
            raise StreamRunError('The stream ended before all audio was acknowledged and finalized',report)
        metrics=done.get('metrics',{})
        expected_output=round(total_samples*RATE/sample_rate)
        if (not metrics.get('coverage_complete') or abs(metrics.get('received_samples',-1)-expected_output)>1
                or metrics.get('analyzed_samples') != metrics.get('received_samples')
                or metrics.get('queued_inference_jobs') != 0
                or metrics.get('inference_max_inflight',0)>1
                or metrics.get('ring_max_samples',0)>30*RATE
                or metrics.get('buffer_capacity_samples') != 30*RATE):
            raise StreamRunError('Coverage, queue or buffer invariants failed',report)
        intervals=metrics.get('coverage_intervals',[])
        if expected_output and (not intervals or intervals[0][0]!=0 or intervals[-1][1]!=metrics['received_samples']
                or any(left[1]!=right[0] for left,right in zip(intervals,intervals[1:]))):
            raise StreamRunError('Coverage intervals contain a hole',report)
        report['model_stage_calls'] = model_stage_calls(metrics.get('stage_timings',[]))
        return report
    except StreamRunError:
        raise
    except Exception as exc:
        report=dict(status='failed',samples_sent=offset,frames_sent=sequence,
                    wall_seconds=round(perf_counter()-began,4),memory_samples=memory,
                    errors=errors+[dict(type=type(exc).__name__,message=str(exc))])
        raise StreamRunError(str(exc),report) from exc
    finally:
        finished.set()
        try:
            client.close()
        except (ConnectionClosed,OSError):
            pass
        if receiver is not None:
            receiver.join(timeout=2)


def tone_bursts(start,end):
    times=np.arange(start,end,dtype=np.float64)/RATE
    return (.15*np.sin(2*np.pi*220*times)*(np.remainder(times,30)<2)).astype(np.float32)


class RecordingSource:
    """Decode a local test recording incrementally, preserving the input clock."""
    def __init__(self, path, sample_rate):
        self.path = path
        self.rate = sample_rate
        self.position = 0
        self.pending = np.empty(0, dtype=np.float32)
        self.frames = self._decode()

    def _decode(self):
        import av
        with self.path.open('rb') as source, av.open(source, options={'protocol_whitelist': ''}) as media:
            if not media.streams.audio:
                raise ValueError('The evaluation recording contains no audio stream')
            stream = media.streams.audio[0]
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=self.rate)
            for frame in media.decode(stream):
                for output in resampler.resample(frame):
                    yield output.to_ndarray().ravel()
            for output in resampler.resample(None):
                yield output.to_ndarray().ravel()

    def __call__(self, start, end):
        if start != self.position or end <= start:
            raise ValueError('Recording must be consumed sequentially')
        pieces = []
        remaining = end-start
        while remaining:
            if not self.pending.size:
                self.pending = next(self.frames, None)
                if self.pending is None:
                    raise ValueError('Recording ended before the requested duration')
            take = min(remaining, self.pending.size)
            pieces.append(self.pending[:take])
            self.pending = self.pending[take:]
            remaining -= take
        self.position = end
        return np.concatenate(pieces)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:5001')
    parser.add_argument('--duration',type=float,default=1800,help='Real wall-clock seconds, at most 1800; default: 1800.')
    parser.add_argument('--server-pid',type=int,help='Optional local server process for RSS samples.')
    parser.add_argument('--recording', type=Path, help='Local human recitation to replay once, without looping.')
    parser.add_argument('--sample-rate', type=int, choices=(16000,44100,48000), default=RATE)
    parser.add_argument('--output',type=Path,default=Path('reports/stream-runtime-v2.json'))
    args=parser.parse_args(argv)
    if not .5<=args.duration<=MAX_SECONDS:
        parser.error('--duration must be between 0.5 and 1800 seconds')
    if args.server_pid is not None and args.server_pid<=0:
        parser.error('--server-pid must be a positive process ID')
    report=dict(kind='synthetic-tone transport soak; not speech throughput or recognition accuracy',
                created_utc=datetime.now(UTC).isoformat(),duration_requested_seconds=args.duration,
                throughput_note='Synthetic tones and silence may be rejected by VAD before Whisper and Muaalem. '
                    'This run verifies transport coverage, ACKs and resource bounds. Use genuine-speech fixtures '
                    'with verify_recitation_cases.py to evaluate heavy-model throughput; this run does not establish it.',
                model_stage_calls_note='ASR and phonetic counts require explicit execution flags; missing legacy flags '
                    'are unknown. VAD counts windows that evaluated model frames, with frame evaluations reported separately.')
    try:
        source=tone_bursts
        if args.recording:
            source=RecordingSource(args.recording, args.sample_rate)
            with args.recording.open('rb') as f:
                digest=file_digest(f,'sha256').hexdigest()
            report.update(kind='Paced continuous human-recording WebSocket soak',
                          recording_sha256=digest, recording=args.recording.name,
                          throughput_note='Actual model-stage flags and transport coverage. No human accuracy labels assumed.')
        elif args.sample_rate!=RATE:
            parser.error('--sample-rate other than 16000 requires --recording')
        report['health']=check_health(args.base_url)
        result=stream_v2(args.base_url,source,round(args.duration*args.sample_rate),server_pid=args.server_pid,
                         sample_rate=args.sample_rate,
                         on_progress=lambda p:print(json.dumps(p),flush=True))
        # The soak publishes counts/metrics, not an invented recitation result.
        report.update({key:value for key,value in result.items() if key not in {'final_tokens','final_gaps'}})
        report['raw_token_count']=len(result['final_tokens'])
        timings=result['done']['metrics'].get('stage_timings',[])
        report['model_stage_calls']=model_stage_calls(timings)
        if args.recording and (not report['model_stage_calls']['asr'] or not report['model_stage_calls']['phonetic']):
            raise RuntimeError('Recording did not exercise both heavy models')
        if args.server_pid is not None and not any(s['rss_kib'] is not None for s in result['memory_samples']):
            raise RuntimeError('Server RSS could not be sampled; check --server-pid')
        samples=[s for s in result['memory_samples'] if s['rss_kib'] is not None]
        warmed=[s['rss_kib'] for s in samples if s['elapsed_seconds']>=min(120,args.duration/4)]
        report['rss_summary_kib']=percentile_summary([s['rss_kib'] for s in samples])
        report['rss_end_minus_warm_median_kib']=(samples[-1]['rss_kib']-median(warmed)) if warmed else None
        report['memory_note']='RSS observations are diagnostic; bounded audio/queue assertions do not prove absence of every memory leak.'
    except StreamRunError as exc:
        report.update({key:value for key,value in exc.report.items() if key not in {'final_tokens','final_gaps'}})
        report['status']='failed'
        report['failure']=str(exc)
    except Exception as exc:
        report.update(status='failed',failure=f'{type(exc).__name__}: {exc}')
    finally:
        if isinstance(source, RecordingSource):
            source.frames.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Runtime verification:',report['status'],'report:',args.output.resolve(),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
