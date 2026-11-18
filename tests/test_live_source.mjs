import {strict as assert} from 'node:assert';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const wire = await readFile(new URL('../static/live-wire.js', import.meta.url),'utf8');
const wireURL = `data:text/javascript;base64,${Buffer.from(wire).toString('base64')}`;
const source = (await readFile(new URL('../static/live-source.js',import.meta.url),'utf8')).replace("'./live-wire.js'",JSON.stringify(wireURL));
const {InputDiagnostics,SessionRecording,prepareAudioSource} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const workletCode = await readFile(new URL('../static/pcm-worklet.js',import.meta.url),'utf8');

test('session replay is independent PCM16, limited to configured duration, and releasable',async()=>{
  const recording = new SessionRecording(16000,1), block = new Float32Array(12000).fill(.5);
  assert.equal(recording.append(block).length,12000); block.fill(-1);
  assert.equal(recording.append(new Float32Array(12000).fill(-.5)).length,4000);
  assert.equal(recording.append(new Float32Array(100)).length,0);
  assert.equal(recording.total,16000);
  const blob = recording.blob(), data = await blob.arrayBuffer(), view = new DataView(data);
  assert.equal(blob.type,'audio/wav'); assert.equal(data.byteLength,32044);
  assert.equal(new TextDecoder().decode(data.slice(0,4)),'RIFF');
  assert.equal(view.getUint32(4,true),32036); assert.equal(view.getUint32(24,true),16000);
  assert.equal(view.getUint32(28,true),32000); assert.equal(view.getUint32(40,true),32000);
  assert.equal(view.getInt16(44,true),16383); assert.equal(view.getInt16(44+12000*2,true),-16384);
  recording.clear(); assert.equal(recording.total,0); assert.equal(recording.chunks.length,0);
  assert.equal(blob.size,32044); // existing replay owns its bytes
});

for (const rate of [16000,44100,48000]) test(`worklet at ${rate} Hz emits ~128ms mono batches then flushes the exact tail`,()=>{
  const posts = []; let Processor;
  const context = vm.createContext({Float32Array,sampleRate:rate,
    AudioWorkletProcessor:class {constructor(){this.port={postMessage:data=>posts.push(data)};}},
    registerProcessor:(name,klass)=>{assert.equal(name,'pcm-capture'); Processor=klass;}});
  vm.runInContext(workletCode,context);
  const processor = new Processor();
  assert.ok(Math.abs(processor.blockSize/rate-.128) <= 128/rate);
  const input = [new Float32Array(128).fill(.75),new Float32Array(128).fill(-.25)];
  const blocks = processor.blockSize/128;
  for (let i=0;i<blocks;i++) assert.equal(processor.process([input]),true);
  assert.equal(posts.length,1);
  const batch = new Float32Array(posts[0]);
  assert.equal(batch.length,processor.blockSize); assert.ok(batch.every(x=>x===.25));
  processor.process([[new Float32Array(17).fill(.125)]]);
  processor.port.onmessage({data:'flush'});
  assert.equal(new Float32Array(posts[1]).length,17); assert.equal(posts[2].flushed,true);
  processor.port.onmessage({data:'flush'});
  assert.equal(posts.length,4); assert.equal(posts[3].flushed,true); // no duplicate tail
});

function installAudioMocks(t,{tail = new Float32Array(0), getUserMedia, addModule = async()=>{}, flush = true} = {}) {
  const originals = new Map(['window','AudioContext','AudioWorkletNode','navigator'].map(k=>[k,Object.getOwnPropertyDescriptor(globalThis,k)]));
  t.after(()=>{for(const [key,descriptor] of originals) {if(descriptor) Object.defineProperty(globalThis,key,descriptor); else delete globalThis[key];}});
  let worklet, context;
  const simpleNode = ()=>({connect(){},disconnect(){}});
  class FakeWorklet {
    constructor() {
      worklet = this;
      this.port = {onmessage:null,postMessage:message=>{
        assert.equal(message,'flush');
        if (!flush) return;
        queueMicrotask(()=>{
          if(tail.length) this.port.onmessage?.({data:tail.buffer});
          this.port.onmessage?.({data:{flushed:true}});
        });
      }};
    }
    connect() {} disconnect() {}
  }
  class FakeContext {
    constructor() { context=this; this.sampleRate=16000; this.state='running'; this.destination={}; this.audioWorklet={addModule}; }
    async resume() {} async close(){this.state='closed';}
    createAnalyser(){return simpleNode();}
    createGain(){return {...simpleNode(),gain:{value:1}};}
    createBuffer(){return {copyToChannel(){}};}
    createBufferSource(){return {...simpleNode(),start(){},stop(){}};}
    createMediaStreamSource(){return simpleNode();}
  }
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{mediaDevices:{getUserMedia}}});
  globalThis.AudioContext=FakeContext; globalThis.AudioWorkletNode=FakeWorklet;
  globalThis.window={AudioWorkletNode:FakeWorklet};
  return {get worklet(){return worklet;},get context(){return context;}};
}

test('stop flushes remaining worklet samples before producing replay and disconnecting',async t=>{
  const audio = installAudioMocks(t,{tail:new Float32Array(17).fill(.25)}), controller = new AbortController(), received=[];
  const source = await prepareAudioSource({fileAudio:{samples:new Float32Array(4096),rate:16000},signal:controller.signal,
    onSamples:chunk=>received.push(chunk.slice()),onEnded:()=>{}});
  source.start();
  audio.worklet.port.onmessage({data:new Float32Array(2048).fill(.5).buffer});
  const stop = source.stop(); assert.equal(source.stop(),stop);
  const replay = await stop;
  assert.deepEqual(received.map(x=>x.length),[2048,17]);
  assert.equal(replay.duration,2065/16000); assert.equal(replay.blob.size,44+2065*2);
  assert.equal(audio.context.state,'closed'); assert.equal(audio.worklet.port.onmessage,null);
});

test('discard closes immediately without flush or retained replay',async t=>{
  const audio = installAudioMocks(t,{tail:new Float32Array(17)}),controller=new AbortController();
  const source = await prepareAudioSource({fileAudio:{samples:new Float32Array(4096),rate:16000},signal:controller.signal,onSamples:()=>{},onEnded:()=>{}});
  source.start(); const replay=await source.stop({discard:true});
  assert.equal(replay.blob,null); assert.equal(audio.context.state,'closed');
});

test('cancelling microphone permission closes a late granted stream',async t=>{
  let grant,stopped=0;
  installAudioMocks(t,{getUserMedia:()=>new Promise(resolve=>{grant=resolve;})});
  const controller=new AbortController();
  const pending=prepareAudioSource({signal:controller.signal,onSamples:()=>{},onEnded:()=>{}});
  const rejected=assert.rejects(pending,{name:'AbortError'});
  controller.abort(); await rejected;
  grant({getTracks:()=>[{stop:()=>{stopped++;}}]});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(stopped,1);
});

test('permission refusal propagates NotAllowedError and leaves file capture usable',async t=>{
  const denied=new DOMException('Permission denied','NotAllowedError');
  let requests=0,samples=0;
  const audio=installAudioMocks(t,{getUserMedia:async()=>{requests++; throw denied;}});
  const controller=new AbortController();
  await assert.rejects(prepareAudioSource({signal:controller.signal,onSamples:()=>samples++,onEnded:()=>{}}),error=>error===denied);
  assert.equal(audio.context,undefined); assert.equal(audio.worklet,undefined);
  assert.equal(samples,0); controller.abort(); // cleanup removed the aborted-operation listener
  const fallback=await prepareAudioSource({fileAudio:{samples:new Float32Array(2048),rate:16000},
    signal:new AbortController().signal,onSamples:()=>samples++,onEnded:()=>{}});
  fallback.start(); await fallback.stop({discard:true});
  assert.equal(requests,1); assert.equal(audio.context.state,'closed');
});

test('worklet setup failure releases a granted microphone and closes its AudioContext',async t=>{
  const failure=new Error('worklet module unavailable');
  let stopped=0,samples=0,failModule=true;
  const track={stop:()=>stopped++,addEventListener(){}};
  const audio=installAudioMocks(t,{getUserMedia:async()=>({getTracks:()=>[track]}),
    addModule:async()=>{if(failModule) throw failure;}});
  const controller=new AbortController();
  await assert.rejects(prepareAudioSource({signal:controller.signal,onSamples:()=>samples++,onEnded:()=>{}}),error=>error===failure);
  assert.equal(stopped,1); assert.equal(audio.context.state,'closed');
  assert.equal(audio.worklet,undefined); assert.equal(samples,0);
  controller.abort(); assert.equal(stopped,1); // no leftover handler re-closes the stream
  failModule=false;
  const fallback=await prepareAudioSource({fileAudio:{samples:new Float32Array(2048),rate:16000},
    signal:new AbortController().signal,onSamples:()=>samples++,onEnded:()=>{}});
  fallback.start(); await fallback.stop({discard:true});
  assert.equal(audio.context.state,'closed'); assert.equal(stopped,1);
});

test('default session cap is thirty minutes while replay remains PCM16 chunks',()=>{
  const recording=new SessionRecording(48000);
  assert.equal(recording.limit,48000*1800);
  recording.append(new Float32Array(128));
  assert.equal(recording.chunks[0].byteLength,128*2);
  recording.clear(); assert.equal(recording.chunks.length,0);
});


test('a missing worklet flush fails explicitly and still releases audio resources',async t=>{
  const audio=installAudioMocks(t,{flush:false});
  const source=await prepareAudioSource({fileAudio:{samples:new Float32Array(4096),rate:16000},signal:new AbortController().signal,onSamples:()=>{},onEnded:()=>{}});
  source.start();
  await assert.rejects(source.stop(),/آخرین بخش صدا/);
  assert.equal(audio.context.state,'closed'); assert.equal(audio.worklet.port.onmessage,null);
});

test('input diagnostics keep only bounded numeric aggregates without changing the samples',()=>{
  const stats=new InputDiagnostics(16000,{sourceKind:'microphone',trackSampleRate:48000,channels:2});
  const block=new Float32Array([.5,-.5,1,-1]);
  stats.append(block);
  const first=stats.snapshot();
  assert.equal(first.rms,Math.sqrt(.625)); assert.equal(first.peak,1); assert.equal(first.clippedFraction,.5);
  assert.equal(first.sampleRate,16000); assert.equal(first.trackSampleRate,48000); assert.equal(first.channels,2);
  assert.deepEqual([...block],[.5,-.5,1,-1]);
  block.fill(0); assert.equal(stats.snapshot().peak,1); // no retained sample alias
  for(let i=0;i<40;i++) stats.append(new Float32Array(2048));
  const quiet=stats.snapshot();
  assert.equal(quiet.rms,0); assert.equal(quiet.peak,0); assert.equal(quiet.clippedFraction,0);
  assert.ok(stats.buckets.length<=9); assert.ok(stats.windowSamples<16000+2048);
  assert.ok(stats.buckets.every(bucket=>Object.values(bucket).every(value=>typeof value==='number')));
  stats.reset(); assert.equal(stats.snapshot().totalSamples,0); assert.equal(stats.buckets.length,0);
  assert.equal(new InputDiagnostics(16000).snapshot().peak,0);
});

test('microphone diagnostics read the already granted track once and preserve capture constraints',async t=>{
  let requests=0,constraints,reads=0;
  const track={stop(){},addEventListener(){},getSettings(){reads++;return {sampleRate:48000,channelCount:2};}};
  const audio=installAudioMocks(t,{getUserMedia:async value=>{requests++;constraints=value;return {getTracks:()=>[track],getAudioTracks:()=>[track]};}});
  const received=[],capture=await prepareAudioSource({signal:new AbortController().signal,onSamples:block=>received.push([...block]),onEnded:()=>{}});
  capture.start();
  const callback=audio.worklet.port.onmessage;
  callback({data:new Float32Array([.25,-.25]).buffer});
  assert.equal(requests,1); assert.equal(reads,1);
  assert.deepEqual(constraints,{audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
  assert.equal(capture.diagnostics.sourceKind,'microphone'); assert.equal(capture.diagnostics.sampleRate,16000);
  assert.equal(capture.diagnostics.trackSampleRate,48000); assert.equal(capture.diagnostics.channels,2);
  assert.equal(capture.diagnostics.rms,.25); assert.deepEqual(received,[[.25,-.25]]);
  await capture.stop({discard:true});
  callback({data:new Float32Array([1]).buffer}); // a late worklet delivery is ignored
  assert.equal(capture.diagnostics.totalSamples,0); assert.equal(received.length,1);
  assert.equal(requests,1); assert.equal(reads,1);
});
test('partial close can recover captured sound even if the last worklet flush is lost', async t=>{
  const audio=installAudioMocks(t,{flush:false});
  const source=await prepareAudioSource({fileAudio:{samples:new Float32Array(4096),rate:16000},signal:new AbortController().signal,onSamples:()=>{},onEnded:()=>{}});
  source.start();
  audio.worklet.port.onmessage({data:new Float32Array(2048).fill(.25).buffer});
  const replay=await source.stop({partial:true});
  assert.equal(replay.duration,2048/16000);
  assert.equal(replay.blob.size,44+2048*2);
  assert.equal(audio.context.state,'closed');
});
