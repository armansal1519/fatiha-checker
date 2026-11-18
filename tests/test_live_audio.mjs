// Development-only check with Node's built-in runner; the app requires no Node.
import {strict as assert} from 'node:assert';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
const code = await readFile(new URL('../static/live-audio.js', import.meta.url), 'utf8');
const {AudioWindow} = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

test('rolling window is bounded and each snapshot is an independent PCM WAV', async () => {
  const ring = new AudioWindow(10);
  for (let i = 0; i < 120; i++) ring.push(new Float32Array(1600).fill(i / 120), 16000);
  assert.equal(ring.total, 192000); assert.equal(ring.length, 160000);
  const data = await ring.wav().arrayBuffer(), view = new DataView(data);
  assert.equal(new TextDecoder().decode(data.slice(0,4)), 'RIFF');
  assert.equal(view.getUint32(24,true), 16000);
  assert.equal(view.getUint32(40,true), 320000);
  assert.equal(view.getInt16(44,true), Math.trunc(20 / 120 * 32767));
  ring.reset(); assert.equal(ring.length,0); assert.equal(ring.total,0);
});

test('48 kHz capture preserves sample rate and clips PCM safely', async () => {
  const ring = new AudioWindow(); ring.push(new Float32Array([-2,0,2]), 48000);
  const view = new DataView(await ring.wav().arrayBuffer());
  assert.equal(view.getUint32(24,true),48000);
  assert.deepEqual([44,46,48].map((i)=>view.getInt16(i,true)),[-32768,0,32767]);
});

test('short live snapshots keep recent speech; final review retains ten seconds', async () => {
  const ring = new AudioWindow(10);
  for (let i = 0; i < 12; i++) ring.push(new Float32Array(16000).fill(i / 20), 16000);
  // Retained [2,12]; choose [5,11] to omit one second of trailing quiet.
  const live = new DataView(await ring.wav({seconds:6,endTotal:11*16000}).arrayBuffer());
  assert.equal(live.getUint32(40,true),6*16000*2);
  assert.equal(live.getInt16(44,true),Math.trunc(5/20*32767));
  assert.equal(live.getInt16(live.byteLength-2,true),Math.trunc(10/20*32767));
  const final = new DataView(await ring.wav().arrayBuffer());
  assert.equal(final.getUint32(40,true),10*16000*2);
  assert.equal(ring.length,10*16000); // Snapshot selection is non-destructive.
});

test('snapshot bounds work across partial blocks and clamp to retained audio', async () => {
  const ring = new AudioWindow(1);
  ring.push(new Float32Array([.1,.2,.3]),4);
  ring.push(new Float32Array([.4,.5,.6]),4);
  const view = new DataView(await ring.wav({seconds:.5,endTotal:5}).arrayBuffer());
  assert.equal(view.getUint32(40,true),4);
  assert.deepEqual([44,46].map(i=>view.getInt16(i,true)),[Math.trunc(Math.fround(.4)*32767),Math.trunc(.5*32767)]);
  const empty = new DataView(await ring.wav({endTotal:0}).arrayBuffer());
  assert.equal(empty.getUint32(40,true),0);
});

test('live file limit is100MiB and duration limit is thirty minutes',async t=>{
  const {readAudioFile}=await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
  const old=globalThis.OfflineAudioContext; t.after(()=>{if(old) globalThis.OfflineAudioContext=old; else delete globalThis.OfflineAudioContext;});
  let duration=1800;
  globalThis.OfflineAudioContext=class {async decodeAudioData(){return {duration,length:16,numberOfChannels:1,sampleRate:16000,getChannelData:()=>new Float32Array(16)};}};
  const file={size:100*1024*1024,arrayBuffer:async()=>new ArrayBuffer(1)};
  assert.equal((await readAudioFile(file)).samples.length,16);
  await assert.rejects(readAudioFile({...file,size:file.size+1}));
  duration=1800.1; await assert.rejects(readAudioFile(file));
});
