// Transport unit tests. No fake model output is used to validate recognition.
import {strict as assert} from 'node:assert';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../static/live-wire.js', import.meta.url), 'utf8');
const {LiveConnection, pcmPacket} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
globalThis.window = {location:{href:'http://127.0.0.1:5000/live'}};

class FakeSocket {
  constructor() { this.readyState = 0; this.bufferedAmount = 0; this.sent = []; }
  send(data) { this.sent.push(data); }
  open() { this.readyState = 1; this.onopen(); }
  event(data) { this.onmessage({data:JSON.stringify(data)}); }
  close() { this.readyState = 3; this.onclose?.(); }
}

async function connected(t) {
  const events = [], errors = [], socket = new FakeSocket();
  let url;
  const connection = new LiveConnection({onEvent:e=>events.push(e),onError:e=>errors.push(e),socketFactory:u=>{url=u; return socket;}});
  const controller = new AbortController();
  const ready = connection.connect({sampleRate:16000,startAyah:5},controller.signal);
  socket.open(); socket.event({type:'ready',session_id:'current',version:1});
  await ready;
  t.after(()=>connection.close());
  return {connection,socket,events,errors,controller,url};
}

test('binary packet carries little-endian sequence, exact sample offset, and clipped PCM16',()=>{
  const data = pcmPacket(new Float32Array([-2,-.5,0,.5,2]),9,44100), view = new DataView(data);
  assert.equal(data.byteLength,18);
  assert.equal(view.getUint32(0,true),9); assert.equal(view.getUint32(4,true),44100);
  assert.deepEqual([8,10,12,14,16].map(i=>view.getInt16(i,true)),[-32768,-16384,0,16383,32767]);
});

test('start precedes ordered PCM and stop contains final sequence/sample totals',async t=>{
  const {connection,socket,url} = await connected(t);
  assert.equal(url,'ws://127.0.0.1:5000/live/ws');
  assert.deepEqual(JSON.parse(socket.sent[0]),{type:'start',version:1,start_ayah:5,sample_rate:16000,channels:1});
  connection.sendSamples(new Float32Array(2048)); connection.sendSamples(new Float32Array(17));
  const first = new DataView(socket.sent[1]), second = new DataView(socket.sent[2]);
  assert.deepEqual([first.getUint32(0,true),first.getUint32(4,true),second.getUint32(0,true),second.getUint32(4,true)],[0,0,1,2048]);
  const done = connection.finish();
  assert.deepEqual(JSON.parse(socket.sent[3]),{type:'stop',last_sequence:1,total_samples:2065});
  assert.equal(connection.finish(),done); // repeated stop never sends another message
  socket.event({type:'done',session_id:'current',revision:1});
  await done;
});

test('ACK frees backlog while over one second fails without sending or dropping a chunk',async t=>{
  const {connection,socket,errors} = await connected(t);
  connection.sendSamples(new Float32Array(16000));
  socket.event({type:'ack',session_id:'current',sequence:0,input_samples:16000});
  connection.sendSamples(new Float32Array(15000));
  assert.equal(connection.total,31000);
  const count = socket.sent.length;
  assert.throws(()=>connection.sendSamples(new Float32Array(1001)));
  assert.equal(socket.sent.length,count); assert.equal(connection.total,31000);
  assert.equal(socket.readyState,3); assert.equal(errors.length,1);
});

test('socket byte backlog also stops capture even when ACK totals look current',async t=>{
  const {connection,socket,errors} = await connected(t);
  socket.bufferedAmount = 32001;
  assert.throws(()=>connection.sendSamples(new Float32Array(128)));
  assert.equal(connection.total,0); assert.equal(errors.length,1);
});

test('stale session or older revision cannot revise current result; progress/result share revision',async t=>{
  const {socket,events,connection} = await connected(t);
  socket.event({type:'progress',session_id:'current',revision:2});
  socket.event({type:'result',session_id:'current',revision:2});
  const count = events.length;
  socket.event({type:'result',session_id:'other',revision:100});
  socket.event({type:'result',session_id:'current',revision:1});
  assert.equal(events.length,count); assert.equal(connection.revision,2);
});

test('foreign fatal error and duplicate ready cannot replace or terminate established session',async t=>{
  const {socket,connection,events,errors} = await connected(t);
  const count = events.length;
  socket.event({type:'error',session_id:'stale',fatal:true,code:'processing',message:'old failure'});
  socket.event({type:'ready',session_id:'stale',version:1});
  assert.equal(connection.failed,false); assert.equal(connection.session,'current');
  assert.equal(errors.length,0); assert.equal(events.length,count);
});

test('impossible future ACK sequence is a protocol failure',async t=>{
  const {connection,socket,errors} = await connected(t);
  connection.sendSamples(new Float32Array(2048));
  socket.event({type:'ack',session_id:'current',sequence:9,input_samples:2048});
  assert.equal(connection.failed,true); assert.equal(errors.length,1);
});

test('ACK must identify its exact sent sample endpoint, not a plausible cumulative count',async t=>{
  const {connection,socket,errors} = await connected(t);
  connection.sendSamples(new Float32Array(2048)); connection.sendSamples(new Float32Array(17));
  socket.event({type:'ack',session_id:'current',sequence:0,input_samples:2065});
  assert.equal(connection.failed,true); assert.equal(errors.length,1);
});

test('stop creates a hard boundary and sends no empty or late PCM chunks',async t=>{
  const {connection,socket} = await connected(t);
  connection.sendSamples(new Float32Array(0));
  assert.equal(socket.sent.length,1); assert.equal(connection.sequence,0);
  const done = connection.finish();
  assert.deepEqual(JSON.parse(socket.sent[1]),{type:'stop',last_sequence:-1,total_samples:0});
  assert.throws(()=>connection.sendSamples(new Float32Array(17)));
  assert.equal(socket.sent.length,2); assert.equal(connection.total,0);
  socket.event({type:'done',session_id:'current',revision:0}); await done;
});

test('after intentional close late result, ready, or error cannot reach UI callbacks',async t=>{
  const {socket,connection,events,errors} = await connected(t);
  connection.close(); const count = events.length;
  socket.event({type:'result',session_id:'current',revision:3});
  socket.event({type:'ready',session_id:'late',version:1});
  socket.event({type:'error',session_id:'current',fatal:false,message:'late'});
  assert.equal(events.length,count); assert.equal(errors.length,0);
});

test('unexpected close rejects a pending final review exactly once',async t=>{
  const {connection,socket,errors} = await connected(t);
  const done = connection.finish();
  const rejection = assert.rejects(done);
  socket.close(); await rejection;
  assert.equal(errors.length,1); assert.equal(connection.failed,true);
});

test('abort during handshake rejects promptly and intentional cancellation stays silent',async()=>{
  const socket = new FakeSocket(), errors = [], controller = new AbortController();
  const connection = new LiveConnection({onEvent:()=>{},onError:e=>errors.push(e),socketFactory:()=>socket});
  const ready = connection.connect({sampleRate:16000,startAyah:1},controller.signal);
  const rejected = assert.rejects(ready,{name:'AbortError'});
  controller.abort(); await rejected;
  assert.equal(socket.readyState,3); assert.deepEqual(errors,[]);
});
