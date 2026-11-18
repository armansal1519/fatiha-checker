import {strict as assert} from 'node:assert';
import {test} from 'node:test';
import {segmentBounds,unreviewedRecording} from '../static/review-audio.js';

test('word playback includes neighbouring context and is bounded by available sound',()=>{
  assert.deepEqual(segmentBounds([{start_ms:100,end_ms:1200}],1.45),{start:0,end:1.45});
  assert.deepEqual(segmentBounds([{start_ms:1500,end_ms:1600}],4),{start:1.2,end:1.9500000000000002});
});
test('gap playback spans both observed anchors and never invents time for absent words',()=>{
  const left={start_ms:800,end_ms:1200},right={start_ms:1500,end_ms:2000};
  assert.deepEqual(segmentBounds([left,right],4),{start:.5,end:2.35});
  for (const item of [null,{start_ms:-2,end_ms:100},{start_ms:500,end_ms:23},{start_ms:0,end_ms:NaN}]) assert.equal(segmentBounds([left,item],4),null);
  assert.equal(segmentBounds([right],0),null);
});

test('retrying a partial recording keeps a short context and an accurate local PCM header',async()=>{
  const header=new ArrayBuffer(44), fields=new DataView(header);
  fields.setUint16(22,1,true); fields.setUint32(24,16000,true); fields.setUint16(34,16,true);
  const samples=Int16Array.from({length:32000},(_,i)=>i%32767);
  const blob=new Blob([header,samples],{type:'audio/wav'});
  const tail=await unreviewedRecording(blob,1.5);
  const wave=new DataView(await tail.arrayBuffer());
  assert.equal(tail.size,44+2*(32000-12000));
  assert.equal(wave.getUint32(40,true),tail.size-44);
  assert.equal(wave.getInt16(44,true),samples[12000]);
  assert.equal((await unreviewedRecording(blob,0)).size,blob.size);
  await assert.rejects(unreviewedRecording(blob,8));
});
