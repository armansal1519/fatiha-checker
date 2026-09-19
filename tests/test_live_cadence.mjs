import {strict as assert} from 'node:assert';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
const code = await readFile(new URL('../static/live-cadence.js', import.meta.url),'utf8');
const {LiveCadence} = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const speaking = {pending:false,voiceSample:16000,sentVoice:-1,lastVoice:800,pauseSent:false};

test('first update after 800ms, never queues during a slow inference', () => {
  const clock = new LiveCadence(); clock.reset(0);
  assert.equal(clock.due(700,speaking),null);
  assert.equal(clock.due(800,speaking),false);
  clock.sent(800);
  assert.equal(clock.due(3000,{...speaking,pending:true,lastVoice:3000}),null);
  clock.finished(3000);
  assert.equal(clock.due(3050,{...speaking,lastVoice:3000}),null);
  assert.equal(clock.due(3100,{...speaking,lastVoice:3000}),false);
});

test('pause review does not wait for periodic timer or repeatedly process silence', () => {
  const clock = new LiveCadence(); clock.sent(1000); clock.finished(1250);
  assert.equal(clock.due(1500,{...speaking,lastVoice:1000}),true);
  assert.equal(clock.due(2000,{...speaking,lastVoice:1000,pauseSent:true}),null);
  assert.equal(clock.due(4000,{...speaking,voiceSample:0}),null);
  assert.equal(clock.due(4000,{...speaking,lastVoice:4000,sentVoice:16000}),null);
});
