// Exercise actual UI handlers plus the actual transcript reducer, without media.
import {strict as assert} from 'node:assert';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const live=await readFile(new URL('../static/live.js',import.meta.url),'utf8');
const stateSource=await readFile(new URL('../static/transcript-state.js',import.meta.url),'utf8');
const {TranscriptState}=await import(`data:text/javascript;base64,${Buffer.from(stateSource).toString('base64')}`);
const diagnosticFunctions=live.slice(live.indexOf('function status('),live.indexOf('function error('));
const acceptFunction=live.slice(live.indexOf('function accept('),live.indexOf('function drawWave('));
const resetFunction=live.slice(live.indexOf('function reset('),live.indexOf('function fail('));

function harness() {
  const nodes=new Map(),transcript=new TranscriptState();
  const $=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'',dataset:{},lastChild:{textContent:''},checked:true,value:'',hidden:false,removeAttribute(){},replaceChildren(){}});return nodes.get(id);};
  const context=vm.createContext({$,transcript,Map,Set,Math,Number,Infinity,
    fa:value=>String(value),cancelAnimationFrame(){},reportUI(){},discardCapture(){},
    clearReplay(){},error(){},markPaint(){},paint(){},summary(){},follow(){},drawWave(){},
    issueHistory:{record(){},clear(){}},resetIssue(){},
    updateReference:async()=>{},finish:async()=>{},fail:message=>{throw new Error(message);}});
  vm.runInContext(`
    let ready=true,generation=1, phase='recording', audioActivity=null, inputWarning='',lowInputSince=null,lastInputPaint=-Infinity,lastStreamStatus='';
    let renderedOrder=[],renderedCommitted=0,keyboardNode=null,lastReviewNode=null,reviewedThrough=0;
    let referenceFetch=0,hasDisplayLocation=false,connection=null,controller=null,capture=null,raf=null,chosenFile=null;
    let selectedWord=null,selectedGap=null,lastReference=null,lastSummary=null,lastAnnouncedCount=0;
    const wordNodes=new Map(),gapNodes=new Map(),active=()=>['recording','file'].includes(phase);
    function setPhase(value){phase=value;}
    ${diagnosticFunctions}\n${acceptFunction}\n${resetFunction}
  `,context);
  const run=code=>vm.runInContext(code,context);
  return {nodes,$,context,transcript,run,accept:(event,generation=1)=>{context.nextEvent=event;run(`accept(nextEvent,${generation})`);}};
}
const spoken={type:'transcript',replace_from:0,committed_count:0,
  tokens:[{id:'heard',text:' ولم يولد',start_ms:0,end_ms:900}],
  audio_activity:{status:'speech',speech_samples:16000,rms:.1,peak:.3}};
const quiet={type:'transcript',replace_from:1,committed_count:1,tokens:[],
  audio_activity:{status:'no_speech',speech_samples:0,rms:.001,peak:.01}};

test('no-speech and uncertain notices preserve raw words and apply monotonic commitment',()=>{
  const app=harness();app.accept(spoken);app.accept(quiet);
  assert.deepEqual(app.transcript.tokens.map(word=>word.text),[' ولم يولد']);
  assert.equal(app.transcript.committedCount,1);
  assert.match(app.$('live-status').textContent,/گفتار روشنی.*نزدیک‌تر/);
  app.accept({type:'feedback',updates:[],gaps:[],message:'متن بررسی شد'});
  assert.match(app.$('live-status').textContent,/گفتار روشنی/); // feedback cannot hide the detector notice
  app.accept({...quiet,audio_activity:{...quiet.audio_activity,status:'uncertain'}});
  assert.match(app.$('live-status').textContent,/قطعی نیست/);
  assert.deepEqual(app.transcript.tokens.map(word=>word.text),[' ولم يولد']);
});

test('empty no-speech session creates no heard text or canonical replacement',()=>{
  const app=harness();app.accept({...quiet,replace_from:0,committed_count:0});
  assert.equal(app.transcript.tokens.length,0);assert.equal(app.transcript.committedCount,0);
  assert.match(app.$('live-status').textContent,/گفتار روشنی/);
});

test('actual reset clears activity and old-generation late events cannot restore it',()=>{
  const app=harness();app.accept(spoken);app.accept(quiet);
  app.run('reset()');
  assert.equal(app.run('generation'),2);assert.equal(app.run('audioActivity'),null);
  assert.equal(app.run('inputWarning'),'');assert.equal(app.transcript.tokens.length,0);
  const readyMessage=app.$('live-status').textContent;
  app.accept(quiet,1);app.accept(spoken,1);
  assert.equal(app.$('live-status').textContent,readyMessage);assert.equal(app.transcript.tokens.length,0);
  app.accept(spoken,2);
  assert.deepEqual(app.transcript.tokens.map(word=>word.text),[' ولم يولد']);
  assert.doesNotMatch(app.$('live-status').textContent,/گفتار روشنی.*پیدا نشد/);
});

test('amplitude never claims speech or correctness and quiet level does not latch',()=>{
  const app=harness();app.run('resetAudioDiagnostics()');
  app.run("paintInputDiagnostics({sourceKind:'microphone',sampleRate:16000,windowSamples:16000,rms:.1,peak:.3,clippedFraction:0},0)");
  assert.equal(app.$('stage-voice').lastChild.textContent,'سطح ورودی');
  assert.equal(app.$('stage-voice').dataset.active,'false');assert.equal(app.$('stage-voice').dataset.level,'present');
  app.run("paintInputDiagnostics({sourceKind:'microphone',sampleRate:16000,windowSamples:16000,rms:0,peak:0,clippedFraction:0},300)");
  assert.equal(app.$('stage-voice').dataset.level,'low');assert.equal(app.$('live-wave-empty').hidden,false);
  app.run("paintInputDiagnostics({sourceKind:'microphone',sampleRate:16000,windowSamples:16000,rms:0,peak:0,clippedFraction:0},2500)");
  assert.match(app.$('live-status').textContent,/صدای ورودی خیلی کم/);
  assert.equal(app.$('stage-voice').dataset.active,'false');
  app.run("paintInputDiagnostics({sourceKind:'microphone',sampleRate:16000,windowSamples:16000,rms:.9,peak:1,clippedFraction:.1},2800)");
  assert.match(app.$('live-status').textContent,/بیش از حد بلند/);
  assert.equal(app.transcript.tokens.length,0);
});
