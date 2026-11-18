// Run against a dedicated, warmed server. The file source exercises the real
// AudioWorklet, browser media clock, WebSocket, model and rendered transcript.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {stressTranscript} from './browser_transcript_stress.mjs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {parseArgs} from 'node:util';

const {values} = parseArgs({options:{
  url:{type:'string'}, corpus:{type:'string'}, output:{type:'string'}, subset:{type:'string'},
  browser:{type:'string',default:'chromium'}, baseline:{type:'boolean'}, quick:{type:'boolean'},
  case:{type:'string'}, mode:{type:'string'}, stress:{type:'boolean'}, 'layout-only':{type:'boolean'}, 'experiences-only':{type:'boolean'}, axe:{type:'string'}, screenshot:{type:'string'},
}});
if (!values.url || !values.output || (!values.corpus && !values['layout-only'] && !values.stress)) throw new Error('Supply --url, --corpus, and --output.');
const base=new URL(values.url);
if (!['127.0.0.1','localhost'].includes(base.hostname)) throw new Error('Use a dedicated local test server.');
const require=createRequire(import.meta.url);
const engine=require(process.env.PLAYWRIGHT_CORE_PATH || 'playwright-core')[values.browser];
if (!engine) throw new Error('Unknown Playwright engine.');
const browser=await engine.launch({headless:true});
const context=await browser.newContext({viewport:{width:1200,height:850}, reducedMotion:'reduce'});
const page=await context.newPage(), failures=[];
page.on('pageerror',error=>failures.push(error.message));
const report={browser:values.browser,version:browser.version(),url:base.href,source:'file; no microphone requested',
  baseline:Boolean(values.baseline),measure:'first binary PCM sent to first nonempty/determinate DOM update plus next two animation frames (presentation proxy)',runs:[],layout:[]};
const save=()=>fs.writeFile(values.output,JSON.stringify(report,null,2));
await context.addInitScript(()=>{
  const stats=window.reviewTest={started:null,text:null,feedback:null,frameCallbacks:0,
    textInViewport:null,feedbackInViewport:null};
  const scrollIntoView=Element.prototype.scrollIntoView;
  window.autoFollowReveals=[];
  Element.prototype.scrollIntoView=function(options) {
    if (this.id==='surah-reading' || this.matches?.('#heard-transcript button')) {
      window.autoFollowReveals.push({id:this.id || this.dataset.tokenId,phase:document.body.dataset.phase,time:performance.now()});
    }
    return scrollIntoView.call(this,options);
  };
  const send=WebSocket.prototype.send;
  WebSocket.prototype.send=function(data){
    if (typeof data!=='string' && stats.started===null) stats.started=performance.now();
    return send.call(this,data);
  };
  const OriginalSocket=window.WebSocket;
  window.WebSocket=class extends OriginalSocket {
    constructor(...args) {
      super(...args);
      this.addEventListener('message',({data})=>{
        if (typeof data!=='string') return;
        let event; try {event=JSON.parse(data);} catch {return;}
        if (event.type==='feedback' || event.type==='done') {
          const reviewed=(event.updates || []).filter(token=>token.state==='difference')
            .map(token=>({id:token.id,types:token.errors?.map(issue=>issue.type)}));
          (stats.serverReports??=[]).push({type:event.type,at:event.end_sample,status:event.status,
            reference:event.reference ? [event.reference.surah,event.reference.ayah] : null,
            summary:event.summary,differences:reviewed});
        }
      });
    }
  };
  addEventListener('DOMContentLoaded',()=>{
    const node=document.getElementById('heard-transcript'); if (!node) return;
    function notice(key,selector){
      if (stats[key]!==null || stats.started===null || !node.querySelector(selector)) return;
      function visibleFrame() {
        const inViewport=[...node.querySelectorAll(selector)].some(el=>{
          const rect=el.getBoundingClientRect();
          return rect.width>0 && rect.height>0 && rect.right>0 && rect.left<innerWidth && rect.bottom>0 && rect.top<innerHeight;
        });
        if (!inViewport) {if (performance.now()-stats.started<60000 && !['review','error'].includes(document.body.dataset.phase)) requestAnimationFrame(visibleFrame);return;}
        requestAnimationFrame(()=>{stats[key+'InViewport']=Math.round(performance.now()-stats.started);});
      }
      requestAnimationFrame(visibleFrame);
      stats[key]=-1;
      requestAnimationFrame(()=>requestAnimationFrame(()=>{
        stats[key]=Math.round(performance.now()-stats.started); stats.frameCallbacks++;
      }));
    }
    new MutationObserver(()=>{
      notice('text','[data-token-id]');
      notice('feedback','[data-state="difference"],[data-state="match"],[data-gap-id]');
    }).observe(node,{subtree:true,childList:true,attributes:true});
  });
});
const inputFiles=['control-word-order-correct.wav','control-repeat-nabudu.wav','control-swap-verbs.wav','control-omit-nabudu.wav','control-full-ayah-reread.wav'];
async function audit(label) {
  if (!values.axe) return;
  await page.addScriptTag({path:path.resolve(values.axe)});
  const outcome=await page.evaluate(async()=> {
    const result=await window.axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}});
    return {executed:result.passes.length+result.violations.length, violations:result.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>n.target)}))};
  });
  (report.accessibility??=[]).push({label,...outcome});
}
async function checkRecovery() {
  // Use deterministic in-memory PCM for WebKit's retry input. The
  // MediaStream-based virtual microphone did not start on this CI host.
  // Every file and the denied-permission check still use the original module.
  if (values.browser==='webkit') await page.route('**/static/live-source.js',route=>route.fulfill({
    contentType:'text/javascript',body:`
      export * from './live-source.js?browser-qa-original';
      import {prepareAudioSource as prepare} from './live-source.js?browser-qa-original';
      export function prepareAudioSource(options) {
        if (window.useRetryPcm && !options.fileAudio) {
          window.retryPcmUsed=true;
          return prepare({...options,fileAudio:{samples:new Float32Array(16_000*4),rate:16_000}});
        }
        return prepare(options);
      }
    `}));
  let sessionCount=0; const starts=[];
  await page.routeWebSocket('**/live/ws',ws=>{
    const session='browser-review-fixture-'+(++sessionCount);
    let rate=16000, stage=0;
    const send=event=>ws.send(JSON.stringify({session_id:session,...event}));
    ws.onMessage(packet=>{
      if (typeof packet==='string') {
        const event=JSON.parse(packet);
        if (event.type==='start') {rate=event.sample_rate; starts.push({id:session,surah:event.start_surah,ayah:event.start_ayah}); send({type:'ready',version:2});}
        else if (event.type==='stop') send({type:'done',revision:4});
        return;
      }
      const seq=packet.readUInt32LE(), end=packet.readUInt32LE(4)+(packet.length-8)/2;
      send({type:'ack',sequence:seq,input_samples:end});
      const token=(id,text,start_ms,end_ms)=>({id,text,start_ms,end_ms});
      if (stage===0 && end/rate>.8) {
        stage++; send({type:'transcript',revision:1,replace_from:0,committed_count:1,tokens:[token('w1',' إياك',200,600),token('old',' من',650,800)]});
      }
      if (stage===1 && end/rate>1.8) {
        stage++; send({type:'transcript',revision:2,replace_from:1,committed_count:3,tokens:[token('w2',' نستعين',800,1150),token('w3',' وإياك',1300,1600)]});
        send({type:'feedback',revision:3,end_sample:28000,status:'tracking',
          updates:[{id:'w1',state:'match'},{id:'w2',state:'difference',reference:{surah:1,ayah:5,word:'نَعْبُدُ'},errors:[{type:'transposition',expected:'نعبد',heard:'نستعين'}]}],
          gaps:[{id:'g',after_token_id:'w3',before_token_id:'w2',expected:'نعبد'}],
          summary:{reviewed:2,differences:1,uncertain:1,missing:1}});
      }
      if (stage===2 && end/rate>4) {stage++; ws.close({code:1011,reason:'intentional QA disconnect'});}
    });
  });
  await page.setViewportSize({width:390,height:844});
  await page.goto(new URL('/',base).href); await page.locator('#precise-surah').selectOption('2');
  await page.locator('#precise-ayah').selectOption('282'); await page.waitForFunction(()=>document.getElementById('verse-text').textContent.length>300);
  assert.ok(await page.locator('#verse-nav button').count()<15); await audit('single-verse long passage');
  await page.goto(new URL('/live',base).href); await audit('idle');
  if (values.screenshot) await page.screenshot({path:path.resolve(values.screenshot+'-idle.png')});
  await page.locator('.file-practice > summary').click();
  await page.locator('#live-file').setInputFiles(path.resolve(values.corpus,inputFiles[0]));
  await page.locator('#file-play').click();
  await page.locator('[data-token-id=old]').waitFor();
  const firstWord=await page.locator('[data-token-id=w1]').boundingBox();
  assert.ok(firstWord.y>=0 && firstWord.y+firstWord.height<=844,'First raw word is visible after using file controls.');
  assert.equal(await page.evaluate(()=>window.autoFollowReveals.filter(r=>r.id==='surah-reading').length),1);
  const prefix=await page.locator('[data-token-id=w1]').elementHandle();
  await page.locator('[data-token-id=w2]').waitFor();
  assert.ok(await prefix.evaluate(node=>node===document.querySelector('[data-token-id=w1]')));
  assert.equal(await page.locator('[data-token-id=old]').count(),0);
  assert.equal(await page.locator('#heard-transcript [tabindex="0"]').count(),1);
  await page.locator('[data-token-id=w1]').focus(); await page.keyboard.press('ArrowLeft');
  assert.equal(await page.evaluate(()=>document.activeElement.dataset.tokenId),'w2');
  const bounds=await page.locator('#heard-transcript').boundingBox(); assert.ok(bounds && bounds.y<844);
  await page.waitForFunction(()=>document.body.dataset.phase==='error');
  await page.waitForFunction(()=>document.getElementById('replay-audio').duration>=3.5);
  const duration=await page.locator('#replay-audio').evaluate(a=>a.duration);
  assert.match(await page.locator('#replay-note').textContent(),/قطع ارتباط/);
  await page.locator('#review-first').click();
  assert.equal((await page.locator('#word-title').textContent()).trim(),'نستعين');
  assert.match(await page.locator('#word-explanation').textContent(),/واژهٔ مورد انتظار/);
  await page.locator('#segment-play').click();
  await page.waitForFunction(()=>!document.getElementById('replay-audio').paused);
  const start=await page.locator('#replay-audio').evaluate(a=>a.currentTime);
  assert.ok(start>=.46 && start<.8,'word replay begins with context');
  await page.waitForFunction(()=>document.getElementById('replay-audio').paused);
  const stop=await page.locator('#replay-audio').evaluate(a=>a.currentTime);
  assert.ok(stop>=1.44 && stop<1.8,'word replay stops near the selected end');
  await page.keyboard.press('Escape'); await page.locator('#difference-next').click();
  assert.match(await page.locator('#word-title').textContent(),/جاافتاده/);
  assert.match(await page.locator('#segment-play').textContent(),/دو سوی/);
  await page.locator('#segment-play').click();
  await page.waitForFunction(()=>!document.getElementById('replay-audio').paused);
  await page.waitForFunction(()=>document.getElementById('replay-audio').paused);
  const gapStop=await page.locator('#replay-audio').evaluate(a=>a.currentTime);
  assert.ok(gapStop>=1.88 && gapStop<2.2);
  if (values.screenshot) await page.screenshot({path:path.resolve(values.screenshot+'-gap.png')});
  const tree=await page.locator('#word-detail').ariaSnapshot();
  assert.match(tree,/dialog.*بخش احتمالی جاافتاده/);
  assert.match(tree,/button.*شنیدن دو سوی جای خالی/);
  await page.locator('#word-close').focus(); await page.keyboard.press('Shift+Tab');
  assert.equal(await page.evaluate(()=>document.activeElement.id),'segment-play');
  await page.keyboard.press('Tab'); assert.equal(await page.evaluate(()=>document.activeElement.id),'word-close');
  await audit('gap review');
  await page.keyboard.press('Escape');
  const raw=await page.locator('#heard-transcript [data-token-id]').allTextContents();
  assert.deepEqual(raw,[' إياك',' نستعين',' وإياك']);
  await page.locator('#check-ready').click();
  await page.waitForFunction(()=>document.body.dataset.ready==='true' && document.getElementById('live-error').hidden);
  const restarted=page.waitForFunction(()=>document.body.dataset.phase==='file');
  await page.locator('#retry-tail').click();
  await restarted;
  await page.waitForFunction(()=>document.body.dataset.phase==='review');
  await page.waitForFunction(()=>Number.isFinite(document.getElementById('replay-audio').duration));
  const repeatDuration=await page.locator('#replay-audio').evaluate(a=>a.duration);
  assert.ok(repeatDuration>1 && repeatDuration<duration);
  await page.evaluate(engine=>{
    if (engine==='webkit') {window.useRetryPcm=true;return;}
    Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:async()=>{
      const context=new AudioContext(), destination=context.createMediaStreamDestination();
      const oscillator=context.createOscillator(), volume=context.createGain();
      volume.gain.value=.01; oscillator.connect(volume); volume.connect(destination);
      oscillator.start(); await context.resume();
      window.retryMicFixture={context,oscillator,stream:destination.stream};
      return destination.stream;
    }});
  },values.browser);
  await page.locator('[data-token-id=w2]').click();
  await page.locator('#word-retry').click(); await page.waitForFunction(()=>document.body.dataset.phase==='recording');
  assert.equal(starts.length,3); assert.equal(new Set(starts.map(start=>start.id)).size,3);
  assert.deepEqual([starts[2].surah,starts[2].ayah],[1,5]);
  assert.equal(await page.locator('#reference-auto').isChecked(),false);
  assert.equal(await page.locator('#replay-section').isVisible(),false);
  await page.locator('#live-stop-mobile').click();
  await page.waitForFunction(()=>document.body.dataset.phase==='review');
  const virtualTrackState=await page.evaluate(async()=>{
    if (window.useRetryPcm) {window.useRetryPcm=false;return window.retryPcmUsed?'local PCM stub stopped':'unused';}
    const {context,oscillator,stream}=window.retryMicFixture;
    oscillator.stop(); await context.close(); return stream.getAudioTracks()[0].readyState;
  });
  assert.equal(virtualTrackState,values.browser==='webkit'?'local PCM stub stopped':'ended');
  await page.locator('#live-reset').click();
  assert.equal(await page.locator('#replay-audio').getAttribute('src'),null);
  await page.evaluate(()=>Object.defineProperty(navigator.mediaDevices,'getUserMedia',{configurable:true,value:()=>Promise.reject(new DOMException('Simulated permission denial','NotAllowedError'))}));
  await page.locator('#live-start').click();
  await page.waitForFunction(()=>document.body.dataset.phase==='error');
  assert.match(await page.locator('#live-error').textContent(),/مجوز میکروفن/);
  const sameAyahSessions=[...starts];
  // Reproduce the desktop scroll position reached by the manual picker and
  // file controls. After the first word, an actual wheel gesture suppresses
  // later programmatic follows while the same session remains active.
  await page.setViewportSize({width:1200,height:850}); await page.goto(new URL('/live',base).href);
  await page.locator('#reference-choice > summary').click(); await page.locator('#reference-auto').uncheck();
  await page.locator('#start-ayah').selectOption('5');
  await page.locator('.file-practice > summary').click();
  await page.locator('#live-file').setInputFiles(path.resolve(values.corpus,'control-omit-nabudu.wav'));
  await page.locator('#file-play').click();
  await page.locator('[data-token-id=w1]').waitFor();
  const manualFirstWord=await page.locator('[data-token-id=w1]').boundingBox();
  assert.ok(manualFirstWord.y>=0 && manualFirstWord.y+manualFirstWord.height<=850);
  const beforeWheel=await page.evaluate(()=>window.autoFollowReveals.length);
  await page.mouse.wheel(0,520); await page.locator('[data-token-id=w3]').waitFor();
  assert.equal(await page.evaluate(()=>window.autoFollowReveals.length),beforeWheel);
  await page.locator('#live-stop').click(); await page.waitForFunction(()=>document.body.dataset.phase==='review');
  return {fixture:'scripted v2 server events; real browser/file/AudioWorklet',partialDuration:duration,wordStart:start,wordStop:stop,gapStop,
    accessibilityTree:tree,modalFocusCycle:true,voiceOverTested:false,focus:true,immutablePrefix:true,revisedTail:true,partialReview:true,firstWordViewport:firstWord,raw,reset:true,offlineRecovery:true,simulatedMicDenial:true,manualFirstWord,manualWheelRespected:true,sameAyahRetry:{sessions:sameAyahSessions,virtualTrackState},retryTailDuration:repeatDuration};
}
function scoreControl(run) {
  const ordinary=['إياك','نعبد','وإياك','نستعين'];
  const controls={
    'control-word-order-correct.wav':[ordinary,0,0],
    'control-repeat-nabudu.wav':[['إياك','نعبد','نعبد','وإياك','نستعين'],1,0],
    'control-swap-verbs.wav':[['إياك','نستعين','وإياك','نعبد'],2,0],
    'control-omit-nabudu.wav':[['إياك','وإياك','نستعين'],0,1],
    'control-full-ayah-reread.wav':[[...ordinary,...ordinary],0,0],
  };
  const [words,differences,gaps]=controls[run.file];
  const kind=run.file.includes('repeat-nabudu')?'تکرار':run.file.includes('swap-verbs')?'جابه‌جایی کلمه':null;
  return {raw:JSON.stringify(run.words.map(word=>word.trim()))===JSON.stringify(words),
    placementCount:run.gaps===gaps && run.differences===differences,
    flaggedOwners:JSON.stringify(run.flaggedIndices)===JSON.stringify(run.file.includes('repeat-nabudu')?[2]:run.file.includes('swap-verbs')?[1,3]:[]),
    errorLabels:!kind || run.renderedDetails?.length===differences && run.renderedDetails.every(label=>label.includes(kind)),
    gapOwners:JSON.stringify(run.gapsAfter)===JSON.stringify(run.file.includes('omit-nabudu')?[0]:[]),
    definitive:run.feedback>0};
}
const sha256=buffer=>createHash('sha256').update(buffer).digest('hex');
async function frozenEntries() {
  const raw=await fs.readFile(values.subset);
  const subset=JSON.parse(raw), pack=await fs.readFile(path.join(values.corpus,'manifest.json'));
  assert.equal(sha256(pack),subset.manifest_sha256,'Unexpected evaluation-pack manifest.');
  const all=JSON.parse(pack), byId=new Map(all.clips.map(row=>[row.id,row]));
  assert.ok(all.complete); assert.equal(new Set(subset.clips.map(row=>row.group)).size,14);
  for (const row of subset.clips) {
    const source=byId.get(row.id);
    assert.ok(source?.source_kind==='professional' && source.duration_s>=4 && row.audio===source.audio);
    assert.equal(sha256(await fs.readFile(path.resolve(values.corpus,row.audio))),row.sha256);
  }
  report.frozen={sha256:sha256(raw),manifest_sha256:sha256(pack),method:subset.measurement,
    plannedIds:subset.clips.map(row=>row.id)};
  return subset.clips;
}
function timingSummary(runs,key) {
  const all=runs.map(row=>row[key]>0?row[key]:Infinity).sort((a,b)=>a-b);
  const valid=all.filter(Number.isFinite);
  const rank=(list,p)=>{const value=list[Math.ceil(list.length*p)-1];return Number.isFinite(value)?value:null;};
  return {total:all.length,observed:valid.length,missing:all.length-valid.length,observedP50ms:rank(valid,.5),
    observedP95ms:rank(valid,.95),allClipsP95ms:rank(all,.95)};
}
try {
  const pageSources=['static/live.js','static/live-source.js','static/transcript-state.js'];
  report.servedClientSha256={};
  for (const key of pageSources) {
    const response=await context.request.get(new URL(key,base).href);
    assert.ok(response.ok()); report.servedClientSha256[key]=sha256(await response.body());
  }
  report.health=await (await context.request.get(new URL('/health',base).href)).json();
  for (const width of [320,390,768,1440]) {
    await page.setViewportSize({width,height:850}); await page.goto(new URL('/live',base).href);
    await page.locator('#live-start').waitFor();
    const dims=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,dir:document.dir,lang:document.documentElement.lang}));
    assert.ok(dims.scrollWidth<=dims.width+1); assert.equal(dims.dir,'rtl'); assert.equal(dims.lang,'fa'); report.layout.push(dims);
  }
  await page.setViewportSize({width:1280,height:900}); await page.evaluate(()=>document.documentElement.style.zoom='2');
  const zoomed=await page.evaluate(()=>({cssZoom:getComputedStyle(document.documentElement).zoom,bodyWidth:document.body.getBoundingClientRect().width,width:innerWidth}));
  assert.ok(zoomed.bodyWidth<=zoomed.width+2); report.reflow200=zoomed;
  const entries=values.subset?await frozenEntries():null;
  const files=entries?entries.map(row=>row.audio):values.case?[values.case]:values.quick?['control-omit-nabudu.wav']:inputFiles;
  if (!entries && files.some(name=>!inputFiles.includes(name))) throw new Error('Unknown control filename.');
  if (!entries && !values.stress && !values['layout-only'] && !values['experiences-only']) {
    const manifest=JSON.parse(await fs.readFile(path.join(values.corpus,'manifest.json')));
    report.checkedAudio={};
    for (const file of files) {
      const fingerprint=sha256(await fs.readFile(path.resolve(values.corpus,file)));
      assert.equal(fingerprint,manifest.source_audio_sha256[file],'Control audio hash differs from its frozen manifest.');
      report.checkedAudio[file]=fingerprint;
    }
  }
  const modes=entries?['auto']:values.mode?[values.mode]:values.quick?['manual']:['auto','manual'];
  if (modes.some(mode=>!['auto','manual'].includes(mode))) throw new Error('Use --mode auto or manual.');
  if (!values.stress && !values['layout-only'] && !values['experiences-only']) for (const file of files) for (const mode of modes) {
    await page.setViewportSize({width:1200,height:850}); await page.goto(new URL('/live?measure',base).href);
    if (mode==='manual') {
      await page.locator('#reference-choice > summary').click(); await page.locator('#reference-auto').uncheck();
      await page.locator('#start-surah').selectOption('1'); await page.locator('#start-ayah').selectOption('5');
    }
    await page.locator('.file-practice > summary').click();
    await page.locator('#live-file').setInputFiles(path.resolve(values.corpus,file));
    await page.locator('#file-play').click();
    if (entries) await page.locator('#surah-reading').scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>['review','error'].includes(document.body.dataset.phase),null,{timeout:100000});
    const phase=await page.locator('body').getAttribute('data-phase');
    assert.equal(phase,'review',await page.locator('#live-error').textContent());
    if (entries) await page.waitForFunction(()=>window.reviewTest.text!==-1);
    else await page.waitForFunction(()=>window.reviewTest.text>0,null,{timeout:10000});
    await page.waitForFunction(()=>window.reviewTest.feedback!==-1);
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    const result={file,mode,...await page.evaluate(()=>window.reviewTest),
      words:await page.locator('#heard-transcript [data-token-id]').allTextContents(),
      gaps:await page.locator('#heard-transcript [data-gap-id]').count(),
      differences:await page.locator('#heard-transcript [data-state="difference"]').count(),
      flaggedIndices:await page.locator('#heard-transcript [data-token-id]').evaluateAll(words=>words.flatMap((word,i)=>word.dataset.state==='difference'?[i]:[])),
      gapsAfter:await page.locator('#heard-transcript').evaluate(box=>[...box.querySelectorAll('[data-gap-id]')].map(gap=>[...box.querySelectorAll('[data-token-id]')].indexOf(gap.previousElementSibling)))};
    report.runs.push(result); await save(); console.log(file,mode,result.text,result.feedback);
    if (entries) {
      const row=entries.find(clip=>clip.audio===file);
      Object.assign(result,{id:row.id,group:row.group,split:row.split,duration:row.duration_s});
      continue;
    }
    const audio=page.locator('#replay-audio');
    await page.locator('#replay-toggle').click();
    await page.waitForFunction(()=>document.getElementById('replay-audio').currentTime>.05);
    await page.locator('#replay-toggle').click();
    await page.locator('#replay-seek').fill('55');
    assert.ok(await audio.evaluate(a=>a.currentTime/a.duration>.50));
    if (!values.baseline && (result.gaps || result.differences)) {
      await page.locator('#difference-next').click();
      await page.locator('#segment-play').click();
      await page.waitForFunction(()=>!document.getElementById('replay-audio').paused);
      await page.waitForFunction(()=>document.getElementById('replay-audio').paused,null,{timeout:18000});
      result.detail=await page.locator('#word-explanation').textContent();
      result.segmentStoppedAt=await audio.evaluate(a=>a.currentTime);
      assert.ok(result.segmentStoppedAt>0); await page.keyboard.press('Escape');
      assert.ok(await page.evaluate(()=>document.activeElement.matches('#heard-transcript button')),'Close returns focus to the reviewed token or gap');
    }
    result.renderedDetails=[];
    for (const word of await page.locator('#heard-transcript [data-state="difference"]').all()) {
      await word.click();
      result.renderedDetails.push(await page.locator('#word-explanation').innerText());
      await page.keyboard.press('Escape');
    }
    result.replayPlaySeek=true; result.scored=scoreControl(result);
  }
  if (values.stress) report.stress=await stressTranscript(page,context,base,values.browser);
  if (values['experiences-only']) report.experience=await checkRecovery();
  assert.deepEqual(failures,[]);
  assert.ok((report.accessibility??[]).every(result=>result.violations.length===0),'Accessibility violations found; inspect selectors.');
  if (report.runs.length) report.timing={firstTextDom:timingSummary(report.runs,'text'),firstVerdictDom:timingSummary(report.runs,'feedback'),
    firstTextInViewport:timingSummary(report.runs,'textInViewport'),firstVerdictInViewport:timingSummary(report.runs,'feedbackInViewport')};
  if (!entries) assert.ok(report.runs.every(run=>Object.values(run.scored).every(Boolean)),'Rendered control mismatches; inspect per-case scores.');
  report.status='passed';
} catch(error) {report.status='failed'; report.error=String(error); process.exitCode=1; console.error(error);}
finally {report.pageErrors=failures; await save(); await browser.close();}
