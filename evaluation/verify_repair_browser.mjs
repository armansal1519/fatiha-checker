// Upload real, pinned audio through /live. Observe the UI and unmodified v2 socket.
// With a warm, dedicated local server and installed Playwright browsers:
// node evaluation/verify_repair_browser.mjs --url http://127.0.0.1:5014
// PLAYWRIGHT_CORE_PATH may point to an external playwright-core installation.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {parseArgs} from 'node:util';

const root = fileURLToPath(new URL('../', import.meta.url));
const {values} = parseArgs({options: {
  url: {type:'string'}, pack: {type:'string', default:path.join(root,'.cache/evaluation-pack')},
  output: {type:'string', default:path.join(root,'reports/repair-browser-final.json')},
  browser: {type:'string', default:'chromium,webkit'}
}});
const url = new URL('/live', values.url);
if (!['127.0.0.1','localhost'].includes(url.hostname)) throw Error('Use a loopback server.');
const engines = createRequire(import.meta.url)(process.env.PLAYWRIGHT_CORE_PATH || 'playwright-core');
const sha = data => createHash('sha256').update(data).digest('hex');
const normalize = text => text.normalize('NFKC').replace(/[\u064b-\u065f\u0670\u06d6-\u06edـ]/gu,'')
  .replace(/[إأآٱ]/gu,'ا').replace(/ى/gu,'ي').trim().split(/\s+/u).filter(Boolean).join(' ');
const manifest = JSON.parse(await fs.readFile(path.join(values.pack,'manifest.json')));
const input = [];
for (const id of ['control-110-2-repetition','learner-e5e672b2-chunk0']) {
  const clip = manifest.clips.find(c => c.id === id);
  const buffer = await fs.readFile(path.join(values.pack,clip.audio));
  if (sha(buffer)!==clip.sha256) throw Error('Changed audio '+id);
  const type = id.startsWith('control')?'repetition':'letter';
  const target = type==='repetition'?clip.spoken_text.split(/\s+/u)[clip.events[0].heard_indices[0]]:
    Object.keys(clip.source_annotations.canonical_word_errors)[0];
  input.push({id, buffer, type, target:normalize(target), sourceHash:clip.sha256,
    gold:clip.spoken_text, surah:clip.surah, ayah:clip.ayah});
}
const chapter = await fs.readFile(path.join(values.pack,'passages/audio/transition-hafs-112-113.wav'));
const lock = JSON.parse(await fs.readFile(path.join(root,'evaluation/passage-challenge-lock.json')));
if (sha(chapter)!==lock.clips['transition-hafs-112-113']) throw Error('Changed chapter fixture');
function tailWav(wav,seconds) {
  if (wav.toString('ascii',12,16)!=='fmt ' || wav.readUInt16LE(20)!==1 || wav.readUInt16LE(22)!==1
      || wav.readUInt32LE(24)!==16000 || wav.readUInt16LE(34)!==16) throw Error('Require mono 16kHz PCM16');
  let i=12;
  while (i+8<wav.length && wav.toString('ascii',i,i+4)!=='data') {
    const size = wav.readUInt32LE(i+4);
    i+=8+size+(size%2);
  }
  if (i+8>=wav.length) throw Error('No PCM data');
  const data=wav.subarray(i+8+seconds*32000,i+8+wav.readUInt32LE(i+4));
  const header=Buffer.from(wav.subarray(0,i+8));
  header.writeUInt32LE(header.length+data.length-8,4); header.writeUInt32LE(data.length,i+4);
  return Buffer.concat([header,data]);
}
input.push({id:'chapter_join_slice_18s',type:'tail',buffer:tailWav(chapter,18),sourceHash:sha(chapter),
  expectedSuffix:normalize('من شر ما خلق'), prohibited:normalize('اشتركوا في القناة')});

async function run(browser,test) {
  const page=await browser.newPage({viewport:{width:1200,height:850},reducedMotion:'reduce'});
  const words=[], history=[], errors=[];
  let done;
  const result={id:test.id, type:test.type, mode:'auto', source_sha256:test.sourceHash,
    uploaded_wav_sha256:sha(test.buffer), source_start_s:test.type==='tail'?18:0};
  page.on('pageerror', e=>errors.push(e.message));
  page.on('websocket', socket=>socket.on('framereceived',({payload})=>{
    if (typeof payload!=='string') return;
    const e=JSON.parse(payload);
    if (e.type==='transcript') {
      words.splice(e.replace_from,words.length-e.replace_from,...e.tokens);
      history.push({revision:e.revision,raw:words.map(w=>w.text).join('')});
    } else if (e.type==='feedback') {
      for (const update of e.updates) {
        const word=words.find(w=>w.id===update.id);
        if (word) Object.assign(word,update);
      }
    } else if (e.type==='done') done=e;
  }));
  try {
    await page.goto(url.href);
    await page.locator('.file-practice > summary').click();
    await page.locator('#live-file').setInputFiles({name:test.id+'.wav',mimeType:'audio/wav',buffer:test.buffer});
    await page.locator('#file-play').click();
    await page.waitForFunction(()=>['review','error'].includes(document.body.dataset.phase),null,{timeout:120000});
    result.phase=await page.locator('body').getAttribute('data-phase');
    if (result.phase!=='review') throw Error(await page.locator('#live-error').innerText());
    result.rendered=await page.locator('#heard-transcript [data-token-id]').evaluateAll(nodes=>nodes.map(n=>({
      id:n.dataset.tokenId,text:n.textContent,state:n.dataset.state})));
    result.session_id=done?.session_id;
    result.coverage_complete=done?.metrics?.coverage_complete===true;
    result.history=history;
    result.dom_matches_wire=result.rendered.length===words.length && result.rendered.every((w,i)=>
      w.id===words[i].id && w.text===words[i].text && w.state===words[i].state);
    const full=normalize(result.rendered.map(w=>w.text).join(' '));
    let target;
    if (test.type==='tail') {
      result.expected_suffix=test.expectedSuffix;
      result.suffix_matches=full.endsWith(test.expectedSuffix);
      result.ever_boilerplate=history.some(r=>normalize(r.raw).includes(test.prohibited));
      target=words.at(-1);
      result.target_pass=result.suffix_matches && !result.ever_boilerplate;
    } else {
      const occurrences=words.filter(w=>normalize(w.text)===test.target);
      const expected=test.type==='letter'?occurrences[0]:occurrences[1];
      target=expected || occurrences[0];
      result.target=test.target;
      result.occurrences=occurrences.map(w=>w.id);
      result.correct_type_on_target=expected?.state==='difference' &&
        expected.errors.some(e=>e.type===test.type) && expected.reference?.surah===test.surah && expected.reference?.ayah===test.ayah;
      if (test.type==='repetition') {
        result.known_spoken_text=test.gold;
        result.exact_spoken_sequence=full===normalize(test.gold);
      }
      result.target_pass=Boolean(result.correct_type_on_target && (test.type!=='repetition' || result.exact_spoken_sequence));
    }
    if (!target) throw Error('No corresponding visible word available for review');
    const node=page.locator('#heard-transcript [data-token-id="'+target.id+'"]');
    await node.click();
    const detail=await page.locator('#word-explanation').innerText();
    const title=await page.locator('#word-title').textContent();
    result.review={token_id:target.id,raw:target.text,title,explanation:detail,start_ms:target.start_ms,
      end_ms:target.end_ms, reference:target.reference, issues:target.errors,
      title_matches:title===target.text, explanation_matches_type:test.type==='tail'?null:
        detail.includes(test.type==='letter'?'حرف':'تکرار')};
    await page.waitForFunction(()=>document.querySelector('#replay-audio').duration>0);
    const audio=page.locator('#replay-audio');
    const duration=await audio.evaluate(a=>{
      window.measuredSegment={};
      a.addEventListener('playing',()=>{window.measuredSegment.start=a.currentTime;},{once:true});
      a.addEventListener('pause',()=>{window.measuredSegment.stop=a.currentTime;},{once:true});
      return a.duration;
    });
    await page.locator('#segment-play').click();
    await page.waitForFunction(()=>Number.isFinite(window.measuredSegment.stop),null,{timeout:18000});
    const clocks=await page.evaluate(()=>window.measuredSegment);
    const expectedStart=Math.max(0,target.start_ms/1000-.3);
    const expectedEnd=Math.min(duration,target.end_ms/1000+.35);
    result.replay={duration_s:duration,expected_start_s:expectedStart,expected_end_s:expectedEnd,
      observed_playing_s:clocks.start,observed_pause_s:clocks.stop,
      within_tolerance:Number.isFinite(clocks.start) && Math.abs(clocks.start-expectedStart)<.25 &&
        Math.abs(clocks.stop-expectedEnd)<.25 && clocks.stop>clocks.start};
    await page.keyboard.press('Escape');
    result.focus_returned=await node.evaluate(n=>n===document.activeElement);
    result.ui_pass=Boolean(result.dom_matches_wire && result.coverage_complete && result.review.title_matches &&
      result.replay.within_tolerance && result.focus_returned && !errors.length);
  } catch (e) { result.error=String(e);result.ui_pass=false; }
  finally { result.page_errors=errors; await page.close(); }
  return result;
}

const report={url:url.href, method:'Real file AudioWorklet, local WebSocket and offline models. Select target in the rendered DOM and observe HTMLMediaElement playing/pause clocks. Auto-reference mode. No network or media mocks.',
  tolerance_s:.25, runs:[]};
for (const name of values.browser.split(',')) {
  const browser=await engines[name].launch({headless:true});
  try {
    for (const test of input) {
      const row=await run(browser,test);
      report.runs.push({browser:name,version:browser.version(),...row});
      await fs.writeFile(values.output,JSON.stringify(report,null,2)+'\n');
      console.log(name,test.id,row.ui_pass,row.target_pass,row.error || '');
    }
  } finally { await browser.close(); }
}
report.complete=true;
report.all_verified=report.runs.every(r=>r.ui_pass && r.target_pass && r.review?.explanation_matches_type!==false);
await fs.writeFile(values.output,JSON.stringify(report,null,2)+'\n');
if (!report.all_verified) process.exitCode=1;
