// Play two frozen public dev recordings through the actual browser file input and live v2 server.
// Run against a warmed, idle local server with PLAYWRIGHT_CORE_PATH set if Playwright is external.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {parseArgs} from 'node:util';

const root = fileURLToPath(new URL('../',import.meta.url));
const {values} = parseArgs({options:{
  url:{type:'string',default:'http://127.0.0.1:5001'},
  pack:{type:'string',default:path.join(root,'.cache/evaluation-pack')},
  output:{type:'string',default:path.join(root,'reports/verse-location-browser.json')}
}});
const url = new URL('/live',values.url);
assert.ok(['localhost','127.0.0.1','[::1]'].includes(url.hostname),'Use a local server.');
const require = createRequire(import.meta.url);
const playwright = require(process.env.PLAYWRIGHT_CORE_PATH || 'playwright-core');
const hash = data=>createHash('sha256').update(data).digest('hex');
const manifest = await readFile(path.join(values.pack,'manifest.json'));
const lock = JSON.parse(await readFile(path.join(root,'evaluation/assets-v1.json'),'utf8'));
assert.equal(hash(manifest),lock.manifest_sha256);
const {clips} = JSON.parse(manifest);
const cases = [
  {browser:'chromium',id:'pro-17-87-14',surah:87,ayah:14,ambiguous:false},
  {browser:'webkit',id:'pro-4-55-13',surah:55,ayah:13,ambiguous:true}
];
const report = {complete:false,url:url.href,manifest_sha256:hash(manifest),results:[]};
await mkdir(path.dirname(values.output),{recursive:true});
const save = ()=>writeFile(values.output,JSON.stringify(report,null,2)+'\n');

// Only observe live v2 traffic and the transcript heading. All media, model
// inference, connections, and updates to the page use normal app entry points.
function observe() {
  const trace = window.verseTrace = {firstPCM:null,words:[],events:[],locations:[],done:null};
  const originalSend = WebSocket.prototype.send;
  WebSocket.prototype.send = function(packet) {
    if (typeof packet === 'string' && JSON.parse(packet).type === 'start') {
      this.addEventListener('message',({data})=>{
        if (typeof data !== 'string') return;
        const msg = JSON.parse(data);
        if (msg.type === 'transcript') {
          trace.words.splice(msg.replace_from,trace.words.length,...msg.tokens.map(t=>t.text));
        }
        if (['transcript','feedback','done'].includes(msg.type)) {
          trace.events.push({type:msg.type,revision:msg.revision,status:msg.display_location?.status});
        }
        if (msg.type === 'done') trace.done = {display:msg.display_location,metrics:msg.metrics};
      });
    } else if (typeof packet !== 'string' && trace.firstPCM === null) trace.firstPCM=performance.now();
    return originalSend.call(this,packet);
  };
  addEventListener('DOMContentLoaded',()=>{
    const node = document.querySelector('#transcript-location');
    new MutationObserver(()=>{
      if (trace.firstPCM !== null) trace.locations.push({
        elapsed_ms:Math.round(performance.now()-trace.firstPCM),
        label:node.textContent,raw:document.querySelector('#heard-transcript').textContent
      });
    }).observe(node,{childList:true,subtree:true,characterData:true});
  },{once:true});
}
async function verify(c) {
  const clip = clips.find(row=>row.id === c.id);
  assert.ok(clip && clip.split === 'dev' && clip.surah === c.surah && clip.ayah === c.ayah);
  const audio = path.join(values.pack,clip.audio);
  assert.equal(hash(await readFile(audio)),clip.sha256);
  const browser = await playwright[c.browser].launch();
  const result = {id:c.id,sha256:clip.sha256,browser:c.browser,version:browser.version(),passed:false};
  report.results.push(result);
  try {
    const page = await browser.newPage({viewport:{width:c.ambiguous?390:1200,height:850},reducedMotion:'reduce'});
    const errors = [];
    page.on('pageerror',error=>errors.push(error.message));
    await page.addInitScript(observe);
    await page.goto(url.href);
    await page.locator('.file-practice > summary').click();
    await page.locator('#live-file').setInputFiles(audio);
    await page.locator('#file-play').click();
    await page.waitForFunction(()=>['review','error'].includes(document.body.dataset.phase),undefined,{timeout:180_000});
    assert.equal(await page.locator('body').getAttribute('data-phase'),'review',await page.locator('#live-error').innerText());
    const trace = await page.evaluate(()=>window.verseTrace);
    const words = await page.locator('#heard-transcript [data-token-id]').allTextContents();
    const label = await page.locator('#transcript-location').innerText();
    Object.assign(result,{label,words,location_updates:trace.locations,events:trace.events,done:trace.done});
    assert.ok(trace.done?.metrics.coverage_complete,'Final analysis must cover the recording.');
    assert.ok(words.length>0);
    assert.deepEqual(words,trace.words,'Visible words must preserve every raw server token.');
    assert.ok(await page.locator('#transcript-location').isVisible());
    const {status,candidates,total_candidates:total} = trace.done.display;
    const first = candidates[0];
    const fa = n=>String(n).replace(/\d/g,d=>'۰۱۲۳۴۵۶۷۸۹'[d]);
    if (c.ambiguous) {
      assert.equal(status,'ambiguous'); assert.equal(total,31);
      assert.ok(label.includes(fa(total)));
      const buttons = page.locator('#location-choices button');
      assert.equal(await buttons.count(),3); assert.ok(await buttons.first().isVisible());
      const second = candidates[1], option = buttons.nth(1);
      assert.ok((await option.innerText()).includes(second.surah_name) && (await option.innerText()).includes(fa(second.ayah)));
      await option.focus(); await page.keyboard.press('Enter');
      assert.match(await page.locator('#location-next').innerText(),/تمرین بعدی/);
      assert.equal(await page.locator('#start-surah').inputValue(),String(second.surah));
      assert.equal(await page.locator('#start-ayah').inputValue(),String(second.ayah));
      assert.equal(await page.locator('#reference-auto').isChecked(),false);
      assert.equal(await page.locator('#transcript-location').innerText(),label);
      assert.deepEqual(await page.locator('#heard-transcript [data-token-id]').allTextContents(),words);
      result.selected_for_next_practice=[second.surah,second.ayah];
    } else {
      assert.ok(['confirmed','probable'].includes(status));
      assert.deepEqual([first.surah,first.ayah],[c.surah,c.ayah]);
      assert.ok(label.includes(first.surah_name) && label.includes(fa(c.ayah)));
      await page.locator('#reference-panel > summary').click();
      assert.equal(await page.locator('#reference-text').textContent(),first.text);
    }
    assert.deepEqual(errors,[]);
    result.passed = true;
    console.log(c.browser,c.id,label,words.join(''));
  } catch(error) {
    result.error=error.message;
    throw error;
  } finally {await browser.close(); await save();}
}
for (const c of cases) await verify(c);
report.complete=true;
await save();
