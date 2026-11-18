// Exercise the live file workflow, the downloaded PCM excerpt, and its offline reader.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';

const root = fileURLToPath(new URL('../', import.meta.url));
const {values} = parseArgs({options:{
  url:{type:'string',default:'http://127.0.0.1:5001'},
  output:{type:'string',default:path.join(root,'reports/local-issue-export.json')},
  screenshot:{type:'string'}
}});
const base = new URL(values.url);
assert.ok(['127.0.0.1','localhost'].includes(base.hostname));
const tools = createRequire(import.meta.url)(process.env.PLAYWRIGHT_CORE_PATH || 'playwright-core');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const pack = path.join(root,'.cache/evaluation-pack');
const manifest = JSON.parse(await fs.readFile(path.join(pack,'manifest.json')));
const cases = [['chromium','control-110-2-repetition','repetition'], ['webkit','learner-e5e672b2-chunk0','letter']];
const temp = await fs.mkdtemp(path.join(tmpdir(),'hamkhan-export-'));
const report = [];
try {
  for (const [engine,id,type] of cases) {
    const clip = manifest.clips.find(c => c.id === id);
    const input = await fs.readFile(path.join(pack,clip.audio));
    assert.equal(sha(input),clip.sha256);
    const browser = await tools[engine].launch({headless:true});
    try {
      const ctx = await browser.newContext({viewport:{width:960,height:850},acceptDownloads:true});
      const page = await ctx.newPage();
      const pageErrors = [];
      page.on('pageerror', error => pageErrors.push(error.message));
      const tokens = new Map();
      page.on('websocket', socket => socket.on('framereceived', frame => {
        if (typeof frame.payload !== 'string') return;
        const event = JSON.parse(frame.payload);
        for (const token of event.updates || []) tokens.set(token.id,token);
      }));
      await page.goto(new URL('/live', base).href);
      await page.locator('.file-practice > summary').click();
      await page.locator('#live-file').setInputFiles({name:'practice.wav',mimeType:'audio/wav',buffer:input});
      await page.locator('#file-play').click();
      await page.waitForFunction(()=>['review','error'].includes(document.body.dataset.phase),null,{timeout:120000});
      assert.equal(await page.locator('body').getAttribute('data-phase'),'review');
      const target = [...tokens.values()].find(token => token.state === 'difference' && token.errors.some(e=>e.type===type));
      assert.ok(target, `Expected ${type} in ${id}`);
      await page.locator(`#heard-transcript [data-token-id="${target.id}"]`).click();
      await page.locator('#issue-open').click();
      const note = 'آزمون ذخیرهٔ نمونه';
      await page.locator('#issue-note').fill(note);
      if (values.screenshot && engine === 'chromium') await page.locator('#issue-panel').screenshot({path:values.screenshot});
      const received = page.waitForEvent('download');
      await page.locator('#issue-save').click();
      const local = path.join(temp,engine+'.json');
      await (await received).saveAs(local);
      const sample = JSON.parse(await fs.readFile(local));
      assert.equal(sample.schema,'hamkhan.issue.v1');
      assert.equal(sample.user_note,note);
      assert.deepEqual(sample.selection.token_ids,[target.id]);
      assert.equal(sample.session_status,'finished');
      assert.ok(sample.revisions.length);
      assert.ok(sample.final_words.find(w=>w.id===target.id)?.errors.some(e=>e.type===type));
      const {audio} = sample;
      const length = audio.session_end_ms - audio.session_start_ms;
      assert.ok(length>0 && length<=8000.01);
      const bytes = Buffer.from(audio.data,'base64');
      const hash = await page.locator('#replay-audio').evaluate(async (player, range) => {
        const original = await (await fetch(player.src)).arrayBuffer();
        const rate = new DataView(original).getUint32(24,true);
        const offset = ms => 44 + 2*Math.round(ms*rate/1000);
        const pcm = original.slice(offset(range[0]),offset(range[1]));
        return [...new Uint8Array(await crypto.subtle.digest('SHA-256',pcm))].map(x=>x.toString(16).padStart(2,'0')).join('');
      },[audio.session_start_ms,audio.session_end_ms]);
      assert.equal(sha(bytes.subarray(44)), hash);
      const viewer = await ctx.newPage();
      const remoteRequests = [];
      viewer.on('request', request => { if (/^https?:/.test(request.url())) remoteRequests.push(request.url()); });
      await viewer.goto(pathToFileURL(path.join(root,'static/issue-review.html')).href);
      await viewer.locator('#issue-file').setInputFiles(local);
      await viewer.locator('#view-result').waitFor();
      assert.equal(await viewer.locator('#issue-note-text').innerText(),note);
      assert.ok((await viewer.locator('#view-final mark').count())>0);
      await viewer.locator('#view-history > summary').click();
      assert.equal(await viewer.locator('#history-list li').count(),sample.revisions.length);
      await viewer.waitForFunction(()=>document.querySelector('#issue-audio').duration>0);
      assert.ok(Math.abs(await viewer.locator('audio').evaluate(a=>a.duration)-length/1000)<.02);
      await viewer.setViewportSize({width:375,height:750});
      assert.ok(await viewer.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      if (values.screenshot && engine === 'chromium') {
        await viewer.screenshot({path:path.join(path.dirname(values.screenshot),'teacher-issue-reader.png')});
      }
      await viewer.locator('#issue-file').setInputFiles({name:'wrong.json',mimeType:'application/json',buffer:Buffer.from('{}')});
      await viewer.locator('#view-error').getByText('فایل گزارش هم‌خوان نسخهٔ ۱ را انتخاب کنید.').waitFor();
      assert.ok(await viewer.locator('#view-result').isHidden());
      assert.deepEqual(remoteRequests,[]);
      assert.deepEqual(pageErrors,[]);
      report.push({engine,id,type,fixture_sha256:clip.sha256,excerpt_ms:Math.round(length),revisions:sample.revisions.length,
        exact_pcm:true,offline_reader:true,page_errors:pageErrors.length});
    } finally { await browser.close(); }
  }
  await fs.writeFile(values.output, JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report));
} finally {
  await fs.rm(temp,{recursive:true,force:true});
}
