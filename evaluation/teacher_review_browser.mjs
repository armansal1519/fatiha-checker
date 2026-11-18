// Verify the actual distributable HTML offline, including native audio and downloaded answers.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';

const {values} = parseArgs({options: {
  file: {type:'string', default:'reports/hamkhan-teacher-review.html'},
  browser: {type:'string', default:'chromium,webkit'},
  output: {type:'string', default:'reports/teacher-review-browser.json'},
  screenshots: {type:'string'},
}});
const engines = createRequire(import.meta.url)(process.env.PLAYWRIGHT_CORE_PATH || 'playwright-core');
const file = pathToFileURL(path.resolve(values.file)).href;
const report = [];
for (const name of values.browser.split(',')) {
  const browser = await engines[name].launch();
  try {
    // Playwright WebKit's offline toggle rejects even file:// navigation.
    // Deny all HTTP(S) on both engines, plus Chromium's offline toggle.
    const makeContext = async viewport => {
      const c = await browser.newContext({offline:name !== 'webkit', viewport, acceptDownloads:true});
      await c.route(/^https?:\/\//, route => route.abort('internetdisconnected'));
      return c;
    };
    const ctx = await makeContext({width:1000,height:850});
    const page = await ctx.newPage(), external = [], errors = [];
    page.on('request', r => { if (/^https?:/.test(r.url())) external.push(r.url()); });
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(file);
    await page.waitForFunction(() => document.querySelector('#audio').readyState >= 1);
    assert.equal(await page.locator('#jump option').count(), 43);
    const words = [];
    for (let i = 0; i < 43; i++) {
      await page.locator('#jump').selectOption({index:i});
      await page.waitForFunction(() => Number.isFinite(document.querySelector('#audio').duration));
      words.push(await page.locator('#verse mark').innerText());
    }
    assert.ok(words.every(Boolean));
    await page.locator('#jump').selectOption('R01');
    await page.locator('input[value=difference]').check();
    await page.locator('#heard').fill('ه');
    await page.locator('#expected').fill('ح');
    await page.locator('#timing > summary').click();
    await page.locator('#start').fill('۰٫۱');
    await page.locator('#end').fill('۰٫۴');
    await page.locator('#listen-range').click();
    await page.waitForFunction(() => {
      const a = document.querySelector('#audio'); return a.paused && Math.abs(a.currentTime-.4)<.04;
    });
    await page.locator('#note').fill('بررسی مستقل');
    await page.locator('#next').click();
    await page.locator('input[value=uncertain]').check();
    const download = page.waitForEvent('download');
    await page.locator('#save').click();
    const downloaded = await download;
    assert.match(downloaded.suggestedFilename(), /^hamkhan-teacher-answers-/);
    const bytes = await fs.readFile(await downloaded.path());
    const payload = JSON.parse(bytes);
    assert.deepEqual(payload.answers.map(r => r.verdict),['difference','uncertain']);
    assert.deepEqual([payload.answers[0].start_s,payload.answers[0].end_s], [.1,.4]);
    const restore = await makeContext({width:390,height:844});
    const page2 = await restore.newPage();
    page2.on('request', r => { if (/^https?:/.test(r.url())) external.push(r.url()); });
    page2.on('pageerror', e => errors.push(e.message));
    await page2.goto(file);
    await page2.locator('#restore').setInputFiles({name:'answers.json',mimeType:'application/json',buffer:bytes});
    await page2.waitForFunction(() => document.querySelector('#jump').value==='R03');
    await page2.locator('#jump').selectOption('R01');
    assert.equal(await page2.locator('#heard').inputValue(),'ه');
    assert.equal(await page2.locator('#note').inputValue(),'بررسی مستقل');
    assert.equal(await page2.locator('input[value=difference]').isChecked(),true);
    await page2.locator('#restore').setInputFiles({name:'wrong.json',mimeType:'application/json',
      buffer:Buffer.from(JSON.stringify({...payload, packet_id:'incorrect'}))});
    await page2.locator('#message.error').waitFor();
    assert.equal(await page2.locator('#note').inputValue(),'بررسی مستقل');
    assert.ok(await page2.evaluate(() => document.documentElement.scrollWidth<=innerWidth+1));
    if (values.screenshots) {
      await fs.mkdir(values.screenshots,{recursive:true});
      await page2.locator('#jump').selectOption('R03');
      await page.screenshot({path:path.join(values.screenshots,name+'-desktop.png')});
      await page2.screenshot({path:path.join(values.screenshots,name+'-phone.png')});
    }
    assert.deepEqual(errors,[]);
    assert.deepEqual(external,[]);
    report.push({browser:name,version:browser.version(),http_blocked:true,offline_toggle:name !== 'webkit',tasks_loaded:words.length,
      range_played:[.1,.4],answers_roundtrip:true,wrong_packet_rejected:true,external_requests:external});
    console.log(name,'passed',words.length,'offline listening tasks');
  } finally { await browser.close(); }
}
await fs.writeFile(values.output, JSON.stringify(report,null,2)+'\n');
