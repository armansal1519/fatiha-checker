import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const src=await readFile(new URL('../static/live.js',import.meta.url),'utf8');
const impl=src.slice(src.indexOf('function revealReadingOnStart'),src.indexOf('async function begin'));
test('session start reveals the heard sheet once, and honours later input, disabled follow, and stale sessions',()=>{
  for (const settings of [{expect:1},{off:true,expect:0},{scroll:true,expect:0},{stale:true,expect:0},{inactive:true,expect:0}]) {
    const frames=[],scrolls=[],ctx=vm.createContext({generation:1,manualScrollIntent:0,manualUntil:5000,autoScrollUntil:0,
      performance:{now:()=>1000},active:()=>!settings.inactive,requestAnimationFrame:f=>frames.push(f),
      $:id=>id==='auto-follow'?{checked:!settings.off}:{scrollIntoView:options=>scrolls.push(options)}});
    vm.runInContext(impl+';revealReadingOnStart(1,0);',ctx);
    if (settings.scroll) vm.runInContext('userScrolled()',ctx);
    if (settings.stale) ctx.generation=2;
    frames.shift()();
    assert.equal(scrolls.length,settings.expect);
    if (settings.expect) {assert.equal(ctx.manualUntil,0);assert.equal(scrolls[0].behavior,'instant');}
    if (settings.scroll) assert.equal(ctx.manualUntil,6000);
  }
});
