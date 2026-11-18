// Browser/UI-only stress test. Intercept every stream and acknowledge a short
// generated WAV locally; fabricated transcript times are not model output.
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';

function silence(seconds) {
  const rate=16000, length=seconds*rate*2, data=Buffer.alloc(44+length);
  data.write('RIFF'); data.writeUInt32LE(36+length,4); data.write('WAVEfmt ',8);
  data.writeUInt32LE(16,16); data.writeUInt16LE(1,20); data.writeUInt16LE(1,22);
  data.writeUInt32LE(rate,24); data.writeUInt32LE(rate*2,28); data.writeUInt16LE(2,32);
  data.writeUInt16LE(16,34); data.write('data',36); data.writeUInt32LE(length,40);
  return data;
}

export async function stressTranscript(page,context,base,engine) {
  const count=3600, committed=count-1, batch=150, rounds=60;
  const session='ui-stress', vocab=['قل','هو','الله','أحد'];
  const word=(id,i)=>({id,text:' '+vocab[i%vocab.length],start_ms:i*500,end_ms:i*500+400});
  let channel,rate=0,revision=0,resolveAudio;
  const recordedEnough=new Promise(resolve=>{resolveAudio=resolve;});
  const publish=payload=>channel.send(JSON.stringify({session_id:session,...payload}));
  await page.routeWebSocket('**/live/ws',route=>{
    channel=route;
    route.onMessage(packet=>{
      if (typeof packet==='string') {
        const event=JSON.parse(packet);
        if (event.type==='start') {rate=event.sample_rate;publish({type:'ready',version:2});}
        else if (event.type==='stop') publish({type:'done',revision:++revision});
        return;
      }
      const end=packet.readUInt32LE(4)+(packet.length-8)/2;
      publish({type:'ack',input_samples:end,sequence:packet.readUInt32LE(0)});
      if (end/rate>=4.5) resolveAudio();
    });
  });
  await page.addInitScript(()=>{
    const Native=WebSocket; window.stressTest={frames:[]};
    window.WebSocket=class extends Native {
      constructor(...args) {
        super(...args); this.addEventListener('message',({data})=>{
          if (typeof data!=='string') return;
          const payload=JSON.parse(data); if (!payload.qaRound) return;
          const start=performance.now();
          requestAnimationFrame(()=>requestAnimationFrame(()=>{
            window.stressTest.frames.push({round:payload.qaRound,ms:performance.now()-start});
          }));
        });
      }
    };
  });
  const cdp=engine==='chromium'?await context.newCDPSession(page):null;
  const heap=async()=>cdp?{...await cdp.send('Runtime.getHeapUsage'),...await cdp.send('Memory.getDOMCounters')}:null;
  await page.setViewportSize({width:1200,height:850}); await page.goto(new URL('/live',base).href);
  await page.locator('#auto-follow').uncheck();
  await page.locator('.file-practice > summary').click();
  await page.locator('#live-file').setInputFiles({name:'empty-test.wav',mimeType:'audio/wav',buffer:silence(35)});
  await page.locator('#file-play').click(); await page.waitForFunction(()=>document.body.dataset.phase==='file');
  assert.equal(await page.evaluate(()=>window.autoFollowReveals.length),0,'Opted-out sessions do not auto-scroll on start.');
  const before=await heap(), seedStart=performance.now();
  for (let i=0;i<count;i+=batch) {
    const end=Math.min(count,i+batch);
    publish({type:'transcript',revision:++revision,replace_from:i,committed_count:Math.min(end,committed),
      tokens:Array.from({length:end-i},(_,j)=>word('w'+(i+j),i+j))});
    await page.waitForFunction(n=>document.getElementById('heard-transcript').childElementCount===n,end);
  }
  const seedMs=Math.round(performance.now()-seedStart), seeded=await heap();
  await page.locator('[data-token-id=w3]').focus();
  await page.evaluate(()=>{
    const box=document.getElementById('heard-transcript'), state=window.stressTest;
    state.prefix=new Map([...box.querySelectorAll('[data-token-id]')].slice(0,-1).map(el=>[el.dataset.tokenId,el]));
    state.mutations={changedExpected:0,unrelatedPrefix:0,removedStableNodes:0,totalRecords:0,stableRawTextRewrites:0};
    state.observer=new MutationObserver(records=>{
      for (const mutation of records) {
        state.mutations.totalRecords++;
        const target=mutation.target.nodeType===1?mutation.target:mutation.target.parentElement;
        const owner=target?.closest('[data-token-id]')?.dataset.tokenId;
        if (state.prefix.has(owner)) {
          if (owner==='w3' || owner==='w18') state.mutations.changedExpected++;
          else state.mutations.unrelatedPrefix++;
          if (mutation.type==='characterData' || mutation.type==='childList') state.mutations.stableRawTextRewrites++;
        }
        for (const node of mutation.removedNodes) if (node.nodeType===1 && state.prefix.has(node.dataset.tokenId)) state.mutations.removedStableNodes++;
      }
    });
    state.observer.observe(box,{attributes:true,characterData:true,childList:true,subtree:true});
  });
  const gap={id:'lasting-gap',before_token_id:'w5',after_token_id:'w6',expected:'من'};
  const volatileGap={id:'old-gap',before_token_id:'w3598',after_token_id:'w3599',expected:'من'};
  const mark={type:'substitution',heard:'أحد',expected:'هو',message:'این کلمه در مرجع متفاوت است.'};
  publish({type:'feedback',revision:++revision,updates:[{id:'w3',state:'difference',errors:[mark]}],gaps:[gap,volatileGap]});
  await page.waitForFunction(()=>!!document.querySelector('[data-gap-id=old-gap]'));
  for (let i=1;i<=rounds;i++) {
    if (i%2) publish({type:'transcript',revision:++revision,qaRound:i,replace_from:committed,committed_count:committed,
      tokens:[word('tail-'+i,count-1)]});
    else publish({type:'feedback',revision:++revision,qaRound:i,
      updates:[{id:'w18',state:i%4?'uncertain':'match',errors:[]}],gaps:[gap]});
    await page.waitForFunction(n=>window.stressTest.frames.length>=n,i);
  }
  const after=await heap();
  const state=await page.evaluate(()=>{
    const s=window.stressTest; s.observer.disconnect();
    const elements=document.getElementById('heard-transcript');
    return {frames:s.frames.map(v=>v.ms),mutations:s.mutations,focus:document.activeElement.dataset.tokenId,
      wordCount:elements.querySelectorAll('[data-token-id]').length,singleTabStop:elements.querySelectorAll('button[tabindex="0"]').length,
      stableNodes:[...s.prefix].every(([id,el])=>el.isConnected && el.dataset.tokenId===id),
      gapRevoked:!elements.querySelector('[data-gap-id=old-gap]'),
      gapAfter:elements.querySelector('[data-gap-id=lasting-gap]').previousElementSibling.dataset.tokenId,
      gapBefore:elements.querySelector('[data-gap-id=lasting-gap]').nextElementSibling.dataset.tokenId};
  });
  assert.equal(state.mutations.unrelatedPrefix,0); assert.equal(state.mutations.removedStableNodes,0);
  assert.equal(state.mutations.stableRawTextRewrites,0); assert.ok(state.mutations.changedExpected>0);
  assert.equal(state.focus,'w3'); assert.equal(state.wordCount,count); assert.equal(state.singleTabStop,1);
  assert.ok(state.stableNodes && state.gapRevoked); assert.equal(state.gapAfter,'w5'); assert.equal(state.gapBefore,'w6');
  await recordedEnough; await page.locator('#live-stop').click();
  await page.waitForFunction(()=>document.body.dataset.phase==='review' && document.getElementById('replay-audio').duration>4);
  const replayRange=async(id)=>{
    await page.locator(id).click(); await page.locator('#segment-play').click();
    await page.waitForFunction(()=>!document.getElementById('replay-audio').paused);
    const start=await page.locator('#replay-audio').evaluate(a=>a.currentTime);
    await page.waitForFunction(()=>document.getElementById('replay-audio').paused);
    const end=await page.locator('#replay-audio').evaluate(a=>a.currentTime);
    await page.keyboard.press('Escape'); return {start,end};
  };
  const wordRange=await replayRange('[data-token-id=w3]'), gapRange=await replayRange('[data-gap-id=lasting-gap]');
  assert.ok(wordRange.start>=1.16 && wordRange.start<1.48 && wordRange.end>=2.2 && wordRange.end<2.55);
  assert.ok(gapRange.start>=2.16 && gapRange.start<2.48 && gapRange.end>=3.7 && gapRange.end<3.98);
  const sorted=[...state.frames].sort((a,b)=>a-b);
  return {source:'Generated browser-only events; real DOM and local generated WAV. No acoustic model or 30-minute elapsed claim.',
    syntheticTranscriptSeconds:count/2,wordCount:count,rounds,seedMs,
    eventToTwoAnimationFrames:{p50ms:sorted[Math.ceil(rounds*.5)-1],p95ms:sorted[Math.ceil(rounds*.95)-1],maxMs:sorted.at(-1)},
    memory:{source:cdp?'Chromium renderer CDP heap and DOM counters; no forced GC':'WebKit engine does not expose a corresponding CDP heap API',before,seeded,after},
    ...state,wordRange,gapRange};
}
