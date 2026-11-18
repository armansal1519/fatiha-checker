import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import vm from 'node:vm';

const read = name=>readFile(new URL('../static/'+name,import.meta.url),'utf8');
const src = await read('live.js');
const {TranscriptState} = await import('data:text/javascript;base64,'+Buffer.from(await read('transcript-state.js')).toString('base64'));
const section = (start,end)=>src.slice(src.indexOf(start),src.indexOf(end,src.indexOf(start)));
const implementation = [
  section('function referenceControls(','function setPhase('),
  section('function locationLabel(','function reset('),
  section('function accept(','function drawWave('),
  section('function applyManualReference()', "$('reference-apply').addEventListener")
].join('\n');

function element() {
  return {children:[], dataset:{}, attributes:{}, textContent:'', hidden:false,
    get childElementCount(){ return this.children.length; },
    setAttribute(key,value){ this.attributes[key]=value; },
    getAttribute(key){ return this.attributes[key]; },
    addEventListener(key,fn){ this[key]=fn; },
    append(node){ this.children.push(node); node.remove=()=>{this.children=this.children.filter(item=>item!==node);}; }
  };
}
function harness() {
  const nodes = new Map(), chosen = [], transcript = new TranscriptState();
  const $ = id=>{if(!nodes.has(id)) nodes.set(id,element()); return nodes.get(id);};
  const context = vm.createContext({$,document:{createElement:element,activeElement:null},transcript,
    surahs:[],catalogueReady:true,generation:1,phase:'recording',hasDisplayLocation:false,lastReference:null,
    referenceFetch:0,reviewedThrough:0,manualReferenceApplied:false,audioActivity:null,
    fillAyahs:ayah=>{$('start-ayah').value=String(ayah);},
    connection:{setReference:(s,a)=>chosen.push([s,a])},active:()=>context.phase==='recording',
    fa:n=>String(n),issueHistory:{record(){}},markPaint(){},paint(){},summary(){},follow(){},
    streamStatus(){},status(){},finish(){},error:e=>assert.fail(e),fail:e=>assert.fail(e)});
  vm.runInContext(implementation,context);
  const accept=(e,run=1)=>{context.e=e;vm.runInContext('accept(e,'+run+')',context);};
  return {$,context,chosen,transcript,accept};
}
const ref=(surah,ayah,surah_name,text)=>({surah,ayah,surah_name,text});
const a=ref(112,3,'الإخلاص','لَمْ يَلِدْ وَلَمْ يُولَدْ');
const b=ref(55,13,'الرحمن','فَبِأَيِّ آلَاءِ رَبِّكُمَا تُكَذِّبَانِ');
const result=(status,...candidates)=>({status,candidates,total_candidates:candidates.length});
const spoken={type:'transcript',replace_from:0,committed_count:0,tokens:[{id:'w',text:' ولم يولد'}]};
const feedback={type:'feedback',updates:[],gaps:[]};

test('first raw words name a probable verse before grading and explicit none clears a revised location',()=>{
  const app=harness(), location=app.$('transcript-location');
  app.accept({...spoken,display_location:result('probable',a)});
  assert.match(location.textContent,/احتمالی.*الإخلاص.*3/);
  assert.equal(app.$('reference-text').textContent,a.text);
  app.accept({...feedback,reference:b});
  assert.match(location.textContent,/الإخلاص/);
  app.accept({...feedback,reference:b,display_location:result('none')});
  assert.doesNotMatch(location.textContent,/الإخلاص|الرحمن/);
  assert.equal(app.$('stage-position').dataset.active,'false');
  app.accept({...spoken,display_location:result('confirmed',a)},0);
  assert.doesNotMatch(location.textContent,/الإخلاص/);
  assert.deepEqual(app.transcript.tokens.map(w=>[w.text,w.state]),[[' ولم يولد','pending']]);
});

test('ambiguous choices stay keyboard-stable and constrain the existing manual reference command',()=>{
  const app=harness(), group=app.$('location-choices');
  const ambiguous={status:'ambiguous',total_candidates:31,candidates:[b,{...b,ayah:16},{...b,ayah:18},{...b,ayah:21}]};
  app.accept({...spoken,display_location:ambiguous});
  assert.match(app.$('transcript-location').textContent,/31.*3/);
  const focused=group.children[1]; app.context.document.activeElement=focused;
  app.accept({...feedback,display_location:ambiguous});
  assert.equal(group.children.length,3); assert.equal(group.children[1],focused);
  focused.click(); assert.deepEqual(app.chosen,[[55,16]]);
  app.accept({...feedback,display_location:result('manual',{...b,ayah:16})});
  assert.match(app.$('transcript-location').textContent,/انتخاب دستی.*16/);
  assert.equal(group.children[0],focused); assert.equal(focused.getAttribute('aria-disabled'),'true');
  focused.click(); assert.equal(app.chosen.length,1);
  focused.onblur(); assert.equal(group.hidden,true);
  assert.equal(app.transcript.tokens[0].state,'pending');
});

test('post-session selection sets the next start and leaves the completed session ambiguous',()=>{
  const app=harness(); app.context.phase='review';
  app.accept({...spoken,display_location:result('ambiguous',a,b)});
  const before=app.$('transcript-location').textContent;
  app.$('location-choices').children[1].click();
  assert.deepEqual(app.chosen,[]);
  assert.equal(app.$('reference-auto').checked,false); assert.equal(app.$('start-ayah').value,'13');
  assert.match(app.$('location-next').textContent,/13.*تمرین بعدی/);
  assert.equal(app.$('transcript-location').textContent,before);
});

test('delayed legacy reference fetch cannot overwrite a newer confirmed display or resurrect an ambiguity',async()=>{
  const app=harness(), waits=[];
  app.context.fetch=()=>new Promise(resolve=>waits.push(resolve));
  const update=e=>{app.context.e=e;return vm.runInContext('updateReference(e)',app.context);};
  const legacy=update({...feedback,reference:{surah:1,ayah:2}});
  await update({type:'done',display_location:result('confirmed',a)});
  waits.shift()({ok:true,json:async()=>({ayahs:[{text:'old'}]})}); await legacy;
  assert.match(app.$('transcript-location').textContent,/تطبیق آوایی.*الإخلاص/);
  assert.equal(app.$('reference-text').textContent,a.text);
  const pending=update({...feedback,display_location:result('probable',{surah:1,ayah:1})});
  await update({...feedback,display_location:result('ambiguous',a,b)});
  waits.shift()({ok:true,json:async()=>({ayahs:[{text:'old'}]})}); await pending;
  assert.notEqual(app.$('reference-text').textContent,'old'); assert.equal(app.$('reference-location').textContent,'');
});
