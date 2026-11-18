import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../static/live.js',import.meta.url),'utf8');
const save = source.slice(source.indexOf('async function saveIssue()'),source.indexOf('function enforceSegment()'));

function ui() {
  let resolve, downloads=0, options;
  const nodes=new Map(), objects=new Map();
  const node=id=>{
    if (!nodes.has(id)) nodes.set(id,{value:'یادداشت',hidden:true});
    return nodes.get(id);
  };
  const ctx=vm.createContext({$:node,issueDraft:{anchor:{}},issueSaving:false,generation:1,phase:'error',
    sessionReplay:{blob:new Blob()},transcript:{tokens:[]},issueHistory:{},Blob,
    paintIssue(){},setTimeout(){},
    URL:{createObjectURL(blob){objects.set('blob:sample',blob);return 'blob:sample';}},
    document:{createElement:()=>({click(){downloads++;},remove(){}}),body:{append(){}}},
    buildIssueSample(args){options=args;return new Promise(done=>{resolve=done;});}});
  vm.runInContext(save,ctx);
  return {ctx,node,objects,resolve:()=>resolve({schema:'hamkhan.issue.v1'}),downloadCount:()=>downloads,
    options:()=>options,save:()=>vm.runInContext('saveIssue()',ctx)};
}

test('an explicit partial-session export is single-flight and reveals the offline viewer after save',async()=>{
  const h=ui(), pending=h.save();
  await h.save(); assert.ok(h.ctx.issueSaving); assert.equal(h.downloadCount(),0);
  assert.equal(h.options().partial,true); assert.equal(h.options().note,'یادداشت');
  h.resolve(); await pending;
  assert.equal(h.downloadCount(),1); assert.equal(h.ctx.issueSaving,false);
  assert.equal(h.node('issue-viewer').hidden,false);
  assert.equal(JSON.parse(await h.objects.get('blob:sample').text()).schema,'hamkhan.issue.v1');
});

test('reset or replacing the selected occurrence invalidates an in-flight download',async()=>{
  for (const invalidate of [ctx=>ctx.generation++,ctx=>{ctx.issueDraft=null;}]) {
    const h=ui(), pending=h.save(); invalidate(h.ctx); h.resolve(); await pending;
    assert.equal(h.downloadCount(),0); assert.equal(h.node('issue-viewer').hidden,true);
  }
});

test('a running session or missing recording cannot export',async()=>{
  const h=ui();h.ctx.phase='recording';await h.save();assert.equal(h.options(),undefined);
  h.ctx.phase='review';h.ctx.sessionReplay=null;await h.save();assert.equal(h.options(),undefined);
});
