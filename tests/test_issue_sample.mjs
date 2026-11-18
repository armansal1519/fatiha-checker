import assert from 'node:assert/strict';
import test from 'node:test';
import {IssueHistory,issueAnchor,wordIssueAnchor,issueBounds,buildIssueSample} from '../static/issue-sample.js';
import {SessionRecording} from '../static/live-source.js';

const word = (id,text,start_ms,end_ms) => ({id,text,start_ms,end_ms});
const delta = (revision,replace_from,...tokens) => ({revision,replace_from,tokens});
const visible = (revision) => revision.words.map(w => w.text);

test('scoped revisions retain disappearance, reappearance and distinct timed duplicates',()=>{
  const h = new IssueHistory();
  const before = word('begin','other private utterance',0,200);
  const one = word('w1',' والضحى',2500,3000), two = word('w2',' والضحى',3100,3900);
  const after = word('after','other private text',9000,9300);
  h.record(delta(2,0,before,one,two));
  h.record(delta(3,1,one));
  h.record(delta(4,1));
  h.record(delta(7,1,one,two,after));
  h.record(delta(8,4));
  const {revisions,complete} = h.excerpt(2200,4100);
  assert.equal(complete,true);
  assert.deepEqual(revisions.map(r => r.revision),[2,3,4,7]);
  assert.deepEqual(revisions.map(visible),[[one.text,two.text],[one.text],[],[one.text,two.text]]);
  assert.deepEqual(revisions[0].words,[one,two]);
  assert.doesNotMatch(JSON.stringify(revisions),/private/);
});

test('history uses bounded patches, reports expiration and resets with the session',()=>{
  const h = new IssueHistory(4,6);
  for (let i=0;i<1800;i++) h.record(delta(i,0,word('a','a',i,i+50),word('b','b',i+50,i+99)));
  assert.ok(h.entries.length-h.head<=3 && h.savedWords<=6);
  assert.equal(h.excerpt(1790,2000).complete,false);
  h.clear();
  assert.deepEqual(h.excerpt(0,1800000),{complete:true,revisions:[]});
  h.record(delta(1,0,...Array.from({length:7},(_,i)=>word(String(i),'oversize',0,1))));
  assert.deepEqual(h.excerpt(0,10),{complete:false,revisions:[]});
});

test('bounded export keeps earliest and latest displayed states when revisions are dense',()=>{
  const h=new IssueHistory();
  for (let i=1;i<=120;i++) h.record(delta(i,0,word('w',String(i),0,300)));
  const sampled=h.excerpt(0,1200);
  assert.equal(sampled.complete,false);
  assert.equal(sampled.revisions.length,96);
  assert.equal(sampled.revisions[0].revision,1);
  assert.equal(sampled.revisions.at(-1).revision,120);
});

for (const rate of [16000,44100,48000]) test(`PCM excerpt exports exact sample boundaries at ${rate} Hz`,async()=>{
  const recording=new SessionRecording(rate);
  const values=Float32Array.from({length:rate*9},(_,i)=>i%2 ? .5 : -.5);
  recording.append(values);
  const initial = word('seen','خَلَق',3800,4000);
  const revised = {...initial, text:' خيرا', state:'difference',
    reference:{surah:105,ayah:3,word_index:2,word:'طيرا',unused:'excluded'},
    errors:[{type:'letter',heard:'خ',expected:'ط',message:'اختلاف حرف',unneeded:'excluded'}]};
  const h = new IssueHistory();
  h.record(delta(6,0,initial)); h.record(delta(9,0,revised));
  const report=await buildIssueSample({recording:recording.blob(),anchor:issueAnchor('word',[initial]),
    history:h,tokens:[word('x','excluded',8100,8300),revised],note:'  این را خواندم  ',partial:true});
  const pcm=Buffer.from(report.audio.data,'base64');
  assert.equal(report.schema,'hamkhan.issue.v1'); assert.equal(report.session_status,'partial');
  assert.equal(report.user_note,'این را خواندم');
  assert.equal(report.audio.session_start_ms,2300); assert.equal(report.audio.session_end_ms,5500);
  assert.equal(report.audio.sample_rate,rate); assert.equal(pcm.readUInt32LE(24),rate);
  assert.equal(pcm.length,44+2*rate*3.2);
  assert.equal(pcm.readUInt32LE(4),pcm.length-8); assert.equal(pcm.readUInt32LE(40),pcm.length-44);
  assert.equal(pcm.readInt16LE(44),-16384); assert.equal(pcm.readInt16LE(46),16383);
  assert.deepEqual(report.selection.initial_words,[initial]);
  assert.deepEqual(report.final_words[0].errors,[{type:'letter',heard:'خ',expected:'ط',message:'اختلاف حرف'}]);
  assert.doesNotMatch(JSON.stringify([report.revisions,report.selection,report.final_words]),/excluded/);
});

test('a gap includes both audible anchors, omission text is isolated, playhead works without ASR',async()=>{
  const left=word('b','ب',1500,1800), right=word('a','ا',2720,3390);
  const gap=issueAnchor('gap',[left,right]); gap.expected='ج';
  const recording=new SessionRecording(16000); recording.append(new Float32Array(16000*4));
  const options={recording:recording.blob(),history:new IssueHistory(),tokens:[left,right]};
  const report=await buildIssueSample({...options,anchor:gap});
  assert.equal(report.audio.session_start_ms,0); assert.equal(report.audio.session_end_ms,4000);
  assert.deepEqual(report.selection.token_ids,['b','a']); assert.equal(report.selection.expected,'ج');
  assert.deepEqual(report.final_words.map(w=>w.text),['ب','ا']);
  const empty=await buildIssueSample({...options,anchor:issueAnchor('playhead',[],4000),tokens:[]});
  assert.deepEqual(empty.final_words,[]); assert.equal(empty.audio.session_end_ms,4000);
  assert.equal(empty.selection.end_ms,4000);
  assert.equal(issueAnchor('word',[word('x','x',null,1200)]),null);
  assert.equal(issueAnchor('gap',[left,undefined]),null);
  assert.equal(issueAnchor('playhead',[],NaN),null);
  assert.deepEqual(issueBounds(issueAnchor('gap',[left,{...right,start_ms:1700000,end_ms:1710000}]),1800000),
    {start:851750,end:859750});
  await assert.rejects(buildIssueSample({...options,anchor:issueAnchor('playhead',[],10000)}));
  await assert.rejects(buildIssueSample({...options,anchor:gap,recording:new Blob([])}));
});

test('an explicitly selected untimed token keeps unknown timestamps and uses an adjacent clock',async()=>{
  const left=word('l','قول',800,1200), target=word('u',' قناة',null,null);
  const privateWord=word('outside','not selected',null,null);
  const tokens=[left,target,privateWord];
  const anchor=wordIssueAnchor(tokens,1);
  assert.equal(anchor.timing_source,'neighbor'); assert.equal(anchor.start_ms,1200);
  assert.equal(wordIssueAnchor([target],0),null);
  const history=new IssueHistory(); history.record(delta(4,0,...tokens));
  const recording=new SessionRecording(16000); recording.append(new Float32Array(16000*3));
  const report=await buildIssueSample({recording:recording.blob(),history,anchor,tokens});
  assert.equal(report.selection.initial_words[0].start_ms,null);
  assert.deepEqual(visible(report.revisions[0]),[left.text,target.text]);
  assert.deepEqual(report.final_words.map(w=>w.id),['l','u']);
  assert.equal(report.selection.timing_source,'neighbor');
});
