import {strict as assert} from 'node:assert';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../static/transcript-state.js',import.meta.url),'utf8');
const {TranscriptState} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const word=(id,text,start_ms=0)=>({id,text,start_ms,end_ms:start_ms+200});

test('heard typo and repeated words remain raw text when canonical feedback arrives',()=>{
  const state=new TranscriptState();
  state.applyTranscript({replace_from:0,tokens:[word('a',' الهمد'),word('b',' الهمد',300)],committed_count:2});
  state.applyFeedback({updates:[{id:'a',text:'الحمد',state:'difference',reference:{word:'ٱلْحَمْدُ'},errors:[{type:'letter',message:'حرف متفاوت'}]}],gaps:[]});
  assert.deepEqual(state.tokens.map(x=>x.text),[' الهمد',' الهمد']);
  assert.equal(state.tokens[0].state,'difference'); assert.equal(state.tokens[1].state,'pending');
});

test('mutable tail replacement revokes old annotations and gap anchors',()=>{
  const state=new TranscriptState();
  state.applyTranscript({replace_from:0,tokens:[word('a','إياك'),word('b','نعبد',300)],committed_count:1});
  state.applyFeedback({updates:[{id:'b',state:'difference',errors:[{type:'letter'}]}],gaps:[{id:'gap',before_token_id:'a',after_token_id:'b',expected:'شيء'}]});
  state.applyTranscript({replace_from:1,tokens:[word('c','نعمل',300)],committed_count:1});
  assert.equal(state.tokens[1].state,'pending'); assert.deepEqual(state.tokens[1].errors,[]); assert.deepEqual(state.gaps,[]);
  state.applyFeedback({updates:[{id:'b',state:'match',errors:[]}],gaps:[]});
  assert.equal(state.tokens[1].id,'c'); assert.equal(state.tokens[1].state,'pending');
});

test('late metadata can annotate committed text without changing it',()=>{
  const state=new TranscriptState(); state.applyTranscript({replace_from:0,tokens:[word('a','رب'),word('b','العالمين',300)],committed_count:2});
  state.applyFeedback({updates:[{id:'a',state:'match',reference:{surah:1,ayah:2,word:'رَبِّ'}}],gaps:[]});
  assert.equal(state.tokens[0].text,'رب'); assert.equal(state.tokens[0].state,'match');
  assert.equal(state.committedCount,2);
  assert.throws(()=>state.applyTranscript({replace_from:0,tokens:[],committed_count:0}));
});

test('gaps are annotations rather than fabricated heard words; snapshots revoke them',()=>{
  const state=new TranscriptState(); state.applyTranscript({replace_from:0,tokens:[word('a','إياك'),word('b','وإياك',300)],committed_count:2});
  state.applyFeedback({updates:[],gaps:[{id:'g',before_token_id:'a',after_token_id:'b',expected:'نعبد'}]});
  assert.deepEqual(state.tokens.map(x=>x.text),['إياك','وإياك']);
  assert.equal(state.summary().missing,1); assert.equal(state.summary().uncertain,2);
  state.applyFeedback({updates:[],gaps:[]}); assert.equal(state.summary().missing,0);
});

test('summary counts only heard words and rejects duplicate IDs transactionally',()=>{
  const state=new TranscriptState(); state.applyTranscript({replace_from:0,tokens:[word('a','قل')],committed_count:0});
  state.applyFeedback({updates:[{id:'a',state:'difference',errors:[]}],gaps:[]});
  assert.deepEqual(state.summary(),{reviewed:1,differences:1,uncertain:0,missing:0});
  assert.throws(()=>state.applyTranscript({replace_from:1,tokens:[word('a','قل')],committed_count:0}));
  assert.equal(state.tokens.length,1); assert.equal(state.tokens[0].text,'قل');
});

test('HTML-like recognition remains a plain string and feedback has no text overwrite path',()=>{
  const state=new TranscriptState(); state.applyTranscript({replace_from:0,tokens:[word('a','<img src=x>')],committed_count:1});
  state.applyFeedback({updates:[{id:'a',text:'canonical',state:'uncertain',errors:[]}],gaps:[]});
  assert.equal(state.tokens[0].text,'<img src=x>');
});

test('a reused ID in a revised tail cannot keep an old gap verdict',()=>{
  const state=new TranscriptState(); state.applyTranscript({replace_from:0,tokens:[word('a','قل'),word('b','أحد',300)],committed_count:1});
  state.applyFeedback({updates:[{id:'b',state:'difference',errors:[{type:'word'}]}],gaps:[{id:'g',before_token_id:'a',after_token_id:'b',expected:'هو'}]});
  state.applyTranscript({replace_from:1,tokens:[word('b','هو',300)],committed_count:1});
  assert.equal(state.tokens[1].text,'هو'); assert.equal(state.tokens[1].state,'pending'); assert.deepEqual(state.gaps,[]);
});
