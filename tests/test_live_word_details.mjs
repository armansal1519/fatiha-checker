// Execute the real popover renderer with a minimal DOM; no model/browser claim.
import {strict as assert} from 'node:assert';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {runInNewContext} from 'node:vm';

const liveSource = await readFile(new URL('../static/live.js',import.meta.url),'utf8');
const stateSource = await readFile(new URL('../static/transcript-state.js',import.meta.url),'utf8');
const {TranscriptState} = await import(`data:text/javascript;base64,${Buffer.from(stateSource).toString('base64')}`);
const labels = liveSource.match(/^const typeLabels = .*;$/m)?.[0];
const paragraph = liveSource.slice(liveSource.indexOf('function detailParagraph('),liveSource.indexOf('function openDetail('));
const popover = liveSource.slice(liveSource.indexOf('function showWord('),liveSource.indexOf('function showGap('));
assert.ok(labels && paragraph && popover,'The actual word-detail renderer must be present.');

function element() {
  return {textContent:'',children:[],dataset:{},hidden:false,
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; }};
}
function renderWord(state,id) {
  const nodes = new Map();
  const $ = name=>{ if (!nodes.has(name)) nodes.set(name,element()); return nodes.get(name); };
  const context = {$,transcript:state,document:{createElement:element},fa:value=>String(value),
    selectedWord:null,selectedGap:null,openDetail:()=>{}};
  runInNewContext(`${labels}\n${paragraph}\n${popover}\nshowWord(${JSON.stringify(id)},false);`,context);
  return {nodes,rows:$('word-explanation').children.map(row=>row.children.map(node=>node.textContent).join(''))};
}

test('swapped word popover keeps the exact heard text and shows its full expected word',()=>{
  const state = new TranscriptState();
  state.applyTranscript({replace_from:0,committed_count:2,tokens:[
    {id:'first',text:' نستعين',start_ms:0,end_ms:900},
    {id:'second',text:' نعبد',start_ms:1000,end_ms:1800}]});
  state.applyFeedback({updates:[{id:'first',state:'difference',text:'نعبد',
    reference:{surah:1,ayah:5,word_index:1,word:'نَعْبُدُ'},
    errors:[{type:'transposition',heard:'نستعين',expected:'نَعْبُدُ',message:'این واژه در جایگاه دیگری خوانده شده است.'}]}],gaps:[]});
  const {nodes,rows} = renderWord(state,'first');
  assert.equal(nodes.get('word-title').textContent,' نستعين');
  assert.ok(rows.includes('شنیده‌شده:  نستعين'));
  assert.ok(rows.includes('مرجع: نَعْبُدُ'));
  assert.ok(rows.includes('جابه‌جایی کلمه: این واژه در جایگاه دیگری خوانده شده است.'));
  assert.ok(rows.includes('واژهٔ مورد انتظار: نَعْبُدُ'));
  assert.deepEqual(state.tokens.map(token=>token.text),[' نستعين',' نعبد']);
});

test('repeat, substitution and uncertain boundary have explicit Persian labels',()=>{
  const names = new Map([['repetition','تکرار'],['substitution','کلمهٔ متفاوت'],['boundary','مرز آوایی نامشخص']]);
  for (const [type,label] of names) {
    const state = new TranscriptState();
    state.applyTranscript({replace_from:0,committed_count:2,tokens:[
      {id:'first',text:' نعبد',start_ms:0,end_ms:900},
      {id:'repeat',text:' نعبد',start_ms:1000,end_ms:1900}]});
    state.applyFeedback({updates:[{id:'repeat',state:type==='boundary' ? 'uncertain' : 'difference',
      reference:{surah:1,ayah:5,word:'نَعْبُدُ'},errors:[{type,heard:'نعبد',expected:type==='repetition' ? '' : 'نَعْبُدُ',message:'توضیح همان اختلاف'}]}],gaps:[]});
    const {nodes,rows} = renderWord(state,'repeat');
    assert.equal(nodes.get('word-title').textContent,' نعبد');
    assert.ok(rows.includes(`${label}: توضیح همان اختلاف`));
    assert.ok(rows.includes('شنیده‌شده:  نعبد'));
    assert.equal(nodes.get('word-detail').dataset.state,type==='boundary' ? 'uncertain' : 'difference');
    assert.deepEqual(state.tokens.map(token=>token.text),[' نعبد',' نعبد']);
  }
});
