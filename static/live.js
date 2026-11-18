import {segmentBounds,unreviewedRecording} from './review-audio.js';
import {readAudioFile,decodePracticeBlob} from './live-audio.js';
import {prepareAudioSource} from './live-source.js';
import {LiveConnection} from './live-wire.js';
import {TranscriptState} from './transcript-state.js';
import {IssueHistory,issueAnchor,wordIssueAnchor,issueBounds,buildIssueSample} from './issue-sample.js';

const $ = id=>document.getElementById(id), fa = value=>String(value).replace(/\d/g,d=>'۰۱۲۳۴۵۶۷۸۹'[d]);
let ready = document.body.dataset.ready === 'true';
const transcript = new TranscriptState();
const issueHistory = new IssueHistory();
let issueDraft = null, issueSaving = false;
let generation = 0, phase = 'idle', capture = null, connection = null, controller = null;
let sessionReplay = null, replayURL = null, chosenFile = null, selectedWord = null, selectedGap = null, lastReference = null;
let manualScrollIntent = 0, manualUntil = 0, autoScrollUntil = 0, raf = null, lastCanvasFrame = 0, lastSummary = null;
let surahs = [], catalogueReady = false, referenceFetch = 0, lastAnnouncedCount = 0;
let audioActivity = null, inputWarning = '', lowInputSince = null, lastInputPaint = -Infinity, lastStreamStatus = '';
const wordNodes = new Map(), gapNodes = new Map();
let renderedOrder = [], renderedCommitted = 0, keyboardNode = null, lastReviewNode=null;
let segmentEnd = null, segmentTimer = 0, reviewedThrough = 0, healthRequest = 0, manualReferenceApplied=false;
const textPaints = [], feedbackPaints = [];

const measureUI = new URLSearchParams(location.search).has('measure');
let frameIntervals = [], controlPaints = [];
const active = ()=>['recording','file'].includes(phase);
const discardCapture = source=>{ if (source) void source.stop({discard:true}).catch(()=>{}); };
const labels = {pending:'هنوز بررسی نشده',uncertain:'شواهد نامشخص',match:'اختلافی پیدا نشد',difference:'اختلاف احتمالی؛ نیازمند مرور'};
const typeLabels = {letter:'حرف',vowel:'حرکت',gemination:'تشدید',deletion:'حذف',insertion:'اضافه',substitution:'کلمهٔ متفاوت',word:'کلمهٔ متفاوت',repetition:'تکرار',transposition:'جابه‌جایی کلمه',boundary:'مرز آوایی نامشخص',transcription:'رونویسی نامشخص'};

function status(text) { const node=$('live-status'); if (node.textContent!==text) node.textContent=text; }
function audioStatusMessage(fallback) {
  if (audioActivity?.status === 'no_speech') return `گفتار روشنی در این بخش پیدا نشد. ${inputWarning || (phase === 'file' ? 'بخش دارای گفتار فایل را بررسی کنید.' : 'اگر می‌خوانید، نزدیک‌تر به میکروفن و پیوسته بخوانید؛ ورودی میکروفن را هم بررسی کنید.')}`;
  if (audioActivity?.status === 'uncertain') return 'تشخیص گفتار این بخش قطعی نیست؛ برای رسیدن صدای واضح‌تر، کمی پیوسته‌تر بخوانید.';
  return inputWarning || fallback;
}
function streamStatus(text) { lastStreamStatus = text; status(audioStatusMessage(text)); }
function resetAudioDiagnostics() {
  audioActivity = null; inputWarning = ''; lowInputSince = null; lastInputPaint = -Infinity; lastStreamStatus = '';
  const stage = $('stage-voice'); stage.dataset.active = 'false'; stage.dataset.level = 'waiting';
  if (stage.lastChild) stage.lastChild.textContent = 'سطح ورودی';
  stage.title = 'بلندی ورودی صدا؛ این نشانگر، تشخیص گفتار یا تأیید قرائت نیست.';
  $('live-wave-empty').textContent = 'در انتظار ورودی صدا';
}
function paintInputDiagnostics(value, now) {
  if (now-lastInputPaint < 250) return; lastInputPaint = now;
  const rms = Number.isFinite(value?.rms) ? value.rms : 0, peak = Number.isFinite(value?.peak) ? value.peak : 0;
  const clipping = Number.isFinite(value?.clippedFraction) ? value.clippedFraction : 0;
  const hasInput = rms > .001, stage = $('stage-voice');
  stage.dataset.active = 'false'; // Amplitude must never look like a speech/correctness verdict.
  stage.dataset.level = clipping > .01 ? 'clipping' : hasInput ? 'present' : 'low';
  stage.title = `سطح ورودی: ${fa((20*Math.log10(Math.max(rms,1e-9))).toFixed(0))} dBFS؛ اوج: ${fa(peak.toFixed(3))}؛ برش: ${fa((clipping*100).toFixed(1))}٪. این مقادیر، تشخیص گفتار نیستند.`;
  $('live-wave-empty').hidden = hasInput && audioActivity?.status !== 'no_speech';
  $('live-wave-empty').textContent = audioActivity?.status === 'no_speech' ? 'در انتظار گفتار واضح' : 'در انتظار ورودی صدا';
  lowInputSince = hasInput ? null : lowInputSince ?? now;
  const previous = inputWarning, warm = (value?.windowSamples || 0) >= (value?.sampleRate || 16000)*.75;
  inputWarning = warm && clipping > .01 ? value?.sourceKind === 'file' ? 'سطح صدای فایل بیش از حد بلند است؛ نسخه‌ای با صدای بدون برش انتخاب کنید.' : 'صدای ورودی بیش از حد بلند است؛ کمی از میکروفن فاصله بگیرید یا سطح ورودی را کم کنید.'
    : warm && lowInputSince !== null && now-lowInputSince >= 2000 && value?.sourceKind !== 'file' ? 'صدای ورودی خیلی کم است؛ اگر می‌خوانید، به میکروفن نزدیک‌تر شوید و ورودی انتخاب‌شده را بررسی کنید.' : '';
  if (previous !== inputWarning && active()) status(audioStatusMessage(lastStreamStatus || 'در انتظار گفتار واضح‌تر…'));
}
function error(text = '') { $('live-error').textContent = text; $('live-error').hidden = !text; }
function referenceControls() {
  const auto = $('reference-auto').checked, unavailable = !catalogueReady || ['connecting','stopping'].includes(phase);
  $('reference-auto').disabled = unavailable;
  $('start-surah').disabled = unavailable || auto; $('start-ayah').disabled = unavailable || auto;
  $('reference-apply').disabled = unavailable || auto;
}
function setPhase(next) {
  if (measureUI) { const changed = performance.now(), run = generation; requestAnimationFrame(()=>{ if (run === generation) controlPaints.push(performance.now()-changed); }); }
  phase = next; document.body.dataset.phase = next;
  const locked = ['connecting','recording','file','stopping'].includes(next);
  $('live-start').disabled = !ready || locked;
  $('check-ready').hidden = locked || (ready && next!=='error');
  $('live-file').disabled = !ready || locked; $('file-play').disabled = !ready || locked || !chosenFile;
  $('live-stop').disabled = !locked || next === 'stopping';
  $('live-stop').textContent = next === 'connecting' ? 'انصراف' : 'پایان خواندن';
  $('live-stop-mobile').hidden = !locked; $('live-stop-mobile').disabled = next === 'stopping';
  $('live-stop-mobile').textContent = next === 'connecting' ? 'انصراف' : next === 'stopping' ? 'در حال جمع‌بندی…' : 'پایان خواندن';
  $('live-pill').textContent = ({idle:'آمادهٔ تمرین',connecting:'در حال آماده‌سازی',recording:'در حال شنیدن',file:'پخش و بررسی',stopping:'بررسی پایانی',review:'تمرین پایان یافت',error:'تمرین متوقف شد'})[next];
  if (next==='idle' && !ready) $('live-pill').textContent='مدل‌ها آماده نیستند';
  $('live-pill').classList.toggle('active',active());
  $('live-state-title').textContent = ({idle:'با صدای خودتان آغاز کنید',connecting:'یک لحظه تا شروع',recording:'صدایتان نوشته می‌شود',file:'همراه صدای شما',stopping:'در حال مرور آخرین کلمه‌ها',review:'این بار را مرور کنید',error:'آمادهٔ یک شروع دوباره'})[next];
  $('live-state-description').textContent = ({idle:'آرام و پیوسته بخوانید؛ متن از صدای شما نوشته می‌شود.',connecting:'اتصال و منبع صدا آماده می‌شوند.',recording:'انتهای متن موقت است؛ در پایان آیه کمی مکث کنید.',file:'همراه پخش فایل، متن و بازخورد می‌رسند.',stopping:'دریافت صدا تمام شده؛ نتیجهٔ آخر در راه است.',review:'کلمه‌های مشخص‌شده را انتخاب و دوباره مرور کنید.',error:'پیام زیر را بررسی کنید و دوباره شروع کنید.'})[next];
  $('word-retry').disabled = locked || !ready || !catalogueReady; referenceControls();
  paintIssue();
}
function reportUI() {
  if (!measureUI || !frameIntervals.length) return;
  const percentile = (items,p)=>[...items].sort((a,b)=>a-b)[Math.min(items.length-1,Math.floor(items.length*p))];
  console.info('demo-ui-metrics',JSON.stringify({wave_frames:frameIntervals.length,wave_interval_p95_ms:percentile(frameIntervals,.95),control_paint_max_ms:Math.max(0,...controlPaints),transcript_frame_p95_ms:percentile(textPaints,.95),feedback_frame_p95_ms:percentile(feedbackPaints,.95)}));
  frameIntervals = []; controlPaints = []; textPaints.length=feedbackPaints.length=0;
}
function clearReplay() {
  cancelSegment(); $('replay-audio').pause(); $('replay-audio').removeAttribute('src'); $('replay-audio').load();
  if (replayURL) URL.revokeObjectURL(replayURL);
  $('retry-tail')?.remove(); sessionReplay=null; replayURL = null; $('replay-section').hidden = true; $('replay-note').textContent='صدای همین نوبت، فقط در این صفحه.'; paintReplay();
}
function cancelSegment() { clearTimeout(segmentTimer); segmentTimer=0; segmentEnd=null; }
function selectedBounds() {
  const byId=id=>transcript.tokens[transcript.index.get(id)];
  if (selectedWord) return segmentBounds([byId(selectedWord)],$('replay-audio').duration);
  const gap=transcript.gaps.find(item=>item.id===selectedGap);
  return gap ? segmentBounds([byId(gap.before_token_id),byId(gap.after_token_id)],$('replay-audio').duration) : null;
}
function segmentButton() {
  $('segment-play').hidden=!selectedBounds();
  $('segment-play').textContent=selectedGap ? 'شنیدن دو سوی جای خالی' : 'شنیدن همین کلمه';
  $('issue-open').hidden=!selectedIssueAnchor();
}
function selectedIssueAnchor() {
  const byId = id => transcript.tokens[transcript.index.get(id)];
  if (selectedWord) return wordIssueAnchor(transcript.tokens,transcript.index.get(selectedWord));
  const gap = selectedGap && transcript.gaps.find(g => g.id === selectedGap);
  if (!gap) return null;
  const anchor = issueAnchor('gap',[byId(gap.before_token_id),byId(gap.after_token_id)]);
  if (anchor) anchor.expected = gap.expected;
  return anchor;
}
function paintIssue() {
  const finished = phase === 'error' || phase === 'review';
  $('issue-at-time').hidden = !sessionReplay || !finished;
  $('issue-panel').hidden = !issueDraft;
  $('issue-viewer').hidden = !issueDraft?.saved;
  $('issue-save').disabled = issueSaving || !sessionReplay || !finished;
  $('issue-listen').hidden = !sessionReplay || !finished;
  if (!issueDraft) return;
  if (!finished) $('issue-status').textContent = 'این بخش نشان شد. با پایان خواندن، آخرین نسخهٔ متن هم در نمونه ثبت می‌شود.';
  else if (!sessionReplay) $('issue-status').textContent = 'صدای این بخش برای دانلود در دسترس نیست.';
  else if (!issueSaving && !issueDraft.saved) $('issue-status').textContent = phase === 'error' ? 'نمونه تا زمان قطع ارتباط در دسترس است.' : '';
}
function openIssue(anchor, label, trigger) {
  if (!anchor) return;
  if (selectedWord || selectedGap) closeWord();
  issueDraft = {anchor,trigger,saved:false};
  issueSaving = false;
  $('issue-note').value = '';
  $('issue-status').textContent = '';
  $('issue-location').textContent = label;
  paintIssue();
  $('issue-title').focus();
}
function resetIssue() {
  issueDraft = null; issueSaving = false;
  $('issue-note').value = ''; $('issue-status').textContent = ''; $('issue-location').textContent = '';
  paintIssue();
}
async function saveIssue() {
  if (!issueDraft || issueSaving || !sessionReplay || !['review','error'].includes(phase)) return;
  const draft = issueDraft, run = generation;
  issueSaving = true; paintIssue(); $('issue-status').textContent = 'در حال آماده‌کردن فایل…';
  try {
    const sample = await buildIssueSample({recording:sessionReplay.blob,anchor:draft.anchor,
      history:issueHistory,tokens:transcript.tokens,note:$('issue-note').value,partial:phase === 'error'});
    if (run !== generation || draft !== issueDraft) return;
    const blob = new Blob([JSON.stringify(sample)], {type:'application/json'});
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = 'hamkhan-issue-v1.json';
    document.body.append(link); link.click(); link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
    draft.saved = true; $('issue-viewer').hidden = false;
    $('issue-status').textContent = 'فایل نمونه برای دانلود آماده شد.';
  } catch (problem) {
    if (run === generation && draft === issueDraft) $('issue-status').textContent = problem.message;
  } finally {
    if (run === generation && draft === issueDraft) {
      issueSaving = false;
      $('issue-save').disabled = false;
    }
  }
}
function enforceSegment() {
  clearTimeout(segmentTimer);
  const audio=$('replay-audio');
  if (segmentEnd===null || audio.paused) return;
  const remaining=segmentEnd-audio.currentTime;
  if (remaining<=.03) { audio.pause(); cancelSegment(); }
  else segmentTimer=setTimeout(enforceSegment, Math.max(25,remaining*1000/audio.playbackRate));
}
async function playSelection() {
  const bounds=selectedBounds(); if (!bounds) return;
  await playBounds(bounds);
}
async function playBounds(bounds) {
  const audio=$('replay-audio'); cancelSegment(); segmentEnd=bounds.end;
  try { audio.currentTime=bounds.start; await audio.play(); enforceSegment(); }
  catch { cancelSegment(); error('این بخش پخش نشد. دوباره دکمهٔ شنیدن را بزنید.'); }
}
function offerRemaining(replay) {
  if (replay.duration-reviewedThrough<.5 || $('retry-tail')) return;
  const button=document.createElement('button'), run=generation;
  button.id='retry-tail'; button.type='button'; button.className='text-button';
  button.textContent='بررسی بخش باقی‌مانده در نوبت تازه';
  $('replay-note').after(button);
  button.addEventListener('click',async()=>{
    button.disabled=true;
    try {
      const tail=await unreviewedRecording(replay.blob,reviewedThrough);
      const audio=await decodePracticeBlob(tail);
      if (generation!==run) return;
      chosenFile={audio,name:'بخش باقی‌مانده از نوبت قبلی'};
      $('file-name').textContent=chosenFile.name;
      await begin(chosenFile);
    } catch {if (generation===run) {error('بخش باقی‌مانده آماده نشد؛ پخش صدای همین نوبت در دسترس است.');button.disabled=false;}}
  });
}
function keepReplay(replay, partial=false) {
  if (!replay?.blob || replay.duration<.2) return;
  clearReplay(); sessionReplay=replay; replayURL=URL.createObjectURL(replay.blob);
  $('replay-audio').src=replayURL; $('replay-section').hidden=false;
  if (partial) {
    $('replay-note').textContent='صدای دریافت‌شده تا قطع ارتباط. زمانِ آخرین بازخورد '+clock(reviewedThrough)+' است.';
    offerRemaining(replay);
  }
  segmentButton();
  paintIssue();
}
function clock(seconds) { return fa(`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(Math.floor(seconds%60)).padStart(2,'0')}`); }
function paintReplay() {
  const audio = $('replay-audio'), duration = Number.isFinite(audio.duration) ? audio.duration : 0;
  $('replay-toggle').disabled = !duration; $('replay-toggle').textContent = audio.paused ? 'پخش صدای من' : 'توقف پخش';
  $('replay-seek').disabled = !duration; $('replay-seek').value = duration ? String(audio.currentTime/duration*100) : '0';
  $('replay-position').textContent = `${clock(audio.currentTime || 0)} / ${clock(duration)}`;
  $('replay-seek').setAttribute('aria-valuetext',`${clock(audio.currentTime || 0)} از ${clock(duration)}`);
  $('replay-seek').setAttribute('aria-describedby','replay-position');
  segmentButton();
}
function summary(value) {
  value = value || transcript.summary(); lastSummary = value;
  $('summary-reviewed').textContent = fa(value.reviewed ?? 0);
  $('summary-differences').textContent = fa(value.differences ?? 0);
  $('summary-unreviewed').textContent = fa(value.uncertain ?? value.unreviewed ?? 0);
  $('summary-missing').textContent = fa(value.missing ?? transcript.gaps.length);
  const count=(value.differences ?? 0)+(value.missing ?? transcript.gaps.length);
  $('difference-nav').hidden=count===0; $('review-first').hidden=count===0;
  $('difference-count').textContent=fa(count)+' بخش برای مرور';
  $('summary-scope').textContent = `${fa(transcript.tokens.length)} کلمهٔ شنیده‌شده در همین نوبت`;
}
function gapButton(gap) {
  const button = document.createElement('button'); button.type = 'button'; button.className = 'transcript-gap'; button.dataset.gapId = gap.id;
  button.textContent = 'بخش جاافتاده'; button.setAttribute('aria-label',`بخش احتمالی جاافتاده: ${gap.expected || ''}`);
  button.addEventListener('click',()=>showGap(gap.id)); return button;
}
function setKeyboardNode(node) {
  if (keyboardNode===node) return;
  if (keyboardNode) keyboardNode.tabIndex=-1;
  keyboardNode=node;
  if (node) node.tabIndex=0;
}
function paint(event={}) {
  const box=$('heard-transcript'), focus=document.activeElement;
  const wasFocused=focus===keyboardNode, touched=new Set();
  const from=Number.isInteger(event.replace_from)?event.replace_from:(event.type ? renderedOrder.length : 0);
  if (from<renderedOrder.length || from<transcript.tokens.length) {
    const current=new Set(transcript.tokens.slice(from).map(word=>word.id));
    for (let i=from;i<renderedOrder.length;i++) {
      const id=renderedOrder[i];
      if (!current.has(id)) {wordNodes.get(id)?.remove(); wordNodes.delete(id);}
    }
    renderedOrder.length=from;
    for (let i=from;i<transcript.tokens.length;i++) {
      const word=transcript.tokens[i];
      let node=wordNodes.get(word.id);
      if (!node) {
        node=document.createElement('button'); node.type='button';
        node.className='live-word transcript-word'; node.dataset.tokenId=word.id; node.tabIndex=-1;
        node.addEventListener('click',()=>showWord(word.id)); wordNodes.set(word.id,node);
      }
      // A revised tail may retain the same IDs in a different order.
      box.append(node); renderedOrder.push(word.id); touched.add(word.id);
    }
  }
  for (let i=renderedCommitted;i<transcript.committedCount;i++) touched.add(transcript.tokens[i].id);
  for (const update of event.updates || []) touched.add(update.id);
  for (const id of touched) {
    const index=transcript.index.get(id), word=transcript.tokens[index], node=wordNodes.get(id);
    if (!node || !word) continue;
    node.dataset.state=word.state;
    node.classList.toggle('tentative',index>=transcript.committedCount);
    if (node.textContent!==word.text) node.textContent=word.text;
    node.setAttribute('aria-label',word.text+'؛ '+(labels[word.state] || labels.pending)+(index>=transcript.committedCount?'؛ متن موقت':''));
    node.title=word.errors.map(item=>item.message || typeLabels[item.type] || 'اختلاف احتمالی').join(' · ') || labels[word.state];
  }
  renderedCommitted=transcript.committedCount;
  const liveGaps=new Set();
  for (const gap of transcript.gaps) {
    liveGaps.add(gap.id);
    let node=gapNodes.get(gap.id);
    if (!node) {node=gapButton(gap); node.tabIndex=-1; gapNodes.set(gap.id,node);}
    const label='بخش احتمالی جاافتاده: '+(gap.expected || '');
    if (node.getAttribute('aria-label')!==label) node.setAttribute('aria-label',label);
    const anchor=wordNodes.get(gap.after_token_id || gap.before_token_id);
    if (!anchor) continue;
    if (gap.after_token_id && node.nextElementSibling!==anchor) anchor.before(node);
    if (!gap.after_token_id && node.previousElementSibling!==anchor) anchor.after(node);
  }
  for (const [id,node] of gapNodes) if (!liveGaps.has(id)) {node.remove(); gapNodes.delete(id);}
  if (!keyboardNode?.isConnected) setKeyboardNode(box.firstElementChild);
  if (wasFocused && keyboardNode && document.activeElement!==keyboardNode) keyboardNode.focus({preventScroll:true});
  $('transcript-empty').hidden=transcript.tokens.length>0;
  if (selectedWord && !wordNodes.has(selectedWord) || selectedGap && !gapNodes.has(selectedGap)) closeWord();
  else if (touched.has(selectedWord)) showWord(selectedWord,false);
  else if (selectedGap) showGap(selectedGap,false);
  if (transcript.committedCount > lastAnnouncedCount) {
    $('transcript-announcement').textContent = `متن ${fa(transcript.committedCount)} کلمه نوشته شد.`; lastAnnouncedCount = transcript.committedCount;
  }
}
function navigateDifference(step, first=false) {
  const nodes=[...$('heard-transcript').querySelectorAll('[data-state="difference"],[data-gap-id]')];
  if (!nodes.length) return;
  const active=(selectedWord ? wordNodes.get(selectedWord) : gapNodes.get(selectedGap)) || lastReviewNode;
  const index=nodes.indexOf(active);
  const next=first ? 0 : index<0 ? (step>0?0:nodes.length-1) : (index+step+nodes.length)%nodes.length;
  const node=nodes[next]; setKeyboardNode(node); node.focus({preventScroll:true});
  node.scrollIntoView({block:'nearest',behavior:'instant'}); node.click();
}
function detailParagraph(label,text,arabic = false) {
  if (!text) return;
  const p = document.createElement('p'), strong = document.createElement('strong'), content = document.createElement('span');
  strong.textContent = `${label}: `; content.textContent = text; if (arabic) content.lang = 'ar'; p.append(strong,content); $('word-explanation').append(p);
}
function openDetail(open) {
  segmentButton();
  if (!open) return;
  const node=selectedWord ? wordNodes.get(selectedWord) : gapNodes.get(selectedGap);
  if (lastReviewNode!==node) lastReviewNode?.removeAttribute('aria-current');
  lastReviewNode=node; node?.setAttribute('aria-current','true');
  const detail=$('word-detail'); detail.dataset.open='true';
  if (matchMedia('(max-width:760px)').matches) {
    detail.setAttribute('role','dialog'); detail.setAttribute('aria-modal','true');
    detail.setAttribute('aria-labelledby','word-title'); $('word-close').focus({preventScroll:true});
  }
}
function showWord(id, open = true) {
  const word = transcript.tokens[transcript.index.get(id)]; if (!word) return;
  selectedWord = id; selectedGap = null;
  $('word-title').textContent = word.text; $('word-title').lang = 'ar'; $('word-explanation').replaceChildren();
  detailParagraph('شنیده‌شده',word.text,true);
  if (word.reference?.word) detailParagraph('مرجع',word.reference.word,true);
  if (word.errors.length) word.errors.forEach(issue=>{
    detailParagraph(typeLabels[issue.type] || 'اختلاف احتمالی',issue.message || 'این بخش را دوباره مرور کنید.');
    const lexical=['word','transposition','repetition','substitution','transcription','boundary'].includes(issue.type) || (issue.type==='insertion' && issue.heard?.trim()===word.text.trim());
    if (issue.heard && !lexical) detailParagraph('آوای تشخیص‌داده‌شده',issue.heard,true);
    if (issue.expected) detailParagraph(lexical ? 'واژهٔ مورد انتظار' : 'آوای مورد انتظار',issue.expected,true);
  });
  else detailParagraph('وضعیت',word.state === 'match' ? 'در موارد بررسی‌شده اختلافی پیدا نشد؛ این نتیجه تأیید قطعی صحت قرائت نیست.' : word.state==='pending' ? 'این واژه هنوز بررسی نشده است.' : word.reference ? 'برای قضاوت تلفظ این واژه شواهد صوتی کافی نیست.' : 'محل این واژه در آیه هنوز مشخص نشده است.');
  $('word-detail').dataset.state = word.state;
  $('word-retry').hidden = !word.reference?.surah || !word.reference?.ayah;
  if (!$('word-retry').hidden) $('word-retry').textContent = `تمرین دوبارهٔ آیهٔ ${fa(word.reference.ayah)}`;
  openDetail(open);
}
function showGap(id, open = true) {
  const gap = transcript.gaps.find(item=>item.id===id); if (!gap) return;
  selectedGap = id; selectedWord = null; $('word-title').textContent = 'بخش احتمالی جاافتاده'; $('word-title').removeAttribute('lang');
  $('word-explanation').replaceChildren(); detailParagraph('در مرجع',gap.expected,true);
  detailParagraph('توضیح',gap.message || 'این بخش بین دو قسمت خوانده‌شده در مرجع وجود دارد، اما در صدا پیدا نشد. این نشانه جزو گفتار شما نیست.');
  $('word-detail').dataset.state = 'difference'; $('word-retry').hidden = true; openDetail(open);
}
function closeWord() {
  const anchor=selectedWord?wordNodes.get(selectedWord):gapNodes.get(selectedGap);
  const detail=$('word-detail'); detail.dataset.open='false'; detail.removeAttribute('role');
  detail.removeAttribute('aria-modal'); detail.removeAttribute('aria-labelledby');
  (anchor || keyboardNode)?.focus({preventScroll:true});
  lastReviewNode?.removeAttribute('aria-current');
  selectedWord = selectedGap = null; segmentButton();
}
function follow() {
  if (issueDraft || !$('auto-follow').checked || performance.now() < manualUntil) return;
  const last = $('heard-transcript').lastElementChild; if (!last) return;
  const bounds = last.getBoundingClientRect();
  if (bounds.top < 75 || bounds.bottom > innerHeight-110) {
    autoScrollUntil = performance.now()+1200;
    last.scrollIntoView({block:'nearest',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  }
}
async function updateReference(reference) {
  if (!reference?.surah || !reference?.ayah) return;
  const identity = `${reference.surah}:${reference.ayah}`;
  const name = surahs.find(surah=>surah.surah === reference.surah)?.name;
  const label = `${name || `سورهٔ ${fa(reference.surah)}`} · آیهٔ ${fa(reference.ayah)}`;
  $('transcript-location').textContent = label; $('reference-location').textContent = label;
  $('stage-position').dataset.active = 'true';
  if (lastReference === identity && !reference.text) return;
  lastReference = identity; const request = ++referenceFetch;
  if (reference.text) { $('reference-text').textContent = reference.text; return; }
  try {
    const response = await fetch(`/api/quran/ayahs?surah=${reference.surah}&start=${reference.ayah}&limit=1`);
    if (!response.ok) throw new Error(); const data = await response.json();
    if (request !== referenceFetch || lastReference !== identity) return;
    $('reference-text').textContent = data.ayahs?.[0]?.text || 'متن مرجع این بخش در دسترس نیست.';
  } catch { if (request === referenceFetch) $('reference-text').textContent = 'دریافت متن مرجع ممکن نشد؛ محل قرائت را دوباره انتخاب کنید.'; }
}
function reset({preserveFile = false} = {}) {
  reportUI(); generation++; referenceFetch++; connection?.close(); controller?.abort(); discardCapture(capture);
  connection = capture = controller = null; cancelAnimationFrame(raf); raf = null;
  transcript.reset(); wordNodes.clear(); gapNodes.clear(); $('heard-transcript').replaceChildren();
  issueHistory.clear(); resetIssue();
  renderedOrder=[]; renderedCommitted=0; keyboardNode=null; lastReviewNode=null; reviewedThrough=0; selectedWord = selectedGap = lastReference = null; lastSummary = null; lastAnnouncedCount = 0;
  resetAudioDiagnostics();
  clearReplay(); error(); $('session-summary').hidden = true; $('word-detail').dataset.open = 'false';
  for (const key of ['role','aria-modal','aria-labelledby']) $('word-detail').removeAttribute(key);
  $('word-title').textContent = 'کلمه‌ها را مرور کنید'; $('word-title').removeAttribute('lang');
  $('word-explanation').textContent = 'کلمهٔ مشخص‌شده را انتخاب کنید تا توضیح آن را ببینید.'; $('word-retry').hidden = true;
  $('transcript-location').textContent = 'محل قرائت هنوز مشخص نشده است'; $('reference-location').textContent = '';
  $('reference-text').textContent = 'پس از مشخص‌شدن محل قرائت، متن همان بخش اینجا نمایش داده می‌شود.';
  for (const id of ['stage-voice','stage-position','stage-result']) $(id).dataset.active = 'false';
  if (!preserveFile) { chosenFile = null; $('live-file').value = ''; $('file-name').textContent = 'تا ۱۰۰ مگابایت · حداکثر ۳۰ دقیقه'; }
  $('live-time').textContent = '۰۰:۰۰'; $('live-wave-empty').hidden = false;
  paint(); summary(); drawWave(); setPhase('idle'); status(!ready ? 'برای شروع، آمادگی مدل‌ها و اتصال محلی را بررسی کنید.' : $('reference-auto').checked ? 'از هر جای قرآن آغاز کنید؛ محل قرائت خودکار پیدا می‌شود.' : 'از سوره و آیهٔ انتخاب‌شده آغاز کنید.');
}
function fail(message) {
  const source=capture, failedController=controller, run=++generation;
  connection?.close(); connection=capture=controller=null;
  cancelAnimationFrame(raf); raf=null;
  // A broken transport still leaves a coherent local recording, including a
  // possible unanalysed tail. Allow the worklet to flush before releasing it.
  if (source) {
    void source.stop({partial:true}).then(replay=>{
      if (run===generation) keepReplay(replay,true);
    }).catch(()=>{}).finally(()=>failedController?.abort());
  } else failedController?.abort();
  if (sessionReplay) {
    $('replay-note').textContent='این نوبت ناقص است. آخرین بازخورد تا '+clock(reviewedThrough)+' دریافت شد.';
    offerRemaining(sessionReplay);
  }
  error(message); setPhase('error'); status('این نوبت کامل بررسی نشد. می‌توانید بخش دریافت‌شده را مرور کنید.');
  if (transcript.tokens.length) { summary(lastSummary); $('session-summary').hidden=false; }
  reportUI(); drawWave();
}
function markPaint(event) {
  if (!measureUI || !['transcript','feedback'].includes(event.type)) return;
  const now=performance.now(), run=generation, target=event.type==='transcript'?textPaints:feedbackPaints;
  requestAnimationFrame(()=>requestAnimationFrame(()=>{if (run===generation && target.length<6000) target.push(performance.now()-now);}));
}
function accept(event, run = generation) {
  if (run !== generation) return;
  if (event.type === 'error') { status(event.message); return; }
  try {
    markPaint(event);
    if (event.type === 'transcript') {
      audioActivity = ['speech','no_speech','uncertain'].includes(event.audio_activity?.status) ? {...event.audio_activity} : null;
      // The server sends a non-destructive delta when speech is absent; apply
      // it normally so final commitment and the client word indexes stay equal.
      transcript.applyTranscript(event); issueHistory.record(event); paint(event); summary(event.summary); follow();
      streamStatus(audioActivity && audioActivity.status !== 'speech' ? 'در انتظار گفتار واضح‌تر…' : 'متن شنیده‌شده نوشته شد؛ بررسی تلفظ ادامه دارد.');
    } else if (event.type === 'feedback') {
      if (Number.isInteger(event.end_sample)) reviewedThrough=Math.max(reviewedThrough,event.end_sample/16000);
      transcript.applyFeedback(event); paint(event); summary(event.summary);
      if (event.updates?.some(word=>['match','difference'].includes(word.state))) $('stage-result').dataset.active = 'true';
      void updateReference(event.reference); streamStatus(event.message || (event.status==='tracking' ? 'قرائت دنبال می‌شود؛ به خواندن ادامه دهید.' : 'برای یافتن محل قرائت، چند کلمهٔ پیوسته بخوانید.'));
      if (event.status === 'ambiguous') $('reference-choice').open = true;
    } else if (event.type === 'progress') streamStatus(event.message || 'در حال بررسی صدا…');
    else if (event.type === 'done') {
      if (event.updates) { transcript.applyFeedback(event); paint(event); }
      summary(event.summary); void updateReference(event.reference);
      if (active()) void finish(true);
    }
  } catch (problem) { fail(problem.message); }
}
function drawWave(data = null) {
  const canvas = $('live-wave'), ctx = canvas.getContext('2d'), rect = canvas.getBoundingClientRect();
  const ratio = Math.min(2,window.devicePixelRatio || 1), width = Math.max(1,Math.round(rect.width*ratio)), height = Math.max(1,Math.round(rect.height*ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  ctx.clearRect(0,0,width,height); const count = 48, step = width/count, bar = Math.max(2,step*.35);
  ctx.fillStyle = getComputedStyle(canvas).getPropertyValue('--wave-color').trim() || '#bbd9bc';
  for (let i = 0; i < count; i++) {
    let peak = 0; if (data) for (let p = Math.floor(i*data.length/count); p < Math.floor((i+1)*data.length/count); p++) peak = Math.max(peak,Math.abs(data[p]));
    const amplitude = Math.max(2*ratio,Math.min(height*.8,peak*height*3));
    ctx.beginPath(); ctx.roundRect(i*step+(step-bar)/2,(height-amplitude)/2,bar,amplitude,bar/2); ctx.fill();
  }
}
function animate(run) {
  const data = new Float32Array(1024);
  const frame = now=>{
    if (run !== generation || !capture) return;
    if (now-lastCanvasFrame >= 30) {
      if (measureUI && lastCanvasFrame && frameIntervals.length < 6000) frameIntervals.push(now-lastCanvasFrame);
      lastCanvasFrame = now; capture.analyser.getFloatTimeDomainData(data); drawWave(data);
      paintInputDiagnostics(capture.diagnostics,now);
      $('live-time').textContent = clock(Math.floor(capture.totalSamples/capture.rate));
    }
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
}
function revealReadingOnStart(run, initialIntent) {
  // The permission/file controls can put the beginning of the transcript
  // outside the viewport before the first word has a chance to be drawn.
  requestAnimationFrame(()=>{
    if (run!==generation || !active() || !$('auto-follow').checked || initialIntent!==manualScrollIntent) return;
    manualUntil=0;
    autoScrollUntil=performance.now()+400;
    $('surah-reading').scrollIntoView({block:'start',behavior:'instant'});
  });
}
function userScrolled() {
  manualScrollIntent++;
  manualUntil=performance.now()+5000;
}
async function begin(file = null) {
  const initialIntent=manualScrollIntent;
  reset({preserveFile:!!file}); const run = generation; controller = new AbortController(); setPhase('connecting');
  status(file ? 'در حال آماده‌سازی پخش و بررسی…' : 'اجازهٔ دسترسی به میکروفن را تأیید کنید.');
  try {
    await new Promise(resolve=>requestAnimationFrame(()=>setTimeout(resolve,0))); if (run !== generation) return;
    const source = await prepareAudioSource({fileAudio:file?.audio,signal:controller.signal,
      onSamples:block=>{ if (run === generation) { try { connection.sendSamples(block); } catch (problem) { if (run === generation) fail(problem.message); } } },
      onEnded:()=>{ if (run === generation) void finish(); }});
    if (run !== generation) { discardCapture(source); return; }
    capture = source;
    connection = new LiveConnection({onEvent:event=>accept(event,run),onError:message=>{ if (run === generation) fail(message); }});
    const manual = !$('reference-auto').checked;
    manualReferenceApplied=manual;
    await connection.connect({sampleRate:source.rate,startSurah:manual ? Number($('start-surah').value) : null,startAyah:manual ? Number($('start-ayah').value) : null},controller.signal);
    if (run !== generation) return;
    setPhase(file ? 'file' : 'recording'); status('چند کلمهٔ اول را پیوسته بخوانید؛ متن از صدای شما نوشته می‌شود.');
    lastCanvasFrame = 0; source.start(); animate(run); revealReadingOnStart(run,initialIntent);
  } catch (problem) {
    if (run !== generation) return;
    const messages = {NotAllowedError:'مجوز میکروفن داده نشد؛ اجازهٔ مرورگر را بررسی کنید یا فایل انتخاب کنید.',NotFoundError:'میکروفن پیدا نشد؛ اتصال آن را بررسی کنید یا فایل انتخاب کنید.',NotReadableError:'میکروفن در دسترس نیست؛ برنامه‌های دیگری را که از آن استفاده می‌کنند بررسی کنید.'};
    fail(messages[problem.name] || (/[\u0600-\u06ff]/.test(problem.message) ? problem.message : 'شروع تمرین ممکن نشد؛ دوباره تلاش کنید.'));
  }
}
async function finish(remoteDone = false) {
  if (phase === 'connecting') { reset(); return; } if (!active()) return;
  const run = generation; setPhase('stopping'); status('دریافت صدا تمام شد؛ آخرین بخش‌ها در حال بررسی‌اند…');
  try {
    const replay = await capture?.stop(); if (run !== generation) return;
    capture = null; cancelAnimationFrame(raf); raf = null; drawWave();
    keepReplay(replay);
    if (!remoteDone) await connection.finish(); if (run !== generation) return;
    connection.close(); setPhase('review'); summary(lastSummary); $('session-summary').hidden = false; reportUI();
    status(transcript.tokens.length ? 'تمرین پایان یافت. متن شنیده‌شده و کلمه‌های مشخص‌شده را مرور کنید.' : 'گفتار کافی برای رونویسی پیدا نشد؛ دوباره بخوانید.');
  } catch (problem) { if (run === generation) fail(problem.message); }
}
function fillAyahs(value = 1) {
  const surah = surahs.find(item=>item.surah === Number($('start-surah').value));
  $('start-ayah').replaceChildren();
  for (let number = 1; number <= (surah?.ayah_count || 1); number++) {
    const option = document.createElement('option'); option.value = String(number); option.textContent = fa(number); $('start-ayah').append(option);
  }
  $('start-ayah').value = String(value <= (surah?.ayah_count || 1) ? value : 1); referenceControls();
}
async function loadCatalogue() {
  try {
    const response = await fetch('/api/quran/surahs'); if (!response.ok) throw new Error();
    const data = await response.json(); if (!Array.isArray(data.surahs) || !data.surahs.length) throw new Error();
    surahs = data.surahs; $('start-surah').replaceChildren();
    for (const surah of surahs) { const option = document.createElement('option'); option.value = String(surah.surah); option.textContent = `${fa(surah.surah)} · ${surah.name}`; $('start-surah').append(option); }
    catalogueReady = true; fillAyahs();
  } catch { $('start-surah').replaceChildren(new Option('فهرست در دسترس نیست','')); $('reference-help').textContent = 'فهرست سوره‌ها دریافت نشد؛ برای انتخاب دستی، صفحه را تازه کنید. تشخیص خودکار همچنان قابل استفاده است.'; referenceControls(); }
}
$('live-start').addEventListener('click',()=>void begin());
$('live-stop').addEventListener('click',()=>void finish()); $('live-stop-mobile').addEventListener('click',()=>void finish());
$('live-reset').addEventListener('click',()=>reset());
$('reference-auto').addEventListener('change',()=>{ referenceControls(); if (active() && $('reference-auto').checked && manualReferenceApplied) { $('reference-auto').checked=false; referenceControls(); status('مرجع دستی این نوبت فعال است. برای تشخیص خودکار نوبت تازه‌ای آغاز کنید.'); } });
$('start-surah').addEventListener('change',()=>fillAyahs());
$('reference-apply').addEventListener('click',()=>{
  if (active()) { try { connection.setReference(Number($('start-surah').value),Number($('start-ayah').value)); manualReferenceApplied=true; status('محل انتخاب‌شده برای مقایسه اعمال شد؛ نوشتهٔ صدای شما حفظ می‌شود.'); } catch (problem) { error(problem.message); } }
  else status('محل شروع انتخاب شد؛ «شروع خواندن» را بزنید.');
});
$('live-file').addEventListener('change',async event=>{
  const file = event.target.files[0]; if (!file) return;
  reset(); const run = generation; setPhase('connecting'); status('در حال آماده‌سازی فایل…');
  try { const audio = await readAudioFile(file); if (run !== generation) return;
    chosenFile = {audio,name:file.name}; $('file-name').textContent = file.name; setPhase('idle'); status('فایل آماده است؛ «پخش و بررسی» را بزنید.');
  } catch (problem) { if (run === generation) fail(problem.message); }
});
$('file-play').addEventListener('click',()=>{ if (chosenFile) void begin(chosenFile); });
$('replay-toggle').addEventListener('click',async()=>{ cancelSegment(); const audio = $('replay-audio'); try { if (audio.paused) await audio.play(); else audio.pause(); } catch { error('بازپخش صدا ممکن نشد؛ دوباره دکمهٔ پخش را بزنید.'); } paintReplay(); });
$('replay-seek').addEventListener('input',()=>{ cancelSegment(); const audio = $('replay-audio'); if (Number.isFinite(audio.duration)) audio.currentTime = Number($('replay-seek').value)/100*audio.duration; });
for (const event of ['loadedmetadata','timeupdate','play','pause','ended','emptied']) $('replay-audio').addEventListener(event,paintReplay);
$('segment-play').addEventListener('click',()=>void playSelection());
$('issue-open').addEventListener('click',()=>{
  const anchor = selectedIssueAnchor();
  const trigger = selectedWord ? wordNodes.get(selectedWord) : gapNodes.get(selectedGap);
  const label = selectedWord ? 'واژهٔ انتخاب‌شده «'+$('word-title').textContent.trim()+'»' : 'بازهٔ میان دو واژهٔ اطراف جای خالی';
  openIssue(anchor,anchor?.timing_source === 'neighbor' ? label+' · زمان تقریبی از واژهٔ همسایه' : label,trigger);
});
$('issue-at-time').addEventListener('click',()=>{
  const audio = $('replay-audio');
  audio.pause();
  openIssue(issueAnchor('playhead',[],audio.currentTime*1000),'زمان انتخاب‌شده در بازپخش '+clock(audio.currentTime),$('issue-at-time'));
});
$('issue-save').addEventListener('click',()=>void saveIssue());
$('issue-listen').addEventListener('click',()=>{
  if (!issueDraft || !sessionReplay) return;
  try {
    const {start,end} = issueBounds(issueDraft.anchor,sessionReplay.duration*1000);
    void playBounds({start:start/1000,end:end/1000});
  } catch (problem) { $('issue-status').textContent = problem.message; }
});
$('issue-close').addEventListener('click',()=>{
  const trigger = issueDraft?.trigger;
  resetIssue();
  (trigger?.isConnected ? trigger : $('surah-reading')).focus();
});
$('replay-audio').addEventListener('timeupdate',enforceSegment);
$('replay-audio').addEventListener('pause',cancelSegment);
$('difference-prev').addEventListener('click',()=>navigateDifference(-1));
$('difference-next').addEventListener('click',()=>navigateDifference(1));
$('review-first').addEventListener('click',()=>navigateDifference(1,true));
$('heard-transcript').addEventListener('focusin',event=>{
  if (event.target.matches('button')) setKeyboardNode(event.target);
});
$('heard-transcript').addEventListener('keydown',event=>{
  if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key) || !event.target.matches('button')) return;
  event.preventDefault();
  const node=event.key==='Home'?event.currentTarget.firstElementChild:event.key==='End'?event.currentTarget.lastElementChild:
    event.key==='ArrowLeft'?event.target.nextElementSibling:event.target.previousElementSibling;
  if (node) {setKeyboardNode(node); node.focus();}
});
$('check-ready').addEventListener('click',async()=>{
  const seq=++healthRequest; $('check-ready').disabled=true;
  try {
    const response=await fetch('/health',{cache:'no-store',signal:AbortSignal.timeout(8000)});
    const data=await response.json(); if (seq!==healthRequest) return;
    ready=response.ok && data.ready===true && data.protocol_version===2;
    if (!catalogueReady && ready) await loadCatalogue();
    status(ready?'مدل‌ها آماده‌اند. می‌توانید خواندن را شروع کنید.':'مدل‌ها آماده نیستند. نصب و اجرای محلی برنامه را بررسی کنید.');
    if (ready) error();
  } catch {ready=false; status('برنامه پاسخ نداد. پس از اجرای دوباره، اتصال را بررسی کنید.');}
  finally {
    if (seq===healthRequest) {
      document.body.dataset.ready=String(ready); $('model-readiness').textContent=ready?'مدل‌ها آماده‌اند':'مدل‌ها آماده نیستند';
      $('model-badge').querySelector('.status-dot').classList.toggle('unready',!ready);
      $('check-ready').disabled=false; setPhase(phase);
    }
  }
});
$('word-detail').addEventListener('keydown',event=>{
  if (event.key!=='Tab' || !$('word-detail').hasAttribute('aria-modal') || !matchMedia('(max-width:760px)').matches) return;
  const buttons=[...event.currentTarget.querySelectorAll('button:not([hidden]):not(:disabled)')];
  if (event.shiftKey && event.target===buttons[0]) {event.preventDefault(); buttons.at(-1)?.focus();}
  else if (!event.shiftKey && event.target===buttons.at(-1)) {event.preventDefault(); buttons[0]?.focus();}
});
$('word-close').addEventListener('click',closeWord);
$('word-retry').addEventListener('click',()=>{
  const word = transcript.tokens.find(token=>token.id === selectedWord); if (!word?.reference) return;
  $('reference-auto').checked = false; $('start-surah').value = String(word.reference.surah); fillAyahs(word.reference.ayah); void begin();
});
$('auto-follow').addEventListener('change',()=>{ if ($('auto-follow').checked) { manualUntil = 0; follow(); } });
window.addEventListener('wheel',userScrolled,{passive:true});
window.addEventListener('touchmove',userScrolled,{passive:true});
window.addEventListener('scroll',()=>{ if (performance.now() > autoScrollUntil) manualUntil = performance.now()+5000; },{passive:true});
window.addEventListener('keydown',event=>{ if (event.key === 'Escape' && $('word-detail').dataset.open === 'true') closeWord();
  if (!event.target.closest('input,select,button,textarea,a') && ['PageDown','PageUp','Home','End','ArrowDown','ArrowUp',' '].includes(event.key)) userScrolled();
});
window.addEventListener('pagehide',()=>{ generation++; connection?.close(); controller?.abort(); discardCapture(capture); if (replayURL) URL.revokeObjectURL(replayURL); });
reset(); void loadCatalogue();
