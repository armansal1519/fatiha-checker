import {readAudioFile} from './live-audio.js';
import {prepareAudioSource} from './live-source.js';
import {LiveConnection} from './live-wire.js';

const $ = id=>document.getElementById(id), fa = value=>String(value).replace(/\d/g,d=>'۰۱۲۳۴۵۶۷۸۹'[d]);
const words = JSON.parse($('live-data').textContent), ready = document.body.dataset.ready === 'true';
const buttons = [...document.querySelectorAll('.live-word')], rows = [...document.querySelectorAll('.live-ayah')];
let generation = 0, phase = 'idle', capture = null, connection = null, controller = null;
let replayURL = null, chosenFile = null, states = new Map(), selectedWord = null, activeIndex = -1;
let followedAyah = null, manualUntil = 0, autoScrollUntil = 0, raf = null, lastCanvasFrame = 0, lastSummary = null;
// Opt-in development measurements; never shown as confidence or a reading score.
const measureUI = new URLSearchParams(location.search).has('measure');
let frameIntervals = [], controlPaints = [];
function reportUI() {
  if (!measureUI || !frameIntervals.length) return;
  const percentile = (items,p)=>[...items].sort((a,b)=>a-b)[Math.min(items.length-1,Math.floor(items.length*p))];
  console.info('demo-ui-metrics',JSON.stringify({wave_frames:frameIntervals.length,wave_interval_p95_ms:percentile(frameIntervals,.95),control_paint_max_ms:Math.max(...controlPaints)}));
  frameIntervals = []; controlPaints = [];
}

function status(text) { $('live-status').textContent = text; }
function error(text = '') { $('live-error').textContent = text; $('live-error').hidden = !text; }
function setPhase(next) {
  if (measureUI) { const changed = performance.now(), run = generation; requestAnimationFrame(()=>{ if (run === generation) controlPaints.push(performance.now()-changed); }); }
  phase = next; document.body.dataset.phase = next;
  const locked = ['connecting','recording','file','stopping'].includes(next);
  $('live-start').disabled = !ready || locked;
  $('start-ayah').disabled = locked; $('live-file').disabled = !ready || locked;
  $('file-play').disabled = !ready || locked || !chosenFile;
  $('live-stop').disabled = !locked || next === 'stopping';
  $('live-stop').textContent = next === 'connecting' ? 'انصراف' : 'پایان خواندن';
  $('live-stop-mobile').hidden = !locked; $('live-stop-mobile').disabled = next === 'stopping';
  $('live-stop-mobile').textContent = next === 'connecting' ? 'انصراف' : next === 'stopping' ? 'در حال جمع‌بندی…' : 'پایان خواندن';
  const labels = {idle:'آمادهٔ تمرین',connecting:'در حال آماده‌سازی',recording:'در حال شنیدن',file:'پخش و بررسی',stopping:'بررسی پایانی',review:'تمرین پایان یافت',error:'تمرین متوقف شد'};
  $('live-pill').textContent = labels[next];
  $('live-pill').classList.toggle('active',['recording','file'].includes(next));
  const titles = {idle:'با صدای خودتان آغاز کنید',connecting:'یک لحظه تا شروع',recording:'صدایتان را می‌شنویم',file:'همراه صدای شما',stopping:'در حال مرور آخرین کلمه‌ها',review:'این بار را مرور کنید',error:'آمادهٔ یک شروع دوباره'};
  if ($('live-state-title')) $('live-state-title').textContent = titles[next];
  if ($('live-state-description')) $('live-state-description').textContent = ({idle:'آرام و پیوسته بخوانید؛ ما همراه متن می‌مانیم.',connecting:'اتصال و میکروفن آماده می‌شوند.',recording:'در پایان هر آیه کمی مکث کنید و ادامه دهید.',file:'فایل شما پخش می‌شود و کلمه‌ها بررسی می‌شوند.',stopping:'میکروفن بسته است؛ نتیجهٔ آخر در راه است.',review:'کلمه‌های مشخص‌شده را انتخاب و دوباره تمرین کنید.',error:'پیام زیر را بررسی کنید؛ می‌توانید دوباره شروع کنید.'})[next];
  $('word-retry').disabled = locked || !ready;
}
function clearReplay() {
  $('replay-audio').pause(); $('replay-audio').removeAttribute('src'); $('replay-audio').load();
  if (replayURL) URL.revokeObjectURL(replayURL);
  replayURL = null; $('replay-section').hidden = true;
  paintReplay();
}
function paintReplay() {
  const audio = $('replay-audio'), duration = Number.isFinite(audio.duration) ? audio.duration : 0;
  const clock = seconds=>fa(`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(Math.floor(seconds%60)).padStart(2,'0')}`);
  $('replay-toggle').disabled = !duration;
  $('replay-toggle').textContent = audio.paused ? 'پخش صدای من' : 'توقف پخش';
  $('replay-seek').disabled = !duration;
  $('replay-seek').value = duration ? String(audio.currentTime/duration*100) : '0';
  $('replay-position').textContent = `${clock(audio.currentTime || 0)} / ${clock(duration)}`;
  $('replay-seek').setAttribute('aria-valuetext',`${clock(audio.currentTime || 0)} از ${clock(duration)}`);
  $('replay-seek').setAttribute('aria-describedby','replay-position');
}
function paint() {
  buttons.forEach((button,index)=>{
    const state = states.get(index);
    if (state) button.dataset.state = state.state; else delete button.dataset.state;
    button.classList.toggle('current',index === activeIndex);
    if (index === activeIndex) button.setAttribute('aria-current','true'); else button.removeAttribute('aria-current');
    const label = state ? state.state === 'match' ? 'اختلافی پیدا نشد' : 'نیازمند مرور؛ اختلاف احتمالی' : 'هنوز بررسی نشده';
    button.setAttribute('aria-label',`${words[index].word}؛ ${label}`);
    button.title = state?.errors.map(e=>e.message).join('\n') || label;
  });
  const activeAyah = words[activeIndex]?.ayah;
  rows.forEach(row=>row.classList.toggle('active',Number(row.dataset.ayah) === activeAyah));
  if (selectedWord !== null) showWord(selectedWord,false);
}
function summary(value) {
  if (value) lastSummary = value;
  value = value || lastSummary;
  const start = words.findIndex(word=>word.ayah === Number($('start-ayah').value));
  $('summary-scope').textContent = `از آیهٔ ${fa($('start-ayah').value)} تا پایان سوره`;
  $('summary-reviewed').textContent = fa(value?.reviewed ?? states.size);
  $('summary-differences').textContent = fa(value?.differences ?? [...states.values()].filter(x=>x.state === 'difference').length);
  $('summary-unreviewed').textContent = fa(value?.unreviewed ?? words.length-Math.max(0,start)-states.size);
}
function showWord(index, open = true) {
  selectedWord = index;
  const word = words[index], state = states.get(index);
  $('word-title').textContent = word.word; $('word-title').lang = 'ar';
  const text = !state ? 'برای بررسی این کلمه هنوز شواهد کافی نداریم. خواندن را پیوسته ادامه دهید.' : state.state === 'match' ? 'در موارد بررسی‌شده اختلافی پیدا نشد. این نتیجه، تأیید قطعی قرائت نیست.' : [...new Set(state.errors.map(e=>e.message))].join('\n');
  if ($('word-explanation').textContent !== text) $('word-explanation').textContent = text;
  $('word-retry').hidden = false; $('word-retry').textContent = `تمرین دوبارهٔ آیهٔ ${fa(word.ayah)}`;
  $('word-detail').dataset.state = state?.state || 'pending';
  if (open) { $('word-detail').dataset.open = 'true'; if (matchMedia('(max-width:760px)').matches) $('word-close').focus({preventScroll:true}); }
}
function closeWord() { $('word-detail').dataset.open = 'false'; if (selectedWord !== null) buttons[selectedWord].focus({preventScroll:true}); }
function follow() {
  const current = words[activeIndex];
  if (!current || current.ayah === followedAyah) return;
  if (!$('auto-follow').checked || performance.now() < manualUntil) return;
  followedAyah = current.ayah;
  const row = rows.find(r=>Number(r.dataset.ayah) === current.ayah), bounds = row.getBoundingClientRect();
  if (bounds.top < 75 || bounds.bottom > innerHeight-100) {
    autoScrollUntil = performance.now()+1200;
    row.scrollIntoView({block:'center',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  }
}
function reset({preserveFile = false} = {}) {
  reportUI();
  frameIntervals = []; controlPaints = [];
  generation++; connection?.close(); controller?.abort(); void capture?.stop({discard:true});
  connection = null; capture = null; controller = null; cancelAnimationFrame(raf); raf = null;
  states = new Map(); selectedWord = null; activeIndex = -1; followedAyah = null; lastSummary = null;
  clearReplay(); error(); $('session-summary').hidden = true;
  $('word-detail').dataset.open = 'false'; delete $('word-detail').dataset.state;
  $('word-title').textContent = 'هر کلمه، فرصتی برای تمرین';
  $('word-title').removeAttribute('lang');
  $('word-explanation').textContent = 'کلمه‌های خط‌دار را انتخاب کنید تا توضیح این بخش را ببینید.';
  $('word-retry').hidden = true;
  for (const id of ['stage-voice','stage-position','stage-result']) $(id).dataset.active = 'false';
  if (!preserveFile) { chosenFile = null; $('live-file').value = ''; $('file-name').textContent = 'فایل صوتی خودتان را انتخاب کنید'; }
  $('live-time').textContent = '۰۰:۰۰'; $('live-wave-empty').hidden = false;
  paint(); drawWave(); setPhase('idle'); status('از آیهٔ انتخاب‌شده شروع کنید.');
}
function fail(message) {
  generation++; connection?.close(); controller?.abort(); void capture?.stop({discard:true});
  capture = null; connection = null; cancelAnimationFrame(raf); raf = null;
  error(message); setPhase('error'); status('این نوبت کامل بررسی نشد. پس از رفع مشکل، دوباره شروع کنید.');
  if (states.size) { summary(); $('session-summary').hidden = false; }
  reportUI();
  drawWave();
}
function accept(event) {
  if (event.type === 'error') { status(event.message); return; }
  if (event.type === 'progress') {
    if (event.phase === 'tracking' && Number.isInteger(event.current_word)) {
      activeIndex = Math.max(activeIndex,event.current_word);
      $('stage-position').dataset.active = 'true'; paint(); follow();
      status(`آیهٔ ${fa(words[activeIndex].ayah)}؛ به خواندن ادامه دهید.`);
    } else status(event.message || 'چند کلمهٔ پیوسته بخوانید تا محل خواندن روشن شود.');
  }
  if (event.type === 'result' || event.type === 'done') {
    states = new Map(event.updates.map(word=>[word.index,word])); paint(); summary(event.summary);
    if (states.size) $('stage-result').dataset.active = 'true';
    if (event.type === 'done' && ['recording','file'].includes(phase)) void finish(true);
  }
}
function drawWave(data = null) {
  const canvas = $('live-wave'), ctx = canvas.getContext('2d'), rect = canvas.getBoundingClientRect();
  const ratio = Math.min(2,window.devicePixelRatio || 1), width = Math.max(1,Math.round(rect.width*ratio)), height = Math.max(1,Math.round(rect.height*ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  ctx.clearRect(0,0,width,height);
  const count = 48, step = width/count, bar = Math.max(2,step*.35);
  ctx.fillStyle = getComputedStyle(canvas).getPropertyValue('--wave-color').trim() || '#bbd9bc';
  for (let i = 0; i < count; i++) {
    let peak = 0;
    if (data) for (let p = Math.floor(i*data.length/count); p < Math.floor((i+1)*data.length/count); p++) peak = Math.max(peak,Math.abs(data[p]));
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
      let energy = 0; for (const value of data) energy += value*value;
      if (Math.sqrt(energy/data.length) > .001) { $('stage-voice').dataset.active = 'true'; $('live-wave-empty').hidden = true; }
      const seconds = Math.floor(capture.totalSamples/capture.rate);
      $('live-time').textContent = fa(`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`);
    }
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
}
async function begin(file = null) {
  reset({preserveFile:!!file}); const run = generation; controller = new AbortController(); setPhase('connecting');
  status(file ? 'در حال آماده‌سازی پخش و بررسی…' : 'اجازهٔ دسترسی به میکروفن را تأیید کنید.');
  try {
    // Paint the pressed/connecting state before opening the audio device.
    await new Promise(resolve=>requestAnimationFrame(()=>setTimeout(resolve,0)));
    if (run !== generation) return;
    const source = await prepareAudioSource({fileAudio:file?.audio,signal:controller.signal,
      onSamples:block=>{ if (run === generation) { try { connection.sendSamples(block); } catch (err) { if (run === generation) fail(err.message); } } },
      onEnded:()=>{ if (run === generation) void finish(); }});
    if (run !== generation) { void source.stop({discard:true}); return; }
    capture = source;
    connection = new LiveConnection({onEvent:event=>{ if (run === generation) accept(event); },onError:message=>{ if (run === generation) fail(message); }});
    status('در حال برقراری ارتباط زنده…');
    await connection.connect({sampleRate:source.rate,startAyah:Number($('start-ayah').value)},controller.signal);
    if (run !== generation) return;
    setPhase(file ? 'file' : 'recording'); status('چند کلمهٔ اول را پیوسته بخوانید؛ هم‌خوان در حال یافتن محل خواندن است.');
    lastCanvasFrame = 0; source.start(); animate(run);
  } catch (err) {
    if (run !== generation) return;
    const messages = {NotAllowedError:'مجوز میکروفن داده نشد؛ اجازهٔ مرورگر را بررسی کنید یا فایل انتخاب کنید.',NotFoundError:'میکروفن پیدا نشد؛ اتصال آن را بررسی کنید یا فایل انتخاب کنید.',NotReadableError:'میکروفن در دسترس نیست؛ برنامه‌های دیگری را که از آن استفاده می‌کنند بررسی کنید.'};
    fail(messages[err.name] || (/[\u0600-\u06ff]/.test(err.message) ? err.message : 'شروع تمرین ممکن نشد؛ دوباره تلاش کنید.'));
  }
}
async function finish(remoteDone = false) {
  if (phase === 'connecting') { reset(); return; }
  if (!['recording','file'].includes(phase)) return;
  const run = generation; setPhase('stopping'); status('صدای آخر دریافت شد؛ در حال جمع‌بندی نتیجه…');
  try {
    const replay = await capture?.stop();
    if (run !== generation) return;
    capture = null; cancelAnimationFrame(raf); raf = null; drawWave();
    if (replay?.blob && replay.duration >= .2) {
      clearReplay(); replayURL = URL.createObjectURL(replay.blob); $('replay-audio').src = replayURL; $('replay-section').hidden = false;
    }
    if (!remoteDone) await connection.finish();
    if (run !== generation) return;
    connection.close(); setPhase('review'); summary(); $('session-summary').hidden = false;
    reportUI();
    status(states.size ? 'تمرین پایان یافت. کلمه‌های مشخص‌شده را مرور کنید.' : 'شواهد کافی برای بررسی پیدا نشد؛ از آیهٔ انتخاب‌شده دوباره بخوانید.');
  } catch (err) { if (run === generation) fail(err.message); }
}

$('live-start').addEventListener('click',()=>void begin());
$('replay-toggle').addEventListener('click',async()=>{
  const audio = $('replay-audio');
  try { if (audio.paused) await audio.play(); else audio.pause(); }
  catch { error('بازپخش صدا ممکن نشد؛ دوباره دکمهٔ پخش را بزنید.'); }
  paintReplay();
});
$('replay-seek').addEventListener('input',()=>{ const audio = $('replay-audio'); if (Number.isFinite(audio.duration)) audio.currentTime = Number($('replay-seek').value)/100*audio.duration; });
for (const event of ['loadedmetadata','timeupdate','play','pause','ended','emptied']) $('replay-audio').addEventListener(event,paintReplay);
$('live-stop').addEventListener('click',()=>void finish());
$('live-stop-mobile').addEventListener('click',()=>void finish());
$('live-reset').addEventListener('click',()=>reset());
$('start-ayah').addEventListener('change',()=>reset({preserveFile:true}));
$('live-file').addEventListener('change',async event=>{
  const file = event.target.files[0]; if (!file) return;
  reset(); const run = generation; setPhase('connecting'); status('در حال آماده‌سازی فایل…');
  try {
    const audio = await readAudioFile(file); if (run !== generation) return;
    chosenFile = {audio,name:file.name}; $('file-name').textContent = file.name;
    setPhase('idle'); status('فایل آماده است؛ «پخش و بررسی» را بزنید.');
  } catch (err) { if (run === generation) fail(err.message); }
});
$('file-play').addEventListener('click',()=>{ if (chosenFile) void begin(chosenFile); });
buttons.forEach((button,index)=>button.addEventListener('click',()=>showWord(index)));
$('word-close').addEventListener('click',closeWord);
$('word-retry').addEventListener('click',()=>{ if (selectedWord !== null) { $('start-ayah').value = String(words[selectedWord].ayah); void begin(); } });
$('auto-follow').addEventListener('change',()=>{ if ($('auto-follow').checked) { followedAyah = null; manualUntil = 0; follow(); } });
window.addEventListener('wheel',()=>{ manualUntil = performance.now()+5000; },{passive:true});
window.addEventListener('touchmove',()=>{ manualUntil = performance.now()+5000; },{passive:true});
window.addEventListener('scroll',()=>{ if (performance.now() > autoScrollUntil) manualUntil = performance.now()+5000; },{passive:true});
window.addEventListener('keydown',event=>{
  if (event.key === 'Escape' && $('word-detail').dataset.open === 'true') closeWord();
  if (!event.target.closest('input,select,button,textarea,a') && ['PageDown','PageUp','Home','End','ArrowDown','ArrowUp',' '].includes(event.key)) manualUntil = performance.now()+5000;
});
window.addEventListener('resize',()=>{ if (!capture) drawWave(); });
window.addEventListener('pagehide',()=>{ generation++; connection?.close(); controller?.abort(); void capture?.stop({discard:true}); cancelAnimationFrame(raf); if (replayURL) URL.revokeObjectURL(replayURL); });
reset();
if (!ready) { $('live-pill').textContent = 'مدل آماده نیست'; status('مدل آماده نیست؛ پیام اجرای برنامه را بررسی کنید.'); }
