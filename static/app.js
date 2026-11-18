"use strict";
let verses = JSON.parse(document.getElementById("verse-data").textContent);
let surah = 1, surahCount = 7, surahs = [], verseRequest = 0;
const ready = document.body.dataset.ready === "true";
const $ = (id) => document.getElementById(id);
const fa = (value) => String(value).replace(/\d/g, (d) => "۰۱۲۳۴۵۶۷۸۹"[d]);
let ayah = 1, phase = "idle", audioFile = null, previewUrl = null;
let recorder = null, stream = null, timer = null, stopTimer = null, started = 0;
let micRequest = 0, waitingForMic = false, permissionTimer = null;

function message(text) { $("message").textContent = text; $("message").hidden = !text; }
function setPhase(next) {
  phase = next;
  const locked = next !== "idle";
  document.querySelectorAll(".verse-button").forEach((b) => b.disabled = locked);
  $("precise-surah").disabled = locked; $("precise-ayah").disabled = locked;
  $("audio-file").disabled = locked || !ready;
  $("upload-zone").classList.toggle("disabled", locked || !ready);
  $("record-button").disabled = !ready || next === "processing" || next === "permission" || next === "loading";
  $("record-button").classList.toggle("recording", next === "recording");
  $("record-button").setAttribute("aria-label", next === "recording" ? "توقف ضبط" : "شروع ضبط");
  $("analyze-button").disabled = !ready || locked || !audioFile;
  $("analyze-label").textContent = next === "processing" ? "در حال بررسی…" : "بررسی قرائت";
  $("processing-hint").hidden = next !== "processing";
  $("retry-button").disabled = locked;
  $("next-button").disabled = locked;
  $("cancel-mic").hidden = !waitingForMic;
}
function renderVerse(flagged = []) {
  const container = $("verse-text");
  container.replaceChildren();
  const selected = verses.find(verse => verse.ayah === ayah);
  if (!selected) return;
  selected.words.forEach((word, i) => {
    const span = document.createElement("span");
    span.textContent = word;
    span.className = "word" + (flagged.includes(i) ? " flagged" : "");
    if (flagged.includes(i)) span.title = "اختلاف احتمالی؛ توضیح را پایین صفحه ببینید.";
    container.append(span);
  });
}
function clearAudio() {
  $("audio-player").pause();
  $("audio-player").removeAttribute("src");
  $("audio-player").load();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null; audioFile = null;
  $("audio-preview").hidden = true;
  $("audio-file").value = "";
  $("file-label").replaceChildren(document.createTextNode("انتخاب فایل صوتی"));
  const small = document.createElement("small");
  small.textContent = "WAV · MP3 · M4A · WebM · OGG — تا ۱۰ مگابایت";
  $("file-label").append(small);
  $("record-time").textContent = "۰۰:۰۰";
}
function paintVerseSelection() {
  $("ayah-label").textContent = `آیهٔ ${fa(ayah)} از ${fa(surahCount)}`;
  $("precise-ayah").value = String(ayah);
  $("verse-nav").replaceChildren();
  verses.forEach(verse => {
    const button = document.createElement("button"); button.className = "verse-button";
    const selected = verse.ayah === ayah; button.classList.toggle("selected",selected); button.setAttribute("aria-current",String(selected));
    button.dataset.ayah = String(verse.ayah);
    const number = document.createElement("span"); number.className = "verse-number"; number.textContent = fa(verse.ayah);
    const preview = document.createElement("span"); preview.className = "verse-preview"; preview.lang = "ar"; preview.textContent = verse.text;
    button.append(number,preview); button.addEventListener("click",()=>void selectAyah(verse.ayah)); $("verse-nav").append(button);
  });
  $("next-button").hidden = ayah >= surahCount;
}
async function selectAyah(number, chosenSurah = surah) {
  if (phase !== "idle") return;
  const request = ++verseRequest;
  message(""); setPhase("loading");
  try {
    const start = Math.max(1, number-2);
    const response = await fetch(`/api/quran/ayahs?surah=${chosenSurah}&start=${start}&limit=5`);
    if (!response.ok) throw new Error(); const result = await response.json();
    if (request !== verseRequest) return;
    if (!Array.isArray(result.ayahs) || !result.ayahs.some(verse=>verse.ayah===number)) throw new Error();
    clearAudio(); $("result-panel").hidden = true;
    surah = chosenSurah; ayah = number; verses = result.ayahs; surahCount = result.total;
    const metadata = surahs.find(item=>item.surah===surah);
    $("precise-surah-number").textContent = `سورهٔ ${fa(surah)}`;
    $("precise-surah-name").textContent = metadata?.name || `سورهٔ ${fa(surah)}`;
    $("precise-ayah-count").textContent = fa(surahCount);
    $("precise-ayah").replaceChildren();
    for (let index=1; index<=surahCount; index++) { const option=document.createElement("option"); option.value=String(index); option.textContent=fa(index); $("precise-ayah").append(option); }
    renderVerse(); paintVerseSelection();
    $("record-label").textContent = "برای شروع ضبط کلیک کنید";
    $("record-detail").textContent = "اجازهٔ دسترسی به میکروفن را تأیید کنید.";
  } catch { if (request===verseRequest) { message("دریافت آیه ممکن نشد؛ انتخاب قبلی حفظ شد. دوباره انتخاب کنید."); $("precise-surah").value=String(surah); $("precise-ayah").value=String(ayah); } }
  finally { if (request === verseRequest) setPhase("idle"); }
}
async function loadCatalogue() {
  try {
    const response=await fetch('/api/quran/surahs'); if (!response.ok) throw new Error();
    const result=await response.json(); if (!Array.isArray(result.surahs)) throw new Error(); surahs=result.surahs;
    $("precise-surah").replaceChildren();
    surahs.forEach(item=>{ const option=document.createElement('option'); option.value=String(item.surah); option.textContent=`${fa(item.surah)} · ${item.name}`; $("precise-surah").append(option); });
    $("precise-catalogue-status").textContent = "۱۱۴ سوره · روایت حفص";
  } catch { $("precise-catalogue-status").textContent = "فهرست سوره‌ها دریافت نشد؛ صفحه را تازه کنید."; }
}
function setAudio(file) {
  clearAudio(); message("");
  $("result-panel").hidden = true; renderVerse();
  if (!file.size) { message("فایل صوتی خالی است. دوباره ضبط کنید."); setPhase("idle"); return; }
  if (file.size > 10 * 1024 * 1024) { message("حجم فایل باید حداکثر ۱۰ مگابایت باشد."); setPhase("idle"); return; }
  audioFile = file;
  previewUrl = URL.createObjectURL(file);
  $("audio-player").src = previewUrl;
  $("audio-preview").hidden = false;
  $("file-label").textContent = file.name;
  $("record-label").textContent = "صدای شما آمادهٔ بررسی است";
  $("record-detail").textContent = "می‌توانید پیش از بررسی، آن را گوش کنید.";
  setPhase("idle");
}
function cleanupStream() {
  clearInterval(timer); clearTimeout(stopTimer);
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
}
function stopRecording() {
  clearInterval(timer); clearTimeout(stopTimer);
  if (recorder && recorder.state === "recording") {
    setPhase("permission");
    $("record-label").textContent = "در حال آماده‌سازی صدا…";
    recorder.stop();
  }
}
function cancelMicRequest(text = "می‌توانید فایل صوتی انتخاب کنید یا دوباره ضبط را شروع کنید.") {
  if (!waitingForMic) return;
  micRequest += 1; waitingForMic = false; clearTimeout(permissionTimer);
  $("record-label").textContent = "ضبط صدا شروع نشد";
  $("record-detail").textContent = "انتخاب فایل صوتی در دسترس است.";
  message(text); setPhase("idle");
}
async function startRecording() {
  message("");
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    message("ضبط در این مرورگر در دسترس نیست. برنامه را در Chrome یا Edge روی localhost باز کنید، یا فایل بارگذاری کنید."); return;
  }
  const requestId = ++micRequest;
  waitingForMic = true;
  setPhase("permission");
  $("record-label").textContent = "در انتظار دسترسی به میکروفن…";
  permissionTimer = setTimeout(() => cancelMicRequest("پاسخی از میکروفن دریافت نشد. مجوز مرورگر را بررسی کنید یا فایل صوتی انتخاب کنید."), 20000);
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
    if (requestId !== micRequest) { acquired.getTracks().forEach((track) => track.stop()); return; }
    clearTimeout(permissionTimer); waitingForMic = false; stream = acquired;
    clearAudio(); $("result-panel").hidden = true; renderVerse();
    const mimeType = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"].find((m) => MediaRecorder.isTypeSupported(m));
    recorder = new MediaRecorder(stream, mimeType ? {mimeType} : {});
    const chunks = []; let failed = false;
    recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
    recorder.onerror = () => { failed = true; cleanupStream(); setPhase("idle"); message("ضبط صدا قطع شد. دوباره ضبط کنید یا فایل انتخاب کنید."); };
    recorder.onstop = () => {
      const type = recorder.mimeType || mimeType || "audio/webm";
      cleanupStream();
      if (failed) return;
      const extension = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
      setAudio(new File(chunks, `ayah-${ayah}.${extension}`, {type}));
    };
    recorder.start(250);
    started = Date.now();
    timer = setInterval(() => { $("record-time").textContent = "۰۰:" + fa(String(Math.min(30, Math.floor((Date.now() - started) / 1000))).padStart(2, "0")); }, 200);
    stopTimer = setTimeout(stopRecording, 29500);
    $("record-label").textContent = "در حال ضبط… برای توقف کلیک کنید";
    $("record-detail").textContent = "فقط همین آیه را بخوانید و در پایان توقف کنید.";
    setPhase("recording");
  } catch (error) {
    if (requestId !== micRequest) return;
    clearTimeout(permissionTimer); waitingForMic = false;
    cleanupStream(); setPhase("idle");
    $("record-label").textContent = "ضبط صدا شروع نشد";
    $("record-detail").textContent = "می‌توانید از گزینهٔ انتخاب فایل استفاده کنید.";
    message(error.name === "NotAllowedError" ? "دسترسی به میکروفن داده نشد. اجازهٔ مرورگر را تغییر دهید یا فایل صوتی انتخاب کنید." : "میکروفن در دسترس نیست. اتصال آن را بررسی کنید یا فایل صوتی انتخاب کنید.");
  }
}
function showResult(result) {
  const errors = result.errors || [];
  renderVerse(errors.map((item) => item.word_index));
  $("result-title").textContent = result.status === "needs_retry" ? "به ضبط دوباره نیاز است" : "نتیجهٔ بررسی";
  $("result-message").textContent = result.message;
  $("elapsed").textContent = `${fa(result.elapsed_seconds)} ثانیه`;
  $("result-errors").replaceChildren();
  const groups = new Map();
  errors.forEach((item) => {
    if (!groups.has(item.word_index)) groups.set(item.word_index, {word: item.word, messages: new Set()});
    groups.get(item.word_index).messages.add(item.message);
  });
  groups.forEach((group) => {
    const section = document.createElement("div"); section.className = "error-group";
    const title = document.createElement("h3"); title.lang = "ar"; title.textContent = group.word; section.append(title);
    group.messages.forEach((text) => { const p = document.createElement("p"); p.textContent = text; section.append(p); });
    $("result-errors").append(section);
  });
  $("next-button").hidden = ayah >= surahCount;
  $("result-panel").hidden = false;
  $("result-panel").scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "nearest"});
}
$("record-button").addEventListener("click", () => phase === "recording" ? stopRecording() : startRecording());
$("cancel-mic").addEventListener("click", () => cancelMicRequest());
$("audio-file").addEventListener("change", (event) => { const file = event.target.files[0]; if (file) setAudio(file); });
$("precise-surah").addEventListener("change",()=>void selectAyah(1,Number($("precise-surah").value)));
$("precise-ayah").addEventListener("change",()=>void selectAyah(Number($("precise-ayah").value)));
$("retry-button").addEventListener("click", () => { selectAyah(ayah); $("record-button").focus(); });
$("next-button").addEventListener("click", () => { if (ayah < surahCount) void selectAyah(ayah + 1); });
$("analyze-button").addEventListener("click", async () => {
  if (!audioFile || phase !== "idle") return;
  message(""); setPhase("processing"); $("audio-player").pause(); $("result-panel").hidden = true;
  const data = new FormData(); data.append("audio", audioFile); data.append("ayah", String(ayah)); data.append("surah", String(surah));
  try {
    const response = await fetch("/analyze", {method: "POST", body: data});
    const result = await response.json();
    if (!response.ok) message(result.message || "بررسی صدا انجام نشد.");
    else showResult(result);
  } catch { message("ارتباط با برنامه برقرار نشد. بازبودن پنجرهٔ اجرای پایتون را بررسی کنید."); }
  finally { setPhase("idle"); }
});
window.addEventListener("pagehide", () => { micRequest += 1; clearTimeout(permissionTimer); cleanupStream(); if (previewUrl) URL.revokeObjectURL(previewUrl); });
renderVerse(); paintVerseSelection(); setPhase("idle"); void loadCatalogue();
