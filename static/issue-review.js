// This reader also runs from a local file in the professor's review package.
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const number = new Intl.NumberFormat('fa', {maximumFractionDigits: 2});
  let audioURL = null;
  let sequence = 0;

  function clear() {
    byId('view-result').hidden = true;
    byId('view-error').textContent = '';
    byId('view-history').open = false;
    const player = byId('issue-audio');
    player.pause();
    player.removeAttribute('src');
    player.load();
    if (audioURL) URL.revokeObjectURL(audioURL);
    audioURL = null;
  }

  function validateWords(words) {
    if (!Array.isArray(words) || words.some(w => !w || typeof w.id !== 'string' || typeof w.text !== 'string')) {
      throw new Error('بخش واژه‌های این گزارش خواندنی نیست.');
    }
  }

  function audioBlob(audio) {
    if (audio?.encoding !== 'base64' || audio.mime_type !== 'audio/wav' || typeof audio.data !== 'string') {
      throw new Error('این گزارش صدای WAV قابل‌پخش ندارد.');
    }
    const bytes = Uint8Array.from(atob(audio.data), c => c.charCodeAt(0));
    const tag = index => String.fromCharCode(...bytes.subarray(index, index + 4));
    const header = new DataView(bytes.buffer);
    if (bytes.length <= 44 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE' || tag(36) !== 'data' ||
        header.getUint16(20, true) !== 1 || header.getUint16(22, true) !== 1 ||
        header.getUint16(34, true) !== 16 || header.getUint32(40, true) !== bytes.length - 44 ||
        ![16000, 44100, 48000].includes(audio.sample_rate) || header.getUint32(24, true) !== audio.sample_rate ||
        !Number.isFinite(audio.session_start_ms) || audio.session_start_ms < 0 ||
        !Number.isFinite(audio.session_end_ms)) throw new Error('اطلاعات صدای این گزارش ناقص است.');
    const duration = (bytes.length - 44) * 500 / audio.sample_rate;
    if (duration > 8000.1 || Math.abs(audio.session_end_ms - audio.session_start_ms - duration) > .1) {
      throw new Error('طول برش صوت با زمان ثبت‌شده هم‌خوانی ندارد.');
    }
    return new Blob([bytes], {type: 'audio/wav'});
  }

  function renderWords(parent, words, selected) {
    parent.replaceChildren();
    for (const word of words) {
      const element = document.createElement(selected.has(word.id) ? 'mark' : 'span');
      element.textContent = word.text;
      parent.append(element);
    }
    if (!words.length) parent.textContent = '—';
  }

  function show(report) {
    if (report?.schema !== 'hamkhan.issue.v1') throw new Error('فایل گزارش هم‌خوان نسخهٔ ۱ را انتخاب کنید.');
    validateWords(report.final_words);
    if (!Array.isArray(report.revisions) || !Array.isArray(report.selection?.token_ids)) {
      throw new Error('تاریخچه یا محل گزارش در این فایل مشخص نیست.');
    }
    for (const revision of report.revisions) validateWords(revision.words);
    const blob = audioBlob(report.audio);
    const selected = new Set(report.selection.token_ids);
    const seconds = t => number.format(t / 1000);
    byId('issue-range').textContent = `از ثانیهٔ ${seconds(report.audio.session_start_ms)} تا ${seconds(report.audio.session_end_ms)} نوبت خواندن` +
      (report.session_status === 'partial' ? ' · جلسه پیش از پایان بررسی قطع شده است.' : '');
    audioURL = URL.createObjectURL(blob);
    byId('issue-audio').src = audioURL;
    byId('view-note').hidden = !report.user_note;
    byId('issue-note-text').textContent = report.user_note || '';
    const selection = report.selection;
    const original = selection.initial_words || [];
    validateWords(original);
    if (selection.kind === 'playhead') {
      byId('view-target').textContent = 'خواننده این بخش را هنگام بازپخش صدا انتخاب کرده است.';
    } else if (selection.kind === 'gap') {
      byId('view-target').textContent = selection.expected ? `خواننده محل نشانهٔ جاافتادگی «${selection.expected}» را انتخاب کرده است.` : 'خواننده محل نشانهٔ جاافتادگی را انتخاب کرده است.';
    } else {
      const present = report.final_words.some(w => selected.has(w.id));
      const first = original[0]?.text?.trim();
      byId('view-target').textContent = present ? 'واژهٔ انتخاب‌شده در متن مشخص است.' :
        first ? `«${first}» هنگام انتخاب دیده می‌شد و در بازبینی بعدی تغییر کرد.` : 'واژهٔ انتخاب‌شده در بازبینی بعدی تغییر کرد.';
    }
    renderWords(byId('view-final'), report.final_words, selected);
    const feedback = byId('view-feedback');
    feedback.replaceChildren();
    for (const word of report.final_words.filter(w => selected.has(w.id))) {
      for (const issue of word.errors || []) {
        if (!issue?.message) continue;
        const line = document.createElement('li');
        line.textContent = `نظر برنامه دربارهٔ «${word.text.trim()}» — ${issue.message}`;
        feedback.append(line);
      }
    }
    const history = byId('history-list');
    history.replaceChildren();
    for (const revision of report.revisions) {
      const row = document.createElement('li');
      const raw = document.createElement('p');
      raw.lang = 'ar';
      renderWords(raw, revision.words, selected);
      row.append(raw);
      history.append(row);
    }
    byId('view-history-status').textContent = report.history_complete ?
      'متن‌های ثبت‌شده به‌ترتیب نمایش آمده‌اند.' : 'بخشی از تاریخچهٔ نمایش برای این بازه در دسترس است.';
    byId('view-result').hidden = false;
  }

  byId('issue-file').addEventListener('change', async event => {
    const request = ++sequence;
    clear();
    const file = event.target.files[0];
    if (!file) return;
    try {
      if (file.size > 12 * 1024 * 1024) throw new Error('فایل بسیار بزرگ است. فایل JSON همان بخش را انتخاب کنید.');
      const report = JSON.parse(await file.text());
      if (request === sequence) show(report);
    } catch (error) {
      if (request !== sequence) return;
      clear();
      byId('view-error').textContent = error instanceof SyntaxError || error.name === 'InvalidCharacterError' ?
        'فایل خوانده نشد. فایل JSON دانلودشده از هم‌خوان را دوباره انتخاب کنید.' : error.message;
    }
  });
  window.addEventListener('pagehide', event => { if (!event.persisted && audioURL) URL.revokeObjectURL(audioURL); });
})();
