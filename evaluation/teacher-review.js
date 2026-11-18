(() => {
  'use strict';
  const get = id => document.getElementById(id);
  const packet = JSON.parse(get('packet').textContent);
  const tasks = packet.tasks;
  const taskIds = new Set(tasks.map(t => t.id));
  const number = n => n.toLocaleString('fa-IR', {useGrouping: false});
  const storageKey = 'hamkhan-teacher-draft:' + packet.id;
  const fields = ['expected', 'heard', 'start', 'end', 'note'];
  const answers = new Map();
  let index = 0;
  let mediaUrl = null;
  let stopAt = null;
  let rangeTimer;
  let needsExport = false;
  const audio = get('audio');
  const radios = [...document.querySelectorAll('input[name="status"]')];
  function message(text, error = false) {
    get('message').textContent = text;
    get('message').classList.toggle('error', error);
  }
  function blank() {
    return {verdict: null, expected: '', heard: '', start: '', end: '', note: ''};
  }
  function hasValue(a) {
    return a.verdict !== null || fields.some(k => a[k] !== '');
  }
  function progress() {
    const count = [...answers.values()].filter(a => a.verdict !== null).length;
    get('progress').textContent = number(count) + ' از ' + number(tasks.length) + ' پاسخ';
    tasks.forEach((t, i) => {
      const reviewed = Boolean(answers.get(t.id)?.verdict);
      get('jump').options[i].textContent = number(i + 1) + (reviewed ? ' · ثبت شده' : '');
    });
  }
  function saveDraft() {
    try {
      localStorage.setItem(storageKey, JSON.stringify({index, rows: [...answers], needsExport}));
    } catch {
      message('برای نگهداری پاسخ‌ها روی این دستگاه از «دانلود پاسخ‌ها» استفاده کنید.');
    }
  }
  function capture() {
    const data = blank();
    data.verdict = radios.find(r => r.checked)?.value || null;
    for (const name of fields) data[name] = get(name).value.trim();
    if (hasValue(data)) answers.set(tasks[index].id, data);
    else answers.delete(tasks[index].id);
    needsExport = true;
    progress();
    saveDraft();
  }
  function readDraft() {
    try {
      const draft = JSON.parse(localStorage.getItem(storageKey));
      if (!draft || !Array.isArray(draft.rows)) return;
      const rows = new Map(draft.rows);
      for (const [id, a] of rows) {
        if (!taskIds.has(id) || !a || ![null, 'match', 'difference', 'uncertain'].includes(a.verdict)
            || !fields.every(k => typeof a[k] === 'string' && a[k].length <= 1000)) return;
      }
      for (const [id, a] of rows) answers.set(id, a);
      if (Number.isInteger(draft.index) && tasks[draft.index]) index = draft.index;
      needsExport = Boolean(draft.needsExport);
    } catch { /* file:// storage can be disabled; JSON export works independently. */ }
  }
  function render(next) {
    index = next;
    const task = tasks[index];
    const a = answers.get(task.id) || blank();
    audio.pause();
    clearTimeout(rangeTimer);
    stopAt = null;
    if (mediaUrl) URL.revokeObjectURL(mediaUrl);
    const data = atob(packet.clips[task.clip].wav);
    const bytes = Uint8Array.from(data, c => c.charCodeAt(0));
    mediaUrl = URL.createObjectURL(new Blob([bytes], {type: 'audio/wav'}));
    audio.src = mediaUrl;
    audio.load();
    get('position').textContent = 'نمونهٔ ' + number(index + 1) + ' از ' + number(tasks.length);
    get('verse-ref').textContent = 'سورهٔ ' + number(task.surah) + '، آیهٔ ' + number(task.ayah);
    const verse = get('verse');
    verse.replaceChildren();
    task.words.forEach((word, i) => {
      if (i) verse.append(' ');
      if (i === task.word_index) {
        const mark = document.createElement('mark');
        mark.textContent = word;
        mark.setAttribute('aria-label', 'واژهٔ مورد بررسی، ' + word);
        verse.append(mark);
      } else verse.append(word);
    });
    get('jump').selectedIndex = index;
    get('prev').disabled = index === 0;
    get('next').disabled = index === tasks.length - 1;
    for (const r of radios) r.checked = r.value === a.verdict;
    for (const name of fields) get(name).value = a[name];
    get('timing').open = Boolean(a.start || a.end);
    progress();
    message('');
  }
  function seconds(raw) {
    const s = raw.replace(/[۰-۹]/g, ch => String(ch.charCodeAt(0) - 0x6f0))
      .replace(/[٠-٩]/g, ch => String(ch.charCodeAt(0) - 0x660)).replace(/[٫,]/g, '.');
    return /^(?:\d+(?:\.\d*)?|\.\d+)$/.test(s) ? Number(s) : NaN;
  }
  function times(a, task) {
    if (!a.start && !a.end) return {start_s: null, end_s: null};
    const start = seconds(a.start), end = seconds(a.end);
    if (!(start >= 0 && start < end && end <= packet.clips[task.clip].duration)) {
      throw new Error('آغاز و پایان را به ثانیه و در محدودهٔ همین فایل ثبت کنید.');
    }
    return {start_s: start, end_s: end};
  }
  function exportAnswers() {
    const rows = [];
    for (let i = 0; i < tasks.length; i++) {
      const task = tasks[i], a = answers.get(task.id);
      if (!a) continue;
      try {
        rows.push({task_id: task.id, verdict: a.verdict, expected: a.expected, heard: a.heard,
          ...times(a, task), note: a.note});
      } catch (e) {
        render(i);
        get('timing').open = true;
        get('start').focus();
        message(e.message, true);
        return;
      }
    }
    if (!rows.length) return message('ابتدا نظر خود را برای دست‌کم یک نمونه وارد کنید.', true);
    const payload = {schema: 'hamkhan-teacher-answers/v1', packet_id: packet.id, answers: rows};
    const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'hamkhan-teacher-answers-' + new Date().toISOString().replace(/[:.]/g, '-') + '.json';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 30000);
    needsExport = false;
    saveDraft();
    message('فایل پاسخ‌ها برای دانلود آماده شد. همین فایل JSON را برای فرستنده بفرستید.');
  }
  function validatedRows(obj) {
    if (obj?.schema !== 'hamkhan-teacher-answers/v1' || obj.packet_id !== packet.id
        || !Array.isArray(obj.answers) || obj.answers.length > tasks.length) throw new Error();
    const rows = new Map();
    for (const r of obj.answers) {
      if (!r || !taskIds.has(r.task_id) || rows.has(r.task_id)
          || ![null, 'match', 'difference', 'uncertain'].includes(r.verdict)) throw new Error();
      for (const [key, limit] of [['expected',200], ['heard',200], ['note',1000]]) {
        if (typeof r[key] !== 'string' || r[key].length > limit) throw new Error();
      }
      for (const key of ['start_s','end_s']) if (r[key] !== null && !Number.isFinite(r[key])) throw new Error();
      const a = {verdict:r.verdict, expected:r.expected, heard:r.heard, note:r.note,
        start:r.start_s === null ? '' : String(r.start_s), end:r.end_s === null ? '' : String(r.end_s)};
      times(a, tasks.find(t => t.id === r.task_id));
      rows.set(r.task_id, a);
    }
    return rows;
  }
  async function restore(event) {
    const file = event.target.files[0];
    if (!file) return;
    event.target.value = '';
    try {
      if (file.size > 500000) throw new Error();
      const rows = validatedRows(JSON.parse(await file.text()));
      const overlaps = [...rows].some(([id, a]) => {
        const old = answers.get(id);
        return old && ['verdict', ...fields].some(k => old[k] !== a[k]);
      });
      if (overlaps && !window.confirm('در این فایل برای بعضی نمونه‌ها پاسخ دیگری ثبت شده است. پاسخ فایل جایگزین پاسخ فعلی همان نمونه‌ها شود؟')) return;
      for (const [id, a] of rows) answers.set(id, a);
      const firstPending = tasks.findIndex(t => !answers.get(t.id)?.verdict);
      render(firstPending >= 0 ? firstPending : 0);
      needsExport = true;
      saveDraft();
      message(number(rows.size) + ' پاسخ از فایل بازیابی شد.');
    } catch {
      message('فایل پاسخ خوانده نشد. فایل JSON همین بسته را انتخاب کنید.', true);
    }
  }
  for (const t of tasks) get('jump').append(new Option('', t.id));
  for (const r of radios) r.addEventListener('change', capture);
  for (const name of fields) get(name).addEventListener('input', capture);
  for (const name of ['start', 'end']) get('stamp-' + name).addEventListener('click', () => {
    if (!Number.isFinite(audio.duration)) return message('ابتدا صدا را بارگذاری کنید.', true);
    const time = Math.min(audio.currentTime, packet.clips[tasks[index].clip].duration);
    // Round down so the last playable instant stays inside the original PCM.
    get(name).value = String(Math.floor(time * 1000) / 1000);
    capture();
  });
  get('listen-range').addEventListener('click', async () => {
    try {
      const {start_s, end_s} = times(answers.get(tasks[index].id) || blank(), tasks[index]);
      if (start_s === null) throw new Error('ابتدا آغاز و پایان بازه را ثبت کنید.');
      audio.currentTime = start_s;
      stopAt = end_s;
      await audio.play();
    } catch (e) { message(e.message, true); }
  });
  function checkRange() {
    clearTimeout(rangeTimer);
    if (stopAt !== null && audio.currentTime >= stopAt) {
      const end = stopAt;
      stopAt = null;
      audio.pause();
      audio.currentTime = end;
    } else if (stopAt !== null && !audio.paused) {
      rangeTimer = setTimeout(checkRange, Math.min(80, Math.max(8, (stopAt - audio.currentTime) * 1000 / audio.playbackRate)));
    }
  }
  audio.addEventListener('playing', checkRange);
  audio.addEventListener('timeupdate', checkRange);
  audio.addEventListener('pause', () => { clearTimeout(rangeTimer); stopAt = null; });
  audio.addEventListener('ended', () => { stopAt = null; });
  function navigate(i) { render(i); saveDraft(); }
  get('prev').addEventListener('click', () => navigate(index - 1));
  get('next').addEventListener('click', () => navigate(index + 1));
  get('jump').addEventListener('change', e => navigate(e.target.selectedIndex));
  get('save').addEventListener('click', exportAnswers);
  get('restore').addEventListener('change', restore);
  window.addEventListener('beforeunload', e => { if (needsExport) { e.preventDefault(); e.returnValue = ''; } });
  readDraft();
  render(index);
})();
