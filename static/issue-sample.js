// The browser retains transcript patches for one session. A report replays only
// the words intersecting its audio excerpt, including words later removed.
const MAX_AUDIO_MS = 8000;
const CONTEXT_MS = 1500;
const validTime = value => Number.isFinite(value) && value >= 0;
const timed = word => word && validTime(word.start_ms) && validTime(word.end_ms) && word.end_ms >= word.start_ms;
const wordFields = ({id, text, start_ms, end_ms}) => ({id, text, start_ms, end_ms});
const intersects = (word, start, end) => timed(word) && word.start_ms < end &&
  (word.end_ms > start || (word.start_ms === word.end_ms && word.start_ms === start));

export function issueAnchor(kind, words, positionMs) {
  if (kind === 'playhead') return validTime(positionMs) ? {kind, words:[], start_ms:positionMs, end_ms:positionMs} : null;
  if (!['word','gap'].includes(kind) || !words.length || !words.every(timed)) return null;
  return {kind, words:words.map(wordFields), start_ms:Math.min(...words.map(w => w.start_ms)),
    end_ms:Math.max(...words.map(w => w.end_ms))};
}

export function wordIssueAnchor(tokens, index) {
  const word = tokens[index];
  if (!word) return null;
  const direct = issueAnchor('word',[word]);
  if (direct) return direct;
  const before = tokens[index-1], after = tokens[index+1];
  const lo = timed(before) ? before.end_ms : timed(after) ? after.start_ms : null;
  const hi = timed(after) ? after.start_ms : lo;
  if (!validTime(lo) || !validTime(hi) || Math.abs(hi-lo) > MAX_AUDIO_MS) return null;
  // An explicit click can preserve this untimed token. Its real timestamps
  // stay unknown; the adjacent timed word only locates the audio excerpt.
  return {kind:'word',words:[wordFields(word)],start_ms:Math.min(lo,hi),end_ms:Math.max(lo,hi),
    timing_source:'neighbor'};
}

export function issueBounds(anchor, durationMs) {
  if (!anchor || !validTime(anchor.start_ms) || !validTime(anchor.end_ms) ||
      anchor.end_ms < anchor.start_ms || !(durationMs > 0) || anchor.start_ms > durationMs) {
    throw new Error('برای این بخش زمان صوت در دسترس نیست. یک واژهٔ همسایه یا زمان بازپخش را انتخاب کنید.');
  }
  const center = Math.min(durationMs, (anchor.start_ms + anchor.end_ms) / 2);
  const start = Math.max(0, anchor.start_ms - CONTEXT_MS, center - MAX_AUDIO_MS / 2);
  const end = Math.min(durationMs, anchor.end_ms + CONTEXT_MS, center + MAX_AUDIO_MS / 2);
  return {start, end};
}

export class IssueHistory {
  constructor(maxEntries = 4096, maxWords = 65536) {
    this.maxEntries = maxEntries;
    this.maxWords = maxWords;
    this.clear();
  }
  clear() {
    this.entries = [];
    this.head = 0;
    this.savedWords = 0;
    this.length = 0;
    this.complete = true;
  }
  record({revision, replace_from:from, tokens}) {
    // Called after the transcript reducer accepts the event for this session.
    const previousLength = this.length;
    this.length = from + tokens.length;
    if (from === previousLength && !tokens.length) return;
    if (tokens.length > this.maxWords) {
      this.clear();
      this.complete = false;
      this.length = from + tokens.length;
      return;
    }
    const words = tokens.map(wordFields);
    this.entries.push({revision, from, words});
    this.savedWords += words.length;
    while (this.entries.length - this.head > this.maxEntries || this.savedWords > this.maxWords) {
      this.savedWords -= this.entries[this.head].words.length;
      this.entries[this.head++] = null;
      this.complete = false;
    }
    if (this.head >= 512) {
      this.entries = this.entries.slice(this.head);
      this.head = 0;
    }
  }
  excerpt(start, end, untimedSelection = new Set()) {
    const current = new Map(), revisions = [];
    let lastText = null, complete = this.complete;
    for (let i = this.head; i < this.entries.length; i++) {
      const {revision, from, words} = this.entries[i];
      let touched = false;
      for (const index of current.keys()) {
        if (index >= from) { current.delete(index); touched = true; }
      }
      words.forEach((word, offset) => {
        if (intersects(word,start,end) || (!timed(word) && untimedSelection.has(word.id))) {
          current.set(from + offset,word); touched = true;
        }
      });
      if (!touched) continue;
      if (current.size > 128) { complete = false; lastText = null; continue; }
      const scoped = [...current.values()];
      const signature = JSON.stringify(scoped);
      if (signature !== lastText) {
        if (revisions.length === 96) {
          revisions.splice(1,1);  // Retain the first displayed version plus the latest changes.
          complete = false;
        }
        revisions.push({revision, words:scoped});
      }
      lastText = signature;
    }
    return {complete, revisions};
  }
}

function reviewedWord(word) {
  const ref = word.reference;
  return {...wordFields(word), state:word.state,
    reference:ref ? {surah:ref.surah, ayah:ref.ayah, word_index:ref.word_index, word:ref.word} : null,
    errors:(word.errors || []).map(({type, expected, heard, message}) => ({type, expected, heard, message}))};
}

export async function buildIssueSample({recording, anchor, history, tokens, note = '', partial = false}) {
  // SessionRecording produces exactly 44 header bytes followed by mono PCM16.
  const header = new Uint8Array(await recording.slice(0,44).arrayBuffer());
  const view = new DataView(header.buffer);
  if (header.length !== 44 || view.getUint32(0) !== 0x52494646 || view.getUint32(8) !== 0x57415645 ||
      view.getUint32(36) !== 0x64617461 || view.getUint16(20,true) !== 1 || view.getUint16(22,true) !== 1 ||
      view.getUint16(34,true) !== 16 || view.getUint32(40,true) !== recording.size - 44) {
    throw new Error('صدای این نوبت برای تهیهٔ نمونه معتبر نیست.');
  }
  const rate = view.getUint32(24,true), total = (recording.size - 44) / 2;
  if (![16000,44100,48000].includes(rate) || !Number.isInteger(total)) throw new Error('نرخ صوت نمونه معتبر نیست.');
  const {start, end} = issueBounds(anchor, total * 1000 / rate);
  const first = Math.floor(start * rate / 1000);
  const last = Math.min(total, first + MAX_AUDIO_MS * rate / 1000, Math.ceil(end * rate / 1000));
  if (last <= first) throw new Error('صدای کافی برای این نمونه در دسترس نیست.');
  const pcm = recording.slice(44 + 2*first, 44 + 2*last);
  view.setUint32(4,36+pcm.size,true);
  view.setUint32(40,pcm.size,true);
  const wav = new Uint8Array(await new Blob([header,pcm]).arrayBuffer());
  let binary = '';
  for (let offset = 0; offset < wav.length; offset += 8192) {
    binary += String.fromCharCode(...wav.subarray(offset, offset + 8192));
  }
  const lo = first * 1000 / rate, hi = last * 1000 / rate;
  const untimedSelection = new Set(anchor.words.filter(word=>!timed(word)).map(word=>word.id));
  const inExcerpt = word => intersects(word,lo,hi) || (!timed(word) && untimedSelection.has(word.id));
  const excerpt = history.excerpt(lo,hi,untimedSelection);
  const selection = {kind:anchor.kind, token_ids:anchor.words.map(w=>w.id),
    start_ms:anchor.start_ms, end_ms:anchor.end_ms, initial_words:anchor.words.filter(inExcerpt)};
  if (anchor.timing_source) selection.timing_source = anchor.timing_source;
  if (anchor.kind === 'gap' && anchor.expected) selection.expected = anchor.expected;
  return {schema:'hamkhan.issue.v1', session_status:partial ? 'partial' : 'finished',
    audio:{mime_type:'audio/wav', encoding:'base64', data:btoa(binary), sample_rate:rate,
      session_start_ms:lo, session_end_ms:hi},
    selection, user_note:note.trim().slice(0,500),
    history_complete:excerpt.complete, revisions:excerpt.revisions,
    final_words:tokens.filter(inExcerpt).map(reviewedWord)};
}
