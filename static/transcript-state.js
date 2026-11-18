// Raw heard text and its annotations have independent lifetimes.
const STATES = new Set(['pending','uncertain','match','difference']);

function token(value) {
  if (!value || typeof value.id !== 'string' || typeof value.text !== 'string') throw new Error('کلمهٔ دریافتی معتبر نیست.');
  return {...value, state:STATES.has(value.state) ? value.state : 'pending',
    errors:Array.isArray(value.errors) ? value.errors.map(error=>({...error})) : [], reference:value.reference || null};
}

export class TranscriptState {
  constructor() { this.reset(); }
  reset() { this.tokens = []; this.gaps = []; this.committedCount = 0; this.index = new Map(); }
  applyTranscript(event) {
    const from = event.replace_from;
    if (!Number.isInteger(from) || from < this.committedCount || from > this.tokens.length || !Array.isArray(event.tokens)) throw new Error('ترتیب متن دریافتی معتبر نیست؛ تمرین را از نو شروع کنید.');
    const replacement = event.tokens.map(token);
    const ids = new Set();
    for (const word of replacement) {
      const prior = this.index.get(word.id);
      if (ids.has(word.id) || (prior !== undefined && prior < from)) throw new Error('شناسهٔ کلمه تکراری است؛ تمرین را از نو شروع کنید.');
      ids.add(word.id);
    }
    const committed = event.committed_count ?? this.committedCount;
    if (!Number.isInteger(committed) || committed < this.committedCount || committed > from+replacement.length) throw new Error('بخش تثبیت‌شدهٔ متن معتبر نیست.');
    const replacedIds = new Set(this.tokens.slice(from).map(word=>word.id));
    for (const id of replacedIds) this.index.delete(id);
    this.tokens.length = from;
    for (const word of replacement) {
      this.index.set(word.id,this.tokens.length);
      this.tokens.push(word);
    }
    this.committedCount = committed;
    this.gaps = this.gaps.filter(gap=>!replacedIds.has(gap.before_token_id) && !replacedIds.has(gap.after_token_id));
    return this.tokens;
  }
  applyFeedback(event) {
    for (const update of event.updates || []) {
      const i = this.index.get(update.id);
      if (i === undefined) continue;
      this.tokens[i] = {...this.tokens[i], state:STATES.has(update.state) ? update.state : 'uncertain',
        errors:Array.isArray(update.errors) ? update.errors.map(error=>({...error})) : [], reference:update.reference || null};
    }
    if (Array.isArray(event.gaps)) {
      const ids = this.index;
      this.gaps = event.gaps.filter(gap=>gap && typeof gap.id === 'string' &&
        (!gap.before_token_id || ids.has(gap.before_token_id)) && (!gap.after_token_id || ids.has(gap.after_token_id))).map(gap=>({...gap}));
    }
  }
  summary() {
    return {reviewed:this.tokens.filter(word=>['match','difference'].includes(word.state)).length,
      differences:this.tokens.filter(word=>word.state==='difference').length,
      uncertain:this.tokens.filter(word=>!['match','difference'].includes(word.state)).length, missing:this.gaps.length};
  }
}
