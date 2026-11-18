// Version 2 JSON; the ordered PCM packet remains unchanged.
export function pcmPacket(samples, sequence, offset) {
  const buffer = new ArrayBuffer(8 + samples.length * 2), view = new DataView(buffer);
  view.setUint32(0, sequence, true); view.setUint32(4, offset, true);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(8 + i * 2, value * (value < 0 ? 32768 : 32767), true);
  }
  return buffer;
}

export class LiveConnection {
  constructor({onEvent, onError, socketFactory = (url) => new WebSocket(url)}) {
    this.onEvent = onEvent; this.onError = onError; this.socketFactory = socketFactory;
    this.sequence = 0; this.total = 0; this.acknowledged = 0; this.session = null;
    this.ackSequence = -1; this.pending = new Map();
    this.finished = false; this.intentionalClose = false; this.failed = false; this.revision = -1;
  }
  async connect({sampleRate, startSurah = null, startAyah = null}, signal) {
    this.rate = sampleRate;
    const url = new URL('/live/ws', window.location.href);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    this.socket = this.socketFactory(url.href);
    this.socket.binaryType = 'arraybuffer';
    const ready = new Promise((resolve, reject) => { this.resolveReady = resolve; this.rejectReady = reject; });
    this.openTimeout = setTimeout(() => this.fail('ارتباط زنده برقرار نشد؛ اجرای برنامه را بررسی کنید.'),10000);
    this.socket.onopen = () => {
      if (!this.intentionalClose && !this.failed) this.socket.send(JSON.stringify({type:'start',version:2,start_surah:startSurah,start_ayah:startAyah,sample_rate:sampleRate,channels:1}));
    };
    this.socket.onmessage = ({data}) => {
      if (this.intentionalClose || this.failed || this.finished) return;
      let event;
      try { event = JSON.parse(data); } catch { this.fail('پاسخ برنامه خوانده نشد؛ دوباره شروع کنید.'); return; }
      if (!event || typeof event !== 'object') { this.fail('پاسخ برنامه معتبر نیست.'); return; }
      if (event.type === 'ready') {
        if (this.session) return;
        if (event.version !== 2 || typeof event.session_id !== 'string') { this.fail('نسخهٔ ارتباط زنده سازگار نیست؛ صفحه را تازه کنید.'); return; }
        this.session = event.session_id; clearTimeout(this.openTimeout); this.resolveReady();
      } else if (event.type === 'error') {
        if (this.session && event.session_id !== this.session) return;
        if (event.fatal) { this.fail(event.message); return; }
      } else {
        if (!this.session || event.session_id !== this.session) return;
        if (event.type === 'ack') {
          if (event.sequence !== this.ackSequence+1 || !this.pending.has(event.sequence) || event.input_samples !== this.pending.get(event.sequence)) {
            this.fail('ترتیب دریافت صدا به هم خورد؛ خواندن را از نو شروع کنید.'); return;
          }
          this.acknowledged = event.input_samples;
          this.ackSequence = event.sequence; this.pending.delete(event.sequence);
        }
        if (Number.isInteger(event.revision)) {
          if (event.revision < this.revision) return;
          this.revision = event.revision;
        }
        if (event.type === 'done') {
          this.finished = true; clearTimeout(this.finishTimeout); this.resolveDone?.(event);
        }
      }
      this.onEvent(event);
    };
    this.socket.onerror = () => this.fail('ارتباط زنده قطع شد. برنامه را بررسی و دوباره شروع کنید.');
    this.socket.onclose = () => { if (!this.finished && !this.intentionalClose) this.fail('ارتباط با برنامه قطع شد؛ نتیجهٔ این نوبت کامل نیست.'); };
    this.abort = () => this.close();
    this.signal = signal; signal.addEventListener('abort',this.abort,{once:true});
    if (signal.aborted) this.close();
    await ready;
  }
  sendSamples(samples) {
    if (!this.session || this.socket.readyState !== 1 || this.failed || this.finished || this.intentionalClose || this.donePromise) throw new Error('ارتباط زنده آماده نیست.');
    if (!samples.length) return;
    if (this.total-this.acknowledged+samples.length > this.rate || this.socket.bufferedAmount > this.rate*2) {
      this.fail('ارسال صدا عقب افتاد و تمرین متوقف شد. دوباره شروع کنید تا بخشی از خواندن جا نیفتد.');
      throw new Error('ارسال صدا عقب افتاد.');
    }
    this.pending.set(this.sequence,this.total+samples.length);
    this.socket.send(pcmPacket(samples,this.sequence,this.total));
    this.total += samples.length; this.sequence++;
  }
  setReference(surah, ayah) {
    if (!this.session || this.socket.readyState !== 1 || this.failed || this.finished || this.intentionalClose || this.donePromise) throw new Error('ارتباط زنده آماده نیست.');
    if (!Number.isInteger(surah) || !Number.isInteger(ayah)) throw new Error('شمارهٔ سوره و آیه معتبر نیست.');
    this.socket.send(JSON.stringify({type:'reference',surah,ayah}));
  }
  finish() {
    if (this.finished) return Promise.resolve();
    if (this.donePromise) return this.donePromise;
    this.donePromise = new Promise((resolve,reject) => { this.resolveDone = resolve; this.rejectDone = reject; });
    if (this.socket.readyState !== 1 || this.failed) {
      this.rejectDone(new Error('ارتباط برای بررسی پایانی در دسترس نیست.'));
    } else {
      this.socket.send(JSON.stringify({type:'stop',last_sequence:this.sequence-1,total_samples:this.total}));
      this.finishTimeout = setTimeout(() => this.fail('بررسی پایانی بیش از حد طول کشید؛ دوباره تلاش کنید.'),60000);
    }
    return this.donePromise;
  }
  fail(message) {
    if (this.failed || this.intentionalClose || this.finished) return;
    this.failed = true;
    const error = new Error(message);
    this.rejectReady?.(error); this.rejectDone?.(error);
    this.dispose(); this.socket?.close(); this.onError(message);
  }
  dispose() {
    clearTimeout(this.openTimeout); clearTimeout(this.finishTimeout);
    this.pending.clear();
    this.signal?.removeEventListener('abort',this.abort);
  }
  close() {
    this.intentionalClose = true; this.dispose();
    const error = new DOMException('Cancelled','AbortError');
    this.rejectReady?.(error); this.rejectDone?.(error);
    this.socket?.close();
  }
}
