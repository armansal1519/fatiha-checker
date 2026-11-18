// Audio only: each request is a self-contained PCM WAV, never a WebM fragment.
export class AudioWindow {
  constructor(seconds = 10) { this.seconds = seconds; this.reset(); }
  reset() { this.blocks = []; this.length = 0; this.total = 0; this.rate = 16000; }
  push(block, rate) {
    if (this.length && rate !== this.rate) throw new Error("Sample rate changed");
    this.rate = rate; this.blocks.push(block.slice()); this.length += block.length; this.total += block.length;
    const limit = Math.round(rate * this.seconds);
    while (this.length > limit) {
      const excess = this.length - limit;
      if (this.blocks[0].length <= excess) this.length -= this.blocks.shift().length;
      else { this.blocks[0] = this.blocks[0].slice(excess); this.length -= excess; }
    }
  }
  wav({seconds = this.seconds, endTotal = this.total} = {}) {
    const retainedStart = this.total - this.length;
    const end = Math.max(retainedStart, Math.min(this.total, Math.floor(endTotal)));
    const start = Math.max(retainedStart, end - Math.round(seconds * this.rate));
    const length = end - start;
    const bytes = new ArrayBuffer(44 + length * 2), view = new DataView(bytes);
    const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
    text(0, "RIFF"); view.setUint32(4, bytes.byteLength - 8, true); text(8, "WAVE"); text(12, "fmt ");
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, this.rate, true); view.setUint32(28, this.rate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, "data"); view.setUint32(40, length * 2, true);
    let offset = 44, blockStart = retainedStart;
    for (const block of this.blocks) {
      const a = Math.max(0, start-blockStart), b = Math.min(block.length, end-blockStart);
      for (let i = a; i < b; i++) {
        const sample = Math.max(-1, Math.min(1, block[i]));
        view.setInt16(offset, sample * (sample < 0 ? 32768 : 32767), true); offset += 2;
      }
      blockStart += block.length;
    }
    return new Blob([bytes], {type: "audio/wav"});
  }
}

export async function openMicrophone(onSamples, signal) {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error("ضبط زنده در دسترس نیست؛ از Chrome یا Edge روی همین دستگاه استفاده کنید.");
  let media = null, context = null, node = null, source = null, mute = null, ended = false, timer, flushed;
  const close = () => {
    ended = true; clearTimeout(timer);
    if (node) { node.port.onmessage = null; node.disconnect(); }
    source?.disconnect(); mute?.disconnect();
    media?.getTracks().forEach((t) => t.stop());
    if (context && context.state !== "closed") void context.close();
  };
  let rejectAbort;
  const aborted = new Promise((_, reject) => { rejectAbort = reject; });
  const onAbort = () => { close(); rejectAbort(new DOMException("Cancelled", "AbortError")); };
  signal.addEventListener("abort", onAbort, {once:true});
  const request = navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false, noiseSuppression:false, autoGainControl:false}}).then((stream) => {
    if (ended || signal.aborted) { stream.getTracks().forEach((t) => t.stop()); throw new DOMException("Cancelled", "AbortError"); }
    return stream;
  });
  try {
    media = await Promise.race([request, aborted, new Promise((_, reject) => {
      timer = setTimeout(() => { close(); reject(new Error("پاسخی از میکروفن دریافت نشد؛ مجوز مرورگر را بررسی کنید یا یک فایل صوتی انتخاب کنید.")); }, 20000);
    })]);
    clearTimeout(timer);
    context = new AudioContext({sampleRate:16000});
    await context.resume();
    await context.audioWorklet.addModule("/static/pcm-worklet.js");
    if (signal.aborted || ended) throw new DOMException("Cancelled", "AbortError");
    source = context.createMediaStreamSource(media);
    node = new AudioWorkletNode(context, "pcm-capture");
    mute = context.createGain(); mute.gain.value = 0;
    node.port.onmessage = ({data}) => {
      if (data.flushed) flushed?.();
      else if (!ended) onSamples(new Float32Array(data), context.sampleRate);
    };
    source.connect(node); node.connect(mute); mute.connect(context.destination);
    return {stop:async () => {
      media.getTracks().forEach((t) => t.stop());
      if (!ended) await new Promise((resolve) => {
        const limit = setTimeout(resolve, 250);
        flushed = () => { clearTimeout(limit); resolve(); };
        node.port.postMessage("flush");
      });
      signal.removeEventListener("abort", onAbort); close();
    }};
  } catch (error) { signal.removeEventListener("abort", onAbort); close(); throw error; }
}

export async function readAudioFile(file) {
  if (file.size > 100 * 1024 * 1024) throw new Error("حجم فایل باید حداکثر ۱۰۰ مگابایت باشد.");
  return decodePracticeBlob(file);
}

export async function decodePracticeBlob(file) {
  const decoder = new OfflineAudioContext(1, 1, 16000);
  let audio;
  try { audio = await decoder.decodeAudioData(await file.arrayBuffer()); }
  catch { throw new Error("مرورگر این فایل را نخواند؛ فایل WAV یا MP3 دیگری انتخاب کنید."); }
  if (audio.duration < .5 || audio.duration > 1800) throw new Error("مدت فایل باید بین نیم ثانیه و ۳۰ دقیقه باشد.");
  const samples = new Float32Array(audio.length);
  for (let channel = 0; channel < audio.numberOfChannels; channel++) {
    const data = audio.getChannelData(channel);
    for (let i = 0; i < data.length; i++) samples[i] += data[i] / audio.numberOfChannels;
  }
  return {samples, rate:audio.sampleRate};
}
