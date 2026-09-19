import {pcmPacket} from './live-wire.js';

// Whole-session replay is browser memory only; the server keeps its short ring.
export class SessionRecording {
  constructor(rate, limitSeconds = 180) { this.rate = rate; this.limit = Math.round(rate*limitSeconds); this.total = 0; this.chunks = []; }
  append(block) {
    const accepted = block.subarray(0,Math.max(0,this.limit-this.total));
    if (accepted.length) { this.chunks.push(pcmPacket(accepted,0,0).slice(8)); this.total += accepted.length; }
    return accepted;
  }
  blob() {
    const header = new ArrayBuffer(44), view = new DataView(header);
    const text = (at,s) => [...s].forEach((c,i) => view.setUint8(at+i,c.charCodeAt(0)));
    text(0,'RIFF'); view.setUint32(4,36+this.total*2,true); text(8,'WAVE'); text(12,'fmt ');
    view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
    view.setUint32(24,this.rate,true); view.setUint32(28,this.rate*2,true);
    view.setUint16(32,2,true); view.setUint16(34,16,true); text(36,'data'); view.setUint32(40,this.total*2,true);
    return new Blob([header,...this.chunks],{type:'audio/wav'});
  }
  clear() { this.chunks = []; this.total = 0; }
}

export async function prepareAudioSource({fileAudio = null, onSamples, onEnded, signal}) {
  if (!window.AudioWorkletNode || (!fileAudio && !navigator.mediaDevices?.getUserMedia)) throw new Error('ضبط زنده در این مرورگر در دسترس نیست؛ از Chrome یا Edge استفاده کنید.');
  let media = null, context = null, source = null, node = null, analyser = null, mute = null;
  let closed = false, started = false, stopping = false, timer, stopPromise, flushed;
  const close = () => {
    closed = true; clearTimeout(timer);
    media?.getTracks().forEach(t=>t.stop());
    if (node) { node.port.onmessage = null; node.disconnect(); }
    if (source) { source.disconnect(); if (fileAudio && started) { try { source.stop(); } catch {} } }
    analyser?.disconnect(); mute?.disconnect();
    if (context && context.state !== 'closed') void context.close();
    signal.removeEventListener('abort',onAbort);
  };
  let rejectAbort;
  const aborted = new Promise((_,reject)=>{ rejectAbort = reject; });
  const onAbort = () => { close(); rejectAbort(new DOMException('Cancelled','AbortError')); };
  signal.addEventListener('abort',onAbort,{once:true});
  try {
    if (signal.aborted) throw new DOMException('Cancelled','AbortError');
    if (!fileAudio) {
      const permission = navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}}).then(stream=>{
        if (closed || signal.aborted) { stream.getTracks().forEach(t=>t.stop()); throw new DOMException('Cancelled','AbortError'); }
        return stream;
      });
      media = await Promise.race([permission,aborted,new Promise((_,reject)=>{
        timer = setTimeout(()=>{ close(); reject(new Error('پاسخی از میکروفن دریافت نشد؛ مجوز مرورگر را بررسی کنید یا فایل انتخاب کنید.')); },20000);
      })]);
      clearTimeout(timer);
    } else aborted.catch(()=>{});
    try { context = new AudioContext({sampleRate:16000}); } catch { context = new AudioContext(); }
    await context.resume();
    await context.audioWorklet.addModule('/static/pcm-worklet.js');
    if (closed || signal.aborted) throw new DOMException('Cancelled','AbortError');
    if (![16000,44100,48000].includes(context.sampleRate)) throw new Error('نرخ صوت این دستگاه پشتیبانی نمی‌شود؛ فایل صوتی دیگری انتخاب کنید.');
    analyser = context.createAnalyser(); analyser.fftSize = 1024; analyser.smoothingTimeConstant = .6;
    node = new AudioWorkletNode(context,'pcm-capture'); mute = context.createGain(); mute.gain.value = 0;
    const recording = new SessionRecording(context.sampleRate);
    if (fileAudio) {
      source = context.createBufferSource();
      const audioBuffer = context.createBuffer(1,fileAudio.samples.length,fileAudio.rate);
      audioBuffer.copyToChannel(fileAudio.samples,0); source.buffer = audioBuffer;
      source.onended = () => { if (started && !stopping && !closed) onEnded(); };
    } else {
      source = context.createMediaStreamSource(media);
      media.getTracks().forEach(t=>t.addEventListener('ended',()=>{ if (started && !stopping && !closed) onEnded(); },{once:true}));
    }
    node.port.onmessage = ({data}) => {
      if (data.flushed) { flushed?.(); return; }
      if (!started || closed) return;
      const accepted = recording.append(new Float32Array(data));
      if (accepted.length) onSamples(accepted);
      if (recording.total === recording.limit && !stopping) onEnded();
    };
    analyser.connect(node); node.connect(mute); mute.connect(context.destination);
    if (fileAudio) analyser.connect(context.destination);
    return {
      rate:context.sampleRate, analyser,
      get totalSamples() { return recording.total; },
      start() {
        if (closed || started) return;
        started = true; source.connect(analyser);
        if (fileAudio) source.start();
      },
      stop({discard = false} = {}) {
        if (stopPromise) return stopPromise;
        stopping = true;
        stopPromise = (async()=>{
          media?.getTracks().forEach(t=>t.stop());
          if (started) { source.disconnect(); if (fileAudio) { try { source.stop(); } catch {} } }
          if (!discard && !closed && started) await new Promise(resolve=>{
            const deadline = setTimeout(resolve,250);
            flushed = ()=>{ clearTimeout(deadline); resolve(); }; node.port.postMessage('flush');
          });
          const result = {blob:discard ? null : recording.blob(),duration:recording.total/context.sampleRate};
          close(); recording.clear(); return result;
        })();
        return stopPromise;
      }
    };
  } catch (error) { close(); throw error; }
}
