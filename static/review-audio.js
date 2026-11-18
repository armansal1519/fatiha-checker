// Bounds use the clock of this session's audio. Omissions have no sound of
// their own; replay the two observed words bracketing the inferred gap.
export function segmentBounds(words, duration) {
  if (!(duration>0) || !words.length || words.some(word=>!word ||
      !Number.isFinite(word.start_ms) || !Number.isFinite(word.end_ms) ||
      word.start_ms<0 || word.end_ms<word.start_ms)) return null;
  const start = Math.max(0,Math.min(...words.map(w=>w.start_ms))/1000-.3);
  const end = Math.min(duration,Math.max(...words.map(w=>w.end_ms))/1000+.35);
  return end>start ? {start,end} : null;
}

export async function unreviewedRecording(blob, reviewedThrough) {
  // SessionRecording creates a mono PCM16 WAV with a 44-byte header. Slice
  // the existing Blob without copying its PCM; the decoder resamples to 16k.
  const header=await blob.slice(0,44).arrayBuffer();
  if (header.byteLength!==44 || !Number.isFinite(reviewedThrough)) throw new Error('صدای نوبت معتبر نیست.');
  const wav=new DataView(header), rate=wav.getUint32(24,true);
  if (!rate || wav.getUint16(22,true)!==1 || wav.getUint16(34,true)!==16) throw new Error('قالب ضبط معتبر نیست.');
  const first=44+2*Math.floor(Math.max(0,reviewedThrough-.75)*rate);
  const data=blob.slice(first);
  if (!data.size) throw new Error('صدای بررسی‌نشده باقی نمانده است.');
  wav.setUint32(4,36+data.size,true); wav.setUint32(40,data.size,true);
  return new Blob([header,data],{type:'audio/wav'});
}
