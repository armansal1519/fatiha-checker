// Pure scheduling policy: one fresh snapshot at a time, with no audio queue.
export const LIVE_OPTIONS = Object.freeze({intervalMs:800, pauseMs:500, windowSeconds:6, contextSeconds:10});

export class LiveCadence {
  constructor() { this.reset(0); }
  reset(now) { this.lastSent = now; this.lastFinished = -Infinity; }
  sent(now) { this.lastSent = now; }
  finished(now) { this.lastFinished = now; }
  due(now, {pending, voiceSample, sentVoice, lastVoice, pauseSent}) {
    if (pending || !voiceSample || now-this.lastFinished < 80) return null;
    const quiet = now-lastVoice >= LIVE_OPTIONS.pauseMs;
    if (quiet && !pauseSent && now-this.lastSent >= 350) return true;
    if (!quiet && voiceSample > sentVoice && now-this.lastSent >= LIVE_OPTIONS.intervalMs) return false;
    return null;
  }
}
