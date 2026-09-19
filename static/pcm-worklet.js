class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super(); this.blockSize = Math.max(128, Math.round(sampleRate * .128 / 128) * 128);
    this.buffer = new Float32Array(this.blockSize); this.offset = 0;
    this.port.onmessage = ({data}) => {
      if (data === "flush") {
        if (this.offset) {
          const tail = this.buffer.slice(0, this.offset);
          this.port.postMessage(tail.buffer, [tail.buffer]); this.offset = 0;
        }
        this.port.postMessage({flushed:true});
      }
    };
  }
  process(inputs) {
    const channels = inputs[0];
    if (channels?.length) for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i] / channels.length;
      this.buffer[this.offset++] = value;
      if (this.offset === this.buffer.length) {
        this.port.postMessage(this.buffer.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(this.blockSize); this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
