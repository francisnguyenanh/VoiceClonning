/**
 * recorder-worklet.js — AudioWorkletProcessor for capturing PCM samples.
 *
 * Runs in the audio render thread. Accumulates samples into a Float32Array
 * and posts them to the main thread every ~100ms.
 */
class PcmRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    // Post every ~100 ms at 16 kHz = 1600 samples
    this._threshold = 1600;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this._buffer.push(channel[i]);
    }

    if (this._buffer.length >= this._threshold) {
      this.port.postMessage(new Float32Array(this._buffer));
      this._buffer = [];
    }
    return true;
  }
}

registerProcessor("pcm-recorder", PcmRecorder);
