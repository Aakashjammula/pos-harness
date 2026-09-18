// PCM16 <-> Float32 conversion and a simple loudness meter, ported
// directly from the arithmetic in the original static/index.html.

export function floatToPcm16(input: Float32Array<ArrayBufferLike>): ArrayBuffer {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out.buffer;
}

export function pcm16ToFloat(buf: ArrayBuffer): Float32Array {
  const view = new Int16Array(buf);
  const out = new Float32Array(view.length);
  for (let i = 0; i < view.length; i++) out[i] = view[i] / 32768;
  return out;
}

export function rms(float32: Float32Array<ArrayBufferLike>): number {
  let sum = 0;
  for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
  return Math.sqrt(sum / float32.length);
}
