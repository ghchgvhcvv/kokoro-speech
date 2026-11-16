// Kokoro-82M speech job (runs on a GitHub-hosted Ubuntu runner, CPU only).
//   node generate.mjs <request.json> <outDir>
// Reads a request, validates it, loads the model ONCE, speaks every line with the voice the app assigned to that character, and writes
// one 24 kHz mono 16-bit WAV per line plus manifest.json. Nothing is uploaded from here: the workflow packs outDir as a short-lived artifact.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const MODEL = process.env.KOKORO_MODEL || 'onnx-community/Kokoro-82M-v1.0-ONNX';
const DTYPE = process.env.KOKORO_DTYPE || 'q8';          // 8-bit quantised ONNX: ~92 MB, fastest on CPU
const SR = 24000, MAX_LINES = 200, MAX_TEXT = 700, MAX_TOTAL = 12000, MAX_SEG = 230;
const clamp = (x, a, b, d) => (Number.isFinite(+x) ? Math.min(b, Math.max(a, +x)) : d);
const fail = (code, msg) => { const e = new Error(msg); e.code = code; throw e; };

export function validate(req, voices) {
  if (!req || typeof req !== 'object') fail('bad_request', 'request is not an object');
  if (!/^[A-Za-z0-9_-]{8,64}$/.test(String(req.job || ''))) fail('bad_job', 'job id must be 8-64 letters, digits, - or _');
  if (!Array.isArray(req.lines) || !req.lines.length) fail('no_lines', 'no lines to speak');
  if (req.lines.length > MAX_LINES) fail('too_many_lines', `at most ${MAX_LINES} lines`);
  const seen = new Set(); let total = 0; const out = [];
  for (const l of req.lines) {
    const id = String(l?.id ?? ''); if (!/^[A-Za-z0-9_-]{1,40}$/.test(id) || seen.has(id)) fail('bad_line_id', `bad or duplicate line id "${id}"`); seen.add(id);
    const text = String(l?.text ?? '').replace(/\s+/g, ' ').trim(); if (!text) fail('empty_text', `line ${id} is empty`);
    if (text.length > MAX_TEXT) fail('text_too_long', `line ${id} is longer than ${MAX_TEXT} characters`);
    total += text.length; if (total > MAX_TOTAL) fail('too_much_text', `more than ${MAX_TOTAL} characters in total`);
    const voice = String(l?.voice ?? ''); if (!voices.includes(voice)) fail('bad_voice', `unknown Kokoro voice "${voice}" for line ${id}`);
    out.push({ id, text, voice, emotion: String(l?.emotion || 'neutral').slice(0, 24), speed: clamp(l?.speed, 0.6, 1.4, 1), gain: clamp(l?.gain, 0.3, 1.2, 1), pause: clamp(l?.pause, 0, 1.2, 0.2), lead: clamp(l?.lead, 0, 1.5, 0) });
  }
  return { job: req.job, lines: out };
}

/** sentence-based segments (kept in order, punctuation kept, never cut inside a word); a long sentence is split at commas, then at a space */
export function segment(text) {
  const sentences = text.match(/[^.!?…]+(?:[.!?…]+["')\]]*|$)/g)?.map(s => s.trim()).filter(Boolean) || [text];
  const out = [];
  for (let s of sentences) {
    while (s.length > MAX_SEG) {
      let cut = Math.max(s.lastIndexOf(', ', MAX_SEG), s.lastIndexOf('; ', MAX_SEG), s.lastIndexOf(': ', MAX_SEG));
      if (cut < 60) cut = s.lastIndexOf(' ', MAX_SEG); if (cut < 1) cut = MAX_SEG;
      out.push(s.slice(0, cut + 1).trim()); s = s.slice(cut + 1).trim();
    }
    if (s) out.push(s);
  }
  return out;
}

export function wavBytes(samples, sr = SR) {
  const n = samples.length, buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write('WAVEfmt ', 8); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sr, 24); buf.writeUInt32LE(sr * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36); buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767))), 44 + i * 2);
  return buf;
}

async function main() {
  const [reqPath, outDir] = process.argv.slice(2); if (!reqPath || !outDir) fail('usage', 'usage: generate.mjs <request.json> <outDir>');
  const T = { started: Date.now() }; const mark = (k, t0) => { T[k] = Math.round(performance.now() - t0); };
  const req = JSON.parse(fs.readFileSync(reqPath, 'utf8'));
  let tts, tr;
  const t0 = performance.now();
  try {
    const { KokoroTTS } = await import('kokoro-js'); tr = await import('@huggingface/transformers');
    if (process.env.KOKORO_CACHE) tr.env.cacheDir = process.env.KOKORO_CACHE;
    tts = await KokoroTTS.from_pretrained(MODEL, { dtype: DTYPE, device: 'cpu' });
  } catch (e) { fail('model_init', `could not load ${MODEL} (${DTYPE}): ${e.message}`); }
  mark('loadMs', t0);
  const job = validate(req, Object.keys(tts.voices));
  fs.mkdirSync(outDir, { recursive: true });
  const manifest = { job: job.job, model: MODEL, dtype: DTYPE, sampleRate: SR, format: 'wav-pcm16-mono', lines: [], timings: T };
  const tg = performance.now(); let audioSec = 0;
  for (const l of job.lines) {
    const parts = [new Float32Array(Math.round(l.lead * SR))];
    for (const seg of segment(l.text)) {
      let a; try { a = await tts.generate(seg, { voice: l.voice, speed: l.speed }); } catch (e) { fail('inference', `line ${l.id}: ${e.message}`); }
      if (!a?.audio?.length) fail('empty_audio', `line ${l.id}: the model returned no audio`);
      parts.push(a.audio, new Float32Array(Math.round(l.pause * SR)));
    }
    parts.pop(); // no silence after the last segment
    const n = parts.reduce((s, p) => s + p.length, 0), all = new Float32Array(n); let o = 0; for (const p of parts) { all.set(p, o); o += p.length; }
    for (let i = 0; i < n; i++) all[i] *= l.gain;
    const bytes = wavBytes(all), file = `${l.id}.wav`; fs.writeFileSync(path.join(outDir, file), bytes);
    audioSec += n / SR;
    manifest.lines.push({ id: l.id, file, duration: +(n / SR).toFixed(3), voice: l.voice, speed: l.speed, emotion: l.emotion, text: l.text, sha256: crypto.createHash('sha256').update(bytes).digest('hex') });
  }
  mark('generateMs', tg); T.audioSeconds = +audioSec.toFixed(2); T.realtimeFactor = +(T.generateMs / 1000 / Math.max(0.01, audioSec)).toFixed(2);
  T.peakRssMB = Math.round(process.memoryUsage().rss / 1e6); T.finished = Date.now();
  fs.writeFileSync(path.join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 1));
  console.log(JSON.stringify({ ok: true, job: job.job, lines: manifest.lines.length, ...T }));
}
if (process.argv[1] && import.meta.url === new URL(`file://${path.resolve(process.argv[1])}`).href) {
  main().catch(e => { try { fs.mkdirSync(process.argv[3], { recursive: true }); fs.writeFileSync(path.join(process.argv[3], 'error.json'), JSON.stringify({ code: e.code || 'error', message: String(e.message).slice(0, 400) })); } catch {} console.error('KOKORO_ERROR', e.code || 'error', e.message); process.exit(1); });
}
