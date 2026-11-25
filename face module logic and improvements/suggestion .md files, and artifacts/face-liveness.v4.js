// face-liveness.v4.js — browser client. Globals: faceapi, ort. Shared logic: ./liveness-core.js
// The browser does UX + early rejection only. The SERVER makes the real decision from the frames this uploads.
import { CFG, dist, iou, shiftedCrop, softmax, makeChallenge, BlinkDetector, YawTracker } from './liveness-core.js';

export const CLIENT = {
  modelUrl: './models', redetectEvery: 6, iouMin: 0.3, stepMs: 7000, maxMissing: 15,
  // anti-spoof models (early-abort on client; server re-scores uploaded frames). Add/remove entries freely.
  spoof: { size: 80, realIndex: 1, bgr: true, norm: 'raw', models: [
    { url: './models/2.7_80x80_MiniFASNetV2.onnx', scale: 2.7 },
    { url: './models/4_0_0_80x80_MiniFASNetV1SE.onnx', scale: 4.0 } ] },
  jpegMaxW: 640, jpegQ: 0.8,
};

let tiny = false;
export async function loadModels() {
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri(CLIENT.modelUrl),
    faceapi.nets.faceLandmark68Net.loadFromUri(CLIENT.modelUrl),
  ]); // no recognition net needed on the client any more: identity is matched on the server
  try { await faceapi.nets.tinyFaceDetector.loadFromUri(CLIENT.modelUrl); tiny = true; } catch { tiny = false; }
}
const detOpts = () => tiny ? new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 }) : new faceapi.SsdMobilenetv1Options({ minConfidence: 0.65 });

export async function startCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user', frameRate: 30 }, audio: false });
  video.srcObject = stream; await video.play();
  return stream;
}
// best-effort: stop auto white-balance/exposure from cancelling the screen-flash signal
async function lockCamera(stream) {
  try {
    const t = stream.getVideoTracks()[0], cap = t.getCapabilities?.() || {}, adv = {};
    if (cap.whiteBalanceMode?.includes('manual')) adv.whiteBalanceMode = 'manual';
    if (cap.exposureMode?.includes('manual')) adv.exposureMode = 'manual';
    if (Object.keys(adv).length) await t.applyConstraints({ advanced: [adv] });
    return Object.keys(adv);
  } catch { return []; }
}
// SIGNALS ONLY (labels are trivially renamed and legit phone-as-webcam apps match too) — the server weighs them as risk
export function deviceSignals(stream) {
  const t = stream.getVideoTracks()[0], s = t.getSettings?.() || {}, label = (t.label || '').toLowerCase();
  return { label: t.label, virtualHint: /obs|virtual|manycam|snap camera|splitcam|xsplit|v4l2loopback/.test(label), fps: s.frameRate || null, w: s.width || null, h: s.height || null };
}

// ---------- fast tracker: SSD every N frames, landmark-net on a crop otherwise ----------
class FaceTracker {
  constructor(v) { this.v = v; this.box = null; this.n = 0; this.c = document.createElement('canvas'); this.x = this.c.getContext('2d', { willReadFrequently: true }); }
  async next() {
    const v = this.v, fresh = !this.box || (this.n++ % CLIENT.redetectEvery === 0);
    let ssdBox = null;
    if (fresh) {
      const d = await faceapi.detectAllFaces(v, detOpts());
      if (d.length > 1) return { status: 'multi' };
      if (!d.length) { this.box = null; return { status: 'none' }; }
      this.box = d[0].box; ssdBox = d[0].box;
    }
    const b = this.box, m = 0.1, x = Math.max(0, b.x - b.width * m), y = Math.max(0, b.y - b.height * m);
    const cw = Math.round(Math.min(b.width * 1.2, v.videoWidth - x)), ch = Math.round(Math.min(b.height * 1.2, v.videoHeight - y));
    if (cw < 40 || ch < 40) { this.box = null; return { status: 'none' }; }
    this.c.width = cw; this.c.height = ch; this.x.drawImage(v, x, y, cw, ch, 0, 0, cw, ch);
    const lm = await faceapi.nets.faceLandmark68Net.detectLandmarks(this.c);
    const pts = lm.positions.map((p) => ({ x: p.x + x, y: p.y + y }));
    const ew = dist(pts[36], pts[39]) / dist(pts[0], pts[16]);
    if (ew < 0.1 || ew > 0.35) { this.box = null; return { status: 'lost' }; }
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), lw = x1 - x0, lh = y1 - y0;
    this.box = new faceapi.Box({ x: x0 - lw * 0.05, y: y0 - lh * 0.15, width: lw * 1.1, height: lh * 1.2 });
    return { status: 'ok', pts, fresh, ssdBox };
  }
}

// ---------- multi-model anti-spoof (probabilities averaged, Silent-Face style fusion) ----------
const sc = document.createElement('canvas'), sx = sc.getContext('2d', { willReadFrequently: true });
export class AntiSpoof {
  constructor() { this.sessions = []; this.scores = []; }
  async load() {
    this.sessions = [];
    for (const m of CLIENT.spoof.models) {
      try { this.sessions.push({ s: await ort.InferenceSession.create(m.url, { executionProviders: ['wasm'] }), scale: m.scale }); } catch (e) { console.warn('spoof model skipped', m.url, e.message); }
    }
    if (!this.sessions.length) throw new Error('No anti-spoof model could be loaded');
  }
  reset() { this.scores = []; }
  async score(video, box) {
    const S = CLIENT.spoof, vals = [];
    for (const { s, scale } of this.sessions) {
      const c = shiftedCrop(video.videoWidth, video.videoHeight, box, scale);
      sc.width = sc.height = S.size; sx.drawImage(video, c.x, c.y, c.w, c.h, 0, 0, S.size, S.size);
      const px = sx.getImageData(0, 0, S.size, S.size).data, n = S.size * S.size, t = new Float32Array(3 * n);
      for (let i = 0; i < n; i++) { const r = px[i * 4], g = px[i * 4 + 1], b = px[i * 4 + 2], q = S.bgr ? [b, g, r] : [r, g, b]; t[i] = q[0]; t[n + i] = q[1]; t[2 * n + i] = q[2]; }
      const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, S.size, S.size]) });
      vals.push(softmax(Array.from(out[s.outputNames[0]].data))[S.realIndex]);
    }
    const v = vals.reduce((a, b) => a + b, 0) / vals.length; this.scores.push(v); return v;
  }
  get mean() { return this.scores.length ? this.scores.reduce((a, b) => a + b, 0) / this.scores.length : 0; }
  get abort() { const S = CFG.spoof; return this.scores.length >= S.abortFrames && this.mean < S.abortMean; }
}

// ---------- helpers ----------
const jc = document.createElement('canvas'), jx = jc.getContext('2d');
function snap(video, tag, extra = {}) {
  const k = Math.min(1, CLIENT.jpegMaxW / video.videoWidth);
  jc.width = Math.round(video.videoWidth * k); jc.height = Math.round(video.videoHeight * k); jx.drawImage(video, 0, 0, jc.width, jc.height);
  return { tag, t: Math.round(performance.now()), jpeg: jc.toDataURL('image/jpeg', CLIENT.jpegQ).split(',')[1], ...extra };
}
const nextFrame = (v) => new Promise((r) => ('requestVideoFrameCallback' in v ? v.requestVideoFrameCallback(() => r()) : requestAnimationFrame(() => r())));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const csprng = (n) => crypto.getRandomValues(new Uint32Array(1))[0] % n;
const fail = (reason, extra = {}) => ({ ok: false, reason, ...extra });
const label = (s) => s.t === 'BLINK' ? (s.n > 1 ? 'Blink twice' : 'Blink once') : s.t === 'TURN_LEFT' ? 'Turn your head to YOUR left' : 'Turn your head to YOUR right';
const css = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

// ---------- main: returns an evidence bundle for the server ----------
export async function runLiveness(video, stream, { challenge = null, antiSpoof, onPrompt = () => {}, setOverlay = () => {} } = {}) {
  const ch = challenge || { id: null, ...makeChallenge(csprng) };
  const tr = new FaceTracker(video), blink = new BlinkDetector(), yaw = new YawTracker(), Y = CFG.yaw;
  const frames = [], tags = { c: 0 }, locked = await lockCamera(stream);
  antiSpoof.reset();
  let prev = null, miss = 0;

  // A) calibrate: centered + open-eye baseline
  onPrompt('Look straight, hold still');
  let tStep = performance.now();
  for (;;) {
    if (performance.now() - tStep > 8000) return fail('Calibration timeout');
    await nextFrame(video);
    const f = await tr.next();
    if (f.status === 'multi') return fail('Multiple faces');
    if (f.status !== 'ok' || yaw.update(f.pts).pose !== 'CENTER') continue;
    if (blink.update(f.pts, performance.now()).ready) { if (f.ssdBox) { await antiSpoof.score(video, f.ssdBox); } break; }
  }
  frames.push(snap(video, 'start')); prev = tr.box;

  // B) random challenges with forced neutral pose after each step
  let i = 0, hold = 0, blinks = 0, neutral = false; tStep = performance.now(); onPrompt(label(ch.steps[0]));
  const blinkLog = [];
  for (;;) {
    if (performance.now() - tStep > CLIENT.stepMs) return fail('Timed out');
    await nextFrame(video);
    const f = await tr.next();
    if (f.status === 'multi') return fail('Multiple faces');
    if (f.status === 'none') { if (++miss > CLIENT.maxMissing) return fail('Face left view'); continue; }
    if (f.status !== 'ok') continue;
    miss = 0;
    if (prev && iou(prev, tr.box) < CLIENT.iouMin) return fail('Sudden movement or frame cut');
    prev = tr.box;
    const now = performance.now(), { pose } = yaw.update(f.pts);
    if (f.fresh && f.ssdBox && pose === 'CENTER') { await antiSpoof.score(video, f.ssdBox); if (antiSpoof.abort) return fail('Spoof detected'); }

    if (neutral) {
      hold = pose === 'CENTER' ? hold + 1 : 0;
      if (hold >= Y.centerHold) {
        neutral = false; hold = 0; tStep = now; blink.closed = false; blink.locked = false;
        if (tags.c < 2) frames.push(snap(video, 'c' + (++tags.c)));
        if (i >= ch.steps.length) break;
        onPrompt(label(ch.steps[i]));
      }
      continue;
    }
    const s = ch.steps[i]; let done = false;
    if (s.t === 'BLINK') {
      if (pose === 'CENTER') { const r = blink.update(f.pts, now); if (r.blinked && ++blinks >= s.n) { done = true; blinkLog.push(Math.round(now)); } }
      else { blink.closed = false; }
    } else {
      hold = pose === (s.t === 'TURN_LEFT' ? 'LEFT' : 'RIGHT') ? hold + 1 : 0;
      if (hold >= Y.hold) { done = true; frames.push(snap(video, s.t === 'TURN_LEFT' ? 'turnL' : 'turnR')); }
    }
    if (done) { i++; hold = 0; blinks = 0; neutral = true; tStep = now; onPrompt('Look straight'); }
  }

  // C) screen-flash challenge: server-chosen random colors; skin must react in sync
  onPrompt('Hold still — light check (turn screen brightness up)');
  setOverlay(css(ch.flash.neutral)); await sleep(350);
  frames.push(snap(video, 'base'));
  const flashT0 = performance.now();
  for (let k = 0; k < ch.flash.slots.length; k++) {
    setOverlay(css(ch.flash.slots[k]));
    await sleep(ch.flash.slotMs - 170);              // let sensor + light settle
    frames.push(snap(video, 'f' + k));
    await sleep(110);
    frames.push(snap(video, 'g' + k));               // 2nd frame per slot: averaged server-side (noise / 1.41)
    await sleep(60);
  }
  setOverlay(null);

  const sp = antiSpoof;
  if (sp.scores.length >= CFG.spoof.minSamples && sp.mean < CFG.spoof.minMean) return fail('Spoof detected (local)', { spoofMean: sp.mean });
  return {
    ok: true,
    evidence: { challengeId: ch.id, frames, signals: { ...deviceSignals(stream), lockedControls: locked, userAgent: navigator.userAgent },
      local: { spoofMean: sp.mean, spoofN: sp.scores.length, blinkTimes: blinkLog, flashMs: Math.round(performance.now() - flashT0) } },
  };
}

// ---------- server calls ----------
const post = async (url, body) => { const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include', body: JSON.stringify(body) }); return r.json(); };
export const getChallenge = (userId, purpose = 'verify') => post('/api/face/challenge', { userId, purpose });
export const submitVerify = (userId, evidence) => post('/api/face/verify', { userId, ...evidence });
export const submitEnroll = (evidence) => post('/api/face/enroll', evidence);

export async function faceLogin(video, stream, antiSpoof, userId, ui) {
  const challenge = await getChallenge(userId, 'verify');
  const r = await runLiveness(video, stream, { challenge, antiSpoof, ...ui });
  if (!r.ok) return r;
  return submitVerify(userId, r.evidence);
}
