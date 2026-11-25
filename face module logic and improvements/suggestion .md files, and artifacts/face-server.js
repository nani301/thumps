// face-server.js — Node >=18. npm i express @tensorflow/tfjs-node @vladmandic/face-api canvas onnxruntime-node
// The server NEVER trusts client-reported scores. It re-runs detection, identity, anti-spoof, head-turn and the
// screen-flash reflection test on the uploaded frames, bound to a single-use server-random challenge.
import '@tensorflow/tfjs-node';
import crypto from 'node:crypto';
import express from 'express';
import * as faceapi from '@vladmandic/face-api';
import canvasLib from 'canvas';
import * as ort from 'onnxruntime-node';
import { CFG, makeChallenge, yawRatio, yawPose, shiftedCrop, softmax, euclid, cheekCenters, bgPatches, flashScore } from './liveness-core.js';

const TTL_MS = 60_000;
const MIN_MS = CFG.flash.slots * CFG.flash.slotMs + 3 * 900;   // a human can't finish faster than flash + 3 steps
const MAX_FAILS = 5, LOCK_MS = 15 * 60_000;
const DEBUG = process.env.FACE_DEBUG === '1';                  // never expose reasons to clients in production

const MAX_PARALLEL = Number(process.env.FACE_MAX_PARALLEL || 2);   // analyses running at once; the rest queue
let active = 0; const waiting = [];
const limited = async (fn) => { if (active >= MAX_PARALLEL) await new Promise((r) => waiting.push(r)); active++; try { return await fn(); } finally { active--; waiting.shift()?.(); } };

let DB_KEY, spoofSessions = [];
const challenges = new Map();   // -> Redis (SET key val NX EX 60) in production
const templates = new Map();    // -> database; stores ONLY the encrypted record
const fails = new Map();

// ---------- init ----------
export async function initFaceServer({ modelDir = './models', spoofModels = [
  { path: './models/2.7_80x80_MiniFASNetV2.onnx', scale: 2.7 }, { path: './models/4_0_0_80x80_MiniFASNetV1SE.onnx', scale: 4.0 }] } = {}) {
  DB_KEY = Buffer.from(process.env.FACE_DB_KEY_HEX || '', 'hex');
  if (DB_KEY.length !== 32) throw new Error('FACE_DB_KEY_HEX must be 64 hex chars (32 bytes) — use a KMS/secret manager');
  faceapi.env.monkeyPatch({ Canvas: canvasLib.Canvas, Image: canvasLib.Image, ImageData: canvasLib.ImageData });
  await Promise.all([faceapi.nets.ssdMobilenetv1, faceapi.nets.faceLandmark68Net, faceapi.nets.faceRecognitionNet].map((n) => n.loadFromDisk(modelDir)));
  spoofSessions = [];
  for (const m of spoofModels) spoofSessions.push({ sess: await ort.InferenceSession.create(m.path), scale: m.scale });
  if (!spoofSessions.length) throw new Error('Anti-spoof model required on the server');
  setInterval(() => { const n = Date.now(); for (const [k, v] of challenges) if (v.expiresAt < n) challenges.delete(k); }, 30_000).unref();
}

// ---------- descriptor encryption at rest (AES-256-GCM, AAD = user id) ----------
const encDesc = (uid, f32) => {
  const iv = crypto.randomBytes(12), c = crypto.createCipheriv('aes-256-gcm', DB_KEY, iv); c.setAAD(Buffer.from('face:' + uid));
  const ct = Buffer.concat([c.update(Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength)), c.final()]);
  return { iv: iv.toString('base64'), tag: c.getAuthTag().toString('base64'), ct: ct.toString('base64') };
};
const decDesc = (uid, r) => {
  const d = crypto.createDecipheriv('aes-256-gcm', DB_KEY, Buffer.from(r.iv, 'base64')); d.setAAD(Buffer.from('face:' + uid)); d.setAuthTag(Buffer.from(r.tag, 'base64'));
  const p = Buffer.concat([d.update(Buffer.from(r.ct, 'base64')), d.final()]);
  return new Float32Array(p.buffer.slice(p.byteOffset, p.byteOffset + p.byteLength)); // copy: Buffer pool offsets aren't 4-byte aligned
};

// ---------- frame analysis ----------
async function decode(b64) {
  if (typeof b64 !== 'string' || b64.length > 400_000) throw new Error('bad frame size');
  const img = await canvasLib.loadImage(Buffer.from(b64, 'base64'));
  if (img.width > 900 || img.height > 900 || img.width < 200 || img.height < 150) throw new Error('bad frame dims');
  const c = canvasLib.createCanvas(img.width, img.height); c.getContext('2d').drawImage(img, 0, 0); return c;
}
async function inspect(c, withDesc) {
  const chain = faceapi.detectAllFaces(c, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.6 })).withFaceLandmarks();
  const dets = withDesc ? await chain.withFaceDescriptors() : await chain;
  if (dets.length !== 1) return null;
  return { box: dets[0].detection.box, pts: dets[0].landmarks.positions, desc: dets[0].descriptor, c };
}
async function spoofReal(c, box) {
  const S = 80, tmp = canvasLib.createCanvas(S, S), x = tmp.getContext('2d'), vals = [];
  for (const { sess, scale } of spoofSessions) {
    const cr = shiftedCrop(c.width, c.height, box, scale); x.drawImage(c, cr.x, cr.y, cr.w, cr.h, 0, 0, S, S);
    const px = x.getImageData(0, 0, S, S).data, n = S * S, t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { t[i] = px[i * 4 + 2]; t[n + i] = px[i * 4 + 1]; t[2 * n + i] = px[i * 4]; } // BGR 0..255
    const out = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, S, S]) });
    vals.push(softmax(Array.from(out[sess.outputNames[0]].data))[1]);
  }
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}
// mean RGB over rects (skips clipped/very dark pixels). null if too few valid pixels.
function rectsRGB(c, rects, minPx = 20) {
  const ctx = c.getContext('2d'); let R = 0, G = 0, B = 0, n = 0;
  for (const q of rects) {
    const x0 = Math.max(0, Math.round(q.x)), y0 = Math.max(0, Math.round(q.y)), w = Math.min(c.width - x0, Math.round(q.w)), h = Math.min(c.height - y0, Math.round(q.h));
    if (w < 2 || h < 2) continue;
    const d = ctx.getImageData(x0, y0, w, h).data;
    for (let i = 0; i < d.length; i += 4) { const l = (d[i] + d[i + 1] + d[i + 2]) / 3; if (l < 25 || l > 235) continue; R += d[i]; G += d[i + 1]; B += d[i + 2]; n++; }
  }
  return n > minPx ? [R / n, G / n, B / n, n] : null;
}
const cheekRects = (pts) => cheekCenters(pts).map(({ x, y, r }) => ({ x: x - r, y: y - r, w: 2 * r, h: 2 * r }));
const avg = (a, b) => (a && b ? [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2] : a || b);
// landmarks only, on a crop around the (still) face from the BASE frame: no SSD per flash frame
async function cheeksOnly(c, box) {
  const m = 0.1, x = Math.max(0, box.x - box.width * m), y = Math.max(0, box.y - box.height * m);
  const cw = Math.round(Math.min(box.width * 1.2, c.width - x)), ch = Math.round(Math.min(box.height * 1.2, c.height - y));
  const t = canvasLib.createCanvas(cw, ch); t.getContext('2d').drawImage(c, x, y, cw, ch, 0, 0, cw, ch);
  const lm = await faceapi.nets.faceLandmark68Net.detectLandmarks(t);
  return lm.positions.map((p) => ({ x: p.x + x, y: p.y + y }));
}

async function analyze(frames, ch) {
  const why = [], M = {};
  if (!Array.isArray(frames) || frames.length > 40) return { ok: false, why: ['frames'] };
  const F = Object.fromEntries(frames.map((f) => [f?.tag, f]));
  const flashTags = ch.flash.slots.map((_, i) => 'f' + i), gTags = ch.flash.slots.map((_, i) => 'g' + i);
  const need = ['start', 'c1', 'c2', 'base', 'turnL', 'turnR', ...flashTags, ...gTags];
  if (frames.length !== need.length || need.some((t) => !F[t])) return { ok: false, why: ['missing/duplicate frames'] };

  // timing: order must follow the challenge, flash cadence must match the server's slot length
  const T = (t) => F[t].t, turnOrder = ch.steps.filter((s) => s.t !== 'BLINK').map((s) => (s.t === 'TURN_LEFT' ? 'turnL' : 'turnR'));
  const seq = ['start', ...turnOrder, 'base', ...flashTags];
  for (let i = 1; i < seq.length; i++) if (!(T(seq[i]) > T(seq[i - 1]))) why.push('order ' + seq[i]);
  for (let i = 1; i < flashTags.length; i++) { const d = T(flashTags[i]) - T(flashTags[i - 1]); if (d < ch.flash.slotMs * 0.6 || d > ch.flash.slotMs * 1.6) why.push('flash cadence'); }
  for (let i = 0; i < flashTags.length; i++) { const d = T(gTags[i]) - T(flashTags[i]); if (d < 40 || d > 250) why.push('g timing'); }
  if (why.length) return { ok: false, why };

  // analyse (sequential: bounded memory/CPU)
  const A = {};
  for (const tag of ['start', 'c1', 'c2', 'base', 'turnL', 'turnR']) {
    A[tag] = await inspect(await decode(F[tag].jpeg), ['start', 'c1', 'c2', 'base'].includes(tag));
    if (!A[tag]) return { ok: false, why: ['face not found exactly once: ' + tag] };
  }
  // pose
  const sy = CFG.serverYaw, pose = (t) => yawPose(yawRatio(A[t].pts), sy);
  if (pose('turnL') !== 'LEFT') why.push('turnL'); if (pose('turnR') !== 'RIGHT') why.push('turnR');
  for (const t of ['start', 'c1', 'c2', 'base']) if (pose(t) !== 'CENTER') why.push('not centered ' + t);

  // passive anti-spoof on frontal frames
  const sc = []; for (const t of ['start', 'c1', 'c2', 'base']) sc.push(await spoofReal(A[t].c, A[t].box));
  M.spoofMean = sc.reduce((a, b) => a + b, 0) / sc.length; M.spoofBad = sc.filter((s) => s < CFG.spoof.badCut).length / sc.length;
  if (sc.length < CFG.spoof.minSamples || M.spoofMean < CFG.spoof.minMean || M.spoofBad > CFG.spoof.maxBad) why.push('antispoof');

  // active illumination: skin chromaticity (minus background chromaticity, which cancels webcam AWB) must follow the random colors
  const W = A.base.c.width, H = A.base.c.height, bgR = bgPatches(W, H, A.base.box);
  const bgOf = (c) => (bgR.length ? rectsRGB(c, bgR, 200) : null);
  const baseFace = rectsRGB(A.base.c, cheekRects(A.base.pts)), baseBg = bgOf(A.base.c), slotsObs = [];
  for (let k = 0; k < flashTags.length; k++) {
    const per = [];
    for (const tag of [flashTags[k], gTags[k]]) {
      const c = await decode(F[tag].jpeg), pts = await cheeksOnly(c, A.base.box);
      per.push({ face: rectsRGB(c, cheekRects(pts)), bg: bgOf(c) });
    }
    const face = avg(per[0].face, per[1].face), bg = avg(per[0].bg, per[1].bg);
    if (!face) { why.push('flash roi'); break; }
    slotsObs.push({ screen: ch.flash.slots[k], face, bg });
  }
  if (!baseFace) why.push('flash roi');
  if (!why.includes('flash roi')) {
    const fs = flashScore({ face: baseFace, bg: baseBg }, slotsObs);
    M.flashCorr = fs.corr; M.flashGain = fs.gain; M.flashTier = fs.tier; M.flashUsedBg = fs.usedBg;
    if (fs.tier !== 'pass') why.push(fs.tier === 'weak' ? 'flash weak' : 'flash reflection');
  }

  // identity continuity across the whole session
  const D = ['start', 'c1', 'c2', 'base'].map((t) => A[t].desc); let maxD = 0;
  for (let i = 0; i < D.length; i++) for (let j = i + 1; j < D.length; j++) maxD = Math.max(maxD, euclid(D[i], D[j]));
  M.continuity = maxD; if (maxD > CFG.identity.continuity) why.push('identity drift');
  const mean = new Float32Array(128); for (let i = 0; i < 128; i++) mean[i] = D.reduce((s, d) => s + d[i], 0) / D.length;
  return { ok: why.length === 0, why, M, mean };
}

// ---------- router ----------
export function createFaceRouter({ requireAuth, issueSession = (uid) => ({ sessionToken: crypto.randomBytes(32).toString('hex') }) }) {
  const r = express.Router(); r.use(express.json({ limit: '8mb' }));
  const locked = (k) => { const f = fails.get(k); return f && f.n >= MAX_FAILS && Date.now() < f.until; };
  const bad = (k) => { const f = fails.get(k) || { n: 0 }; f.n++; f.until = Date.now() + LOCK_MS; fails.set(k, f); };
  const deny = (res, why, code = 401) => res.status(code).json({ ok: false, error: 'Verification failed', ...(DEBUG ? { why } : {}) });

  const issue = (uid, purpose, res) => {
    const id = crypto.randomUUID(), ch = makeChallenge((n) => crypto.randomInt(n)), now = Date.now();
    challenges.set(id, { uid, purpose, ch, issuedAt: now, expiresAt: now + TTL_MS });
    res.json({ id, steps: ch.steps, flash: ch.flash, expiresAt: now + TTL_MS });
  };
  r.post('/api/face/challenge', (req, res) => {
    const purpose = req.body?.purpose === 'enroll' ? 'enroll' : 'verify';
    if (purpose === 'enroll') return requireAuth(req, res, () => issue(String(req.user.id), 'enroll', res));
    const uid = String(req.body?.userId || ''); if (!uid || locked(uid)) return deny(res, ['locked'], 429);
    issue(uid, 'verify', res);   // always issued, even for unknown users (no enrollment oracle)
  });

  const consume = (id, uid, purpose) => {            // single-use, bound to user + purpose + time window
    const c = challenges.get(id); challenges.delete(id); const now = Date.now();
    if (!c || c.uid !== uid || c.purpose !== purpose || now > c.expiresAt || now - c.issuedAt < MIN_MS) return null; return c;
  };

  r.post('/api/face/verify', async (req, res) => {
    const uid = String(req.body?.userId || ''), ipk = 'ip:' + req.ip;
    if (locked(uid) || locked(ipk)) return deny(res, ['locked'], 429);
    try {
      const c = consume(req.body?.challengeId, uid, 'verify'); if (!c) { bad(ipk); return deny(res, ['challenge']); }
      const rec = templates.get(uid), a = await limited(() => analyze(req.body.frames, c.ch));
      let why = [...(a.why || [])], dist = null;
      if (!rec) why.push('not enrolled'); else if (a.mean) { dist = euclid(a.mean, decDesc(uid, rec)); if (dist > CFG.identity.match) why.push('face mismatch'); }
      if (req.body.signals?.virtualHint) a.M && (a.M.risk = ['virtual_cam_hint']);   // logged risk signal; flash test is the real guard
      // weak (not absent) screen reflection with everything else fine => ask for a second factor instead of failing daylight users
      if (why.length === 1 && why[0] === 'flash weak') { console.info('[face] step-up', uid, a.M); return res.status(202).json({ ok: false, stepUp: 'second_factor' }); }
      if (why.length) { bad(uid); bad(ipk); console.warn('[face] deny', uid, why, a.M); return deny(res, why); }
      fails.delete(uid); console.info('[face] ok', uid, a.M, { dist });
      return res.json({ ok: true, ...issueSession(uid, req, res) });
    } catch (e) { console.error('[face] error', e.message); return deny(res, ['error'], 400); }
  });

  // enrollment: ONLY for an already-authenticated user (otherwise an attacker enrolls their own face on a victim)
  r.post('/api/face/enroll', requireAuth, async (req, res) => {
    const uid = String(req.user.id);
    try {
      const c = consume(req.body?.challengeId, uid, 'enroll'); if (!c) return deny(res, ['challenge']);
      const a = await limited(() => analyze(req.body.frames, c.ch)); if (!a.ok) return deny(res, a.why);
      templates.set(uid, encDesc(uid, a.mean)); return res.json({ ok: true });
    } catch (e) { console.error('[face] enroll error', e.message); return deny(res, ['error'], 400); }
  });
  return r;
}
