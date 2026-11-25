// liveness-core.js — pure logic shared by the browser client and the Node server (no DOM, no globals)

export const CFG = {
  blink: { calFrames: 14, minOpenEAR: 0.19, close: 0.70, open: 0.84, maxAsym: 0.085, maxLowerShift: 0.14, minMs: 40, maxMs: 450 },
  // client yaw (ratio = dist(nose,pt0)/dist(nose,pt16); <1 = user's RIGHT, >1 = user's LEFT)
  yaw: { alpha: 0.55, left: 1 / 0.6, right: 0.6, cMin: 0.78, cMax: 1.28, hold: 4, centerHold: 4 },
  // server only needs to see a clear turn in a still frame
  serverYaw: { left: 1.45, right: 0.69, cMin: 0.7, cMax: 1.43 },
  flash: { slots: 10, slotMs: 450, minCorr: 0.65, minGain: 0.004, weakCorr: 0.45, weakGain: 0.002 }, // weak => step-up (passkey/OTP), not hard fail
  spoof: { minMean: 0.85, maxBad: 0.25, badCut: 0.55, minSamples: 3, abortFrames: 4, abortMean: 0.3 },
  identity: { continuity: 0.45, match: 0.48 },
};

// ---------- geometry ----------
export const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
export const euclid = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); };
export const eyeEAR = (p, o) => { const h = dist(p[o], p[o + 3]); return h < 1e-4 ? 0 : (dist(p[o + 1], p[o + 5]) + dist(p[o + 2], p[o + 4])) / (2 * h); };
// stays ~constant during a real eyelid blink, moves on nods / talking / grimaces
export const lowerFace = (p) => (dist(p[30], p[8]) + dist(p[62], p[66])) / Math.max(dist(p[36], p[45]), 1);
export const yawRatio = (p) => Math.max(dist(p[30], p[0]), 1) / Math.max(dist(p[30], p[16]), 1);
export function yawPose(r, c = CFG.yaw) {
  if (r >= c.left) return 'LEFT';
  if (r <= c.right) return 'RIGHT';
  return r >= c.cMin && r <= c.cMax ? 'CENTER' : 'TRANSITION';
}
export function iou(a, b) {
  const w = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const i = w * h; return i ? i / (a.width * a.height + b.width * b.height - i) : 0;
}
// Silent-Face CropImage: scale box around its center, SHIFT (never squash) back inside the frame
export function shiftedCrop(W, H, box, scale) {
  const k = Math.min((H - 1) / box.height, (W - 1) / box.width, scale);
  const w = box.width * k, h = box.height * k, cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  let x1 = cx - w / 2, y1 = cy - h / 2, x2 = cx + w / 2, y2 = cy + h / 2;
  if (x1 < 0) { x2 -= x1; x1 = 0; } if (y1 < 0) { y2 -= y1; y1 = 0; }
  if (x2 > W - 1) { x1 -= x2 - (W - 1); x2 = W - 1; } if (y2 > H - 1) { y1 -= y2 - (H - 1); y2 = H - 1; }
  x1 = Math.max(0, x1); y1 = Math.max(0, y1);
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}
export function softmax(v) {
  const m = Math.max(...v), e = Array.from(v, (q) => Math.exp(q - m)), s = e.reduce((a, q) => a + q, 0);
  return e.map((q) => q / s);
}

// ---------- blink (raw EAR, hysteresis, depth, symmetry, rigid lower face) ----------
export class BlinkDetector {
  constructor(c = CFG.blink) { this.c = c; this.reset(); }
  reset() { this.cal = []; this.rig = []; this.base = null; this.rig0 = null; this.closed = false; this.locked = false; this.t0 = 0; this.min = Infinity; }
  update(p, now) {
    const c = this.c, l = eyeEAR(p, 36), r = eyeEAR(p, 42), e = (l + r) / 2, asym = Math.abs(l - r), rg = lowerFace(p);
    if (this.base === null) {
      if (e >= c.minOpenEAR && asym <= c.maxAsym) { this.cal.push(e); this.rig.push(rg); }
      if (this.cal.length >= c.calFrames) {
        const s = [...this.cal].sort((a, b) => a - b), k = Math.floor(s.length * 0.2), m = s.slice(k, s.length - k);
        this.base = m.reduce((a, b) => a + b, 0) / m.length;
        this.rig0 = this.rig.reduce((a, b) => a + b, 0) / this.rig.length;
      }
      return { ready: false, blinked: false, ear: e };
    }
    const rigid = Math.abs(rg - this.rig0) / this.rig0 <= c.maxLowerShift;
    let blinked = false;
    if (this.locked) { if (e >= this.base * c.open) this.locked = false; return { ready: true, blinked: false, ear: e }; } // must fully reopen before re-arming
    if (!this.closed) {
      if (e > this.base * 0.92) this.base = this.base * 0.98 + e * 0.02;
      if (e < this.base * c.close && asym <= c.maxAsym && rigid) { this.closed = true; this.t0 = now; this.min = e; }
    } else {
      this.min = Math.min(this.min, e);
      if (!rigid || asym > c.maxAsym * 1.4 || now - this.t0 > c.maxMs * 1.5) { this.closed = false; this.locked = true; }
      else if (e >= this.base * c.open) {
        this.closed = false;
        const d = now - this.t0;
        blinked = d >= c.minMs && d <= c.maxMs && this.min < this.base * 0.65;
      }
    }
    return { ready: true, blinked, ear: e };
  }
}
export class YawTracker {
  constructor(c = CFG.yaw) { this.c = c; this.r = null; }
  update(p) {
    const raw = yawRatio(p);
    this.r = this.r === null ? raw : this.c.alpha * raw + (1 - this.c.alpha) * this.r;
    return { ratio: this.r, pose: yawPose(this.r, this.c) };
  }
}

// ---------- active-illumination (screen flash) check ----------
export const FLASH_PALETTE = [[255, 45, 45], [45, 255, 90], [45, 107, 255], [255, 208, 45], [45, 230, 255], [255, 45, 220]];
export const NEUTRAL = [128, 128, 128];

export function makeChallenge(randInt) {
  const steps = [{ t: 'BLINK', n: 1 + randInt(2) }, { t: 'TURN_LEFT' }, { t: 'TURN_RIGHT' }];
  for (let i = steps.length - 1; i > 0; i--) { const j = randInt(i + 1); [steps[i], steps[j]] = [steps[j], steps[i]]; }
  const slots = []; let last = -1;
  while (slots.length < CFG.flash.slots) { const k = randInt(FLASH_PALETTE.length); if (k !== last) { slots.push([...FLASH_PALETTE[k]]); last = k; } }
  return { steps, flash: { slotMs: CFG.flash.slotMs, neutral: NEUTRAL, slots } };
}

export const chroma = (c) => { const s = (c[0] + c[1] + c[2]) || 1; return { r: c[0] / s, g: c[1] / s, b: c[2] / s }; };
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
export function pearson(xs, ys) {
  const mx = mean(xs), my = mean(ys); let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < xs.length; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
  return sxx * syy < 1e-18 ? 0 : sxy / Math.sqrt(sxx * syy);
}
export const slope = (xs, ys) => { const mx = mean(xs), my = mean(ys); let sxy = 0, sxx = 0; for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; } return sxx < 1e-12 ? 0 : sxy / sxx; };
// cheek ROIs from 68 landmarks (beside nose wings toward the jaw): returns [{x,y,r}]
export function cheekCenters(p) {
  const r = Math.max(3, dist(p[0], p[16]) * 0.07);
  return [mid(p[31], p[3]), mid(p[35], p[13])].map((c) => ({ ...c, r }));
}
// Background patches (frame corners not overlapping the expanded face box). Fixed from the BASE frame, reused for all slots.
export function bgPatches(W, H, box) {
  const w = Math.round(W * 0.14), h = Math.round(H * 0.14);
  const ex = { x: box.x - box.width * 0.3, y: box.y - box.height * 0.3, width: box.width * 1.6, height: box.height * 1.6 };
  return [[0, 0], [W - w, 0], [0, H - h], [W - w, H - h]].map(([x, y]) => ({ x, y, w, h }))
    .filter((r) => !(r.x < ex.x + ex.width && r.x + r.w > ex.x && r.y < ex.y + ex.height && r.y + r.h > ex.y));
}
// base: {face:[r,g,b], bg:[r,g,b]|null} under the neutral screen; slots: [{screen, face, bg}] under each random flash.
// Observed signal = (face chroma shift) - (background chroma shift): webcam AWB/auto-exposure moves both equally and
// cancels; screen light (strong at ~40cm, weak on the far wall) does not. Also cancels a frame-wide color-cast attack.
export function flashScore(base, slots, c = CFG.flash) {
  const useBg = !!base.bg && slots.every((s) => s.bg);
  const f0 = chroma(base.face), b0 = useBg ? chroma(base.bg) : null, xs = [], ys = [];
  for (const s of slots) {
    const ce = chroma(s.screen), cf = chroma(s.face), cb = useBg ? chroma(s.bg) : null;
    xs.push(ce.r - 1 / 3, ce.b - 1 / 3);
    ys.push((cf.r - f0.r) - (useBg ? cb.r - b0.r : 0), (cf.b - f0.b) - (useBg ? cb.b - b0.b : 0));
  }
  const corr = pearson(xs, ys), gain = slope(xs, ys);
  const tier = corr >= c.minCorr && gain >= c.minGain ? 'pass' : corr >= c.weakCorr && gain >= c.weakGain ? 'weak' : 'fail';
  return { corr, gain, tier, usedBg: useBg, pass: tier === 'pass' };
}
