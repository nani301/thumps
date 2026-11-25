/**
 * livenessCore.ts — pure logic (no DOM, no TensorFlow). Unit-tested; the engine imports from here.
 * Everything that decides "is this a live person / the same person / already marked" lives in this file
 * so it can be tested without a camera.
 */

export interface Pt {
  x: number;
  y: number;
}
export type RGB = [number, number, number];

// ---------------------------------------------------------------------------
// Randomness (CSPRNG) + challenge sequence
// ---------------------------------------------------------------------------
export function secureRandomInt(maxExclusive: number): number {
  if (maxExclusive <= 1) return 0;
  const limit = Math.floor(0x100000000 / maxExclusive) * maxExclusive; // rejection sampling: no modulo bias
  const buf = new Uint32Array(1);
  do {
    crypto.getRandomValues(buf);
  } while (buf[0] >= limit);
  return buf[0] % maxExclusive;
}

/** Draws `count` DISTINCT items in random order (Fisher–Yates over a copy). */
export function pickDistinct<T>(pool: readonly T[], count: number, rand = secureRandomInt): T[] {
  const a = [...pool];
  for (let i = a.length - 1; i > 0; i--) {
    const j = rand(i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, Math.min(count, a.length));
}

// ---------------------------------------------------------------------------
// Blink tracker: raw EAR, hysteresis, depth check, and a LATCH so a long eye-closure can't re-arm itself
// ---------------------------------------------------------------------------
export class BlinkTracker {
  private closedAt = 0;
  private minEar = Infinity;
  private locked = false;

  reset(): void {
    this.closedAt = 0;
    this.minEar = Infinity;
    this.locked = false;
  }

  /** Returns true exactly once per completed, human-plausible blink. */
  update(ear: number, symmetryDiff: number, now: number, o: { earPeak: number; lowFps: boolean }): boolean {
    const closeF = o.lowFps ? 0.77 : 0.74;
    const openF = o.lowFps ? 0.88 : 0.87;
    const maxSym = o.lowFps ? 0.088 : 0.072;
    const deepF = o.lowFps ? 0.72 : 0.64; // a real blink goes deep; landmark jitter on a photo does not

    if (this.locked) {
      // eyes were held shut too long: must fully reopen before a new blink can start
      if (ear >= o.earPeak * openF) this.locked = false;
      return false;
    }
    if (this.closedAt === 0) {
      if (ear <= o.earPeak * closeF && symmetryDiff < maxSym) {
        this.closedAt = now;
        this.minEar = ear;
      }
      return false;
    }
    this.minEar = Math.min(this.minEar, ear);
    const dur = now - this.closedAt;
    if (dur > 720) {
      this.closedAt = 0;
      this.locked = true;
      return false;
    }
    if (ear >= o.earPeak * openF) {
      this.closedAt = 0;
      const minDur = o.lowFps ? 30 : 50;
      return dur >= minDur && dur <= 680 && this.minEar <= o.earPeak * deepF;
    }
    return false;
  }
}

// ---------------------------------------------------------------------------
// Identity continuity (defeats "bait-and-switch": photo for the ID frame, live face for the challenge)
// ---------------------------------------------------------------------------
export const CONTINUITY_MAX_DISTANCE = 0.4;

export function l2(vec: ArrayLike<number>): Float32Array {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  const inv = 1 / (Math.sqrt(s) + 1e-12);
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] * inv;
  return out;
}
export function dist128(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length || a.length === 0) return 1;
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}
export function meanNormalized(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const m = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) m[i] = (a[i] + b[i]) / 2;
  return l2(m);
}
export function identityContinuity(
  start: ArrayLike<number>,
  end: ArrayLike<number>,
  max = CONTINUITY_MAX_DISTANCE
): { ok: boolean; distance: number } {
  const d = dist128(start, end);
  return { ok: d <= max, distance: d };
}

// ---------------------------------------------------------------------------
// Active illumination ("screen flash") check
// The kiosk screen shows a random colour sequence; skin must change colour in sync (relative to the background,
// which cancels webcam auto-white-balance). Defeats pre-recorded video, injected virtual cameras and frame-wide
// colour-cast filters. It does NOT prove "live skin" (a printed photo reflects light too) — keep passive PAD + challenges.
// ---------------------------------------------------------------------------
export const FLASH_PALETTE: readonly RGB[] = [
  [255, 45, 45],
  [45, 255, 90],
  [45, 107, 255],
  [255, 208, 45],
  [45, 230, 255],
  [255, 45, 220],
];
export const FLASH_NEUTRAL: RGB = [128, 128, 128];

export function makeFlashPlan(slots: number, rand = secureRandomInt): RGB[] {
  const plan: RGB[] = [];
  let last = -1;
  while (plan.length < slots) {
    const k = rand(FLASH_PALETTE.length);
    if (k !== last) {
      plan.push([...FLASH_PALETTE[k]] as RGB);
      last = k;
    }
  }
  return plan;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
const mid = (a: Pt, b: Pt): Pt => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);

/** Cheek ROIs (beside the nose wings toward the jaw) from the 68 landmarks. */
export function cheekRects(p: Pt[]): Rect[] {
  const r = Math.max(3, dist(p[0], p[16]) * 0.07);
  return [mid(p[31], p[3]), mid(p[35], p[13])].map((c) => ({ x: c.x - r, y: c.y - r, w: 2 * r, h: 2 * r }));
}
/** Frame-corner patches that don't overlap the (expanded) face box. Fixed from the base frame, reused for all slots. */
export function backgroundRects(W: number, H: number, box: Rect): Rect[] {
  const w = Math.round(W * 0.14);
  const h = Math.round(H * 0.14);
  const ex = { x: box.x - box.w * 0.3, y: box.y - box.h * 0.3, w: box.w * 1.6, h: box.h * 1.6 };
  return ([[0, 0], [W - w, 0], [0, H - h], [W - w, H - h]] as const)
    .map(([x, y]) => ({ x, y, w, h }))
    .filter((r) => !(r.x < ex.x + ex.w && r.x + r.w > ex.x && r.y < ex.y + ex.h && r.y + r.h > ex.y));
}

const chroma = (c: ArrayLike<number>) => {
  const s = c[0] + c[1] + c[2] || 1;
  return { r: c[0] / s, g: c[1] / s, b: c[2] / s };
};
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
export function pearson(xs: number[], ys: number[]): number {
  const mx = mean(xs), my = mean(ys);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  return sxx * syy < 1e-18 ? 0 : sxy / Math.sqrt(sxx * syy);
}
function slope(xs: number[], ys: number[]): number {
  const mx = mean(xs), my = mean(ys);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
  }
  return sxx < 1e-12 ? 0 : sxy / sxx;
}

export const FLASH_THRESHOLDS = { minCorr: 0.65, minGain: 0.004, weakCorr: 0.45, weakGain: 0.002 };
export type FlashTier = 'pass' | 'weak' | 'fail';
export interface FlashObservation {
  screen: RGB;
  face: RGB;
  bg: RGB | null;
}
/**
 * observed = (face chroma shift) − (background chroma shift), both vs. the neutral base frame.
 * tier: pass = auto-accept; weak = send to operator review (do not auto-accept); fail = reject.
 */
export function flashScore(
  base: { face: RGB; bg: RGB | null },
  slots: FlashObservation[],
  t = FLASH_THRESHOLDS
): { corr: number; gain: number; tier: FlashTier; usedBackground: boolean } {
  const useBg = !!base.bg && slots.every((s) => s.bg);
  const f0 = chroma(base.face);
  const b0 = useBg ? chroma(base.bg as RGB) : null;
  const xs: number[] = [];
  const ys: number[] = [];
  for (const s of slots) {
    const ce = chroma(s.screen), cf = chroma(s.face), cb = useBg ? chroma(s.bg as RGB) : null;
    xs.push(ce.r - 1 / 3, ce.b - 1 / 3);
    ys.push(cf.r - f0.r - (cb && b0 ? cb.r - b0.r : 0), cf.b - f0.b - (cb && b0 ? cb.b - b0.b : 0));
  }
  const corr = pearson(xs, ys);
  const gain = slope(xs, ys);
  const tier: FlashTier =
    corr >= t.minCorr && gain >= t.minGain ? 'pass' : corr >= t.weakCorr && gain >= t.weakGain ? 'weak' : 'fail';
  return { corr, gain, tier, usedBackground: useBg };
}

// ---------------------------------------------------------------------------
// Attendance guards
// ---------------------------------------------------------------------------
export interface RecordLike {
  attendanceId: string;
  timestamp: string;
  status: string;
}
/** Latest accepted (present/late) record for this student inside the cooldown window, else null. */
export function findPriorCheckIn<T extends RecordLike>(
  records: readonly T[],
  attendanceId: string,
  now: Date,
  cooldownMinutes: number
): T | null {
  const key = attendanceId.toLowerCase();
  const windowMs = Math.max(0, cooldownMinutes) * 60_000;
  let best: T | null = null;
  for (const r of records) {
    if (r.attendanceId.toLowerCase() !== key) continue;
    if (r.status !== 'present' && r.status !== 'late') continue;
    const t = Date.parse(r.timestamp);
    if (Number.isNaN(t) || now.getTime() - t > windowMs || t > now.getTime() + 60_000) continue;
    if (!best || t > Date.parse(best.timestamp)) best = r;
  }
  return best;
}

export interface EnrolledLike {
  id: string;
  attendanceId: string;
  fullName: string;
  faceVectors?: number[][] | null;
  faceDescriptor?: number[] | null;
}
/** Enrollment de-duplication (the Aadhaar idea): is this face already enrolled under another ID? */
export function findDuplicateFace(
  newVectors: readonly ArrayLike<number>[],
  roster: readonly EnrolledLike[],
  threshold: number
): { attendanceId: string; fullName: string; distance: number } | null {
  const probes = newVectors.filter((v) => v.length === 128).map(l2);
  let best: { attendanceId: string; fullName: string; distance: number } | null = null;
  for (const s of roster) {
    const stored = (s.faceVectors?.length ? s.faceVectors : s.faceDescriptor ? [s.faceDescriptor] : []).filter(
      (v) => v.length === 128
    );
    for (const sv of stored) {
      const n = l2(sv);
      for (const p of probes) {
        const d = dist128(p, n);
        if (d <= threshold && (!best || d < best.distance)) best = { attendanceId: s.attendanceId, fullName: s.fullName, distance: d };
      }
    }
  }
  return best;
}

/** Is this frame fit to be ENROLLED (frontal, eyes open, sharp)? Used for burst frames after head-turn challenges. */
export function isEnrollableFrame(f: { rollDeg: number; noseCheekRatio: number; ear: number; earPeak: number; sharpness: number }): boolean {
  return (
    f.rollDeg <= 10 &&
    f.noseCheekRatio >= 0.8 &&
    f.noseCheekRatio <= 1.25 &&
    f.ear >= Math.max(0.2, f.earPeak * 0.85) &&
    f.sharpness >= 14
  );
}
