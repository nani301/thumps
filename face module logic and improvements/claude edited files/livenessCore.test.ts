import assert from 'node:assert/strict';
import * as C from './services/livenessCore';

let n = 0; const ok = (s: string) => { n++; console.log('PASS', s); };

// ---------- CSPRNG / sequence ----------
{
  const seen = new Set<string>();
  for (let i = 0; i < 4000; i++) seen.add(C.pickDistinct(['a', 'b', 'c'] as const, 2).join(''));
  assert.equal(seen.size, 6); ok('2-of-3 ordered sequences: all 6 reachable');
  const cnt = [0, 0, 0]; for (let i = 0; i < 30000; i++) cnt[C.secureRandomInt(3)]++;
  assert.ok(Math.max(...cnt) - Math.min(...cnt) < 1000); ok('secureRandomInt unbiased');
}

// ---------- blink tracker ----------
function run(seq: number[], dt: number, lowFps: boolean) {
  const b = new C.BlinkTracker(); let t = 1000, n = 0;
  for (const e of seq) { t += dt; if (b.update(e, 0.01, t, { earPeak: 0.3, lowFps })) n++; }
  return n;
}
const open = (k: number) => Array(k).fill(0.3);
assert.equal(run([...open(10), 0.12, 0.1, 0.14, ...open(10)], 33, false), 1); ok('real blink @30fps');
assert.equal(run([...open(10), 0.14, ...open(10)], 66, true), 1); ok('real blink @15fps, 1 deep frame');
assert.equal(run([...open(10), 0.226, ...open(10)], 66, true), 0); ok('shallow 1-frame landmark dip rejected (photo jitter)');
assert.equal(run([...open(10), ...Array(30).fill(0.1), ...open(10)], 33, false), 0); ok('long eye-close rejected');
assert.equal(run([...open(10), ...Array(30).fill(0.1), 0.1, ...open(5), 0.1, 0.1, 0.3, ...open(5)], 33, false) <= 1, true); ok('no instant re-arm after long close');
// the original bug: held >720ms then reopen must NOT count
assert.equal(run([...open(5), ...Array(26).fill(0.1), ...open(8)], 33, false), 0); ok('held-closed 860ms then open: not a blink');

// ---------- identity continuity ----------
{
  const rnd = () => Float32Array.from({ length: 128 }, () => Math.random() - 0.5);
  const a = C.l2(rnd()), b = C.l2(rnd());
  const a2 = C.l2(a.map((v, i) => v + 0.01 * (i % 3)));
  assert.equal(C.identityContinuity(a, a2).ok, true);
  assert.equal(C.identityContinuity(a, b).ok, false); ok('continuity: same person ok, different person rejected');
  const m = C.meanNormalized(a, a2); assert.ok(Math.abs(Math.hypot(...m) - 1) < 1e-4); ok('meanNormalized is unit length');
}

// ---------- flash score (statistical) ----------
const gauss = () => Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random());
const rgb = (r: number, b: number): C.RGB => [r * 450, (1 - r - b) * 450, b * 450];
const B = { r: 0.36, b: 0.28 };
function trial(slotsN: number, mode: 'live' | 'replay' | 'global' | 'faceonly', o: { gain?: number; noise?: number; bgNoise?: number; awb?: number; drift?: number } = {}) {
  const { gain = 0.02, noise = 0.004, bgNoise = 0.0012, awb = 0.8, drift = 0.006 } = o;
  const plan = C.makeFlashPlan(slotsN), obs: C.FlashObservation[] = [];
  for (const s of plan) {
    const s3 = s[0] + s[1] + s[2]; const er = s[0] / s3 - 1 / 3, eb = s[2] / s3 - 1 / 3;
    let fr = 0, fb = 0, wr = 0, wb = 0;
    if (mode === 'live' || mode === 'faceonly') { fr = gain * er; fb = gain * eb; if (mode === 'live') { wr = 0.15 * fr; wb = 0.15 * fb; } }
    if (mode === 'global') { fr = wr = gain * er; fb = wb = gain * eb; }
    const ar = -awb * (0.5 * fr + 0.5 * wr) + drift * gauss(), ab = -awb * (0.5 * fb + 0.5 * wb) + drift * gauss();
    obs.push({ screen: s, face: rgb(B.r + fr + ar + noise * gauss(), B.b + fb + ab + noise * gauss()), bg: rgb(B.r + wr + ar + bgNoise * gauss(), B.b + wb + ab + bgNoise * gauss()) });
  }
  return C.flashScore({ face: rgb(B.r, B.b), bg: rgb(B.r, B.b) }, obs).tier;
}
const rate = (slots: number, mode: any, o: any = {}, N = 15000) => { const t = { pass: 0, weak: 0, fail: 0 } as Record<string, number>; for (let i = 0; i < N; i++) t[trial(slots, mode, o)]++; return t; };
const pct = (t: Record<string, number>, k: string) => (t[k] / 150).toFixed(2) + '%';
// averaged reads per slot (~5 frames @ 15fps in a 340ms slot): noise/sqrt(5)
const avg = { noise: 0.004 / Math.sqrt(5), bgNoise: 0.0012 / Math.sqrt(5) };
console.log('\nflash tiers, strong AWB, 5 reads averaged per slot:');
for (const slots of [6, 8, 10]) {
  const live = rate(slots, 'live', avg), liveWeak = rate(slots, 'live', { ...avg, gain: 0.01 }), rep = rate(slots, 'replay', avg), glob = rate(slots, 'global', avg);
  console.log(`  ${slots} slots | live .02: pass ${pct(live, 'pass')} | live .01: pass ${pct(liveWeak, 'pass')} weak ${pct(liveWeak, 'weak')} | replay: AUTO-ACCEPT ${pct(rep, 'pass')} (review ${pct(rep, 'weak')}) | frame-wide tint attack auto-accept ${pct(glob, 'pass')}`);
}
const r8 = rate(8, 'replay', avg, 30000); assert.ok(r8.pass / 30000 < 0.006); ok('8 slots: replay auto-accept < 0.6%');
const l8 = rate(8, 'live', avg); assert.ok(l8.pass / 15000 > 0.9); ok('8 slots: live face auto-accept > 90% (assumed gain .02)');
const g8 = rate(8, 'global', avg); assert.ok(g8.pass / 15000 < 0.01); ok('frame-wide tint attack rejected');

// ---------- guards ----------
{
  const now = new Date('2026-11-20T09:30:00Z');
  const recs = [
    { attendanceId: '23210-ec-001', timestamp: '2026-11-20T09:10:00Z', status: 'present' },
    { attendanceId: '23210-ec-002', timestamp: '2026-11-20T09:20:00Z', status: 'flagged' },
    { attendanceId: '23210-ec-003', timestamp: '2026-11-20T04:00:00Z', status: 'present' },
  ];
  assert.ok(C.findPriorCheckIn(recs, '23210-EC-001', now, 120)); ok('prior check-in found (case-insensitive)');
  assert.equal(C.findPriorCheckIn(recs, '23210-ec-002', now, 120), null); ok('flagged record does not block a real check-in');
  assert.equal(C.findPriorCheckIn(recs, '23210-ec-003', now, 120), null); ok('old record outside cooldown ignored');
  const v = Array.from({ length: 128 }, (_, i) => Math.sin(i)), w = Array.from({ length: 128 }, (_, i) => Math.cos(i * 1.7));
  const roster = [{ id: '1', attendanceId: 'A', fullName: 'Asha', faceVectors: [v] }, { id: '2', attendanceId: 'B', fullName: 'Bala', faceVectors: [w] }];
  assert.equal(C.findDuplicateFace([v.map((x) => x + 0.001)], roster, 0.45)?.attendanceId, 'A'); ok('duplicate enrollment detected');
  assert.equal(C.findDuplicateFace([Array.from({ length: 128 }, (_, i) => Math.sin(i * 3.3))], roster, 0.45), null); ok('new person not flagged duplicate');
  assert.equal(C.isEnrollableFrame({ rollDeg: 3, noseCheekRatio: 1, ear: 0.3, earPeak: 0.3, sharpness: 30 }), true);
  assert.equal(C.isEnrollableFrame({ rollDeg: 3, noseCheekRatio: 1.5, ear: 0.3, earPeak: 0.3, sharpness: 30 }), false); ok('enrollable-frame gate rejects turned head');
  const bg = C.backgroundRects(640, 480, { x: 200, y: 100, w: 240, h: 300 });
  assert.ok(bg.length >= 2); ok('background patches avoid face');
}
console.log(`\n${n} checks passed`);
