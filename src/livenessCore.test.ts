import assert from 'node:assert/strict';
import * as C from './services/livenessCore';

let passed = 0;
const ok = (label: string) => {
  passed++;
  console.log('PASS', label);
};

// 1. CSPRNG unbiasedness
{
  const counts = [0, 0, 0];
  for (let i = 0; i < 30_000; i++) counts[C.secureRandomInt(3)]++;
  assert.ok(Math.max(...counts) - Math.min(...counts) < 1200);
  ok('secureRandomInt unbiased across 30k draws');
}

// 2. Personal EAR calibration (glasses / narrow-eye safe, no hardcoded 0.26 floor)
{
  assert.equal(C.calibrateEarPeak([0.17, 0.19, 0.18, 0.2, 0.185]), 0.185);
  assert.equal(C.calibrateEarPeak([0.08, 0.11]), 0);
  ok('calibrateEarPeak computes median without hardcoded 0.26 floor');
}

// 3. Bilateral BlinkTracker (glasses-safe, 15fps/30fps, long-close rejection)
{
  const run = (seq: number[], dt: number, lowFps: boolean, earPeak = 0.3) => {
    const tracker = new C.BlinkTracker();
    let t = 1000;
    let count = 0;
    for (const ear of seq) {
      t += dt;
      if (tracker.update(ear, 0.01, t, { earPeak, lowFps })) count++;
    }
    return count;
  };
  const open = (n: number, v = 0.3) => Array<number>(n).fill(v);

  assert.equal(run([...open(8), 0.14, 0.11, 0.15, ...open(8)], 33, false), 1);
  ok('real natural blink @30fps');

  assert.equal(run([...open(8), 0.16, ...open(8)], 66, true), 1);
  ok('real natural blink @15fps WASM');

  assert.equal(
    run([...open(8, 0.185), 0.135, 0.125, ...open(8, 0.185)], 50, false, 0.185),
    1
  );
  ok('glasses / narrow-eye blink (earPeak=0.185)');

  assert.equal(run([...open(8), 0.255, ...open(8)], 66, true), 0);
  ok('shallow landmark jitter rejected');

  assert.equal(run([...open(6), ...Array(32).fill(0.1), ...open(8)], 33, false), 0);
  ok('held-closed >950ms rejected and not counted on reopen');
}

// 4. Pre-blink vs Post-blink Identity Continuity (0.42 bait-and-switch guard)
{
  const rnd = (seed: number) =>
    Float32Array.from({ length: 128 }, (_, i) => Math.sin((i + 1) * seed));
  const a = C.l2(rnd(1.37));
  const b = C.l2(rnd(4.91));
  const aSlightlyMoved = C.l2(a.map((v, i) => v + 0.008 * ((i % 3) - 1)));

  assert.equal(C.identityContinuity(a, aSlightlyMoved, 0.42).ok, true);
  assert.equal(C.identityContinuity(a, b, 0.42).ok, false);
  ok('identityContinuity accepts same face and rejects bait-and-switch (>0.42)');

  const blended = C.meanNormalized(a, aSlightlyMoved, 0.7);
  assert.ok(Math.abs(Math.hypot(...blended) - 1) < 1e-4);
  ok('meanNormalized returns unit-length 128-D L2 vector');
}

// 5. Vertical 9:16 Phone Pillar-Box & Hair Black-Point Veil Guards
{
  const W = 256;
  const H = 192;
  const phoneFrame = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      // Bright vertical 9:16 phone screen in center (x in [76..180]), dark room outside
      phoneFrame[y * W + x] = x >= 76 && x <= 180 ? 155 : 22;
    }
  }
  const phoneRes = C.detectVerticalPhonePillarBox(phoneFrame, W, H, {
    x: 92,
    y: 42,
    w: 72,
    h: 96,
  });
  assert.equal(phoneRes.detected, true);
  assert.equal(phoneRes.bilateral, true);
  ok('detectVerticalPhonePillarBox blocks bilateral 9:16 vertical phone video');

  const liveRoom = new Float32Array(W * H).fill(128);
  const liveRes = C.detectVerticalPhonePillarBox(liveRoom, W, H, {
    x: 92,
    y: 42,
    w: 72,
    h: 96,
  });
  assert.equal(liveRes.detected, false);
  ok('detectVerticalPhonePillarBox allows normal evenly lit live frame');

  assert.equal(
    C.evaluateHairBlackPointVeil({
      hairDarkP05: 62,
      p10Luma: 78,
      p90Luma: 158,
      satMean: 0.11,
    }).detected,
    true
  );
  assert.equal(
    C.evaluateHairBlackPointVeil({
      hairDarkP05: 24,
      p10Luma: 38,
      p90Luma: 182,
      satMean: 0.21,
    }).detected,
    false
  );
  ok('evaluateHairBlackPointVeil distinguishes glass reflection veil from real hair');
}

// 6. Check-In Cooldown (180m default) & Duplicate Face De-Duplication
{
  const now = new Date('2026-11-24T10:30:00Z');
  const recs = [
    { attendanceId: '23210-ec-053', timestamp: '2026-11-24T09:10:00Z', status: 'present' },
    { attendanceId: '23210-ec-054', timestamp: '2026-11-24T10:15:00Z', status: 'flagged' },
    { attendanceId: '23210-ec-055', timestamp: '2026-11-24T05:00:00Z', status: 'present' },
  ];

  assert.ok(C.findPriorCheckIn(recs, '23210-EC-053', now, 180));
  ok('findPriorCheckIn blocks duplicate within 180m cooldown (case-insensitive)');

  assert.equal(C.findPriorCheckIn(recs, '23210-ec-054', now, 180), null);
  ok('findPriorCheckIn ignores flagged records');

  assert.equal(C.findPriorCheckIn(recs, '23210-ec-055', now, 180), null);
  ok('findPriorCheckIn allows check-in after cooldown expires');

  const v = Array.from({ length: 128 }, (_, i) => Math.sin(i + 1));
  const w = Array.from({ length: 128 }, (_, i) => Math.cos((i + 1) * 1.7));
  const roster = [
    { id: '1', attendanceId: '23210-ec-001', fullName: 'Asha', faceVectors: [v] },
    { id: '2', attendanceId: '23210-ec-002', fullName: 'Bala', faceVectors: [w] },
  ];
  assert.equal(
    C.findDuplicateFace([v.map((x) => x + 0.001)], roster, 0.4)?.attendanceId,
    '23210-ec-001'
  );
  assert.equal(
    C.findDuplicateFace(
      [Array.from({ length: 128 }, (_, i) => Math.sin((i + 1) * 3.3))],
      roster,
      0.4
    ),
    null
  );
  ok('findDuplicateFace catches duplicate biometric enrollment and allows new students');

  assert.equal(
    C.isEnrollableFrame({
      rollDeg: 4,
      noseCheekRatio: 1.0,
      ear: 0.22,
      earPeak: 0.24,
      sharpness: 28,
    }),
    true
  );
  assert.equal(
    C.isEnrollableFrame({
      rollDeg: 4,
      noseCheekRatio: 1.5,
      ear: 0.22,
      earPeak: 0.24,
      sharpness: 28,
    }),
    false
  );
  ok('isEnrollableFrame accepts sharp frontal frame and rejects turned head');
}

console.log(`\n${passed} checks passed`);
