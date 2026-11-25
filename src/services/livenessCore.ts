/**
 * src/services/livenessCore.ts
 * Pure mathematical anti-spoof, glasses-friendly single-blink liveness, and deduplication core.
 */

export interface Pt {
  x: number;
  y: number;
}

export type RGB = [number, number, number];

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

// 1. CSPRNG (Zero Modulo Bias) for Unpredictable Challenges
export function secureRandomInt(maxExclusive: number): number {
  if (maxExclusive <= 1) return 0;
  const limit = Math.floor(0x100000000 / maxExclusive) * maxExclusive;
  const buf = new Uint32Array(1);
  do {
    crypto.getRandomValues(buf);
  } while (buf[0] >= limit);
  return buf[0] % maxExclusive;
}

// 2. Glasses-Friendly Bilateral Single-Blink Tracker (Blue-Cut Lens & 12–15 FPS WASM Safe)
export class BlinkTracker {
  private closedAt = 0;
  private minEar = Infinity;
  private locked = false;

  public reset(): void {
    this.closedAt = 0;
    this.minEar = Infinity;
    this.locked = false;
  }

  public update(
    ear: number,
    symmetryDiff: number,
    now: number,
    opts: { earPeak: number; lowFps: boolean }
  ): boolean {
    const closeF = opts.lowFps ? 0.83 : 0.80;
    const openF = 0.87;
    const maxSym = opts.lowFps ? 0.13 : 0.12;
    const deepF = closeF;

    if (this.locked) {
      if (ear >= opts.earPeak * openF) this.locked = false;
      return false;
    }

    if (this.closedAt === 0) {
      if (ear <= opts.earPeak * closeF && symmetryDiff < maxSym) {
        this.closedAt = now;
        this.minEar = ear;
      }
      return false;
    }

    this.minEar = Math.min(this.minEar, ear);
    const dur = now - this.closedAt;

    if (dur > 950) {
      this.closedAt = 0;
      this.minEar = Infinity;
      this.locked = true;
      return false;
    }

    if (ear >= opts.earPeak * openF) {
      const deepest = this.minEar;
      this.closedAt = 0;
      this.minEar = Infinity;
      const minDur = 20;
      return (
        dur >= minDur &&
        dur <= 900 &&
        deepest <= opts.earPeak * deepF
      );
    }

    return false;
  }
}

/**
 * Computes a personal open-eye EAR baseline (median of clean frontal samples >= 0.14)
 * so naturally narrow eyes or glasses calibrate accurately without a hardcoded 0.26 floor.
 */
export function calibrateEarPeak(samples: readonly number[]): number {
  const valid = samples.filter((v) => Number.isFinite(v) && v >= 0.14);
  if (valid.length === 0) return 0;
  const sorted = [...valid].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) * 0.5;
}

// 3. L2 Vector Math & Bait-and-Switch Identity Continuity (0.46 accommodates glasses + slight yaw)
export const CONTINUITY_MAX_DISTANCE = 0.46;

export function l2(vec: ArrayLike<number>): Float32Array {
  const len = vec.length || 128;
  let s = 0;
  for (let i = 0; i < len; i++) s += vec[i] * vec[i];
  const inv = 1 / (Math.sqrt(s) + 1e-12);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) out[i] = vec[i] * inv;
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

/**
 * Returns an L2-normalized weighted blend of two 128-D vectors.
 * Defaults to 50/50 mean; pass weightA = 0.75 for 75% pre-blink + 25% post-turn blend.
 */
export function meanNormalized(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  weightA = 0.5
): Float32Array {
  const len = Math.min(a.length, b.length) || 128;
  const wA = Math.min(1, Math.max(0, weightA));
  const wB = 1 - wA;
  const blended = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    blended[i] = a[i] * wA + b[i] * wB;
  }
  return l2(blended);
}

export function identityContinuity(
  preChallengeVec: ArrayLike<number>,
  postChallengeVec: ArrayLike<number>,
  maxDist = CONTINUITY_MAX_DISTANCE
): { ok: boolean; distance: number } {
  const distance = dist128(preChallengeVec, postChallengeVec);
  return { ok: distance <= maxDist, distance: Number(distance.toFixed(4)) };
}

// 4. Borderless Screen Spoof & Active Chromatic Illumination Geometry Utilities
export const FLASH_PALETTE: readonly RGB[] = [
  [255, 52, 52],   // Vivid Red (high +R-B)
  [48, 152, 255],  // Vivid Blue/Cyan (high -R+B)
  [52, 232, 124],  // Vivid Green (high +G)
  [244, 72, 255],  // Vivid Magenta (high R+B, low G)
  [255, 224, 68],  // Vivid Amber/Yellow (high R+G, low B)
];

/**
 * Generates a CSPRNG sequence of high-contrast chromatic flash colors so adjacent
 * slots always maximize inter-channel chromatic separation (ideal for R-B skin reflection).
 */
export function makeFlashPlan(slots = 2): RGB[] {
  const plan: RGB[] = [];
  let lastIdx = -1;
  let guard = 0;
  while (plan.length < slots && guard < 64) {
    guard++;
    const k = secureRandomInt(FLASH_PALETTE.length);
    if (k === lastIdx) continue;
    if (lastIdx >= 0 && guard < 40) {
      const prev = FLASH_PALETTE[lastIdx];
      const next = FLASH_PALETTE[k];
      const rbGap = Math.abs((next[0] - next[2]) - (prev[0] - prev[2]));
      const rgbDist = Math.hypot(
        next[0] - prev[0],
        next[1] - prev[1],
        next[2] - prev[2]
      );
      if (rbGap < 180 && rgbDist < 240) continue;
    }
    plan.push([...FLASH_PALETTE[k]] as RGB);
    lastIdx = k;
  }
  while (plan.length < slots) {
    const fallbackIdx = (lastIdx + 1) % FLASH_PALETTE.length;
    plan.push([...FLASH_PALETTE[fallbackIdx]] as RGB);
    lastIdx = fallbackIdx;
  }
  return plan;
}

export interface FlashScoreResult {
  corr: number;
  rbSlope: number;
  projGain: number;
  diffuseRatio: number;
  passed: boolean;
  reason: string;
}

/**
 * Evaluates synchronized chromatic skin reflection against the projected flash plan.
 * Real human skin diffusely reflects the monitor's RGB color shift relative to ambient
 * background, whereas backlit OLED/LCD screens emit their own fixed light and fail
 * synchronized R-B chromatic correlation.
 */
export function flashScore(params: {
  plan: readonly RGB[];
  cheekSamples: readonly RGB[];
  bgSamples: readonly RGB[];
  innerLumaMean?: number;
}): FlashScoreResult {
  const { plan, cheekSamples, bgSamples, innerLumaMean = 120 } = params;
  if (plan.length < 2 || cheekSamples.length < 2) {
    return {
      corr: 0,
      rbSlope: 0,
      projGain: 0,
      diffuseRatio: 0,
      passed: false,
      reason: 'Insufficient chromatic reflection samples captured',
    };
  }

  const c0 = plan[0];
  const c1 = plan[1];
  const ch0 = cheekSamples[0];
  const ch1 = cheekSamples[1];
  const bg0 = bgSamples[0] ?? [0, 0, 0];
  const bg1 = bgSamples[1] ?? [0, 0, 0];

  // Expected stimulus shift (Color 0 -> Color 1)
  const dStim: RGB = [c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]];
  const stimNorm = Math.hypot(dStim[0], dStim[1], dStim[2]) + 1e-6;

  // Measured cheek RGB shift (partially compensated for global background AWB drift)
  const rawCheek: RGB = [ch1[0] - ch0[0], ch1[1] - ch0[1], ch1[2] - ch0[2]];
  const dBg: RGB = [bg1[0] - bg0[0], bg1[1] - bg0[1], bg1[2] - bg0[2]];
  const dCheek: RGB = [
    rawCheek[0] - dBg[0] * 0.25,
    rawCheek[1] - dBg[1] * 0.25,
    rawCheek[2] - dBg[2] * 0.25,
  ];
  const cheekNorm = Math.hypot(dCheek[0], dCheek[1], dCheek[2]) + 1e-6;

  // Zero-mean chromatic components (isolates hue shift from pure brightness exposure)
  const sMean = (dStim[0] + dStim[1] + dStim[2]) / 3;
  const cMean = (dCheek[0] + dCheek[1] + dCheek[2]) / 3;
  const sChroma: RGB = [dStim[0] - sMean, dStim[1] - sMean, dStim[2] - sMean];
  const cChroma: RGB = [dCheek[0] - cMean, dCheek[1] - cMean, dCheek[2] - cMean];
  const sChromaNorm = Math.hypot(sChroma[0], sChroma[1], sChroma[2]) + 1e-6;
  const cChromaNorm = Math.hypot(cChroma[0], cChroma[1], cChroma[2]) + 1e-6;

  const chromaCorr =
    (sChroma[0] * cChroma[0] +
      sChroma[1] * cChroma[1] +
      sChroma[2] * cChroma[2]) /
    (sChromaNorm * cChromaNorm);
  const fullCorr =
    (dStim[0] * dCheek[0] + dStim[1] * dCheek[1] + dStim[2] * dCheek[2]) /
    (stimNorm * cheekNorm);

  // Use the stronger of zero-mean chroma correlation and full RGB correlation
  const corr = Number(Math.max(chromaCorr, fullCorr).toFixed(3));
  const projGain = Number(
    ((sChroma[0] * cChroma[0] +
      sChroma[1] * cChroma[1] +
      sChroma[2] * cChroma[2]) /
      sChromaNorm).toFixed(3)
  );

  // Signed R-B chromatic slope in the direction of the stimulus R-B flip
  const stimRB = (c1[0] - c1[2]) - (c0[0] - c0[2]);
  const obsRB = (ch1[0] - ch1[2]) - (ch0[0] - ch0[2]);
  const rbSlope = Number(
    (Math.abs(stimRB) >= 40
      ? obsRB * Math.sign(stimRB)
      : projGain
    ).toFixed(3)
  );

  const diffuseRatio = Number(
    (projGain / Math.max(40, innerLumaMean)).toFixed(4)
  );

  // Real skin reflects synchronized screen chroma (corr >= 0.35 and positive slope/response).
  // Emissive phone/tablet displays fail synchronized R-B reflection or show negative/zero slope.
  const passed =
    corr >= 0.35 &&
    rbSlope > 0.35 &&
    projGain >= 0.65 &&
    diffuseRatio >= 0.004;

  const reason = passed
    ? `Chromatic skin reflection verified (corr ${corr.toFixed(2)}, Δ ${projGain.toFixed(1)})`
    : corr < 0.35
    ? `Emissive screen detected — failed chromatic skin reflection (corr ${corr.toFixed(2)} < 0.35)`
    : rbSlope <= 0.35
    ? `Emissive screen detected — negative/flat R-B reflection slope (${rbSlope.toFixed(2)})`
    : `Emissive screen detected — zero diffuse skin response (${projGain.toFixed(2)})`;

  return {
    corr,
    rbSlope,
    projGain,
    diffuseRatio,
    passed,
    reason,
  };
}

export function cheekRects(pts: Pt[]): Rect[] {
  const r = Math.max(
    4,
    Math.hypot(pts[0].x - pts[16].x, pts[0].y - pts[16].y) * 0.075
  );
  const left = {
    x: (pts[31].x + pts[3].x) * 0.5,
    y: (pts[31].y + pts[3].y) * 0.5,
  };
  const right = {
    x: (pts[35].x + pts[13].x) * 0.5,
    y: (pts[35].y + pts[13].y) * 0.5,
  };
  return [left, right].map((c) => ({
    x: c.x - r,
    y: c.y - r,
    w: r * 2,
    h: r * 2,
  }));
}

export function backgroundRects(W: number, H: number, box: Rect): Rect[] {
  const w = Math.round(W * 0.14);
  const h = Math.round(H * 0.14);
  const ex = {
    x: box.x - box.w * 0.3,
    y: box.y - box.h * 0.3,
    w: box.w * 1.6,
    h: box.h * 1.6,
  };
  return (
    [
      [0, 0],
      [W - w, 0],
      [0, H - h],
      [W - w, H - h],
    ] as const
  )
    .map(([x, y]) => ({ x, y, w, h }))
    .filter(
      (r) =>
        !(
          r.x < ex.x + ex.w &&
          r.x + r.w > ex.x &&
          r.y < ex.y + ex.h &&
          r.y + r.h > ex.y
        )
    );
}

/**
 * Zero-delay borderless OLED/LCD emissive screen classifier (used with 2-strike confirmation).
 * Tuned for zero indoor real-face false positives while catching close-up backlit displays.
 */
export function isBorderlessEmissiveScreen(stats: {
  clippedRatio: number;
  satStd: number;
  brightRatio: number;
  lumaStd: number;
  blueExcess: number;
}): boolean {
  return (
    (stats.clippedRatio > 0.065 && stats.satStd < 0.045) ||
    (stats.brightRatio > 0.52 && stats.lumaStd < 28 && stats.blueExcess > 7.5)
  );
}

/**
 * OLED/LCD Glass Veil (Lifted Black-Point) Classifier (used with 2-strike confirmation).
 * Tuned conservatively so normal laptop webcam indoor haze never triggers false positives.
 */
export function isGlassBlackPointVeil(stats: {
  p03Luma: number;
  p95Luma: number;
  satStd: number;
  highFreqSpike: boolean;
}): boolean {
  const dynamicRange = stats.p95Luma - stats.p03Luma;
  return (
    stats.p03Luma >= 68 &&
    dynamicRange < 92 &&
    (stats.satStd < 0.045 || stats.highFreqSpike)
  );
}

export interface VerticalPillarBoxResult {
  detected: boolean;
  bilateral: boolean;
  leftCoherence: number;
  rightCoherence: number;
  centerLuma: number;
  leftStepDelta: number;
  rightStepDelta: number;
  reason: string;
}

/**
 * Full-Frame 4:3 vs 9:16 Vertical Smartphone Pillar-Box Detector.
 * Scans full-height vertical columns on the 256x192 grayscale frame to catch
 * vertical (9:16) phone screens held in front of a 4:3 laptop webcam—where a
 * bright central phone display (> 95 luma) is flanked on left & right by sharp
 * vertical luminance drops (>= 38 across >= 55% of frame height) into darker
 * off-phone room bars (< 58 luma).
 */
export function detectVerticalPhonePillarBox(
  gray: Float32Array,
  W: number,
  H: number,
  faceBox: Rect
): VerticalPillarBoxResult {
  const yStart = Math.max(4, Math.round(H * 0.08));
  const yEnd = Math.min(H - 5, Math.round(H * 0.92));

  // 1. Measure mean luma of the bright central face band
  const cL = Math.max(4, Math.floor(faceBox.x - faceBox.w * 0.08));
  const cR = Math.min(W - 5, Math.ceil(faceBox.x + faceBox.w * 1.08));
  const cT = Math.max(yStart, Math.floor(faceBox.y));
  const cB = Math.min(yEnd, Math.ceil(faceBox.y + faceBox.h));
  let centerSum = 0;
  let centerCnt = 0;
  for (let y = cT; y <= cB; y += 2) {
    const row = y * W;
    for (let x = cL; x <= cR; x += 2) {
      centerSum += gray[row + x];
      centerCnt++;
    }
  }
  const centerLuma = centerSum / (centerCnt || 1);

  if (centerLuma <= 95) {
    return {
      detected: false,
      bilateral: false,
      leftCoherence: 0,
      rightCoherence: 0,
      centerLuma,
      leftStepDelta: 0,
      rightStepDelta: 0,
      reason: '',
    };
  }

  // Slight tilt slopes (±7°) accommodate hand-held vertical phones
  const slopes = [-0.12, -0.06, 0, 0.06, 0.12];

  const scanSide = (xMin: number, xMax: number, isLeft: boolean) => {
    let bestCoherence = 0;
    let bestStep = 0;
    let bestOuterMean = 255;
    let bestInnerMean = 0;

    if (xMax <= xMin) {
      return { coherence: 0, stepDelta: 0, outerMean: 255, innerMean: 0 };
    }

    for (const slope of slopes) {
      for (let x0 = xMin; x0 <= xMax; x0 += 2) {
        let hits = 0;
        let stepSum = 0;
        let outerSum = 0;
        let innerSum = 0;
        let samples = 0;

        for (let y = yStart; y <= yEnd; y += 2) {
          const x = Math.round(x0 + (y - H * 0.5) * slope);
          if (x < 6 || x >= W - 6) continue;
          const row = y * W;

          const outVal = isLeft
            ? (gray[row + x - 3] + gray[row + x - 4] + gray[row + x - 5]) / 3
            : (gray[row + x + 3] + gray[row + x + 4] + gray[row + x + 5]) / 3;
          const inVal = isLeft
            ? (gray[row + x + 2] + gray[row + x + 3] + gray[row + x + 4]) / 3
            : (gray[row + x - 2] + gray[row + x - 3] + gray[row + x - 4]) / 3;
          const delta = inVal - outVal;

          if (outVal < 58 && inVal > 82 && delta >= 38) {
            hits++;
          }
          stepSum += Math.max(0, delta);
          outerSum += outVal;
          innerSum += inVal;
          samples++;
        }

        if (samples < 24) continue;
        const coherence = hits / samples;
        const stepDelta = stepSum / samples;
        const outerMean = outerSum / samples;
        const innerMean = innerSum / samples;

        // Verify the far-outer pillar-box bar stays dark (not just a narrow shadow line)
        const farBarX = isLeft ? Math.max(2, x0 - 10) : Math.min(W - 3, x0 + 10);
        let farSum = 0;
        let farCnt = 0;
        for (let y = yStart; y <= yEnd; y += 4) {
          farSum += gray[y * W + farBarX];
          farCnt++;
        }
        const farBarMean = farSum / (farCnt || 1);
        if (farBarMean >= 64) continue;

        if (
          coherence > bestCoherence ||
          (coherence >= bestCoherence - 0.02 && stepDelta > bestStep)
        ) {
          bestCoherence = coherence;
          bestStep = stepDelta;
          bestOuterMean = outerMean;
          bestInnerMean = innerMean;
        }
      }
    }

    return {
      coherence: bestCoherence,
      stepDelta: bestStep,
      outerMean: bestOuterMean,
      innerMean: bestInnerMean,
    };
  };

  const leftEdge = scanSide(
    6,
    Math.min(W - 24, Math.max(14, Math.floor(faceBox.x) + 4)),
    true
  );
  const rightEdge = scanSide(
    Math.max(24, Math.min(W - 14, Math.ceil(faceBox.x + faceBox.w) - 4)),
    W - 7,
    false
  );

  // Bilateral 9:16 vertical phone pillar-box (both left & right dark bars flanking bright screen)
  const bilateral =
    leftEdge.coherence >= 0.55 &&
    rightEdge.coherence >= 0.55 &&
    leftEdge.stepDelta >= 38 &&
    rightEdge.stepDelta >= 38 &&
    leftEdge.outerMean < 58 &&
    rightEdge.outerMean < 58;

  // Extreme single-side vertical phone edge (when phone is shifted partly off one edge)
  const strongUnilateral =
    (leftEdge.coherence >= 0.74 &&
      leftEdge.stepDelta >= 52 &&
      leftEdge.outerMean < 44 &&
      leftEdge.innerMean > 108) ||
    (rightEdge.coherence >= 0.74 &&
      rightEdge.stepDelta >= 52 &&
      rightEdge.outerMean < 44 &&
      rightEdge.innerMean > 108);

  if (bilateral || strongUnilateral) {
    return {
      detected: true,
      bilateral,
      leftCoherence: Number(leftEdge.coherence.toFixed(2)),
      rightCoherence: Number(rightEdge.coherence.toFixed(2)),
      centerLuma: Math.round(centerLuma),
      leftStepDelta: Math.round(leftEdge.stepDelta),
      rightStepDelta: Math.round(rightEdge.stepDelta),
      reason: bilateral
        ? `Vertical 9:16 phone pillar-box bars detected (L ${(leftEdge.coherence * 100).toFixed(0)}% Δ${Math.round(leftEdge.stepDelta)}, R ${(rightEdge.coherence * 100).toFixed(0)}% Δ${Math.round(rightEdge.stepDelta)})`
        : `Sharp full-height smartphone edge detected (coh ${(Math.max(leftEdge.coherence, rightEdge.coherence) * 100).toFixed(0)}%, Δ${Math.round(Math.max(leftEdge.stepDelta, rightEdge.stepDelta))})`,
    };
  }

  return {
    detected: false,
    bilateral: false,
    leftCoherence: Number(leftEdge.coherence.toFixed(2)),
    rightCoherence: Number(rightEdge.coherence.toFixed(2)),
    centerLuma: Math.round(centerLuma),
    leftStepDelta: Math.round(leftEdge.stepDelta),
    rightStepDelta: Math.round(rightEdge.stepDelta),
    reason: '',
  };
}

export interface HairBlackPointVeilResult {
  detected: boolean;
  hairDarkP05: number;
  contrastSpan: number;
  satMean: number;
}

/**
 * Evaluates smartphone glass reflection veil / lifted shadows in upper hair & eyebrows.
 * Real human hair & eyebrows absorb indoor light (deep dark black-point < 45), whereas
 * phone glass re-photographed by a webcam lifts blacks into milky gray (>= 54) while
 * compressing midtone dynamic range (< 105) and desaturating skin chroma (< 0.145).
 */
export function evaluateHairBlackPointVeil(stats: {
  hairDarkP05: number;
  p10Luma: number;
  p90Luma: number;
  satMean: number;
}): HairBlackPointVeilResult {
  const contrastSpan = Math.max(0, stats.p90Luma - stats.p10Luma);
  const detected =
    stats.hairDarkP05 >= 54 && contrastSpan < 105 && stats.satMean < 0.145;
  return {
    detected,
    hairDarkP05: Math.round(stats.hairDarkP05),
    contrastSpan: Math.round(contrastSpan),
    satMean: Number(stats.satMean.toFixed(3)),
  };
}

// 5. Attendance Cooldown & Aadhaar-Style Enrollment De-Duplication
export interface RecordLike {
  attendanceId: string;
  timestamp: string;
  status: string;
}

export function findPriorCheckIn<T extends RecordLike>(
  records: readonly T[],
  attendanceId: string,
  now: Date,
  cooldownMinutes = 480
): T | null {
  const key = attendanceId.trim().toLowerCase();
  const windowMs = Math.max(0, cooldownMinutes) * 60_000;
  let latest: T | null = null;
  for (const r of records) {
    if (r.attendanceId.toLowerCase() !== key) continue;
    if (r.status !== 'present' && r.status !== 'late') continue;
    const t = Date.parse(r.timestamp);
    if (Number.isNaN(t) || now.getTime() - t > windowMs) continue;
    if (!latest || t > Date.parse(latest.timestamp)) latest = r;
  }
  return latest;
}

export interface EnrolledLike {
  id: string;
  attendanceId: string;
  fullName: string;
  faceVectors?: number[][] | null;
  faceDescriptor?: number[] | null;
}

export function findDuplicateFace(
  newVectors: readonly ArrayLike<number>[],
  roster: readonly EnrolledLike[],
  threshold = 0.4
): { attendanceId: string; fullName: string; distance: number } | null {
  const probes = newVectors.filter((v) => v.length === 128).map(l2);
  let best: { attendanceId: string; fullName: string; distance: number } | null =
    null;

  for (const s of roster) {
    const stored = (
      s.faceVectors?.length
        ? s.faceVectors
        : s.faceDescriptor
        ? [s.faceDescriptor]
        : []
    ).filter((v) => v.length === 128);

    for (const sv of stored) {
      const n = l2(sv);
      for (const p of probes) {
        const d = dist128(p, n);
        if (d <= threshold && (!best || d < best.distance)) {
          best = {
            attendanceId: s.attendanceId,
            fullName: s.fullName,
            distance: Number(d.toFixed(4)),
          };
        }
      }
    }
  }
  return best;
}

export function isEnrollableFrame(f: {
  rollDeg: number;
  noseCheekRatio: number;
  ear: number;
  earPeak: number;
  sharpness: number;
}): boolean {
  const minEar = f.earPeak >= 0.14 ? Math.max(0.145, f.earPeak * 0.82) : 0.145;
  return (
    f.rollDeg <= 12 &&
    f.noseCheekRatio >= 0.78 &&
    f.noseCheekRatio <= 1.28 &&
    f.ear >= minEar &&
    f.sharpness >= 15
  );
}
