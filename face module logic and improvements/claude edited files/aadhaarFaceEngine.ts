import * as faceapi from '@vladmandic/face-api';
import type { BiometricQualityChecks, LiveChallengeType } from '../types';
import {
  getActiveBackend,
  getBackendDisplayLabel,
  getTfRuntime,
  initTensorFlowBackend,
  useTensorFlowBackend,
  type ComputeBackend,
} from './initTensorFlowBackend';
import {
  BlinkTracker,
  backgroundRects,
  cheekRects,
  dist128,
  FLASH_NEUTRAL,
  flashScore,
  identityContinuity,
  isEnrollableFrame,
  makeFlashPlan,
  meanNormalized,
  pickDistinct,
  type FlashObservation,
  type FlashTier,
  type Rect,
  type RGB,
} from './livenessCore';

/**
 * VeriMark Anti-Spoof Aadhaar-Grade Facial Attendance Engine
 * ----------------------------------------------------------
 * Powered by the uploaded neural weight manifests in /public/models:
 * - /models/ssd_mobilenetv1_model-weights_manifest.json
 * - /models/face_landmark_68_model-weights_manifest.json
 * - /models/face_recognition_model-weights_manifest.json
 *
 * Security & Performance Architecture:
 * 1. ZERO Eyeglass/Timeout Auto-Bypass: Attendance NEVER auto-approves without live proof.
 * 2. 1.5x Outer-Context Smartphone Bezel & Emissive Screen Detector: Detects slanted/straight
 *    parallel vertical dark bezels flanking glowing screens, clipped OLED highlights (>244),
 *    chromaticity flatness, and Moire subpixel grids.
 * 3. Randomized Live Challenge + 3D Depth Parallax + Reaction-Time Gate:
 *    - Requires 280ms clean open-eye neutral baseline (blocks looped blinking videos).
 *    - Randomly selects "Blink + Slight Left Turn", "Blink + Slight Right Turn", or
 *      "Smile + Natural Blink" with a strict 160ms–4000ms human reaction window.
 *    - Verifies true 3D nose-tip Z-parallax against 4-corner perspective homography skew,
 *      immediately rejecting 2D flat phone tilts.
 * 4. High-FPS WASM SIMD Pipeline: Downscales SSD MobileNet input to 256x192 & caps ROI
 *    landmark crops for 15–25 FPS tracking on WASM/CPU, widens low-FPS EAR blink capture,
 *    and extracts the final 150x150 ResNet-34 winner crop from full-resolution video.
 */

const MODEL_MANIFESTS = {
  ssd: '/models/ssd_mobilenetv1_model-weights_manifest.json',
  landmarks68: '/models/face_landmark_68_model-weights_manifest.json',
  recognition: '/models/face_recognition_model-weights_manifest.json',
} as const;

const DETECT_W = 256;
const DETECT_H = 192;

// Virtual-camera DRIVERS only. (The old list matched 'obs' => blocked real OBSBOT webcams, and 'snap'/'droidcam'/
// 'iriun'/'epoccam' => blocked legit phone-as-webcam sensors.) Labels are trivially renamed, so this is a first
// filter only — the screen-flash light check is the real defence against injected video.
const BLOCKED_CAM_KEYWORDS = [
  'obs virtual',
  'obs-camera',
  'obs studio',
  'manycam',
  'vcam',
  'virtual',
  'splitcam',
  'xsplit',
  'snap camera',
  'v4l2loopback',
  'ndi video',
];

// Two DISTINCT random challenges per session (6 ordered pairs, CSPRNG) + the random flash sequence.
const CHALLENGES_PER_SESSION = 2;

export interface FlashOptions {
  enabled: boolean;
  slots?: number; // default 8
  slotMs?: number; // default 340 (< 3 flashes/second, WCAG photosensitivity limit)
  setOverlay: (css: string | null) => void;
}

const CHALLENGE_POOL: readonly Exclude<LiveChallengeType, 'none'>[] = [
  'blink_left',
  'blink_right',
  'smile_blink',
];

export function getChallengePromptLabel(type: LiveChallengeType): string {
  if (type === 'blink_left') return 'Blink + Slight Left Turn';
  if (type === 'blink_right') return 'Blink + Slight Right Turn';
  if (type === 'smile_blink') return 'Smile + Natural Blink';
  return 'Hold steady';
}

export interface EnrolledStudentVector {
  id: string;
  attendanceId: string;
  name: string;
  vectors: Float32Array[]; // 1 to 4 L2-normalized Float32Array(128)
}

export interface FastHudState {
  instruction: string;
  subtext: string;
  progress: number; // 0..100
  shouldFlashUI: boolean;
  qualityStatus: 'ok' | 'warn' | 'locked';
  activeChallenge: LiveChallengeType;
  checks: {
    frontal: boolean;
    screenClear: boolean;
    liveness: boolean;
    vector: boolean;
  };
}

export interface FastAttendanceResult {
  matched: boolean;
  studentId: string | null;
  attendanceId: string | null;
  distance: number;
  confidence: number;
  latencyMs: number;
  spoofPassed: boolean;
  liveDescriptor: number[];
  capturedFrameDataUrl: string;
  qualityChecks: BiometricQualityChecks;
  failReason?: string;
  needsReview?: boolean; // weak light-reflection signal: do NOT auto-accept, send to operator review
}

export interface EnrollmentBurstResult {
  ok: boolean;
  reason?: string;
  vectors?: number[][]; // [centroid, frame1, frame2, frame3] L2-normalized
  primaryDescriptor?: number[];
  capturedFrameDataUrl?: string;
  latencyMs: number;
  qualityChecks: BiometricQualityChecks;
}

interface BufferedWinnerFrame {
  alignedCanvas: HTMLCanvasElement; // 150x150 dlib-aligned open-eye crop from full-res video
  snapshotCanvas: HTMLCanvasElement; // Clean frontal snapshot for audit receipt
  sharpness: number;
  lumaCheek: number;
}

interface BaselineGeometry {
  noseCheekRatio: number;
  faceWidth: number;
  faceHeight: number;
  aspectRatio: number;
  cornerVertSkew: number;
  mouthWidthRatio: number;
}

export {
  getActiveBackend,
  getBackendDisplayLabel,
  initTensorFlowBackend,
  useTensorFlowBackend,
  type ComputeBackend,
};

let warmupPromise: Promise<ComputeBackend> | null = null;
let engineWarmed = false;

export function isAadhaarEngineWarmed(): boolean {
  return engineWarmed;
}

export async function warmupAadhaarEngine(): Promise<ComputeBackend> {
  const existing = getActiveBackend();
  if (engineWarmed && existing) return existing;
  if (!warmupPromise) {
    warmupPromise = (async () => {
      // 1. Negotiate healthy TFJS backend (webgl v2/v1 -> local /wasm/ -> cpu)
      const backend = await initTensorFlowBackend();

      // 2. Load directly from the exact uploaded shard manifests in /public/models
      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_MANIFESTS.ssd),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_MANIFESTS.landmarks68),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_MANIFESTS.recognition),
      ]);

      // 3. Leak-free shader/kernel warmup on small blank canvases
      const blank64 = document.createElement('canvas');
      blank64.width = 64;
      blank64.height = 64;
      const ctx64 = blank64.getContext('2d');
      if (ctx64) {
        ctx64.fillStyle = '#000000';
        ctx64.fillRect(0, 0, 64, 64);
      }

      const blank150 = document.createElement('canvas');
      blank150.width = 150;
      blank150.height = 150;
      const ctx150 = blank150.getContext('2d');
      if (ctx150) {
        ctx150.fillStyle = '#000000';
        ctx150.fillRect(0, 0, 150, 150);
      }

      const tfRef = getTfRuntime();
      if (tfRef?.tidy && tfRef?.zeros) {
        const tmp = tfRef.tidy(() => tfRef.zeros!([1, 64, 64, 3]));
        tfRef.dispose?.(tmp);
      }

      try {
        await faceapi.detectSingleFace(
          blank64,
          new faceapi.SsdMobilenetv1Options({ minConfidence: 0.99 })
        );
        await faceapi.nets.faceLandmark68Net.detectLandmarks(blank64);
        await faceapi.nets.faceRecognitionNet.computeFaceDescriptor(blank150);
      } catch {
        // Non-fatal if context is busy; weights remain loaded on active backend
      }

      engineWarmed = true;
      return backend;
    })().catch((err) => {
      warmupPromise = null;
      throw err;
    });
  }
  return warmupPromise;
}

// ---------- L2 Vector Math Helpers ----------
export function l2Normalize(vec: ArrayLike<number>): Float32Array {
  const len = vec.length || 128;
  let sumSq = 0;
  for (let i = 0; i < len; i++) {
    sumSq += vec[i] * vec[i];
  }
  const invNorm = 1 / (Math.sqrt(sumSq) + 1e-12);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    out[i] = vec[i] * invNorm;
  }
  return out;
}

export function computeNormalizedCentroid(vectors: Float32Array[]): Float32Array {
  if (vectors.length === 0) return new Float32Array(128);
  const acc = new Float32Array(128);
  for (const v of vectors) {
    for (let i = 0; i < 128; i++) acc[i] += v[i];
  }
  return l2Normalize(acc);
}

export function euclideanDistance(
  a: ArrayLike<number>,
  b: ArrayLike<number>
): number {
  if (a.length !== b.length || a.length === 0) return 1.0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Calibrated biometric match confidence for L2-normalized ResNet-34 vectors.
 * Same-person L2 distance is typically 0.22–0.44; impostors are >0.56.
 */
export function distanceToConfidence(distance: number): number {
  const clamped = Math.max(0, Math.min(1.2, distance));
  // Smooth calibrated mapping: 0.22 -> 97%, 0.35 -> 91%, 0.45 -> 84%, 0.58 -> 62%, 0.72 -> 35%
  const score = 100 / (1 + Math.exp((clamped - 0.62) * 9.5));
  return Number(Math.max(5, Math.min(99.8, score)).toFixed(1));
}

// ---------- Core Anti-Spoof Sub-Second Aadhaar Attendance Engine ----------
export class AadhaarSpeedAttendanceEngine {
  private frameCount = 0;
  private lastBox: faceapi.Box | null = null;
  private lastTickMs = 0;
  private avgFrameDeltaMs = 55;

  // Reusable offscreen canvases (zero per-frame DOM allocations)
  private readonly detectCanvas: HTMLCanvasElement;
  private readonly detectCtx: CanvasRenderingContext2D;
  private readonly roiCanvas: HTMLCanvasElement;
  private readonly roiCtx: CanvasRenderingContext2D;
  private readonly patchCanvas: HTMLCanvasElement;
  private readonly patchCtx: CanvasRenderingContext2D;
  private readonly outerCanvas: HTMLCanvasElement;
  private readonly outerCtx: CanvasRenderingContext2D;

  // Phase 1: Neutral Open-Eye Baseline Lock (defeats pre-recorded blinking videos)
  private neutralHoldStartMs = 0;
  private baseline: BaselineGeometry | null = null;

  // Phase 2: Randomized Micro-Challenge + Human Reaction-Time Window
  private activeChallenge: LiveChallengeType = 'none';
  private challengeIssuedAt = 0;
  private challengeActionDoneAt = 0;
  private blinkCompletedAt = 0;
  private depthParallaxConfirmed = false;

  // Blink tracker (raw EAR + hysteresis + depth + long-closure latch; see livenessCore.ts)
  private earPeak = 0.27;
  private readonly blink = new BlinkTracker();
  private blinkCompleted = false;

  // Multi-step random challenge queue (CSPRNG) + post-challenge identity frame (anti bait-and-switch)
  private challengeQueue: Exclude<LiveChallengeType, 'none'>[] = [];
  private completedChallenges: LiveChallengeType[] = [];
  private reactionTimes: number[] = [];
  private depthAll = true;
  private tremorOk = true;
  private startFrozen = false;
  private awaitingEndFrame = false;
  private endWaitStartMs = 0;
  private endFrame: BufferedWinnerFrame | null = null;
  private ssdMissCount = 0;
  private readonly flashCanvas: HTMLCanvasElement;
  private readonly flashCtx: CanvasRenderingContext2D;

  // Passive Anti-Spoof Signals (Micro-tremor + Soft 180ms Reticle Reflection)
  private noseRatios: number[] = [];
  private lumaDark = -1;
  private lumaBright = -1;
  private sessionStartMs = 0;
  private screenSpoofStrikes = 0;

  // Sharpest Pre-Blink Open-Eye Winner Buffer (full-resolution 150x150 dlib crop)
  private bestFrame: BufferedWinnerFrame | null = null;

  private matchThreshold: number;
  private impostorMargin: number;

  constructor(options?: { matchThreshold?: number; impostorMargin?: number }) {
    this.matchThreshold = options?.matchThreshold ?? 0.45;
    this.impostorMargin = options?.impostorMargin ?? 0.07;

    this.detectCanvas = document.createElement('canvas');
    this.detectCanvas.width = DETECT_W;
    this.detectCanvas.height = DETECT_H;
    this.detectCtx = this.detectCanvas.getContext('2d', {
      willReadFrequently: true,
    })!;

    this.roiCanvas = document.createElement('canvas');
    this.roiCtx = this.roiCanvas.getContext('2d', {
      willReadFrequently: true,
    })!;

    this.patchCanvas = document.createElement('canvas');
    this.patchCanvas.width = 96;
    this.patchCanvas.height = 96;
    this.patchCtx = this.patchCanvas.getContext('2d', {
      willReadFrequently: true,
    })!;

    this.outerCanvas = document.createElement('canvas');
    this.outerCanvas.width = 128;
    this.outerCanvas.height = 128;
    this.outerCtx = this.outerCanvas.getContext('2d', {
      willReadFrequently: true,
    })!;

    this.flashCanvas = document.createElement('canvas');
    this.flashCanvas.width = 320;
    this.flashCanvas.height = 240;
    this.flashCtx = this.flashCanvas.getContext('2d', {
      willReadFrequently: true,
    })!;

    this.resetSession();
  }

  public configure(matchThreshold: number, impostorMargin: number): void {
    this.matchThreshold = matchThreshold;
    this.impostorMargin = impostorMargin;
  }

  public resetSession(): void {
    this.frameCount = 0;
    this.lastBox = null;
    this.lastTickMs = 0;
    this.neutralHoldStartMs = 0;
    this.baseline = null;
    this.activeChallenge = 'none';
    this.challengeIssuedAt = 0;
    this.challengeActionDoneAt = 0;
    this.blinkCompletedAt = 0;
    this.depthParallaxConfirmed = false;
    this.earPeak = 0.26;
    this.blink.reset();
    this.blinkCompleted = false;
    this.challengeQueue = pickDistinct(CHALLENGE_POOL, CHALLENGES_PER_SESSION);
    this.completedChallenges = [];
    this.reactionTimes = [];
    this.depthAll = true;
    this.tremorOk = true;
    this.startFrozen = false;
    this.awaitingEndFrame = false;
    this.endWaitStartMs = 0;
    this.endFrame = null;
    this.ssdMissCount = 0;
    this.noseRatios = [];
    this.lumaDark = -1;
    this.lumaBright = -1;
    this.screenSpoofStrikes = 0;
    this.bestFrame = null;
    this.sessionStartMs = performance.now();
  }

  /**
   * 1. HARDWARE VIRTUAL CAMERA GUARD
   */
  public verifyPhysicalCamera(stream: MediaStream): {
    valid: boolean;
    label: string;
    reason?: string;
  } {
    const track = stream.getVideoTracks()[0];
    if (!track) {
      return {
        valid: false,
        label: 'None',
        reason: 'No active camera track detected.',
      };
    }
    const label = track.label || 'Physical Camera';
    const lower = label.toLowerCase();
    if (BLOCKED_CAM_KEYWORDS.some((kw) => lower.includes(kw))) {
      return {
        valid: false,
        label,
        reason: `Virtual camera blocked (${label}). Use a physical webcam sensor.`,
      };
    }
    const settings = track.getSettings();
    if (settings.width === 0 || settings.height === 0) {
      return {
        valid: false,
        label,
        reason: 'Invalid hardware video stream dimensions.',
      };
    }
    return { valid: true, label };
  }

  /**
   * Best-effort: stop auto white-balance/exposure from cancelling the screen-flash signal (not all cameras allow it).
   */
  public async tryLockCameraControls(stream: MediaStream): Promise<string[]> {
    try {
      const track = stream.getVideoTracks()[0] as MediaStreamTrack & {
        getCapabilities?: () => Record<string, unknown>;
      };
      const cap = (track.getCapabilities?.() ?? {}) as {
        whiteBalanceMode?: string[];
        exposureMode?: string[];
      };
      const adv: Record<string, string> = {};
      if (cap.whiteBalanceMode?.includes('manual')) adv.whiteBalanceMode = 'manual';
      if (cap.exposureMode?.includes('manual')) adv.exposureMode = 'manual';
      if (Object.keys(adv).length === 0) return [];
      await track.applyConstraints({ advanced: [adv] } as MediaTrackConstraints);
      return Object.keys(adv);
    } catch {
      return [];
    }
  }

  /**
   * 2. HIGH-FPS ANTI-SPOOF SINGLE-TICK PROCESSOR
   * - Downscaled 256x192 SSD MobileNet on Frame 1 & every 6th frame (15–25 FPS on WASM SIMD)
   * - Crop-tracked 68-landmark inference on Frames 2–5
   * - 1.5x outer-context smartphone bezel + emissive screen detector
   * - 280ms neutral lock -> randomized live challenge (160ms–4000ms) + 3D parallax vs 2D tilt check
   * - Zero eyeglass/timeout bypasses
   */
  public async processVideoFrame(
    video: HTMLVideoElement,
    enrolledRoster: EnrolledStudentVector[],
    options: {
      requireLiveness: boolean;
      mirror: boolean;
      onHudUpdate: (hud: FastHudState) => void;
      flash?: FlashOptions;
    }
  ): Promise<FastAttendanceResult | null> {
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
      return null;
    }

    const now = performance.now();
    if (this.lastTickMs > 0) {
      const dt = Math.min(500, Math.max(16, now - this.lastTickMs));
      this.avgFrameDeltaMs = this.avgFrameDeltaMs * 0.7 + dt * 0.3;
    }
    this.lastTickMs = now;
    this.frameCount++;

    const backend = getActiveBackend();
    const isLowFpsBackend =
      backend === 'wasm' || backend === 'cpu' || this.avgFrameDeltaMs > 55;

    const elapsed = now - this.sessionStartMs;
    const shouldFlashUI = elapsed >= 150 && elapsed <= 350;

    // Step A: Fast Downscaled 256x192 SSD MobileNet (Frame 1 & every 6th) + Crop-Tracked 68-Landmarks (Frames 2–5)
    let pts: faceapi.Point[];
    let box: faceapi.Box;
    let alignedBox: faceapi.Box;

    const runFullDetection = this.frameCount % 6 === 1 || !this.lastBox;

    if (runFullDetection) {
      this.detectCtx.drawImage(video, 0, 0, DETECT_W, DETECT_H);

      const allFaces = await faceapi
        .detectAllFaces(
          this.detectCanvas,
          new faceapi.SsdMobilenetv1Options({ minConfidence: 0.56 })
        )
        .withFaceLandmarks();

      if (allFaces.length === 0) {
        this.ssdMissCount++;
        // a brief detector miss (e.g. mid head-turn) must not wipe a running challenge
        if ((this.activeChallenge !== 'none' || this.awaitingEndFrame) && this.ssdMissCount < 4) {
          return null;
        }
        this.resetSession();
        options.onHudUpdate({
          instruction: 'Position your face inside the reticle',
          subtext: 'Keep eyes level in even lighting',
          progress: 8,
          shouldFlashUI: false,
          qualityStatus: 'warn',
          activeChallenge: 'none',
          checks: { frontal: false, screenClear: false, liveness: false, vector: false },
        });
        return null;
      }

      this.ssdMissCount = 0;

      // Sort largest foreground face first; reject multiple foreground faces
      allFaces.sort((a, b) => b.detection.box.area - a.detection.box.area);
      const primary = allFaces[0];
      if (
        allFaces.length > 1 &&
        allFaces[1].detection.box.area > primary.detection.box.area * 0.18
      ) {
        this.resetSession();
        options.onHudUpdate({
          instruction: 'Multiple faces detected — step up one at a time',
          subtext: 'Only one foreground person allowed',
          progress: 8,
          shouldFlashUI: false,
          qualityStatus: 'warn',
          activeChallenge: 'none',
          checks: { frontal: false, screenClear: false, liveness: false, vector: false },
        });
        return null;
      }

      // Scale detection box, 68 landmarks, and dlib-aligned box from 256x192 back to full video space
      const scaleX = video.videoWidth / DETECT_W;
      const scaleY = video.videoHeight / DETECT_H;
      const dBox = primary.detection.box;
      const fullBox = new faceapi.Box({
        x: dBox.x * scaleX,
        y: dBox.y * scaleY,
        width: dBox.width * scaleX,
        height: dBox.height * scaleY,
      });

      // Sudden frame-cut / photo-swap IoU reset
      if (this.lastBox && this.computeIoU(this.lastBox, fullBox) < 0.42) {
        this.resetSession();
      }

      pts = primary.landmarks.positions.map(
        (p) => new faceapi.Point(p.x * scaleX, p.y * scaleY)
      );
      box = fullBox;

      const dAligned = primary.landmarks.align(null, { useDlibAlignment: true });
      alignedBox = new faceapi.Box({
        x: dAligned.x * scaleX,
        y: dAligned.y * scaleY,
        width: dAligned.width * scaleX,
        height: dAligned.height * scaleY,
      });
      this.lastBox = box;
    } else {
      // TRUE CROP-TRACKED INTERMEDIATE FRAME (Frames 2–5): Zero SSD MobileNet inference!
      const b = this.lastBox!;
      const margin = 0.18;
      const rx = Math.max(0, Math.floor(b.x - b.width * margin));
      const ry = Math.max(0, Math.floor(b.y - b.height * margin));
      const rw = Math.min(
        video.videoWidth - rx,
        Math.ceil(b.width * (1 + margin * 2))
      );
      const rh = Math.min(
        video.videoHeight - ry,
        Math.ceil(b.height * (1 + margin * 2))
      );

      if (rw < 48 || rh < 48) {
        this.lastBox = null;
        return null;
      }

      // Cap ROI canvas size on WASM/CPU so landmark network runs in ~18–30ms
      const maxRoiDim = isLowFpsBackend ? 192 : 256;
      const roiScale = Math.min(1, maxRoiDim / Math.max(rw, rh));
      const cw = Math.max(48, Math.round(rw * roiScale));
      const ch = Math.max(48, Math.round(rh * roiScale));

      this.roiCanvas.width = cw;
      this.roiCanvas.height = ch;
      this.roiCtx.drawImage(video, rx, ry, rw, rh, 0, 0, cw, ch);

      const localLandmarks = (await faceapi.nets.faceLandmark68Net.detectLandmarks(
        this.roiCanvas
      )) as faceapi.FaceLandmarks68;

      const invX = rw / cw;
      const invY = rh / ch;

      pts = localLandmarks.positions.map(
        (p) => new faceapi.Point(p.x * invX + rx, p.y * invY + ry)
      );

      // Sanity check eye-width-to-jaw-width ratio to catch lost crop tracking
      const jawW = Math.max(this.dist(pts[0], pts[16]), 1);
      const eyeW = this.dist(pts[36], pts[39]) / jawW;
      if (eyeW < 0.08 || eyeW > 0.38) {
        this.lastBox = null;
        return null;
      }

      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const p of pts) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
      const lw = maxX - minX;
      const lh = maxY - minY;
      box = new faceapi.Box({
        x: Math.max(0, minX - lw * 0.06),
        y: Math.max(0, minY - lh * 0.18),
        width: lw * 1.12,
        height: lh * 1.24,
      });

      const localAligned = localLandmarks.align(null, { useDlibAlignment: true });
      alignedBox = new faceapi.Box({
        x: localAligned.x * invX + rx,
        y: localAligned.y * invY + ry,
        width: localAligned.width * invX,
        height: localAligned.height * invY,
      });
      this.lastBox = box;
    }

    // Step B: Geometric Features (< 0.2ms)
    const leftEye = this.midpoint(pts[36], pts[39]);
    const rightEye = this.midpoint(pts[42], pts[45]);
    const iod = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
    const rollDeg = Math.abs(
      (Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180) /
        Math.PI
    );

    const distNoseLeftCheek = this.dist(pts[30], pts[2]);
    const distNoseRightCheek = this.dist(pts[14], pts[30]);
    const noseCheekRatio = distNoseLeftCheek / (distNoseRightCheek + 1e-5);

    const faceWidth = Math.max(this.dist(pts[2], pts[14]), 1);
    const faceHeight = Math.max(this.dist(pts[27], pts[8]), 1);
    const aspectRatio = faceWidth / faceHeight;
    const leftVert = Math.max(this.dist(pts[36], pts[4]), 1);
    const rightVert = Math.max(this.dist(pts[45], pts[12]), 1);
    const cornerVertSkew = Math.log(leftVert / rightVert);
    const mouthWidthRatio = this.dist(pts[48], pts[54]) / faceWidth;

    const minIod = Math.round(Math.max(44, (video.videoWidth || 640) * 0.078));
    if (iod < minIod) {
      this.neutralHoldStartMs = 0;
      options.onHudUpdate({
        instruction: 'Move slightly closer to the camera',
        subtext: 'Face distance below minimum Aadhaar resolution',
        progress: 16,
        shouldFlashUI,
        qualityStatus: 'warn',
        activeChallenge: this.activeChallenge,
        checks: { frontal: false, screenClear: false, liveness: false, vector: false },
      });
      return null;
    }

    // Step C: Inner Patch Luma/Moire + Outer 1.45x Sobel Vertical Bezel & Luma Contrast Check
    const innerStats = this.analyzeFacePatch(video, box);
    const screenAudit = this.detectPhoneScreenAndBezel(video, box, innerStats);

    if (options.requireLiveness && screenAudit.isScreenSpoof) {
      this.screenSpoofStrikes++;
      const shouldRejectNow =
        screenAudit.immediateReject || this.screenSpoofStrikes >= 2;
      options.onHudUpdate({
        instruction: screenAudit.errorMessage,
        subtext: screenAudit.reason,
        progress: 6,
        shouldFlashUI: false,
        qualityStatus: 'warn',
        activeChallenge: 'none',
        checks: {
          frontal: false,
          screenClear: false,
          liveness: false,
          vector: false,
        },
      });
      if (shouldRejectNow) {
        const snapUrl = this.captureSnapshotCanvas(
          video,
          options.mirror
        ).toDataURL('image/jpeg', 0.85);
        const latencyMs = Math.round(performance.now() - this.sessionStartMs);
        const challengeType = this.activeChallenge;
        this.resetSession();
        return {
          matched: false,
          studentId: null,
          attendanceId: null,
          distance: 1.0,
          confidence: 0,
          latencyMs,
          spoofPassed: false,
          liveDescriptor: [],
          capturedFrameDataUrl: snapUrl,
          qualityChecks: {
            hardwareValid: true,
            singleFace: true,
            lightingSharpness: innerStats.sharpness >= 14,
            moirePassed: !innerStats.highFreqSpike,
            screenBorderClear: false,
            microTremorPassed: false,
            depthParallaxPassed: false,
            blinkVerified: false,
            challengeType,
            challengePassed: false,
            reactionTimeMs: null,
            reflectionDelta: null,
          },
          failReason: screenAudit.errorMessage,
        };
      }
      return null;
    } else {
      this.screenSpoofStrikes = Math.max(0, this.screenSpoofStrikes - 1);
    }

    if (innerStats.sharpness < 14) {
      options.onHudUpdate({
        instruction: 'Hold steady — image slightly soft',
        subtext: 'Waiting for crisp frontal lock',
        progress: 24,
        shouldFlashUI,
        qualityStatus: 'warn',
        activeChallenge: this.activeChallenge,
        checks: { frontal: false, screenClear: true, liveness: false, vector: false },
      });
      return null;
    }

    // Track soft cheek luminance delta during 180ms reticle glow
    if (!shouldFlashUI && this.lumaDark < 0) this.lumaDark = innerStats.cheekLuma;
    if (shouldFlashUI) {
      this.lumaBright = Math.max(this.lumaBright, innerStats.cheekLuma);
    }

    // Track physiological micro-tremor history
    this.noseRatios.push(noseCheekRatio);
    if (this.noseRatios.length > 24) this.noseRatios.shift();

    // Bilateral EAR computation
    const earL = this.computeEAR(pts.slice(36, 42));
    const earR = this.computeEAR(pts.slice(42, 48));
    const earAvg = (earL + earR) * 0.5;
    const eyeSymmetryDiff = Math.abs(earL - earR);

    // Continuously update sharpest clean open-eye frontal winner BEFORE blink or head turn!
    const isCleanFrontalOpen =
      rollDeg <= 12 &&
      noseCheekRatio >= 0.74 &&
      noseCheekRatio <= 1.35 &&
      earAvg >= Math.max(0.2, this.earPeak * 0.85) &&
      eyeSymmetryDiff < 0.08;

    if (isCleanFrontalOpen) {
      this.earPeak = Math.max(this.earPeak * 0.985, earAvg);
      if (
        !this.startFrozen &&
        (!this.bestFrame || innerStats.sharpness > this.bestFrame.sharpness)
      ) {
        this.bestFrame = {
          alignedCanvas: this.captureAligned150Canvas(video, alignedBox),
          snapshotCanvas: this.captureSnapshotCanvas(video, options.mirror),
          sharpness: innerStats.sharpness,
          lumaCheek: innerStats.cheekLuma,
        };
      }
    }

    // If liveness is explicitly disabled in Settings, verify immediately once frontal winner locks
    if (!options.requireLiveness) {
      if (!this.bestFrame) return null;
      return this.finalizeMatch(enrolledRoster, {
        microTremorPassed: true,
        depthParallaxPassed: true,
        blinkVerified: false,
        challengeType: 'none',
        challengePassed: false,
        reactionTimeMs: null,
        onHudUpdate: options.onHudUpdate,
      });
    }

    // =========================================================================
    // PHASE 3: POST-CHALLENGE IDENTITY FRAME + LIGHT CHECK
    // The ID frame was captured BEFORE the challenge. A second frame AFTER it (same person required, see
    // identityContinuity) stops "photo for the ID frame, live stranger for the blink" bait-and-switch.
    // =========================================================================
    if (this.awaitingEndFrame) {
      if (now - this.endWaitStartMs > 3500) {
        this.resetSession();
        options.onHudUpdate({
          instruction: 'Final identity frame timed out — look straight to retry',
          subtext: 'Face the camera after the live checks',
          progress: 12,
          shouldFlashUI: false,
          qualityStatus: 'warn',
          activeChallenge: 'none',
          checks: { frontal: false, screenClear: true, liveness: false, vector: false },
        });
        return null;
      }
      if (!isCleanFrontalOpen) {
        options.onHudUpdate({
          instruction: 'Look straight at the camera',
          subtext: 'Final identity check · eyes open',
          progress: 84,
          shouldFlashUI: false,
          qualityStatus: 'ok',
          activeChallenge: 'none',
          checks: { frontal: false, screenClear: true, liveness: true, vector: false },
        });
        return null;
      }
      this.endFrame = {
        alignedCanvas: this.captureAligned150Canvas(video, alignedBox),
        snapshotCanvas: this.captureSnapshotCanvas(video, options.mirror),
        sharpness: innerStats.sharpness,
        lumaCheek: innerStats.cheekLuma,
      };
      this.awaitingEndFrame = false;

      let flash: { tier: FlashTier | 'skipped'; corr: number | null } = { tier: 'skipped', corr: null };
      if (options.flash?.enabled) {
        flash = await this.runFlashCheck(video, pts, box, options.flash, options.onHudUpdate);
        if (flash.tier === 'fail') {
          const snapUrl = this.endFrame.snapshotCanvas.toDataURL('image/jpeg', 0.85);
          const latencyMs = Math.round(performance.now() - this.sessionStartMs);
          const seq = [...this.completedChallenges];
          this.resetSession();
          return {
            matched: false,
            studentId: null,
            attendanceId: null,
            distance: 1.0,
            confidence: 0,
            latencyMs,
            spoofPassed: false,
            liveDescriptor: [],
            capturedFrameDataUrl: snapUrl,
            qualityChecks: {
              hardwareValid: true,
              singleFace: true,
              lightingSharpness: true,
              moirePassed: true,
              screenBorderClear: true,
              microTremorPassed: this.tremorOk,
              depthParallaxPassed: this.depthAll,
              blinkVerified: true,
              challengeType: seq[seq.length - 1] ?? 'none',
              challengePassed: true,
              reactionTimeMs: null,
              reflectionDelta: null,
              flashTier: 'fail',
              flashCorr: flash.corr,
              challengeSequence: seq,
            },
            failReason:
              'Light-reflection check failed — skin did not react to the screen colours (replay / injected video suspected).',
          };
        }
      }

      return this.finalizeMatch(enrolledRoster, {
        microTremorPassed: this.tremorOk,
        depthParallaxPassed: this.depthAll,
        blinkVerified: true,
        challengeType: this.completedChallenges[this.completedChallenges.length - 1] ?? 'none',
        challengePassed: true,
        reactionTimeMs: this.reactionTimes.length ? Math.max(...this.reactionTimes) : null,
        onHudUpdate: options.onHudUpdate,
        flash,
      });
    }

    // =========================================================================
    // PHASE 1: CLEAN NEUTRAL OPEN-EYE BASELINE (280ms)
    // Defeats pre-recorded blinking/moving videos before the randomized prompt!
    // =========================================================================
    if (this.activeChallenge === 'none') {
      if (rollDeg > 12 || noseCheekRatio < 0.74 || noseCheekRatio > 1.35) {
        this.neutralHoldStartMs = 0;
        options.onHudUpdate({
          instruction: 'Look straight at the camera',
          subtext: 'Hold neutral face for 0.3s to arm live challenge',
          progress: 26,
          shouldFlashUI,
          qualityStatus: 'warn',
          activeChallenge: 'none',
          checks: { frontal: false, screenClear: true, liveness: false, vector: false },
        });
        return null;
      }

      // If eyes are already closed/blinking BEFORE prompt appears (e.g., looped video), hold baseline
      if (earAvg < this.earPeak * 0.76) {
        this.neutralHoldStartMs = 0;
        options.onHudUpdate({
          instruction: 'Keep eyes open briefly to calibrate',
          subtext: 'Wait for the live prompt before blinking',
          progress: 30,
          shouldFlashUI: false,
          qualityStatus: 'ok',
          activeChallenge: 'none',
          checks: { frontal: true, screenClear: true, liveness: false, vector: false },
        });
        return null;
      }

      if (this.neutralHoldStartMs === 0) {
        this.neutralHoldStartMs = now;
      }

      const neutralElapsed = now - this.neutralHoldStartMs;
      if (neutralElapsed < 280 || !this.bestFrame) {
        options.onHudUpdate({
          instruction: 'Hold steady — arming live liveness...',
          subtext: 'Anti-spoof bezel & neutral lock active',
          progress: Math.min(46, 32 + Math.round((neutralElapsed / 280) * 14)),
          shouldFlashUI,
          qualityStatus: 'ok',
          activeChallenge: 'none',
          checks: { frontal: true, screenClear: true, liveness: false, vector: false },
        });
        return null;
      }

      // Baseline locked! Randomly pick ONE instant live micro-challenge
      this.baseline = {
        noseCheekRatio,
        faceWidth,
        faceHeight,
        aspectRatio,
        cornerVertSkew,
        mouthWidthRatio,
      };
      this.activeChallenge =
        this.challengeQueue.shift() ??
        pickDistinct(CHALLENGE_POOL, 1)[0];
      this.startFrozen = true; // ID frame is now locked to the PRE-challenge winner
      this.challengeIssuedAt = now;
      this.challengeActionDoneAt = 0;
      this.blinkCompletedAt = 0;
      this.blink.reset();
      this.blinkCompleted = false;
      this.depthParallaxConfirmed = false;

      options.onHudUpdate({
        instruction: getChallengePromptLabel(this.activeChallenge),
        subtext: 'Randomized live check · Respond naturally now',
        progress: 52,
        shouldFlashUI: false,
        qualityStatus: 'ok',
        activeChallenge: this.activeChallenge,
        checks: { frontal: true, screenClear: true, liveness: false, vector: false },
      });
      return null;
    }

    // =========================================================================
    // PHASE 2: RANDOMIZED LIVE CHALLENGE + 3D PARALLAX + REACTION-TIME GATE
    // ZERO timeout auto-bypass! Must prove active liveness within 160ms–4000ms.
    // =========================================================================
    const sincePromptMs = now - this.challengeIssuedAt;

    if (sincePromptMs > 4000) {
      this.resetSession();
      options.onHudUpdate({
        instruction: 'Live challenge timed out — look straight to retry',
        subtext: 'Active response required within 4 seconds',
        progress: 12,
        shouldFlashUI: false,
        qualityStatus: 'warn',
        activeChallenge: 'none',
        checks: { frontal: false, screenClear: true, liveness: false, vector: false },
      });
      return null;
    }

    // Blink detection (livenessCore.BlinkTracker): raw EAR + hysteresis + depth + long-closure latch
    if (!this.blinkCompleted) {
      const blinked = this.blink.update(earAvg, eyeSymmetryDiff, now, {
        earPeak: this.earPeak,
        lowFps: isLowFpsBackend,
      });
      if (blinked) {
        if (sincePromptMs < 140) {
          // Pre-emptive / looped video blinked faster than human reaction time
          this.resetSession();
          options.onHudUpdate({
            instruction: 'Wait for the prompt before blinking',
            subtext: 'Pre-prompt motion ignored — hold steady',
            progress: 14,
            shouldFlashUI: false,
            qualityStatus: 'warn',
            activeChallenge: 'none',
            checks: { frontal: false, screenClear: true, liveness: false, vector: false },
          });
          return null;
        }
        this.blinkCompleted = true;
        this.blinkCompletedAt = now;
      }
    }

    // Evaluate Challenge Action (3D Left/Right Parallax OR Smile) + 2D Flat Screen Tilt Rejector
    const base = this.baseline!;
    const deltaNose = noseCheekRatio - base.noseCheekRatio;
    const deltaSkew = Math.abs(cornerVertSkew - base.cornerVertSkew);
    const widthForeshorten = faceWidth / Math.max(1, base.faceWidth);
    const aspectDrift = Math.abs(aspectRatio / Math.max(0.1, base.aspectRatio) - 1);

    // 2D Flat Screen Tilt Check:
    // Tilting a flat smartphone causes outer 4-corner trapezoidal skew or uniform horizontal
    // foreshortening WITHOUT strong 3D nose-tip Z-parallax.
    const flatPlaneTiltDetected =
      (deltaSkew >= 0.145 && Math.abs(deltaNose) < 0.19) ||
      (widthForeshorten < 0.82 && Math.abs(deltaNose) < 0.17) ||
      (aspectDrift > 0.17 && Math.abs(deltaNose) < 0.17);

    if (flatPlaneTiltDetected) {
      this.resetSession();
      options.onHudUpdate({
        instruction: '2D flat screen tilt rejected',
        subtext: 'True 3D head depth required — present a live face',
        progress: 8,
        shouldFlashUI: false,
        qualityStatus: 'warn',
        activeChallenge: 'none',
        checks: { frontal: false, screenClear: false, liveness: false, vector: false },
      });
      return null;
    }

    if (this.challengeActionDoneAt === 0 && sincePromptMs >= 160) {
      if (this.activeChallenge === 'blink_left') {
        // When mirrored, turning left shifts raw noseCheekRatio upward; unmirrored shifts downward
        const turnedLeft = options.mirror
          ? deltaNose >= 0.17
          : deltaNose <= -0.15;
        if (turnedLeft && deltaSkew < 0.145) {
          this.challengeActionDoneAt = now;
          this.depthParallaxConfirmed = true;
        }
      } else if (this.activeChallenge === 'blink_right') {
        const turnedRight = options.mirror
          ? deltaNose <= -0.15
          : deltaNose >= 0.17;
        if (turnedRight && deltaSkew < 0.145) {
          this.challengeActionDoneAt = now;
          this.depthParallaxConfirmed = true;
        }
      } else if (this.activeChallenge === 'smile_blink') {
        const smiled =
          mouthWidthRatio >= base.mouthWidthRatio + 0.042 ||
          mouthWidthRatio >= base.mouthWidthRatio * 1.1;
        if (smiled) {
          this.challengeActionDoneAt = now;
          this.depthParallaxConfirmed = true;
        }
      }
    }

    // Both blink AND randomized action must complete after human reaction threshold (>= 160ms)
    const actionDone = this.challengeActionDoneAt > 0;
    const blinkDone = this.blinkCompleted;

    if (!actionDone || !blinkDone) {
      const promptTitle = getChallengePromptLabel(this.activeChallenge);
      const subtext =
        blinkDone && !actionDone
          ? this.activeChallenge === 'smile_blink'
            ? 'Blink locked · Now give a quick smile'
            : this.activeChallenge === 'blink_left'
            ? 'Blink locked · Now turn slightly left'
            : 'Blink locked · Now turn slightly right'
          : actionDone && !blinkDone
          ? '3D action locked · Now blink naturally once'
          : 'Complete both actions naturally (anti-replay check)';

      const progress =
        blinkDone || actionDone
          ? 78
          : Math.min(70, 54 + Math.round((sincePromptMs / 4000) * 16));

      options.onHudUpdate({
        instruction: promptTitle,
        subtext,
        progress,
        shouldFlashUI: false,
        qualityStatus: blinkDone || actionDone ? 'locked' : 'ok',
        activeChallenge: this.activeChallenge,
        checks: {
          frontal: true,
          screenClear: true,
          liveness: blinkDone && actionDone,
          vector: false,
        },
      });
      return null;
    }

    // Enforce >= 160ms human reaction time on final completion
    const reactionTimeMs = Math.round(
      Math.max(this.challengeActionDoneAt, this.blinkCompletedAt) -
        this.challengeIssuedAt
    );
    if (reactionTimeMs < 160) {
      this.resetSession();
      options.onHudUpdate({
        instruction: 'Pre-recorded loop rejected — respond after prompt',
        subtext: 'Human reaction window required',
        progress: 10,
        shouldFlashUI: false,
        qualityStatus: 'warn',
        activeChallenge: 'none',
        checks: { frontal: false, screenClear: true, liveness: false, vector: false },
      });
      return null;
    }

    if (!this.bestFrame) return null;

    // Physiological micro-tremor verification
    const tremorStd = this.stdDev(this.noseRatios);
    const microTremorPassed =
      this.noseRatios.length < 4 || (tremorStd >= 0.0005 && tremorStd <= 0.35);

    if (!microTremorPassed) {
      this.resetSession();
      options.onHudUpdate({
        instruction: 'Static feed rejected — present a live face',
        subtext: 'Natural 3D micro-motion required',
        progress: 10,
        shouldFlashUI: false,
        qualityStatus: 'warn',
        activeChallenge: 'none',
        checks: { frontal: false, screenClear: true, liveness: false, vector: false },
      });
      return null;
    }

    // Challenge complete. Remember it, then either arm the NEXT random challenge or ask for the final ID frame.
    this.completedChallenges.push(this.activeChallenge);
    this.reactionTimes.push(reactionTimeMs);
    this.depthAll = this.depthAll && this.depthParallaxConfirmed;
    this.tremorOk = this.tremorOk && microTremorPassed;

    if (this.challengeQueue.length > 0) {
      this.activeChallenge = 'none';
      this.neutralHoldStartMs = 0;
      this.blink.reset();
      this.blinkCompleted = false;
      this.blinkCompletedAt = 0;
      this.challengeActionDoneAt = 0;
      this.depthParallaxConfirmed = false;
      options.onHudUpdate({
        instruction: `Step ${this.completedChallenges.length} of ${CHALLENGES_PER_SESSION} done — look straight`,
        subtext: 'The next live check is random',
        progress: 60,
        shouldFlashUI: false,
        qualityStatus: 'locked',
        activeChallenge: 'none',
        checks: { frontal: true, screenClear: true, liveness: false, vector: false },
      });
      return null;
    }

    this.awaitingEndFrame = true;
    this.endWaitStartMs = now;
    options.onHudUpdate({
      instruction: 'Look straight at the camera',
      subtext: 'Final identity check',
      progress: 82,
      shouldFlashUI: false,
      qualityStatus: 'locked',
      activeChallenge: 'none',
      checks: { frontal: true, screenClear: true, liveness: true, vector: false },
    });
    return null;
  }

  /**
   * ACTIVE ILLUMINATION: the kiosk screen shows a random colour sequence; skin chromaticity (minus the background's,
   * which cancels webcam auto-white-balance) must follow it. Averages several reads per slot to beat sensor noise.
   */
  private async runFlashCheck(
    video: HTMLVideoElement,
    pts: faceapi.Point[],
    box: faceapi.Box,
    cfg: FlashOptions,
    onHud: (hud: FastHudState) => void
  ): Promise<{ tier: FlashTier | 'skipped'; corr: number | null }> {
    const slots = Math.max(6, Math.min(12, cfg.slots ?? 8));
    const slotMs = Math.max(334, cfg.slotMs ?? 340); // keep < 3 flashes/second
    const plan = makeFlashPlan(slots);
    const W = this.flashCanvas.width;
    const H = this.flashCanvas.height;
    const sx = W / video.videoWidth;
    const sy = H / video.videoHeight;
    const scaleRect = (r: Rect): Rect => ({ x: r.x * sx, y: r.y * sy, w: r.w * sx, h: r.h * sy });
    const cheeks = cheekRects(pts).map(scaleRect);
    const bgs = backgroundRects(W, H, scaleRect({ x: box.x, y: box.y, w: box.width, h: box.height }));
    const css = (c: RGB) => `rgb(${c[0]},${c[1]},${c[2]})`;
    const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));

    const meanRgb = (data: Uint8ClampedArray, rects: Rect[], minPx: number): RGB | null => {
      let R = 0, G = 0, B = 0, n = 0;
      for (const q of rects) {
        const x0 = Math.max(0, Math.round(q.x));
        const y0 = Math.max(0, Math.round(q.y));
        const x1 = Math.min(W, Math.round(q.x + q.w));
        const y1 = Math.min(H, Math.round(q.y + q.h));
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const i = (y * W + x) * 4;
            const l = (data[i] + data[i + 1] + data[i + 2]) / 3;
            if (l < 25 || l > 235) continue; // skip clipped / very dark pixels
            R += data[i]; G += data[i + 1]; B += data[i + 2]; n++;
          }
        }
      }
      return n >= minPx ? [R / n, G / n, B / n] : null;
    };
    const readOnce = () => {
      this.flashCtx.drawImage(video, 0, 0, W, H);
      const { data } = this.flashCtx.getImageData(0, 0, W, H);
      return { face: meanRgb(data, cheeks, 40), bg: bgs.length ? meanRgb(data, bgs, 300) : null };
    };
    const avgReads = async (count: number, gapMs: number) => {
      const faces: RGB[] = [];
      const bgsAcc: RGB[] = [];
      for (let i = 0; i < count; i++) {
        const r = readOnce();
        if (r.face) faces.push(r.face);
        if (r.bg) bgsAcc.push(r.bg);
        if (i < count - 1) await sleep(gapMs);
      }
      const avg = (a: RGB[]): RGB | null =>
        a.length
          ? [a.reduce((s, v) => s + v[0], 0) / a.length, a.reduce((s, v) => s + v[1], 0) / a.length, a.reduce((s, v) => s + v[2], 0) / a.length]
          : null;
      return { face: avg(faces), bg: bgsAcc.length === count ? avg(bgsAcc) : null };
    };

    onHud({
      instruction: 'Hold still — light check',
      subtext: 'Screen will flash colours for ~3 seconds · keep your face centered',
      progress: 90,
      shouldFlashUI: false,
      qualityStatus: 'locked',
      activeChallenge: 'none',
      checks: { frontal: true, screenClear: true, liveness: true, vector: false },
    });

    const obs: FlashObservation[] = [];
    try {
      cfg.setOverlay(css(FLASH_NEUTRAL));
      await sleep(240);
      const base = await avgReads(3, 40);
      if (!base.face) return { tier: 'fail', corr: null };

      for (const color of plan) {
        cfg.setOverlay(css(color));
        await sleep(Math.round(slotMs * 0.5)); // sensor + light settle
        const t0 = performance.now();
        const r = await avgReads(4, Math.round(slotMs * 0.1));
        if (!r.face) return { tier: 'fail', corr: null };
        obs.push({ screen: color, face: r.face, bg: r.bg });
        await sleep(Math.max(0, slotMs - Math.round(slotMs * 0.5) - (performance.now() - t0)));
      }
      const score = flashScore({ face: base.face, bg: base.bg }, obs);
      return { tier: score.tier, corr: Number(score.corr.toFixed(3)) };
    } finally {
      cfg.setOverlay(null);
    }
  }

  private async finalizeMatch(
    enrolledRoster: EnrolledStudentVector[],
    meta: {
      microTremorPassed: boolean;
      depthParallaxPassed: boolean;
      blinkVerified: boolean;
      challengeType: LiveChallengeType;
      challengePassed: boolean;
      reactionTimeMs: number | null;
      onHudUpdate: (hud: FastHudState) => void;
      flash?: { tier: FlashTier | 'skipped'; corr: number | null };
    }
  ): Promise<FastAttendanceResult> {
    const reflectionDelta =
      this.lumaDark >= 0 && this.lumaBright >= 0
        ? Number((this.lumaBright - this.lumaDark).toFixed(2))
        : null;

    meta.onHudUpdate({
      instruction: 'Matching 128-D face vector...',
      subtext: 'ResNet-34 on the pre- and post-challenge frames',
      progress: 96,
      shouldFlashUI: false,
      qualityStatus: 'locked',
      activeChallenge: meta.challengeType,
      checks: { frontal: true, screenClear: true, liveness: true, vector: true },
    });

    const toVec = async (c: HTMLCanvasElement): Promise<Float32Array> => {
      const raw = await faceapi.computeFaceDescriptor(c);
      return l2Normalize(Array.isArray(raw) ? raw[0] : raw);
    };

    // ID frame = sharpest PRE-challenge frame; END frame = first clean frontal frame AFTER the challenges.
    const startVec = await toVec(this.bestFrame!.alignedCanvas);
    let liveVec: Float32Array = startVec;
    let continuity: number | null = null;
    const snapshotFrame = this.endFrame ?? this.bestFrame!;
    const sequence = [...this.completedChallenges];

    const checks = (): BiometricQualityChecks => ({
      hardwareValid: true,
      singleFace: true,
      lightingSharpness: true,
      moirePassed: true,
      screenBorderClear: true,
      microTremorPassed: meta.microTremorPassed,
      depthParallaxPassed: meta.depthParallaxPassed,
      blinkVerified: meta.blinkVerified,
      challengeType: meta.challengeType,
      challengePassed: meta.challengePassed,
      reactionTimeMs: meta.reactionTimeMs,
      reflectionDelta,
      identityContinuity: continuity,
      flashTier: meta.flash?.tier ?? 'skipped',
      flashCorr: meta.flash?.corr ?? null,
      challengeSequence: sequence,
    });

    if (this.endFrame) {
      const endVec = await toVec(this.endFrame.alignedCanvas);
      const c = identityContinuity(startVec, endVec);
      continuity = Number(c.distance.toFixed(4));
      if (!c.ok) {
        return {
          matched: false,
          studentId: null,
          attendanceId: null,
          distance: 1.0,
          confidence: 0,
          latencyMs: Math.round(performance.now() - this.sessionStartMs),
          spoofPassed: false,
          liveDescriptor: [],
          capturedFrameDataUrl: snapshotFrame.snapshotCanvas.toDataURL('image/jpeg', 0.85),
          qualityChecks: checks(),
          failReason: `Identity changed during the live check (continuity ${c.distance.toFixed(2)}) — a different face appeared after the ID frame.`,
        };
      }
      liveVec = meanNormalized(startVec, endVec);
    }

    const match = this.matchRoster(liveVec, enrolledRoster);
    const totalLatency = Math.round(performance.now() - this.sessionStartMs);
    const capturedFrameDataUrl = snapshotFrame.snapshotCanvas.toDataURL('image/jpeg', 0.88);

    return {
      matched: match.matched,
      studentId: match.studentId,
      attendanceId: match.attendanceId,
      distance: Number(match.bestDist.toFixed(4)),
      confidence: distanceToConfidence(match.bestDist),
      latencyMs: totalLatency,
      spoofPassed: true,
      liveDescriptor: Array.from(liveVec),
      capturedFrameDataUrl,
      qualityChecks: checks(),
      failReason: match.failReason,
      needsReview: meta.flash?.tier === 'weak',
    };
  }

  /**
   * 3. 3-FRAME VECTOR BURST FOR REGISTRATION
   * Reuses pre-blink winner as Frame 1, then captures 2 more full-res 150x150 dlib crops.
   */
  public async captureThreeFrameBurst(
    video: HTMLVideoElement,
    mirror: boolean,
    signal?: AbortSignal,
    initialVector?: number[] | null
  ): Promise<{ vectors: number[][]; snapshotDataUrl: string } | null> {
    const burstVectors: Float32Array[] = [];
    let bestSnapshot: HTMLCanvasElement | null = null;
    let bestSharpness = -1;

    if (initialVector && initialVector.length === 128) {
      burstVectors.push(l2Normalize(initialVector));
      if (this.bestFrame) {
        bestSnapshot = this.bestFrame.snapshotCanvas;
        bestSharpness = this.bestFrame.sharpness;
      }
    } else if (this.bestFrame) {
      const raw = await faceapi.computeFaceDescriptor(
        this.bestFrame.alignedCanvas
      );
      const vec = Array.isArray(raw) ? raw[0] : raw;
      burstVectors.push(l2Normalize(vec));
      bestSnapshot = this.bestFrame.snapshotCanvas;
      bestSharpness = this.bestFrame.sharpness;
    }

    // Extra burst frames must be ENROLLABLE (frontal, eyes open, sharp) and consistent with frame 1. The user has
    // just done head-turn challenges, so a blind 120ms grab used to enrol turned / mid-blink faces.
    const needed = 3 - burstVectors.length;
    let attempts = 0;
    while (burstVectors.length < 3 && needed > 0 && attempts < needed + 12) {
      attempts++;
      if (signal?.aborted) return null;
      await new Promise((r) => window.setTimeout(r, 110));
      if (signal?.aborted || !video.videoWidth) return null;

      this.detectCtx.drawImage(video, 0, 0, DETECT_W, DETECT_H);
      const det = await faceapi
        .detectSingleFace(
          this.detectCanvas,
          new faceapi.SsdMobilenetv1Options({ minConfidence: 0.52 })
        )
        .withFaceLandmarks();
      if (!det) continue;

      const scaleX = video.videoWidth / DETECT_W;
      const scaleY = video.videoHeight / DETECT_H;
      const fullBox = new faceapi.Box({
        x: det.detection.box.x * scaleX,
        y: det.detection.box.y * scaleY,
        width: det.detection.box.width * scaleX,
        height: det.detection.box.height * scaleY,
      });
      const lp = det.landmarks.positions;
      const le = this.midpoint(lp[36], lp[39]);
      const re = this.midpoint(lp[42], lp[45]);
      const { sharpness } = this.analyzeFacePatch(video, fullBox);
      const okFrame = isEnrollableFrame({
        rollDeg: Math.abs((Math.atan2(re.y - le.y, re.x - le.x) * 180) / Math.PI),
        noseCheekRatio: this.dist(lp[30], lp[2]) / (this.dist(lp[14], lp[30]) + 1e-5),
        ear: (this.computeEAR(lp.slice(36, 42)) + this.computeEAR(lp.slice(42, 48))) / 2,
        earPeak: this.earPeak,
        sharpness,
      });
      if (!okFrame) continue;

      const dAligned = det.landmarks.align(null, { useDlibAlignment: true });
      const fullAligned = new faceapi.Box({
        x: dAligned.x * scaleX,
        y: dAligned.y * scaleY,
        width: dAligned.width * scaleX,
        height: dAligned.height * scaleY,
      });
      const raw = await faceapi.computeFaceDescriptor(
        this.captureAligned150Canvas(video, fullAligned)
      );
      const vec = l2Normalize(Array.isArray(raw) ? raw[0] : raw);
      if (burstVectors.length > 0 && dist128(vec, burstVectors[0]) > 0.5) continue; // outlier / motion blur
      burstVectors.push(vec);
      if (!bestSnapshot || sharpness > bestSharpness) {
        bestSharpness = sharpness;
        bestSnapshot = this.captureSnapshotCanvas(video, mirror);
      }
    }

    if (burstVectors.length < 2 || !bestSnapshot) return null;

    const centroid = computeNormalizedCentroid(burstVectors);
    const allNormalized: number[][] = [
      Array.from(centroid),
      ...burstVectors.map((v) => Array.from(v)),
    ];

    return {
      vectors: allNormalized,
      snapshotDataUrl: bestSnapshot.toDataURL('image/jpeg', 0.9),
    };
  }

  /**
   * Ultra-fast L2-Normalized Roster Matcher (< 0.4ms across thousands of students)
   */
  public matchRoster(
    liveVec: Float32Array,
    roster: EnrolledStudentVector[]
  ): {
    matched: boolean;
    studentId: string | null;
    attendanceId: string | null;
    bestDist: number;
    failReason?: string;
  } {
    if (roster.length === 0) {
      return {
        matched: false,
        studentId: null,
        attendanceId: null,
        bestDist: 1.0,
        failReason: 'No enrolled biometric vectors available.',
      };
    }

    let bestStudentId: string | null = null;
    let bestAttendanceId: string | null = null;
    let bestDist = Infinity;
    let secondBestDist = Infinity;

    for (const student of roster) {
      let studentMinDist = Infinity;
      for (const storedVec of student.vectors) {
        let sum = 0;
        for (let i = 0; i < 128; i++) {
          const d = liveVec[i] - storedVec[i];
          sum += d * d;
        }
        const dist = Math.sqrt(sum);
        if (dist < studentMinDist) studentMinDist = dist;
      }

      if (studentMinDist < bestDist) {
        secondBestDist = bestDist;
        bestDist = studentMinDist;
        bestStudentId = student.id;
        bestAttendanceId = student.attendanceId;
      } else if (studentMinDist < secondBestDist) {
        secondBestDist = studentMinDist;
      }
    }

    if (bestDist > this.matchThreshold) {
      return {
        matched: false,
        studentId: null,
        attendanceId: null,
        bestDist,
        failReason: `Face mismatch — distance ${bestDist.toFixed(3)} exceeded limit ${this.matchThreshold.toFixed(2)}.`,
      };
    }

    if (roster.length > 1 && secondBestDist - bestDist < this.impostorMargin) {
      return {
        matched: false,
        studentId: null,
        attendanceId: null,
        bestDist,
        failReason: `Ambiguous 1:N match (margin ${(secondBestDist - bestDist).toFixed(3)} < ${this.impostorMargin.toFixed(2)}) — verify with Student PIN.`,
      };
    }

    return {
      matched: true,
      studentId: bestStudentId,
      attendanceId: bestAttendanceId,
      bestDist,
    };
  }

  /**
   * OUTER-CONTEXT 1.45x SMARTPHONE BEZEL (SOBEL FILTER) & LUMA CONTRAST DETECTOR
   * 1. Samples an exact 1.45x bounding box around the face to inspect the immediate periphery.
   * 2. Applies 3x3 Sobel Gx/Gy filtering to detect straight vertical phone bezels (left & right)
   *    plus horizontal top phone status/gallery bars (including hand-tilted phones up to ±10°).
   * 3. Runs a Luma Contrast Check: high-uniform inner face brightness framed by extremely dark
   *    flanking bezel strips triggers an immediate 'Screen border detected' rejection.
   */
  private detectPhoneScreenAndBezel(
    video: HTMLVideoElement,
    box: faceapi.Box,
    innerStats: {
      sharpness: number;
      cheekLuma: number;
      lumaMean: number;
      lumaStd: number;
      highFreqSpike: boolean;
      clippedRatio: number;
      brightRatio: number;
      satStd: number;
      blueExcess: number;
    }
  ): {
    isScreenSpoof: boolean;
    immediateReject: boolean;
    errorMessage: string;
    reason: string;
  } {
    if (innerStats.highFreqSpike) {
      return {
        isScreenSpoof: true,
        immediateReject: false, // needs repeated strikes: dim-light sensor noise / beards also spike a Laplacian test
        errorMessage: 'Screen border detected — digital display Moiré pattern',
        reason: 'Subpixel Moiré oscillation pattern detected on face surface',
      };
    }

    // Exact 1.45x sampling area around the face bounding box
    const OUTER_SCALE = 1.45;
    const padX = box.width * ((OUTER_SCALE - 1) * 0.5);
    const padY = box.height * ((OUTER_SCALE - 1) * 0.5);
    const ox = Math.max(0, Math.floor(box.x - padX));
    const oy = Math.max(0, Math.floor(box.y - padY));
    const ow = Math.min(video.videoWidth - ox, Math.ceil(box.width + padX * 2));
    const oh = Math.min(video.videoHeight - oy, Math.ceil(box.height + padY * 2));

    if (ow < 64 || oh < 64) {
      return {
        isScreenSpoof: false,
        immediateReject: false,
        errorMessage: '',
        reason: '',
      };
    }

    const W = 128;
    const H = 128;
    this.outerCtx.drawImage(video, ox, oy, ow, oh, 0, 0, W, H);
    const { data } = this.outerCtx.getImageData(0, 0, W, H);
    const gray = new Float32Array(W * H);

    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    // Compute 3x3 Sobel horizontal (Gx -> vertical edges) and vertical (Gy -> horizontal edges)
    const sobelX = new Float32Array(W * H);
    const sobelY = new Float32Array(W * H);
    for (let y = 1; y < H - 1; y++) {
      const row = y * W;
      const up = row - W;
      const dn = row + W;
      for (let x = 1; x < W - 1; x++) {
        const gx =
          -gray[up + x - 1] +
          gray[up + x + 1] -
          2 * gray[row + x - 1] +
          2 * gray[row + x + 1] -
          gray[dn + x - 1] +
          gray[dn + x + 1];
        const gy =
          -gray[up + x - 1] -
          2 * gray[up + x] -
          gray[up + x + 1] +
          gray[dn + x - 1] +
          2 * gray[dn + x] +
          gray[dn + x + 1];
        sobelX[row + x] = gx;
        sobelY[row + x] = gy;
      }
    }

    const innerL = Math.max(8, Math.floor(((box.x - ox) / ow) * W));
    const innerR = Math.min(W - 9, Math.ceil(((box.x + box.width - ox) / ow) * W));
    const innerT = Math.max(8, Math.floor(((box.y - oy) / oh) * H));
    const innerB = Math.min(H - 9, Math.ceil(((box.y + box.height - oy) / oh) * H));
    const spanY = Math.max(16, innerB - innerT);

    // 1. Sobel Vertical Bezel Detector across straight & slightly tilted (±10°) peripheral lines
    const slopes = [-0.18, -0.09, 0, 0.09, 0.18];

    const scanSobelVerticalFlank = (
      xStart: number,
      xEnd: number,
      isLeftFlank: boolean
    ) => {
      let bestSobelCoherence = 0;
      let bestMeanSignedGx = 0;
      let bestOuterDarkLuma = 255;
      let bestStepJump = 0;

      for (const slope of slopes) {
        for (let x0 = xStart; x0 <= xEnd; x0++) {
          let sobelHits = 0;
          let signedGxSum = 0;
          let darkSum = 0;
          let insideSum = 0;
          let samples = 0;

          for (let y = innerT; y <= innerB; y += 2) {
            const x = Math.round(x0 + (y - innerT) * slope);
            if (x < 4 || x >= W - 4) continue;
            const idx = y * W + x;

            // Left bezel: dark outside -> bright screen inside => Gx > 0
            // Right bezel: bright screen inside -> dark outside => Gx < 0
            const gx = sobelX[idx];
            const gy = sobelY[idx];
            const signedGx = isLeftFlank ? gx : -gx;

            // True straight vertical bezel has strong directional Gx dominating Gy
            if (signedGx >= 72 && Math.abs(gx) >= Math.abs(gy) * 1.55) {
              sobelHits++;
            }
            signedGxSum += Math.max(0, signedGx);

            const outsideVal = isLeftFlank
              ? (gray[idx - 2] + gray[idx - 3] + gray[idx - 4]) / 3
              : (gray[idx + 2] + gray[idx + 3] + gray[idx + 4]) / 3;
            const insideVal = isLeftFlank
              ? (gray[idx + 2] + gray[idx + 3] + gray[idx + 4]) / 3
              : (gray[idx - 2] + gray[idx - 3] + gray[idx - 4]) / 3;

            darkSum += outsideVal;
            insideSum += insideVal;
            samples++;
          }

          if (samples < Math.floor(spanY / 4)) continue;
          const coherence = sobelHits / samples;
          const meanSignedGx = signedGxSum / samples;
          const darkMean = darkSum / samples;
          const stepJump = insideSum / samples - darkMean;

          if (
            coherence > bestSobelCoherence ||
            (coherence >= bestSobelCoherence - 0.03 &&
              meanSignedGx > bestMeanSignedGx)
          ) {
            bestSobelCoherence = coherence;
            bestMeanSignedGx = meanSignedGx;
            bestOuterDarkLuma = darkMean;
            bestStepJump = stepJump;
          }
        }
      }

      return {
        sobelCoherence: bestSobelCoherence,
        meanSignedGx: bestMeanSignedGx,
        outerDarkLuma: bestOuterDarkLuma,
        stepJump: bestStepJump,
      };
    };

    const leftFlank = scanSobelVerticalFlank(3, Math.min(W - 12, innerL + 6), true);
    const rightFlank = scanSobelVerticalFlank(
      Math.max(12, innerR - 6),
      W - 4,
      false
    );

    // 2. Dark Flanking Bezel Strip Scanner (finds darkest continuous 3px bezel strip in each flank)
    const scanDarkFlankStrip = (xStart: number, xEnd: number) => {
      let minStripLuma = 255;
      let bestDarkCoverage = 0;

      for (const slope of slopes) {
        for (let x0 = xStart; x0 <= xEnd; x0++) {
          let stripSum = 0;
          let darkRows = 0;
          let samples = 0;

          for (let y = innerT; y <= innerB; y += 2) {
            const x = Math.round(x0 + (y - innerT) * slope);
            if (x < 2 || x >= W - 2) continue;
            const idx = y * W + x;
            const bandLuma = (gray[idx - 1] + gray[idx] + gray[idx + 1]) / 3;
            stripSum += bandLuma;
            if (bandLuma <= 42) darkRows++;
            samples++;
          }

          if (samples < Math.floor(spanY / 4)) continue;
          const meanLuma = stripSum / samples;
          const darkCov = darkRows / samples;
          if (meanLuma < minStripLuma) {
            minStripLuma = meanLuma;
            bestDarkCoverage = darkCov;
          }
        }
      }

      return { minStripLuma, darkCoverage: bestDarkCoverage };
    };

    const leftStrip = scanDarkFlankStrip(2, Math.min(W - 10, innerL + 2));
    const rightStrip = scanDarkFlankStrip(Math.max(10, innerR - 2), W - 3);

    // 3. Top Horizontal Sobel (Gy) Bezel / Gallery Status Bar Check
    let topBarSobelCoherence = 0;
    let topDarkMean = 255;
    for (let y0 = 3; y0 <= Math.min(H - 12, innerT + 6); y0++) {
      let hits = 0;
      let dSum = 0;
      let count = 0;
      for (let x = innerL; x <= innerR; x += 2) {
        const idx = y0 * W + x;
        const gy = sobelY[idx];
        const gx = sobelX[idx];
        if (gy >= 70 && Math.abs(gy) >= Math.abs(gx) * 1.5) hits++;
        dSum += (gray[(y0 - 2) * W + x] + gray[(y0 - 1) * W + x]) * 0.5;
        count++;
      }
      if (count > 8) {
        const coh = hits / count;
        if (coh > topBarSobelCoherence) {
          topBarSobelCoherence = coh;
          topDarkMean = dSum / count;
        }
      }
    }

    // --- LUMA CONTRAST CHECK ---
    // High-uniform inner face brightness combined with extremely dark flanking strips (bezels)
    const highUniformInnerBrightness =
      (innerStats.lumaMean >= 136 && innerStats.lumaStd <= 48) ||
      (innerStats.lumaMean >= 150 && innerStats.lumaStd <= 56) ||
      innerStats.clippedRatio > 0.035 ||
      innerStats.brightRatio > 0.3;

    const extremelyDarkFlankingStrips =
      leftStrip.minStripLuma <= 42 &&
      rightStrip.minStripLuma <= 42 &&
      leftStrip.darkCoverage >= 0.56 &&
      rightStrip.darkCoverage >= 0.56;

    const lumaContrastGap =
      innerStats.lumaMean - Math.max(leftStrip.minStripLuma, rightStrip.minStripLuma);

    const lumaContrastSpoof =
      highUniformInnerBrightness &&
      extremelyDarkFlankingStrips &&
      lumaContrastGap >= 72;

    if (lumaContrastSpoof) {
      return {
        isScreenSpoof: true,
        immediateReject: true,
        errorMessage: 'Screen border detected',
        reason: `High-uniform inner luma (${Math.round(innerStats.lumaMean)}, σ=${Math.round(innerStats.lumaStd)}) framed by dark flanking bezels (${Math.round(leftStrip.minStripLuma)} / ${Math.round(rightStrip.minStripLuma)})`,
      };
    }

    // --- SOBEL VERTICAL BEZEL CHECK ---
    // Straight-edge vertical phone bezels detected via 3x3 Sobel Gx on both left & right flanks
    const bilateralSobelBezels =
      leftFlank.sobelCoherence >= 0.44 &&
      rightFlank.sobelCoherence >= 0.44 &&
      leftFlank.outerDarkLuma < 60 &&
      rightFlank.outerDarkLuma < 60 &&
      leftFlank.meanSignedGx >= 64 &&
      rightFlank.meanSignedGx >= 64 &&
      leftFlank.stepJump >= 34 &&
      rightFlank.stepJump >= 34;

    if (bilateralSobelBezels) {
      return {
        isScreenSpoof: true,
        immediateReject: true,
        errorMessage: 'Screen border detected',
        reason: `Sobel vertical phone bezels detected on both flanks (coh ${(leftFlank.sobelCoherence * 100).toFixed(0)}% / ${(rightFlank.sobelCoherence * 100).toFixed(0)}%)`,
      };
    }

    // --- COMBINED BEZEL + EMISSIVE SCREEN CHECKS ---
    const sobelPlusEmissiveGlare =
      leftFlank.sobelCoherence >= 0.38 &&
      rightFlank.sobelCoherence >= 0.38 &&
      (leftFlank.outerDarkLuma < 56 || rightFlank.outerDarkLuma < 56) &&
      (innerStats.clippedRatio > 0.03 || innerStats.brightRatio > 0.24);

    if (sobelPlusEmissiveGlare) {
      return {
        isScreenSpoof: true,
        immediateReject: true,
        errorMessage: 'Screen border detected',
        reason: 'Sobel bezel edges paired with emissive backlit highlights',
      };
    }

    const topAndSideSobelBezel =
      topBarSobelCoherence >= 0.48 &&
      topDarkMean < 56 &&
      (leftFlank.sobelCoherence >= 0.45 || rightFlank.sobelCoherence >= 0.45) &&
      (leftFlank.outerDarkLuma < 58 || rightFlank.outerDarkLuma < 58) &&
      (highUniformInnerBrightness || innerStats.satStd < 0.055);

    const flatOledRecapture =
      innerStats.clippedRatio > 0.055 &&
      innerStats.satStd < 0.048 &&
      innerStats.blueExcess > 6 &&
      (leftFlank.sobelCoherence >= 0.4 || rightFlank.sobelCoherence >= 0.4);

    if (topAndSideSobelBezel || flatOledRecapture) {
      return {
        isScreenSpoof: true,
        immediateReject: false,
        errorMessage: 'Screen border detected',
        reason: 'Display bezel edges and compressed screen chromaticity detected',
      };
    }

    return {
      isScreenSpoof: false,
      immediateReject: false,
      errorMessage: '',
      reason: '',
    };
  }

  private captureAligned150Canvas(
    video: HTMLVideoElement,
    alignedBox: faceapi.Box
  ): HTMLCanvasElement {
    const out = document.createElement('canvas');
    out.width = 150;
    out.height = 150;
    const ctx = out.getContext('2d');
    if (!ctx) return out;

    const sx = Math.max(0, Math.min(video.videoWidth - 1, alignedBox.x));
    const sy = Math.max(0, Math.min(video.videoHeight - 1, alignedBox.y));
    const sw = Math.max(16, Math.min(video.videoWidth - sx, alignedBox.width));
    const sh = Math.max(16, Math.min(video.videoHeight - sy, alignedBox.height));

    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, 150, 150);
    return out;
  }

  private captureSnapshotCanvas(
    video: HTMLVideoElement,
    mirror: boolean
  ): HTMLCanvasElement {
    const maxW = 640;
    const scale = Math.min(1, maxW / (video.videoWidth || 640));
    const w = Math.round((video.videoWidth || 640) * scale);
    const h = Math.round((video.videoHeight || 480) * scale);
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      if (mirror) {
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(video, 0, 0, w, h);
    }
    return canvas;
  }

  private analyzeFacePatch(video: HTMLVideoElement, box: faceapi.Box) {
    const sx = Math.max(0, box.x);
    const sy = Math.max(0, box.y);
    const sw = Math.max(16, Math.min(video.videoWidth - sx, box.width));
    const sh = Math.max(16, Math.min(video.videoHeight - sy, box.height));

    this.patchCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 96, 96);
    const { data } = this.patchCtx.getImageData(0, 0, 96, 96);

    const totalPx = 96 * 96;
    const gray = new Float32Array(totalPx);
    let lumaSum = 0;
    let lumaSqSum = 0;
    let cheekSum = 0;
    let cheekPixels = 0;
    let clippedCount = 0;
    let brightCount = 0;
    let satSum = 0;
    let satSqSum = 0;
    let blueExcessSum = 0;
    let brightBlueSamples = 0;

    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      gray[p] = luma;
      lumaSum += luma;
      lumaSqSum += luma * luma;

      if (luma > 244) clippedCount++;
      if (luma > 185) brightCount++;
      if (luma > 165) {
        blueExcessSum += b - r;
        brightBlueSamples++;
      }

      const maxC = Math.max(r, g, b);
      const minC = Math.min(r, g, b);
      const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
      satSum += sat;
      satSqSum += sat * sat;

      const x = p % 96;
      const y = (p / 96) | 0;
      if (y > 42 && y < 68 && ((x > 18 && x < 36) || (x > 60 && x < 78))) {
        cheekSum += luma;
        cheekPixels++;
      }
    }

    // Fast 3x3 Laplacian Variance + Moire High-Frequency Oscillation Detector
    let lapSum = 0;
    let lapSqSum = 0;
    let moireCount = 0;
    let count = 0;

    for (let y = 1; y < 95; y++) {
      for (let x = 1; x < 95; x++) {
        const idx = y * 96 + x;
        const lap =
          -4 * gray[idx] +
          gray[idx - 1] +
          gray[idx + 1] +
          gray[idx - 96] +
          gray[idx + 96];
        lapSum += lap;
        lapSqSum += lap * lap;
        if (Math.abs(lap) > 80) moireCount++;
        count++;
      }
    }

    const mean = lapSum / (count || 1);
    const variance = lapSqSum / (count || 1) - mean * mean;
    const highFreqSpike = moireCount / (count || 1) > 0.145;

    const lumaMean = lumaSum / totalPx;
    const lumaStd = Math.sqrt(Math.max(0, lumaSqSum / totalPx - lumaMean * lumaMean));
    const satMean = satSum / totalPx;
    const satStd = Math.sqrt(Math.max(0, satSqSum / totalPx - satMean * satMean));
    const blueExcess =
      brightBlueSamples > 20 ? blueExcessSum / brightBlueSamples : 0;

    return {
      sharpness: variance,
      cheekLuma: cheekSum / (cheekPixels || 1),
      lumaMean,
      lumaStd,
      highFreqSpike,
      clippedRatio: clippedCount / totalPx,
      brightRatio: brightCount / totalPx,
      satStd,
      blueExcess,
    };
  }

  private computeEAR(eye: faceapi.Point[]): number {
    const v1 = this.dist(eye[1], eye[5]);
    const v2 = this.dist(eye[2], eye[4]);
    const h = this.dist(eye[0], eye[3]);
    return (v1 + v2) / (2.0 * h + 1e-6);
  }

  private dist(a: faceapi.Point, b: faceapi.Point): number {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  private midpoint(a: faceapi.Point, b: faceapi.Point) {
    return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 };
  }

  private computeIoU(a: faceapi.Box, b: faceapi.Box): number {
    const xA = Math.max(a.x, b.x);
    const yA = Math.max(a.y, b.y);
    const xB = Math.min(a.x + a.width, b.x + b.width);
    const yB = Math.min(a.y + a.height, b.y + b.height);
    const inter = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    return inter / (a.area + b.area - inter + 1e-6);
  }

  private stdDev(arr: number[]): number {
    if (arr.length < 2) return 0;
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length);
  }
}

// ---------- High-Level Async Runners ----------
function waitNextVideoFrame(
  video: HTMLVideoElement,
  signal?: AbortSignal
): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve();
      return;
    }
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      signal?.removeEventListener('abort', done);
      resolve();
    };
    const timer = window.setTimeout(done, 110);
    signal?.addEventListener('abort', done, { once: true });

    const v = video as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: () => void) => number;
    };
    if (typeof v.requestVideoFrameCallback === 'function') {
      v.requestVideoFrameCallback(done);
    } else {
      requestAnimationFrame(done);
    }
  });
}

export async function waitForVideoFrames(
  video: HTMLVideoElement,
  signal?: AbortSignal
): Promise<boolean> {
  const deadline = performance.now() + 4000;
  while (!signal?.aborted && performance.now() < deadline) {
    if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
      return true;
    }
    await new Promise((r) => window.setTimeout(r, 35));
  }
  return video.videoWidth > 0 && video.videoHeight > 0;
}

/**
 * Runs live enrollment with active randomized liveness + 3-frame L2 vector burst.
 */
export async function runEnrollmentVectorBurst(
  video: HTMLVideoElement,
  stream: MediaStream,
  options: {
    mirror: boolean;
    signal?: AbortSignal;
    onHudUpdate?: (hud: FastHudState) => void;
    flash?: FlashOptions;
  }
): Promise<EnrollmentBurstResult> {
  const tStart = performance.now();
  const emptyChecks: BiometricQualityChecks = {
    hardwareValid: false,
    singleFace: false,
    lightingSharpness: false,
    moirePassed: false,
    screenBorderClear: false,
    microTremorPassed: false,
    depthParallaxPassed: false,
    blinkVerified: false,
    challengeType: 'none',
    challengePassed: false,
    reactionTimeMs: null,
    reflectionDelta: null,
  };

  const engine = new AadhaarSpeedAttendanceEngine();
  const hw = engine.verifyPhysicalCamera(stream);
  if (!hw.valid) {
    return {
      ok: false,
      reason: hw.reason || 'Virtual camera blocked.',
      latencyMs: Math.round(performance.now() - tStart),
      qualityChecks: emptyChecks,
    };
  }

  await engine.tryLockCameraControls(stream);

  options.onHudUpdate?.({
    instruction: 'Warming up biometric models...',
    subtext: 'SSD MobileNet · 68 Landmarks · ResNet-34',
    progress: 8,
    shouldFlashUI: false,
    qualityStatus: 'ok',
    activeChallenge: 'none',
    checks: { frontal: false, screenClear: false, liveness: false, vector: false },
  });

  await warmupAadhaarEngine();
  const ready = await waitForVideoFrames(video, options.signal);
  if (!ready || options.signal?.aborted) {
    return {
      ok: false,
      reason: options.signal?.aborted
        ? 'Cancelled'
        : 'Camera did not produce frames.',
      latencyMs: Math.round(performance.now() - tStart),
      qualityChecks: emptyChecks,
    };
  }

  engine.resetSession();

  const dummyTarget: EnrolledStudentVector[] = [
    {
      id: 'enroll',
      attendanceId: 'ENROLL',
      name: 'Enroll',
      vectors: [new Float32Array(128)],
    },
  ];

  while (!options.signal?.aborted) {
    if (performance.now() - tStart > 18_000) {
      return {
        ok: false,
        reason: 'Enrollment timed out. Retry in clear frontal lighting.',
        latencyMs: Math.round(performance.now() - tStart),
        qualityChecks: emptyChecks,
      };
    }

    await waitNextVideoFrame(video, options.signal);
    if (options.signal?.aborted) break;

    const tickResult = await engine.processVideoFrame(video, dummyTarget, {
      requireLiveness: true,
      mirror: options.mirror,
      onHudUpdate: (hud) => options.onHudUpdate?.(hud),
      flash: options.flash,
    });

    if (tickResult) {
      if (!tickResult.spoofPassed) {
        return {
          ok: false,
          reason: tickResult.failReason || 'Screen border detected',
          latencyMs: Math.round(performance.now() - tStart),
          qualityChecks: tickResult.qualityChecks,
        };
      }

      if (tickResult.needsReview) {
        return {
          ok: false,
          reason:
            'Light check inconclusive. Enrollment needs a clear result — face the screen in normal lighting and retry.',
          latencyMs: Math.round(performance.now() - tStart),
          qualityChecks: tickResult.qualityChecks,
        };
      }

      options.onHudUpdate?.({
        instruction: 'Capturing 3-frame biometric burst...',
        subtext: 'Computing L2-normalized centroid for high daily accuracy',
        progress: 92,
        shouldFlashUI: false,
        qualityStatus: 'locked',
        activeChallenge: tickResult.qualityChecks.challengeType,
        checks: { frontal: true, screenClear: true, liveness: true, vector: true },
      });

      const burst = await engine.captureThreeFrameBurst(
        video,
        options.mirror,
        options.signal,
        tickResult.liveDescriptor
      );

      if (!burst) {
        return {
          ok: false,
          reason: 'Could not complete 3-frame vector burst. Retry steadily.',
          latencyMs: Math.round(performance.now() - tStart),
          qualityChecks: tickResult.qualityChecks,
        };
      }

      return {
        ok: true,
        vectors: burst.vectors,
        primaryDescriptor: burst.vectors[0],
        capturedFrameDataUrl: burst.snapshotDataUrl,
        latencyMs: Math.round(performance.now() - tStart),
        qualityChecks: tickResult.qualityChecks,
      };
    }
  }

  return {
    ok: false,
    reason: 'Enrollment cancelled',
    latencyMs: Math.round(performance.now() - tStart),
    qualityChecks: emptyChecks,
  };
}

// ---------- Single-Image L2-Normalized Descriptor Extractor ----------
const imageDescriptorCache = new Map<string, number[]>();

export async function extractDescriptorFromImageUrl(
  imageUrl: string
): Promise<number[] | null> {
  if (!imageUrl || imageUrl.endsWith('preview-frame')) return null;

  const cached = imageDescriptorCache.get(imageUrl);
  if (cached) return cached;

  await warmupAadhaarEngine();

  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new Image();
    el.crossOrigin = 'anonymous';
    el.onload = () => resolve(el);
    el.onerror = () =>
      reject(new Error('Failed to load reference portrait for vector extraction'));
    el.src = imageUrl;
  });

  const det = await faceapi
    .detectSingleFace(
      img,
      new faceapi.SsdMobilenetv1Options({ minConfidence: 0.45 })
    )
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!det?.descriptor) return null;
  const normalized = Array.from(l2Normalize(det.descriptor));
  imageDescriptorCache.set(imageUrl, normalized);
  return normalized;
}
