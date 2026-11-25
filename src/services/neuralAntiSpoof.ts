import * as ort from 'onnxruntime-web/wasm';
import { getTfRuntime, initTensorFlowBackend } from './initTensorFlowBackend';

/**
 * VeriMark On-Device Neural Presentation Attack Detection (PAD) Service
 * ---------------------------------------------------------------------
 * Primary Engine: MiniFASNetV2-SE INT8 ONNX (/models/minifasv2_se_int8.onnx, 612 KB)
 *   - CelebA-Spoof trained (98.20% accuracy, 0.9984 ROC-AUC)
 *   - Input:  [batch, 3, 128, 128] Float32 NCHW, RGB in [0, 1]
 *   - Output: [batch, 2] Float32 logits -> [real_logit, spoof_logit]
 *   - Multi-scale 2-crop batched inference:
 *       Crop 0 (1.15x): Tight face texture, skin speculars, glass veil & moiré
 *       Crop 1 (2.20x): Wide context capturing phone edges, bezels, pillar-box & screen glow
 *
 * Secondary Automatic Fallback: TFJS Anti-Spoof Graph Model (/models/antispoof.json)
 *   - Input:  [batch, 128, 128, 3] Float32 NHWC, RGB in [0, 1]
 *   - Output: [batch, 1] Float32 sigmoid real probability
 */

const ONNX_MODEL_URL = '/models/minifasv2_se_int8.onnx';
const TFJS_MODEL_URL = '/models/antispoof.json';
const CROP_SIZE = 128;
const MULTI_SCALES: readonly [number, number] = [1.15, 2.2];

export type NeuralPadBackend = 'minifasv2-onnx' | 'tfjs-antispoof' | 'unavailable';

export interface FaceBoxLike {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface NeuralPadResult {
  realScore: number; // Combined 0..1 probability of a live 3D human face
  tightScore: number; // 1.15x tight face crop real probability
  contextScore: number; // 2.2x wide context crop real probability
  minScaleScore: number; // min(tightScore, contextScore)
  backend: NeuralPadBackend;
  latencyMs: number;
}

interface TfGraphModelLike {
  execute: (input: unknown) => unknown;
  predict?: (input: unknown) => unknown;
}

interface ExtendedTfRuntime {
  loadGraphModel?: (url: string) => Promise<TfGraphModelLike>;
  tensor4d?: (
    values: Float32Array,
    shape: [number, number, number, number]
  ) => { dispose?: () => void };
  dispose?: (t: unknown) => void;
}

let onnxSession: ort.InferenceSession | null = null;
let tfjsModel: TfGraphModelLike | null = null;
let activePadBackend: NeuralPadBackend = 'unavailable';
let padWarmupPromise: Promise<NeuralPadBackend> | null = null;

// Reusable scratch canvas for single-read pixel extraction
let scratchCanvas: HTMLCanvasElement | null = null;
let scratchCtx: CanvasRenderingContext2D | null = null;

function getScratchCanvas(w: number, h: number): {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
} {
  if (!scratchCanvas) {
    scratchCanvas = document.createElement('canvas');
    scratchCtx = scratchCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (scratchCanvas.width !== w || scratchCanvas.height !== h) {
    scratchCanvas.width = w;
    scratchCanvas.height = h;
  }
  return { canvas: scratchCanvas, ctx: scratchCtx! };
}

/**
 * Exact OpenCV BORDER_REFLECT_101 index reflection (gfedcb|abcdefgh|gfedcba).
 */
function reflect101(idx: number, len: number): number {
  if (len <= 1) return 0;
  let i = idx;
  let guard = 0;
  while ((i < 0 || i >= len) && guard < 12) {
    if (i < 0) i = -i;
    if (i >= len) i = 2 * len - 2 - i;
    guard++;
  }
  return Math.max(0, Math.min(len - 1, i));
}

/**
 * Extracts multi-scale square crops around `box` with exact BORDER_REFLECT_101 padding
 * and bilinear resizing into BOTH NCHW (ONNX) and NHWC (TFJS) Float32 [0..1] tensors.
 */
function extractMultiScaleTensors(
  source: HTMLVideoElement | HTMLCanvasElement,
  box: FaceBoxLike,
  layout: 'nchw' | 'nhwc'
): Float32Array {
  const srcRawW =
    source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const srcRawH =
    source instanceof HTMLVideoElement ? source.videoHeight : source.height;

  // Cap scratch buffer to <= 384px wide for <1.2ms pixel readback while staying >2x sharper than 128px crop
  const maxDim = 384;
  const downScale = Math.min(1, maxDim / Math.max(srcRawW, srcRawH, 1));
  const sw = Math.max(32, Math.round(srcRawW * downScale));
  const sh = Math.max(32, Math.round(srcRawH * downScale));

  const { ctx } = getScratchCanvas(sw, sh);
  ctx.drawImage(source, 0, 0, sw, sh);
  const { data } = ctx.getImageData(0, 0, sw, sh);

  const bx = box.x * downScale;
  const by = box.y * downScale;
  const bw = Math.max(8, box.width * downScale);
  const bh = Math.max(8, box.height * downScale);
  const cx = bx + bw * 0.5;
  const cy = by + bh * 0.5;
  const baseSide = Math.max(bw, bh);

  const batch = MULTI_SCALES.length;
  const planeSize = CROP_SIZE * CROP_SIZE;
  const out = new Float32Array(batch * 3 * planeSize);
  const inv255 = 1 / 255;

  for (let b = 0; b < batch; b++) {
    const side = Math.max(16, baseSide * MULTI_SCALES[b]);
    const x0 = cx - side * 0.5;
    const y0 = cy - side * 0.5;
    const step = side / CROP_SIZE;

    const batchOffset = b * 3 * planeSize;
    const rPlane = batchOffset;
    const gPlane = batchOffset + planeSize;
    const bPlane = batchOffset + planeSize * 2;

    for (let dy = 0; dy < CROP_SIZE; dy++) {
      const fy = y0 + (dy + 0.5) * step - 0.5;
      const iy0 = Math.floor(fy);
      const wy = fy - iy0;
      const ry0 = reflect101(iy0, sh);
      const ry1 = reflect101(iy0 + 1, sh);
      const row0 = ry0 * sw;
      const row1 = ry1 * sw;
      const dyRow = dy * CROP_SIZE;

      for (let dx = 0; dx < CROP_SIZE; dx++) {
        const fx = x0 + (dx + 0.5) * step - 0.5;
        const ix0 = Math.floor(fx);
        const wx = fx - ix0;
        const rx0 = reflect101(ix0, sw);
        const rx1 = reflect101(ix0 + 1, sw);

        const p00 = (row0 + rx0) * 4;
        const p10 = (row0 + rx1) * 4;
        const p01 = (row1 + rx0) * 4;
        const p11 = (row1 + rx1) * 4;

        const w00 = (1 - wx) * (1 - wy);
        const w10 = wx * (1 - wy);
        const w01 = (1 - wx) * wy;
        const w11 = wx * wy;

        const r =
          (data[p00] * w00 +
            data[p10] * w10 +
            data[p01] * w01 +
            data[p11] * w11) *
          inv255;
        const g =
          (data[p00 + 1] * w00 +
            data[p10 + 1] * w10 +
            data[p01 + 1] * w01 +
            data[p11 + 1] * w11) *
          inv255;
        const bl =
          (data[p00 + 2] * w00 +
            data[p10 + 2] * w10 +
            data[p01 + 2] * w01 +
            data[p11 + 2] * w11) *
          inv255;

        const pixIdx = dyRow + dx;
        if (layout === 'nchw') {
          out[rPlane + pixIdx] = r;
          out[gPlane + pixIdx] = g;
          out[bPlane + pixIdx] = bl;
        } else {
          const hwcIdx = batchOffset + pixIdx * 3;
          out[hwcIdx] = r;
          out[hwcIdx + 1] = g;
          out[hwcIdx + 2] = bl;
        }
      }
    }
  }

  return out;
}

function configureOnnxWasmRuntime(): void {
  // Single-threaded SIMD WASM avoids SharedArrayBuffer / COOP-COEP iframe restrictions
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  if (typeof window !== 'undefined' && window.location?.origin) {
    ort.env.wasm.wasmPaths = new URL('/ort/', window.location.origin).href;
  }
}

async function initOnnxMiniFasNet(): Promise<boolean> {
  try {
    configureOnnxWasmRuntime();
    const resp = await fetch(ONNX_MODEL_URL);
    if (!resp.ok) return false;
    const modelBytes = new Uint8Array(await resp.arrayBuffer());

    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    // Dry warmup pass on [2, 3, 128, 128] so first real user verification has ~15ms latency
    const dummy = new Float32Array(2 * 3 * CROP_SIZE * CROP_SIZE).fill(0.5);
    const inputName = session.inputNames[0] || 'input';
    const feeds: Record<string, ort.Tensor> = {
      [inputName]: new ort.Tensor('float32', dummy, [2, 3, CROP_SIZE, CROP_SIZE]),
    };
    const out = await session.run(feeds);
    Object.values(out).forEach((t) => {
      if (typeof (t as { dispose?: () => void }).dispose === 'function') {
        (t as { dispose?: () => void }).dispose?.();
      }
    });

    onnxSession = session;
    activePadBackend = 'minifasv2-onnx';
    return true;
  } catch {
    onnxSession = null;
    return false;
  }
}

async function initTfjsAntiSpoofFallback(): Promise<boolean> {
  try {
    await initTensorFlowBackend();
    const tf = getTfRuntime() as ExtendedTfRuntime | undefined;
    if (!tf?.loadGraphModel || !tf?.tensor4d) return false;

    const model = await tf.loadGraphModel(TFJS_MODEL_URL);
    // Dry warmup pass
    const dummy = tf.tensor4d(
      new Float32Array(2 * CROP_SIZE * CROP_SIZE * 3).fill(0.5),
      [2, CROP_SIZE, CROP_SIZE, 3]
    );
    const out = model.execute(dummy);
    tf.dispose?.(dummy);
    tf.dispose?.(out);

    tfjsModel = model;
    activePadBackend = 'tfjs-antispoof';
    return true;
  } catch {
    tfjsModel = null;
    return false;
  }
}

export function isNeuralPadWarmed(): boolean {
  return activePadBackend !== 'unavailable';
}

export function getNeuralPadBackend(): NeuralPadBackend {
  return activePadBackend;
}

/**
 * Warms up MiniFASNetV2-SE ONNX locally, with automatic fallback to TFJS Anti-Spoof CNN.
 */
export async function warmupNeuralAntiSpoof(): Promise<NeuralPadBackend> {
  if (activePadBackend !== 'unavailable') return activePadBackend;
  if (!padWarmupPromise) {
    padWarmupPromise = (async () => {
      const okOnnx = await initOnnxMiniFasNet();
      if (okOnnx) return activePadBackend;
      const okTfjs = await initTfjsAntiSpoofFallback();
      if (okTfjs) return activePadBackend;
      activePadBackend = 'unavailable';
      return activePadBackend;
    })().catch(() => {
      padWarmupPromise = null;
      activePadBackend = 'unavailable';
      return 'unavailable';
    });
  }
  return padWarmupPromise;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

/**
 * Runs silent 2-scale (1.15x tight face + 2.2x wide context) Neural PAD inference.
 * Returns realScore in [0, 1] (higher = live 3D human face; lower = screen/video/print spoof).
 */
export async function scoreFaceAntiSpoof(
  source: HTMLVideoElement | HTMLCanvasElement,
  box: FaceBoxLike
): Promise<NeuralPadResult | null> {
  const t0 = performance.now();
  if (activePadBackend === 'unavailable') {
    await warmupNeuralAntiSpoof();
  }

  // Tier 1: MiniFASNetV2-SE INT8 ONNX (verified polarity: index 0 = real_logit, index 1 = spoof_logit)
  if (onnxSession && activePadBackend === 'minifasv2-onnx') {
    try {
      const nchw = extractMultiScaleTensors(source, box, 'nchw');
      const inputName = onnxSession.inputNames[0] || 'input';
      const outputName = onnxSession.outputNames[0] || 'output';
      const inputTensor = new ort.Tensor('float32', nchw, [
        MULTI_SCALES.length,
        3,
        CROP_SIZE,
        CROP_SIZE,
      ]);

      const results = await onnxSession.run({ [inputName]: inputTensor });
      const outTensor = results[outputName];
      const logits = outTensor?.data as Float32Array | undefined;

      let tightScore = 0.5;
      let contextScore = 0.5;
      if (logits && logits.length >= 4) {
        // Softmax probability for class 0 (REAL) = sigmoid(real_logit - spoof_logit)
        tightScore = sigmoid(logits[0] - logits[1]);
        contextScore = sigmoid(logits[2] - logits[3]);
      }

      if (typeof (outTensor as { dispose?: () => void })?.dispose === 'function') {
        (outTensor as { dispose?: () => void }).dispose?.();
      }
      if (typeof (inputTensor as { dispose?: () => void })?.dispose === 'function') {
        (inputTensor as { dispose?: () => void }).dispose?.();
      }

      const realScore = tightScore * 0.55 + contextScore * 0.45;
      return {
        realScore: Number(realScore.toFixed(4)),
        tightScore: Number(tightScore.toFixed(4)),
        contextScore: Number(contextScore.toFixed(4)),
        minScaleScore: Number(Math.min(tightScore, contextScore).toFixed(4)),
        backend: 'minifasv2-onnx',
        latencyMs: Math.round(performance.now() - t0),
      };
    } catch {
      // Seamless fallback to TFJS graph model if ONNX session fails mid-flight
      await initTfjsAntiSpoofFallback();
    }
  }

  // Tier 2: TFJS Anti-Spoof Graph Model fallback
  if (tfjsModel && activePadBackend === 'tfjs-antispoof') {
    const tf = getTfRuntime() as ExtendedTfRuntime | undefined;
    if (tf?.tensor4d) {
      let inputTensor: { dispose?: () => void } | null = null;
      let outputTensor: unknown = null;
      try {
        const nhwc = extractMultiScaleTensors(source, box, 'nhwc');
        inputTensor = tf.tensor4d(nhwc, [
          MULTI_SCALES.length,
          CROP_SIZE,
          CROP_SIZE,
          3,
        ]);
        outputTensor = tfjsModel.execute(inputTensor);
        const outObj = outputTensor as { data?: () => Promise<Float32Array> };
        const scores = outObj.data ? await outObj.data() : null;

        const tightScore = scores && scores.length >= 1 ? scores[0] : 0.5;
        const contextScore = scores && scores.length >= 2 ? scores[1] : tightScore;
        const realScore = tightScore * 0.55 + contextScore * 0.45;

        return {
          realScore: Number(realScore.toFixed(4)),
          tightScore: Number(tightScore.toFixed(4)),
          contextScore: Number(contextScore.toFixed(4)),
          minScaleScore: Number(Math.min(tightScore, contextScore).toFixed(4)),
          backend: 'tfjs-antispoof',
          latencyMs: Math.round(performance.now() - t0),
        };
      } catch {
        return null;
      } finally {
        if (inputTensor) tf.dispose?.(inputTensor);
        if (outputTensor) tf.dispose?.(outputTensor);
      }
    }
  }

  return null;
}
