import { useEffect, useState } from 'react';
import * as faceapi from '@vladmandic/face-api';

export type ComputeBackend = 'webgl' | 'wasm' | 'cpu';

export interface BackendState {
  backend: ComputeBackend | null;
  webglVersion: 2 | 1 | null;
  ready: boolean;
  label: string;
}

interface FaceApiTfRuntime {
  wasm?: {
    setWasmPaths?: (
      paths: string | Record<string, string>,
      usePlatformFetch?: boolean
    ) => void;
  };
  setWasmPaths?: (
    paths: string | Record<string, string>,
    usePlatformFetch?: boolean
  ) => void;
  setBackend?: (backendName: string) => Promise<boolean>;
  getBackend?: () => string | undefined;
  ready?: () => Promise<void>;
  env?: () => {
    set?: (key: string, value: unknown) => void;
    getBool?: (key: string) => boolean;
    getNumber?: (key: string) => number;
  };
  version_core?: string;
  tidy?: <T>(fn: () => T) => T;
  zeros?: (shape: number[]) => {
    dataSync?: () => ArrayLike<number>;
    dispose?: () => void;
  };
  dispose?: (t: unknown) => void;
}

let currentState: BackendState = {
  backend: null,
  webglVersion: null,
  ready: false,
  label: 'Initializing...',
};

let initPromise: Promise<ComputeBackend> | null = null;
const listeners = new Set<(state: BackendState) => void>();

function notifyListeners(next: BackendState): void {
  currentState = next;
  listeners.forEach((cb) => cb(next));
}

export function getTfRuntime(): FaceApiTfRuntime | undefined {
  return (faceapi as unknown as { tf?: FaceApiTfRuntime }).tf;
}

export function getActiveBackend(): ComputeBackend | null {
  return currentState.backend;
}

export function getBackendState(): BackendState {
  return currentState;
}

export function getBackendDisplayLabel(
  backend: ComputeBackend | null,
  webglVersion: 2 | 1 | null = currentState.webglVersion
): string {
  if (backend === 'webgl') {
    return webglVersion === 1 ? 'WebGL 1 GPU' : 'WebGL 2 GPU';
  }
  if (backend === 'wasm') return 'WASM SIMD';
  if (backend === 'cpu') return 'CPU Fallback';
  return 'Initializing...';
}

export function subscribeTensorFlowBackend(
  listener: (state: BackendState) => void
): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * React hook exposing the live TensorFlow.js compute backend status for UI panels.
 */
export function useTensorFlowBackend(): BackendState {
  const [state, setState] = useState<BackendState>(() => getBackendState());

  useEffect(() => {
    setState(getBackendState());
    return subscribeTensorFlowBackend(setState);
  }, []);

  return state;
}

const SOFTWARE_GL_PATTERNS = [
  'swiftshader',
  'llvmpipe',
  'softpipe',
  'mesa offscreen',
  'microsoft basic render',
  'google swiftshader',
];

function isSoftwareRenderer(gl: WebGLRenderingContext | WebGL2RenderingContext): boolean {
  try {
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    if (ext) {
      const renderer = String(
        gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || ''
      ).toLowerCase();
      if (SOFTWARE_GL_PATTERNS.some((p) => renderer.includes(p))) {
        return true;
      }
    }
    const baseRenderer = String(gl.getParameter(gl.RENDERER) || '').toLowerCase();
    return SOFTWARE_GL_PATTERNS.some((p) => baseRenderer.includes(p));
  } catch {
    return false;
  }
}

/**
 * Probes hardware-accelerated WebGL (WebGL2 first, then WebGL1).
 * Tests strict hardware context first; if Chrome's caveat flag falsely rejects a real
 * integrated GPU, retries without the caveat flag and filters out slow software GL
 * renderers (SwiftShader/llvmpipe) so GPU-less devices still use faster WASM SIMD.
 */
function probeHardwareWebGL(): 2 | 1 | 0 {
  if (typeof document === 'undefined') return 0;

  const tryCreate = (
    version: 2 | 1,
    strictCaveat: boolean
  ): boolean => {
    const attrs: WebGLContextAttributes = {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
      failIfMajorPerformanceCaveat: strictCaveat,
    };

    try {
      const canvas = document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;
      const gl = (
        version === 2
          ? canvas.getContext('webgl2', attrs)
          : canvas.getContext('webgl', attrs) ||
            canvas.getContext('experimental-webgl', attrs)
      ) as WebGL2RenderingContext | WebGLRenderingContext | null;

      if (!gl) return false;
      const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
      const software = isSoftwareRenderer(gl);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return !software && maxTex >= 2048;
    } catch {
      return false;
    }
  };

  if (tryCreate(2, true)) return 2;
  if (tryCreate(1, true)) return 1;
  if (tryCreate(2, false)) return 2;
  if (tryCreate(1, false)) return 1;
  return 0;
}

function verifyTensorExecution(tf: FaceApiTfRuntime): boolean {
  if (!tf.zeros) return true;
  try {
    const t = tf.zeros([1]);
    t.dataSync?.();
    if (typeof t.dispose === 'function') {
      t.dispose();
    } else {
      tf.dispose?.(t);
    }
    return true;
  } catch {
    return false;
  }
}

async function resolveWasmBasePath(tf: FaceApiTfRuntime): Promise<string> {
  const tfVersion = tf.version_core || '4.22.0';
  const cdnUrl = `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfVersion}/dist/`;

  if (typeof window === 'undefined' || !window.location?.origin) {
    return cdnUrl;
  }

  const localUrl = `${window.location.origin}/wasm/`;
  try {
    const probe = await fetch(`${localUrl}tfjs-backend-wasm.wasm`, {
      method: 'HEAD',
    });
    const contentType = probe.headers.get('content-type') || '';
    if (
      probe.ok &&
      !contentType.includes('text/html') &&
      (contentType.includes('wasm') ||
        contentType.includes('octet-stream') ||
        contentType === '')
    ) {
      return localUrl;
    }
  } catch {
    // Fall back to reliable CDN if local /wasm/ check fails
  }

  return cdnUrl;
}

function configureWasmPaths(tf: FaceApiTfRuntime, wasmPath: string): void {
  try {
    if (typeof tf.wasm?.setWasmPaths === 'function') {
      tf.wasm.setWasmPaths(wasmPath);
    } else if (typeof tf.setWasmPaths === 'function') {
      tf.setWasmPaths(wasmPath);
    }
  } catch {
    // Ignore if WASM backend was already initialized
  }
}

/**
 * 3-Tier TensorFlow.js Backend Negotiator (`webgl` [v2 -> v1] -> `wasm` -> `cpu`).
 * Must be called once before loading face-api models or allocating any tensors.
 */
export async function initTensorFlowBackend(): Promise<ComputeBackend> {
  if (currentState.ready && currentState.backend) {
    return currentState.backend;
  }

  if (!initPromise) {
    initPromise = (async (): Promise<ComputeBackend> => {
      const tf = getTfRuntime();
      if (!tf || !tf.setBackend || !tf.ready) {
        notifyListeners({
          backend: 'cpu',
          webglVersion: null,
          ready: true,
          label: getBackendDisplayLabel('cpu', null),
        });
        return 'cpu';
      }

      // Configure valid WASM binary directory BEFORE any backend initialization
      const wasmPath = await resolveWasmBasePath(tf);
      configureWasmPaths(tf, wasmPath);

      // Tier 1: Hardware WebGL (WebGL2, with automatic WebGL1 fallback)
      // Run our own hardware probe first so WebGL1 GPUs aren't skipped when TFJS defaults
      // WEBGL_VERSION=2, while GPU-less/software environments skip cleanly to WASM SIMD.
      const hwWebGL = probeHardwareWebGL();
      if (hwWebGL > 0) {
        const versionsToTry: (2 | 1)[] = hwWebGL === 2 ? [2, 1] : [1];
        for (const version of versionsToTry) {
          try {
            tf.env?.().set?.('WEBGL_VERSION', version);
            tf.env?.().set?.('HAS_WEBGL', true);
            const ok = await tf.setBackend('webgl');
            if (ok) {
              await tf.ready();
              if (
                tf.getBackend?.() === 'webgl' &&
                verifyTensorExecution(tf)
              ) {
                notifyListeners({
                  backend: 'webgl',
                  webglVersion: version,
                  ready: true,
                  label: getBackendDisplayLabel('webgl', version),
                });
                return 'webgl';
              }
            }
          } catch {
            // Try next WebGL version or fall through cleanly to WASM
          }
        }
      }

      // Tier 2: WebAssembly SIMD (local /wasm/ or verified CDN fallback)
      try {
        const okWasm = await tf.setBackend('wasm');
        if (okWasm) {
          await tf.ready();
          if (tf.getBackend?.() === 'wasm' && verifyTensorExecution(tf)) {
            notifyListeners({
              backend: 'wasm',
              webglVersion: null,
              ready: true,
              label: getBackendDisplayLabel('wasm', null),
            });
            return 'wasm';
          }
        }
      } catch {
        // Fall through cleanly to CPU
      }

      // Tier 3: Universal CPU fallback (guaranteed zero crash)
      try {
        await tf.setBackend('cpu');
        await tf.ready();
      } catch {
        // CPU backend already active
      }

      notifyListeners({
        backend: 'cpu',
        webglVersion: null,
        ready: true,
        label: getBackendDisplayLabel('cpu', null),
      });
      return 'cpu';
    })();
  }

  return initPromise;
}
