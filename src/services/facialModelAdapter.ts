import type {
  FacialVerificationRequest,
  FacialVerificationResponse,
  Student,
  SystemConfig,
} from '../types';
import {
  distanceToConfidence,
  euclideanDistance,
  extractDescriptorFromImageUrl,
  getChallengePromptLabel,
  l2Normalize,
} from './aadhaarFaceEngine';

/**
 * Helper to gather all L2-normalized vectors stored on a Student profile.
 */
export function getStudentVectors(student: Student): Float32Array[] {
  if (student.faceVectors && student.faceVectors.length > 0) {
    return student.faceVectors.map((v) => l2Normalize(v));
  }
  if (student.faceDescriptor && student.faceDescriptor.length === 128) {
    return [l2Normalize(student.faceDescriptor)];
  }
  return [];
}

/**
 * Facial Attendance Verification Adapter
 * --------------------------------------
 * Supports:
 * 1. 'aadhaar_local' (Default): On-device L2-normalized multi-vector matching (<0.4ms)
 * 2. 'custom_api': Forwards Attendance ID, L2-normalized 128-D vector, and JPEG frame
 *    to a custom backend API (FastAPI / Node / PostgreSQL pgvector).
 */
export async function verifyFaceFrame(
  request: FacialVerificationRequest,
  config: SystemConfig
): Promise<FacialVerificationResponse> {
  const start = performance.now();

  const localLivenessPassed =
    request.livenessPassed === true &&
    request.qualityChecks?.challengePassed === true &&
    request.qualityChecks?.blinkVerified === true;

  // Attendance always requires local liveness; a remote matcher cannot bypass it.
  if (!localLivenessPassed) {
    return {
      verified: false,
      confidence: 0,
      euclideanDistance: undefined,
      livenessPassed: false,
      qualityChecks: request.qualityChecks,
      processingMs: request.latencyMs ?? Math.round(performance.now() - start),
      mode: config.modelMode,
      message: request.spoofReason || 'Live one-blink verification did not pass.',
    };
  }

  // Do not ask a remote API to verify a face that failed the local PIN-specific match.
  if (!request.identityMatchPassed) {
    return {
      verified: false,
      confidence: 0,
      euclideanDistance: request.identityMatchDistance,
      livenessPassed: true,
      qualityChecks: request.qualityChecks,
      processingMs: request.latencyMs ?? Math.round(performance.now() - start),
      mode: config.modelMode,
      message:
        'This live face does not match the enrolled face for the selected Student PIN. No attendance was saved.',
    };
  }

  if (config.modelMode === 'custom_api' && config.modelEndpointUrl.trim()) {
    try {
      const response = await fetch(config.modelEndpointUrl.trim(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          attendance_id: request.attendanceId,
          student_name: request.student.fullName,
          captured_frame: request.capturedFrameDataUrl,
          descriptor: request.liveDescriptor ?? null,
          reference_photo: request.student.referencePhotoUrl,
          confidence_threshold: config.confidenceThreshold,
          max_euclidean_distance: config.maxEuclideanDistance,
          require_liveness: true,
          liveness_passed: request.livenessPassed ?? false,
          quality_checks: request.qualityChecks ?? null,
          timestamp: request.timestamp,
        }),
      });

      if (!response.ok) {
        throw new Error(`Model API returned HTTP ${response.status}`);
      }

      const data = (await response.json()) as Partial<{
        verified: boolean;
        confidence: number;
        distance: number;
        liveness_passed: boolean;
        message: string;
      }>;

      const confidence =
        typeof data.confidence === 'number'
          ? Math.min(100, Math.max(0, data.confidence))
          : typeof data.distance === 'number'
          ? distanceToConfidence(data.distance)
          : 0;
      const livenessPassed =
        localLivenessPassed && data.liveness_passed !== false;
      const verified =
        (typeof data.verified === 'boolean'
          ? data.verified
          : confidence >= config.confidenceThreshold) &&
        livenessPassed;

      return {
        verified,
        confidence: Number(confidence.toFixed(1)),
        euclideanDistance: data.distance,
        livenessPassed,
        qualityChecks: request.qualityChecks,
        processingMs:
          request.latencyMs ?? Math.round(performance.now() - start),
        mode: 'custom_api',
        message:
          data.message ||
          (verified
            ? 'Identity confirmed by connected model API.'
            : 'Match score fell below configured threshold.'),
      };
    } catch (error) {
      const reason =
        error instanceof Error ? error.message : 'Unable to reach model endpoint';
      return {
        verified: false,
        confidence: 0,
        livenessPassed: false,
        qualityChecks: request.qualityChecks,
        processingMs: Math.round(performance.now() - start),
        mode: 'custom_api',
        message: `Model API error: ${reason}. Switch to On-Device Aadhaar Engine in Settings to verify locally.`,
      };
    }
  }

  // On-Device Aadhaar FaceRD-Speed Local Matching
  try {
    let liveVecArr =
      request.liveDescriptor && request.liveDescriptor.length === 128
        ? request.liveDescriptor
        : null;
    if (!liveVecArr && request.capturedFrameDataUrl) {
      liveVecArr = await extractDescriptorFromImageUrl(
        request.capturedFrameDataUrl
      );
    }

    if (!liveVecArr) {
      return {
        verified: false,
        confidence: 0,
        livenessPassed: Boolean(request.livenessPassed),
        qualityChecks: request.qualityChecks,
        processingMs: Math.round(performance.now() - start),
        mode: 'aadhaar_local',
        message:
          'No clear frontal face could be encoded from the captured frame. Retry in better lighting.',
      };
    }

    const liveVec = l2Normalize(liveVecArr);

    let enrolledVectors = getStudentVectors(request.student);
    let extractedRefVectors: number[][] | null = null;

    if (enrolledVectors.length === 0 && request.student.referencePhotoUrl) {
      const extracted = await extractDescriptorFromImageUrl(
        request.student.referencePhotoUrl
      );
      if (extracted) {
        const norm = l2Normalize(extracted);
        enrolledVectors = [norm];
        extractedRefVectors = [Array.from(norm)];
      }
    }

    if (enrolledVectors.length === 0) {
      return {
        verified: false,
        confidence: 0,
        livenessPassed: Boolean(request.livenessPassed),
        qualityChecks: request.qualityChecks,
        processingMs: Math.round(performance.now() - start),
        mode: 'aadhaar_local',
        message:
          'This student has no enrolled reference face yet. Capture their 3-frame biometric burst in Register Student first.',
      };
    }

    // Best L2-normalized distance across centroid + burst vectors
    let minDistance = Infinity;
    for (const stored of enrolledVectors) {
      const d = euclideanDistance(liveVec, stored);
      if (d < minDistance) minDistance = d;
    }

    const distance = Number(minDistance.toFixed(4));
    const confidence = distanceToConfidence(distance);
    const blinkOk = Boolean(request.qualityChecks?.blinkVerified);
    const challengeOk = Boolean(request.qualityChecks?.challengePassed);
    const livenessOk = Boolean(request.livenessPassed) && blinkOk && challengeOk;
    const distanceMatch = distance <= config.maxEuclideanDistance;
    const verified = livenessOk && distanceMatch;

    let message = '';
    if (verified) {
      const challengeSuffix =
        request.qualityChecks?.challengeType &&
        request.qualityChecks.challengeType !== 'none'
          ? ` · ${getChallengePromptLabel(request.qualityChecks.challengeType)}`
          : '';
      const reactSuffix =
        typeof request.qualityChecks?.reactionTimeMs === 'number'
          ? ` in ${request.qualityChecks.reactionTimeMs}ms`
          : '';
      const padSuffix =
        typeof request.qualityChecks?.padRealScore === 'number'
          ? ` · PAD ${request.qualityChecks.padRealScore}%`
          : '';
      message = `Aadhaar 128-D match & blink liveness confirmed (distance ${distance.toFixed(3)} ≤ ${config.maxEuclideanDistance.toFixed(2)}${challengeSuffix}${reactSuffix}${padSuffix}).`;
    } else if (!blinkOk) {
      message = 'The required one-blink liveness check was not completed.';
    } else if (!livenessOk) {
      message = 'Live anti-spoof verification did not complete.';
    } else {
      message = `Face mismatch — Euclidean distance ${distance.toFixed(3)} exceeded limit ${config.maxEuclideanDistance.toFixed(2)}.`;
    }

    return {
      verified,
      confidence,
      euclideanDistance: distance,
      livenessPassed: Boolean(request.livenessPassed),
      qualityChecks: request.qualityChecks,
      processingMs:
        request.latencyMs ?? Math.round(performance.now() - start),
      mode: 'aadhaar_local',
      message,
      extractedReferenceVectors: extractedRefVectors,
    };
  } catch (err) {
    const reason =
      err instanceof Error ? err.message : 'Unexpected biometric engine error';
    return {
      verified: false,
      confidence: 0,
      livenessPassed: Boolean(request.livenessPassed),
      qualityChecks: request.qualityChecks,
      processingMs: Math.round(performance.now() - start),
      mode: 'aadhaar_local',
      message: `Biometric verification error: ${reason}`,
    };
  }
}

/**
 * Enrolls either a 3-frame L2-normalized vector burst (live capture) or extracts
 * a normalized vector from an uploaded reference portrait.
 */
export async function enrollFaceReference(
  attendanceId: string,
  referenceImageDataUrl: string | null,
  config: SystemConfig,
  precomputedVectors?: number[][] | null
): Promise<{
  enrolled: boolean;
  descriptor: number[] | null;
  vectors: number[][] | null;
  message: string;
}> {
  if (
    !referenceImageDataUrl &&
    (!precomputedVectors || precomputedVectors.length === 0)
  ) {
    return {
      enrolled: false,
      descriptor: null,
      vectors: null,
      message:
        'Registered without biometric vectors. Use Live Aadhaar Capture to lock a 3-frame burst.',
    };
  }

  let vectors =
    precomputedVectors && precomputedVectors.length > 0
      ? precomputedVectors
      : null;

  if (!vectors && referenceImageDataUrl) {
    try {
      const single = await extractDescriptorFromImageUrl(referenceImageDataUrl);
      if (single) vectors = [single];
    } catch {
      vectors = null;
    }
  }

  const descriptor = vectors && vectors.length > 0 ? vectors[0] : null;

  if (config.modelMode === 'custom_api' && config.enrollEndpointUrl.trim()) {
    try {
      const response = await fetch(config.enrollEndpointUrl.trim(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          attendance_id: attendanceId,
          reference_photo: referenceImageDataUrl,
          descriptor,
          vectors,
        }),
      });
      if (response.ok) {
        return {
          enrolled: Boolean(descriptor),
          descriptor,
          vectors,
          message:
            vectors && vectors.length > 1
              ? `${vectors.length - 1}-frame L2 centroid registered on custom API.`
              : '128-D biometric vector registered on custom API.',
        };
      }
    } catch {
      // Fall through cleanly to local storage
    }
  }

  if (!descriptor || !vectors) {
    return {
      enrolled: false,
      descriptor: null,
      vectors: null,
      message:
        'Portrait saved, but no clear frontal face was detected. Use Live Aadhaar Capture for best accuracy.',
    };
  }

  return {
    enrolled: true,
    descriptor,
    vectors,
    message:
      vectors.length > 1
        ? `3-frame L2 biometric burst & centroid locked for sub-second matching.`
        : '128-D L2-normalized vector enrolled and ready for sub-second matching.',
  };
}
