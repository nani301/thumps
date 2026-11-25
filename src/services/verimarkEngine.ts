/**
 * VeriMark Standalone Biometric Verification & Anti-Spoof Engine
 * --------------------------------------------------------------
 * Single-import reusable engine bundling:
 * - On-device SSD MobileNet + 68-Landmark + ResNet-34 128-D L2 Matching
 * - Glasses-friendly Single-Blink Liveness (TSTS ABAS style)
 * - MiniFASNetV2-SE INT8 ONNX + TFJS Anti-Spoof Neural PAD (2-scale 1.15x / 2.2x)
 * - Full-Frame 4:3 vs 9:16 Vertical Phone Pillar-Box & Hair Black-Point Veil Guards
 * - Enrollment 3-Frame Vector Burst & Duplicate-Face De-Duplication
 */

export {
  AadhaarSpeedAttendanceEngine,
  computeNormalizedCentroid,
  distanceToConfidence,
  euclideanDistance,
  extractDescriptorFromImageUrl,
  getActiveBackend,
  getBackendDisplayLabel,
  getChallengePromptLabel,
  initTensorFlowBackend,
  isAadhaarEngineWarmed,
  l2Normalize,
  runEnrollmentVectorBurst,
  useTensorFlowBackend,
  waitForVideoFrames,
  warmupAadhaarEngine,
  type ComputeBackend,
  type EnrolledStudentVector,
  type EnrollmentBurstResult,
  type FastAttendanceResult,
  type FastHudState,
  type LightingStatus,
} from './aadhaarFaceEngine';

export {
  getNeuralPadBackend,
  isNeuralPadWarmed,
  scoreFaceAntiSpoof,
  warmupNeuralAntiSpoof,
  type FaceBoxLike,
  type NeuralPadBackend,
  type NeuralPadResult,
} from './neuralAntiSpoof';

export {
  BlinkTracker,
  CONTINUITY_MAX_DISTANCE,
  calibrateEarPeak,
  detectVerticalPhonePillarBox,
  dist128,
  evaluateHairBlackPointVeil,
  findDuplicateFace,
  findPriorCheckIn,
  identityContinuity,
  isBorderlessEmissiveScreen,
  isEnrollableFrame,
  isGlassBlackPointVeil,
  l2,
  meanNormalized,
  type EnrolledLike,
  type HairBlackPointVeilResult,
  type Pt,
  type RecordLike,
  type Rect,
  type VerticalPillarBoxResult,
} from './livenessCore';

export {
  enrollFaceReference,
  getStudentVectors,
  verifyFaceFrame,
} from './facialModelAdapter';

export { shrinkDataUrl } from './imageUtils';

export type {
  AttendanceRecord,
  AttendanceStatus,
  BiometricQualityChecks,
  CheckInMode,
  FacialVerificationRequest,
  FacialVerificationResponse,
  LiveChallengeType,
  ModelMode,
  Student,
  SystemConfig,
} from '../types';
