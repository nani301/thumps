export type NavSection = 'dashboard' | 'mark' | 'register' | 'logs' | 'settings';

export type AttendanceStatus = 'present' | 'late' | 'flagged';

export type ModelMode = 'aadhaar_local' | 'custom_api';

export type CheckInMode = '1to1' | 'walkup';

export type LiveChallengeType = 'blink_left' | 'blink_right' | 'smile_blink' | 'none';

export interface BiometricQualityChecks {
  hardwareValid: boolean;
  singleFace: boolean;
  lightingSharpness: boolean;
  moirePassed: boolean;
  screenBorderClear: boolean;
  microTremorPassed: boolean;
  depthParallaxPassed: boolean;
  blinkVerified: boolean;
  challengeType: LiveChallengeType;
  challengePassed: boolean;
  reactionTimeMs: number | null;
  reflectionDelta: number | null; // Soft cheek luminance delta during 180ms glow
  identityContinuity?: number | null; // L2 distance between pre- and post-challenge face vectors (<= 0.40)
  flashTier?: 'pass' | 'weak' | 'fail' | 'skipped'; // screen-flash light-reflection check
  flashCorr?: number | null; // correlation of skin colour shift with the random flash sequence
  challengeSequence?: LiveChallengeType[]; // the random challenges actually completed, in order
}

export interface Student {
  id: string;
  attendanceId: string;
  fullName: string;
  email: string;
  department: string;
  cohort: string;
  referencePhotoUrl: string | null;
  faceDescriptor?: number[] | null; // Primary L2-normalized 128-D centroid vector
  faceVectors?: number[][] | null; // 1-4 L2-normalized 128-D vectors (centroid + 3-frame burst)
  enrolledAt: string; // ISO date string
  biometricEnrolled: boolean;
}

export interface AttendanceRecord {
  id: string;
  attendanceId: string;
  studentName: string;
  department: string;
  timestamp: string; // ISO string
  status: AttendanceStatus;
  confidence: number; // 0 to 100
  euclideanDistance?: number; // ResNet-34 128-D L2-normalized distance (<= 0.45 Aadhaar strict)
  latencyMs?: number; // End-to-end verification time in ms
  checkInMode?: CheckInMode;
  livenessPassed: boolean;
  qualityChecks?: BiometricQualityChecks;
  capturedFrameUrl: string | null;
  referencePhotoUrl: string | null;
  verificationMode: ModelMode;
  notes?: string;
}

export interface DailyTrendPoint {
  date: string; // YYYY-MM-DD
  label: string; // e.g., "Nov 12"
  present: number;
  late: number;
  absent: number;
  total: number;
}

export interface UserProfile {
  fullName: string;
  role: string;
  institution: string;
  email: string;
  avatarUrl: string | null;
}

export interface SystemConfig {
  theme: 'light' | 'dark';
  confidenceThreshold: number; // 65 - 99
  maxEuclideanDistance: number; // 0.36 - 0.56 (Aadhaar FaceRD default: 0.45)
  impostorMargin: number; // 0.03 - 0.14 (1:N twin/lookalike margin, default: 0.07)
  defaultCheckInMode: CheckInMode;
  lateCutoffTime: string; // "09:15"
  requireLiveness: boolean;
  enableAudioChime: boolean;
  mirrorCamera: boolean;
  cameraResolution: '480p' | '720p';
  modelMode: ModelMode;
  modelEndpointUrl: string;
  enrollEndpointUrl: string;
  flashSlots: number; // screen-flash light-check slots (0 = off, 8 recommended; ~2.7s)
  checkInCooldownMinutes: number; // one accepted check-in per student inside this window (blocks duplicate logs)
  allowPhotoEnrollment: boolean; // false = enrollment only from a LIVE liveness-checked capture
}

export interface FacialVerificationRequest {
  attendanceId: string;
  student: Student;
  capturedFrameDataUrl: string;
  liveDescriptor?: number[] | null;
  livenessPassed?: boolean;
  spoofReason?: string;
  qualityChecks?: BiometricQualityChecks;
  latencyMs?: number;
  timestamp: string;
}

export interface FacialVerificationResponse {
  verified: boolean;
  confidence: number;
  euclideanDistance?: number;
  livenessPassed: boolean;
  qualityChecks?: BiometricQualityChecks;
  processingMs: number;
  mode: ModelMode;
  message: string;
  extractedReferenceVectors?: number[][] | null;
}
