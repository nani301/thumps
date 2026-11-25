import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from 'react';
import {
  AlertTriangle,
  Camera,
  CameraOff,
  Check,
  CheckCircle2,
  Clock,
  Play,
  RefreshCw,
  ScanFace,
  ShieldCheck,
  Square,
  Upload,
  UserCheck,
  Users,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react';
import type {
  AttendanceRecord,
  AttendanceStatus,
  BiometricQualityChecks,
  CheckInMode,
  FacialVerificationResponse,
  Student,
  SystemConfig,
} from '../types';
import { formatDateKey } from '../data/seedData';
import {
  getStudentVectors,
  verifyFaceFrame,
} from '../services/facialModelAdapter';
import {
  AadhaarSpeedAttendanceEngine,
  extractDescriptorFromImageUrl,
  getActiveBackend,
  getBackendDisplayLabel,
  getChallengePromptLabel,
  isAadhaarEngineWarmed,
  l2Normalize,
  waitForVideoFrames,
  warmupAadhaarEngine,
  type ComputeBackend,
  type EnrolledStudentVector,
  type FastHudState,
} from '../services/aadhaarFaceEngine';
import {
  playVerificationFlagTone,
  playVerificationSuccessChime,
} from '../services/soundFeedback';
import { Avatar } from './Avatar';
import { findPriorCheckIn } from '../services/livenessCore';

interface MarkAttendanceViewProps {
  students: Student[];
  records: AttendanceRecord[];
  config: SystemConfig;
  initialAttendanceId?: string;
  onRecordAttendance: (record: AttendanceRecord) => void;
  onCacheStudentVectors: (studentId: string, vectors: number[][]) => void;
  onGoToRegister: () => void;
}

type VerificationPhase =
  | 'standby'
  | 'requesting_camera'
  | 'camera_active'
  | 'verifying'
  | 'completed'
  | 'camera_error';

interface QueueReceiptToast {
  id: string;
  studentName: string;
  attendanceId: string;
  avatarUrl: string | null;
  status: AttendanceStatus;
  confidence: number;
  distance: number;
  latencyMs: number;
  matched: boolean;
  message: string;
}

function determineAttendanceStatus(
  verified: boolean,
  now: Date,
  lateCutoffTime: string
): AttendanceStatus {
  if (!verified) return 'flagged';
  const [cutHour, cutMin] = lateCutoffTime.split(':').map((n) => Number(n) || 0);
  const currentMins = now.getHours() * 60 + now.getMinutes();
  const cutoffMins = cutHour * 60 + cutMin;
  return currentMins > cutoffMins ? 'late' : 'present';
}

export function MarkAttendanceView({
  students,
  records,
  config,
  initialAttendanceId = '',
  onRecordAttendance,
  onCacheStudentVectors,
  onGoToRegister,
}: MarkAttendanceViewProps) {
  const [checkInMode, setCheckInMode] = useState<CheckInMode>(() =>
    initialAttendanceId ? '1to1' : config.defaultCheckInMode
  );
  const [attendanceIdInput, setAttendanceIdInput] = useState(initialAttendanceId);
  const [phase, setPhase] = useState<VerificationPhase>('standby');
  const [engineWarmed, setEngineWarmed] = useState<boolean>(() =>
    isAadhaarEngineWarmed()
  );
  const [activeBackend, setActiveBackend] = useState<ComputeBackend | null>(() =>
    getActiveBackend()
  );
  const [audioEnabled, setAudioEnabled] = useState<boolean>(
    config.enableAudioChime
  );
  const [indexingRoster, setIndexingRoster] = useState(false);

  const [liveHud, setLiveHud] = useState<FastHudState | null>(null);
  const [cameraErrorMessage, setCameraErrorMessage] = useState<string>('');
  const [capturedFrameUrl, setCapturedFrameUrl] = useState<string | null>(null);
  const [verificationResult, setVerificationResult] =
    useState<FacialVerificationResponse | null>(null);
  const [lastLoggedRecord, setLastLoggedRecord] =
    useState<AttendanceRecord | null>(null);

  // Walk-Up Queue (1:N) session state
  const [queueReceipt, setQueueReceipt] = useState<QueueReceiptToast | null>(null);
  const [queueSessionLog, setQueueSessionLog] = useState<QueueReceiptToast[]>([]);
  const [queueElapsedSec, setQueueElapsedSec] = useState(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const queueCooldownMapRef = useRef<Map<string, number>>(new Map());
  const queueToastTimerRef = useRef<number | null>(null);
  const studentsRef = useRef<Student[]>(students);
  studentsRef.current = students;
  const recordsRef = useRef<AttendanceRecord[]>(records);
  recordsRef.current = records;
  // Screen-flash light check overlay (the kiosk screen itself is the light source)
  const [flashColor, setFlashColor] = useState<string | null>(null);
  useEffect(() => {
    if (phase !== 'camera_active') setFlashColor(null);
  }, [phase]);
  const audioEnabledRef = useRef<boolean>(audioEnabled);
  audioEnabledRef.current = audioEnabled;

  useEffect(() => {
    setAudioEnabled(config.enableAudioChime);
  }, [config.enableAudioChime]);

  // Warm neural engine & negotiate backend silently on view entry (camera stays 100% OFF)
  useEffect(() => {
    let active = true;
    if (!isAadhaarEngineWarmed()) {
      warmupAadhaarEngine()
        .then((backend) => {
          if (active) {
            setActiveBackend(backend);
            setEngineWarmed(true);
          }
        })
        .catch(() => {});
    } else {
      setActiveBackend(getActiveBackend());
    }
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (initialAttendanceId) {
      setCheckInMode('1to1');
      setAttendanceIdInput(initialAttendanceId);
    }
  }, [initialAttendanceId]);

  const normalizedQuery = attendanceIdInput.trim().toLowerCase();

  const matchedStudent = useMemo(
    () =>
      students.find((s) => s.attendanceId.toLowerCase() === normalizedQuery) ??
      null,
    [students, normalizedQuery]
  );

  const todayKey = formatDateKey(new Date());

  const existingTodayRecord = useMemo(() => {
    if (!matchedStudent) return null;
    return (
      records.find(
        (r) =>
          r.attendanceId.toLowerCase() ===
            matchedStudent.attendanceId.toLowerCase() &&
          formatDateKey(new Date(r.timestamp)) === todayKey
      ) ?? null
    );
  }, [records, matchedStudent, todayKey]);

  const enrolledCount = useMemo(
    () =>
      students.filter(
        (s) =>
          (s.faceVectors && s.faceVectors.length > 0) ||
          Boolean(s.faceDescriptor) ||
          Boolean(s.referencePhotoUrl)
      ).length,
    [students]
  );

  /**
   * Critical Detail 4: Always stop ALL MediaStreamTracks immediately on unmount,
   * mode switch, cancel, completion, or error.
   */
  const stopCameraStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // Ignore track teardown errors
        }
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stopCameraStream();
      if (queueToastTimerRef.current !== null) {
        window.clearTimeout(queueToastTimerRef.current);
      }
    };
  }, [stopCameraStream]);

  // Walk-Up Queue live elapsed timer
  useEffect(() => {
    if (checkInMode !== 'walkup' || phase !== 'camera_active') {
      setQueueElapsedSec(0);
      return;
    }
    const timer = window.setInterval(() => {
      setQueueElapsedSec((s) => s + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [checkInMode, phase]);

  const handleModeSwitch = (nextMode: CheckInMode) => {
    if (nextMode === checkInMode) return;
    stopCameraStream();
    setLiveHud(null);
    setCameraErrorMessage('');
    setVerificationResult(null);
    setLastLoggedRecord(null);
    setCapturedFrameUrl(null);
    setQueueReceipt(null);
    setPhase('standby');
    setCheckInMode(nextMode);
  };

  /**
   * Lazily ensures target student(s) have L2-normalized Float32Array(128) vectors ready
   * so the live matching loop completes in < 0.4ms.
   */
  const prepareRosterVectors = useCallback(
    async (targetStudents: Student[]): Promise<EnrolledStudentVector[]> => {
      const prepared: EnrolledStudentVector[] = [];
      for (const stu of targetStudents) {
        let vecs = getStudentVectors(stu);
        if (vecs.length === 0 && stu.referencePhotoUrl) {
          try {
            const extracted = await extractDescriptorFromImageUrl(
              stu.referencePhotoUrl
            );
            if (extracted) {
              const norm = l2Normalize(extracted);
              vecs = [norm];
              onCacheStudentVectors(stu.id, [Array.from(norm)]);
            }
          } catch {
            // Skip unencodable reference image cleanly
          }
        }
        if (vecs.length > 0) {
          prepared.push({
            id: stu.id,
            attendanceId: stu.attendanceId,
            name: stu.fullName,
            vectors: vecs,
          });
        }
      }
      return prepared;
    },
    [onCacheStudentVectors]
  );

  const completeOneToOneAndLog = useCallback(
    async (
      student: Student,
      payload: {
        capturedFrameDataUrl: string;
        liveDescriptor?: number[] | null;
        livenessPassed: boolean;
        spoofReason?: string;
        qualityChecks?: BiometricQualityChecks;
        latencyMs?: number;
      }
    ) => {
      stopCameraStream();
      setCapturedFrameUrl(payload.capturedFrameDataUrl);
      setPhase('verifying');

      const now = new Date();
      const response = await verifyFaceFrame(
        {
          attendanceId: student.attendanceId,
          student,
          capturedFrameDataUrl: payload.capturedFrameDataUrl,
          liveDescriptor: payload.liveDescriptor,
          livenessPassed: payload.livenessPassed,
          spoofReason: payload.spoofReason,
          qualityChecks: payload.qualityChecks,
          latencyMs: payload.latencyMs,
          timestamp: now.toISOString(),
        },
        config
      );

      if (response.extractedReferenceVectors && !student.faceVectors?.length) {
        onCacheStudentVectors(student.id, response.extractedReferenceVectors);
      }

      // One accepted check-in per student inside the cooldown window (no duplicate log spam)
      const prior = response.verified
        ? findPriorCheckIn(
            recordsRef.current,
            student.attendanceId,
            now,
            config.checkInCooldownMinutes
          )
        : null;
      if (prior) {
        const when = new Date(prior.timestamp).toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        });
        if (audioEnabledRef.current) playVerificationSuccessChime();
        setVerificationResult({
          ...response,
          message: `Already marked ${prior.status} at ${when}. No duplicate record created.`,
        });
        setLastLoggedRecord(prior);
        setPhase('completed');
        return;
      }

      // Weak (not absent) screen-light reflection: never auto-accept, send to operator review
      const needsReview =
        response.verified && payload.qualityChecks?.flashTier === 'weak';
      const status: AttendanceStatus = needsReview
        ? 'flagged'
        : determineAttendanceStatus(response.verified, now, config.lateCutoffTime);

      const newRecord: AttendanceRecord = {
        id: `rec-${Date.now()}`,
        attendanceId: student.attendanceId,
        studentName: student.fullName,
        department: student.department,
        timestamp: now.toISOString(),
        status,
        confidence: response.confidence,
        euclideanDistance: response.euclideanDistance,
        latencyMs: response.processingMs,
        checkInMode: '1to1',
        livenessPassed: response.livenessPassed,
        qualityChecks: response.qualityChecks,
        capturedFrameUrl: payload.capturedFrameDataUrl,
        referencePhotoUrl: student.referencePhotoUrl,
        verificationMode: response.mode,
        notes: needsReview
          ? `${response.message} · Light check weak — operator review required.`
          : response.message,
      };

      if (audioEnabledRef.current) {
        if (response.verified) playVerificationSuccessChime();
        else playVerificationFlagTone();
      }

      setVerificationResult(response);
      setLastLoggedRecord(newRecord);
      onRecordAttendance(newRecord);
      setPhase('completed');
    },
    [
      config,
      onCacheStudentVectors,
      onRecordAttendance,
      stopCameraStream,
    ]
  );

  /**
   * Live Biometric Loop Effect:
   * - In 1:1 Mode: runs until randomized 3D live challenge + blink locks, then immediately shuts down camera.
   * - In 1:N Walk-Up Queue Mode: runs continuously, logs each student with a 4s dedup cooldown, plays chime,
   *   shows a non-blocking receipt banner, and immediately resets for the next student in line.
   */
  useEffect(() => {
    if (phase !== 'camera_active') return;
    if (checkInMode === '1to1' && !matchedStudent) return;

    const videoEl = videoRef.current;
    const stream = streamRef.current;
    if (!videoEl || !stream) return;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    videoEl.srcObject = stream;
    videoEl.play().catch(() => {});

    let cancelled = false;

    void (async () => {
      try {
        const engine = new AadhaarSpeedAttendanceEngine({
          matchThreshold: config.maxEuclideanDistance,
          impostorMargin: config.impostorMargin,
        });

        // 1. Hardware virtual camera guard
        const hw = engine.verifyPhysicalCamera(stream);
        if (!hw.valid) {
          stopCameraStream();
          setCameraErrorMessage(hw.reason || 'Virtual camera blocked.');
          setPhase('camera_error');
          return;
        }

        // 1b. Best-effort manual white-balance/exposure so the light check isn't cancelled by auto-WB
        await engine.tryLockCameraControls(stream);

        // 2. Ensure models + negotiated backend are warmed
        const resolvedBackend = await warmupAadhaarEngine();
        if (!cancelled) {
          setActiveBackend(resolvedBackend);
          setEngineWarmed(true);
        }
        if (cancelled || controller.signal.aborted) return;

        // 3. Prepare L2-normalized target vector(s)
        setIndexingRoster(true);
        const targets =
          checkInMode === '1to1' && matchedStudent
            ? [matchedStudent]
            : studentsRef.current;
        const rosterVectors = await prepareRosterVectors(targets);
        if (!cancelled) setIndexingRoster(false);

        if (cancelled || controller.signal.aborted) return;

        if (checkInMode === 'walkup' && rosterVectors.length === 0) {
          stopCameraStream();
          setCameraErrorMessage(
            'No students have enrolled biometric portraits yet. Enroll at least one student in Register Student before starting Walk-Up Queue.'
          );
          setPhase('camera_error');
          return;
        }

        const framesReady = await waitForVideoFrames(videoEl, controller.signal);
        if (!framesReady || cancelled || controller.signal.aborted) {
          if (!cancelled && !controller.signal.aborted) {
            stopCameraStream();
            setCameraErrorMessage(
              'Camera stream did not produce video frames. Retry or check device permissions.'
            );
            setPhase('camera_error');
          }
          return;
        }

        engine.resetSession();

        const waitTick = () =>
          new Promise<void>((resolve) => {
            if (controller.signal.aborted) {
              resolve();
              return;
            }
            const timer = window.setTimeout(resolve, 90);
            const v = videoEl as HTMLVideoElement & {
              requestVideoFrameCallback?: (cb: () => void) => number;
            };
            if (typeof v.requestVideoFrameCallback === 'function') {
              v.requestVideoFrameCallback(() => {
                window.clearTimeout(timer);
                resolve();
              });
            } else {
              requestAnimationFrame(() => {
                window.clearTimeout(timer);
                resolve();
              });
            }
          });

        // Main frame loop
        while (!cancelled && !controller.signal.aborted) {
          await waitTick();
          if (cancelled || controller.signal.aborted) break;

          // If in 1:1 mode and student has no reference vector yet, pass a placeholder
          // so liveness runs and adapter returns a clear guidance receipt
          const effectiveRoster: EnrolledStudentVector[] =
            rosterVectors.length > 0
              ? rosterVectors
              : matchedStudent
              ? [
                  {
                    id: matchedStudent.id,
                    attendanceId: matchedStudent.attendanceId,
                    name: matchedStudent.fullName,
                    vectors: [new Float32Array(128)],
                  },
                ]
              : [];

          const result = await engine.processVideoFrame(videoEl, effectiveRoster, {
            requireLiveness: config.requireLiveness,
            mirror: config.mirrorCamera,
            flash: {
              enabled: config.requireLiveness && config.flashSlots > 0,
              slots: config.flashSlots,
              setOverlay: setFlashColor,
            },
            onHudUpdate: (hud) => {
              if (!cancelled) setLiveHud(hud);
            },
          });

          if (!result || cancelled || controller.signal.aborted) continue;

          // --- 1:1 MODE COMPLETION (Shuts off camera immediately) ---
          if (checkInMode === '1to1' && matchedStudent) {
            await completeOneToOneAndLog(matchedStudent, {
              capturedFrameDataUrl: result.capturedFrameDataUrl,
              liveDescriptor: result.liveDescriptor,
              livenessPassed: result.spoofPassed,
              spoofReason: result.spoofPassed ? undefined : result.failReason,
              qualityChecks: result.qualityChecks,
              latencyMs: result.latencyMs,
            });
            return;
          }

          // --- 1:N WALK-UP QUEUE MODE COMPLETION (Continuous auto-reset) ---
          if (checkInMode === 'walkup') {
            const nowMs = Date.now();
            if (result.matched && result.studentId) {
              const stu = studentsRef.current.find(
                (s) => s.id === result.studentId
              );
              if (stu) {
                const lastCheckMs =
                  queueCooldownMapRef.current.get(stu.id) ?? 0;
                const priorCheckIn = findPriorCheckIn(
                  recordsRef.current,
                  stu.attendanceId,
                  new Date(nowMs),
                  config.checkInCooldownMinutes
                );
                if (!priorCheckIn && nowMs - lastCheckMs > 4000) {
                  queueCooldownMapRef.current.set(stu.id, nowMs);
                  const nowDate = new Date();
                  const needsReview = result.needsReview === true;
                  const status: AttendanceStatus = needsReview
                    ? 'flagged'
                    : determineAttendanceStatus(true, nowDate, config.lateCutoffTime);
                  const record: AttendanceRecord = {
                    id: `rec-${nowMs}`,
                    attendanceId: stu.attendanceId,
                    studentName: stu.fullName,
                    department: stu.department,
                    timestamp: nowDate.toISOString(),
                    status,
                    confidence: result.confidence,
                    euclideanDistance: result.distance,
                    latencyMs: result.latencyMs,
                    checkInMode: 'walkup',
                    livenessPassed: result.spoofPassed,
                    qualityChecks: result.qualityChecks,
                    capturedFrameUrl: result.capturedFrameDataUrl,
                    referencePhotoUrl: stu.referencePhotoUrl,
                    verificationMode: config.modelMode,
                    notes: `Walk-Up Queue 1:N match & live check confirmed (dist ${result.distance.toFixed(3)} ≤ ${config.maxEuclideanDistance.toFixed(2)}).${needsReview ? ' Light check weak — operator review required.' : ''}`,
                  };

                  onRecordAttendance(record);
                  if (audioEnabledRef.current) playVerificationSuccessChime();

                  const toast: QueueReceiptToast = {
                    id: record.id,
                    studentName: stu.fullName,
                    attendanceId: stu.attendanceId,
                    avatarUrl: stu.referencePhotoUrl,
                    status,
                    confidence: result.confidence,
                    distance: result.distance,
                    latencyMs: result.latencyMs,
                    matched: true,
                    message: needsReview
                      ? 'Flagged — light check weak, operator review'
                      : status === 'late'
                      ? 'Logged (Late Arrival)'
                      : 'Attendance Verified',
                  };

                  setQueueReceipt(toast);
                  setQueueSessionLog((prev) => [toast, ...prev].slice(0, 12));
                  if (queueToastTimerRef.current !== null) {
                    window.clearTimeout(queueToastTimerRef.current);
                  }
                  queueToastTimerRef.current = window.setTimeout(() => {
                    setQueueReceipt((curr) =>
                      curr?.id === toast.id ? null : curr
                    );
                  }, 3200);
                } else {
                  const cooldownToast: QueueReceiptToast = {
                    id: `cool-${nowMs}`,
                    studentName: stu.fullName,
                    attendanceId: stu.attendanceId,
                    avatarUrl: stu.referencePhotoUrl,
                    status: 'present',
                    confidence: result.confidence,
                    distance: result.distance,
                    latencyMs: result.latencyMs,
                    matched: true,
                    message: priorCheckIn
                      ? `Already marked at ${new Date(priorCheckIn.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
                      : 'Already marked · 4s cooldown active',
                  };
                  setQueueReceipt(cooldownToast);
                  if (queueToastTimerRef.current !== null) {
                    window.clearTimeout(queueToastTimerRef.current);
                  }
                  queueToastTimerRef.current = window.setTimeout(() => {
                    setQueueReceipt((curr) =>
                      curr?.id === cooldownToast.id ? null : curr
                    );
                  }, 1800);
                }
              }
            } else {
              if (audioEnabledRef.current) playVerificationFlagTone();
              const missToast: QueueReceiptToast = {
                id: `miss-${nowMs}`,
                studentName: 'Unmatched Face',
                attendanceId: 'Use 1:1 ID',
                avatarUrl: null,
                status: 'flagged',
                confidence: result.confidence,
                distance: result.distance,
                latencyMs: result.latencyMs,
                matched: false,
                message:
                  result.failReason || 'Not matched — step up again or use 1:1 ID',
              };
              setQueueReceipt(missToast);
              if (queueToastTimerRef.current !== null) {
                window.clearTimeout(queueToastTimerRef.current);
              }
              queueToastTimerRef.current = window.setTimeout(() => {
                setQueueReceipt((curr) =>
                  curr?.id === missToast.id ? null : curr
                );
              }, 2600);
            }

            // Short handoff pause before arming for the next student in queue
            await new Promise((r) => window.setTimeout(r, 900));
            engine.resetSession();
          }
        }
      } catch (err) {
        if (cancelled || controller.signal.aborted) return;
        stopCameraStream();
        setCameraErrorMessage(
          err instanceof Error
            ? err.message
            : 'Unexpected biometric camera verification error.'
        );
        setPhase('camera_error');
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [
    phase,
    checkInMode,
    matchedStudent,
    config.maxEuclideanDistance,
    config.impostorMargin,
    config.requireLiveness,
    config.mirrorCamera,
    config.lateCutoffTime,
    config.modelMode,
    completeOneToOneAndLog,
    onRecordAttendance,
    prepareRosterVectors,
    stopCameraStream,
  ]);

  /**
   * Camera is STRICTLY activated ONLY when user clicks "Mark Attendance" (1:1)
   * or explicitly clicks "Start Queue Session" (1:N).
   * Requests 640x480 @ 30fps by default for sub-second WebGL frame latency.
   */
  const handleStartCamera = async () => {
    if (checkInMode === '1to1' && !matchedStudent) return;

    setCameraErrorMessage('');
    setVerificationResult(null);
    setLastLoggedRecord(null);
    setCapturedFrameUrl(null);
    setLiveHud(null);
    setQueueReceipt(null);
    setPhase('requesting_camera');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraErrorMessage(
        'Camera API is unavailable in this browser context. Upload a clear frontal snapshot in 1:1 mode to verify via 128-D matching.'
      );
      setPhase('camera_error');
      return;
    }

    try {
      const videoConstraints: MediaTrackConstraints =
        config.cameraResolution === '720p'
          ? {
              facingMode: 'user',
              width: { ideal: 1280 },
              height: { ideal: 720 },
              frameRate: { ideal: 30 },
            }
          : {
              facingMode: 'user',
              width: { ideal: 640 },
              height: { ideal: 480 },
              frameRate: { ideal: 30 },
            };

      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });

      streamRef.current = stream;
      setPhase('camera_active');
    } catch (err) {
      stopCameraStream();
      const message =
        err instanceof Error && err.name === 'NotAllowedError'
          ? 'Camera permission was denied. Allow camera access in your browser settings or upload a snapshot.'
          : 'No active physical webcam detected on this device. Connect a camera or upload a snapshot to verify.';
      setCameraErrorMessage(message);
      setPhase('camera_error');
    }
  };

  const handleCancelCamera = () => {
    stopCameraStream();
    setLiveHud(null);
    setQueueReceipt(null);
    setPhase('standby');
    setCameraErrorMessage('');
  };

  const handleResetToStandby = () => {
    stopCameraStream();
    setLiveHud(null);
    setPhase('standby');
    setCapturedFrameUrl(null);
    setVerificationResult(null);
    setLastLoggedRecord(null);
    setCameraErrorMessage('');
  };

  const handleFallbackFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !matchedStudent) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        void completeOneToOneAndLog(matchedStudent, {
          capturedFrameDataUrl: reader.result,
          liveDescriptor: null,
          livenessPassed: !config.requireLiveness,
          qualityChecks: {
            hardwareValid: false,
            singleFace: true,
            lightingSharpness: true,
            screenBorderClear: false,
            moirePassed: true,
            microTremorPassed: false,
            depthParallaxPassed: false,
            blinkVerified: false,
            challengeType: 'none',
            challengePassed: false,
            reactionTimeMs: null,
            reflectionDelta: null,
          },
        });
      }
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const isCameraLive =
    phase === 'camera_active' || phase === 'requesting_camera';

  const reticleColorClass = liveHud?.shouldFlashUI
    ? 'border-[#FFF9E6] shadow-[0_0_24px_rgba(255,244,206,0.9)]'
    : liveHud?.qualityStatus === 'locked'
    ? 'border-emerald-400'
    : liveHud?.qualityStatus === 'warn'
    ? 'border-amber-400'
    : 'border-[#E8CB80]/95';

  const formatTimer = (sec: number) => {
    const m = String(Math.floor(sec / 60)).padStart(2, '0');
    const s = String(sec % 60).padStart(2, '0');
    return `${m}:${s}`;
  };

  return (
    <div className="space-y-5">
      {flashColor && (
        <div
          aria-hidden="true"
          className="fixed inset-0 z-[9999] pointer-events-none"
          style={{ background: flashColor }}
        />
      )}
      {/* Workspace Mode Bar: 1:1 Attendance ID (Default) vs 1:N Walk-Up Queue */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div
          role="group"
          aria-label="Attendance verification mode"
          className="inline-flex items-center gap-1 p-1 rounded-xl neu-inset"
        >
          <button
            type="button"
            onClick={() => handleModeSwitch('1to1')}
            className={`h-8 px-3.5 rounded-lg text-xs font-medium inline-flex items-center gap-1.5 transition-all whitespace-nowrap shrink-0 cursor-pointer ${
              checkInMode === '1to1'
                ? 'neu-raised-sm text-slate-900 dark:text-[#E5C675] font-semibold'
                : 'text-slate-500 dark:text-stone-400 hover:text-slate-800 dark:hover:text-stone-200'
            }`}
          >
            <UserCheck className="w-3.5 h-3.5" />
            1:1 Attendance ID
          </button>
          <button
            type="button"
            onClick={() => handleModeSwitch('walkup')}
            className={`h-8 px-3.5 rounded-lg text-xs font-medium inline-flex items-center gap-1.5 transition-all whitespace-nowrap shrink-0 cursor-pointer ${
              checkInMode === 'walkup'
                ? 'neu-raised-sm text-slate-900 dark:text-[#E5C675] font-semibold'
                : 'text-slate-500 dark:text-stone-400 hover:text-slate-800 dark:hover:text-stone-200'
            }`}
          >
            <Users className="w-3.5 h-3.5" />
            Walk-Up Queue (1:N)
          </button>
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-stone-400">
          <span className="font-mono tabular-nums">
            {config.cameraResolution} @ 30fps · Dist ≤{' '}
            {config.maxEuclideanDistance.toFixed(2)}
          </span>
          <button
            type="button"
            onClick={() => setAudioEnabled((v) => !v)}
            aria-label={
              audioEnabled ? 'Mute verification chime' : 'Enable verification chime'
            }
            className="h-8 px-2.5 rounded-lg neu-btn text-slate-600 dark:text-stone-300 inline-flex items-center gap-1.5 cursor-pointer whitespace-nowrap"
          >
            {audioEnabled ? (
              <>
                <Volume2 className="w-3.5 h-3.5 text-blue-600 dark:text-[#E2C275]" />
                Chime On
              </>
            ) : (
              <>
                <VolumeX className="w-3.5 h-3.5" />
                Muted
              </>
            )}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left Column: 1:1 ID Lookup OR 1:N Walk-Up Queue Control Deck */}
        <section className="lg:col-span-5 neu-surface gold-edge-top rounded-2xl p-6 flex flex-col justify-between">
          {checkInMode === '1to1' ? (
            <>
              <div className="space-y-6">
                <div>
                  <h2 className="font-display text-lg font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6]">
                    Verify with Student PIN
                  </h2>
                  <p className="text-xs leading-relaxed text-slate-500 dark:text-stone-400 mt-1">
                    Enter a registered Student PIN (e.g. 23210-ec-053). Camera stays off until you press Mark Attendance, then verifies with a randomized 3D live check.
                  </p>
                </div>

                {/* Recessed Input Well */}
                <div>
                  <label
                    htmlFor="attendance-id-input"
                    className="block text-xs font-medium text-slate-600 dark:text-stone-300 mb-2"
                  >
                    Student PIN Number
                  </label>
                  <div className="relative">
                    <input
                      id="attendance-id-input"
                      type="text"
                      value={attendanceIdInput}
                      disabled={isCameraLive || phase === 'verifying'}
                      onChange={(e) => {
                        setAttendanceIdInput(e.target.value.toLowerCase());
                        if (phase === 'completed' || phase === 'camera_error') {
                          handleResetToStandby();
                        }
                      }}
                      placeholder="23210-ec-053"
                      autoComplete="off"
                      spellCheck={false}
                      className="w-full h-12 px-4 pr-10 rounded-xl font-mono text-sm tracking-wider neu-inset text-slate-900 dark:text-[#F5EFE2] placeholder:text-slate-400 dark:placeholder:text-stone-600 focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45 disabled:opacity-60 transition-all"
                    />
                    {attendanceIdInput &&
                      !isCameraLive &&
                      phase !== 'verifying' && (
                        <button
                          type="button"
                          onClick={() => {
                            setAttendanceIdInput('');
                            handleResetToStandby();
                          }}
                          aria-label="Clear Student PIN"
                          className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-stone-200 cursor-pointer"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      )}
                  </div>

                  {students.length > 0 && (
                    <div className="mt-3.5">
                      <p className="text-[11px] text-slate-500 dark:text-stone-400 mb-2">
                        Quick select PIN
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {students.slice(0, 5).map((stu) => {
                          const active =
                            normalizedQuery === stu.attendanceId.toLowerCase();
                          return (
                            <button
                              key={stu.id}
                              type="button"
                              disabled={isCameraLive || phase === 'verifying'}
                              onClick={() => {
                                setAttendanceIdInput(stu.attendanceId);
                                if (phase !== 'standby') handleResetToStandby();
                              }}
                              className={`px-2.5 py-1.5 rounded-lg font-mono text-[11px] transition-all whitespace-nowrap cursor-pointer disabled:opacity-50 ${
                                active
                                  ? 'neu-inset text-blue-700 dark:text-[#E5C675] font-semibold'
                                  : 'neu-btn text-slate-600 dark:text-stone-300'
                              }`}
                            >
                              {stu.attendanceId}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>

                <div className="neu-divider" />

                {/* Matched Student Profile */}
                <div>
                  {matchedStudent ? (
                    <div className="space-y-3">
                      <div className="flex items-center gap-4">
                        <Avatar
                          src={matchedStudent.referencePhotoUrl}
                          name={matchedStudent.fullName}
                          size="lg"
                        />
                        <div className="min-w-0 flex-1">
                          <p className="font-display text-base font-semibold text-slate-900 dark:text-[#F6F1E6] truncate">
                            {matchedStudent.fullName}
                          </p>
                          <p className="text-xs text-slate-500 dark:text-stone-400 truncate mt-0.5">
                            {matchedStudent.department} · {matchedStudent.cohort}
                          </p>
                          <p className="text-[11px] text-slate-500 dark:text-stone-400 mt-1 font-mono tabular-nums">
                            {matchedStudent.attendanceId} ·{' '}
                            {matchedStudent.faceVectors &&
                            matchedStudent.faceVectors.length > 1
                              ? '3-Vector L2 Burst Ready'
                              : matchedStudent.faceDescriptor
                              ? '128-D L2 Vector Ready'
                              : matchedStudent.referencePhotoUrl
                              ? 'Portrait Enrolled'
                              : 'No Reference Face'}
                          </p>
                        </div>
                      </div>

                      {!matchedStudent.referencePhotoUrl &&
                        !matchedStudent.faceDescriptor &&
                        !matchedStudent.faceVectors?.length &&
                        phase === 'standby' && (
                          <p className="text-xs text-amber-700 dark:text-amber-400 flex items-center gap-1.5 pt-1">
                            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                            No reference face enrolled. Capture a biometric burst in Register Student first.
                          </p>
                        )}

                      {existingTodayRecord && phase === 'standby' && (
                        <p className="text-xs text-amber-700 dark:text-amber-400 flex items-center gap-1.5 pt-1">
                          <Clock className="w-3.5 h-3.5 shrink-0" />
                          Checked in today at{' '}
                          {new Date(
                            existingTodayRecord.timestamp
                          ).toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                          . Marking again logs a new entry.
                        </p>
                      )}
                    </div>
                  ) : normalizedQuery.length > 0 ? (
                    <div className="space-y-1.5">
                      <p className="text-sm font-medium text-slate-800 dark:text-stone-200">
                        No student matched &ldquo;{normalizedQuery}&rdquo;
                      </p>
                      <p className="text-xs text-slate-500 dark:text-stone-400">
                        Check the Student PIN (e.g. 23210-ec-053) or enroll this student first.
                      </p>
                      <button
                        type="button"
                        onClick={onGoToRegister}
                        className="text-xs font-medium text-blue-600 dark:text-[#E2C275] hover:underline cursor-pointer pt-1"
                      >
                        Register new student
                      </button>
                    </div>
                  ) : students.length === 0 ? (
                    <div className="space-y-2">
                      <p className="text-xs text-slate-600 dark:text-stone-300 leading-relaxed">
                        No students are registered yet. Enroll your first student with a PIN (e.g. 23210-ec-053) and a 3-frame biometric burst to start marking attendance.
                      </p>
                      <button
                        type="button"
                        onClick={onGoToRegister}
                        className="text-xs font-semibold text-blue-600 dark:text-[#E2C275] hover:underline cursor-pointer"
                      >
                        Go to Register Student →
                      </button>
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 dark:text-stone-400 leading-relaxed">
                      Enter a registered Student PIN to load the student profile and unlock Aadhaar-grade 3D live anti-spoof verification.
                    </p>
                  )}
                </div>
              </div>

              {/* 1:1 Trigger Footer */}
              <div className="mt-6 pt-5 space-y-3">
                <div className="neu-divider mb-4" />

                {phase === 'standby' && (
                  <button
                    type="button"
                    disabled={!matchedStudent}
                    onClick={handleStartCamera}
                    className={`w-full h-12 px-4 rounded-xl text-sm font-semibold inline-flex items-center justify-center gap-2 transition-all cursor-pointer disabled:cursor-not-allowed ${
                      matchedStudent
                        ? 'neu-btn-primary'
                        : 'neu-inset text-slate-400 dark:text-stone-500'
                    }`}
                  >
                    <Camera className="w-4 h-4" />
                    Mark Attendance
                  </button>
                )}

                {phase === 'requesting_camera' && (
                  <button
                    type="button"
                    onClick={handleCancelCamera}
                    className="w-full h-12 px-4 rounded-xl text-sm font-medium neu-btn text-slate-700 dark:text-stone-200 inline-flex items-center justify-center gap-2 cursor-pointer"
                  >
                    Cancel camera request
                  </button>
                )}

                {phase === 'camera_active' && (
                  <button
                    type="button"
                    onClick={handleCancelCamera}
                    className="w-full h-12 px-4 rounded-xl text-sm font-medium neu-btn text-slate-700 dark:text-stone-200 inline-flex items-center justify-center gap-2 cursor-pointer"
                  >
                    <X className="w-4 h-4" />
                    Cancel &amp; Power Off Camera
                  </button>
                )}

                {phase === 'verifying' && (
                  <div className="w-full h-12 px-4 rounded-xl text-sm font-medium neu-inset text-slate-600 dark:text-stone-300 inline-flex items-center justify-center gap-2">
                    <RefreshCw className="w-4 h-4 animate-spin text-blue-600 dark:text-[#E2C275]" />
                    Comparing 128-D L2 Vector...
                  </div>
                )}

                {phase === 'completed' && (
                  <button
                    type="button"
                    onClick={() => {
                      setAttendanceIdInput('');
                      handleResetToStandby();
                    }}
                    className="w-full h-12 px-4 rounded-xl text-sm font-semibold neu-btn-ink inline-flex items-center justify-center gap-2 cursor-pointer"
                  >
                    <UserCheck className="w-4 h-4" />
                    Mark Next Student
                  </button>
                )}

                {phase === 'camera_error' && (
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={handleStartCamera}
                      className="flex-1 h-12 px-4 rounded-xl text-sm font-semibold neu-btn-primary inline-flex items-center justify-center gap-2 cursor-pointer"
                    >
                      <RefreshCw className="w-4 h-4" />
                      Retry Scan
                    </button>
                    <button
                      type="button"
                      onClick={handleResetToStandby}
                      className="h-12 px-4 rounded-xl text-sm font-medium neu-btn text-slate-700 dark:text-stone-200 cursor-pointer"
                    >
                      Standby
                    </button>
                  </div>
                )}

                <p className="text-[11px] text-slate-500 dark:text-stone-400 flex items-center justify-between gap-2 pt-0.5">
                  <span>3D Parallax + Randomized Live Check</span>
                  <span className="font-mono tabular-nums">
                    {engineWarmed
                      ? `${getBackendDisplayLabel(activeBackend)} Ready`
                      : 'Warming Engine...'}
                  </span>
                </p>
              </div>
            </>
          ) : (
            /* 1:N WALK-UP QUEUE MODE PANEL */
            <>
              <div className="space-y-5">
                <div>
                  <h2 className="font-display text-lg font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6]">
                    Walk-Up Kiosk Queue (1:N)
                  </h2>
                  <p className="text-xs leading-relaxed text-slate-500 dark:text-stone-400 mt-1">
                    Hands-free morning attendance. Students step up, complete the instant randomized blink/3D prompt, hear the chime, and move on with zero typing.
                  </p>
                </div>

                {/* Live Queue Summary Metrics */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="rounded-xl neu-inset p-3">
                    <p className="text-[11px] text-slate-500 dark:text-stone-400">
                      Roster Ready
                    </p>
                    <p className="font-mono text-lg font-semibold text-slate-900 dark:text-[#F6F1E6] mt-0.5 tabular-nums">
                      {enrolledCount}
                    </p>
                  </div>
                  <div className="rounded-xl neu-inset p-3">
                    <p className="text-[11px] text-slate-500 dark:text-stone-400">
                      Checked In
                    </p>
                    <p className="font-mono text-lg font-semibold text-emerald-700 dark:text-emerald-400 mt-0.5 tabular-nums">
                      {queueSessionLog.length}
                    </p>
                  </div>
                  <div className="rounded-xl neu-inset p-3">
                    <p className="text-[11px] text-slate-500 dark:text-stone-400">
                      Session
                    </p>
                    <p className="font-mono text-lg font-semibold text-slate-900 dark:text-[#E5C675] mt-0.5 tabular-nums">
                      {formatTimer(queueElapsedSec)}
                    </p>
                  </div>
                </div>

                <div className="neu-divider" />

                {/* Recent Walk-Up Check-ins */}
                <div className="space-y-2.5">
                  <p className="text-xs font-medium text-slate-600 dark:text-stone-300">
                    Live Queue Activity
                  </p>
                  {queueSessionLog.length === 0 ? (
                    <p className="text-xs text-slate-500 dark:text-stone-400 leading-relaxed py-2">
                      Press <span className="font-medium text-slate-800 dark:text-[#E5C675]">Start Queue Session</span> to activate the kiosk sensor. Impostor margin ({config.impostorMargin.toFixed(2)}) protects against lookalike collisions.
                    </p>
                  ) : (
                    <div className="space-y-2 max-h-[210px] overflow-y-auto pr-1">
                      {queueSessionLog.map((item) => (
                        <div
                          key={item.id}
                          className="flex items-center justify-between gap-3 py-2 border-b border-slate-300/30 dark:border-[#D4AF37]/10 last:border-b-0"
                        >
                          <div className="flex items-center gap-2.5 min-w-0">
                            <Avatar
                              src={item.avatarUrl}
                              name={item.studentName}
                              size="xs"
                            />
                            <div className="min-w-0">
                              <p className="text-xs font-medium text-slate-900 dark:text-[#F6F1E6] truncate">
                                {item.studentName}
                              </p>
                              <p className="text-[11px] font-mono text-slate-500 dark:text-stone-400 tabular-nums">
                                {item.attendanceId} · {item.latencyMs}ms
                              </p>
                            </div>
                          </div>
                          <span className="text-[11px] font-mono font-medium text-emerald-700 dark:text-emerald-400 tabular-nums shrink-0">
                            d={item.distance.toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* 1:N Queue Session Controls */}
              <div className="mt-6 pt-5 space-y-3">
                <div className="neu-divider mb-4" />

                {phase === 'standby' || phase === 'camera_error' ? (
                  <button
                    type="button"
                    disabled={enrolledCount === 0}
                    onClick={handleStartCamera}
                    className={`w-full h-12 px-4 rounded-xl text-sm font-semibold inline-flex items-center justify-center gap-2 transition-all cursor-pointer disabled:cursor-not-allowed ${
                      enrolledCount > 0
                        ? 'neu-btn-primary'
                        : 'neu-inset text-slate-400 dark:text-stone-500'
                    }`}
                  >
                    <Play className="w-4 h-4" />
                    Start Queue Session
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleCancelCamera}
                    className="w-full h-12 px-4 rounded-xl text-sm font-semibold neu-btn text-red-600 dark:text-red-400 inline-flex items-center justify-center gap-2 cursor-pointer"
                  >
                    <Square className="w-3.5 h-3.5 fill-current" />
                    Stop Queue &amp; Power Off Camera
                  </button>
                )}

                <p className="text-[11px] text-slate-500 dark:text-stone-400 flex items-center justify-between gap-2 pt-0.5">
                  <span>1:N L2 Matcher (&lt;0.4ms) · Margin ≥ {config.impostorMargin.toFixed(2)}</span>
                  <span className="font-mono tabular-nums">
                    {indexingRoster
                      ? 'Indexing...'
                      : engineWarmed
                      ? 'Ready'
                      : 'Warming...'}
                  </span>
                </p>
              </div>
            </>
          )}
        </section>

        {/* Right Column: Precision Hardware-Inspired Optical Housing */}
        <section className="lg:col-span-7 neu-surface gold-edge-top rounded-2xl p-6 flex flex-col">
          <div className="flex items-center justify-between gap-4 pb-4">
            <div>
              <h2 className="font-display text-base font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6]">
                {checkInMode === 'walkup'
                  ? 'Walk-Up Queue Optical Stage'
                  : 'Aadhaar FaceRD Optical Stage'}
              </h2>
              <p className="text-xs text-slate-500 dark:text-stone-400 mt-0.5">
                {phase === 'camera_active'
                  ? indexingRoster
                    ? 'Indexing enrolled portraits into L2 vectors...'
                    : liveHud?.instruction || 'Tracking frontal landmarks...'
                  : phase === 'requesting_camera'
                  ? 'Activating physical webcam sensor...'
                  : phase === 'verifying'
                  ? 'Camera powered off · Matching pre-blink winner vector'
                  : phase === 'completed'
                  ? 'Camera powered off · Verification complete'
                  : engineWarmed
                  ? `Camera off (standby) · ${getBackendDisplayLabel(activeBackend)} pre-warmed`
                  : 'Camera off (standby) · Warming up neural engine'}
              </p>
            </div>
            <span
              className={`text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap ${
                phase === 'camera_active'
                  ? 'text-blue-600 dark:text-[#E2C275]'
                  : phase === 'completed' && verificationResult?.verified
                  ? 'text-emerald-700 dark:text-emerald-400'
                  : phase === 'camera_error' ||
                    (phase === 'completed' && !verificationResult?.verified)
                  ? 'text-amber-700 dark:text-amber-400'
                  : 'text-slate-500 dark:text-stone-400'
              }`}
            >
              {phase === 'camera_active' ? (
                <>
                  <Camera className="w-3.5 h-3.5" />
                  Live Sensor
                </>
              ) : (
                <>
                  <CameraOff className="w-3.5 h-3.5" />
                  Camera Off
                </>
              )}
            </span>
          </div>

          {/* Recessed Sculpted Lens Well */}
          <div
            className={`flex-1 min-h-[395px] rounded-2xl p-2.5 flex transition-all duration-150 ${
              liveHud?.shouldFlashUI
                ? 'bg-gradient-to-br from-[#FFF9E6]/90 via-[#F5E1A4]/75 to-[#FFF9E6]/90 shadow-[0_0_28px_rgba(245,225,164,0.45)]'
                : 'neu-inset'
            }`}
          >
            <div className="flex-1 rounded-xl overflow-hidden bg-[#08090C] relative flex items-center justify-center border border-[#D4AF37]/15">
              {/* STANDBY VIEW — Camera is strictly OFF */}
              {phase === 'standby' && (
                <div className="text-center px-8 py-12 max-w-md">
                  <div className="w-14 h-14 mx-auto rounded-2xl bg-[#111318] border border-[#D4AF37]/25 shadow-inner flex items-center justify-center text-[#D4AF37]/80">
                    <CameraOff className="w-6 h-6" strokeWidth={1.7} />
                  </div>
                  <h3 className="mt-4 font-display text-base font-semibold text-[#F6F1E6]">
                    {checkInMode === 'walkup'
                      ? 'Walk-Up Queue on standby'
                      : 'Camera on standby'}
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-stone-400">
                    {checkInMode === 'walkup' ? (
                      <>
                        Camera remains off until you click{' '}
                        <span className="text-[#E5C675] font-medium">
                          Start Queue Session
                        </span>
                        . Students walk up, complete a randomized 3D live prompt, and check in automatically while phone screens &amp; replays are blocked.
                      </>
                    ) : (
                      <>
                        Camera remains completely off until a registered Student PIN is matched and you press{' '}
                        <span className="text-[#E5C675] font-medium">
                          Mark Attendance
                        </span>
                        . Randomized 3D live prompt captures the sharpest open-eye frame and powers off immediately.
                      </>
                    )}
                  </p>
                  <div className="mt-6 pt-5 border-t border-[#D4AF37]/15 flex flex-wrap items-center justify-center gap-x-4 gap-y-1.5 text-[11px] text-stone-400 font-mono tabular-nums">
                    <span>Randomized 3D Prompt</span>
                    <span>·</span>
                    <span>Phone Bezel Guard</span>
                    <span>·</span>
                    <span>Zero Auto-Bypass</span>
                  </div>
                </div>
              )}

              {/* REQUESTING CAMERA */}
              {phase === 'requesting_camera' && (
                <div className="text-center px-8 py-12">
                  <RefreshCw className="w-7 h-7 mx-auto text-[#E2C275] animate-spin" />
                  <p className="mt-4 text-sm font-medium text-[#F6F1E6]">
                    Activating physical camera sensor...
                  </p>
                  <p className="mt-1 text-xs text-stone-400">
                    Locking {config.cameraResolution} @ 30fps stream and hardware check.
                  </p>
                </div>
              )}

              {/* LIVE CAMERA FEED WITH SUBTLE GOLD-WHITE REFLECTION PULSE & HUD */}
              {phase === 'camera_active' && (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`w-full h-full object-cover min-h-[375px] ${
                      config.mirrorCamera ? '-scale-x-100' : ''
                    }`}
                  />

                  {/* Subtle Gold-White Rim Reflection Frame (150ms–350ms soft illumination) */}
                  <div
                    className={`pointer-events-none absolute inset-0 rounded-xl border-4 transition-opacity duration-150 ${
                      liveHud?.shouldFlashUI
                        ? 'opacity-90 border-[#FFF7DE] shadow-[inset_0_0_48px_rgba(255,246,214,0.38)]'
                        : 'opacity-0 border-transparent'
                    }`}
                  />

                  {/* Top Biometric Gate Indicators */}
                  <div className="pointer-events-none absolute inset-x-0 top-0 bg-gradient-to-b from-black/82 via-black/40 to-transparent px-4 pt-3.5 pb-6 space-y-2.5">
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      {(
                        [
                          {
                            id: 'frontal',
                            label: 'ISO Pose',
                            done: Boolean(liveHud?.checks.frontal),
                          },
                          {
                            id: 'screenClear',
                            label: 'Bezel Clear',
                            done: Boolean(liveHud?.checks.screenClear),
                          },
                          {
                            id: 'liveness',
                            label: '3D Live Check',
                            done: Boolean(liveHud?.checks.liveness),
                          },
                          {
                            id: 'vector',
                            label: '128-D L2',
                            done: Boolean(liveHud?.checks.vector),
                          },
                        ] as const
                      ).map((gate) => (
                        <div
                          key={gate.id}
                          className="inline-flex items-center gap-1.5 text-[11px] font-medium"
                        >
                          <span
                            className={`w-4 h-4 rounded-full flex items-center justify-center border ${
                              gate.done
                                ? 'bg-emerald-500/20 border-emerald-400 text-emerald-300'
                                : 'bg-white/5 border-white/20 text-stone-500'
                            }`}
                          >
                            {gate.done ? (
                              <Check className="w-2.5 h-2.5" strokeWidth={2.5} />
                            ) : (
                              <span className="w-1.5 h-1.5 rounded-full bg-stone-500" />
                            )}
                          </span>
                          <span
                            className={
                              gate.done ? 'text-emerald-300' : 'text-stone-300'
                            }
                          >
                            {gate.label}
                          </span>
                        </div>
                      ))}
                    </div>

                    {liveHud?.activeChallenge &&
                      liveHud.activeChallenge !== 'none' &&
                      !liveHud.checks.liveness && (
                        <div className="flex justify-center">
                          <div className="px-3.5 py-1.5 rounded-lg bg-black/75 border border-[#E8CB80]/45 backdrop-blur-sm text-xs font-semibold text-[#F6E4B0] tracking-tight shadow-md">
                            Live Prompt: {getChallengePromptLabel(liveHud.activeChallenge)}
                          </div>
                        </div>
                      )}
                  </div>

                  {/* Walk-Up Queue Non-Blocking Receipt Banner */}
                  {checkInMode === 'walkup' && queueReceipt && (
                    <div className="pointer-events-none absolute inset-x-4 top-12 z-20 flex justify-center">
                      <div
                        className={`px-4 py-2.5 rounded-xl backdrop-blur-md border flex items-center gap-3 shadow-lg ${
                          queueReceipt.matched
                            ? 'bg-emerald-950/90 border-emerald-400/40 text-emerald-100'
                            : 'bg-amber-950/90 border-amber-400/40 text-amber-100'
                        }`}
                      >
                        {queueReceipt.matched ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                        ) : (
                          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                        )}
                        <div className="min-w-0">
                          <p className="text-xs font-semibold truncate">
                            {queueReceipt.studentName} · {queueReceipt.message}
                          </p>
                          <p className="text-[11px] font-mono opacity-85 tabular-nums">
                            {queueReceipt.attendanceId} · {queueReceipt.latencyMs}ms · dist{' '}
                            {queueReceipt.distance.toFixed(3)}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Minimalist Dynamic Optical Face Reticle */}
                  <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                    <div className="relative w-52 h-64 sm:w-56 sm:h-72">
                      <span
                        className={`absolute top-0 left-0 w-7 h-7 border-t-2 border-l-2 rounded-tl-2xl transition-all duration-150 ${reticleColorClass}`}
                      />
                      <span
                        className={`absolute top-0 right-0 w-7 h-7 border-t-2 border-r-2 rounded-tr-2xl transition-all duration-150 ${reticleColorClass}`}
                      />
                      <span
                        className={`absolute bottom-0 left-0 w-7 h-7 border-b-2 border-l-2 rounded-bl-2xl transition-all duration-150 ${reticleColorClass}`}
                      />
                      <span
                        className={`absolute bottom-0 right-0 w-7 h-7 border-b-2 border-r-2 rounded-br-2xl transition-all duration-150 ${reticleColorClass}`}
                      />
                      <div className="absolute inset-x-3 top-0 bottom-0 overflow-hidden">
                        <div className="w-full h-0.5 bg-[#E8CB80]/90 shadow-[0_0_12px_rgba(212,175,55,0.85)] animate-scan-sweep" />
                      </div>
                    </div>
                  </div>

                  {/* Scrim Footer Live Instruction & Progress */}
                  <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/92 via-black/65 to-transparent px-5 pb-4 pt-10 space-y-2.5">
                    <div className="flex items-end justify-between gap-4">
                      <div className="min-w-0">
                        <p
                          className={`text-sm font-semibold leading-snug ${
                            liveHud?.qualityStatus === 'warn'
                              ? 'text-amber-300'
                              : liveHud?.qualityStatus === 'locked'
                              ? 'text-emerald-300'
                              : 'text-[#F6F1E6]'
                          }`}
                        >
                          {indexingRoster
                            ? 'Preparing enrolled biometric index...'
                            : liveHud?.instruction ||
                              'Look straight and blink naturally'}
                        </p>
                        <p className="text-[11px] font-mono text-stone-300 mt-0.5 tabular-nums truncate">
                          {liveHud?.subtext ||
                            (checkInMode === 'walkup'
                              ? 'Walk-Up Queue · Next student ready'
                              : matchedStudent?.attendanceId)}
                        </p>
                      </div>
                      <span className="font-mono text-lg font-semibold text-[#E8CB80] tabular-nums shrink-0">
                        {liveHud?.progress ?? 10}%
                      </span>
                    </div>
                    <div className="h-1 w-full rounded-full bg-white/10 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-[#B88928] via-[#E8CB80] to-emerald-400 transition-all duration-150"
                        style={{ width: `${liveHud?.progress ?? 10}%` }}
                      />
                    </div>
                  </div>
                </>
              )}

              {/* VERIFYING FROZEN PRE-BLINK FRAME */}
              {phase === 'verifying' && (
                <div className="w-full h-full relative flex items-center justify-center min-h-[375px]">
                  {capturedFrameUrl && (
                    <img
                      src={capturedFrameUrl}
                      alt="Sharpest pre-blink capture frame"
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover opacity-35"
                    />
                  )}
                  <div className="absolute inset-0 bg-black/65 flex flex-col items-center justify-center px-6 text-center">
                    <RefreshCw className="w-7 h-7 text-[#E8CB80] animate-spin" />
                    <p className="mt-3 font-display text-sm font-semibold text-[#F6F1E6]">
                      Matching Pre-Blink 128-D Winner
                    </p>
                    <p className="mt-1 text-xs text-stone-300 font-mono">
                      {matchedStudent?.attendanceId} · Camera powered off
                    </p>
                  </div>
                </div>
              )}

              {/* COMPLETED 1:1 RECEIPT */}
              {phase === 'completed' &&
                verificationResult &&
                lastLoggedRecord && (
                  <div className="w-full h-full bg-[#101217] p-6 flex flex-col justify-between min-h-[375px]">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-start gap-3">
                        {verificationResult.verified ? (
                          <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                        ) : (
                          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                        )}
                        <div>
                          <h3 className="font-display text-base font-semibold text-[#F6F1E6]">
                            {verificationResult.verified
                              ? lastLoggedRecord.status === 'late'
                                ? 'Attendance Logged (Late Arrival)'
                                : 'Aadhaar FaceRD Match Verified'
                              : 'Verification Flagged for Review'}
                          </h3>
                          <p className="text-xs text-stone-400 mt-1">
                            {verificationResult.message}
                          </p>
                        </div>
                      </div>
                      <span className="font-mono text-xs font-semibold text-[#E8CB80] tabular-nums shrink-0">
                        {verificationResult.processingMs} ms
                      </span>
                    </div>

                    <div className="my-5 grid grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <p className="text-[11px] font-medium text-stone-400">
                          Enrolled Reference
                        </p>
                        <div className="aspect-[4/3] rounded-xl overflow-hidden bg-[#171A21] border border-[#D4AF37]/15 flex items-center justify-center">
                          {lastLoggedRecord.referencePhotoUrl ? (
                            <img
                              src={lastLoggedRecord.referencePhotoUrl}
                              alt={`${lastLoggedRecord.studentName} enrolled portrait`}
                              referrerPolicy="no-referrer"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <span className="text-xs text-stone-500">
                              No reference stored
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="space-y-1.5">
                        <p className="text-[11px] font-medium text-stone-400">
                          Pre-Blink Sharpest Winner
                        </p>
                        <div className="aspect-[4/3] rounded-xl overflow-hidden bg-[#171A21] border border-[#D4AF37]/15 flex items-center justify-center">
                          {capturedFrameUrl &&
                          !capturedFrameUrl.endsWith('preview-frame') ? (
                            <img
                              src={capturedFrameUrl}
                              alt="Sharpest pre-blink check-in frame"
                              referrerPolicy="no-referrer"
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <span className="text-xs text-stone-500">
                              Frame recorded
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="pt-4 border-t border-[#D4AF37]/15 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                      <div>
                        <p className="text-stone-400">Confidence</p>
                        <p className="font-mono font-semibold text-[#F6F1E6] mt-0.5 tabular-nums">
                          {verificationResult.confidence.toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-stone-400">128-D Distance</p>
                        <p className="font-mono font-semibold text-[#E8CB80] mt-0.5 tabular-nums">
                          {typeof verificationResult.euclideanDistance ===
                          'number'
                            ? verificationResult.euclideanDistance.toFixed(3)
                            : '—'}
                        </p>
                      </div>
                      <div>
                        <p className="text-stone-400">Check-in Time</p>
                        <p className="font-mono font-semibold text-[#F6F1E6] mt-0.5 tabular-nums">
                          {new Date(
                            lastLoggedRecord.timestamp
                          ).toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit',
                          })}
                        </p>
                      </div>
                      <div>
                        <p className="text-stone-400">Liveness</p>
                        <p
                          className={`font-medium mt-0.5 inline-flex items-center gap-1 ${
                            verificationResult.livenessPassed
                              ? 'text-emerald-400'
                              : 'text-amber-400'
                          }`}
                        >
                          <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
                          {verificationResult.livenessPassed
                            ? typeof verificationResult.qualityChecks
                                ?.reactionTimeMs === 'number'
                              ? `3D Live · ${verificationResult.qualityChecks.reactionTimeMs}ms`
                              : '3D Live OK'
                            : 'Failed'}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

              {/* CAMERA ERROR / SPOOF ABORT FALLBACK */}
              {phase === 'camera_error' && (
                <div className="text-center px-8 py-10 max-w-md">
                  <div className="w-12 h-12 mx-auto rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
                    <AlertTriangle className="w-5 h-5" />
                  </div>
                  <h3 className="mt-4 font-display text-base font-semibold text-[#F6F1E6]">
                    Verification halted
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-stone-400">
                    {cameraErrorMessage}
                  </p>

                  <input
                    ref={uploadInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFallbackFileUpload}
                    className="hidden"
                  />

                  <div className="mt-6 flex flex-wrap items-center justify-center gap-2.5">
                    <button
                      type="button"
                      onClick={handleStartCamera}
                      className="h-9 px-3.5 rounded-xl text-xs font-semibold bg-[#E2C275] text-[#0B0C10] hover:bg-[#EDD28F] inline-flex items-center gap-1.5 cursor-pointer"
                    >
                      <ScanFace className="w-3.5 h-3.5" />
                      Try Live Scan Again
                    </button>
                    {checkInMode === '1to1' && matchedStudent && (
                      <button
                        type="button"
                        onClick={() => uploadInputRef.current?.click()}
                        className="h-9 px-3.5 rounded-xl text-xs font-medium bg-[#1A1D24] text-stone-200 hover:bg-[#232730] border border-white/10 inline-flex items-center gap-1.5 cursor-pointer"
                      >
                        <Upload className="w-3.5 h-3.5" />
                        Match Uploaded Photo
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
