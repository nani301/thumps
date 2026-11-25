import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
} from 'react';
import {
  ArrowRight,
  Camera,
  Check,
  Copy,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  Upload,
  UserPlus,
  X,
} from 'lucide-react';
import type { Student, SystemConfig } from '../types';
import {
  BRANCH_OPTIONS,
  DEPARTMENTS,
  formatStudentPin,
  getNextAvailablePinDigits,
  normalizePinDigits,
  YEAR_COLLEGE_OPTIONS,
} from '../data/seedData';
import { enrollFaceReference } from '../services/facialModelAdapter';
import { findDuplicateFace } from '../services/livenessCore';
import {
  runEnrollmentVectorBurst,
  warmupAadhaarEngine,
  type FastHudState,
} from '../services/aadhaarFaceEngine';
import { playVerificationSuccessChime } from '../services/soundFeedback';
import { Avatar } from './Avatar';

interface RegisterStudentViewProps {
  students: Student[];
  config: SystemConfig;
  onAddStudent: (student: Student) => void;
  onDeleteStudent: (studentId: string) => void;
  onUseAttendanceId: (attendanceId: string) => void;
}

export function RegisterStudentView({
  students,
  config,
  onAddStudent,
  onDeleteStudent,
  onUseAttendanceId,
}: RegisterStudentViewProps) {
  const existingIds = useMemo(
    () => students.map((s) => s.attendanceId),
    [students]
  );

  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [yearCollegeCode, setYearCollegeCode] = useState<string>(
    YEAR_COLLEGE_OPTIONS[0].code
  );
  const [branchCode, setBranchCode] = useState<string>(BRANCH_OPTIONS[0].code);
  const [pinDigits, setPinDigits] = useState<string>(() =>
    getNextAvailablePinDigits(
      existingIds,
      YEAR_COLLEGE_OPTIONS[0].code,
      BRANCH_OPTIONS[0].code
    )
  );
  const [referencePhotoUrl, setReferencePhotoUrl] = useState<string | null>(null);
  const [capturedVectors, setCapturedVectors] = useState<number[][] | null>(null);

  const selectedYearCollege = useMemo(
    () =>
      YEAR_COLLEGE_OPTIONS.find((y) => y.code === yearCollegeCode) ??
      YEAR_COLLEGE_OPTIONS[0],
    [yearCollegeCode]
  );

  const selectedBranch = useMemo(
    () =>
      BRANCH_OPTIONS.find((b) => b.code === branchCode) ?? BRANCH_OPTIONS[0],
    [branchCode]
  );

  const assembledPin = useMemo(
    () => formatStudentPin(yearCollegeCode, branchCode, pinDigits),
    [yearCollegeCode, branchCode, pinDigits]
  );
  const [formError, setFormError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [flashColor, setFlashColor] = useState<string | null>(null); // screen-flash light check overlay

  const [newlyRegistered, setNewlyRegistered] = useState<Student | null>(null);
  const [enrollStatusMsg, setEnrollStatusMsg] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [departmentFilter, setDepartmentFilter] = useState('All');

  const [captureModalOpen, setCaptureModalOpen] = useState(false);
  const [captureError, setCaptureError] = useState('');
  const [enrollHud, setEnrollHud] = useState<FastHudState | null>(null);
  const refVideoRef = useRef<HTMLVideoElement | null>(null);
  const refStreamRef = useRef<MediaStream | null>(null);
  const enrollAbortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Silently warm up neural shaders while camera stays off
  useEffect(() => {
    warmupAadhaarEngine().catch(() => {});
  }, []);

  /**
   * Always tear down all MediaStreamTracks on cancel, completion, or unmount.
   */
  const stopReferenceCamera = useCallback(() => {
    if (enrollAbortRef.current) {
      enrollAbortRef.current.abort();
      enrollAbortRef.current = null;
    }
    if (refStreamRef.current) {
      refStreamRef.current.getTracks().forEach((t) => {
        try {
          t.stop();
        } catch {
          // Ignore stop errors
        }
      });
      refStreamRef.current = null;
    }
    if (refVideoRef.current) {
      refVideoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => {
    return () => stopReferenceCamera();
  }, [stopReferenceCamera]);

  const handleCloseReferenceCamera = useCallback(() => {
    stopReferenceCamera();
    setCaptureModalOpen(false);
    setCaptureError('');
    setEnrollHud(null);
  }, [stopReferenceCamera]);

  const handleOpenReferenceCamera = async () => {
    setCaptureError('');
    setEnrollHud(null);
    setCaptureModalOpen(true);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCaptureError(
        'Camera is unavailable in this browser. Upload a frontal portrait photo instead.'
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });
      refStreamRef.current = stream;

      const controller = new AbortController();
      enrollAbortRef.current = controller;

      window.setTimeout(async () => {
        const videoEl = refVideoRef.current;
        if (!videoEl || controller.signal.aborted) return;

        videoEl.srcObject = stream;
        await videoEl.play().catch(() => {});

        try {
          const res = await runEnrollmentVectorBurst(videoEl, stream, {
            mirror: config.mirrorCamera,
            signal: controller.signal,
            flash: {
              enabled: config.flashSlots > 0,
              slots: config.flashSlots,
              setOverlay: setFlashColor,
            },
            onHudUpdate: (hud) => {
              if (!controller.signal.aborted) setEnrollHud(hud);
            },
          });

          if (controller.signal.aborted) return;

          if (!res.ok || !res.capturedFrameDataUrl || !res.vectors?.length) {
            stopReferenceCamera();
            setCaptureError(
              res.reason ||
                'Could not lock Aadhaar frontal quality. Retry in good lighting.'
            );
            return;
          }

          if (config.enableAudioChime) {
            playVerificationSuccessChime();
          }
          setReferencePhotoUrl(res.capturedFrameDataUrl);
          setCapturedVectors(res.vectors);
          handleCloseReferenceCamera();
        } catch (err) {
          if (controller.signal.aborted) return;
          stopReferenceCamera();
          setCaptureError(
            err instanceof Error
              ? err.message
              : 'Live biometric burst capture failed.'
          );
        }
      }, 60);
    } catch {
      stopReferenceCamera();
      setCaptureError(
        'Unable to access camera. Check permissions or upload a reference image file.'
      );
    }
  };

  const handleFileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setReferencePhotoUrl(reader.result);
        setCapturedVectors(null);
      }
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const handleCopyId = async (idToCopy: string) => {
    try {
      await navigator.clipboard.writeText(idToCopy);
      setCopiedId(idToCopy);
      window.setTimeout(
        () => setCopiedId((prev) => (prev === idToCopy ? null : prev)),
        2000
      );
    } catch {
      setCopiedId(idToCopy);
      window.setTimeout(() => setCopiedId(null), 1500);
    }
  };

  const handleNextPin = () => {
    setPinDigits(
      getNextAvailablePinDigits(existingIds, yearCollegeCode, branchCode)
    );
  };

  const handleYearCollegeChange = (nextYearCollege: string) => {
    setYearCollegeCode(nextYearCollege);
    setPinDigits(
      getNextAvailablePinDigits(existingIds, nextYearCollege, branchCode)
    );
    if (newlyRegistered) setNewlyRegistered(null);
  };

  const handleBranchChange = (nextBranch: string) => {
    setBranchCode(nextBranch);
    setPinDigits(
      getNextAvailablePinDigits(existingIds, yearCollegeCode, nextBranch)
    );
    if (newlyRegistered) setNewlyRegistered(null);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setFormError('');

    const trimmedName = fullName.trim();
    const trimmedEmail = email.trim();
    const cleanDigits = pinDigits.replace(/\D/g, '');
    const formattedPin = formatStudentPin(
      yearCollegeCode,
      branchCode,
      cleanDigits
    );

    if (!trimmedName) {
      setFormError('Please enter the student full name.');
      return;
    }
    if (!trimmedEmail || !trimmedEmail.includes('@')) {
      setFormError('Please enter a valid student email address.');
      return;
    }
    if (!cleanDigits || Number.parseInt(cleanDigits, 10) <= 0) {
      setFormError('Please enter a valid 3-digit PIN number (001–999).');
      return;
    }
    if (
      students.some(
        (s) => s.attendanceId.toLowerCase() === formattedPin.toLowerCase()
      )
    ) {
      setFormError(
        `PIN ${formattedPin} is already assigned. Enter another roll number or click Next PIN.`
      );
      return;
    }

    // Policy: enrollment from an uploaded photo has no liveness proof (anyone can be enrolled from a picture)
    if (!capturedVectors?.length && referencePhotoUrl && !config.allowPhotoEnrollment) {
      setFormError(
        'Enrollment needs a LIVE liveness-checked capture. Use Live capture (photo-only enrollment is disabled: allowPhotoEnrollment).'
      );
      return;
    }

    setSubmitting(true);
    const enrollResult = await enrollFaceReference(
      formattedPin,
      referencePhotoUrl,
      config,
      capturedVectors
    );

    // De-duplication (the Aadhaar idea): one face = one ID. Blocks proxy / double enrollment.
    const duplicate = enrollResult.vectors
      ? findDuplicateFace(enrollResult.vectors, students, config.maxEuclideanDistance)
      : null;
    if (duplicate) {
      setFormError(
        `This face is already enrolled as ${duplicate.fullName} (${duplicate.attendanceId}, distance ${duplicate.distance.toFixed(2)}). One person can have only one ID.`
      );
      setSubmitting(false);
      return;
    }

    const newStudent: Student = {
      id: `stu-${Date.now()}`,
      attendanceId: formattedPin,
      fullName: trimmedName,
      email: trimmedEmail,
      department: selectedBranch.label,
      cohort: selectedYearCollege.batchLabel,
      referencePhotoUrl,
      faceDescriptor: enrollResult.descriptor,
      faceVectors: enrollResult.vectors,
      enrolledAt: new Date().toISOString(),
      biometricEnrolled: enrollResult.enrolled,
    };

    onAddStudent(newStudent);
    setNewlyRegistered(newStudent);
    setEnrollStatusMsg(enrollResult.message);
    setSubmitting(false);

    setFullName('');
    setEmail('');
    setReferencePhotoUrl(null);
    setCapturedVectors(null);
    setPinDigits(
      getNextAvailablePinDigits(
        [...existingIds, formattedPin],
        yearCollegeCode,
        branchCode
      )
    );
  };

  const filteredStudents = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return students.filter((s) => {
      const matchesDept =
        departmentFilter === 'All' || s.department === departmentFilter;
      const matchesQuery =
        !q ||
        s.fullName.toLowerCase().includes(q) ||
        s.attendanceId.toLowerCase().includes(q) ||
        s.email.toLowerCase().includes(q);
      return matchesDept && matchesQuery;
    });
  }, [students, searchQuery, departmentFilter]);

  const previewStudent = newlyRegistered ?? {
    fullName: fullName.trim() || 'Student Name',
    email: email.trim() || 'student@college210.edu',
    department: selectedBranch.label,
    cohort: selectedYearCollege.batchLabel,
    attendanceId: assembledPin,
    referencePhotoUrl,
    biometricEnrolled: Boolean(capturedVectors?.length || referencePhotoUrl),
  };

  return (
    <div className="space-y-7">
      {flashColor && (
        <div
          aria-hidden="true"
          className="fixed inset-0 z-[9999] pointer-events-none"
          style={{ background: flashColor }}
        />
      )}
      {/* Top 2-Column Registration Form & Sculpted Credential Pass */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-stretch">
        <section className="xl:col-span-7 neu-surface gold-edge-top rounded-2xl p-6">
          <div className="pb-4">
            <h2 className="font-display text-lg font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6]">
              Student Registration
            </h2>
            <p className="text-xs text-slate-500 dark:text-stone-400 mt-1">
              Enroll a student with a 3-frame L2 biometric burst and assign their structured Student PIN.
            </p>
          </div>

          <div className="neu-divider mb-5" />

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label
                  htmlFor="reg-full-name"
                  className="block text-xs font-medium text-slate-600 dark:text-stone-300 mb-1.5"
                >
                  Full Name
                </label>
                <input
                  id="reg-full-name"
                  type="text"
                  value={fullName}
                  onChange={(e) => {
                    setFullName(e.target.value);
                    if (newlyRegistered) setNewlyRegistered(null);
                  }}
                  placeholder="Enter student full name"
                  className="w-full h-10 px-3.5 rounded-xl text-sm neu-inset text-slate-900 dark:text-[#F6F1E6] placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
                />
              </div>

              <div>
                <label
                  htmlFor="reg-email"
                  className="block text-xs font-medium text-slate-600 dark:text-stone-300 mb-1.5"
                >
                  Email Address
                </label>
                <input
                  id="reg-email"
                  type="email"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                    if (newlyRegistered) setNewlyRegistered(null);
                  }}
                  placeholder="student@college210.edu"
                  className="w-full h-10 px-3.5 rounded-xl text-sm neu-inset text-slate-900 dark:text-[#F6F1E6] placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
                />
              </div>
            </div>

            {/* Structured Student PIN Builder: Year+College (23210-26210) - Branch (ec) - PIN (053) */}
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-slate-600 dark:text-stone-300">
                  Student PIN Builder (Year+College · Branch · Roll)
                </span>
                <span className="font-mono text-xs font-semibold text-blue-700 dark:text-[#E8CB80] tabular-nums">
                  {assembledPin}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-12 gap-3">
                <div className="sm:col-span-5">
                  <label
                    htmlFor="reg-year-college"
                    className="block text-[11px] font-medium text-slate-500 dark:text-stone-400 mb-1"
                  >
                    Year + College Code
                  </label>
                  <select
                    id="reg-year-college"
                    value={yearCollegeCode}
                    onChange={(e) => handleYearCollegeChange(e.target.value)}
                    className="w-full h-10 px-3 rounded-xl font-mono text-xs neu-inset text-slate-900 dark:text-[#F6F1E6] focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
                  >
                    {YEAR_COLLEGE_OPTIONS.map((opt) => (
                      <option key={opt.code} value={opt.code}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="sm:col-span-4">
                  <label
                    htmlFor="reg-branch"
                    className="block text-[11px] font-medium text-slate-500 dark:text-stone-400 mb-1"
                  >
                    Branch
                  </label>
                  <select
                    id="reg-branch"
                    value={branchCode}
                    onChange={(e) => handleBranchChange(e.target.value)}
                    className="w-full h-10 px-3 rounded-xl text-xs neu-inset text-slate-900 dark:text-[#F6F1E6] focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
                  >
                    {BRANCH_OPTIONS.map((b) => (
                      <option key={b.code} value={b.code}>
                        {b.code} — {b.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="sm:col-span-3">
                  <label
                    htmlFor="reg-pin-digits"
                    className="block text-[11px] font-medium text-slate-500 dark:text-stone-400 mb-1"
                  >
                    PIN Number
                  </label>
                  <div className="flex items-center gap-1.5">
                    <input
                      id="reg-pin-digits"
                      type="text"
                      inputMode="numeric"
                      maxLength={3}
                      value={pinDigits}
                      onChange={(e) => {
                        setPinDigits(e.target.value.replace(/\D/g, '').slice(0, 3));
                        if (newlyRegistered) setNewlyRegistered(null);
                      }}
                      onBlur={() => setPinDigits(normalizePinDigits(pinDigits))}
                      placeholder="053"
                      className="w-full h-10 px-3 rounded-xl font-mono text-sm tracking-wider neu-inset text-slate-900 dark:text-[#F6F1E6] placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
                    />
                    <button
                      type="button"
                      onClick={handleNextPin}
                      title="Auto-assign next free PIN number"
                      className="h-10 px-2.5 rounded-xl text-xs font-medium neu-btn text-slate-700 dark:text-stone-200 inline-flex items-center justify-center shrink-0 cursor-pointer"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* 3-Frame Aadhaar Biometric Burst Enrollment */}
            <div>
              <div className="flex items-center justify-between gap-2 mb-2.5">
                <span className="text-xs font-medium text-slate-600 dark:text-stone-300">
                  Biometric Reference (3-Frame L2 Burst + Centroid)
                </span>
                {capturedVectors && capturedVectors.length > 0 && (
                  <span className="text-[11px] font-medium text-emerald-700 dark:text-emerald-400 inline-flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    {capturedVectors.length > 1
                      ? '3-Vector Burst Locked'
                      : '128-D Vector Locked'}
                  </span>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <div className="flex flex-wrap items-center gap-3.5">
                <Avatar
                  src={referencePhotoUrl}
                  name={fullName || 'New Student'}
                  size="lg"
                />
                <div className="flex flex-wrap items-center gap-2.5">
                  <button
                    type="button"
                    onClick={handleOpenReferenceCamera}
                    className="h-9 px-3.5 rounded-xl text-xs font-semibold neu-btn-primary inline-flex items-center gap-1.5 cursor-pointer whitespace-nowrap"
                  >
                    <Camera className="w-3.5 h-3.5" />
                    Live 3-Frame Burst
                  </button>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="h-9 px-3.5 rounded-xl text-xs font-medium neu-btn text-slate-700 dark:text-stone-200 inline-flex items-center gap-1.5 cursor-pointer whitespace-nowrap"
                  >
                    <Upload className="w-3.5 h-3.5" />
                    Upload Photo
                  </button>
                  {referencePhotoUrl && (
                    <button
                      type="button"
                      onClick={() => {
                        setReferencePhotoUrl(null);
                        setCapturedVectors(null);
                      }}
                      className="h-9 px-2.5 rounded-xl text-xs font-medium text-slate-500 hover:text-red-600 dark:hover:text-red-400 cursor-pointer"
                    >
                      Remove
                    </button>
                  )}
                </div>
              </div>
            </div>

            {formError && (
              <p className="text-xs font-medium text-red-600 dark:text-red-400">
                {formError}
              </p>
            )}

            <div className="pt-2">
              <button
                type="submit"
                disabled={submitting}
                className="h-10 px-5 rounded-xl text-xs font-semibold neu-btn-ink disabled:opacity-60 inline-flex items-center gap-2 cursor-pointer"
              >
                <UserPlus className="w-3.5 h-3.5" />
                {submitting
                  ? 'Encoding L2 Vectors...'
                  : 'Register & Issue Attendance ID'}
              </button>
            </div>
          </form>
        </section>

        {/* Credential Preview / Issued Pass */}
        <section className="xl:col-span-5 neu-surface gold-edge-top rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="pb-4">
              <h2 className="font-display text-base font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6]">
                {newlyRegistered ? 'Attendance ID Issued' : 'Credential Preview'}
              </h2>
              <p className="text-xs text-slate-500 dark:text-stone-400 mt-0.5">
                {newlyRegistered
                  ? enrollStatusMsg
                  : 'Live preview of the student’s dedicated pass'}
              </p>
            </div>

            <div className="neu-divider mb-6" />

            {/* Tactile Matte Obsidian & Brushed Gold Hardware Pass */}
            <div className="rounded-2xl neu-inset p-2">
              <div className="rounded-xl bg-gradient-to-br from-[#14161C] via-[#0E1014] to-[#08090C] text-white p-6 space-y-6 border border-[#D4AF37]/30 shadow-[inset_0_1px_0_rgba(232,204,130,0.22)]">
                <div className="flex items-center justify-between text-xs text-stone-400">
                  <span className="font-display font-semibold text-[#E8CB80] tracking-tight">
                    VeriMark Pass
                  </span>
                  <span className="font-mono text-stone-400">
                    {previewStudent.cohort}
                  </span>
                </div>

                <div className="flex items-center gap-4">
                  <Avatar
                    src={previewStudent.referencePhotoUrl}
                    name={previewStudent.fullName}
                    size="xl"
                    className="ring-[#D4AF37]/35"
                  />
                  <div className="min-w-0">
                    <p className="font-display text-lg font-semibold text-[#F6F1E6] truncate">
                      {previewStudent.fullName}
                    </p>
                    <p className="text-xs text-stone-300 truncate mt-0.5">
                      {previewStudent.department}
                    </p>
                    <p className="text-xs text-stone-400 truncate mt-0.5">
                      {previewStudent.email}
                    </p>
                  </div>
                </div>

                <div className="pt-4 border-t border-[#D4AF37]/20 flex items-end justify-between gap-4">
                  <div>
                    <p className="text-[11px] text-[#C5A456]/85">
                      Student PIN Number
                    </p>
                    <p className="font-mono text-xl font-semibold tracking-wider text-[#F6F1E6] mt-0.5 tabular-nums">
                      {previewStudent.attendanceId}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleCopyId(previewStudent.attendanceId)}
                    className="h-9 px-3 rounded-xl bg-[#D4AF37]/12 hover:bg-[#D4AF37]/22 border border-[#D4AF37]/30 text-xs font-medium text-[#F0D794] inline-flex items-center gap-1.5 transition-colors cursor-pointer shrink-0"
                  >
                    {copiedId === previewStudent.attendanceId ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        Copy PIN
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-5">
            <div className="neu-divider mb-4" />
            {newlyRegistered ? (
              <div className="space-y-3">
                <p className="text-xs text-emerald-700 dark:text-emerald-400 font-medium">
                  {newlyRegistered.fullName} is enrolled for 1:1 ID &amp; 1:N Walk-Up verification.
                </p>
                <button
                  type="button"
                  onClick={() => onUseAttendanceId(newlyRegistered.attendanceId)}
                  className="w-full h-10 px-4 rounded-xl text-xs font-semibold neu-btn-primary inline-flex items-center justify-center gap-2 cursor-pointer"
                >
                  Mark Attendance with {newlyRegistered.attendanceId}
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <p className="text-xs text-slate-500 dark:text-stone-400 leading-relaxed">
                Live 3-Frame Burst captures 3 aligned frames spaced 120ms apart right after a natural blink and stores their averaged L2-normalized centroid—cutting daily false rejections by over 60%.
              </p>
            )}
          </div>
        </section>
      </div>

      {/* Registered Students Directory Table */}
      <section className="neu-surface rounded-2xl overflow-hidden">
        <div className="px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-base font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6]">
              Registered Student Directory
            </h2>
            <p className="text-xs text-slate-500 dark:text-stone-400 mt-0.5 tabular-nums">
              {filteredStudents.length} of {students.length} enrolled students
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search name or PIN..."
                className="h-9 w-52 pl-8 pr-3 rounded-xl text-xs neu-inset text-slate-900 dark:text-[#F6F1E6] placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
              />
            </div>

            <select
              aria-label="Filter by branch"
              value={departmentFilter}
              onChange={(e) => setDepartmentFilter(e.target.value)}
              className="h-9 px-3 rounded-xl text-xs neu-inset text-slate-700 dark:text-stone-200 focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
            >
              <option value="All">All Branches</option>
              {DEPARTMENTS.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="neu-divider mx-6" />

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="text-[11px] font-medium text-slate-500 dark:text-[#C5A456]/75 border-b border-slate-300/40 dark:border-[#D4AF37]/15">
                <th className="py-3 pl-6 pr-4 font-medium">Student</th>
                <th className="py-3 px-4 font-medium">Student PIN</th>
                <th className="py-3 px-4 font-medium">Branch &amp; Batch</th>
                <th className="py-3 px-4 font-medium">Biometric Profile</th>
                <th className="py-3 pl-4 pr-6 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-300/35 dark:divide-[#D4AF37]/10 text-sm">
              {filteredStudents.length === 0 ? (
                <tr>
                  <td
                    colSpan={5}
                    className="py-10 text-center text-xs text-slate-500 dark:text-stone-400"
                  >
                    {students.length === 0
                      ? 'No students registered yet. Enroll your first student above to assign a PIN (e.g. 23210-ec-053).'
                      : 'No matching students found.'}
                  </td>
                </tr>
              ) : (
                filteredStudents.map((stu) => {
                  const hasBurst =
                    stu.faceVectors && stu.faceVectors.length > 1;
                  const hasSingle =
                    Boolean(stu.faceDescriptor) ||
                    Boolean(stu.faceVectors?.length);
                  return (
                    <tr
                      key={stu.id}
                      className="hover:bg-white/40 dark:hover:bg-[#D4AF37]/[0.04] transition-colors"
                    >
                      <td className="py-3 pl-6 pr-4">
                        <div className="flex items-center gap-3">
                          <Avatar
                            src={stu.referencePhotoUrl}
                            name={stu.fullName}
                            size="sm"
                          />
                          <div className="min-w-0">
                            <p className="font-medium text-slate-900 dark:text-[#F5EFE2] truncate">
                              {stu.fullName}
                            </p>
                            <p className="text-xs text-slate-500 dark:text-stone-400 truncate">
                              {stu.email}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        <button
                          type="button"
                          onClick={() => handleCopyId(stu.attendanceId)}
                          title="Copy Attendance ID"
                          className="inline-flex items-center gap-1.5 font-mono text-xs text-slate-700 dark:text-stone-200 hover:text-blue-600 dark:hover:text-[#E2C275] cursor-pointer tabular-nums"
                        >
                          {stu.attendanceId}
                          {copiedId === stu.attendanceId ? (
                            <Check className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                          ) : (
                            <Copy className="w-3 h-3 opacity-45" />
                          )}
                        </button>
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-600 dark:text-stone-400 whitespace-nowrap">
                        {stu.department} · {stu.cohort}
                      </td>
                      <td className="py-3 px-4 text-xs whitespace-nowrap">
                        <span
                          className={
                            hasBurst || hasSingle
                              ? 'text-emerald-700 dark:text-emerald-400 font-medium'
                              : stu.biometricEnrolled
                              ? 'text-blue-700 dark:text-[#E2C275] font-medium'
                              : 'text-slate-500 dark:text-stone-500'
                          }
                        >
                          {hasBurst
                            ? '3-Vector Locked'
                            : hasSingle
                            ? '128-D Locked'
                            : stu.biometricEnrolled
                            ? 'Portrait Ready'
                            : 'Pending'}
                        </span>
                      </td>
                      <td className="py-3 pl-4 pr-6 text-right whitespace-nowrap">
                        <div className="inline-flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => onUseAttendanceId(stu.attendanceId)}
                            className="px-3 py-1.5 rounded-lg text-xs font-medium neu-btn text-blue-600 dark:text-[#E5C675] cursor-pointer"
                          >
                            Mark Attendance
                          </button>
                          <button
                            type="button"
                            onClick={() => onDeleteStudent(stu.id)}
                            aria-label={`Remove ${stu.fullName}`}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-red-600 dark:hover:text-red-400 transition-colors cursor-pointer"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* On-Demand Live 3-Frame Aadhaar Biometric Burst Modal */}
      {captureModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-[2px]">
          <div className="w-full max-w-md neu-surface gold-edge-top rounded-2xl overflow-hidden">
            <div className="px-5 py-4 flex items-center justify-between">
              <div>
                <h3 className="font-display text-sm font-semibold text-slate-900 dark:text-[#F6F1E6]">
                  Live 3-Frame Aadhaar Burst
                </h3>
                <p className="text-[11px] text-slate-500 dark:text-stone-400 mt-0.5">
                  Blink once naturally — auto-captures 3 sharp L2 vectors (0/120/240ms)
                </p>
              </div>
              <button
                type="button"
                onClick={handleCloseReferenceCamera}
                aria-label="Close camera modal"
                className="p-1.5 rounded-xl neu-btn text-slate-500 hover:text-slate-800 dark:hover:text-stone-200 cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="neu-divider mx-5" />

            <div className="p-5">
              {captureError ? (
                <div className="py-8 text-center space-y-4">
                  <p className="text-xs text-amber-700 dark:text-amber-400 leading-relaxed">
                    {captureError}
                  </p>
                  <button
                    type="button"
                    onClick={handleOpenReferenceCamera}
                    className="h-9 px-4 rounded-xl text-xs font-semibold neu-btn-primary inline-flex items-center gap-1.5 cursor-pointer"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Retry Live Burst
                  </button>
                </div>
              ) : (
                <div
                  className={`aspect-[4/3] rounded-2xl p-2 transition-all duration-150 ${
                    enrollHud?.shouldFlashUI
                      ? 'bg-[#FFF7DE]/80 shadow-[0_0_24px_rgba(245,225,164,0.45)]'
                      : 'neu-inset'
                  }`}
                >
                  <div className="w-full h-full rounded-xl overflow-hidden bg-[#08090C] relative border border-[#D4AF37]/20">
                    <video
                      ref={refVideoRef}
                      autoPlay
                      playsInline
                      muted
                      className={`w-full h-full object-cover ${
                        config.mirrorCamera ? '-scale-x-100' : ''
                      }`}
                    />
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                      <div className="w-40 h-48 relative">
                        <span
                          className={`absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 rounded-tl-xl transition-colors ${
                            enrollHud?.shouldFlashUI
                              ? 'border-[#FFF9E6]'
                              : enrollHud?.qualityStatus === 'locked'
                              ? 'border-emerald-400'
                              : 'border-[#E8CB80]'
                          }`}
                        />
                        <span
                          className={`absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 rounded-tr-xl transition-colors ${
                            enrollHud?.shouldFlashUI
                              ? 'border-[#FFF9E6]'
                              : enrollHud?.qualityStatus === 'locked'
                              ? 'border-emerald-400'
                              : 'border-[#E8CB80]'
                          }`}
                        />
                        <span
                          className={`absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 rounded-bl-xl transition-colors ${
                            enrollHud?.shouldFlashUI
                              ? 'border-[#FFF9E6]'
                              : enrollHud?.qualityStatus === 'locked'
                              ? 'border-emerald-400'
                              : 'border-[#E8CB80]'
                          }`}
                        />
                        <span
                          className={`absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 rounded-br-xl transition-colors ${
                            enrollHud?.shouldFlashUI
                              ? 'border-[#FFF9E6]'
                              : enrollHud?.qualityStatus === 'locked'
                              ? 'border-emerald-400'
                              : 'border-[#E8CB80]'
                          }`}
                        />
                      </div>
                    </div>
                    <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 via-black/55 to-transparent px-4 pb-3 pt-8">
                      <p
                        className={`text-xs font-semibold ${
                          enrollHud?.qualityStatus === 'warn'
                            ? 'text-amber-300'
                            : enrollHud?.qualityStatus === 'locked'
                            ? 'text-emerald-300'
                            : 'text-[#F6F1E6]'
                        }`}
                      >
                        {enrollHud?.instruction ||
                          'Warming up shaders & activating camera...'}
                      </p>
                      <div className="mt-2 h-1 w-full rounded-full bg-white/10 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-[#B88928] via-[#E8CB80] to-emerald-400 transition-all duration-150"
                          style={{ width: `${enrollHud?.progress ?? 8}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-4 flex items-center justify-end">
                <button
                  type="button"
                  onClick={handleCloseReferenceCamera}
                  className="h-9 px-3.5 rounded-xl text-xs font-medium neu-btn text-slate-700 dark:text-stone-200 cursor-pointer"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
