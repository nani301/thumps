import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Camera, Moon, Sun, UserPlus, X } from 'lucide-react';
import type {
  AttendanceRecord,
  AttendanceStatus,
  NavSection,
  Student,
  SystemConfig,
  UserProfile,
} from './types';
import {
  createInitialAttendanceRecords,
  DEFAULT_SYSTEM_CONFIG,
  DEFAULT_USER_PROFILE,
  INITIAL_STUDENTS,
} from './data/seedData';
import { MobileMenuButton, Sidebar } from './components/Sidebar';
import { DashboardView } from './components/DashboardView';
import { MarkAttendanceView } from './components/MarkAttendanceView';
import { RegisterStudentView } from './components/RegisterStudentView';
import { AttendanceLogsView } from './components/AttendanceLogsView';
import { SettingsView } from './components/SettingsView';
import {
  extractDescriptorFromImageUrl,
  getActiveBackend,
  warmupAadhaarEngine,
  type ComputeBackend,
} from './services/aadhaarFaceEngine';
import { shrinkDataUrl } from './services/imageUtils';

const STORAGE_KEYS = {
  students: 'verimark_students_v2',
  records: 'verimark_records_v2',
  profile: 'verimark_profile_v2',
  config: 'verimark_config_v2',
};

const LEGACY_MOCK_KEYS = [
  'verimark_students_v1',
  'verimark_records_v1',
  'verimark_profile_v1',
];

function readStored<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

const SECTION_META: Record<
  NavSection,
  { title: string; breadcrumb: string }
> = {
  dashboard: {
    title: 'Attendance Overview',
    breadcrumb: 'Overview / Daily Analytics',
  },
  mark: {
    title: 'Mark Attendance',
    breadcrumb: 'Verification / Facial Check-in',
  },
  register: {
    title: 'Register Student',
    breadcrumb: 'Directory / Attendance ID',
  },
  logs: {
    title: 'Attendance Logs',
    breadcrumb: 'Records / Historical Audit',
  },
  settings: {
    title: 'Settings',
    breadcrumb: 'Workspace / Profile & Config',
  },
};

export default function App() {
  const [activeSection, setActiveSection] = useState<NavSection>('dashboard');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [prefilledAttendanceId, setPrefilledAttendanceId] = useState('');
  const [storageWarning, setStorageWarning] = useState('');
  const [activeBackend, setActiveBackend] = useState<ComputeBackend | null>(() =>
    getActiveBackend()
  );

  const [students, setStudents] = useState<Student[]>(() =>
    readStored<Student[]>(STORAGE_KEYS.students, INITIAL_STUDENTS)
  );
  const [records, setRecords] = useState<AttendanceRecord[]>(() => {
    const stored = readStored<AttendanceRecord[]>(
      STORAGE_KEYS.records,
      createInitialAttendanceRecords()
    );
    return stored.map((r) => ({
      ...r,
      verificationMode:
        r.verificationMode === 'custom_api' ? 'custom_api' : 'aadhaar_local',
    }));
  });
  const [userProfile, setUserProfile] = useState<UserProfile>(() =>
    readStored<UserProfile>(STORAGE_KEYS.profile, DEFAULT_USER_PROFILE)
  );
  const [config, setConfig] = useState<SystemConfig>(() => {
    const raw = readStored<Partial<SystemConfig>>(
      STORAGE_KEYS.config,
      DEFAULT_SYSTEM_CONFIG
    );
    const merged: SystemConfig = {
      ...DEFAULT_SYSTEM_CONFIG,
      ...raw,
      modelMode:
        raw.modelMode === 'custom_api' ? 'custom_api' : 'aadhaar_local',
      defaultCheckInMode:
        raw.defaultCheckInMode === 'walkup' ? 'walkup' : '1to1',
      cameraResolution: raw.cameraResolution === '720p' ? '720p' : '480p',
      requireLiveness: true,
      maxEuclideanDistance:
        typeof raw.maxEuclideanDistance === 'number'
          ? raw.maxEuclideanDistance
          : DEFAULT_SYSTEM_CONFIG.maxEuclideanDistance,
      impostorMargin:
        typeof raw.impostorMargin === 'number'
          ? raw.impostorMargin
          : DEFAULT_SYSTEM_CONFIG.impostorMargin,
      allowPhotoEnrollment:
        typeof raw.allowPhotoEnrollment === 'boolean'
          ? raw.allowPhotoEnrollment
          : DEFAULT_SYSTEM_CONFIG.allowPhotoEnrollment,
      checkInCooldownMinutes:
        typeof raw.checkInCooldownMinutes === 'number'
          ? raw.checkInCooldownMinutes
          : DEFAULT_SYSTEM_CONFIG.checkInCooldownMinutes,
      enableAudioChime:
        typeof raw.enableAudioChime === 'boolean'
          ? raw.enableAudioChime
          : true,
    };
    return merged;
  });

  // Remove legacy v1 mock storage keys on startup & warm TFJS backend silently (camera OFF)
  useEffect(() => {
    let active = true;
    try {
      LEGACY_MOCK_KEYS.forEach((k) => window.localStorage.removeItem(k));
    } catch {
      // Ignore storage errors
    }
    warmupAadhaarEngine()
      .then((backend) => {
        if (active) setActiveBackend(backend);
      })
      .catch(() => {
        // Retries cleanly when verification begins if offline/busy
      });
    return () => {
      active = false;
    };
  }, []);

  const attemptedBackfillIdsRef = useRef<Set<string>>(new Set());

  // Auto-heal enrollment vectors: if a student registered with a photo while the backend
  // previously failed (empty faceVectors), re-extract & cache their 128-D vector automatically.
  useEffect(() => {
    if (!activeBackend) return;
    const missing = students.filter(
      (s) =>
        Boolean(s.referencePhotoUrl) &&
        !attemptedBackfillIdsRef.current.has(s.id) &&
        (!s.faceVectors || s.faceVectors.length === 0) &&
        (!s.faceDescriptor || s.faceDescriptor.length !== 128)
    );
    if (missing.length === 0) return;

    let cancelled = false;
    void (async () => {
      for (const stu of missing) {
        if (cancelled || !stu.referencePhotoUrl) break;
        attemptedBackfillIdsRef.current.add(stu.id);
        try {
          const vec = await extractDescriptorFromImageUrl(stu.referencePhotoUrl);
          if (!cancelled && vec && vec.length === 128) {
            setStudents((prev) =>
              prev.map((item) =>
                item.id === stu.id
                  ? {
                      ...item,
                      faceDescriptor: vec,
                      faceVectors: [vec],
                      biometricEnrolled: true,
                    }
                  : item
              )
            );
          }
        } catch {
          // Skip unencodable photo cleanly
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeBackend, students]);

  // Sync dark mode class on <html>
  useEffect(() => {
    const root = document.documentElement;
    if (config.theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [config.theme]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.students, JSON.stringify(students));
    } catch {
      setStorageWarning(
        'Browser storage quota reached while saving student profiles. Export or clear old attendance logs in Settings to free space.'
      );
    }
  }, [students]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.records, JSON.stringify(records));
    } catch {
      setStorageWarning(
        'Browser storage quota reached while saving attendance logs. Export CSV or clear older records in Settings so new check-ins persist.'
      );
    }
  }, [records]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.profile, JSON.stringify(userProfile));
    } catch {
      // Ignore quota exceeded
    }
  }, [userProfile]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEYS.config, JSON.stringify(config));
    } catch {
      // Ignore quota exceeded
    }
  }, [config]);

  const handleAddStudent = (student: Student) => {
    void shrinkDataUrl(student.referencePhotoUrl, 320, 0.72).then(
      (compressedPhoto) => {
        setStudents((prev) => [
          { ...student, referencePhotoUrl: compressedPhoto },
          ...prev,
        ]);
      }
    );
  };

  const handleCacheStudentVectors = (
    studentId: string,
    vectors: number[][]
  ) => {
    if (vectors.length === 0) return;
    setStudents((prev) =>
      prev.map((s) =>
        s.id === studentId
          ? {
              ...s,
              faceDescriptor: vectors[0],
              faceVectors: vectors,
              biometricEnrolled: true,
            }
          : s
      )
    );
  };

  const handleDeleteStudent = (studentId: string) => {
    setStudents((prev) => prev.filter((s) => s.id !== studentId));
  };

  const handleRecordAttendance = (record: AttendanceRecord) => {
    void Promise.all([
      shrinkDataUrl(record.capturedFrameUrl, 240, 0.6),
      shrinkDataUrl(record.referencePhotoUrl, 96, 0.55),
    ]).then(([capturedThumb, referenceThumb]) => {
      setRecords((prev) => [
        {
          ...record,
          capturedFrameUrl: capturedThumb,
          referencePhotoUrl: referenceThumb,
        },
        ...prev,
      ]);
    });
  };

  const handleDeleteRecord = (recordId: string) => {
    setRecords((prev) => prev.filter((r) => r.id !== recordId));
  };

  const handleUpdateRecordStatus = (
    recordId: string,
    status: AttendanceStatus
  ) => {
    setRecords((prev) =>
      prev.map((r) => (r.id === recordId ? { ...r, status } : r))
    );
  };

  const handleQuickMarkStudent = (attendanceId: string) => {
    setPrefilledAttendanceId(attendanceId);
    setActiveSection('mark');
  };

  const handleToggleTheme = (theme: 'light' | 'dark') => {
    setConfig((prev) => ({ ...prev, theme }));
  };

  const handleClearWorkspaceData = () => {
    setStudents([]);
    setRecords([]);
    setUserProfile(DEFAULT_USER_PROFILE);
    setConfig(DEFAULT_SYSTEM_CONFIG);
    setStorageWarning('');
  };

  const meta = SECTION_META[activeSection];

  return (
    <div className="min-h-screen flex neu-canvas text-[#18202C] dark:text-[#F1ECE1]">
      <Sidebar
        activeSection={activeSection}
        onSelectSection={setActiveSection}
        userProfile={userProfile}
        mobileOpen={mobileMenuOpen}
        onToggleMobile={setMobileMenuOpen}
      />

      <div className="flex-1 min-w-0 flex flex-col">
        {/* Ultra-Slim Header with Dark/White Switcher & Primary CTA */}
        <header className="relative z-20 bg-[#F2F4F8] dark:bg-[#0C0D10] px-5 sm:px-8 lg:px-10 pt-5 pb-3.5 transition-colors duration-200">
          <div className="max-w-[1360px] mx-auto flex items-center justify-between gap-4">
            <div className="flex items-center gap-3.5 min-w-0">
              <MobileMenuButton onClick={() => setMobileMenuOpen(true)} />
              <div className="min-w-0">
                <p className="text-[11px] font-medium text-slate-500 dark:text-[#C5A456]/80 truncate leading-none">
                  {userProfile.institution} · {meta.breadcrumb}
                </p>
                <h1 className="font-display text-lg sm:text-xl font-bold tracking-tight text-slate-900 dark:text-[#F6F1E6] truncate mt-1 leading-tight">
                  {meta.title}
                </h1>
              </div>
            </div>

            <div className="flex items-center gap-3 shrink-0">
              {/* Tactile Dark / White Mode Switcher */}
              <div
                role="group"
                aria-label="Dark and White mode switcher"
                className="inline-flex items-center gap-1 p-1 rounded-xl neu-inset shrink-0"
              >
                <button
                  type="button"
                  onClick={() => handleToggleTheme('dark')}
                  aria-pressed={config.theme === 'dark'}
                  title="Dark Mode (Obsidian Gold)"
                  className={`h-7 px-2.5 rounded-lg text-xs font-medium inline-flex items-center gap-1.5 transition-all whitespace-nowrap cursor-pointer ${
                    config.theme === 'dark'
                      ? 'neu-raised-sm text-[#E5C675] font-semibold'
                      : 'text-slate-500 dark:text-stone-400 hover:text-slate-800 dark:hover:text-stone-200'
                  }`}
                >
                  <Moon className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Dark</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleToggleTheme('light')}
                  aria-pressed={config.theme === 'light'}
                  title="White Mode"
                  className={`h-7 px-2.5 rounded-lg text-xs font-medium inline-flex items-center gap-1.5 transition-all whitespace-nowrap cursor-pointer ${
                    config.theme === 'light'
                      ? 'neu-raised-sm text-slate-900 font-semibold'
                      : 'text-slate-500 dark:text-stone-400 hover:text-slate-800 dark:hover:text-stone-200'
                  }`}
                >
                  <Sun className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">White</span>
                </button>
              </div>

              {activeSection !== 'mark' ? (
                <button
                  type="button"
                  onClick={() => setActiveSection('mark')}
                  className="h-9 px-4 rounded-xl text-xs font-semibold neu-btn-primary inline-flex items-center gap-2 whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <Camera className="w-3.5 h-3.5" />
                  Mark Attendance
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => setActiveSection('register')}
                  className="h-9 px-4 rounded-xl text-xs font-medium neu-btn text-slate-700 dark:text-[#E5C675] inline-flex items-center gap-2 whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <UserPlus className="w-3.5 h-3.5" />
                  Register Student
                </button>
              )}
            </div>
          </div>
          <div className="max-w-[1360px] mx-auto mt-3.5 neu-divider" />
        </header>

        {/* Main Viewport */}
        <main className="flex-1 px-5 sm:px-8 lg:px-10 pt-5 pb-8 sm:pt-6 sm:pb-10 max-w-[1360px] w-full mx-auto">
          {storageWarning && (
            <div
              role="alert"
              className="mb-5 px-4 py-3 rounded-xl border border-amber-500/35 bg-amber-500/10 text-amber-800 dark:text-amber-300 text-xs font-medium flex items-center justify-between gap-3"
            >
              <div className="flex items-center gap-2.5 min-w-0">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{storageWarning}</span>
              </div>
              <button
                type="button"
                onClick={() => setStorageWarning('')}
                aria-label="Dismiss storage warning"
                className="p-1 rounded-lg hover:bg-amber-500/15 transition-colors cursor-pointer shrink-0"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {activeSection === 'dashboard' && (
            <DashboardView
              students={students}
              records={records}
              onNavigate={setActiveSection}
              onQuickMarkStudent={handleQuickMarkStudent}
            />
          )}

          {activeSection === 'mark' && (
            <MarkAttendanceView
              students={students}
              records={records}
              config={config}
              initialAttendanceId={prefilledAttendanceId}
              onRecordAttendance={handleRecordAttendance}
              onCacheStudentVectors={handleCacheStudentVectors}
              onGoToRegister={() => setActiveSection('register')}
            />
          )}

          {activeSection === 'register' && (
            <RegisterStudentView
              students={students}
              config={config}
              onAddStudent={handleAddStudent}
              onDeleteStudent={handleDeleteStudent}
              onUseAttendanceId={handleQuickMarkStudent}
            />
          )}

          {activeSection === 'logs' && (
            <AttendanceLogsView
              records={records}
              onDeleteRecord={handleDeleteRecord}
              onUpdateRecordStatus={handleUpdateRecordStatus}
            />
          )}

          {activeSection === 'settings' && (
            <SettingsView
              userProfile={userProfile}
              config={config}
              activeBackend={activeBackend}
              onUpdateProfile={setUserProfile}
              onUpdateConfig={setConfig}
              onClearWorkspaceData={handleClearWorkspaceData}
            />
          )}
        </main>
      </div>
    </div>
  );
}
