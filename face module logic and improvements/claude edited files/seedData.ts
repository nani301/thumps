import type {
  AttendanceRecord,
  DailyTrendPoint,
  Student,
  SystemConfig,
  UserProfile,
} from '../types';

export interface YearCollegeOption {
  code: string;
  admissionYear: string;
  collegeCode: string;
  label: string;
  batchLabel: string;
}

export interface BranchOption {
  code: string;
  shortName: string;
  label: string;
}

export const YEAR_COLLEGE_OPTIONS: readonly YearCollegeOption[] = [
  {
    code: '23210',
    admissionYear: '2023',
    collegeCode: '210',
    label: '23210 · 2023 Admission (College 210)',
    batchLabel: '2023 Batch · 210',
  },
  {
    code: '24210',
    admissionYear: '2024',
    collegeCode: '210',
    label: '24210 · 2024 Admission (College 210)',
    batchLabel: '2024 Batch · 210',
  },
  {
    code: '25210',
    admissionYear: '2025',
    collegeCode: '210',
    label: '25210 · 2025 Admission (College 210)',
    batchLabel: '2025 Batch · 210',
  },
  {
    code: '26210',
    admissionYear: '2026',
    collegeCode: '210',
    label: '26210 · 2026 Admission (College 210)',
    batchLabel: '2026 Batch · 210',
  },
] as const;

export const BRANCH_OPTIONS: readonly BranchOption[] = [
  { code: 'ec', shortName: 'EC', label: 'Electronics & Communication (EC)' },
  { code: 'cm', shortName: 'CM', label: 'Computer Engineering (CM)' },
  { code: 'cs', shortName: 'CS', label: 'Computer Science (CS)' },
  { code: 'ee', shortName: 'EE', label: 'Electrical & Electronics (EE)' },
  { code: 'me', shortName: 'ME', label: 'Mechanical Engineering (ME)' },
  { code: 'ce', shortName: 'CE', label: 'Civil Engineering (CE)' },
  { code: 'ai', shortName: 'AI', label: 'AI & Machine Learning (AI)' },
  { code: 'it', shortName: 'IT', label: 'Information Technology (IT)' },
] as const;

export const DEPARTMENTS: readonly string[] = BRANCH_OPTIONS.map((b) => b.label);

export const COHORTS: readonly string[] = YEAR_COLLEGE_OPTIONS.map(
  (y) => y.batchLabel
);

export const DEFAULT_USER_PROFILE: UserProfile = {
  fullName: 'Administrator',
  role: 'Attendance Operator',
  institution: 'College 210',
  email: 'admin@college210.edu',
  avatarUrl: null,
};

export const DEFAULT_SYSTEM_CONFIG: SystemConfig = {
  theme: 'dark',
  confidenceThreshold: 84,
  maxEuclideanDistance: 0.45,
  impostorMargin: 0.07,
  defaultCheckInMode: '1to1',
  lateCutoffTime: '09:15',
  requireLiveness: true,
  enableAudioChime: true,
  mirrorCamera: true,
  cameraResolution: '480p',
  modelMode: 'aadhaar_local',
  modelEndpointUrl: 'http://localhost:8000/api/v1/verify-face',
  enrollEndpointUrl: 'http://localhost:8000/api/v1/enroll-face',
  flashSlots: 8,
  checkInCooldownMinutes: 180,
  allowPhotoEnrollment: false,
};

export const INITIAL_STUDENTS: Student[] = [];

export function formatDateKey(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

export function normalizePinDigits(raw: string | number): string {
  const digits = String(raw).replace(/\D/g, '').slice(0, 3);
  if (!digits) return '001';
  return digits.padStart(3, '0');
}

export function formatStudentPin(
  yearCollegeCode: string,
  branchCode: string,
  pinNumber: string | number
): string {
  return `${yearCollegeCode.trim()}-${branchCode.trim().toLowerCase()}-${normalizePinDigits(pinNumber)}`;
}

export function getNextAvailablePinDigits(
  existingIds: string[],
  yearCollegeCode: string,
  branchCode: string
): string {
  const prefix = `${yearCollegeCode.trim()}-${branchCode.trim().toLowerCase()}-`;
  const usedNumbers = new Set<number>();

  for (const id of existingIds) {
    const normalized = id.trim().toLowerCase();
    if (normalized.startsWith(prefix)) {
      const suffix = normalized.slice(prefix.length);
      const parsed = Number.parseInt(suffix, 10);
      if (!Number.isNaN(parsed) && parsed > 0) {
        usedNumbers.add(parsed);
      }
    }
  }

  for (let n = 1; n <= 999; n++) {
    if (!usedNumbers.has(n)) {
      return String(n).padStart(3, '0');
    }
  }

  return '001';
}

export function createInitialAttendanceRecords(): AttendanceRecord[] {
  return [];
}

/**
 * Builds daily attendance trend series strictly from real logged attendance records.
 * Zero synthetic or mock historical values are generated.
 */
export function buildDailyTrendSeries(
  days: number,
  students: Student[],
  records: AttendanceRecord[]
): DailyTrendPoint[] {
  const todayKey = formatDateKey(new Date());
  const rosterCount = students.length;
  const series: DailyTrendPoint[] = [];

  // Pre-group unique student records per calendar day (latest record per student wins)
  const recordsByDay = new Map<string, Map<string, AttendanceRecord>>();
  for (const rec of records) {
    const dayKey = formatDateKey(new Date(rec.timestamp));
    let dayMap = recordsByDay.get(dayKey);
    if (!dayMap) {
      dayMap = new Map<string, AttendanceRecord>();
      recordsByDay.set(dayKey, dayMap);
    }
    const studentKey = rec.attendanceId.toLowerCase();
    if (!dayMap.has(studentKey)) {
      dayMap.set(studentKey, rec);
    }
  }

  for (let offset = days - 1; offset >= 0; offset--) {
    const d = new Date();
    d.setDate(d.getDate() - offset);
    const key = formatDateKey(d);
    const label =
      key === todayKey
        ? 'Today'
        : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

    const dayMap = recordsByDay.get(key);
    let present = 0;
    let late = 0;

    if (dayMap) {
      dayMap.forEach((rec) => {
        if (rec.status === 'present') present++;
        else if (rec.status === 'late') late++;
      });
    }

    const total = Math.max(rosterCount, present + late);
    const absent = Math.max(0, total - present - late);
    series.push({ date: key, label, present, late, absent, total });
  }

  return series;
}
