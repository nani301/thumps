import { useMemo, useState } from 'react';
import { ArrowRight, ArrowUpRight, Camera, UserPlus } from 'lucide-react';
import type { AttendanceRecord, NavSection, Student } from '../types';
import { buildDailyTrendSeries, formatDateKey } from '../data/seedData';
import { Avatar } from './Avatar';

interface DashboardViewProps {
  students: Student[];
  records: AttendanceRecord[];
  onNavigate: (section: NavSection) => void;
  onQuickMarkStudent: (attendanceId: string) => void;
}

type RangeDays = 7 | 14 | 30;

export function DashboardView({
  students,
  records,
  onNavigate,
  onQuickMarkStudent,
}: DashboardViewProps) {
  const [rangeDays, setRangeDays] = useState<RangeDays>(14);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const todayKey = formatDateKey(new Date());

  const stats = useMemo(() => {
    const todayUnique = new Map<string, AttendanceRecord>();
    for (const r of records) {
      if (formatDateKey(new Date(r.timestamp)) === todayKey) {
        if (!todayUnique.has(r.attendanceId)) {
          todayUnique.set(r.attendanceId, r);
        }
      }
    }

    let presentCount = 0;
    let lateCount = 0;
    let flaggedCount = 0;

    todayUnique.forEach((rec) => {
      if (rec.status === 'present') presentCount++;
      else if (rec.status === 'late') lateCount++;
      else flaggedCount++;
    });

    const totalAttended = presentCount + lateCount;
    const totalStudents = students.length;
    const attendanceRate =
      totalStudents > 0 ? Math.round((totalAttended / totalStudents) * 100) : 0;

    const avgConfidence =
      records.length > 0
        ? (
            records.reduce((acc, item) => acc + item.confidence, 0) / records.length
          ).toFixed(1)
        : '0.0';

    return {
      totalStudents,
      presentToday: totalAttended,
      onTimeToday: presentCount,
      lateToday: lateCount,
      flaggedToday: flaggedCount,
      attendanceRate,
      avgConfidence,
    };
  }, [students, records, todayKey]);

  const trendSeries = useMemo(
    () => buildDailyTrendSeries(rangeDays, students, records),
    [rangeDays, students, records]
  );

  const departmentBreakdown = useMemo(() => {
    const map = new Map<string, { total: number; presentToday: number }>();
    const todaySet = new Set(
      records
        .filter(
          (r) =>
            formatDateKey(new Date(r.timestamp)) === todayKey &&
            r.status !== 'flagged'
        )
        .map((r) => r.attendanceId)
    );

    for (const s of students) {
      const current = map.get(s.department) ?? { total: 0, presentToday: 0 };
      current.total += 1;
      if (todaySet.has(s.attendanceId)) {
        current.presentToday += 1;
      }
      map.set(s.department, current);
    }

    return Array.from(map.entries())
      .map(([department, counts]) => ({
        department,
        total: counts.total,
        presentToday: counts.presentToday,
        rate:
          counts.total > 0
            ? Math.round((counts.presentToday / counts.total) * 100)
            : 0,
      }))
      .sort((a, b) => b.total - a.total);
  }, [students, records, todayKey]);

  const recentRecords = useMemo(() => records.slice(0, 6), [records]);

  // SVG Chart Geometry
  const chartWidth = 760;
  const chartHeight = 225;
  const padLeft = 36;
  const padRight = 18;
  const padTop = 18;
  const padBottom = 30;
  const plotWidth = chartWidth - padLeft - padRight;
  const plotHeight = chartHeight - padTop - padBottom;

  const chartPoints = useMemo(() => {
    const count = Math.max(trendSeries.length, 1);
    return trendSeries.map((item, index) => {
      const attended = item.present + item.late;
      const rate = item.total > 0 ? Math.round((attended / item.total) * 100) : 0;
      const x =
        count === 1
          ? padLeft + plotWidth / 2
          : padLeft + (index / (count - 1)) * plotWidth;
      const y = padTop + plotHeight - (rate / 100) * plotHeight;
      return {
        ...item,
        attended,
        rate,
        x,
        y,
      };
    });
  }, [trendSeries, plotWidth, plotHeight]);

  const linePath = useMemo(() => {
    if (chartPoints.length === 0) return '';
    return chartPoints
      .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`)
      .join(' ');
  }, [chartPoints]);

  const areaPath = useMemo(() => {
    if (chartPoints.length === 0) return '';
    const first = chartPoints[0];
    const last = chartPoints[chartPoints.length - 1];
    const baselineY = padTop + plotHeight;
    return `${linePath} L ${last.x.toFixed(1)} ${baselineY} L ${first.x.toFixed(1)} ${baselineY} Z`;
  }, [chartPoints, linePath, plotHeight]);

  const activePoint =
    hoveredIndex !== null && chartPoints[hoveredIndex]
      ? chartPoints[hoveredIndex]
      : chartPoints[chartPoints.length - 1];

  return (
    <div className="space-y-7">
      {/* KPI Strip — Sculpted Minimalist Raised Tiles */}
      <section
        aria-label="Daily attendance summary"
        className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5"
      >
        <div className="neu-surface gold-edge-top rounded-2xl p-5">
          <p className="text-xs font-medium text-slate-500 dark:text-stone-400">
            Attendance Rate Today
          </p>
          <div className="mt-2.5 flex items-baseline justify-between gap-2">
            <span className="font-display text-3xl font-bold tracking-tight text-slate-900 dark:text-[#F6F1E6] tabular-nums">
              {stats.attendanceRate}%
            </span>
            <span className="text-xs font-medium text-emerald-700 dark:text-[#E2C275] tabular-nums">
              {stats.presentToday} of {stats.totalStudents} checked in
            </span>
          </div>
        </div>

        <div className="neu-surface gold-edge-top rounded-2xl p-5">
          <p className="text-xs font-medium text-slate-500 dark:text-stone-400">
            Registered Students
          </p>
          <div className="mt-2.5 flex items-baseline justify-between gap-2">
            <span className="font-display text-3xl font-bold tracking-tight text-slate-900 dark:text-[#F6F1E6] tabular-nums">
              {stats.totalStudents}
            </span>
            <span className="text-xs text-slate-500 dark:text-stone-400 tabular-nums">
              {students.filter((s) => s.biometricEnrolled).length} enrolled
            </span>
          </div>
        </div>

        <div className="neu-surface gold-edge-top rounded-2xl p-5">
          <p className="text-xs font-medium text-slate-500 dark:text-stone-400">
            Late &amp; Flagged Today
          </p>
          <div className="mt-2.5 flex items-baseline justify-between gap-2">
            <span className="font-display text-3xl font-bold tracking-tight text-slate-900 dark:text-[#F6F1E6] tabular-nums">
              {stats.lateToday + stats.flaggedToday}
            </span>
            <span className="text-xs text-amber-700 dark:text-amber-400 tabular-nums">
              {stats.lateToday} late · {stats.flaggedToday} flagged
            </span>
          </div>
        </div>

        <div className="neu-surface gold-edge-top rounded-2xl p-5">
          <p className="text-xs font-medium text-slate-500 dark:text-stone-400">
            Mean Match Confidence
          </p>
          <div className="mt-2.5 flex items-baseline justify-between gap-2">
            <span className="font-display text-3xl font-bold tracking-tight text-slate-900 dark:text-[#F6F1E6] tabular-nums">
              {stats.avgConfidence}%
            </span>
            <span className="text-xs text-slate-500 dark:text-stone-400 tabular-nums">
              {records.length} logs
            </span>
          </div>
        </div>
      </section>

      {/* Main Analytics Split: Daily Trend Chart + Department Breakdown */}
      <section className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-stretch">
        {/* Daily Attendance Trend Chart */}
        <div className="xl:col-span-8 neu-surface gold-edge-top rounded-2xl p-6 flex flex-col">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4">
            <div>
              <h2 className="font-display text-base font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6]">
                Daily Attendance Trends
              </h2>
              {activePoint && (
                <p className="text-xs text-slate-500 dark:text-stone-400 mt-1 tabular-nums">
                  <span className="font-semibold text-slate-800 dark:text-[#E5C675]">
                    {activePoint.label}
                  </span>
                  {' · '}
                  {activePoint.rate}% rate
                  {' · '}
                  {activePoint.present} on time
                  {' · '}
                  {activePoint.late} late
                  {' · '}
                  {activePoint.absent} absent
                </p>
              )}
            </div>

            {/* Tactile Recessed Segmented Control */}
            <div
              role="group"
              aria-label="Trend timeframe"
              className="inline-flex items-center gap-1 p-1 rounded-xl neu-inset self-start sm:self-auto"
            >
              {([7, 14, 30] as RangeDays[]).map((days) => (
                <button
                  key={days}
                  type="button"
                  onClick={() => {
                    setRangeDays(days);
                    setHoveredIndex(null);
                  }}
                  className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-150 whitespace-nowrap shrink-0 cursor-pointer ${
                    rangeDays === days
                      ? 'neu-raised-sm text-slate-900 dark:text-[#E5C675] font-semibold'
                      : 'text-slate-500 dark:text-stone-400 hover:text-slate-800 dark:hover:text-stone-200'
                  }`}
                >
                  {days} Days
                </button>
              ))}
            </div>
          </div>

          <div className="neu-divider mb-5" />

          <div className="flex-1 min-h-[220px] flex items-center">
            <svg
              viewBox={`0 0 ${chartWidth} ${chartHeight}`}
              className="w-full h-auto overflow-visible select-none"
              role="img"
              aria-label={`Daily attendance rate over the last ${rangeDays} days`}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <defs>
                <linearGradient id="neuAreaGradLight" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#2563EB" stopOpacity="0.18" />
                  <stop offset="100%" stopColor="#2563EB" stopOpacity="0.0" />
                </linearGradient>
                <linearGradient id="neuAreaGradGold" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#D4AF37" stopOpacity="0.22" />
                  <stop offset="100%" stopColor="#D4AF37" stopOpacity="0.0" />
                </linearGradient>
              </defs>

              {[0, 25, 50, 75, 100].map((tick) => {
                const y = padTop + plotHeight - (tick / 100) * plotHeight;
                return (
                  <g key={tick}>
                    <line
                      x1={padLeft}
                      y1={y}
                      x2={chartWidth - padRight}
                      y2={y}
                      className="stroke-slate-300/70 dark:stroke-[#D4AF37]/12"
                      strokeWidth="1"
                      strokeDasharray={tick === 0 ? undefined : '3 4'}
                    />
                    <text
                      x={padLeft - 8}
                      y={y + 4}
                      textAnchor="end"
                      className="fill-slate-400 dark:fill-stone-500 text-[10px] font-mono"
                    >
                      {tick}%
                    </text>
                  </g>
                );
              })}

              {areaPath && (
                <>
                  <path d={areaPath} fill="url(#neuAreaGradLight)" className="dark:hidden" />
                  <path d={areaPath} fill="url(#neuAreaGradGold)" className="hidden dark:block" />
                </>
              )}

              {linePath && (
                <path
                  d={linePath}
                  fill="none"
                  className="stroke-blue-600 dark:stroke-[#E2C275]"
                  strokeWidth="2.1"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {activePoint && (
                <line
                  x1={activePoint.x}
                  y1={padTop}
                  x2={activePoint.x}
                  y2={padTop + plotHeight}
                  className="stroke-slate-400/70 dark:stroke-[#D4AF37]/35"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                />
              )}

              {chartPoints.map((pt, index) => {
                const isSelected = activePoint?.date === pt.date;
                const labelStep = rangeDays === 30 ? 5 : rangeDays === 14 ? 2 : 1;
                const showXLabel =
                  index === 0 ||
                  index === chartPoints.length - 1 ||
                  index % labelStep === 0;
                const hitWidth = plotWidth / Math.max(chartPoints.length, 1);

                return (
                  <g key={pt.date}>
                    <rect
                      x={pt.x - hitWidth / 2}
                      y={padTop}
                      width={hitWidth}
                      height={plotHeight + 24}
                      fill="transparent"
                      className="cursor-pointer"
                      onMouseEnter={() => setHoveredIndex(index)}
                    />
                    <circle
                      cx={pt.x}
                      cy={pt.y}
                      r={isSelected ? 5 : rangeDays <= 14 ? 2.75 : 0}
                      className={
                        isSelected
                          ? 'fill-blue-600 dark:fill-[#E8CB80] stroke-[#EBF0F6] dark:stroke-[#0C0D10]'
                          : 'fill-[#EBF0F6] dark:fill-[#111318] stroke-blue-600 dark:stroke-[#D4AF37]'
                      }
                      strokeWidth={isSelected ? 2.5 : 1.65}
                    />
                    {showXLabel && (
                      <text
                        x={pt.x}
                        y={chartHeight - 6}
                        textAnchor="middle"
                        className={`text-[10px] font-mono ${
                          isSelected
                            ? 'fill-slate-900 dark:fill-[#E5C675] font-semibold'
                            : 'fill-slate-400 dark:fill-stone-500'
                        }`}
                      >
                        {pt.label}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Department Attendance Breakdown */}
        <div className="xl:col-span-4 neu-surface gold-edge-top rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="pb-4">
              <h2 className="font-display text-base font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6]">
                Branch Breakdown
              </h2>
              <p className="text-xs text-slate-500 dark:text-stone-400 mt-1">
                Today&apos;s verified check-ins by branch
              </p>
            </div>

            <div className="neu-divider mb-5" />

            {departmentBreakdown.length === 0 ? (
              <div className="py-8 text-center">
                <p className="text-xs font-medium text-slate-600 dark:text-stone-300">
                  No students enrolled yet
                </p>
                <p className="text-[11px] text-slate-500 dark:text-stone-400 mt-1 leading-relaxed">
                  Register students by branch (EC, CM, CS, EE, ME, CE, AI, IT) to see live branch attendance rates here.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {departmentBreakdown.map((item) => (
                  <div key={item.department} className="space-y-1.5">
                    <div className="flex items-center justify-between gap-2 text-xs">
                      <span className="font-medium text-slate-700 dark:text-stone-200 truncate">
                        {item.department}
                      </span>
                      <span className="font-mono text-slate-500 dark:text-stone-400 tabular-nums shrink-0">
                        {item.presentToday}/{item.total} ({item.rate}%)
                      </span>
                    </div>
                    {/* Carved Neumorphic Progress Trough */}
                    <div className="h-2 w-full rounded-full neu-inset overflow-hidden p-[1px]">
                      <div
                        className="h-full rounded-full bg-blue-600 dark:bg-gradient-to-r dark:from-[#B88B28] dark:to-[#E8CB80] transition-all duration-300"
                        style={{ width: `${item.rate}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-6 pt-5">
            <div className="neu-divider mb-4" />
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => onNavigate('mark')}
                className="flex-1 h-9 px-3.5 rounded-xl text-xs font-semibold neu-btn-ink inline-flex items-center justify-center gap-1.5 whitespace-nowrap cursor-pointer"
              >
                <Camera className="w-3.5 h-3.5" />
                Mark Attendance
              </button>
              <button
                type="button"
                onClick={() => onNavigate('register')}
                className="h-9 px-3.5 rounded-xl text-xs font-medium neu-btn text-slate-700 dark:text-slate-200 inline-flex items-center justify-center gap-1.5 whitespace-nowrap cursor-pointer"
              >
                <UserPlus className="w-3.5 h-3.5" />
                Enroll
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Recent Verification Activity Table */}
      <section className="neu-surface rounded-2xl overflow-hidden">
        <div className="px-6 py-4 flex items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-base font-semibold tracking-tight text-slate-900 dark:text-white">
              Recent Verification Activity
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Latest check-in records across all registered Student PINs
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('logs')}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-600 dark:text-[#E2C275] hover:text-blue-700 dark:hover:text-[#F3DCA0] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            View full history
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="neu-divider mx-6" />

        {recentRecords.length === 0 ? (
          <div className="px-6 py-12 text-center">
            <p className="text-sm font-medium text-slate-700 dark:text-stone-200">
              No attendance records logged yet
            </p>
            <p className="text-xs text-slate-500 dark:text-stone-400 mt-1">
              {students.length === 0
                ? 'Register your first student with a PIN (e.g. 23210-ec-053) to begin facial check-ins.'
                : 'Enter a Student PIN or start Walk-Up Queue to capture the first facial check-in.'}
            </p>
            <button
              type="button"
              onClick={() => onNavigate(students.length === 0 ? 'register' : 'mark')}
              className="mt-4 h-9 px-4 text-xs font-semibold rounded-xl neu-btn-primary cursor-pointer"
            >
              {students.length === 0 ? 'Register First Student' : 'Mark First Attendance'}
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="text-[11px] font-medium text-slate-500 dark:text-[#C5A456]/75 border-b border-slate-300/40 dark:border-[#D4AF37]/15">
                  <th className="py-3 pl-6 pr-4 font-medium">Student</th>
                  <th className="py-3 px-4 font-medium">Student PIN</th>
                  <th className="py-3 px-4 font-medium">Branch</th>
                  <th className="py-3 px-4 font-medium">Timestamp</th>
                  <th className="py-3 px-4 font-medium text-right">Confidence</th>
                  <th className="py-3 pl-4 pr-6 font-medium text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-300/35 dark:divide-[#D4AF37]/10 text-sm">
                {recentRecords.map((rec) => {
                  const dateObj = new Date(rec.timestamp);
                  const timeFormatted = dateObj.toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit',
                  });
                  const dateFormatted =
                    formatDateKey(dateObj) === todayKey
                      ? 'Today'
                      : dateObj.toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                        });

                  return (
                    <tr
                      key={rec.id}
                      className="hover:bg-white/40 dark:hover:bg-[#D4AF37]/[0.04] transition-colors"
                    >
                      <td className="py-3 pl-6 pr-4">
                        <div className="flex items-center gap-3">
                          <Avatar
                            src={rec.referencePhotoUrl || rec.capturedFrameUrl}
                            name={rec.studentName}
                            size="xs"
                          />
                          <button
                            type="button"
                            onClick={() => onQuickMarkStudent(rec.attendanceId)}
                            className="font-medium text-slate-900 dark:text-[#F5EFE2] hover:text-blue-600 dark:hover:text-[#E2C275] transition-colors inline-flex items-center gap-1 cursor-pointer text-left"
                          >
                            {rec.studentName}
                            <ArrowUpRight className="w-3 h-3 opacity-45" />
                          </button>
                        </div>
                      </td>
                      <td className="py-3 px-4 font-mono text-xs text-slate-600 dark:text-slate-300 tabular-nums whitespace-nowrap">
                        {rec.attendanceId}
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-600 dark:text-slate-400 whitespace-nowrap">
                        {rec.department}
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-600 dark:text-slate-400 tabular-nums whitespace-nowrap">
                        {dateFormatted} · {timeFormatted}
                      </td>
                      <td className="py-3 px-4 font-mono text-xs text-slate-700 dark:text-slate-300 text-right tabular-nums whitespace-nowrap">
                        {rec.confidence.toFixed(1)}%
                      </td>
                      <td className="py-3 pl-4 pr-6 text-right whitespace-nowrap">
                        <span
                          className={`text-xs font-medium ${
                            rec.status === 'present'
                              ? 'text-emerald-700 dark:text-emerald-400'
                              : rec.status === 'late'
                              ? 'text-amber-700 dark:text-amber-400'
                              : 'text-red-700 dark:text-red-400'
                          }`}
                        >
                          {rec.status === 'present'
                            ? 'Verified'
                            : rec.status === 'late'
                            ? 'Late'
                            : 'Flagged'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
