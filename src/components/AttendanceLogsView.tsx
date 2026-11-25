import { useMemo, useState } from 'react';
import {
  CheckCircle2,
  Clock,
  Download,
  Eye,
  Search,
  ShieldAlert,
  Trash2,
  X,
} from 'lucide-react';
import type { AttendanceRecord, AttendanceStatus } from '../types';
import { DEPARTMENTS, formatDateKey } from '../data/seedData';
import { getChallengePromptLabel } from '../services/aadhaarFaceEngine';
import { Avatar } from './Avatar';

interface AttendanceLogsViewProps {
  records: AttendanceRecord[];
  onDeleteRecord: (recordId: string) => void;
  onUpdateRecordStatus: (recordId: string, status: AttendanceStatus) => void;
}

type StatusFilter = 'all' | AttendanceStatus;

export function AttendanceLogsView({
  records,
  onDeleteRecord,
  onUpdateRecordStatus,
}: AttendanceLogsViewProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [departmentFilter, setDepartmentFilter] = useState('All');
  const [dateFilter, setDateFilter] = useState('');
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);

  const filteredRecords = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return records.filter((r) => {
      const matchesStatus = statusFilter === 'all' || r.status === statusFilter;
      const matchesDept =
        departmentFilter === 'All' || r.department === departmentFilter;
      const matchesDate =
        !dateFilter || formatDateKey(new Date(r.timestamp)) === dateFilter;
      const matchesQuery =
        !q ||
        r.studentName.toLowerCase().includes(q) ||
        r.attendanceId.toLowerCase().includes(q) ||
        r.department.toLowerCase().includes(q);
      return matchesStatus && matchesDept && matchesDate && matchesQuery;
    });
  }, [records, searchQuery, statusFilter, departmentFilter, dateFilter]);

  const selectedRecord = useMemo(
    () => records.find((r) => r.id === selectedRecordId) ?? null,
    [records, selectedRecordId]
  );

  const handleExportCsv = () => {
    const headers = [
      'Record ID',
      'Student PIN',
      'Student Name',
      'Branch',
      'Timestamp',
      'Status',
      'Confidence (%)',
      '128-D Distance',
      'Latency (ms)',
      'Check-In Mode',
      'Liveness Passed',
      'Verification Mode',
    ];

    const rows = filteredRecords.map((r) => [
      r.id,
      r.attendanceId,
      `"${r.studentName.replace(/"/g, '""')}"`,
      `"${r.department.replace(/"/g, '""')}"`,
      r.timestamp,
      r.status,
      r.confidence.toFixed(1),
      typeof r.euclideanDistance === 'number'
        ? r.euclideanDistance.toFixed(3)
        : '',
      typeof r.latencyMs === 'number' ? String(r.latencyMs) : '',
      r.checkInMode === 'walkup' ? '1:N Walk-Up' : '1:1 ID',
      r.livenessPassed ? 'Yes' : 'No',
      r.verificationMode,
    ]);

    const csvContent = [headers.join(','), ...rows.map((row) => row.join(','))].join(
      '\n'
    );
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `verimark-attendance-${formatDateKey(new Date())}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const counts = useMemo(() => {
    return {
      all: records.length,
      present: records.filter((r) => r.status === 'present').length,
      late: records.filter((r) => r.status === 'late').length,
      flagged: records.filter((r) => r.status === 'flagged').length,
    };
  }, [records]);

  return (
    <div className="space-y-6">
      {/* Sculpted Filter Toolbar */}
      <section className="neu-surface rounded-2xl p-4 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div
          role="group"
          aria-label="Filter by attendance status"
          className="inline-flex items-center gap-1 p-1 rounded-xl neu-inset self-start"
        >
          {(
            [
              { id: 'all', label: `All (${counts.all})` },
              { id: 'present', label: `Verified (${counts.present})` },
              { id: 'late', label: `Late (${counts.late})` },
              { id: 'flagged', label: `Flagged (${counts.flagged})` },
            ] as { id: StatusFilter; label: string }[]
          ).map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setStatusFilter(tab.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all whitespace-nowrap shrink-0 cursor-pointer tabular-nums ${
                statusFilter === tab.id
                  ? 'neu-raised-sm text-slate-900 dark:text-[#E5C675] font-semibold'
                  : 'text-slate-500 dark:text-stone-400 hover:text-slate-800 dark:hover:text-stone-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <div className="relative flex-1 sm:flex-initial">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search student or PIN..."
              className="h-9 w-full sm:w-52 pl-8 pr-3 rounded-xl text-xs neu-inset text-slate-900 dark:text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-600/35"
            />
          </div>

          <select
            aria-label="Filter by branch"
            value={departmentFilter}
            onChange={(e) => setDepartmentFilter(e.target.value)}
            className="h-9 px-3 rounded-xl text-xs neu-inset text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-600/35"
          >
            <option value="All">All Branches</option>
            {DEPARTMENTS.map((dept) => (
              <option key={dept} value={dept}>
                {dept}
              </option>
            ))}
          </select>

          <input
            type="date"
            aria-label="Filter by date"
            value={dateFilter}
            onChange={(e) => setDateFilter(e.target.value)}
            className="h-9 px-3 rounded-xl text-xs font-mono neu-inset text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-600/35"
          />

          {dateFilter && (
            <button
              type="button"
              onClick={() => setDateFilter('')}
              className="text-xs text-slate-500 hover:text-slate-900 dark:hover:text-white cursor-pointer"
            >
              Clear date
            </button>
          )}

          <button
            type="button"
            onClick={handleExportCsv}
            disabled={filteredRecords.length === 0}
            className="h-9 px-3.5 rounded-xl text-xs font-semibold neu-btn-ink disabled:opacity-40 inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV
          </button>
        </div>
      </section>

      {/* Historical Audit Table */}
      <section className="neu-surface gold-edge-top rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="text-[11px] font-medium text-slate-500 dark:text-[#C5A456]/75 border-b border-slate-300/40 dark:border-[#D4AF37]/15">
                <th className="py-3.5 pl-6 pr-4 font-medium">Student</th>
                <th className="py-3.5 px-4 font-medium">Student PIN</th>
                <th className="py-3.5 px-4 font-medium">Branch</th>
                <th className="py-3.5 px-4 font-medium">Date &amp; Time</th>
                <th className="py-3.5 px-4 font-medium text-right">Confidence</th>
                <th className="py-3.5 px-4 font-medium">Status</th>
                <th className="py-3.5 pl-4 pr-6 font-medium text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-300/35 dark:divide-[#D4AF37]/10 text-sm">
              {filteredRecords.length === 0 ? (
                <tr>
                  <td
                    colSpan={7}
                    className="py-12 text-center text-xs text-slate-500 dark:text-stone-400"
                  >
                    {records.length === 0
                      ? 'No attendance records yet. Verified check-ins will be logged here automatically.'
                      : 'No historical attendance records match your active filters.'}
                  </td>
                </tr>
              ) : (
                filteredRecords.map((rec) => {
                  const dt = new Date(rec.timestamp);
                  return (
                    <tr
                      key={rec.id}
                      onClick={() => setSelectedRecordId(rec.id)}
                      className="hover:bg-white/40 dark:hover:bg-[#D4AF37]/[0.04] transition-colors cursor-pointer"
                    >
                      <td className="py-3 pl-6 pr-4">
                        <div className="flex items-center gap-3">
                          <Avatar
                            src={rec.capturedFrameUrl || rec.referencePhotoUrl}
                            name={rec.studentName}
                            size="xs"
                          />
                          <span className="font-medium text-slate-900 dark:text-slate-100 whitespace-nowrap">
                            {rec.studentName}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4 font-mono text-xs text-slate-600 dark:text-slate-300 tabular-nums whitespace-nowrap">
                        {rec.attendanceId}
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-600 dark:text-slate-400 whitespace-nowrap">
                        {rec.department}
                      </td>
                      <td className="py-3 px-4 font-mono text-xs text-slate-600 dark:text-slate-400 tabular-nums whitespace-nowrap">
                        {dt.toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric',
                        })}{' '}
                        ·{' '}
                        {dt.toLocaleTimeString('en-US', {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </td>
                      <td className="py-3 px-4 font-mono text-xs text-slate-700 dark:text-slate-200 text-right tabular-nums whitespace-nowrap">
                        {rec.confidence.toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        <span
                          className={`inline-flex items-center gap-1 text-xs font-medium ${
                            rec.status === 'present'
                              ? 'text-emerald-700 dark:text-emerald-400'
                              : rec.status === 'late'
                              ? 'text-amber-700 dark:text-amber-400'
                              : 'text-red-700 dark:text-red-400'
                          }`}
                        >
                          {rec.status === 'present' ? (
                            <CheckCircle2 className="w-3.5 h-3.5" />
                          ) : rec.status === 'late' ? (
                            <Clock className="w-3.5 h-3.5" />
                          ) : (
                            <ShieldAlert className="w-3.5 h-3.5" />
                          )}
                          {rec.status === 'present'
                            ? 'Verified'
                            : rec.status === 'late'
                            ? 'Late'
                            : 'Flagged'}
                        </span>
                      </td>
                      <td className="py-3 pl-4 pr-6 text-right whitespace-nowrap">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedRecordId(rec.id);
                          }}
                          className="inline-flex items-center gap-1 text-xs font-medium text-slate-600 dark:text-stone-300 hover:text-blue-600 dark:hover:text-[#E2C275] cursor-pointer"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          Details
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Sculpted Slide-Over Record Inspector */}
      {selectedRecord && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/55 backdrop-blur-[2px]">
          <div
            className="fixed inset-0"
            onClick={() => setSelectedRecordId(null)}
            aria-hidden="true"
          />
          <aside className="relative w-full max-w-md neu-canvas border-l border-white/60 dark:border-[#D4AF37]/20 h-full overflow-y-auto p-6 flex flex-col justify-between z-10 shadow-2xl">
            <div className="space-y-6">
              <div className="flex items-start justify-between gap-4 pb-3">
                <div>
                  <h3 className="font-display text-base font-semibold text-slate-900 dark:text-white">
                    Verification Record
                  </h3>
                  <p className="font-mono text-xs text-slate-500 dark:text-slate-400 mt-0.5 tabular-nums">
                    {selectedRecord.attendanceId}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedRecordId(null)}
                  aria-label="Close inspector"
                  className="p-2 rounded-xl neu-btn text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="neu-divider" />

              {/* Comparison Photos inside Recessed Frames */}
              <div className="grid grid-cols-2 gap-3.5">
                <div className="space-y-1.5">
                  <p className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                    Reference Portrait
                  </p>
                  <div className="aspect-square rounded-2xl neu-inset p-1.5 flex">
                    <div className="w-full h-full rounded-xl overflow-hidden bg-slate-200/60 dark:bg-slate-950 flex items-center justify-center">
                      {selectedRecord.referencePhotoUrl ? (
                        <img
                          src={selectedRecord.referencePhotoUrl}
                          alt="Reference portrait"
                          referrerPolicy="no-referrer"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-xs text-slate-400">Not stored</span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <p className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                    Captured Snapshot
                  </p>
                  <div className="aspect-square rounded-2xl neu-inset p-1.5 flex">
                    <div className="w-full h-full rounded-xl overflow-hidden bg-slate-200/60 dark:bg-slate-950 flex items-center justify-center">
                      {selectedRecord.capturedFrameUrl &&
                      !selectedRecord.capturedFrameUrl.endsWith('preview-frame') ? (
                        <img
                          src={selectedRecord.capturedFrameUrl}
                          alt="Captured attendance frame"
                          referrerPolicy="no-referrer"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-xs text-slate-400">No frame saved</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <dl className="divide-y divide-slate-300/40 dark:divide-white/[0.06] text-xs">
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-slate-500 dark:text-slate-400">Student</dt>
                  <dd className="font-medium text-slate-900 dark:text-white text-right">
                    {selectedRecord.studentName}
                  </dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-slate-500 dark:text-slate-400">Department</dt>
                  <dd className="text-slate-700 dark:text-slate-200 text-right">
                    {selectedRecord.department}
                  </dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-slate-500 dark:text-slate-400">Timestamp</dt>
                  <dd className="font-mono text-slate-700 dark:text-slate-200 text-right tabular-nums">
                    {new Date(selectedRecord.timestamp).toLocaleString('en-US')}
                  </dd>
                </div>
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-slate-500 dark:text-stone-400">
                    Match Confidence
                  </dt>
                  <dd className="font-mono font-semibold text-slate-900 dark:text-[#F6F1E6] text-right tabular-nums">
                    {selectedRecord.confidence.toFixed(1)}%
                  </dd>
                </div>
                {typeof selectedRecord.euclideanDistance === 'number' && (
                  <div className="py-2.5 flex justify-between gap-4">
                    <dt className="text-slate-500 dark:text-stone-400">
                      128-D Distance
                    </dt>
                    <dd className="font-mono font-semibold text-slate-900 dark:text-[#E5C675] text-right tabular-nums">
                      {selectedRecord.euclideanDistance.toFixed(3)}
                    </dd>
                  </div>
                )}
                {typeof selectedRecord.latencyMs === 'number' && (
                  <div className="py-2.5 flex justify-between gap-4">
                    <dt className="text-slate-500 dark:text-stone-400">
                      Verification Speed
                    </dt>
                    <dd className="font-mono text-slate-700 dark:text-stone-200 text-right tabular-nums">
                      {selectedRecord.latencyMs} ms ·{' '}
                      {selectedRecord.checkInMode === 'walkup'
                        ? '1:N Walk-Up'
                        : '1:1 ID'}
                    </dd>
                  </div>
                )}
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-slate-500 dark:text-stone-400">
                    Liveness Check
                  </dt>
                  <dd className="text-slate-700 dark:text-stone-200 text-right">
                    {selectedRecord.livenessPassed
                      ? selectedRecord.qualityChecks?.challengeType &&
                        selectedRecord.qualityChecks.challengeType !== 'none'
                        ? `Passed (${getChallengePromptLabel(selectedRecord.qualityChecks.challengeType)})`
                        : 'Passed (Single Blink)'
                      : 'Failed'}
                  </dd>
                </div>
                {selectedRecord.qualityChecks && (
                  <div className="py-2.5 space-y-1.5">
                    <dt className="text-slate-500 dark:text-stone-400">
                      Biometric Anti-Spoof Audit
                    </dt>
                    <dd className="text-[11px] font-mono text-slate-600 dark:text-stone-300 leading-relaxed">
                      Sensor {selectedRecord.qualityChecks.hardwareValid ? 'OK' : 'Bypass'} · Bezel{' '}
                      {selectedRecord.qualityChecks.screenBorderClear ? 'Clear' : 'Blocked'} · Blink{' '}
                      {selectedRecord.qualityChecks.blinkVerified ? 'OK' : 'Fail'} · Moiré{' '}
                      {selectedRecord.qualityChecks.moirePassed ? 'Clean' : 'Fail'}
                      {typeof selectedRecord.qualityChecks.padRealScore === 'number'
                        ? ` · PAD ${selectedRecord.qualityChecks.padRealScore}%`
                        : ''}
                      {typeof selectedRecord.qualityChecks.reactionTimeMs === 'number'
                        ? ` · Blink ${selectedRecord.qualityChecks.reactionTimeMs}ms`
                        : ''}
                    </dd>
                  </div>
                )}
                <div className="py-2.5 flex justify-between gap-4">
                  <dt className="text-slate-500 dark:text-stone-400">Engine Mode</dt>
                  <dd className="text-slate-700 dark:text-stone-200 text-right">
                    {selectedRecord.verificationMode === 'custom_api'
                      ? 'Custom Backend API'
                      : 'Aadhaar FaceRD Local'}
                  </dd>
                </div>
                {selectedRecord.notes && (
                  <div className="py-2.5 space-y-1">
                    <dt className="text-slate-500 dark:text-slate-400">
                      Verification Note
                    </dt>
                    <dd className="text-slate-700 dark:text-slate-300 leading-relaxed">
                      {selectedRecord.notes}
                    </dd>
                  </div>
                )}
              </dl>

              {/* Tactile Status Override */}
              <div className="space-y-2">
                <p className="text-xs font-medium text-slate-600 dark:text-slate-300">
                  Update Record Status
                </p>
                <div className="grid grid-cols-3 gap-2 p-1 rounded-xl neu-inset">
                  {(
                    [
                      { id: 'present', label: 'Verified' },
                      { id: 'late', label: 'Late' },
                      { id: 'flagged', label: 'Flagged' },
                    ] as { id: AttendanceStatus; label: string }[]
                  ).map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => onUpdateRecordStatus(selectedRecord.id, opt.id)}
                      className={`h-8 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                        selectedRecord.status === opt.id
                          ? 'neu-raised-sm text-slate-900 dark:text-[#E5C675] font-semibold'
                          : 'text-slate-500 dark:text-stone-400 hover:text-slate-800 dark:hover:text-stone-200'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="pt-5 mt-6">
              <div className="neu-divider mb-4" />
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={() => {
                    onDeleteRecord(selectedRecord.id);
                    setSelectedRecordId(null);
                  }}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-red-600 dark:text-red-400 hover:underline cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Delete entry
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedRecordId(null)}
                  className="h-9 px-4 rounded-xl text-xs font-medium neu-btn text-slate-800 dark:text-slate-200 cursor-pointer"
                >
                  Done
                </button>
              </div>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
