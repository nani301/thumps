import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react';
import {
  Check,
  Code2,
  Moon,
  RotateCcw,
  Save,
  Sun,
  Upload,
} from 'lucide-react';
import type { SystemConfig, UserProfile } from '../types';
import {
  getBackendDisplayLabel,
  useTensorFlowBackend,
  type ComputeBackend,
} from '../services/initTensorFlowBackend';
import { Avatar } from './Avatar';

interface SettingsViewProps {
  userProfile: UserProfile;
  config: SystemConfig;
  activeBackend?: ComputeBackend | null;
  onUpdateProfile: (profile: UserProfile) => void;
  onUpdateConfig: (config: SystemConfig) => void;
  onClearWorkspaceData: () => void;
}

export function SettingsView({
  userProfile,
  config,
  activeBackend = null,
  onUpdateProfile,
  onUpdateConfig,
  onClearWorkspaceData,
}: SettingsViewProps) {
  const backendState = useTensorFlowBackend();
  const resolvedBackendLabel = backendState.ready
    ? backendState.label
    : getBackendDisplayLabel(activeBackend);
  const [profileDraft, setProfileDraft] = useState<UserProfile>(userProfile);
  const [configDraft, setConfigDraft] = useState<SystemConfig>(config);
  const [savedNotice, setSavedNotice] = useState<string>('');
  const avatarFileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    setConfigDraft((prev) =>
      prev.theme === config.theme ? prev : { ...prev, theme: config.theme }
    );
  }, [config.theme]);

  const showSavedFeedback = (message: string) => {
    setSavedNotice(message);
    window.setTimeout(() => setSavedNotice(''), 2400);
  };

  const handleAvatarUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        const updated = { ...profileDraft, avatarUrl: reader.result };
        setProfileDraft(updated);
        onUpdateProfile(updated);
        showSavedFeedback('Profile portrait updated');
      }
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const handleSaveProfile = (e: FormEvent) => {
    e.preventDefault();
    onUpdateProfile(profileDraft);
    showSavedFeedback('Operator profile saved');
  };

  const handleSaveConfig = (e: FormEvent) => {
    e.preventDefault();
    onUpdateConfig(configDraft);
    showSavedFeedback('System configuration saved');
  };

  const handleThemeSwitch = (theme: 'light' | 'dark') => {
    const next = { ...configDraft, theme };
    setConfigDraft(next);
    onUpdateConfig(next);
  };

  return (
    <div className="max-w-4xl space-y-7">
      {savedNotice && (
        <div
          role="status"
          className="px-4 py-3 rounded-xl neu-raised-sm text-xs font-medium text-emerald-700 dark:text-emerald-400 flex items-center gap-2"
        >
          <Check className="w-4 h-4 shrink-0" />
          {savedNotice}
        </div>
      )}

      {/* 1. Operator Profile Section */}
      <section className="neu-surface gold-edge-top rounded-2xl p-6">
        <div className="pb-4">
          <h2 className="font-display text-base font-semibold tracking-tight text-slate-900 dark:text-white">
            Operator Profile
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Manage the administrator profile displayed across attendance audits and workspace headers.
          </p>
        </div>

        <div className="neu-divider mb-5" />

        <form onSubmit={handleSaveProfile} className="space-y-5">
          <div className="flex flex-wrap items-center gap-4">
            <Avatar
              src={profileDraft.avatarUrl}
              name={profileDraft.fullName}
              size="lg"
            />
            <input
              ref={avatarFileRef}
              type="file"
              accept="image/*"
              onChange={handleAvatarUpload}
              className="hidden"
            />
            <div className="space-y-1">
              <button
                type="button"
                onClick={() => avatarFileRef.current?.click()}
                className="h-9 px-3.5 rounded-xl text-xs font-medium neu-btn text-slate-700 dark:text-slate-200 inline-flex items-center gap-1.5 cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5" />
                Change Photo
              </button>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                Square JPG or PNG recommended.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="profile-name"
                className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1.5"
              >
                Full Name
              </label>
              <input
                id="profile-name"
                type="text"
                value={profileDraft.fullName}
                onChange={(e) =>
                  setProfileDraft({ ...profileDraft, fullName: e.target.value })
                }
                className="w-full h-10 px-3.5 rounded-xl text-sm neu-inset text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-600/35"
              />
            </div>

            <div>
              <label
                htmlFor="profile-role"
                className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1.5"
              >
                Role / Title
              </label>
              <input
                id="profile-role"
                type="text"
                value={profileDraft.role}
                onChange={(e) =>
                  setProfileDraft({ ...profileDraft, role: e.target.value })
                }
                className="w-full h-10 px-3.5 rounded-xl text-sm neu-inset text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-600/35"
              />
            </div>

            <div>
              <label
                htmlFor="profile-institution"
                className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1.5"
              >
                Institution Name
              </label>
              <input
                id="profile-institution"
                type="text"
                value={profileDraft.institution}
                onChange={(e) =>
                  setProfileDraft({
                    ...profileDraft,
                    institution: e.target.value,
                  })
                }
                className="w-full h-10 px-3.5 rounded-xl text-sm neu-inset text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-600/35"
              />
            </div>

            <div>
              <label
                htmlFor="profile-email"
                className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1.5"
              >
                Email Address
              </label>
              <input
                id="profile-email"
                type="email"
                value={profileDraft.email}
                onChange={(e) =>
                  setProfileDraft({ ...profileDraft, email: e.target.value })
                }
                className="w-full h-10 px-3.5 rounded-xl text-sm neu-inset text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-600/35"
              />
            </div>
          </div>

          <div className="pt-2">
            <button
              type="submit"
              className="h-9 px-4 rounded-xl text-xs font-semibold neu-btn-ink inline-flex items-center gap-1.5 cursor-pointer"
            >
              <Save className="w-3.5 h-3.5" />
              Save Profile
            </button>
          </div>
        </form>
      </section>

      {/* 2. Capture & Attendance Policy */}
      <form onSubmit={handleSaveConfig} className="space-y-7">
        <section className="neu-surface gold-edge-top rounded-2xl p-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h2 className="font-display text-base font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6]">
                Capture &amp; Attendance Configuration
              </h2>
              <p className="text-xs text-slate-500 dark:text-stone-400 mt-1">
                Control threshold strictness, capture countdown, late cutoff, and appearance mode.
              </p>
            </div>

            {/* Tactile Dark / White Mode Switch */}
            <div
              role="group"
              aria-label="Interface theme"
              className="inline-flex items-center gap-1 p-1 rounded-xl neu-inset self-start"
            >
              <button
                type="button"
                onClick={() => handleThemeSwitch('dark')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium inline-flex items-center gap-1.5 cursor-pointer transition-all ${
                  configDraft.theme === 'dark'
                    ? 'neu-raised-sm text-[#E5C675] font-semibold'
                    : 'text-slate-500 dark:text-stone-400 hover:text-slate-800 dark:hover:text-stone-200'
                }`}
              >
                <Moon className="w-3.5 h-3.5" />
                Dark
              </button>
              <button
                type="button"
                onClick={() => handleThemeSwitch('light')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium inline-flex items-center gap-1.5 cursor-pointer transition-all ${
                  configDraft.theme === 'light'
                    ? 'neu-raised-sm text-slate-900 font-semibold'
                    : 'text-slate-500 dark:text-stone-400 hover:text-slate-800 dark:hover:text-stone-200'
                }`}
              >
                <Sun className="w-3.5 h-3.5" />
                White
              </button>
            </div>
          </div>

          <div className="neu-divider" />

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <label
                  htmlFor="cfg-distance"
                  className="font-medium text-slate-600 dark:text-stone-300"
                >
                  Max 128-D Euclidean Distance
                </label>
                <span className="font-mono font-semibold text-slate-900 dark:text-[#E5C675] tabular-nums">
                  ≤ {configDraft.maxEuclideanDistance.toFixed(2)}
                </span>
              </div>
              <input
                id="cfg-distance"
                type="range"
                min={0.36}
                max={0.56}
                step={0.01}
                value={configDraft.maxEuclideanDistance}
                onChange={(e) =>
                  setConfigDraft({
                    ...configDraft,
                    maxEuclideanDistance: Number(e.target.value),
                  })
                }
                className="w-full neu-range"
              />
              <p className="text-[11px] text-slate-500 dark:text-stone-400">
                TSTS ABAS default is 0.47 on L2-normalized ResNet-34 vectors (tuned for glasses + indoor light).
              </p>
            </div>

            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <label
                  htmlFor="cfg-impostor"
                  className="font-medium text-slate-600 dark:text-stone-300"
                >
                  1:N Walk-Up Impostor Margin
                </label>
                <span className="font-mono font-semibold text-slate-900 dark:text-[#E5C675] tabular-nums">
                  ≥ {configDraft.impostorMargin.toFixed(2)}
                </span>
              </div>
              <input
                id="cfg-impostor"
                type="range"
                min={0.03}
                max={0.14}
                step={0.01}
                value={configDraft.impostorMargin}
                onChange={(e) =>
                  setConfigDraft({
                    ...configDraft,
                    impostorMargin: Number(e.target.value),
                  })
                }
                className="w-full neu-range"
              />
              <p className="text-[11px] text-slate-500 dark:text-stone-400">
                Minimum distance gap to 2nd-closest student in Walk-Up Queue mode to block lookalike collisions.
              </p>
            </div>

            <div className="space-y-2">
              <label
                htmlFor="cfg-default-mode"
                className="block text-xs font-medium text-slate-600 dark:text-stone-300"
              >
                Default Check-In Mode
              </label>
              <select
                id="cfg-default-mode"
                value={configDraft.defaultCheckInMode}
                onChange={(e) =>
                  setConfigDraft({
                    ...configDraft,
                    defaultCheckInMode: e.target.value as '1to1' | 'walkup',
                  })
                }
                className="w-full h-10 px-3.5 rounded-xl text-sm neu-inset text-slate-900 dark:text-[#F6F1E6] focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
              >
                <option value="1to1">
                  1:1 Attendance ID (On-Demand Camera Default)
                </option>
                <option value="walkup">
                  1:N Walk-Up Queue Mode (Hands-Free Kiosk)
                </option>
              </select>
              <p className="text-[11px] text-slate-500 dark:text-stone-400">
                You can still switch instantly between 1:1 and Walk-Up Queue inside Mark Attendance.
              </p>
            </div>

            <div className="space-y-2">
              <label
                htmlFor="cfg-resolution"
                className="block text-xs font-medium text-slate-600 dark:text-stone-300"
              >
                Camera Sensor Resolution
              </label>
              <select
                id="cfg-resolution"
                value={configDraft.cameraResolution}
                onChange={(e) =>
                  setConfigDraft({
                    ...configDraft,
                    cameraResolution: e.target.value as '480p' | '720p',
                  })
                }
                className="w-full h-10 px-3.5 rounded-xl text-sm neu-inset text-slate-900 dark:text-[#F6F1E6] focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
              >
                <option value="480p">
                  640×480 @ 30fps (Aadhaar Sub-Second Default)
                </option>
                <option value="720p">
                  1280×720 HD (Higher Snapshot Detail)
                </option>
              </select>
              <p className="text-[11px] text-slate-500 dark:text-stone-400">
                480p cuts WebGL frame upload latency by 3× with identical 150×150 ResNet-34 accuracy.
              </p>
            </div>

            <div className="space-y-2">
              <label
                htmlFor="cfg-cutoff"
                className="block text-xs font-medium text-slate-600 dark:text-stone-300"
              >
                Daily Late Arrival Cutoff
              </label>
              <input
                id="cfg-cutoff"
                type="time"
                value={configDraft.lateCutoffTime}
                onChange={(e) =>
                  setConfigDraft({
                    ...configDraft,
                    lateCutoffTime: e.target.value,
                  })
                }
                className="w-full h-10 px-3.5 rounded-xl font-mono text-sm neu-inset text-slate-900 dark:text-[#F6F1E6] focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
              />
              <p className="text-[11px] text-slate-500 dark:text-stone-400">
                Check-ins recorded after this local time are marked Late.
              </p>
            </div>

            <div className="space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <label
                  htmlFor="cfg-cooldown"
                  className="font-medium text-slate-600 dark:text-stone-300"
                >
                  Duplicate Check-In Cooldown
                </label>
                <span className="font-mono font-semibold text-slate-900 dark:text-[#E5C675] tabular-nums">
                  {configDraft.checkInCooldownMinutes === 0
                    ? 'Off'
                    : `${configDraft.checkInCooldownMinutes} min`}
                </span>
              </div>
              <input
                id="cfg-cooldown"
                type="range"
                min={0}
                max={480}
                step={15}
                value={configDraft.checkInCooldownMinutes}
                onChange={(e) =>
                  setConfigDraft({
                    ...configDraft,
                    checkInCooldownMinutes: Number(e.target.value),
                  })
                }
                className="w-full neu-range"
              />
              <p className="text-[11px] text-slate-500 dark:text-stone-400">
                Blocks duplicate attendance logs for the same student across both 1:1 PIN and Walk-Up Queue modes (default 180 min).
              </p>
            </div>
          </div>

          <div className="neu-divider" />

          {/* Tactile Sculpted Toggle Switches */}
          <div className="space-y-4">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-slate-900 dark:text-[#F6F1E6]">
                  One-Blink Liveness &amp; Screen Guard
                </p>
                <p className="text-xs text-slate-500 dark:text-stone-400">
                  Always required: one blink and local PIN-to-face matching. Phone and tablet browsers are blocked; phone-screen replays are checked automatically.
                </p>
              </div>
              <span className="shrink-0 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-semibold text-emerald-700 dark:text-emerald-300">
                Always on
              </span>
            </div>

            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-slate-900 dark:text-[#F6F1E6]">
                  Allow Static Photo Enrollment
                </p>
                <p className="text-xs text-slate-500 dark:text-stone-400">
                  Off by default: enforces Live 3-Frame Biometric Capture with blink liveness so unverified static photos cannot be enrolled.
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={configDraft.allowPhotoEnrollment}
                onClick={() =>
                  setConfigDraft({
                    ...configDraft,
                    allowPhotoEnrollment: !configDraft.allowPhotoEnrollment,
                  })
                }
                className={`w-11 h-6 rounded-full neu-inset p-0.5 transition-colors cursor-pointer shrink-0 flex items-center ${
                  configDraft.allowPhotoEnrollment ? 'justify-end' : 'justify-start'
                }`}
              >
                <span
                  className={`w-5 h-5 rounded-full transition-all ${
                    configDraft.allowPhotoEnrollment
                      ? 'bg-blue-600 dark:bg-gradient-to-br dark:from-[#E8CB80] dark:to-[#C29632] shadow-sm'
                      : 'neu-raised-sm'
                  }`}
                />
              </button>
            </div>

            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-slate-900 dark:text-[#F6F1E6]">
                  Verification Success Chime
                </p>
                <p className="text-xs text-slate-500 dark:text-stone-400">
                  Plays a crisp native harmonic chime when a student is verified (ideal for Walk-Up Queue).
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={configDraft.enableAudioChime}
                onClick={() =>
                  setConfigDraft({
                    ...configDraft,
                    enableAudioChime: !configDraft.enableAudioChime,
                  })
                }
                className={`w-11 h-6 rounded-full neu-inset p-0.5 transition-colors cursor-pointer shrink-0 flex items-center ${
                  configDraft.enableAudioChime ? 'justify-end' : 'justify-start'
                }`}
              >
                <span
                  className={`w-5 h-5 rounded-full transition-all ${
                    configDraft.enableAudioChime
                      ? 'bg-blue-600 dark:bg-gradient-to-br dark:from-[#E8CB80] dark:to-[#C29632] shadow-sm'
                      : 'neu-raised-sm'
                  }`}
                />
              </button>
            </div>

            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-slate-900 dark:text-[#F6F1E6]">
                  Mirror Live Camera Preview
                </p>
                <p className="text-xs text-slate-500 dark:text-stone-400">
                  Flips the front-facing webcam horizontally for natural framing.
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={configDraft.mirrorCamera}
                onClick={() =>
                  setConfigDraft({
                    ...configDraft,
                    mirrorCamera: !configDraft.mirrorCamera,
                  })
                }
                className={`w-11 h-6 rounded-full neu-inset p-0.5 transition-colors cursor-pointer shrink-0 flex items-center ${
                  configDraft.mirrorCamera ? 'justify-end' : 'justify-start'
                }`}
              >
                <span
                  className={`w-5 h-5 rounded-full transition-all ${
                    configDraft.mirrorCamera
                      ? 'bg-blue-600 dark:bg-gradient-to-br dark:from-[#E8CB80] dark:to-[#C29632] shadow-sm'
                      : 'neu-raised-sm'
                  }`}
                />
              </button>
            </div>
          </div>
        </section>

        {/* 3. Custom Facial Model Adapter */}
        <section className="neu-surface gold-edge-top rounded-2xl p-6 space-y-5">
          <div>
            <h2 className="font-display text-base font-semibold tracking-tight text-slate-900 dark:text-[#F6F1E6] flex items-center gap-2">
              <Code2 className="w-4 h-4 text-blue-600 dark:text-[#E2C275]" />
              Facial Recognition Model Adapter
            </h2>
            <p className="text-xs text-slate-500 dark:text-stone-400 mt-1">
              Drop in your custom facial recognition backend or local model anytime without rewriting UI components.
            </p>
          </div>

          <div className="neu-divider" />

          <div
            role="group"
            aria-label="Model integration mode"
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          >
            <button
              type="button"
              onClick={() =>
                setConfigDraft({ ...configDraft, modelMode: 'aadhaar_local' })
              }
              className={`p-4 rounded-2xl text-left transition-all cursor-pointer ${
                configDraft.modelMode === 'aadhaar_local'
                  ? 'neu-inset ring-1 ring-blue-600/35 dark:ring-[#D4AF37]/50'
                  : 'neu-btn'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-semibold text-slate-900 dark:text-[#F6F1E6]">
                  On-Device Aadhaar Engine
                </p>
                <span className="font-mono text-[11px] text-emerald-700 dark:text-[#E5C675] tabular-nums">
                  {resolvedBackendLabel}
                </span>
              </div>
              <p className="text-xs text-slate-500 dark:text-stone-400 mt-1 leading-relaxed">
                Runs your SSD MobileNet, 68-Landmark Liveness, and ResNet-34 128-D matching directly in browser with automatic WebGL → WASM → CPU fallback.
              </p>
            </button>

            <button
              type="button"
              onClick={() =>
                setConfigDraft({ ...configDraft, modelMode: 'custom_api' })
              }
              className={`p-4 rounded-2xl text-left transition-all cursor-pointer ${
                configDraft.modelMode === 'custom_api'
                  ? 'neu-inset ring-1 ring-blue-600/35 dark:ring-[#D4AF37]/50'
                  : 'neu-btn'
              }`}
            >
              <p className="text-sm font-semibold text-slate-900 dark:text-[#F6F1E6]">
                Custom Backend API
              </p>
              <p className="text-xs text-slate-500 dark:text-stone-400 mt-1 leading-relaxed">
                POSTs Attendance ID, live 128-D descriptor, and JPEG frame to your server (FastAPI / Node / pgvector).
              </p>
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="cfg-verify-endpoint"
                className="block text-xs font-medium text-slate-600 dark:text-stone-300 mb-1.5"
              >
                Verify Face Endpoint (POST)
              </label>
              <input
                id="cfg-verify-endpoint"
                type="url"
                value={configDraft.modelEndpointUrl}
                onChange={(e) =>
                  setConfigDraft({
                    ...configDraft,
                    modelEndpointUrl: e.target.value,
                  })
                }
                placeholder="http://localhost:8000/api/v1/verify-face"
                className="w-full h-10 px-3.5 rounded-xl font-mono text-xs neu-inset text-slate-900 dark:text-[#F6F1E6] focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
              />
            </div>

            <div>
              <label
                htmlFor="cfg-enroll-endpoint"
                className="block text-xs font-medium text-slate-600 dark:text-stone-300 mb-1.5"
              >
                Enroll Reference Endpoint (POST)
              </label>
              <input
                id="cfg-enroll-endpoint"
                type="url"
                value={configDraft.enrollEndpointUrl}
                onChange={(e) =>
                  setConfigDraft({
                    ...configDraft,
                    enrollEndpointUrl: e.target.value,
                  })
                }
                placeholder="http://localhost:8000/api/v1/enroll-face"
                className="w-full h-10 px-3.5 rounded-xl font-mono text-xs neu-inset text-slate-900 dark:text-[#F6F1E6] focus:outline-none focus:ring-2 focus:ring-blue-600/35 dark:focus:ring-[#D4AF37]/45"
              />
            </div>
          </div>

          <div className="rounded-xl bg-[#08090C] text-stone-200 p-4 font-mono text-[11px] leading-relaxed overflow-x-auto border border-[#D4AF37]/20">
            <p className="text-[#C5A456]/85 mb-2">
              Custom API JSON contract (128-D descriptor included):
            </p>
            <pre>{`// Request sent after live Aadhaar liveness lock:
{ "attendance_id": "23210-ec-053", "descriptor": [0.021, -0.14, ...], "max_euclidean_distance": ${configDraft.maxEuclideanDistance.toFixed(2)}, "liveness_passed": true }

// Expected JSON response:
{ "verified": true, "distance": 0.294, "confidence": 96.4, "liveness_passed": true, "message": "Match confirmed" }`}</pre>
          </div>

          <div className="pt-2 flex flex-wrap items-center justify-between gap-3">
            <button
              type="submit"
              className="h-10 px-5 rounded-xl text-xs font-semibold neu-btn-primary inline-flex items-center gap-1.5 cursor-pointer"
            >
              <Save className="w-3.5 h-3.5" />
              Save Configuration
            </button>

            <button
              type="button"
              onClick={() => {
                onClearWorkspaceData();
                showSavedFeedback('All local students and attendance logs cleared');
              }}
              className="h-9 px-3.5 rounded-xl text-xs font-medium neu-btn text-slate-600 dark:text-slate-300 inline-flex items-center gap-1.5 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Clear All Local Data
            </button>
          </div>
        </section>
      </form>
    </div>
  );
}
