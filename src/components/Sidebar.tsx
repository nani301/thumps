import {
  BarChart3,
  Camera,
  ClipboardList,
  Menu,
  ScanFace,
  Settings,
  UserPlus,
  X,
} from 'lucide-react';
import type { NavSection, UserProfile } from '../types';
import { Avatar } from './Avatar';

interface SidebarProps {
  activeSection: NavSection;
  onSelectSection: (section: NavSection) => void;
  userProfile: UserProfile;
  mobileOpen: boolean;
  onToggleMobile: (open: boolean) => void;
}

const NAV_ITEMS: {
  id: NavSection;
  label: string;
  icon: typeof BarChart3;
}[] = [
  { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
  { id: 'mark', label: 'Mark Attendance', icon: Camera },
  { id: 'register', label: 'Register Student', icon: UserPlus },
  { id: 'logs', label: 'Attendance Logs', icon: ClipboardList },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export function Sidebar({
  activeSection,
  onSelectSection,
  userProfile,
  mobileOpen,
  onToggleMobile,
}: SidebarProps) {
  const handleSelect = (id: NavSection) => {
    onSelectSection(id);
    onToggleMobile(false);
  };

  const sidebarContent = (
    <div className="flex flex-col h-full neu-canvas px-4 py-5 select-none dark:border-r dark:border-[#D4AF37]/15">
      {/* Slim Minimal Brand Lockup */}
      <div className="px-2 pb-6 flex items-center justify-between shrink-0">
        <button
          type="button"
          onClick={() => handleSelect('dashboard')}
          className="flex items-center gap-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-600 dark:focus-visible:ring-[#D4AF37] rounded-xl cursor-pointer group"
        >
          <span className="w-9 h-9 rounded-xl neu-raised-sm text-blue-600 dark:text-[#E2C275] flex items-center justify-center shrink-0 transition-transform duration-150 group-active:scale-95">
            <ScanFace className="w-4 h-4" strokeWidth={2.1} />
          </span>
          <span className="font-display text-[15px] font-bold tracking-tight text-slate-800 dark:text-[#F5EFE2] whitespace-nowrap">
            VeriMark
          </span>
        </button>

        <button
          type="button"
          onClick={() => onToggleMobile(false)}
          aria-label="Close navigation"
          className="lg:hidden p-2 rounded-xl neu-btn text-slate-600 dark:text-stone-300"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="neu-divider mb-5 mx-1" />

      {/* Minimalist Tactile Navigation */}
      <nav
        aria-label="Primary navigation"
        className="flex-1 space-y-2 overflow-y-auto px-0.5 py-1"
      >
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeSection === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => handleSelect(item.id)}
              aria-current={isActive ? 'page' : undefined}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-[13px] font-medium transition-all duration-150 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600 dark:focus-visible:ring-[#D4AF37] ${
                isActive
                  ? 'neu-inset text-blue-700 dark:text-[#E5C675] font-semibold'
                  : 'text-slate-600 dark:text-stone-400 hover:text-slate-900 dark:hover:text-[#F5EFE2] hover:bg-white/35 dark:hover:bg-white/[0.025]'
              }`}
            >
              <Icon
                className={`w-4 h-4 shrink-0 ${
                  isActive
                    ? 'text-blue-600 dark:text-[#E2C275]'
                    : 'text-slate-500 dark:text-stone-500'
                }`}
                strokeWidth={isActive ? 2.15 : 1.8}
              />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Sculpted Operator Profile Footer */}
      <div className="pt-4 shrink-0">
        <div className="neu-divider mb-4 mx-1" />

        <button
          type="button"
          onClick={() => handleSelect('settings')}
          className="w-full flex items-center gap-3 p-2.5 rounded-2xl neu-btn text-left cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600 dark:focus-visible:ring-[#D4AF37]"
        >
          <Avatar src={userProfile.avatarUrl} name={userProfile.fullName} size="sm" />
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-slate-800 dark:text-[#F5EFE2] truncate leading-tight">
              {userProfile.fullName}
            </p>
            <p className="text-[11px] text-slate-500 dark:text-stone-400 truncate mt-0.5">
              {userProfile.role}
            </p>
          </div>
        </button>
      </div>
    </div>
  );

  return (
    <>
      <aside className="hidden lg:block w-56 shrink-0 h-screen sticky top-0">
        {sidebarContent}
      </aside>

      {mobileOpen && (
        <div className="lg:hidden fixed inset-0 z-50 flex">
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-[2px]"
            onClick={() => onToggleMobile(false)}
            aria-hidden="true"
          />
          <div className="relative w-64 max-w-[82vw] h-full z-10 shadow-2xl">
            {sidebarContent}
          </div>
        </div>
      )}
    </>
  );
}

export function MobileMenuButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Open navigation menu"
      className="lg:hidden p-2 rounded-xl neu-btn text-slate-700 dark:text-[#E2C275] cursor-pointer"
    >
      <Menu className="w-4 h-4" />
    </button>
  );
}
