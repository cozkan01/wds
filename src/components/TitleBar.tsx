import { useEffect, useState, useCallback } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

function WinBtn({
  onClick,
  danger = false,
  children,
  label,
}: {
  onClick: () => void;
  danger?: boolean;
  children: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      className={[
        "flex items-center justify-center w-11 h-full cursor-default",
        "transition-colors duration-100 border-0 outline-none",
        danger
          ? "hover:bg-red-500/80 text-white/40 hover:text-white/90"
          : "hover:bg-white/[0.07] text-white/35 hover:text-white/75",
      ].join(" ")}
      // prevent drag from intercepting the click
      onMouseDown={e => e.stopPropagation()}
    >
      {children}
    </button>
  );
}

interface TitleBarProps {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
}

export default function TitleBar({ title, subtitle, right }: TitleBarProps) {
  const [isMaximized, setIsMaximized] = useState(false);
  const win = getCurrentWindow();

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    win.isMaximized().then(setIsMaximized);
    win.onResized(async () => {
      setIsMaximized(await win.isMaximized());
    }).then(fn => { unlisten = fn; });
    return () => { unlisten?.(); };
  }, [win]);

  const handleClose    = useCallback(() => void win.close(),          [win]);
  const handleMinimize = useCallback(() => void win.minimize(),       [win]);
  const handleMaximize = useCallback(() => void win.toggleMaximize(), [win]);

  return (
    <div
      data-tauri-drag-region
      onDoubleClick={handleMaximize}
      className="flex items-center h-9 flex-shrink-0 select-none"
      style={{
        background: "#0c0d11",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      <div
        data-tauri-drag-region
        className="flex items-center gap-2 px-4 pointer-events-none"
      >
        <span className="text-[11.5px] font-light tracking-[-0.01em] text-white/50">
          OmniParser WDS
        </span>
        {subtitle && (
          <span className="text-[9.5px] font-mono text-white/20">{subtitle}</span>
        )}
      </div>

      <div data-tauri-drag-region className="flex-1 h-full" />

      {right && (
        <div
          className="flex items-center px-3"
          onMouseDown={e => e.stopPropagation()}
        >
          {right}
        </div>
      )}

      {/* ── Window controls ─────────────────────────────── */}
      <div className="flex items-stretch h-full">
        {/* Minimize — ─ */}
        <WinBtn onClick={handleMinimize} label="Minimize">
          <svg width="10" height="1" viewBox="0 0 10 1">
            <line x1="0" y1="0.5" x2="10" y2="0.5" stroke="currentColor" strokeWidth="1" />
          </svg>
        </WinBtn>

        {/* Maximize / Restore */}
        <WinBtn onClick={handleMaximize} label={isMaximized ? "Restore" : "Maximize"}>
          {isMaximized ? (
            /* Restore icon — two overlapping squares */
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <rect x="3" y="0" width="7" height="7" rx="0.5" stroke="currentColor" strokeWidth="1" />
              <path d="M0 3h3v7h7" stroke="currentColor" strokeWidth="1" fill="none" />
            </svg>
          ) : (
            /* Maximize icon — single square */
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <rect x="0.5" y="0.5" width="9" height="9" rx="0.5" stroke="currentColor" strokeWidth="1" />
            </svg>
          )}
        </WinBtn>

        {/* Close — × */}
        <WinBtn onClick={handleClose} label="Close" danger>
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <line x1="1" y1="1" x2="9" y2="9" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
            <line x1="9" y1="1" x2="1" y2="9" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
          </svg>
        </WinBtn>
      </div>
    </div>
  );
}
