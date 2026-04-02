import { useState, useCallback, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { motion, AnimatePresence } from "framer-motion";
import {
  FolderOpen, Clipboard, Play, Layers, Eye,
  Loader2, X, ZoomIn, Scan,
} from "lucide-react";
import TitleBar from "../components/TitleBar";

interface AnalysisResult {
  omni_count: number;
  wds_count: number;
  total_count: number;
  output_path: string;
  mask_path?: string;
}

type TabId = "detections" | "mask";
type StatusKind = "idle" | "loading" | "running" | "done" | "error";

const BRIDGE = "http://127.0.0.1:8991";
const toSrc = (p: string) =>
  `${BRIDGE}/image?path=${encodeURIComponent(p)}&t=${Date.now()}`;

function Btn({
  onClick, disabled = false, children, primary = false, id,
}: {
  onClick: () => void; disabled?: boolean; children: React.ReactNode;
  primary?: boolean; id?: string;
}) {
  return (
    <button
      id={id}
      onClick={onClick}
      disabled={disabled}
      className={[
        "cursor-pointer inline-flex items-center gap-1.5 h-7 px-3 rounded-lg",
        "text-[11px] font-medium transition-all duration-200 flex-shrink-0",
        "border outline-none disabled:opacity-30 disabled:cursor-not-allowed",
        primary
          ? "bg-white/[0.09] border-white/[0.14] text-white/80 hover:bg-white/[0.14] hover:text-white"
          : "bg-white/[0.04] border-white/[0.08] text-white/45 hover:bg-white/[0.08] hover:text-white/75",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

// ─── Mono label ───────────────────────────────────────────────────────────────
function MonoLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[9.5px] font-mono tracking-[0.18em] uppercase text-white/25">
      {children}
    </span>
  );
}

// ─── Status bar ───────────────────────────────────────────────────────────────
function StatusBar({ message, kind, running }: {
  message: string; kind: StatusKind; running: boolean;
}) {
  return (
    <div className="flex items-center gap-2.5 px-4 h-7 border-t border-white/[0.05] bg-black/20 flex-shrink-0">
      <span className={[
        "w-1 h-1 rounded-full flex-shrink-0",
        kind === "done"    ? "bg-white/40"
        : kind === "error" ? "bg-red-400/60"
        : kind === "running" || kind === "loading" ? "bg-white/25"
        : "bg-white/10",
      ].join(" ")} />
      <span className="text-[10.5px] font-mono text-white/30 flex-1 truncate">
        {message}
      </span>
      {running && (
        <div className="w-16 h-[2px] rounded-full bg-white/[0.06] overflow-hidden flex-shrink-0">
          <div className="h-full w-1/3 bg-white/20 rounded-full ind-bar" />
        </div>
      )}
    </div>
  );
}

// ─── Stat cell ────────────────────────────────────────────────────────────────
function StatCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 px-3 border-r border-white/[0.05]">
      <MonoLabel>{label}</MonoLabel>
      <span key={value} className="fade-up text-[13px] font-light text-white/70 tabular-nums">
        {value}
      </span>
    </div>
  );
}

// ─── Image panel ──────────────────────────────────────────────────────────────
function ImagePanel({
  title, src, placeholder, onEnlarge,
}: {
  title: string; src?: string; placeholder?: string; onEnlarge?: () => void;
}) {
  return (
    <div className="flex-1 flex flex-col border border-white/[0.06] rounded-lg overflow-hidden min-w-0 bg-white/[0.01]">
      {/* header */}
      <div className="flex items-center justify-between px-3 h-8 border-b border-white/[0.06] bg-white/[0.025] flex-shrink-0">
        <MonoLabel>{title}</MonoLabel>
        {src && onEnlarge && (
          <button
            onClick={onEnlarge}
            className="inline-flex items-center gap-1 text-[10px] text-white/25 hover:text-white/55 transition-colors duration-150 cursor-pointer"
          >
            <ZoomIn size={9} /> expand
          </button>
        )}
      </div>

      {/* body */}
      <div className="flex-1 relative overflow-hidden flex items-center justify-center">
        {src ? (
          <motion.img
            key={src} src={src} alt={title}
            className="img-full"
            style={{ cursor: onEnlarge ? "zoom-in" : "default" }}
            onClick={onEnlarge}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2 }}
          />
        ) : (
          <div className="flex flex-col items-center gap-2.5 text-white/[0.08]">
            <Scan size={20} strokeWidth={1} />
            <span className="text-[10.5px] font-mono text-white/[0.15]">
              {placeholder || "—"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Lightbox ─────────────────────────────────────────────────────────────────
function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }} onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 cursor-zoom-out"
      style={{ backdropFilter: "blur(8px)" }}
    >
      <motion.div
        initial={{ scale: 0.97 }} animate={{ scale: 1 }} exit={{ scale: 0.97 }}
        transition={{ duration: 0.15 }} onClick={e => e.stopPropagation()}
        className="relative cursor-default"
      >
        <img src={src} alt="" className="max-w-[90vw] max-h-[88vh] rounded block" />
        <button
          onClick={onClose}
          className="absolute top-2 right-2 inline-flex items-center justify-center w-6 h-6 rounded bg-black/60 border border-white/[0.08] text-white/40 hover:text-white/70 transition-colors cursor-pointer"
        >
          <X size={11} />
        </button>
      </motion.div>
    </motion.div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function HybridAnalyzer() {
  const [tab, setTab]         = useState<TabId>("detections");
  const [imagePath, setPath]  = useState("");
  const [inputSrc, setInput]  = useState<string | undefined>();
  const [outputSrc, setOutput]= useState<string | undefined>();
  const [maskSrc, setMask]    = useState<string | undefined>();
  const [lightbox, setLB]     = useState<string | undefined>();
  const [stats, setStats]     = useState({ omni: "—", wds: "—", total: "—" });
  const [status, setStatus]   = useState("Ready.");
  const [kind, setKind]       = useState<StatusKind>("idle");
  const [running, setRunning] = useState(false);
  const [ready, setReady]     = useState(false);
  const [isDrag, setDrag]     = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // Model load
  useEffect(() => {
    setStatus("Loading models…"); setKind("loading");
    invoke<string>("load_models")
      .then(() => { setReady(true); setStatus("Models ready."); setKind("idle"); })
      .catch(e  => { setStatus(String(e)); setKind("error"); });
  }, []);

  const loadPath = useCallback((path: string) => {
    setPath(path);
    setOutput(undefined); setMask(undefined);
    setStats({ omni: "—", wds: "—", total: "—" });
    setInput(toSrc(path));
    setStatus(path.split(/[\\/]/).pop()!);
    setKind("idle");
  }, []);

  // File picker
  const pick = useCallback(async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const s = await open({ multiple: false, filters: [{ name: "Images", extensions: ["png","jpg","jpeg","bmp","webp"] }] });
      if (s && typeof s === "string") loadPath(s);
    } catch { fileRef.current?.click(); }
  }, [loadPath]);

  const onFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    const p = f && (f as File & { path?: string }).path;
    if (p) loadPath(p);
    e.target.value = "";
  }, [loadPath]);

  // Paste
  const paste = useCallback(async () => {
    setStatus("Reading clipboard…"); setKind("loading");
    try {
      const path = await invoke<string>("paste_clipboard_image");
      if (path) loadPath(path);
    } catch (e) { setStatus(`Clipboard: ${e}`); setKind("error"); }
  }, [loadPath]);

  useEffect(() => {
    const h = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.key === "v") paste(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [paste]);

  // Drag
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDrag(false);
    const p = (e.dataTransfer.files[0] as File & { path?: string })?.path;
    if (p) loadPath(p);
  }, [loadPath]);

  // Run
  const STEPS = [
    "Step 1/4 — WDS Laplacian…",
    "Step 2/4 — OmniParser YOLO…",
    "Step 3/4 — Filtering misses…",
    "Step 4/4 — Captioning…",
  ];

  const run = useCallback(async () => {
    if (!imagePath || !ready || running) return;
    setRunning(true); setKind("running");
    let i = 0; setStatus(STEPS[0]);
    const t = setInterval(() => { i = Math.min(i + 1, STEPS.length - 1); setStatus(STEPS[i]); }, 8000);
    try {
      const r = await invoke<AnalysisResult>("run_analysis", { imagePath: imagePath });
      clearInterval(t);
      setStats({ omni: String(r.omni_count), wds: String(r.wds_count), total: String(r.total_count) });
      setOutput(toSrc(r.output_path));
      if (r.mask_path) setMask(toSrc(r.mask_path));
      setStatus(`Saved → ${r.output_path.split(/[\\/]/).pop()}`);
      setKind("done");
    } catch (e) {
      clearInterval(t);
      setStatus(String(e)); setKind("error");
    } finally { setRunning(false); }
  }, [imagePath, ready, running]);

  const tabs = [
    { id: "detections" as TabId, label: "Detections", Icon: Eye },
    { id: "mask"       as TabId, label: "Mask",       Icon: Layers },
  ];

  return (
    <div
      className="flex flex-col h-screen bg-[#0c0d11] overflow-hidden"
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
    >
      {/* Drop overlay */}
      <AnimatePresence>
        {isDrag && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-2 z-50 pointer-events-none rounded-xl border border-dashed border-white/[0.12] bg-white/[0.02] flex items-center justify-center"
          >
            <span className="text-[11px] font-mono text-white/25">Drop image</span>
          </motion.div>
        )}
      </AnimatePresence>

      <TitleBar
        title="OmniParser WDS"
        right={
          <div className="flex items-center gap-1.5">
            {ready
              ? <span className="text-[10px] font-mono text-white/25"></span>
              : <><Loader2 size={10} className="text-white/25 spin" /><span className="text-[10px] font-mono text-white/25">loading</span></>}
          </div>
        }
      />

      {/* ── Toolbar ───────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 px-3 h-10 border-b border-white/[0.05] bg-white/[0.01] flex-shrink-0">
        <Btn id="btn-open" onClick={pick}>
          <FolderOpen size={11} strokeWidth={1.5} /> Open
        </Btn>
        <Btn id="btn-paste" onClick={paste}>
          <Clipboard size={11} strokeWidth={1.5} /> Paste
        </Btn>

        <div className="flex-1 text-[10.5px] font-mono text-white/25 truncate px-2 leading-none">
          {imagePath ? imagePath.split(/[\\/]/).pop() : "No image — open, paste (Ctrl+V), or drag & drop"}
        </div>

        <Btn id="btn-run" onClick={run} disabled={!imagePath || !ready || running} primary>
          {running
            ? <><Loader2 size={11} className="spin" strokeWidth={1.5} /> Running…</>
            : <><Play size={11} strokeWidth={1.5} style={{ fill: "currentColor" }} /> Run</>}
        </Btn>
      </div>

      {/* ── Tab row + stats ───────────────────────────────────────────────── */}
      <div className="flex items-stretch h-8 border-b border-white/[0.05] bg-white/[0.01] flex-shrink-0">
        {tabs.map(({ id, label, Icon }) => {
          const active = tab === id;
          return (
            <button
              key={id} id={`tab-${id}`}
              onClick={() => setTab(id)}
              className={[
                "inline-flex items-center gap-1.5 px-3 h-full transition-all duration-150 cursor-pointer font-medium",
                "border-b-[1.5px] text-[11px] border border-t-0 border-l-0 border-r-0",
                active
                  ? "border-white/40 text-white/70"
                  : "border-transparent text-white/25 hover:text-white/45",
              ].join(" ")}
            >
              <Icon size={10} strokeWidth={active ? 2 : 1.5} />
              {label}
            </button>
          );
        })}

        {/* divider */}
        <div className="w-px bg-white/[0.05] my-1.5 mx-1" />

        {/* stats */}
        <div className="flex items-center">
          <StatCell label="omni"  value={stats.omni} />
          <StatCell label="wds"   value={stats.wds} />
          <StatCell label="total" value={stats.total} />
        </div>
      </div>

      {/* ── Content ───────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-hidden p-2.5">
        <AnimatePresence mode="wait">
          {tab === "detections" && (
            <motion.div
              key="det" className="flex gap-2.5 h-full"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.12 }}
            >
              <ImagePanel
                title="Original"
                src={inputSrc}
                placeholder="open or drop an image"
                onEnlarge={inputSrc ? () => setLB(inputSrc) : undefined}
              />
              <ImagePanel
                title="Hybrid Result"
                src={outputSrc}
                placeholder={imagePath ? "press run" : "—"}
                onEnlarge={outputSrc ? () => setLB(outputSrc) : undefined}
              />
            </motion.div>
          )}

          {tab === "mask" && (
            <motion.div
              key="mask" className="flex gap-2.5 h-full"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.12 }}
            >
              <ImagePanel
                title="Laplacian Mask + Boxes"
                src={maskSrc}
                placeholder={imagePath ? "run analysis first" : "—"}
                onEnlarge={maskSrc ? () => setLB(maskSrc) : undefined}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Status ────────────────────────────────────────────────────────── */}
      <StatusBar message={status} kind={kind} running={running} />

      {/* Lightbox */}
      <AnimatePresence>
        {lightbox && <Lightbox src={lightbox} onClose={() => setLB(undefined)} />}
      </AnimatePresence>

      <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={onFile} />
    </div>
  );
}
