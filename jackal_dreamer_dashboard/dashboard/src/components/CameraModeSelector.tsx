import { CAMERA_MODES, TERRAIN_QUALITIES, useCameraMode } from "../hooks/useCameraMode";

/**
 * Compact mode + quality selector. Rendered inside the chase-view
 * overlay so the user can switch camera angle and terrain detail
 * without leaving the 3D panel.
 */
export function CameraModeSelector() {
  const { mode, setMode, quality, setQuality, showLidar, toggleLidar,
          showRoutePath, toggleRoutePath, showGrid, toggleGrid } = useCameraMode();

  return (
    <div className="flex flex-col gap-1.5 text-xs">
      <div className="flex gap-1">
        {CAMERA_MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            title={m.hint}
            className={
              "px-2 py-1 rounded-sm font-mono uppercase tracking-wider " +
              (mode === m.id
                ? "bg-cyan-400 text-black"
                : "bg-slate-800/80 text-slate-200 hover:bg-slate-700/80")
            }
          >
            {m.label}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2 text-slate-300">
        <span className="text-[10px] uppercase tracking-widest">Quality</span>
        {TERRAIN_QUALITIES.map((q) => (
          <button
            key={q.id}
            onClick={() => setQuality(q.id)}
            className={
              "px-1.5 py-0.5 rounded-sm font-mono uppercase " +
              (quality === q.id
                ? "bg-amber-400/90 text-black"
                : "bg-slate-800/80 text-slate-300 hover:bg-slate-700/80")
            }
          >
            {q.label}
          </button>
        ))}
      </div>
      <div className="flex gap-2 text-slate-300">
        <Toggle on={showLidar} onClick={toggleLidar} label="LIDAR" />
        <Toggle on={showRoutePath} onClick={toggleRoutePath} label="PATH" />
        <Toggle on={showGrid} onClick={toggleGrid} label="GRID" />
      </div>
    </div>
  );
}

function Toggle({ on, onClick, label }:
                { on: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      className={
        "px-1.5 py-0.5 rounded-sm font-mono uppercase " +
        (on ? "bg-emerald-500/80 text-black"
            : "bg-slate-800/80 text-slate-400 hover:bg-slate-700/80")
      }
    >
      {label}
    </button>
  );
}
