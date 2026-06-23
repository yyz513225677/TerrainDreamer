import { useAppStore } from "../lib/modeFSM";
import { StatusPill } from "./StatusPill";
import { EmergencyStop } from "./EmergencyStop";
import type { Mode, WSOutbound } from "../lib/protocol";

interface TopBarProps {
  send: (m: WSOutbound) => void;
}

function modeColor(mode: Mode): "ok" | "warn" | "danger" | "idle" | "info" {
  switch (mode) {
    case "manual":
      return "info";
    case "autonomous":
      return "ok";
    case "recording":
      return "warn";
    case "replay":
      return "info";
    case "training":
      return "ok";
    case "estop":
      return "danger";
    default:
      return "idle";
  }
}

function fmtTime(secs: number) {
  const s = Math.floor(secs);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

export function TopBar({ send }: TopBarProps) {
  const t = useAppStore((s) => s.telemetry);
  const mode = useAppStore((s) => s.mode);
  const estop = useAppStore((s) => s.estop);
  const connected = useAppStore((s) => s.connected);
  const mockMode = useAppStore((s) => s.mockMode);

  const rosOk = !!t?.ros_connected || mockMode;
  const gzOk = !!t?.gazebo_connected || mockMode;
  const battery = t?.battery ?? null;
  const collisionCount = t?.collision_count ?? 0;

  return (
    <header className="flex items-center justify-between gap-6 px-4 h-12 border-b border-border-subtle bg-bg-panel">
      <div className="flex items-center gap-3">
        <div className="flex items-baseline gap-2">
          <span className="font-mono font-semibold tracking-widest text-text-primary text-[13px]">
            JACKAL · DREAMER
          </span>
          <span className="text-text-muted text-label">DASHBOARD v0.1</span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <StatusPill label={`ROS ${rosOk ? "OK" : "NC"}`} state={rosOk ? "ok" : "danger"} title="ROS bridge" />
        <StatusPill label={`GZ ${gzOk ? "OK" : "NC"}`} state={gzOk ? "ok" : "danger"} title="Gazebo sim" />
        <StatusPill
          label={mockMode ? "MOCK" : connected ? "WS LIVE" : "WS DOWN"}
          state={mockMode ? "warn" : connected ? "ok" : "danger"}
          blink={mockMode}
        />
        <StatusPill label={`MODE · ${mode.toUpperCase()}`} state={modeColor(mode)} blink={mode === "recording" || estop} />
      </div>

      <div className="flex items-center gap-4">
        <div className="flex flex-col items-end leading-tight">
          <span className="metric-label">sim-time</span>
          <span className="font-mono text-[13px] text-text-primary tabular-nums">
            {fmtTime(t?.sim_time ?? 0)}
          </span>
        </div>
        <div className="flex flex-col items-end leading-tight">
          <span className="metric-label">battery</span>
          <span className="font-mono text-[13px] tabular-nums" style={{ color: battery !== null && battery < 0.2 ? "#ef4444" : "#e3e8ee" }}>
            {battery !== null ? `${(battery * 100).toFixed(0)}%` : "—"}
          </span>
        </div>
        <div className="flex flex-col items-end leading-tight">
          <span className="metric-label">collisions</span>
          <span className="font-mono text-[13px] tabular-nums" style={{ color: collisionCount > 0 ? "#fbbf24" : "#e3e8ee" }}>
            {collisionCount}
          </span>
        </div>
        <EmergencyStop send={send} />
      </div>
    </header>
  );
}
