import { useAppStore } from "../lib/modeFSM";
import { CameraModeSelector } from "./CameraModeSelector";

/**
 * 2D HUD drawn on top of the chase-view <Canvas>. Shows:
 *   • top-left  — camera mode selector
 *   • top-right — live data-feed status
 *   • mid-right — Sensors inset: VN-100 IMU readings (ω, a, RPY) +
 *                 LiDAR summary (sample count, min/mean/max range)
 *   • bottom    — speed / yaw / goal HUD + collision banner
 */
export function ChaseViewOverlay() {
  const t = useAppStore((s) => s.telemetry);
  const lidar = useAppStore((s) => s.lidar);
  const mockMode = useAppStore((s) => s.mockMode);

  const speed = Math.abs(t?.odom?.linear_x ?? 0);
  const yaw_deg = t ? (t.pose.yaw * 180 / Math.PI) : 0;
  const goal_dist = t?.goal_xy
    ? Math.hypot(t.goal_xy[0] - t.pose.x, t.goal_xy[1] - t.pose.y)
    : null;
  const collision = !!t?.collision;

  const imu = t?.imu ?? null;
  const w = imu?.angular_velocity ?? { x: 0, y: 0, z: 0 };
  const a = imu?.linear_acceleration ?? { x: 0, y: 0, z: 0 };
  const rpy_deg = imu
    ? { r: imu.roll * 180 / Math.PI,
        p: imu.pitch * 180 / Math.PI,
        y: imu.yaw   * 180 / Math.PI }
    : null;

  // LiDAR aggregate stats from the latest scan.
  const lidarStats = (() => {
    if (!lidar?.ranges?.length) return null;
    let n = 0, min = Infinity, max = -Infinity, sum = 0;
    for (const r of lidar.ranges) {
      if (!isFinite(r) || r <= 0) continue;
      n++; sum += r;
      if (r < min) min = r;
      if (r > max) max = r;
    }
    return n
      ? { n, min, mean: sum / n, max, total: lidar.ranges.length }
      : null;
  })();

  return (
    <>
      {/* Top-left: mode selector */}
      <div className="absolute top-3 left-3 z-10">
        <CameraModeSelector />
      </div>

      {/* Top-right: data feed status */}
      <div className="absolute top-3 right-3 z-10 text-xs font-mono
                      bg-slate-900/80 px-2 py-1 rounded-sm
                      text-slate-300 tracking-wider">
        {mockMode ? (
          <span className="text-amber-300">MOCK</span>
        ) : (
          <span className="text-emerald-400">LIVE /odom</span>
        )}
      </div>

      {/* Mid-right: Sensors inset */}
      <SensorsInset
        w={w} a={a} rpy={rpy_deg} lidarStats={lidarStats} live={!mockMode}
      />

      {/* Bottom HUD bar */}
      <div className="absolute bottom-3 left-3 right-3 z-10
                      flex items-end justify-between
                      font-mono text-slate-100 pointer-events-none">
        <div className="flex gap-4">
          <HudItem label="SPEED" value={`${speed.toFixed(2)} m/s`} />
          <HudItem label="YAW" value={`${yaw_deg.toFixed(1)}°`} />
          <HudItem
            label="GOAL"
            value={goal_dist !== null ? `${goal_dist.toFixed(1)} m` : "—"}
          />
        </div>
        {collision && (
          <div className="bg-red-500/80 px-3 py-1.5 rounded
                          text-white font-bold tracking-widest animate-pulse">
            ⚠ COLLISION
          </div>
        )}
      </div>
    </>
  );
}

/**
 * Mid-right inset: live VectorNav VN-100 IMU readings + LiDAR
 * aggregate stats. Tabular monospace so the values don't jitter as
 * digits change.
 */
function SensorsInset({ w, a, rpy, lidarStats, live }: {
  w: { x: number; y: number; z: number };
  a: { x: number; y: number; z: number };
  rpy: { r: number; p: number; y: number } | null;
  lidarStats: { n: number; total: number;
                min: number; mean: number; max: number } | null;
  live: boolean;
}) {
  return (
    <div className="absolute top-16 right-3 z-10
                    bg-slate-900/85 border border-slate-700/80
                    rounded-sm font-mono text-[11px] text-slate-200
                    px-2.5 py-2 w-[230px] tabular-nums">
      <SensorHeader title="VectorNav VN-100" online={live} />
      <Row3 label="ω rad/s" v={w} colors={["#22d3ee", "#22d3ee", "#22d3ee"]} />
      <Row3 label="a m/s²"  v={a} colors={["#f59e0b", "#f59e0b", "#f59e0b"]} />
      {rpy && (
        <div className="grid grid-cols-[3.4em_repeat(3,1fr)] gap-x-1
                        text-[10.5px] mt-0.5">
          <span className="text-slate-400 uppercase">RPY°</span>
          <span className="text-right">{rpy.r.toFixed(1)}</span>
          <span className="text-right">{rpy.p.toFixed(1)}</span>
          <span className="text-right">{rpy.y.toFixed(1)}</span>
        </div>
      )}

      <div className="border-t border-slate-700/70 my-1.5"></div>
      <SensorHeader title="GPU LiDAR (16 rings)" online={lidarStats !== null} />
      {lidarStats ? (
        <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
          <span className="text-slate-400">samples</span>
          <span className="text-right">{lidarStats.n} / {lidarStats.total}</span>
          <span className="text-slate-400">min</span>
          <span className="text-right">{lidarStats.min.toFixed(2)} m</span>
          <span className="text-slate-400">mean</span>
          <span className="text-right">{lidarStats.mean.toFixed(2)} m</span>
          <span className="text-slate-400">max</span>
          <span className="text-right">{lidarStats.max.toFixed(2)} m</span>
        </div>
      ) : (
        <div className="text-slate-500 italic">no scan</div>
      )}
    </div>
  );
}

function SensorHeader({ title, online }: { title: string; online: boolean }) {
  return (
    <div className="flex items-center justify-between mb-1">
      <span className="text-[10px] uppercase tracking-widest text-slate-400">
        {title}
      </span>
      <span className={
        "w-1.5 h-1.5 rounded-full " +
        (online ? "bg-emerald-400 animate-pulse" : "bg-slate-600")
      } />
    </div>
  );
}

function Row3({ label, v, colors }: {
  label: string;
  v: { x: number; y: number; z: number };
  colors: [string, string, string];
}) {
  const fmt = (n: number) => (Math.abs(n) < 0.005 ? "0.00" : n.toFixed(2));
  return (
    <div className="grid grid-cols-[3.4em_repeat(3,1fr)] gap-x-1
                    text-[10.5px]">
      <span className="text-slate-400 uppercase">{label}</span>
      <span className="text-right" style={{ color: colors[0] }}>{fmt(v.x)}</span>
      <span className="text-right" style={{ color: colors[1] }}>{fmt(v.y)}</span>
      <span className="text-right" style={{ color: colors[2] }}>{fmt(v.z)}</span>
    </div>
  );
}

function HudItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-900/80 rounded-sm px-2 py-1">
      <div className="text-[9px] text-slate-400 tracking-widest uppercase">
        {label}
      </div>
      <div className="text-base font-bold tabular-nums">{value}</div>
    </div>
  );
}
