import { useAppStore } from "../lib/modeFSM";
import { MetricRow } from "./MetricRow";
import { Panel } from "./Panel";
import { StatusPill } from "./StatusPill";

function RiskBar({ value, label }: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(1, value));
  const color = pct > 0.7 ? "#ef4444" : pct > 0.4 ? "#fbbf24" : "#10b981";
  return (
    <div className="px-3 py-1.5 border-b border-border-subtle/60 last:border-b-0">
      <div className="flex items-center justify-between">
        <span className="metric-label">{label}</span>
        <span className="font-mono text-label tabular-nums" style={{ color }}>
          {(pct * 100).toFixed(0)}%
        </span>
      </div>
      <div className="mt-1 h-1 bg-bg-base border border-border-subtle">
        <div
          className="h-full transition-all"
          style={{ width: `${pct * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export function TelemetryPanel() {
  const t = useAppStore((s) => s.telemetry);
  const pose = t?.pose;
  const odom = t?.odom;
  const imu = t?.imu;

  return (
    <Panel title="Robot Telemetry" accent="#22d3ee" className="h-full">
      <div className="flex flex-col h-full overflow-y-auto">
        <div className="px-3 py-2 border-b border-border-subtle/60 flex items-center justify-between">
          <span className="metric-label">collision</span>
          <StatusPill
            label={t?.collision ? "CONTACT" : "CLEAR"}
            state={t?.collision ? "danger" : "ok"}
            blink={t?.collision}
          />
        </div>

        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">pose</span>
          </div>
          <MetricRow label="x" value={pose?.x} unit="m" precision={2} />
          <MetricRow label="y" value={pose?.y} unit="m" precision={2} />
          <MetricRow label="yaw" value={pose?.yaw !== undefined ? (pose.yaw * 180) / Math.PI : null} unit="°" precision={1} />
        </div>

        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">velocity</span>
          </div>
          <MetricRow label="lin v" value={odom?.linear_x} unit="m/s" precision={3} />
          <MetricRow label="ang ω" value={odom?.angular_z} unit="rad/s" precision={3} />
        </div>

        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">imu attitude</span>
          </div>
          <MetricRow label="roll" value={imu ? (imu.roll * 180) / Math.PI : null} unit="°" precision={1} />
          <MetricRow label="pitch" value={imu ? (imu.pitch * 180) / Math.PI : null} unit="°" precision={1} />
          <MetricRow label="yaw" value={imu ? (imu.yaw * 180) / Math.PI : null} unit="°" precision={1} />
        </div>

        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">imu angular vel</span>
          </div>
          <MetricRow label="wx" value={imu?.angular_velocity.x} unit="rad/s" precision={3} />
          <MetricRow label="wy" value={imu?.angular_velocity.y} unit="rad/s" precision={3} />
          <MetricRow label="wz" value={imu?.angular_velocity.z} unit="rad/s" precision={3} />
        </div>

        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">imu linear accel</span>
          </div>
          <MetricRow label="ax" value={imu?.linear_acceleration.x} unit="m/s²" precision={2} />
          <MetricRow label="ay" value={imu?.linear_acceleration.y} unit="m/s²" precision={2} />
          <MetricRow label="az" value={imu?.linear_acceleration.z} unit="m/s²" precision={2} />
        </div>

        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">risk</span>
          </div>
          <RiskBar label="stuck" value={t?.stuck_score ?? 0} />
          <RiskBar label="rollover" value={t?.rollover_risk ?? 0} />
        </div>
      </div>
    </Panel>
  );
}
