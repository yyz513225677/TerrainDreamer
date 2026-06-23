import { useAppStore } from "../lib/modeFSM";
import { MetricRow } from "./MetricRow";
import { Panel } from "./Panel";
import { Sparkline } from "./Sparkline";

function LossRow({
  label,
  value,
  history,
  color,
}: {
  label: string;
  value: number | undefined;
  history: number[];
  color: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-1.5 border-b border-border-subtle/60">
      <span className="metric-label flex-shrink-0 w-20">{label}</span>
      <Sparkline data={history} stroke={color} width={120} height={22} />
      <span className="metric-num tabular-nums w-16 text-right" style={{ color }}>
        {value !== undefined && Number.isFinite(value) ? value.toFixed(4) : "—"}
      </span>
    </div>
  );
}

export function DreamerPanel() {
  const d = useAppStore((s) => s.dreamer);
  const rh = useAppStore((s) => s.rewardHistory);
  const vh = useAppStore((s) => s.valueHistory);
  const wm = useAppStore((s) => s.wmLossHistory);
  const ac = useAppStore((s) => s.actorLossHistory);
  const cr = useAppStore((s) => s.criticLossHistory);
  const bc = useAppStore((s) => s.bcLossHistory);

  return (
    <Panel title="Dreamer Policy" accent="#3b82f6" className="h-full">
      <div className="flex flex-col h-full overflow-y-auto">
        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">current action</span>
          </div>
          <MetricRow label="lin v" value={d?.action_linear} unit="m/s" precision={3} accent="#3b82f6" />
          <MetricRow label="ang ω" value={d?.action_angular} unit="rad/s" precision={3} accent="#3b82f6" />
        </div>

        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">estimates</span>
          </div>
          <div className="flex items-center justify-between gap-3 px-3 py-1 border-b border-border-subtle/60">
            <span className="metric-label">reward est</span>
            <Sparkline data={rh} stroke="#10b981" width={90} height={20} />
            <span className="metric-num tabular-nums w-14 text-right" style={{ color: "#10b981" }}>
              {d?.reward_est !== undefined ? d.reward_est.toFixed(3) : "—"}
            </span>
          </div>
          <div className="flex items-center justify-between gap-3 px-3 py-1 border-b border-border-subtle/60">
            <span className="metric-label">value est</span>
            <Sparkline data={vh} stroke="#3b82f6" width={90} height={20} />
            <span className="metric-num tabular-nums w-14 text-right" style={{ color: "#3b82f6" }}>
              {d?.value_est !== undefined ? d.value_est.toFixed(3) : "—"}
            </span>
          </div>
          <MetricRow label="uncertainty" value={d?.uncertainty} precision={4} accent="#fbbf24" />
        </div>

        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">losses</span>
          </div>
          <LossRow label="world model" value={d?.world_model_loss} history={wm} color="#22d3ee" />
          <LossRow label="actor" value={d?.actor_loss} history={ac} color="#3b82f6" />
          <LossRow label="critic" value={d?.critic_loss} history={cr} color="#a855f7" />
          <LossRow label="bc" value={d?.bc_loss} history={bc} color="#f59e0b" />
        </div>

        <div>
          <div className="px-3 pt-2 pb-1">
            <span className="panel-header text-[10px]">replay buffer</span>
          </div>
          <MetricRow
            label="size"
            value={d?.replay_buffer_size !== undefined ? d.replay_buffer_size.toFixed(0) : null}
            unit="samples"
          />
          <MetricRow
            label="human ratio"
            value={d?.human_demo_ratio !== undefined ? d.human_demo_ratio * 100 : null}
            unit="%"
            precision={1}
            accent="#f59e0b"
          />
        </div>
      </div>
    </Panel>
  );
}
