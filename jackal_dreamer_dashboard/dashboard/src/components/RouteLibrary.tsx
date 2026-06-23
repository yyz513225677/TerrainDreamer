import { useState } from "react";
import { useAppStore } from "../lib/modeFSM";
import { Panel } from "./Panel";
import { StatusPill } from "./StatusPill";
import type { RouteMeta, WSOutbound } from "../lib/protocol";

interface RouteLibraryProps {
  send: (m: WSOutbound) => void;
}

function outcomeState(o: RouteMeta["outcome"]): "ok" | "warn" | "danger" | "idle" {
  switch (o) {
    case "reached_goal":
      return "ok";
    case "timeout":
      return "warn";
    case "collision":
      return "danger";
    default:
      return "idle";
  }
}

export function RouteLibrary({ send }: RouteLibraryProps) {
  const routes = useAppStore((s) => s.routes);
  const replay = useAppStore((s) => s.replay);
  const setReplay = useAppStore((s) => s.setReplay);
  const setMode = useAppStore((s) => s.setMode);
  const [selected, setSelected] = useState<string | null>(null);

  const onReplay = (r: RouteMeta) => {
    setMode("replay");
    send({ cmd: "set_mode", mode: "replay" });
    setReplay(true, r.route_id);
    send({ cmd: "replay", action: "start", route_id: r.route_id });
  };

  const onStopReplay = () => {
    setReplay(false, null);
    send({ cmd: "replay", action: "stop", route_id: replay.routeId ?? "" });
  };

  return (
    <Panel
      title="Route Library"
      accent="#f59e0b"
      className="h-full"
      right={
        <span className="metric-label">
          <span className="font-mono text-text-primary">{routes.length}</span> saved
        </span>
      }
    >
      <div className="overflow-y-auto h-full">
        {routes.length === 0 && (
          <div className="p-4 metric-label text-center">No routes recorded yet</div>
        )}
        <ul className="divide-y divide-border-subtle/60">
          {routes.map((r) => {
            const isActive = replay.routeId === r.route_id;
            const isSelected = selected === r.route_id;
            return (
              <li
                key={r.route_id}
                className={`p-2 cursor-pointer hover:bg-bg-raised/60 ${isSelected ? "bg-bg-raised" : ""}`}
                onClick={() => setSelected(r.route_id)}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-label text-text-primary truncate" title={r.route_id}>
                    {r.route_id}
                  </span>
                  <StatusPill label={r.outcome.replace("_", " ")} state={outcomeState(r.outcome)} />
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1 text-label text-text-muted font-mono tabular-nums">
                  <span>op·{r.operator_id}</span>
                  <span>env·{r.environment_id}</span>
                  <span>{r.duration_s.toFixed(1)}s</span>
                  <span>{r.frame_count}f</span>
                  <span style={{ color: r.collisions > 0 ? "#fbbf24" : "#7a8590" }}>
                    coll·{r.collisions}
                  </span>
                  {r.success_rate !== undefined && (
                    <span style={{ color: r.success_rate > 0.8 ? "#10b981" : r.success_rate > 0.5 ? "#fbbf24" : "#ef4444" }}>
                      sr·{(r.success_rate * 100).toFixed(0)}%
                    </span>
                  )}
                  {r.route_following_error !== undefined && (
                    <span>err·{r.route_following_error.toFixed(2)}m</span>
                  )}
                  {r.priority === 1.0 && (
                    <span style={{ color: "#f59e0b" }}>★ expert</span>
                  )}
                </div>
                {isSelected && (
                  <div className="flex gap-1 mt-2">
                    <button className="btn" onClick={() => setSelected(r.route_id)}>
                      PREVIEW
                    </button>
                    {isActive ? (
                      <button className="btn btn-danger" onClick={onStopReplay}>
                        STOP REPLAY
                      </button>
                    ) : (
                      <button className="btn btn-active" onClick={() => onReplay(r)}>
                        ▸ REPLAY
                      </button>
                    )}
                    <button className="btn" disabled title="Compare vs Dreamer (Phase 4)">
                      COMPARE
                    </button>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </Panel>
  );
}
