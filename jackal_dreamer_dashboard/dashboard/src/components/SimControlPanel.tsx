import { useAppStore } from "../lib/modeFSM";
import { Panel } from "./Panel";
import type { WSOutbound } from "../lib/protocol";

interface SimControlPanelProps {
  send: (m: WSOutbound) => void;
}

export function SimControlPanel({ send }: SimControlPanelProps) {
  const setMode = useAppStore((s) => s.setMode);
  const mode = useAppStore((s) => s.mode);
  const estop = useAppStore((s) => s.estop);
  const clearTrail = useAppStore((s) => s.clearTrail);
  const clearRecordedData = useAppStore((s) => s.clearRecordedData);

  const disabled = estop;

  return (
    <Panel title="Sim Control" accent="#7a8590" className="h-full">
      <div className="grid grid-cols-2 gap-2 p-3">
        <button
          className="btn btn-ok"
          disabled={disabled}
          onClick={() => {
            setMode("autonomous");
            send({ cmd: "set_mode", mode: "autonomous" });
          }}
          title="Start simulation"
        >
          ▸ START
        </button>
        <button
          className="btn"
          disabled={disabled}
          onClick={() => {
            setMode("idle");
            send({ cmd: "set_mode", mode: "idle" });
          }}
          title="Pause simulation"
        >
          ▌▌ PAUSE
        </button>
        <button
          className="btn"
          disabled={disabled}
          onClick={() => send({ cmd: "reset" })}
          title="Reset Gazebo world"
        >
          RESET WORLD
        </button>
        <button
          className="btn"
          disabled={disabled}
          onClick={() => send({ cmd: "reset" })}
          title="Reset robot pose to spawn"
        >
          RESET ROBOT
        </button>
        <button
          className="btn"
          disabled={disabled}
          onClick={() => send({ cmd: "reset" })}
          title="Respawn robot"
        >
          SPAWN ROBOT
        </button>
        <button
          className="btn"
          onClick={() => clearTrail()}
          title="Clear trajectory overlay"
        >
          CLEAR ROUTE
        </button>
        <button
          className="btn btn-danger col-span-2"
          onClick={() => clearRecordedData()}
          title="Clear current recorded buffer"
        >
          CLEAR RECORDED DATA
        </button>
      </div>
      <div className="px-3 pb-2 metric-label">
        active mode · <span className="font-mono uppercase text-text-primary">{mode}</span>
      </div>
    </Panel>
  );
}
