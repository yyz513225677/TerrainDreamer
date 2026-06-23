import { useAppStore } from "../lib/modeFSM";
import { useKeyboardControl } from "../hooks/useKeyboardControl";
import { Panel } from "./Panel";
import type { Mode, WSOutbound } from "../lib/protocol";

interface ManualControlProps {
  send: (m: WSOutbound) => void;
}

const MODES: Array<{ id: Mode; label: string }> = [
  { id: "manual", label: "Manual" },
  { id: "autonomous", label: "Autonomous" },
  { id: "recording", label: "Recording" },
  { id: "replay", label: "Replay" },
  { id: "training", label: "Training" },
];

function ArrowButton({
  label,
  send,
  vec,
  disabled,
}: {
  label: string;
  send: (m: WSOutbound) => void;
  vec: { linear: number; angular: number };
  disabled: boolean;
}) {
  return (
    <button
      className="btn"
      disabled={disabled}
      onMouseDown={() => send({ cmd: "manual_cmd", linear: vec.linear, angular: vec.angular })}
      onMouseUp={() => send({ cmd: "manual_cmd", linear: 0, angular: 0 })}
      onMouseLeave={() => send({ cmd: "manual_cmd", linear: 0, angular: 0 })}
      title={`Drive ${label}`}
    >
      {label}
    </button>
  );
}

export function ManualControl({ send }: ManualControlProps) {
  const mode = useAppStore((s) => s.mode);
  const estop = useAppStore((s) => s.estop);
  const setMode = useAppStore((s) => s.setMode);
  const { ownsCmdVel } = useKeyboardControl(send);

  const onMode = (m: Mode) => {
    setMode(m);
    send({ cmd: "set_mode", mode: m });
  };

  const disabled = !ownsCmdVel;

  return (
    <Panel title="Mode · Manual Control" accent={ownsCmdVel ? "#22d3ee" : "#3a4350"} className="h-full">
      <div className="flex flex-col gap-3 p-3 h-full">
        <div className="grid grid-cols-5 gap-1">
          {MODES.map((m) => (
            <button
              key={m.id}
              className={`btn ${mode === m.id ? "btn-active" : ""}`}
              disabled={estop}
              onClick={() => onMode(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 metric-label">
          <span>cmd_vel owner:</span>
          <span
            className="font-mono uppercase tracking-wider"
            style={{ color: ownsCmdVel ? "#22d3ee" : "#7a8590" }}
          >
            {ownsCmdVel ? "DASHBOARD (WASD)" : mode === "autonomous" ? "DREAMER" : mode === "replay" ? "REPLAY" : "—"}
          </span>
        </div>

        <div className="grid grid-cols-3 gap-1 max-w-[200px]">
          <span />
          <ArrowButton label="W ▲" send={send} vec={{ linear: 1.0, angular: 0 }} disabled={disabled} />
          <span />
          <ArrowButton label="A ◀" send={send} vec={{ linear: 0, angular: 1.5 }} disabled={disabled} />
          <button
            className="btn"
            disabled={disabled}
            onClick={() => send({ cmd: "manual_cmd", linear: 0, angular: 0 })}
            title="Stop (Space)"
          >
            STOP
          </button>
          <ArrowButton label="D ▶" send={send} vec={{ linear: 0, angular: -1.5 }} disabled={disabled} />
          <span />
          <ArrowButton label="S ▼" send={send} vec={{ linear: -1.0, angular: 0 }} disabled={disabled} />
          <span />
        </div>

        <div className="metric-label leading-tight">
          Keyboard: <span className="font-mono text-text-primary">W</span>/<span className="font-mono text-text-primary">S</span> drive · <span className="font-mono text-text-primary">A</span>/<span className="font-mono text-text-primary">D</span> turn · <span className="font-mono text-text-primary">SHIFT</span> boost · <span className="font-mono text-text-primary">SPACE</span> stop
        </div>
      </div>
    </Panel>
  );
}
