import { useEffect, useState } from "react";
import { useAppStore } from "../lib/modeFSM";
import { Panel } from "./Panel";
import { StatusPill } from "./StatusPill";
import type { WSOutbound } from "../lib/protocol";

interface RecordingPanelProps {
  send: (m: WSOutbound) => void;
}

function fmtDuration(secs: number) {
  const s = Math.floor(secs);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

export function RecordingPanel({ send }: RecordingPanelProps) {
  const rec = useAppStore((s) => s.recording);
  const setRecordingMeta = useAppStore((s) => s.setRecordingMeta);
  const startRecording = useAppStore((s) => s.startRecording);
  const stopRecording = useAppStore((s) => s.stopRecording);
  const markAddedToBuffer = useAppStore((s) => s.markAddedToBuffer);
  const setMode = useAppStore((s) => s.setMode);
  const mode = useAppStore((s) => s.mode);

  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!rec.active) return;
    const id = setInterval(() => setTick((t) => t + 1), 250);
    return () => clearInterval(id);
  }, [rec.active]);

  const elapsed = rec.active && rec.startedAt ? Date.now() / 1000 - rec.startedAt : 0;
  void tick;

  const onStart = () => {
    if (mode !== "recording") {
      setMode("recording");
      send({ cmd: "set_mode", mode: "recording" });
    }
    startRecording();
    send({
      cmd: "recording",
      action: "start",
      route_id: rec.routeId || undefined,
      operator_id: rec.operatorId,
      environment_id: rec.environmentId,
    });
  };

  const onStop = () => {
    stopRecording();
    send({ cmd: "recording", action: "stop" });
  };

  const onSave = () => {
    send({ cmd: "recording", action: "save", route_id: rec.routeId });
    markAddedToBuffer();
  };

  return (
    <Panel
      title="Recording"
      accent="#f59e0b"
      className="h-full"
      right={
        <StatusPill
          label={rec.active ? "REC" : "IDLE"}
          state={rec.active ? "warn" : "idle"}
          blink={rec.active}
        />
      }
    >
      <div className="flex flex-col gap-2 p-3">
        <label className="flex items-center gap-2">
          <span className="metric-label w-20">route_id</span>
          <input
            className="input-field flex-1"
            placeholder="auto"
            value={rec.routeId}
            onChange={(e) => setRecordingMeta({ routeId: e.target.value })}
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="metric-label w-20">operator</span>
          <input
            className="input-field flex-1"
            value={rec.operatorId}
            onChange={(e) => setRecordingMeta({ operatorId: e.target.value })}
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="metric-label w-20">environ.</span>
          <input
            className="input-field flex-1"
            value={rec.environmentId}
            onChange={(e) => setRecordingMeta({ environmentId: e.target.value })}
          />
        </label>

        <div className="flex items-center justify-between gap-3 px-1 py-1">
          <div className="flex flex-col">
            <span className="metric-label">duration</span>
            <span className="metric-num tabular-nums" style={{ color: rec.active ? "#fbbf24" : undefined }}>
              {fmtDuration(elapsed)}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="metric-label">frames</span>
            <span className="metric-num tabular-nums">{rec.frames}</span>
          </div>
          <div className="flex flex-col">
            <span className="metric-label">priority</span>
            <span className="metric-num tabular-nums" style={{ color: "#f59e0b" }}>
              {rec.priority.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <button className="btn btn-ok" onClick={onStart} disabled={rec.active}>
            ● START
          </button>
          <button className="btn btn-danger" onClick={onStop} disabled={!rec.active}>
            ▌▌ STOP
          </button>
          <button className="btn" onClick={onSave} disabled={rec.active || rec.frames === 0}>
            SAVE
          </button>
        </div>

        <button
          className="btn btn-active"
          onClick={() => {
            markAddedToBuffer();
            send({ cmd: "recording", action: "save", route_id: rec.routeId });
          }}
          disabled={rec.active || rec.frames === 0 || rec.addedToBuffer}
        >
          {rec.addedToBuffer ? "✓ ADDED · priority=1.0" : "ADD TO EXPERT BUFFER"}
        </button>
      </div>
    </Panel>
  );
}
