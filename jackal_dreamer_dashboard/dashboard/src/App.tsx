import { useWebSocket } from "./hooks/useWebSocket";
import { useMockGenerator } from "./hooks/useMockGenerator";
import { useAppStore } from "./lib/modeFSM";
import { TopBar } from "./components/TopBar";
import { TelemetryPanel } from "./components/TelemetryPanel";
import { BEVMap } from "./components/BEVMap";
import { DreamerPanel } from "./components/DreamerPanel";
import { SimControlPanel } from "./components/SimControlPanel";
import { ManualControl } from "./components/ManualControl";
import { RecordingPanel } from "./components/RecordingPanel";
import { RouteLibrary } from "./components/RouteLibrary";
import { JackalChaseView } from "./components/JackalChaseView";

export default function App() {
  const { send } = useWebSocket();
  useMockGenerator();
  const estop = useAppStore((s) => s.estop);

  return (
    <div className="min-h-screen flex flex-col bg-bg-base text-text-primary">
      <TopBar send={send} />

      <main className="flex-1 grid grid-cols-12 gap-3 p-3 min-h-0">
        {/* Row 1: large center chase view + minimap + right column */}
        <div className="col-span-8 min-h-0 min-w-0" style={{ minHeight: 520 }}>
          <JackalChaseView className="w-full h-full rounded border border-slate-700" />
        </div>
        <div className="col-span-4 min-h-0 min-w-0 flex flex-col gap-3">
          <div className="min-h-0" style={{ height: 240 }}>
            <BEVMap />
          </div>
          <div className="min-h-0 flex-1">
            <TelemetryPanel />
          </div>
        </div>

        {/* Row 2: Dreamer · Sim Control · Manual · Recording · Route Library */}
        <div className="col-span-3 min-h-0 min-w-0">
          <DreamerPanel />
        </div>
        <div className="col-span-2 min-h-0 min-w-0">
          <SimControlPanel send={send} />
        </div>
        <div className="col-span-2 min-h-0 min-w-0">
          <ManualControl send={send} />
        </div>
        <div className="col-span-2 min-h-0 min-w-0">
          <RecordingPanel send={send} />
        </div>
        <div className="col-span-3 min-h-0 min-w-0">
          <RouteLibrary send={send} />
        </div>
      </main>

      {estop && <div className="estop-shroud" />}
    </div>
  );
}
