import { useAppStore } from "../lib/modeFSM";
import type { WSOutbound } from "../lib/protocol";

interface EmergencyStopProps {
  send: (m: WSOutbound) => void;
  compact?: boolean;
}

export function EmergencyStop({ send, compact = false }: EmergencyStopProps) {
  const estop = useAppStore((s) => s.estop);
  const setEstop = useAppStore((s) => s.setEstop);
  const resetEstop = useAppStore((s) => s.resetEstop);

  const onClick = () => {
    if (estop) {
      resetEstop();
      send({ cmd: "reset" });
      send({ cmd: "estop", active: false });
    } else {
      setEstop(true);
      send({ cmd: "estop", active: true });
    }
  };

  return (
    <button
      onClick={onClick}
      className={
        compact
          ? `font-mono font-bold uppercase tracking-widest text-[12px] px-3 py-1 border-2 ${estop ? "bg-accent-danger text-white border-accent-danger" : "border-accent-danger text-accent-danger hover:bg-accent-danger/15"}`
          : `font-mono font-bold uppercase tracking-widest text-[14px] px-4 py-2 border-2 ${estop ? "bg-accent-danger text-white border-accent-danger animate-pulse" : "border-accent-danger text-accent-danger hover:bg-accent-danger/15"}`
      }
      title={estop ? "E-STOP latched. Click to reset." : "Emergency stop"}
    >
      {estop ? "● E-STOP LATCHED · RESET" : "E-STOP"}
    </button>
  );
}
