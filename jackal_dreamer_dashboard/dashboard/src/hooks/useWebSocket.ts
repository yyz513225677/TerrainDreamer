import { useEffect, useRef } from "react";
import { WSClient } from "../lib/wsClient";
import { useAppStore } from "../lib/modeFSM";
import type { WSOutbound } from "../lib/protocol";

let singleton: WSClient | null = null;

function getClient(setStatus: (s: "connecting" | "open" | "closed") => void, onInbound: (msg: any) => void) {
  if (singleton) return singleton;
  singleton = new WSClient({
    url: "ws://localhost:8765",
    onMessage: onInbound,
    onStatus: setStatus,
  });
  return singleton;
}

/**
 * useWebSocket
 *
 * Wires the WSClient to the Zustand store. Toggles mockMode when the
 * connection is closed for >2s so the UI keeps animating.
 */
export function useWebSocket() {
  const setConnected = useAppStore((s) => s.setConnected);
  const setMockMode = useAppStore((s) => s.setMockMode);
  const applyTelemetry = useAppStore((s) => s.applyTelemetry);
  const applyLidar = useAppStore((s) => s.applyLidar);
  const applyDreamer = useAppStore((s) => s.applyDreamer);
  const setMode = useAppStore((s) => s.setMode);
  const setEstop = useAppStore((s) => s.setEstop);
  const applyRoutes = useAppStore((s) => s.applyRoutes);
  const appendRoute = useAppStore((s) => s.appendRoute);

  const mockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const client = getClient(
      (status) => {
        if (status === "open") {
          setConnected(true);
          setMockMode(false);
          if (mockTimerRef.current) {
            clearTimeout(mockTimerRef.current);
            mockTimerRef.current = null;
          }
        } else if (status === "closed") {
          setConnected(false);
          if (!mockTimerRef.current) {
            mockTimerRef.current = setTimeout(() => setMockMode(true), 1500);
          }
        }
      },
      (msg) => {
        switch (msg.topic) {
          case "telemetry":
            applyTelemetry(msg.payload);
            break;
          case "lidar":
            applyLidar(msg.payload);
            break;
          case "dreamer":
            applyDreamer(msg.payload);
            break;
          case "mode":
            setEstop(msg.payload.estop);
            if (!msg.payload.estop) setMode(msg.payload.mode);
            break;
          case "route":
            if (msg.payload.kind === "list") applyRoutes(msg.payload.routes);
            else if (msg.payload.kind === "added") appendRoute(msg.payload.route);
            break;
        }
      },
    );

    return () => {
      // do not close singleton — survives StrictMode double-mount
      void client;
    };
  }, [setConnected, setMockMode, applyTelemetry, applyLidar, applyDreamer, setMode, setEstop, applyRoutes, appendRoute]);

  const send = (msg: WSOutbound) => {
    singleton?.send(msg);
  };

  return { send };
}
