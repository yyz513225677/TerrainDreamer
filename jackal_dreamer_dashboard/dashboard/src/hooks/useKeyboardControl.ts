import { useEffect, useRef } from "react";
import { useAppStore } from "../lib/modeFSM";
import type { WSOutbound } from "../lib/protocol";

const MAX_LIN = 1.0;
const MAX_ANG = 1.5;
const SHIFT_BOOST = 1.6;
const PUBLISH_HZ = 20;

interface KeyState {
  W: boolean;
  S: boolean;
  A: boolean;
  D: boolean;
  SHIFT: boolean;
}

/**
 * useKeyboardControl
 *
 * Listens for W/S/A/D + SPACE keys and emits `manual_cmd` at 20 Hz when
 * the active mode owns /cmd_vel (Manual or Recording). SHIFT boosts speed.
 */
export function useKeyboardControl(send: (msg: WSOutbound) => void) {
  const mode = useAppStore((s) => s.mode);
  const estop = useAppStore((s) => s.estop);
  const ownsCmdVel = !estop && (mode === "manual" || mode === "recording");
  const keysRef = useRef<KeyState>({ W: false, S: false, A: false, D: false, SHIFT: false });

  useEffect(() => {
    if (!ownsCmdVel) {
      keysRef.current = { W: false, S: false, A: false, D: false, SHIFT: false };
      return;
    }

    const onDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      const k = e.key.toLowerCase();
      if (k === "w") keysRef.current.W = true;
      else if (k === "s") keysRef.current.S = true;
      else if (k === "a") keysRef.current.A = true;
      else if (k === "d") keysRef.current.D = true;
      else if (k === "shift") keysRef.current.SHIFT = true;
      else if (k === " ") {
        keysRef.current = { W: false, S: false, A: false, D: false, SHIFT: keysRef.current.SHIFT };
        send({ cmd: "manual_cmd", linear: 0, angular: 0 });
        e.preventDefault();
      }
    };

    const onUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (k === "w") keysRef.current.W = false;
      else if (k === "s") keysRef.current.S = false;
      else if (k === "a") keysRef.current.A = false;
      else if (k === "d") keysRef.current.D = false;
      else if (k === "shift") keysRef.current.SHIFT = false;
    };

    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, [ownsCmdVel, send]);

  // Publish loop @ 20 Hz
  useEffect(() => {
    if (!ownsCmdVel) return;
    const id = setInterval(() => {
      const k = keysRef.current;
      const boost = k.SHIFT ? SHIFT_BOOST : 1.0;
      let lin = 0;
      let ang = 0;
      if (k.W) lin += MAX_LIN;
      if (k.S) lin -= MAX_LIN;
      if (k.A) ang += MAX_ANG;
      if (k.D) ang -= MAX_ANG;
      send({ cmd: "manual_cmd", linear: lin * boost, angular: ang * boost });
    }, 1000 / PUBLISH_HZ);
    return () => clearInterval(id);
  }, [ownsCmdVel, send]);

  return { ownsCmdVel };
}
