import { create } from "zustand";
import type {
  DreamerMetrics,
  LidarSample,
  Mode,
  RouteMeta,
  TelemetrySample,
} from "./protocol";

export interface AppState {
  mode: Mode;
  estop: boolean;
  connected: boolean;
  mockMode: boolean;

  telemetry: TelemetrySample | null;
  lidar: LidarSample | null;
  dreamer: DreamerMetrics | null;

  rewardHistory: number[];
  valueHistory: number[];
  wmLossHistory: number[];
  actorLossHistory: number[];
  criticLossHistory: number[];
  bcLossHistory: number[];

  pathTrail: Array<[number, number]>;
  collisionMarkers: Array<[number, number]>;

  routes: RouteMeta[];

  recording: {
    active: boolean;
    startedAt: number | null;
    frames: number;
    routeId: string;
    operatorId: string;
    environmentId: string;
    addedToBuffer: boolean;
    priority: number;
  };

  replay: {
    active: boolean;
    routeId: string | null;
  };

  setMode: (mode: Mode) => void;
  setEstop: (active: boolean) => void;
  resetEstop: () => void;
  setConnected: (c: boolean) => void;
  setMockMode: (m: boolean) => void;

  applyTelemetry: (t: TelemetrySample) => void;
  applyLidar: (l: LidarSample) => void;
  applyDreamer: (d: DreamerMetrics) => void;
  applyRoutes: (r: RouteMeta[]) => void;
  appendRoute: (r: RouteMeta) => void;

  setRecordingMeta: (
    patch: Partial<{ routeId: string; operatorId: string; environmentId: string }>,
  ) => void;
  startRecording: () => void;
  stopRecording: () => void;
  bumpRecordingFrame: () => void;
  markAddedToBuffer: () => void;

  setReplay: (active: boolean, routeId: string | null) => void;

  clearTrail: () => void;
  clearRecordedData: () => void;
}

const MAX_HISTORY = 90;
const MAX_TRAIL = 600;

function push(arr: number[], v: number) {
  const next = arr.length >= MAX_HISTORY ? arr.slice(arr.length - MAX_HISTORY + 1) : arr.slice();
  next.push(v);
  return next;
}

export const useAppStore = create<AppState>((set) => ({
  mode: "idle",
  estop: false,
  connected: false,
  mockMode: false,

  telemetry: null,
  lidar: null,
  dreamer: null,

  rewardHistory: [],
  valueHistory: [],
  wmLossHistory: [],
  actorLossHistory: [],
  criticLossHistory: [],
  bcLossHistory: [],

  pathTrail: [],
  collisionMarkers: [],

  routes: [],

  recording: {
    active: false,
    startedAt: null,
    frames: 0,
    routeId: "",
    operatorId: "leo",
    environmentId: "varied",
    addedToBuffer: false,
    priority: 1.0,
  },

  replay: { active: false, routeId: null },

  setMode: (mode) =>
    set((s) => {
      if (s.estop) return s; // E-STOP supersedes
      return { mode };
    }),

  setEstop: (active) =>
    set((s) => ({
      estop: active,
      mode: active ? "estop" : s.mode === "estop" ? "idle" : s.mode,
    })),

  resetEstop: () => set({ estop: false, mode: "idle" }),

  setConnected: (connected) => set({ connected }),
  setMockMode: (mockMode) => set({ mockMode }),

  applyTelemetry: (t) =>
    set((s) => {
      const trail = s.pathTrail.slice();
      const last = trail[trail.length - 1];
      const pt: [number, number] = [t.pose.x, t.pose.y];
      if (!last || Math.hypot(last[0] - pt[0], last[1] - pt[1]) > 0.05) {
        trail.push(pt);
        if (trail.length > MAX_TRAIL) trail.shift();
      }
      const markers = s.collisionMarkers.slice();
      if (t.collision && (!markers.length || Math.hypot(markers[markers.length - 1][0] - pt[0], markers[markers.length - 1][1] - pt[1]) > 0.3)) {
        markers.push(pt);
      }
      return {
        telemetry: t,
        pathTrail: trail,
        collisionMarkers: markers,
        recording: s.recording.active
          ? { ...s.recording, frames: s.recording.frames + 1 }
          : s.recording,
      };
    }),

  applyLidar: (l) => set({ lidar: l }),

  applyDreamer: (d) =>
    set((s) => ({
      dreamer: d,
      rewardHistory: push(s.rewardHistory, d.reward_est),
      valueHistory: push(s.valueHistory, d.value_est),
      wmLossHistory: push(s.wmLossHistory, d.world_model_loss),
      actorLossHistory: push(s.actorLossHistory, d.actor_loss),
      criticLossHistory: push(s.criticLossHistory, d.critic_loss),
      bcLossHistory: push(s.bcLossHistory, d.bc_loss),
    })),

  applyRoutes: (routes) => set({ routes }),
  appendRoute: (r) => set((s) => ({ routes: [r, ...s.routes] })),

  setRecordingMeta: (patch) =>
    set((s) => ({
      recording: {
        ...s.recording,
        ...(patch.routeId !== undefined && { routeId: patch.routeId }),
        ...(patch.operatorId !== undefined && { operatorId: patch.operatorId }),
        ...(patch.environmentId !== undefined && { environmentId: patch.environmentId }),
      },
    })),

  startRecording: () =>
    set((s) => ({
      recording: {
        ...s.recording,
        active: true,
        startedAt: Date.now() / 1000,
        frames: 0,
        addedToBuffer: false,
        routeId:
          s.recording.routeId ||
          `demo_${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`,
      },
    })),

  stopRecording: () =>
    set((s) => ({
      recording: { ...s.recording, active: false },
    })),

  bumpRecordingFrame: () =>
    set((s) => ({
      recording: { ...s.recording, frames: s.recording.frames + 1 },
    })),

  markAddedToBuffer: () =>
    set((s) => ({
      recording: { ...s.recording, addedToBuffer: true },
    })),

  setReplay: (active, routeId) => set({ replay: { active, routeId } }),

  clearTrail: () => set({ pathTrail: [], collisionMarkers: [] }),

  clearRecordedData: () =>
    set((s) => ({
      pathTrail: [],
      collisionMarkers: [],
      recording: { ...s.recording, frames: 0, addedToBuffer: false, startedAt: null },
    })),
}));
