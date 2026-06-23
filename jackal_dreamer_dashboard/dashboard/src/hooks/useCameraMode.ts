import { create } from "zustand";

export type CameraMode = "chase" | "topdown" | "orbit" | "driver";
export type TerrainQuality = "low" | "medium" | "high";

/** Mapping quality → grid resolution in vertices per side. */
export const TERRAIN_GRID_BY_QUALITY: Record<TerrainQuality, number> = {
  low: 129,
  medium: 257,
  high: 513,
};

interface CameraModeStore {
  mode: CameraMode;
  quality: TerrainQuality;
  showLidar: boolean;
  showRoutePath: boolean;
  showWaypointArrow: boolean;
  showGrid: boolean;
  setMode: (m: CameraMode) => void;
  setQuality: (q: TerrainQuality) => void;
  toggleLidar: () => void;
  toggleRoutePath: () => void;
  toggleWaypointArrow: () => void;
  toggleGrid: () => void;
}

export const useCameraMode = create<CameraModeStore>((set) => ({
  mode: "chase",
  quality: "medium",
  showLidar: true,
  showRoutePath: true,
  showWaypointArrow: true,
  showGrid: false,
  setMode: (mode) => set({ mode }),
  setQuality: (quality) => set({ quality }),
  toggleLidar: () => set((s) => ({ showLidar: !s.showLidar })),
  toggleRoutePath: () => set((s) => ({ showRoutePath: !s.showRoutePath })),
  toggleWaypointArrow: () => set((s) => ({ showWaypointArrow: !s.showWaypointArrow })),
  toggleGrid: () => set((s) => ({ showGrid: !s.showGrid })),
}));

/** Available modes — exported separately so tests don't have to construct the store. */
export const CAMERA_MODES: { id: CameraMode; label: string; hint: string }[] = [
  { id: "chase", label: "Chase", hint: "Third-person follow (default)" },
  { id: "topdown", label: "Top-down", hint: "BEV map view" },
  { id: "orbit", label: "Free orbit", hint: "User-controlled orbit" },
  { id: "driver", label: "Driver", hint: "Cockpit / forward-looking" },
];

export const TERRAIN_QUALITIES: { id: TerrainQuality; label: string; verts: number }[] = [
  { id: "low",    label: "Low",    verts: TERRAIN_GRID_BY_QUALITY.low },
  { id: "medium", label: "Medium", verts: TERRAIN_GRID_BY_QUALITY.medium },
  { id: "high",   label: "High",   verts: TERRAIN_GRID_BY_QUALITY.high },
];
