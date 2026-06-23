# jackal_dreamer_dashboard — frontend

Web dashboard for the Jackal-on-lunar-DEM rover. Vite + React +
TypeScript + Tailwind + Zustand. Phase 2.5 adds the
**Jackal Chase View** — an in-browser 3D third-person panel that
renders the rover on the actual LOLA-derived heightmap, independent
of the Gazebo Sim GUI.

## Layout

| Region | Content |
|---|---|
| Top bar | mode, ROS/gz connection, sim time, E-STOP |
| Centre (8 / 12 cols) | **JackalChaseView** — Three.js + R3F |
| Right column (4 / 12 cols) | BEV minimap, telemetry |
| Bottom row | Dreamer · SimControl · ManualControl · Recording · RouteLibrary |

## Jackal Chase View — how it works without Gazebo

1. On mount, the panel fetches:
   - `/public/dem/heightmap.png` (16-bit grey) — the same PNG written
     by `scripts/normalize_heightmap.py`.
   - `/public/dem/tile.yaml` — flat YAML produced by the same script.
2. A 2-D RGBA canvas decodes the PNG; `lib/terrainHeight.makeSampler`
   builds a Float32 intensity buffer.
3. `<Canvas>` (React Three Fiber) constructs a `PlaneGeometry`
   (`grid × grid` vertices — 129 / 257 / 513 by quality setting),
   displaces each vertex Y by the bilinearly sampled elevation, and
   applies a lunar-gray PBR material with low-angle sun.
4. Live rover pose comes from `useAppStore.telemetry` (driven by the
   wsClient bridge); when no telemetry arrives or `mockMode` is set,
   `hooks/useJackalPose.ts` falls back to a figure-eight trajectory.
5. The chase camera math is in `lib/chaseCamera.ts` — `chasePoseFor`
   computes a target pose 6 m behind / 3 m above the rover and 23°
   downward, then `smoothPose` exponentially smooths each frame.

**No custom 3D engine, no custom shader.** The displacement is plain
`PlaneGeometry` + per-vertex Y, the rendering is bog-standard Three.js
+ R3F (the project's reuse-first rule).

## Camera modes

| Mode | Description |
|---|---|
| **Chase** | third-person follow, 6 m behind + 3 m up |
| **Top-down** | BEV from 45 m straight above the rover |
| **Free orbit** | drei `<OrbitControls>` — drag to rotate, scroll to zoom |
| **Driver** | cockpit / forward — eye height 0.6 m, slightly forward of base_link |

Switch via the top-left button bar in the 3D panel.

## Terrain quality (perf knob)

| Quality | Grid | Vertices |
|---|---|---|
| Low | 129 × 129 | 16 641 |
| Medium *(default)* | 257 × 257 | 66 049 |
| High | 513 × 513 | 263 169 |

A 1025 × 1025 mesh is intentionally avoided — it stalls on integrated
GPUs.

## Mock mode

The chase view works fully **offline**:

```bash
cd jackal_dreamer_dashboard/dashboard
npm run dev
# open http://localhost:5173
```

Without a wsClient connection the app store has `mockMode=true` after
1.5 s; `useJackalPose` then drives the synthetic figure-eight
trajectory. The terrain, camera, lidar overlay, and route trail all
render against this fake pose so you can validate the visualization
in isolation.

## Live ROS mode

When the `ros_gz_bridge` is up and the bridge sends `telemetry`
messages on the websocket, `mockMode` flips false and the chase view
follows `/lunar_jackal/odom` (relayed by the bridge → telemetry
envelope). No extra config — same component.

## Sensor overlays

* LiDAR rays (cyan) — sampled from `/lunar_jackal/scan` via the
  store's `lidar` slice.
* Current goal — amber translucent cylinder at `telemetry.goal_xy`.
* Route trail — rolling 1024-point amber line behind the rover.
* HUD — speed, yaw, distance-to-goal at the bottom; collision banner
  pulses red if `telemetry.collision`.

## Limitations vs the Gazebo GUI camera

| | Gazebo GUI | JackalChaseView |
|---|---|---|
| Rendering fidelity | high (Ogre2, shadows, lighting model) | medium (R3F flat-shaded) |
| Physics-true contacts | yes | no — pose-only visualization |
| Sensor visualization | physics-accurate | overlay-only |
| Works without gz process | no | **yes (uses cached DEM PNG)** |
| Works in remote browser | only via VNC / streaming | **yes (just HTTP)** |
| Runs on a laptop GPU | heavy | light |

The chase view is a *visualization* of pose + sensor topics; physics
remains Gazebo's job.

## Build / dev / test

```bash
npm install           # first time
npm run dev           # http://localhost:5173
npm run build         # production bundle in dist/
npm test              # vitest (21 chase-view + existing tests)
npx tsc --noEmit      # type-check only
```

Production bundle: ≈ 1.1 MB JS / **308 KB gzipped**, up from the
Phase-2 baseline of 185 KB JS — the +120 KB delta is Three.js + R3F.

## Files added in Phase 2.5

```
src/lib/coordinateTransform.ts     DEM-metre ↔ pixel ↔ Three.js ↔ UV
src/lib/terrainHeight.ts           heightmap loader + bilinear sampler
src/lib/chaseCamera.ts             chase / driver / topdown pose math
src/hooks/useJackalPose.ts         live telemetry → world pose (+ mock fallback)
src/hooks/useCameraMode.ts         mode + quality Zustand store
src/components/JackalChaseView.tsx the R3F panel
src/components/CameraModeSelector.tsx
src/components/ChaseViewOverlay.tsx
src/test/chaseView.test.ts         21 unit tests
public/dem/heightmap.png           copied from lunar_south_pole_gazebo
public/dem/color.png               (future shader use)
public/dem/tile.yaml               metadata
```
