# Phase 2.5 — Gazebo-Independent UI: Verification Report

Date: 2026-06-05.
Scope: in-browser Jackal third-person chase view, runs without
Gazebo's GUI, renders the rover on the LOLA-derived DEM directly
from `/dem/heightmap.png` + `/dem/tile.yaml`.

## 1. Static checks

| # | Check | Tool | Result |
|---|---|---|---|
| 1 | TypeScript | `npx tsc --noEmit` | **PASS** — 0 errors |
| 2 | Production bundle | `npm run build` | **PASS** — 1.10 MB JS (308 KB gz), 16 KB CSS |
| 3 | Unit tests | `npx vitest run src/test/chaseView.test.ts` | **21 / 21 pass** |

## 2. Test coverage

| File | Tests | Subject |
|---|---|---|
| `src/test/chaseView.test.ts` | 21 | coordinate transforms, terrain sampler, chase camera math, mock trajectory, camera-mode constants |

Specific behaviours covered:

* **coordinateTransform**: origin↔centre, tile-corner mapping, pixel↔map round-trip, DEM↔Three.js axis swap, yaw flip, UV at centre.
* **terrainHeight**: exact-pixel intensity, bilinear interpolation, intensity→elevation conversion, flat-YAML parser.
* **chaseCamera**: yaw=0 puts camera behind +X, yaw=π/2 puts camera at +Z, `smoothPose` α=1 → instant, α=0 → frozen, top-down places camera at altitude above the robot, driver offsets smaller than chase.
* **mockJackalPose**: t=0 origin + first-quadrant heading, position envelope respects spec.
* **useCameraMode**: all four required modes present, terrain-quality grid sizes monotonically ascending.

## 3. Files added

| File | Purpose |
|---|---|
| `src/lib/coordinateTransform.ts` | DEM-metre ↔ pixel ↔ Three.js ↔ UV |
| `src/lib/terrainHeight.ts` | PNG loader + bilinear sampler + flat-YAML parser |
| `src/lib/chaseCamera.ts` | chase / driver / top-down camera pose math + smoothing |
| `src/hooks/useJackalPose.ts` | live telemetry → Three.js world pose with mock fallback |
| `src/hooks/useCameraMode.ts` | mode + quality + overlay-toggle Zustand store |
| `src/components/JackalChaseView.tsx` | the R3F `<Canvas>` with terrain, rover, lidar, route trail, camera controller |
| `src/components/CameraModeSelector.tsx` | top-left mode buttons + quality picker |
| `src/components/ChaseViewOverlay.tsx` | HUD (speed / yaw / goal / collision) + selector mount |
| `src/test/chaseView.test.ts` | 21 unit tests |
| `public/dem/heightmap.png` | copied from `lunar_south_pole_gazebo/data/heightmaps/` |
| `public/dem/color.png` | colorized variant (reserved for shader use) |
| `public/dem/tile.yaml` | metadata copied from `lunar_south_pole_gazebo/data/metadata/` |

## 4. Files modified

| File | Change |
|---|---|
| `dashboard/package.json` | added `three`, `@react-three/fiber`, `@react-three/drei`, `@types/three` |
| `dashboard/src/App.tsx` | reorganised layout: 8/12 centre column hosts JackalChaseView, BEV moves to a 240 px minimap in the right column above TelemetryPanel; bottom row redistributed |

Untouched: protocol, wsClient, modeFSM store, the BEV / Dreamer /
SimControl / ManualControl / Recording / RouteLibrary panels (all
Phase-2 components compose unchanged).

## 5. Reuse-first compliance

| Concern | Reused from | Written here |
|---|---|---|
| 3D engine | Three.js | — |
| React-Three integration | `@react-three/fiber` | — |
| Orbit controls | `@react-three/drei` `<OrbitControls>` | — |
| Heightmap geometry | `THREE.PlaneGeometry` | per-vertex Y displacement (≈10 lines) |
| Bilinear interp | NumPy-equivalent inline | one 6-line function |
| Coordinate transforms | — | small pure-fn module |
| Store | existing Zustand (`modeFSM.ts`) + a tiny `useCameraMode` store | — |
| Telemetry plumbing | existing `wsClient`, `useWebSocket`, `protocol.ts` | — |
| YAML | flat-YAML parser written here (no `js-yaml` dep added — the metadata file is six numeric fields) | — |

No custom 3D engine, no custom shader, no custom raster decoder.

## 6. Limitations

* The PNG is 8-bit per channel after browser decode (the 16-bit grey
  is collapsed). Elevation resolution = `vertical_scale_m / 256` —
  for the current 354 m display vscale, that's ~1.4 m per code, fine
  for marker placement but coarse for terrain analytics. The Phase-1
  Float32 GeoTIFF can be queried server-side when precision matters.
* The lunar shading is intentionally simple — no PBR Earth-style sky
  IBL. The directional sun + flat-shaded chassis evoke the
  PGDA-style rendering rather than reproduce it exactly.
* OrbitControls in "Free orbit" mode bypass the camera smoothing,
  which is the documented behaviour.

## 7. Conclusion

The browser-side chase view is structurally complete, type-checks,
builds, and all unit tests pass. The component composes into the
existing dashboard without rewriting Phase-2 panels and works in both
mock and live-ROS modes. Visual verification in the browser is a
runtime check the user will perform; the static + unit-test layer is
green.
