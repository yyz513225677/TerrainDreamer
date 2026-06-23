# Jackal-Dreamer Dashboard — Implementation Plan

This is the execution plan for the MVP. See `dashboard_design.md` for the
"why" + the visual/architectural spec. This doc is the "how" + the
sequenced checklist.

## Definition of MVP done

- [ ] `npm run build` in `dashboard/` produces a clean production bundle.
- [ ] `cd dashboard && npm run dev` opens a polished dashboard at
      <http://localhost:5173> that renders **every** module described in
      §4 of the design doc.
- [ ] Mock-mode renders smooth telemetry + LiDAR scan + a pre-baked
      expert route by default (no live ROS required).
- [ ] All 5 mode-switch buttons toggle the UI state machine; the
      currently-owning mode is unambiguously indicated in the top bar.
- [ ] E-STOP button latches red across the whole UI; reset button
      requires explicit click to unlatch; while latched all manual
      controls visually disabled.
- [ ] Manual control panel accepts W/A/S/D + SPACE and sends
      `manual_cmd` over the websocket (or to the mock generator).
- [ ] Recording panel produces a JSON-Lines file under
      `data/demonstrations/` matching `RouteSample` schema, and writes a
      route manifest under `data/routes/`, and appends a `priority=1.0`
      entry to `data/replay_buffer/index.json`.
- [ ] Route Library lists at least one mock route, lets the user preview
      it on the BEV map, and start a (mock) replay.
- [ ] `colcon build` succeeds for the 5 ROS 2 packages.
- [ ] `pytest tests/` passes (≥3 test files: mode FSM, E-STOP filter,
      recording schema).
- [ ] README has 4 copy-pasteable run commands (dev, prod, ros build,
      run nodes) and 1 troubleshooting section.
- [ ] Code review notes from a `general-purpose` subagent are addressed
      or explicitly deferred.

## Sprint plan

### Sprint 1 — Skeleton + design (this commit)

| Step | Item | Owner | Done? |
|------|------|-------|------|
| 1.1  | Create project tree under `jackal_dreamer_dashboard/` | main | ✓ |
| 1.2  | Write design doc (`docs/dashboard_design.md`) | main | ✓ |
| 1.3  | Write impl plan (this doc) | main | ✓ |

### Sprint 2 — Parallel build (delegated to subagents)

| Step | Item | Owner |
|------|------|-------|
| 2.A  | Build React+TS+Tailwind frontend with all 9 modules + mock-mode generator | agent A (general-purpose) |
| 2.B  | Build ROS 2 packages (msgs + 5 nodes) | agent B (general-purpose) |
| 2.A and 2.B run **in parallel** to compress wall-clock time. Both must produce a passing build before integration. |

### Sprint 3 — Integration + tests + docs (back to main)

| Step | Item |
|------|------|
| 3.1  | Verify the websocket protocol contract matches between bridge + frontend; fix mismatches |
| 3.2  | Write `tests/test_mode_fsm.py`, `tests/test_estop.py`, `tests/test_recording_schema.py` |
| 3.3  | Write `README.md` |
| 3.4  | Smoke: `npm run build`, `pytest tests/`, dev server up |
| 3.5  | Subagent code review (general-purpose, ML-research perspective + frontend perspective) |
| 3.6  | Address any blocker findings; defer the rest |

## Risks + mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Vite + Tailwind 4 + TS 5.x toolchain churn breaks `npm install` | medium | pin Vite 5.x + Tailwind 3.4 + TS 5.4 (battle-tested combo) |
| Subagent produces non-buildable code | medium | brief includes explicit "verify `npm run build` succeeds before returning" / "verify `colcon build` succeeds" |
| ROS 2 nodes can't run without live Jazzy install | low (we have it) | bridge has a `--mock` flag that produces fake telemetry, satisfies MVP |
| Mock-mode and real-mode diverge | medium | bridge + frontend share message types via a single Python and a single TS file that mirror each other |

## Toolchain pin

- Node 20.20.2 + npm 10.8.2 (installed at start of session)
- Vite 5.4.x
- React 18.3.x
- TypeScript 5.5.x
- Tailwind CSS 3.4.x
- ROS 2 Jazzy (already on the box)
- Python 3.12 (sim venv)
