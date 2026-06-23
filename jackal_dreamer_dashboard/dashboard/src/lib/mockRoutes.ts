import type { RouteMeta } from "./protocol";

function makeArcPath(
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  curvature: number,
  steps: number,
): Array<[number, number]> {
  const path: Array<[number, number]> = [];
  const cx = (startX + endX) / 2 + curvature * (endY - startY);
  const cy = (startY + endY) / 2 - curvature * (endX - startX);
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const mt = 1 - t;
    const x = mt * mt * startX + 2 * mt * t * cx + t * t * endX;
    const y = mt * mt * startY + 2 * mt * t * cy + t * t * endY;
    path.push([x, y]);
  }
  return path;
}

export const MOCK_ROUTES: RouteMeta[] = [
  {
    route_id: "demo_2026-06-02T14-08-11",
    operator_id: "leo",
    environment_id: "varied",
    started_at: "2026-06-02T14:08:11Z",
    duration_s: 88.4,
    frame_count: 884,
    spawn_xy: [0.0, 0.0],
    goal_xy: [38.0, -12.0],
    outcome: "reached_goal",
    collisions: 0,
    priority: 1.0,
    success_rate: 1.0,
    route_following_error: 0.21,
    path: makeArcPath(0, 0, 38, -12, 0.18, 80),
  },
  {
    route_id: "demo_2026-06-01T11-43-22",
    operator_id: "leo",
    environment_id: "rocks",
    started_at: "2026-06-01T11:43:22Z",
    duration_s: 124.7,
    frame_count: 1247,
    spawn_xy: [-5.0, 2.0],
    goal_xy: [25.0, 18.0],
    outcome: "reached_goal",
    collisions: 1,
    priority: 1.0,
    success_rate: 0.92,
    route_following_error: 0.47,
    path: makeArcPath(-5, 2, 25, 18, -0.22, 80),
  },
  {
    route_id: "demo_2026-05-30T09-12-04",
    operator_id: "mira",
    environment_id: "slopes",
    started_at: "2026-05-30T09:12:04Z",
    duration_s: 62.1,
    frame_count: 621,
    spawn_xy: [0.0, 0.0],
    goal_xy: [-20.0, 14.0],
    outcome: "timeout",
    collisions: 2,
    priority: 0.6,
    success_rate: 0.45,
    route_following_error: 0.83,
    path: makeArcPath(0, 0, -20, 14, 0.3, 80),
  },
];
