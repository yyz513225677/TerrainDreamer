import { useEffect, useRef, useState } from "react";
import { useAppStore } from "../lib/modeFSM";
import type { HeightSampler } from "../lib/terrainHeight";
import { sampleHeight } from "../lib/terrainHeight";
import { mapToWorld, dem_yaw_to_three_yaw } from "../lib/coordinateTransform";
import type { PoseW } from "../lib/chaseCamera";

/**
 * Mock trajectory: a figure-eight across a 100m × 60m area. Used when
 * no telemetry is flowing (chase view should work standalone — see
 * Phase 2.5 spec §10).
 */
export function mockJackalPose(t_s: number): { x: number; y: number; yaw: number } {
  const A = 50.0;
  const B = 25.0;
  const ω = 0.06; // rad/s
  const x = A * Math.sin(ω * t_s);
  const y = B * Math.sin(2 * ω * t_s);
  // Heading = tangent direction.
  const dx = A * ω * Math.cos(ω * t_s);
  const dy = 2 * B * ω * Math.cos(2 * ω * t_s);
  return { x, y, yaw: Math.atan2(dy, dx) };
}

export interface UseJackalPoseOpts {
  sampler: HeightSampler | null;
  /** Z height to add on top of the terrain so the rover sits ON the ground. */
  rover_ride_height_m?: number;
}

/**
 * Returns the live Jackal pose in Three.js world coordinates, updated
 * each animation frame. Falls back to a mock trajectory whenever the
 * store has no telemetry or mockMode is set.
 */
export function useJackalPose(opts: UseJackalPoseOpts): PoseW {
  const telemetry = useAppStore((s) => s.telemetry);
  const mockMode = useAppStore((s) => s.mockMode);
  const start_t = useRef<number>(performance.now() / 1000);
  const [pose, setPose] = useState<PoseW>({
    x: 0, y: 0, z: 0, yaw_rad: 0,
  });

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const t = performance.now() / 1000 - start_t.current;
      let x_m: number, y_m: number, yaw_dem: number;
      if (telemetry && !mockMode) {
        x_m = telemetry.pose.x;
        y_m = telemetry.pose.y;
        yaw_dem = telemetry.pose.yaw;
      } else {
        const mp = mockJackalPose(t);
        x_m = mp.x;
        y_m = mp.y;
        yaw_dem = mp.yaw;
      }
      const ride = opts.rover_ride_height_m ?? 0.12;
      const z_m = (opts.sampler
        ? sampleHeight(opts.sampler, { x: x_m, y: y_m })
        : 0) + ride;
      const w = mapToWorld({ x: x_m, y: y_m, z: z_m });
      setPose({
        x: w.x, y: w.y, z: w.z,
        yaw_rad: dem_yaw_to_three_yaw(yaw_dem),
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [telemetry, mockMode, opts.sampler, opts.rover_ride_height_m]);

  return pose;
}
