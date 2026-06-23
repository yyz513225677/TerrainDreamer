import { useEffect, useRef } from "react";
import { useAppStore } from "../lib/modeFSM";
import { MOCK_ROUTES } from "../lib/mockRoutes";
import type {
  DreamerMetrics,
  LidarSample,
  TelemetrySample,
} from "../lib/protocol";

/**
 * useMockGenerator
 *
 * When the dashboard is in mockMode, this hook drives the store with
 * synthetic telemetry, LiDAR, and Dreamer metrics so the UI renders
 * smoothly without a backend.
 *
 * The mock rover drives a slow lissajous path while LiDAR is a 360-ray
 * sweep with simulated obstacles arranged in a ring.
 */
export function useMockGenerator() {
  const mockMode = useAppStore((s) => s.mockMode);
  const mode = useAppStore((s) => s.mode);
  const estop = useAppStore((s) => s.estop);
  const applyTelemetry = useAppStore((s) => s.applyTelemetry);
  const applyLidar = useAppStore((s) => s.applyLidar);
  const applyDreamer = useAppStore((s) => s.applyDreamer);
  const applyRoutes = useAppStore((s) => s.applyRoutes);
  const routesLoaded = useRef(false);

  const tRef = useRef(0);
  const collisionCountRef = useRef(0);
  const trainRef = useRef(0);

  // Seed routes once
  useEffect(() => {
    if (!routesLoaded.current && mockMode) {
      applyRoutes(MOCK_ROUTES);
      routesLoaded.current = true;
    }
  }, [mockMode, applyRoutes]);

  // Telemetry @ 20 Hz
  useEffect(() => {
    if (!mockMode) return;
    const id = setInterval(() => {
      tRef.current += 0.05;
      const t = tRef.current;
      const moving = !estop && (mode === "manual" || mode === "autonomous" || mode === "recording" || mode === "replay");
      const v = moving ? 0.4 + 0.15 * Math.sin(t * 0.5) : 0;
      const w = moving ? 0.2 * Math.sin(t * 0.3) : 0;

      const x = 8 * Math.cos(t * 0.1);
      const y = 6 * Math.sin(t * 0.13);
      const yaw = Math.atan2(Math.cos(t * 0.13) * 6 * 0.13, -Math.sin(t * 0.1) * 8 * 0.1);

      const collide = Math.sin(t * 0.7) > 0.985;
      if (collide) collisionCountRef.current += 1;

      const sample: TelemetrySample = {
        timestamp: Date.now() / 1000,
        sim_time: t,
        pose: { x, y, yaw },
        odom: { linear_x: v, angular_z: w },
        imu: {
          roll: 0.04 * Math.sin(t * 0.4),
          pitch: 0.06 * Math.cos(t * 0.3),
          yaw,
          angular_velocity: {
            x: 0.02 * Math.sin(t * 0.7),
            y: 0.03 * Math.cos(t * 0.6),
            z: w,
          },
          linear_acceleration: {
            x: 0.1 * Math.sin(t * 1.2),
            y: 0.05 * Math.cos(t * 0.9),
            z: -9.81 + 0.2 * Math.sin(t * 0.5),
          },
        },
        collision: collide,
        stuck_score: Math.max(0, 0.05 + 0.3 * Math.sin(t * 0.2)),
        rollover_risk: Math.max(0, 0.05 + 0.25 * Math.cos(t * 0.18)),
        collision_count: collisionCountRef.current,
        goal_xy: [12, -4],
        ros_connected: true,
        gazebo_connected: true,
        battery: 0.95 - 0.0001 * t,
      };
      applyTelemetry(sample);
    }, 50);
    return () => clearInterval(id);
  }, [mockMode, mode, estop, applyTelemetry]);

  // LiDAR @ 10 Hz
  useEffect(() => {
    if (!mockMode) return;
    const id = setInterval(() => {
      const N = 360;
      const ranges = new Array<number>(N);
      const intensities = new Array<number>(N);
      const t = tRef.current;
      for (let i = 0; i < N; i++) {
        const angle = (i / N) * Math.PI * 2;
        // Base wall ring at ~10 m with noise + a few obstacles
        let r = 9.5 + 0.3 * Math.sin(angle * 3 + t * 0.2) + Math.random() * 0.2;
        // Two simulated obstacles at fixed bearings (rotating slowly)
        const obs1 = ((angle - t * 0.05 + Math.PI * 2) % (Math.PI * 2));
        if (Math.abs(obs1 - 0.6) < 0.15) r = 3.5 + Math.random() * 0.1;
        const obs2 = ((angle - t * 0.03 + Math.PI * 2) % (Math.PI * 2));
        if (Math.abs(obs2 - 2.2) < 0.2) r = 5.2 + Math.random() * 0.1;
        const obs3 = ((angle - t * 0.07 + Math.PI * 2) % (Math.PI * 2));
        if (Math.abs(obs3 - 4.4) < 0.18) r = 4.0 + Math.random() * 0.1;
        ranges[i] = r;
        intensities[i] = 150 + Math.random() * 100;
      }
      const sample: LidarSample = {
        ranges,
        intensities,
        angle_min: 0,
        angle_increment: (Math.PI * 2) / N,
      };
      applyLidar(sample);
    }, 100);
    return () => clearInterval(id);
  }, [mockMode, applyLidar]);

  // Dreamer metrics @ 4 Hz
  useEffect(() => {
    if (!mockMode) return;
    const id = setInterval(() => {
      trainRef.current += 1;
      const k = trainRef.current;
      const decay = Math.exp(-k / 500);
      const d: DreamerMetrics = {
        action_linear: 0.4 + 0.1 * Math.sin(k * 0.05),
        action_angular: 0.15 * Math.cos(k * 0.07),
        reward_est: 0.05 + 0.4 * Math.sin(k * 0.04) + 0.05 * Math.random(),
        value_est: 1.2 + 0.6 * Math.sin(k * 0.02),
        uncertainty: 0.2 * decay + 0.05 + 0.02 * Math.random(),
        world_model_loss: 0.6 * decay + 0.08 + 0.02 * Math.random(),
        actor_loss: -0.4 + 0.5 * decay + 0.04 * Math.random(),
        critic_loss: 0.3 * decay + 0.05 + 0.02 * Math.random(),
        bc_loss: 0.4 * decay + 0.02 + 0.01 * Math.random(),
        replay_buffer_size: Math.min(50000, 200 + k * 12),
        human_demo_ratio: 0.32 + 0.02 * Math.sin(k * 0.03),
      };
      applyDreamer(d);
    }, 250);
    return () => clearInterval(id);
  }, [mockMode, applyDreamer]);
}
