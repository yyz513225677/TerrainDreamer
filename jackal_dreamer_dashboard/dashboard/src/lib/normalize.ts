// Adapter: bridge wire format → frontend protocol format.
// The ROS bridge serializes ROS msgs with field names that match the .msg
// definitions (wm_loss, stuck, imu_rpy, etc.). The frontend protocol uses
// more readable names (world_model_loss, stuck_score, imu.angular_velocity).
// This file is the ONLY place that knows about both shapes.

import type { WSInbound } from "./protocol";

type AnyMsg = { topic?: string; payload?: Record<string, unknown> } & Record<
  string,
  unknown
>;

function vec(a: unknown): { x: number; y: number; z: number } {
  if (Array.isArray(a) && a.length >= 3) {
    return { x: Number(a[0]) || 0, y: Number(a[1]) || 0, z: Number(a[2]) || 0 };
  }
  if (a && typeof a === "object") {
    const o = a as Record<string, number>;
    return { x: o.x ?? 0, y: o.y ?? 0, z: o.z ?? 0 };
  }
  return { x: 0, y: 0, z: 0 };
}

function num(v: unknown, fallback = 0): number {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return fallback;
}

/**
 * Translate a single inbound bridge message into the frontend's WSInbound
 * shape. Returns null if the message can't be understood (caller drops it).
 */
export function normalizeInbound(raw: unknown): WSInbound | null {
  if (!raw || typeof raw !== "object") return null;
  const m = raw as AnyMsg;
  const topic = m.topic;
  const p = (m.payload ?? {}) as Record<string, unknown>;
  if (!topic) return null;

  switch (topic) {
    case "telemetry": {
      const rpy = (p.imu_rpy as number[]) ?? [0, 0, 0];
      const angvel = vec(p.imu_angvel);
      const linacc = vec(p.imu_linacc);
      return {
        topic: "telemetry",
        payload: {
          timestamp: num(p.timestamp ?? p.stamp ?? 0),
          sim_time: num(p.sim_time ?? p.timestamp ?? 0),
          pose: {
            x: num((p.pose as Record<string, number>)?.x),
            y: num((p.pose as Record<string, number>)?.y),
            yaw: num((p.pose as Record<string, number>)?.yaw),
          },
          odom: {
            linear_x: num((p.odom as Record<string, number>)?.linear_x),
            angular_z: num((p.odom as Record<string, number>)?.angular_z),
          },
          imu: {
            roll: rpy[0] ?? 0,
            pitch: rpy[1] ?? 0,
            yaw: rpy[2] ?? 0,
            angular_velocity: angvel,
            linear_acceleration: linacc,
          },
          collision: Boolean(p.collision),
          stuck_score: num(p.stuck_score ?? p.stuck),
          rollover_risk: num(p.rollover_risk ?? Math.min(1, Math.abs(rpy[0] ?? 0) / 1.0)),
          collision_count: num(p.collision_count),
          goal_xy: (p.goal_xy as [number, number] | null) ?? null,
          ros_connected: p.ros_connected !== false,
          gazebo_connected: p.gazebo_connected !== false,
          battery: num(p.battery ?? 1.0, 1.0),
        },
      };
    }
    case "lidar": {
      const angle_min = num(p.angle_min ?? -Math.PI, -Math.PI);
      const angle_max = num(p.angle_max ?? Math.PI, Math.PI);
      const ranges = (p.ranges as number[]) ?? [];
      const N = Math.max(ranges.length, 1);
      const angle_increment = num(
        p.angle_increment,
        (angle_max - angle_min) / N,
      );
      return {
        topic: "lidar",
        payload: {
          ranges,
          intensities: p.intensities as number[] | undefined,
          angle_min,
          angle_increment,
        },
      };
    }
    case "dreamer": {
      return {
        topic: "dreamer",
        payload: {
          action_linear: num(p.action_linear),
          action_angular: num(p.action_angular),
          reward_est: num(p.reward_est),
          value_est: num(p.value_est),
          uncertainty: num(p.uncertainty ?? p.uncertainty_est),
          world_model_loss: num(p.world_model_loss ?? p.wm_loss),
          actor_loss: num(p.actor_loss),
          critic_loss: num(p.critic_loss),
          bc_loss: num(p.bc_loss),
          replay_buffer_size: num(p.replay_buffer_size ?? p.buffer_size),
          human_demo_ratio: num(p.human_demo_ratio ?? p.human_ratio),
          planned_route: p.planned_route as Array<[number, number]> | undefined,
        },
      };
    }
    case "mode": {
      // The bridge enforces the enum server-side; we trust the string.
      return {
        topic: "mode",
        payload: {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          mode: (p.mode as any) ?? "idle",
          estop: Boolean(p.estop ?? p.estop_active),
        },
      } as WSInbound;
    }
    case "route":
      // Route events come through unchanged — backend and frontend agree
      // on the shape (kind: "list" | "added" | "replay_pose").
      return { topic: "route", payload: p } as unknown as WSInbound;
    default:
      return null;
  }
}
