/**
 * Chase-camera math (pure functions — no Three.js imports so this file
 * is testable in plain ts-node).
 *
 * Frame: Three.js world (+X east, +Y up, +Z south).
 */

export interface CameraOffsetConfig {
  /** Metres behind the robot (along its local -X). */
  behind_m: number;
  /** Metres lateral (positive = robot's left). */
  lateral_m: number;
  /** Metres above the robot. */
  height_m: number;
  /** Look-at target metres above base_link. */
  lookat_height_m: number;
  /** Smoothing alpha — `new = α·target + (1−α)·old`. 1.0 = instant. */
  smooth_alpha: number;
}

export const CHASE_DEFAULTS: CameraOffsetConfig = {
  behind_m: 6.0,
  lateral_m: 0.0,
  height_m: 3.0,
  lookat_height_m: 0.5,
  smooth_alpha: 0.18,
};

export const DRIVER_DEFAULTS: CameraOffsetConfig = {
  behind_m: -0.4,         // slightly *forward* of base_link
  lateral_m: 0.0,
  height_m: 0.6,
  lookat_height_m: 0.5,
  smooth_alpha: 0.35,
};

export interface PoseW {
  x: number;
  y: number;        // world Y (up — usually the terrain elevation)
  z: number;        // world Z
  yaw_rad: number;  // about world Y axis (Three.js convention)
}

export interface CameraPose {
  position: { x: number; y: number; z: number };
  lookAt: { x: number; y: number; z: number };
}

/**
 * Compute the target camera pose given the robot pose and an offset
 * config. The camera sits behind / above the robot's body frame,
 * looking at a point slightly above the robot.
 */
export function chasePoseFor(robot: PoseW, cfg: CameraOffsetConfig): CameraPose {
  // Robot forward direction (Three.js world). yaw=0 → +X.
  const fwd_x = Math.cos(robot.yaw_rad);
  const fwd_z = -Math.sin(robot.yaw_rad);   // -sin because Three.js +Y rotates +X→-Z
  const right_x = -fwd_z;
  const right_z = fwd_x;

  const px = robot.x - fwd_x * cfg.behind_m + right_x * cfg.lateral_m;
  const py = robot.y + cfg.height_m;
  const pz = robot.z - fwd_z * cfg.behind_m + right_z * cfg.lateral_m;

  return {
    position: { x: px, y: py, z: pz },
    lookAt: {
      x: robot.x,
      y: robot.y + cfg.lookat_height_m,
      z: robot.z,
    },
  };
}

/** Exponential smoothing between two camera poses. α=1 → instant, 0 → frozen. */
export function smoothPose(current: CameraPose, target: CameraPose,
                           alpha: number): CameraPose {
  const a = Math.max(0, Math.min(1, alpha));
  const lerp = (a0: number, a1: number) => a0 + (a1 - a0) * a;
  return {
    position: {
      x: lerp(current.position.x, target.position.x),
      y: lerp(current.position.y, target.position.y),
      z: lerp(current.position.z, target.position.z),
    },
    lookAt: {
      x: lerp(current.lookAt.x, target.lookAt.x),
      y: lerp(current.lookAt.y, target.lookAt.y),
      z: lerp(current.lookAt.z, target.lookAt.z),
    },
  };
}

/**
 * Top-down "BEV" camera looking straight down at the robot. Useful
 * for the minimap and for the Top-down camera mode.
 */
export function topDownPoseFor(robot: PoseW,
                               altitude_m: number = 40): CameraPose {
  return {
    position: { x: robot.x, y: robot.y + altitude_m, z: robot.z },
    lookAt: { x: robot.x, y: robot.y, z: robot.z },
  };
}
