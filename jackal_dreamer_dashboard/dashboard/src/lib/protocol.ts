// WebSocket message protocol — mirrors design doc §4.5

export type Mode =
  | "idle"
  | "manual"
  | "autonomous"
  | "recording"
  | "replay"
  | "training"
  | "estop";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Pose2D {
  x: number;
  y: number;
  yaw: number;
}

export interface OdomSample {
  linear_x: number;
  angular_z: number;
}

export interface IMUSample {
  roll: number;
  pitch: number;
  yaw: number;
  angular_velocity: Vec3;
  linear_acceleration: Vec3;
}

export interface TelemetrySample {
  timestamp: number;
  sim_time: number;
  pose: Pose2D;
  odom: OdomSample;
  imu: IMUSample;
  collision: boolean;
  stuck_score: number;
  rollover_risk: number;
  collision_count: number;
  goal_xy: [number, number] | null;
  ros_connected: boolean;
  gazebo_connected: boolean;
  battery: number;
}

export interface LidarSample {
  ranges: number[];
  intensities?: number[];
  angle_min?: number;
  angle_increment?: number;
}

export interface DreamerMetrics {
  action_linear: number;
  action_angular: number;
  reward_est: number;
  value_est: number;
  uncertainty: number;
  world_model_loss: number;
  actor_loss: number;
  critic_loss: number;
  bc_loss: number;
  replay_buffer_size: number;
  human_demo_ratio: number;
  planned_route?: Array<[number, number]>;
}

export type RouteOutcome = "reached_goal" | "collision" | "timeout" | "aborted";

export interface RouteMeta {
  route_id: string;
  operator_id: string;
  environment_id: string;
  started_at: string;
  duration_s: number;
  frame_count: number;
  spawn_xy: [number, number];
  goal_xy: [number, number];
  outcome: RouteOutcome;
  collisions: number;
  priority: number;
  success_rate?: number;
  route_following_error?: number;
  path?: Array<[number, number]>;
}

export type RouteEvent =
  | { kind: "list"; routes: RouteMeta[] }
  | { kind: "added"; route: RouteMeta }
  | { kind: "replay_pose"; xy: [number, number] };

export type WSInbound =
  | { topic: "telemetry"; payload: TelemetrySample }
  | { topic: "lidar"; payload: LidarSample }
  | { topic: "dreamer"; payload: DreamerMetrics }
  | { topic: "mode"; payload: { mode: Mode; estop: boolean } }
  | { topic: "route"; payload: RouteEvent };

export type WSOutbound =
  | { cmd: "set_mode"; mode: Mode }
  | { cmd: "estop"; active: boolean }
  | { cmd: "reset" }
  | { cmd: "manual_cmd"; linear: number; angular: number }
  | {
      cmd: "recording";
      action: "start" | "stop" | "save";
      route_id?: string;
      operator_id?: string;
      environment_id?: string;
    }
  | { cmd: "replay"; action: "start" | "stop"; route_id: string };
