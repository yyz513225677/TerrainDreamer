"""Real-time viewer for the TerrainDreamer training stack.

Window layout:
  ┌──────────────────────────┬────────────────────────┐
  │  LiDAR top-down scatter  │  IMU time-series (3 row)│
  │  (rover frame)           │  - linear accel (x/y/z) │
  │                          │  - angular vel  (x/y/z) │
  │                          │  - tilt (rad)           │
  ├──────────────────────────┴────────────────────────┤
  │           Travel path (XY world frame)            │
  └────────────────────────────────────────────────────┘

This runs as a standalone ROS 2 node — it does NOT touch the trainer. Just
launch it in another terminal while training is going:

  source /opt/ros/jazzy/setup.bash
  source ros2_ws/install/setup.bash  # if needed
  python3 scripts/td_viewer.py

Subscribes:
  /j100_0001/sensors/lidar3d_0/points  (PointCloud2)
  /j100_0001/sensors/imu_0/data        (Imu)
  /j100_0001/platform/odom             (Odometry)  — path source

Requires matplotlib + numpy + rclpy + sensor_msgs_py.
"""
from __future__ import annotations

import math
import threading
from collections import deque

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import Imu, PointCloud2

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation


def _apply_style():
    plt.style.use("dark_background")
    rcParams.update({
        "figure.facecolor":  "#11151c",
        "axes.facecolor":    "#181d27",
        "axes.edgecolor":    "#3a414d",
        "axes.labelcolor":   "#cad3df",
        "xtick.color":       "#9aa3b1",
        "ytick.color":       "#9aa3b1",
        "text.color":        "#e3e8ef",
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   "#e3e8ef",
        "axes.titlepad":     8,
        "axes.labelsize":    9,
        "axes.labelweight":  "normal",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "grid.color":        "#2b3140",
        "grid.linewidth":    0.6,
        "grid.alpha":        0.6,
        "legend.frameon":    False,
        "legend.fontsize":   9,
    })


# ── Display tunables ───────────────────────────────────────────────────────
LIDAR_RANGE_M = 15.0           # half-extent shown in the LiDAR scatter
LIDAR_MAX_POINTS = 8000        # downsample if more than this in a frame
IMU_HISTORY_S = 30.0           # seconds shown in IMU time-series
IMU_BUFFER_LEN = 600           # at ~10 Hz → 60 s of history
PATH_MAX_POINTS = 5000
IMU_TOPIC = "/j100_0001/sensors/imu_0/data"


def quat_to_rpy(x, y, z, w):
    """Return (roll, pitch, yaw)."""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class ViewerNode(Node):
    def __init__(self):
        super().__init__("td_viewer")
        sensor_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            PointCloud2, "/j100_0001/sensors/lidar3d_0/points",
            self._on_points, sensor_qos)
        # IMU bridge from gz publishes RELIABLE — using BEST_EFFORT here
        # silently fails to match. Try RELIABLE first; if no msgs, the user
        # can swap topic via the IMU_TOPIC constant.
        self.create_subscription(
            Imu, IMU_TOPIC, self._on_imu, reliable_qos)
        self._imu_count = 0
        # Ground-truth pose published by the trainer (/td/rover_gt). Falls
        # back to /j100_0001/platform/odom (drifting) if trainer isn't up.
        self.create_subscription(
            PoseStamped, "/td/rover_gt", self._on_gt, reliable_qos)
        self.create_subscription(
            Odometry, "/j100_0001/platform/odom",
            self._on_odom, reliable_qos)
        self._have_gt = False

        self.lock = threading.Lock()
        self.latest_points = np.zeros((0, 3), dtype=np.float32)   # x, y, z
        self.imu_t = deque(maxlen=IMU_BUFFER_LEN)
        self.imu_acc = deque(maxlen=IMU_BUFFER_LEN)
        self.imu_gyro = deque(maxlen=IMU_BUFFER_LEN)
        self.imu_tilt = deque(maxlen=IMU_BUFFER_LEN)
        self.latest_vel = (0.0, 0.0, 0.0)         # rover linear velocity, odom
        self.latest_gyro = (0.0, 0.0, 0.0)         # rover angular velocity, IMU
        self.path_xy = deque(maxlen=PATH_MAX_POINTS)
        self._t_start = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info("td_viewer ready, subscribed to lidar/imu/odom")

    # ── Subscribers ────────────────────────────────────────────────────
    def _on_points(self, msg: PointCloud2) -> None:
        try:
            struct = pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            xyz = np.stack(
                [struct["x"], struct["y"], struct["z"]],
                axis=-1).astype(np.float32)
        except Exception:
            return
        if xyz.shape[0] == 0:
            return
        # Drop the ground plane: keep points above the rover base.
        xyz = xyz[xyz[:, 2] > -0.4]
        if xyz.shape[0] > LIDAR_MAX_POINTS:
            idx = np.random.choice(xyz.shape[0], LIDAR_MAX_POINTS, replace=False)
            xyz = xyz[idx]
        with self.lock:
            self.latest_points = xyz   # keep z for height-based coloring

    def _on_imu(self, msg: Imu) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - self._t_start
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z
        q = msg.orientation
        roll, pitch, _yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
        tilt = math.hypot(roll, pitch)
        with self.lock:
            self.imu_t.append(t)
            self.imu_acc.append((ax, ay, az))
            self.imu_gyro.append((wx, wy, wz))
            self.imu_tilt.append(tilt)
            self.latest_gyro = (wx, wy, wz)
        self._imu_count += 1
        if self._imu_count == 1 or self._imu_count % 200 == 0:
            self.get_logger().info(
                f"IMU msgs received: {self._imu_count} (latest tilt={math.degrees(tilt):.1f}°)")

    def _on_gt(self, msg: PoseStamped) -> None:
        x = msg.pose.position.x
        y = msg.pose.position.y
        with self.lock:
            self.path_xy.append((x, y))
            self._have_gt = True

    def _on_odom(self, msg: Odometry) -> None:
        # Position: fallback when trainer isn't publishing /td/rover_gt.
        # Velocity: always from odom (IMU has acceleration only).
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        with self.lock:
            self.latest_vel = (vx, vy, vz)
            if not self._have_gt:
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                self.path_xy.append((x, y))


# ── Plotting ───────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = ViewerNode()

    # Spin rclpy in a daemon thread; matplotlib must own the main thread.
    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(node), name="rclpy_spin", daemon=True)
    spin_thread.start()

    _apply_style()
    fig = plt.figure(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title("TerrainDreamer • Live Telemetry")
    fig.suptitle("TerrainDreamer  •  Live Telemetry",
                 fontsize=15, fontweight="bold", color="#e3e8ef", y=0.985)
    # Left half: LiDAR (full height). Right half: travel path (top 2/3) +
    # compact IMU (bottom 1/3).
    gs = fig.add_gridspec(2, 2,
                          width_ratios=[1.0, 1.0],
                          height_ratios=[2.0, 1.0],
                          left=0.05, right=0.97, top=0.93, bottom=0.07,
                          wspace=0.20, hspace=0.28)

    # LiDAR scatter — large left half, top-down in rover frame.
    # Points colored by Z (height) using turbo colormap so it matches the
    # terrain colormap. Low = blue, mid = green/yellow, high = red.
    import matplotlib.cm as cm
    LIDAR_Z_MIN, LIDAR_Z_MAX = -0.4, 2.0
    ax_lidar = fig.add_subplot(gs[:, 0])
    ax_lidar.set_aspect("equal")
    ax_lidar.set_xlim(-LIDAR_RANGE_M, LIDAR_RANGE_M)
    ax_lidar.set_ylim(-LIDAR_RANGE_M, LIDAR_RANGE_M)
    ax_lidar.set_title("LiDAR  ·  rover frame, height-colored")
    ax_lidar.set_xlabel("x [m]"); ax_lidar.set_ylabel("y [m]")
    ax_lidar.grid(True)
    rover_dot = ax_lidar.plot([0], [0], marker="^", ms=14,
                              mfc="#ff5252", mec="#fff", mew=1.4,
                              ls="None", label="rover")[0]
    lidar_scatter = ax_lidar.scatter(
        [], [], s=4, c=[], cmap="turbo",
        vmin=LIDAR_Z_MIN, vmax=LIDAR_Z_MAX, alpha=0.85)
    cbar = fig.colorbar(lidar_scatter, ax=ax_lidar, label="z [m]",
                        shrink=0.75, pad=0.02, aspect=14)
    cbar.ax.tick_params(labelsize=7, colors="#9aa3b1")
    cbar.set_label("z [m]", color="#cad3df", fontsize=8)
    ax_lidar.legend(loc="upper right")

    # IMU — combined view: forward speed (vx) + angular velocity (ωx/ωy/ωz).
    # The rover is a diff-drive platform so vy / vz are uninformative.
    ax_imu = fig.add_subplot(gs[1, 1])
    ax_imu.set_title("Speed + angular velocity", fontsize=10)
    vel_x = [0]
    gyro_x = [2, 3, 4]   # gap at index 1 to separate groups
    VEL_C  = "#42a5f5"   # cool blue for linear velocity
    GYRO_C = ["#ef5350", "#66bb6a", "#ffa726"]    # red / green / orange triad
    vel_bars = ax_imu.bar(vel_x, [0], color=[VEL_C], width=0.65,
                          edgecolor="#0c2d52", linewidth=1.0)
    ax_imu.axhline(0, color="#3a414d", lw=0.7)
    ax_imu.set_xticks(vel_x + gyro_x)
    ax_imu.set_xticklabels(["vx", "ωx", "ωy", "ωz"], fontweight="bold")
    ax_imu.set_ylim(-1.0, 1.0)
    ax_imu.set_ylabel("Linear vel [m/s]", color="#90caf9")
    ax_imu.tick_params(axis="y", colors="#90caf9")
    ax_imu.grid(True, axis="y")
    ax_imu_r = ax_imu.twinx()
    gyro_bars = ax_imu_r.bar(gyro_x, [0, 0, 0], color=GYRO_C, width=0.65,
                             edgecolor="#3a1010", linewidth=1.0, alpha=0.9)
    ax_imu_r.set_ylim(-1.5, 1.5)
    ax_imu_r.set_ylabel("Angular vel [rad/s]", color="#ffab91")
    ax_imu_r.tick_params(axis="y", colors="#ffab91")
    ax_imu_r.spines["right"].set_visible(True)
    ax_imu_r.spines["right"].set_color("#3a414d")
    vel_labels = [ax_imu.text(x, 0, "", ha="center", va="bottom",
                              fontsize=9, fontweight="bold", color=VEL_C)
                  for x in vel_x]
    gyro_labels = [ax_imu_r.text(x, 0, "", ha="center", va="bottom",
                                 fontsize=9, fontweight="bold", color=c)
                   for x, c in zip(gyro_x, GYRO_C)]

    # Travel path — top of the right column (2/3 of right-side height).
    ax_path = fig.add_subplot(gs[0, 1])
    ax_path.set_aspect("equal")
    ax_path.set_title("Travel path  ·  world frame (GT)")
    ax_path.set_xlabel("x [m]"); ax_path.set_ylabel("y [m]")
    ax_path.grid(True)
    path_line = ax_path.plot([], [], "-", lw=1.8, color="#42a5f5",
                             alpha=0.85)[0]
    path_head = ax_path.plot([], [], marker="^", ms=12,
                             mfc="#ff5252", mec="#fff", mew=1.4, ls="None")[0]
    xy_label = ax_path.text(
        0, 0, "", fontsize=9, fontweight="bold", color="#ff7373",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#11151c",
                  edgecolor="#ff5252", alpha=0.92))
    tilt_text = ax_path.text(
        0.02, 0.96, "", transform=ax_path.transAxes,
        fontsize=9, fontweight="bold", color="#cad3df",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#11151c",
                  edgecolor="#3a414d", alpha=0.9))

    def update(_frame):
        with node.lock:
            pts = node.latest_points.copy() if node.latest_points.shape[0] else None
            tilt_arr = np.array(node.imu_tilt) if node.imu_tilt else np.zeros(0)
            path = np.array(node.path_xy) if node.path_xy else np.zeros((0, 2))
            vx, vy, vz = node.latest_vel
            wx, wy, wz = node.latest_gyro

        if pts is not None and pts.shape[0]:
            lidar_scatter.set_offsets(pts[:, :2])
            lidar_scatter.set_array(pts[:, 2])

        # Velocity bars + labels (m/s) — forward speed only.
        for bar, lbl, val in zip(vel_bars, vel_labels, (vx,)):
            bar.set_height(val)
            y_off = 0.05 if val >= 0 else -0.05
            lbl.set_position((bar.get_x() + bar.get_width() / 2, val + y_off))
            lbl.set_text(f"{val:+.2f} m/s")
            lbl.set_va("bottom" if val >= 0 else "top")

        # Gyro bars + labels (rad/s)
        for bar, lbl, val in zip(gyro_bars, gyro_labels, (wx, wy, wz)):
            bar.set_height(val)
            y_off = 0.07 if val >= 0 else -0.07
            lbl.set_position((bar.get_x() + bar.get_width() / 2, val + y_off))
            lbl.set_text(f"{val:+.2f} rad/s")
            lbl.set_va("bottom" if val >= 0 else "top")

        if path.shape[0] >= 1:
            path_line.set_data(path[:, 0], path[:, 1])
            path_head.set_data([path[-1, 0]], [path[-1, 1]])
            cx, cy = path[-1]
            ax_path.set_xlim(cx - 12, cx + 12)
            ax_path.set_ylim(cy - 12, cy + 12)
            xy_label.set_position((cx + 0.4, cy + 0.4))
            xy_label.set_text(f"({cx:+.2f}, {cy:+.2f})")
        if tilt_arr.size:
            tilt_text.set_text(
                f"tilt={math.degrees(tilt_arr[-1]):.1f}°"
                f"  |v|={math.sqrt(vx*vx+vy*vy+vz*vz):.2f} m/s"
                f"  |ω|={math.sqrt(wx*wx+wy*wy+wz*wz):.2f} rad/s")

        return (lidar_scatter, *vel_bars, *gyro_bars,
                *vel_labels, *gyro_labels,
                path_line, path_head, xy_label, tilt_text)

    anim = FuncAnimation(fig, update, interval=200, blit=False,
                         cache_frame_data=False)
    try:
        plt.show()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
