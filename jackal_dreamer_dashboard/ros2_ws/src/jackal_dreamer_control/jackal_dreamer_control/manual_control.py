"""manual_control_node — bridges /dashboard/manual_cmd to /cmd_vel.

Only republishes to /cmd_vel when current mode is "manual" or "recording"
AND E-STOP is not latched. Includes a 0.5 s stale-input watchdog that
emits a zero Twist if no manual_cmd arrives in time.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from jackal_dreamer_dashboard_msgs.msg import DashboardMode


ACTIVE_MODES = {"manual", "recording"}
WATCHDOG_TIMEOUT_S = 0.5
PUBLISH_HZ = 20.0


class ManualControlNode(Node):
    def __init__(self) -> None:
        super().__init__("manual_control_node")

        self._last_cmd = Twist()
        self._last_cmd_time = self.get_clock().now()
        self._current_mode = "idle"
        self._estop_active = False

        self.sub_cmd = self.create_subscription(
            Twist, "/dashboard/manual_cmd", self._on_manual_cmd, 10)
        self.sub_mode = self.create_subscription(
            DashboardMode, "/dashboard/mode/current",
            self._on_mode_current, 10)

        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self._tick)
        self.get_logger().info("manual_control_node started.")

    def _on_manual_cmd(self, msg: Twist) -> None:
        self._last_cmd = msg
        self._last_cmd_time = self.get_clock().now()

    def _on_mode_current(self, msg: DashboardMode) -> None:
        self._current_mode = (msg.mode or "").strip().lower()
        self._estop_active = bool(msg.estop_active)

    def _tick(self) -> None:
        if self._estop_active or self._current_mode not in ACTIVE_MODES:
            return  # not our turn to publish

        now = self.get_clock().now()
        dt = (now - self._last_cmd_time).nanoseconds * 1e-9
        if dt > WATCHDOG_TIMEOUT_S:
            # Stale input — emit zero (deadman).
            self.pub_cmd_vel.publish(Twist())
        else:
            self.pub_cmd_vel.publish(self._last_cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManualControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
