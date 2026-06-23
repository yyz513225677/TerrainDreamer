"""safety_supervisor_node — last line of defense on /cmd_vel.

Runs the supervisory loop at 100 Hz. Latches E-STOP on the rising edge
of /dashboard/estop=true. While latched, publishes a zero Twist on
/cmd_vel at 50 Hz so anything else publishing simultaneously is
overwritten. Only /dashboard/reset clears the latch.

Also publishes a heartbeat on /dashboard/safety_status (DashboardMode
with mode='estop' if latched, else 'idle').
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from geometry_msgs.msg import Twist

from jackal_dreamer_dashboard_msgs.msg import DashboardMode


SUPERVISOR_HZ = 100.0
ESTOP_PUBLISH_HZ = 50.0


class SafetySupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_supervisor_node")

        self._latched = False
        self._seq = 0
        self._zero_period_s = 1.0 / ESTOP_PUBLISH_HZ
        self._last_zero_pub_t = self.get_clock().now()
        self._heartbeat_counter = 0

        self.create_subscription(Bool, "/dashboard/estop",
                                 self._on_estop, 10)
        self.create_subscription(Empty, "/dashboard/reset",
                                 self._on_reset, 10)

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_status = self.create_publisher(
            DashboardMode, "/dashboard/safety_status", 10)

        self.timer = self.create_timer(1.0 / SUPERVISOR_HZ, self._tick)
        self.get_logger().info(
            "safety_supervisor_node started — 100 Hz; E-STOP overrides /cmd_vel.")

    def _on_estop(self, msg: Bool) -> None:
        if msg.data and not self._latched:
            self._latched = True
            self.get_logger().error("E-STOP LATCHED — zeroing /cmd_vel.")
            # Immediate zero publish.
            self.pub_cmd.publish(Twist())
            self._last_zero_pub_t = self.get_clock().now()

    def _on_reset(self, _msg: Empty) -> None:
        if self._latched:
            self.get_logger().warn(
                "Reset received — releasing E-STOP latch.")
        self._latched = False

    def _tick(self) -> None:
        now = self.get_clock().now()

        if self._latched:
            dt = (now - self._last_zero_pub_t).nanoseconds * 1e-9
            if dt >= self._zero_period_s:
                self.pub_cmd.publish(Twist())
                self._last_zero_pub_t = now

        # Heartbeat at ~10 Hz so dashboard sees status even with no events.
        self._heartbeat_counter += 1
        if self._heartbeat_counter >= int(SUPERVISOR_HZ // 10):
            self._heartbeat_counter = 0
            status = DashboardMode()
            status.mode = "estop" if self._latched else "idle"
            status.estop_active = self._latched
            status.seq = self._seq
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            self.pub_status.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetySupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
