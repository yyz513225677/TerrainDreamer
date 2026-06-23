"""mode_manager_node — owns the dashboard FSM and latches E-STOP.

Subscribes:
  /dashboard/mode  (DashboardMode)         — requested mode
  /dashboard/estop (std_msgs/Bool)         — engage / release (latching)
  /dashboard/reset (std_msgs/Empty)        — explicit unlatch
Publishes:
  /dashboard/mode/current (DashboardMode)  — authoritative current mode
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from jackal_dreamer_dashboard_msgs.msg import DashboardMode


VALID_MODES = {"idle", "manual", "autonomous", "recording", "replay",
               "training", "estop"}


class ModeManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("mode_manager_node")

        self._mode = "idle"
        self._estop_latched = False
        self._seq = 0

        self.sub_mode = self.create_subscription(
            DashboardMode, "/dashboard/mode", self._on_mode_request, 10)
        self.sub_estop = self.create_subscription(
            Bool, "/dashboard/estop", self._on_estop, 10)
        self.sub_reset = self.create_subscription(
            Empty, "/dashboard/reset", self._on_reset, 10)

        self.pub_current = self.create_publisher(
            DashboardMode, "/dashboard/mode/current", 10)

        # Heartbeat: republish current mode at 5 Hz so late subscribers
        # don't have to wait for the next transition.
        self.timer = self.create_timer(0.2, self._publish_current)

        self.get_logger().info("mode_manager_node started — mode=idle")

    # ---- callbacks ----
    def _on_mode_request(self, msg: DashboardMode) -> None:
        if self._estop_latched:
            self.get_logger().warn(
                f"Ignoring mode request '{msg.mode}' — E-STOP latched.")
            return
        requested = (msg.mode or "").strip().lower()
        if requested not in VALID_MODES:
            self.get_logger().warn(f"Invalid mode '{requested}', ignoring.")
            return
        if requested == "estop":
            # Mode-channel route into E-STOP also latches.
            self._estop_latched = True
            self._mode = "estop"
        else:
            self._mode = requested
        self._publish_current()

    def _on_estop(self, msg: Bool) -> None:
        if msg.data and not self._estop_latched:
            self._estop_latched = True
            self._mode = "estop"
            self.get_logger().error("E-STOP LATCHED.")
            self._publish_current()
        # We DO NOT unlatch on Bool=False. Reset topic is the only exit.

    def _on_reset(self, _msg: Empty) -> None:
        if self._estop_latched:
            self.get_logger().warn("Reset received — releasing E-STOP latch.")
        self._estop_latched = False
        self._mode = "idle"
        self._publish_current()

    # ---- publish ----
    def _publish_current(self) -> None:
        out = DashboardMode()
        out.mode = self._mode
        out.estop_active = self._estop_latched
        out.seq = self._seq
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        self.pub_current.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ModeManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
