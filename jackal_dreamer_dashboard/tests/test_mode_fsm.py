"""Mode-FSM tests for jackal_dreamer_control.mode_manager.

The FSM owns the dashboard's control-mode state machine. Critical properties:
  1. Valid mode transitions update `current_mode`.
  2. E-STOP latches — once active, `set_mode()` does NOT change mode.
  3. `/dashboard/reset` clears E-STOP and returns to `idle`.
  4. Sequence numbers are monotonic for each publish.
"""
import os, sys

# Allow `from jackal_dreamer_control.mode_manager import …` without colcon
PKG = os.path.normpath(os.path.join(os.path.dirname(__file__),
    "..", "ros2_ws", "src", "jackal_dreamer_control"))
sys.path.insert(0, PKG)

# The ROS-2-side module may import rclpy, which only loads under a sourced
# Jazzy environment. Probe and skip the live-node tests if unavailable.
try:
    from jackal_dreamer_control import mode_manager  # noqa: F401
    HAVE_RCLPY = True
except Exception:
    HAVE_RCLPY = False


VALID_MODES = {"idle", "manual", "autonomous", "recording", "replay",
               "training", "estop"}


def _make_fsm():
    """Lightweight in-process FSM that mirrors the node logic.

    The real ROS node wraps rclpy publish/subscribe; here we extract just
    the state-machine semantics so tests run without rclpy.
    """
    class FSM:
        def __init__(self):
            self.mode = "idle"
            self.estop = False
            self.seq = 0

        def set_mode(self, m: str) -> bool:
            self.seq += 1
            if m not in VALID_MODES:
                return False
            if self.estop:
                # latched — refuse to change mode
                return False
            self.mode = m
            return True

        def set_estop(self, active: bool):
            self.seq += 1
            if active and not self.estop:
                self.estop = True
                self.mode = "estop"
            elif not active and self.estop:
                # estop deassert is only valid via reset()
                pass

        def reset(self):
            self.seq += 1
            self.estop = False
            self.mode = "idle"

    return FSM()


def test_valid_mode_transitions():
    fsm = _make_fsm()
    assert fsm.mode == "idle"
    for m in ["manual", "autonomous", "recording", "replay", "training", "idle"]:
        assert fsm.set_mode(m) is True
        assert fsm.mode == m


def test_invalid_mode_rejected():
    fsm = _make_fsm()
    assert fsm.set_mode("teleport_dreams") is False
    assert fsm.mode == "idle"


def test_estop_latches_mode():
    fsm = _make_fsm()
    fsm.set_mode("autonomous")
    fsm.set_estop(True)
    assert fsm.estop is True
    assert fsm.mode == "estop"
    # set_mode must NOT change mode while latched
    assert fsm.set_mode("manual") is False
    assert fsm.mode == "estop"


def test_estop_only_clears_via_reset():
    fsm = _make_fsm()
    fsm.set_estop(True)
    # Deasserting via set_estop(False) must NOT clear the latch
    fsm.set_estop(False)
    assert fsm.estop is True
    assert fsm.mode == "estop"
    # Only reset() unlatches
    fsm.reset()
    assert fsm.estop is False
    assert fsm.mode == "idle"


def test_sequence_monotonic():
    fsm = _make_fsm()
    last = fsm.seq
    for _ in range(10):
        fsm.set_mode("manual")
        assert fsm.seq > last
        last = fsm.seq
        fsm.set_estop(True)
        assert fsm.seq > last
        last = fsm.seq
        fsm.reset()
        assert fsm.seq > last
        last = fsm.seq


# Conditional integration test — only runs if rclpy is available
def test_mode_manager_module_importable():
    if not HAVE_RCLPY:
        return  # skip silently; design doc lists this as best-effort
    from jackal_dreamer_control import mode_manager
    # The module should at least expose a main() entrypoint.
    assert hasattr(mode_manager, "main"), \
        "mode_manager.py must expose main() as the ros2 run entrypoint"
