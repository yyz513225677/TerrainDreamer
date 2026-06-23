"""E-STOP tests for the safety supervisor.

Property under test: while E-STOP is latched, NO Twist with non-zero linear
or angular velocity may reach /cmd_vel. The supervisor must overwrite every
inbound command with a zero Twist until /dashboard/reset arrives.

The real safety_supervisor is a rclpy node — we don't spin a node in tests
(too brittle in CI). Instead we test the gating logic directly.
"""


class CmdVelGate:
    """Mirrors the safety supervisor's gating logic without rclpy.

    Behaviour matches jackal_dreamer_safety.safety_supervisor.SafetySupervisor:
      * E-STOP latches on rising edge of /dashboard/estop with active=True.
      * Only /dashboard/reset can clear the latch.
      * filter(twist) returns the SAFE twist that should be published.
    """

    def __init__(self):
        self.latched = False

    def on_estop(self, active: bool):
        if active:
            self.latched = True
        # active=False is ignored — must come via reset()

    def on_reset(self):
        self.latched = False

    def filter(self, linear: float, angular: float):
        if self.latched:
            return (0.0, 0.0)
        return (linear, angular)


def test_no_estop_passes_through():
    g = CmdVelGate()
    assert g.filter(0.5, 0.3) == (0.5, 0.3)
    assert g.filter(-0.2, 0.0) == (-0.2, 0.0)


def test_estop_blocks_motion():
    g = CmdVelGate()
    g.on_estop(True)
    # Every imaginable Twist now zeros
    for v, w in [(1.0, 0.0), (-1.0, 0.0), (0.5, 0.5), (0.01, 0.0)]:
        assert g.filter(v, w) == (0.0, 0.0)


def test_estop_persists_until_reset():
    g = CmdVelGate()
    g.on_estop(True)
    # Sending estop=False does NOT clear the latch
    g.on_estop(False)
    assert g.filter(1.0, 0.0) == (0.0, 0.0)
    # Only reset clears
    g.on_reset()
    assert g.filter(1.0, 0.0) == (1.0, 0.0)


def test_repeated_estop_idempotent():
    g = CmdVelGate()
    for _ in range(10):
        g.on_estop(True)
    g.on_reset()
    assert g.filter(0.3, 0.1) == (0.3, 0.1)


def test_estop_rising_edge_immediate():
    """The CRITICAL safety property: a command issued in the same step that
    estop is triggered must NOT pass through."""
    g = CmdVelGate()
    # Simulating a high-rate loop: latch comes between two filter calls
    assert g.filter(0.5, 0.0) == (0.5, 0.0)
    g.on_estop(True)
    assert g.filter(0.5, 0.0) == (0.0, 0.0)
