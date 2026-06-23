"""HumanTakeover — algorithm-side interface for swapping between autonomous
and manual control during data collection.

Does NOT implement keyboard/joystick reading; the ROS 2 wrapper is
expected to call `set_mode()` and pass `human_action` into the model's
`select_action()` when a human is driving. Records the last time a manual
action arrived so callers can auto-fall back after `manual_control_timeout`.
"""
from __future__ import annotations

import time
from typing import Optional

import torch


MODE_AUTONOMOUS = "autonomous"
MODE_HUMAN = "human"


class HumanTakeover:
    def __init__(
        self,
        default_mode: str = MODE_AUTONOMOUS,
        manual_control_timeout: float = 1.0,
    ):
        self._validate_mode(default_mode)
        self._mode = default_mode
        self.manual_control_timeout = float(manual_control_timeout)
        self._last_human_at: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in (MODE_AUTONOMOUS, MODE_HUMAN):
            raise ValueError(
                f"HumanTakeover: mode must be '{MODE_AUTONOMOUS}' or "
                f"'{MODE_HUMAN}', got '{mode}'")

    # ──────────────────────────────────────────────────────────────────────
    def set_mode(self, mode: str) -> None:
        self._validate_mode(mode)
        self._mode = mode

    def get_mode(self) -> str:
        return self._mode

    def is_human_control(self) -> bool:
        return self._mode == MODE_HUMAN

    # ──────────────────────────────────────────────────────────────────────
    def select_action(
        self,
        autonomous_action: torch.Tensor,
        human_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Choose between autonomous and human action based on current mode.

        Auto-falls-back to autonomous if mode is HUMAN but no human_action
        was provided and the manual_control_timeout has elapsed since the
        last manual command.
        """
        if self._mode == MODE_HUMAN and human_action is not None:
            self._last_human_at = time.monotonic()
            return human_action

        if self._mode == MODE_HUMAN and self._last_human_at is not None:
            stale = (time.monotonic() - self._last_human_at) > self.manual_control_timeout
            if not stale:
                # Mode is HUMAN but caller didn't supply an action right now —
                # hold the rover (return zeros) to make the takeover obvious.
                return torch.zeros_like(autonomous_action)
            # Stale: silently fall back to autonomous to avoid getting stuck.
        return autonomous_action
