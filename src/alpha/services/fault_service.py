"""
Deterministic fault injection for Weeks 3-4.

The injector corrupts execution at the ControlService boundary. It does not
reason about tasks or plans; it only applies configured chaos behavior and
records what happened for later metrics.
"""

from __future__ import annotations

import random

from alpha.core.entities import BlockColor, FaultType, Vec3
from alpha.core.interfaces import FaultInjector
from alpha.models.fault_config import FaultConfig


class ScriptedFaultInjector(FaultInjector):
    def __init__(self, config: FaultConfig | None = None) -> None:
        self.config = config or FaultConfig()
        self._rng = random.Random(self.config.seed)
        self._events: list[dict] = []

    @property
    def fault_type(self) -> FaultType:
        return self.config.fault_type

    @property
    def seed(self) -> int | None:
        return self.config.seed

    def event_log(self) -> list[dict]:
        return list(self._events)

    def maybe_corrupt_target(self, target_pos: Vec3) -> Vec3:
        if self.config.fault_type not in {FaultType.TARGET_CORRUPTION, FaultType.POSITION_OFFSET}:
            self._events.append(
                {
                    "event": "move_to",
                    "fault_type": self.config.fault_type.value,
                    "fault_applied": False,
                    "requested_target": target_pos,
                    "executed_target": target_pos,
                }
            )
            return target_pos

        if not self._should_fire():
            self._events.append(
                {
                    "event": "move_to",
                    "fault_type": self.config.fault_type.value,
                    "fault_applied": False,
                    "requested_target": target_pos,
                    "executed_target": target_pos,
                }
            )
            return target_pos

        offset = self._resolved_offset()
        corrupted = (
            target_pos[0] + offset[0],
            target_pos[1] + offset[1],
            target_pos[2] + offset[2],
        )
        self._events.append(
            {
                "event": "move_to",
                "fault_type": self.config.fault_type.value,
                "fault_applied": True,
                "requested_target": target_pos,
                "executed_target": corrupted,
                "offset": offset,
                "seed": self.config.seed,
            }
        )
        return corrupted

    def maybe_drop_grip(self, gripped: bool) -> bool:
        if self.config.fault_type != FaultType.GRIP_FAILURE or not gripped:
            return gripped
        return not self._should_fire()

    def should_fail_grip(self, color: BlockColor) -> bool:
        failed = self.config.fault_type == FaultType.GRIP_FAILURE and self._should_fire()
        self._events.append(
            {
                "event": "grip",
                "fault_type": self.config.fault_type.value,
                "fault_applied": failed,
                "block": color.value,
                "seed": self.config.seed,
            }
        )
        return failed

    def should_fail_release(self) -> bool:
        failed = self.config.fault_type == FaultType.RELEASE_FAILURE and self._should_fire()
        self._events.append(
            {
                "event": "release",
                "fault_type": self.config.fault_type.value,
                "fault_applied": failed,
                "seed": self.config.seed,
            }
        )
        return failed

    def _should_fire(self) -> bool:
        if not self.config.active:
            return False
        if self.config.probability >= 1.0:
            return True
        if self.config.probability <= 0.0:
            return False
        return self._rng.random() < self.config.probability

    def _resolved_offset(self) -> Vec3:
        if self.config.magnitude > 0:
            base = self.config.offset
            return (
                self.config.magnitude if base[0] >= 0 else -self.config.magnitude,
                self.config.magnitude if base[1] >= 0 else -self.config.magnitude,
                self.config.magnitude if base[2] > 0 else (0.0 if base[2] == 0 else -self.config.magnitude),
            )
        return self.config.offset
