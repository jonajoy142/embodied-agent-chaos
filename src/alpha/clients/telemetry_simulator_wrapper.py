"""
Optional simulator wrapper that feeds sim-step ticks and safety scans into
``ChaosTelemetryLogger`` without modifying agent or control logic.
"""

from __future__ import annotations

from typing import Any

from alpha.clients.safety_monitor import PyBulletSafetyMonitor
from alpha.config import settings
from alpha.core.entities import BlockColor, BlockState, Vec3
from alpha.core.interfaces import SimulatorClient
from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger


class TelemetrySimulatorWrapper(SimulatorClient):
    """Delegate to an inner simulator while recording telemetry hooks."""

    def __init__(
        self,
        inner: SimulatorClient,
        telemetry: ChaosTelemetryLogger,
        safety_monitor: PyBulletSafetyMonitor | None = None,
    ) -> None:
        self._inner = inner
        self._telemetry = telemetry
        self._safety = safety_monitor
        self._scene_handles: dict[str, Any] = {}

    @property
    def inner(self) -> SimulatorClient:
        return self._inner

    def connect(self, gui: bool = False) -> None:
        self._inner.connect(gui=gui)

    def disconnect(self) -> None:
        self._inner.disconnect()

    def build_scene(self) -> dict[str, Any]:
        handles = self._inner.build_scene()
        self._scene_handles = handles
        if self._safety is not None:
            self._safety.set_scene(handles)
        return handles

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self._inner.step(1)
            self._telemetry.tick_sim_step(1)
            if self._safety is not None:
                snapshot = self._safety.scan(sim_step=self._telemetry.sim_step_count)
                self._telemetry.update_safety(snapshot)

    def get_block_state(self, color: BlockColor) -> BlockState:
        return self._inner.get_block_state(color)

    def move_end_effector(self, target_pos: Vec3, target_orn: Any = None) -> dict[str, Any]:
        result = self._inner.move_end_effector(target_pos, target_orn)
        self._telemetry.tick_sim_step(settings.MOVE_STEPS)
        if self._safety is not None:
            snapshot = self._safety.scan(sim_step=self._telemetry.sim_step_count)
            self._telemetry.update_safety(snapshot)
        return result

    def grip(self, color: BlockColor) -> Any:
        return self._inner.grip(color)

    def release(self, grip_handle: Any) -> None:
        self._inner.release(grip_handle)

    def observe(self) -> dict[str, Any]:
        return self._inner.observe()
