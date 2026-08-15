"""
ChaosRoboticsWrapper — decoupled middleware for embodied-AI fault injection.

This module wraps any ``SimulatorClient`` (typically ``PyBulletSimulatorClient``)
and exposes explicit injection hooks at the boundaries between physics, sensing,
planning, and control. The Brain/Body agent planner is never modified; callers
wire these hooks in scripts or orchestration code.

Fault classes (Weeks 3–4):
    1. Sensor lag        — ``wrap_observation``
    2. Grip slip         — ``step`` while a grip constraint is active
    3. Unreachable IK    — ``move_end_effector`` target offset
    4. Planner corruption — ``intercept_planner_output``
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from alpha.core.entities import BlockColor, BlockState, FaultType, Vec3
from alpha.core.interfaces import SimulatorClient
from alpha.models.chaos_config import ChaosConfig, FaultSpec, InjectionMode, TriggerKind


@dataclass
class ChaosRuntimeContext:
    """Mutable episode context used by trigger predicates."""

    elapsed_seconds: float = 0.0
    sim_step_count: int = 0
    phase: str = "idle"
    event_step: int = 0


@dataclass
class _ObservationSample:
    timestamp: float
    state: dict[str, Any]


class ChaosRoboticsWrapper(SimulatorClient):
    """
    Middleware wrapper that delegates to an inner simulator while injecting
    configured faults at well-defined hook points.

    Parameters
    ----------
    inner:
        The real physics backend (e.g. PyBulletSimulatorClient).
    config:
        Chaos experiment configuration describing faults, triggers, and intensity.
    """

    def __init__(self, inner: SimulatorClient, config: ChaosConfig | None = None, safety_monitor: Any = None) -> None:
        self._inner = inner
        self.config = config or ChaosConfig()
        self._rng = random.Random(self.config.seed)
        self._context = ChaosRuntimeContext()
        self._observation_history: deque[_ObservationSample] = deque()
        self._grip_handle: Any | None = None
        self._grip_active: bool = False
        self._events: list[dict[str, Any]] = []
        self._armed_faults: set[FaultType] = set()
        self._latched_single_fault: FaultType | None = None
        self._episode_started_at = time.monotonic()
        self._safety_monitor = safety_monitor

    # ------------------------------------------------------------------
    # Runtime context API (call from orchestration layer, not from agent)
    # ------------------------------------------------------------------

    @property
    def context(self) -> ChaosRuntimeContext:
        return self._context

    def set_phase(self, phase: str) -> None:
        """Update the high-level task phase used by event triggers (e.g. ``pick``)."""
        self._context.phase = phase

    def advance_event_step(self, step: int | None = None) -> None:
        """Increment or set the logical event step counter."""
        if step is None:
            self._context.event_step += 1
        else:
            self._context.event_step = step

    def event_log(self) -> list[dict[str, Any]]:
        """Structured record of injected faults for metrics / episode logs."""
        return list(self._events)

    @property
    def grip_is_active(self) -> bool:
        return self._grip_active

    @property
    def grip_handle(self) -> Any | None:
        return self._grip_handle

    # ------------------------------------------------------------------
    # SimulatorClient delegation with physics/control hooks
    # ------------------------------------------------------------------

    def connect(self, gui: bool = False) -> None:
        self._inner.connect(gui=gui)

    def disconnect(self) -> None:
        self._inner.disconnect()

    def build_scene(self) -> dict[str, Any]:
        return self._inner.build_scene()

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self._context.sim_step_count += 1
            self._context.elapsed_seconds += self.config.sim_timestep
            self._maybe_inject_grip_slip()
            self._inner.step(1)

    def get_block_state(self, color: BlockColor) -> BlockState:
        return self._inner.get_block_state(color)

    def move_end_effector(self, target_pos: Vec3, target_orn: Any = None) -> dict[str, Any]:
        requested = target_pos
        executed = self._maybe_inject_unreachable_ik(target_pos)
        
        # Inject grip_slip during move operation if we're in lift phase
        if self._grip_active and self._context.phase == "lift":
            self._maybe_inject_grip_slip()
        
        result = self._inner.move_end_effector(executed, target_orn)
        result["requested_target"] = requested
        result["executed_target"] = executed
        return result

    def grip(self, color: BlockColor) -> Any:
        handle = self._inner.grip(color)
        if handle is not None:
            self._grip_handle = handle
            self._grip_active = True
            if self._safety_monitor is not None:
                self._safety_monitor.set_gripped(color, gripped=True)
        return handle

    def release(self, grip_handle: Any) -> None:
        if grip_handle is None:
            return
        self._inner.release(grip_handle)
        if grip_handle == self._grip_handle:
            self._grip_handle = None
            self._grip_active = False
            if self._safety_monitor is not None:
                # Find which block was gripped and mark as released
                for color in [BlockColor.RED, BlockColor.GREEN, BlockColor.BLUE]:
                    if self._safety_monitor._gripped_blocks.intersection({color.value}):
                        self._safety_monitor.set_gripped(color, gripped=False)
                        break

    # ------------------------------------------------------------------
    # Agent-boundary hooks (Brain/Body planner remains untouched)
    # ------------------------------------------------------------------

    def wrap_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Intercept environment state before it reaches the planner.

        Applies sensor-lag faults by returning a deep-copied snapshot from
        ``lag_seconds`` ago instead of the freshest observation.
        """
        timestamp = self._context.elapsed_seconds
        snapshot = copy.deepcopy(state)
        self._observation_history.append(_ObservationSample(timestamp=timestamp, state=snapshot))

        specs = self._active_specs(FaultType.SENSOR_LAG)
        if not specs:
            return snapshot

        spec = specs[0]
        if not self._should_apply_spec(spec):
            return snapshot

        lag_seconds = spec.intensity.lag_seconds
        cutoff = timestamp - lag_seconds
        delayed = snapshot
        for sample in reversed(self._observation_history):
            if sample.timestamp <= cutoff:
                delayed = copy.deepcopy(sample.state)
                break

        self._record_event(
            fault_type=FaultType.SENSOR_LAG,
            hook="wrap_observation",
            fault_applied=True,
            lag_seconds=lag_seconds,
            returned_timestamp=cutoff,
        )
        return delayed

    def intercept_planner_output(self, raw_text: str) -> str:
        """
        Intercept raw LLM planner output before JSON/schema validation.

        Applies planner-output corruption faults by deleting critical keys or
        injecting malformed characters.
        """
        specs = self._active_specs(FaultType.PLANNER_OUTPUT_CORRUPTION)
        if not specs:
            return raw_text

        spec = specs[0]
        if not self._should_apply_spec(spec):
            return raw_text

        corrupted = self._corrupt_planner_text(raw_text, spec)
        self._record_event(
            fault_type=FaultType.PLANNER_OUTPUT_CORRUPTION,
            hook="intercept_planner_output",
            fault_applied=True,
            keys_deleted=list(spec.intensity.keys_to_delete),
        )
        return corrupted

    # ------------------------------------------------------------------
    # Fault implementations
    # ------------------------------------------------------------------

    def _maybe_inject_grip_slip(self) -> None:
        if not self._grip_active or self._grip_handle is None:
            return

        specs = self._active_specs(FaultType.GRIP_SLIP)
        if not specs:
            return

        spec = specs[0]
        if not self._trigger_satisfied(spec):
            return
        if not self._roll_probability(spec.probability):
            self._record_event(
                fault_type=FaultType.GRIP_SLIP,
                hook="step",
                fault_applied=False,
                slip_probability=spec.intensity.slip_probability,
            )
            return
        if not self._roll_probability(spec.intensity.slip_probability):
            self._record_event(
                fault_type=FaultType.GRIP_SLIP,
                hook="step",
                fault_applied=False,
                slip_probability=spec.intensity.slip_probability,
            )
            return

        handle = self._grip_handle
        weaken = spec.intensity.weaken_force_factor
        if weaken <= 0.0:
            self._inner.release(handle)
            self._grip_handle = None
            self._grip_active = False
            dropped = True
        else:
            dropped = self._weaken_grip_constraint(handle, weaken)
            if dropped:
                self._grip_handle = None
                self._grip_active = False

        self._record_event(
            fault_type=FaultType.GRIP_SLIP,
            hook="step",
            fault_applied=True,
            constraint_id=handle,
            weaken_force_factor=weaken,
            constraint_removed=dropped,
        )

    def _maybe_inject_unreachable_ik(self, target_pos: Vec3) -> Vec3:
        specs = self._active_specs(FaultType.UNREACHABLE_IK)
        if not specs:
            return target_pos

        spec = specs[0]
        if not self._should_apply_spec(spec):
            return target_pos

        offset = self._resolved_unreachable_offset(spec)
        corrupted = (
            target_pos[0] + offset[0],
            target_pos[1] + offset[1],
            target_pos[2] + offset[2],
        )
        self._record_event(
            fault_type=FaultType.UNREACHABLE_IK,
            hook="move_end_effector",
            fault_applied=True,
            requested_target=target_pos,
            executed_target=corrupted,
            offset=offset,
        )
        return corrupted

    def _corrupt_planner_text(self, raw_text: str, spec: FaultSpec) -> str:
        intensity = spec.intensity
        if intensity.inject_malformed_chars and self._rng.random() < 0.5:
            return self._inject_malformed_json(raw_text)

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            for key in intensity.keys_to_delete:
                payload.pop(key, None)
                if key == "target" and "actions" in payload:
                    for action in payload.get("actions", []):
                        if isinstance(action, dict):
                            action.pop("target", None)
            return json.dumps(payload)

        return self._inject_malformed_json(raw_text)

    @staticmethod
    def _inject_malformed_json(raw_text: str) -> str:
        stripped = raw_text.strip()
        if stripped.endswith("}"):
            return stripped[:-1] + ",\n"
        return stripped + "{{"

    def _weaken_grip_constraint(self, handle: Any, weaken_force_factor: float) -> bool:
        """
        Attempt to weaken an active grip constraint.

        When ``weaken_force_factor`` is near zero the constraint is removed
        outright (full slip). Otherwise this falls back to constraint removal
        because the portable SimulatorClient API does not expose per-step force
        tuning; subclasses may override for backend-specific weakening.
        """
        if weaken_force_factor < 0.05:
            self._inner.release(handle)
            return True
        return False

    def _resolved_unreachable_offset(self, spec: FaultSpec) -> Vec3:
        direction = spec.intensity.offset
        magnitude = spec.intensity.offset_magnitude
        norm = math.sqrt(sum(component * component for component in direction))
        if norm == 0:
            direction = (1.0, 0.0, 0.0)
            norm = 1.0
        scale = magnitude / norm
        return (direction[0] * scale, direction[1] * scale, direction[2] * scale)

    # ------------------------------------------------------------------
    # Trigger / selection helpers
    # ------------------------------------------------------------------

    def _active_specs(self, fault_type: FaultType) -> list[FaultSpec]:
        enabled = [spec for spec in self.config.enabled_faults() if spec.fault_type == fault_type]
        if self.config.mode == InjectionMode.CONCURRENT:
            return [spec for spec in enabled if self._trigger_satisfied(spec)]

        if not enabled:
            return []

        if self._latched_single_fault is not None and self._latched_single_fault != fault_type:
            return []

        ready = [spec for spec in enabled if self._trigger_satisfied(spec)]
        if not ready:
            return []

        if self._latched_single_fault is None:
            self._latched_single_fault = fault_type
        return ready[:1]

    def _should_apply_spec(self, spec: FaultSpec) -> bool:
        if not self._trigger_satisfied(spec):
            return False
        return self._roll_probability(spec.probability)

    def _trigger_satisfied(self, spec: FaultSpec) -> bool:
        trigger = spec.trigger
        if trigger.kind == TriggerKind.IMMEDIATE:
            return True
        if trigger.kind == TriggerKind.TIME:
            after = trigger.after_seconds if trigger.after_seconds is not None else 0.0
            return self._context.elapsed_seconds >= after
        if trigger.kind == TriggerKind.EVENT:
            phase_ok = trigger.phase is None or self._context.phase == trigger.phase
            step_ok = trigger.step is None or self._context.event_step == trigger.step
            return phase_ok and step_ok
        return False

    def _roll_probability(self, probability: float) -> bool:
        if probability >= 1.0:
            return True
        if probability <= 0.0:
            return False
        return self._rng.random() < probability

    def _record_event(self, *, fault_type: FaultType, hook: str, fault_applied: bool, **details: Any) -> None:
        self._events.append(
            {
                "fault_type": fault_type.value,
                "hook": hook,
                "fault_applied": fault_applied,
                "phase": self._context.phase,
                "event_step": self._context.event_step,
                "sim_step": self._context.sim_step_count,
                "elapsed_seconds": round(self._context.elapsed_seconds, 6),
                "seed": self.config.seed,
                **details,
            }
        )


class PyBulletChaosRoboticsWrapper(ChaosRoboticsWrapper):
    """
    PyBullet-aware chaos wrapper.

    Uses ``pybullet.changeConstraint`` for partial grip weakening when
    ``weaken_force_factor`` is between 0 and 1.
    """

    def _weaken_grip_constraint(self, handle: Any, weaken_force_factor: float) -> bool:
        if weaken_force_factor <= 0.0:
            self._inner.release(handle)
            return True

        try:
            import pybullet as p
        except ImportError:
            return super()._weaken_grip_constraint(handle, weaken_force_factor)

        max_force = max(1.0, 500.0 * weaken_force_factor)
        try:
            p.changeConstraint(handle, maxForce=max_force)
        except Exception:
            self._inner.release(handle)
            return True
        return False
