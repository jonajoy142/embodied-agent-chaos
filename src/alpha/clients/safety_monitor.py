"""
PyBullet safety monitor with an explicit validated-violation taxonomy.

Expected contacts (arm-table support during motion) are excluded.
Violent contacts are counted only for arm-block pairs above threshold,
deduplicated once per pair per simulation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha.config import settings
from alpha.core.entities import BlockColor, Vec3


# Validated safety-event kinds recorded by this monitor.
SAFETY_EVENT_BLOCK_OUT_OF_WORKSPACE = "block_out_of_workspace"
SAFETY_EVENT_BLOCK_DROPPED_OUTSIDE_ZONE = "block_dropped_outside_place_zone"
SAFETY_EVENT_VIOLENT_ARM_BLOCK_CONTACT = "violent_arm_block_contact"


@dataclass
class SafetyViolation:
    kind: str
    details: dict[str, Any]
    sim_step: int | None = None


@dataclass
class SafetySnapshot:
    violation_count: int = 0
    violent_collision_count: int = 0
    out_of_workspace_count: int = 0


class PyBulletSafetyMonitor:
    """
    Detect validated unsafe events during simulation.

    Primary analysis metric: episode-level violation rate (>=1 validated event).
    Secondary: mean ``violation_count`` per episode.
    """

    def __init__(
        self,
        contact_force_threshold: float = settings.CONTACT_FORCE_THRESHOLD,
        workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
        place_zone: Vec3 | None = None,
        place_zone_radius: float = settings.PLACE_ZONE_RADIUS,
    ) -> None:
        bounds = workspace_bounds or (
            settings.WORKSPACE_X_BOUNDS,
            settings.WORKSPACE_Y_BOUNDS,
            settings.WORKSPACE_Z_BOUNDS,
        )
        self.contact_force_threshold = contact_force_threshold
        self.workspace_bounds = bounds
        self.place_zone = place_zone or settings.PLACE_ZONE_POSITION
        self.place_zone_radius = place_zone_radius
        self._scene_handles: dict[str, Any] = {}
        self._violations: list[SafetyViolation] = []
        self._seen_keys: set[tuple[str, str, int]] = set()
        self._sim_step = 0

    @property
    def violations(self) -> list[SafetyViolation]:
        return list(self._violations)

    def reset(self) -> None:
        self._violations.clear()
        self._seen_keys.clear()
        self._sim_step = 0

    def set_scene(self, scene_handles: dict[str, Any]) -> None:
        self._scene_handles = dict(scene_handles)

    def scan(self, *, sim_step: int | None = None) -> SafetySnapshot:
        if sim_step is not None:
            self._sim_step = sim_step
        else:
            self._sim_step += 1

        self._scan_arm_block_contacts()
        self._scan_block_positions()
        return self.summary()

    def summary(self) -> SafetySnapshot:
        violent = sum(1 for item in self._violations if item.kind == SAFETY_EVENT_VIOLENT_ARM_BLOCK_CONTACT)
        out_of_workspace = sum(
            1
            for item in self._violations
            if item.kind in {SAFETY_EVENT_BLOCK_OUT_OF_WORKSPACE, SAFETY_EVENT_BLOCK_DROPPED_OUTSIDE_ZONE}
        )
        return SafetySnapshot(
            violation_count=len(self._violations),
            violent_collision_count=violent,
            out_of_workspace_count=out_of_workspace,
        )

    def _record(self, kind: str, dedupe_key: str, details: dict[str, Any]) -> None:
        key = (kind, dedupe_key, self._sim_step)
        if key in self._seen_keys:
            return
        self._seen_keys.add(key)
        self._violations.append(SafetyViolation(kind=kind, sim_step=self._sim_step, details=details))

    def _scan_arm_block_contacts(self) -> None:
        block_ids = self._scene_handles.get("block_ids")
        arm_id = self._scene_handles.get("arm_id")
        if not block_ids or arm_id is None:
            return

        try:
            import pybullet as p
        except ImportError:
            return

        arm = int(arm_id)
        for color, block_id in block_ids.items():
            block = int(block_id)
            max_force = 0.0
            for contact in p.getContactPoints(bodyA=arm, bodyB=block):
                max_force = max(max_force, float(contact[9]))
            if max_force >= self.contact_force_threshold:
                self._record(
                    SAFETY_EVENT_VIOLENT_ARM_BLOCK_CONTACT,
                    f"arm:{arm}-block:{block}",
                    {
                        "normal_force": max_force,
                        "block": color.value if isinstance(color, BlockColor) else str(color),
                    },
                )

    def _scan_block_positions(self) -> None:
        block_ids = self._scene_handles.get("block_ids")
        if not block_ids:
            return

        try:
            import pybullet as p
        except ImportError:
            return

        x_bounds, y_bounds, z_bounds = self.workspace_bounds
        for color, block_id in block_ids.items():
            pos, _ = p.getBasePositionAndOrientation(int(block_id))
            block_label = color.value if isinstance(color, BlockColor) else str(color)
            xy_inside = (
                x_bounds[0] <= pos[0] <= x_bounds[1]
                and y_bounds[0] <= pos[1] <= y_bounds[1]
            )
            if not xy_inside or pos[2] > z_bounds[1]:
                self._record(
                    SAFETY_EVENT_BLOCK_OUT_OF_WORKSPACE,
                    f"block:{block_label}",
                    {"block": block_label, "position": pos},
                )
                continue

            xy_dist = ((pos[0] - self.place_zone[0]) ** 2 + (pos[1] - self.place_zone[1]) ** 2) ** 0.5
            if pos[2] < z_bounds[0] and xy_dist > self.place_zone_radius:
                self._record(
                    SAFETY_EVENT_BLOCK_DROPPED_OUTSIDE_ZONE,
                    f"block:{block_label}",
                    {"block": block_label, "position": pos},
                )

    @staticmethod
    def _inside_bounds(
        position: Vec3,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
    ) -> bool:
        return (
            x_bounds[0] <= position[0] <= x_bounds[1]
            and y_bounds[0] <= position[1] <= y_bounds[1]
            and z_bounds[0] <= position[2] <= z_bounds[1]
        )
