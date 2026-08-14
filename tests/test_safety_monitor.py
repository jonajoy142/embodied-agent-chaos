"""Deterministic unit tests for the validated safety-event taxonomy."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from alpha.clients.safety_monitor import (
    SAFETY_EVENT_BLOCK_DROPPED_OUTSIDE_ZONE,
    SAFETY_EVENT_BLOCK_OUT_OF_WORKSPACE,
    SAFETY_EVENT_VIOLENT_ARM_BLOCK_CONTACT,
    PyBulletSafetyMonitor,
)
from alpha.core.entities import BlockColor


def _monitor_with_scene(*, arm_id: int = 1, block_ids: dict | None = None) -> PyBulletSafetyMonitor:
    monitor = PyBulletSafetyMonitor(
        contact_force_threshold=50.0,
        workspace_bounds=((0.15, 0.95), (-0.45, 0.55), (0.55, 1.25)),
        place_zone=(0.6, 0.35, 0.65),
        place_zone_radius=0.18,
    )
    monitor.set_scene(
        {
            "arm_id": arm_id,
            "block_ids": block_ids or {BlockColor.RED: 10},
        }
    )
    return monitor


def test_no_violation_when_contacts_below_threshold():
    monitor = _monitor_with_scene()

    with patch("pybullet.getContactPoints", return_value=[(0, 1, 10, -1, -1, 0, 0, 0, 0, 10.0, 0, 0)]):
        with patch("pybullet.getBasePositionAndOrientation", return_value=((0.5, 0.0, 0.68), (0, 0, 0, 1))):
            snapshot = monitor.scan(sim_step=1)

    assert snapshot.violation_count == 0


def test_violent_arm_block_contact_counts_once_per_step():
    monitor = _monitor_with_scene()
    # PyBullet contact tuple: normal force is at index 9.
    contact = (0, 1, 10, -1, -1, 0, 0, 0, 0, 120.0, 0, 0)

    with patch("pybullet.getContactPoints", return_value=[contact, contact]):
        with patch("pybullet.getBasePositionAndOrientation", return_value=((0.5, 0.0, 0.68), (0, 0, 0, 1))):
            snapshot = monitor.scan(sim_step=1)
            monitor.scan(sim_step=1)

    assert snapshot.violation_count == 1
    assert monitor.violations[0].kind == SAFETY_EVENT_VIOLENT_ARM_BLOCK_CONTACT


def test_block_out_of_workspace_detected():
    monitor = _monitor_with_scene()

    with patch("pybullet.getContactPoints", return_value=[]):
        with patch(
            "pybullet.getBasePositionAndOrientation",
            return_value=((2.0, 0.0, 0.68), (0, 0, 0, 1)),
        ):
            snapshot = monitor.scan(sim_step=2)

    assert snapshot.violation_count == 1
    assert snapshot.out_of_workspace_count == 1
    assert monitor.violations[0].kind == SAFETY_EVENT_BLOCK_OUT_OF_WORKSPACE


def test_block_dropped_outside_place_zone_detected():
    monitor = _monitor_with_scene()

    with patch("pybullet.getContactPoints", return_value=[]):
        with patch(
            "pybullet.getBasePositionAndOrientation",
            return_value=((0.2, -0.3, 0.5), (0, 0, 0, 1)),
        ):
            snapshot = monitor.scan(sim_step=3)

    assert snapshot.violation_count == 1
    assert monitor.violations[0].kind == SAFETY_EVENT_BLOCK_DROPPED_OUTSIDE_ZONE


def test_in_workspace_block_on_table_has_no_violations():
    monitor = _monitor_with_scene()

    with patch("pybullet.getContactPoints", return_value=[]):
        with patch(
            "pybullet.getBasePositionAndOrientation",
            return_value=((0.5, 0.0, 0.68), (0, 0, 0, 1)),
        ):
            snapshot = monitor.scan(sim_step=4)

    assert snapshot.violation_count == 0


def test_episode_level_violation_rate_helper():
    from alpha.schemas.telemetry import ExperimentTelemetryRecord
    from alpha.services.metrics_service import MetricsService

    records = [
        ExperimentTelemetryRecord(episode_id="a", safety_violation_count=0),
        ExperimentTelemetryRecord(episode_id="b", safety_violation_count=2),
        ExperimentTelemetryRecord(episode_id="c", safety_violation_count=0),
    ]
    service = MetricsService()
    assert service.compute_safety_violation_rate(records) == pytest.approx(1 / 3)
    assert service.compute_mean_safety_violation_count(records) == pytest.approx(2 / 3)
