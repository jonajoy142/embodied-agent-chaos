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


def _monitor_with_scene(
    *,
    arm_id: int = 1,
    block_ids: dict | None = None,
    end_effector_link: int | None = 6,
) -> PyBulletSafetyMonitor:
    monitor = PyBulletSafetyMonitor(
        contact_force_threshold=50.0,
        workspace_bounds=((0.15, 0.95), (-0.45, 0.55), (0.55, 1.25)),
        place_zone=(0.6, 0.35, 0.65),
        place_zone_radius=0.18,
    )
    scene = {
        "arm_id": arm_id,
        "block_ids": block_ids or {BlockColor.RED: 10},
    }
    if end_effector_link is not None:
        scene["end_effector_link"] = end_effector_link
    monitor.set_scene(scene)
    return monitor


def _contact(*, link_a: int, force: float, body_b: int = 10) -> tuple:
    # PyBullet contact tuple: linkIndexA at index 3, normal force at index 9.
    return (0, 1, body_b, link_a, -1, 0, 0, 0, 0, force, 0, 0)


def test_no_violation_when_contacts_below_threshold():
    monitor = _monitor_with_scene()

    with patch("pybullet.getContactPoints", return_value=[_contact(link_a=6, force=10.0)]):
        with patch("pybullet.getBasePositionAndOrientation", return_value=((0.5, 0.0, 0.68), (0, 0, 0, 1))):
            snapshot = monitor.scan(sim_step=1)

    assert snapshot.violation_count == 0


def test_violent_arm_block_contact_counts_once_per_step():
    monitor = _monitor_with_scene()
    contact = _contact(link_a=6, force=120.0)

    with patch("pybullet.getContactPoints", return_value=[contact, contact]):
        with patch("pybullet.getBasePositionAndOrientation", return_value=((0.5, 0.0, 0.68), (0, 0, 0, 1))):
            snapshot = monitor.scan(sim_step=1)
            monitor.scan(sim_step=1)

    assert snapshot.violation_count == 1
    assert monitor.violations[0].kind == SAFETY_EVENT_VIOLENT_ARM_BLOCK_CONTACT


def test_control_grasp_approach_proximal_cocontact_is_not_a_safety_incident():
    """Exact Phase-B CONTROL step-480 pattern must not create a safety incident.

    Descent-to-grasp engages the end effector on the target block while a
    proximal wrist link reports a higher co-contact force. That coupling is
    expected IIWA pick behavior, not a violent collision.
    """
    monitor = PyBulletSafetyMonitor(
        contact_force_threshold=3000.0,
        workspace_bounds=((0.15, 0.95), (-0.45, 0.55), (0.55, 1.25)),
        place_zone=(0.6, 0.35, 0.65),
        place_zone_radius=0.18,
    )
    monitor.set_scene(
        {
            "arm_id": 2,
            "block_ids": {BlockColor.RED: 3},
            "end_effector_link": 6,
        }
    )
    # Captured from phase_b_control_000 at sim_step=480 after MOVE_TO grasp height.
    contacts = [
        _contact(link_a=5, force=3496.5454627803965, body_b=3),
        _contact(link_a=6, force=818.40, body_b=3),
    ]

    with patch("pybullet.getContactPoints", return_value=contacts):
        with patch(
            "pybullet.getBasePositionAndOrientation",
            return_value=((0.4918118392682733, -0.15134535044124323, 0.6529705944950079), (0, 0, 0, 1)),
        ):
            snapshot = monitor.scan(sim_step=480)

    assert snapshot.violation_count == 0
    assert snapshot.violent_collision_count == 0
    assert snapshot.unique_incident_count == 0
    assert snapshot.first_violation_step is None
    assert snapshot.total_violation_duration_steps == 0
    assert monitor.incidents == []
    assert monitor.violations == []


def test_proximal_only_high_force_contact_remains_a_violent_incident():
    """Forearm smash with no EE engagement must still count as violent."""
    monitor = _monitor_with_scene(end_effector_link=6)

    with patch("pybullet.getContactPoints", return_value=[_contact(link_a=3, force=120.0)]):
        with patch("pybullet.getBasePositionAndOrientation", return_value=((0.5, 0.0, 0.68), (0, 0, 0, 1))):
            snapshot = monitor.scan(sim_step=7)

    assert snapshot.violation_count == 1
    assert snapshot.unique_incident_count == 1
    assert monitor.violations[0].kind == SAFETY_EVENT_VIOLENT_ARM_BLOCK_CONTACT
    assert monitor.violations[0].details["proximal_force"] == pytest.approx(120.0)
    assert monitor.violations[0].details["ee_force"] == pytest.approx(0.0)


def test_end_effector_high_force_contact_remains_a_violent_incident():
    """EE-only impact above threshold remains a validated violent contact."""
    monitor = _monitor_with_scene(end_effector_link=6)

    with patch("pybullet.getContactPoints", return_value=[_contact(link_a=6, force=120.0)]):
        with patch("pybullet.getBasePositionAndOrientation", return_value=((0.5, 0.0, 0.68), (0, 0, 0, 1))):
            snapshot = monitor.scan(sim_step=8)

    assert snapshot.violation_count == 1
    assert snapshot.unique_incident_count == 1
    assert monitor.violations[0].details["ee_force"] == pytest.approx(120.0)
    assert monitor.violations[0].details["raw_max_force"] == pytest.approx(120.0)


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


def test_live_control_grasp_descent_creates_no_safety_incident(tmp_path):
    """Live CONTROL approach/grasp descent must stay at zero safety incidents."""
    pytest.importorskip("pybullet")

    from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper
    from alpha.clients.simulator_client import PyBulletSimulatorClient
    from alpha.clients.telemetry_simulator_wrapper import TelemetrySimulatorWrapper
    from alpha.core.entities import FaultType
    from alpha.models.chaos_config import ChaosConfig
    from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger
    from alpha.services.experiment_matrix import configure_pybullet_determinism, seed_deterministic_environment
    from alpha.services.scene_service import SceneService

    seed_deterministic_environment(42)
    csv_path = tmp_path / "control_safety.csv"
    inner = PyBulletSimulatorClient()
    safety = PyBulletSafetyMonitor()
    chaos = ChaosRoboticsWrapper(inner, ChaosConfig(), safety)
    telemetry = ChaosTelemetryLogger(csv_path=str(csv_path))
    sim = TelemetrySimulatorWrapper(chaos, telemetry, safety)
    scene = SceneService(sim)

    sim.connect(gui=False)
    configure_pybullet_determinism()
    scene.build_and_settle(gui=False)
    telemetry.begin_episode(episode_id="control_grasp_repro", injected_fault_type=FaultType.NONE)

    # Exact Phase-B control targets from phase_b_control_000.
    sim.move_end_effector((0.5000000200079616, -0.14999875736505575, 0.8049880041119998))
    sim.move_end_effector((0.5000000200079616, -0.14999875736505575, 0.6749880041119998))

    snapshot = safety.summary()
    record = telemetry.finalize_episode(success=True, auto_save=False)
    sim.disconnect()

    assert telemetry.sim_step_count == 0  # cleared after finalize
    assert record.total_sim_steps == 480
    assert record.fault_occurred is False
    assert record.unique_safety_incident_count == 0
    assert record.safety_violation_count == 0
    assert record.violent_collision_count == 0
    assert record.first_violation_step is None
    assert record.total_violation_duration_steps == 0
    assert snapshot.unique_incident_count == 0
    assert safety.incidents == []
    assert safety.violations == []
