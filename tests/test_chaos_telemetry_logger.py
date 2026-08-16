"""Unit tests for Week 5 ChaosTelemetryLogger and MetricsService."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha.clients.safety_monitor import PyBulletSafetyMonitor, SafetySnapshot
from alpha.core.entities import FaultType
from alpha.config import settings
from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger
from alpha.services.metrics_service import MetricsService


@pytest.fixture
def telemetry_csv(tmp_path: Path) -> str:
    return str(tmp_path / "experiments_log.csv")


def test_post_fault_completion_time_none_when_no_fault(telemetry_csv: str):
    logger = ChaosTelemetryLogger(csv_path=telemetry_csv)
    logger.begin_episode(agent_name="mock", block="red")
    logger.tick_sim_step(120)
    record = logger.finalize_episode(success=True)

    assert record.post_fault_completion_time_sim_steps is None
    assert record.post_fault_completion_time_wall_seconds is None
    assert record.fault_occurred is False


def test_post_fault_completion_time_on_success_after_fault(telemetry_csv: str):
    logger = ChaosTelemetryLogger(csv_path=telemetry_csv)
    logger.begin_episode(
        agent_name="openai",
        block="red",
        injected_fault_type=FaultType.GRIP_SLIP,
        fault_intensity=0.85,
    )
    logger.record_fault_injection({"fault_applied": True, "fault_type": "grip_slip", "sim_step": 40})
    logger.tick_sim_step(100)
    record = logger.finalize_episode(success=True)

    assert record.post_fault_completion_time_sim_steps == 60.0
    assert record.post_fault_completion_time_wall_seconds is not None
    assert record.post_fault_completion_time_wall_seconds >= 0.0


def test_post_fault_completion_time_none_on_failure(telemetry_csv: str):
    logger = ChaosTelemetryLogger(csv_path=telemetry_csv)
    logger.begin_episode(injected_fault_type=FaultType.UNREACHABLE_IK, fault_intensity=0.5)
    logger.record_fault_injection({"fault_applied": True, "fault_type": "unreachable_ik", "sim_step": 10})
    logger.tick_sim_step(50)
    record = logger.finalize_episode(success=False)

    assert record.post_fault_completion_time_sim_steps is None
    assert record.post_fault_completion_time_wall_seconds is None


def test_post_fault_completion_time_none_on_crash_or_timeout(telemetry_csv: str):
    logger = ChaosTelemetryLogger(csv_path=telemetry_csv)
    logger.begin_episode(injected_fault_type=FaultType.SENSOR_LAG)
    logger.record_fault_injection({"fault_applied": True, "fault_type": "sensor_lag", "sim_step": 5})

    crashed = logger.finalize_episode(success=False, crashed=True)
    assert crashed.post_fault_completion_time_sim_steps is None

    logger.begin_episode(injected_fault_type=FaultType.SENSOR_LAG)
    logger.record_fault_injection({"fault_applied": True, "fault_type": "sensor_lag", "sim_step": 5})
    timed_out = logger.finalize_episode(success=False, timed_out=True)
    assert timed_out.post_fault_completion_time_sim_steps is None


def test_fault_classification_accuracy(telemetry_csv: str):
    logger = ChaosTelemetryLogger(csv_path=telemetry_csv)
    logger.begin_episode(injected_fault_type=FaultType.GRIP_SLIP, fault_intensity=0.9)
    logger.record_fault_injection({"fault_applied": True, "fault_type": "grip_slip", "sim_step": 1})
    logger.record_agent_diagnosis("grip_slip")
    correct = logger.finalize_episode(success=True)
    assert correct.fault_classification_correct is True

    logger.begin_episode(injected_fault_type=FaultType.GRIP_SLIP)
    logger.record_fault_injection({"fault_applied": True, "fault_type": "grip_slip", "sim_step": 1})
    logger.record_agent_diagnosis("sensor_lag")
    wrong = logger.finalize_episode(success=False)
    assert wrong.fault_classification_correct is False


def test_degradation_curve_payload_and_csv_autosave(telemetry_csv: str):
    logger = ChaosTelemetryLogger(csv_path=telemetry_csv)
    logger.begin_episode(
        agent_name="openai",
        block="red",
        injected_fault_type=FaultType.POSITION_OFFSET,
        fault_intensity=0.25,
    )
    logger.record_llm_usage(prompt_tokens=120, completion_tokens=80)
    logger.tick_sim_step(300)
    record = logger.finalize_episode(success=True)

    payload = json.loads(record.degradation_curve_json)
    assert payload["fault_intensity"] == 0.25
    assert payload["llm_total_tokens"] == 200
    assert payload["total_sim_steps"] == 300
    assert Path(telemetry_csv).is_file()

    loaded = ChaosTelemetryLogger.load_records(telemetry_csv)
    assert len(loaded) == 1
    assert loaded[0].llm_total_tokens == 200


def test_safety_violation_counts_are_logged(telemetry_csv: str):
    logger = ChaosTelemetryLogger(csv_path=telemetry_csv)
    logger.begin_episode(block="red")
    logger.update_safety(
        SafetySnapshot(
            violation_count=3,
            violent_collision_count=2,
            out_of_workspace_count=1,
            unique_incident_count=1,
            first_violation_step=4,
            total_violation_duration_steps=7,
        )
    )
    logger.record_observation_lag(requested_lag=1.25, observed_lag=0.8)
    record = logger.finalize_episode(success=False)

    assert record.safety_violation_count == 3
    assert record.violent_collision_count == 2
    assert record.out_of_workspace_count == 1
    assert record.unique_safety_incident_count == 1
    assert record.first_violation_step == 4
    assert record.total_violation_duration_steps == 7
    assert record.observation_lag_requested == pytest.approx(1.25)
    assert record.observation_lag_observed == pytest.approx(0.8)


def test_metrics_service_mean_post_fault_completion_time_and_degradation(telemetry_csv: str):
    service = MetricsService()

    for intensity, pfct, tokens in [(0.2, 10.0, 100), (0.8, 30.0, 250), (0.8, 50.0, 300)]:
        logger = ChaosTelemetryLogger(csv_path=telemetry_csv)
        logger.begin_episode(
            injected_fault_type=FaultType.GRIP_SLIP,
            fault_intensity=intensity,
        )
        logger.record_fault_injection({"fault_applied": True, "fault_type": "grip_slip", "sim_step": 0})
        logger.record_llm_usage(prompt_tokens=tokens // 2, completion_tokens=tokens // 2)
        logger.tick_sim_step(int(pfct))
        logger.finalize_episode(success=True)

    records = ChaosTelemetryLogger.load_records(telemetry_csv)
    mean_pfct = service.compute_mean_post_fault_completion_time(records)
    curve = service.compute_degradation_curve(records)

    assert mean_pfct == pytest.approx(30.0)
    assert len(curve) == 2
    high_intensity = next(point for point in curve if point.fault_intensity == 0.8)
    assert high_intensity.mean_llm_total_tokens == pytest.approx(275.0)
    assert high_intensity.sample_count == 2


def test_pybullet_safety_monitor_flags_out_of_workspace():
    pytest.importorskip("pybullet")
    from alpha.clients.simulator_client import PyBulletSimulatorClient

    inner = PyBulletSimulatorClient()
    monitor = PyBulletSafetyMonitor(
        workspace_bounds=((0.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
        place_zone=settings.PLACE_ZONE_POSITION,
        contact_force_threshold=10_000.0,
    )
    inner.connect(gui=False)
    handles = inner.build_scene()
    monitor.set_scene(handles)
    snapshot = monitor.scan()

    assert snapshot.violation_count == 0
    inner.disconnect()
