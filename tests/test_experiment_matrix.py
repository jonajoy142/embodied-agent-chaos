"""Tests for Week 6 experiment matrix scheduling and summaries."""

from __future__ import annotations

from alpha.core.entities import FaultType
from alpha.models.chaos_config import InjectionMode
from alpha.schemas.telemetry import ExperimentTelemetryRecord
from alpha.services.experiment_matrix import (
    IntensityLabel,
    MatrixScenario,
    build_chaos_config,
    generate_matrix_scenarios,
    intensity_params_for_fault,
    matrix_run_record_from_telemetry,
    render_summary_markdown,
    seed_deterministic_environment,
    summarize_matrix_runs,
    MatrixRunRecord,
)


def test_generate_matrix_scenario_count():
    scenarios = generate_matrix_scenarios(
        control_runs=10,
        episodes_per_variation=5,
        concurrent_runs=5,
    )
    assert len(scenarios) == 75
    assert sum(1 for item in scenarios if item.scenario_group == "control") == 10
    assert sum(1 for item in scenarios if item.scenario_group == "concurrent") == 5
    assert sum(1 for item in scenarios if item.scenario_group == "fault_matrix") == 60


def test_intensity_mapping_monotonic():
    low = intensity_params_for_fault(FaultType.GRIP_SLIP, IntensityLabel.LOW)
    high = intensity_params_for_fault(FaultType.GRIP_SLIP, IntensityLabel.HIGH)
    assert low.slip_probability < high.slip_probability


def test_build_chaos_config_concurrent_mode():
    config = build_chaos_config(
        fault_types=(FaultType.SENSOR_LAG, FaultType.GRIP_SLIP),
        intensity_label=IntensityLabel.MEDIUM,
        episode_seed=7,
        concurrent=True,
    )
    assert config.mode == InjectionMode.CONCURRENT
    assert len(config.faults) == 2


def test_summary_markdown_table():
    records = [
        MatrixRunRecord(
            run_id="sensor_lag_low_000",
            scenario_group="fault_matrix",
            fault_type="sensor_lag",
            intensity_label="low",
            intensity_value=0.33,
            episode_id="ep-1",
            success=True,
            crashed=False,
            timed_out=False,
            post_fault_completion_time_sim_steps=12.0,
            safety_violation_count=0,
            total_wall_seconds=1.0,
            total_sim_steps=100,
            fault_occurred=True,
        ),
        MatrixRunRecord(
            run_id="sensor_lag_low_001",
            scenario_group="fault_matrix",
            fault_type="sensor_lag",
            intensity_label="low",
            intensity_value=0.33,
            episode_id="ep-2",
            success=False,
            crashed=False,
            timed_out=False,
            post_fault_completion_time_sim_steps=None,
            safety_violation_count=2,
            total_wall_seconds=1.5,
            total_sim_steps=120,
            fault_occurred=True,
        ),
    ]
    rows = summarize_matrix_runs(records)
    markdown = render_summary_markdown(rows)

    assert "sensor_lag" in markdown
    assert "50.0%" in markdown
    assert "12.0" in markdown
    assert "1.00" in markdown


def test_matrix_run_record_from_telemetry():
    scenario = MatrixScenario(
        run_id="control_000",
        scenario_group="control",
        fault_type="none",
        intensity_label="none",
        intensity_value=0.0,
        episode_index=0,
    )
    telemetry = ExperimentTelemetryRecord(
        episode_id="control_000",
        success=True,
        total_sim_steps=240,
        total_wall_seconds=0.5,
    )
    record = matrix_run_record_from_telemetry(scenario, telemetry)
    assert record.run_id == "control_000"
    assert record.success is True


def test_load_pilot_config():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pilot_path = root / "configs" / "experiments" / "pilot.yaml"
    from alpha.services.experiment_matrix import generate_pilot_scenarios, load_pilot_config

    pilot = load_pilot_config(str(pilot_path))
    scenarios = generate_pilot_scenarios(pilot)
    assert pilot.experiment_id == "phase_a_pilot"
    assert pilot.agent_provider == "openai"
    assert len(scenarios) == 70  # 5 control + 4*3*5 faults + 5 concurrent


def test_seed_deterministic_environment_is_repeatable():
    seed_deterministic_environment(42)
    import random

    first = random.random()
    seed_deterministic_environment(42)
    second = random.random()
    assert first == second
