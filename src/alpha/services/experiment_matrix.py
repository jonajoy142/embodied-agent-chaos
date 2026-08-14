"""
Week 6 experiment matrix definitions and helpers.

Builds the full factorial schedule (control, single-fault matrix, concurrent
runs), enforces deterministic seeding, and aggregates markdown summaries.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import yaml

from alpha.config import settings
from alpha.core.entities import FaultType
from alpha.models.chaos_config import (
    ChaosConfig,
    FaultIntensity,
    FaultSpec,
    InjectionMode,
    TriggerConfig,
    TriggerKind,
)
from alpha.schemas.telemetry import ExperimentTelemetryRecord


MATRIX_FAULT_TYPES: tuple[FaultType, ...] = (
    FaultType.SENSOR_LAG,
    FaultType.GRIP_SLIP,
    FaultType.UNREACHABLE_IK,
    FaultType.PLANNER_OUTPUT_CORRUPTION,
)

CONCURRENT_FAULT_TYPES: tuple[FaultType, ...] = (
    FaultType.SENSOR_LAG,
    FaultType.GRIP_SLIP,
)


class IntensityLabel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


INTENSITY_SCALAR: dict[IntensityLabel, float] = {
    IntensityLabel.LOW: 0.33,
    IntensityLabel.MEDIUM: 0.66,
    IntensityLabel.HIGH: 1.0,
}


@dataclass(frozen=True)
class MatrixScenario:
    run_id: str
    scenario_group: str
    fault_type: str
    intensity_label: str
    intensity_value: float
    episode_index: int
    concurrent: bool = False
    master_seed: int = 42


@dataclass(frozen=True)
class MatrixRunRecord:
    run_id: str
    scenario_group: str
    fault_type: str
    intensity_label: str
    intensity_value: float
    episode_id: str
    success: bool
    crashed: bool
    timed_out: bool
    post_fault_completion_time_sim_steps: float | None
    safety_violation_count: int
    total_wall_seconds: float
    total_sim_steps: int
    fault_occurred: bool


@dataclass(frozen=True)
class MatrixSummaryRow:
    fault_type: str
    intensity: str
    success_rate: float
    average_post_fault_completion_time: float | None
    safety_violations: float


def seed_deterministic_environment(master_seed: int) -> None:
    """Seed Python, NumPy, and the standard-library RNG used by chaos faults."""
    random.seed(master_seed)
    np.random.seed(master_seed)


def configure_pybullet_determinism() -> None:
    """Apply deterministic physics parameters after ``pybullet.connect()``."""
    try:
        import pybullet as p
    except ImportError:
        return

    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / 240.0,
        numSolverIterations=150,
        deterministicOverlappingPairs=1,
        useSplitImpulse=1,
    )
    if hasattr(p, "setRandomSeed"):
        p.setRandomSeed(0)


def intensity_params_for_fault(fault_type: FaultType, label: IntensityLabel) -> FaultIntensity:
    """Map qualitative intensity labels to fault-specific severity knobs."""
    scalar = INTENSITY_SCALAR[label]
    if fault_type == FaultType.SENSOR_LAG:
        return FaultIntensity(lag_seconds=0.25 + 0.75 * scalar)
    if fault_type == FaultType.GRIP_SLIP:
        return FaultIntensity(slip_probability=min(0.99, 0.2 + 0.75 * scalar), weaken_force_factor=0.0)
    if fault_type == FaultType.UNREACHABLE_IK:
        return FaultIntensity(offset_magnitude=0.08 + 0.32 * scalar, offset=(1.0, 0.0, 0.0))
    if fault_type == FaultType.PLANNER_OUTPUT_CORRUPTION:
        return FaultIntensity(
            corruption_probability=min(1.0, 0.2 + 0.8 * scalar),
            inject_malformed_chars=label != IntensityLabel.LOW,
        )
    return FaultIntensity()


def build_chaos_config(
    *,
    fault_types: Iterable[FaultType],
    intensity_label: IntensityLabel,
    episode_seed: int,
    concurrent: bool = False,
) -> ChaosConfig:
    """Construct a ``ChaosConfig`` for one matrix scenario."""
    specs: list[FaultSpec] = []
    for fault_type in fault_types:
        intensity = intensity_params_for_fault(fault_type, intensity_label)
        trigger = TriggerConfig(kind=TriggerKind.IMMEDIATE)
        if fault_type == FaultType.GRIP_SLIP:
            trigger = TriggerConfig(kind=TriggerKind.EVENT, phase="lift")
        specs.append(
            FaultSpec(
                fault_type=fault_type,
                enabled=True,
                trigger=trigger,
                intensity=intensity,
                probability=1.0,
            )
        )

    return ChaosConfig(
        faults=tuple(specs),
        mode=InjectionMode.CONCURRENT if concurrent else InjectionMode.SINGLE,
        seed=episode_seed,
    )


def generate_matrix_scenarios(
    *,
    master_seed: int = 42,
    control_runs: int = 10,
    episodes_per_variation: int = 5,
    concurrent_runs: int = 5,
) -> list[MatrixScenario]:
    """Build the full Week 6 schedule."""
    scenarios: list[MatrixScenario] = []

    for index in range(control_runs):
        scenarios.append(
            MatrixScenario(
                run_id=f"control_{index:03d}",
                scenario_group="control",
                fault_type="none",
                intensity_label="none",
                intensity_value=0.0,
                episode_index=index,
                master_seed=master_seed,
            )
        )

    for fault_type in MATRIX_FAULT_TYPES:
        for label in IntensityLabel:
            for index in range(episodes_per_variation):
                scenarios.append(
                    MatrixScenario(
                        run_id=f"{fault_type.value}_{label.value}_{index:03d}",
                        scenario_group="fault_matrix",
                        fault_type=fault_type.value,
                        intensity_label=label.value,
                        intensity_value=INTENSITY_SCALAR[label],
                        episode_index=index,
                        master_seed=master_seed,
                    )
                )

    for index in range(concurrent_runs):
        scenarios.append(
            MatrixScenario(
                run_id=f"concurrent_{index:03d}",
                scenario_group="concurrent",
                fault_type="sensor_lag+grip_slip",
                intensity_label=IntensityLabel.MEDIUM.value,
                intensity_value=INTENSITY_SCALAR[IntensityLabel.MEDIUM],
                episode_index=index,
                concurrent=True,
                master_seed=master_seed,
            )
        )

    return scenarios


def load_checkpoint(path: str) -> set[str]:
    file_path = Path(path)
    if not file_path.is_file():
        return set()
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    return set(payload.get("completed_run_ids", []))


def save_checkpoint(path: str, completed_run_ids: set[str], master_seed: int) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(
            {
                "master_seed": master_seed,
                "completed_run_ids": sorted(completed_run_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


MATRIX_LOG_FIELDS = [
    "run_id",
    "scenario_group",
    "fault_type",
    "intensity_label",
    "intensity_value",
    "episode_id",
    "success",
    "crashed",
    "timed_out",
    "post_fault_completion_time_sim_steps",
    "safety_violation_count",
    "total_wall_seconds",
    "total_sim_steps",
    "fault_occurred",
]


def append_matrix_run_log(path: str, record: MatrixRunRecord) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    exists = file_path.is_file()
    row = {
        "run_id": record.run_id,
        "scenario_group": record.scenario_group,
        "fault_type": record.fault_type,
        "intensity_label": record.intensity_label,
        "intensity_value": record.intensity_value,
        "episode_id": record.episode_id,
        "success": record.success,
        "crashed": record.crashed,
        "timed_out": record.timed_out,
        "post_fault_completion_time_sim_steps": record.post_fault_completion_time_sim_steps,
        "safety_violation_count": record.safety_violation_count,
        "total_wall_seconds": record.total_wall_seconds,
        "total_sim_steps": record.total_sim_steps,
        "fault_occurred": record.fault_occurred,
    }
    with file_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_matrix_run_log(path: str) -> list[MatrixRunRecord]:
    file_path = Path(path)
    if not file_path.is_file():
        return []

    records: list[MatrixRunRecord] = []
    with file_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pfct_raw = row.get("post_fault_completion_time_sim_steps", row.get("ttr_sim_steps"))
            if pfct_raw in ("", None):
                pfct_value = None
            elif pfct_raw == "inf":
                pfct_value = None
            else:
                pfct_value = float(pfct_raw)
            records.append(
                MatrixRunRecord(
                    run_id=row["run_id"],
                    scenario_group=row["scenario_group"],
                    fault_type=row["fault_type"],
                    intensity_label=row["intensity_label"],
                    intensity_value=float(row["intensity_value"]),
                    episode_id=row["episode_id"],
                    success=str(row["success"]).lower() in {"1", "true", "yes"},
                    crashed=str(row["crashed"]).lower() in {"1", "true", "yes"},
                    timed_out=str(row["timed_out"]).lower() in {"1", "true", "yes"},
                    post_fault_completion_time_sim_steps=pfct_value,
                    safety_violation_count=int(float(row.get("safety_violation_count") or 0)),
                    total_wall_seconds=float(row.get("total_wall_seconds") or 0.0),
                    total_sim_steps=int(float(row.get("total_sim_steps") or 0)),
                    fault_occurred=str(row.get("fault_occurred")).lower() in {"1", "true", "yes"},
                )
            )
    return records


def summarize_matrix_runs(records: Iterable[MatrixRunRecord]) -> list[MatrixSummaryRow]:
    """Aggregate matrix rows for markdown reporting."""
    buckets: dict[tuple[str, str], list[MatrixRunRecord]] = {}
    for record in records:
        key = (record.fault_type, record.intensity_label)
        buckets.setdefault(key, []).append(record)

    summary_rows: list[MatrixSummaryRow] = []
    for (fault_type, intensity), batch in sorted(buckets.items()):
        successes = sum(1 for item in batch if item.success)
        success_rate = successes / len(batch) if batch else 0.0
        finite_pfct = [
            float(item.post_fault_completion_time_sim_steps)
            for item in batch
            if item.post_fault_completion_time_sim_steps is not None
        ]
        average_pfct = mean(finite_pfct) if finite_pfct else None
        safety_violations = mean(item.safety_violation_count for item in batch) if batch else 0.0
        summary_rows.append(
            MatrixSummaryRow(
                fault_type=fault_type,
                intensity=intensity,
                success_rate=success_rate,
                average_post_fault_completion_time=average_pfct,
                safety_violations=safety_violations,
            )
        )
    return summary_rows


def format_post_fault_completion_time(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def render_summary_markdown(rows: Iterable[MatrixSummaryRow]) -> str:
    lines = [
        "# Experiment Matrix Summary",
        "",
        "| Fault Type | Intensity | Success Rate | Mean Post-Fault Completion Time (sim steps) | Safety Violations |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {fault} | {intensity} | {success:.1%} | {pfct} | {safety:.2f} |".format(
                fault=row.fault_type,
                intensity=row.intensity,
                success=row.success_rate,
                pfct=format_post_fault_completion_time(row.average_post_fault_completion_time),
                safety=row.safety_violations,
            )
        )
    lines.append("")
    return "\n".join(lines)


def matrix_run_record_from_telemetry(
    scenario: MatrixScenario,
    telemetry_record: ExperimentTelemetryRecord,
) -> MatrixRunRecord:
    return MatrixRunRecord(
        run_id=scenario.run_id,
        scenario_group=scenario.scenario_group,
        fault_type=scenario.fault_type,
        intensity_label=scenario.intensity_label,
        intensity_value=scenario.intensity_value,
        episode_id=telemetry_record.episode_id,
        success=telemetry_record.success,
        crashed=telemetry_record.crashed,
        timed_out=telemetry_record.timed_out,
        post_fault_completion_time_sim_steps=telemetry_record.post_fault_completion_time_sim_steps,
        safety_violation_count=telemetry_record.safety_violation_count,
        total_wall_seconds=telemetry_record.total_wall_seconds,
        total_sim_steps=telemetry_record.total_sim_steps,
        fault_occurred=telemetry_record.fault_occurred,
    )


def default_matrix_paths() -> dict[str, str]:
    return {
        "matrix_log": os.path.join(settings.RESULTS_DIR, "experiment_matrix_runs.csv"),
        "checkpoint": os.path.join(settings.RESULTS_DIR, "experiment_matrix_checkpoint.json"),
        "summary": os.path.join(settings.RESULTS_DIR, "experiment_matrix_summary.md"),
    }


@dataclass(frozen=True)
class PilotConfig:
    experiment_id: str
    phase: str
    master_seed: int
    episodes_per_variation: int
    control_runs: int
    concurrent_runs: int
    concurrent_intensity: str
    agent_provider: str
    agent_model: str
    llm_min_interval: float
    block: str
    results_dir: str


def load_pilot_config(path: str) -> PilotConfig:
    """Load Phase A pilot settings from YAML."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    agent = payload.get("agent", {})
    output = payload.get("output", {})
    return PilotConfig(
        experiment_id=str(payload["experiment_id"]),
        phase=str(payload.get("phase", "pilot")),
        master_seed=int(payload.get("master_seed", 42)),
        episodes_per_variation=int(payload.get("episodes_per_variation", 5)),
        control_runs=int(payload.get("control_runs", 5)),
        concurrent_runs=int(payload.get("concurrent_runs", 5)),
        concurrent_intensity=str(payload.get("concurrent_intensity", "medium")),
        agent_provider=str(agent.get("provider", "openai")),
        agent_model=str(agent.get("model", "gpt-4o")),
        llm_min_interval=float(agent.get("min_interval_seconds", 0.5)),
        block=str(payload.get("block", "red")),
        results_dir=str(output.get("results_dir", settings.PILOT_RESULTS_DIR)),
    )


def pilot_paths(pilot: PilotConfig) -> dict[str, str]:
    base = pilot.results_dir
    return {
        "results_dir": base,
        "matrix_log": os.path.join(base, "experiment_matrix_runs.csv"),
        "checkpoint": os.path.join(base, "pilot_checkpoint.json"),
        "summary": os.path.join(base, "pilot_summary.md"),
    }


def generate_pilot_scenarios(pilot: PilotConfig) -> list[MatrixScenario]:
    """Build the Phase A pilot schedule from ``configs/experiments/pilot.yaml``."""
    return generate_matrix_scenarios(
        master_seed=pilot.master_seed,
        control_runs=pilot.control_runs,
        episodes_per_variation=pilot.episodes_per_variation,
        concurrent_runs=pilot.concurrent_runs,
    )
