"""
Week 5 aggregation helpers over experiment telemetry records.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from alpha.core.entities import EpisodeResult
from alpha.schemas.telemetry import ExperimentTelemetryRecord
from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger


@dataclass(frozen=True)
class DegradationPoint:
    fault_intensity: float
    mean_total_sim_steps: float
    mean_total_wall_seconds: float
    mean_llm_total_tokens: float
    sample_count: int


@dataclass(frozen=True)
class MetricsSummary:
    mean_post_fault_completion_time_sim_steps: float | None
    mean_post_fault_completion_time_wall_seconds: float | None
    fault_classification_accuracy: float | None
    safety_violation_rate: float
    mean_safety_violation_count: float
    degradation_curve: tuple[DegradationPoint, ...]
    episode_count: int


class MetricsService:
    """Aggregate reliability metrics from telemetry records."""

    def compute_mean_post_fault_completion_time(
        self,
        records: Iterable[ExperimentTelemetryRecord | EpisodeResult],
        *,
        use_wall_clock: bool = False,
    ) -> float | None:
        values: list[float] = []
        for record in records:
            if isinstance(record, EpisodeResult):
                continue
            if not record.fault_occurred:
                continue
            metric = (
                record.post_fault_completion_time_wall_seconds
                if use_wall_clock
                else record.post_fault_completion_time_sim_steps
            )
            if metric is None:
                continue
            values.append(float(metric))
        return mean(values) if values else None

    def compute_degradation_curve(
        self,
        records: Iterable[ExperimentTelemetryRecord],
    ) -> list[DegradationPoint]:
        buckets: dict[float, list[ExperimentTelemetryRecord]] = {}
        for record in records:
            key = round(record.fault_intensity, 4)
            buckets.setdefault(key, []).append(record)

        points: list[DegradationPoint] = []
        for intensity in sorted(buckets):
            batch = buckets[intensity]
            points.append(
                DegradationPoint(
                    fault_intensity=intensity,
                    mean_total_sim_steps=mean(item.total_sim_steps for item in batch),
                    mean_total_wall_seconds=mean(item.total_wall_seconds for item in batch),
                    mean_llm_total_tokens=mean(item.llm_total_tokens for item in batch),
                    sample_count=len(batch),
                )
            )
        return points

    def compute_fault_classification_accuracy(
        self,
        records: Iterable[ExperimentTelemetryRecord],
    ) -> float | None:
        labeled = [record for record in records if record.fault_classification_correct is not None]
        if not labeled:
            return None
        correct = sum(1 for record in labeled if record.fault_classification_correct)
        return correct / len(labeled)

    def compute_safety_violation_rate(
        self,
        records: Iterable[ExperimentTelemetryRecord],
    ) -> float:
        items = list(records)
        if not items:
            return 0.0
        violated = sum(1 for record in items if record.safety_violation_count > 0)
        return violated / len(items)

    def compute_mean_safety_violation_count(
        self,
        records: Iterable[ExperimentTelemetryRecord],
    ) -> float:
        items = list(records)
        if not items:
            return 0.0
        return mean(record.safety_violation_count for record in items)

    def summarize(
        self,
        csv_path: str | None = None,
        records: Iterable[ExperimentTelemetryRecord] | None = None,
    ) -> MetricsSummary:
        data = list(records) if records is not None else ChaosTelemetryLogger.load_records(csv_path or "")
        return MetricsSummary(
            mean_post_fault_completion_time_sim_steps=self.compute_mean_post_fault_completion_time(
                data, use_wall_clock=False
            ),
            mean_post_fault_completion_time_wall_seconds=self.compute_mean_post_fault_completion_time(
                data, use_wall_clock=True
            ),
            fault_classification_accuracy=self.compute_fault_classification_accuracy(data),
            safety_violation_rate=self.compute_safety_violation_rate(data),
            mean_safety_violation_count=self.compute_mean_safety_violation_count(data),
            degradation_curve=tuple(self.compute_degradation_curve(data)),
            episode_count=len(data),
        )
