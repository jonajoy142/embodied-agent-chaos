"""
Week 5 metrics harness: episode telemetry logging and CSV export.

Tracks task outcome, post-fault completion time (single-shot agent),
degradation inputs, and validated safety violations.

This is **not** a recovery/MTTR logger until a replanning loop exists.
"""

from __future__ import annotations

import csv
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha.clients.safety_monitor import SafetySnapshot
from alpha.config import settings
from alpha.core.entities import EpisodeResult, FaultType
from alpha.schemas.telemetry import ExperimentTelemetryRecord


@dataclass
class _EpisodeState:
    experiment_id: str
    episode_id: str
    seed: int | None = None
    agent_name: str = "scripted"
    model_identifier: str = ""
    block: str = ""
    injected_fault_type: FaultType = FaultType.NONE
    fault_intensity: float = 0.0
    fault_trigger: str = ""
    fault_duration: str = ""
    fault_occurred: bool = False
    replan_count: int = 0
    agent_diagnosed_fault: str = ""
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    sim_step_count: int = 0
    fault_injected_at_step: int | None = None
    fault_injected_at_wall: float | None = None
    safety_violation_count: int = 0
    violent_collision_count: int = 0
    out_of_workspace_count: int = 0
    started_at_wall: float = field(default_factory=time.monotonic)
    fault_events: list[dict[str, Any]] = field(default_factory=list)


class ChaosTelemetryLogger:
    """Tracks reliability metrics for one episode at a time and appends a CSV row."""

    def __init__(
        self,
        csv_path: str = settings.EXPERIMENTS_CSV,
        max_post_fault_completion_sim_steps: int = settings.MAX_POST_FAULT_COMPLETION_SIM_STEPS,
        max_post_fault_completion_wall_seconds: float = settings.MAX_POST_FAULT_COMPLETION_WALL_SECONDS,
    ) -> None:
        self.csv_path = csv_path
        self.max_post_fault_completion_sim_steps = max_post_fault_completion_sim_steps
        self.max_post_fault_completion_wall_seconds = max_post_fault_completion_wall_seconds
        self._state: _EpisodeState | None = None
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)

    @property
    def sim_step_count(self) -> int:
        return 0 if self._state is None else self._state.sim_step_count

    def begin_episode(
        self,
        *,
        episode_id: str | None = None,
        experiment_id: str = "",
        agent_name: str = "scripted",
        model_identifier: str = "",
        block: str = "",
        injected_fault_type: FaultType = FaultType.NONE,
        fault_intensity: float = 0.0,
        seed: int | None = None,
        fault_trigger: str = "",
        fault_duration: str = "",
    ) -> str:
        episode = episode_id or f"ep-{uuid.uuid4().hex[:12]}"
        self._state = _EpisodeState(
            experiment_id=experiment_id,
            episode_id=episode,
            seed=seed,
            agent_name=agent_name,
            model_identifier=model_identifier,
            block=block,
            injected_fault_type=injected_fault_type,
            fault_intensity=fault_intensity,
            fault_trigger=fault_trigger,
            fault_duration=fault_duration,
        )
        return episode

    def tick_sim_step(self, steps: int = 1) -> None:
        if self._state is not None:
            self._state.sim_step_count += steps

    def record_fault_injection(self, event: dict[str, Any]) -> None:
        if self._state is None or not event.get("fault_applied"):
            return
        self._state.fault_events.append(dict(event))
        self._state.fault_occurred = True
        if self._state.fault_injected_at_step is None:
            self._state.fault_injected_at_step = event.get("sim_step", self._state.sim_step_count)
            self._state.fault_injected_at_wall = time.monotonic() - self._state.started_at_wall
        fault_type = event.get("fault_type")
        if fault_type and self._state.injected_fault_type == FaultType.NONE:
            try:
                self._state.injected_fault_type = FaultType(str(fault_type))
            except ValueError:
                pass

    def ingest_fault_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.record_fault_injection(event)

    def record_replan(self, *, llm_prompt_tokens: int = 0, llm_completion_tokens: int = 0) -> None:
        if self._state is None:
            return
        self._state.replan_count += 1
        self._state.llm_prompt_tokens += llm_prompt_tokens
        self._state.llm_completion_tokens += llm_completion_tokens

    def record_llm_usage(self, *, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        if self._state is None:
            return
        self._state.llm_prompt_tokens += prompt_tokens
        self._state.llm_completion_tokens += completion_tokens

    def record_agent_diagnosis(self, diagnosed_fault: str) -> None:
        if self._state is not None:
            self._state.agent_diagnosed_fault = diagnosed_fault.strip()

    def update_safety(self, snapshot: SafetySnapshot) -> None:
        if self._state is None:
            return
        self._state.safety_violation_count = snapshot.violation_count
        self._state.violent_collision_count = snapshot.violent_collision_count
        self._state.out_of_workspace_count = snapshot.out_of_workspace_count

    def finalize_episode(
        self,
        *,
        success: bool,
        crashed: bool = False,
        timed_out: bool = False,
        episode_result: EpisodeResult | None = None,
        failure_reason: str = "",
        auto_save: bool = True,
    ) -> ExperimentTelemetryRecord:
        """
        Finalize episode metrics.

        ``post_fault_completion_time_*`` is recorded only when:
        - a fault occurred,
        - execution continued after injection, and
        - the task completed successfully.

        Failed or aborted episodes leave the metric empty (``None``).
        """
        if self._state is None:
            raise RuntimeError("begin_episode() must be called before finalize_episode()")

        state = self._state
        total_wall = time.monotonic() - state.started_at_wall

        if episode_result is not None:
            state.agent_name = episode_result.agent_name or state.agent_name
            state.block = episode_result.task.block.value
            state.fault_occurred = state.fault_occurred or episode_result.fault_occurred
            if episode_result.fault_type != FaultType.NONE:
                state.injected_fault_type = episode_result.fault_type

        pfct_steps, pfct_wall = self._compute_post_fault_completion_time(
            success=success,
            crashed=crashed,
            timed_out=timed_out,
            total_sim_steps=state.sim_step_count,
            total_wall_seconds=total_wall,
        )

        classification = self._classification_correct(
            actual=state.injected_fault_type,
            diagnosed=state.agent_diagnosed_fault,
            fault_occurred=state.fault_occurred,
        )

        degradation_payload = {
            "fault_intensity": state.fault_intensity,
            "total_sim_steps": state.sim_step_count,
            "total_wall_seconds": round(total_wall, 6),
            "llm_total_tokens": state.llm_prompt_tokens + state.llm_completion_tokens,
            "llm_prompt_tokens": state.llm_prompt_tokens,
            "llm_completion_tokens": state.llm_completion_tokens,
            "replan_count": state.replan_count,
        }

        record = ExperimentTelemetryRecord(
            experiment_id=state.experiment_id,
            episode_id=state.episode_id,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            seed=state.seed,
            agent_name=state.agent_name,
            model_identifier=state.model_identifier,
            block=state.block,
            injected_fault_type=state.injected_fault_type,
            fault_intensity=state.fault_intensity,
            fault_trigger=state.fault_trigger,
            fault_duration=state.fault_duration,
            fault_occurred=state.fault_occurred,
            success=success,
            crashed=crashed,
            timed_out=timed_out,
            failure_reason=failure_reason,
            post_fault_completion_time_sim_steps=pfct_steps,
            post_fault_completion_time_wall_seconds=pfct_wall,
            total_sim_steps=state.sim_step_count,
            total_wall_seconds=round(total_wall, 6),
            llm_total_tokens=state.llm_prompt_tokens + state.llm_completion_tokens,
            llm_prompt_tokens=state.llm_prompt_tokens,
            llm_completion_tokens=state.llm_completion_tokens,
            replan_count=state.replan_count,
            agent_diagnosed_fault=state.agent_diagnosed_fault,
            fault_classification_correct=classification,
            safety_violation_count=state.safety_violation_count,
            violent_collision_count=state.violent_collision_count,
            out_of_workspace_count=state.out_of_workspace_count,
            degradation_curve_json=json.dumps(degradation_payload),
        )

        if auto_save:
            self.save_record(record)

        self._state = None
        return record

    def save_record(self, record: ExperimentTelemetryRecord) -> None:
        row = record.to_csv_row()
        fieldnames = list(ExperimentTelemetryRecord.model_fields.keys())
        file_exists = os.path.isfile(self.csv_path)

        if file_exists:
            with open(self.csv_path, newline="", encoding="utf-8") as handle:
                existing_header = next(csv.reader(handle), [])
            if existing_header and existing_header != fieldnames:
                self._migrate_csv_header(self.csv_path, existing_header, fieldnames)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @classmethod
    def load_records(cls, csv_path: str = settings.EXPERIMENTS_CSV) -> list[ExperimentTelemetryRecord]:
        path = Path(csv_path)
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return [ExperimentTelemetryRecord.from_csv_row(row) for row in csv.DictReader(handle)]

    def _compute_post_fault_completion_time(
        self,
        *,
        success: bool,
        crashed: bool,
        timed_out: bool,
        total_sim_steps: int,
        total_wall_seconds: float,
    ) -> tuple[float | None, float | None]:
        if self._state is None or not self._state.fault_occurred:
            return None, None
        if crashed or timed_out or not success:
            return None, None

        fault_step = self._state.fault_injected_at_step or 0
        fault_wall = self._state.fault_injected_at_wall or 0.0
        pfct_steps = float(max(0, total_sim_steps - fault_step))
        pfct_wall = float(max(0.0, total_wall_seconds - fault_wall))

        if pfct_steps > self.max_post_fault_completion_sim_steps:
            return None, None
        if pfct_wall > self.max_post_fault_completion_wall_seconds:
            return None, None
        return pfct_steps, pfct_wall

    @staticmethod
    def _classification_correct(
        *,
        actual: FaultType,
        diagnosed: str,
        fault_occurred: bool,
    ) -> bool | None:
        if not diagnosed:
            return None
        if not fault_occurred and actual == FaultType.NONE:
            return diagnosed.lower() in {"none", "no_fault", "healthy", ""}
        return diagnosed.lower() == actual.value.lower()

    @staticmethod
    def _migrate_csv_header(csv_path: str, existing_header: list[str], expected_fields: list[str]) -> None:
        path = Path(csv_path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=expected_fields)
            writer.writeheader()
            for old_row in rows:
                writer.writerow({field: old_row.get(field, "") for field in expected_fields})
