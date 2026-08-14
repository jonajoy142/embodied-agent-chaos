"""
I/O schema for Week 5+ experiment telemetry records.

``post_fault_completion_time_*`` measures elapsed time from first confirmed
fault injection until task completion in the **single-shot** agent. It is
**not** time-to-recovery and must not be labeled MTTR/TTR until a genuine
replanning loop exists.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from alpha.core.entities import FaultType


class ExperimentTelemetryRecord(BaseModel):
    """One episode summary row for experiment analysis."""

    experiment_id: str = ""
    episode_id: str
    recorded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seed: int | None = None
    agent_name: str = "scripted"
    model_identifier: str = ""
    block: str = ""
    injected_fault_type: FaultType = FaultType.NONE
    fault_intensity: float = 0.0
    fault_trigger: str = ""
    fault_duration: str = ""
    fault_occurred: bool = False
    success: bool = False
    crashed: bool = False
    timed_out: bool = False
    failure_reason: str = ""
    post_fault_completion_time_sim_steps: float | None = None
    post_fault_completion_time_wall_seconds: float | None = None
    total_sim_steps: int = 0
    total_wall_seconds: float = 0.0
    llm_total_tokens: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    replan_count: int = 0
    agent_diagnosed_fault: str = ""
    fault_classification_correct: bool | None = None
    safety_violation_count: int = 0
    violent_collision_count: int = 0
    out_of_workspace_count: int = 0
    degradation_curve_json: str = ""

    def degradation_payload(self) -> dict[str, Any]:
        if not self.degradation_curve_json:
            return {
                "fault_intensity": self.fault_intensity,
                "total_sim_steps": self.total_sim_steps,
                "total_wall_seconds": self.total_wall_seconds,
                "llm_total_tokens": self.llm_total_tokens,
            }
        return json.loads(self.degradation_curve_json)

    def to_csv_row(self) -> dict[str, Any]:
        row = self.model_dump(mode="json")
        row["injected_fault_type"] = self.injected_fault_type.value
        if row.get("fault_classification_correct") is None:
            row["fault_classification_correct"] = ""
        if row.get("seed") is None:
            row["seed"] = ""
        return row

    @classmethod
    def from_csv_row(cls, row: dict[str, Any]) -> ExperimentTelemetryRecord:
        payload = dict(row)
        for key in ("post_fault_completion_time_sim_steps", "post_fault_completion_time_wall_seconds"):
            if payload.get(key) in ("", None):
                payload[key] = None
            else:
                payload[key] = float(payload[key])
        legacy_ttr_map = {
            "ttr_sim_steps": "post_fault_completion_time_sim_steps",
            "ttr_wall_seconds": "post_fault_completion_time_wall_seconds",
        }
        for legacy, modern in legacy_ttr_map.items():
            if legacy in payload and payload.get(modern) in ("", None):
                raw = payload.pop(legacy)
                if raw in ("", None):
                    payload[modern] = None
                elif raw == "inf":
                    payload[modern] = None
                else:
                    payload[modern] = float(raw)
        if payload.get("fault_classification_correct") in ("", None):
            payload["fault_classification_correct"] = None
        else:
            payload["fault_classification_correct"] = str(payload["fault_classification_correct"]).lower() in {
                "1",
                "true",
                "yes",
            }
        if payload.get("seed") in ("", None):
            payload["seed"] = None
        else:
            payload["seed"] = int(float(payload["seed"]))
        payload["injected_fault_type"] = FaultType(str(payload["injected_fault_type"]))
        payload["fault_intensity"] = float(payload.get("fault_intensity") or 0.0)
        payload["total_wall_seconds"] = float(payload.get("total_wall_seconds") or 0.0)
        payload["total_sim_steps"] = int(float(payload.get("total_sim_steps") or 0))
        payload["llm_total_tokens"] = int(float(payload.get("llm_total_tokens") or 0))
        payload["llm_prompt_tokens"] = int(float(payload.get("llm_prompt_tokens") or 0))
        payload["llm_completion_tokens"] = int(float(payload.get("llm_completion_tokens") or 0))
        payload["replan_count"] = int(float(payload.get("replan_count") or 0))
        payload["safety_violation_count"] = int(float(payload.get("safety_violation_count") or 0))
        payload["violent_collision_count"] = int(float(payload.get("violent_collision_count") or 0))
        payload["out_of_workspace_count"] = int(float(payload.get("out_of_workspace_count") or 0))
        payload["fault_occurred"] = str(payload.get("fault_occurred")).lower() in {"1", "true", "yes"}
        payload["success"] = str(payload.get("success")).lower() in {"1", "true", "yes"}
        payload["crashed"] = str(payload.get("crashed")).lower() in {"1", "true", "yes"}
        payload["timed_out"] = str(payload.get("timed_out")).lower() in {"1", "true", "yes"}
        payload.setdefault("experiment_id", "")
        payload.setdefault("failure_reason", "")
        payload.setdefault("fault_trigger", "")
        payload.setdefault("fault_duration", "")
        payload.setdefault("model_identifier", "")
        return cls(**payload)
