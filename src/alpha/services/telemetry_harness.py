"""
Orchestration helper for running agent episodes with Week 5 telemetry.

Keeps ``AgentService`` internals untouched — callers pass an
already-constructed ``AgentService`` whose simulator may be wrapped with
``TelemetrySimulatorWrapper``.
"""

from __future__ import annotations

from typing import Any, Callable

from alpha.core.entities import EpisodeResult, FaultType, TaskSpec
from alpha.schemas.telemetry import ExperimentTelemetryRecord
from alpha.services.agent_service import AgentService
from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger


def run_instrumented_episode(
    agent_service: AgentService,
    task: TaskSpec,
    telemetry: ChaosTelemetryLogger,
    *,
    scene_info: dict[str, Any] | None = None,
    injected_fault_type: FaultType = FaultType.NONE,
    fault_intensity: float = 0.0,
    fault_events_provider: Callable[[], list[dict[str, Any]]] | None = None,
    agent_diagnosis: str | None = None,
    llm_prompt_tokens: int = 0,
    llm_completion_tokens: int = 0,
    replan_count: int = 0,
) -> tuple[EpisodeResult, ExperimentTelemetryRecord]:
    """
    Run one agent episode and finalize telemetry metrics.

    ``fault_events_provider`` should return chaos / fault injector events, e.g.
    ``lambda: chaos_wrapper.event_log()`` or
    ``lambda: control_service.fault_injector.event_log()``.
    """
    telemetry.begin_episode(
        agent_name=getattr(agent_service.brain, "name", "agent"),
        block=task.block.value,
        injected_fault_type=injected_fault_type,
        fault_intensity=fault_intensity,
    )

    if replan_count:
        for _ in range(replan_count):
            telemetry.record_replan(
                llm_prompt_tokens=llm_prompt_tokens // max(replan_count, 1),
                llm_completion_tokens=llm_completion_tokens // max(replan_count, 1),
            )
    elif llm_prompt_tokens or llm_completion_tokens:
        telemetry.record_llm_usage(
            prompt_tokens=llm_prompt_tokens,
            completion_tokens=llm_completion_tokens,
        )

    if agent_diagnosis:
        telemetry.record_agent_diagnosis(agent_diagnosis)

    crashed = False
    try:
        result = agent_service.run_task(task, scene_info=scene_info)
    except Exception:
        crashed = True
        if fault_events_provider is not None:
            telemetry.ingest_fault_events(fault_events_provider())
        record = telemetry.finalize_episode(
            success=False,
            crashed=True,
        )
        raise
    else:
        if fault_events_provider is not None:
            telemetry.ingest_fault_events(fault_events_provider())
        record = telemetry.finalize_episode(
            success=result.success,
            episode_result=result,
        )
        return result, record
