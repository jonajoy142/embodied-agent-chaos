"""
I/O schema for EpisodeResult.

Why this exists separately from core.entities.EpisodeResult: the core
entity is the internal domain shape services pass around in memory: this
schema is the serialization boundary — what actually gets written to CSV,
returned by the future dashboard API, or logged. Keeping them separate
means the internal shape can evolve without silently changing a stored
data format, and vice versa.
"""

from pydantic import BaseModel

from alpha.core.entities import EpisodeResult, FaultType


class EpisodeResultSchema(BaseModel):
    task_description: str
    block: str
    success: bool
    final_error: float
    final_position: tuple[float, float, float]
    fault_type: FaultType = FaultType.NONE
    steps_taken: int = 0
    time_to_recovery_steps: int | None = None
    notes: str = ""
    agent_name: str = "scripted"
    action_plan_json: str = ""
    execution_log_json: str = ""
    fault_occurred: bool = False
    fault_seed: int | None = None

    @classmethod
    def from_entity(cls, result: EpisodeResult) -> "EpisodeResultSchema":
        return cls(
            task_description=result.task.description,
            block=result.task.block.value,
            success=result.success,
            final_error=result.final_error,
            final_position=result.final_position,
            fault_type=result.fault_type,
            steps_taken=result.steps_taken,
            time_to_recovery_steps=result.time_to_recovery_steps,
            notes=result.notes,
            agent_name=result.agent_name,
            action_plan_json=result.action_plan_json,
            execution_log_json=result.execution_log_json,
            fault_occurred=result.fault_occurred,
            fault_seed=result.fault_seed,
        )
