"""
Domain entities for Alpha.

This module is the innermost layer of the codebase: plain dataclasses with
zero external dependencies (no pybullet, no pydantic, no LangGraph). Every
other layer (clients, services, schemas) is allowed to import from here;
this module must never import from them. That's what keeps the domain model
stable while the physics engine, the agent framework, or the persistence
format change underneath it across the 12 weeks.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


Vec3 = tuple[float, float, float]


class BlockColor(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclass(frozen=True)
class BlockSpec:
    """A block's identity and where it starts. Immutable — spawning a block
    doesn't mutate this, it produces a BlockState."""
    color: BlockColor
    spawn_position: Vec3


@dataclass
class BlockState:
    """A block's live pose during/after simulation."""
    color: BlockColor
    position: Vec3
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class TaskSpec:
    """What the arm is being asked to do. In Week 1 this is hardcoded
    (move `block` to `place_zone`); from Week 2 onward this is what the
    agent planner receives as its natural-language-derived goal."""
    block: BlockColor
    place_zone: Vec3
    description: str = "pick block and move it to the place zone"


class FaultType(str, Enum):
    """Fault taxonomy. Fixed here in Week 1 even though the injector
    (services/fault_service.py) isn't implemented until Weeks 3-4, so the
    taxonomy is a first-class, citable part of the domain model rather than
    a string someone invents ad hoc later."""
    NONE = "none"
    SENSOR_LAG = "sensor_lag"
    GRIP_SLIP = "grip_slip"
    UNREACHABLE_IK = "unreachable_ik"
    PLANNER_OUTPUT_CORRUPTION = "planner_output_corruption"
    TARGET_CORRUPTION = "target_corruption"
    POSITION_OFFSET = "position_offset"
    GRIP_FAILURE = "grip_failure"
    RELEASE_FAILURE = "release_failure"


class ActionType(str, Enum):
    """High-level body actions produced by the Brain and executed by ControlService."""
    MOVE_TO = "MOVE_TO"
    GRIP = "GRIP"
    RELEASE = "RELEASE"


@dataclass
class EpisodeResult:
    """Outcome of one run of the task, with or without faults injected.
    This is the record metrics_service (Week 5) will aggregate across
    50+ episodes — the shape is fixed now so nothing downstream needs to
    change when faults/agent logic are added, only how this gets populated.
    """
    task: TaskSpec
    success: bool
    final_error: float
    final_position: Vec3
    fault_type: FaultType = FaultType.NONE
    steps_taken: int = 0
    time_to_recovery_steps: Optional[int] = None  # reserved; use post_fault_completion_time in telemetry
    notes: str = ""
    agent_name: str = "scripted"
    action_plan_json: str = ""
    execution_log_json: str = ""
    fault_occurred: bool = False
    fault_seed: Optional[int] = None
