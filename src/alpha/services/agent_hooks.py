"""Optional hooks for ``AgentService.run_task`` experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from alpha.core.entities import ActionType, FaultType
from alpha.schemas.action_plan import ActionStep


@dataclass(frozen=True)
class AgentRunHooks:
    """
    Experiment-layer callbacks that keep fault injection and phase tagging
    outside ``AgentService`` while using a single execution code path.
    """

    on_before_action: Callable[[int, ActionStep, Any | None], None] | None = None
    fault_events_provider: Callable[[], list[dict[str, Any]]] | None = None
    fault_type: FaultType = FaultType.NONE
    fault_seed: int | None = None
