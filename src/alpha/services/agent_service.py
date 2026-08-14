"""
Week 2 Brain/Body orchestration.

The Brain produces a validated ActionPlan. The Body is still ControlService;
this service only coordinates planning, validation, execution, and result
recording.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from alpha.clients.llm_client import LLMClient
from alpha.core.entities import ActionType, EpisodeResult, FaultType, TaskSpec
from alpha.core.interfaces import AgentPlanner, SimulatorClient
from alpha.schemas.action_plan import ActionPlan
from alpha.services.agent_hooks import AgentRunHooks
from alpha.services.control_service import ControlService


class AgentBrain(AgentPlanner):
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    @property
    def name(self) -> str:
        return self.llm_client.name

    def plan(self, task: TaskSpec, scene_info: dict | None = None) -> ActionPlan:
        return self.llm_client.generate_action_plan(task, scene_info)


class AgentService:
    def __init__(
        self,
        brain: AgentBrain,
        control_service: ControlService,
        simulator: SimulatorClient,
    ) -> None:
        self.brain = brain
        self.control_service = control_service
        self.simulator = simulator

    def plan_task(self, task: TaskSpec, scene_info: dict[str, Any] | None = None) -> ActionPlan:
        plan = self.brain.plan(task, scene_info)
        return ActionPlan.validate_raw_plan(plan, agent_name=self.brain.name)

    def run_task(
        self,
        task: TaskSpec,
        scene_info: dict[str, Any] | None = None,
        hooks: AgentRunHooks | None = None,
    ) -> EpisodeResult:
        try:
            plan = self.plan_task(task, scene_info)
        except ValidationError:
            raise

        execution_log: list[dict[str, Any]] = []
        grip_handle = None

        for index, step in enumerate(plan.actions):
            if hooks is not None and hooks.on_before_action is not None:
                hooks.on_before_action(index, step, grip_handle)

            if step.action == ActionType.MOVE_TO:
                assert step.target is not None
                move_result = self.control_service.move_to(step.target)
                execution_log.append(
                    {
                        "index": index,
                        "action": step.action.value,
                        "requested_target": move_result.get("requested_target", step.target),
                        "executed_target": move_result.get("executed_target", step.target),
                        "reached_pos": move_result.get("reached_pos"),
                        "error": move_result.get("error"),
                    }
                )
            elif step.action == ActionType.GRIP:
                assert step.block is not None
                grip_handle = self.control_service.grip(step.block)
                execution_log.append(
                    {
                        "index": index,
                        "action": step.action.value,
                        "block": step.block.value,
                        "grip_acquired": grip_handle is not None,
                    }
                )
            elif step.action == ActionType.RELEASE:
                self.control_service.release(grip_handle)
                execution_log.append({"index": index, "action": step.action.value})

        self.simulator.step(60)
        success, final_error, final_position = self.control_service.evaluate_task(task)

        fault_injector = self.control_service.fault_injector
        if hooks is not None and hooks.fault_events_provider is not None:
            fault_log = hooks.fault_events_provider()
            fault_type = hooks.fault_type
            fault_seed = hooks.fault_seed
        elif fault_injector is not None:
            fault_log = fault_injector.event_log()
            fault_type = fault_injector.fault_type
            fault_seed = fault_injector.seed
        else:
            fault_log = []
            fault_type = FaultType.NONE
            fault_seed = None
        fault_occurred = any(event.get("fault_applied") for event in fault_log)

        return EpisodeResult(
            task=task,
            success=success,
            final_error=final_error,
            final_position=final_position,
            fault_type=fault_type,
            steps_taken=len(plan.actions),
            agent_name=plan.agent_name,
            action_plan_json=json.dumps(plan.model_dump(mode="json")),
            execution_log_json=json.dumps(
                {
                    "actions": execution_log,
                    "fault_events": fault_log,
                },
                default=str,
            ),
            fault_occurred=fault_occurred,
            fault_seed=fault_seed,
        )
