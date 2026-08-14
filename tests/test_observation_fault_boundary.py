"""Tests that F1 observation faults reach the planner boundary."""

from __future__ import annotations

from typing import Any

import pytest

from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper
from alpha.config import settings
from alpha.core.entities import BlockColor, FaultType, TaskSpec
from alpha.models.chaos_config import ChaosConfig, FaultIntensity, FaultSpec, TriggerConfig, TriggerKind
from alpha.services.agent_service import AgentBrain, AgentService
from alpha.services.control_service import ControlService
from tests.fakes import FakeSimulator, RecordingLLMClient


def test_sensor_lag_reaches_planner_via_openai_style_client():
    """
    Formal-study path: observation dict is wrapped by chaos middleware, then
    passed to an LLM client that consumes scene_info (unlike MockLLMClient).
    """
    config = ChaosConfig(
        seed=1,
        sim_timestep=1.0,
        faults=(
            FaultSpec(
                fault_type=FaultType.SENSOR_LAG,
                enabled=True,
                trigger=TriggerConfig(kind=TriggerKind.IMMEDIATE),
                intensity=FaultIntensity(lag_seconds=2.0),
            ),
        ),
    )
    sim = FakeSimulator()
    chaos = ChaosRoboticsWrapper(sim, config)
    recorder = RecordingLLMClient()
    brain = AgentBrain(recorder)
    service = AgentService(brain, ControlService(sim), sim)

    fresh = {"blocks": {"red": [0.9, 0.8, 0.7], "green": [0.5, 0.0, 0.68]}}
    chaos.wrap_observation({"blocks": {"red": [0.1, 0.2, 0.3], "green": [0.5, 0.0, 0.68]}})
    chaos.context.elapsed_seconds = 3.0
    delayed = chaos.wrap_observation(fresh)

    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)
    service.plan_task(task, scene_info=delayed)

    assert recorder.last_scene_info is not None
    assert recorder.last_scene_info["blocks"]["red"] == pytest.approx([0.1, 0.2, 0.3])
    assert chaos.event_log()[-1]["fault_applied"] is True


def test_mock_llm_does_not_consume_scene_info_by_design():
    """MockLLMClient is engineering-only and intentionally ignores observations."""
    from alpha.clients.llm_client import MockLLMClient

    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)
    plan = MockLLMClient().generate_action_plan(task, scene_info={"blocks": {"red": [9, 9, 9]}})
    assert plan.actions[0].target == pytest.approx(
        (
            settings.BLOCK_SPAWN_POSITIONS[BlockColor.RED][0],
            settings.BLOCK_SPAWN_POSITIONS[BlockColor.RED][1],
            settings.BLOCK_SPAWN_POSITIONS[BlockColor.RED][2] + 0.15,
        )
    )
