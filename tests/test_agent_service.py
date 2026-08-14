import pytest
from pydantic import ValidationError

from alpha.clients.llm_client import MockLLMClient
from alpha.config import settings
from alpha.core.entities import BlockColor, FaultType, TaskSpec
from alpha.schemas.action_plan import ActionPlan
from alpha.services.agent_service import AgentBrain, AgentService
from alpha.services.control_service import ControlService
from tests.fakes import FakeSimulator


def test_mock_brain_generates_valid_action_plan():
    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)
    brain = AgentBrain(MockLLMClient())

    plan = brain.plan(task)

    assert plan.agent_name == "mock"
    assert len(plan.actions) == 7
    assert plan.actions[0].target is not None


def test_invalid_brain_output_is_rejected_by_schema_validation():
    with pytest.raises(ValidationError):
        ActionPlan.validate_raw_plan([{"action": "MOVE_TO"}])


def test_agent_executes_valid_plan_through_control_service():
    simulator = FakeSimulator()
    control_service = ControlService(simulator)
    agent_service = AgentService(AgentBrain(MockLLMClient()), control_service, simulator)
    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)

    result = agent_service.run_task(task)

    assert result.agent_name == "mock"
    assert result.fault_type == FaultType.NONE
    assert result.steps_taken == 7
    assert len(simulator.moves) == 5
    assert result.action_plan_json
    assert result.execution_log_json


def test_agent_run_completes_without_external_llm_api_key():
    simulator = FakeSimulator()
    agent_service = AgentService(AgentBrain(MockLLMClient()), ControlService(simulator), simulator)
    task = TaskSpec(block=BlockColor.GREEN, place_zone=settings.PLACE_ZONE_POSITION)

    result = agent_service.run_task(task)

    assert result.agent_name == "mock"
    assert isinstance(result.success, bool)
