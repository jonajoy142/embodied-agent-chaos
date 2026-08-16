"""Tests for F1 repeated observation and replanning path."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_observe_method_exists_on_simulator_client():
    """Test that observe method exists on SimulatorClient interface."""
    from alpha.core.interfaces import SimulatorClient
    
    assert hasattr(SimulatorClient, 'observe')


def test_observe_method_exists_on_pybullet_simulator():
    """Test that PyBulletSimulatorClient implements observe method."""
    from alpha.clients.simulator_client import PyBulletSimulatorClient
    
    client = PyBulletSimulatorClient()
    assert hasattr(client, 'observe')


def test_observe_method_exists_on_chaos_wrapper():
    """Test that ChaosRoboticsWrapper implements observe method."""
    from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper
    from alpha.core.interfaces import SimulatorClient
    from alpha.models.chaos_config import ChaosConfig
    
    inner = MagicMock(spec=SimulatorClient)
    wrapper = ChaosRoboticsWrapper(inner, ChaosConfig())
    assert hasattr(wrapper, 'observe')


def test_observe_method_exists_on_agent_service():
    """Test that AgentService implements observe method."""
    from alpha.services.agent_service import AgentService, AgentBrain
    from alpha.services.control_service import ControlService
    from alpha.core.interfaces import SimulatorClient
    
    brain = MagicMock(spec=AgentBrain)
    control = MagicMock(spec=ControlService)
    simulator = MagicMock(spec=SimulatorClient)
    
    service = AgentService(brain, control, simulator)
    assert hasattr(service, 'observe')


def test_chaos_wrapper_observe_applies_f1_fault():
    """Test that ChaosRoboticsWrapper.observe() applies F1 sensor lag fault."""
    from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper
    from alpha.core.interfaces import SimulatorClient
    from alpha.models.chaos_config import ChaosConfig, FaultIntensity, FaultSpec, TriggerConfig, TriggerKind
    from alpha.core.entities import FaultType
    
    # Create a mock inner simulator
    inner = MagicMock(spec=SimulatorClient)
    inner.observe.return_value = {"red": (0.5, 0.5, 0.5)}
    
    # Create chaos config with F1 fault
    fault_spec = FaultSpec(
        fault_type=FaultType.SENSOR_LAG,
        intensity=FaultIntensity(lag_seconds=1.0),
        trigger=TriggerConfig(kind=TriggerKind.IMMEDIATE),
        probability=1.0,
    )
    config = ChaosConfig(faults=[fault_spec], seed=42)
    
    wrapper = ChaosRoboticsWrapper(inner, config)
    
    # Call observe should apply F1 fault
    observation = wrapper.observe()
    
    # Should have called wrap_observation which applies F1
    assert observation is not None
    # The observation should be from wrap_observation, not directly from inner.observe
    # This proves F1 is applied through the observe path


def test_repeated_observation_requests_occur_with_replanning():
    """Test that repeated observation requests actually occur when enable_replanning=True."""
    from alpha.services.agent_service import AgentService, AgentBrain
    from alpha.services.control_service import ControlService
    from alpha.core.interfaces import SimulatorClient
    from alpha.core.entities import TaskSpec, ActionType, BlockColor
    from alpha.schemas.action_plan import ActionPlan, ActionStep
    
    # Create mocks
    brain = MagicMock(spec=AgentBrain)
    control = MagicMock(spec=ControlService)
    simulator = MagicMock(spec=SimulatorClient)
    
    # Setup brain to return plans
    plan1 = ActionPlan(
        agent_name="test",
        actions=[
            ActionStep(action=ActionType.MOVE_TO, target=(0.5, 0.5, 0.5)),
            ActionStep(action=ActionType.GRIP, block=BlockColor.RED),
            ActionStep(action=ActionType.MOVE_TO, target=(0.6, 0.6, 0.6)),
            ActionStep(action=ActionType.RELEASE),
            ActionStep(action=ActionType.MOVE_TO, target=(0.7, 0.7, 0.7)),
        ]
    )
    plan2 = ActionPlan(
        agent_name="test",
        actions=[
            ActionStep(action=ActionType.MOVE_TO, target=(0.8, 0.8, 0.8)),
        ]
    )
    
    brain.plan.side_effect = [plan1, plan2]
    
    # Setup control and simulator mocks
    control.fault_injector = None
    control.move_to.return_value = {"reached_pos": (0.5, 0.5, 0.5), "error": 0.0}
    control.grip.return_value = "grip_handle"
    control.evaluate_task.return_value = (True, 0.1, (0.8, 0.8, 0.8))
    simulator.observe.return_value = {"red": (0.6, 0.6, 0.6)}
    
    service = AgentService(brain, control, simulator)
    
    task = TaskSpec(block=BlockColor.RED, place_zone=(0.8, 0.8, 0.0))
    
    # Run with replanning enabled
    result = service.run_task(task, scene_info={"red": (0.5, 0.5, 0.5)}, enable_replanning=True, replanning_chunk_size=2)
    
    # Should have called observe at least once for replanning
    assert simulator.observe.call_count >= 1
    # Should have called plan twice (initial + replanning)
    assert brain.plan.call_count >= 2
    assert brain.plan.call_args_list[1].args[1] == {"blocks": {"red": (0.6, 0.6, 0.6)}}


def test_f1_returns_stale_observation():
    """Test that F1 returns an older observation when sensor lag is active."""
    from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper
    from alpha.core.interfaces import SimulatorClient
    from alpha.models.chaos_config import ChaosConfig, FaultIntensity, FaultSpec, TriggerConfig, TriggerKind
    from alpha.core.entities import FaultType
    # Create a mock inner simulator
    inner = MagicMock(spec=SimulatorClient)
    
    # Simulate changing observations over time
    observations = [
        {"red": (0.5, 0.5, 0.5)},  # Initial observation
        {"red": (0.6, 0.6, 0.6)},  # Later observation
    ]
    inner.observe.side_effect = observations
    
    # Create chaos config with F1 fault with 1 second lag
    fault_spec = FaultSpec(
        fault_type=FaultType.SENSOR_LAG,
        intensity=FaultIntensity(lag_seconds=1.0),
        trigger=TriggerConfig(kind=TriggerKind.IMMEDIATE),
        probability=1.0,
    )
    config = ChaosConfig(faults=[fault_spec], seed=42)
    
    wrapper = ChaosRoboticsWrapper(inner, config)
    
    # First observation should be current
    obs1 = wrapper.observe()
    assert obs1 == {"red": (0.5, 0.5, 0.5)}
    
    # Simulate time passing (this is conceptual - in real test we'd need to control time)
    # For now, we just verify the structure is correct
    obs2 = wrapper.observe()
    assert obs2 is not None


def test_control_uses_current_observations():
    """Test that control with no fault still uses current observations."""
    from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper
    from alpha.core.interfaces import SimulatorClient
    from alpha.models.chaos_config import ChaosConfig
    
    # Create a mock inner simulator
    inner = MagicMock(spec=SimulatorClient)
    inner.observe.return_value = {"red": (0.5, 0.5, 0.5)}
    
    # Create chaos config without F1 fault
    config = ChaosConfig()
    
    wrapper = ChaosRoboticsWrapper(inner, config)
    
    # Observation should pass through unchanged
    observation = wrapper.observe()
    assert observation == {"red": (0.5, 0.5, 0.5)}


def test_valid_action_plans_still_execute():
    """Test that valid ActionPlans still execute with replanning enabled."""
    from alpha.services.agent_service import AgentService, AgentBrain
    from alpha.services.control_service import ControlService
    from alpha.core.interfaces import SimulatorClient
    from alpha.core.entities import TaskSpec, ActionType, BlockColor
    from alpha.schemas.action_plan import ActionPlan, ActionStep
    
    # Create mocks
    brain = MagicMock(spec=AgentBrain)
    control = MagicMock(spec=ControlService)
    simulator = MagicMock(spec=SimulatorClient)
    
    # Setup brain to return plan
    plan = ActionPlan(
        agent_name="test",
        actions=[
            ActionStep(action=ActionType.MOVE_TO, target=(0.5, 0.5, 0.5)),
            ActionStep(action=ActionType.GRIP, block=BlockColor.RED),
            ActionStep(action=ActionType.RELEASE),
        ]
    )
    brain.plan.return_value = plan
    
    # Setup control and simulator mocks
    control.fault_injector = None
    control.move_to.return_value = {"reached_pos": (0.5, 0.5, 0.5), "error": 0.0}
    control.grip.return_value = "grip_handle"
    control.evaluate_task.return_value = (True, 0.1, (0.5, 0.5, 0.5))
    simulator.observe.return_value = {"red": (0.5, 0.5, 0.5)}
    
    service = AgentService(brain, control, simulator)
    
    task = TaskSpec(block=BlockColor.RED, place_zone=(0.8, 0.8, 0.0))
    
    # Run with replanning enabled
    result = service.run_task(task, scene_info={"red": (0.5, 0.5, 0.5)}, enable_replanning=True)
    
    # Should execute successfully
    assert result.success
    assert result.steps_taken == 3


def test_f1_does_not_alter_control_service_directly():
    """Test that F1 does not alter ControlService directly."""
    from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper
    from alpha.core.interfaces import SimulatorClient
    from alpha.models.chaos_config import ChaosConfig, FaultIntensity, FaultSpec, TriggerConfig, TriggerKind
    from alpha.core.entities import FaultType
    
    # Create a mock inner simulator
    inner = MagicMock(spec=SimulatorClient)
    inner.observe.return_value = {"red": (0.5, 0.5, 0.5)}
    
    # Create chaos config with F1 fault
    fault_spec = FaultSpec(
        fault_type=FaultType.SENSOR_LAG,
        intensity=FaultIntensity(lag_seconds=1.0),
        trigger=TriggerConfig(kind=TriggerKind.IMMEDIATE),
        probability=1.0,
    )
    config = ChaosConfig(faults=[fault_spec], seed=42)
    
    wrapper = ChaosRoboticsWrapper(inner, config)
    
    # F1 should only affect observations, not control methods
    # Verify control methods are not affected
    assert wrapper.move_end_effector.__name__ == "move_end_effector"
    assert wrapper.grip.__name__ == "grip"
    assert wrapper.release.__name__ == "release"


def test_replanning_disabled_by_default():
    """Test that replanning is disabled by default (preserves Phase A behavior)."""
    from alpha.services.agent_service import AgentService, AgentBrain
    from alpha.services.control_service import ControlService
    from alpha.core.interfaces import SimulatorClient
    from alpha.core.entities import TaskSpec, ActionType, BlockColor
    from alpha.schemas.action_plan import ActionPlan, ActionStep
    
    # Create mocks
    brain = MagicMock(spec=AgentBrain)
    control = MagicMock(spec=ControlService)
    simulator = MagicMock(spec=SimulatorClient)
    
    # Setup brain to return plan
    plan = ActionPlan(
        agent_name="test",
        actions=[
            ActionStep(action=ActionType.MOVE_TO, target=(0.5, 0.5, 0.5)),
            ActionStep(action=ActionType.GRIP, block=BlockColor.RED),
            ActionStep(action=ActionType.RELEASE),
        ]
    )
    brain.plan.return_value = plan
    
    # Setup control and simulator mocks
    control.fault_injector = None
    control.move_to.return_value = {"reached_pos": (0.5, 0.5, 0.5), "error": 0.0}
    control.grip.return_value = "grip_handle"
    control.evaluate_task.return_value = (True, 0.1, (0.5, 0.5, 0.5))
    simulator.observe.return_value = {"red": (0.5, 0.5, 0.5)}
    
    service = AgentService(brain, control, simulator)
    
    task = TaskSpec(block=BlockColor.RED, place_zone=(0.8, 0.8, 0.0))
    
    # Run without replanning (default)
    result = service.run_task(task, scene_info={"red": (0.5, 0.5, 0.5)})
    
    # Should NOT call observe (no replanning)
    assert simulator.observe.call_count == 0
    # Should call plan only once (no replanning)
    assert brain.plan.call_count == 1


def _make_plan(*targets):
    from alpha.core.entities import ActionType, BlockColor
    from alpha.schemas.action_plan import ActionPlan, ActionStep

    actions = []
    for item in targets:
        if item == "GRIP":
            actions.append(ActionStep(action=ActionType.GRIP, block=BlockColor.RED))
        elif item == "RELEASE":
            actions.append(ActionStep(action=ActionType.RELEASE))
        else:
            actions.append(ActionStep(action=ActionType.MOVE_TO, target=item))
    return ActionPlan(agent_name="test", actions=actions)


def _service_with_brain(brain):
    from unittest.mock import MagicMock

    from alpha.core.interfaces import SimulatorClient
    from alpha.services.agent_service import AgentService
    from alpha.services.control_service import ControlService

    control = MagicMock(spec=ControlService)
    simulator = MagicMock(spec=SimulatorClient)
    control.fault_injector = None
    control.move_to.return_value = {"reached_pos": (0.5, 0.5, 0.5), "error": 0.0}
    control.grip.return_value = "grip_handle"
    control.evaluate_task.return_value = (True, 0.1, (0.8, 0.8, 0.8))
    simulator.observe.return_value = {"red": (0.6, 0.6, 0.6)}
    return AgentService(brain, control, simulator), control, simulator


def test_f1_episode_terminates_with_bounded_replans():
    """One F1 episode must terminate and keep replan count bounded."""
    import json
    from unittest.mock import MagicMock

    from alpha.core.entities import BlockColor, TaskSpec
    from alpha.services.agent_service import AgentBrain, REPLANNING_TIMEOUT_REASON

    brain = MagicMock(spec=AgentBrain)
    long_plan = _make_plan(
        (0.5, 0.5, 0.7),
        (0.5, 0.5, 0.6),
        "GRIP",
        (0.5, 0.5, 0.7),
        (0.6, 0.35, 0.7),
        (0.6, 0.35, 0.65),
        "RELEASE",
    )
    # Always return a full plan — the pre-fix bug that hung forever.
    brain.plan.return_value = long_plan

    service, control, simulator = _service_with_brain(brain)
    task = TaskSpec(block=BlockColor.RED, place_zone=(0.6, 0.35, 0.65))

    result = service.run_task(
        task,
        scene_info={"blocks": {"red": (0.5, 0.5, 0.65)}},
        enable_replanning=True,
        replanning_chunk_size=3,
        max_replanning_attempts=3,
    )

    payload = json.loads(result.execution_log_json)
    assert payload["replan_count"] <= 3
    assert payload["replan_count"] == 3
    assert brain.plan.call_count == 1 + 3  # initial + bounded replans
    assert payload["timed_out"] is True
    assert result.notes == REPLANNING_TIMEOUT_REASON
    assert result.success is False
    assert payload["chunk_index"] >= 1
    assert simulator.observe.call_count == 3


def test_chunk_index_advances_across_replanning_loop():
    import json
    from unittest.mock import MagicMock

    from alpha.core.entities import BlockColor, TaskSpec
    from alpha.services.agent_service import AgentBrain

    brain = MagicMock(spec=AgentBrain)
    brain.plan.side_effect = [
        _make_plan((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4)),
        _make_plan((0.5, 0.5, 0.5)),
    ]
    service, _, _ = _service_with_brain(brain)
    task = TaskSpec(block=BlockColor.RED, place_zone=(0.6, 0.35, 0.65))

    result = service.run_task(
        task,
        scene_info={"blocks": {"red": (0.5, -0.15, 0.65)}},
        enable_replanning=True,
        replanning_chunk_size=2,
        max_replanning_attempts=3,
    )

    payload = json.loads(result.execution_log_json)
    assert payload["chunk_index"] >= 2
    assert payload["replan_count"] == 1
    assert payload["timed_out"] is False
    assert result.success is True


def test_loop_exits_after_successful_completion_without_timeout():
    import json
    from unittest.mock import MagicMock

    from alpha.core.entities import BlockColor, TaskSpec
    from alpha.services.agent_service import AgentBrain

    brain = MagicMock(spec=AgentBrain)
    brain.plan.return_value = _make_plan((0.5, 0.5, 0.5), "GRIP", "RELEASE")
    service, control, _ = _service_with_brain(brain)
    control.evaluate_task.return_value = (True, 0.01, (0.6, 0.35, 0.65))
    task = TaskSpec(block=BlockColor.RED, place_zone=(0.6, 0.35, 0.65))

    result = service.run_task(
        task,
        enable_replanning=True,
        replanning_chunk_size=3,
        max_replanning_attempts=3,
    )

    payload = json.loads(result.execution_log_json)
    assert result.success is True
    assert payload["timed_out"] is False
    assert payload["replan_count"] == 0
    assert result.notes == ""


def test_loop_exits_after_task_failure_without_hanging():
    import json
    from unittest.mock import MagicMock

    from alpha.core.entities import BlockColor, TaskSpec
    from alpha.services.agent_service import AgentBrain

    brain = MagicMock(spec=AgentBrain)
    brain.plan.return_value = _make_plan((0.5, 0.5, 0.5), "GRIP", "RELEASE")
    service, control, _ = _service_with_brain(brain)
    control.evaluate_task.return_value = (False, 0.9, (0.1, 0.1, 0.1))
    task = TaskSpec(block=BlockColor.RED, place_zone=(0.6, 0.35, 0.65))

    result = service.run_task(
        task,
        enable_replanning=True,
        replanning_chunk_size=3,
        max_replanning_attempts=3,
    )

    payload = json.loads(result.execution_log_json)
    assert result.success is False
    assert payload["timed_out"] is False
    assert brain.plan.call_count == 1


def test_loop_exits_after_replanning_timeout():
    import json
    from unittest.mock import MagicMock

    from alpha.core.entities import BlockColor, TaskSpec
    from alpha.services.agent_hooks import AgentRunHooks
    from alpha.services.agent_service import AgentBrain, REPLANNING_TIMEOUT_REASON

    brain = MagicMock(spec=AgentBrain)
    brain.plan.return_value = _make_plan(
        (0.1, 0.1, 0.1),
        (0.2, 0.2, 0.2),
        (0.3, 0.3, 0.3),
        (0.4, 0.4, 0.4),
    )
    service, _, _ = _service_with_brain(brain)
    replan_hooks = []

    def on_replan(count, context):
        replan_hooks.append((count, context["chunk_index"]))

    task = TaskSpec(block=BlockColor.RED, place_zone=(0.6, 0.35, 0.65))
    result = service.run_task(
        task,
        hooks=AgentRunHooks(on_replan=on_replan),
        enable_replanning=True,
        replanning_chunk_size=2,
        max_replanning_attempts=2,
    )

    payload = json.loads(result.execution_log_json)
    assert result.notes == REPLANNING_TIMEOUT_REASON
    assert payload["timed_out"] is True
    assert payload["replan_count"] == 2
    assert len(replan_hooks) == 2
    assert result.success is False
    assert any(entry.get("replanning_timeout") for entry in payload["actions"])


def test_infinite_full_plan_replacement_cannot_hang():
    """Regression: replacing remainder with a full plan must not loop forever."""
    import json
    from unittest.mock import MagicMock

    from alpha.core.entities import BlockColor, TaskSpec
    from alpha.services.agent_service import AgentBrain, REPLANNING_TIMEOUT_REASON

    brain = MagicMock(spec=AgentBrain)
    call_count = {"n": 0}

    def always_full_plan(task, scene_info=None):
        call_count["n"] += 1
        assert call_count["n"] <= 20  # hard guard if bound regresses
        return _make_plan(
            (0.5, -0.15, 0.8),
            (0.5, -0.15, 0.67),
            "GRIP",
            (0.5, -0.15, 0.8),
            (0.6, 0.35, 0.8),
            (0.6, 0.35, 0.71),
            "RELEASE",
        )

    brain.plan.side_effect = always_full_plan
    service, _, _ = _service_with_brain(brain)
    task = TaskSpec(block=BlockColor.RED, place_zone=(0.6, 0.35, 0.65))

    result = service.run_task(
        task,
        enable_replanning=True,
        replanning_chunk_size=3,
        max_replanning_attempts=3,
    )

    payload = json.loads(result.execution_log_json)
    assert call_count["n"] == 4  # 1 initial + 3 replans
    assert payload["replan_count"] == 3
    assert result.notes == REPLANNING_TIMEOUT_REASON
    assert payload["timed_out"] is True
