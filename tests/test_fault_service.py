from alpha.config import settings
from alpha.core.entities import BlockColor, FaultType, TaskSpec
from alpha.models.fault_config import FaultConfig
from alpha.services.agent_service import AgentBrain, AgentService
from alpha.services.control_service import ControlService
from alpha.services.fault_service import ScriptedFaultInjector
from alpha.clients.llm_client import MockLLMClient
from tests.fakes import FakeSimulator


def test_fault_type_none_behaves_like_controller_without_injector():
    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)

    plain_sim = FakeSimulator()
    plain_result = AgentService(AgentBrain(MockLLMClient()), ControlService(plain_sim), plain_sim).run_task(task)

    none_sim = FakeSimulator()
    none_injector = ScriptedFaultInjector(FaultConfig(fault_type=FaultType.NONE, enabled=False))
    none_result = AgentService(
        AgentBrain(MockLLMClient()),
        ControlService(none_sim, fault_injector=none_injector),
        none_sim,
    ).run_task(task)

    assert none_sim.moves == plain_sim.moves
    assert none_result.final_position == plain_result.final_position
    assert none_result.fault_type == FaultType.NONE
    assert none_result.fault_occurred is False


def test_position_offset_modifies_target_deterministically():
    config = FaultConfig(fault_type=FaultType.POSITION_OFFSET, enabled=True, magnitude=0.05, seed=7)
    injector_a = ScriptedFaultInjector(config)
    injector_b = ScriptedFaultInjector(config)
    target = (0.5, 0.3, 0.7)

    assert injector_a.maybe_corrupt_target(target) == (0.55, 0.25, 0.7)
    assert injector_b.maybe_corrupt_target(target) == (0.55, 0.25, 0.7)
    assert injector_a.event_log()[0]["requested_target"] == target


def test_grip_failure_is_injected_deterministically():
    simulator = FakeSimulator()
    injector = ScriptedFaultInjector(FaultConfig(fault_type=FaultType.GRIP_FAILURE, enabled=True, seed=42))
    control_service = ControlService(simulator, fault_injector=injector)

    handle = control_service.grip(BlockColor.RED)

    assert handle is None
    assert simulator.gripped is None
    assert injector.event_log()[0]["fault_applied"] is True


def test_release_failure_is_injected_deterministically():
    simulator = FakeSimulator()
    injector = ScriptedFaultInjector(FaultConfig(fault_type=FaultType.RELEASE_FAILURE, enabled=True, seed=42))
    control_service = ControlService(simulator, fault_injector=injector)
    handle = control_service.grip(BlockColor.RED)

    control_service.release(handle)

    assert simulator.released is False
    assert simulator.gripped == BlockColor.RED
    assert injector.event_log()[-1]["fault_applied"] is True


def test_fault_seed_reproducibility_with_probability():
    config = FaultConfig(
        fault_type=FaultType.POSITION_OFFSET,
        enabled=True,
        probability=0.5,
        magnitude=0.05,
        seed=123,
    )
    target = (0.5, 0.3, 0.7)
    sequence_a = [ScriptedFaultInjector(config).maybe_corrupt_target(target) for _ in range(3)]
    sequence_b = [ScriptedFaultInjector(config).maybe_corrupt_target(target) for _ in range(3)]

    assert sequence_a == sequence_b


def test_episode_result_records_injected_fault():
    simulator = FakeSimulator()
    injector = ScriptedFaultInjector(FaultConfig(fault_type=FaultType.GRIP_FAILURE, enabled=True, seed=42))
    service = AgentService(
        AgentBrain(MockLLMClient()),
        ControlService(simulator, fault_injector=injector),
        simulator,
    )
    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)

    result = service.run_task(task)

    assert result.fault_type == FaultType.GRIP_FAILURE
    assert result.fault_occurred is True
    assert result.fault_seed == 42
    assert "fault_events" in result.execution_log_json
