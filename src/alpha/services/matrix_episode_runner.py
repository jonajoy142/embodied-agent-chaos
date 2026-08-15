"""
Rewrite of matrix episode runner using ``AgentService.run_task`` hooks.
"""

from __future__ import annotations

import traceback
from typing import Any

from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper, PyBulletChaosRoboticsWrapper
from alpha.clients.experiment_llm_clients import ChaosPlannerLLMClient, RateLimitedLLMClient, TelemetryLLMClient
from alpha.clients.llm_client import LLMClient, build_llm_client
from alpha.clients.safety_monitor import PyBulletSafetyMonitor
from alpha.clients.simulator_client import PyBulletSimulatorClient
from alpha.clients.telemetry_simulator_wrapper import TelemetrySimulatorWrapper
from alpha.config import settings
from alpha.core.entities import ActionType, BlockColor, FaultType, TaskSpec
from alpha.core.interfaces import SimulatorClient
from alpha.models.agent_config import AgentConfig
from alpha.models.chaos_config import ChaosConfig
from alpha.repos.episode_repository import EpisodeRepository
from alpha.repos.experiment_data_repository import ExperimentDataRepository
from alpha.schemas.telemetry import ExperimentTelemetryRecord
from alpha.services.agent_hooks import AgentRunHooks
from alpha.services.agent_service import AgentBrain, AgentService
from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger
from alpha.services.control_service import ControlService
from alpha.services.experiment_matrix import (
    CONCURRENT_FAULT_TYPES,
    IntensityLabel,
    MatrixScenario,
    build_chaos_config,
    configure_pybullet_determinism,
    seed_deterministic_environment,
)
from alpha.services.scene_service import SceneService


def execute_matrix_episode(
    scenario: MatrixScenario,
    *,
    experiment_id: str = "dev",
    telemetry: ChaosTelemetryLogger,
    episode_repo: EpisodeRepository,
    data_repo: ExperimentDataRepository | None = None,
    agent_config: AgentConfig,
    block: BlockColor = BlockColor.RED,
    llm_min_interval: float = 0.5,
    master_seed: int | None = None,
    model_identifier: str = "",
) -> tuple[Any | None, ExperimentTelemetryRecord]:
    """Run one matrix scenario through the canonical ``AgentService`` path."""
    seed_deterministic_environment(master_seed or scenario.master_seed)
    episode_seed = (master_seed or scenario.master_seed) + (abs(hash(scenario.run_id)) % 10_000)

    inner = PyBulletSimulatorClient()
    chaos_wrapper: ChaosRoboticsWrapper | None = None
    chaos_config: ChaosConfig | None = None

    safety_monitor = PyBulletSafetyMonitor()

    if scenario.scenario_group == "control":
        # Use minimal chaos wrapper for control to enable safety monitor grip tracking
        chaos_wrapper = ChaosRoboticsWrapper(inner, ChaosConfig(), safety_monitor)
        simulator = chaos_wrapper
    else:
        if scenario.concurrent:
            chaos_config = build_chaos_config(
                fault_types=CONCURRENT_FAULT_TYPES,
                intensity_label=IntensityLabel(scenario.intensity_label),
                episode_seed=episode_seed,
                concurrent=True,
            )
        else:
            chaos_config = build_chaos_config(
                fault_types=(FaultType(scenario.fault_type),),
                intensity_label=IntensityLabel(scenario.intensity_label),
                episode_seed=episode_seed,
                concurrent=False,
            )
        chaos_wrapper = PyBulletChaosRoboticsWrapper(inner, chaos_config, safety_monitor)
        simulator = chaos_wrapper

    instrumented = TelemetrySimulatorWrapper(simulator, telemetry, safety_monitor)

    scene_service = SceneService(instrumented)
    instrumented.connect(gui=False)
    configure_pybullet_determinism()
    scene_service.build_and_settle(gui=False)

    llm_client: LLMClient = RateLimitedLLMClient(
        build_llm_client(agent_config),
        min_interval_seconds=llm_min_interval,
    )
    llm_client = TelemetryLLMClient(llm_client, telemetry)
    if chaos_wrapper is not None and _needs_planner_wrapper(scenario, chaos_config):
        llm_client = ChaosPlannerLLMClient(llm_client, chaos_wrapper)

    brain = AgentBrain(llm_client)
    control_service = ControlService(instrumented)
    agent_service = AgentService(brain, control_service, instrumented)

    task = TaskSpec(block=block, place_zone=settings.PLACE_ZONE_POSITION)
    scene_info = {
        "blocks": {
            color_key: state.position for color_key, state in scene_service.block_states().items()
        }
    }
    if chaos_wrapper is not None:
        scene_info = chaos_wrapper.wrap_observation(scene_info)

    injected_fault = _injected_fault_type(scenario)
    fault_trigger = _fault_trigger_label(chaos_config)

    telemetry.begin_episode(
        episode_id=scenario.run_id,
        experiment_id=experiment_id,
        agent_name=brain.name,
        block=block.value,
        injected_fault_type=injected_fault,
        fault_intensity=scenario.intensity_value,
        seed=episode_seed,
        model_identifier=model_identifier or agent_config.model_name,
        fault_trigger=fault_trigger,
    )

    hooks = _build_hooks(chaos_wrapper, episode_seed)

    result = None
    crashed = False
    failure_reason = ""
    try:
        result = agent_service.run_task(task, scene_info, hooks=hooks)
    except Exception as exc:
        crashed = True
        failure_reason = _format_exception_failure(exc)
    finally:
        if chaos_wrapper is not None:
            telemetry.ingest_fault_events(chaos_wrapper.event_log())
        if result is not None and not result.success and not failure_reason:
            failure_reason = "task_failed"
        record = telemetry.finalize_episode(
            success=bool(result and result.success),
            crashed=crashed,
            episode_result=result,
            failure_reason=failure_reason,
        )
        if result is not None:
            episode_repo.save(result)
        if data_repo is not None:
            data_repo.append_raw_episode(
                experiment_id=experiment_id,
                episode_id=scenario.run_id,
                record=record,
                execution_log_json=result.execution_log_json if result else "",
                action_plan_json=result.action_plan_json if result else "",
            )
        instrumented.disconnect()

    return result, record


def _format_exception_failure(exc: Exception) -> str:
    summary = f"exception:{type(exc).__name__}"
    message = str(exc).strip().replace("\n", " ")
    if message:
        summary = f"{summary}: {message}"

    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        frame = frames[-1]
        summary = f"{summary} @ {frame.filename}:{frame.lineno}"

    return summary[:1000]


def _build_hooks(chaos_wrapper: ChaosRoboticsWrapper | None, fault_seed: int) -> AgentRunHooks | None:
    if chaos_wrapper is None:
        return None

    def on_before_action(index: int, step, grip_handle) -> None:
        if step.action == ActionType.GRIP:
            chaos_wrapper.set_phase("pick")
        elif step.action == ActionType.RELEASE:
            chaos_wrapper.set_phase("place")
        chaos_wrapper.advance_event_step(index)

    def on_after_action(index: int, step, grip_handle) -> None:
        # After GRIP succeeds, set lift phase for subsequent MOVE_TO actions
        if step.action == ActionType.GRIP and grip_handle is not None:
            chaos_wrapper.set_phase("lift")

    return AgentRunHooks(
        on_before_action=on_before_action,
        on_after_action=on_after_action,
        fault_events_provider=chaos_wrapper.event_log,
        fault_type=_fault_type_from_wrapper(chaos_wrapper),
        fault_seed=fault_seed,
    )


def _injected_fault_type(scenario: MatrixScenario) -> FaultType:
    if scenario.scenario_group == "control":
        return FaultType.NONE
    if scenario.concurrent:
        return FaultType.SENSOR_LAG
    return FaultType(scenario.fault_type.split("+")[0])


def _fault_type_from_wrapper(chaos_wrapper: ChaosRoboticsWrapper) -> FaultType:
    enabled = chaos_wrapper.config.enabled_faults()
    if not enabled:
        return FaultType.NONE
    return enabled[0].fault_type


def _fault_trigger_label(chaos_config: ChaosConfig | None) -> str:
    if chaos_config is None or not chaos_config.faults:
        return "none"
    trigger = chaos_config.faults[0].trigger
    if trigger.phase:
        return f"event:{trigger.phase}"
    return trigger.kind.value


def _needs_planner_wrapper(scenario: MatrixScenario, chaos_config: ChaosConfig | None) -> bool:
    if scenario.concurrent:
        return False
    if scenario.fault_type == FaultType.PLANNER_OUTPUT_CORRUPTION.value:
        return True
    if chaos_config is None:
        return False
    return any(spec.fault_type == FaultType.PLANNER_OUTPUT_CORRUPTION for spec in chaos_config.faults)
