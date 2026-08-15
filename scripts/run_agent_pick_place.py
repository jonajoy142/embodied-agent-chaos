"""
CLI entrypoint for Week 2 agent pick/place and Weeks 3-4 fault injection.

Usage:
    python scripts/run_agent_pick_place.py --block red
    python scripts/run_agent_pick_place.py --block red --fault position_offset
    python scripts/run_agent_pick_place.py --block red --fault grip_failure --gui
"""

import argparse

from alpha.clients.llm_client import build_llm_client
from alpha.config import settings
from alpha.core.entities import BlockColor, FaultType, TaskSpec
from alpha.models.agent_config import AgentConfig
from alpha.models.fault_config import FaultConfig
from alpha.repos.episode_repository import EpisodeRepository
from alpha.services.agent_service import AgentBrain, AgentService
from alpha.services.control_service import ControlService
from alpha.services.fault_service import ScriptedFaultInjector
from alpha.services.scene_service import SceneService
from alpha.clients.simulator_client import PyBulletSimulatorClient


FAULT_CHOICES = {
    "none": FaultType.NONE,
    "target_corruption": FaultType.TARGET_CORRUPTION,
    "position_offset": FaultType.POSITION_OFFSET,
    "grip_failure": FaultType.GRIP_FAILURE,
    "release_failure": FaultType.RELEASE_FAILURE,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 2 agent pick/place with optional Weeks 3-4 faults.")
    parser.add_argument("--block", choices=[c.value for c in BlockColor], default="red")
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI window.")
    parser.add_argument("--agent", choices=["mock", "openai", "ollama"], default="mock")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fault", choices=list(FAULT_CHOICES), default="none")
    parser.add_argument("--fault-magnitude", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    block = BlockColor(args.block)
    task = TaskSpec(block=block, place_zone=settings.PLACE_ZONE_POSITION)

    fault_type = FAULT_CHOICES[args.fault]
    fault_config = FaultConfig(
        fault_type=fault_type,
        enabled=fault_type != FaultType.NONE,
        magnitude=args.fault_magnitude,
        seed=args.seed,
    )
    fault_injector = ScriptedFaultInjector(fault_config) if fault_type != FaultType.NONE else None

    simulator = PyBulletSimulatorClient()
    scene_service = SceneService(simulator)
    scene_service.build_and_settle(gui=args.gui)

    control_service = ControlService(simulator, fault_injector=fault_injector)
    llm_client = build_llm_client(AgentConfig(provider=args.agent, model_name=args.model))
    brain = AgentBrain(llm_client)
    agent_service = AgentService(brain, control_service, simulator)
    repo = EpisodeRepository()

    scene_info = {"blocks": {color: state.position for color, state in scene_service.block_states().items()}}
    result = agent_service.run_task(task, scene_info=scene_info)
    simulator.disconnect()

    repo.save(result)

    print("Week 2 agent pick/place result:")
    print(f"  block: {result.task.block.value}")
    print(f"  success: {result.success}")
    print(f"  final_error: {result.final_error}")
    print(f"  final_position: {result.final_position}")
    print(f"  actions: {result.steps_taken}")
    print(f"  agent: {result.agent_name}")
    print(f"  fault: {result.fault_type.value}")
    print(f"Saved to {settings.EPISODES_CSV}")


if __name__ == "__main__":
    main()
