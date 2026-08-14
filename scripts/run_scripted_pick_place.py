"""
CLI entrypoint for Week 1's scripted (non-agentic) pick/place task.

Usage:
    python scripts/run_scripted_pick_place.py --block red
    python scripts/run_scripted_pick_place.py --block green --gui
"""

import argparse

from alpha.clients.simulator_client import PyBulletSimulatorClient
from alpha.core.entities import BlockColor, TaskSpec
from alpha.repos.episode_repository import EpisodeRepository
from alpha.services.control_service import ControlService
from alpha.services.scene_service import SceneService
from alpha.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 1 scripted pick/place.")
    parser.add_argument("--block", choices=[c.value for c in BlockColor], default="red")
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI window.")
    args = parser.parse_args()

    block = BlockColor(args.block)
    task = TaskSpec(block=block, place_zone=settings.PLACE_ZONE_POSITION)

    simulator = PyBulletSimulatorClient()
    scene_service = SceneService(simulator)
    control_service = ControlService(simulator)  # no fault_injector yet — that's Weeks 3-4
    repo = EpisodeRepository()

    scene_service.build_and_settle(gui=args.gui)
    result = control_service.run_scripted_pick_place(task)
    simulator.disconnect()

    repo.save(result)

    print("Week 1 scripted pick/place result:")
    print(f"  block: {result.task.block.value}")
    print(f"  success: {result.success}")
    print(f"  final_error: {result.final_error}")
    print(f"  final_position: {result.final_position}")
    print(f"Saved to {settings.EPISODES_CSV}")


if __name__ == "__main__":
    main()
