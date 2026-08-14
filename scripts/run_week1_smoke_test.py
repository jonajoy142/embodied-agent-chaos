"""
CLI entrypoint for Week 1's acceptance check.

Usage:
    python scripts/run_week1_smoke_test.py
"""

from alpha.clients.simulator_client import PyBulletSimulatorClient
from alpha.services.scene_service import SceneService


def main() -> None:
    simulator = PyBulletSimulatorClient()
    scene_service = SceneService(simulator)
    report = scene_service.smoke_test(gui=False)

    print("Week 1 smoke test result:")
    for color, info in report.items():
        print(f"  {color}: position={info['position']}, drift_from_spawn={info['drift_from_spawn']:.4f}")
    print("Scene built and settled without error.")


if __name__ == "__main__":
    main()
