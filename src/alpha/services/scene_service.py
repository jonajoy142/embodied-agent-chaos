"""
Scene orchestration: build the scene, let physics settle, report state.
Depends on core.interfaces.SimulatorClient — not on PyBullet directly —
so it works unchanged if the simulator backend is ever swapped.
"""

from alpha.core.entities import BlockState
from alpha.core.interfaces import SimulatorClient
from alpha.config import settings


class SceneService:
    def __init__(self, simulator: SimulatorClient) -> None:
        self.simulator = simulator

    def build_and_settle(self, gui: bool = False, settle_steps: int = settings.SETTLE_STEPS) -> dict:
        self.simulator.connect(gui=gui)
        handles = self.simulator.build_scene()
        self.simulator.step(settle_steps)
        return handles

    def block_states(self) -> dict[str, BlockState]:
        return {
            color.value: self.simulator.get_block_state(color)
            for color in settings.BLOCK_SPAWN_POSITIONS
        }

    def smoke_test(self, gui: bool = False) -> dict:
        """
        Week 1 acceptance check: scene loads, all bodies exist, physics
        steps without error, blocks stay close to their spawn position
        after settling.
        """
        self.build_and_settle(gui=gui)
        report = {}
        for color, spawn_pos in settings.BLOCK_SPAWN_POSITIONS.items():
            state = self.simulator.get_block_state(color)
            drift = sum((a - b) ** 2 for a, b in zip(state.position, spawn_pos)) ** 0.5
            report[color.value] = {"position": state.position, "drift_from_spawn": drift}
        self.simulator.disconnect()
        return report
