from typing import Any

from alpha.clients.llm_client import LLMClient
from alpha.config import settings
from alpha.core.entities import BlockColor, BlockState, TaskSpec, Vec3
from alpha.core.interfaces import SimulatorClient
from alpha.schemas.action_plan import ActionPlan


class RecordingLLMClient(LLMClient):
    """Test double that records the observation dict passed to the planner."""

    name = "recording"

    def __init__(self) -> None:
        self.last_scene_info: dict[str, Any] | None = None

    def generate_action_plan(
        self,
        task: TaskSpec,
        scene_info: dict[str, Any] | None = None,
    ) -> ActionPlan:
        self.last_scene_info = scene_info
        spawn = settings.BLOCK_SPAWN_POSITIONS[task.block]
        return ActionPlan(
            agent_name=self.name,
            actions=[
                {"action": "MOVE_TO", "target": (spawn[0], spawn[1], spawn[2] + 0.15)},
                {"action": "GRIP", "block": task.block},
                {"action": "RELEASE"},
            ],
        )


class FakeSimulator(SimulatorClient):
    def __init__(self) -> None:
        self.block_positions = dict(settings.BLOCK_SPAWN_POSITIONS)
        self.moves: list[Vec3] = []
        self.gripped: BlockColor | None = None
        self.released = False

    def connect(self, gui: bool) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def build_scene(self) -> dict[str, Any]:
        return {}

    def step(self, steps: int) -> None:
        pass

    def get_block_state(self, color: BlockColor) -> BlockState:
        return BlockState(color=color, position=self.block_positions[color], orientation=(0, 0, 0, 1))

    def move_end_effector(self, target_pos: Vec3, target_orn: Any = None) -> dict[str, Any]:
        self.moves.append(target_pos)
        if self.gripped is not None:
            self.block_positions[self.gripped] = target_pos
        return {"reached_pos": target_pos, "target_pos": target_pos, "error": 0.0}

    def grip(self, color: BlockColor) -> Any:
        self.gripped = color
        return {"block": color.value}

    def release(self, grip_handle: Any) -> None:
        self.released = True
        self.gripped = None
