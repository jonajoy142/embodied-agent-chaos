"""
Abstract interfaces that clients and services implement.

Depending on abstractions here (rather than concrete classes like
PyBulletSimulatorClient directly) is what lets tests run against a fake
simulator, and what lets a future swap (e.g. Isaac Sim instead of PyBullet,
or a different LLM provider for the agent) happen inside clients/ without
touching services/ or scripts/.
"""

from abc import ABC, abstractmethod
from typing import Any

from alpha.core.entities import BlockColor, BlockState, TaskSpec, Vec3


class SimulatorClient(ABC):
    """Contract for any physics backend (PyBullet now, possibly others later)."""

    @abstractmethod
    def connect(self, gui: bool) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def build_scene(self) -> dict[str, Any]:
        """Load arm, table, and blocks. Returns backend-specific handles."""

    @abstractmethod
    def step(self, steps: int) -> None: ...

    @abstractmethod
    def get_block_state(self, color: BlockColor) -> BlockState: ...

    @abstractmethod
    def move_end_effector(self, target_pos: Vec3, target_orn: Any = None) -> dict[str, Any]: ...

    @abstractmethod
    def grip(self, color: BlockColor) -> Any:
        """Attach a block to the end effector. Returns a handle needed to release it."""

    @abstractmethod
    def release(self, grip_handle: Any) -> None: ...

    @abstractmethod
    def observe(self) -> dict[str, Any]:
        """Return current scene observation (block positions, etc.)."""


class AgentPlanner(ABC):
    """Contract for the Week 2 Brain.

    Implementations decide what high-level actions should be taken. They do
    not execute the actions and must not talk to PyBullet directly.
    """

    @abstractmethod
    def plan(self, task: TaskSpec, scene_info: dict | None = None):
        """Return a structured action plan for the Body/ControlService to execute."""


class FaultInjector(ABC):
    """Legacy control-boundary fault contract (``ScriptedFaultInjector``).

    Formal experiments use ``ChaosRoboticsWrapper`` instead — see
    ``docs/FAULT_INJECTION.md``.
    """

    @abstractmethod
    def maybe_corrupt_target(self, target_pos: Vec3) -> Vec3: ...

    @abstractmethod
    def maybe_drop_grip(self, gripped: bool) -> bool: ...

    @abstractmethod
    def should_fail_grip(self, color: BlockColor) -> bool: ...

    @abstractmethod
    def should_fail_release(self) -> bool: ...

    @abstractmethod
    def event_log(self) -> list[dict]: ...

    @property
    @abstractmethod
    def fault_type(self): ...

    @property
    @abstractmethod
    def seed(self) -> int | None: ...
