"""
PyBullet implementation of SimulatorClient.

This is the ONLY module in the codebase that imports pybullet directly.
Every other layer (services, scripts, tests) depends on
core.interfaces.SimulatorClient, not on this class or on pybullet itself.
That's deliberate: if Week 6+ ever needs a faster headless backend, or
tests need a fake simulator, only this file changes.
"""

from typing import Any

import pybullet as p
import pybullet_data

from alpha.core.entities import BlockColor, BlockState, Vec3
from alpha.core.exceptions import SimulationError
from alpha.core.interfaces import SimulatorClient
from alpha.config import settings


class PyBulletSimulatorClient(SimulatorClient):
    def __init__(self) -> None:
        self._arm_id: int | None = None
        self._table_id: int | None = None
        self._block_ids: dict[BlockColor, int] = {}
        self._end_effector_link: int | None = None
        self._connected = False

    def connect(self, gui: bool = False) -> None:
        p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*settings.GRAVITY)
        self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            p.disconnect()
            self._connected = False

    def build_scene(self) -> dict[str, Any]:
        if not self._connected:
            raise SimulationError("connect() must be called before build_scene()")

        p.loadURDF("plane.urdf")
        self._table_id = p.loadURDF(settings.TABLE_URDF, basePosition=settings.TABLE_BASE_POSITION)
        self._arm_id = p.loadURDF(settings.ARM_URDF, basePosition=settings.ARM_BASE_POSITION, useFixedBase=True)
        for link_index in range(-1, p.getNumJoints(self._arm_id)):
            p.setCollisionFilterPair(self._arm_id, self._table_id, link_index, -1, enableCollision=0)

        self._block_ids = {}
        for color, pos in settings.BLOCK_SPAWN_POSITIONS.items():
            block_id = p.loadURDF(settings.BLOCK_URDF, basePosition=pos, globalScaling=1.2)
            p.changeVisualShape(block_id, -1, rgbaColor=settings.BLOCK_RGBA[color])
            self._block_ids[color] = block_id

        num_joints = p.getNumJoints(self._arm_id)
        self._end_effector_link = num_joints - 1

        return {
            "arm_id": self._arm_id,
            "table_id": self._table_id,
            "block_ids": dict(self._block_ids),
            "end_effector_link": self._end_effector_link,
        }

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            p.stepSimulation()

    def get_block_state(self, color: BlockColor) -> BlockState:
        block_id = self._block_ids[color]
        pos, orn = p.getBasePositionAndOrientation(block_id)
        return BlockState(color=color, position=pos, orientation=orn)

    def move_end_effector(self, target_pos: Vec3, target_orn: Any = None) -> dict[str, Any]:
        self.set_end_effector_target(target_pos, target_orn)
        self.step(settings.MOVE_STEPS)
        return self.end_effector_tracking_result(target_pos)

    def set_end_effector_target(self, target_pos: Vec3, target_orn: Any = None) -> None:
        """Apply IK setpoints without stepping — enables wrappers to sample mid-move."""
        if self._arm_id is None or self._end_effector_link is None:
            raise SimulationError("build_scene() must be called before move_end_effector()")

        ik_kwargs = {
            "maxNumIterations": 200,
            "residualThreshold": 1e-5,
        }
        if target_orn is None:
            joint_positions = p.calculateInverseKinematics(
                self._arm_id, self._end_effector_link, target_pos, **ik_kwargs
            )
        else:
            joint_positions = p.calculateInverseKinematics(
                self._arm_id, self._end_effector_link, target_pos, target_orn, **ik_kwargs
            )

        num_arm_joints = min(len(joint_positions), p.getNumJoints(self._arm_id))
        for i in range(num_arm_joints):
            p.setJointMotorControl2(
                self._arm_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=2000,
                positionGain=0.1,
                velocityGain=1.0,
            )

    def end_effector_tracking_result(self, target_pos: Vec3) -> dict[str, Any]:
        if self._arm_id is None or self._end_effector_link is None:
            raise SimulationError("build_scene() must be called before move_end_effector()")
        ee_state = p.getLinkState(self._arm_id, self._end_effector_link)
        reached_pos = ee_state[0]
        error = sum((a - b) ** 2 for a, b in zip(reached_pos, target_pos)) ** 0.5
        return {"reached_pos": reached_pos, "target_pos": target_pos, "error": error}

    def grip(self, color: BlockColor) -> Any:
        block_id = self._block_ids[color]
        constraint_id = p.createConstraint(
            parentBodyUniqueId=self._arm_id,
            parentLinkIndex=self._end_effector_link,
            childBodyUniqueId=block_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0),
        )
        return constraint_id

    def release(self, grip_handle: Any) -> None:
        p.removeConstraint(grip_handle)

    def observe(self) -> dict[str, Any]:
        """Return current scene observation with block positions."""
        observation = {}
        for color, block_id in self._block_ids.items():
            pos, orn = p.getBasePositionAndOrientation(block_id)
            observation[color.value] = pos
        return observation
