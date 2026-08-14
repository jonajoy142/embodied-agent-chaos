"""
Integration test for the Week 1 scripted pick/place, run against the real
PyBullet backend headless. Skipped automatically if pybullet isn't
installed in whatever environment runs pytest.
"""

import pytest

pybullet = pytest.importorskip("pybullet")

from alpha.clients.simulator_client import PyBulletSimulatorClient
from alpha.core.entities import BlockColor, TaskSpec
from alpha.services.control_service import ControlService
from alpha.services.scene_service import SceneService
from alpha.config import settings


def test_scripted_pick_place_moves_block_toward_zone():
    simulator = PyBulletSimulatorClient()
    scene_service = SceneService(simulator)
    control_service = ControlService(simulator)

    scene_service.build_and_settle(gui=False)
    task = TaskSpec(block=BlockColor.RED, place_zone=settings.PLACE_ZONE_POSITION)
    result = control_service.run_scripted_pick_place(task)
    simulator.disconnect()

    # Loose assertion deliberately: Week 1's bar is "the primitives work and
    # the block ends up near the zone," not exact placement.
    assert result.final_error < 0.2
