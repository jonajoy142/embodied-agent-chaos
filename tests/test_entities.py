"""
Tests for the core domain layer. These import nothing but core.entities,
so they run even in environments where pybullet isn't installed — a
useful CI smoke check independent of the physics stack.
"""

from alpha.core.entities import BlockColor, EpisodeResult, FaultType, TaskSpec


def test_task_spec_defaults():
    task = TaskSpec(block=BlockColor.RED, place_zone=(0.6, 0.35, 0.65))
    assert task.block == BlockColor.RED
    assert "pick block" in task.description


def test_episode_result_defaults_to_no_fault():
    task = TaskSpec(block=BlockColor.GREEN, place_zone=(0.6, 0.35, 0.65))
    result = EpisodeResult(task=task, success=True, final_error=0.01, final_position=(0.6, 0.35, 0.65))
    assert result.fault_type == FaultType.NONE
    assert result.time_to_recovery_steps is None
