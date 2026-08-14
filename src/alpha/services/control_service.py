"""
Control orchestration: move-to, grip, release, and the Week 1 scripted
(non-agentic) pick/place task.

No LLM, no fault injection here — that's the point of Week 1. But every
method takes plain waypoints/targets as arguments rather than deciding
them internally, so:
  - Week 2 can swap "hardcoded waypoints" for "agent.plan(task)" output
    without changing move_to/grip/release.
  - Weeks 3-4 can wrap these same calls with a FaultInjector (see
    core.interfaces.FaultInjector) without rewriting this service.
"""

from alpha.core.entities import BlockColor, EpisodeResult, FaultType, TaskSpec, Vec3
from alpha.core.interfaces import FaultInjector, SimulatorClient
from alpha.config import settings


class ControlService:
    def __init__(self, simulator: SimulatorClient, fault_injector: FaultInjector | None = None) -> None:
        self.simulator = simulator
        self.fault_injector = fault_injector  # None in Week 1; wired up in Weeks 3-4

    def move_to(self, target: Vec3) -> dict:
        requested_target = target
        if self.fault_injector is not None:
            target = self.fault_injector.maybe_corrupt_target(target)
        result = self.simulator.move_end_effector(target)
        result["requested_target"] = requested_target
        result["executed_target"] = target
        return result

    def grip(self, color: BlockColor):
        if self.fault_injector is not None and self.fault_injector.should_fail_grip(color):
            return None
        return self.simulator.grip(color)

    def release(self, grip_handle) -> None:
        if grip_handle is None:
            return
        if self.fault_injector is not None and self.fault_injector.should_fail_release():
            return
        self.simulator.release(grip_handle)

    def evaluate_task(self, task: TaskSpec) -> tuple[bool, float, Vec3]:
        final_state = self.simulator.get_block_state(task.block)
        final_error = sum((a - b) ** 2 for a, b in zip(final_state.position, task.place_zone)) ** 0.5
        success = final_error < settings.PLACE_SUCCESS_TOLERANCE
        return success, round(final_error, 4), final_state.position

    def run_scripted_pick_place(self, task: TaskSpec) -> EpisodeResult:
        """
        Week 1 deliverable: hardcoded waypoint sequence, no agent, no faults.
        Returns an EpisodeResult in the exact shape metrics_service (Week 5)
        will consume, so this function's output contract doesn't change later.
        """
        spawn_pos = settings.BLOCK_SPAWN_POSITIONS[task.block]
        above_block = (spawn_pos[0], spawn_pos[1], spawn_pos[2] + 0.15)
        grasp_pos = (spawn_pos[0], spawn_pos[1], spawn_pos[2] + 0.02)
        above_place = (task.place_zone[0], task.place_zone[1], task.place_zone[2] + 0.15)
        release_pos = (
            task.place_zone[0],
            task.place_zone[1],
            task.place_zone[2] + settings.PLACE_RELEASE_HEIGHT_OFFSET,
        )

        steps_taken = 0
        for waypoint in (above_block, grasp_pos):
            self.move_to(waypoint)
            steps_taken += 1

        grip_handle = self.grip(task.block)

        for waypoint in (above_block, above_place, release_pos):
            self.move_to(waypoint)
            steps_taken += 1

        self.release(grip_handle)
        self.simulator.step(60)  # let the released block settle

        success, final_error, final_position = self.evaluate_task(task)

        return EpisodeResult(
            task=task,
            success=success,
            final_error=final_error,
            final_position=final_position,
            fault_type=FaultType.NONE,
            steps_taken=steps_taken,
        )
