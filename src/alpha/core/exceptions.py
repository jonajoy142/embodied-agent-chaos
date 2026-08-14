class AlphaError(Exception):
    """Base class for all Alpha domain errors."""


class SimulationError(AlphaError):
    """Raised when the physics backend fails to build or step the scene."""


class GraspFailedError(AlphaError):
    """Raised when a grip attempt fails (e.g. arm out of reach of the block)."""


class TaskNotAchievedError(AlphaError):
    """Raised when an episode completes but the task success condition isn't met."""
