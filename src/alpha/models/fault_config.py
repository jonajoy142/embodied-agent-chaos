"""Configuration models for the deterministic fault injection layer."""

from dataclasses import dataclass

from alpha.core.entities import FaultType, Vec3


@dataclass(frozen=True)
class FaultConfig:
    fault_type: FaultType = FaultType.NONE
    enabled: bool = False
    probability: float = 1.0      # chance the fault fires when its trigger condition is hit
    magnitude: float = 0.0        # fault-specific severity, usually meters for target offsets
    offset: Vec3 = (0.05, -0.05, 0.0)
    seed: int | None = 42
    concurrent_with: tuple[FaultType, ...] = ()  # other faults to inject simultaneously

    @property
    def active(self) -> bool:
        return self.enabled and self.fault_type != FaultType.NONE


DEFAULT_FAULT_MATRIX: tuple[FaultConfig, ...] = (
    FaultConfig(FaultType.NONE, enabled=False, probability=0.0),
    FaultConfig(FaultType.TARGET_CORRUPTION, enabled=True, magnitude=0.05),
    FaultConfig(FaultType.POSITION_OFFSET, enabled=True, magnitude=0.05),
    FaultConfig(FaultType.GRIP_FAILURE, enabled=True),
    FaultConfig(FaultType.RELEASE_FAILURE, enabled=True),
)
