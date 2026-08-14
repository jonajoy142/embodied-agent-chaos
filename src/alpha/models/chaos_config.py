"""
Configuration models for the ChaosRoboticsWrapper fault injection layer.

Supports programmatic construction via dataclasses and declarative loading
from YAML experiment manifests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from alpha.core.entities import FaultType, Vec3


class InjectionMode(str, Enum):
    """Whether one fault or many may be active in the same timestep."""

    SINGLE = "single"
    CONCURRENT = "concurrent"


class TriggerKind(str, Enum):
    """How a fault spec decides it is allowed to fire."""

    IMMEDIATE = "immediate"
    TIME = "time"
    EVENT = "event"


@dataclass(frozen=True)
class TriggerConfig:
    """
    Trigger predicate for a fault spec.

    - ``immediate``: active as soon as the episode starts (subject to probability).
    - ``time``: active after ``after_seconds`` of simulated wall time elapse.
    - ``event``: active when runtime ``phase`` matches ``phase`` and/or
      ``step_index`` equals ``step``.
    """

    kind: TriggerKind = TriggerKind.IMMEDIATE
    after_seconds: float | None = None
    phase: str | None = None
    step: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> TriggerConfig:
        if not data:
            return cls()
        kind_raw = str(data.get("kind", TriggerKind.IMMEDIATE.value)).lower()
        return cls(
            kind=TriggerKind(kind_raw),
            after_seconds=_optional_float(data.get("after_seconds")),
            phase=data.get("phase"),
            step=_optional_int(data.get("step")),
        )


@dataclass(frozen=True)
class FaultIntensity:
    """
    Fault-specific severity knobs.

    Unused fields are ignored per fault type:
    - sensor_lag: ``lag_seconds``
    - grip_slip: ``slip_probability``, ``weaken_force_factor`` (0 clears constraint)
    - unreachable_ik: ``offset_magnitude``, ``offset`` (Vec3 direction hint)
    - planner_output_corruption: ``corruption_probability``, ``keys_to_delete``,
      ``inject_malformed_chars``
    """

    lag_seconds: float = 0.5
    slip_probability: float = 1.0
    weaken_force_factor: float = 0.0
    offset_magnitude: float = 0.25
    offset: Vec3 = (1.0, 1.0, 0.0)
    corruption_probability: float = 1.0
    keys_to_delete: tuple[str, ...] = ("actions", "target", "block")
    inject_malformed_chars: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> FaultIntensity:
        if not data:
            return cls()
        offset_raw = data.get("offset", cls.offset)
        offset: Vec3
        if isinstance(offset_raw, (list, tuple)) and len(offset_raw) == 3:
            offset = (float(offset_raw[0]), float(offset_raw[1]), float(offset_raw[2]))
        else:
            offset = cls.offset
        keys = data.get("keys_to_delete", cls.keys_to_delete)
        return cls(
            lag_seconds=float(data.get("lag_seconds", cls.lag_seconds)),
            slip_probability=float(data.get("slip_probability", cls.slip_probability)),
            weaken_force_factor=float(data.get("weaken_force_factor", cls.weaken_force_factor)),
            offset_magnitude=float(data.get("offset_magnitude", cls.offset_magnitude)),
            offset=offset,
            corruption_probability=float(data.get("corruption_probability", cls.corruption_probability)),
            keys_to_delete=tuple(keys),
            inject_malformed_chars=bool(data.get("inject_malformed_chars", cls.inject_malformed_chars)),
        )


@dataclass(frozen=True)
class FaultSpec:
    """One injectable fault scenario."""

    fault_type: FaultType
    enabled: bool = True
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    intensity: FaultIntensity = field(default_factory=FaultIntensity)
    probability: float = 1.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> FaultSpec:
        fault_raw = data["fault_type"]
        fault_type = fault_raw if isinstance(fault_raw, FaultType) else FaultType(str(fault_raw))
        return cls(
            fault_type=fault_type,
            enabled=bool(data.get("enabled", True)),
            trigger=TriggerConfig.from_mapping(data.get("trigger")),
            intensity=FaultIntensity.from_mapping(data.get("intensity")),
            probability=float(data.get("probability", 1.0)),
        )


@dataclass(frozen=True)
class ChaosConfig:
    """
    Top-level chaos experiment configuration.

    ``mode`` controls fault composition:
    - ``single``: at most one fault spec may fire per hook invocation.
    - ``concurrent``: every spec whose trigger is satisfied may fire together.
    """

    faults: tuple[FaultSpec, ...] = ()
    mode: InjectionMode = InjectionMode.SINGLE
    seed: int | None = 42
    sim_timestep: float = 1.0 / 240.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ChaosConfig:
        mode_raw = str(data.get("mode", InjectionMode.SINGLE.value)).lower()
        faults_raw = data.get("faults", [])
        return cls(
            faults=tuple(FaultSpec.from_mapping(item) for item in faults_raw),
            mode=InjectionMode(mode_raw),
            seed=data.get("seed", 42),
            sim_timestep=float(data.get("sim_timestep", 1.0 / 240.0)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChaosConfig:
        """Load a chaos manifest from YAML."""
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load chaos manifests. Install with: pip install pyyaml"
            ) from exc

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Chaos config root must be a mapping, got {type(payload)!r}")
        return cls.from_mapping(payload)

    def enabled_faults(self) -> tuple[FaultSpec, ...]:
        return tuple(spec for spec in self.faults if spec.enabled)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
