"""Unit tests for ChaosRoboticsWrapper and ChaosConfig."""

from __future__ import annotations

import json

import pytest

from alpha.clients.chaos_robotics_wrapper import ChaosRoboticsWrapper, PyBulletChaosRoboticsWrapper
from alpha.core.entities import BlockColor, FaultType
from alpha.models.chaos_config import (
    ChaosConfig,
    FaultIntensity,
    FaultSpec,
    InjectionMode,
    TriggerConfig,
    TriggerKind,
)
from alpha.config import settings
from tests.fakes import FakeSimulator

pybullet = pytest.importorskip("pybullet", reason="PyBullet required for integration tests")

from alpha.clients.simulator_client import PyBulletSimulatorClient


def _grip_slip_config(*, phase: str = "lift", slip_probability: float = 1.0, seed: int = 7) -> ChaosConfig:
    return ChaosConfig(
        seed=seed,
        mode=InjectionMode.SINGLE,
        faults=(
            FaultSpec(
                fault_type=FaultType.GRIP_SLIP,
                enabled=True,
                trigger=TriggerConfig(kind=TriggerKind.EVENT, phase=phase),
                intensity=FaultIntensity(slip_probability=slip_probability, weaken_force_factor=0.0),
                probability=1.0,
            ),
        ),
    )


def test_chaos_config_from_mapping_round_trip():
    config = ChaosConfig.from_mapping(
        {
            "mode": "concurrent",
            "seed": 99,
            "faults": [
                {
                    "fault_type": "sensor_lag",
                    "enabled": True,
                    "trigger": {"kind": "time", "after_seconds": 2.0},
                    "intensity": {"lag_seconds": 0.4},
                }
            ],
        }
    )

    assert config.mode == InjectionMode.CONCURRENT
    assert config.seed == 99
    assert len(config.enabled_faults()) == 1
    assert config.enabled_faults()[0].fault_type == FaultType.SENSOR_LAG


def test_sensor_lag_returns_stale_observation():
    config = ChaosConfig(
        seed=1,
        sim_timestep=1.0,
        faults=(
            FaultSpec(
                fault_type=FaultType.SENSOR_LAG,
                enabled=True,
                trigger=TriggerConfig(kind=TriggerKind.IMMEDIATE),
                intensity=FaultIntensity(lag_seconds=2.0),
            ),
        ),
    )
    wrapper = ChaosRoboticsWrapper(FakeSimulator(), config)

    first = {"blocks": {"red": [0.1, 0.2, 0.3]}}
    second = {"blocks": {"red": [0.9, 0.8, 0.7]}}

    assert wrapper.wrap_observation(first) == first
    wrapper.context.elapsed_seconds = 3.0
    delayed = wrapper.wrap_observation(second)

    assert delayed == first
    assert wrapper.event_log()[-1]["fault_applied"] is True


def test_unreachable_ik_offsets_move_target():
    config = ChaosConfig(
        faults=(
            FaultSpec(
                fault_type=FaultType.UNREACHABLE_IK,
                enabled=True,
                trigger=TriggerConfig(kind=TriggerKind.IMMEDIATE),
                intensity=FaultIntensity(offset_magnitude=0.1, offset=(1.0, 0.0, 0.0)),
            ),
        )
    )
    sim = FakeSimulator()
    wrapper = ChaosRoboticsWrapper(sim, config)
    target = (0.5, 0.3, 0.7)

    wrapper.move_end_effector(target)

    assert sim.moves[0] == pytest.approx((0.6, 0.3, 0.7))
    assert wrapper.event_log()[0]["fault_applied"] is True


def test_planner_output_corruption_deletes_actions_key():
    config = ChaosConfig(
        faults=(
            FaultSpec(
                fault_type=FaultType.PLANNER_OUTPUT_CORRUPTION,
                enabled=True,
                trigger=TriggerConfig(kind=TriggerKind.IMMEDIATE),
                intensity=FaultIntensity(
                    keys_to_delete=("actions",),
                    inject_malformed_chars=False,
                ),
            ),
        ),
        seed=0,
    )
    wrapper = ChaosRoboticsWrapper(FakeSimulator(), config)
    raw = json.dumps({"agent_name": "mock", "actions": [{"action": "RELEASE"}]})

    corrupted = wrapper.intercept_planner_output(raw)
    payload = json.loads(corrupted)

    assert "actions" not in payload
    assert wrapper.event_log()[0]["fault_applied"] is True


def test_grip_slip_on_fake_simulator_clears_active_grip():
    sim = FakeSimulator()
    wrapper = ChaosRoboticsWrapper(sim, _grip_slip_config())
    wrapper.set_phase("lift")

    handle = wrapper.grip(BlockColor.RED)
    assert handle is not None
    assert wrapper.grip_is_active is True

    wrapper.step(1)

    assert wrapper.grip_is_active is False
    assert wrapper.grip_handle is None
    assert wrapper.event_log()[-1]["fault_applied"] is True
    assert wrapper.event_log()[-1]["constraint_removed"] is True


def test_grip_slip_does_not_fire_before_event_phase():
    sim = FakeSimulator()
    wrapper = ChaosRoboticsWrapper(sim, _grip_slip_config(phase="lift"))
    wrapper.set_phase("pick")

    wrapper.grip(BlockColor.RED)
    wrapper.step(1)

    assert wrapper.grip_is_active is True
    assert not wrapper.event_log()


def test_grip_slip_pybullet_removes_fixed_constraint():
    inner = PyBulletSimulatorClient()
    wrapper = PyBulletChaosRoboticsWrapper(inner, _grip_slip_config(seed=11))
    inner.connect(gui=False)
    inner.build_scene()
    inner.step(settings.SETTLE_STEPS)

    wrapper.set_phase("lift")
    constraint_id = wrapper.grip(BlockColor.RED)
    assert constraint_id is not None
    assert wrapper.grip_is_active is True
    pybullet.getConstraintInfo(constraint_id)

    wrapper.step(1)

    assert wrapper.grip_is_active is False
    assert wrapper.event_log()[-1]["fault_applied"] is True
    assert wrapper.event_log()[-1]["constraint_removed"] is True

    with pytest.raises(pybullet.error):
        pybullet.getConstraintInfo(constraint_id)

    inner.disconnect()


def test_chaos_config_from_yaml_example_manifest():
    pytest.importorskip("yaml")
    config = ChaosConfig.from_yaml("config/chaos_example.yaml")

    assert config.mode == InjectionMode.CONCURRENT
    fault_types = {spec.fault_type for spec in config.enabled_faults()}
    assert FaultType.SENSOR_LAG in fault_types
    assert FaultType.GRIP_SLIP in fault_types
