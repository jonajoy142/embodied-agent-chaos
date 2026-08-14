# Fault injection in Alpha

Alpha has **two** fault-injection implementations. Only one is authoritative for
formal experiments.

## Canonical path (formal experiments)

**`ChaosRoboticsWrapper`** + **`ChaosConfig`** is the authoritative fault
injection interface for Phase A and the experiment matrix.

| Fault | ID | Injection boundary | Notes |
| --- | --- | --- | --- |
| Sensor lag | F1 | `wrap_observation()` before planning | OpenAI agent consumes `scene_info`; MockLLMClient does not |
| Grip slip | F2 | `move_end_effector()` / grip constraint | Triggered at lift phase in matrix runs |
| Unreachable IK | F3 | Target offset at control boundary | Kinematic/unreachable-target fault |
| Planner corruption | F4 | `ChaosPlannerLLMClient` wrapper | **Planner/schema robustness** — do not claim general planner degradation |

Execution flow:

```text
SceneService.block_states()
  -> ChaosRoboticsWrapper.wrap_observation()   # F1
  -> AgentBrain.plan(scene_info)               # OpenAI reads observation
  -> AgentService.run_task(hooks=...)
  -> ChaosRoboticsWrapper (F2/F3 on actuation)
  -> ChaosPlannerLLMClient (F4 on planner output)
  -> PyBullet
```

Matrix and pilot runners use `execute_matrix_episode()` which wires this path
through `AgentService.run_task(hooks=...)`.

## Legacy path (CLI / compatibility)

**`ScriptedFaultInjector`** + **`FaultConfig`** injects faults at the
`ControlService` boundary only. It is used by:

- `scripts/run_agent_pick_place.py` (Week 3–4 CLI demos)
- `tests/test_fault_service.py` (unit tests)

It does **not** implement F1 observation lag or F4 planner corruption. Keep it
for backward-compatible CLI testing; do not use it for published experiment
results.

## MockLLMClient scope

`MockLLMClient` intentionally ignores `scene_info`. It is for engineering,
unit, and integration tests only. Formal studies must use the OpenAI agent path
so F1 faults reach the planner.

## Concurrent faults (Phase A)

Phase A uses **F1 + F2** only (`sensor_lag` + `grip_slip`). Do not expand to
F3/F4 combinations unless later evidence justifies it.

See `configs/experiments/pilot.yaml` for the Phase A schedule.
