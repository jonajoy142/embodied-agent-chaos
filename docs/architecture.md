# Architecture

Alpha is one repo for the full 12-week build, laid out as layered/clean
architecture so weeks 2–12 add code into an existing structure instead of
each week becoming its own throwaway repo.

```
src/alpha/
  config/        Centralized constants (URDF names, positions, tolerances).
                 Every other layer reads from here — no magic numbers elsewhere.

  core/          Domain layer. Plain dataclasses/enums + abstract interfaces.
                 Zero external dependencies (no pybullet, no pydantic).
                 Every other layer may import from core; core imports from
                 nothing else in this repo.

  schemas/       Pydantic validation and I/O schemas — ActionPlan is the
                 Brain/Body contract; EpisodeResultSchema is the CSV/API
                 boundary. Separate from core entities on purpose: internal
                 shape and stored/external shape are allowed to diverge.

  models/        Config data models for agent/provider selection and
                 deterministic fault experiments.

  clients/       Thin wrappers around external systems. simulator_client.py
                 is the ONLY module that imports pybullet directly. llm_client.py
                 owns LLM provider integration and provides a deterministic
                 mock. Swapping PyBullet or the LLM provider means editing
                 here only.

  services/      Orchestration / business logic. scene_service.py and
                 control_service.py are live from Week 1. agent_service.py
                 coordinates Brain -> ActionPlan -> Body execution (single-shot;
                 no LangGraph graph). fault_service.py implements legacy CLI
                 faults via ScriptedFaultInjector. Formal experiments use
                 chaos_robotics_wrapper.py via matrix_episode_runner.py.
                 metrics_service.py aggregates telemetry including
                 post_fault_completion_time and safety violation rate.

  repos/         Persistence. episode_repository.py writes EpisodeResult
                 records to CSV. Swapping CSV for SQLite/Postgres later
                 means editing this file only — nothing upstream changes.

  dashboard/     Presentation layer stub for Week 8.

scripts/         CLI entrypoints. This is where you actually run things —
                 they wire clients -> services -> repos together.

tests/           Unit tests cover action-plan validation, mock Brain,
                 deterministic faults, and fake-simulator execution.
                 test_control_service.py is the PyBullet integration check
                 and skips automatically if pybullet isn't installed.

docker/          Dockerfile stub for Week 7's reproducibility milestone.
```

## Why this shape

The core risk called out in the PRD is scope creep and rework as faults,
an agent, and metrics get layered on week over week. The layering fixes the
seams before they're needed:

- `ControlService` takes an optional `fault_injector`, so Week 1 callers can
  still use `ControlService(simulator)` while fault experiments use
  `ControlService(simulator, fault_injector=...)`.
- `ActionPlan` is validated before execution. LLM output is data, not code.
- `EpisodeResult` and `EpisodeResultSchema` now preserve Week 1 fields while
  adding agent/fault/action-plan metadata needed by the Week 5 metrics harness.
- `AgentPlanner` is implemented by `AgentBrain`; `ControlService` does not
  care whether actions came from a hardcoded mock Brain or an optional real
  LLM provider.

## Current data flows

Week 1 scripted baseline:

```text
scripts/run_scripted_pick_place.py
  -> SceneService
  -> ControlService.run_scripted_pick_place()
  -> PyBulletSimulatorClient
  -> EpisodeRepository
  -> results/episodes.csv
```

Week 2 agent run:

```text
TaskSpec
  -> AgentBrain
  -> LLMClient / MockLLMClient
  -> ActionPlan schema validation
  -> AgentService
  -> ControlService
  -> PyBulletSimulatorClient
  -> EpisodeRepository
```

Weeks 3-4 fault run:

```text
TaskSpec
  -> AgentBrain
  -> ActionPlan
  -> AgentService
  -> ControlService
  -> ScriptedFaultInjector
  -> PyBulletSimulatorClient
  -> EpisodeResult(action_plan_json, execution_log_json, fault metadata)
  -> results/episodes.csv
```

Week 6 formal experiment (pilot / matrix):

```text
configs/experiments/pilot.yaml
  -> execute_matrix_episode()
  -> ChaosRoboticsWrapper (canonical faults)
  -> AgentService.run_task(hooks=...)
  -> OpenAI LLMClient (formal studies; Mock excluded)
  -> TelemetrySimulatorWrapper + PyBulletSafetyMonitor
  -> ExperimentDataRepository (data/raw/, data/processed/)
  -> results/pilot/ or results/
```

See `docs/FAULT_INJECTION.md` for canonical vs legacy fault paths.
