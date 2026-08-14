# Research Engineering Audit

**Project:** Chaos Engineering Harness for Embodied AI Agents (Alpha)  
**Audit date:** 2026-08-13  
**Auditor role:** Senior research engineer / systems researcher (pre-experiment review)  
**Scope:** Full repository inspection before further experimentation or paper claims  
**Action taken:** Read-only audit. No code modified.

---

## Executive Summary

Alpha is a **real, layered research codebase** with a working PyBullet pick-and-place environment, a Brain/Body agent loop, two partially overlapping fault-injection implementations, a telemetry layer, and a batch experiment runner. The repository is **not empty** and **not a toy single-file demo**.

However, several **critical gaps** currently prevent this system from supporting scientifically strong claims about LLM-driven embodied-agent reliability:

1. **LangGraph is documented but not implemented.** The agent is a single-shot plan → execute loop with no replanning, reflection, or self-diagnosis.
2. **The default experiment agent (`MockLLMClient`) ignores observations**, making sensor/planner-layer faults largely irrelevant to planning behavior.
3. **Recovery metrics (TTR/MTTR) are defined but not measurable** in the current agent architecture because there is no recovery loop.
4. **Two fault-injection systems coexist** without a unified interface or single source of truth.
5. **Pilot matrix results already in `results/` are not publication-ready** and should not be interpreted as experimental findings.
6. **Safety metrics appear miscalibrated** (constant 120 violations/episode in smoke runs).

**Recommendation:** Do **not** run or report the full 75-episode matrix for paper-quality results until a formal **Phase A pilot** validates fault injection, metrics, baselines, and agent conditions. The infrastructure is a strong foundation; the **experimental design and agent fidelity** need hardening first.

---

## 1. Current Architecture

### 1.1 Repository layout (actual)

```
alpha_repo/
├── PRD.md                          # 12-week roadmap and research positioning
├── README.md                       # User-facing setup (partially stale vs code)
├── pyproject.toml                  # Package definition; optional deps: dev, agent, analytics, dashboard
├── requirements.txt                # Alternate install path (unpinned)
├── config/
│   ├── chaos_example.yaml          # Example chaos manifest (not used by matrix runner)
│   └── settings.py                 # Legacy/alternate config pointer
├── docker/Dockerfile               # Week 7 stub; runs scripted baseline only
├── docs/
│   └── architecture.md             # Layering rationale (partially stale)
├── scripts/
│   ├── run_scripted_pick_place.py  # Week 1 baseline CLI
│   ├── run_agent_pick_place.py     # Week 2/3-4 single-episode CLI
│   ├── run_experiment_matrix.py    # Week 6 batch runner
│   └── analyze_experiments.py      # Week 5 analysis helper (pandas/matplotlib)
├── results/                        # Mutable CSV outputs (not immutable raw store)
│   ├── episodes.csv
│   ├── experiments_log.csv
│   ├── experiment_matrix_runs.csv
│   ├── experiment_matrix_checkpoint.json
│   └── experiment_matrix_summary.md
├── src/alpha/
│   ├── config/settings.py          # Scene constants, thresholds, CSV paths
│   ├── core/
│   │   ├── entities.py             # Domain types: TaskSpec, FaultType, EpisodeResult
│   │   ├── interfaces.py           # SimulatorClient, AgentPlanner, FaultInjector ABCs
│   │   └── exceptions.py
│   ├── schemas/
│   │   ├── action_plan.py          # Pydantic Brain/Body contract
│   │   ├── episode.py              # Episode CSV schema
│   │   └── telemetry.py            # Experiment telemetry CSV schema
│   ├── models/
│   │   ├── agent_config.py
│   │   ├── chaos_config.py         # ChaosRoboticsWrapper config + YAML loader
│   │   └── fault_config.py         # ScriptedFaultInjector config
│   ├── clients/
│   │   ├── simulator_client.py     # ONLY direct pybullet import (PyBulletSimulatorClient)
│   │   ├── llm_client.py           # Mock / OpenAI / Ollama Brain providers
│   │   ├── chaos_robotics_wrapper.py
│   │   ├── safety_monitor.py
│   │   ├── telemetry_simulator_wrapper.py
│   │   └── experiment_llm_clients.py # Rate limit + planner corruption wrappers
│   ├── services/
│   │   ├── scene_service.py
│   │   ├── control_service.py      # Body: move/grip/release
│   │   ├── agent_service.py        # Brain/Body orchestration
│   │   ├── fault_service.py        # ScriptedFaultInjector
│   │   ├── chaos_telemetry_logger.py
│   │   ├── metrics_service.py
│   │   ├── experiment_matrix.py
│   │   ├── matrix_episode_runner.py
│   │   └── telemetry_harness.py
│   ├── repos/
│   │   └── episode_repository.py   # Append-only episodes CSV
│   └── dashboard/app.py            # Stub (Week 8)
└── tests/                          # 37 pytest tests
```

### 1.2 Conceptual architecture (as implemented)

```text
TaskSpec (NL-derived goal + block + place zone)
        ↓
AgentBrain → LLMClient (Mock | OpenAI | Ollama)
        ↓
ActionPlan (Pydantic validation)
        ↓
AgentService.run_task()  OR  matrix_episode_runner._run_with_phase_hooks()
        ↓
ControlService (optional ScriptedFaultInjector OR none)
        ↓
[Optional stack]
  TelemetrySimulatorWrapper
    → PyBulletChaosRoboticsWrapper (Week 3-4 middleware faults)
      → PyBulletSimulatorClient
        ↓
ChaosTelemetryLogger + PyBulletSafetyMonitor
        ↓
EpisodeRepository (episodes.csv) + experiments_log.csv + experiment_matrix_runs.csv
```

### 1.3 Intended vs actual architecture

| Documented (PRD / comments) | Actual implementation |
|-----------------------------|------------------------|
| LangGraph 2-stage Brain/Body | **Not present.** Simple `AgentService` orchestration |
| Replanning / reflection loop | **Not present.** Single plan, single execution |
| One fault injection layer | **Two systems** (see §4) |
| Week 5 metrics stub | **Implemented** (`ChaosTelemetryLogger`, `MetricsService`) |
| Week 6 full matrix | **Script exists**; pilot data quality insufficient |
| Config-driven experiments (`configs/experiments/*.yaml`) | **Partial.** Only `config/chaos_example.yaml`; matrix is Python-hardcoded |

---

## 2. Current Execution Flow

### 2.1 Scripted baseline (Week 1)

```text
run_scripted_pick_place.py
  → SceneService.build_and_settle()
  → ControlService.run_scripted_pick_place()   # hardcoded waypoints
  → PyBulletSimulatorClient
  → EpisodeRepository → results/episodes.csv
```

### 2.2 Single agent episode (Week 2 CLI)

```text
run_agent_pick_place.py
  → TaskSpec
  → AgentBrain.plan() → LLMClient.generate_action_plan()
  → ActionPlan.validate_raw_plan()
  → AgentService.run_task()
  → ControlService (optional ScriptedFaultInjector)
  → PyBulletSimulatorClient
  → EpisodeRepository
```

### 2.3 Matrix episode (Week 6 batch)

```text
run_experiment_matrix.py
  → generate_matrix_scenarios()   # 75 scenarios by default
  → execute_matrix_episode()
      → seed_deterministic_environment(master_seed)
      → [optional] PyBulletChaosRoboticsWrapper
      → TelemetrySimulatorWrapper + PyBulletSafetyMonitor
      → RateLimitedLLMClient (+ ChaosPlannerLLMClient if planner fault)
      → _run_with_phase_hooks()   # duplicates AgentService.run_task with phase tags
      → ChaosTelemetryLogger.finalize_episode() → experiments_log.csv
      → EpisodeRepository → episodes.csv
      → append_matrix_run_log() → experiment_matrix_runs.csv
  → summarize_matrix_runs() → experiment_matrix_summary.md
```

### 2.4 Agent state flow (critical)

There is **no persistent agent state machine**. State per episode consists of:

| State | Location | Persisted? |
|-------|----------|------------|
| Task goal | `TaskSpec` | In episode result |
| Scene observation | `scene_info` dict passed once at plan time | Embedded in prompts only if LLM uses it |
| Action plan | `ActionPlan` | `action_plan_json` in EpisodeResult |
| Grip handle | Local variable in run loop | No |
| Fault events | Chaos wrapper / injector logs | `execution_log_json` |
| Sim step counters | Chaos wrapper context / telemetry | Telemetry CSV |

**There is no replanning cycle.** After fault injection during execution, the agent does not re-invoke the Brain. Therefore:

- Failures are **not recovered from** at the planning level.
- TTR currently measures time from fault injection to **episode end while nominally successful**, not time to **replan-based recovery**.

---

## 3. Existing Interfaces

### 3.1 Core abstractions (`core/interfaces.py`)

| Interface | Purpose | Implementations |
|-----------|---------|-----------------|
| `SimulatorClient` | Physics backend contract | `PyBulletSimulatorClient`, `FakeSimulator`, `ChaosRoboticsWrapper`, `TelemetrySimulatorWrapper` |
| `AgentPlanner` | High-level planning | `AgentBrain` |
| `FaultInjector` | Control-boundary faults | `ScriptedFaultInjector` only |

**Gap:** `ChaosRoboticsWrapper` implements `SimulatorClient` but **not** `FaultInjector`. The matrix runner bypasses `FaultInjector` entirely.

### 3.2 Brain/Body contract (`schemas/action_plan.py`)

- Actions: `MOVE_TO`, `GRIP`, `RELEASE`
- Structured JSON validated by Pydantic before execution
- LLM output is **data, not code** (good separation)

### 3.3 Chaos middleware hooks (`ChaosRoboticsWrapper`)

| Hook | Fault class | Injection point |
|------|-------------|-----------------|
| `wrap_observation()` | Sensor lag | Pre-planner observation dict |
| `move_end_effector()` | Unreachable IK | Actuation target |
| `step()` | Grip slip | Physics constraint during carry |
| `intercept_planner_output()` | Planner corruption | Raw JSON string pre-validation |

These hooks are **correctly decoupled** from `AgentService`, but **require orchestration wiring** (done in matrix runner, not in default CLI).

### 3.4 Telemetry schema (`schemas/telemetry.py`)

CSV row per episode via `ExperimentTelemetryRecord`. Fields include success, TTR, fault metadata, safety counts, LLM token counts, degradation JSON blob.

**Missing vs research spec:** no immutable JSONL event stream; no explicit `experiment_id`, `failure_reason`, `fault_trigger_time` columns; no versioned environment snapshot.

---

## 4. PyBullet Environment Interface

### 4.1 `PyBulletSimulatorClient` (sole pybullet import site)

| Method | Behavior |
|--------|----------|
| `connect(gui)` | DIRECT or GUI mode |
| `build_scene()` | plane, table, Kuka arm, 3 colored blocks at fixed spawn coords |
| `step(n)` | `p.stepSimulation()` × n |
| `get_block_state(color)` | Base pose |
| `move_end_effector(target)` | IK → joint position control → `MOVE_STEPS` (240) sim steps |
| `grip(color)` | Fixed constraint arm EE ↔ block |
| `release(handle)` | Remove constraint |

### 4.2 Observations (what the agent actually sees)

Produced in scripts as:

```python
scene_info = {"blocks": {color_name: (x, y, z), ...}}
```

**Properties:**

- **Low-dimensional** block positions only (no vision, no joint state, no EE pose)
- Passed **once** before planning
- **Not refreshed** during execution in `AgentService.run_task()`
- `MockLLMClient` **ignores `scene_info` entirely**

This is adequate for pipeline testing but ** insufficient** to study perception-layer fault effects on LLM reasoning.

### 4.3 Action execution

- Open-loop waypoint following via IK
- No collision-aware planning
- Success = final block XY distance to `place_zone` < `PLACE_SUCCESS_TOLERANCE` (0.08 m)

---

## 5. How Plans Are Generated (LLM Entry Points)

| Provider | Entry | Deterministic? | Uses scene_info? |
|----------|-------|----------------|------------------|
| `MockLLMClient` | Hardcoded waypoints from `settings.BLOCK_SPAWN_POSITIONS` | Yes | **No** |
| `OpenAIActionPlanClient` | OpenAI Responses API | **No** (model/API dependent) | Yes (in prompt) |
| `OllamaActionPlanClient` | Local HTTP API | Mostly (temperature=0) | Yes (in prompt) |

**LLM output path:**

```text
LLMClient.generate_action_plan()
  → ActionPlan (or dict/list)
  → ActionPlan.validate_raw_plan()
  → AgentService executes steps
```

**Planner corruption path (matrix only):**

```text
Mock/OpenAI/Ollama → ActionPlan → JSON serialize
  → ChaosRoboticsWrapper.intercept_planner_output()
  → re-parse → validate
```

There is **no LangGraph graph**, **no tool loop**, **no memory**, **no self-diagnosis node**.

---

## 6. Fault Injection: Dual Systems (Technical Debt)

### 6.1 System A — `ScriptedFaultInjector` (ControlService boundary)

Used by: `run_agent_pick_place.py`

| Fault type | Mechanism |
|------------|-----------|
| `TARGET_CORRUPTION` / `POSITION_OFFSET` | Offset `move_to` targets |
| `GRIP_FAILURE` | Block grip acquisition |
| `RELEASE_FAILURE` | Block release |

Does **not** implement PRD fault classes: sensor lag, grip slip, unreachable IK, planner corruption.

### 6.2 System B — `ChaosRoboticsWrapper` (simulator middleware)

Used by: `run_experiment_matrix.py` / `matrix_episode_runner.py`

Implements PRD-aligned fault taxonomy:

| Fault | Layer |
|-------|-------|
| `SENSOR_LAG` | Observation |
| `GRIP_SLIP` | Actuation / physics |
| `UNREACHABLE_IK` | Kinematic |
| `PLANNER_OUTPUT_CORRUPTION` | Planner / communication |

### 6.3 Confounders from dual systems

- README documents System A faults; matrix uses System B
- `FaultType` enum merges both taxonomies (`GRIP_FAILURE` vs `GRIP_SLIP`)
- `AgentService` fault metadata reads `control_service.fault_injector` — **null under chaos wrapper path**
- Comparisons between CLI fault runs and matrix runs are **not directly comparable** without harmonization

---

## 7. Logging, Data, and Reproducibility

### 7.1 Current outputs

| File | Content | Mutable? | Raw/immutable? |
|------|---------|----------|----------------|
| `results/episodes.csv` | EpisodeResult summary + JSON blobs | Append + header migration | No |
| `results/experiments_log.csv` | Telemetry per episode | Append | No |
| `results/experiment_matrix_runs.csv` | Matrix scenario metadata | Append | No |
| `results/experiment_matrix_checkpoint.json` | Completed run IDs | Overwritten | No |
| `results/experiment_matrix_summary.md` | Aggregated markdown table | Overwritten | Derived |

**No JSONL event log.** Detailed traces live inside JSON strings in `episodes.csv` (`execution_log_json`, `action_plan_json`).

### 7.2 Seeding mechanisms

| Layer | Mechanism | Status |
|-------|-----------|--------|
| Block spawn positions | Fixed constants in `settings.py` | ✅ Deterministic |
| Python `random` / NumPy | `seed_deterministic_environment(master_seed)` in matrix | ✅ Called |
| PyBullet | `configure_pybullet_determinism()` (fixed timestep, deterministic pairs) | ⚠️ Best-effort; platform/Bullet build dependent |
| Chaos faults | Per-episode seed derived from `hash(run_id)` | ✅ Deterministic per run_id |
| LLM (OpenAI) | Not seeded | ❌ Non-reproducible |
| Mock agent | Fully deterministic | ✅ |

### 7.3 Reproducibility gaps

- No lockfile / pinned dependency hashes
- No recorded Python/package versions in experiment metadata
- No `experiment_id` linking all artifacts for a batch run
- `results/` CSVs can be appended across different code versions without provenance
- Docker image not validated for headless PyBullet matrix runs
- README still lists Weeks 5–6 as incomplete despite implementation

---

## 8. Existing Tests (37 total)

| Test module | Coverage |
|-------------|----------|
| `test_entities.py` | Domain defaults |
| `test_control_service.py` | PyBullet scripted pick/place integration |
| `test_agent_service.py` | Mock brain, schema validation, fake sim execution |
| `test_fault_service.py` | ScriptedFaultInjector determinism |
| `test_chaos_robotics_wrapper.py` | 4 chaos fault hooks + PyBullet grip slip |
| `test_chaos_telemetry_logger.py` | TTR, classification, safety, MTTR aggregation |
| `test_experiment_matrix.py` | Schedule generation, summary markdown |

### 8.1 Test gaps (research-critical)

- No integration test: full matrix episode end-to-end under mock **and** real LLM
- No test that sensor lag changes mock agent behavior (currently impossible by design)
- No test that TTR reflects replanning (feature absent)
- No calibration test for safety monitor thresholds
- No validation that control vs fault conditions differ only in fault variable
- No statistical analysis tests
- No pilot protocol gate

---

## 9. Existing Assumptions (Explicit and Implicit)

### 9.1 Explicit

- Single manipulation task: pick one block → place zone
- Simulated PyBullet only (no hardware)
- Contribution is **evaluation methodology**, not novel planning algorithms
- Brain/Body separation with validated structured actions

### 9.2 Implicit (often unstated — risky)

| Assumption | Risk if false |
|------------|---------------|
| Mock agent ≈ LLM agent for fault experiments | **High.** Mock ignores observations and always emits scripted plan |
| Single-shot execution can measure "recovery" | **High.** No replan loop exists |
| Sensor lag affects task outcome | **Low with mock agent**; moderate with real LLM if scene_info used |
| 75 episodes sufficient for inference | **Unknown.** No variance analysis or power estimate |
| Safety contact threshold 50 N is meaningful | **Unvalidated.** Pilot shows 120 violations/episode even on success |
| Success tolerance 0.08 m is appropriate | Moderate. Loose placement metric may mask actuation faults |
| One block color (RED) generalizes | Moderate. Matrix hardcodes `BlockColor.RED` |
| Chaos wrapper and scripted injector faults are interchangeable | **Invalid.** Different mechanisms and taxonomies |

---

## 10. Reusable Components (Keep and Build On)

| Component | Research value |
|-----------|----------------|
| Layered `core/` interfaces | Enables swap of sim, agent, faults without rewrite |
| `ActionPlan` Pydantic schema | Clear Brain/Body contract; good for logging |
| `PyBulletSimulatorClient` isolation | Clean physics backend boundary |
| `ChaosRoboticsWrapper` hook design | Correct middleware pattern for fault injection |
| `ChaosConfig` + YAML loader | Configuration-driven faults (extend to full experiments) |
| `ChaosTelemetryLogger` | Solid episode-level metrics skeleton |
| `ExperimentTelemetryRecord` schema | Analysis-ready CSV boundary |
| `run_experiment_matrix.py` + checkpointing | Batch runner with crash isolation |
| `FakeSimulator` | Fast deterministic unit tests |
| Scripted baseline | Essential control condition (Baseline A) |

---

## 11. Technical Debt

| ID | Issue | Severity |
|----|-------|----------|
| TD-01 | LangGraph referenced everywhere but not implemented | **Critical** (documentation honesty) |
| TD-02 | Dual fault injection systems | **High** |
| TD-03 | `matrix_episode_runner._run_with_phase_hooks` duplicates `AgentService.run_task` | **High** |
| TD-04 | `MockLLMClient` ignores observations | **Critical** for perception-fault experiments |
| TD-05 | No replanning / recovery loop | **Critical** for TTR/MTTR claims |
| TD-06 | Safety monitor counts not calibrated | **High** |
| TD-07 | `architecture.md` / README stale | Medium |
| TD-08 | No immutable raw JSONL event store | Medium |
| TD-09 | Matrix parameters hardcoded in Python, not YAML experiment configs | Medium |
| TD-10 | `results/` overwritten summary + append logs without batch provenance | Medium |
| TD-11 | Legacy fault types mixed with PRD types in enum | Medium |
| TD-12 | No `analysis/` pipeline directory; only one script | Medium |
| TD-13 | Docker stub not aligned with matrix runner | Low (Week 7) |

---

## 12. Experimental Risks

### 12.1 Internal validity (does the experiment test what we claim?)

| Risk | Description |
|------|-------------|
| **R-01 Agent not LLM-driven in default matrix** | Mock agent dominates test runs; degradations reflect physics layer only |
| **R-02 No recovery loop** | Measuring TTR/MTTR implies recovery; agent cannot recover via replanning |
| **R-03 Observation faults don't affect planner (mock)** | Sensor lag wired but planner uses spawn constants |
| **R-04 Planner corruption → validation failure, not graceful degradation** | Corrupted plans often hard-fail Pydantic parse; this measures parser rejection, not in-task robustness |
| **R-05 Unreachable IK always fails open-loop** | Large offsets may make success rate ≈ 0 regardless of agent intelligence |
| **R-06 Single block / single task** | Limited generalization |
| **R-07 Grip slip may not fire if phase hooks misaligned** | Event-triggered faults depend on duplicated runner logic |

### 12.2 Construct validity (do metrics mean what we say?)

| Risk | Description |
|------|-------------|
| **R-08 TTR ≠ recovery time** | Currently time from fault to episode end, not to successful replan |
| **R-09 MTTR reported from sparse/invalid TTR samples** | Many episodes yield TTR=0, None, or inf |
| **R-10 Fault classification accuracy unmeasurable** | No self-diagnosis / reflection step exists |
| **R-11 Safety violations constant across conditions** | Suggests metric captures nominal contact, not unsafe events |
| **R-12 Degradation curve with zero LLM tokens (mock)** | Token-based degradation meaningless for mock runs |

### 12.3 External validity (generalization)

| Risk | Description |
|------|-------------|
| **R-13 PyBullet ≠ real robot dynamics** | Expected limitation; must be stated clearly |
| **R-14 Single arm / table / blocks scene** | Narrow embodiment |
| **R-15 OpenAI results non-reproducible** | Cannot claim exact replication with API models |

---

## 13. Reproducibility Risks

1. **Unpinned dependencies** — `pyproject.toml` uses minimum versions, not exact pins.
2. **No experiment manifest** — Cannot reconstruct "matrix run X" from config file alone.
3. **Append-only CSV without run batch ID** — Mixing pilot and production data undetectable.
4. **Summary markdown overwritten** — Only latest aggregate survives.
5. **Platform-specific PyBullet wheels** (`pybullet` vs `pybullet-arm64`).
6. **LLM API nondeterminism** not logged per episode (model version, prompt hash, temperature).

---

## 14. Research Validity Risks (What Would Invalidate Paper Claims)

| If we claimed… | Current support | Verdict |
|----------------|-----------------|---------|
| "LangGraph agent under fault injection" | No LangGraph | **Unsupported — do not claim** |
| "Agent recovers via replanning" | No replan loop | **Unsupported** |
| "Sensor faults degrade LLM planning" | Mock ignores scene | **Unsupported with default agent** |
| "Production-style MTTR for embodied agents" | TTR definition doesn't match recovery | **Partially supported; redefine or add replan** |
| "Fault-class distinguishable failure signatures" | Possible at physics layer; not validated statistically | **Hypothesis only** |
| "Concurrent faults worse than isolated" | Mechanism exists; no valid pilot analysis yet | **Hypothesis only** |
| "50+ episode matrix with quantified degradation" | Script supports it; existing summary is smoke test | **Not yet scientifically valid** |

### 14.1 Observed pilot artifacts (DO NOT cite as findings)

From `results/experiment_matrix_summary.md` (14-episode smoke configuration):

- Control and sensor_lag: 100% success (mock agent, physics-only)
- grip_slip: 100% success at all intensities (slip may not prevent place success under tolerance)
- unreachable_ik: 0% success (expected — offsets break IK)
- planner_output_corruption: 0% success (plan validation failure, not execution failure)
- Safety violations ≈ 120 for most conditions → metric saturation

**These numbers validate plumbing, not research hypotheses.**

---

## 15. Mapping to Research Questions (RQ1–RQ5)

| RQ | Can current system answer? | Blocker |
|----|----------------------------|---------|
| **RQ1** Degradation under controlled failures | **Partially** (actuation/kinematic layer only with mock) | Need real LLM + observation-dependent planner |
| **RQ2** Distinguishable failure signatures | **Unknown** | Need labeled failure taxonomy + qualitative traces + stats |
| **RQ3** Intensity vs success/time/recovery/safety | **Partially** | Recovery and safety metrics unreliable |
| **RQ4** Concurrent vs isolated faults | **Mechanism ready** | Need valid baselines and sample size rationale |
| **RQ5** Production-style reliability metrics useful? | **Open question** | MTTR/TTR need semantic alignment with actual recovery |

All three outcomes (positive, negative, inconclusive) remain possible — but **inconclusive is the likely outcome** if experiments proceed without addressing blockers.

---

## 16. Baselines (Current State)

| Baseline | Implemented? | Used in matrix? | Notes |
|----------|--------------|-----------------|-------|
| **A: Scripted pick/place** | ✅ `ControlService.run_scripted_pick_place` | ❌ Not in matrix runner | Required for fair comparison |
| **B: LLM agent, no fault** | ✅ | ✅ (control group) | Uses mock by default — not true LLM baseline |
| **C: LLM agent + faults** | ✅ | ✅ | |
| Recovery disabled / replan disabled ablation | ❌ | — | Cannot ablate what doesn't exist |

**Critical gap:** Matrix control group is "agent without fault wrapper," not "scripted baseline." Comparing fault runs to **scripted** and **no-fault agent** separately is necessary.

---

## 17. Recommended Architecture Before Experimentation

### 17.1 Phase 0 — Documentation and honesty (no code required first)

Create (per research notebook plan):

- `docs/RESEARCH_QUESTIONS.md` — RQ1–RQ5 with falsifiable hypotheses
- `docs/FAULT_MODEL.md` — unified taxonomy table
- `docs/METRICS.md` — formal definitions (TTR vs MTTR)
- `docs/EXPERIMENT_PROTOCOL.md` — pilot → full matrix gate criteria
- `docs/FAILURE_ANALYSIS.md` — failure mode taxonomy
- `docs/LIMITATIONS.md` — simulation, agent, metric limits
- `docs/RELATED_WORK.md` — verified citations only

Update `README.md` and `docs/architecture.md` to reflect **actual** agent (Brain/Body, not LangGraph) unless LangGraph is implemented.

### 17.2 Phase A — Pilot (5–10 episodes/condition)

**Gate criteria before full matrix:**

1. Fault injection confirmed via `fault_events` logs (fault_applied=true at expected hook/step)
2. Control vs fault conditions differ **only** in fault config (same agent, task, seed policy)
3. Safety metric discriminates nominal vs intentionally unsafe episode (calibrate threshold)
4. TTR definition matches implemented agent capabilities **or** replan loop added
5. Real LLM agent (`openai` or `ollama`) used for at least one pilot condition
6. Scripted baseline recorded under same telemetry schema

### 17.3 Phase B — Minimal architectural fixes (smallest coherent changes)

Priority order:

1. **Unify fault injection** behind one controller interface consumed by both CLI and matrix
2. **Remove or isolate `matrix_episode_runner` duplication** — extend `AgentService` via hooks/callbacks instead of copying run loop
3. **Add optional replan loop** (even 1 retry) OR **redefine TTR** as time-to-task-completion-after-fault without claiming "recovery"
4. **Make mock agent optionally observation-aware** for pilot tests, or **disallow mock** for paper experiments
5. **Add experiment YAML** (`configs/experiments/pilot.yaml`) as single source of truth
6. **Add immutable raw log** (`results/raw/<experiment_id>/episodes.jsonl`)
7. **Calibrate `PyBulletSafetyMonitor`** — deduplicate contacts, ignore nominal table support forces
8. **Include scripted baseline in matrix** as explicit arm

### 17.4 Phase C — Full matrix (only after pilot pass)

- Episode count determined from pilot variance (report mean, std, CI width target)
- Separate batches by `experiment_id` with provenance metadata
- Analysis pipeline regenerates all figures from raw data
- No overwriting summary without archiving prior run

### 17.5 Target data flow (research-grade)

```text
configs/experiments/<id>.yaml
        ↓
ExperimentRunner (validates config, records provenance)
        ↓
Agent (LLM) ←→ FaultController ←→ InstrumentedSimulator
        ↓
JSONL raw events (immutable) + CSV derived metrics
        ↓
analysis/ (pandas) → figures/ + tables/
        ↓
dashboard/ (read-only, post-validation)
```

Adapt folder names to existing `src/alpha/` layout rather than literal restructure — e.g.:

- `configs/experiments/` (new)
- `src/alpha/services/experiment_runner.py` (evolve from matrix)
- `analysis/` (new)
- `results/raw/` (new)

---

## 18. What Should NOT Be Done Next

1. **Do not** publish or cite current `results/experiment_matrix_summary.md` as experimental findings.
2. **Do not** claim LangGraph, replanning, or self-diagnosis without implementation.
3. **Do not** run 75+ episodes with `MockLLMClient` and interpret as LLM embodied-agent reliability.
4. **Do not** merge more features (dashboard, paper) before pilot validation.
5. **Do not** rewrite working Week 1–2 code for stylistic reasons.

---

## 19. Recommended Immediate Next Step

**Execute Phase A pilot protocol** (document in `docs/EXPERIMENT_PROTOCOL.md`):

1. Fix safety metric calibration.
2. Run 5 episodes × {scripted, mock-agent control, mock+sensor_lag, mock+grip_slip, real-LLM control, real-LLM+sensor_lag}.
3. Inspect raw `fault_events`, success criteria, and safety counts manually.
4. Decide: add minimal replan loop **or** narrow paper claims to open-loop fault tolerance.
5. Only then scale episode count and lock experiment YAML.

---

## 20. Audit Checklist (Completed)

| # | Item | Status |
|---|------|--------|
| 1 | Inspect entire repository | ✅ |
| 2 | Identify current architecture | ✅ §1 |
| 3 | Identify agent state flow | ✅ §2.4 |
| 4 | Identify PyBullet interface | ✅ §4 |
| 5 | Identify observation production | ✅ §4.2 |
| 6 | Identify plan generation | ✅ §5 |
| 7 | Identify LLM entry points | ✅ §5 |
| 8 | Identify logging / reproducibility | ✅ §7 |
| 9 | Identify existing tests | ✅ §8 |
| 10 | Identify architectural weaknesses | ✅ §11 |
| 11 | Identify scientific invalidity risks | ✅ §12–14 |

---

## Appendix A — File / Component Index

| Concern | Primary files |
|---------|---------------|
| Physics | `src/alpha/clients/simulator_client.py` |
| Agent orchestration | `src/alpha/services/agent_service.py` |
| Body control | `src/alpha/services/control_service.py` |
| Chaos faults | `src/alpha/clients/chaos_robotics_wrapper.py`, `src/alpha/models/chaos_config.py` |
| Legacy faults | `src/alpha/services/fault_service.py` |
| Telemetry | `src/alpha/services/chaos_telemetry_logger.py` |
| Matrix batch | `scripts/run_experiment_matrix.py`, `src/alpha/services/matrix_episode_runner.py` |
| Analysis | `scripts/analyze_experiments.py`, `src/alpha/services/metrics_service.py` |
| Tests | `tests/test_*.py` (8 modules, 37 tests) |

## Appendix B — Glossary Alignment

| Term | Current implementation meaning | Production reliability meaning | Alignment |
|------|-------------------------------|--------------------------------|-----------|
| **TTR** | Sim steps from first fault to episode end (if success) | Time until service restored after fault | ⚠️ Misaligned without replan |
| **MTTR** | Mean of finite TTR across episodes | Mean time to restore service | ⚠️ Depends on TTR definition |
| **Fault injection** | Middleware hooks + control-boundary injector | Controlled production fault | ✅ Concept aligned |
| **Agent** | Single-shot LLM planner | Often includes monitor/replan loop | ⚠️ Partial |

---

*End of audit. Next deliverable: `docs/EXPERIMENT_PROTOCOL.md` and Phase A pilot — not additional fault injector implementation.*
