# Experiment Protocol

**Project:** Chaos Engineering Harness for Embodied AI Agents (Alpha)  
**Document version:** 0.2 (Phase A pilot — protocol frozen)  
**Status:** Phase A — pilot/calibration study authorized **after** validation gates; pilot data are **not** final study evidence  
**Related documents:** `docs/RESEARCH_ENGINEERING_AUDIT.md`, `docs/PILOT_VALIDATION_CHECKLIST.md`, `docs/FAULT_INJECTION.md`, `docs/SAFETY_TAXONOMY.md`, `PRD.md`, `docs/architecture.md`  
**Last updated:** 2026-08-13

---

## Document purpose

This protocol defines **how** experiments will be conducted **before** any pilot or full-matrix data are collected for publication. It is a scientific notebook entry, not a results report.

**Explicit exclusions for this milestone:**

- No fault-injector implementation in this step
- No pilot execution in this step
- No full experiment matrix execution in this step
- No fabricated or assumed experimental outcomes

Any data under `results/dev_smoke/` from development smoke runs are **not** pilot or final experimental data and must not be cited as findings. Pilot output belongs under `results/pilot/` and `data/raw/pilot/` only after gate validation.

---

## 1. Research objective

### 1.1 Central research objective

To evaluate whether **production-style fault injection and reliability metrics** can be applied systematically to an **LLM-driven embodied manipulation agent** in simulation, and to characterize how **controlled failures at different system layers** affect **task outcome, execution cost, post-fault completion interval (where valid), and validated safety-related events**.

The current agent is **single-shot** (one plan, open-loop execution). There is **no replanning loop**. Metrics labeled recovery, TTR, or MTTR are **not** valid primary outcomes in Phase A.

This project does **not** claim a novel robot-planning algorithm or foundation model. The intended contribution is a **reproducible evaluation methodology and software framework** for studying failure behavior in embodied AI agents.

### 1.2 Research questions

The following questions bound what the current system can attempt to measure. They are **not** assumed to be answerable affirmatively.

| ID | Question | What the current system can measure | What it cannot yet measure |
|----|----------|--------------------------------------|----------------------------|
| **RQ1** | How does an LLM-driven embodied agent degrade under controlled failures at different layers? | Task success, completion time, execution steps, layer-specific fault events (when chaos middleware is wired) | Full-stack degradation with mock agent that ignores observations; cross-layer causal attribution without trace analysis |
| **RQ2** | Do different fault classes produce distinguishable failure signatures? | Per-episode execution logs, fault event records, failure taxonomy labels | Statistically validated separability (requires pilot + adequate N) |
| **RQ3** | How does fault intensity affect success, execution time, post-fault completion interval, and safety? | Success rate, wall/sim time, `post_fault_completion_time_*`, safety violation counts by intensity scalar | Recovery/replanning behavior (no replan loop exists) |
| **RQ4** | How does the agent behave under concurrent vs isolated faults? | Comparative outcomes when `InjectionMode.CONCURRENT` is used (Phase A: F1+F2 only) | Isolated comparison unless confounders (dual fault activation timing) are controlled |
| **RQ5** | Can production-style reliability metrics usefully characterize embodied-agent robustness? | Post-fault completion time (conditional), episode-level safety violation rate, degradation vs intensity | TTR/MTTR as recovery metrics; industry-calibrated MTTR; fleet-scale reliability |

**Valid outcomes:** positive findings, negative findings, and **inconclusive** findings are all scientifically acceptable.

---

## 2. Hypotheses

Hypotheses are **testable statements**, not expected results. Each may be **supported, refuted, or left inconclusive** by the data.

| ID | Hypothesis | Notes / falsification |
|----|------------|------------------------|
| **H1** | Increasing fault intensity will generally reduce task success and/or increase execution cost (time, steps, or LLM token usage where applicable). | Falsified if success and cost are flat across intensity levels after calibration. Inconclusive if variance is too high or N too small. |
| **H2** | Different fault classes will produce different observable failure signatures (telemetry events, failure categories, success patterns). | Falsified if failure distributions are indistinguishable. Requires manual trace review plus aggregate stats. |
| **H3** | Concurrent faults will produce greater degradation than isolated faults under comparable intensity and agent conditions. | Falsified if concurrent conditions perform equal to or better than isolated. Confounded if fault triggers are not aligned. |
| **H4** | Agent recovery/replanning will mitigate some injected failure classes but not others. | **DEFERRED — not testable in Phase A.** The current system has **no replan/recovery loop**. H4 may become a secondary hypothesis only after genuine replanning is implemented. Do not collect or report H4 evidence from pilot or single-shot runs. |
| **H5** | Production-style metrics (post-fault completion time where valid, safety violation rate, degradation curves) will correlate with fault type and intensity in interpretable ways. | Falsified if metrics are constant, saturated, or uncorrelated with injected faults. Do not use TTR/MTTR labels. |

**Explicit statement:** The experiments may **fail to support any hypothesis**. That outcome must be reported honestly.

---

## 3. System under test

This section describes the **actual** repository state as of the research audit. Values are taken from `src/alpha/config/settings.py` and related modules unless marked otherwise.

### 3.1 Simulation environment

| Parameter | Value | Source |
|-----------|-------|--------|
| Engine | PyBullet (headless `DIRECT` mode for experiments) | `PyBulletSimulatorClient` |
| Gravity | `(0, 0, -9.8)` m/s² | `settings.GRAVITY` |
| Fixed simulation timestep (when determinism helper applied) | `1/240` s (~4.17 ms) | `configure_pybullet_determinism()`, `ChaosConfig.sim_timestep` default |
| Scene settle steps after build | `120` | `settings.SETTLE_STEPS` |
| Sim steps per `move_end_effector` call | `240` | `settings.MOVE_STEPS` |

**PyBullet determinism:** Best-effort via `fixedTimeStep`, `deterministicOverlappingPairs=1`, `useSplitImpulse=1`, and `setRandomSeed(0)` when available. **Full cross-platform determinism is not guaranteed** — TO BE CONFIRMED FROM IMPLEMENTATION during pilot Gate 6.

### 3.2 Robot configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Arm URDF | `kuka_iiwa/model.urdf` (PyBullet data) | `settings.ARM_URDF` |
| Arm base position | `(0.0, 0.0, 0.0)` | `settings.ARM_BASE_POSITION` |
| Table URDF | `table/table.urdf` | `settings.TABLE_URDF` |
| Table base position | `(0.6, 0.0, 0.0)` | `settings.TABLE_BASE_POSITION` |
| End-effector control | Inverse kinematics → joint position control | `PyBulletSimulatorClient.move_end_effector()` |
| IK max iterations | `200` | `simulator_client.py` |
| Joint control force | `2000` | `simulator_client.py` |

### 3.3 Objects

| Object | URDF / scaling | Spawn positions (x, y, z) |
|--------|----------------|---------------------------|
| Block (red) | `cube_small.urdf`, globalScaling `1.2` | `(0.5, -0.15, 0.68)` |
| Block (green) | same | `(0.5, 0.0, 0.68)` |
| Block (blue) | same | `(0.5, 0.15, 0.68)` |
| Place zone (target) | N/A — coordinate only | `(0.6, 0.35, 0.65)` |

**Pilot default task block:** The Week 6 matrix runner hardcodes `BlockColor.RED`. Multi-block generalization is **not** in the current matrix — TO BE CONFIRMED whether pilot expands to green/blue.

### 3.4 Manipulation task

| Field | Value |
|-------|-------|
| Task type | Pick-and-place |
| Goal | Move specified block to `place_zone` |
| Default description | `"pick block and move it to the place zone"` |
| Success criterion | Final block position within Euclidean distance `< 0.08` m of `place_zone` |
| Success tolerance | `settings.PLACE_SUCCESS_TOLERANCE = 0.08` |
| Release height offset | `0.06` m above place zone Z | `settings.PLACE_RELEASE_HEIGHT_OFFSET` |

### 3.5 Agent architecture

| Component | Implementation | Present? |
|-----------|----------------|----------|
| Orchestration | `AgentService` (Brain plans, Body executes) | Yes |
| Brain | `AgentBrain` → `LLMClient` | Yes |
| Body | `ControlService` → `SimulatorClient` | Yes |
| LangGraph | — | **No — not implemented; Brain/Body `AgentService` only** |
| Replanning / reflection loop | — | **No — single-shot agent** |
| Self-diagnosis / fault classification by agent | — | **No** (telemetry field exists but agent does not populate it) |

**Planning model:** Single-shot: one plan generated, then executed open-loop.

### 3.6 LLM / planner providers

| Provider ID | Class | Deterministic? | Uses `scene_info`? |
|-------------|-------|----------------|---------------------|
| `mock` | `MockLLMClient` | Yes | **No** — engineering/tests only; **excluded from formal results** |
| `openai` | `OpenAIActionPlanClient` | No (API/model dependent) | Yes (in prompt) |
| `ollama` | `OllamaActionPlanClient` | Partial (`temperature=0`) | Yes (in prompt) — **not used in Phase A primary matrix** |

**Default for CI/dev:** `mock` (engineering only).  
**Formal Phase A pilot and primary study:** **`openai` only** (`configs/experiments/pilot.yaml`).

**Pilot model:** `gpt-4o` (pilot config). Matrix dev script default may differ; formal runs must record `model_identifier`.

### 3.7 Observation interface

Observations passed to the planner:

```python
scene_info = {"blocks": {color_name: (x, y, z), ...}}
```

| Property | Status |
|----------|--------|
| Modalities | Block positions only (no vision, no joint state, no EE pose) |
| Update frequency | **Once** before planning; not refreshed during execution in `AgentService.run_task()` |
| Sensor lag hook | `ChaosRoboticsWrapper.wrap_observation()` (matrix runner only) |

### 3.8 Action interface

Validated structured plan (`ActionPlan` schema):

| Action | Required fields |
|--------|-----------------|
| `MOVE_TO` | `target: (x, y, z)` |
| `GRIP` | `block: BlockColor` |
| `RELEASE` | none |

Execution is sequential; no parallel actions.

### 3.9 Fault injection (existing, not modified in this protocol step)

Two implementations coexist (see `docs/FAULT_INJECTION.md`):

1. **`ScriptedFaultInjector`** — legacy ControlService boundary (`run_agent_pick_place.py`); CLI/compatibility only
2. **`ChaosRoboticsWrapper`** — **canonical** simulator middleware (`execute_matrix_episode`, pilot runner)

**Primary fault path for this protocol:** `ChaosRoboticsWrapper` only. Legacy injector is out of scope for pilot and formal study.

### 3.10 Termination conditions

| Condition | Definition |
|-----------|------------|
| Normal completion | All plan actions executed; settle step (`60` sim steps) applied |
| Task success / failure | Evaluated post-execution via block distance to place zone |
| Crash | Uncaught exception during episode (logged as `crashed=True` in telemetry) |
| Timeout | Field exists (`timed_out`) but episode timeout policy — **TO BE CONFIRMED FROM IMPLEMENTATION** (not enforced in `AgentService` today) |

---

## 4. Baseline conditions

### 4.1 BASELINE 0 — Scripted non-agentic pick-and-place

| Field | Value |
|-------|-------|
| Entry point | `scripts/run_scripted_pick_place.py` / `ControlService.run_scripted_pick_place()` |
| Agent | None — hardcoded waypoint sequence |
| Fault injection | None |
| Purpose | Physical/simulation feasibility ceiling; isolates environment and Body primitives from planner failures |

**Why it exists:** Without Baseline 0, improvements or degradations cannot be separated from "the task is impossible in sim" vs "the agent/fault layer failed."

### 4.2 BASELINE 1 — LLM-driven agent, zero injected faults

| Field | Value |
|-------|-------|
| Entry point | `run_agent_pick_place.py --fault none` or matrix `scenario_group=control` |
| Agent | Selected LLM provider (mock, openai, ollama) |
| Fault injection | None (no chaos wrapper, or wrapper with empty fault list) |
| Purpose | Agent capability under nominal conditions |

**Why it exists:** Experimental fault conditions must be compared against the **same agent and task** without faults.

### 4.3 EXPERIMENTAL — LLM-driven agent + controlled faults

| Field | Value |
|-------|-------|
| Entry point | Matrix runner with `ChaosRoboticsWrapper` |
| Faults | One of four PRD classes, or concurrent combination per §5 |
| Purpose | Measure degradation under controlled layer-specific failures |

### 4.4 Equivalence requirements

For scientifically valid comparison, the following **must match** across Baseline 1 and Experimental conditions unless the independent variable requires otherwise:

| Variable | Policy |
|----------|--------|
| Task definition (`TaskSpec`) | Same block target and `place_zone` |
| Initial block spawn coordinates | Same constants (`settings.BLOCK_SPAWN_POSITIONS`) |
| Master random seed (Python/NumPy) | Same `master_seed` per protocol §8 |
| Simulation build/settle procedure | Same `SceneService.build_and_settle()` |
| Agent provider and model settings | Same provider; document model ID |
| Success tolerance | Same `PLACE_SUCCESS_TOLERANCE` |

**Known exception:** Fault conditions intentionally differ in fault configuration — that is the independent variable.

**Resolved:** Matrix and pilot runners use `AgentService.run_task(hooks=...)` — baseline and fault arms share the same orchestration path.

---

## 5. Fault taxonomy

Four **primary** fault classes for the formal study. Implementation references describe **existing design intent** in `ChaosRoboticsWrapper`; this protocol does not implement them.

### 5.1 Fault summary table

| Fault ID | Layer | Injection point | Mechanism (existing design) |
|----------|-------|-----------------|----------------------------|
| **F1** `sensor_lag` | Sensor / observation | `wrap_observation()` before planner | Return stale cached observation |
| **F2** `grip_slip` | Actuation / physical | `step()` while grip active | Remove/weaken fixed constraint |
| **F3** `unreachable_ik` | Kinematic / environmental | `move_end_effector()` target | Apply directional offset to EE target |
| **F4** `planner_output_corruption` | Planner / communication | `intercept_planner_output()` | Delete JSON keys or inject malformed chars |

Legacy fault types (`grip_failure`, `position_offset`, etc.) in `FaultType` enum are **not** part of the primary matrix.

### 5.2 F1 — Sensor / observation fault (`sensor_lag`)

| Field | Specification |
|-------|---------------|
| **System layer** | Perception / observation stream |
| **Injection point** | `ChaosRoboticsWrapper.wrap_observation(state_dict)` |
| **Mechanism** | Maintain observation history; return snapshot from `lag_seconds` ago |
| **Intensity parameter** | `lag_seconds` (seconds of simulated elapsed time) |
| **Trigger** | `immediate`, `time` (`after_seconds`), or `event` (`phase`, `step`) per `TriggerConfig` |
| **Duration** | Per observation call while trigger active; not a one-shot unless configured |
| **Expected manifestation** | Planner receives outdated block positions |
| **Observable telemetry** | `fault_events` with `hook=wrap_observation`, `lag_seconds`; degraded plan quality (LLM agents only) |
| **Potential confounders** | Mock agent ignores `scene_info` → fault has **no effect on planning**; observation passed only once |
| **Safety implications** | Indirect — stale state may cause collisions or missed grasps |

### 5.3 F2 — Actuation / grip-slip fault (`grip_slip`)

| Field | Specification |
|-------|---------------|
| **System layer** | Actuation / physics coupling |
| **Injection point** | `ChaosRoboticsWrapper.step()` when `_grip_active` |
| **Mechanism** | With probability `slip_probability`, remove constraint (`weaken_force_factor=0`) or weaken via PyBullet |
| **Intensity parameter** | `slip_probability` ∈ [0, 1]; optional `weaken_force_factor` |
| **Trigger** | Matrix default: `event`, `phase=lift` (set by phase hooks in matrix runner) |
| **Duration** | Instantaneous constraint drop per triggering step |
| **Expected manifestation** | Block detaches during carry phase |
| **Observable telemetry** | `fault_events` with `hook=step`, `constraint_removed`; block pose divergence |
| **Potential confounders** | Success tolerance may still mark task successful if block lands near zone by chance; phase hooks must align with actual carry |
| **Safety implications** | Dropped object; out-of-workspace monitor may fire |

### 5.4 F3 — Kinematic / unreachable-target fault (`unreachable_ik`)

| Field | Specification |
|-------|---------------|
| **System layer** | Kinematic / motion execution |
| **Injection point** | `ChaosRoboticsWrapper.move_end_effector()` |
| **Mechanism** | Add offset vector to commanded target: `offset_magnitude` along `offset` direction |
| **Intensity parameter** | `offset_magnitude` (meters); direction `offset` (unit-scaled, default `(1,0,0)`) |
| **Trigger** | Configurable; matrix default: `immediate` |
| **Duration** | Per affected move command |
| **Expected manifestation** | EE reaches wrong pose; increased IK error; missed grasp or place |
| **Observable telemetry** | `requested_target` vs `executed_target` in execution log; `fault_events` on `move_end_effector` |
| **Potential confounders** | Large offsets cause immediate task failure — intensity must be calibrated (§6) |
| **Safety implications** | Arm may reach toward workspace bounds; collision monitor may fire |

### 5.5 F4 — Planner/schema robustness (`planner_output_corruption`)

F4 measures **planner output and schema validation robustness**, not general closed-loop planner degradation under in-task replanning.

| Field | Specification |
|-------|---------------|
| **System layer** | Planner / communication |
| **Injection point** | `intercept_planner_output(raw_json_string)` via `ChaosPlannerLLMClient` |
| **Mechanism** | Delete keys (`actions`, `target`, `block`) or inject malformed JSON |
| **Intensity parameter** | `corruption_probability`; `inject_malformed_chars` (bool); `keys_to_delete` |
| **Trigger** | Configurable; matrix default: `immediate` |
| **Duration** | One-shot at plan time |
| **Expected manifestation** | Validation failure or structurally invalid plan |
| **Observable telemetry** | Parse/validation errors; `fault_events` on `intercept_planner_output` |
| **Potential confounders** | Hard validation failure ≠ in-task graceful degradation; measures parser robustness, not closed-loop recovery |
| **Safety implications** | Low direct risk — episode may abort before motion |

### 5.6 Concurrent fault condition

| Field | Specification |
|-------|---------------|
| **Combination (primary study)** | `sensor_lag` + `grip_slip` |
| **Mode** | `InjectionMode.CONCURRENT` |
| **Purpose** | RQ4 — overlapping failures at observation and actuation layers |

Additional concurrent combinations are **OPTIONAL** secondary experiments (§16).

---

## 6. Fault intensity

### 6.1 Intensity representation

Intensity is defined at **two levels**:

1. **Normalized scalar** (for telemetry grouping): `0.33` (LOW), `0.66` (MEDIUM), `1.0` (HIGH) — from `INTENSITY_SCALAR` in `experiment_matrix.py`
2. **Fault-specific physical/semantic parameters** mapped from the scalar

### 6.2 Proposed parameter mapping (existing code — subject to pilot calibration)

These values are **starting points**, not validated experimental constants:

| Label | Scalar | F1 `lag_seconds` | F2 `slip_probability` | F3 `offset_magnitude` (m) | F4 `corruption_probability` |
|-------|--------|------------------|------------------------|---------------------------|-------------------------------|
| LOW | 0.33 | 0.4975 | 0.4475 | 0.1856 | 0.464 |
| MEDIUM | 0.66 | 0.745 | 0.695 | 0.2912 | 0.728 |
| HIGH | 1.0 | 1.0 | 0.99 | 0.40 | 1.0 |

Formulas (from `intensity_params_for_fault()`):

- F1: `lag_seconds = 0.25 + 0.75 × scalar`
- F2: `slip_probability = min(0.99, 0.2 + 0.75 × scalar)`
- F3: `offset_magnitude = 0.08 + 0.32 × scalar`
- F4: `corruption_probability = min(1.0, 0.2 + 0.8 × scalar)`; `inject_malformed_chars = (label != LOW)`

### 6.3 Calibration procedure (Phase A — mandatory)

Before accepting intensity labels for the full matrix, the pilot must classify each fault × intensity cell as:

| Classification | Criterion |
|----------------|-----------|
| **Negligible** | No measurable change vs Baseline 1 on success, time, or fault events |
| **Measurable degradation** | Significant change in ≥1 primary metric without trivial instant failure |
| **Severe / invalid** | Immediate crash, 0% success with no informative variance, or fault does not activate |

**Action rules:**

- **Negligible** → increase intensity parameter or fix injection wiring; do not use for full study until measurable
- **Measurable** → candidate for full matrix
- **Severe / invalid** → reduce intensity or redesign fault; exclude from matrix until fixed

**Protocol does not fix final numerical ranges** until pilot calibration completes.

---

## 7. Experiment variables

### 7.1 Independent variables

| Variable | Levels (initial) |
|----------|------------------|
| Fault type | none, F1, F2, F3, F4, concurrent (F1+F2) |
| Fault intensity | LOW, MEDIUM, HIGH (parameterized per §6) |
| Fault timing / phase | `immediate`, `time`, `event` (phase: `pick`, `lift`, `place`) |
| Concurrent vs isolated | Isolated single fault vs concurrent mode |

### 7.2 Dependent variables

| Variable | Source | Validity note |
|----------|--------|---------------|
| Task success | `EpisodeResult.success` / telemetry `success` | Valid |
| Completion time | `total_wall_seconds`, sim steps | Valid |
| Post-fault completion time | `post_fault_completion_time_sim_steps`, `post_fault_completion_time_wall_seconds` | Valid **only** when fault occurred and episode succeeded; `null` otherwise — see §12.3 |
| Failure reason | `failure_reason` | Present in telemetry |
| Safety violations (primary) | Episode-level rate: proportion with `safety_violation_count ≥ 1` | Validated taxonomy — see `docs/SAFETY_TAXONOMY.md` |
| Safety violations (secondary) | Mean `safety_violation_count` per episode | Same taxonomy |
| Replanning events | `replan_count` | Always 0 — no replan loop |
| Execution steps | `steps_taken`, `total_sim_steps` | Valid |
| LLM token usage | `llm_total_tokens` | Valid for real LLM; 0 for mock |
| Fault occurred | `fault_occurred`, fault event log | Valid if Gate 1 passes |
| Fault classification accuracy | `fault_classification_correct` | **Not applicable** until agent self-diagnosis exists |

### 7.3 Controlled variables

| Variable | Control method |
|----------|----------------|
| Task definition | Fixed `TaskSpec` per experiment batch |
| Robot / URDF | `settings.ARM_URDF`, etc. |
| Environment layout | Fixed spawn constants |
| Initial state | Same seed policy + settle steps |
| Simulation parameters | `configure_pybullet_determinism()` |
| Agent provider / model | Fixed per batch; recorded in metadata |
| LLM temperature | `0` for Ollama; OpenAI settings TO BE CONFIRMED |
| Random seed policy | §8 |

### 7.4 Potential confounders

| Confounder | Mitigation |
|------------|------------|
| LLM nondeterminism | Record model ID, prompt hash, temperature; prefer fixed mock for engineering validation only |
| API latency / rate limits | `RateLimitedLLMClient`; record wall time separately from sim time |
| Model version drift | Record exact model string and date; pin for final batch |
| Simulation nondeterminism | Fixed seeds, deterministic Bullet params, same platform when possible |
| Initial-state differences | Fixed spawn coords; verify Gate 6 |
| Mock agent ignores observations | **Do not use mock for RQ1/RQ2 LLM claims** |
| Dual fault injectors | Use chaos wrapper only for formal study |
| Orchestration code path divergence | Unify matrix and single-episode runners before full study |
| Network failures | Retry policy; mark episode `crashed` with reason |
| Safety metric saturation | Calibrate threshold in pilot |

---

## 8. Randomization and reproducibility

### 8.1 Seed hierarchy

| Layer | Policy | Reproducible? |
|-------|--------|---------------|
| **Master seed** | Fixed integer per experiment batch (default `42` in matrix script) | Yes |
| **Python `random`** | `random.seed(master_seed)` at episode start | Yes |
| **NumPy** | `np.random.seed(master_seed)` at episode start | Yes |
| **Chaos fault RNG** | `episode_seed = master_seed + (abs(hash(run_id)) % 10000)` | Yes per `run_id` |
| **PyBullet** | `configure_pybullet_determinism()`; `setRandomSeed(0)` if available | Best-effort |
| **Block spawn positions** | Deterministic constants (not sampled) | Yes |
| **LLM output** | Provider-dependent | **No** for OpenAI; partial for Ollama `temperature=0` |

### 8.2 Simulation vs LLM determinism

| Subsystem | Expectation |
|-----------|-------------|
| **Simulation determinism** | Same master seed + same code path should yield identical initial scene and, under best-effort Bullet settings, highly similar physics **on the same platform/build** |
| **LLM/API determinism** | **Not guaranteed.** Exact replication of OpenAI outputs across time is not claimed. Experiments using real LLMs must report model identifier, date, and acknowledge stochasticity |

### 8.3 Provenance to record per batch

| Field | Required |
|-------|----------|
| `experiment_id` | Unique batch identifier |
| Git commit hash | TO BE CONFIRMED — not auto-recorded today |
| Python version | Yes |
| Package versions | Yes (pip freeze or lockfile — **lockfile not yet present**) |
| Agent provider + model | Yes |
| `master_seed` | Yes |
| Config file path / hash | Yes (when YAML experiment configs exist) |

---

## 9. Pilot study — Phase A (PILOT / CALIBRATION STUDY)

### 9.1 Purpose and status

Phase A is a **pilot / calibration study**, not the final statistically justified experiment.

**Phase A does not produce definitive evidence for paper hypotheses.** Its outputs estimate variance, validate instrumentation, and inform the sample-size rationale for a subsequent formal study.

### 9.2 Phase A objectives

1. Validate fault injection at the intended boundaries (F1–F4, concurrent F1+F2)
2. Validate telemetry completeness and append-only raw storage
3. Validate safety monitoring against the validated taxonomy
4. Validate metrics definitions (especially `post_fault_completion_time` vs recovery)
5. Characterize baseline variance under OpenAI agent, no-fault control
6. Identify implementation failures before scaling N
7. Estimate appropriate sample size for the subsequent experiment (~±10 percentage-point 95% CI on success rate as initial planning target, subject to feasibility)

### 9.3 Frozen pilot configuration

| Field | Value |
|-------|-------|
| Config file | `configs/experiments/pilot.yaml` |
| Runner | `scripts/run_pilot.py` |
| `experiment_id` | `phase_a_pilot` |
| Agent | OpenAI only (`gpt-4o`) — MockLLMClient excluded |
| Block | RED |
| Master seed | `42` |
| Episodes per fault×intensity cell | `5` |
| Control episodes | `5` |
| Concurrent episodes (F1+F2, medium) | `5` |
| **Total scheduled episodes** | **70** |

**Episode arithmetic:**

```text
Control:     5
F1:          3 intensities × 5 = 15
F2:          3 intensities × 5 = 15
F3:          3 intensities × 5 = 15
F4:          3 intensities × 5 = 15
Concurrent:  5  (F1+F2 @ medium only)
─────────────────────────────────
Total:       70
```

This N is **not** the final sample size. Final N will be chosen after pilot variance analysis (§11).

### 9.4 Pilot scope (by condition)

| Condition ID | Description | Episodes | Agent |
|--------------|-------------|----------|-------|
| CONTROL | Baseline — no fault | 5 | OpenAI |
| F1 | `sensor_lag` @ LOW/MEDIUM/HIGH | 15 | OpenAI |
| F2 | `grip_slip` @ LOW/MEDIUM/HIGH | 15 | OpenAI |
| F3 | `unreachable_ik` @ LOW/MEDIUM/HIGH | 15 | OpenAI |
| F4 | `planner_output_corruption` (schema robustness) @ LOW/MEDIUM/HIGH | 15 | OpenAI |
| CONCURRENT | F1+F2 @ MEDIUM | 5 | OpenAI |

Mock agent runs are permitted for **engineering plumbing only** and must not be written to pilot/final data paths.

### 9.5 Pilot data labeling and separation

| Requirement | Policy |
|-------------|--------|
| Results path | `results/pilot/` |
| Raw traces | `data/raw/pilot/<experiment_id>/episodes.jsonl` (append-only) |
| Processed summaries | `data/processed/pilot/<experiment_id>/episodes.csv` |
| Dev smoke | `results/dev_smoke/` — never mixed with pilot |
| Analysis | Pilot data **excluded** from paper hypothesis figures unless explicitly labeled as pilot/calibration validation |
| Metadata | Every record includes `experiment_id=phase_a_pilot`, `phase=pilot` |

### 9.6 Existing smoke data

Files under `results/dev_smoke/` are **development artifacts**, not pilot or final data.

---

## 10. Pilot gates

Full experiment (Phase C) **must not begin** until all gates pass.

| Gate | Name | Pass criterion | Fail action |
|------|------|----------------|-------------|
| **G1** | Fault validity | ≥80% of fault episodes show `fault_applied=true` in event log at expected hook | Fix injection wiring / triggers |
| **G2** | Control stability | Baseline 1 success rate stable across pilot episodes (exact threshold TO BE SET from P1 variance) | Fix agent/task/sim before fault study |
| **G3** | Telemetry integrity | 100% episodes have `episode_id`, `success`, `fault_type`, timestamps | Fix logger |
| **G4** | Metric validity | Manual audit of ≥3 episodes per condition: `post_fault_completion_time`, success, steps match traces; no TTR/MTTR labels | Fix metric definitions or code |
| **G5** | Safety monitor validity | Nominal successful Baseline 0/1 episodes: violations near zero; intentional drop test: violations >0 | Calibrate `CONTACT_FORCE_THRESHOLD` and deduplication |
| **G6** | Reproducibility | Repeat 3 seeded P1 episodes: identical initial block positions | Document platform limits; fix seed policy |
| **G7** | Data integrity | Kill runner mid-batch; prior rows intact; checkpoint resumes | Fix checkpoint / append logic |
| **G8** | Fault separability | Fault event hooks differ across F1–F4; failure categories not identical by inspection | Redesign faults or metrics |

**Gate 2 threshold:** TO BE CONFIRMED — propose ≥80% success for Baseline 1 on RED block task if variance supports it.

---

## 11. Sample size strategy

### 11.1 Policy

**No fixed final sample size is prescribed in this protocol.** The PRD mention of "≥50 episodes" is an aspirational milestone, not a statistical justification.

Final N will be chosen **after Phase A** based on:

| Factor | How it informs N |
|--------|------------------|
| Observed success rate variance | Binomial CI width |
| Effect size between intensity levels | Practical significance |
| Runtime per episode | Feasibility |
| Failure mode diversity | Qualitative trace needs |
| LLM cost | Budget constraint |

### 11.2 Power analysis

Formal a priori power analysis is **not justified** until pilot estimates baseline variance. If post-pilot data remain insufficient for strong inferential statistics, the protocol requires stating that explicitly and emphasizing effect sizes + CIs over p-values.

### 11.3 Decision rule (post-pilot)

Document in `docs/EXPERIMENT_PROTOCOL.md` appendix or `docs/SAMPLE_SIZE_RATIONALE.md` (future):

- Target CI width for success rate (e.g., ±10% at 95% confidence)
- Resulting N per condition
- Expected total runtime

---

## 12. Metrics

### 12.1 Task success rate

| Field | Definition |
|-------|------------|
| **Definition** | Proportion of episodes where final block error `< PLACE_SUCCESS_TOLERANCE` |
| **Formula** | `success_rate = (# success) / N` |
| **Unit** | Proportion ∈ [0, 1] |
| **Data source** | `EpisodeResult.success`, telemetry `success` |
| **Interpretation** | Primary task outcome |
| **Limitations** | Loose tolerance; single task; binary |

### 12.2 Completion time

| Field | Definition |
|-------|------------|
| **Definition** | Wall-clock seconds from episode start to finalize |
| **Formula** | `total_wall_seconds` |
| **Unit** | Seconds |
| **Data source** | `ExperimentTelemetryRecord.total_wall_seconds` |
| **Interpretation** | End-to-end cost including LLM latency |
| **Limitations** | Hardware/API dependent; not pure sim time |

**Secondary:** `total_sim_steps` for simulation-only cost.

### 12.3 Post-fault completion time (`post_fault_completion_time`)

This is the **only** timing metric for post-fault execution in the single-shot agent. It is **not** recovery time, TTR, or MTTR.

| Field | Definition |
|-------|------------|
| **Definition** | Elapsed time from the **first confirmed fault injection** until **successful episode completion** (task success criterion met), for episodes where a valid interval exists |
| **Fields** | `post_fault_completion_time_sim_steps`, `post_fault_completion_time_wall_seconds` |
| **Formula (sim steps)** | `total_sim_steps − fault_injected_at_step` when valid |
| **Formula (wall)** | `total_wall_seconds − fault_injected_at_wall` when valid |
| **Unit** | Sim steps and/or seconds |
| **Data source** | `ChaosTelemetryLogger` → `ExperimentTelemetryRecord` |
| **When recorded** | Fault occurred **and** episode completed successfully **and** interval within configured caps |
| **When null** | No fault; episode failed; crashed; timed out; interval exceeds cap |
| **Interpretation** | Time to finish the **pre-existing single plan** after fault injection — **not** time to replan or recover |
| **Prohibited labels** | Do **not** call this recovery time, TTR, or MTTR in reports or figures |

Failed, crashed, or timed-out episodes: **`post_fault_completion_time = null`** (empty in CSV).

### 12.4 Deferred recovery metrics (TTR / MTTR)

| Field | Status |
|-------|--------|
| **TTR (time to recovery)** | **Not reported in Phase A.** Requires a genuine replanning/recovery loop. |
| **MTTR (mean time to recovery)** | **Not reported in Phase A.** Same prerequisite as TTR. |
| **H4 (recovery/replanning hypothesis)** | **DEFERRED** until replanning exists (§2). |

When replanning is implemented in a future phase, recovery metrics may be defined separately from `post_fault_completion_time`.

### 12.5 Failure rate

| Field | Definition |
|-------|------------|
| **Definition** | `1 − success_rate` for task outcome |
| **Unit** | Proportion |
| **Data source** | Telemetry |
| **Limitations** | Does not distinguish failure modes without taxonomy (§13) |

### 12.6 Safety violation rate

| Field | Definition |
|-------|------------|
| **Primary metric** | Episode-level violation rate: `(# episodes with safety_violation_count > 0) / N` |
| **Secondary metric** | Mean `safety_violation_count` per episode |
| **Data source** | `PyBulletSafetyMonitor` (validated taxonomy — `docs/SAFETY_TAXONOMY.md`) |
| **Interpretation** | Validated unsafe events only (violent arm–block contact, block OOW, block dropped outside zone) |
| **Limitations** | Live pilot validation required to confirm rates on OpenAI control episodes |

### 12.7 Replanning count

| Field | Definition |
|-------|------------|
| **Definition** | Number of planner re-invocations after initial plan |
| **Unit** | Count |
| **Data source** | `replan_count` |
| **Limitations** | **Always 0 in current system** — metric reserved for future use |

### 12.8 Fault detection / classification accuracy

| Field | Definition |
|-------|------------|
| **Definition** | Whether agent-reported fault label matches injected fault |
| **Unit** | Proportion of labeled episodes |
| **Data source** | `agent_diagnosed_fault`, `fault_classification_correct` |
| **Limitations** | **Not applicable today** — agent does not diagnose faults |

### 12.9 Execution steps

| Field | Definition |
|-------|------------|
| **Definition** | Count of high-level actions in plan (`len(plan.actions)`) and/or sim steps |
| **Unit** | Count |
| **Data source** | `EpisodeResult.steps_taken`, `total_sim_steps` |

---

## 13. Failure taxonomy

Failures will be assigned **post hoc** by inspecting execution logs, fault events, and outcomes. Multiple labels may apply. If evidence is insufficient:

**`UNKNOWN / UNRESOLVED`**

| Category | Indicators (examples) |
|----------|----------------------|
| **Perception / observation failure** | Planner acted on stale/wrong block positions (F1; LLM agents) |
| **Stale-state reasoning** | Plan inconsistent with true state after fault |
| **Planner failure** | LLM error, timeout, or unparseable output |
| **Malformed action** | Pydantic validation failure (F4 common) |
| **Kinematic failure** | Large EE error; unreachable targets (F3) |
| **Physical execution failure** | Grip not acquired, slip during carry (F2) |
| **Recovery failure** | Fault occurred, no successful task completion |
| **Cascading failure** | Initial fault led to secondary collision or drop |
| **Timeout** | `timed_out=True` when implemented |
| **Safety violation** | Nonzero safety count with qualifying event |
| **Unrecoverable state** | Block off table / outside workspace with no success |
| **UNKNOWN / UNRESOLVED** | Insufficient trace evidence |

Coding procedure TO BE CONFIRMED — propose dual coding on ≥10% of pilot episodes for reliability check.

---

## 14. Data schema

### 14.1 Minimum episode-level schema

| Field | Status in current codebase |
|-------|---------------------------|
| `experiment_id` | Present |
| `episode_id` | Present |
| `seed` | Present |
| `timestamp` | `recorded_at` |
| `task_id` | Partial (block color only) |
| `agent_version` | `agent_name` |
| `model_identifier` | Present |
| `fault_type` | `injected_fault_type` |
| `fault_intensity` | Present |
| `fault_trigger` | Present |
| `fault_duration` | Present (may be empty) |
| `success` | Present |
| `failure_reason` | Present |
| `completion_time` | `total_wall_seconds` |
| `post_fault_completion_time` | `post_fault_completion_time_sim_steps` / `_wall_seconds` |
| `safety_violation_count` | Present |
| `replanning_count` | `replan_count` (always 0) |
| `execution_steps` | `total_sim_steps`, `steps_taken` |

### 14.2 Storage tiers

| Tier | Format | Mutability | Contents |
|------|--------|------------|----------|
| **Raw event log** | JSONL | **Append-only, immutable** | Fault events, action log, execution traces via `ExperimentDataRepository` |
| **Episode summary** | CSV | Append-only per batch | One row per episode |
| **Aggregated analysis** | CSV/Parquet | Derived, regenerable | Condition-level stats |
| **Figures** | PNG/SVG | Derived, regenerable | From analysis pipeline |

**Current state:** Pilot/final runs write append-only JSONL under `data/raw/` and CSV summaries under `data/processed/`. Legacy dev smoke remains in `results/dev_smoke/`.

### 14.3 Pilot vs final data separation

| Tier | Pilot path | Final path (future) |
|------|------------|----------------------|
| Raw | `data/raw/pilot/<experiment_id>/` | `data/raw/final/<experiment_id>/` |
| Processed | `data/processed/pilot/` | `data/processed/final/` |
| Runner summaries | `results/pilot/` | `results/final/` |
| Dev smoke | `results/dev_smoke/` (never pilot/final) | — |

Existing `results/dev_smoke/` files must not be overwritten or merged into pilot/final batches.

---

## 15. Statistical analysis plan

### 15.1 Reporting standards (all conditions)

Report at minimum:

- **N** (episodes per condition; exclusions documented)
- **Mean** and **median** for continuous metrics
- **Standard deviation** or IQR
- **95% confidence intervals** for proportions and means where N permits
- **Full distribution plots** where sample size allows

### 15.2 Comparisons

| Outcome type | Planned approach |
|--------------|------------------|
| Success rate (binary) | Report proportions + CI; compare conditions with chi-square or Fisher's exact **if** expected counts valid; otherwise simulation/exact methods |
| Completion time / post-fault completion time | Report distribution; prefer nonparametric comparison if normality rejected |
| Safety violation rate (episode-level) | Proportions + CI |
| Intensity gradient | Trend visualization; regression **only if** assumptions checked |

### 15.3 Hypothesis testing policy

- Apply tests **only after** checking assumptions
- Report **effect sizes** alongside p-values where tests are used
- **No p-hacking** — primary outcomes pre-specified: success rate, completion time, post-fault completion time (where valid), episode-level safety violation rate
- **H4 excluded** from Phase A primary analysis (deferred)
- **No selective reporting** — all conditions run per frozen matrix must appear in results or exclusion log

### 15.4 Inconclusive results

If N is insufficient or variance too high, report CIs wide enough to span practical equivalence and state **inconclusive** rather than over-interpreting noise.

---

## 16. Ablation / secondary experiments

| ID | Experiment | Priority | Status |
|----|------------|----------|--------|
| **A1** | Recovery/replanning enabled vs disabled | SECONDARY | **Blocked** — replan not implemented |
| **A2** | Isolated vs concurrent faults | PRIMARY (RQ4) | Mechanism exists |
| **A3** | LOW vs MEDIUM vs HIGH intensity | PRIMARY (RQ3) | Planned |
| **A4** | Fault injection at different phases (`pick`, `lift`, `place`) | SECONDARY | Phase hooks exist in matrix runner |
| **A5** | Observation delay magnitude sweep (F1) | OPTIONAL | — |
| **A6** | Scripted vs LLM baseline comparison | SECONDARY | Baseline 0 vs 1 |
| **A7** | Mock vs real LLM (engineering validation only) | OPTIONAL | Not for paper claims |

Do **not** implement secondary experiments until pilot gates pass.

---

## 17. Threats to validity

### 17.1 Internal validity

| Threat | Severity | Notes |
|--------|----------|-------|
| Mock agent ignores observations | **High** | Invalidates perception-fault study on mock |
| No replan loop | **High** | Recovery/H4 claims invalid; use `post_fault_completion_time` only |
| Dual fault injection systems | **Low** | Documented — chaos wrapper canonical (`docs/FAULT_INJECTION.md`) |
| Orchestration duplication | **Resolved** | Unified `AgentService.run_task(hooks=...)` path |
| Planner corruption → validation failure | **Medium** | Different failure mode than in-task degradation |

### 17.2 External validity

| Threat | Notes |
|--------|-------|
| Single PyBullet scene | Results may not generalize to other environments |
| Single arm / task | Pick-and-place only |
| Simulation-to-reality gap | No hardware validation claimed |
| LLM-specific behavior | One model ≠ all LLM agents |

### 17.3 Construct validity

| Threat | Notes |
|--------|-------|
| Mislabeling post-fault completion as recovery | Use `post_fault_completion_time` only; defer TTR/MTTR |
| Safety violations counting nominal contact | Addressed by validated taxonomy — live pilot confirmation required |
| Success tolerance 0.08 m | Loose placement may mask faults |

### 17.4 Other limitations

- LLM/API version drift
- Small sample size until justified post-pilot
- Fault model bias (four faults ≠ exhaustive production fault space)
- Concurrent fault interaction may not represent independent production failures

---

## 18. Research ethics / integrity

| Principle | Commitment |
|-----------|------------|
| No fabricated results | All reported numbers trace to logged episodes |
| No fabricated citations | Literature in `docs/RELATED_WORK.md` verified before use |
| No selective deletion | Failed/crashed episodes retained and reported |
| Pilot vs final separation | Clearly labeled in metadata and analysis |
| Protocol deviations | Documented with reason and date |
| AI assistance | Coding/writing tools may assist; researcher verifies all claims against raw data |

---

## 19. Experiment directory structure (proposed)

Adapted to existing repo — **do not create empty directories until Phase B**:

```text
alpha_repo/
├── configs/
│   └── experiments/              # NEW: pilot.yaml, final_matrix.yaml
├── data/                         # NEW
│   ├── raw/
│   │   ├── pilot/<experiment_id>/episodes.jsonl
│   │   └── final/<experiment_id>/episodes.jsonl
│   └── processed/
├── analysis/                     # NEW: regenerate figures/tables from raw
├── figures/                      # NEW: derived outputs (gitignored or committed selectively)
├── results/                      # EXISTING: migrate to data/ over time
├── src/alpha/                    # EXISTING: core implementation
├── scripts/
│   ├── run_scripted_pick_place.py
│   ├── run_agent_pick_place.py
│   ├── run_experiment_matrix.py
│   ├── run_pilot.py              # Phase A pilot runner
│   └── analyze_experiments.py
├── tests/
└── docs/
    ├── RESEARCH_ENGINEERING_AUDIT.md
    ├── EXPERIMENT_PROTOCOL.md    # this document
    ├── FAULT_MODEL.md            # Phase 0 — future
    ├── METRICS.md                # Phase 0 — future
    └── ...
```

---

## 20. Final experiment checklist

Use before Phase C (full matrix):

- [ ] Research questions finalized
- [ ] Baselines validated (P0, P1)
- [ ] Fault definitions finalized (F1–F4)
- [ ] Fault intensity calibrated (§6.3)
- [ ] Telemetry validated (G3, G4)
- [ ] Safety monitor validated (G5)
- [ ] Seed policy validated (G6)
- [ ] Pilot configuration created (`configs/experiments/pilot.yaml`)
- [ ] Pilot completed
- [ ] Pilot gates G1–G8 passed
- [ ] Final sample-size rationale documented
- [ ] Final experiment matrix frozen
- [ ] Raw-data storage verified (immutable JSONL)
- [ ] Analysis pipeline verified (regenerates figures from raw)
- [ ] Real LLM arm selected and model ID recorded
- [ ] Orchestration code path unified (baseline == fault runner)
- [ ] Recovery metric definition aligned with agent capabilities (**post_fault_completion_time only; H4 deferred**)

---

## Appendix A — Resolved protocol items (audit blockers)

| Item | Resolution |
|------|------------|
| Real LLM for formal study | `configs/experiments/pilot.yaml` — OpenAI only |
| LangGraph agent | Not implemented; documentation updated to Brain/Body |
| Replan/recovery (H4, TTR, MTTR) | **Deferred**; `post_fault_completion_time` implemented |
| Immutable JSONL raw log | `ExperimentDataRepository` append-only JSONL |
| `experiment_id` and schema fields | Present in `ExperimentTelemetryRecord` |
| Unified fault injector | `ChaosRoboticsWrapper` canonical; legacy documented |
| Safety metric taxonomy | `docs/SAFETY_TAXONOMY.md` + unit tests |
| Pilot data separation | `results/pilot/`, `results/dev_smoke/` |
| Orchestration path | `AgentService.run_task(hooks=...)` |

## Appendix B — Remaining items (post-pilot)

| Item | Status |
|------|--------|
| Final sample-size rationale | After Phase A variance analysis |
| Timeout enforcement | Field exists; policy optional |
| Git commit hash auto-recorded | Manual provenance for now |
| Failure taxonomy dual coding | Post-pilot qualitative review |

## Appendix C — Version history

| Version | Date | Change |
|---------|------|--------|
| 0.1 | 2026-08-13 | Initial pre-experiment protocol |
| 0.2 | 2026-08-13 | Phase A frozen: `post_fault_completion_time`, H4 deferred, pilot N=70, OpenAI-only formal agent |

---

*End of protocol. Phase A pilot authorized only after validation gates in `docs/PILOT_VALIDATION_CHECKLIST.md` pass. Pilot data are calibration evidence, not final hypothesis tests.*
