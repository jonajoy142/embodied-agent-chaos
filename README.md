# Alpha — Chaos Engineering Harness for Embodied AI Agents

Fault-injection / chaos-engineering test harness for LLM-agent-driven robot
manipulation in simulation. See `PRD.md` for the full 12-week plan and
`docs/architecture.md` for how the repo is laid out and why.

One repo for the whole project — each week adds code into the existing
`core / schemas / models / clients / services / repos` layers below rather
than becoming a new repo.

## Status

- [x] Week 1 — Baseline sim + scripted (non-agentic) pick/place
- [x] Week 2 — Minimal 2-stage LLM agent loop (Brain/Body)
- [x] Weeks 3-4 — Fault injection layer
- [ ] Week 5 — Metrics harness
- [ ] Week 6 — Full experiment matrix
- [ ] Week 7 — Public repo cleanup (Docker, README, demo video)
- [ ] Week 8 — Hosted dashboard
- [ ] Week 9 — Medium series (parts 1-2)
- [ ] Week 10 — Paper draft (IEEE/RA-L template)
- [ ] Week 11 — Review pass + Medium part 3
- [ ] Week 12 — arXiv preprint + final public links

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[dev]"           # installs alpha as an editable package + pytest
```

(`requirements.txt` also works if you'd rather not install the package: `pip install -r requirements.txt`.)

## Running Week 1

```bash
python scripts/run_week1_smoke_test.py     # scene loads, blocks settle — no error
python scripts/run_scripted_pick_place.py --block red
python scripts/run_scripted_pick_place.py --block green --gui   # watch it in a GUI window
```

Results get appended to `results/episodes.csv` via the episode repository —
this is the same file Week 5's metrics harness will read.

## Running Week 2 — Agent Brain/Body

Week 2 changes where the plan comes from, not how robot motion is executed:

```text
Brain / LLM client
  -> validated ActionPlan
  -> Body / ControlService
  -> PyBullet
```

The Brain decides high-level actions (`MOVE_TO`, `GRIP`, `RELEASE`). The
Body (`ControlService`) executes those actions. The Brain never imports or
talks to PyBullet directly.

The default agent is deterministic and local, so no API key is required:

```bash
python scripts/run_agent_pick_place.py --block red
python scripts/run_agent_pick_place.py --block red --gui
```

Optional real provider wiring is behind `clients/llm_client.py`:

```bash
pip install -e ".[agent]"
OPENAI_API_KEY=... python scripts/run_agent_pick_place.py --agent openai --block red
```

The Week 1 scripted controller is still useful as the deterministic baseline.
The Week 2 agent path produces the same kind of `EpisodeResult`, but also
logs the structured action plan and execution trace to `results/episodes.csv`.

## Running Weeks 3-4 — Fault Injection

Weeks 3-4 add a composable fault injector at the execution boundary:

```text
Agent / Brain
  -> Body / ControlService
  -> FaultInjector
  -> SimulatorClient
  -> PyBullet
```

Fault injection is intentional chaos engineering for embodied-AI evaluation:
the system corrupts execution in controlled, reproducible ways and records
what happened for later metrics.

### Legacy CLI faults (ScriptedFaultInjector)

The CLI uses `ScriptedFaultInjector` for backward-compatible testing:

Supported faults:

- `none` — no fault; default behavior.
- `target_corruption` — offsets requested `MOVE_TO` targets before execution.
- `position_offset` — deterministic target offset, useful for degradation tests.
- `grip_failure` — prevents `GRIP` from acquiring the block.
- `release_failure` — prevents `RELEASE` from releasing the block.

Examples:

```bash
python scripts/run_agent_pick_place.py --block red --fault none
python scripts/run_agent_pick_place.py --block red --fault position_offset
python scripts/run_agent_pick_place.py --block red --fault grip_failure
python scripts/run_agent_pick_place.py --block red --fault release_failure --seed 42
```

### Production fault injection (ChaosRoboticsWrapper)

Formal experiments use **`ChaosRoboticsWrapper`** — a decoupled middleware wrapper
that implements 4 fault classes at different boundaries:

**Fault Classes:**

1. **Sensor Lag (F1)** — Intercepts environment state before it reaches the planner,
   returning cached observations from `lag_seconds` ago to simulate delayed sensor streams.

2. **Grip Slip (F2)** — Intercepts PyBullet simulation steps when an object is held,
   randomly weakening or clearing grip constraints based on `slip_probability`.

3. **Unreachable IK (F3)** — Intercepts end-effector target coordinates from the LLM
   Body plan and applies mathematical offsets, pushing targets outside the arm's
   operational workspace.

4. **Planner-Output Corruption (F4)** — Intercepts raw text/JSON from the LLM Brain
   and randomly deletes critical JSON keys or injects malformed characters to test
   parsing error handling.

**Configuration:**

```python
from alpha.models.chaos_config import ChaosConfig

config = ChaosConfig.from_yaml("configs/experiments/pilot.yaml")
```

Supports:
- Single and concurrent fault injection modes
- Time-based triggers (seconds elapsed) or event-based triggers (phase == "pick", step == 3)
- Intensity parameters (lag duration, slip probability, offset magnitude, corruption probability)

See `docs/FAULT_INJECTION.md` for detailed architecture and usage.

Fault runs append to the same `results/episodes.csv` file and include the
fault type, action plan, and execution log.

## Phase A pilot (preparation — do not treat as final results)

After completing `docs/PILOT_VALIDATION_CHECKLIST.md`:

```bash
python scripts/run_pilot.py --dry-run
OPENAI_API_KEY=... python scripts/run_pilot.py
```

Config: `configs/experiments/pilot.yaml`. Output: `results/pilot/` and
`data/raw/pilot/`. Dev smoke data lives in `results/dev_smoke/`.

MockLLMClient is for tests only; the pilot uses OpenAI. Metrics use
`post_fault_completion_time` (not TTR/MTTR) until a replanning loop exists.

## Tests

```bash
pytest
```

`test_entities.py` runs with no dependencies beyond the package itself.
`test_control_service.py` is skipped automatically if `pybullet` isn't
installed in the current environment.

## Repo layout

```
src/alpha/
  config/         Centralized constants
  core/           Domain entities + interfaces (no external deps)
  schemas/        Pydantic I/O schemas (serialization boundary)
  models/         Config models for not-yet-built pieces (fault matrix, agent)
  clients/        Wrappers around external systems (PyBullet, future LLM API)
  services/       Orchestration/business logic
  repos/          Persistence (CSV now, swappable later)
  dashboard/      Presentation layer stub (Week 8)
scripts/          CLI entrypoints
tests/            Unit + integration tests
docker/           Reproducible container (Week 7)
docs/             Architecture notes
```

See `docs/architecture.md` for the full rationale.
