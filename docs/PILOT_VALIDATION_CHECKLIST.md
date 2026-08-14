# Phase A pilot validation checklist

Complete every gate **before** running `scripts/run_pilot.py`. If any gate fails,
stop and fix the blocker.

**Legend**

- **READY** — satisfied by code, config, or offline/unit tests without live OpenAI execution
- **REQUIRES LIVE VALIDATION** — must be confirmed with at least one real pilot episode (OpenAI + PyBullet)

## Gates

| # | Gate | Status | How to verify |
| --- | --- | --- | --- |
| 1 | Fault actually occurs | **REQUIRES LIVE VALIDATION** | Run one episode per fault type; inspect `fault_occurred=true` in telemetry and chaos `event_log` |
| 2 | Fault reaches intended injection boundary | **Mixed** | F1 boundary: **READY** (`tests/test_observation_fault_boundary.py`). F2/F3 actuation hooks: **REQUIRES LIVE VALIDATION**. F4 via real OpenAI planner path: **REQUIRES LIVE VALIDATION** |
| 3 | Control condition unchanged | **REQUIRES LIVE VALIDATION** | Control episodes must show `fault_occurred=false` and no chaos fault events in live runs |
| 4 | Telemetry is complete | **Mixed** | Schema/fields: **READY** (unit tests). End-to-end row population on live OpenAI episodes: **REQUIRES LIVE VALIDATION** |
| 5 | Safety monitor calibrated | **Mixed** | Taxonomy + deterministic tests: **READY** (`tests/test_safety_monitor.py`). Baseline scene ≈ 0 violations on live control episodes: **REQUIRES LIVE VALIDATION** |
| 6 | Seeds work as intended | **READY** | `test_seed_deterministic_environment_is_repeatable`; episode seed formula in code |
| 7 | Raw data preserved | **Mixed** | Append-only repository implementation: **READY**. First live write to `data/raw/pilot/`: **REQUIRES LIVE VALIDATION** |
| 8 | Pilot and smoke data separated | **READY** | Dev smoke in `results/dev_smoke/`; pilot output paths in `configs/experiments/pilot.yaml` → `results/pilot/` |
| 9 | Mock agent excluded | **READY** | `pilot.yaml` sets `agent.provider: openai`; `run_pilot.py` rejects non-OpenAI providers |
| 10 | No TTR/MTTR claim | **READY** | Telemetry, summaries, and `docs/EXPERIMENT_PROTOCOL.md` use `post_fault_completion_time` only |
| 11 | OpenAI only in formal matrix | **READY** | No Ollama in `configs/experiments/pilot.yaml` |
| 12 | F4 labelled correctly | **READY** | F4 documented as planner/schema robustness in `docs/FAULT_INJECTION.md` and protocol §5.5 |

### Properties requiring live OpenAI execution (summary)

These cannot be marked READY from unit tests alone:

1. **OpenAI end-to-end execution** — full plan → execute → telemetry on control and fault episodes
2. **F4 corruption through the real planner path** — `ChaosPlannerLLMClient` + live API response handling
3. **Real telemetry completeness** — all fields populated after a live episode (including `execution_log_json` in raw JSONL)
4. **Actual safety-event behavior** — violation rates on live control vs fault episodes in PyBullet

## Automated preflight (READY gates)

```bash
pytest tests/test_observation_fault_boundary.py tests/test_safety_monitor.py \
       tests/test_chaos_telemetry_logger.py tests/test_experiment_matrix.py
python scripts/run_pilot.py --dry-run
```

## Manual spot checks (before live pilot)

1. Confirm output paths point to `results/pilot/` not `results/dev_smoke/`
2. Confirm `OPENAI_API_KEY` is set
3. Run **one** control episode manually if desired before full 70-episode batch (optional smoke — label separately if not using pilot runner)

## Post-run (pilot only — not final study)

- Review violation rate: should be low on control, not ~100/episode
- Inspect 1–2 raw JSONL traces per condition
- Use variance to estimate final sample size (target ~±10 pp on success rate)
- **Do not publish pilot results as final research outcomes**
