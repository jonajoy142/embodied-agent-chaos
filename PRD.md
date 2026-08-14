# PRD — Chaos Engineering Harness for Embodied AI Agents

**Project codename:** Alpha · **Owner:** Jona · **Duration:** 12 weeks · **Status:** Kickoff

## 1. One-line pitch

A fault-injection / chaos-engineering test harness that stress-tests LLM-agent-driven robot manipulation in simulation, applying production reliability-engineering rigor (fault taxonomies, MTTR, degradation curves) that current embodied-AI research doesn't — proving the "software systems engineer → physical AI" transition with a real artifact, not a novel algorithm claim.

## 2. Why this, why now

- Planning/replanning, edge-LLM-for-robotics, and even fault-injection-for-embodied-AI *as concepts* all have prior art (BrainBody-LLM, ReplanVLM, LiteVLA-Edge, "Harnessing Embodied Agents"). Competing on algorithmic novelty is a losing game against robotics PhDs.
- The gap is evaluation rigor, not architecture. No public project combines: (a) a working agentic manipulation demo, (b) a systematic multi-layer fault matrix, (c) production-style metrics (MTTR, degradation curve, blast radius), built by someone with real distributed-systems/fintech reliability background.
- Goal is dual: a working public artifact for R&D interviews (BMW/Bosch/Airbus/Werkstudent) **and** a systems-track paper (ICRA/IROS workshop or arXiv preprint), not a flagship-conference novelty claim.

## 3. Scope

**In scope:** simulated manipulation only (PyBullet, arm + blocks), one baseline LLM-agent planner (reproduced, not invented), a fault-injection layer, an eval/metrics harness, a public repo + demo dashboard, a written systems paper.
**Out of scope:** physical hardware, novel planning algorithms, multi-robot, real vehicles/driving stack.

## 4. Architecture (v1)

```
[Task goal, NL] -> [Agent Planner: Brain/Body via AgentService — single-shot plan then execute]
                          |
                 [Fault Injection Layer] -> F1 sensor lag | F2 grip slip | F3 unreachable IK | F4 planner/schema corruption | concurrent F1+F2
                          |
                 [PyBullet Env: arm + blocks] -> outcome
                          |
                 [Metrics: post_fault_completion_time, failure-type classification, degradation curve, safety-violation rate]
```

**Implementation note:** The agent is **not** a LangGraph graph. Orchestration is
`AgentService` coordinating `AgentBrain` (LLM) and `ControlService` (Body). A
replanning loop may be added in a later phase; until then do not report TTR/MTTR.

See docs/architecture.md for how this maps to the actual repo layout.

## 5. 12-Week Roadmap

| Weeks | Milestone | Deliverable |
| --- | --- | --- |
| 1 | Baseline sim + scripted (non-agentic) pick/place working | Working PyBullet env, 1 task |
| 2 | Minimal 2-stage LLM agent loop (Brain/Body) replicated | Agent solves task via NL command, no faults yet |
| 3-4 | Build fault injection layer: 4 fault classes (sensor lag, grip slip, unreachable IK, planner-output corruption) + single/concurrent injection modes | Fault matrix + injector module, unit-tested |
| 5 | Metrics harness: MTTR, degradation curve, fault-classification accuracy, safety-violation logging | CSV/dashboard of results across 50+ episodes |
| 6 | Run full experiment matrix (baseline agent vs. faults, single vs. concurrent) | Results table + plots |
| 7 | Public repo cleanup: Docker/reproducible setup, README, demo video | GitHub repo live |
| 8 | Minimal live dashboard (hosted, e.g. Streamlit/Vercel) showing a run + metrics | Public demo link |
| 9 | Medium series drafted (3 parts: architecture, fault matrix, results) | Posts 1-2 published |
| 10 | Paper draft in Overleaf (IEEE/RA-L template): abstract, related work (cite BrainBody-LLM, ReplanVLM, Harnessing Embodied Agents), method, results | Full draft |
| 11 | Paper review pass + Medium post 3 + share with TUM postdoc contact for feedback | Revised draft + outside review |
| 12 | Submit preprint to arXiv; finalize public repo/demo links for resume/LinkedIn | Live paper + live demo + live repo |

## 6. Success metrics

- Public repo with reproducible setup (not "coming soon")
- Dashboard/demo reachable via a link, not just local
- >=50 experiment episodes across >=4 fault types, with quantified degradation
- arXiv preprint submitted by week 12
- 3-part Medium series published
- One concrete artifact citable in Werkstudent/Master's applications and interviews

## 7. Key risks

- **Scope creep toward "novel architecture"** — stay disciplined: the agent loop is *reproduced*, the harness is the contribution.
- **Week 3-6 is the crunch** (fault layer + metrics) — protect this against GRE/dMAT prep load; this is the block that can't slip.
- **"Live deployment"** here means a public, hosted demo + repo — not a real robot. Keep that framing explicit in the paper and pitch so it's never oversold in an interview.

## 8. Immediate next action

Week 1 build: PyBullet scene (arm + 3 blocks) + scripted pick/place, no agent, no faults yet. Done — see `src/alpha/services/scene_service.py` and `control_service.py`.
