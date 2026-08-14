#!/usr/bin/env python3
"""
Phase A pilot runner — loads ``configs/experiments/pilot.yaml``.

Formal studies use OpenAI only. MockLLMClient is excluded.

Usage:
    python scripts/run_pilot.py --dry-run
    OPENAI_API_KEY=... python scripts/run_pilot.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alpha.config import settings
from alpha.models.agent_config import AgentConfig
from alpha.repos.episode_repository import EpisodeRepository
from alpha.repos.experiment_data_repository import ExperimentDataRepository
from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger
from alpha.services.experiment_matrix import (
    append_matrix_run_log,
    generate_pilot_scenarios,
    load_checkpoint,
    load_pilot_config,
    load_matrix_run_log,
    matrix_run_record_from_telemetry,
    pilot_paths,
    render_summary_markdown,
    save_checkpoint,
    summarize_matrix_runs,
)
from alpha.services.matrix_episode_runner import execute_matrix_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase A pilot experiment.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "pilot.yaml"),
        help="Path to pilot YAML config",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pilot = load_pilot_config(args.config)
    paths = pilot_paths(pilot)
    scenarios = generate_pilot_scenarios(pilot)

    print("Phase A pilot")
    print(f"  experiment_id: {pilot.experiment_id}")
    print(f"  scheduled episodes: {len(scenarios)}")
    print(f"  agent: {pilot.agent_provider} / {pilot.agent_model}")
    print(f"  results: {paths['results_dir']}")

    if pilot.agent_provider != "openai":
        raise SystemExit("Formal pilot requires agent.provider=openai in pilot.yaml")

    if args.dry_run:
        for scenario in scenarios:
            print(f"  - {scenario.run_id}: {scenario.fault_type} @ {scenario.intensity_label}")
        return

    completed = load_checkpoint(paths["checkpoint"]) if args.resume else set()
    telemetry = ChaosTelemetryLogger(csv_path=str(Path(paths["results_dir"]) / "experiments_log.csv"))
    episode_repo = EpisodeRepository()
    data_repo = ExperimentDataRepository(experiment_id=pilot.experiment_id, phase="pilot")
    agent_config = AgentConfig(provider=pilot.agent_provider, model_name=pilot.agent_model)

    failures: list[str] = []
    for index, scenario in enumerate(scenarios, start=1):
        if scenario.run_id in completed:
            print(f"[{index}/{len(scenarios)}] skip {scenario.run_id} (checkpoint)")
            continue

        print(
            f"[{index}/{len(scenarios)}] running {scenario.run_id} "
            f"({scenario.fault_type}, {scenario.intensity_label})"
        )
        try:
            _result, record = execute_matrix_episode(
                scenario,
                experiment_id=pilot.experiment_id,
                telemetry=telemetry,
                episode_repo=episode_repo,
                data_repo=data_repo,
                agent_config=agent_config,
                llm_min_interval=pilot.llm_min_interval,
                master_seed=pilot.master_seed,
                model_identifier=pilot.agent_model,
            )
        except Exception as exc:
            failures.append(scenario.run_id)
            print(f"  ERROR: {scenario.run_id} crashed: {exc}")
            traceback.print_exc()
            completed.add(scenario.run_id)
            save_checkpoint(paths["checkpoint"], completed, pilot.master_seed)
            continue

        matrix_record = matrix_run_record_from_telemetry(scenario, record)
        append_matrix_run_log(paths["matrix_log"], matrix_record)
        completed.add(scenario.run_id)
        save_checkpoint(paths["checkpoint"], completed, pilot.master_seed)

        status = "SUCCESS" if record.success else "FAIL"
        print(
            f"  done: {status} | pfct={record.post_fault_completion_time_sim_steps} | "
            f"safety={record.safety_violation_count}"
        )

    run_records = load_matrix_run_log(paths["matrix_log"])
    summary_rows = summarize_matrix_runs(run_records)
    markdown = render_summary_markdown(summary_rows)
    Path(paths["summary"]).write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Saved summary to {paths['summary']}")
    if failures:
        print(f"Completed with {len(failures)} crashed scenarios: {', '.join(failures)}")


if __name__ == "__main__":
    main()
