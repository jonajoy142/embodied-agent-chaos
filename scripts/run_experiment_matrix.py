#!/usr/bin/env python3
"""
Week 6 — full experiment matrix automation.

Runs:
  - 10 control episodes (zero faults)
  - 4 fault types × 3 intensities × 5 episodes = 60 fault-matrix episodes
  - 5 concurrent episodes (Sensor Lag + Grip Slip)

Each episode uses deterministic seeds for fair comparison, persists telemetry
after every run, checkpoints progress, and emits a markdown summary table.

Usage:
    python scripts/run_experiment_matrix.py
    python scripts/run_experiment_matrix.py --agent mock --resume
    python scripts/run_experiment_matrix.py --dry-run
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
from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger
from alpha.services.experiment_matrix import (
    append_matrix_run_log,
    default_matrix_paths,
    generate_matrix_scenarios,
    load_checkpoint,
    load_matrix_run_log,
    matrix_run_record_from_telemetry,
    render_summary_markdown,
    save_checkpoint,
    summarize_matrix_runs,
)
from alpha.services.matrix_episode_runner import execute_matrix_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Week 6 chaos experiment matrix.")
    parser.add_argument("--agent", choices=["mock", "openai", "ollama"], default="mock",
                        help="LLM provider. Mock is for engineering/tests only; use openai for formal studies.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--control-runs", type=int, default=10)
    parser.add_argument("--episodes-per-variation", type=int, default=5)
    parser.add_argument("--concurrent-runs", type=int, default=5)
    parser.add_argument("--llm-min-interval", type=float, default=0.5, help="Seconds between LLM calls.")
    parser.add_argument("--resume", action="store_true", help="Skip runs already present in the checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print the schedule without executing episodes.")
    parser.add_argument("--matrix-log", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--summary", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_matrix_paths()
    matrix_log_path = args.matrix_log or paths["matrix_log"]
    checkpoint_path = args.checkpoint or paths["checkpoint"]
    summary_path = args.summary or paths["summary"]

    scenarios = generate_matrix_scenarios(
        master_seed=args.master_seed,
        control_runs=args.control_runs,
        episodes_per_variation=args.episodes_per_variation,
        concurrent_runs=args.concurrent_runs,
    )
    expected_total = args.control_runs + (4 * 3 * args.episodes_per_variation) + args.concurrent_runs

    print("Week 6 experiment matrix")
    print(f"  master seed: {args.master_seed}")
    print(f"  scheduled episodes: {len(scenarios)} (expected {expected_total})")
    print(f"  agent provider: {args.agent}")
    print(f"  matrix log: {matrix_log_path}")
    print(f"  telemetry log: {settings.EXPERIMENTS_CSV}")
    print(f"  checkpoint: {checkpoint_path}")

    if args.dry_run:
        for scenario in scenarios:
            print(f"  - {scenario.run_id}: {scenario.fault_type} @ {scenario.intensity_label}")
        return

    completed = load_checkpoint(checkpoint_path) if args.resume else set()
    telemetry = ChaosTelemetryLogger()
    episode_repo = EpisodeRepository()
    agent_config = AgentConfig(provider=args.agent, model_name=args.model)

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
                telemetry=telemetry,
                episode_repo=episode_repo,
                agent_config=agent_config,
                llm_min_interval=args.llm_min_interval,
            )
        except Exception as exc:
            failures.append(scenario.run_id)
            print(f"  ERROR: {scenario.run_id} crashed: {exc}")
            traceback.print_exc()
            completed.add(scenario.run_id)
            save_checkpoint(checkpoint_path, completed, args.master_seed)
            continue

        matrix_record = matrix_run_record_from_telemetry(scenario, record)
        append_matrix_run_log(matrix_log_path, matrix_record)
        completed.add(scenario.run_id)
        save_checkpoint(checkpoint_path, completed, args.master_seed)

        status = "SUCCESS" if record.success else "FAIL"
        print(
            f"  done: {status} | pfct={record.post_fault_completion_time_sim_steps} | "
            f"safety={record.safety_violation_count} | wall={record.total_wall_seconds:.2f}s"
        )

    run_records = load_matrix_run_log(matrix_log_path)
    summary_rows = summarize_matrix_runs(run_records)
    markdown = render_summary_markdown(summary_rows)

    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"Saved summary to {summary_file}")
    if failures:
        print(f"Completed with {len(failures)} crashed scenarios: {', '.join(failures)}")


if __name__ == "__main__":
    main()
