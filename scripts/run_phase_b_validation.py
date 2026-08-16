#!/usr/bin/env python3
"""
Phase B validation runner — loads ``configs/experiments/phase_b_validation.yaml``.

This engineering validation runs 21 episodes:
  - 3 control
  - 3 intensities x 3 F1 sensor-lag episodes
  - 3 intensities x 3 F3 unreachable-IK episodes

Usage:
    python scripts/run_phase_b_validation.py --dry-run
    python scripts/run_phase_b_validation.py
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

from alpha.clients.llm_client import validate_llm_runtime
from alpha.core.entities import BlockColor, FaultType
from alpha.models.agent_config import AgentConfig
from alpha.repos.episode_repository import EpisodeRepository
from alpha.repos.experiment_data_repository import ExperimentDataRepository
from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger
from alpha.services.experiment_matrix import (
    append_matrix_run_log,
    generate_phase_b_validation_scenarios,
    load_checkpoint,
    load_matrix_run_log,
    load_pilot_config,
    matrix_run_record_from_telemetry,
    phase_b_validation_paths,
    render_summary_markdown,
    save_checkpoint,
    summarize_matrix_runs,
)
from alpha.services.matrix_episode_runner import execute_matrix_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase B validation experiment.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "phase_b_validation.yaml"),
        help="Path to Phase B validation YAML config",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pilot_config(args.config)
    paths = phase_b_validation_paths(config)
    scenarios = generate_phase_b_validation_scenarios(config)

    print("Phase B validation")
    print(f"  experiment_id: {config.experiment_id}")
    print(f"  scheduled episodes: {len(scenarios)}")
    print(f"  agent: {config.agent_provider} / {config.agent_model}")
    print(f"  results: {paths['results_dir']}")

    if config.agent_provider != "ollama" or config.agent_model != "llama3":
        raise SystemExit("Phase B validation is configured for Ollama llama3 only")

    if args.dry_run:
        for scenario in scenarios:
            print(f"  - {scenario.run_id}: {scenario.fault_type} @ {scenario.intensity_label}")
        return

    agent_config = AgentConfig(
        provider=config.agent_provider,
        model_name=config.agent_model,
        ollama_base_url=config.ollama_base_url,
    )
    try:
        validate_llm_runtime(agent_config)
    except RuntimeError as exc:
        raise SystemExit(f"LLM runtime preflight failed: {exc}") from exc

    completed = load_checkpoint(paths["checkpoint"]) if args.resume else set()
    telemetry = ChaosTelemetryLogger(csv_path=str(Path(paths["results_dir"]) / "experiments_log.csv"))
    episode_repo = EpisodeRepository()
    data_repo = ExperimentDataRepository(experiment_id=config.experiment_id, phase="phase_b")
    block = BlockColor(config.block)

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
                experiment_id=config.experiment_id,
                telemetry=telemetry,
                episode_repo=episode_repo,
                data_repo=data_repo,
                agent_config=agent_config,
                block=block,
                llm_min_interval=config.llm_min_interval,
                master_seed=config.master_seed,
                model_identifier=config.agent_model,
                enable_replanning=scenario.fault_type == FaultType.SENSOR_LAG.value,
                replanning_chunk_size=3,
            )
        except Exception as exc:
            failures.append(scenario.run_id)
            print(f"  ERROR: {scenario.run_id} crashed: {exc}")
            traceback.print_exc()
            completed.add(scenario.run_id)
            save_checkpoint(paths["checkpoint"], completed, config.master_seed)
            continue

        matrix_record = matrix_run_record_from_telemetry(scenario, record)
        append_matrix_run_log(paths["matrix_log"], matrix_record)
        completed.add(scenario.run_id)
        save_checkpoint(paths["checkpoint"], completed, config.master_seed)

        status = "SUCCESS" if record.success else "FAIL"
        print(
            f"  done: {status} | fault_observed={record.fault_occurred} | "
            f"raw_safety={record.safety_violation_count} | unique_safety={record.unique_safety_incident_count}"
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
