#!/usr/bin/env python3
"""
Run exactly ONE live Alpha experiment scenario.

Examples:

    python scripts/run_spot_check.py --fault none --provider openai --model gpt-4o

    python scripts/run_spot_check.py --fault none --provider ollama --model llama3.2

    python scripts/run_spot_check.py \
        --fault sensor_lag \
        --intensity low \
        --provider ollama \
        --model llama3.2

    python scripts/run_spot_check.py \
        --fault grip_slip \
        --intensity low

    python scripts/run_spot_check.py \
        --fault unreachable_ik \
        --intensity low

    python scripts/run_spot_check.py \
        --fault planner_output_corruption \
        --intensity low

This uses the same production execution path as the Phase A pilot,
but executes exactly one scenario.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from alpha.clients.llm_client import validate_llm_runtime
from alpha.models.agent_config import AgentConfig
from alpha.repos.episode_repository import EpisodeRepository
from alpha.repos.experiment_data_repository import ExperimentDataRepository
from alpha.services.chaos_telemetry_logger import ChaosTelemetryLogger
from alpha.services.experiment_matrix import (
    generate_pilot_scenarios,
    load_pilot_config,
)
from alpha.services.matrix_episode_runner import execute_matrix_episode


VALID_FAULTS = {
    "none",
    "sensor_lag",
    "grip_slip",
    "unreachable_ik",
    "planner_output_corruption",
}

VALID_INTENSITIES = {
    "low",
    "medium",
    "high",
}

VALID_PROVIDERS = {
    "openai",
    "ollama",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly ONE live Alpha experiment scenario."
    )

    parser.add_argument(
        "--fault",
        required=True,
        choices=sorted(VALID_FAULTS),
        help="Fault to inject. Use 'none' for the control condition.",
    )

    parser.add_argument(
        "--intensity",
        choices=sorted(VALID_INTENSITIES),
        help="Fault intensity. Required for F1-F4.",
    )

    parser.add_argument(
        "--provider",
        default="openai",
        choices=sorted(VALID_PROVIDERS),
        help="LLM provider to use explicitly for this spot check.",
    )

    parser.add_argument(
        "--model",
        help="Provider model name. Defaults to gpt-4o for OpenAI or OLLAMA_MODEL for Ollama.",
    )

    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for local Ollama when --provider ollama.",
    )

    return parser.parse_args()


def select_scenario(
    fault: str,
    intensity: str | None,
):
    """
    Select exactly one matching scenario from the existing pilot matrix.

    We reuse pilot.yaml's scenario generation so the spot check uses
    exactly the same fault definitions/intensity calibration as the
    formal pilot.
    """

    pilot = load_pilot_config(
        str(ROOT / "configs" / "experiments" / "pilot.yaml")
    )

    scenarios = generate_pilot_scenarios(pilot)

    if fault == "none":
        if intensity is not None:
            raise SystemExit(
                "ERROR: --intensity must not be supplied with --fault none."
            )

        matches = [
            scenario
            for scenario in scenarios
            if scenario.scenario_group == "control"
        ]

    else:
        if intensity is None:
            raise SystemExit(
                f"ERROR: --intensity is required for fault '{fault}'."
            )

        matches = [
            scenario
            for scenario in scenarios
            if scenario.fault_type == fault
            and scenario.intensity_label == intensity
            and not scenario.concurrent
        ]

    if not matches:
        raise SystemExit(
            f"ERROR: No scenario found for fault={fault}, "
            f"intensity={intensity}."
        )

    # We deliberately run exactly one scenario.
    return matches[0]


def print_result(
    result,
    record,
    scenario,
    raw_path: Path,
    *,
    provider: str,
    model: str,
) -> None:
    print()
    print("=" * 60)
    print("ALPHA — LIVE SPOT CHECK")
    print("=" * 60)

    print(f"run_id:              {scenario.run_id}")
    print(f"fault:               {scenario.fault_type}")
    print(f"intensity:           {scenario.intensity_label}")
    print(f"agent:               {provider}")
    print(f"model:               {model}")

    print()
    print("--- LLM ---")

    print(f"prompt_tokens:       {getattr(record, 'llm_prompt_tokens', 0)}")
    print(
        f"completion_tokens:   "
        f"{getattr(record, 'llm_completion_tokens', 0)}"
    )
    print(f"total_tokens:        {getattr(record, 'llm_total_tokens', 0)}")

    print()
    print("--- SIMULATION ---")

    print(f"total_sim_steps:     {getattr(record, 'total_sim_steps', 0)}")

    success = result.success if result is not None else None

    print(f"success:             {success}")
    print(f"crashed:             {record.crashed}")
    print(f"failure_reason:      {record.failure_reason}")

    print()
    print("--- CHAOS ---")

    print(f"fault_occurred:      {record.fault_occurred}")

    print()
    print("--- SAFETY ---")

    print(
        f"safety_violations:   "
        f"{getattr(record, 'safety_violation_count', 0)}"
    )

    print()
    print("--- RAW DATA ---")

    print(f"raw_jsonl:           {raw_path}")

    print("=" * 60)
    print()


def main() -> int:
    args = parse_args()

    # ---------------------------------------------------------
    # Validate provider configuration without printing secrets.
    # ---------------------------------------------------------

    model = args.model
    if args.provider == "openai" and not model:
        model = "gpt-4o"
    if args.provider == "ollama" and not model:
        model = os.environ.get("OLLAMA_MODEL")

    if not model:
        print(
            f"ERROR: --model is required for provider '{args.provider}' "
            "unless the provider has an environment default.",
            file=sys.stderr,
        )
        return 1

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set in this shell.", file=sys.stderr)
        print("Export it in your terminal and rerun this command.", file=sys.stderr)
        return 1

    agent_config = AgentConfig(
        provider=args.provider,
        model_name=model,
        ollama_base_url=args.ollama_base_url,
    )
    try:
        validate_llm_runtime(agent_config)
    except RuntimeError as exc:
        print(f"ERROR: LLM runtime preflight failed: {exc}", file=sys.stderr)
        return 1

    # ---------------------------------------------------------
    # Select ONE scenario.
    # ---------------------------------------------------------

    scenario = select_scenario(
        fault=args.fault,
        intensity=args.intensity,
    )

    # Give the spot check its own namespace so it does not get
    # mixed with the failed 70-episode pilot.
    spot_id = uuid.uuid4().hex[:8]

    experiment_id = f"spot_{args.provider}_{args.fault}_{spot_id}"

    results_dir = ROOT / "results" / "spot_checks" / experiment_id
    raw_root = ROOT / "data" / "raw" / "spot_checks"
    processed_root = ROOT / "data" / "processed" / "spot_checks"

    results_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = results_dir / "experiments_log.csv"

    # ---------------------------------------------------------
    # Build the same production components used by the pilot.
    # ---------------------------------------------------------

    telemetry = ChaosTelemetryLogger(
        csv_path=str(telemetry_path)
    )

    episode_repo = EpisodeRepository(
        csv_path=str(results_dir / "episodes.csv")
    )

    data_repo = ExperimentDataRepository(
        experiment_id=experiment_id,
        phase="pilot",
        raw_root=str(raw_root),
        processed_root=str(processed_root),
    )
    raw_path = data_repo.raw_jsonl

    print()
    print("Starting ONE live spot check...")
    print(f"experiment_id: {experiment_id}")
    print(f"fault: {scenario.fault_type}")
    print(f"intensity: {scenario.intensity_label}")
    print(f"agent: {args.provider} / {model}")
    print()
    if args.provider == "openai":
        print("The API key value will NOT be printed.")
    if args.provider == "ollama":
        print(f"ollama_base_url: {args.ollama_base_url}")
    print()

    # ---------------------------------------------------------
    # Execute the exact production matrix path.
    # ---------------------------------------------------------

    try:
        result, record = execute_matrix_episode(
            scenario,
            experiment_id=experiment_id,
            telemetry=telemetry,
            episode_repo=episode_repo,
            data_repo=data_repo,
            agent_config=agent_config,
            llm_min_interval=0.5,
            master_seed=42,
            model_identifier=model,
        )

    except Exception as exc:
        print()
        print("SPOT CHECK CRASHED")
        print(f"exception_type: {type(exc).__name__}")
        print(f"exception: {exc}")
        print()
        return 1

    print_result(
        result=result,
        record=record,
        scenario=scenario,
        raw_path=raw_path,
        provider=args.provider,
        model=model,
    )

    # ---------------------------------------------------------
    # Validate that the episode actually executed.
    # ---------------------------------------------------------

    token_count = getattr(record, "llm_total_tokens", 0) or 0
    sim_steps = getattr(record, "total_sim_steps", 0) or 0

    if record.crashed:
        print(
            "WARNING: Episode crashed. "
            "This is NOT a valid experimental result."
        )
        return 2

    if token_count == 0:
        print(
            "WARNING: Zero LLM tokens recorded. "
            f"The live {args.provider} planner path may not have reported token usage."
        )
        return 3

    if sim_steps == 0:
        print(
            "WARNING: Zero simulation steps recorded. "
            "The robot body did not meaningfully execute."
        )
        return 4

    if args.fault != "none" and not record.fault_occurred:
        print(
            "WARNING: Requested fault was not observed in telemetry."
        )
        return 5

    print("SPOT CHECK VALIDATION: PASSED")
    print(
        "The episode reached meaningful LLM + "
        "simulation execution."
    )

    if args.fault != "none":
        print(
            f"Fault '{args.fault}' was observed by telemetry."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
