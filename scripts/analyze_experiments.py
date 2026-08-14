#!/usr/bin/env python3
"""
Analyze experiment telemetry.

Reads an experiments CSV, computes post-fault completion time (single-shot agent),
safety violation rate, and plots a basic degradation curve.

This script does **not** report MTTR/TTR — those labels require a genuine
replanning/recovery loop that is not implemented in Phase A.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha.config import settings
from alpha.services.metrics_service import MetricsService


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze chaos experiment telemetry.")
    parser.add_argument("--csv", default=settings.EXPERIMENTS_CSV, help="Path to experiments_log.csv")
    parser.add_argument(
        "--output",
        default=f"{settings.RESULTS_DIR}/degradation_curve.png",
        help="Where to save the degradation curve plot",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required for analysis. Install with: pip install -e '.[analytics]'"
        ) from exc

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"No telemetry CSV found at {csv_path}. Run instrumented episodes first.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"Telemetry CSV at {csv_path} is empty.")

    pfct_col = "post_fault_completion_time_sim_steps"
    if pfct_col not in df.columns and "ttr_sim_steps" in df.columns:
        pfct_col = "ttr_sim_steps"

    summary = MetricsService().summarize(csv_path=str(csv_path))

    fault_rows = df[df["fault_occurred"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if pfct_col in fault_rows.columns:
        finite = fault_rows[fault_rows[pfct_col].astype(str).notna()]
        finite = finite[finite[pfct_col].astype(str) != "inf"]
        finite[pfct_col] = pd.to_numeric(finite[pfct_col], errors="coerce")
        mean_pfct = finite[pfct_col].mean() if not finite.empty else None
    else:
        mean_pfct = None

    print("Experiment summary")
    print(f"  episodes: {summary.episode_count}")
    print(f"  mean post-fault completion time (sim steps): {summary.mean_post_fault_completion_time_sim_steps}")
    print(f"  mean post-fault completion time (wall seconds): {summary.mean_post_fault_completion_time_wall_seconds}")
    if mean_pfct is not None:
        print(f"  pandas mean post-fault completion time (sim steps): {mean_pfct}")
    if summary.fault_classification_accuracy is not None:
        print(f"  fault-classification accuracy: {summary.fault_classification_accuracy:.2%}")
    print(f"  safety-violation rate (episode-level): {summary.safety_violation_rate:.2%}")
    print(f"  mean safety-violation count per episode: {summary.mean_safety_violation_count:.2f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install -e '.[analytics]'"
        ) from exc

    curve_df = (
        df.groupby("fault_intensity", as_index=False)
        .agg(
            mean_total_sim_steps=("total_sim_steps", "mean"),
            mean_total_wall_seconds=("total_wall_seconds", "mean"),
            mean_llm_total_tokens=("llm_total_tokens", "mean"),
            sample_count=("episode_id", "count"),
        )
        .sort_values("fault_intensity")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Degradation Curve Matrix")

    axes[0].plot(
        curve_df["fault_intensity"],
        curve_df["mean_total_sim_steps"],
        marker="o",
        label="Sim steps",
    )
    axes[0].plot(
        curve_df["fault_intensity"],
        curve_df["mean_total_wall_seconds"],
        marker="s",
        label="Wall seconds",
    )
    axes[0].set_xlabel("Fault intensity")
    axes[0].set_ylabel("Execution time")
    axes[0].set_title("Time cost vs intensity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        curve_df["fault_intensity"].astype(str),
        curve_df["mean_llm_total_tokens"],
        alpha=0.8,
    )
    axes[1].set_xlabel("Fault intensity")
    axes[1].set_ylabel("Mean LLM tokens")
    axes[1].set_title("Token cost vs intensity")
    axes[1].grid(True, axis="y", alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

    print(f"Saved degradation curve plot to {output_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
