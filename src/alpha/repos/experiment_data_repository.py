"""
Append-only experiment data storage for pilot and final study batches.

Raw JSONL traces are immutable from the runner's perspective. Processed CSV
summaries are written alongside under ``data/processed/``.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha.config import settings
from alpha.schemas.telemetry import ExperimentTelemetryRecord


class ExperimentDataRepository:
    """Write raw episode traces and mirror summary rows to processed storage."""

    def __init__(
        self,
        *,
        experiment_id: str,
        phase: str = "pilot",
        raw_root: str = settings.RAW_DATA_DIR,
        processed_root: str = settings.PROCESSED_DATA_DIR,
    ) -> None:
        if phase not in {"pilot", "phase_b", "final"}:
            raise ValueError("phase must be 'pilot', 'phase_b', or 'final'")
        self.experiment_id = experiment_id
        self.phase = phase
        self.raw_dir = Path(raw_root) / phase / experiment_id
        self.processed_dir = Path(processed_root) / phase / experiment_id
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_jsonl = self.raw_dir / "episodes.jsonl"
        self.summary_csv = self.processed_dir / "episodes.csv"

    def append_raw_episode(
        self,
        *,
        experiment_id: str,
        episode_id: str,
        record: ExperimentTelemetryRecord,
        execution_log_json: str = "",
        action_plan_json: str = "",
    ) -> None:
        payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "episode_id": episode_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python_version": sys.version,
                "platform": platform.platform(),
            },
            "summary": record.model_dump(mode="json"),
            "execution_log_json": execution_log_json,
            "action_plan_json": action_plan_json,
        }
        payload["summary"]["injected_fault_type"] = record.injected_fault_type.value
        with self.raw_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

        self._append_summary_csv(record)

    def _append_summary_csv(self, record: ExperimentTelemetryRecord) -> None:
        import csv

        row = record.to_csv_row()
        fieldnames = list(row.keys())
        exists = self.summary_csv.is_file()
        if exists:
            with self.summary_csv.open(newline="", encoding="utf-8") as handle:
                existing = next(csv.reader(handle), [])
            if existing and existing != fieldnames:
                self._migrate_header(existing, fieldnames)
        with self.summary_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _migrate_header(self, existing_header: list[str], expected_fields: list[str]) -> None:
        import csv

        with self.summary_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with self.summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=expected_fields)
            writer.writeheader()
            for old_row in rows:
                writer.writerow({field: old_row.get(field, "") for field in expected_fields})
