"""
Persistence for episode results.

CSV for now (fine at 50-500 episodes per the PRD's success metric) — this
is the only module that knows that. Weeks 5-6 read through this repository,
not through raw file I/O, so switching to SQLite/Postgres later means
changing this file only.
"""

import csv
import os
from pathlib import Path

from alpha.core.entities import EpisodeResult
from alpha.schemas.episode import EpisodeResultSchema
from alpha.config import settings


class EpisodeRepository:
    def __init__(self, csv_path: str = settings.EPISODES_CSV) -> None:
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)

    def save(self, result: EpisodeResult) -> None:
        schema = EpisodeResultSchema.from_entity(result)
        row = schema.model_dump(mode="json")
        file_exists = os.path.isfile(self.csv_path)
        expected_fields = list(row.keys())
        if file_exists:
            with open(self.csv_path, newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
            if existing_header and existing_header != expected_fields:
                self._migrate_header(existing_header, expected_fields)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=expected_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _migrate_header(self, existing_header: list[str], expected_fields: list[str]) -> None:
        path = Path(self.csv_path)
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=expected_fields)
            writer.writeheader()
            for old_row in rows:
                writer.writerow({field: old_row.get(field, "") for field in expected_fields})

    def load_all(self) -> list[dict]:
        if not os.path.isfile(self.csv_path):
            return []
        with open(self.csv_path, newline="") as f:
            return list(csv.DictReader(f))
