"""Observability: structured logging only (section 5 — no separate monitoring
stack). Each generation step is recorded as one row; rows are dumped to
CSV/JSON for later analysis (section 11's evaluation chapter)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class StepEvent:
    step_index: int
    route: str
    token_id: int
    confidence: float
    switched: bool


class EventLog:
    def __init__(self) -> None:
        self._events: list[StepEvent] = []

    def record(self, event: StepEvent) -> None:
        self._events.append(event)

    def events(self) -> list[StepEvent]:
        return list(self._events)

    def switch_count(self) -> int:
        return sum(1 for event in self._events if event.switched)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([asdict(event) for event in self._events], ensure_ascii=False, indent=2)
        )

    def to_csv(self, path: str | Path) -> None:
        rows = [asdict(event) for event in self._events]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
