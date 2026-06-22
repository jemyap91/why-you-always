"""Append-only SQLite event log. Totals are DERIVED by querying the log, so a
bad frame can never corrupt cumulative state (event sourcing)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from dishcounter.domain import WashEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person    TEXT    NOT NULL,
    timestamp REAL    NOT NULL,
    confidence REAL   NOT NULL,
    counted   INTEGER NOT NULL
);
"""


class CountStore:
    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def record(self, event: WashEvent) -> None:
        counted = 1 if event.person in ("You", "Wife") else 0
        self._conn.execute(
            "INSERT INTO events (person, timestamp, confidence, counted) "
            "VALUES (?, ?, ?, ?)",
            (event.person, event.timestamp, event.confidence, counted),
        )
        self._conn.commit()

    def totals(self, now: float) -> dict:
        all_time = self._tally()
        start, end = self._day_bounds(now)
        today = self._tally(start, end)
        return {"today": today, "all_time": all_time}

    def _tally(self, start: float | None = None, end: float | None = None) -> dict:
        sql = (
            "SELECT person, COUNT(*) FROM events "
            "WHERE counted = 1 AND person IN ('You', 'Wife')"
        )
        params: list[float] = []
        if start is not None:
            sql += " AND timestamp >= ? AND timestamp < ?"
            params += [start, end]
        sql += " GROUP BY person"
        result = {"You": 0, "Wife": 0}
        for person, count in self._conn.execute(sql, params):
            result[person] = count
        return result

    @staticmethod
    def _day_bounds(now: float) -> tuple[float, float]:
        d = datetime.fromtimestamp(now)
        start = datetime(d.year, d.month, d.day).timestamp()
        return (start, start + 86400.0)
