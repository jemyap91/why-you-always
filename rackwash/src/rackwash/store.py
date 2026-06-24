"""Append-only SQLite event log. Totals are DERIVED by querying the log, so a
bad frame can never corrupt cumulative state (event sourcing)."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from rackwash.domain import WashEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person    TEXT    NOT NULL,
    timestamp REAL    NOT NULL,
    confidence REAL   NOT NULL,
    counted   INTEGER NOT NULL
);
"""


_RESET_MARKER = "__reset__"


class CountStore:
    def __init__(self, db_path: str | Path) -> None:
        # Built on the main thread, written by the engine thread, and now also
        # written by the web thread (reset). check_same_thread is off and a lock
        # serializes every public access so the shared connection is safe.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def record(self, event: WashEvent) -> None:
        counted = 1 if event.person in ("You", "Wife") else 0
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (person, timestamp, confidence, counted) "
                "VALUES (?, ?, ?, ?)",
                (event.person, event.timestamp, event.confidence, counted),
            )
            self._conn.commit()

    def reset(self, now: float) -> None:
        # Non-destructive: append a marker; totals only count events after it.
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (person, timestamp, confidence, counted) "
                "VALUES (?, ?, ?, ?)",
                (_RESET_MARKER, now, 0.0, 0),
            )
            self._conn.commit()

    def totals(self, now: float) -> dict:
        with self._lock:
            floor = self._latest_reset()
            all_time = self._tally(floor)
            start, end = self._day_bounds(now)
            today = self._tally(max(start, floor), end)
            return {"today": today, "all_time": all_time}

    def _latest_reset(self) -> float:
        row = self._conn.execute(
            "SELECT MAX(timestamp) FROM events WHERE person = ?", (_RESET_MARKER,)
        ).fetchone()
        return row[0] if row and row[0] is not None else 0.0

    def _tally(self, start: float, end: float | None = None) -> dict:
        sql = (
            "SELECT person, COUNT(*) FROM events "
            "WHERE counted = 1 AND person IN ('You', 'Wife') AND timestamp >= ?"
        )
        params: list[float] = [start]
        if end is not None:
            sql += " AND timestamp < ?"
            params.append(end)
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
