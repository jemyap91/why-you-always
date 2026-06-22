from datetime import datetime

from dishcounter.domain import WashEvent
from dishcounter.store import CountStore


def _ts(y, mo, d, h=12):
    return datetime(y, mo, d, h, 0, 0).timestamp()


def test_counts_attributed_events_today():
    store = CountStore(":memory:")
    now = _ts(2026, 6, 22)
    store.record(WashEvent("You", now, 0.9, 1))
    store.record(WashEvent("You", now, 0.9, 1))
    store.record(WashEvent("Wife", now, 0.8, 2))
    totals = store.totals(now)
    assert totals["today"] == {"You": 2, "Wife": 1}
    assert totals["all_time"] == {"You": 2, "Wife": 1}


def test_uncertain_events_are_recorded_but_not_counted():
    store = CountStore(":memory:")
    now = _ts(2026, 6, 22)
    store.record(WashEvent("uncertain", now, 0.0, 3))
    totals = store.totals(now)
    assert totals["today"] == {"You": 0, "Wife": 0}
    assert totals["all_time"] == {"You": 0, "Wife": 0}


def test_today_excludes_other_days_but_all_time_includes_them():
    store = CountStore(":memory:")
    yesterday = _ts(2026, 6, 21)
    today = _ts(2026, 6, 22)
    store.record(WashEvent("You", yesterday, 0.9, 1))
    store.record(WashEvent("You", today, 0.9, 1))
    totals = store.totals(today)
    assert totals["today"]["You"] == 1
    assert totals["all_time"]["You"] == 2


def test_counts_survive_reopening_the_database(tmp_path):
    path = tmp_path / "dishcounter.db"
    now = _ts(2026, 6, 22)
    CountStore(path).record(WashEvent("Wife", now, 0.8, 2))
    reopened = CountStore(path)
    assert reopened.totals(now)["all_time"] == {"You": 0, "Wife": 1}
