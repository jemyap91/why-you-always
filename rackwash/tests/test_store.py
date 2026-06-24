from datetime import datetime

from rackwash.domain import WashEvent
from rackwash.store import CountStore


def test_store_usable_from_a_thread_other_than_the_one_that_created_it():
    """The store is built on the main thread but driven by the engine thread."""
    import threading

    store = CountStore(":memory:")
    now = _ts(2026, 6, 22)
    errors: list[Exception] = []

    def use_store() -> None:
        try:
            store.record(WashEvent("You", now, 0.9, 1))
            store.totals(now)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=use_store)
    t.start()
    t.join()

    assert errors == []


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
    path = tmp_path / "rackwash.db"
    now = _ts(2026, 6, 22)
    CountStore(path).record(WashEvent("Wife", now, 0.8, 2))
    reopened = CountStore(path)
    assert reopened.totals(now)["all_time"] == {"You": 0, "Wife": 1}


def test_reset_zeroes_totals_but_keeps_counting_after():
    store = CountStore(":memory:")
    t = _ts(2026, 6, 22)
    store.record(WashEvent("You", t, 1.0, 1))
    store.record(WashEvent("Wife", t, 1.0, 2))
    assert store.totals(t)["all_time"] == {"You": 1, "Wife": 1}

    store.reset(t + 1)
    assert store.totals(t + 2)["all_time"] == {"You": 0, "Wife": 0}
    assert store.totals(t + 2)["today"] == {"You": 0, "Wife": 0}

    store.record(WashEvent("You", t + 3, 1.0, 3))
    assert store.totals(t + 4)["all_time"] == {"You": 1, "Wife": 0}
    assert store.totals(t + 4)["today"] == {"You": 1, "Wife": 0}


def test_reset_marker_is_not_counted():
    store = CountStore(":memory:")
    t = _ts(2026, 6, 22)
    store.reset(t)
    assert store.totals(t)["all_time"] == {"You": 0, "Wife": 0}
