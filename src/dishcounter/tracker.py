"""Greedy IoU tracker: assigns stable integer ids to any boxed object across
frames so downstream logic can reason about 'the same object' over time."""

from __future__ import annotations

Box = tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


class IouTracker:
    def __init__(self, iou_threshold: float = 0.3) -> None:
        self._iou_threshold = iou_threshold
        self._next_id = 0
        self._tracks: dict[int, Box] = {}  # id -> last bbox

    def update(self, items: list) -> list:
        unmatched_tracks = dict(self._tracks)
        new_tracks: dict[int, Box] = {}

        # Greedy: best (item, track) IoU pairs first.
        candidates = [
            (iou(it.bbox, box), idx, tid)
            for idx, it in enumerate(items)
            for tid, box in unmatched_tracks.items()
        ]
        candidates.sort(reverse=True)

        assigned: dict[int, int] = {}  # item idx -> track id
        for score, idx, tid in candidates:
            if score < self._iou_threshold:
                break
            if idx in assigned or tid not in unmatched_tracks:
                continue
            assigned[idx] = tid
            del unmatched_tracks[tid]

        for idx, item in enumerate(items):
            if idx in assigned:
                item.id = assigned[idx]
            else:
                item.id = self._next_id
                self._next_id += 1
            new_tracks[item.id] = item.bbox

        self._tracks = new_tracks
        return items


HandTracker = IouTracker  # back-compat: hands are just boxed objects
