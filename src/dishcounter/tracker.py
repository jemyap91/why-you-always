"""Greedy IoU tracker: assigns stable integer ids to hands across frames so
the ZoneEventEngine can reason about 'the same hand' over time."""

from __future__ import annotations

from dishcounter.domain import Hand

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


class HandTracker:
    def __init__(self, iou_threshold: float = 0.3) -> None:
        self._iou_threshold = iou_threshold
        self._next_id = 0
        self._tracks: dict[int, Box] = {}  # id -> last bbox

    def update(self, hands: list[Hand]) -> list[Hand]:
        unmatched_tracks = dict(self._tracks)
        new_tracks: dict[int, Box] = {}

        # Greedy: best (hand, track) IoU pairs first.
        candidates = [
            (iou(h.bbox, box), idx, tid)
            for idx, h in enumerate(hands)
            for tid, box in unmatched_tracks.items()
        ]
        candidates.sort(reverse=True)

        assigned_hand: dict[int, int] = {}  # hand idx -> track id
        for score, idx, tid in candidates:
            if score < self._iou_threshold:
                break
            if idx in assigned_hand or tid not in unmatched_tracks:
                continue
            assigned_hand[idx] = tid
            del unmatched_tracks[tid]

        for idx, hand in enumerate(hands):
            if idx in assigned_hand:
                hand.id = assigned_hand[idx]
            else:
                hand.id = self._next_id
                self._next_id += 1
            new_tracks[hand.id] = hand.bbox

        self._tracks = new_tracks
        return hands
