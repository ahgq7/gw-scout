from __future__ import annotations

from typing import List, Tuple


def good_segments_for_block(start: float, end: float) -> List[Tuple[float, float]]:
    # Placeholder: treat entire block as good. Extend with DQ masks if available.
    return [(start, end)]


def clip_segments(segments: List[Tuple[float, float]], window: Tuple[float, float]) -> List[Tuple[float, float]]:
    ws, we = window
    out = []
    for s, e in segments:
        s2, e2 = max(s, ws), min(e, we)
        if s2 < e2:
            out.append((s2, e2))
    return out
