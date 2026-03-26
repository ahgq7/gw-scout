import math

from gwsearch import dq


def test_clip_segments():
    segs = [(0, 10), (15, 25)]
    window = (5, 20)
    clipped = dq.clip_segments(segs, window)
    assert clipped == [(5, 10), (15, 20)]


def test_good_segments_for_block():
    start, end = 100.0, 200.0
    segs = dq.good_segments_for_block(start, end)
    assert segs == [(start, end)]
