"""Reading order tests for MangaLens (P1: right-to-left, top-to-bottom).

Tests cover:
- 4 bubbles in 2x2 grid → RTL, top-to-bottom order
- Same-row grouping (y-center difference ≤ 50% of avg height)
- Single bubble → trivial case
- Empty list → empty result
- Three rows with varying bubble counts
"""

from __future__ import annotations

import pytest

from server.pipeline.bubble_detector import BubbleInfo
from server.utils.reading_order import sort_bubbles_rtl


# ---------------------------------------------------------------------------
# Basic ordering
# ---------------------------------------------------------------------------

class TestReadingOrderBasic:
    def test_empty_list(self):
        result = sort_bubbles_rtl([])
        assert result == []

    def test_single_bubble(self):
        b = BubbleInfo(id=1, bbox=(100, 100, 80, 100))
        result = sort_bubbles_rtl([b])
        assert len(result) == 1
        assert result[0].id == 1
        assert result[0].reading_order == 1

    def test_two_bubbles_same_row_rtl(self):
        """Two bubbles at same y → right one first."""
        left = BubbleInfo(id=1, bbox=(100, 100, 80, 100))
        right = BubbleInfo(id=2, bbox=(800, 100, 80, 100))
        result = sort_bubbles_rtl([left, right])
        assert [b.id for b in result] == [2, 1]
        assert result[0].reading_order == 1
        assert result[1].reading_order == 2


# ---------------------------------------------------------------------------
# P1 Principle: 4 bubbles in 2x2 grid, RTL + top-to-bottom
# ---------------------------------------------------------------------------

class TestReadingOrder2x2Grid:
    def test_four_bubbles_rtl_ttb(self):
        """
        Grid layout:
          top-left(1)    top-right(2)
          bot-left(3)    bot-right(4)
        Expected reading order: 2 → 1 → 4 → 3
        """
        bubbles = [
            BubbleInfo(id=1, bbox=(100, 100, 80, 100)),   # top-left
            BubbleInfo(id=2, bbox=(800, 100, 80, 100)),   # top-right
            BubbleInfo(id=3, bbox=(100, 500, 80, 100)),   # bottom-left
            BubbleInfo(id=4, bbox=(800, 500, 80, 100)),   # bottom-right
        ]
        result = sort_bubbles_rtl(bubbles)
        ids = [b.id for b in result]
        assert ids == [2, 1, 4, 3], f"Expected [2, 1, 4, 3], got {ids}"

    def test_reading_order_assigned_correctly(self):
        bubbles = [
            BubbleInfo(id=1, bbox=(100, 100, 80, 100)),
            BubbleInfo(id=2, bbox=(800, 100, 80, 100)),
            BubbleInfo(id=3, bbox=(100, 500, 80, 100)),
            BubbleInfo(id=4, bbox=(800, 500, 80, 100)),
        ]
        result = sort_bubbles_rtl(bubbles)
        orders = [(b.id, b.reading_order) for b in result]
        assert orders == [(2, 1), (1, 2), (4, 3), (3, 4)]


# ---------------------------------------------------------------------------
# Same-row detection (y-center difference ≤ 50% of avg height)
# ---------------------------------------------------------------------------

class TestSameRowGrouping:
    def test_slightly_offset_same_row(self):
        """Bubbles with small vertical offset should be in the same row."""
        # height=100, 50% = 50. cy difference = 20 (< 50) → same row
        b1 = BubbleInfo(id=1, bbox=(100, 100, 80, 100))  # cy=150
        b2 = BubbleInfo(id=2, bbox=(800, 120, 80, 100))  # cy=170, diff=20
        result = sort_bubbles_rtl([b1, b2])
        # Same row → RTL: right(2) first
        assert [b.id for b in result] == [2, 1]

    def test_large_offset_different_rows(self):
        """Bubbles with large vertical offset should be in different rows."""
        # height=100, 50% = 50. cy difference = 300 (> 50) → different rows
        b1 = BubbleInfo(id=1, bbox=(800, 100, 80, 100))   # cy=150, right
        b2 = BubbleInfo(id=2, bbox=(100, 400, 80, 100))   # cy=450, left
        result = sort_bubbles_rtl([b1, b2])
        # Different rows → top-to-bottom: 1 first (cy=150), then 2 (cy=450)
        assert [b.id for b in result] == [1, 2]


# ---------------------------------------------------------------------------
# Multiple rows with varying bubble counts
# ---------------------------------------------------------------------------

class TestMultipleRows:
    def test_three_rows(self):
        """
        Row 1 (top): 3 bubbles
        Row 2 (mid): 1 bubble
        Row 3 (bot): 2 bubbles
        """
        bubbles = [
            BubbleInfo(id=1, bbox=(100, 50, 60, 80)),     # row1, left
            BubbleInfo(id=2, bbox=(400, 50, 60, 80)),     # row1, center
            BubbleInfo(id=3, bbox=(700, 50, 60, 80)),     # row1, right
            BubbleInfo(id=4, bbox=(400, 300, 60, 80)),    # row2, center
            BubbleInfo(id=5, bbox=(200, 550, 60, 80)),    # row3, left
            BubbleInfo(id=6, bbox=(600, 550, 60, 80)),    # row3, right
        ]
        result = sort_bubbles_rtl(bubbles)
        ids = [b.id for b in result]
        # Row 1 RTL: 3, 2, 1; Row 2: 4; Row 3 RTL: 6, 5
        assert ids == [3, 2, 1, 4, 6, 5]


# ---------------------------------------------------------------------------
# Input order independence
# ---------------------------------------------------------------------------

class TestInputOrderIndependence:
    def test_shuffled_input_gives_same_result(self):
        """Result should be deterministic regardless of input order."""
        bubbles_a = [
            BubbleInfo(id=1, bbox=(100, 100, 80, 100)),
            BubbleInfo(id=2, bbox=(800, 100, 80, 100)),
            BubbleInfo(id=3, bbox=(100, 500, 80, 100)),
            BubbleInfo(id=4, bbox=(800, 500, 80, 100)),
        ]
        bubbles_b = [
            BubbleInfo(id=4, bbox=(800, 500, 80, 100)),
            BubbleInfo(id=1, bbox=(100, 100, 80, 100)),
            BubbleInfo(id=3, bbox=(100, 500, 80, 100)),
            BubbleInfo(id=2, bbox=(800, 100, 80, 100)),
        ]
        result_a = [b.id for b in sort_bubbles_rtl(bubbles_a)]
        result_b = [b.id for b in sort_bubbles_rtl(bubbles_b)]
        assert result_a == result_b == [2, 1, 4, 3]
