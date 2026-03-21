"""Reading order sort for manga bubbles (P1: right-to-left, top-to-bottom)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.pipeline.bubble_detector import BubbleInfo


def sort_bubbles_rtl(bubbles: list[BubbleInfo]) -> list[BubbleInfo]:
    """Sort bubbles in Japanese manga reading order: right-to-left, top-to-bottom.

    Bubbles on the same row (y-center difference < 50% of bbox height) are
    sorted by x descending (right-to-left).  Rows themselves are sorted by
    y ascending (top-to-bottom).
    """
    if not bubbles:
        return []

    # Build list of (bubble, cx, cy, h)
    items = []
    for b in bubbles:
        x, y, w, h = b.bbox
        cx = x + w / 2
        cy = y + h / 2
        items.append((b, cx, cy, h))

    # Sort by cy first to group into rows
    items.sort(key=lambda t: t[2])

    # Group into rows: consecutive bubbles whose cy difference < 50% of avg height
    rows: list[list[tuple]] = []
    current_row: list[tuple] = [items[0]]

    for item in items[1:]:
        prev_cy = current_row[-1][2]
        prev_h = current_row[-1][3]
        cur_cy = item[2]
        cur_h = item[3]
        avg_h = (prev_h + cur_h) / 2
        if abs(cur_cy - prev_cy) <= avg_h * 0.5:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    # Within each row, sort by cx descending (right-to-left)
    sorted_bubbles: list[BubbleInfo] = []
    for row in rows:
        row.sort(key=lambda t: t[1], reverse=True)
        sorted_bubbles.extend(item[0] for item in row)

    # Assign reading_order
    for idx, bubble in enumerate(sorted_bubbles):
        bubble.reading_order = idx + 1

    return sorted_bubbles
