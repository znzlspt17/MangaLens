"""Text rendering using Pillow + FreeType (CJK engine).

OpenCV putText is NOT used (P6).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_PADDING = 6
_MIN_FONT_SIZE = 12
_TEXT_COLOR = (0, 0, 0)
_STROKE_COLOR = (255, 255, 255)
_FONT_WEIGHT = 700  # Bold weight for variable fonts

# B-4: Dynamic line height ratio based on number of lines
_LINE_HEIGHT_RATIO_FEW = 1.3    # 1-2 lines
_LINE_HEIGHT_RATIO_MANY = 1.2   # 3+ lines

# B-3: Small bubble threshold — no scale-down below this size
_SMALL_BUBBLE_THRESHOLD = 60

# P2: Module-level font object cache keyed by (font_path, size).
# _find_best_font_size() calls _load_font() O(log N) times during binary
# search; caching avoids re-reading the font file on every iteration.
_font_cache: dict[tuple[str | None, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


class TextRenderer:
    """Render translated text onto bubble images using Pillow.

    Features:
    - CJK font support via FreeType.
    - Auto font-size fitting within the bubble bbox.
    - Automatic line wrapping.
    - Supports both vertical and horizontal text layout.
    """

    def __init__(self, font_dir: str = "./fonts") -> None:
        self.font_dir = font_dir
        self._font_path: str | None = None

        font_path = Path(font_dir)
        if font_path.is_dir():
            # Prefer variable font, then Bold, then any .ttf/.otf
            candidates = sorted(font_path.iterdir())
            for f in candidates:
                if f.suffix.lower() in (".ttf", ".otf") and "variable" in f.stem.lower():
                    self._font_path = str(f)
                    break
            if self._font_path is None:
                for f in candidates:
                    if f.suffix.lower() in (".ttf", ".otf") and "bold" in f.stem.lower():
                        self._font_path = str(f)
                        break
            if self._font_path is None:
                for f in candidates:
                    if f.suffix.lower() in (".ttf", ".otf"):
                        self._font_path = str(f)
                        break
        self._is_variable_font: bool = False
        if self._font_path:
            logger.info("Using font: %s", self._font_path)
            # Detect variable font once at init
            try:
                _probe = ImageFont.truetype(self._font_path, 12)
                _probe.set_variation_by_axes([_FONT_WEIGHT])
                self._is_variable_font = True
            except Exception:
                self._is_variable_font = False

        if self._font_path is None:
            logger.warning(
                "No .ttf/.otf font found in %s — falling back to Pillow default font",
                font_dir,
            )

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load font at the given size with bold weight (cached)."""
        cache_key = (self._font_path, size)
        if cache_key in _font_cache:
            return _font_cache[cache_key]
        if self._font_path:
            font = ImageFont.truetype(self._font_path, size)
            if self._is_variable_font:
                font.set_variation_by_axes([_FONT_WEIGHT])
        else:
            font = ImageFont.load_default(size=size)
        _font_cache[cache_key] = font
        return font

    async def render(
        self,
        bubble_image: np.ndarray,
        text: str,
        bbox: tuple[int, int, int, int],
        text_direction: str = "horizontal",
    ) -> tuple[np.ndarray, int]:
        """Render translated text onto the bubble image.

        Args:
            bubble_image: Cleaned (inpainted) bubble region (H, W, 3) BGR numpy.
            text: Translated text string to render.
            bbox: Bounding box ``(x, y, w, h)`` within the bubble.
            text_direction: ``"vertical"`` or ``"horizontal"``.

        Returns:
            Tuple of (RGBA numpy array (H, W, 4) with rendered text, font_size used).
        """
        if not text or not text.strip():
            h, w = bubble_image.shape[:2]
            return np.zeros((h, w, 4), dtype=np.uint8), 0

        _, _, bw, bh = bbox
        usable_w = max(bw - _PADDING * 2, 1)
        usable_h = max(bh - _PADDING * 2, 1)

        # Korean (and most translated target languages) use horizontal writing only.
        # Vertical layout is only appropriate for the source Japanese — never for
        # the rendered translation.
        use_vertical = False

        font_size = self._find_best_font_size(text, usable_w, usable_h, use_vertical)
        # B-3: Adaptive scale-down — skip for small bubbles where space is precious
        if bw > _SMALL_BUBBLE_THRESHOLD and bh > _SMALL_BUBBLE_THRESHOLD:
            font_size = max(int(font_size * 0.85), _MIN_FONT_SIZE)
        font = self._load_font(font_size)

        # B-3: Adaptive stroke width proportional to font size
        stroke_width = max(1, font_size // 20)

        # Pre-compute lines to know actual height needed
        # Subtract stroke width from both sides so the stroke outline never
        # bleeds past the overlay edge (left/right clipping fix).
        wrap_w = max(usable_w - 2 * stroke_width, 1)
        lines = self._wrap_text(text, font, wrap_w) if not use_vertical else None

        # B-4: Dynamic line height ratio
        num_lines = len(lines) if lines else 1
        line_height_ratio = _LINE_HEIGHT_RATIO_FEW if num_lines <= 2 else _LINE_HEIGHT_RATIO_MANY
        line_height = int(font_size * line_height_ratio)

        # Narrow-bubble fix: reduce font size until text fits inside bh.
        # Never expand the overlay beyond the actual bubble height — doing so
        # causes text to bleed outside the speech bubble.
        if not use_vertical and lines is not None:
            needed_h = len(lines) * line_height + _PADDING * 2
            while needed_h > bh and font_size > _MIN_FONT_SIZE:
                font_size -= 1
                font = self._load_font(font_size)
                stroke_width = max(1, font_size // 20)
                wrap_w = max(usable_w - 2 * stroke_width, 1)
                lines = self._wrap_text(text, font, wrap_w)
                num_lines = len(lines)
                line_height_ratio = _LINE_HEIGHT_RATIO_FEW if num_lines <= 2 else _LINE_HEIGHT_RATIO_MANY
                line_height = int(font_size * line_height_ratio)
                needed_h = len(lines) * line_height + _PADDING * 2

        actual_bh = bh  # Never exceed bubble bounds

        # Create RGBA overlay — may be taller than bbox to prevent text clipping
        overlay = Image.new("RGBA", (bw, actual_bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if use_vertical:
            self._draw_vertical(draw, font, text, usable_w, actual_bh - _PADDING * 2, font_size, stroke_width)
        else:
            self._draw_horizontal(draw, font, lines or [""], usable_w, actual_bh - _PADDING * 2, font_size, stroke_width)

        return np.array(overlay, dtype=np.uint8), font_size

    def _draw_horizontal(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        lines: list[str],
        usable_w: int,
        usable_h: int,
        font_size: int,
        stroke_width: int = 2,
    ) -> None:
        """Draw lines of text horizontally, centered in the bbox."""
        num_lines = len(lines)
        line_height_ratio = _LINE_HEIGHT_RATIO_FEW if num_lines <= 2 else _LINE_HEIGHT_RATIO_MANY
        line_height = int(font_size * line_height_ratio)
        total_text_height = line_height * num_lines
        y_start = _PADDING + max((usable_h - total_text_height) // 2, 0)

        for i, line in enumerate(lines):
            line_w = font.getlength(line)
            # Ensure x is at least stroke_width so the left stroke outline
            # never falls outside (or right at the edge of) the overlay.
            x = max(stroke_width, _PADDING + max((usable_w - int(line_w)) // 2, 0))
            y = y_start + i * line_height
            draw.text(
                (x, y),
                line,
                fill=(*_TEXT_COLOR, 255),
                font=font,
                stroke_width=stroke_width,
                stroke_fill=(*_STROKE_COLOR, 255),
            )

    def _draw_vertical(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        text: str,
        usable_w: int,
        usable_h: int,
        font_size: int,
        stroke_width: int = 2,
    ) -> None:
        """Draw text vertically: columns right-to-left, 1 char per cell."""
        col_width = int(font_size * _LINE_HEIGHT_RATIO_FEW)
        line_height = col_width

        # How many columns actually fit in usable_w
        max_cols = max(1, usable_w // col_width)
        # Adapt chars_per_col so all text is distributed across available columns
        import math
        chars_per_col = max(usable_h // line_height, 1)
        needed_cols = math.ceil(len(text) / chars_per_col)
        if needed_cols > max_cols:
            # Increase chars per column to fit within max_cols
            chars_per_col = math.ceil(len(text) / max_cols)

        columns: list[str] = []
        for i in range(0, len(text), chars_per_col):
            columns.append(text[i : i + chars_per_col])

        total_cols_width = col_width * len(columns)
        # Right-to-left column order: first column on the right
        x_start = _PADDING + min(usable_w, total_cols_width) - col_width

        for col_idx, col_text in enumerate(columns):
            x = x_start - col_idx * col_width
            if x < _PADDING:
                break
            line_height = int(font_size * _LINE_HEIGHT_RATIO_FEW)
            total_col_height = line_height * len(col_text)
            y_offset = _PADDING + max((usable_h - total_col_height) // 2, 0)
            for char_idx, ch in enumerate(col_text):
                char_w = font.getlength(ch)
                cx = x + max((col_width - int(char_w)) // 2, 0)
                cy = y_offset + char_idx * line_height
                draw.text(
                    (cx, cy),
                    ch,
                    fill=(*_TEXT_COLOR, 255),
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill=(*_STROKE_COLOR, 255),
                )

    def _find_best_font_size(
        self,
        text: str,
        usable_w: int,
        usable_h: int,
        vertical: bool,
    ) -> int:
        """Binary-search for the largest font size that fits the text in the bbox."""
        lo = _MIN_FONT_SIZE
        hi = min(usable_w, usable_h)
        if hi < lo:
            return lo

        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._text_fits(text, mid, usable_w, usable_h, vertical):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _text_fits(
        self,
        text: str,
        font_size: int,
        usable_w: int,
        usable_h: int,
        vertical: bool,
    ) -> bool:
        """Check whether text at the given font size fits in the usable area."""
        font = self._load_font(font_size)

        if vertical:
            line_height = int(font_size * _LINE_HEIGHT_RATIO_MANY)
            chars_per_col = max(usable_h // line_height, 1)
            num_cols = -(-len(text) // chars_per_col)  # ceil division
            col_width = int(font_size * _LINE_HEIGHT_RATIO_FEW)
            return num_cols * col_width <= usable_w
        else:
            lines = self._wrap_text(text, font, usable_w)
            num_lines = len(lines)
            # Use the same ratio that render() will actually use
            ratio = _LINE_HEIGHT_RATIO_FEW if num_lines <= 2 else _LINE_HEIGHT_RATIO_MANY
            line_height = int(font_size * ratio)
            total_height = line_height * num_lines
            return total_height <= usable_h

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> list[str]:
        """Wrap Korean text with word-level awareness.

        Breaks preferably at spaces, commas, and periods to keep words
        intact.  Falls back to character-level breaking only when a
        single word exceeds *max_width*.
        """
        lines: list[str] = []

        for paragraph in text.split("\n"):
            words = self._split_words(paragraph)
            current_line = ""
            for word in words:
                test_line = current_line + word
                if font.getlength(test_line) > max_width and current_line:
                    lines.append(current_line.rstrip())
                    current_line = word.lstrip()
                elif font.getlength(test_line) > max_width and not current_line:
                    # Single word exceeds max_width — break by character
                    for ch in word:
                        test_ch = current_line + ch
                        if font.getlength(test_ch) > max_width and current_line:
                            lines.append(current_line)
                            current_line = ch
                        else:
                            current_line = test_ch
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line.rstrip())

        return lines if lines else [""]

    @staticmethod
    def _split_words(text: str) -> list[str]:
        """Split Korean/CJK text into word-level chunks for wrapping.

        Splits at spaces (keeping the space attached to the preceding
        chunk) and after punctuation so line breaks appear at natural
        boundaries rather than in the middle of a word.
        """
        import re
        # Split keeping delimiters attached: "안녕하세요, 반갑습니다!"
        # → ["안녕하세요, ", "반갑습니다!"]
        tokens = re.split(r'(?<=[\s,\.!?~…·\-、。！？])', text)
        return [t for t in tokens if t]
