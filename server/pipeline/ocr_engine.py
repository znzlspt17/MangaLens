"""OCR engine using manga-ocr (TrOCR-based)."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Pattern matching CJK Unified Ideographs + Hiragana + Katakana
_CJK_RE = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]"
)


@dataclass
class OCRResult:
    """Result of OCR recognition on a single bubble crop."""

    text: str
    confidence: float


def _estimate_confidence(text: str) -> float:
    """Heuristic confidence based on recognised text length and content."""
    if not text:
        return 0.0
    length = len(text)
    if length <= 2:
        base = 0.5
    else:
        base = 0.9
    # Boost if most characters are CJK (expected for manga)
    cjk_ratio = len(_CJK_RE.findall(text)) / length
    if cjk_ratio >= 0.8:
        return min(base + 0.05, 1.0)
    return base


class OCREngine:
    """Japanese manga OCR engine.

    Uses manga-ocr (TrOCR-based) for recognising vertical/horizontal
    Japanese text in manga speech bubbles.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._model_loaded = False
        self._ocr = None
        try:
            from manga_ocr import MangaOcr
            from huggingface_hub import snapshot_download
            from server.config import settings

            _MANGA_OCR_REPO = "kha-white/manga-ocr-base"
            local_path = snapshot_download(
                _MANGA_OCR_REPO,
                cache_dir=settings.model_cache_dir,
            )
            self._ocr = MangaOcr(pretrained_model_name_or_path=local_path)
            self._model_loaded = True
            log.info("manga-ocr model loaded successfully")
        except Exception:
            log.exception("Failed to load manga-ocr model; OCR will be unavailable")

    async def recognize(self, crop_image: np.ndarray) -> OCRResult:
        """Recognise text in a cropped (and upscaled) bubble image.

        Args:
            crop_image: Upscaled crop of a single bubble (H, W, 3).

        Returns:
            OCRResult with recognised text and confidence score.
        """
        if not self._model_loaded:
            return OCRResult(text="", confidence=0.0)

        pil_img = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
        text: str = await asyncio.to_thread(self._ocr, pil_img)
        text = text.strip()
        confidence = _estimate_confidence(text)
        return OCRResult(text=text, confidence=confidence)
