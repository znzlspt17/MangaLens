"""Magi v2 based bubble/text detector.

Wraps ``ragavsachdeva/magiv2`` from HuggingFace as a drop-in
alternative to :class:`BubbleDetector`.  The model performs detection,
reading-order sorting, and essential-text classification in one pass.

Requires ``transformers>=5.0`` and ``accelerate``.

Fixes vs original implementation:
  1. BGR→RGB conversion before inference (OpenCV reads BGR; Magi expects RGB).
  2. Blocking inference is offloaded to a thread via ``asyncio.to_thread``
     so the ASGI event loop is never stalled.
  3. ``torch.float16`` is used on non-CPU devices to match memory usage of
     other pipeline components.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from server.pipeline.bubble_detector import BubbleInfo

logger = logging.getLogger(__name__)


def _xyxy_to_xywh(bbox: list[float]) -> tuple[int, int, int, int]:
    """Convert ``[x1, y1, x2, y2]`` to ``(x, y, w, h)``."""
    x1, y1, x2, y2 = bbox
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


class MagiDetector:
    """Detect speech bubbles using Magi v2 (HuggingFace).

    Falls back to :class:`BubbleDetector` (YOLOv5) when:
      - ``transformers`` / ``accelerate`` are not installed
      - Available VRAM is below ``magi_vram_threshold_mb``
      - Model download / load fails
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._model: Any | None = None
        self._model_loaded = False

        try:
            from server.config import settings
            from server.gpu import get_vram_mb
            vram = get_vram_mb()
            if vram > 0 and vram < settings.magi_vram_threshold_mb:
                logger.warning(
                    "VRAM %d MB < threshold %d MB; skipping Magi v2",
                    vram,
                    settings.magi_vram_threshold_mb,
                )
                return
        except Exception:
            pass

        try:
            import torch
            from transformers import AutoModel

            torch_dtype = torch.float16 if device != "cpu" else torch.float32
            from server.config import settings

            model = AutoModel.from_pretrained(
                "ragavsachdeva/magiv2",
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                cache_dir=settings.model_cache_dir,
            )
            if device != "cpu":
                model = model.to(device)
            model.eval()
            self._model = model
            self._model_loaded = True
            logger.info("Magi v2 detector loaded (device=%s)", device)
        except Exception:
            logger.warning(
                "Failed to load Magi v2; will fall back to YOLOv5",
                exc_info=True,
            )

    async def detect(self, image: np.ndarray) -> list[BubbleInfo]:
        """Detect text regions and return :class:`BubbleInfo` list.

        The returned list is already sorted in Magi's panel-aware
        reading order, so the caller should **not** re-sort with
        ``sort_bubbles_rtl``.
        """
        if not self._model_loaded or self._model is None:
            logger.info("Magi v2 not loaded; returning empty list")
            return []

        # OpenCV provides BGR; Magi v2 expects RGB — channel order matters.
        rgb_image = image[:, :, ::-1].copy()

        import torch

        def _infer(img: np.ndarray):
            with torch.no_grad():
                return self._model.predict_detections_and_associations([img])

        results = await asyncio.to_thread(_infer, rgb_image)

        if not results:
            return []

        page = results[0]
        text_bboxes: list[list[float]] = page.get("texts", [])
        is_essential: list[bool] = page.get("is_essential_text", [])

        bubbles: list[BubbleInfo] = []
        for idx, bbox_xyxy in enumerate(text_bboxes):
            x, y, w, h = _xyxy_to_xywh(bbox_xyxy)
            if w <= 0 or h <= 0:
                continue

            # Classify bubble type
            essential = is_essential[idx] if idx < len(is_essential) else True
            btype = "speech" if essential else "effect"

            # Text direction heuristic: tall → vertical, wide → horizontal
            direction = "vertical" if h > w * 1.2 else "horizontal"

            bubbles.append(
                BubbleInfo(
                    id=idx + 1,
                    bbox=(x, y, w, h),
                    mask=None,
                    text_direction=direction,
                    bubble_type=btype,
                    reading_order=idx + 1,  # Magi's own order
                )
            )

        logger.info("Magi v2 detected %d text regions", len(bubbles))
        return bubbles
