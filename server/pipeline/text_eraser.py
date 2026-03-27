"""Text eraser using LaMa inpainting (torch direct inference)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from server.config import settings

logger = logging.getLogger(__name__)


def _pad_to_multiple(
    image: np.ndarray,
    mask: np.ndarray,
    multiple: int = 8,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Pad image and mask so H and W are multiples of *multiple*.

    Returns:
        (padded_image, padded_mask, (pad_h, pad_w))
    """
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, mask, (0, 0)
    padded_image = cv2.copyMakeBorder(
        image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )
    padded_mask = cv2.copyMakeBorder(
        mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )
    return padded_image, padded_mask, (pad_h, pad_w)


class TextEraser:
    """Remove text from bubble regions using LaMa inpainting.

    Uses the comic-text-detector text mask to precisely erase only the
    text inside speech bubbles while preserving the bubble shape and
    surrounding artwork.

    Primary backend: LaMa TorchScript model loaded via ``torch.jit.load``.
    Fallback: OpenCV Telea inpainting when the model is unavailable.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._model: torch.jit.ScriptModule | None = None
        self._model_loaded = False

        model_path = Path(settings.model_cache_dir) / "big-lama.pt"
        try:
            self._model = torch.jit.load(
                str(model_path), map_location=self.device
            )
            self._model.eval()
            self._model_loaded = True
            logger.info("LaMa model loaded from %s", model_path)
        except Exception:
            logger.warning(
                "Failed to load LaMa model from %s — falling back to cv2.inpaint",
                model_path,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def erase(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Erase text from the image using the provided mask.

        Args:
            image: Original manga page as numpy array (H, W, 3) BGR.
            mask: Binary mask where white (255) indicates text to remove.

        Returns:
            Inpainted image with text removed (H, W, 3) BGR.
        """
        if mask.max() == 0:
            return image

        # Dilate the mask to ensure all text ink (including anti-aliased
        # edges and strokes that bleed past the bbox boundary) is covered.
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, dilate_kernel, iterations=2)

        if self._model_loaded:
            return await asyncio.to_thread(self._inpaint_lama, image, mask)
        return await asyncio.to_thread(self._inpaint_cv2, image, mask)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run LaMa TorchScript inference."""
        assert self._model is not None
        orig_h, orig_w = image.shape[:2]

        # BGR → RGB, uint8 → float32 [0, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_f = (mask.astype(np.float32) / 255.0)

        # Pad to multiple of 8
        rgb, mask_f, (pad_h, pad_w) = _pad_to_multiple(rgb, mask_f, 8)

        # numpy → torch tensors
        img_t = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )  # (1, 3, H, W)
        mask_t = (
            torch.from_numpy(mask_f)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )  # (1, 1, H, W)

        with torch.no_grad():
            result = self._model(img_t, mask_t)

        # P1: explicitly release GPU tensors to prevent VRAM accumulation
        out_np = result[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        del img_t, mask_t, result
        if self.device != "cpu":
            torch.cuda.empty_cache()

        # tensor → numpy
        out = (out_np * 255).astype(np.uint8)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:orig_h, :orig_w]

        # RGB → BGR
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _inpaint_cv2(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback inpainting via OpenCV Telea algorithm."""
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
