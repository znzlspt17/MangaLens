"""Crop & upscale using Real-ESRGAN."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def remove_furigana(image: np.ndarray) -> np.ndarray:
    """Remove furigana (ruby text) from a bubble crop image before OCR.

    Furigana characters are visually small — typically less than 40% of the
    median glyph height in the same region.  By filtering connected components
    below that threshold we mask them out with the background colour (white)
    without touching okurigana or regular hiragana words, which share the same
    character size as kanji.

    **Dakuten / handakuten protection**: Small dot-like marks (゛ ゜) that are
    part of a character (e.g. が, ぱ) appear as tiny separate connected
    components after binarisation.  To avoid deleting them we check whether a
    small component's bounding box overlaps or nearly touches any main-glyph
    bounding box.  If it does, the component is likely a diacritical mark and
    is preserved.

    Args:
        image: BGR crop of a single bubble (already upscaled).

    Returns:
        Copy of the image with furigana regions replaced by white pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if n_labels <= 1:
        return image  # no components found

    # Exclude background (label 0); collect heights/areas of all glyphs
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(heights) == 0:
        return image

    median_h = float(np.median(heights))
    median_area = float(np.median(areas))
    threshold_h = median_h * 0.4
    threshold_area = median_area * 0.25

    # --- Build list of main-glyph bounding boxes (non-small components) ---
    large_rects: list[tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)
    for i in range(1, n_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if not (h < threshold_h and a < threshold_area):
            lx = int(stats[i, cv2.CC_STAT_LEFT])
            ly = int(stats[i, cv2.CC_STAT_TOP])
            lw = int(stats[i, cv2.CC_STAT_WIDTH])
            lh = int(stats[i, cv2.CC_STAT_HEIGHT])
            large_rects.append((lx, ly, lx + lw, ly + lh))

    # Proximity padding — dakuten marks sit within a few pixels of the
    # parent character body.  15 % of the median glyph height (min 3 px)
    # is enough to catch them without reaching into a furigana sub-column.
    pad = max(median_h * 0.15, 3.0)

    result = image.copy()
    for label_idx in range(1, n_labels):
        h = stats[label_idx, cv2.CC_STAT_HEIGHT]
        a = stats[label_idx, cv2.CC_STAT_AREA]
        # Both height AND area must be small to be considered furigana
        if not (h < threshold_h and a < threshold_area):
            continue

        # Bounding box of this small component
        sx1 = int(stats[label_idx, cv2.CC_STAT_LEFT])
        sy1 = int(stats[label_idx, cv2.CC_STAT_TOP])
        sx2 = sx1 + int(stats[label_idx, cv2.CC_STAT_WIDTH])
        sy2 = sy1 + int(h)

        # If this small component overlaps (with padding) any main glyph,
        # it is likely a dakuten/handakuten mark — preserve it.
        near_large = False
        for (lx1, ly1, lx2, ly2) in large_rects:
            if (sx1 - pad < lx2 and sx2 + pad > lx1 and
                    sy1 - pad < ly2 and sy2 + pad > ly1):
                near_large = True
                break

        if near_large:
            continue

        component_mask = labels == label_idx
        result[component_mask] = (255, 255, 255)

    return result


def _patch_basicsr() -> None:
    """Patch basicsr's broken torchvision import at runtime if needed."""
    try:
        import importlib
        mod = importlib.import_module("basicsr.data.degradations")
    except ImportError:
        # basicsr not installed — nothing to patch
        return
    except ModuleNotFoundError:
        path = Path("basicsr/data/degradations.py")
        # Find the actual installed file
        import basicsr.data as _bd
        path = Path(_bd.__file__).parent / "degradations.py"
        if not path.exists():
            return
        src = path.read_text()
        old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        if old not in src:
            return
        new = (
            "try:\n"
            "    from torchvision.transforms.functional_tensor import rgb_to_grayscale\n"
            "except ImportError:\n"
            "    from torchvision.transforms.functional import rgb_to_grayscale"
        )
        path.write_text(src.replace(old, new))
        logger.info("Patched basicsr torchvision import")


_patch_basicsr()

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _pick_x4_variant() -> tuple[Path, int]:
    """Return (weight_path, num_block) for the configured x4 upscaler.

    Prefers anime_6B (lighter, manga-optimised); falls back to x4plus.
    """
    from server.config import settings

    anime_path = _MODELS_DIR / "RealESRGAN_x4plus_anime_6B.pth"
    x4plus_path = _MODELS_DIR / "RealESRGAN_x4plus.pth"

    if settings.upscaler_variant == "anime_6b" and anime_path.exists():
        return anime_path, 6
    if x4plus_path.exists():
        if settings.upscaler_variant == "anime_6b":
            logger.warning(
                "anime_6B weights not found; falling back to x4plus"
            )
        return x4plus_path, 23
    if anime_path.exists():
        return anime_path, 6
    return x4plus_path, 23  # will trigger "not found" later


class Preprocessor:
    """Crop bubble regions and upscale them for better OCR accuracy.

    Uses Real-ESRGAN with x2 by default; automatically switches to x4
    for small bubbles (width or height < ``min_size``).
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._model_loaded = False
        self._upsampler_x2 = None
        self._upsampler_x4 = None

        x2_path = _MODELS_DIR / "RealESRGAN_x2plus.pth"
        x4_path, x4_num_block = _pick_x4_variant()

        if not x2_path.exists() or not x4_path.exists():
            logger.warning(
                "Real-ESRGAN weights not found (%s, %s). "
                "Falling back to cv2.resize.",
                x2_path,
                x4_path,
            )
            return

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            net_x2 = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2,
            )
            self._upsampler_x2 = RealESRGANer(
                scale=2, model_path=str(x2_path), model=net_x2,
                half=False, device=device,
            )

            net_x4 = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=x4_num_block, num_grow_ch=32, scale=4,
            )
            self._upsampler_x4 = RealESRGANer(
                scale=4, model_path=str(x4_path), model=net_x4,
                half=False, device=device,
            )

            self._model_loaded = True
            logger.info(
                "Real-ESRGAN models loaded (x4=%s, blocks=%d, device=%s)",
                x4_path.name, x4_num_block, device,
            )
        except Exception:
            logger.warning(
                "Failed to initialise Real-ESRGAN. "
                "Falling back to cv2.resize.",
                exc_info=True,
            )

    async def crop_and_upscale(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        min_size: int = 64,
    ) -> np.ndarray:
        """Crop the bbox region from image and upscale it.

        Args:
            image: Full manga page as numpy array (H, W, 3).
            bbox: Bounding box ``(x, y, w, h)``.
            min_size: If width or height of the crop is below this value,
                      use x4 upscale instead of x2.

        Returns:
            Upscaled crop as numpy array.
        """
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        # Clamp bbox to image boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        crop = image[y1:y2, x1:x2]

        crop_h, crop_w = crop.shape[:2]
        scale = 4 if crop_w < min_size or crop_h < min_size else 2

        # Real-ESRGAN uses pre_pad=10 with reflect mode — input dims must
        # exceed the pad size, otherwise fall back to cv2.resize.
        _MIN_ESRGAN_DIM = 12

        if self._model_loaded and crop_h >= _MIN_ESRGAN_DIM and crop_w >= _MIN_ESRGAN_DIM:
            import torch
            upsampler = self._upsampler_x4 if scale == 4 else self._upsampler_x2

            def _upscale_sync():
                with torch.no_grad():
                    out, _ = upsampler.enhance(crop, outscale=scale)
                return out

            output = await asyncio.to_thread(_upscale_sync)
            return remove_furigana(output)

        upscaled = cv2.resize(
            crop, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        return remove_furigana(upscaled)
