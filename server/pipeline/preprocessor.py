"""Crop & upscale using Real-ESRGAN."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
        x4_path = _MODELS_DIR / "RealESRGAN_x4plus.pth"

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
                num_block=23, num_grow_ch=32, scale=4,
            )
            self._upsampler_x4 = RealESRGANer(
                scale=4, model_path=str(x4_path), model=net_x4,
                half=False, device=device,
            )

            self._model_loaded = True
            logger.info("Real-ESRGAN models loaded (device=%s)", device)
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
            with torch.no_grad():
                output, _ = upsampler.enhance(crop, outscale=scale)
            return output

        return cv2.resize(
            crop, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
