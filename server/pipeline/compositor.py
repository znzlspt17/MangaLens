"""Compositor — alpha-blending rendered bubbles onto the original image."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RenderedBubble:
    """A single rendered bubble ready for compositing."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    image: np.ndarray = field(repr=False)  # RGBA (H, W, 4)


class Compositor:
    """Composite rendered bubbles back onto the original manga page.

    Uses alpha blending with feathering for smooth edges.
    Output resolution always matches the original image (upscaling is
    for OCR preprocessing only and never affects the final output).
    """

    @staticmethod
    async def composite(
        original: np.ndarray,
        rendered_bubbles: list[RenderedBubble],
    ) -> np.ndarray:
        """Overlay rendered bubbles onto the original image.

        Args:
            original: Original manga page as numpy array (H, W, 3).
            rendered_bubbles: List of RenderedBubble with RGBA images
                              and their corresponding bounding boxes.

        Returns:
            Composited image at original resolution (H, W, 3).
        """
        if not rendered_bubbles:
            return original.copy()

        result = original.copy()
        img_h, img_w = result.shape[:2]

        for bubble in rendered_bubbles:
            x, y, w, h = bubble.bbox
            overlay = bubble.image

            # Compute overlap between bbox and image bounds
            src_x0 = max(0, -x)
            src_y0 = max(0, -y)
            dst_x0 = max(0, x)
            dst_y0 = max(0, y)
            dst_x1 = min(img_w, x + overlay.shape[1])  # use actual overlay width
            dst_y1 = min(img_h, y + overlay.shape[0])  # use actual overlay height (may exceed bbox h)

            if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
                continue

            src_x1 = src_x0 + (dst_x1 - dst_x0)
            src_y1 = src_y0 + (dst_y1 - dst_y0)

            # Clamp source region to overlay dimensions
            ov_h, ov_w = overlay.shape[:2]
            src_x1 = min(src_x1, ov_w)
            src_y1 = min(src_y1, ov_h)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
                continue

            overlay_region = overlay[src_y0:src_y1, src_x0:src_x1]
            alpha = overlay_region[:, :, 3].astype(np.float32) / 255.0

            # Feathering: Gaussian blur on alpha for smooth edges
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1, sigmaY=1)

            alpha_3ch = alpha[:, :, np.newaxis]
            overlay_rgb = overlay_region[:, :, :3].astype(np.float32)
            original_region = result[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)

            blended = alpha_3ch * overlay_rgb + (1.0 - alpha_3ch) * original_region
            result[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(blended, 0, 255).astype(np.uint8)

        return result
