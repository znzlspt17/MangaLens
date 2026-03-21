"""Bubble / text-block detection using comic-text-detector weights.

Uses the ``blk_det`` YOLOv5s object-detection head stored inside
``comictextdetector.pt`` to predict bounding boxes for text regions.
The backbone + FPN is reconstructed via ``ultralytics`` and the Detect
convolutions are loaded manually so we get standard YOLO box output.

Falls back to an OpenCV contour detector when the model is unavailable.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from server.config import settings

logger = logging.getLogger(__name__)

_MODEL_INPUT_SIZE = 1024
_CONF_THRESH = 0.35
_NMS_IOU = 0.45
_MIN_AREA = 100


@dataclass
class BubbleInfo:
    """Detected speech bubble metadata."""

    id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    mask: np.ndarray | None = field(default=None, repr=False)
    text_direction: str = "vertical"
    bubble_type: str = "speech"
    reading_order: int = 0


# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------

def _build_detector(weight_path: str, device: str):
    """Build a YOLOv5s backbone + detect head from blk_det weights."""
    try:
        import torch
        import torch.nn as nn
        from ultralytics.nn.tasks import parse_model
        from torchvision.ops import nms  # noqa: F401 – availability check
    except ImportError:
        logger.warning("ultralytics / torchvision not installed — ML detection unavailable")
        return None, False

    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=False)
    except Exception:
        logger.exception("Failed to load %s", weight_path)
        return None, False

    if not isinstance(state, dict) or "blk_det" not in state:
        return None, False

    blk = state["blk_det"]
    cfg = blk["cfg"]

    yaml_cfg = {
        "nc": cfg["nc"],
        "depth_multiple": cfg["depth_multiple"],
        "width_multiple": cfg["width_multiple"],
        "backbone": cfg["backbone"],
        "head": cfg["head"][:-1],  # exclude Detect layer
    }

    class _Backbone(nn.Module):
        def __init__(self, ycfg, ch=3):
            super().__init__()
            self.model, self.save = parse_model(deepcopy(ycfg), ch=ch, verbose=False)
            self.save = set(range(len(self.model)))

        def forward(self, x):
            y: list = []
            for m in self.model:
                if m.f != -1:
                    x = (
                        y[m.f]
                        if isinstance(m.f, int)
                        else [x if j == -1 else y[j] for j in m.f]
                    )
                x = m(x)
                y.append(x)
            return y

    backbone = _Backbone(yaml_cfg, ch=3)
    bb_weights = {k: v.float() for k, v in blk["weights"].items() if not k.startswith("model.24")}
    backbone.load_state_dict(bb_weights, strict=False)
    backbone.to(device).eval()

    # 3 detect convolutions (P3, P4, P5)
    nc = cfg["nc"]
    na = 3  # anchors per scale
    no = na * (5 + nc)  # outputs per grid cell
    feat_channels = [128, 256, 512]
    det_convs = nn.ModuleList([nn.Conv2d(ch, no, 1) for ch in feat_channels])
    for i in range(3):
        det_convs[i].weight.data = blk["weights"][f"model.24.m.{i}.weight"].float()
        det_convs[i].bias.data = blk["weights"][f"model.24.m.{i}.bias"].float()
    det_convs.to(device).eval()

    anchors = blk["weights"]["model.24.anchors"].float()
    strides = torch.tensor([8.0, 16.0, 32.0])

    class _DetectorModel(nn.Module):
        FEAT_IDX = [17, 20, 23]

        def __init__(self, bb, convs, anch, strd, nc_):
            super().__init__()
            self.backbone = bb
            self.det_convs = convs
            self.register_buffer("anchors", anch)
            self.register_buffer("strides", strd)
            self.nc = nc_

        def forward(self, x):
            feats = self.backbone(x)
            all_preds: list = []
            for i, (fi, conv, stride) in enumerate(
                zip(self.FEAT_IDX, self.det_convs, self.strides)
            ):
                pred = conv(feats[fi])
                bs, _, ny, nx = pred.shape
                na_ = 3
                pred = pred.view(bs, na_, 5 + self.nc, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                pred_sig = pred.sigmoid()
                dev = pred.device
                yv, xv = torch.meshgrid(
                    torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij"
                )
                grid = torch.stack([xv, yv], dim=-1).float().view(1, 1, ny, nx, 2)
                anchor = self.anchors[i].to(dev).view(1, 3, 1, 1, 2)
                xy = (pred_sig[..., :2] * 2 - 0.5 + grid) * stride
                wh = (pred_sig[..., 2:4] * 2) ** 2 * anchor * stride
                rest = pred_sig[..., 4:]
                boxes = torch.cat([xy, wh, rest], dim=-1).view(bs, -1, 5 + self.nc)
                all_preds.append(boxes)
            return torch.cat(all_preds, dim=1)

    model = _DetectorModel(backbone, det_convs, anchors, strides, nc)
    model.to(device).eval()
    logger.info("comic-text-detector (blk_det YOLOv5) loaded on %s", device)
    return model, True


# ------------------------------------------------------------------
# BubbleDetector
# ------------------------------------------------------------------

class BubbleDetector:
    """Detect speech bubbles / text blocks in manga page images."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._model_loaded = False
        self._model = None

        weight_path = Path(settings.model_cache_dir) / "comictextdetector.pt"
        if not weight_path.exists():
            logger.warning("comic-text-detector weights not found at %s", weight_path)
            return

        model, ok = _build_detector(str(weight_path), device)
        if ok and model is not None:
            self._model = model
            self._model_loaded = True

    async def detect(self, image: np.ndarray) -> list[BubbleInfo]:
        if not self._model_loaded or self._model is None:
            logger.info("Using OpenCV fallback bubble detector")
            return self._detect_cv2(image)

        import torch
        from torchvision.ops import nms

        orig_h, orig_w = image.shape[:2]

        tensor = self._preprocess(image).to(self.device)
        with torch.no_grad():
            preds = self._model(tensor)  # (1, N, 5+nc)

        preds = preds[0]  # first (only) batch item
        obj_conf = preds[:, 4]
        mask = obj_conf > _CONF_THRESH
        preds = preds[mask]
        if preds.shape[0] == 0:
            logger.info("No text regions detected (conf > %.2f)", _CONF_THRESH)
            return []

        cls_conf, cls_id = preds[:, 5:].max(dim=1)
        scores = preds[:, 4] * cls_conf
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        keep = nms(boxes, scores, _NMS_IOU)
        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        cls_ids = cls_id[keep].cpu().numpy()

        sx, sy = orig_w / _MODEL_INPUT_SIZE, orig_h / _MODEL_INPUT_SIZE
        bubbles: list[BubbleInfo] = []
        for i, (box, score, cid) in enumerate(zip(boxes, scores, cls_ids)):
            bx1 = int(max(0, box[0] * sx))
            by1 = int(max(0, box[1] * sy))
            bx2 = int(min(orig_w, box[2] * sx))
            by2 = int(min(orig_h, box[3] * sy))
            bw, bh = bx2 - bx1, by2 - by1
            if bw * bh < _MIN_AREA:
                continue
            mask_img = np.zeros((orig_h, orig_w), dtype=np.uint8)
            mask_img[by1:by2, bx1:bx2] = 255
            text_dir = "vertical" if bh > bw * 1.2 else "horizontal"
            bubble_type = "text" if cid == 0 else "speech"
            bubbles.append(
                BubbleInfo(
                    id=i + 1,
                    bbox=(bx1, by1, bw, bh),
                    mask=mask_img,
                    text_direction=text_dir,
                    bubble_type=bubble_type,
                )
            )
        logger.info("Detected %d text regions (model)", len(bubbles))
        return bubbles

    @staticmethod
    def _preprocess(image: np.ndarray) -> "torch.Tensor":
        import torch

        resized = cv2.resize(image, (_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    @staticmethod
    def _detect_cv2(image: np.ndarray) -> list[BubbleInfo]:
        orig_h, orig_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, grad_kernel)
        _, grad_binary = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        grad_closed = cv2.dilate(grad_binary, close_kernel, iterations=2)
        filled = grad_closed.copy()
        mask_ff = np.zeros((orig_h + 2, orig_w + 2), dtype=np.uint8)
        for px in range(0, orig_w, 5):
            for py in [0, orig_h - 1]:
                if filled[py, px] == 0:
                    cv2.floodFill(filled, mask_ff, (px, py), 128)
        for py in range(0, orig_h, 5):
            for px in [0, orig_w - 1]:
                if filled[py, px] == 0:
                    cv2.floodFill(filled, mask_ff, (px, py), 128)
        enclosed = (filled == 0).astype(np.uint8) * 255
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        enclosed = cv2.morphologyEx(enclosed, cv2.MORPH_CLOSE, merge_kernel, iterations=2)
        contours, _ = cv2.findContours(enclosed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = orig_h * orig_w
        bubbles: list[BubbleInfo] = []
        bubble_id = 1
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area_ratio = (w * h) / image_area
            if area_ratio < 0.003 or area_ratio > 0.3:
                continue
            roi = gray[y : y + h, x : x + w]
            if np.mean(roi) < 150 or np.sum(roi < 100) / roi.size < 0.01:
                continue
            bubble_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.drawContours(bubble_mask, [cnt], -1, 255, cv2.FILLED)
            dark_pixels = (gray < 128).astype(np.uint8) * 255
            text_mask = cv2.bitwise_and(dark_pixels, bubble_mask)
            text_dir = "vertical" if h > w * 1.2 else "horizontal"
            bubbles.append(
                BubbleInfo(
                    id=bubble_id,
                    bbox=(x, y, w, h),
                    mask=text_mask,
                    text_direction=text_dir,
                    bubble_type="speech",
                )
            )
            bubble_id += 1
        logger.info("OpenCV fallback detected %d bubbles", len(bubbles))
        return bubbles
