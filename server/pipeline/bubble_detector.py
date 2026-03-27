"""Bubble / text-block detection using comic-text-detector weights.

Uses the ``blk_det`` YOLOv5s object-detection head stored inside
``comictextdetector.pt`` to predict bounding boxes for text regions.
The backbone + FPN is reconstructed via ``ultralytics`` and the Detect
convolutions are loaded manually so we get standard YOLO box output.

Falls back to an OpenCV contour detector when the model is unavailable.
"""

from __future__ import annotations

import asyncio
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
_MERGE_IOU = 0.25  # post-NMS overlap threshold: merge boxes that overlap this much
_PROXIMITY_GAP = 25  # max edge-to-edge gap (px, model space) for proximity merge
_MIN_AREA = 100
_SEG_THRESHOLD = 0.4  # text_seg sigmoid threshold for bubble interior


# ------------------------------------------------------------------
# UnetHead helper modules (matching comic-text-detector's architecture)
# ------------------------------------------------------------------

class _CTDConv:
    """Placeholder — defined later inside torch-available scope."""


class _CTDBottleneck:
    """Placeholder — defined later inside torch-available scope."""


def _build_unet_classes():
    """Build UnetHead module classes after torch/nn is confirmed available."""
    import torch.nn as nn

    class CTDConv(nn.Module):
        """Conv + BN + LeakyReLU(0.1), matching comic-text-detector Conv."""
        def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None) -> None:
            super().__init__()
            if p is None:
                p = (k - 1) // 2
            self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.LeakyReLU(0.1, inplace=True)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class CTDBottleneck(nn.Module):
        """YOLOv5-style Bottleneck (cv1=1×1, cv2=3×3)."""
        def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5) -> None:
            super().__init__()
            import math
            c_ = int(c2 * e)
            self.cv1 = CTDConv(c1, c_, 1, 1)
            self.cv2 = CTDConv(c_, c2, 3, 1)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            import torch
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    class CTDC3(nn.Module):
        """Cross-stage partial (C3) block matching comic-text-detector C3."""
        def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5) -> None:
            super().__init__()
            c_ = int(c2 * e)
            self.cv1 = CTDConv(c1, c_, 1, 1)
            self.cv2 = CTDConv(c1, c_, 1, 1)
            self.cv3 = CTDConv(2 * c_, c2, 1, 1)
            self.m = nn.Sequential(*(CTDBottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))

        def forward(self, x):
            import torch
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

    class UnetDown(nn.Module):
        """double_conv_c3 with stride=2: AvgPool2d → C3."""
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.down = nn.AvgPool2d(2, stride=2)
            self.conv = CTDC3(in_ch, out_ch, n=1)

        def forward(self, x):
            return self.conv(self.down(x))

    class UnetUp(nn.Module):
        """double_conv_up_c3: C3 → ConvTranspose2d (×2 upsample)."""
        def __init__(self, c_in: int, c_mid: int, c_out: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                CTDC3(c_in, c_mid, n=1),
                nn.ConvTranspose2d(c_mid, c_out, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class UnetHead(nn.Module):
        """Segmentation decoder from comictextdetector.pt 'text_seg' head.

        Consumes backbone features at model output indices [1, 3, 5, 7, 9]
        and returns a (B, 1, H, W) float mask in [0, 1] at input resolution.
        """
        # Model output indices for the five backbone feature levels
        FEAT_IDX = [1, 3, 5, 7, 9]

        def __init__(self) -> None:
            super().__init__()
            self.down_conv1 = UnetDown(512, 512)
            self.upconv0 = UnetUp(512, 512, 256)
            self.upconv2 = UnetUp(768, 512, 256)   # cat(f20=512, u20=256) → 768
            self.upconv3 = UnetUp(512, 512, 256)   # cat(f40=256, u40=256) → 512
            self.upconv4 = UnetUp(384, 256, 128)   # cat(f80=128, u80=256) → 384
            self.upconv5 = UnetUp(192, 128, 64)    # cat(f160=64, u160=128) → 192
            self.upconv6 = nn.Sequential(
                nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, feats: list) -> "torch.Tensor":
            import torch
            f160 = feats[1]   # (B, 64,  H/4,  W/4)
            f80  = feats[3]   # (B, 128, H/8,  W/8)
            f40  = feats[5]   # (B, 256, H/16, W/16)
            f20  = feats[7]   # (B, 512, H/32, W/32)
            f3   = feats[9]   # (B, 512, H/32, W/32) after SPPF
            d10  = self.down_conv1(f3)
            u20  = self.upconv0(d10)
            u40  = self.upconv2(torch.cat([f20, u20], dim=1))
            u80  = self.upconv3(torch.cat([f40, u40], dim=1))
            u160 = self.upconv4(torch.cat([f80, u80],  dim=1))
            u320 = self.upconv5(torch.cat([f160, u160], dim=1))
            return self.upconv6(u320)

    return UnetHead

_UnetHead = None  # populated lazily when torch is available


def _merge_overlapping_boxes(
    boxes: "np.ndarray",
    scores: "np.ndarray",
    cls_ids: "np.ndarray",
    iou_thresh: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Iteratively merge post-NMS boxes whose IoU exceeds *iou_thresh*.

    When two boxes overlap enough (e.g., a tall speech balloon fragmented
    into two detections), they are merged into the union bounding box and
    the higher-confidence class id is kept.  Merging repeats until no more
    pairs exceed the threshold.
    """
    if len(boxes) == 0:
        return boxes, scores, cls_ids

    merged = True
    boxes = boxes.copy()
    scores = scores.copy()
    cls_ids = cls_ids.copy()

    while merged:
        merged = False
        keep = list(range(len(boxes)))
        used = [False] * len(boxes)
        new_boxes, new_scores, new_cls = [], [], []

        for i in range(len(boxes)):
            if used[i]:
                continue
            b1 = boxes[i]
            best_j = -1
            best_iou = iou_thresh
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                b2 = boxes[j]
                # Compute IoU
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter == 0:
                    continue
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                b2 = boxes[best_j]
                merged_box = np.array([
                    min(b1[0], b2[0]),
                    min(b1[1], b2[1]),
                    max(b1[2], b2[2]),
                    max(b1[3], b2[3]),
                ], dtype=boxes.dtype)
                new_boxes.append(merged_box)
                new_scores.append(max(scores[i], scores[best_j]))
                # Keep the class of the larger box
                if (b1[2]-b1[0])*(b1[3]-b1[1]) >= (b2[2]-b2[0])*(b2[3]-b2[1]):
                    new_cls.append(cls_ids[i])
                else:
                    new_cls.append(cls_ids[best_j])
                used[i] = True
                used[best_j] = True
                merged = True
            else:
                new_boxes.append(b1)
                new_scores.append(scores[i])
                new_cls.append(cls_ids[i])
                used[i] = True

        boxes = np.array(new_boxes, dtype=boxes.dtype) if new_boxes else boxes[:0]
        scores = np.array(new_scores, dtype=scores.dtype) if new_scores else scores[:0]
        cls_ids = np.array(new_cls, dtype=cls_ids.dtype) if new_cls else cls_ids[:0]

    return boxes, scores, cls_ids


def _merge_proximity_boxes(
    boxes: "np.ndarray",
    scores: "np.ndarray",
    cls_ids: "np.ndarray",
    gap_thresh: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Merge boxes whose edges are within *gap_thresh* px even if IoU == 0.

    Two boxes are merged when:
    1. Their edge-to-edge gap ≤ *gap_thresh* on BOTH axes, AND
    2. They share significant overlap on at least one axis (≥ 30 % of the
       smaller box's span on that axis).

    This catches vertically-stacked narrow text strips that the YOLO
    detector fragments into separate boxes for the same speech bubble.
    """
    if len(boxes) < 2:
        return boxes, scores, cls_ids

    merged = True
    boxes = boxes.copy()
    scores = scores.copy()
    cls_ids = cls_ids.copy()

    while merged:
        merged = False
        used = [False] * len(boxes)
        new_boxes, new_scores, new_cls = [], [], []

        for i in range(len(boxes)):
            if used[i]:
                continue
            b1 = boxes[i]
            best_j = -1
            best_gap = gap_thresh + 1

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                b2 = boxes[j]

                # Edge-to-edge gap on each axis (0 if overlapping)
                h_gap = max(0.0, max(b2[0] - b1[2], b1[0] - b2[2]))
                v_gap = max(0.0, max(b2[1] - b1[3], b1[1] - b2[3]))

                if h_gap > gap_thresh or v_gap > gap_thresh:
                    continue

                # Axis-overlap check: at least 30 % overlap on one axis
                x_overlap = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                y_overlap = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                w1, w2 = b1[2] - b1[0], b2[2] - b2[0]
                h1, h2 = b1[3] - b1[1], b2[3] - b2[1]
                min_w = min(w1, w2) if min(w1, w2) > 0 else 1
                min_h = min(h1, h2) if min(h1, h2) > 0 else 1

                x_ratio = x_overlap / min_w
                y_ratio = y_overlap / min_h

                if x_ratio < 0.3 and y_ratio < 0.3:
                    continue

                gap = max(h_gap, v_gap)
                if gap < best_gap:
                    best_gap = gap
                    best_j = j

            if best_j >= 0:
                b2 = boxes[best_j]
                merged_box = np.array([
                    min(b1[0], b2[0]),
                    min(b1[1], b2[1]),
                    max(b1[2], b2[2]),
                    max(b1[3], b2[3]),
                ], dtype=boxes.dtype)
                new_boxes.append(merged_box)
                new_scores.append(max(scores[i], scores[best_j]))
                if (b1[2] - b1[0]) * (b1[3] - b1[1]) >= (b2[2] - b2[0]) * (b2[3] - b2[1]):
                    new_cls.append(cls_ids[i])
                else:
                    new_cls.append(cls_ids[best_j])
                used[i] = True
                used[best_j] = True
                merged = True
            else:
                new_boxes.append(b1)
                new_scores.append(scores[i])
                new_cls.append(cls_ids[i])
                used[i] = True

        boxes = np.array(new_boxes, dtype=boxes.dtype) if new_boxes else boxes[:0]
        scores = np.array(new_scores, dtype=scores.dtype) if new_scores else scores[:0]
        cls_ids = np.array(new_cls, dtype=cls_ids.dtype) if new_cls else cls_ids[:0]

    return boxes, scores, cls_ids


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
    """Build YOLOv5s backbone + detect head + UnetHead seg from comictextdetector.pt."""
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

    # ── Build text_seg (UnetHead) ──────────────────────────────────────────
    seg_head = None
    if "text_seg" in state:
        try:
            global _UnetHead
            if _UnetHead is None:
                _UnetHead = _build_unet_classes()
            seg_head = _UnetHead()
            seg_head.load_state_dict(state["text_seg"], strict=True)
            seg_head.to(device).eval()
            logger.info("text_seg (UnetHead) loaded on %s", device)
        except Exception:
            logger.warning("text_seg load failed — bubble masks will be rectangular", exc_info=True)
            seg_head = None
    else:
        logger.warning("text_seg key not found in weights — bubble masks will be rectangular")

    class _DetectorModel(nn.Module):
        FEAT_IDX = [17, 20, 23]

        def __init__(self, bb, convs, anch, strd, nc_, seg):
            super().__init__()
            self.backbone = bb
            self.det_convs = convs
            self.register_buffer("anchors", anch)
            self.register_buffer("strides", strd)
            self.nc = nc_
            self.seg_head = seg  # _UnetHead or None

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
            det = torch.cat(all_preds, dim=1)
            # Run text_seg decoder on backbone features (indices 1,3,5,7,9)
            seg_mask = self.seg_head(feats) if self.seg_head is not None else None
            return det, seg_mask

    model = _DetectorModel(backbone, det_convs, anchors, strides, nc, seg_head)
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

        def _infer_sync():
            tensor = self._preprocess(image).to(self.device)
            with torch.no_grad():
                det, seg_mask = self._model(tensor)  # det: (1, N, 5+nc), seg_mask: (1,1,H,W)|None
            return det, seg_mask

        raw_det, seg_mask = await asyncio.to_thread(_infer_sync)

        # ── Build full-resolution segmentation map ────────────────────────
        seg_full: np.ndarray | None = None
        if seg_mask is not None:
            seg_np = seg_mask[0, 0].cpu().float().numpy()  # (1024, 1024) float in [0,1]
            seg_full = cv2.resize(seg_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        preds = raw_det[0]  # first (only) batch item
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

        # Post-NMS: merge remaining boxes that still overlap significantly.
        # This handles cases where one speech balloon is detected as two
        # overlapping boxes (e.g., a tall vertical bubble split into two).
        boxes, scores, cls_ids = _merge_overlapping_boxes(boxes, scores, cls_ids, _MERGE_IOU)

        # Proximity merge: combine fragments that don't overlap but are
        # within _PROXIMITY_GAP px of each other and share an axis span.
        boxes, scores, cls_ids = _merge_proximity_boxes(boxes, scores, cls_ids, _PROXIMITY_GAP)

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
            text_dir = "vertical" if bh > bw * 1.2 else "horizontal"
            bubble_type = "text" if cid == 0 else ("effect" if cid == 2 else "speech")

            # ── Build per-bubble mask ────────────────────────────────────
            if seg_full is not None:
                # Use text_seg segmentation for accurate bubble interior mask
                roi = seg_full[by1:by2, bx1:bx2]
                mask_roi = (roi >= _SEG_THRESHOLD).astype(np.uint8) * 255
                # If seg is empty inside bbox fall back to solid fill
                if mask_roi.any():
                    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    full_mask[by1:by2, bx1:bx2] = mask_roi
                else:
                    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    full_mask[by1:by2, bx1:bx2] = 255
            else:
                full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                full_mask[by1:by2, bx1:bx2] = 255

            bubbles.append(
                BubbleInfo(
                    id=i + 1,
                    bbox=(bx1, by1, bw, bh),
                    mask=full_mask,
                    text_direction=text_dir,
                    bubble_type=bubble_type,
                )
            )
        logger.info("Detected %d text regions (model, seg=%s)", len(bubbles), seg_full is not None)
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
