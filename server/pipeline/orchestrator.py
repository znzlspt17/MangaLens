"""Pipeline orchestrator — coordinates all 7 translation stages."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from server.pipeline.bubble_detector import BubbleDetector, BubbleInfo
from server.pipeline.compositor import Compositor, RenderedBubble
from server.pipeline.ocr_engine import OCREngine, OCRResult
from server.pipeline.preprocessor import Preprocessor
from server.pipeline.text_eraser import TextEraser
from server.pipeline.text_renderer import TextRenderer
from server.pipeline.translator import Translator
from server.config import settings as _server_settings
from server.utils.reading_order import sort_bubbles_rtl

logger = logging.getLogger(__name__)

# Minimum OCR confidence to proceed with translation
_MIN_OCR_CONFIDENCE = 0.3

# ---------------------------------------------------------------------------
# Global model cache — avoids reloading 100s of MB per request (§13 GPU memory)
# ---------------------------------------------------------------------------
_model_cache: dict[str, object] = {}
_model_cache_lock: asyncio.Lock = asyncio.Lock()  # P0: prevents double-loading on concurrent first requests


async def _get_cached(cls: type, key: str, **kwargs) -> object:
    """Return a cached instance of *cls*, creating on first call.

    Uses double-checked locking so that concurrent pipeline starts do not
    race to instantiate the same model (which would waste VRAM / cause OOM).
    """
    # Fast path — model already loaded
    if key in _model_cache:
        return _model_cache[key]
    # Slow path — acquire lock and re-check (double-checked locking)
    async with _model_cache_lock:
        if key not in _model_cache:
            _model_cache[key] = await asyncio.to_thread(cls, **kwargs)
            logger.info("Model cached: %s", key)
    return _model_cache[key]


def clear_model_cache() -> None:
    """Release all cached models (useful for testing / shutdown)."""
    _model_cache.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class UserTranslationSettings:
    """Per-request translation settings supplied by the user."""

    target_lang: str = "KO"
    source_lang: str = "JA"


@dataclass
class PipelineResult:
    """Final output of the translation pipeline."""

    translated_image_path: Path
    translation_log_path: Path


@dataclass
class _BubbleContext:
    """Internal context accumulated per bubble across stages."""

    info: BubbleInfo
    crop: np.ndarray | None = field(default=None, repr=False)
    ocr: OCRResult | None = None
    translated_text: str = ""
    translation_engine: str = ""
    skipped: bool = False
    skip_reason: str = ""
    font_size_used: int = 0


async def run_pipeline(
    image_path: Path,
    settings: UserTranslationSettings,
    output_dir: Path | None = None,
) -> PipelineResult:
    """Execute the full 7-stage manga translation pipeline.

    Stages:
        1. Bubble detection
        2. Crop & upscale
        3. OCR
        4. Translation  }  ← parallel via asyncio.gather
        5. Text erasure  }
        6. Text rendering
        7. Compositing

    Args:
        image_path: Path to the source manga page image.
        settings: User-provided translation settings.
        output_dir: Directory for output files. Defaults to
                    ``./output/<stem>/``.

    Returns:
        PipelineResult with paths to the translated image and log.
    """
    import cv2  # local import — only needed at runtime

    t_start = time.monotonic()
    stem = image_path.stem

    if output_dir is None:
        output_dir = Path("./output") / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    device = "cpu"  # will be injected by caller / GPU agent
    try:
        from server.gpu import get_device
        device = get_device()
    except Exception:
        logger.warning("[pipeline] GPU device detection failed, falling back to CPU")

    # Load image
    original = cv2.imread(str(image_path))
    if original is None:
        logger.error("[pipeline] cv2.imread failed: %s (exists=%s, size=%s)",
                     image_path, image_path.exists(),
                     image_path.stat().st_size if image_path.exists() else 'N/A')
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_h, img_w = original.shape[:2]
    logger.info("[pipeline] Start: %s (%dx%d, device=%s)", image_path.name, img_w, img_h, device)

    # ── Stage 1: Bubble Detection ──────────────────────────────────────
    logger.info("[1/7] Detecting bubbles")
    _using_magi = False
    if _server_settings.use_magi_detector:
        try:
            from server.pipeline.magi_detector import MagiDetector
            magi = await _get_cached(MagiDetector, "magi_detector", device=device)
            if magi._model_loaded:
                bubbles_raw = await magi.detect(original)
                _using_magi = True
            else:
                logger.info("Magi v2 not available; falling back to YOLOv5")
        except Exception:
            logger.warning("Magi v2 import failed; falling back to YOLOv5", exc_info=True)

    if not _using_magi:
        detector = await _get_cached(BubbleDetector, "bubble_detector", device=device)
        bubbles_raw = await detector.detect(original)

    # Sort in reading order — skip if Magi already sorted
    if _using_magi:
        bubbles = bubbles_raw  # Magi sorts by panels internally
    else:
        bubbles = sort_bubbles_rtl(bubbles_raw)
    logger.info("  Found %d bubbles", len(bubbles))

    # Filter out effect-type bubbles (sound effects are NOT translated)
    translatable = [b for b in bubbles if b.bubble_type != "effect"]
    effects_skipped = len(bubbles) - len(translatable)
    if effects_skipped:
        logger.info("  Skipped %d effect bubbles", effects_skipped)

    # ── Stage 2: Crop & Upscale ────────────────────────────────────────
    logger.info("[2/7] Cropping & upscaling")
    preprocessor = await _get_cached(Preprocessor, "preprocessor", device=device)
    contexts: list[_BubbleContext] = []

    for bubble in translatable:
        crop = await preprocessor.crop_and_upscale(original, bubble.bbox)
        crop_path = crops_dir / f"{stem}_bubble_{bubble.id:03d}.png"
        cv2.imwrite(str(crop_path), crop)
        ctx = _BubbleContext(info=bubble, crop=crop)
        contexts.append(ctx)

    # ── Stage 3: OCR ───────────────────────────────────────────────────
    logger.info("[3/7] Running OCR")
    ocr_engine = await _get_cached(OCREngine, "ocr_engine", device=device)

    for ctx in contexts:
        assert ctx.crop is not None
        result = await ocr_engine.recognize(ctx.crop)
        ctx.ocr = result

        if result.confidence < _MIN_OCR_CONFIDENCE:
            ctx.skipped = True
            ctx.skip_reason = "low_ocr_confidence"
            logger.warning(
                "  Bubble %d: OCR confidence %.2f < %.2f — keeping original",
                ctx.info.id,
                result.confidence,
                _MIN_OCR_CONFIDENCE,
            )

    # Collect texts for batch translation (only non-skipped bubbles)
    texts_to_translate = [
        ctx.ocr.text for ctx in contexts if not ctx.skipped and ctx.ocr
    ]

    # ── Stage 4 & 5: Translation + Text Erasure (parallel) ─────────────
    logger.info("[4-5/7] Translation & text erasure (parallel)")
    translator = Translator(
        target_lang=settings.target_lang,
        source_lang=settings.source_lang,
        device=device,
    )
    text_eraser = await _get_cached(TextEraser, "text_eraser", device=device)

    async def _do_translation() -> list[str]:
        if not texts_to_translate:
            logger.info("[pipeline] No texts to translate (all skipped)")
            return []
        logger.info("[pipeline] Translating %d text(s): %s → %s",
                    len(texts_to_translate), settings.source_lang, settings.target_lang)
        try:
            results = await translator.translate_batch(
                texts_to_translate,
                context=texts_to_translate,
            )
            logger.info("[pipeline] Translation success: %d text(s) translated", len(results))
            return results
        except Exception as exc:
            logger.exception(
                "[pipeline] Translation FAILED (%s: %s) — falling back to original texts",
                type(exc).__name__, exc,
            )
            # Fallback: return original texts (§11 error recovery)
            return list(texts_to_translate)

    async def _do_erasure() -> np.ndarray:
        # Build a combined mask for all translatable bubbles
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        mask_sources = {"provided": 0, "fallback_bbox": 0}
        for ctx in contexts:
            if ctx.info.mask is not None:
                mask = np.maximum(mask, ctx.info.mask)
                mask_sources["provided"] += 1
            else:
                bx, by, bw, bh = ctx.info.bbox
                mask[by:by + bh, bx:bx + bw] = 255
                mask_sources["fallback_bbox"] += 1
        logger.info("[pipeline] Erasure mask: %d provided, %d bbox fallback",
                    mask_sources["provided"], mask_sources["fallback_bbox"])
        return await text_eraser.erase(original, mask)

    translated_texts, erased_image = await asyncio.gather(
        _do_translation(),
        _do_erasure(),
    )

    # Map translated texts back to contexts
    tr_idx = 0
    for ctx in contexts:
        if ctx.skipped:
            continue
        if tr_idx < len(translated_texts):
            ctx.translated_text = translated_texts[tr_idx]
            ctx.translation_engine = "hunyuan"  # local Hunyuan-MT-7B
            tr_idx += 1
        else:
            # Translation failed for this bubble
            ctx.translated_text = ctx.ocr.text if ctx.ocr else ""
            ctx.translation_engine = "translation_failed"

    # ── Stage 6: Text Rendering ────────────────────────────────────────
    logger.info("[6/7] Rendering translated text")
    renderer = await _get_cached(TextRenderer, "text_renderer", font_dir=_server_settings.font_dir)
    rendered_bubbles: list[RenderedBubble] = []

    for ctx in contexts:
        if ctx.skipped:
            continue
        # Use only the translated text. If empty (translation discarded by postprocess),
        # render blank so the erased bubble stays clean rather than showing original Japanese.
        text = ctx.translated_text
        rendered, font_size, adj_bbox = await renderer.render(
            erased_image,
            text,
            ctx.info.bbox,
            ctx.info.text_direction,
        )
        ctx.font_size_used = font_size
        rendered_bubbles.append(
            RenderedBubble(bbox=adj_bbox, image=rendered)
        )

    # ── Stage 7: Compositing ───────────────────────────────────────────
    logger.info("[7/7] Compositing final image")
    final_image = await Compositor.composite(erased_image, rendered_bubbles)

    # Save output
    output_image_path = output_dir / f"{stem}_translated.png"
    cv2.imwrite(str(output_image_path), final_image)

    # ── Translation log (PLAN.md §4) ───────────────────────────────────
    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    confidences = [
        ctx.ocr.confidence for ctx in contexts if ctx.ocr is not None
    ]
    avg_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )

    log_data = {
        "version": "1.0",
        "source_file": image_path.name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "reading_direction": "rtl",
        "image_size": {"width": img_w, "height": img_h},
        "bubbles": [],
        "summary": {
            "total_bubbles": len(translatable),
            "effects_skipped": effects_skipped,
            "avg_confidence": round(avg_confidence, 3),
            "processing_time_ms": elapsed_ms,
        },
    }

    for ctx in contexts:
        x, y, w, h = ctx.info.bbox
        bubble_entry = {
            "id": ctx.info.id,
            "reading_order": ctx.info.reading_order,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "text_direction": ctx.info.text_direction,
            "bubble_type": ctx.info.bubble_type,
            "original_text": ctx.ocr.text if ctx.ocr else "",
            "ocr_confidence": ctx.ocr.confidence if ctx.ocr else 0.0,
            "translated_text": ctx.translated_text,
            "translation_engine": ctx.translation_engine,
            "font_size_used": ctx.font_size_used,
            "crop_file": f"crops/{stem}_bubble_{ctx.info.id:03d}.png",
        }
        if ctx.skipped:
            bubble_entry["skipped"] = True
            bubble_entry["skip_reason"] = ctx.skip_reason
        log_data["bubbles"].append(bubble_entry)

    log_path = output_dir / "translation_log.json"
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "[pipeline] Complete: %s → %s (%d ms, %d bubbles, %d translated)",
        image_path.name,
        output_image_path.name,
        elapsed_ms,
        len(translatable),
        len([c for c in contexts if not c.skipped]),
    )

    # Free intermediate GPU tensors
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("[pipeline] CUDA cache clear failed: %s", exc)

    return PipelineResult(
        translated_image_path=output_image_path,
        translation_log_path=log_path,
    )
