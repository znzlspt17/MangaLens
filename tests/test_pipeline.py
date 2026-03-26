"""Phase 2 — Pipeline module unit tests.

Tests each pipeline module in isolation using mocks for all ML models
and external API calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# TestBubbleDetector
# ---------------------------------------------------------------------------


class TestBubbleDetector:
    """BubbleDetector: model-not-loaded fallback, text_direction detection."""

    async def test_returns_empty_when_model_not_loaded(self):
        """Model weights absent → detect() returns []."""
        with patch("server.pipeline.bubble_detector.Path.exists", return_value=False):
            from server.pipeline.bubble_detector import BubbleDetector

            det = BubbleDetector(device="cpu")
        assert det._model_loaded is False

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = await det.detect(img)
        assert result == []

    async def test_detect_cv2_fallback_returns_bubbles(self):
        """OpenCV fallback detects contours from synthetic image."""
        from server.pipeline.bubble_detector import BubbleDetector

        # Create an image with a white rectangle on black → contour-detectable
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 120), (255, 255, 255), -1)
        bubbles = BubbleDetector._detect_cv2(img)
        assert isinstance(bubbles, list)
        # Should find at least the drawn rectangle
        for b in bubbles:
            assert hasattr(b, "bbox")
            assert hasattr(b, "text_direction")

    def test_preprocess_output_shape(self):
        """_preprocess returns (1, 3, 1024, 1024) float tensor."""
        from server.pipeline.bubble_detector import BubbleDetector

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = BubbleDetector._preprocess(img)
        assert tensor.shape == (1, 3, 1024, 1024)
        assert tensor.dtype.is_floating_point
        assert tensor.max() <= 1.0

    async def test_detect_cv2_blank_image_empty(self):
        """All-black image → no contours detected."""
        from server.pipeline.bubble_detector import BubbleDetector

        img = np.zeros((200, 300, 3), dtype=np.uint8)
        bubbles = BubbleDetector._detect_cv2(img)
        assert bubbles == []


# ---------------------------------------------------------------------------
# TestPreprocessor
# ---------------------------------------------------------------------------


class TestPreprocessor:
    """Preprocessor: cv2.resize fallback, bbox clamping, scale selection."""

    @pytest.fixture()
    def preprocessor(self):
        """Build a Preprocessor with models disabled."""
        with patch(
            "server.pipeline.preprocessor.Path.exists", return_value=False
        ):
            from server.pipeline.preprocessor import Preprocessor

            return Preprocessor(device="cpu")

    async def test_cv2_resize_fallback_x2(self, preprocessor):
        """Model absent → cv2.resize with scale=2 for large crops."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        bbox = (10, 10, 100, 100)  # w=100,h=100 both ≥ 64 → scale 2
        result = await preprocessor.crop_and_upscale(img, bbox)
        assert result.shape[:2] == (200, 200)  # 100*2, 100*2

    async def test_cv2_resize_fallback_x4(self, preprocessor):
        """Crop smaller than min_size → x4 upscale."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        bbox = (10, 10, 30, 30)  # w=30 < 64 → scale 4
        result = await preprocessor.crop_and_upscale(img, bbox)
        assert result.shape[:2] == (120, 120)  # 30*4, 30*4

    async def test_bbox_clamped_to_image(self, preprocessor):
        """Negative / out-of-bound bbox values are clamped."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (-10, -10, 50, 50)  # x1=0, y1=0, x2=40, y2=40
        result = await preprocessor.crop_and_upscale(img, bbox)
        # crop is 40x40 → < 64 → x4 → 160x160
        assert result.shape[0] > 0 and result.shape[1] > 0

    async def test_bbox_exceeds_image(self, preprocessor):
        """bbox extends past image boundaries → clamped to edge."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        bbox = (30, 30, 100, 100)  # x2=130→50, y2=130→50 → crop 20x20
        result = await preprocessor.crop_and_upscale(img, bbox)
        assert result.shape[0] > 0 and result.shape[1] > 0

    async def test_scale_selection_boundary(self, preprocessor):
        """Exactly min_size → uses x2."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = (0, 0, 64, 64)  # w=h=64 exactly → x2
        result = await preprocessor.crop_and_upscale(img, bbox)
        assert result.shape[:2] == (128, 128)

    def test_pick_x4_variant_anime_6b(self, tmp_path):
        """_pick_x4_variant prefers anime_6B when configured and file exists."""
        from unittest.mock import patch as _patch
        import server.pipeline.preprocessor as _mod

        anime = tmp_path / "RealESRGAN_x4plus_anime_6B.pth"
        anime.write_bytes(b"fake")
        x4 = tmp_path / "RealESRGAN_x4plus.pth"
        x4.write_bytes(b"fake")

        with _patch.object(_mod, "_MODELS_DIR", tmp_path), \
             _patch("server.config.settings") as mock_s:
            mock_s.upscaler_variant = "anime_6b"
            path, blocks = _mod._pick_x4_variant()
        assert path == anime
        assert blocks == 6

    def test_pick_x4_variant_fallback_to_x4plus(self, tmp_path):
        """anime_6B missing → falls back to x4plus (num_block=23)."""
        from unittest.mock import patch as _patch
        import server.pipeline.preprocessor as _mod

        x4 = tmp_path / "RealESRGAN_x4plus.pth"
        x4.write_bytes(b"fake")

        with _patch.object(_mod, "_MODELS_DIR", tmp_path), \
             _patch("server.config.settings") as mock_s:
            mock_s.upscaler_variant = "anime_6b"
            path, blocks = _mod._pick_x4_variant()
        assert path == x4
        assert blocks == 23

    def test_pick_x4_variant_explicit_x4plus(self, tmp_path):
        """upscaler_variant='x4plus' → uses x4plus even if anime_6B exists."""
        from unittest.mock import patch as _patch
        import server.pipeline.preprocessor as _mod

        anime = tmp_path / "RealESRGAN_x4plus_anime_6B.pth"
        anime.write_bytes(b"fake")
        x4 = tmp_path / "RealESRGAN_x4plus.pth"
        x4.write_bytes(b"fake")

        with _patch.object(_mod, "_MODELS_DIR", tmp_path), \
             _patch("server.config.settings") as mock_s:
            mock_s.upscaler_variant = "x4plus"
            path, blocks = _mod._pick_x4_variant()
        assert path == x4
        assert blocks == 23

    def test_remove_furigana_preserves_dakuten(self):
        """Dakuten-like dots near a large glyph must NOT be erased."""
        from server.pipeline.preprocessor import remove_furigana

        # Create a 200x200 white image with a large glyph and small dots nearby
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255

        # Main character body — a large black rectangle
        img[60:140, 70:130] = 0  # 80x60 block (simulated character body)

        # Dakuten-like dots — two small dots near the upper-right of the glyph
        img[55:62, 132:138] = 0  # dot 1 (7x6, adjacent to glyph)
        img[65:72, 132:138] = 0  # dot 2 (7x6, adjacent to glyph)

        result = remove_furigana(img)

        # The dots should be preserved (still black) because they're near the glyph
        assert (result[58, 135] < 128).all(), "dakuten dot 1 was incorrectly erased"
        assert (result[68, 135] < 128).all(), "dakuten dot 2 was incorrectly erased"

    def test_remove_furigana_removes_distant_small_components(self):
        """Small components far from any main glyph should be erased."""
        from server.pipeline.preprocessor import remove_furigana

        img = np.ones((200, 200, 3), dtype=np.uint8) * 255

        # Main character body
        img[60:140, 70:130] = 0

        # Distant small component (simulated furigana, far from main text)
        img[10:18, 10:16] = 0  # 8x6 block, far away

        result = remove_furigana(img)

        # The distant small component should be erased (white)
        assert (result[14, 13] > 200).all(), "distant small component should be erased"
        # Main glyph should remain
        assert (result[100, 100] < 128).all(), "main glyph was incorrectly erased"


# ---------------------------------------------------------------------------
# TestOCREngine
# ---------------------------------------------------------------------------


class TestOCREngine:
    """OCREngine: model-not-loaded fallback, confidence heuristic."""

    async def test_returns_empty_when_model_not_loaded(self):
        """Model absent → recognize() returns empty OCRResult."""
        from server.pipeline.ocr_engine import OCREngine

        with patch.dict("sys.modules", {"manga_ocr": None}):
            engine = OCREngine.__new__(OCREngine)
            engine.device = "cpu"
            engine._model_loaded = False
            engine._ocr = None

        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        result = await engine.recognize(crop)
        assert result.text == ""
        assert result.confidence == 0.0

    def test_confidence_empty_text(self):
        from server.pipeline.ocr_engine import _estimate_confidence

        assert _estimate_confidence("") == 0.0

    def test_confidence_short_text(self):
        from server.pipeline.ocr_engine import _estimate_confidence

        c = _estimate_confidence("AB")
        assert 0.0 < c <= 1.0

    def test_confidence_boost_cjk(self):
        from server.pipeline.ocr_engine import _estimate_confidence

        cjk = "これはテスト"  # all CJK
        c = _estimate_confidence(cjk)
        assert c >= 0.5

    def test_confidence_long_text(self):
        from server.pipeline.ocr_engine import _estimate_confidence

        c = _estimate_confidence("明日は晴れるでしょう")
        assert c >= 0.9


# ---------------------------------------------------------------------------
# TestTranslator
# ---------------------------------------------------------------------------


class TestTranslator:
    """Translator: Hunyuan-MT-7B local model singleton."""

    async def test_empty_input_returns_empty(self):
        """translate_batch([]) should immediately return [] without loading model."""
        import server.pipeline.translator as trans_mod
        from server.pipeline.translator import Translator
        from unittest.mock import patch

        with patch.object(trans_mod, "_ensure_model_loaded"):
            t = Translator()
        result = await t.translate_batch([])
        assert result == []
        await t.close()

    async def test_translate_calls_model(self):
        """translate_batch() offloads to thread and returns model results."""
        import server.pipeline.translator as trans_mod
        from server.pipeline.translator import Translator
        from unittest.mock import patch

        with patch.object(trans_mod, "_ensure_model_loaded"):
            with patch.object(
                trans_mod,
                "_translate_texts_sync",
                return_value=["안녕하세요"],
            ):
                t = Translator(target_lang="KO", source_lang="JA")
                result = await t.translate_batch(["こんにちは"])

        assert result == ["안녕하세요"]
        await t.close()

    async def test_translate_batch_preserves_order(self):
        """translate_batch returns results in the same order as input."""
        import server.pipeline.translator as trans_mod
        from server.pipeline.translator import Translator
        from unittest.mock import patch

        translated = ["안녕하세요", "감사합니다", "잘 자요"]
        with patch.object(trans_mod, "_ensure_model_loaded"):
            with patch.object(
                trans_mod,
                "_translate_texts_sync",
                return_value=translated,
            ):
                t = Translator()
                result = await t.translate_batch(
                    ["こんにちは", "ありがとう", "おやすみ"]
                )

        assert result == translated
        await t.close()

    async def test_inference_failure_returns_originals(self):
        """When inference raises, translate_batch falls back to originals."""
        import server.pipeline.translator as trans_mod
        from server.pipeline.translator import Translator
        from unittest.mock import patch

        with patch.object(trans_mod, "_ensure_model_loaded"):
            with patch.object(
                trans_mod,
                "_translate_texts_sync",
                side_effect=RuntimeError("CUDA OOM"),
            ):
                t = Translator()
                result = await t.translate_batch(["テスト"])

        assert result == ["テスト"]
        await t.close()

    async def test_close_is_noop(self):
        """close() completes without raising any exception."""
        import server.pipeline.translator as trans_mod
        from server.pipeline.translator import Translator
        from unittest.mock import patch

        with patch.object(trans_mod, "_ensure_model_loaded"):
            t = Translator()
        await t.close()  # must not raise

    def test_unload_model_clears_globals(self):
        """unload_model() resets module-level singletons."""
        import server.pipeline.translator as trans_mod

        # Inject fake model/tokenizer
        trans_mod._model = object()
        trans_mod._tokenizer = object()

        trans_mod.unload_model()

        assert trans_mod._model is None
        assert trans_mod._tokenizer is None

    async def test_legacy_kwargs_accepted(self):
        """deepl_key / google_key kwargs are silently ignored (backward compat)."""
        import server.pipeline.translator as trans_mod
        from server.pipeline.translator import Translator
        from unittest.mock import patch

        with patch.object(trans_mod, "_ensure_model_loaded"):
            t = Translator(
                deepl_key="legacy-key",
                google_key="legacy-key",
                target_lang="KO",
                source_lang="JA",
            )
        assert t.target_lang == "KO"
        await t.close()
# ---------------------------------------------------------------------------
# TestTextEraser
# ---------------------------------------------------------------------------


class TestTextEraser:
    """TextEraser: cv2.inpaint fallback, empty mask returns original."""

    @pytest.fixture()
    def eraser(self):
        with patch("server.pipeline.text_eraser.torch.jit.load", side_effect=Exception("no model")):
            from server.pipeline.text_eraser import TextEraser

            return TextEraser(device="cpu")

    async def test_empty_mask_returns_original(self, eraser):
        """mask.max() == 0 → return image unchanged."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = await eraser.erase(img, mask)
        np.testing.assert_array_equal(result, img)

    async def test_cv2_inpaint_fallback(self, eraser):
        """Model absent → uses cv2.inpaint, result shape matches input."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        assert eraser._model_loaded is False
        result = await eraser.erase(img, mask)
        assert result.shape == img.shape

    async def test_lama_model_path(self, eraser):
        """When model fails to load, _model_loaded is False."""
        assert eraser._model_loaded is False
        assert eraser._model is None


# ---------------------------------------------------------------------------
# TestTextRenderer
# ---------------------------------------------------------------------------


class TestTextRenderer:
    """TextRenderer: font fallback, RGBA output, direction branching."""

    @pytest.fixture()
    def renderer(self, tmp_path):
        from server.pipeline.text_renderer import TextRenderer

        return TextRenderer(font_dir=str(tmp_path))  # empty → default font

    async def test_empty_text_returns_zeros(self, renderer):
        """Empty text should produce an all-zero RGBA image."""
        img = np.zeros((80, 60, 3), dtype=np.uint8)
        bbox = (0, 0, 60, 80)
        result, font_size = await renderer.render(img, "", bbox)
        assert result.shape == (80, 60, 4)
        assert result.max() == 0
        assert font_size == 0

    async def test_output_is_rgba(self, renderer):
        """render() always returns (RGBA array, font_size) tuple."""
        img = np.zeros((80, 60, 3), dtype=np.uint8)
        bbox = (0, 0, 60, 80)
        result, font_size = await renderer.render(img, "테스트", bbox)
        assert result.ndim == 3
        assert result.shape[2] == 4
        assert isinstance(font_size, int)
        assert font_size > 0

    async def test_horizontal_direction(self, renderer):
        """Horizontal text renders without error and returns positive font size."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        bbox = (0, 0, 200, 100)
        result, font_size = await renderer.render(img, "테스트 문장", bbox, text_direction="horizontal")
        assert result.shape[2] == 4
        assert font_size > 0

    async def test_vertical_source_rendered_horizontal(self, renderer):
        """Korean translation from a vertical Japanese bubble must render horizontal."""
        img = np.zeros((300, 50, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 300)
        result, font_size = await renderer.render(
            img, "한국어 번역 테스트", bbox, text_direction="vertical"
        )
        assert result.shape[2] == 4
        # Overlay must contain some non-zero pixels (text was actually drawn)
        assert result.max() > 0, "Korean text must be rendered even on vertical-source bbox"

    async def test_narrow_bbox_does_not_produce_empty_overlay(self, renderer):
        """Regression: narrow Japanese speech bubble bbox (w=23) rendered Korean text
        as empty because usable_w=11px < _MIN_FONT_SIZE → text got truncated to 0 lines.
        font_size must be >= _MIN_FONT_SIZE (12) and overlay must be non-empty.
        """
        from server.pipeline.text_renderer import _MIN_FONT_SIZE

        img = np.zeros((82, 23, 3), dtype=np.uint8)
        # Typical narrow vertical speech bubble bbox from Japanese manga
        bbox = (0, 0, 23, 82)
        text = "그럼 유리하마로 가볼게요~!"
        result, font_size = await renderer.render(img, text, bbox, text_direction="vertical")
        assert result.shape[2] == 4
        assert font_size >= _MIN_FONT_SIZE, (
            f"font_size={font_size} < _MIN_FONT_SIZE={_MIN_FONT_SIZE}: "
            "renderer fell back to no-render on narrow bbox"
        )
        assert result.max() > 0, (
            "Overlay is completely empty — Korean text not rendered in narrow bbox"
        )

    async def test_very_narrow_bbox_width_10(self, renderer):
        """Regression: bbox w=10 (usable_w=1) should still render text.
        Bubble id=9 in e347732c had w=10 and produced zero output.
        """
        img = np.zeros((91, 10, 3), dtype=np.uint8)
        bbox = (0, 0, 10, 91)
        text = "슈바츠메우~!"
        result, font_size = await renderer.render(img, text, bbox, text_direction="vertical")
        assert result.shape[2] == 4
        assert result.max() > 0, "w=10 bbox must still produce visible Korean text"

    async def test_narrow_bbox_text_not_severely_truncated(self, renderer):
        """Regression (e347732c): for a narrow vertical speech bubble (w=23, h=82),
        a 16-char Korean translation must not be clipped to 5 visible characters.

        Root cause:
          usable_w = 23 - 2*PADDING = 11px
          _wrap_text wraps to 1 char/line → 16 lines
          line_height = MIN_FONT_SIZE * LINE_HEIGHT_RATIO = 12*1.4 ≈ 16px
          16 lines × 16px = 256px needed, but overlay is only 82px
          → PIL clips chars 5-16 silently, 69% of text is lost

        After pipeline fix: overlay height must expand to fit all wrapped lines,
        or font size must shrink until all lines fit within original bbox height.
        """
        from server.pipeline.text_renderer import _LINE_HEIGHT_RATIO, _PADDING

        img = np.zeros((82, 23, 3), dtype=np.uint8)
        bbox = (0, 0, 23, 82)
        text = "그럼 유리하마로 가볼게요~!"  # 16 chars → 16 lines at usable_w=11px

        result, font_size = await renderer.render(img, text, bbox, text_direction="vertical")

        # Compute actual lines that WOULD be needed at the returned font_size
        font = renderer._load_font(font_size)
        usable_w = max(23 - _PADDING * 2, 1)
        lines = renderer._wrap_text(text, font, usable_w)
        line_height = int(font_size * _LINE_HEIGHT_RATIO)
        needed_h = len(lines) * line_height + _PADDING * 2

        assert result.shape[0] >= needed_h, (
            f"Overlay height {result.shape[0]}px cannot fit {len(lines)} wrapped lines "
            f"(needs {needed_h}px at font_size={font_size}). "
            f"Characters beyond line {result.shape[0] // line_height} are silently clipped. "
            "Pipeline fix: expand overlay height to `needed_h` and update compositor "
            "to use overlay.shape[0] instead of bbox h."
        )

    async def test_font_size_used_nonzero_for_nonempty_text(self, renderer):
        """font_size_used must be > 0 whenever text is non-empty (P5 log correctness)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (0, 0, 100, 100)
        _, font_size = await renderer.render(img, "테스트", bbox)
        assert font_size > 0, "font_size_used must not be 0 when text is non-empty"

    def test_font_not_found_uses_default(self, tmp_path):
        """No font file in font_dir → _font_path is None, uses Pillow default."""
        from server.pipeline.text_renderer import TextRenderer

        r = TextRenderer(font_dir=str(tmp_path))
        assert r._font_path is None


# ---------------------------------------------------------------------------
# TestCompositor
# ---------------------------------------------------------------------------


class TestCompositor:
    """Compositor: empty list → original, single composite, original immutable."""

    async def test_empty_list_returns_copy(self):
        from server.pipeline.compositor import Compositor

        orig = np.ones((100, 200, 3), dtype=np.uint8) * 42
        result = await Compositor.composite(orig, [])
        np.testing.assert_array_equal(result, orig)
        # must be a copy, not the same object
        assert result is not orig

    async def test_single_bubble_composited(self):
        from server.pipeline.compositor import Compositor, RenderedBubble

        orig = np.zeros((100, 200, 3), dtype=np.uint8)
        # White opaque overlay patch at (10,10,20,20)
        overlay = np.zeros((20, 20, 4), dtype=np.uint8)
        overlay[:, :, :3] = 255
        overlay[:, :, 3] = 255

        rb = RenderedBubble(bbox=(10, 10, 20, 20), image=overlay)
        result = await Compositor.composite(orig, [rb])
        # The region should no longer be black (original was all zeros)
        region = result[10:30, 10:30]
        assert region.mean() > 0

    async def test_original_not_mutated(self):
        from server.pipeline.compositor import Compositor, RenderedBubble

        orig = np.zeros((100, 100, 3), dtype=np.uint8)
        orig_copy = orig.copy()

        overlay = np.full((10, 10, 4), 200, dtype=np.uint8)
        rb = RenderedBubble(bbox=(5, 5, 10, 10), image=overlay)
        await Compositor.composite(orig, [rb])
        np.testing.assert_array_equal(orig, orig_copy)

    async def test_out_of_bounds_bubble_safe(self):
        """Bubble partially outside image → composited without error."""
        from server.pipeline.compositor import Compositor, RenderedBubble

        orig = np.zeros((50, 50, 3), dtype=np.uint8)
        overlay = np.full((30, 30, 4), 128, dtype=np.uint8)
        rb = RenderedBubble(bbox=(40, 40, 30, 30), image=overlay)
        result = await Compositor.composite(orig, [rb])
        assert result.shape == orig.shape


# ---------------------------------------------------------------------------
# TestModelCache
# ---------------------------------------------------------------------------


class TestModelCache:
    """Orchestrator model cache: singleton instances, cache clearing."""

    @pytest.mark.asyncio
    async def test_get_cached_returns_same_instance(self):
        """_get_cached returns the same object on repeated calls."""
        from server.pipeline.orchestrator import _get_cached, _model_cache

        _model_cache.clear()

        class _Dummy:
            def __init__(self, **kwargs):
                pass

        a = await _get_cached(_Dummy, "test-key")
        b = await _get_cached(_Dummy, "test-key")
        assert a is b
        _model_cache.clear()

    @pytest.mark.asyncio
    async def test_clear_model_cache_empties(self):
        """clear_model_cache removes all entries."""
        from server.pipeline.orchestrator import (
            _get_cached,
            _model_cache,
            clear_model_cache,
        )

        class _Dummy:
            def __init__(self, **kwargs):
                pass

        await _get_cached(_Dummy, "test-clear")
        assert len(_model_cache) > 0
        clear_model_cache()
        assert len(_model_cache) == 0


# ---------------------------------------------------------------------------
# TestMagiDetector
# ---------------------------------------------------------------------------


class TestMagiDetector:
    """MagiDetector: bbox conversion, model-not-loaded fallback, reading order."""

    def test_xyxy_to_xywh_conversion(self):
        """Magi bbox [x1,y1,x2,y2] → BubbleInfo (x,y,w,h)."""
        from server.pipeline.magi_detector import _xyxy_to_xywh

        assert _xyxy_to_xywh([10.0, 20.0, 110.0, 80.0]) == (10, 20, 100, 60)
        assert _xyxy_to_xywh([0.0, 0.0, 50.5, 30.9]) == (0, 0, 50, 30)

    async def test_returns_empty_when_model_not_loaded(self):
        """Model unavailable → detect() returns []."""
        with patch("server.pipeline.magi_detector.MagiDetector.__init__", lambda self, device: None):
            from server.pipeline.magi_detector import MagiDetector

            det = MagiDetector.__new__(MagiDetector)
            det.device = "cpu"
            det._model = None
            det._model_loaded = False

        img = np.zeros((200, 300, 3), dtype=np.uint8)
        result = await det.detect(img)
        assert result == []

    async def test_detect_produces_xywh_bubbles(self):
        """Mocked Magi output → BubbleInfo with correct (x,y,w,h)."""
        from server.pipeline.magi_detector import MagiDetector

        det = MagiDetector.__new__(MagiDetector)
        det.device = "cpu"
        det._model_loaded = True

        mock_model = MagicMock()
        mock_model.predict_detections_and_associations.return_value = [{
            "texts": [[10.0, 20.0, 110.0, 80.0], [200.0, 50.0, 300.0, 250.0]],
            "is_essential_text": [True, False],
            "panels": [],
            "characters": [],
        }]
        det._model = mock_model

        img = np.zeros((300, 400, 3), dtype=np.uint8)
        bubbles = await det.detect(img)

        assert len(bubbles) == 2
        assert bubbles[0].bbox == (10, 20, 100, 60)
        assert bubbles[0].bubble_type == "speech"
        assert bubbles[0].text_direction == "horizontal"  # w=100>h=60

        assert bubbles[1].bbox == (200, 50, 100, 200)
        assert bubbles[1].bubble_type == "effect"
        assert bubbles[1].text_direction == "vertical"  # h=200>w=100*1.2

    async def test_reading_order_preserved_from_magi(self):
        """Magi's internal sort order is preserved (not re-sorted)."""
        from server.pipeline.magi_detector import MagiDetector

        det = MagiDetector.__new__(MagiDetector)
        det.device = "cpu"
        det._model_loaded = True

        mock_model = MagicMock()
        mock_model.predict_detections_and_associations.return_value = [{
            "texts": [[300.0, 10.0, 400.0, 50.0], [10.0, 10.0, 100.0, 50.0]],
            "is_essential_text": [True, True],
            "panels": [],
            "characters": [],
        }]
        det._model = mock_model

        img = np.zeros((100, 500, 3), dtype=np.uint8)
        bubbles = await det.detect(img)

        # Magi returns right bubble first → reading_order should stay 1, 2
        assert bubbles[0].reading_order == 1
        assert bubbles[0].bbox[0] == 300  # right bubble first
        assert bubbles[1].reading_order == 2
        assert bubbles[1].bbox[0] == 10

    def test_zero_size_bbox_skipped(self):
        """Zero-width or zero-height bboxes are filtered out."""
        from server.pipeline.magi_detector import _xyxy_to_xywh

        x, y, w, h = _xyxy_to_xywh([50.0, 50.0, 50.0, 100.0])
        assert w == 0  # would be filtered in detect()


class TestMagiCacheKeySeparation:
    """Magi and YOLOv5 use different cache keys in orchestrator."""

    @pytest.mark.asyncio
    async def test_separate_cache_keys(self):
        """magi_detector and bubble_detector use separate cache keys."""
        from server.pipeline.orchestrator import _get_cached, _model_cache

        _model_cache.clear()

        class _FakeYOLO:
            def __init__(self, **kw):
                self.name = "yolo"

        class _FakeMagi:
            def __init__(self, **kw):
                self.name = "magi"

        yolo = await _get_cached(_FakeYOLO, "bubble_detector", device="cpu")
        magi = await _get_cached(_FakeMagi, "magi_detector", device="cpu")

        assert yolo is not magi
        assert yolo.name == "yolo"
        assert magi.name == "magi"
        assert "bubble_detector" in _model_cache
        assert "magi_detector" in _model_cache
        _model_cache.clear()
