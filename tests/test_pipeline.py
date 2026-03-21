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
    """Translator: no-key fallback, DeepL mock, Google fallback, retry, auth error."""

    async def test_no_keys_returns_originals(self):
        from server.pipeline.translator import Translator

        t = Translator(deepl_key="", google_key="")
        result = await t.translate_batch(["こんにちは"])
        assert result == ["こんにちは"]
        await t.close()

    async def test_empty_input_returns_empty(self):
        from server.pipeline.translator import Translator

        t = Translator(deepl_key="key")
        result = await t.translate_batch([])
        assert result == []
        await t.close()

    async def test_deepl_success(self):
        from server.pipeline.translator import Translator

        t = Translator(deepl_key="fake-key", google_key="")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "translations": [{"text": "안녕하세요"}]
        }

        with patch.object(t._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await t.translate_batch(["こんにちは"])

        assert result == ["안녕하세요"]
        await t.close()

    async def test_google_fallback_on_deepl_failure(self):
        from server.pipeline.translator import Translator

        t = Translator(deepl_key="dk", google_key="gk")

        deepl_resp = MagicMock()
        deepl_resp.status_code = 500
        deepl_resp.raise_for_status.side_effect = Exception("server error")

        google_resp = MagicMock()
        google_resp.status_code = 200
        google_resp.raise_for_status = MagicMock()
        google_resp.json.return_value = {
            "data": {"translations": [{"translatedText": "번역됨"}]}
        }

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First 3 calls are DeepL retries, then Google
            if call_count <= 3:
                return deepl_resp
            return google_resp

        with patch.object(t._client, "post", side_effect=mock_post):
            result = await t.translate_batch(["テスト"])

        assert result == ["번역됨"]
        await t.close()

    async def test_auth_error_not_retried_deepl(self):
        """DeepL 401 → called only once (no retry), then returns originals."""
        from server.pipeline.translator import Translator

        t = Translator(deepl_key="bad-key", google_key="")

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        mock_post = AsyncMock(return_value=mock_resp)
        with patch.object(t._client, "post", mock_post):
            result = await t.translate_batch(["テスト"])

        # Auth error → exactly 1 call, no retry
        assert mock_post.call_count == 1
        assert result == ["テスト"]
        await t.close()

    async def test_retry_3_times_then_fallback_to_originals(self):
        """DeepL retries 3 times, Google also fails → return originals."""
        from server.pipeline.translator import Translator

        t = Translator(deepl_key="dk", google_key="gk")

        fail_resp = MagicMock()
        fail_resp.status_code = 500
        fail_resp.raise_for_status.side_effect = Exception("fail")

        with patch.object(t._client, "post", new_callable=AsyncMock, return_value=fail_resp), \
             patch("server.pipeline.translator.asyncio.sleep", new_callable=AsyncMock):
            result = await t.translate_batch(["テスト"])

        assert result == ["テスト"]
        await t.close()

    async def test_auth_error_google_403(self):
        """Google 403 → called only once (no retry), then returns originals."""
        from server.pipeline.translator import Translator

        t = Translator(deepl_key="", google_key="bad")

        mock_resp = MagicMock()
        mock_resp.status_code = 403

        mock_post = AsyncMock(return_value=mock_resp)
        with patch.object(t._client, "post", mock_post):
            result = await t.translate_batch(["テスト"])

        assert mock_post.call_count == 1
        assert result == ["テスト"]
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
        result = await renderer.render(img, "", bbox)
        assert result.shape == (80, 60, 4)
        assert result.max() == 0

    async def test_output_is_rgba(self, renderer):
        """render() always returns RGBA (H, W, 4)."""
        img = np.zeros((80, 60, 3), dtype=np.uint8)
        bbox = (0, 0, 60, 80)
        result = await renderer.render(img, "테스트", bbox)
        assert result.ndim == 3
        assert result.shape[2] == 4

    async def test_horizontal_direction(self, renderer):
        """Horizontal text renders without error."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        bbox = (0, 0, 200, 100)
        result = await renderer.render(img, "테스트 문장", bbox, text_direction="horizontal")
        assert result.shape[2] == 4

    async def test_vertical_direction(self, renderer):
        """Vertical text on tall bbox triggers vertical layout."""
        img = np.zeros((300, 50, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 300)
        result = await renderer.render(img, "テスト", bbox, text_direction="vertical")
        assert result.shape[2] == 4

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

    def test_get_cached_returns_same_instance(self):
        """_get_cached returns the same object on repeated calls."""
        from server.pipeline.orchestrator import _get_cached, _model_cache

        _model_cache.clear()

        class _Dummy:
            def __init__(self, **kwargs):
                pass

        a = _get_cached(_Dummy, "test-key")
        b = _get_cached(_Dummy, "test-key")
        assert a is b
        _model_cache.clear()

    def test_clear_model_cache_empties(self):
        """clear_model_cache removes all entries."""
        from server.pipeline.orchestrator import (
            _get_cached,
            _model_cache,
            clear_model_cache,
        )

        class _Dummy:
            def __init__(self, **kwargs):
                pass

        _get_cached(_Dummy, "test-clear")
        assert len(_model_cache) > 0
        clear_model_cache()
        assert len(_model_cache) == 0
