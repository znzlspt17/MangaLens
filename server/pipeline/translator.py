"""Translation — DeepL (primary) / Google Translate (fallback)."""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

# Sentinel for auth errors that should NOT be retried.
_NON_RETRYABLE_STATUS = frozenset({401, 403})

_DEEPL_FREE_URL = "https://api-free.deepl.com/v2/translate"
_DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
_GOOGLE_URL = "https://translation.googleapis.com/language/translate/v2"


class TranslationAuthError(Exception):
    """Raised on 401/403 — do not retry."""


class Translator:
    """Batch translator with DeepL-first, Google-fallback strategy.

    Features:
    - Batch translation to minimise API calls.
    - Context-aware: surrounding bubble texts passed as context.
    - Automatic retry with exponential backoff (max 3 attempts).
    """

    def __init__(
        self,
        deepl_key: str = "",
        google_key: str = "",
        target_lang: str = "KO",
        source_lang: str = "JA",
    ) -> None:
        self.deepl_key = deepl_key
        self.google_key = google_key
        self.target_lang = target_lang
        self.source_lang = source_lang
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    async def translate_batch(
        self,
        texts: list[str],
        context: list[str] | None = None,
    ) -> list[str]:
        """Translate a batch of texts, preserving dialogue context.

        Tries DeepL first; falls back to Google Translate when DeepL is
        unavailable or errors out.  Retries up to 3 times per engine
        with exponential backoff.

        Args:
            texts: List of source-language strings to translate.
            context: Optional surrounding texts for contextual translation.

        Returns:
            List of translated strings in the same order as ``texts``.
        """
        if not texts:
            return []

        # --- Try DeepL ---
        if self.deepl_key:
            try:
                return await self._translate_with_retry("deepl", texts, context)
            except Exception:
                logger.warning("DeepL translation failed, trying Google fallback")

        # --- Try Google ---
        if self.google_key:
            try:
                return await self._translate_with_retry("google", texts, context)
            except Exception:
                logger.warning("Google translation also failed, returning originals")

        # Both unavailable / failed → return originals
        if not self.deepl_key and not self.google_key:
            logger.warning("No translation API keys configured, returning originals")
        return list(texts)

    async def _translate_with_retry(
        self,
        engine: str,
        texts: list[str],
        context: list[str] | None,
        max_retries: int = 3,
    ) -> list[str]:
        """Call a translation engine with exponential-backoff retry.

        Args:
            engine: ``"deepl"`` or ``"google"``.
            texts: Texts to translate.
            context: Optional context.
            max_retries: Maximum retry attempts.

        Returns:
            Translated texts.

        Raises:
            Exception: If all retries are exhausted.
        """
        for attempt in range(max_retries):
            try:
                if engine == "deepl":
                    return await self._call_deepl(texts, context)
                else:
                    return await self._call_google(texts, context)
            except TranslationAuthError:
                # 401/403 — no point retrying
                raise
            except Exception:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "Translation attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1,
                    max_retries,
                    engine,
                    wait,
                )
                await asyncio.sleep(wait)
        # Unreachable but keeps type-checkers happy
        raise RuntimeError("Retry loop exited unexpectedly")

    async def _call_deepl(
        self,
        texts: list[str],
        context: list[str] | None,
    ) -> list[str]:
        """Call DeepL API v2."""
        url = (
            _DEEPL_FREE_URL
            if self.deepl_key.endswith(":fx")
            else _DEEPL_PRO_URL
        )
        headers = {"Authorization": f"DeepL-Auth-Key {self.deepl_key}"}
        body: dict = {
            "text": texts,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
        }
        if context:
            body["context"] = "\n".join(context)

        resp = await self._client.post(url, json=body, headers=headers)

        if resp.status_code in _NON_RETRYABLE_STATUS:
            raise TranslationAuthError(
                f"DeepL auth error: {resp.status_code}"
            )
        resp.raise_for_status()

        data = resp.json()
        return [t["text"] for t in data["translations"]]

    async def _call_google(
        self,
        texts: list[str],
        context: list[str] | None,
    ) -> list[str]:
        """Call Google Cloud Translation API v2."""
        params = {"key": self.google_key}
        body = {
            "q": texts,
            "source": self.source_lang.lower(),
            "target": self.target_lang.lower(),
            "format": "text",
        }

        resp = await self._client.post(_GOOGLE_URL, params=params, json=body)

        if resp.status_code in _NON_RETRYABLE_STATUS:
            raise TranslationAuthError(
                f"Google auth error: {resp.status_code}"
            )
        resp.raise_for_status()

        data = resp.json()
        return [
            t["translatedText"]
            for t in data["data"]["translations"]
        ]

    async def close(self) -> None:
        """Shut down the underlying HTTP client."""
        await self._client.aclose()
