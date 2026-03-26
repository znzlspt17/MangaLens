"""Translation — Hunyuan-MT-7B local model (JA→KO).

Model : tencent/Hunyuan-MT-7B  (CausalLM, ~7 B parameters)
Prompt: "Translate the following segment into Korean, without additional
         explanation.\n<source_text>"   (per model card)

The model is loaded **once** at module level behind a threading.Lock and
reused across every request.  asyncio.to_thread() offloads the blocking
inference so the ASGI event loop is never stalled.

Public interface (Translator class, translate_batch, close) is identical
to the previous API-based implementation so the orchestrator needs only
one-line changes.
"""

from __future__ import annotations

import asyncio
import logging
import threading

logger = logging.getLogger(__name__)

_HUNYUAN_MODEL_ID = "tencent/Hunyuan-MT-7B"
_MAX_NEW_TOKENS = 256
# Per model card — do NOT alter without re-validating translation quality.
_PROMPT_TEMPLATE = (
    "Translate the following segment into Korean, "
    "without additional explanation.\n{text}"
)

# ---------------------------------------------------------------------------
# Module-level singleton  (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_model_device: str = "cpu"
_init_lock = threading.Lock()


def _ensure_model_loaded(device: str = "cpu") -> None:
    """Load Hunyuan-MT-7B exactly once; subsequent calls are no-ops."""
    global _model, _tokenizer, _model_device

    if _model is not None:
        return

    with _init_lock:
        if _model is not None:  # second check inside the lock
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for local translation. "
                "Run: uv add transformers"
            ) from exc

        logger.info("Loading %s on device=%s ...", _HUNYUAN_MODEL_ID, device)

        _tokenizer = AutoTokenizer.from_pretrained(
            _HUNYUAN_MODEL_ID,
            trust_remote_code=True,
        )

        # fp16 on GPU, fp32 on CPU (fp16 unsupported on most CPU builds).
        torch_dtype = torch.float16 if device != "cpu" else torch.float32

        _model = AutoModelForCausalLM.from_pretrained(
            _HUNYUAN_MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=device,          # "auto" also works when accelerate is installed
            trust_remote_code=True,
        )
        _model.eval()
        _model_device = device
        logger.info("Hunyuan-MT-7B ready on %s", device)


def unload_model() -> None:
    """Release the model from memory (useful for testing / graceful shutdown)."""
    global _model, _tokenizer
    with _init_lock:
        _model = None
        _tokenizer = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("Hunyuan-MT-7B unloaded")


# ---------------------------------------------------------------------------
# Synchronous inference (runs inside asyncio.to_thread)
# ---------------------------------------------------------------------------

def _translate_texts_sync(texts: list[str]) -> list[str]:
    """Translate *texts* one-by-one using the loaded model.

    This function is synchronous and CPU/GPU-bound.  It must be called
    via ``asyncio.to_thread()`` to avoid blocking the event loop.
    """
    import torch

    results: list[str] = []
    for text in texts:
        prompt = _PROMPT_TEMPLATE.format(text=text)
        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(_model_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=_tokenizer.eos_token_id,
                pad_token_id=(
                    _tokenizer.pad_token_id
                    if _tokenizer.pad_token_id is not None
                    else _tokenizer.eos_token_id
                ),
            )

        # Slice off the prompt tokens — decode only newly generated tokens.
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        decoded = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        results.append(decoded)

    return results


# ---------------------------------------------------------------------------
# Public Translator class
# ---------------------------------------------------------------------------

class Translator:
    """Local Hunyuan-MT-7B translator (JA->KO).

    The public interface is intentionally identical to the previous
    API-based Translator so the orchestrator can adopt it with minimal
    changes.

    Args:
        deepl_key:  Unused -- kept for API compatibility with orchestrator.
        google_key: Unused -- kept for API compatibility with orchestrator.
        target_lang: Target language code (default: "KO").
        source_lang: Source language code (default: "JA").
        device:     Torch device string passed by the orchestrator after
                    GPU detection (e.g. "cpu", "cuda", "cuda:0").
    """

    def __init__(
        self,
        deepl_key: str = "",
        google_key: str = "",
        target_lang: str = "KO",
        source_lang: str = "JA",
        device: str = "cpu",
    ) -> None:
        self.target_lang = target_lang
        self.source_lang = source_lang
        self._device = device
        # Trigger model load -- no-op on subsequent Translator instantiations.
        _ensure_model_loaded(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def translate_batch(
        self,
        texts: list[str],
        context: list[str] | None = None,
    ) -> list[str]:
        """Translate a batch of texts using Hunyuan-MT-7B.

        Inference runs in a thread pool via asyncio.to_thread() so the
        ASGI event loop remains unblocked.

        Args:
            texts:   Source-language strings to translate.
            context: Reserved for future context-aware prompting; currently
                     not incorporated into the LLM prompt.

        Returns:
            Translated strings in the same order as texts.
        """
        if not texts:
            return []

        if context:
            logger.debug(
                "translate_batch: context arg received (%d items) -- "
                "not yet incorporated into LLM prompt",
                len(context),
            )

        try:
            results = await asyncio.to_thread(_translate_texts_sync, texts)
        except Exception:
            logger.exception("Hunyuan-MT-7B inference failed; returning originals")
            return list(texts)

        return results

    async def close(self) -> None:
        """No-op: the model remains in memory for reuse across requests.

        Call unload_model() at shutdown if explicit memory release is needed.
        """
