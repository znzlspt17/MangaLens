"""Translation — Hunyuan-MT-7B local model (JA→KO).

Model : tencent/Hunyuan-MT-7B  (CausalLM, ~7 B parameters)

Fixes applied vs. original naive implementation:
  1. ``apply_chat_template`` is used when available so the model receives
     properly formatted system/user messages instead of raw completion text.
  2. ``max_new_tokens`` is capped dynamically at 4× the source token length
     (floor 32, ceil 256) to prevent runaway generation on short inputs.
  3. ``repetition_penalty=1.15`` and ``no_repeat_ngram_size=5`` guard against
     repetition loops (e.g. "하하하하하…").
  4. Post-processing strips common LLM noise: leading '. ', '* ', '**Note:**…',
     and lines that are longer than 5× the expected output length.

The model is loaded **once** at module level behind a threading.Lock and
reused across every request.  asyncio.to_thread() offloads the blocking
inference so the ASGI event loop is never stalled.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading

logger = logging.getLogger(__name__)

_HUNYUAN_MODEL_ID = "tencent/Hunyuan-MT-7B"
_MAX_NEW_TOKENS_HARD_CAP = 256
_MAX_NEW_TOKENS_FLOOR = 32
# Multiplier applied to source token length to compute a dynamic cap.
_MAX_NEW_TOKENS_RATIO = 4
# A-1: Context-window batch limit — max tokens before falling back to individual
_BATCH_CONTEXT_MAX_TOKENS = 900
# A-1: Hard cap for batch context generation (all lines combined)
_BATCH_MAX_NEW_TOKENS = 1024

_SYSTEM_MSG_TEMPLATE = (
    "You are a professional manga translator specializing in {src} to {tgt} translation. "
    "Rules:\n"
    "1. Translate dialogue into natural, spoken {tgt}. Preserve the tone (casual/formal) of each line.\n"
    "2. Handle dialects: Kansai-ben (〜やん, 〜とんねん, 〜やねん), Tohoku-ben, etc. "
    "Convey the regional flavor naturally.\n"
    "3. Translate cultural terms accurately: 家系ラーメン→이에케이 라면, 割りばし→나무젓가락, "
    "コンビニ→편의점. Do NOT guess — use the correct meaning.\n"
    "4. Transliterate proper nouns using standard conventions: キアヌ→키아누, ジョン→존.\n"
    "5. Translate onomatopoeia/mimetic words into natural {tgt} equivalents.\n"
    "6. Output ONLY the {tgt} translation — no explanations, no notes, no original text."
)

# A-1: Context-window batch prompt — instructs model to translate all lines together
_BATCH_SYSTEM_MSG_TEMPLATE = (
    "You are a professional manga translator specializing in {src} to {tgt} translation. "
    "You will receive multiple dialogue lines from the same manga page, numbered <1> to <N>. "
    "These lines form a continuous conversation — use the full context to produce accurate, "
    "consistent translations.\n"
    "Rules:\n"
    "1. Translate each line into natural, spoken {tgt}. Preserve the tone (casual/formal).\n"
    "2. Handle dialects: Kansai-ben (〜やん, 〜とんねん), Tohoku-ben, etc.\n"
    "3. Translate cultural terms accurately: 家系ラーメン→이에케이 라면, 割りばし→나무젓가락.\n"
    "4. Transliterate proper nouns consistently: キアヌ→키아누, ジョン→존.\n"
    "5. Output ONLY the translations, one per line, numbered <1> to <N> matching the input.\n"
    "6. No explanations, no notes, no original text."
)
_PROMPT_TEMPLATE = (
    "Translate the following {src} manga dialogue into {tgt}. "
    "Output only the {tgt} translation without any explanation.\n{text}"
)

# Few-shot examples injected into chat context for better quality.
_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {"user": "お待たせしましたー", "assistant": "오래 기다리셨습니다~"},
    {"user": "最近家系ラーメンにはまっとんねん", "assistant": "요즘 이에케이 라면에 빠져있거든"},
    {"user": "割りばし一本で殺し屋を返り討ちにした", "assistant": "나무젓가락 하나로 킬러를 역관광 시켰어"},
]

_LANG_NAMES = {
    "JA": "Japanese", "KO": "Korean", "EN": "English",
    "ZH": "Chinese",  "ES": "Spanish", "FR": "French",
}

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

def _build_input_ids(text: str, source_lang: str = "JA", target_lang: str = "KO") -> tuple["torch.Tensor", int]:
    """Return ``(input_ids_tensor, token_count)`` for *text*.

    Uses ``apply_chat_template`` when the tokenizer supports it so the model
    receives properly structured system/user turns.  Falls back to the plain
    prompt template for tokenizers that lack the method.
    """
    src_name = _LANG_NAMES.get(source_lang, source_lang)
    tgt_name = _LANG_NAMES.get(target_lang, target_lang)
    system_msg = _SYSTEM_MSG_TEMPLATE.format(src=src_name, tgt=tgt_name)

    has_chat = (
        hasattr(_tokenizer, "apply_chat_template")
        and _tokenizer.chat_template is not None
    )
    if has_chat:
        messages = [
            {"role": "system", "content": system_msg},
        ]
        # Inject few-shot examples for better translation quality
        for ex in _FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": text})
        result = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        # apply_chat_template may return a BatchEncoding (dict-like) or a
        # plain tensor depending on the transformers version.
        if hasattr(result, "input_ids"):
            encoded = result["input_ids"]
        else:
            encoded = result
    else:
        prompt = _PROMPT_TEMPLATE.format(src=src_name, tgt=tgt_name, text=text)
        encoded = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )["input_ids"]

    return encoded, int(encoded.shape[1])


def _dynamic_max_new_tokens(src_token_len: int) -> int:
    """Compute a reasonable upper bound for generated tokens.

    Short inputs (≤ 8 tokens) easily produce hallucinated walls of text when
    given the full 256-token budget.  Cap at ``ratio × source_len`` while
    keeping a floor so very short inputs still get enough budget.
    """
    dynamic = src_token_len * _MAX_NEW_TOKENS_RATIO
    return max(_MAX_NEW_TOKENS_FLOOR, min(dynamic, _MAX_NEW_TOKENS_HARD_CAP))


_NOISE_PREFIXES = (". ", "* ", "- ", "**", "# ")
_NOTE_MARKERS = ("**note:", "(note:", "[note:", "※", "translation note", "참고:")


def _postprocess(raw: str, src_text: str) -> str:
    """Remove common LLM noise artefacts from a translated string.

    Operations (in order):
    1. Strip surrounding whitespace.
    2. Remove a leading '. ' or '* ' that Hunyuan sometimes prepends.
    3. Drop everything from the first 'Note:' / '※' line onwards.
    4. If the result is longer than 6× the source character length,
       keep only the first sentence — a strong hallucination signal.
    5. Strip again.
    """
    text = raw.strip()

    # 1. Remove noisy leading markers
    for prefix in _NOISE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break

    # 2. Truncate at Note/※ markers (model started adding explanations)
    lines = text.splitlines()
    clean_lines: list[str] = []
    for line in lines:
        if any(line.strip().lower().startswith(m) for m in _NOTE_MARKERS):
            break
        clean_lines.append(line)
    text = "\n".join(clean_lines).strip()

    # 3. Hallucination guard — output vastly longer than source is suspicious
    hard_limit = max(len(src_text) * 6, 80)
    if len(text) > hard_limit:
        # Keep only up to the first sentence-ending punctuation
        for i, ch in enumerate(text):
            if ch in (".", "!", "?", "…", "。", "！", "？") and i >= len(src_text):
                text = text[: i + 1].strip()
                break
        else:
            text = text[:hard_limit].rstrip() + "…"

    return text.strip()


def _translate_texts_sync(texts: list[str], source_lang: str = "JA", target_lang: str = "KO") -> list[str]:
    """Translate *texts* one-by-one using the loaded model.

    This function is synchronous and CPU/GPU-bound.  It must be called
    via ``asyncio.to_thread()`` to avoid blocking the event loop.
    """
    import torch

    results: list[str] = []
    for text in texts:
        try:
            input_ids, src_len = _build_input_ids(text, source_lang, target_lang)
            input_ids = input_ids.to(_model_device)
            max_new = _dynamic_max_new_tokens(src_len)

            with torch.no_grad():
                output_ids = _model.generate(
                    input_ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=5,
                    eos_token_id=_tokenizer.eos_token_id,
                    pad_token_id=(
                        _tokenizer.pad_token_id
                        if _tokenizer.pad_token_id is not None
                        else _tokenizer.eos_token_id
                    ),
                )

            # Slice off the prompt tokens — decode only newly generated tokens.
            generated_ids = output_ids[0][src_len:]
            decoded = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append(_postprocess(decoded, text))

            # Free GPU tensors promptly
            del input_ids, output_ids, generated_ids
        except Exception:
            logger.warning("Translation failed for text: %s", text[:40], exc_info=True)
            results.append(text)  # fallback: return original

    return results


# ---------------------------------------------------------------------------
# A-1: Context-window batch translation
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<(\d+)>\s*")


def _build_batch_input(texts: list[str]) -> str:
    """Format multiple texts as numbered lines for batch context translation."""
    return "\n".join(f"<{i+1}> {t}" for i, t in enumerate(texts))


def _parse_batch_output(raw: str, count: int) -> list[str] | None:
    """Parse numbered output lines from batch translation.

    Returns None if parsing fails (not enough lines recovered).
    """
    lines_out: dict[int, str] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _TAG_RE.match(line)
        if m:
            idx = int(m.group(1))
            content = line[m.end():].strip()
            if 1 <= idx <= count and content:
                lines_out[idx] = content
        elif len(lines_out) == 0:
            # Model may omit tags for single-line or first line
            continue

    if len(lines_out) < count * 0.7:
        # Too many lines missing — parsing failed
        return None

    results: list[str] = []
    for i in range(1, count + 1):
        results.append(lines_out.get(i, ""))
    return results


def _translate_batch_context_sync(
    texts: list[str], source_lang: str = "JA", target_lang: str = "KO"
) -> list[str] | None:
    """Translate all texts in a single context-aware LLM call.

    Returns None if the batch approach fails, signaling the caller to
    fall back to individual translation.
    """
    import torch

    if not texts:
        return []

    src_name = _LANG_NAMES.get(source_lang, source_lang)
    tgt_name = _LANG_NAMES.get(target_lang, target_lang)
    system_msg = _BATCH_SYSTEM_MSG_TEMPLATE.format(src=src_name, tgt=tgt_name)
    user_msg = _build_batch_input(texts)

    has_chat = (
        hasattr(_tokenizer, "apply_chat_template")
        and _tokenizer.chat_template is not None
    )

    if not has_chat:
        # Batch mode requires chat template for clear role separation
        return None

    messages = [
        {"role": "system", "content": system_msg},
    ]
    # Few-shot: show batch format example
    messages.append({
        "role": "user",
        "content": "<1> お待たせしましたー\n<2> 最近家系ラーメンにはまっとんねん\n<3> 割りばし一本で殺し屋を返り討ちにした",
    })
    messages.append({
        "role": "assistant",
        "content": "<1> 오래 기다리셨습니다~\n<2> 요즘 이에케이 라면에 빠져있거든\n<3> 나무젓가락 하나로 킬러를 역관광 시켰어",
    })
    messages.append({"role": "user", "content": user_msg})

    try:
        result = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        if hasattr(result, "input_ids"):
            input_ids = result["input_ids"]
        else:
            input_ids = result

        src_len = int(input_ids.shape[1])

        if src_len > _BATCH_CONTEXT_MAX_TOKENS:
            logger.info(
                "[batch-translate] Input too long (%d tokens > %d) — falling back to individual",
                src_len, _BATCH_CONTEXT_MAX_TOKENS,
            )
            return None

        input_ids = input_ids.to(_model_device)

        with torch.no_grad():
            output_ids = _model.generate(
                input_ids,
                max_new_tokens=_BATCH_MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=5,
                eos_token_id=_tokenizer.eos_token_id,
                pad_token_id=(
                    _tokenizer.pad_token_id
                    if _tokenizer.pad_token_id is not None
                    else _tokenizer.eos_token_id
                ),
            )

        generated_ids = output_ids[0][src_len:]
        decoded = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        del input_ids, output_ids, generated_ids

        parsed = _parse_batch_output(decoded, len(texts))
        if parsed is None:
            logger.warning(
                "[batch-translate] Failed to parse batch output — falling back to individual. Raw: %s",
                decoded[:200],
            )
            return None

        # Post-process each line
        results = [_postprocess(t, src) for t, src in zip(parsed, texts)]

        # Fill any empty slots with fallback
        for i, (r, orig) in enumerate(zip(results, texts)):
            if not r.strip():
                results[i] = orig

        logger.info("[batch-translate] Success: %d/%d lines parsed", len(results), len(texts))
        return results

    except Exception:
        logger.warning("[batch-translate] Exception — falling back to individual", exc_info=True)
        return None


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

        A-1 Context Window: First attempts batch context translation
        (all lines in one LLM call for conversation coherence).  Falls
        back to individual per-line translation on failure.

        Args:
            texts:   Source-language strings to translate.
            context: Reserved for future use.

        Returns:
            Translated strings in the same order as texts.
        """
        if not texts:
            return []

        # A-1: Try batch context translation first (all lines in one call)
        if len(texts) >= 2:
            try:
                batch_results = await asyncio.to_thread(
                    _translate_batch_context_sync, texts, self.source_lang, self.target_lang
                )
                if batch_results is not None:
                    return batch_results
                logger.info("Batch context returned None — falling back to individual translation")
            except Exception:
                logger.warning("Batch context translation failed — falling back", exc_info=True)

        # Fallback: individual per-line translation
        try:
            results = await asyncio.to_thread(
                _translate_texts_sync, texts, self.source_lang, self.target_lang
            )
        except Exception:
            logger.exception("Hunyuan-MT-7B inference failed; returning originals")
            return list(texts)

        return results

    async def close(self) -> None:
        """No-op: the model remains in memory for reuse across requests.

        Call unload_model() at shutdown if explicit memory release is needed.
        """
