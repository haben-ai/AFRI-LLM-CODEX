"""
Offline translation layer: takes English text and translates it into a
low-resource African language using Meta's NLLB-200 (distilled 600M),
pre-quantized to CTranslate2 INT8 format. Runs entirely from a local model
directory -- no network calls, no dependency on src/engine.py's llama-cli
pipeline.

Why a dedicated translation model instead of asking the primary LLM to
write the regional-language section itself: earlier testing in this project
(adtc-2026-submission/scripts/test_multilingual_inference.py, and this
engine's own PedagogicalEngine runs) showed Qwen2.5-Coder-1.5B-Instruct does
not reliably produce genuine Swahili/Amharic/Tigrinya text on its own -- it
either falls back to English, or degenerates into repetitive nonsense under
grammar pressure. NLLB-200 is a model actually trained for translation
across 200 languages, including these; delegating translation to it instead
of the code-focused instruct model is the point of this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import ctranslate2
except ImportError:
    ctranslate2 = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

UNAVAILABLE_NOTICE = "[Translation unavailable]"

# FLORES-200 / NLLB language codes for the languages this project targets.
# resolve_language_code() also accepts a FLORES code passed directly, so
# this map doesn't need to be exhaustive for the class to be usable with
# any of NLLB's other 200 languages.
LANGUAGE_CODES = {
    "swahili": "swh_Latn",
    "amharic": "amh_Ethi",
    "tigrinya": "tir_Ethi",
}

SOURCE_LANG = "eng_Latn"


class OfflineTranslator:
    """Wraps a local CTranslate2 NLLB-200-distilled-600M model for
    English -> low-resource-African-language translation.

    Fails soft, not hard: if the model directory is missing or incomplete,
    or the ctranslate2/transformers packages aren't installed, translate()
    returns the original English text prefixed with an
    "[Translation unavailable]" notice instead of raising. This module is a
    secondary enhancement layer; a broken translator should never take down
    a pipeline that already has a perfectly good English explanation to
    fall back to.
    """

    def __init__(self, model_dir: str = "model/nllb_ct2", device: str = "cpu", threads: int = 4):
        self.model_dir = Path(model_dir)
        self.device = device
        self.threads = threads
        self._translator = None
        self._tokenizer = None
        self._init_error: Optional[str] = None
        self._load()

    @property
    def available(self) -> bool:
        return self._translator is not None and self._tokenizer is not None

    def _load(self) -> None:
        if ctranslate2 is None:
            self._init_error = "ctranslate2 is not installed (pip install ctranslate2)."
            return
        if AutoTokenizer is None:
            self._init_error = "transformers is not installed (pip install transformers)."
            return
        if not self.model_dir.is_dir() or not (self.model_dir / "model.bin").exists():
            self._init_error = (
                f"NLLB model directory not found or incomplete: {self.model_dir} "
                f"(run download_model.sh first)."
            )
            return

        try:
            self._translator = ctranslate2.Translator(
                str(self.model_dir),
                device=self.device,
                compute_type="int8",
                inter_threads=1,
                intra_threads=self.threads,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), clean_up_tokenization_spaces=True,
            )
        except Exception as exc:
            # Model directory exists but files are corrupt/incompatible --
            # fail soft here too, same as a missing directory.
            self._translator = None
            self._tokenizer = None
            self._init_error = f"Failed to load NLLB model from {self.model_dir}: {exc}"

    @staticmethod
    def resolve_language_code(target_lang: str) -> Optional[str]:
        """Map a user-friendly name ("Swahili") to its FLORES-200 code
        ("swh_Latn"). Also accepts an already-valid FLORES code directly
        (e.g. "swh_Latn"), so callers aren't limited to LANGUAGE_CODES'
        three entries if they know the code for another NLLB language."""
        key = target_lang.strip().lower()
        if key in LANGUAGE_CODES:
            return LANGUAGE_CODES[key]
        if target_lang.strip() in LANGUAGE_CODES.values():
            return target_lang.strip()
        return None

    def translate(self, text: str, target_lang: str) -> str:
        """Translate `text` (English) into `target_lang`. Never raises --
        returns an "[Translation unavailable]"-prefixed fallback on any
        failure (missing model, unsupported language, or a runtime error
        from ctranslate2 itself)."""
        if not text or not text.strip():
            return text

        if not self.available:
            return f"{UNAVAILABLE_NOTICE} ({self._init_error})\n{text}"

        tgt_code = self.resolve_language_code(target_lang)
        if tgt_code is None:
            supported = ", ".join(sorted(LANGUAGE_CODES))
            return (
                f"{UNAVAILABLE_NOTICE} (unknown target language '{target_lang}'; "
                f"supported names: {supported}, or pass a FLORES-200 code directly)\n{text}"
            )

        try:
            self._tokenizer.src_lang = SOURCE_LANG
            source_tokens = self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))

            results = self._translator.translate_batch(
                [source_tokens],
                target_prefix=[[tgt_code]],
                beam_size=2,
            )

            output_tokens = results[0].hypotheses[0][1:]  # drop the target-language prefix token
            translated_ids = self._tokenizer.convert_tokens_to_ids(output_tokens)
            return self._tokenizer.decode(
                translated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
            ).strip()
        except Exception as exc:
            return f"{UNAVAILABLE_NOTICE} (translation failed: {exc})\n{text}"
