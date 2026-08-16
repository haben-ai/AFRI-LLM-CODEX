"""
Pipeline orchestrator: llama-cli (via src.engine.PedagogicalEngine) generates
code + an English explanation; src.translator.OfflineTranslator translates
the English explanation into the requested regional language using a local
NLLB-200 CTranslate2 model, independent of the primary LLM.

This intentionally reuses PedagogicalEngine/pedagogy.gbnf as-is -- they
already produce a grammar-enforced "### 1. CODE SOLUTION" /
"### 2. ENGLISH EXPLANATION" / "### 3. REGIONAL LANGUAGE EXPLANATION"
structure -- rather than introducing a second, narrower prompt/grammar pair.
Section 3 from the LLM is parsed out but deliberately NOT used in the final
output; OfflineTranslator's result replaces it. Why: earlier testing in this
project (test_multilingual_inference.py, and this same engine's own runs)
showed the 1.5B instruct model does not reliably produce genuine
Swahili/Amharic/Tigrinya text on its own -- it falls back to English, or
degenerates into repetitive nonsense -- which is exactly the gap this
translation layer exists to close. Note this means the LLM call still pays
the cost of generating a section that gets discarded; trimming
pedagogy.gbnf/prompts.py down to a two-section grammar would remove that
waste, but that's a change to files this task didn't ask for.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

# Section 3 output is Ethiopic/Latin script text (Amharic, Tigrinya, Swahili).
# Windows consoles default to a legacy codepage (cp1252) that can't encode
# Ethiopic characters and crashes on print() -- reconfigure stdout to UTF-8
# unconditionally so this works the same on a default `python main.py ...`
# invocation as it does anywhere UTF-8 is already the default. Guarded
# because reconfigure() isn't available if stdout has been replaced with
# something that doesn't support it (e.g. some test harnesses).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.engine import EngineError, PedagogicalEngine
from src.translator import OfflineTranslator

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GRAMMAR = PROJECT_ROOT / "grammars" / "pedagogy.gbnf"
DEFAULT_NLLB_DIR = PROJECT_ROOT / "model" / "nllb_ct2"

_CODE_RE = re.compile(r"### 1\. CODE SOLUTION\s*\n```python\s*\n(.*?)```", re.DOTALL)
_ENGLISH_RE = re.compile(r"### 2\. ENGLISH EXPLANATION\s*\n(.*?)\n\n### 3\.", re.DOTALL)


def parse_llm_output(raw: str) -> Tuple[str, str]:
    """Extract (code, english_explanation) from PedagogicalEngine's
    grammar-enforced output.

    Raises:
        ValueError: if either section can't be located -- that should only
            happen if grammar-constrained generation itself failed in some
            unexpected way, which is worth surfacing loudly rather than
            silently passing empty strings downstream to the translator.
    """
    code_match = _CODE_RE.search(raw)
    english_match = _ENGLISH_RE.search(raw)
    if not code_match:
        raise ValueError("Could not locate '### 1. CODE SOLUTION' section in LLM output.")
    if not english_match:
        raise ValueError("Could not locate '### 2. ENGLISH EXPLANATION' section in LLM output.")
    return code_match.group(1).strip(), english_match.group(1).strip()


def assemble_markdown(query: str, code: str, english: str, regional_text: str, target_lang: str) -> str:
    return (
        f"# {query}\n\n"
        f"## Code\n\n```python\n{code}\n```\n\n"
        f"## English Explanation\n\n{english}\n\n"
        f"## {target_lang} Explanation\n\n{regional_text}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM code generation + offline NLLB translation pipeline: code and an "
                    "English explanation come from the local Qwen GGUF model via llama-cli; "
                    "the regional-language explanation comes from a local NLLB-200 "
                    "CTranslate2 model, not the LLM.",
    )
    parser.add_argument("query", type=str, help="The coding question or task to solve and explain.")
    parser.add_argument("--model", type=str, required=True, help="Path to the primary .gguf model file.")
    parser.add_argument("--grammar", type=str, default=str(DEFAULT_GRAMMAR),
                         help=f"Path to the GBNF grammar file (default: {DEFAULT_GRAMMAR}).")
    parser.add_argument("--llama-cli", type=str, default="llama-cli",
                         help="Path to the llama-cli executable (default: 'llama-cli' on PATH).")
    parser.add_argument("--nllb-dir", type=str, default=str(DEFAULT_NLLB_DIR),
                         help=f"Path to the local NLLB CTranslate2 model directory "
                              f"(default: {DEFAULT_NLLB_DIR}).")
    parser.add_argument("--lang", type=str, default="Swahili",
                         help="Target regional language, e.g. Swahili, Amharic, Tigrinya "
                              "(default: Swahili).")
    parser.add_argument("--threads", type=int, default=4,
                         help="CPU threads for both llama-cli and the NLLB translator (default: 4).")
    parser.add_argument("--context-size", type=int, default=2048,
                         help="llama-cli context window size (default: 2048).")
    parser.add_argument("--max-tokens", type=int, default=500,
                         help="Max tokens for the LLM call (default: 500).")
    parser.add_argument("--timeout", type=int, default=120,
                         help="llama-cli subprocess timeout in seconds (default: 120).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        engine = PedagogicalEngine(
            model_path=args.model,
            grammar_path=args.grammar,
            threads=args.threads,
            context_size=args.context_size,
            llama_cli=args.llama_cli,
        )
    except EngineError as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    print("Generating code + English explanation (this can take a while on CPU)...", file=sys.stderr)
    try:
        raw_output = engine.generate(
            query=args.query,
            target_language=args.lang,  # still asked of the LLM; its section 3 is discarded below
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
    except EngineError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1

    try:
        code, english = parse_llm_output(raw_output)
    except ValueError as exc:
        print(f"Parsing failed: {exc}\n\n--- raw output ---\n{raw_output}", file=sys.stderr)
        return 1

    print(f"Translating explanation into {args.lang} via local NLLB-200...", file=sys.stderr)
    translator = OfflineTranslator(model_dir=args.nllb_dir, threads=args.threads)
    regional_text = translator.translate(english, args.lang)

    print(assemble_markdown(args.query, code, english, regional_text, args.lang))
    return 0


if __name__ == "__main__":
    sys.exit(main())
