"""
ChatML prompt templates for the Dual-Language Pedagogical Engine.

Prompt construction is kept separate from execution (engine.py) and from
structural enforcement (grammars/pedagogy.gbnf). The system prompt is what
*asks* the model to follow the three-section template and write section 3
in the requested language; the grammar is what *guarantees* the structural
shape regardless of whether the model actually complies with the language
request. Neither layer can guarantee section 3 is genuinely written in
`target_language` -- that depends on the model's own capability.
"""

from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = (
    "You are a patient, precise coding tutor for African developers learning to "
    "program. For every request, respond with EXACTLY three sections, in this "
    "exact order, with nothing before the first section header:\n\n"
    "### 1. CODE SOLUTION\n"
    "```python\n"
    "<a complete, correct, executable Python solution>\n"
    "```\n\n"
    "### 2. ENGLISH EXPLANATION\n"
    "<a clear explanation of how the code works, in English>\n\n"
    "### 3. REGIONAL LANGUAGE EXPLANATION ({language})\n"
    "<the same explanation, translated into {language}, using {language}'s own "
    "script where applicable>\n"
)


def build_pedagogical_prompt(user_query: str, target_language: str) -> str:
    """Build the full ChatML-formatted prompt for a single-turn completion.

    Args:
        user_query: The coding question or task to solve and explain.
        target_language: Regional language for section 3 (e.g. "Swahili",
            "Amharic", "Tigrinya"). Only referenced in prompt text -- it
            does not alter grammars/pedagogy.gbnf, which accepts any
            language name in the section 3 header.

    Raises:
        ValueError: if either argument is empty/whitespace-only.
    """
    if not user_query or not user_query.strip():
        raise ValueError("user_query must be a non-empty string")
    if not target_language or not target_language.strip():
        raise ValueError("target_language must be a non-empty string")

    system = SYSTEM_PROMPT_TEMPLATE.format(language=target_language.strip())
    user = user_query.strip()

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
