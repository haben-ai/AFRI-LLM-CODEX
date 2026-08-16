"""
Execution layer for the Dual-Language Pedagogical Engine: drives llama-cli
as a subprocess, relying on grammars/pedagogy.gbnf for structural
guarantees and defending against real-world llama-cli quirks.

Two llama-cli behaviors this engine specifically works around (verified
empirically against a real llama-cli build in this project, not assumed
from documentation):

1. Recent llama-cli builds default to an interactive "conversation" mode
   that auto-applies the model's chat template and, after producing one
   reply, waits indefinitely for further chat turns from stdin. Under
   subprocess with no attached terminal, this hangs forever rather than
   exiting. `--single-turn` (paired with `-p`) forces exactly one reply and
   a clean exit. Older llama-cli/`main` builds never had conversation mode
   and don't recognize this flag at all -- passing it there would abort
   with "unrecognized argument". `_detect_flags()` probes `llama-cli --help`
   once at construction time so this engine adds the flag only when the
   installed binary actually supports it, instead of assuming one behavior.

2. The same interactive builds can truncate their own echo of a long input
   prompt mid-token when displaying it (observed directly: a literal
   "... (truncated)" spliced into the middle of a control token like
   "<|im_start|>assistant"). That makes anchoring output-extraction on any
   literal from the tail of the *prompt* unreliable. Anchoring instead on
   the LAST occurrence of the grammar's own first required literal
   ("### 1. CODE SOLUTION") sidesteps this entirely: the system prompt's
   instructions contain that heading once, as an example; the model's real,
   grammar-enforced reply contains it once, for real; the real one is
   always the later occurrence, regardless of how the echo gets mangled.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Union

from .prompts import build_pedagogical_prompt

PRIMARY_SECTION_MARKER = "### 1. CODE SOLUTION"

_PROBE_FLAGS = ("--single-turn", "--simple-io", "--no-display-prompt")


class EngineError(Exception):
    """Base class for all PedagogicalEngine failures."""


class ModelNotFoundError(EngineError):
    """Raised when the .gguf model file doesn't exist at construction time."""


class GrammarNotFoundError(EngineError):
    """Raised when the .gbnf grammar file doesn't exist at construction time."""


class BinaryNotFoundError(EngineError):
    """Raised when the llama-cli executable can't be found/launched."""


class GenerationTimeoutError(EngineError):
    """Raised when llama-cli doesn't finish within the requested timeout."""


class GenerationFailedError(EngineError):
    """Raised when llama-cli exits with a non-zero status."""


class PedagogicalEngine:
    """Drives llama-cli to produce grammar-constrained, three-section
    (code / English / regional-language) pedagogical responses.

    Example:
        engine = PedagogicalEngine(
            model_path="model.gguf",
            grammar_path="grammars/pedagogy.gbnf",
        )
        reply = engine.generate("Write a function to reverse a linked list",
                                 target_language="Amharic")
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        grammar_path: Union[str, Path],
        threads: int = 4,
        context_size: int = 2048,
        llama_cli: str = "llama-cli",
    ) -> None:
        self.model_path = Path(model_path)
        self.grammar_path = Path(grammar_path)
        self.threads = threads
        self.context_size = context_size
        self.llama_cli = llama_cli

        if not self.model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {self.model_path}")
        if not self.grammar_path.exists():
            raise GrammarNotFoundError(f"Grammar file not found: {self.grammar_path}")

        self._supported_flags = self._detect_flags()

    # ------------------------------------------------------------------ #

    def _detect_flags(self) -> set:
        """Probe `llama-cli --help` once to see which flags this particular
        build supports, so this engine never passes a flag an older/
        different llama-cli build would reject outright. Returns an empty
        set (most conservative command line) if the binary can't be found
        or probed in time -- that failure surfaces clearly and specifically
        via BinaryNotFoundError the first time generate() actually tries to
        run it, rather than being swallowed here."""
        try:
            result = subprocess.run(
                [self.llama_cli, "--help"],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return set()

        help_text = result.stdout + result.stderr
        return {flag for flag in _PROBE_FLAGS if flag in help_text}

    def _build_command(self, prompt: str, max_tokens: int, temperature: float) -> list:
        cmd = [
            self.llama_cli,
            "-m", str(self.model_path),
            "-p", prompt,
            "-t", str(self.threads),
            "-c", str(self.context_size),
            "-n", str(max_tokens),
            "--grammar-file", str(self.grammar_path),
            "--temp", str(temperature),
        ]
        # Only add build-specific flags this installed binary actually
        # understands (see _detect_flags docstring).
        for flag in _PROBE_FLAGS:
            if flag in self._supported_flags:
                cmd.append(flag)
        return cmd

    @staticmethod
    def _extract_reply(raw_stdout: str) -> str:
        """Isolate the model's actual reply from banner/echo noise -- see
        the module docstring (point 2) for why this anchors on the
        grammar's own first required literal rather than on the prompt's
        trailing tokens."""
        idx = raw_stdout.rfind(PRIMARY_SECTION_MARKER)
        if idx == -1:
            # The grammar guarantees this literal appears in a successful
            # generation; if it's genuinely absent (e.g. llama-cli errored
            # before generating), return the raw text rather than silently
            # discarding output the caller might still want to inspect.
            return raw_stdout.strip()
        return raw_stdout[idx:].strip()

    # ------------------------------------------------------------------ #

    def generate(
        self,
        query: str,
        target_language: str = "Swahili",
        timeout: int = 60,
        max_tokens: int = 700,
        temperature: float = 0.3,
    ) -> str:
        """Generate a grammar-constrained, three-section pedagogical
        response for `query`, with section 3 requested in
        `target_language`.

        Raises:
            BinaryNotFoundError: llama-cli isn't on PATH / can't be launched.
            GenerationTimeoutError: llama-cli didn't finish within `timeout`.
            GenerationFailedError: llama-cli exited with a non-zero status.
        """
        prompt = build_pedagogical_prompt(query, target_language)
        cmd = self._build_command(prompt, max_tokens, temperature)

        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise BinaryNotFoundError(
                f"'{self.llama_cli}' not found. Build/install llama.cpp and ensure llama-cli "
                f"is on PATH, or pass llama_cli='/path/to/llama-cli' to PedagogicalEngine()."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GenerationTimeoutError(
                f"llama-cli did not finish within {timeout}s for query={query!r}."
            ) from exc

        if result.returncode != 0:
            raise GenerationFailedError(
                f"llama-cli exited with code {result.returncode}.\n"
                f"stderr:\n{result.stderr.strip()}"
            )

        return self._extract_reply(result.stdout)
