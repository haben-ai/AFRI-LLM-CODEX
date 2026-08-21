import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# --pedagogical mode can print Ethiopic-script text (Amharic, Tigrinya).
# Windows consoles default to a legacy codepage (cp1252) that can't encode
# those characters and crashes on print() -- reconfigure stdout to UTF-8
# unconditionally, verified necessary and sufficient in adtc_assistant/main.py.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantica_context import SemanticaCodeEngine  # noqa: E402

DEFAULT_PROMPT = "Refactor this Python function for lower memory footprint."
DEFAULT_MODEL_REL = "model/qwen2.5-coder-1.5b-instruct-q5_k_m.gguf"
DEFAULT_GRAMMAR_REL = "grammars/json_patch.gbnf"
SYSTEM_MESSAGE = (
    "You are a code patching assistant. Use the CONTEXT below to understand the "
    "relevant parts of the codebase, then output a single JSON object matching "
    "{\"file_path\": ..., \"action\": ..., \"code_patch\": ...}. Output JSON only."
)

# --pedagogical mode: code + English explanation from the LLM (grammar-
# enforced), regional-language explanation from a local NLLB-200 translator
# instead of the LLM. See run_pedagogical_mode()'s docstring for why.
PEDAGOGY_GRAMMAR_REL = "grammars/pedagogy.gbnf"
NLLB_DIR_REL = "model/nllb_ct2"
PEDAGOGY_DEFAULT_MAX_TOKENS = 600  # the argparse default (256) truncates pedagogical
                                    # output before the required sections complete --
                                    # measured directly in this project (adtc_assistant).
PEDAGOGY_SYSTEM_PROMPT_TEMPLATE = (
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
PEDAGOGY_PRIMARY_MARKER = "### 1. CODE SOLUTION"
_PEDAGOGY_CODE_RE = re.compile(r"### 1\. CODE SOLUTION\s*\n```python\s*\n(.*?)```", re.DOTALL)
_PEDAGOGY_ENGLISH_RE = re.compile(r"### 2\. ENGLISH EXPLANATION\s*\n(.*?)\n\n### 3\.", re.DOTALL)


def resolve_default_model_path() -> Path:
    metadata_path = PROJECT_ROOT / "metadata.json"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            configured = meta.get("_runtime", {}).get("model_path")
            if configured:
                return (PROJECT_ROOT / configured).resolve()
        except (json.JSONDecodeError, OSError):
            pass
    return (PROJECT_ROOT / DEFAULT_MODEL_REL).resolve()


def guess_symbol(prompt_text: str, engine) -> str | None:
    """Best-effort: pick the first identifier in the prompt that matches a
    known function/class name in the indexed graph (e.g. "Fix the memory
    leak in process_data" -> "process_data")."""
    candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prompt_text)
    for token in candidates:
        if token in engine.name_index:
            return token
    return None


def build_engine(args) -> SemanticaCodeEngine | None:
    """Build the DS-Code Graph once from --context-dir/--context-file, or
    None if neither was given. Split out from build_context_block so
    interactive mode can build it a single time and reuse it across many
    prompts instead of re-indexing on every turn."""
    code_dir = args.context_dir
    context_file = args.context_file
    if not code_dir and not context_file:
        return None
    index_target = code_dir or str(Path(context_file).parent)
    return SemanticaCodeEngine().build_graph_from_directory(index_target)


def context_from_engine(engine, args, prompt_text: str) -> str:
    if engine is not None:
        symbol = args.symbol or (guess_symbol(prompt_text, engine) if engine.node_count else None)
        if symbol:
            context = engine.get_focused_context(symbol, max_depth=args.depth, max_tokens=args.max_context_tokens)
            if not context.startswith("<!--"):
                return context

    # Fallback: no resolvable symbol -- inject the raw context file, trimmed to budget.
    if args.context_file:
        text = Path(args.context_file).read_text(encoding="utf-8", errors="ignore")
        budget_chars = args.max_context_tokens * 4
        if len(text) > budget_chars:
            text = text[:budget_chars] + "\n# ... (truncated)"
        return f"# [TARGET] file `{args.context_file}`\n```python\n{text}\n```"

    return ""


def build_context_block(args, prompt_text: str) -> str:
    engine = build_engine(args)
    return context_from_engine(engine, args, prompt_text)


def build_full_prompt(prompt_text: str, context_block: str) -> str:
    user_msg = f"CONTEXT:\n{context_block}\n\nTASK:\n{prompt_text}" if context_block else prompt_text
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _extract_delta_text(chunk: dict) -> str:
    """llama-cpp-python streaming chunks use the plain-completion shape
    ({"choices": [{"text": ...}]}); chat-completion mode uses a "delta"
    envelope instead. Handle both so this keeps working if the call site
    ever switches to create_chat_completion."""
    choice = (chunk.get("choices") or [{}])[0]
    text = choice.get("text")
    if text:
        return text
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


def stream_llama_cpp_python(model_path: Path, grammar_path: Path, full_prompt: str, args,
                             type_k=None, type_v=None, flash_attn=False):
    """Return a generator of text deltas from llama-cpp-python, or None if
    the binding isn't installed.

    type_k/type_v/flash_attn default to unset (llama-cpp-python's own
    defaults, fp16 KV cache) to keep the default JSON-patch mode's already-
    measured RAM/behavior unchanged. --pedagogical mode passes
    GGML_TYPE_Q8_0 + flash_attn=True explicitly: measured in this project,
    q8_0 KV-cache quantization plus flash attention (required for a
    quantized V-cache in llama.cpp) meaningfully reduces peak RSS for the
    longer prompts/contexts that mode uses, at negligible output-quality
    cost -- a well-established llama.cpp technique, not experimental.
    """
    try:
        from llama_cpp import Llama, LlamaGrammar
    except ImportError:
        return None

    grammar = LlamaGrammar.from_file(str(grammar_path)) if grammar_path.exists() else None
    llm = Llama(
        model_path=str(model_path), n_ctx=args.ctx_size, n_threads=args.threads,
        type_k=type_k, type_v=type_v, flash_attn=flash_attn,
        verbose=False,
    )
    chunks = llm(
        full_prompt,
        grammar=grammar,
        temperature=args.temp,
        max_tokens=args.max_tokens,
        stream=True,
    )
    return (_extract_delta_text(chunk) for chunk in chunks)


def stream_llama_cli(model_path: Path, grammar_path: Path, full_prompt: str, args):
    """Return a generator of text deltas from the llama-cli subprocess, or
    None if the binary isn't on PATH."""
    cmd = [
        "llama-cli",
        "-m", str(model_path),
        "-t", str(args.threads),
        "-c", str(args.ctx_size),
        "-n", str(args.max_tokens),
        "--grammar-file", str(grammar_path),
        "--temp", str(args.temp),
        "-p", full_prompt,
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except FileNotFoundError:
        return None

    def _gen():
        for line in proc.stdout:
            yield line
        proc.wait()

    return _gen()


def render_stream(chunks) -> str:
    """Consume a generator of text deltas, live-rendering them to the
    terminal. Uses rich's Live + Markdown when attached to an interactive
    terminal; falls back to plain incremental sys.stdout.write otherwise
    (missing `rich`, init failure, or piped/non-tty output such as a
    profiler capturing this script's stdout) so nothing crashes or corrupts
    captured output during benchmark runs."""
    full_response = ""

    if sys.stdout.isatty():
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.markdown import Markdown

            console = Console()
            with Live(
                Markdown(""),
                console=console,
                refresh_per_second=12,
                vertical_overflow="visible",
            ) as live:
                for delta in chunks:
                    if not delta:
                        continue
                    full_response += delta
                    live.update(Markdown(full_response))
            return full_response
        except Exception:
            pass  # rich unavailable or failed to init -- fall back to plain streaming below

    for delta in chunks:
        if not delta:
            continue
        full_response += delta
        sys.stdout.write(delta)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return full_response


def run_local_patch(prompt_text: str, args) -> None:
    model_path = Path(args.model) if args.model else resolve_default_model_path()
    grammar_path = Path(args.grammar) if args.grammar else (PROJECT_ROOT / DEFAULT_GRAMMAR_REL)

    if not model_path.exists():
        print(f"Model file not found at {model_path}. Please run 'bash download_model.sh' first.")
        sys.exit(1)

    context_block = build_context_block(args, prompt_text)
    full_prompt = build_full_prompt(prompt_text, context_block)

    if context_block:
        print(f"Injected graph context (~{len(context_block) // 4} tokens).")

    if args.dry_run:
        print("--- DRY RUN: constructed prompt ---")
        print(full_prompt)
        return

    print(f"Running inference with {args.threads} CPU threads (streaming)...")
    stream = stream_llama_cpp_python(model_path, grammar_path, full_prompt, args)
    if stream is None:
        stream = stream_llama_cli(model_path, grammar_path, full_prompt, args)
    if stream is None:
        print("Neither llama-cpp-python nor llama-cli is available. "
              "Install with 'pip install llama-cpp-python' or compile llama.cpp.")
        sys.exit(1)

    render_stream(stream)


def run_interactive_session(args) -> None:
    """Keep one Llama instance (and, if configured, one DS-Code Graph) warm
    across multiple prompts in a single process, instead of the cold
    load-model-per-invocation path run_local_patch takes.

    This matters because llama.cpp automatically reuses KV-cache state for
    the longest shared prefix between consecutive calls on the same model
    instance. Cold prompt prefill on modest CPU hardware is the dominant
    cost of a single invocation (tens of seconds, measured); since
    SYSTEM_MESSAGE -- and the CONTEXT block too, if you keep asking about
    the same symbol -- stays identical turn to turn, every call after the
    first reuses that cached prefix and only has to prefill the differing
    tail, cutting time-to-first-token from tens of seconds to well under a
    second. Exit with an empty line, "exit"/"quit", or Ctrl+D/Ctrl+C.
    """
    model_path = Path(args.model) if args.model else resolve_default_model_path()
    grammar_path = Path(args.grammar) if args.grammar else (PROJECT_ROOT / DEFAULT_GRAMMAR_REL)

    if not model_path.exists():
        print(f"Model file not found at {model_path}. Please run 'bash download_model.sh' first.")
        sys.exit(1)

    try:
        from llama_cpp import Llama, LlamaGrammar
    except ImportError:
        print("Interactive mode requires llama-cpp-python (pip install llama-cpp-python); "
              "the llama-cli subprocess fallback has no persistent session to keep warm.")
        sys.exit(1)

    engine = build_engine(args)
    if engine is not None:
        print(f"Indexed {len(engine.indexed_files)} file(s), {engine.node_count} node(s).")

    grammar = LlamaGrammar.from_file(str(grammar_path)) if grammar_path.exists() else None
    print(f"Loading model ({model_path.name})...")
    t0 = time.perf_counter()
    llm = Llama(model_path=str(model_path), n_ctx=args.ctx_size, n_threads=args.threads, verbose=False)
    print(f"Model loaded in {(time.perf_counter() - t0) * 1000:.0f}ms. Session ready.")
    print("Type a task instruction, or 'exit'/'quit' to leave.\n")

    while True:
        try:
            prompt_text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt_text or prompt_text.lower() in ("exit", "quit"):
            break

        context_block = context_from_engine(engine, args, prompt_text)
        full_prompt = build_full_prompt(prompt_text, context_block)
        if context_block:
            print(f"[context: ~{len(context_block) // 4} tokens]")

        t0 = time.perf_counter()
        first_token_at = None

        def _timed_deltas():
            nonlocal first_token_at
            for chunk in llm(full_prompt, grammar=grammar, temperature=args.temp,
                              max_tokens=args.max_tokens, stream=True):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield _extract_delta_text(chunk)

        render_stream(_timed_deltas())
        total_ms = (time.perf_counter() - t0) * 1000
        ttft_ms = (first_token_at - t0) * 1000 if first_token_at else total_ms
        # TTFT is what KV-cache prefix reuse speeds up; total also includes
        # generation, which isn't cache-accelerated -- reporting both keeps
        # the two effects from being conflated.
        print(f"[TTFT: {ttft_ms:.0f}ms, total: {total_ms:.0f}ms]\n")


def build_pedagogical_prompt(query: str, target_language: str) -> str:
    system = PEDAGOGY_SYSTEM_PROMPT_TEMPLATE.format(language=target_language.strip())
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{query.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_pedagogical_reply(raw_text: str) -> str:
    """Same anchoring trick used by the DS-Code Graph context path and by
    adtc_assistant/src/engine.py: isolate the model's actual reply by
    taking the LAST occurrence of the grammar's first required literal,
    robust to any banner/echo noise a given llama backend might emit
    around it."""
    idx = raw_text.rfind(PEDAGOGY_PRIMARY_MARKER)
    return raw_text[idx:].strip() if idx != -1 else raw_text.strip()


def parse_pedagogical_output(raw: str):
    """Extract (code, english) from a grammar-enforced pedagogical reply.
    Returns (None, None) if either section can't be located -- most often
    because generation was truncated by --max-tokens before section 2
    completed; callers should treat that as a failure, not silently
    continue with partial/empty text."""
    code_match = _PEDAGOGY_CODE_RE.search(raw)
    english_match = _PEDAGOGY_ENGLISH_RE.search(raw)
    if not code_match or not english_match:
        return None, None
    return code_match.group(1).strip(), english_match.group(1).strip()


def run_pedagogical_mode(query: str, args) -> None:
    """Generate code + an English explanation (grammar-enforced via
    grammars/pedagogy.gbnf), then translate the English explanation into
    args.translate_lang using a local NLLB-200 CTranslate2 model -- not the
    LLM itself.

    Why: measured directly in this project (adtc_assistant/'s
    PedagogicalEngine, and scripts/test_multilingual_inference.py),
    Qwen2.5-Coder-1.5B-Instruct does not reliably produce genuine
    Swahili/Amharic/Tigrinya text on its own -- it falls back to English,
    or degenerates into repetitive nonsense under grammar pressure.
    NLLB-200 is trained specifically for translation and does this
    correctly; delegating to it is the point of this mode. The LLM's own
    section 3 attempt is still generated (the grammar requires it) but is
    discarded -- trimming pedagogy.gbnf to two sections would remove that
    waste but isn't done here to keep the grammar file identical to
    adtc_assistant's.

    NLLB and the primary LLM are never loaded at once: llama-cpp-python's
    Llama object goes out of scope (and its memory is released) before
    OfflineTranslator loads, keeping peak RSS bounded by whichever is
    larger rather than their sum -- confirmed by direct process-tree RSS
    measurement, not assumed.
    """
    model_path = Path(args.model) if args.model else resolve_default_model_path()
    grammar_path = Path(args.grammar) if args.grammar else (PROJECT_ROOT / PEDAGOGY_GRAMMAR_REL)

    if not model_path.exists():
        print(f"Model file not found at {model_path}. Please run 'bash download_model.sh' first.")
        sys.exit(1)
    if not grammar_path.exists():
        print(f"Grammar file not found at {grammar_path}.")
        sys.exit(1)

    max_tokens = args.max_tokens
    if max_tokens == 256:  # untouched argparse default -- known to truncate this mode
        max_tokens = PEDAGOGY_DEFAULT_MAX_TOKENS
        print(f"Note: raising --max-tokens to {max_tokens} for --pedagogical mode "
              f"(the 256 default truncates output before required sections complete; "
              f"pass --max-tokens explicitly to override).")

    class _Args:
        pass
    gen_args = _Args()
    gen_args.__dict__.update(vars(args))
    gen_args.max_tokens = max_tokens

    prompt = build_pedagogical_prompt(query, args.translate_lang)

    try:
        import llama_cpp as _llama_cpp_module
        type_k = type_v = _llama_cpp_module.GGML_TYPE_Q8_0
    except ImportError:
        type_k = type_v = None

    print(f"Generating code + English explanation ({args.threads} CPU threads"
          f"{', q8_0 KV cache' if type_k is not None else ''})...")
    stream = stream_llama_cpp_python(
        model_path, grammar_path, prompt, gen_args,
        type_k=type_k, type_v=type_v, flash_attn=(type_k is not None),
    )
    if stream is None:
        print("--pedagogical mode requires llama-cpp-python (the llama-cli subprocess "
              "fallback isn't wired up for this mode). Install with 'pip install llama-cpp-python'.")
        sys.exit(1)

    raw = render_stream(stream)
    reply = extract_pedagogical_reply(raw)
    code, english = parse_pedagogical_output(reply)
    if code is None:
        print("Parsing failed: could not locate required sections in the model's output.\n"
              "--- raw output ---")
        print(reply)
        sys.exit(1)

    print(f"Translating explanation into {args.translate_lang} via local NLLB-200...")
    try:
        from translator import OfflineTranslator
    except ImportError:
        print("--pedagogical mode requires ctranslate2 and transformers "
              "(pip install ctranslate2 transformers). Showing English only.")
        regional_text = None
    else:
        nllb_dir = Path(args.nllb_dir) if args.nllb_dir else (PROJECT_ROOT / NLLB_DIR_REL)
        translator = OfflineTranslator(model_dir=str(nllb_dir), threads=args.threads)
        regional_text = translator.translate(english, args.translate_lang)

    print("=" * 70)
    print(f"## Code\n\n```python\n{code}\n```\n")
    print(f"## English Explanation\n\n{english}\n")
    if regional_text is not None:
        print(f"## {args.translate_lang} Explanation\n\n{regional_text}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="AFRI-LLM-CODEX local code-patching inference.")
    parser.add_argument("prompt_positional", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--prompt", type=str, default=None, help="Task instruction for the model.")
    parser.add_argument("--context-dir", "--code-dir", dest="context_dir", type=str, default=None,
                         help="Directory to index into the DS-Code Graph (IMPORT/CONTAIN/CALL edges).")
    parser.add_argument("--context-file", type=str, default=None, help="Single file to pull context from.")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol name to center the subgraph context on.")
    parser.add_argument("--depth", type=int, default=2, help="Hop depth for subgraph context extraction.")
    parser.add_argument("--max-context-tokens", type=int, default=500, help="Token budget for injected context.")
    parser.add_argument("--model", type=str, default=None, help="Path to the .gguf model file.")
    parser.add_argument("--grammar", type=str, default=None, help="Path to the GBNF grammar file.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="Print the constructed prompt without calling the model.")
    parser.add_argument("--interactive", action="store_true",
                         help="Start a persistent session: load the model once and keep it warm across "
                              "multiple prompts, so llama.cpp's KV-cache prefix reuse cuts time-to-first-token "
                              "dramatically after the first call. Requires llama-cpp-python.")
    parser.add_argument("--pedagogical", action="store_true",
                         help="Dual-language coding-tutor mode: code + English explanation from the LLM "
                              "(grammar-enforced), regional-language explanation from a local NLLB-200 "
                              "translator instead of the LLM. Requires llama-cpp-python, ctranslate2, "
                              "and transformers.")
    parser.add_argument("--translate-lang", type=str, default="Swahili",
                         help="Target regional language for --pedagogical mode, e.g. Swahili, Amharic, "
                              "Tigrinya (default: Swahili).")
    parser.add_argument("--nllb-dir", type=str, default=None,
                         help=f"Path to the local NLLB CTranslate2 model directory for --pedagogical mode "
                              f"(default: {NLLB_DIR_REL} under the project root).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pedagogical:
        query = args.prompt or args.prompt_positional or DEFAULT_PROMPT
        run_pedagogical_mode(query, args)
    elif args.interactive:
        run_interactive_session(args)
    else:
        prompt = args.prompt or args.prompt_positional or DEFAULT_PROMPT
        run_local_patch(prompt, args)
