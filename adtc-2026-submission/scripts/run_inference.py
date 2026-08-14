import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

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


def build_context_block(args, prompt_text: str) -> str:
    code_dir = args.context_dir
    context_file = args.context_file

    if not code_dir and not context_file:
        return ""

    index_target = code_dir or str(Path(context_file).parent)
    engine = SemanticaCodeEngine().build_graph_from_directory(index_target)

    symbol = args.symbol or (guess_symbol(prompt_text, engine) if engine.node_count else None)

    if symbol:
        context = engine.get_focused_context(symbol, max_depth=args.depth, max_tokens=args.max_context_tokens)
        if not context.startswith("<!--"):
            return context

    # Fallback: no resolvable symbol -- inject the raw context file, trimmed to budget.
    if context_file:
        text = Path(context_file).read_text(encoding="utf-8", errors="ignore")
        budget_chars = args.max_context_tokens * 4
        if len(text) > budget_chars:
            text = text[:budget_chars] + "\n# ... (truncated)"
        return f"# [TARGET] file `{context_file}`\n```python\n{text}\n```"

    return ""


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


def stream_llama_cpp_python(model_path: Path, grammar_path: Path, full_prompt: str, args):
    """Return a generator of text deltas from llama-cpp-python, or None if
    the binding isn't installed."""
    try:
        from llama_cpp import Llama, LlamaGrammar
    except ImportError:
        return None

    grammar = LlamaGrammar.from_file(str(grammar_path)) if grammar_path.exists() else None
    llm = Llama(model_path=str(model_path), n_ctx=args.ctx_size, n_threads=args.threads, verbose=False)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prompt = args.prompt or args.prompt_positional or DEFAULT_PROMPT
    run_local_patch(prompt, args)
