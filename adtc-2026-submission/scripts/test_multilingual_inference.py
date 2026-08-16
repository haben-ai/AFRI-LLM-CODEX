"""
Tests Qwen2.5-Coder-1.5B-Instruct's multilingual coding-tutorial capability
by driving the llama-cli binary directly (not llama-cpp-python), measuring
TTFT / generation throughput / peak RSS, and validating output structure --
including an Ethiopic-script check for the Amharic test case.

Implementation note on llama-cli behavior (verified empirically against the
actual binary in this environment, build b10375-ba360efe1): modern llama-cli
builds default to an interactive "conversation" mode that auto-applies the
model's chat template and drops into a REPL after each reply -- `-f`
(prompt-from-file) and `--no-conversation` did NOT prevent this here, and it
hung waiting for further chat turns. The combination that reliably produces
one clean response and a clean exit on this build is `-p "<prompt>"
--single-turn` with stdin closed -- this is also what the binary's own
`--single-turn` help text recommends ("will not be interactive if first turn
is predefined with --prompt"). Performance stats and the model's reply both
land on **stdout** in this mode (not the classic stderr
`llama_perf_context_print` block from older llama.cpp `main`), as a compact
`[ Prompt: X t/s | Generation: Y t/s ]` line -- this script parses that
format, with a regex fallback for the older stderr format in case a
different llama-cli build is used to run it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_REL = "model/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
PAYLOAD_DIR = PROJECT_ROOT / "test_payloads"
RESULTS_PATH = PROJECT_ROOT / "test_results.json"
RAM_BUDGET_MB = 2048.0

ASSISTANT_MARKER = "<|im_start|>assistant"

SYSTEM_PROMPT_TEMPLATE = (
    "You are a coding tutor. When asked to write and explain code, respond using "
    "EXACTLY this structure, with each header on its own line and nothing before "
    "the first header:\n\n"
    "### PYTHON CODE\n"
    "```python\n<the code>\n```\n\n"
    "### ENGLISH EXPLANATION\n<explanation in English>\n\n"
    "### {lang} EXPLANATION\n<the same explanation translated into {lang_name}>\n"
)


@dataclass
class TestCase:
    case_id: str
    description: str
    task_prompt: str
    lang_code: str    # marker used in the section header, e.g. "SWAHILI"
    lang_name: str     # human-readable name used in the instruction text
    ethiopic_check: bool = False


TEST_CASES = [
    TestCase(
        case_id="case_a_palindrome_swahili",
        description="English + Swahili explanation (palindrome check)",
        task_prompt="Write a Python function `is_palindrome(s)` that checks whether a string is a palindrome.",
        lang_code="SWAHILI",
        lang_name="Swahili",
        ethiopic_check=False,
    ),
    TestCase(
        case_id="case_b_binary_search_amharic",
        description="English + Amharic explanation (binary search algorithm)",
        task_prompt="Write a Python function `binary_search(arr, target)` that implements the binary search algorithm.",
        lang_code="AMHARIC",
        lang_name="Amharic",
        ethiopic_check=True,
    ),
]


# ---------------------------------------------------------------------- #
# Prompt construction
# ---------------------------------------------------------------------- #

def build_prompt(case: TestCase) -> str:
    system = SYSTEM_PROMPT_TEMPLATE.format(lang=case.lang_code, lang_name=case.lang_name)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{case.task_prompt}<|im_end|>\n"
        f"{ASSISTANT_MARKER}\n"
    )


def resolve_model_path(args) -> Path:
    if args.model:
        return Path(args.model)
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


# ---------------------------------------------------------------------- #
# llama-cli execution: streamed so TTFT is genuinely measured, not guessed
# ---------------------------------------------------------------------- #

_STATS_RE = re.compile(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]")
_LEGACY_PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens?.*?([\d.]+)\s*tokens per second"
)
_LEGACY_GEN_EVAL_RE = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs?.*?([\d.]+)\s*tokens per second"
)
_LEGACY_LOAD_RE = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
_PY_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_ETHIOPIC_RANGE = (0x1200, 0x137F)


def run_llama_cli(model_path: Path, prompt: str, args) -> dict:
    cmd = [
        args.llama_cli,
        "-m", str(model_path),
        "-p", prompt,
        "--single-turn",
        "-t", str(args.threads),
        "-c", str(args.ctx_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "--cache-type-k", args.cache_type_k,
        "--cache-type-v", args.cache_type_v,
        "-n", str(args.max_tokens),
        "--temp", str(args.temp),
        "--simple-io",
        "--no-display-prompt",
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return {"error": f"'{args.llama_cli}' not found. Build/install llama.cpp and ensure llama-cli "
                          f"is on PATH, or pass --llama-cli /path/to/llama-cli."}

    state = {"stdout": "", "first_token_time": None, "marker_seen": False}
    peak_rss_mb = [0.0]
    stop_rss = threading.Event()

    def _sample_rss():
        if psutil is None:
            return
        try:
            p = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            return
        while not stop_rss.is_set():
            try:
                rss = p.memory_info().rss
                for child in p.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_rss_mb[0] = max(peak_rss_mb[0], rss / (1024 * 1024))
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)

    def _read_stdout():
        # Character-mode read (portable across platforms) so the marker
        # timestamp isn't delayed by line buffering.
        while True:
            ch = proc.stdout.read(1)
            if ch == "":
                break
            state["stdout"] += ch
            if not state["marker_seen"] and ASSISTANT_MARKER in state["stdout"]:
                state["marker_seen"] = True
                state["first_token_time"] = time.perf_counter()

    rss_thread = threading.Thread(target=_sample_rss, daemon=True)
    reader_thread = threading.Thread(target=_read_stdout, daemon=True)
    t0 = time.perf_counter()
    rss_thread.start()
    reader_thread.start()

    try:
        stderr = proc.stderr.read()
        proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        stop_rss.set()
        return {"error": f"llama-cli timed out after {args.timeout}s (process killed)."}

    t_end = time.perf_counter()
    reader_thread.join(timeout=2)
    stop_rss.set()
    rss_thread.join(timeout=1)

    return {
        "returncode": proc.returncode,
        "stdout": state["stdout"],
        "stderr": stderr,
        "wall_time_s": t_end - t0,
        "first_token_at_s": (state["first_token_time"] - t0) if state["first_token_time"] else None,
        "marker_observed_in_output": state["marker_seen"],
        "peak_rss_mb": round(peak_rss_mb[0], 1),
    }


def parse_perf_stats(stdout: str, stderr: str) -> dict:
    """Modern short-form stats live on stdout; fall back to the classic
    llama_perf_context_print stderr block for older llama-cli builds."""
    m = _STATS_RE.search(stdout) or _STATS_RE.search(stderr)
    if m:
        return {"format": "modern", "prompt_tok_s": float(m.group(1)), "gen_tok_s": float(m.group(2))}

    metrics = {"format": "legacy"}
    combined = stderr + "\n" + stdout
    m = _LEGACY_PROMPT_EVAL_RE.search(combined)
    if m:
        metrics["prompt_eval_ms"] = float(m.group(1))
        metrics["prompt_tokens"] = int(m.group(2))
        metrics["prompt_tok_s"] = float(m.group(3))
    m = _LEGACY_GEN_EVAL_RE.search(combined)
    if m:
        metrics["gen_eval_ms"] = float(m.group(1))
        metrics["gen_tokens"] = int(m.group(2))
        metrics["gen_tok_s"] = float(m.group(3))
    m = _LEGACY_LOAD_RE.search(combined)
    if m:
        metrics["load_ms"] = float(m.group(1))
    return metrics if len(metrics) > 1 else {}


def extract_completion(raw_stdout: str, prompt: str) -> str:
    """Strip everything up to and including the echoed prompt (this build
    echoes it despite --no-display-prompt) and the trailing stats/banner,
    leaving just the model's actual reply."""
    text = raw_stdout
    idx = text.rfind(ASSISTANT_MARKER)
    if idx != -1:
        text = text[idx + len(ASSISTANT_MARKER):]
    elif prompt in text:
        text = text.split(prompt, 1)[1]

    stats_idx = _STATS_RE.search(text)
    if stats_idx:
        text = text[:stats_idx.start()]

    return text.strip()


# ---------------------------------------------------------------------- #
# Output validation
# ---------------------------------------------------------------------- #

def validate_output(completion: str, case: TestCase) -> dict:
    required_sections = ["### PYTHON CODE", "### ENGLISH EXPLANATION", f"### {case.lang_code} EXPLANATION"]
    sections_present = {s: s in completion for s in required_sections}

    blocks = _PY_BLOCK_RE.findall(completion)
    syntax_errors = []
    for i, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as e:
            syntax_errors.append(f"block {i}: {e}")
    python_syntax_valid = bool(blocks) and not syntax_errors

    result = {
        "sections_present": sections_present,
        "python_blocks_found": len(blocks),
        "python_syntax_valid": python_syntax_valid,
        "python_syntax_errors": syntax_errors or None,
    }

    if case.ethiopic_check:
        marker = f"### {case.lang_code} EXPLANATION"
        idx = completion.find(marker)
        section_text = completion[idx + len(marker):] if idx != -1 else ""
        ethiopic_chars = [ch for ch in section_text if _ETHIOPIC_RANGE[0] <= ord(ch) <= _ETHIOPIC_RANGE[1]]
        result["ethiopic_char_count"] = len(ethiopic_chars)
        result["ethiopic_script_detected"] = len(ethiopic_chars) > 0

    result["overall_pass"] = (
        all(sections_present.values())
        and python_syntax_valid
        and (result.get("ethiopic_script_detected", True) if case.ethiopic_check else True)
    )
    return result


# ---------------------------------------------------------------------- #
# Orchestration
# ---------------------------------------------------------------------- #

def run_case(case: TestCase, model_path: Path, args) -> dict:
    print(f"=== {case.case_id}: {case.description} ===")
    prompt = build_prompt(case)

    PAYLOAD_DIR.mkdir(exist_ok=True)
    payload_path = PAYLOAD_DIR / f"{case.case_id}.txt"
    payload_path.write_text(prompt, encoding="utf-8")

    entry = {
        "case_id": case.case_id,
        "description": case.description,
        "target_language": case.lang_code,
        "model_path": str(model_path),
        "payload_file": str(payload_path),
        "flags": {
            "threads": args.threads, "ctx_size": args.ctx_size,
            "batch_size": args.batch_size, "ubatch_size": args.ubatch_size,
            "cache_type_k": args.cache_type_k, "cache_type_v": args.cache_type_v,
            "max_tokens": args.max_tokens, "temp": args.temp,
        },
    }

    if args.dry_run:
        entry["dry_run"] = True
        print(f"  [dry-run] wrote payload to {payload_path}, skipping execution.")
        return entry

    run_info = run_llama_cli(model_path, prompt, args)
    if "error" in run_info:
        entry["success"] = False
        entry["error"] = run_info["error"]
        print(f"  FAILED: {run_info['error']}")
        return entry

    perf = parse_perf_stats(run_info["stdout"], run_info["stderr"])
    completion = extract_completion(run_info["stdout"], prompt)
    validation = validate_output(completion, case)

    ttft_s = run_info["first_token_at_s"]
    entry.update({
        "success": run_info["returncode"] == 0,
        "returncode": run_info["returncode"],
        "wall_time_s": round(run_info["wall_time_s"], 2),
        "ttft_ms": round(ttft_s * 1000, 1) if ttft_s is not None else None,
        "ttft_method": "measured (stream timestamp at end of echoed prompt)" if ttft_s is not None
                        else "unavailable (prompt-echo marker not observed in output)",
        "tokens_per_second": perf.get("gen_tok_s"),
        "prompt_tokens_per_second": perf.get("prompt_tok_s"),
        "perf_raw": perf,
        "peak_rss_mb": run_info["peak_rss_mb"],
        "rss_within_budget": run_info["peak_rss_mb"] < RAM_BUDGET_MB,
        "validation": validation,
        "completion": completion,
    })

    print(f"  TTFT: {entry['ttft_ms']} ms | tok/s: {entry['tokens_per_second']} | "
          f"peak RSS: {entry['peak_rss_mb']} MB (budget: {'OK' if entry['rss_within_budget'] else 'EXCEEDED'})")
    print(f"  validation: {'PASS' if validation['overall_pass'] else 'FAIL'} -- {validation}")
    print()
    return entry


def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual coding-tutorial capability test via llama-cli.")
    parser.add_argument("--llama-cli", type=str, default="llama-cli", help="Path to the llama-cli executable.")
    parser.add_argument("--model", type=str, default=None, help="Path to the .gguf model file.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--cache-type-k", type=str, default="q8_0")
    parser.add_argument("--cache-type-v", type=str, default="q8_0")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=300, help="Per-case subprocess timeout, seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Write payload files but don't invoke llama-cli.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args)

    if not args.dry_run and not model_path.exists():
        print(f"Model file not found at {model_path}.")
        sys.exit(1)
    if not args.dry_run and psutil is None:
        print("Warning: psutil not installed -- peak RSS will not be measured (pip install psutil).")

    results = [run_case(case, model_path, args) for case in TEST_CASES]

    RESULTS_PATH.write_text(
        json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved results to {RESULTS_PATH}")

    if args.dry_run:
        return

    overall_ok = all(r.get("success") and r.get("validation", {}).get("overall_pass") for r in results)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
