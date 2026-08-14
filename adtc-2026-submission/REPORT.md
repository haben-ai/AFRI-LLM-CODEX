# ADTC 2026 Technical Report: On-Device Code Patching Assistant

## 1. Problem Statement
Developers in low-bandwidth or offline environments across Africa require fast, private, on-device code refactoring tools that run reliably on budget hardware (4 vCPUs, 8 GB RAM).

## 2. Technical Architecture & Design Decisions
- **Base Model:** `Qwen2.5-Coder-1.5B-Instruct`
- **Quantization:** `GGUF Q5_K_M` (~1.2 GB file size, fits well under the 8 GB RAM threshold).
- **Execution Engine:** Native `llama.cpp` using strict thread affinity (`-t 4`) and context capping (`-c 2048`).
- **Zero-Chatter Enforcement:** Custom GBNF grammar (`json_patch.gbnf`) forces direct JSON diff patch outputs, reducing generated output tokens by ~80% and mitigating CPU latency.

## 3. DS-Code Graph Context Engine (Version 2)

Version 1 passed whole files to the model, wasting context window on
boilerplate and irrelevant functions. Version 2 introduces
`SemanticaCodeEngine` (alias `DSCodeGraphEngine`, `src/semantica_context.py`),
a local dependency graph whose build-graph -> focused-subgraph workflow
follows the **DS-Code Graph (Dependency & Semantic Code Graph)** strategy
described in CodeRAG (Ugare et al., 2024/2025): represent a codebase as
nodes (modules, classes, functions) connected by `IMPORT`, `CONTAIN`, and
`CALL` edges, then retrieve only the small sub-graph relevant to the symbol
being patched instead of whole files. (The module keeps its original name
from an earlier design pass shaped after `semantica-agi/semantica`'s
`ContextGraph` workflow; the graph model itself now follows CodeRAG's edge
taxonomy.)

This is a from-scratch, zero-dependency implementation -- only `ast`, `re`,
and the standard library, no `torch`/`transformers`/`sentence-transformers`/
vector databases:

- **`build_graph_from_directory(repo_path)`** walks a file or directory and
  parses every `.py`/`.c`/`.h` file it finds. Python files get a real AST
  pass (`ast` module); C/H files -- which the stdlib has no parser for --
  get a lightweight regex-based scanner that extracts `#include` directives,
  function signatures via brace-matched body scanning, and naive call sites.
  This matters directly: one of this project's own two test prompts
  (`tp_001`) is a C memory-leak fix, so C-aware context extraction isn't
  academic here.
- **`get_focused_context(symbol_name, max_depth=2)`** traverses 1-2 hops
  along `CALL` and `IMPORT` edges from the target symbol. The target itself
  is returned with its full body (the model needs to see what it's patching);
  every dependency node is trimmed to just its signature + docstring/comment
  -- never its full implementation -- which is what keeps the sub-graph in
  the ~300-500 token range regardless of how large a dependency function is.
- `scripts/run_inference.py` auto-detects the target symbol from the prompt
  text (e.g. "Fix memory leak in `process_data`" -> `process_data`), builds
  the graph via `--context-dir`/`--context-file`, injects the focused context
  into the system prompt, and applies `grammars/json_patch.gbnf` during
  `llama-cpp-python` sampling so generation is constrained to a structured
  JSON patch.

## 4. Benchmarks & Performance Profile

Measured on the reference participant laptop (Intel Kaby Lake-class CPU, 7.9
GB RAM, no GPU, Windows) via `adtc-profiler run --mode participant` unless
noted otherwise:

- **Peak RAM Usage:** 1253.5 MB RSS end-to-end (`submission.json`, profiler-measured), and 1276.2 MB RSS measured manually for a full context-build + grammar-constrained generation pass -- both comfortably under the 1.5 GB target and the 2 GB hard ceiling. The context engine itself accounts for a few MB of pure-Python graph structures; the model dominates the footprint.
- **Context engine latency:** graph construction over a small target directory (2 files, ~13 nodes) takes ~4-8ms; `get_focused_context` (the actual sub-graph traversal + Markdown formatting) takes **under 0.1ms** -- both well inside the 50ms budget. Graph construction time scales with total source size (~30-50ms for a couple of larger, ~400-line files), dominated by one-time `ast.parse` per file rather than the traversal/extraction step itself.
- **Throughput:** 8.83 tokens/sec generation on 4 CPU threads (`submission.json`, measured -- an earlier draft of this report cited an unverified ~35 tok/s; corrected here to the profiler's actual number).
- **First-token latency:** ~35.6s, dominated by CPU prompt-prefill of the profiler's 512-token benchmark prompt, not by the context engine (which adds low-single-digit milliseconds) or by generation itself.
- **Offline Compliance:** 100% offline runtime execution -- no network calls for parsing, graph construction, or inference.
