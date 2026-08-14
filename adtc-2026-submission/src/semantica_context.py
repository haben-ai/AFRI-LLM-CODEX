"""
DS-Code Graph context engine for AFRI-LLM-CODEX.

`SemanticaCodeEngine` (alias: `DSCodeGraphEngine`) is a zero-dependency,
100% offline implementation of the Dependency & Semantic Code Graph (DS-Code
Graph) strategy described in CodeRAG (Ugare et al., 2024/2025): build a
lightweight graph of code entities connected by IMPORT / CONTAIN / CALL
edges, then pull a small, high-density sub-graph around a target symbol
instead of feeding the model whole files.

Design constraints (ADTC 2026, budget hardware, offline):
    - Only the Python standard library is used (`ast`, `re`, `pathlib`,
      `dataclasses`, `collections`) -- no torch, transformers,
      sentence-transformers, or vector databases.
    - Python files are parsed with a real AST (`ast` module). C/H files are
      parsed with a lightweight regex-based heuristic scanner (the stdlib
      has no C parser) that extracts #include directives, function
      signatures, and naive call sites via brace-matched body scanning --
      good enough for the ~50-500 line source-patching prompts this
      pipeline targets, not a full C front end.
    - Dependency nodes returned by `get_focused_context` carry only their
      signature + docstring (not their full body), keeping the injected
      context in the ~300-500 token range regardless of how large the
      dependency function actually is.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

DEFAULT_EXCLUDE_DIRS = {
    ".git", "__pycache__", "venv", ".venv", "env", ".env",
    "node_modules", "site-packages", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode",
}
SOURCE_SUFFIXES = {".py", ".c", ".h"}
CHARS_PER_TOKEN = 4  # rough heuristic, avoids pulling in a tokenizer dependency

# DS-Code Graph edge relations (CodeRAG naming)
IMPORT = "IMPORT"
CONTAIN = "CONTAIN"
CALL = "CALL"


@dataclass
class CodeNode:
    id: str
    name: str
    kind: str  # "module" | "class" | "function" | "method" | "import"
    file: str
    lineno: int
    end_lineno: int
    language: str  # "python" | "c"
    signature: str = ""
    doc: str = ""
    source: str = ""


@dataclass
class SemanticaCodeEngine:
    """DS-Code Graph engine: build a local IMPORT/CONTAIN/CALL graph from a
    directory of Python/C source, then extract focused sub-graph context.

        engine = SemanticaCodeEngine()
        engine.build_graph_from_directory("path/to/repo_or_file")
        context = engine.get_focused_context("some_function")
    """

    max_files: int = 500
    nodes: Dict[str, CodeNode] = field(default_factory=dict)
    out_edges: Dict[str, List[Tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))
    in_edges: Dict[str, List[Tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))
    name_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    indexed_files: List[str] = field(default_factory=list)
    _pending_calls: List[Tuple[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def build_graph_from_directory(self, repo_path: str) -> "SemanticaCodeEngine":
        """Walk `repo_path` (a file or directory), parsing every .py/.c/.h
        file into the IMPORT/CONTAIN/CALL graph."""
        path = Path(repo_path)
        if path.is_file():
            files = [path] if path.suffix in SOURCE_SUFFIXES else []
        elif path.is_dir():
            files = [
                f for f in sorted(path.rglob("*"))
                if f.suffix in SOURCE_SUFFIXES and not any(part in DEFAULT_EXCLUDE_DIRS for part in f.parts)
            ][: self.max_files]
        else:
            files = []

        for f in files:
            if f.suffix == ".py":
                self._index_python_file(f)
            else:
                self._index_c_file(f)
            self.indexed_files.append(str(f))

        self._resolve_pending_calls()
        return self

    def _resolve_pending_calls(self) -> None:
        for caller_id, callee_name in self._pending_calls:
            for target_id in self.name_index.get(callee_name, []):
                if target_id != caller_id:
                    self._add_edge(caller_id, target_id, CALL)
        self._pending_calls.clear()

    # -- internal mutators ----------------------------------------------- #

    def _add_node(self, node: CodeNode) -> None:
        if node.id in self.nodes:
            return
        self.nodes[node.id] = node
        self.name_index[node.name].append(node.id)

    def _add_edge(self, src_id: str, dst_id: str, relation: str) -> None:
        self.out_edges[src_id].append((relation, dst_id))
        self.in_edges[dst_id].append((relation, src_id))

    def _add_import(self, scope_id: str, module_name: str) -> None:
        import_id = f"import::{module_name}"
        if import_id not in self.nodes:
            self._add_node(CodeNode(import_id, module_name, "import", "<external>", 0, 0, "external"))
        self._add_edge(scope_id, import_id, IMPORT)

    def _add_call(self, caller_id: str, callee_name: str) -> None:
        self._pending_calls.append((caller_id, callee_name))

    # -- Python (ast) ------------------------------------------------------ #

    def _index_python_file(self, path: Path) -> None:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError, ValueError):
            return
        _PyIndexer(self, str(path), source).visit(tree)

    # -- C / H (regex heuristic scanner) ----------------------------------- #

    _C_INCLUDE_RE = re.compile(r'#\s*include\s*[<"]([^>"]+)[>"]')
    _C_FUNC_RE = re.compile(
        r'^[ \t]*(?:(?:static|inline|extern|const)\s+)*'
        r'([A-Za-z_][\w ]*?[\w\*])\s*'          # return type (handles "char *name" and "char* name")
        r'([A-Za-z_]\w*)\s*'                     # function name
        r'\(([^;{}]*)\)\s*'                      # parameters
        r'\{',                                    # a definition body, not a prototype
        re.MULTILINE,
    )
    _C_CALL_RE = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
    _C_KEYWORDS = {
        "if", "for", "while", "switch", "return", "sizeof", "else", "do",
        "break", "continue", "goto", "typedef", "struct", "union", "enum",
    }

    def _index_c_file(self, path: Path) -> None:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return

        file_key = str(path)
        module_id = f"{file_key}::<module>"
        self._add_node(CodeNode(module_id, path.name, "module", file_key, 1, 1, "c"))

        for m in self._C_INCLUDE_RE.finditer(source):
            self._add_import(module_id, m.group(1))

        for m in self._C_FUNC_RE.finditer(source):
            return_type, name, params = m.group(1).strip(), m.group(2), m.group(3).strip()
            brace_start = m.end() - 1  # index of the opening '{'
            body_end = self._match_brace(source, brace_start)
            if body_end is None:
                continue

            lineno = source.count("\n", 0, m.start()) + 1
            end_lineno = source.count("\n", 0, body_end) + 1
            signature = f"{return_type} {name}({params})"
            doc = self._preceding_c_comment(source, m.start())
            body = source[m.start():body_end + 1]

            node_id = f"{file_key}::{name}"
            self._add_node(CodeNode(
                node_id, name, "function", file_key, lineno, end_lineno, "c",
                signature=signature, doc=doc, source=body,
            ))
            self._add_edge(module_id, node_id, CONTAIN)

            for call_match in self._C_CALL_RE.finditer(source, brace_start, body_end):
                callee = call_match.group(1)
                if callee != name and callee not in self._C_KEYWORDS:
                    self._add_call(node_id, callee)

    @staticmethod
    def _match_brace(source: str, open_index: int) -> int | None:
        """Given the index of an opening '{', return the index of its
        matching closing '}' (naive counter, ignores braces in string/char
        literals -- acceptable for a lightweight heuristic scanner)."""
        depth = 0
        for i in range(open_index, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
        return None

    @staticmethod
    def _preceding_c_comment(source: str, decl_start: int) -> str:
        """Best-effort: pull a `/* ... */` or contiguous `//` block sitting
        directly above a function signature, C's equivalent of a docstring."""
        line_start = source.rfind("\n", 0, decl_start) + 1
        lines_above = source[:line_start].splitlines()
        collected: List[str] = []
        for line in reversed(lines_above):
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith("//"):
                collected.insert(0, stripped.lstrip("/").strip())
                continue
            if stripped.endswith("*/") or stripped.startswith("/*") or stripped.startswith("*"):
                collected.insert(0, stripped.strip("/*").strip())
                if stripped.startswith("/*"):
                    break
                continue
            break
        return " ".join(c for c in collected if c)

    # ------------------------------------------------------------------ #
    # Sub-graph context extraction
    # ------------------------------------------------------------------ #

    def _resolve_symbol(self, query: str) -> List[str]:
        if query in self.nodes:
            return [query]
        exact = self.name_index.get(query)
        if exact:
            return list(exact)
        q = query.lower()
        matches = [nid for name, ids in self.name_index.items() if q in name.lower() for nid in ids]
        return matches[:3]

    def get_focused_context(self, symbol_name: str, max_depth: int = 2, max_tokens: int = 400) -> str:
        """Locate `symbol_name` in the graph and traverse 1-2 hops along
        CALL and IMPORT edges to pull its dependency neighborhood. The
        target itself is returned with its full source; every dependency
        node is trimmed to just its signature + docstring so the total
        stays within `max_tokens`."""
        target_ids = self._resolve_symbol(symbol_name)
        if not target_ids:
            return f"<!-- semantica_context: no symbol matching '{symbol_name}' found in indexed codebase -->"

        allowed_relations = {CALL, IMPORT}
        visited: Set[str] = set(target_ids)
        order: List[Tuple[str, int]] = [(tid, 0) for tid in target_ids]
        frontier = list(target_ids)
        for hop in range(1, max(max_depth, 0) + 1):
            next_frontier: List[str] = []
            for nid in frontier:
                neighbors = [
                    (r, o) for r, o in self.out_edges.get(nid, []) if r in allowed_relations
                ] + [
                    (r, o) for r, o in self.in_edges.get(nid, []) if r in allowed_relations
                ]
                for relation, other in neighbors:
                    if other not in visited and other in self.nodes:
                        visited.add(other)
                        order.append((other, hop))
                        next_frontier.append(other)
            frontier = next_frontier
            if not frontier:
                break

        budget_chars = max_tokens * CHARS_PER_TOKEN
        sections: List[str] = []
        used_chars = 0
        imports: Set[str] = set()
        for nid, hop in order:
            node = self.nodes[nid]
            if node.kind == "import":
                imports.add(node.name)
                continue
            block = self._format_node(node, primary=(hop == 0))
            if hop != 0 and used_chars + len(block) > budget_chars:
                continue
            sections.append(block)
            used_chars += len(block)

        header = f"**Relevant imports:** {', '.join(sorted(imports))}\n\n" if imports else ""
        return header + "\n\n".join(sections)

    @staticmethod
    def _format_node(node: CodeNode, primary: bool) -> str:
        label = "TARGET" if primary else "RELATED"
        loc = f"{node.file}:{node.lineno}"
        doc_line = f"\n# {node.doc.strip().splitlines()[0]}" if node.doc else ""
        if primary:
            body = (node.source or node.signature).strip()
            if len(body) > 1500:
                body = body[:1500] + "\n# ... (truncated)"
        else:
            # Dependencies: signature + docstring only, never the full body --
            # this is what keeps the sub-graph in the ~300-500 token range.
            body = node.signature.strip()
        lang = "python" if node.language == "python" else "c"
        return f"# [{label}] {node.kind} `{node.name}` -- {loc}{doc_line}\n```{lang}\n{body}\n```"

    @property
    def node_count(self) -> int:
        return len(self.nodes)


# Alias matching the CodeRAG DS-Code Graph terminology used in the spec.
DSCodeGraphEngine = SemanticaCodeEngine


def _fast_source_segment(lines: List[str], node: ast.AST) -> str:
    """Equivalent to ast.get_source_segment(source, node), but takes a
    pre-split lines array instead of re-splitting the whole file on every
    call. ast.get_source_segment() internally calls source.splitlines()
    from scratch each time it's invoked -- calling it once per node turns
    graph construction into O(nodes x file_size) and dominates indexing
    time on anything past a handful of functions; splitting once per file
    and reusing that array is what keeps indexing fast."""
    if getattr(node, "end_lineno", None) is None:
        return ""
    start_line, end_line = node.lineno - 1, node.end_lineno - 1
    start_col, end_col = node.col_offset, node.end_col_offset
    if start_line == end_line:
        return lines[start_line][start_col:end_col]
    parts = [lines[start_line][start_col:]]
    parts.extend(lines[start_line + 1:end_line])
    parts.append(lines[end_line][:end_col])
    return "\n".join(parts)


class _PyIndexer(ast.NodeVisitor):
    """Single-pass AST walk populating nodes + CONTAIN/IMPORT edges, and
    queuing CALL edges for cross-file resolution once all files are
    indexed."""

    def __init__(self, engine: SemanticaCodeEngine, filepath: str, source: str):
        self.engine = engine
        self.filepath = filepath
        self.source = source
        self.lines = source.splitlines()
        self.name_path: List[str] = []
        self.module_id = f"{filepath}::<module>"
        self.func_id_stack: List[str] = []
        engine._add_node(CodeNode(self.module_id, Path(filepath).name, "module", filepath, 1, 1, "python"))

    def _current_id(self) -> str:
        if not self.name_path:
            return self.module_id
        return f"{self.filepath}::{'.'.join(self.name_path)}"

    def _child_id(self, name: str) -> str:
        return f"{self.filepath}::{'.'.join(self.name_path + [name])}"

    def _current_caller(self) -> str:
        return self.func_id_stack[-1] if self.func_id_stack else self.module_id

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.engine._add_import(self._current_caller(), alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            full = f"{mod}.{alias.name}" if mod else alias.name
            self.engine._add_import(self._current_caller(), full)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        node_id = self._child_id(node.name)
        parent_id = self._current_id()
        try:
            bases = ", ".join(ast.unparse(b) for b in node.bases)
        except Exception:
            bases = ""
        signature = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
        src = _fast_source_segment(self.lines, node)
        self.engine._add_node(CodeNode(
            node_id, node.name, "class", self.filepath,
            node.lineno, getattr(node, "end_lineno", node.lineno), "python",
            signature=signature, doc=ast.get_docstring(node) or "", source=src,
        ))
        self.engine._add_edge(parent_id, node_id, CONTAIN)
        self.name_path.append(node.name)
        self.generic_visit(node)
        self.name_path.pop()

    def _visit_func(self, node) -> None:
        node_id = self._child_id(node.name)
        parent_id = self._current_id()
        parent_node = self.engine.nodes.get(parent_id)
        kind = "method" if parent_node and parent_node.kind == "class" else "function"
        try:
            args = ast.unparse(node.args)
        except Exception:
            args = ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        returns = ""
        if getattr(node, "returns", None) is not None:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except Exception:
                returns = ""
        signature = f"{prefix} {node.name}({args}){returns}:"
        src = _fast_source_segment(self.lines, node)
        self.engine._add_node(CodeNode(
            node_id, node.name, kind, self.filepath,
            node.lineno, getattr(node, "end_lineno", node.lineno), "python",
            signature=signature, doc=ast.get_docstring(node) or "", source=src,
        ))
        self.engine._add_edge(parent_id, node_id, CONTAIN)
        self.name_path.append(node.name)
        self.func_id_stack.append(node_id)
        self.generic_visit(node)
        self.func_id_stack.pop()
        self.name_path.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name:
            self.engine._add_call(self._current_caller(), name)
        self.generic_visit(node)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Inspect SemanticaCodeEngine (DS-Code Graph) context extraction.")
    parser.add_argument("path", help="File or directory to index")
    parser.add_argument("symbol", help="Symbol name to extract focused context for")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    engine = SemanticaCodeEngine()
    t0 = time.perf_counter()
    engine.build_graph_from_directory(args.path)
    build_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    context = engine.get_focused_context(args.symbol, max_depth=args.max_depth, max_tokens=args.max_tokens)
    extract_ms = (time.perf_counter() - t0) * 1000

    print(f"Indexed {len(engine.indexed_files)} file(s), {engine.node_count} node(s) "
          f"in {build_ms:.2f}ms; context extraction took {extract_ms:.2f}ms.")
    print(context)
