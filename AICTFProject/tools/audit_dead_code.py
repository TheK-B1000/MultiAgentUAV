"""Phase 11 Stage 8: Dead-code analysis for the AICTFProject codebase.

Scans Python source in rl/, gpu_env/, tests/, tools/, and experiments/ for:
  - Unused imports (imported names never referenced after the import statement)
  - Functions/classes defined but never referenced anywhere in the scanned tree
  - Private functions (_-prefixed) defined in rl/ and gpu_env/ but never called

Outputs:
  artifacts/cleanup_audit/dead_code_report.json
  artifacts/cleanup_audit/unused_imports.csv

Usage:
    uv run python tools/audit_dead_code.py
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SCAN_PACKAGES = ["rl", "gpu_env", "tests", "tools", "experiments"]
SKIP_DIRS = {".venv", ".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}

# Names that look unused but are intentional (protocol stubs, re-exports, __all__, etc.)
WHITELIST_NAMES = {
    # Dunder names are always intentional
    "__all__", "__version__", "__author__", "__getattr__", "__init__",
    "__repr__", "__str__", "__len__", "__eq__", "__hash__", "__call__",
    "__enter__", "__exit__", "__iter__", "__next__", "__contains__",
    "__getitem__", "__setitem__", "__delitem__", "__bool__", "__add__",
    "__mul__", "__sub__",
    # pytest fixtures and markers are collected by name
    "pytest_configure", "pytest_collection_modifyitems",
}


def iter_python_files(root: Path) -> list[Path]:
    result = []
    for pkg in SCAN_PACKAGES:
        pkg_dir = root / pkg
        if not pkg_dir.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(pkg_dir):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                if fname.endswith(".py"):
                    result.append(Path(dirpath) / fname)
    return result


def extract_definitions(tree: ast.Module, path: Path) -> list[dict]:
    """Extract top-level and class-level function/class definitions."""
    defs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            lineno = node.lineno
            # Determine type
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            defs.append({"name": name, "kind": kind, "file": str(path), "line": lineno})
    return defs


def extract_imports(tree: ast.Module, path: Path) -> list[dict]:
    """Extract all import statements with their bound names."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname if alias.asname else alias.name.split(".")[0]
                imports.append({
                    "file": str(path),
                    "line": node.lineno,
                    "module": alias.name,
                    "bound_name": bound,
                    "import_style": "import",
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname if alias.asname else alias.name
                imports.append({
                    "file": str(path),
                    "line": node.lineno,
                    "module": f"{module}.{alias.name}",
                    "bound_name": bound,
                    "import_style": "from",
                })
    return imports


def collect_name_usages(tree: ast.Module) -> set[str]:
    """Collect all Name and Attribute node ids used in the file (after definitions)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def scan_all_usages(files: list[Path]) -> set[str]:
    """Union of all name usages across all files (for cross-file reference checking)."""
    all_usages: set[str] = set()
    for path in files:
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(path))
            all_usages |= collect_name_usages(tree)
        except SyntaxError:
            pass
    return all_usages


def find_unused_imports(files: list[Path]) -> list[dict]:
    """Find imports whose bound name is never used in the same file."""
    unused = []
    for path in files:
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue

        file_imports = extract_imports(tree, path)
        file_usages = collect_name_usages(tree)

        # Remove the bound name from usages because the import itself contributes
        # a Name node in some AST visitors. We check src text as fallback.
        for imp in file_imports:
            bound = imp["bound_name"]
            if bound in WHITELIST_NAMES:
                continue
            # Check if the name appears anywhere else in the file beyond the import line
            # Simple heuristic: count non-import occurrences of the name
            usage_count = src.count(bound)
            import_occurrences = sum(1 for ln in src.splitlines() if re.search(
                r"\b(import|from)\b", ln) and bound in ln)
            if usage_count <= import_occurrences:
                unused.append({
                    "file": str(path.relative_to(path.parents[len(path.parts) - len(Path.cwd().parts) - 1])).replace("\\", "/"),
                    "line": imp["line"],
                    "module": imp["module"],
                    "bound_name": bound,
                    "confidence": "high" if usage_count == import_occurrences else "medium",
                })
    return unused


def find_unreferenced_privates(files: list[Path], root: Path, all_usages: set[str]) -> list[dict]:
    """Find private functions/classes (_-prefixed) not referenced anywhere in the codebase."""
    unreferenced = []
    for path in files:
        # Only check rl/ and gpu_env/ source, not tests/tools/experiments
        rel = path.relative_to(root)
        if rel.parts[0] not in ("rl", "gpu_env"):
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue

        defs = extract_definitions(tree, path)
        for d in defs:
            name = d["name"]
            if not name.startswith("_"):
                continue
            if name in WHITELIST_NAMES:
                continue
            if name not in all_usages:
                unreferenced.append({
                    "file": str(rel).replace("\\", "/"),
                    "line": d["line"],
                    "name": name,
                    "kind": d["kind"],
                    "confidence": "high",
                })
    return unreferenced


def find_unreferenced_public(files: list[Path], root: Path, all_usages: set[str]) -> list[dict]:
    """Find public functions defined in tools/ and experiments/ not referenced anywhere."""
    unreferenced = []
    for path in files:
        rel = path.relative_to(root)
        if rel.parts[0] not in ("tools", "experiments"):
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue

        defs = extract_definitions(tree, path)
        for d in defs:
            name = d["name"]
            if name in WHITELIST_NAMES or name.startswith("__"):
                continue
            if name == "main":
                continue
            if name not in all_usages:
                unreferenced.append({
                    "file": str(rel).replace("\\", "/"),
                    "line": d["line"],
                    "name": name,
                    "kind": d["kind"],
                    "confidence": "low",  # tools/experiments may be run directly
                })
    return unreferenced


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "artifacts" / "cleanup_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning Python files...")
    files = iter_python_files(root)
    print(f"  {len(files)} files found")

    print("Collecting all name usages (cross-file)...")
    all_usages = scan_all_usages(files)
    print(f"  {len(all_usages)} unique names referenced across codebase")

    print("Finding unused imports...")
    unused_imports = find_unused_imports(files)
    print(f"  {len(unused_imports)} potentially unused imports")

    print("Finding unreferenced private symbols in rl/ and gpu_env/...")
    unreferenced_privates = find_unreferenced_privates(files, root, all_usages)
    print(f"  {len(unreferenced_privates)} unreferenced private symbols")

    print("Finding unreferenced public symbols in tools/ and experiments/...")
    unreferenced_public = find_unreferenced_public(files, root, all_usages)
    print(f"  {len(unreferenced_public)} unreferenced public symbols in tools/experiments")

    # Write unused imports CSV
    csv_path = out_dir / "unused_imports.csv"
    if unused_imports:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "line", "module", "bound_name", "confidence"])
            w.writeheader()
            w.writerows(unused_imports)
    else:
        csv_path.write_text("(none found)\n", encoding="utf-8")

    # Write full report
    report = {
        "stage": 8,
        "title": "Dead-Code Analysis",
        "date": "2026-06-28",
        "files_scanned": len(files),
        "unique_names_in_codebase": len(all_usages),
        "summary": {
            "unused_imports_found": len(unused_imports),
            "unreferenced_private_symbols": len(unreferenced_privates),
            "unreferenced_public_tools_experiments": len(unreferenced_public),
        },
        "unused_imports": unused_imports[:50],  # cap to avoid enormous JSON
        "unreferenced_privates": unreferenced_privates,
        "unreferenced_public_tools_experiments": unreferenced_public[:50],
        "methodology": [
            "AST-based scan — no runtime import, no torch required",
            "Unused imports: bound name appears in source only on import lines",
            "Unreferenced privates: _-prefixed symbols in rl/ and gpu_env/ not found in any file's name-usage set",
            "Unreferenced public tools/experiments: excludes 'main' and __dunder__ names; confidence=low (may be entry points)",
            "Cross-file usage set is union of all Name/Attribute nodes across all scanned files",
        ],
        "known_false_positive_categories": [
            "Re-export facades: names imported and re-exported via __all__ look unused in-file",
            "TYPE_CHECKING guards: imports inside 'if TYPE_CHECKING:' are annotation-only",
            "Dynamic attribute access: getattr(obj, name) references won't appear as Name nodes",
            "Pytest fixtures: consumed by test framework, not by explicit name calls",
            "Entry-point scripts in tools/experiments: called via CLI, not imported",
        ],
    }

    report_path = out_dir / "dead_code_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(f"Unused imports CSV: {csv_path}")


if __name__ == "__main__":
    main()
