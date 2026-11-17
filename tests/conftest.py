"""Pytest configuration and notebook code extraction helpers."""

import json
import os
import re
from typing import Any, Dict


def extract_symbols_from_notebook(nb_path: str, symbols: list[str]) -> Dict[str, Any]:
    """Extract specified function and class definitions from a Jupyter notebook.

    Args:
        nb_path: Relative or absolute path to the .ipynb file.
        symbols: List of function/class names to extract.

    Returns:
        Dictionary mapping symbol names to their executable objects.
    """
    if not os.path.isabs(nb_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nb_path = os.path.join(root_dir, nb_path)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Patterns to match only definitions of requested symbols
    def_patterns = [
        re.compile(rf"^\s*(def|class)\s+{re.escape(sym)}\b", re.MULTILINE)
        for sym in symbols
    ]

    code_blocks: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            clean_lines = [
                line
                for line in src.splitlines()
                if not line.strip().startswith("!")
                and not line.strip().startswith("%")
            ]
            code_str = "\n".join(clean_lines)
            if any(pattern.search(code_str) for pattern in def_patterns):
                code_blocks.append(code_str)

    namespace: Dict[str, Any] = {}
    for block in code_blocks:
        try:
            exec(block, namespace)
        except Exception:
            pass

    return {sym: namespace[sym] for sym in symbols if sym in namespace}
