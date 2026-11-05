"""Only saddle.cuest may import cuEST (spec §14 as amended by ADR 0001)."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "saddle"


def test_only_the_adapter_imports_cuest() -> None:
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        if path.parent.name == "cuest" and path.name == "bindings.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            if any(n == "cuest" or n.startswith("cuest.") for n in names):
                offenders.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert not offenders, offenders
