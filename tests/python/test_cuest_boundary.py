"""Only saddle.cuest may import cuEST."""

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


def test_cpu_layers_import_without_cupy_or_cuest() -> None:
    """A clean checkout passes all non-GPU tests, so the workflow,
    gate and CLI layers must not import cupy/cuest at module load."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('cupy', 'cuest'):\n"
        "            raise ImportError(f'blocked {name}')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import saddle.workflows.pipeline, saddle.gates.gates, saddle.cli\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
