from pathlib import Path

from saddle import _core

REPO_STUB = Path(__file__).resolve().parents[2] / "python" / "saddle" / "_core.pyi"


def test_committed_stub_matches_generated_stub() -> None:
    generated = Path(_core.__file__).with_name("_core.pyi")
    assert generated.exists(), "build did not generate _core.pyi"
    assert REPO_STUB.read_text() == generated.read_text(), (
        "python/saddle/_core.pyi is stale; copy the generated stub after rebuilding"
    )
