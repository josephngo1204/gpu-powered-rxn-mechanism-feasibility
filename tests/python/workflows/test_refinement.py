import numpy as np

from saddle.workflows.pipeline import _refined


def fake_search(results: list[dict[str, object]]):  # type: ignore[no-untyped-def]
    calls = iter(results)

    def search(model, coords, numbers, **kw):  # type: ignore[no-untyped-def]
        return dict(next(calls))

    return search


def stage(rms: float, converged: bool) -> dict[str, object]:
    return {
        "coords": np.zeros(3),
        "gradient": np.full(3, rms),
        "converged": converged,
        "iterations": 5,
        "n_gradients": 7,
    }


def test_converged_refinement_is_used() -> None:
    r = _refined(fake_search([stage(1e-5, True), stage(1e-8, True)]), None, np.zeros(3), [])
    assert r["refined"] and np.allclose(r["gradient"], 1e-8)
    assert r["iterations"] == 10 and r["n_gradients"] == 14


def test_wandering_refinement_keeps_the_better_stage() -> None:
    r = _refined(fake_search([stage(1e-6, True), stage(3e-3, False)]), None, np.zeros(3), [])
    assert not r["refined"] and r["converged"]
    assert np.allclose(r["gradient"], 1e-6)


def test_unconverged_but_better_refinement_is_kept() -> None:
    r = _refined(fake_search([stage(1e-4, False), stage(2e-7, False)]), None, np.zeros(3), [])
    assert np.allclose(r["gradient"], 2e-7) and not r["converged"]


def test_random_rotation_is_proper_and_preserves_distances() -> None:
    from saddle.workflows.pipeline import _random_rotation, _rotate

    q = _random_rotation(7)
    assert np.allclose(q @ q.T, np.eye(3)) and np.isclose(np.linalg.det(q), 1.0)
    x = np.array([0.0, 0, 0, 1.0, 0.2, 0, -0.3, 1.1, 0.4])
    y = _rotate(x, q).reshape(-1, 3)
    d = lambda z: np.linalg.norm(z[:, None] - z[None], axis=-1)  # noqa: E731
    assert np.allclose(d(x.reshape(-1, 3)), d(y))


def test_dft_detection() -> None:
    from saddle.workflows.pipeline import _is_dft

    assert _is_dft({"type": "cuest", "functional": "WB97X-V"})
    assert not _is_dft({"type": "cuest", "functional": "HF"})
    assert not _is_dft({"type": "fake:pair_model"})
