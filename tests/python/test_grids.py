import math

import numpy as np
import pytest

from saddle.grids import (
    LEBEDEV_ORDERS,
    GridError,
    GridSpec,
    becke_radial,
    bragg_radius_bohr,
    parse_grid,
)


@pytest.mark.parametrize("alpha", [0.05, 0.3, 1.0, 10.0, 100.0, 1000.0])
def test_radial_quadrature_integrates_gaussians(alpha: float) -> None:
    r, w = becke_radial(99, 1.0)
    exact = math.sqrt(math.pi) / (4.0 * alpha**1.5)
    assert float(np.sum(w * np.exp(-alpha * r * r))) == pytest.approx(exact, rel=1e-9)


def test_radial_weights_include_r_squared_and_are_positive() -> None:
    r, w = becke_radial(75, 1.3)
    assert np.all(w > 0) and np.all(r > 0)
    assert float(np.sum(w * np.exp(-r))) == pytest.approx(2.0, rel=1e-9)


def test_more_points_converge() -> None:
    errs = []
    for n in (25, 50, 99):
        r, w = becke_radial(n, 1.0)
        errs.append(
            abs(float(np.sum(w * np.exp(-5.0 * r * r))) - math.sqrt(math.pi) / (4 * 5.0**1.5))
        )
    assert errs[0] > errs[1] > errs[2]


def test_bragg_radii() -> None:
    assert bragg_radius_bohr("H") == pytest.approx(0.35 / 0.529177210903)
    assert bragg_radius_bohr("C") == pytest.approx(0.70 / 0.529177210903)
    with pytest.raises(GridError):
        bragg_radius_bohr("Og")


def test_parse_spec_grids() -> None:
    assert parse_grid("(75,302)") == GridSpec(75, 302)
    assert parse_grid("(99,590)") == GridSpec(99, 590)
    assert parse_grid("(150, 974)") == GridSpec(150, 974)
    assert str(GridSpec(99, 590)) == "(99,590)"


def test_angular_counts_must_be_lebedev_numbers() -> None:
    assert 302 in LEBEDEV_ORDERS and 590 in LEBEDEV_ORDERS
    with pytest.raises(GridError, match="Lebedev"):
        GridSpec(75, 300)
    with pytest.raises(GridError, match="radial"):
        GridSpec(0, 302)
    with pytest.raises(GridError):
        parse_grid("75x302")
