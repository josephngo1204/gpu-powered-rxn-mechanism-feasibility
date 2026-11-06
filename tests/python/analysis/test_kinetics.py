import pytest

from saddle.analysis import kinetics

T = 298.15


def test_23_kcal_half_life_is_about_two_hours() -> None:
    hours = kinetics.half_life_s(23.0, T) / 3600.0
    assert 1.8 < hours < 2.6


def test_25_kcal_half_life_is_about_three_days() -> None:
    days = kinetics.half_life_s(25.0, T) / 86400.0
    assert 2.5 < days < 3.5


def test_one_kcal_changes_rate_about_5_4_times() -> None:
    assert kinetics.rate_ratio(1.0, T) == pytest.approx(5.4, abs=0.05)


def test_1_36_kcal_gap_gives_ten_to_one() -> None:
    assert kinetics.rate_ratio(1.36, T) == pytest.approx(10.0, rel=0.02)


def test_barrier_limit_inverts_half_life() -> None:
    for t_half in (1.0, 3600.0, 86400.0 * 7):
        limit = kinetics.barrier_limit_kcal_mol(T, t_half)
        assert kinetics.half_life_s(limit, T) == pytest.approx(t_half, rel=1e-10)


def test_eyring_prefactor() -> None:
    assert kinetics.eyring_rate(0.0, T) == pytest.approx(6.2124e12, rel=1e-4)


def test_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError):
        kinetics.eyring_rate(10.0, 0.0)


def test_energetic_span_linear_picks_ts_after_lowest_prior_intermediate() -> None:
    states = [
        kinetics.State("R", 0.0, is_ts=False),
        kinetics.State("TS1", 10.0, is_ts=True),
        kinetics.State("I", -5.0, is_ts=False),
        kinetics.State("TS2", 12.0, is_ts=True),
        kinetics.State("P", -20.0, is_ts=False),
    ]
    span = kinetics.energetic_span(states)
    assert span.value == pytest.approx(17.0)
    assert span.tdts == "TS2"
    assert span.tdi == "I"


def test_energetic_span_cycle_adds_reaction_energy_when_ts_precedes_intermediate() -> None:
    states = [
        kinetics.State("I1", 0.0, is_ts=False),
        kinetics.State("TS1", 20.0, is_ts=True),
        kinetics.State("I2", -5.0, is_ts=False),
        kinetics.State("TS2", 8.0, is_ts=True),
    ]
    span = kinetics.energetic_span(states, reaction_energy=-10.0)
    assert span.value == pytest.approx(20.0)
    assert (span.tdts, span.tdi) == ("TS1", "I1")


def test_energetic_span_requires_a_ts() -> None:
    with pytest.raises(ValueError):
        kinetics.energetic_span([kinetics.State("R", 0.0, is_ts=False)])
