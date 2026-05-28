"""Tests for the parametric return/inflation generator.

Distribution moments are checked against the closed-form portfolio mean/variance
and the moment-matched inflation parameters, with large samples and loose
tolerances so the asserts are about correctness, not RNG noise.
"""

import numpy as np
import pytest

from finplan.models import InflationAssumptions, ReturnAssumptions
from finplan.returns import (
    ReturnSequences,
    generate_parametric,
    make_rng,
    resolve_seed,
)


def _returns(alloc: float = 0.6) -> ReturnAssumptions:
    return ReturnAssumptions(
        equity_real_return=0.06,
        equity_volatility=0.18,
        bond_real_return=0.02,
        bond_volatility=0.07,
        equity_bond_correlation=-0.10,
        equity_allocation=alloc,
    )


def _infl(mean: float = 0.025, vol: float = 0.01) -> InflationAssumptions:
    return InflationAssumptions(mean=mean, volatility=vol)


class TestReproducibility:
    def test_same_seed_identical(self) -> None:
        a = generate_parametric(_returns(), _infl(), 100, 30, make_rng(7))
        b = generate_parametric(_returns(), _infl(), 100, 30, make_rng(7))
        assert np.array_equal(a.portfolio_real, b.portfolio_real)
        assert np.array_equal(a.inflation, b.inflation)

    def test_different_seed_differs(self) -> None:
        a = generate_parametric(_returns(), _infl(), 100, 30, make_rng(1))
        b = generate_parametric(_returns(), _infl(), 100, 30, make_rng(2))
        assert not np.array_equal(a.portfolio_real, b.portfolio_real)


class TestShapes:
    def test_shape(self) -> None:
        s = generate_parametric(_returns(), _infl(), 123, 17, make_rng(0))
        assert s.portfolio_real.shape == (123, 17)
        assert s.inflation.shape == (123, 17)
        assert s.n_trials == 123
        assert s.n_years == 17

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            ReturnSequences(portfolio_real=np.zeros((3, 4)), inflation=np.zeros((3, 5)))

    def test_nonpositive_counts_raise(self) -> None:
        with pytest.raises(ValueError):
            generate_parametric(_returns(), _infl(), 0, 10, make_rng(0))


class TestDistribution:
    def test_portfolio_moments_match_closed_form(self) -> None:
        r = _returns(alloc=0.6)
        s = generate_parametric(r, _infl(), 200_000, 1, make_rng(0))
        x = s.portfolio_real.ravel()
        w = 0.6
        exp_mean = w * r.equity_real_return + (1 - w) * r.bond_real_return
        exp_var = (
            w**2 * r.equity_volatility**2
            + (1 - w) ** 2 * r.bond_volatility**2
            + 2
            * w
            * (1 - w)
            * r.equity_bond_correlation
            * r.equity_volatility
            * r.bond_volatility
        )
        assert x.mean() == pytest.approx(exp_mean, abs=2e-3)
        assert x.std() == pytest.approx(exp_var**0.5, rel=0.02)

    def test_inflation_moments_match_request(self) -> None:
        s = generate_parametric(_returns(), _infl(0.03, 0.012), 200_000, 1, make_rng(0))
        i = s.inflation.ravel()
        assert i.mean() == pytest.approx(0.03, abs=1e-3)
        assert i.std() == pytest.approx(0.012, rel=0.03)

    def test_zero_inflation_vol_is_constant(self) -> None:
        s = generate_parametric(_returns(), _infl(0.025, 0.0), 1000, 5, make_rng(0))
        assert np.allclose(s.inflation, 0.025)


class TestResolveSeed:
    def test_passthrough(self) -> None:
        assert resolve_seed(12345) == 12345

    def test_none_returns_reproducible_int(self) -> None:
        seed = resolve_seed(None)
        assert isinstance(seed, int)
        a = make_rng(seed).standard_normal(5)
        b = make_rng(seed).standard_normal(5)
        assert np.array_equal(a, b)
