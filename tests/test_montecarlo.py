"""Tests for the Monte Carlo runner and result aggregation.

Covers result shape, percentile ordering, reproducibility (including the
resolve-seed-when-None path), and a near-deterministic limit where tiny
volatilities must collapse the Monte Carlo back onto the deterministic engine.
"""

import numpy as np
import pytest

from finplan import montecarlo
from finplan.models import (
    AccountInput,
    AccountType,
    FilingStatus,
    InflationAssumptions,
    PlanInput,
    ReturnAssumptions,
    SocialSecurityInput,
    SourcingPolicy,
)
from finplan.montecarlo import _median_index
from finplan.simulation import simulate_deterministic

_N_YEARS = 2055 - 2025 + 1


def _plan(**kw: object) -> PlanInput:
    defaults: dict[str, object] = dict(
        birth_year=1965,
        plan_start_year=2025,
        retirement_year=2030,
        plan_end_year=2055,
        accounts=[
            AccountInput(
                account_type=AccountType.TAXABLE, balance=400_000, cost_basis=200_000
            ),
            AccountInput(account_type=AccountType.TAX_DEFERRED, balance=600_000),
            AccountInput(account_type=AccountType.TAX_FREE, balance=200_000),
        ],
        annual_spending_real=70_000,
        sourcing_policy=SourcingPolicy.CONVENTIONAL,
        returns=ReturnAssumptions(
            equity_real_return=0.05,
            equity_volatility=0.16,
            bond_real_return=0.01,
            bond_volatility=0.06,
            equity_allocation=0.6,
        ),
        inflation=InflationAssumptions(mean=0.025, volatility=0.01),
        social_security=SocialSecurityInput(claim_age=67, annual_benefit=28_000),
        filing_status=FilingStatus.MARRIED_FILING_JOINTLY,
        n_trials=300,
        random_seed=123,
    )
    defaults.update(kw)
    return PlanInput(**defaults)  # type: ignore[arg-type]


class TestResultShape:
    def test_years_and_path_lengths(self) -> None:
        r = montecarlo.run(_plan())
        assert len(r.years) == _N_YEARS
        assert r.years[0] == 2025
        assert r.years[-1] == 2055
        for key in ("p10", "p25", "p50", "p75", "p90"):
            assert len(r.percentile_paths[key]) == _N_YEARS
        assert len(r.median_year_details) == _N_YEARS

    def test_counts_consistent(self) -> None:
        r = montecarlo.run(_plan())
        assert r.n_trials == 300
        assert 0 <= r.n_successes <= 300
        assert r.success_probability == pytest.approx(r.n_successes / 300)


class TestPercentileMonotonicity:
    def test_ordered_each_year(self) -> None:
        r = montecarlo.run(_plan())
        p = r.percentile_paths
        for t in range(_N_YEARS):
            assert (
                p["p10"][t] <= p["p25"][t] <= p["p50"][t] <= p["p75"][t] <= p["p90"][t]
            )


class TestReproducibility:
    def test_explicit_seed_repeatable(self) -> None:
        a = montecarlo.run(_plan(random_seed=999))
        b = montecarlo.run(_plan(random_seed=999))
        assert a.success_probability == b.success_probability
        assert a.percentile_paths == b.percentile_paths

    def test_none_seed_recorded_and_replayable(self) -> None:
        a = montecarlo.run(_plan(random_seed=None))
        # The resolved seed is stamped onto the returned plan...
        assert isinstance(a.plan_input.random_seed, int)
        # ...and feeding that plan back reproduces the run exactly.
        b = montecarlo.run(a.plan_input)
        assert a.success_probability == b.success_probability
        assert a.percentile_paths == b.percentile_paths


class TestMedianIndex:
    def test_picks_a_real_middle_trial(self) -> None:
        # sorted order of values -> indices [0,3,2,4,1]; middle (idx 2) -> trial 2.
        fw = np.array([10.0, 50.0, 30.0, 20.0, 40.0])
        assert _median_index(fw) == 2


class TestDeterministicLimit:
    def test_near_zero_vol_collapses_to_deterministic(self) -> None:
        plan = _plan(
            returns=ReturnAssumptions(
                equity_real_return=0.05,
                equity_volatility=1e-7,
                bond_real_return=0.01,
                bond_volatility=1e-7,
                equity_allocation=0.6,
            ),
            inflation=InflationAssumptions(mean=0.02, volatility=0.0),
            n_trials=40,
            random_seed=5,
        )
        det = simulate_deterministic(plan)
        mc = montecarlo.run(plan)
        assert mc.percentile_paths["p50"][-1] == pytest.approx(
            det.final_wealth, rel=1e-3
        )
        # With effectively no randomness every trial agrees on success.
        assert mc.success_probability in (0.0, 1.0)
