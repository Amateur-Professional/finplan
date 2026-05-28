"""Benchmark validation against published retirement research (CLAUDE.md).

Reproduces the "4% rule": a 4% initial withdrawal, held constant in real terms,
over a 30-year retirement on a balanced portfolio.

To isolate the *projection mechanics* from the tax model, the portfolio is held
entirely in a tax-free (Roth) bucket -- no tax drag, no Social Security -- which
is the setup the original studies assume (pre-tax withdrawals from a single
portfolio). For a Roth-only, fixed-real plan, inflation cancels in real terms,
so success depends only on the real-return distribution.

Why a Monte Carlo band (not ~100%)
----------------------------------
The 4% rule originates from HISTORICAL backtests -- Bengen (1994) and the
Trinity Study (Cooley, Hubbard & Walz, 1998) -- which never failed over any
rolling 30-year US period (~95-100% success).

Monte Carlo draws returns i.i.d. and so lacks the mean reversion present in
historical sequences; it projects more long-run tail risk and therefore reports
*lower* success at the same 4% rate. This is well documented:

  * Wade Pfau (retirementresearcher.com): Monte Carlo calibrated to the same
    historical moments yields ~6% failure (~94% success) for the 4% strategy
    vs. ~0% historically, and MC "pretty much always produces lower numbers."
  * Morningstar, "The State of Retirement Income" (Benz, Rekenthaler, Arnott,
    Pfau; annual since 2021): forward-looking MC with a 90%-success target put
    the safe starting rate at 3.3% in 2021, rising toward ~4% in later higher-
    yield years -- i.e. 4% sits near the high-80s/low-90s success under
    reasonable capital-market assumptions, and lower under pessimistic ones.

This engine, with moderately conservative real-return assumptions, lands at
~0.84-0.87 -- inside that Monte Carlo band. The exact figure is CMA-dependent,
so the test asserts a band, not a point. A historical block-bootstrap that would
reproduce the ~95% Bengen/Trinity figure is out of v0 scope (see returns.py).
"""

import pytest

from finplan import montecarlo
from finplan.models import (
    AccountInput,
    AccountType,
    FilingStatus,
    InflationAssumptions,
    PlanInput,
    ReturnAssumptions,
    SourcingPolicy,
)


def _bengen(alloc: float, *, spending: float = 40_000, seed: int = 42, n: int = 4000):
    plan = PlanInput(
        birth_year=1960,
        plan_start_year=2025,
        retirement_year=2025,
        plan_end_year=2054,  # 30-year horizon, inclusive
        accounts=[AccountInput(account_type=AccountType.TAX_FREE, balance=1_000_000)],
        annual_spending_real=spending,
        sourcing_policy=SourcingPolicy.CONVENTIONAL,
        returns=ReturnAssumptions(
            equity_real_return=0.066,
            equity_volatility=0.185,
            bond_real_return=0.018,
            bond_volatility=0.075,
            equity_bond_correlation=-0.10,
            equity_allocation=alloc,
        ),
        inflation=InflationAssumptions(mean=0.025, volatility=0.01),
        filing_status=FilingStatus.SINGLE,
        n_trials=n,
        random_seed=seed,
    )
    return montecarlo.run(plan)


class TestFourPercentRule:
    def test_30_year_horizon(self) -> None:
        assert len(_bengen(0.6).years) == 30

    @pytest.mark.parametrize("alloc", [0.5, 0.6, 0.75])
    def test_success_in_parametric_research_band(self, alloc: float) -> None:
        r = _bengen(alloc)
        # Parametric-MC band; historical mode would be higher (~0.95).
        # See module docstring for the documented comparison.
        assert 0.80 <= r.success_probability <= 0.95

    def test_higher_withdrawal_lowers_success(self) -> None:
        base = _bengen(0.6, spending=40_000)
        worse = _bengen(0.6, spending=60_000)  # 6% withdrawal rate
        assert worse.success_probability < base.success_probability
