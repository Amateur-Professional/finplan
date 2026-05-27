"""Tests for the deterministic single-path engine.

Scenarios use zero inflation and zero volatility so balances, taxes, and the
after-tax gross-up are hand-calculable from first principles. No mocks.
"""

import pytest

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
from finplan.simulation import simulate_deterministic


def _plan(
    *,
    accounts: list[AccountInput],
    spending: float = 1.0,
    start: int = 2030,
    retire: int = 2030,
    end: int = 2030,
    birth: int = 1965,
    real_return: float = 0.0,
    inflation: float = 0.0,
    sourcing: SourcingPolicy = SourcingPolicy.CONVENTIONAL,
    filing: FilingStatus = FilingStatus.SINGLE,
    ss: SocialSecurityInput | None = None,
    contributions: dict[AccountType, float] | None = None,
) -> PlanInput:
    return PlanInput(
        birth_year=birth,
        plan_start_year=start,
        retirement_year=retire,
        plan_end_year=end,
        accounts=accounts,
        annual_contributions=contributions or {},
        annual_spending_real=spending,
        sourcing_policy=sourcing,
        returns=ReturnAssumptions(
            equity_real_return=real_return,
            equity_volatility=0.15,
            bond_real_return=0.0,
            bond_volatility=0.05,
            equity_allocation=1.0,
        ),
        inflation=InflationAssumptions(mean=inflation, volatility=0.0),
        social_security=ss,
        filing_status=filing,
    )


# ---------------------------------------------------------------------------
# Compound growth must match B*(1+r)^n exactly (CLAUDE.md correctness req).
# ---------------------------------------------------------------------------


class TestCompoundGrowth:
    def test_pure_growth_matches_hand_calc(self) -> None:
        # Roth 100k, no flows, 5% real, 0% inflation -> 5% nominal.
        # Working years 2025-2029; check end of 2029 (index 4) == 100k*1.05^5.
        plan = _plan(
            accounts=[AccountInput(account_type=AccountType.TAX_FREE, balance=100_000)],
            start=2025,
            retire=2030,
            end=2030,
            birth=1960,
            real_return=0.05,
        )
        result = simulate_deterministic(plan)
        assert result.year_details[0].balances.tax_free == pytest.approx(105_000)
        assert result.year_details[4].balances.tax_free == pytest.approx(127_628.15625)

    def test_portfolio_return_is_nominal(self) -> None:
        # 4% real + 3% inflation -> (1.04)(1.03)-1 = 0.0712 nominal.
        plan = _plan(
            accounts=[AccountInput(account_type=AccountType.TAX_FREE, balance=100_000)],
            start=2025,
            retire=2026,
            end=2026,
            real_return=0.04,
            inflation=0.03,
        )
        result = simulate_deterministic(plan)
        assert result.year_details[0].portfolio_return == pytest.approx(0.0712)


# ---------------------------------------------------------------------------
# Tax-deferred withdrawals are ORDINARY income, never capital gains.
# ---------------------------------------------------------------------------


class TestTaxDeferredIsOrdinaryIncome:
    def test_wiring_end_to_end(self) -> None:
        # Single, age 65 (no RMD), tax-deferred only, net spend 50k, no growth.
        # Solve gross W: net = 0.88*W + 2128.5 = 50000 -> W = 54399.43 (in 12%).
        # ordinary tax = W - 50000 = 4399.43; LTCG tax must be 0.
        plan = _plan(
            accounts=[
                AccountInput(account_type=AccountType.TAX_DEFERRED, balance=1_000_000)
            ],
            spending=50_000,
        )
        result = simulate_deterministic(plan)
        detail = result.year_details[0]
        assert detail.capital_gains_tax == 0.0
        assert detail.withdrawal_tax_deferred == pytest.approx(54_399.43, abs=1.0)
        assert detail.ordinary_income_tax == pytest.approx(4_399.43, abs=1.0)
        assert result.success is True

    def test_net_spending_target_is_met(self) -> None:
        plan = _plan(
            accounts=[
                AccountInput(account_type=AccountType.TAX_DEFERRED, balance=1_000_000)
            ],
            spending=50_000,
        )
        detail = simulate_deterministic(plan).year_details[0]
        net = (
            detail.social_security_income
            + detail.withdrawal_taxable
            + detail.withdrawal_tax_deferred
            + detail.withdrawal_tax_free
            - detail.total_taxes
        )
        assert net == pytest.approx(50_000, abs=1.0)


# ---------------------------------------------------------------------------
# Conventional sourcing: taxable drained before tax-deferred.
# ---------------------------------------------------------------------------


class TestConventionalSourcing:
    def test_taxable_first_no_tax_when_no_gain(self) -> None:
        # Taxable 40k with basis 40k (no embedded gain) covers a 30k net spend
        # with zero tax; the tax-deferred bucket is untouched.
        plan = _plan(
            accounts=[
                AccountInput(
                    account_type=AccountType.TAXABLE, balance=40_000, cost_basis=40_000
                ),
                AccountInput(account_type=AccountType.TAX_DEFERRED, balance=1_000_000),
            ],
            spending=30_000,
        )
        detail = simulate_deterministic(plan).year_details[0]
        assert detail.withdrawal_taxable == pytest.approx(30_000, abs=0.5)
        assert detail.withdrawal_tax_deferred == 0.0
        assert detail.total_taxes == 0.0

    def test_ltcg_gross_up_meets_target(self) -> None:
        # Taxable 500k, basis 100k (80% embedded gain). Net spend 200k forces
        # withdrawals into the 15% LTCG band; the gross-up must still net 200k.
        plan = _plan(
            accounts=[
                AccountInput(
                    account_type=AccountType.TAXABLE,
                    balance=500_000,
                    cost_basis=100_000,
                )
            ],
            spending=200_000,
        )
        detail = simulate_deterministic(plan).year_details[0]
        net = detail.withdrawal_taxable - detail.total_taxes
        assert net == pytest.approx(200_000, abs=1.0)
        assert detail.capital_gains_tax > 1_000  # genuinely in the 15% band


# ---------------------------------------------------------------------------
# RMDs are forced; surplus cash is reinvested into the taxable bucket.
# ---------------------------------------------------------------------------


class TestRMD:
    def test_forced_rmd_with_low_spending_reinvests_surplus(self) -> None:
        # Born 1950 -> RMD age 73; at 75 the period is 24.6.
        # RMD = 1,000,000 / 24.6 = 40,650.41 (forced, fully ordinary).
        # Spending is tiny, so after-tax surplus is reinvested into taxable.
        plan = _plan(
            accounts=[
                AccountInput(account_type=AccountType.TAX_DEFERRED, balance=1_000_000)
            ],
            spending=1_000,
            start=2025,
            retire=2025,
            end=2025,
            birth=1950,
        )
        detail = simulate_deterministic(plan).year_details[0]
        rmd = 1_000_000 / 24.6
        assert detail.rmd_amount == pytest.approx(rmd, abs=1.0)
        assert detail.withdrawal_tax_deferred == pytest.approx(rmd, abs=1.0)
        # tax on RMD: taxable = rmd - 15,750; 10% to 11,925 then 12%.
        tax = 1_192.50 + 0.12 * (rmd - 15_750 - 11_925)
        net_rmd = rmd - tax
        assert detail.contribution_taxable == pytest.approx(net_rmd - 1_000, abs=1.0)
        assert detail.balances.taxable == pytest.approx(net_rmd - 1_000, abs=1.0)


# ---------------------------------------------------------------------------
# Social Security reduces required withdrawals.
# ---------------------------------------------------------------------------


class TestSocialSecurity:
    def test_ss_covers_spending_no_withdrawal(self) -> None:
        # SS 30k fully covers a 20k net spend (SS alone untaxed at this level),
        # so no portfolio withdrawal is needed and the surplus is reinvested.
        plan = _plan(
            accounts=[
                AccountInput(account_type=AccountType.TAX_DEFERRED, balance=500_000)
            ],
            spending=20_000,
            birth=1960,
            start=2030,
            retire=2030,
            end=2030,
            ss=SocialSecurityInput(claim_age=67, annual_benefit=30_000),
        )
        detail = simulate_deterministic(plan).year_details[0]
        assert detail.social_security_income == pytest.approx(30_000)
        assert detail.withdrawal_tax_deferred == 0.0
        assert detail.total_taxes == 0.0  # SS not taxable with no other income here


# ---------------------------------------------------------------------------
# Depletion -> the trial fails.
# ---------------------------------------------------------------------------


class TestDepletion:
    def test_running_out_marks_failure(self) -> None:
        # Roth 50k, 30k/yr spend, no growth: funds run dry in year 2.
        plan = _plan(
            accounts=[AccountInput(account_type=AccountType.TAX_FREE, balance=50_000)],
            spending=30_000,
            start=2030,
            retire=2030,
            end=2034,
        )
        result = simulate_deterministic(plan)
        assert result.success is False
        assert result.final_wealth == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# Tax-efficient sourcing differs from conventional.
# ---------------------------------------------------------------------------


class TestTaxEfficientSourcing:
    def test_pulls_tax_deferred_before_taxable(self) -> None:
        # Both buckets funded; tax-efficient fills the 12% bracket from
        # tax-deferred first, so it withdraws MORE from tax-deferred than
        # conventional (which would drain taxable first).
        accounts = [
            AccountInput(
                account_type=AccountType.TAXABLE, balance=500_000, cost_basis=250_000
            ),
            AccountInput(account_type=AccountType.TAX_DEFERRED, balance=500_000),
        ]
        conv = simulate_deterministic(
            _plan(
                accounts=accounts, spending=40_000, sourcing=SourcingPolicy.CONVENTIONAL
            )
        ).year_details[0]
        teff = simulate_deterministic(
            _plan(
                accounts=accounts,
                spending=40_000,
                sourcing=SourcingPolicy.TAX_EFFICIENT,
            )
        ).year_details[0]
        assert conv.withdrawal_tax_deferred == 0.0
        assert teff.withdrawal_tax_deferred > 0.0
