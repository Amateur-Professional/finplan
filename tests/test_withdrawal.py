"""Tests for withdrawal.py.

Allocation amounts and realized gains are hand-calculated from the pro-rata
cost-basis math in accounts.py. No mocks.
"""

import pytest

from finplan.accounts import Account
from finplan.models import AccountType, SourcingPolicy
from finplan.withdrawal import (
    WithdrawalPlan,
    fixed_real_spending,
    source_withdrawals,
)


def _accounts(
    taxable: tuple[float, float] | None = None,
    tax_deferred: float | None = None,
    tax_free: float | None = None,
) -> dict[AccountType, Account]:
    out: dict[AccountType, Account] = {}
    if taxable is not None:
        out[AccountType.TAXABLE] = Account(
            AccountType.TAXABLE, balance=taxable[0], cost_basis=taxable[1]
        )
    if tax_deferred is not None:
        out[AccountType.TAX_DEFERRED] = Account(
            AccountType.TAX_DEFERRED, balance=tax_deferred
        )
    if tax_free is not None:
        out[AccountType.TAX_FREE] = Account(AccountType.TAX_FREE, balance=tax_free)
    return out


# ---------------------------------------------------------------------------
# fixed_real_spending
# ---------------------------------------------------------------------------


class TestFixedRealSpending:
    def test_no_inflation(self) -> None:
        assert fixed_real_spending(40_000, 1.0) == pytest.approx(40_000)

    def test_with_cumulative_inflation(self) -> None:
        # 40,000 real * 1.21 cumulative factor = 48,400 nominal
        assert fixed_real_spending(40_000, 1.21) == pytest.approx(48_400)


# ---------------------------------------------------------------------------
# Conventional sourcing: taxable -> tax-deferred -> Roth
# ---------------------------------------------------------------------------


class TestConventionalSourcing:
    def test_funds_entirely_from_taxable(self) -> None:
        # taxable balance 100k (basis 60k -> 40% gain); withdraw 25k
        # realized gain = 25k * 0.40 = 10k; nothing else touched
        accts = _accounts(
            taxable=(100_000, 60_000), tax_deferred=50_000, tax_free=50_000
        )
        plan = source_withdrawals(accts, 25_000, SourcingPolicy.CONVENTIONAL)
        assert plan.withdrawal_taxable == pytest.approx(25_000)
        assert plan.withdrawal_tax_deferred == 0.0
        assert plan.withdrawal_tax_free == 0.0
        assert plan.realized_gain == pytest.approx(10_000)
        assert plan.total_withdrawn == pytest.approx(25_000)
        assert plan.shortfall == 0.0

    def test_spills_into_tax_deferred(self) -> None:
        # taxable only has 30k (all gain since basis 0); need 50k
        # -> 30k taxable (gain 30k) + 20k tax-deferred
        accts = _accounts(taxable=(30_000, 0), tax_deferred=80_000, tax_free=40_000)
        plan = source_withdrawals(accts, 50_000, SourcingPolicy.CONVENTIONAL)
        assert plan.withdrawal_taxable == pytest.approx(30_000)
        assert plan.withdrawal_tax_deferred == pytest.approx(20_000)
        assert plan.withdrawal_tax_free == 0.0
        assert plan.realized_gain == pytest.approx(30_000)
        assert plan.total_withdrawn == pytest.approx(50_000)

    def test_spills_through_all_three(self) -> None:
        accts = _accounts(
            taxable=(10_000, 10_000), tax_deferred=20_000, tax_free=40_000
        )
        plan = source_withdrawals(accts, 45_000, SourcingPolicy.CONVENTIONAL)
        assert plan.withdrawal_taxable == pytest.approx(10_000)
        assert plan.withdrawal_tax_deferred == pytest.approx(20_000)
        assert plan.withdrawal_tax_free == pytest.approx(15_000)
        assert plan.realized_gain == pytest.approx(0.0)  # basis == balance
        assert plan.total_withdrawn == pytest.approx(45_000)

    def test_shortfall_when_funds_exhausted(self) -> None:
        accts = _accounts(taxable=(10_000, 0), tax_deferred=5_000, tax_free=5_000)
        plan = source_withdrawals(accts, 30_000, SourcingPolicy.CONVENTIONAL)
        assert plan.total_withdrawn == pytest.approx(20_000)
        assert plan.shortfall == pytest.approx(10_000)


# ---------------------------------------------------------------------------
# Tax-efficient sourcing: fill tax-deferred headroom first, then taxable,
# then remaining tax-deferred, then Roth.
# ---------------------------------------------------------------------------


class TestTaxEfficientSourcing:
    def test_fills_headroom_then_taxable(self) -> None:
        # headroom 30k: pull 30k tax-deferred first, then 20k from taxable
        accts = _accounts(
            taxable=(100_000, 60_000), tax_deferred=80_000, tax_free=40_000
        )
        plan = source_withdrawals(
            accts, 50_000, SourcingPolicy.TAX_EFFICIENT, tax_deferred_headroom=30_000
        )
        assert plan.withdrawal_tax_deferred == pytest.approx(30_000)
        assert plan.withdrawal_taxable == pytest.approx(20_000)
        # taxable gain: 20k * 40% = 8k
        assert plan.realized_gain == pytest.approx(8_000)
        assert plan.total_withdrawn == pytest.approx(50_000)

    def test_remaining_need_spills_back_to_tax_deferred(self) -> None:
        # headroom 10k; need 60k. 10k deferred + 30k taxable (exhausts it)
        # + 20k more deferred. tax_deferred total = 30k.
        accts = _accounts(
            taxable=(30_000, 30_000), tax_deferred=80_000, tax_free=40_000
        )
        plan = source_withdrawals(
            accts, 60_000, SourcingPolicy.TAX_EFFICIENT, tax_deferred_headroom=10_000
        )
        assert plan.withdrawal_taxable == pytest.approx(30_000)
        assert plan.withdrawal_tax_deferred == pytest.approx(30_000)
        assert plan.withdrawal_tax_free == 0.0
        assert plan.total_withdrawn == pytest.approx(60_000)

    def test_zero_headroom_behaves_like_taxable_first(self) -> None:
        # headroom 0 -> first step draws nothing; taxable funds it.
        accts = _accounts(
            taxable=(100_000, 60_000), tax_deferred=80_000, tax_free=40_000
        )
        plan = source_withdrawals(
            accts, 25_000, SourcingPolicy.TAX_EFFICIENT, tax_deferred_headroom=0.0
        )
        assert plan.withdrawal_taxable == pytest.approx(25_000)
        assert plan.withdrawal_tax_deferred == 0.0

    def test_headroom_capped_by_balance(self) -> None:
        # headroom 50k but only 20k in tax-deferred; the rest comes from taxable.
        accts = _accounts(
            taxable=(100_000, 100_000), tax_deferred=20_000, tax_free=40_000
        )
        plan = source_withdrawals(
            accts, 50_000, SourcingPolicy.TAX_EFFICIENT, tax_deferred_headroom=50_000
        )
        assert plan.withdrawal_tax_deferred == pytest.approx(20_000)
        assert plan.withdrawal_taxable == pytest.approx(30_000)


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------


class TestGeneral:
    def test_zero_gross_is_noop(self) -> None:
        accts = _accounts(taxable=(100_000, 60_000))
        plan = source_withdrawals(accts, 0, SourcingPolicy.CONVENTIONAL)
        assert plan.total_withdrawn == 0.0
        assert plan.shortfall == 0.0
        assert plan.withdrawn == {}

    def test_original_accounts_not_mutated(self) -> None:
        accts = _accounts(taxable=(100_000, 60_000), tax_deferred=50_000)
        source_withdrawals(accts, 25_000, SourcingPolicy.CONVENTIONAL)
        assert accts[AccountType.TAXABLE].balance == 100_000  # frozen + copied dict
        assert accts[AccountType.TAX_DEFERRED].balance == 50_000

    def test_plan_balances_reflect_withdrawals(self) -> None:
        accts = _accounts(taxable=(100_000, 60_000), tax_deferred=50_000)
        plan: WithdrawalPlan = source_withdrawals(
            accts, 25_000, SourcingPolicy.CONVENTIONAL
        )
        assert plan.accounts[AccountType.TAXABLE].balance == pytest.approx(75_000)
