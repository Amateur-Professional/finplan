"""Tests for accounts.py.

Cost-basis math is hand-calculated from the average-cost / pro-rata method.
No mocks -- every assertion is derivable from first principles.
"""

import pytest

from finplan.accounts import Account, contribute, grow, withdraw
from finplan.models import AccountInput, AccountType


def _taxable(balance: float, cost_basis: float) -> Account:
    return Account(AccountType.TAXABLE, balance=balance, cost_basis=cost_basis)


def _tax_deferred(balance: float) -> Account:
    return Account(AccountType.TAX_DEFERRED, balance=balance)


def _tax_free(balance: float) -> Account:
    return Account(AccountType.TAX_FREE, balance=balance)


# ---------------------------------------------------------------------------
# from_input
# ---------------------------------------------------------------------------


class TestFromInput:
    def test_carries_fields(self) -> None:
        acct = Account.from_input(
            AccountInput(
                account_type=AccountType.TAXABLE, balance=100_000, cost_basis=60_000
            )
        )
        assert acct.account_type == AccountType.TAXABLE
        assert acct.balance == 100_000
        assert acct.cost_basis == 60_000


# ---------------------------------------------------------------------------
# grow -- growth is unrealised; cost basis is untouched
# ---------------------------------------------------------------------------


class TestGrow:
    def test_balance_compounds(self) -> None:
        acct = grow(_taxable(100_000, 60_000), 0.07)
        assert acct.balance == pytest.approx(107_000)

    def test_cost_basis_unchanged_by_growth(self) -> None:
        acct = grow(_taxable(100_000, 60_000), 0.07)
        assert acct.cost_basis == 60_000  # embedded gain grew from 40k to 47k

    def test_negative_return(self) -> None:
        acct = grow(_taxable(100_000, 60_000), -0.20)
        assert acct.balance == pytest.approx(80_000)
        assert acct.cost_basis == 60_000

    def test_multi_year_compounding(self) -> None:
        acct = _tax_deferred(10_000)
        for _ in range(3):
            acct = grow(acct, 0.05)
        # 10,000 * 1.05^3 = 11,576.25
        assert acct.balance == pytest.approx(11_576.25)


# ---------------------------------------------------------------------------
# contribute
# ---------------------------------------------------------------------------


class TestContribute:
    def test_taxable_adds_to_basis(self) -> None:
        acct = contribute(_taxable(100_000, 60_000), 10_000)
        assert acct.balance == pytest.approx(110_000)
        assert acct.cost_basis == pytest.approx(70_000)

    def test_tax_deferred_no_basis(self) -> None:
        acct = contribute(_tax_deferred(50_000), 10_000)
        assert acct.balance == pytest.approx(60_000)
        assert acct.cost_basis == 0.0

    def test_zero_contribution_is_noop(self) -> None:
        original = _taxable(100_000, 60_000)
        assert contribute(original, 0) == original


# ---------------------------------------------------------------------------
# withdraw -- pro-rata realised gain
# gain_fraction = (balance - cost_basis) / balance
# ---------------------------------------------------------------------------


class TestWithdrawTaxable:
    def test_pro_rata_gain(self) -> None:
        # balance=100k, basis=60k -> 40% embedded gain
        # withdraw 25k -> realized_gain = 25k * 0.40 = 10k
        # basis_removed = 25k - 10k = 15k -> new basis = 45k
        acct, result = withdraw(_taxable(100_000, 60_000), 25_000)
        assert result.amount_withdrawn == pytest.approx(25_000)
        assert result.realized_gain == pytest.approx(10_000)
        assert result.shortfall == 0.0
        assert acct.balance == pytest.approx(75_000)
        assert acct.cost_basis == pytest.approx(45_000)

    def test_full_liquidation_realizes_all_gain(self) -> None:
        acct, result = withdraw(_taxable(100_000, 60_000), 100_000)
        assert result.realized_gain == pytest.approx(40_000)
        assert acct.balance == pytest.approx(0.0)
        assert acct.cost_basis == pytest.approx(0.0)

    def test_gain_fraction_invariant_after_partial_withdrawal(self) -> None:
        # Pro-rata withdrawals leave the gain fraction unchanged.
        acct, _ = withdraw(_taxable(100_000, 60_000), 25_000)
        assert (acct.balance - acct.cost_basis) / acct.balance == pytest.approx(0.40)

    def test_all_basis_no_gain(self) -> None:
        # basis == balance -> no embedded gain -> withdrawal is pure return of capital
        acct, result = withdraw(_taxable(50_000, 50_000), 10_000)
        assert result.realized_gain == pytest.approx(0.0)
        assert acct.cost_basis == pytest.approx(40_000)

    def test_overdraw_caps_and_reports_shortfall(self) -> None:
        acct, result = withdraw(_taxable(30_000, 20_000), 50_000)
        assert result.amount_withdrawn == pytest.approx(30_000)
        assert result.shortfall == pytest.approx(20_000)
        assert result.realized_gain == pytest.approx(10_000)  # all embedded gain
        assert acct.balance == pytest.approx(0.0)


class TestWithdrawNonTaxable:
    def test_tax_deferred_no_realized_gain(self) -> None:
        acct, result = withdraw(_tax_deferred(80_000), 20_000)
        assert result.amount_withdrawn == pytest.approx(20_000)
        assert result.realized_gain == 0.0
        assert acct.balance == pytest.approx(60_000)

    def test_tax_free_no_realized_gain(self) -> None:
        acct, result = withdraw(_tax_free(80_000), 20_000)
        assert result.amount_withdrawn == pytest.approx(20_000)
        assert result.realized_gain == 0.0
        assert acct.balance == pytest.approx(60_000)


class TestWithdrawEdgeCases:
    def test_withdraw_from_empty_account(self) -> None:
        acct, result = withdraw(_taxable(0, 0), 10_000)
        assert result.amount_withdrawn == 0.0
        assert result.shortfall == pytest.approx(10_000)
        assert acct.balance == 0.0

    def test_zero_withdrawal_is_noop(self) -> None:
        original = _taxable(100_000, 60_000)
        acct, result = withdraw(original, 0)
        assert acct == original
        assert result.amount_withdrawn == 0.0
