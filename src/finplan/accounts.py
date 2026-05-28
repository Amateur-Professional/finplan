"""Account state and the balance-affecting operations of one simulation year.

This module owns the three account buckets and the *cost-basis tracking* that
makes taxable withdrawals tax-aware.  RMD logic lives in ``taxes.py`` because it
is a tax-table lookup; everything here is pure balance arithmetic.

Design
------
``Account`` is an immutable snapshot of one bucket.  Every operation returns a
*new* ``Account`` rather than mutating in place, so a year's evolution is a
readable chain (grow -> contribute -> withdraw) and each step is trivially
testable.  All dollars are *nominal* for the simulated year.

Cost basis
----------
Taxable withdrawals realise a capital gain equal to the embedded-gain fraction
of the amount withdrawn (the average-cost / pro-rata method):

    gain_fraction = (balance - cost_basis) / balance
    realized_gain = amount_withdrawn * gain_fraction

This is the standard simplification used by retirement calculators; it does not
model specific-lot identification.  Tax-deferred and tax-free buckets do not
track basis: tax-deferred withdrawals are fully ordinary income and tax-free
(Roth) withdrawals are assumed to be qualified distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

from finplan.models import AccountInput, AccountType


@dataclass(frozen=True)
class WithdrawalResult:
    """What a withdrawal produced, for the tax engine to consume."""

    amount_withdrawn: float
    """Cash actually removed (<= the requested amount, capped at balance)."""

    realized_gain: float
    """Capital gain realised by the withdrawal; nonzero only for taxable."""

    shortfall: float
    """Requested minus withdrawn; positive when the bucket ran dry."""


@dataclass(frozen=True)
class Account:
    """Immutable snapshot of one account bucket."""

    account_type: AccountType
    balance: float
    cost_basis: float = 0.0

    @classmethod
    def from_input(cls, account_input: AccountInput) -> Account:
        return cls(
            account_type=account_input.account_type,
            balance=account_input.balance,
            cost_basis=account_input.cost_basis,
        )

    @property
    def unrealized_gain(self) -> float:
        """Embedded gain that would be realised if fully liquidated (taxable only)."""
        if self.account_type != AccountType.TAXABLE:
            return 0.0
        return self.balance - self.cost_basis


def grow(account: Account, return_rate: float) -> Account:
    """Apply a one-year return to the balance.

    Growth is unrealised: the balance moves but cost basis does not, so the
    embedded gain (and thus future tax) grows with the account.
    """
    return Account(
        account.account_type, account.balance * (1.0 + return_rate), account.cost_basis
    )


def contribute(account: Account, amount: float) -> Account:
    """Add a contribution to the balance.

    For a taxable account the contribution is purchased at the current price, so
    it adds to cost basis dollar-for-dollar (no embedded gain on new money).
    """
    if amount <= 0.0:
        return account
    new_balance = account.balance + amount
    if account.account_type == AccountType.TAXABLE:
        return Account(account.account_type, new_balance, account.cost_basis + amount)
    return Account(account.account_type, new_balance, account.cost_basis)


def withdraw(account: Account, amount: float) -> tuple[Account, WithdrawalResult]:
    """Withdraw up to *amount* from the account.

    The withdrawal is capped at the available balance; any unmet request is
    reported as ``shortfall`` so the withdrawal strategy can move to the next
    bucket.  For a taxable account, a pro-rata slice of the embedded gain is
    realised and the cost basis is reduced by the return-of-capital portion.
    """
    if amount <= 0.0 or account.balance <= 0.0:
        return account, WithdrawalResult(
            amount_withdrawn=0.0, realized_gain=0.0, shortfall=max(0.0, amount)
        )

    actual = min(amount, account.balance)
    shortfall = amount - actual

    if account.account_type != AccountType.TAXABLE:
        new_account = Account(
            account.account_type, account.balance - actual, account.cost_basis
        )
        return new_account, WithdrawalResult(
            amount_withdrawn=actual, realized_gain=0.0, shortfall=shortfall
        )

    gain_fraction = (account.balance - account.cost_basis) / account.balance
    realized_gain = actual * gain_fraction
    basis_removed = actual - realized_gain
    new_account = Account(
        account.account_type,
        account.balance - actual,
        account.cost_basis - basis_removed,
    )
    return new_account, WithdrawalResult(
        amount_withdrawn=actual, realized_gain=realized_gain, shortfall=shortfall
    )
