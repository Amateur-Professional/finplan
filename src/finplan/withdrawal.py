"""Spending and sourcing policies.

Two orthogonal axes (see CLAUDE.md):

* **Spending policy** -- *how much* to spend each year.  v0 supports only
  fixed-real spending: a constant inflation-adjusted, after-tax target.
* **Sourcing policy** -- *which accounts* fund that spending, and in what order.
  v0 supports conventional ordering and a tax-efficient bracket-aware ordering.

Scope boundary
--------------
This module is *pure*: it allocates a known **gross** dollar amount across the
accounts.  It does not compute taxes and does not solve the after-tax gross-up
(net target -> gross withdrawal), because that requires the year's full tax
context -- Social Security, RMDs, brackets -- which lives in ``simulation.py``.

The tax-efficient policy's only tax awareness is the ``tax_deferred_headroom``
argument: the dollar amount of tax-deferred withdrawal that fits under the
caller's target ordinary-income ceiling (e.g. the top of a low bracket).  The
caller decides that ceiling; we just fill it first.  Keeping the bracket choice
out of this module avoids baking a tax-rule judgement into pure allocation code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from finplan.accounts import Account, withdraw
from finplan.models import AccountType, SourcingPolicy


def fixed_real_spending(
    annual_spending_real: float, cumulative_inflation_factor: float
) -> float:
    """Nominal after-tax spending target for the year under fixed-real policy.

    Spending is constant in real terms, so the nominal target is the real
    target scaled by cumulative inflation since plan start.
    """
    return annual_spending_real * cumulative_inflation_factor


@dataclass(frozen=True)
class _DrawStep:
    """One ordered draw: take up to ``cap`` dollars from ``account_type``."""

    account_type: AccountType
    cap: float = math.inf


def _conventional_steps() -> list[_DrawStep]:
    # Spend taxable first (only the gain is taxed), preserve tax-advantaged
    # compounding, and leave tax-free Roth for last.
    return [
        _DrawStep(AccountType.TAXABLE),
        _DrawStep(AccountType.TAX_DEFERRED),
        _DrawStep(AccountType.TAX_FREE),
    ]


def _tax_efficient_steps(tax_deferred_headroom: float) -> list[_DrawStep]:
    # Fill the caller's low-bracket headroom with tax-deferred dollars first
    # (cheap ordinary income now, smaller forced RMDs later), then spend
    # taxable, then any remaining tax-deferred, and Roth last.
    return [
        _DrawStep(AccountType.TAX_DEFERRED, cap=max(0.0, tax_deferred_headroom)),
        _DrawStep(AccountType.TAXABLE),
        _DrawStep(AccountType.TAX_DEFERRED),
        _DrawStep(AccountType.TAX_FREE),
    ]


@dataclass
class WithdrawalPlan:
    """Result of sourcing a gross amount across the accounts.

    ``accounts`` are the post-withdrawal balances.  The per-type amounts and
    ``realized_gain`` map directly onto ``taxes.compute_year_taxes`` inputs.
    """

    accounts: dict[AccountType, Account]
    withdrawn: dict[AccountType, float]
    realized_gain: float = 0.0
    total_withdrawn: float = 0.0
    shortfall: float = 0.0

    @property
    def withdrawal_taxable(self) -> float:
        return self.withdrawn.get(AccountType.TAXABLE, 0.0)

    @property
    def withdrawal_tax_deferred(self) -> float:
        return self.withdrawn.get(AccountType.TAX_DEFERRED, 0.0)

    @property
    def withdrawal_tax_free(self) -> float:
        return self.withdrawn.get(AccountType.TAX_FREE, 0.0)


def _allocate(
    accounts: dict[AccountType, Account],
    gross_amount: float,
    steps: list[_DrawStep],
) -> WithdrawalPlan:
    """Raise *gross_amount* by walking *steps* in order, capping at balances."""
    working = dict(accounts)
    withdrawn: dict[AccountType, float] = {}
    realized_gain = 0.0
    remaining = max(0.0, gross_amount)

    for step in steps:
        if remaining <= 0.0:
            break
        account = working.get(step.account_type)
        if account is None:
            continue
        target = min(remaining, step.cap)
        if target <= 0.0:
            continue
        new_account, result = withdraw(account, target)
        working[step.account_type] = new_account
        withdrawn[step.account_type] = (
            withdrawn.get(step.account_type, 0.0) + result.amount_withdrawn
        )
        realized_gain += result.realized_gain
        remaining -= result.amount_withdrawn

    total = gross_amount - remaining if gross_amount > 0.0 else 0.0
    return WithdrawalPlan(
        accounts=working,
        withdrawn=withdrawn,
        realized_gain=realized_gain,
        total_withdrawn=total,
        shortfall=remaining if gross_amount > 0.0 else 0.0,
    )


def source_withdrawals(
    accounts: dict[AccountType, Account],
    gross_amount: float,
    sourcing_policy: SourcingPolicy,
    tax_deferred_headroom: float = 0.0,
) -> WithdrawalPlan:
    """Allocate *gross_amount* across *accounts* per the sourcing policy.

    *tax_deferred_headroom* is used only by the tax-efficient policy; it is the
    dollar amount of tax-deferred withdrawal the caller wants taken first (to
    fill a low ordinary-income bracket).  Ignored by the conventional policy.
    """
    if sourcing_policy == SourcingPolicy.TAX_EFFICIENT:
        steps = _tax_efficient_steps(tax_deferred_headroom)
    else:
        steps = _conventional_steps()
    return _allocate(accounts, gross_amount, steps)
