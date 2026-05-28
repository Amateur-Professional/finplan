"""Single-path, year-by-year deterministic projection engine.

This is the spine of the library: it walks one lifetime year by year, applying
contributions, growth, Social Security, RMDs, spending withdrawals, and taxes,
and emits a fully inspectable ``YearDetail`` for every year (transparency
requirement).  ``montecarlo.py`` runs this same path engine N times with
stochastic return/inflation sequences.

Annual convention
------------------
Within each year, *flows happen at the start of the year and growth applies to
the post-flow balance* -- the Bengen/Trinity convention, so the 4%-rule
benchmark reproduces.  With no flows, a balance compounds as ``B*(1+r)^n``
exactly.

Real vs nominal
---------------
``ReturnAssumptions`` holds *real* returns and inflation is modelled separately,
so the nominal return applied in year *k* is ``(1+real_k)*(1+infl_k) - 1``.  All
balances and dollar amounts in ``YearDetail`` are nominal for that year.  Real
spending and the tax brackets are scaled by cumulative inflation since plan
start (the plan-start year is the indexing base; the embedded 2025 tables are
treated as that base -- a v0 simplification).

The after-tax gross-up
-----------------------
Spending is an *after-tax* target.  Because withdrawals create taxable income,
the gross withdrawal needed to net the target is solved by fixed-point
iteration: each step adds the remaining net shortfall to the gross draw, which
converges geometrically since the marginal tax rate is < 1.
"""

from __future__ import annotations

from collections.abc import Sequence

from finplan._tax_tables import get_ordinary_brackets, get_standard_deduction
from finplan.accounts import Account, contribute, grow, withdraw
from finplan.models import (
    AccountBalances,
    AccountType,
    FilingStatus,
    PlanInput,
    SourcingPolicy,
    TrialResult,
    YearDetail,
)
from finplan.taxes import (
    build_year_tax_context,
    compute_rmd,
    compute_ss_taxable_amount,
    compute_year_taxes_ctx,
)
from finplan.withdrawal import allocate, build_draw_steps, fixed_real_spending

# Map the schema's filing-status enum onto the tax-table keys.
_FILING_KEY: dict[FilingStatus, str] = {
    FilingStatus.SINGLE: "single",
    FilingStatus.MARRIED_FILING_JOINTLY: "mfj",
}

_GROSS_UP_TOLERANCE = 0.01  # dollars
_MAX_GROSS_UP_ITERS = 200


def _rmd_start_age(birth_year: int) -> int:
    """SECURE 2.0 RMD start age: 73 for 1951-1959, 75 for 1960+."""
    return 75 if birth_year >= 1960 else 73


def _initial_accounts(plan: PlanInput) -> dict[AccountType, Account]:
    """Collapse the input account list into one bucket per account type."""
    accounts = {t: Account(t, balance=0.0, cost_basis=0.0) for t in AccountType}
    for ai in plan.accounts:
        current = accounts[ai.account_type]
        accounts[ai.account_type] = Account(
            ai.account_type,
            balance=current.balance + ai.balance,
            cost_basis=current.cost_basis + ai.cost_basis,
        )
    return accounts


def _twelve_pct_ceiling(filing_key: str) -> float:
    """Top of the 12% ordinary bracket (== where the 22% bracket begins)."""
    for rate, threshold in get_ordinary_brackets(filing_key):
        if rate == 0.22:
            return float(threshold)
    return float("inf")


def _tax_deferred_headroom(
    rmd_taxable: float, ss_gross: float, filing_key: str, inflation_factor: float
) -> float:
    """Discretionary tax-deferred dollars that still fit under the 12% bracket.

    HEURISTIC (flagged): the tax-efficient sourcing policy fills ordinary income
    up to the top of the 12% bracket with tax-deferred withdrawals before
    tapping taxable accounts -- the common "fill the 12% bracket" rule, which
    smooths income and shrinks future RMDs.  This is an *ordering* heuristic, not
    a true lifetime-tax optimiser (out of v0 scope).  It approximates the
    bracket room by ignoring the feedback of the extra withdrawal on Social
    Security taxability; the exact tax is still computed afterward.
    """
    ceiling = _twelve_pct_ceiling(filing_key) * inflation_factor
    std_ded = get_standard_deduction(filing_key) * inflation_factor
    ss_taxable = compute_ss_taxable_amount(ss_gross, rmd_taxable, filing_key)
    existing_ordinary_taxable = max(0.0, rmd_taxable + ss_taxable - std_ded)
    return max(0.0, ceiling - existing_ordinary_taxable)


def _portfolio_real_return(plan: PlanInput) -> float:
    r = plan.returns
    return (
        r.equity_allocation * r.equity_real_return
        + (1.0 - r.equity_allocation) * r.bond_real_return
    )


def simulate_deterministic(plan: PlanInput) -> TrialResult:
    """Run the single mean-return, mean-inflation path (no randomness)."""
    n_years = plan.plan_end_year - plan.plan_start_year + 1
    real_returns = [_portfolio_real_return(plan)] * n_years
    inflation = [plan.inflation.mean] * n_years
    return simulate_path(plan, real_returns, inflation)


def simulate_path(
    plan: PlanInput,
    portfolio_real_returns: Sequence[float],
    inflation_rates: Sequence[float],
) -> TrialResult:
    """Walk one path given per-year portfolio real returns and inflation rates.

    Both sequences must have length ``plan_end_year - plan_start_year + 1``.
    ``montecarlo.py`` supplies stochastic sequences; the deterministic wrapper
    supplies constant ones.
    """
    n_years = plan.plan_end_year - plan.plan_start_year + 1
    if len(portfolio_real_returns) != n_years or len(inflation_rates) != n_years:
        raise ValueError("return/inflation sequences must have length n_years")

    filing_key = _FILING_KEY[plan.filing_status]
    rmd_start_age = _rmd_start_age(plan.birth_year)

    accounts = _initial_accounts(plan)
    cumulative_inflation = 1.0  # price level at start of the current year
    year_details: list[YearDetail] = []
    all_retirement_years_funded = True

    for k in range(n_years):
        year = plan.plan_start_year + k
        age = year - plan.birth_year
        real_return = portfolio_real_returns[k]
        inflation_rate = inflation_rates[k]
        nominal_return = (1.0 + real_return) * (1.0 + inflation_rate) - 1.0
        is_retired = year >= plan.retirement_year

        if is_retired:
            detail, accounts, funded = _retirement_year(
                plan=plan,
                accounts=accounts,
                year=year,
                age=age,
                nominal_return=nominal_return,
                inflation_rate=inflation_rate,
                inflation_factor=cumulative_inflation,
                rmd_start_age=rmd_start_age,
                filing_key=filing_key,
            )
            all_retirement_years_funded = all_retirement_years_funded and funded
        else:
            detail, accounts = _working_year(
                plan=plan,
                accounts=accounts,
                year=year,
                age=age,
                nominal_return=nominal_return,
                inflation_rate=inflation_rate,
            )

        year_details.append(detail)
        cumulative_inflation *= 1.0 + inflation_rate

    final_wealth = year_details[-1].total_wealth
    return TrialResult(
        trial_index=0,
        success=all_retirement_years_funded,
        year_details=year_details,
        final_wealth=final_wealth,
    )


def _balances(accounts: dict[AccountType, Account]) -> AccountBalances:
    return AccountBalances(
        taxable=accounts[AccountType.TAXABLE].balance,
        tax_deferred=accounts[AccountType.TAX_DEFERRED].balance,
        tax_free=accounts[AccountType.TAX_FREE].balance,
    )


def _grow_all(
    accounts: dict[AccountType, Account], nominal_return: float
) -> dict[AccountType, Account]:
    return {t: grow(a, nominal_return) for t, a in accounts.items()}


def _working_year(
    plan: PlanInput,
    accounts: dict[AccountType, Account],
    year: int,
    age: int,
    nominal_return: float,
    inflation_rate: float,
) -> tuple[YearDetail, dict[AccountType, Account]]:
    """Pre-retirement year: apply nominal contributions, then grow."""
    accounts = dict(accounts)
    contributions = {t: 0.0 for t in AccountType}
    for account_type, amount in plan.annual_contributions.items():
        if amount > 0.0:
            accounts[account_type] = contribute(accounts[account_type], amount)
            contributions[account_type] = amount

    accounts = _grow_all(accounts, nominal_return)
    balances = _balances(accounts)
    detail = YearDetail(
        year=year,
        age=age,
        portfolio_return=nominal_return,
        inflation_rate=inflation_rate,
        contribution_taxable=contributions[AccountType.TAXABLE],
        contribution_tax_deferred=contributions[AccountType.TAX_DEFERRED],
        contribution_tax_free=contributions[AccountType.TAX_FREE],
        balances=balances,
        total_wealth=balances.total,
    )
    return detail, accounts


def _retirement_year(
    plan: PlanInput,
    accounts: dict[AccountType, Account],
    year: int,
    age: int,
    nominal_return: float,
    inflation_rate: float,
    inflation_factor: float,
    rmd_start_age: int,
    filing_key: str,
) -> tuple[YearDetail, dict[AccountType, Account], bool]:
    """Retirement year: SS + forced RMD, gross-up to fund after-tax spending."""
    accounts = dict(accounts)

    # 1. Social Security (nominal) once the claim age is reached.
    ss_gross = 0.0
    if plan.social_security is not None and age >= plan.social_security.claim_age:
        ss_gross = plan.social_security.annual_benefit * inflation_factor

    # 2. Forced RMD from the tax-deferred bucket (based on start-of-year balance).
    rmd_due = compute_rmd(
        accounts[AccountType.TAX_DEFERRED].balance, age, rmd_start_age
    )
    accounts[AccountType.TAX_DEFERRED], rmd_result = withdraw(
        accounts[AccountType.TAX_DEFERRED], rmd_due
    )
    rmd_withdrawn = rmd_result.amount_withdrawn

    # 3. After-tax spending target for the year.
    net_target = fixed_real_spending(plan.annual_spending_real, inflation_factor)

    # 4. Tax-efficient headroom (ignored by the conventional policy).
    headroom = 0.0
    if plan.sourcing_policy == SourcingPolicy.TAX_EFFICIENT:
        total_headroom = _tax_deferred_headroom(
            rmd_withdrawn, ss_gross, filing_key, inflation_factor
        )
        headroom = max(0.0, total_headroom - rmd_withdrawn)

    # 5. Gross-up: solve discretionary withdrawals so net cash meets the target.
    #    The year's inflation factor is constant, so inflate the tax tables once
    #    and reuse them across every iteration (compute_year_taxes_ctx).
    tax_ctx = build_year_tax_context(filing_key, inflation_factor)
    draw_steps = build_draw_steps(plan.sourcing_policy, headroom)
    g = max(0.0, net_target - ss_gross - rmd_withdrawn)
    plan_result = allocate(accounts, 0.0, draw_steps)
    tax = compute_year_taxes_ctx(rmd_withdrawn, ss_gross, 0.0, tax_ctx)
    net = ss_gross + rmd_withdrawn - tax.total_taxes

    for _ in range(_MAX_GROSS_UP_ITERS):
        plan_result = allocate(accounts, g, draw_steps)
        td_total = rmd_withdrawn + plan_result.withdrawal_tax_deferred
        tax = compute_year_taxes_ctx(
            td_total, ss_gross, plan_result.realized_gain, tax_ctx
        )
        gross_cash = ss_gross + rmd_withdrawn + plan_result.total_withdrawn
        net = gross_cash - tax.total_taxes
        gap = net_target - net
        if gap <= _GROSS_UP_TOLERANCE:
            break
        if plan_result.shortfall > _GROSS_UP_TOLERANCE:
            break  # accounts drained; cannot raise more
        g += gap

    funded = (net_target - net) <= _GROSS_UP_TOLERANCE
    accounts = dict(plan_result.accounts)

    # 6. Reinvest any surplus cash (e.g. RMD exceeding need) into the taxable bucket.
    contribution_taxable = 0.0
    surplus = net - net_target
    if surplus > _GROSS_UP_TOLERANCE:
        accounts[AccountType.TAXABLE] = contribute(
            accounts[AccountType.TAXABLE], surplus
        )
        contribution_taxable = surplus

    # 7. Grow post-flow balances for the rest of the year.
    accounts = _grow_all(accounts, nominal_return)
    balances = _balances(accounts)

    td_total = rmd_withdrawn + plan_result.withdrawal_tax_deferred
    detail = YearDetail(
        year=year,
        age=age,
        portfolio_return=nominal_return,
        inflation_rate=inflation_rate,
        social_security_income=ss_gross,
        rmd_amount=rmd_withdrawn,
        withdrawal_taxable=plan_result.withdrawal_taxable,
        withdrawal_tax_deferred=td_total,
        withdrawal_tax_free=plan_result.withdrawal_tax_free,
        contribution_taxable=contribution_taxable,
        ordinary_income_tax=round(tax.ordinary_income_tax, 2),
        capital_gains_tax=round(tax.capital_gains_tax, 2),
        total_taxes=round(tax.total_taxes, 2),
        balances=balances,
        total_wealth=balances.total,
    )
    return detail, accounts, funded
