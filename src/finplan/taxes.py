"""Federal income tax, long-term capital gains, standard deduction,
Social Security benefit taxation, and RMD calculations.

All dollar inputs are *nominal* for the simulated year.
Bracket thresholds are inflated from 2025 base values using the
cumulative inflation factor supplied by the caller (Option A).

When unsure about a tax rule, flag it — do not guess.
"""

from __future__ import annotations

from finplan._ss_thresholds import get_ss_tiers
from finplan._tax_tables import (
    get_ltcg_brackets,
    get_ordinary_brackets,
    get_rmd_distribution_period,
    get_standard_deduction,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# A bracket list after inflation scaling uses float thresholds.
_Brackets = list[tuple[float, float]]


def _inflate(brackets: list[tuple[float, int]], factor: float) -> _Brackets:
    """Scale bracket min-income thresholds by cumulative inflation factor."""
    return [(rate, threshold * factor) for rate, threshold in brackets]


def _marginal_tax(income: float, brackets: _Brackets) -> float:
    """Apply marginal-rate brackets to *income*.

    *brackets* must be sorted lowest-to-highest by threshold and must start
    at 0 (i.e., the first entry covers income from 0 up to the next threshold).
    """
    tax = 0.0
    for i, (rate, floor) in enumerate(brackets):
        ceiling = brackets[i + 1][1] if i + 1 < len(brackets) else float("inf")
        amount_in_bracket = max(0.0, min(income, ceiling) - floor)
        tax += rate * amount_in_bracket
    return tax


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_rmd(
    prior_year_end_balance: float,
    age: int,
    rmd_start_age: int = 73,
) -> float:
    """Return the Required Minimum Distribution for the year.

    RMD = prior-year-end balance / Uniform Lifetime Table distribution period.

    SECURE 2.0 (P.L. 117-328, eff. 2023) sets the RMD starting age at 73 for
    people born 1951-1959 and 75 for people born 1960+.  The caller should pass
    the correct *rmd_start_age* based on the planner's birth year; the default
    of 73 covers the most common near-retiree cohort.

    Source: IRS Pub. 590-B; Treas. Reg. §1.401(a)(9)-9.
    """
    if prior_year_end_balance <= 0 or age < rmd_start_age:
        return 0.0
    period = get_rmd_distribution_period(age)
    return prior_year_end_balance / period


def compute_ss_taxable_amount(
    ss_benefit: float,
    other_income: float,
    filing_status: str,
) -> float:
    """Return the taxable portion of Social Security benefits.

    Uses the provisional-income formula from IRS Pub. 915:
        Provisional income = other_income + 0.5 * ss_benefit

    Two-tier system (OBRA 1993, effective 1994, thresholds NOT inflation-indexed):
        Below lower threshold      -> 0% taxable
        Lower threshold to upper   -> up to 50% taxable
        Above upper threshold      -> up to 85% taxable

    *other_income* is AGI excluding SS: all tax-deferred withdrawals plus any
    realized capital gains.  Roth withdrawals are tax-free and excluded.
    """
    if ss_benefit <= 0:
        return 0.0

    tiers = get_ss_tiers(filing_status)
    provisional = other_income + 0.5 * ss_benefit

    lower = tiers[0].combined_income_threshold
    upper = tiers[1].combined_income_threshold  # always present for current law

    if provisional <= lower:
        return 0.0

    if provisional <= upper:
        # Only the 50% tier applies.
        return min(0.50 * ss_benefit, 0.50 * (provisional - lower))

    # 85% tier: the lower band ($lower -> $upper) contributes at 50%.
    lower_band_contribution = 0.50 * (upper - lower)
    return min(
        0.85 * ss_benefit,
        0.85 * (provisional - upper) + lower_band_contribution,
    )


def compute_ordinary_income_tax(
    ordinary_taxable_income: float,
    filing_status: str,
    inflation_factor: float = 1.0,
) -> float:
    """Return federal ordinary income tax on *ordinary_taxable_income*.

    *ordinary_taxable_income* is after deductions and excludes any
    preferentially taxed long-term capital gains.

    Brackets are 2025 nominal values scaled by *inflation_factor*.
    Source: IRS Rev. Proc. 2024-61; IRC §1.
    """
    if ordinary_taxable_income <= 0:
        return 0.0
    brackets = _inflate(get_ordinary_brackets(filing_status), inflation_factor)
    return _marginal_tax(ordinary_taxable_income, brackets)


def compute_ltcg_tax(
    ltcg_amount: float,
    ordinary_taxable_income: float,
    filing_status: str,
    inflation_factor: float = 1.0,
) -> float:
    """Return federal long-term capital gains tax.

    LTCGs are "stacked" on top of ordinary taxable income to determine the
    applicable rate.  The marginal LTCG rate on any dollar of gain is the rate
    that corresponds to where (ordinary_taxable_income + that dollar) falls in
    the LTCG brackets.

    The 3.8% Net Investment Income Tax (IRC §1411) is NOT included here.
    Source: IRS Rev. Proc. 2024-61; IRC §1(h).
    """
    if ltcg_amount <= 0:
        return 0.0
    brackets = _inflate(get_ltcg_brackets(filing_status), inflation_factor)
    # Stacking: tax on (ordinary + LTCG) minus tax on ordinary alone.
    tax_on_total = _marginal_tax(ordinary_taxable_income + ltcg_amount, brackets)
    tax_on_ordinary = _marginal_tax(ordinary_taxable_income, brackets)
    return tax_on_total - tax_on_ordinary


def compute_year_taxes(
    withdrawal_tax_deferred: float,
    ss_income: float,
    ltcg_realized: float,
    filing_status: str,
    inflation_factor: float = 1.0,
) -> dict[str, float]:
    """Compute all federal taxes for one simulation year.

    Income treatment by source
    --------------------------
    withdrawal_tax_deferred  Fully ordinary income (IRC §72).
    ss_income                Partially ordinary per the Pub. 915 formula.
    ltcg_realized            Net realized long-term capital gains from the
                             taxable account; taxed at preferential rates.
    Roth withdrawals         Tax-free — do not pass in.

    Returns
    -------
    Dict with keys:
        ordinary_income_tax   Federal ordinary income tax owed.
        capital_gains_tax     Federal LTCG tax owed.
        total_taxes           Sum of the above.
        ss_taxable_amount     Portion of ss_income included in gross income.
        taxable_income        AGI minus standard deduction (floor 0).
    """
    # 1. SS taxability.
    #    "Other income" for the provisional-income formula includes LTCG because
    #    realized gains are part of AGI.  Source: IRS Pub. 915, Worksheet 1.
    other_income = withdrawal_tax_deferred + ltcg_realized
    ss_taxable = compute_ss_taxable_amount(ss_income, other_income, filing_status)

    # 2. Adjusted Gross Income (no above-the-line deductions modeled in v0).
    agi = withdrawal_tax_deferred + ss_taxable + ltcg_realized

    # 3. Standard deduction inflated from 2025 base.
    std_ded = get_standard_deduction(filing_status) * inflation_factor

    # 4. Taxable income after deduction.
    taxable_income = max(0.0, agi - std_ded)

    # 5. Carve out LTCG from taxable income; ordinary income fills brackets first.
    ltcg_taxable = min(ltcg_realized, taxable_income)
    ordinary_taxable = max(0.0, taxable_income - ltcg_taxable)

    # 6. Compute taxes.
    ordinary_tax = compute_ordinary_income_tax(
        ordinary_taxable, filing_status, inflation_factor
    )
    ltcg_tax = compute_ltcg_tax(
        ltcg_taxable, ordinary_taxable, filing_status, inflation_factor
    )

    return {
        "ordinary_income_tax": round(ordinary_tax, 2),
        "capital_gains_tax": round(ltcg_tax, 2),
        "total_taxes": round(ordinary_tax + ltcg_tax, 2),
        "ss_taxable_amount": round(ss_taxable, 2),
        "taxable_income": round(taxable_income, 2),
    }
