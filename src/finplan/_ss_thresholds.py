"""Social Security benefit taxation thresholds, indexed by effective tax year.

Combined income = AGI + tax-exempt interest + 50% of SS benefits.
Source: IRS Publication 915 (Social Security and Equivalent Railroad
Retirement Benefits).

History
-------
1935-1983  Social Security Act enacted 1935; benefits first paid 1940.
           No federal income tax on SS benefits during this period.
1984-1993  TEFRA 1983 (P.L. 98-21): up to 50% of benefits includable in gross
           income if combined income exceeds the lower threshold.
1994-pres. OBRA 1993 (P.L. 103-66): added an 85% tier for combined income
           above the upper threshold. Lower thresholds unchanged from 1984.

The dollar thresholds are NOT indexed for inflation — they are frozen by statute.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SSTaxTier:
    """One combined-income tier for SS benefit taxation."""

    combined_income_threshold: float
    """Combined income floor at which this tier applies."""

    max_taxable_fraction: float
    """Maximum fraction of SS benefit includable in gross income at this tier."""


@dataclass(frozen=True)
class SSTaxRule:
    """SS benefit taxation rule covering a range of tax years.

    ``year_through=None`` means the rule applies through the present.
    An empty tier list means SS benefits are not taxed at all.
    """

    year_from: int
    year_through: int | None
    single: list[SSTaxTier] = field(default_factory=list)
    mfj: list[SSTaxTier] = field(default_factory=list)
    note: str = ""


# Ordered from earliest to most recent.  ``get_ss_tax_rule`` searches in reverse.
SS_TAX_RULES: list[SSTaxRule] = [
    SSTaxRule(
        year_from=1935,
        year_through=1983,
        single=[],
        mfj=[],
        note=(
            "Social Security Act signed 14 Aug 1935; first monthly benefits paid"
            " Jan 1940. No federal income tax on SS benefits. TEFRA 1983"
            " (P.L. 98-21) introduced SS taxation effective tax year 1984."
        ),
    ),
    SSTaxRule(
        year_from=1984,
        year_through=1993,
        single=[
            SSTaxTier(combined_income_threshold=25_000, max_taxable_fraction=0.50),
        ],
        mfj=[
            SSTaxTier(combined_income_threshold=32_000, max_taxable_fraction=0.50),
        ],
        note=(
            "TEFRA 1983 (P.L. 98-21), effective tax year 1984. Up to 50% of"
            " benefits includable in gross income if combined income exceeds"
            " threshold. Thresholds: single $25,000; MFJ $32,000."
            " Source: IRS Pub. 915."
        ),
    ),
    SSTaxRule(
        year_from=1994,
        year_through=None,  # current law as of 2025
        single=[
            # Lower tier: same as 1984 law
            SSTaxTier(combined_income_threshold=25_000, max_taxable_fraction=0.50),
            # Upper tier: added by OBRA 1993
            SSTaxTier(combined_income_threshold=34_000, max_taxable_fraction=0.85),
        ],
        mfj=[
            SSTaxTier(combined_income_threshold=32_000, max_taxable_fraction=0.50),
            SSTaxTier(combined_income_threshold=44_000, max_taxable_fraction=0.85),
        ],
        note=(
            "OBRA 1993 (P.L. 103-66), effective tax year 1994. Added 85% tier"
            " for combined income above upper threshold; lower thresholds"
            " unchanged from 1984. Thresholds: single $25k/$34k; MFJ $32k/$44k."
            " Thresholds are NOT inflation-indexed (frozen by statute)."
            " Source: IRS Pub. 915."
        ),
    ),
]


def get_ss_tax_rule(year: int) -> SSTaxRule:
    """Return the SS taxation rule applicable for *year*.

    Raises ``ValueError`` if *year* predates the Social Security Act (1935).
    """
    for rule in reversed(SS_TAX_RULES):
        if year >= rule.year_from and (
            rule.year_through is None or year <= rule.year_through
        ):
            return rule
    raise ValueError(f"No SS tax rule found for year {year} (SS Act enacted 1935)")
