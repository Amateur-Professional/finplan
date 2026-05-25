"""Social Security benefit taxation thresholds (current law).

Combined income = AGI + tax-exempt interest + 50% of SS benefits.
Source: IRS Publication 915; OBRA 1993 (P.L. 103-66), effective 1994.

The dollar thresholds are NOT indexed for inflation — frozen by statute.

Tiers
-----
Lower tier  up to 50% of benefits includable if combined income > threshold.
Upper tier  up to 85% of benefits includable if combined income > threshold.
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
class SSTaxThresholds:
    """SS benefit taxation tiers for a given filing status."""

    single: list[SSTaxTier] = field(default_factory=list)
    mfj: list[SSTaxTier] = field(default_factory=list)


# Current law: OBRA 1993 (P.L. 103-66), effective tax year 1994.
# Thresholds: single $25,000 / $34,000; MFJ $32,000 / $44,000.
SS_TAX_THRESHOLDS = SSTaxThresholds(
    single=[
        SSTaxTier(combined_income_threshold=25_000, max_taxable_fraction=0.50),
        SSTaxTier(combined_income_threshold=34_000, max_taxable_fraction=0.85),
    ],
    mfj=[
        SSTaxTier(combined_income_threshold=32_000, max_taxable_fraction=0.50),
        SSTaxTier(combined_income_threshold=44_000, max_taxable_fraction=0.85),
    ],
)


def get_ss_tiers(filing_status: str) -> list[SSTaxTier]:
    """Return SS taxation tiers for *filing_status* ("single" or "mfj")."""
    return getattr(SS_TAX_THRESHOLDS, filing_status)
