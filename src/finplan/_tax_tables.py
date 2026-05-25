"""Embedded federal tax tables for the finplan engine.

All dollar amounts are 2025 nominal values.
The engine inflates these brackets forward using simulated CPI each year (Option A).

Sources
-------
ORDINARY_BRACKETS   IRS Rev. Proc. 2024-61 (tax year 2025).
                    Rates: 10%, 12%, 22%, 24%, 32%, 35%, 37%.

STANDARD_DEDUCTION  IRS Rev. Proc. 2024-61 (tax year 2025).

LTCG_BRACKETS       IRS Rev. Proc. 2024-61 (tax year 2025).
                    Rates: 0% / 15% / 20%.
                    The 3.8% NIIT (IRC §1411) is NOT included here;
                    it is applied separately in taxes.py.

UNIFORM_LIFETIME_TABLE  IRS Publication 590-B (2025), Appendix B, Table III.
                    Effective for distribution calendar years on or after
                    Jan 1, 2022 per Treas. Reg. §1.401(a)(9)-9 (TD 9930).
                    Used by IRA *owners* to calculate RMDs; NOT the Single
                    Life Expectancy table (Table I, for beneficiaries).

Format
------
ORDINARY_BRACKETS and LTCG_BRACKETS
    {"single": [(rate, min_income), ...], "mfj": [...]}
    Brackets are ordered lowest-to-highest by min_income.
    rate is a decimal (0.10, not 10.0).

STANDARD_DEDUCTION
    {"single": amount, "mfj": amount}

UNIFORM_LIFETIME_TABLE
    {age: distribution_period}
    Ages 72-120.  Use age 120 for any age >= 120.
    RMD = prior_year_end_balance / distribution_period
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 2025 Ordinary income brackets
# Source: IRS Rev. Proc. 2024-61
# ---------------------------------------------------------------------------
ORDINARY_BRACKETS: dict[str, list[tuple[float, int]]] = {
    "single": [
        (0.10, 0),
        (0.12, 11_925),
        (0.22, 48_475),
        (0.24, 103_350),
        (0.32, 197_300),
        (0.35, 250_525),
        (0.37, 626_350),
    ],
    "mfj": [
        (0.10, 0),
        (0.12, 23_850),
        (0.22, 96_950),
        (0.24, 206_700),
        (0.32, 394_600),
        (0.35, 501_050),
        (0.37, 751_600),
    ],
}

# ---------------------------------------------------------------------------
# 2025 Standard deduction
# Source: IRS Rev. Proc. 2024-61
# ---------------------------------------------------------------------------
STANDARD_DEDUCTION: dict[str, int] = {
    "single": 15_750,
    "mfj": 31_500,
}

# ---------------------------------------------------------------------------
# 2025 Long-term capital gains brackets
# Source: IRS Rev. Proc. 2024-61
# ---------------------------------------------------------------------------
LTCG_BRACKETS: dict[str, list[tuple[float, int]]] = {
    "single": [
        (0.00, 0),
        (0.15, 48_350),
        (0.20, 533_400),
    ],
    "mfj": [
        (0.00, 0),
        (0.15, 96_700),
        (0.20, 600_050),
    ],
}

# ---------------------------------------------------------------------------
# Uniform Lifetime Table (IRS Pub. 590-B, Appendix B, Table III)
# Effective Jan 1, 2022. Used by IRA owners for RMD calculations.
# Source: Treas. Reg. §1.401(a)(9)-9, TD 9930 (Nov 2020).
# Spot-check: age 75 -> 24.6 (confirmed in IRS Pub. 590-B 2025 example).
# ---------------------------------------------------------------------------
UNIFORM_LIFETIME_TABLE: dict[int, float] = {
    72: 27.4,
    73: 26.5,
    74: 25.5,
    75: 24.6,
    76: 23.7,
    77: 22.9,
    78: 22.0,
    79: 21.1,
    80: 20.2,
    81: 19.4,
    82: 18.5,
    83: 17.7,
    84: 16.8,
    85: 16.0,
    86: 15.2,
    87: 14.4,
    88: 13.7,
    89: 12.9,
    90: 12.2,
    91: 11.5,
    92: 10.8,
    93: 10.1,
    94: 9.5,
    95: 8.9,
    96: 8.4,
    97: 7.8,
    98: 7.3,
    99: 6.8,
    100: 6.4,
    101: 6.0,
    102: 5.6,
    103: 5.2,
    104: 4.9,
    105: 4.6,
    106: 4.3,
    107: 4.1,
    108: 3.9,
    109: 3.7,
    110: 3.5,
    111: 3.4,
    112: 3.3,
    113: 3.1,
    114: 3.0,
    115: 2.9,
    116: 2.8,
    117: 2.7,
    118: 2.5,
    119: 2.3,
    120: 2.0,  # 120+ uses 2.0 per the table
}

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

FilingStatusKey = str  # "single" or "mfj"
BracketList = list[tuple[float, int]]  # [(rate, min_income), ...]


def get_ordinary_brackets(filing_status: FilingStatusKey) -> BracketList:
    """Return 2025 ordinary income brackets for *filing_status*."""
    return ORDINARY_BRACKETS[filing_status]


def get_standard_deduction(filing_status: FilingStatusKey) -> int:
    """Return the 2025 standard deduction for *filing_status*."""
    return STANDARD_DEDUCTION[filing_status]


def get_ltcg_brackets(filing_status: FilingStatusKey) -> BracketList:
    """Return 2025 LTCG brackets for *filing_status*."""
    return LTCG_BRACKETS[filing_status]


def get_rmd_distribution_period(age: int) -> float:
    """Return the Uniform Lifetime Table distribution period for *age*.

    Ages below 72 return ``float("inf")`` (no RMD required).
    Ages above 120 use 2.0 per the table.
    """
    if age < 72:
        return float("inf")
    return UNIFORM_LIFETIME_TABLE.get(min(age, 120), 2.0)
