"""Tests for taxes.py.

All expected values are hand-calculated against published 2025 federal tax
parameters (IRS Rev. Proc. 2024-61) and IRS Pub. 915 / 590-B formulas.
No mocks -- every assertion must be derivable from first principles.
"""

import pytest

from finplan.taxes import (
    compute_ltcg_tax,
    compute_ordinary_income_tax,
    compute_rmd,
    compute_ss_taxable_amount,
    compute_year_taxes,
)

# ---------------------------------------------------------------------------
# compute_ordinary_income_tax -- 2025 single brackets
# 10%: $0-$11,925 | 12%: $11,925-$48,475 | 22%: $48,475-$103,350
# 24%: $103,350-$197,300 | 32%: $197,300-$250,525 | 35%: $250,525-$626,350
# 37%: $626,350+
# ---------------------------------------------------------------------------


class TestOrdinaryIncomeTaxSingle:
    def test_zero_income(self) -> None:
        assert compute_ordinary_income_tax(0, "single") == 0.0

    def test_negative_income(self) -> None:
        assert compute_ordinary_income_tax(-1_000, "single") == 0.0

    def test_exactly_top_of_10pct_bracket(self) -> None:
        # 10% x $11,925 = $1,192.50
        assert compute_ordinary_income_tax(11_925, "single") == pytest.approx(1_192.50)

    def test_in_12pct_bracket(self) -> None:
        # 10% x $11,925 + 12% x ($30,000 - $11,925)
        # = $1,192.50 + 12% x $18,075 = $1,192.50 + $2,169.00 = $3,361.50
        assert compute_ordinary_income_tax(30_000, "single") == pytest.approx(3_361.50)

    def test_in_22pct_bracket(self) -> None:
        # 10% x $11,925 + 12% x $36,550 + 22% x ($50,000 - $48,475)
        # = $1,192.50 + $4,386.00 + 22% x $1,525
        # = $1,192.50 + $4,386.00 + $335.50 = $5,914.00
        assert compute_ordinary_income_tax(50_000, "single") == pytest.approx(5_914.00)

    def test_deep_in_22pct_bracket(self) -> None:
        # 10% x $11,925 + 12% x $36,550 + 22% x ($100,000 - $48,475)
        # = $1,192.50 + $4,386.00 + 22% x $51,525
        # = $1,192.50 + $4,386.00 + $11,335.50 = $16,914.00
        assert compute_ordinary_income_tax(100_000, "single") == pytest.approx(
            16_914.00
        )


# ---------------------------------------------------------------------------
# compute_ordinary_income_tax -- 2025 MFJ brackets
# 10%: $0-$23,850 | 12%: $23,850-$96,950 | 22%: $96,950-$206,700
# ---------------------------------------------------------------------------


class TestOrdinaryIncomeTaxMFJ:
    def test_in_22pct_bracket(self) -> None:
        # 10% x $23,850 + 12% x ($96,950 - $23,850) + 22% x ($100,000 - $96,950)
        # = $2,385.00 + 12% x $73,100 + 22% x $3,050
        # = $2,385.00 + $8,772.00 + $671.00 = $11,828.00
        assert compute_ordinary_income_tax(100_000, "mfj") == pytest.approx(11_828.00)

    def test_mfj_lower_than_single_same_income(self) -> None:
        # MFJ brackets are wider so tax should always be <= single at same income.
        assert compute_ordinary_income_tax(
            150_000, "mfj"
        ) < compute_ordinary_income_tax(150_000, "single")


# ---------------------------------------------------------------------------
# compute_ordinary_income_tax -- inflation factor
# ---------------------------------------------------------------------------


class TestOrdinaryIncomeTaxInflation:
    def test_inflation_scales_brackets(self) -> None:
        # With factor=2.0 all thresholds double; 12% bracket starts at $23,850.
        # So $23,850 sits in the 10% bracket under inflation_factor=2.0.
        # Inflated: 10% x $23,850 = $2,385.00
        # Base:     10% x $11,925 + 12% x ($23,850 - $11,925)
        #         = $1,192.50 + $1,431.00 = $2,623.50
        assert compute_ordinary_income_tax(
            23_850, "single", inflation_factor=2.0
        ) == pytest.approx(2_385.00)
        assert compute_ordinary_income_tax(
            23_850, "single", inflation_factor=1.0
        ) == pytest.approx(2_623.50)


# ---------------------------------------------------------------------------
# compute_ss_taxable_amount
# Provisional income (PI) = other_income + 0.5 x ss_benefit
# Single thresholds: $25,000 (50% tier) / $34,000 (85% tier)
# MFJ thresholds:    $32,000 (50% tier) / $44,000 (85% tier)
# ---------------------------------------------------------------------------


class TestSSTaxableSingle:
    SS = 20_000.0

    def test_below_lower_threshold(self) -> None:
        # PI = $10,000 + $10,000 = $20,000 < $25,000 -> $0 taxable
        result = compute_ss_taxable_amount(
            self.SS, other_income=10_000, filing_status="single"
        )
        assert result == 0.0

    def test_exactly_at_lower_threshold(self) -> None:
        # PI = $15,000 + $10,000 = $25,000 -> $0 taxable (not strictly above)
        result = compute_ss_taxable_amount(
            self.SS, other_income=15_000, filing_status="single"
        )
        assert result == 0.0

    def test_in_50pct_tier(self) -> None:
        # PI = $20,000 + $10,000 = $30,000 (between $25,000 and $34,000)
        # min(50% x $20,000, 50% x ($30,000 - $25,000)) = min($10,000, $2,500) = $2,500
        result = compute_ss_taxable_amount(
            self.SS, other_income=20_000, filing_status="single"
        )
        assert result == pytest.approx(2_500.0)

    def test_in_85pct_tier_partial(self) -> None:
        # PI = $25,000 + $10,000 = $35,000 (just above $34,000)
        # min(85% x $20,000, 85% x $1,000 + 50% x $9,000)
        # = min($17,000, $850 + $4,500) = $5,350
        result = compute_ss_taxable_amount(
            self.SS, other_income=25_000, filing_status="single"
        )
        assert result == pytest.approx(5_350.0)

    def test_in_85pct_tier_maxed(self) -> None:
        # PI = $40,000 + $10,000 = $50,000 (well above $34,000)
        # min(85% x $20,000, 85% x $16,000 + 50% x $9,000)
        # = min($17,000, $13,600 + $4,500) = min($17,000, $18,100) = $17,000
        result = compute_ss_taxable_amount(
            self.SS, other_income=40_000, filing_status="single"
        )
        assert result == pytest.approx(17_000.0)

    def test_zero_benefit(self) -> None:
        result = compute_ss_taxable_amount(
            0, other_income=100_000, filing_status="single"
        )
        assert result == 0.0


class TestSSTaxableMFJ:
    SS = 30_000.0

    def test_in_85pct_tier_maxed(self) -> None:
        # PI = $50,000 + $15,000 = $65,000 (above $44,000)
        # lower band = 50% x ($44,000 - $32,000) = $6,000
        # min(85% x $30,000, 85% x $21,000 + $6,000)
        # = min($25,500, $17,850 + $6,000) = min($25,500, $23,850) = $23,850
        result = compute_ss_taxable_amount(
            self.SS, other_income=50_000, filing_status="mfj"
        )
        assert result == pytest.approx(23_850.0)


# ---------------------------------------------------------------------------
# compute_ltcg_tax
# 2025 single LTCG: 0% up to $48,350 | 15% up to $533,400 | 20% above
# Stacking: LTCG sits on top of ordinary taxable income.
# ---------------------------------------------------------------------------


class TestLTCGTax:
    def test_no_ltcg(self) -> None:
        assert (
            compute_ltcg_tax(0, ordinary_taxable_income=50_000, filing_status="single")
            == 0.0
        )

    def test_all_in_zero_pct_bracket(self) -> None:
        # ordinary=$20,000 + LTCG=$10,000 = $30,000 < $48,350 -> 0% tax
        result = compute_ltcg_tax(
            10_000, ordinary_taxable_income=20_000, filing_status="single"
        )
        assert result == pytest.approx(0.0)

    def test_ltcg_straddles_zero_and_15pct(self) -> None:
        # ordinary=$40,000; LTCG=$10,000; total=$50,000
        # 0% bracket: $40,000 to $48,350 -> $8,350 at 0%
        # 15% bracket: $48,350 to $50,000 -> $1,650 at 15% = $247.50
        result = compute_ltcg_tax(
            10_000, ordinary_taxable_income=40_000, filing_status="single"
        )
        assert result == pytest.approx(247.50)

    def test_ltcg_fully_in_15pct_bracket(self) -> None:
        # ordinary=$50,000 (already above $48,350); LTCG=$10,000
        # All $10,000 stacked above $48,350 -> 15% x $10,000 = $1,500
        result = compute_ltcg_tax(
            10_000, ordinary_taxable_income=50_000, filing_status="single"
        )
        assert result == pytest.approx(1_500.0)

    def test_ltcg_zero_ordinary(self) -> None:
        # ordinary=$0; LTCG=$100,000
        # 0% x $48,350 + 15% x ($100,000 - $48,350) = 15% x $51,650 = $7,747.50
        result = compute_ltcg_tax(
            100_000, ordinary_taxable_income=0, filing_status="single"
        )
        assert result == pytest.approx(7_747.50)

    def test_mfj_all_in_zero_pct(self) -> None:
        # MFJ 0% threshold = $96,700; ordinary=$80,000; LTCG=$10,000 -> all at 0%
        result = compute_ltcg_tax(
            10_000, ordinary_taxable_income=80_000, filing_status="mfj"
        )
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_rmd
# Uniform Lifetime Table (IRS Pub. 590-B, Table III, effective 2022)
# ---------------------------------------------------------------------------


class TestRMD:
    def test_below_rmd_age(self) -> None:
        assert compute_rmd(500_000, age=72) == 0.0

    def test_exactly_at_rmd_age(self) -> None:
        # age=73: distribution period = 26.5; RMD = $500,000 / 26.5
        assert compute_rmd(500_000, age=73) == pytest.approx(500_000 / 26.5)

    def test_age_75_matches_irs_example(self) -> None:
        # IRS Pub. 590-B (2025) example: age 75 -> period 24.6
        assert compute_rmd(500_000, age=75) == pytest.approx(500_000 / 24.6)

    def test_age_80(self) -> None:
        # period = 20.2
        assert compute_rmd(1_000_000, age=80) == pytest.approx(1_000_000 / 20.2)

    def test_zero_balance(self) -> None:
        assert compute_rmd(0, age=80) == 0.0

    def test_custom_rmd_start_age_for_1960_cohort(self) -> None:
        # SECURE 2.0: people born 1960+ have RMD start age of 75.
        assert compute_rmd(500_000, age=74, rmd_start_age=75) == 0.0
        assert compute_rmd(500_000, age=75, rmd_start_age=75) == pytest.approx(
            500_000 / 24.6
        )


# ---------------------------------------------------------------------------
# compute_year_taxes -- integration
# ---------------------------------------------------------------------------


class TestComputeYearTaxes:
    def test_no_income_no_tax(self) -> None:
        result = compute_year_taxes(
            withdrawal_tax_deferred=0,
            ss_income=0,
            ltcg_realized=0,
            filing_status="single",
        )
        assert result["total_taxes"] == 0.0

    def test_only_tax_deferred_withdrawal(self) -> None:
        # withdrawal=$50,000, no SS, no LTCG, single
        # AGI = $50,000; std_ded = $15,750; taxable = $34,250
        # 10% x $11,925 + 12% x ($34,250 - $11,925)
        # = $1,192.50 + 12% x $22,325 = $1,192.50 + $2,679.00 = $3,871.50
        result = compute_year_taxes(
            withdrawal_tax_deferred=50_000,
            ss_income=0,
            ltcg_realized=0,
            filing_status="single",
        )
        assert result["ordinary_income_tax"] == pytest.approx(3_871.50)
        assert result["capital_gains_tax"] == 0.0
        assert result["ss_taxable_amount"] == 0.0
        assert result["taxable_income"] == pytest.approx(34_250.0)

    def test_ss_partially_taxable(self) -> None:
        # withdrawal=$20,000, SS=$20,000, single
        # PI = $20,000 (other) + $10,000 (50% SS) = $30,000 -> 50% tier
        # ss_taxable = min($10,000, 50% x $5,000) = $2,500
        # AGI = $20,000 + $2,500 = $22,500
        # taxable = $22,500 - $15,750 = $6,750
        # tax = 10% x $6,750 = $675
        result = compute_year_taxes(
            withdrawal_tax_deferred=20_000,
            ss_income=20_000,
            ltcg_realized=0,
            filing_status="single",
        )
        assert result["ss_taxable_amount"] == pytest.approx(2_500.0)
        assert result["taxable_income"] == pytest.approx(6_750.0)
        assert result["ordinary_income_tax"] == pytest.approx(675.0)

    def test_ltcg_sheltered_by_standard_deduction(self) -> None:
        # withdrawal=$0, SS=$0, LTCG=$10,000, single
        # AGI = $10,000; std_ded = $15,750; taxable = $0 -> no tax
        result = compute_year_taxes(
            withdrawal_tax_deferred=0,
            ss_income=0,
            ltcg_realized=10_000,
            filing_status="single",
        )
        assert result["total_taxes"] == 0.0
        assert result["taxable_income"] == 0.0

    def test_mixed_income_types(self) -> None:
        # withdrawal=$40,000, SS=$0, LTCG=$10,000, single
        # AGI = $50,000; std_ded = $15,750; taxable = $34,250
        # ltcg_taxable = $10,000; ordinary_taxable = $24,250
        # ordinary tax: 10% x $11,925 + 12% x ($24,250 - $11,925)
        #             = $1,192.50 + 12% x $12,325 = $1,192.50 + $1,479.00 = $2,671.50
        # LTCG: stacked above $24,250; total $34,250 < $48,350 -> 0%
        result = compute_year_taxes(
            withdrawal_tax_deferred=40_000,
            ss_income=0,
            ltcg_realized=10_000,
            filing_status="single",
        )
        assert result["ordinary_income_tax"] == pytest.approx(2_671.50)
        assert result["capital_gains_tax"] == pytest.approx(0.0)
        assert result["taxable_income"] == pytest.approx(34_250.0)
