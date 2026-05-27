"""Pydantic input/output schemas for the finplan engine.

All financial logic lives in other modules; this file is schema-only.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AccountType(StrEnum):
    TAXABLE = "taxable"
    TAX_DEFERRED = "tax_deferred"  # Traditional 401(k) / IRA
    TAX_FREE = "tax_free"  # Roth 401(k) / Roth IRA


class FilingStatus(StrEnum):
    SINGLE = "single"
    MARRIED_FILING_JOINTLY = "married_filing_jointly"


class SpendingPolicy(StrEnum):
    """How much to spend each year (the after-tax target)."""

    FIXED_REAL = "fixed_real"  # Constant inflation-adjusted spending


class SourcingPolicy(StrEnum):
    """Which accounts fund the spending, in what order."""

    CONVENTIONAL = "conventional"  # taxable -> tax-deferred -> Roth
    TAX_EFFICIENT = "tax_efficient"  # bracket-aware sequencing to lower lifetime tax


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class AccountInput(BaseModel):
    """Starting state for one account bucket."""

    account_type: AccountType
    balance: float = Field(ge=0.0, description="Current balance in today's dollars")
    cost_basis: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost basis for taxable accounts (ignored for tax-deferred/free)",
    )

    @model_validator(mode="after")
    def cost_basis_le_balance(self) -> AccountInput:
        if self.account_type == AccountType.TAXABLE and self.cost_basis > self.balance:
            raise ValueError("cost_basis cannot exceed balance for a taxable account")
        return self


class ReturnAssumptions(BaseModel):
    """Capital-market assumptions for a two-asset (equity/bond) portfolio.

    Returns are expressed as *real* annualised values so that inflation is
    modelled separately and the two sources of uncertainty are independent.
    """

    equity_real_return: float = Field(
        description="Mean real annual return for equities (e.g. 0.05 for 5%)"
    )
    equity_volatility: float = Field(
        gt=0.0, description="Annual std-dev for equity returns (e.g. 0.15)"
    )
    bond_real_return: float = Field(
        description="Mean real annual return for bonds (e.g. 0.01 for 1%)"
    )
    bond_volatility: float = Field(
        gt=0.0, description="Annual std-dev for bond returns (e.g. 0.05)"
    )
    equity_bond_correlation: float = Field(
        default=-0.10,
        ge=-1.0,
        le=1.0,
        description="Correlation between annual equity and bond returns",
    )
    equity_allocation: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of portfolio held in equities (0-1)",
    )


class InflationAssumptions(BaseModel):
    """Stochastic inflation parameters (log-normal)."""

    mean: float = Field(
        default=0.025,
        ge=0.0,
        description="Mean annual inflation rate (e.g. 0.025 for 2.5%)",
    )
    volatility: float = Field(
        default=0.010,
        ge=0.0,
        description="Annual std-dev of inflation (e.g. 0.01)",
    )


class SocialSecurityInput(BaseModel):
    """Social Security parameters. Claiming optimisation is out of v0 scope."""

    claim_age: int = Field(
        ge=62,
        le=70,
        description="Age at which benefits begin",
    )
    annual_benefit: float = Field(
        gt=0.0,
        description="Annual PIA in today's real dollars",
    )


class PlanInput(BaseModel):
    """Top-level input to the simulation engine."""

    # --- Demographics ---
    birth_year: int = Field(description="Planner's birth year")
    plan_start_year: int = Field(description="First year of the projection")
    retirement_year: int = Field(
        description="Year retirement begins (contributions stop)"
    )
    plan_end_year: int = Field(description="Last year of the projection (e.g. age 95)")

    # --- Accounts ---
    accounts: list[AccountInput] = Field(
        min_length=1,
        description="One entry per account bucket at plan start",
    )

    # --- Contributions (pre-retirement, nominal dollars per year) ---
    annual_contributions: dict[AccountType, float] = Field(
        default_factory=dict,
        description="Annual contributions by account type while still working",
    )

    # --- Spending ---
    annual_spending_real: float = Field(
        gt=0.0,
        description="Desired after-tax annual spending in today's real dollars",
    )
    spending_policy: SpendingPolicy = SpendingPolicy.FIXED_REAL
    sourcing_policy: SourcingPolicy = SourcingPolicy.CONVENTIONAL

    # --- Capital-market assumptions ---
    returns: ReturnAssumptions
    inflation: InflationAssumptions = Field(default_factory=InflationAssumptions)

    # --- Social Security (optional) ---
    social_security: SocialSecurityInput | None = None

    # --- Tax ---
    filing_status: FilingStatus = FilingStatus.SINGLE

    # --- Monte Carlo controls ---
    n_trials: int = Field(default=10_000, gt=0)
    random_seed: int | None = None

    @model_validator(mode="after")
    def year_ordering(self) -> PlanInput:
        if not (self.plan_start_year <= self.retirement_year <= self.plan_end_year):
            raise ValueError(
                "plan_start_year <= retirement_year <= plan_end_year required"
            )
        return self


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class AccountBalances(BaseModel):
    """End-of-year balances for each account bucket."""

    taxable: float
    tax_deferred: float
    tax_free: float

    @property
    def total(self) -> float:
        return self.taxable + self.tax_deferred + self.tax_free


class YearDetail(BaseModel):
    """Full year-by-year detail for one simulation path.

    Every number here must be traceable to a PlanInput field — no black boxes.
    Values are in *nominal* dollars for that calendar year unless stated.
    """

    year: int
    age: int

    # --- Returns & inflation applied this year ---
    portfolio_return: float = Field(description="Weighted nominal return applied")
    inflation_rate: float = Field(description="Inflation rate used for this year")

    # --- Income ---
    social_security_income: float = Field(default=0.0)
    rmd_amount: float = Field(
        default=0.0,
        description="Required minimum distribution from tax-deferred accounts",
    )

    # --- Withdrawals by account (positive = withdrawal from portfolio) ---
    withdrawal_taxable: float = Field(default=0.0)
    withdrawal_tax_deferred: float = Field(default=0.0)
    withdrawal_tax_free: float = Field(default=0.0)

    # --- Contributions (pre-retirement; zero after retirement_year) ---
    contribution_taxable: float = Field(default=0.0)
    contribution_tax_deferred: float = Field(default=0.0)
    contribution_tax_free: float = Field(default=0.0)

    # --- Taxes ---
    ordinary_income_tax: float = Field(default=0.0)
    capital_gains_tax: float = Field(default=0.0)
    total_taxes: float = Field(default=0.0)

    # --- End-of-year state ---
    balances: AccountBalances
    total_wealth: float = Field(description="Sum of all account balances")


class TrialResult(BaseModel):
    """Result of a single Monte Carlo trial (one stochastic path)."""

    trial_index: int
    success: bool = Field(
        description="Wealth remained non-negative through plan_end_year"
    )
    year_details: list[YearDetail]
    final_wealth: float


class SimulationResult(BaseModel):
    """Aggregated result of all Monte Carlo trials.

    Satisfies the transparency requirement: both aggregate statistics and
    the median path's year-by-year detail are exposed.
    """

    # --- Top-line summary ---
    success_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of trials in which wealth lasted through plan_end_year",
    )

    # --- Percentile wealth paths ---
    # Each list has length == len(years).
    # Keys are "p10", "p25", "p50", "p75", "p90".
    years: list[int] = Field(description="Calendar years covered by the projection")
    percentile_paths: dict[str, list[float]] = Field(
        description="Total wealth at each year for key percentiles"
    )

    # --- Inspectable median path ---
    median_year_details: list[YearDetail] = Field(
        description="Full year-by-year detail for the median-outcome trial"
    )

    # --- Metadata ---
    n_trials: int
    n_successes: int
    plan_input: PlanInput
