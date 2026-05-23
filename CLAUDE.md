# Financial Planning Engine — Claude Code Context

## What this is
An open-source, US-focused financial planning ENGINE (a library, NOT an app).
It produces rigorous, tax-aware, transparent retirement projections via
Monte Carlo simulation. Core differentiators: correctness, transparency
(every output number traceable to inputs), and a clean, embeddable API.

## v0 scope — build ONLY these
- Lifetime year-by-year cash-flow projection
- Monte Carlo simulation: stochastic returns AND stochastic inflation
- Three account types: taxable (with cost-basis tracking), tax-deferred
  (traditional), tax-free (Roth)
- Federal tax modeling: ordinary income brackets, long-term capital gains,
  standard deduction, simplified Social Security taxation, RMDs from
  tax-deferred accounts
- Withdrawal strategies: (1) fixed real spending, (2) tax-efficient
  account-ordering
- Outputs: probability of success, percentile wealth paths, and FULL
  year-by-year detail that is inspectable

## OUT of scope for v0 — do NOT build these unless I explicitly ask
- State taxes (federal only; note as roadmap)
- Social Security claiming OPTIMIZATION (use a fixed claim age input)
- Equity comp, estate, insurance, education funding
- Roth conversion OPTIMIZATION (basic modeling only)
- Any web UI, frontend, or API server (library + demo notebook only)
- Microservices, Docker, cloud infra

## Correctness requirements (CRITICAL — this is the whole value)
- All financial math MUST be covered by tests.
- Deterministic mode must match hand-calculated compound growth exactly.
- Reproduce the 4% rule: ~30yr horizon, balanced portfolio, success rate
  consistent with Bengen/Trinity research.
- Monte Carlo success rates must land near cFIREsim/FIRECalc for identical
  inputs (document the comparison in tests).
- Tax calcs must match published federal brackets for sample incomes.
- When unsure about a tax rule, STOP and flag it for me — do not guess.

## Transparency requirement
- No black-box outputs. Results objects must expose year-by-year detail:
  income, withdrawals by account, taxes paid, balances, returns applied.

## Conventions
- Python 3.11+, full type hints.
- pydantic models for all inputs and outputs.
- numpy for the simulation (vectorize trials; avoid Python loops over trials).
- pytest; write tests alongside each module, not after.
- Small, pure functions where possible. Financial logic must be readable.
- Cite the source (IRS pub, paper) in a comment for any non-obvious formula.

## Module layout
finplan/
  accounts.py      # account types, cost-basis tracking, RMD logic
  taxes.py         # federal income, cap gains, std deduction, SS taxation
  returns.py       # CMA-based and historical-bootstrap return generators
  simulation.py    # single-path year-by-year engine (deterministic)
  montecarlo.py    # runs N stochastic paths; aggregates results
  withdrawal.py    # withdrawal strategies
  models.py        # pydantic input/output schemas
  results.py       # result objects + success prob + percentiles
tests/             # mirror of the above + benchmark validation
notebooks/         # demo only

## Commands
- Install deps: `uv sync`
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type-check: `uv run pyright`