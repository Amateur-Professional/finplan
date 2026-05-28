"""Return generators: CMA-based parametric and historical-bootstrap.

Parametric generator (v0)
-------------------------
Draws annual *real* equity and bond returns from a correlated bivariate normal
(means, vols, correlation from :class:`ReturnAssumptions`) and blends them by the
equity allocation to get a portfolio real return for each trial-year. Inflation
is drawn independently per :class:`InflationAssumptions`.

All randomness is vectorised: one call fills an ``(n_trials, n_years)`` array, so
the expensive RNG step has no Python loop over trials. (The per-path engine loop
lives in ``montecarlo.py`` -- a deliberate design choice; see that module.)

Distribution choices (flagged, not guessed)
-------------------------------------------
* **Asset returns -- normal.** Real equity/bond returns are drawn from a
  bivariate normal -- the standard assumption for parametric Monte Carlo
  retirement studies (e.g. Morningstar's "The State of Retirement Income,"
  which uses forward-looking simulated returns). Normal real returns can in
  principle be < -100% in an extreme tail; at realistic vols this is negligible
  and the engine clamps balances at zero anyway.
* **Inflation -- shifted log-normal of the price factor.** The annual inflation
  rate ``i`` is modelled as ``exp(N(m, s)) - 1`` so the gross price factor
  ``1+i`` is log-normal (strictly positive) per the ``InflationAssumptions``
  docstring, while still allowing mild deflation (``i`` slightly negative).
  ``m, s`` are moment-matched so the drawn rate has exactly the requested
  arithmetic ``mean`` and ``volatility``. With ``volatility == 0`` every draw is
  exactly ``mean`` (useful for deterministic-limit checks).

Historical bootstrap (roadmap -- out of scope for now)
------------------------------------------------------
Resampling contiguous blocks from a historical real-return + inflation series
captures the sequence-of-returns risk and mean reversion that i.i.d. parametric
draws miss (and is how the original Bengen/Trinity backtests were run). It is
**out of scope for v0**: the repository ships no vetted market-return/inflation
dataset, and fabricating one would violate the correctness contract.
``generate_historical`` raises until a series is wired in.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass

import numpy as np

from finplan.models import InflationAssumptions, ReturnAssumptions


@dataclass(frozen=True)
class ReturnSequences:
    """Generated stochastic inputs for one Monte Carlo batch.

    Both arrays have shape ``(n_trials, n_years)``. ``portfolio_real`` is the
    allocation-blended real return; ``inflation`` is the annual inflation rate.
    The nominal return the engine applies in a given year is
    ``(1 + real) * (1 + inflation) - 1``; it is combined per-path rather than
    stored here so the two sources of uncertainty stay independent and
    inspectable.
    """

    portfolio_real: np.ndarray
    inflation: np.ndarray

    def __post_init__(self) -> None:
        if self.portfolio_real.shape != self.inflation.shape:
            raise ValueError("portfolio_real and inflation must share a shape")

    @property
    def n_trials(self) -> int:
        return int(self.portfolio_real.shape[0])

    @property
    def n_years(self) -> int:
        return int(self.portfolio_real.shape[1])


def resolve_seed(seed: int | None) -> int:
    """Return a concrete integer seed, drawing fresh OS entropy if ``None``.

    Every run is reproducible: when the caller leaves ``random_seed`` unset we
    still pick a definite seed here (numpy's entropy source) and return it, so
    the chosen value can be recorded on the plan and replayed later. Passing the
    returned seed back to ``make_rng`` reconstructs the identical generator.
    """
    if seed is not None:
        return seed
    return secrets.randbits(64)


def make_rng(seed: int | None) -> np.random.Generator:
    """Build a numpy Generator from a (already resolved) seed."""
    return np.random.default_rng(seed)


def _lognormal_rate_params(mean: float, volatility: float) -> tuple[float, float]:
    """Solve (m, s) so that ``exp(N(m,s)) - 1`` has the given arithmetic moments.

    For the gross factor ``F = 1 + i = exp(N(m, s))``:
        E[i] = exp(m + s^2/2) - 1            == mean
        Var[i] = (exp(s^2) - 1) * exp(2m + s^2) == volatility^2

    Let ``A = 1 + mean = E[F]``. Then ``exp(s^2) = 1 + (volatility / A)^2`` and
    ``m = ln(A) - s^2/2``. With ``volatility == 0`` this gives ``s = 0`` and
    ``exp(m) = A``, i.e. every draw equals ``mean`` exactly.
    """
    a = 1.0 + mean
    s_sq = math.log1p((volatility / a) ** 2)
    m = math.log(a) - 0.5 * s_sq
    return m, math.sqrt(s_sq)


def generate_parametric(
    returns: ReturnAssumptions,
    inflation: InflationAssumptions,
    n_trials: int,
    n_years: int,
    rng: np.random.Generator,
) -> ReturnSequences:
    """Draw correlated real returns + inflation for ``n_trials`` x ``n_years``.

    The two equity/bond draws share a per-cell correlation
    (``equity_bond_correlation``); inflation is drawn independently. Returns a
    :class:`ReturnSequences` with both arrays shaped ``(n_trials, n_years)``.
    """
    if n_trials <= 0 or n_years <= 0:
        raise ValueError("n_trials and n_years must be positive")

    # --- Correlated real equity/bond returns (bivariate normal) ---
    se = returns.equity_volatility
    sb = returns.bond_volatility
    rho = returns.equity_bond_correlation
    cov = se * sb * rho
    mean_vec = [returns.equity_real_return, returns.bond_real_return]
    cov_mat = [[se * se, cov], [cov, sb * sb]]
    draws = rng.multivariate_normal(mean_vec, cov_mat, size=(n_trials, n_years))
    equity = draws[..., 0]
    bond = draws[..., 1]
    w = returns.equity_allocation
    portfolio_real = w * equity + (1.0 - w) * bond

    # --- Inflation (shifted log-normal of the price factor) ---
    m, s = _lognormal_rate_params(inflation.mean, inflation.volatility)
    z = rng.standard_normal(size=(n_trials, n_years))
    inflation_rates = np.exp(m + s * z) - 1.0

    return ReturnSequences(portfolio_real=portfolio_real, inflation=inflation_rates)


def generate_historical(*_args: object, **_kwargs: object) -> ReturnSequences:
    """Historical block-bootstrap generator -- not yet implemented (roadmap).

    Deferred until a vetted historical real-return + inflation series is wired
    into the library; see the module docstring.
    """
    raise NotImplementedError(
        "Historical-bootstrap returns require a vetted market-return/inflation "
        "series, which is not yet bundled. Use generate_parametric for v0."
    )
