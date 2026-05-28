"""Result aggregation: success probability, percentile wealth paths, summary stats.

Turns the raw per-trial outcomes collected by ``montecarlo.py`` into the
inspectable :class:`SimulationResult`. Two notions of "the middle" coexist and
are intentionally distinct:

* **Percentile paths** are *cross-sectional*: at each calendar year we take the
  p10/p25/p50/p75/p90 of total wealth across all trials. The p50 path is a
  per-year median and does NOT correspond to any single trial's trajectory.
* **median_year_details** is one real trial -- the trial whose terminal wealth
  is the median -- kept in full so every number stays traceable (transparency
  requirement). ``montecarlo.py`` selects and supplies it.
"""

from __future__ import annotations

import numpy as np

from finplan.models import PlanInput, SimulationResult, YearDetail

# Percentiles exposed in SimulationResult.percentile_paths, as (key, q) pairs.
_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("p10", 10.0),
    ("p25", 25.0),
    ("p50", 50.0),
    ("p75", 75.0),
    ("p90", 90.0),
)


def percentile_paths(wealth_paths: np.ndarray) -> dict[str, list[float]]:
    """Cross-sectional wealth percentiles per year.

    ``wealth_paths`` has shape ``(n_trials, n_years)``. Returns one list of
    length ``n_years`` per percentile key.
    """
    quantiles = np.percentile(wealth_paths, [q for _, q in _PERCENTILES], axis=0)
    return {key: quantiles[i].tolist() for i, (key, _) in enumerate(_PERCENTILES)}


def aggregate(
    plan: PlanInput,
    years: list[int],
    wealth_paths: np.ndarray,
    success: np.ndarray,
    median_year_details: list[YearDetail],
) -> SimulationResult:
    """Assemble a :class:`SimulationResult` from raw per-trial arrays.

    ``wealth_paths``  (n_trials, n_years) total wealth at each year-end.
    ``success``       (n_trials,) bool -- trial funded every retirement year.
    """
    n_trials = int(wealth_paths.shape[0])
    n_successes = int(np.count_nonzero(success))
    return SimulationResult(
        success_probability=n_successes / n_trials,
        years=years,
        percentile_paths=percentile_paths(wealth_paths),
        median_year_details=median_year_details,
        n_trials=n_trials,
        n_successes=n_successes,
        plan_input=plan,
    )
