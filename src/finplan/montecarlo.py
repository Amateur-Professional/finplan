"""Monte Carlo runner: executes N stochastic paths and aggregates results.

Vectorises the *return/inflation RNG* in numpy (one batch fills an
``(n_trials, n_years)`` array), then runs the validated scalar engine
``simulation.simulate_path`` once per trial. The per-trial Python loop is a
deliberate design choice: it keeps the deterministic and Monte Carlo paths a
single, tested source of truth rather than maintaining a second vectorised
engine. See the project notes; vectorising the engine is a deferred optimisation
to revisit only if profiling shows the loop is a real bottleneck.

Memory footprint
----------------
Only an ``(n_trials, n_years)`` wealth matrix plus per-trial ``success`` /
terminal-wealth vectors are retained -- not 10k full ``TrialResult`` objects.
The median-outcome trial's full year-by-year detail is recovered by re-running
that single path (cheap) so the transparency requirement is met without holding
every path's detail in memory.
"""

from __future__ import annotations

import numpy as np

from finplan.models import PlanInput, SimulationResult
from finplan.results import aggregate
from finplan.returns import (
    ReturnSequences,
    generate_parametric,
    make_rng,
    resolve_seed,
)
from finplan.simulation import simulate_path


def _median_index(final_wealth: np.ndarray) -> int:
    """Index of the trial with the median terminal wealth.

    Uses the upper-median element of a stable argsort so the result is always a
    *real* trial (not an interpolated value) and is reproducible under a seed.
    """
    order = np.argsort(final_wealth, kind="stable")
    return int(order[final_wealth.shape[0] // 2])


def _simulate_batch(
    plan: PlanInput, sequences: ReturnSequences
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the scalar engine over every trial in ``sequences``.

    Returns ``(wealth_paths, success, final_wealth)``:
        wealth_paths  (n_trials, n_years) total wealth at each year-end
        success       (n_trials,) bool
        final_wealth  (n_trials,) terminal total wealth
    """
    n_trials = sequences.n_trials
    n_years = sequences.n_years
    wealth_paths = np.empty((n_trials, n_years), dtype=float)
    success = np.empty(n_trials, dtype=bool)
    final_wealth = np.empty(n_trials, dtype=float)

    real = sequences.portfolio_real
    inflation = sequences.inflation
    for i in range(n_trials):
        trial = simulate_path(plan, real[i], inflation[i])
        wealth_paths[i] = [yd.total_wealth for yd in trial.year_details]
        success[i] = trial.success
        final_wealth[i] = trial.final_wealth

    return wealth_paths, success, final_wealth


def run(plan: PlanInput) -> SimulationResult:
    """Run the full Monte Carlo simulation and aggregate the outcome.

    Draws ``plan.n_trials`` stochastic return/inflation sequences (seeded by
    ``plan.random_seed``), walks each through the deterministic engine, and
    returns a :class:`SimulationResult` with success probability, percentile
    wealth paths, and the median trial's full year-by-year detail.

    Every run is reproducible: if ``plan.random_seed`` is ``None`` a concrete
    seed is drawn and recorded on ``result.plan_input.random_seed``, so feeding
    that plan back reproduces the run exactly.
    """
    n_years = plan.plan_end_year - plan.plan_start_year + 1
    years = list(range(plan.plan_start_year, plan.plan_end_year + 1))

    # Pin the seed (drawing fresh entropy if unset) and stamp it onto the plan
    # the result carries, so the exact run can always be replayed.
    seed = resolve_seed(plan.random_seed)
    plan = plan.model_copy(update={"random_seed": seed})

    rng = make_rng(seed)
    sequences = generate_parametric(
        plan.returns, plan.inflation, plan.n_trials, n_years, rng
    )

    wealth_paths, success, final_wealth = _simulate_batch(plan, sequences)

    median_idx = _median_index(final_wealth)
    median_trial = simulate_path(
        plan,
        sequences.portfolio_real[median_idx],
        sequences.inflation[median_idx],
    )

    return aggregate(plan, years, wealth_paths, success, median_trial.year_details)
